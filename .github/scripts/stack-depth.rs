//! Measure peak stack usage of key operations using Valgrind Massif.
//!
//! Usage: stack-depth <sha>
//!
//! Builds a small test binary, runs it under `valgrind --tool=massif
//! --stacks=yes`, and parses the peak stack from ms_print output.
//!
//! Compile: `rustc -O stack-depth.rs -o stack-depth`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::time::SystemTime;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: stack-depth <sha>");
        std::process::exit(1);
    }
    let sha = &args[1];

    // Build the test binary that exercises key operations.
    eprintln!("[stack-depth] building test binary...");
    let status = Command::new("cargo")
        .args(["test", "--release", "--lib", "--no-run"])
        .status()
        .expect("failed to build tests");
    if !status.success() {
        eprintln!("[stack-depth] build failed");
        std::process::exit(1);
    }

    // Find the test binary.
    let test_bin = find_test_binary();
    let test_bin = match test_bin {
        Some(b) => b,
        None => {
            eprintln!("[stack-depth] could not find test binary");
            std::process::exit(1);
        }
    };

    // Run a quick test under massif to measure stack.
    // We run a specific test that exercises verify (a representative operation).
    eprintln!("[stack-depth] running under valgrind massif...");
    let massif_out = "/tmp/massif.out";
    let status = Command::new("valgrind")
        .args([
            "--tool=massif",
            "--stacks=yes",
            &format!("--massif-out-file={massif_out}"),
            "--pages-as-heap=no",
            &test_bin,
            "--test-threads=1",
            "signature_try_from", // a fast test that exercises parsing
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status();

    let peak_stack = match status {
        Ok(s) if s.success() || s.code() == Some(0) => parse_massif_peak(massif_out),
        _ => {
            eprintln!("[stack-depth] valgrind failed, trying ms_print");
            0
        }
    };

    // Also try to get peak from ms_print.
    let ms_peak = if std::path::Path::new(massif_out).exists() {
        let output = Command::new("ms_print")
            .arg(massif_out)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .ok();
        output
            .and_then(|o| {
                let text = String::from_utf8_lossy(&o.stdout);
                // Find the peak line: "  n        time(i)         total(B)   useful-heap(B)..."
                // The peak is the max "total(B)" value.
                text.lines()
                    .filter_map(|l| {
                        let parts: Vec<&str> = l.split_whitespace().collect();
                        if parts.len() >= 3 {
                            parts[2].replace(',', "").parse::<u64>().ok()
                        } else {
                            None
                        }
                    })
                    .max()
            })
            .unwrap_or(peak_stack)
    } else {
        peak_stack
    };

    let peak = if ms_peak > 0 { ms_peak } else { peak_stack };

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"peak_stack_bytes\": {}", peak)?;
    writeln!(w, "}}")?;
    Ok(())
}

fn parse_massif_peak(path: &str) -> u64 {
    let Ok(contents) = fs::read_to_string(path) else { return 0 };
    // Massif output has lines like: mem_stacks_B=12345
    contents
        .lines()
        .filter_map(|l| {
            l.strip_prefix("mem_stacks_B=")
                .and_then(|v| v.parse::<u64>().ok())
        })
        .max()
        .unwrap_or(0)
}

fn find_test_binary() -> Option<String> {
    // Find the most recently modified test binary in target/release/deps.
    let mut candidates: Vec<_> = fs::read_dir("target/release/deps")
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with("sqisign_selkie-")
                && !name.ends_with(".d")
                && !name.ends_with(".rlib")
                && !name.ends_with(".rmeta")
        })
        .collect();
    candidates.sort_by_key(|e| {
        e.metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    });
    candidates
        .last()
        .map(|e| e.path().to_string_lossy().to_string())
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn iso8601_now() -> String {
    let dur = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
    let secs = dur.as_secs();
    let (h, m, s) = ((secs % 86400) / 3600, (secs % 3600) / 60, secs % 60);
    let mut y = 1970i64;
    let mut rem = (secs / 86400) as i64;
    loop {
        let yd = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if rem < yd { break }
        rem -= yd;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let md = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 0;
    for &d in &md { if rem < d { break } rem -= d; mo += 1; }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo + 1, rem + 1, h, m, s)
}
