//! Count panic-related symbols in a release binary.
//!
//! Usage: panic-count <sha>
//!
//! Builds the release binary, runs `nm -C` on the rlib, and counts
//! panic-related symbols by category. Outputs JSON to stdout.
//!
//! Compile: `rustc -O panic-count.rs -o panic-count`

use std::env;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::time::SystemTime;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: panic-count <sha>");
        std::process::exit(1);
    }
    let sha = &args[1];

    // Build release.
    eprintln!("[panic-count] building release...");
    let status = Command::new("cargo")
        .args(["build", "--release"])
        .status()
        .expect("failed to run cargo build");
    if !status.success() {
        eprintln!("[panic-count] cargo build failed");
        std::process::exit(1);
    }

    // Find the rlib.
    let lib = find_rlib();
    let lib = match lib {
        Some(l) => l,
        None => {
            eprintln!("[panic-count] could not find rlib");
            std::process::exit(1);
        }
    };
    eprintln!("[panic-count] analyzing {lib}");

    // Run nm -C and collect output.
    let nm_output = Command::new("nm")
        .args(["-C", &lib])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();

    // Count by category.
    let bounds = nm_output.matches("panic_bounds_check").count();
    let asserts = nm_output.matches("assert_failed").count();
    let expects = nm_output.matches("expect_failed").count();
    let unwraps = nm_output.matches("unwrap_failed").count();
    let panic_fmt = nm_output.matches("panic_fmt").count();
    let total = bounds + asserts + expects + unwraps + panic_fmt;

    // Output JSON.
    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"total\": {},", total)?;
    writeln!(w, "  \"bounds_check\": {},", bounds)?;
    writeln!(w, "  \"assert\": {},", asserts)?;
    writeln!(w, "  \"expect\": {},", expects)?;
    writeln!(w, "  \"unwrap\": {},", unwraps)?;
    writeln!(w, "  \"panic_fmt\": {}", panic_fmt)?;
    writeln!(w, "}}")?;
    Ok(())
}

fn find_rlib() -> Option<String> {
    // Try direct path first.
    for entry in std::fs::read_dir("target/release").ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name == "libsqisign_selkie.rlib" {
            return Some(entry.path().to_string_lossy().to_string());
        }
    }
    // Try deps/.
    for entry in std::fs::read_dir("target/release/deps").ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("libsqisign_selkie-") && name.ends_with(".rlib") {
            return Some(entry.path().to_string_lossy().to_string());
        }
    }
    None
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
