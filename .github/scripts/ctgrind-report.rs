//! Run ctgrind tests under Valgrind memcheck and output JSON.
//!
//! Builds the ctgrind test binary, then runs each test function
//! individually under Valgrind so errors map to specific functions.
//!
//! Usage: ctgrind-report <sha>
//!
//! Compile: `rustc -O ctgrind-report.rs -o ctgrind-report`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::time::SystemTime;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: ctgrind-report <sha>");
        std::process::exit(1);
    }
    let sha = &args[1];

    // Build the ctgrind test binary.
    eprintln!("[ctgrind-report] building...");
    let status = Command::new("cargo")
        .args([
            "test", "--test", "ctgrind", "--features", "expose-internals", "--no-run",
        ])
        .status()
        .expect("failed to build");
    if !status.success() {
        eprintln!("[ctgrind-report] build failed");
        std::process::exit(1);
    }

    let bin = match find_binary("ctgrind") {
        Some(b) => b,
        None => {
            eprintln!("[ctgrind-report] binary not found");
            std::process::exit(1);
        }
    };

    // Discover test names by running with --list.
    let list_output = Command::new(&bin)
        .args(["--list", "--format=terse"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .expect("failed to list tests");
    let test_names: Vec<String> = String::from_utf8_lossy(&list_output.stdout)
        .lines()
        .filter_map(|l| l.strip_suffix(": test").map(|s| s.trim().to_string()))
        .collect();

    eprintln!(
        "[ctgrind-report] found {} tests: {:?}",
        test_names.len(),
        test_names
    );

    // Run each test individually under Valgrind.
    let mut results = Vec::new();
    let mut total_errors = 0;

    for name in &test_names {
        eprintln!("[ctgrind-report] running {name}...");
        let xml_file = format!("/tmp/ctgrind-{name}.xml");
        let output = Command::new("valgrind")
            .args([
                "--tool=memcheck",
                "--error-exitcode=0",
                "--xml=yes",
                &format!("--xml-file={xml_file}"),
                &bin,
                "--exact",
                name,
                "--test-threads=1",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .expect("failed to run valgrind");

        let errors = fs::read_to_string(&xml_file)
            .ok()
            .map(|xml| xml.matches("<error>").count())
            .unwrap_or(0);

        let stdout = String::from_utf8_lossy(&output.stdout);
        let passed = stdout.contains("test result: ok");
        let status = if errors == 0 && passed { "pass" } else { "fail" };

        total_errors += errors;
        results.push((name.clone(), status.to_string(), errors));
        eprintln!("[ctgrind-report]   {name}: {status} ({errors} errors)");
    }

    let overall = if total_errors == 0 { "pass" } else { "fail" };

    // Write JSON.
    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"status\": {},", json_str(overall))?;
    writeln!(w, "  \"errors\": {},", total_errors)?;
    writeln!(w, "  \"tests\": {},", results.len())?;
    writeln!(w, "  \"results\": [")?;
    for (i, (name, status, errors)) in results.iter().enumerate() {
        write!(
            w,
            "    {{\"name\": {}, \"status\": {}, \"errors\": {}}}",
            json_str(name),
            json_str(status),
            errors
        )?;
        if i + 1 < results.len() {
            writeln!(w, ",")?;
        } else {
            writeln!(w)?;
        }
    }
    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    Ok(())
}

fn find_binary(prefix: &str) -> Option<String> {
    for entry in fs::read_dir("target/debug/deps").ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with(prefix) && !name.contains('.') {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() && meta.len() > 0 {
                    return Some(entry.path().to_string_lossy().to_string());
                }
            }
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
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    let secs = dur.as_secs();
    let (h, m, s) = ((secs % 86400) / 3600, (secs % 3600) / 60, secs % 60);
    let mut y = 1970i64;
    let mut rem = (secs / 86400) as i64;
    loop {
        let yd = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
            366
        } else {
            365
        };
        if rem < yd {
            break;
        }
        rem -= yd;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let md = [
        31,
        if leap { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];
    let mut mo = 0;
    for &d in &md {
        if rem < d {
            break;
        }
        rem -= d;
        mo += 1;
    }
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        mo + 1,
        rem + 1,
        h,
        m,
        s
    )
}
