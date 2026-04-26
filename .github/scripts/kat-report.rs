//! Run KAT and Wycheproof test vectors, output structured JSON for
//! the CI dashboard.
//!
//! Usage: kat-report <sha>
//!
//! This is a standalone script compiled with `rustc -O` in CI.
//! It shells out to `cargo test` and parses the output to extract
//! per-test pass/fail results.
//!
//! Compile: `rustc -O kat-report.rs -o kat-report`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::time::SystemTime;

struct TestResult {
    name: String,
    status: String, // "pass", "fail", "skip"
    detail: String,
}

fn run_tests(args: &[&str]) -> Vec<TestResult> {
    let output = Command::new("cargo")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("failed to run cargo test");

    let combined = stdout_and_stderr(&output);
    let stdout = String::from_utf8_lossy(&combined);
    let mut results = Vec::new();

    for line in stdout.lines() {
        // Match: test <name> ... ok
        // Match: test <name> ... FAILED
        // Match: test <name> ... ignored
        if line.starts_with("test ") && (line.contains(" ... ") || line.contains("...")) {
            let parts: Vec<&str> = line.splitn(2, " ... ").collect();
            if parts.len() != 2 {
                continue;
            }
            let name = parts[0].trim_start_matches("test ").trim().to_string();
            let outcome = parts[1].trim();
            let status = if outcome == "ok" {
                "pass"
            } else if outcome == "FAILED" {
                "fail"
            } else if outcome == "ignored" {
                "skip"
            } else {
                continue;
            };
            results.push(TestResult {
                name,
                status: status.to_string(),
                detail: String::new(),
            });
        }
        // Capture panic messages for failures.
        if line.contains("panicked at") {
            if let Some(last) = results.last_mut() {
                if last.status == "fail" && last.detail.is_empty() {
                    last.detail = line.trim().to_string();
                }
            }
        }
    }

    results
}

fn stdout_and_stderr(output: &std::process::Output) -> Vec<u8> {
    let mut combined = output.stdout.clone();
    combined.extend_from_slice(&output.stderr);
    combined
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
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
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

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: kat-report <sha>");
        std::process::exit(1);
    }
    let sha = &args[1];

    eprintln!("[kat-report] Running KAT tests...");
    let kat_results = run_tests(&["test", "--lib", "--", "kat_"]);

    eprintln!("[kat-report] Running Wycheproof tests...");
    let wyche_results = run_tests(&["test", "--test", "wycheproof"]);

    // Count individual test vectors from the JSON/source files.
    let kat_vectors = count_kat_vectors();
    let wycheproof_vectors = count_wycheproof_vectors();

    let all_results: Vec<(&str, &[TestResult], u64)> = vec![
        ("kat", &kat_results[..], kat_vectors),
        ("wycheproof", &wyche_results[..], wycheproof_vectors),
    ];

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;

    // Per-suite summary and results.
    writeln!(w, "  \"suites\": [")?;
    for (si, &(suite_name, results, vectors)) in all_results.iter().enumerate() {
        let pass = results.iter().filter(|r| r.status == "pass").count();
        let fail = results.iter().filter(|r| r.status == "fail").count();
        let skip = results.iter().filter(|r| r.status == "skip").count();

        writeln!(w, "    {{")?;
        writeln!(w, "      \"name\": {},", json_str(suite_name))?;
        writeln!(w, "      \"pass\": {},", pass)?;
        writeln!(w, "      \"fail\": {},", fail)?;
        writeln!(w, "      \"skip\": {},", skip)?;
        writeln!(w, "      \"total\": {},", results.len())?;
        writeln!(w, "      \"vectors\": {},", vectors)?;
        writeln!(w, "      \"tests\": [")?;

        for (i, r) in results.iter().enumerate() {
            write!(
                w,
                "        {{\"name\": {}, \"status\": {}",
                json_str(&r.name),
                json_str(&r.status)
            )?;
            if !r.detail.is_empty() {
                write!(w, ", \"detail\": {}", json_str(&r.detail))?;
            }
            write!(w, "}}")?;
            if i + 1 < results.len() {
                writeln!(w, ",")?;
            } else {
                writeln!(w)?;
            }
        }

        write!(w, "      ]\n    }}")?;
        if si + 1 < all_results.len() {
            writeln!(w, ",")?;
        } else {
            writeln!(w)?;
        }
    }
    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;

    Ok(())
}

/// Counts individual KAT vectors from `src/keys/kat_data.rs`.
/// Each vector is a tuple starting with `(` on its own indented line.
fn count_kat_vectors() -> u64 {
    let content = fs::read_to_string("src/keys/kat_data.rs").unwrap_or_default();
    content
        .lines()
        .filter(|line| {
            let t = line.trim();
            t == "("
        })
        .count() as u64
}

/// Counts individual Wycheproof test vectors by summing
/// `numberOfTests` across all JSON files in `tests/vectors/`.
fn count_wycheproof_vectors() -> u64 {
    let mut total = 0u64;
    let dir = match fs::read_dir("tests/vectors") {
        Ok(d) => d,
        Err(_) => return 0,
    };
    for entry in dir.flatten() {
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "json") {
            continue;
        }
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        // Extract "numberOfTests": N from the JSON.
        total += extract_num_u64(&content, "numberOfTests");
    }
    total
}

fn extract_num_u64(json: &str, key: &str) -> u64 {
    let needle = format!("\"{}\"", key);
    let Some(idx) = json.find(&needle) else {
        return 0;
    };
    let rest = &json[idx + needle.len()..];
    let Some(colon) = rest.find(':') else {
        return 0;
    };
    let after = rest[colon + 1..].trim_start();
    let end = after
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after.len());
    after[..end].parse().unwrap_or(0)
}
