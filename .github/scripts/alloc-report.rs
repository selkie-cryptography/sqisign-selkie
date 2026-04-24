//! Wrap the alloc-report.json from the alloc_budget test with
//! sha/timestamp metadata. Output JSON to stdout.
//!
//! Usage: alloc-report <sha> <status> [alloc-report.json]
//!
//! Compile: `rustc -O alloc-report.rs -o alloc-report`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::SystemTime;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: alloc-report <sha> <status> [alloc-report.json]");
        std::process::exit(1);
    }
    let sha = &args[1];
    let status = &args[2];
    let report_path = args.get(3).map(|s| s.as_str()).unwrap_or("alloc-report.json");

    // Read the operations array from the test's report file.
    let operations = fs::read_to_string(report_path)
        .ok()
        .and_then(|contents| {
            // Extract the "operations" array value from the JSON.
            let key = "\"operations\":";
            contents.find(key).map(|idx| {
                let rest = &contents[idx + key.len()..];
                // Find the matching closing bracket.
                let mut depth = 0;
                let mut end = 0;
                for (i, c) in rest.chars().enumerate() {
                    match c {
                        '[' => depth += 1,
                        ']' => {
                            depth -= 1;
                            if depth == 0 {
                                end = i + 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                rest[..end].to_string()
            })
        })
        .unwrap_or_else(|| "[]".to_string());

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"status\": {},", json_str(status))?;
    writeln!(w, "  \"operations\": {}", operations)?;
    writeln!(w, "}}")?;
    Ok(())
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            _ => o.push(c),
        }
    }
    o.push('"');
    o
}

fn iso8601_now() -> String {
    let dur = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
    let s = dur.as_secs();
    let (h, m, sc) = ((s % 86400) / 3600, (s % 3600) / 60, s % 60);
    let mut y = 1970i64;
    let mut r = (s / 86400) as i64;
    loop {
        let yd = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if r < yd { break }
        r -= yd;
        y += 1;
    }
    let lp = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let md = [31, if lp { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 0;
    for &d in &md { if r < d { break } r -= d; mo += 1; }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo + 1, r + 1, h, m, sc)
}
