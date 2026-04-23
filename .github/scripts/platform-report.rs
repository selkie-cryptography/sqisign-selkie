//! Collect platform test matrix results and output JSON.
//!
//! Usage: platform-report <run-id> <sha>
//!
//! Uses `gh` CLI to fetch job results from the CI workflow run.
//!
//! Compile: `rustc -O platform-report.rs -o platform-report`

use std::env;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::time::SystemTime;

struct Platform {
    name: String,
    target: String,
    status: String,
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: platform-report <run-id> <sha>");
        std::process::exit(1);
    }
    let run_id = &args[1];
    let sha = &args[2];

    // Query GitHub API via gh CLI for job results.
    let output = Command::new("gh")
        .args([
            "run", "view", run_id,
            "--json", "jobs",
            "-q", ".jobs[] | select(.name | startswith(\"Test\")) | .name + \"|\" + .conclusion",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("failed to run gh");

    let text = String::from_utf8_lossy(&output.stdout);
    let mut platforms = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let parts: Vec<&str> = line.splitn(2, '|').collect();
        if parts.len() != 2 { continue; }

        let name = parts[0].trim().to_string();
        let conclusion = parts[1].trim();

        // Extract target from name like "Test (x86_64-unknown-linux-gnu, 64-bit)"
        let target = name
            .find('(')
            .and_then(|start| name.find(',').map(|end| &name[start + 1..end]))
            .unwrap_or(&name)
            .trim()
            .to_string();

        let status = match conclusion {
            "success" => "pass",
            "failure" => "fail",
            "skipped" => "skip",
            _ => "unknown",
        };

        platforms.push(Platform {
            name,
            target,
            status: status.to_string(),
        });
    }

    let pass = platforms.iter().filter(|p| p.status == "pass").count();
    let fail = platforms.iter().filter(|p| p.status == "fail").count();

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"pass\": {},", pass)?;
    writeln!(w, "  \"fail\": {},", fail)?;
    writeln!(w, "  \"total\": {},", platforms.len())?;
    writeln!(w, "  \"platforms\": [")?;

    for (i, p) in platforms.iter().enumerate() {
        write!(
            w,
            "    {{\"name\": {}, \"target\": {}, \"status\": {}}}",
            json_str(&p.name),
            json_str(&p.target),
            json_str(&p.status)
        )?;
        if i + 1 < platforms.len() { writeln!(w, ",")?; } else { writeln!(w)?; }
    }

    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    Ok(())
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
