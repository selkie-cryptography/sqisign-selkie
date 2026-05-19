//! Collect platform test matrix results and output JSON.
//!
//! Usage: platform-report <run-id> <sha>
//!
//! Queries the GitHub REST API (via curl + jq, both present on every
//! runner we use) for the workflow run's job list, filters to the
//! platform-matrix jobs, and emits a summary JSON for the dashboard.
//!
//! Previously shelled out to `gh`, but `gh` isn't on the Fly self-
//! hosted runner image and pulling it in just for one API call is
//! overkill — curl + jq are universally available.
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

    let repo = env::var("GITHUB_REPOSITORY").unwrap_or_default();
    if repo.is_empty() {
        eprintln!("GITHUB_REPOSITORY is not set");
        std::process::exit(1);
    }
    // Accept either GH_TOKEN (what `gh` reads) or GITHUB_TOKEN (default
    // in GHA jobs) so this works in either env shape.
    let token = env::var("GH_TOKEN")
        .or_else(|_| env::var("GITHUB_TOKEN"))
        .unwrap_or_default();
    if token.is_empty() {
        eprintln!("no GH_TOKEN / GITHUB_TOKEN in env");
        std::process::exit(1);
    }

    // The platform matrix in ci.yml emits jobs named
    // `lib + doc tests (<target>, <bits>-bit[, <variant>])`. Match that
    // prefix.
    //
    // The Actions REST jobs endpoint paginates at 30 per page by default;
    // bump per_page to 100 since CI never approaches that. If it ever
    // does, switch to following the `next` Link header.
    let url = format!(
        "https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"
    );
    let auth = format!("Authorization: Bearer {token}");
    let jq_filter = r#".jobs[] | select(.name | startswith("lib + doc tests")) | "\(.name)|\(.conclusion)""#;

    // curl -> jq via shell so the pipe stays inside one process tree.
    let pipeline = format!(
        "curl -fsSL -H 'Accept: application/vnd.github+json' -H \"$AUTH\" {url} | jq -r {filter}",
        url = shell_escape(&url),
        filter = shell_escape(jq_filter),
    );
    let output = Command::new("bash")
        .arg("-c")
        .arg(&pipeline)
        .env("AUTH", &auth)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("failed to spawn bash for curl|jq pipeline");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("curl|jq pipeline failed (status {}): {}", output.status, stderr);
    }

    let text = String::from_utf8_lossy(&output.stdout);
    eprintln!("REST API returned {} matching job line(s)", text.lines().count());
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

// Single-quote-wrap for safe interpolation into a bash command. The
// inputs we splice (REST URL, jq filter) never contain `'`, but we
// still quote rigorously: any `'` becomes `'\''`, the canonical way to
// embed a literal single quote inside a single-quoted shell string.
fn shell_escape(s: &str) -> String {
    let escaped = s.replace('\'', "'\\''");
    format!("'{}'", escaped)
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
