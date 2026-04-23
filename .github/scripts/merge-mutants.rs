//! Merge incremental mutants results into an existing baseline.
//!
//! Usage: merge-mutants <incremental.json> <sha> [baseline-url]
//!
//! Fetches the existing baseline from the CI site, replaces survivors
//! for files touched in the incremental run, keeps the rest, and
//! recomputes summary stats. Outputs merged JSON to stdout.
//!
//! Compile: `rustc -O merge-mutants.rs -o merge-mutants`

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::{Command, Stdio};

const DEFAULT_BASELINE: &str = "https://sqisign-selkie-ci.fly.dev/mutants/latest.json";

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("usage: merge-mutants <incremental.json> <sha> [baseline-url]");
        std::process::exit(1);
    }

    let inc_path = &args[1];
    let sha = &args[2];
    let baseline_url = args.get(3).map(|s| s.as_str()).unwrap_or(DEFAULT_BASELINE);

    let inc = fs::read_to_string(inc_path)?;
    let baseline = fetch_url(baseline_url).unwrap_or_else(|| "{}".to_string());

    // Parse survivors from both.
    let inc_survivors = extract_survivors(&inc);
    let base_survivors = extract_survivors(&baseline);

    // Files touched in the incremental run.
    let touched: BTreeSet<&str> = inc_survivors.iter().map(|s| s.file.as_str()).collect();

    // Keep baseline survivors for untouched files.
    let mut merged: Vec<&Survivor> = base_survivors
        .iter()
        .filter(|s| !touched.contains(s.file.as_str()))
        .collect();

    // Add all incremental survivors.
    merged.extend(inc_survivors.iter());

    // Sort by file, line.
    merged.sort_by(|a, b| a.file.cmp(&b.file).then(a.line.cmp(&b.line)));

    // Parse summary stats from both.
    let bs = extract_summary(&baseline);
    let is = extract_summary(&inc);

    // Recompute merged summary.
    let missed = std::cmp::max(merged.len() as u64, bs.missed + is.missed);

    // Extract updated_at from incremental.
    let updated_at = extract_string(&inc, "updated_at");

    // Write merged JSON.
    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&updated_at))?;
    writeln!(w, "  \"summary\": {{")?;
    writeln!(w, "    \"caught\": {},", bs.caught + is.caught)?;
    writeln!(w, "    \"missed\": {},", missed)?;
    writeln!(w, "    \"timeout\": {},", bs.timeout + is.timeout)?;
    writeln!(w, "    \"unviable\": {},", bs.unviable + is.unviable)?;
    writeln!(w, "    \"total\": {}", bs.total + is.total)?;
    writeln!(w, "  }},")?;
    writeln!(w, "  \"survivors\": [")?;

    for (i, s) in merged.iter().enumerate() {
        write!(
            w,
            "    {{\"name\": {}, \"file\": {}, \"function\": {}, \"line\": {}}}",
            json_str(&s.name),
            json_str(&s.file),
            json_str(&s.function),
            s.line
        )?;
        if i + 1 < merged.len() {
            writeln!(w, ",")?;
        } else {
            writeln!(w)?;
        }
    }

    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;

    Ok(())
}

struct Survivor {
    name: String,
    file: String,
    function: String,
    line: u64,
}

struct Summary {
    caught: u64,
    missed: u64,
    timeout: u64,
    unviable: u64,
    total: u64,
}

/// Very simple JSON string extraction: find "key": "value" pairs.
fn extract_string(json: &str, key: &str) -> String {
    let needle = format!("\"{}\"", key);
    let Some(idx) = json.find(&needle) else {
        return String::new();
    };
    let rest = &json[idx + needle.len()..];
    let Some(colon) = rest.find(':') else {
        return String::new();
    };
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('"') {
        return String::new();
    }
    let start = 1;
    let mut end = start;
    let bytes = after.as_bytes();
    while end < bytes.len() {
        if bytes[end] == b'"' && (end == start || bytes[end - 1] != b'\\') {
            break;
        }
        end += 1;
    }
    after[start..end].to_string()
}

/// Extract a number value for a key.
fn extract_num(json: &str, key: &str) -> u64 {
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

fn extract_summary(json: &str) -> Summary {
    // Find the "summary" object and extract fields from within it.
    let section = json
        .find("\"summary\"")
        .and_then(|i| json[i..].find('{').map(|j| &json[i + j..]))
        .and_then(|s| s.find('}').map(|j| &s[..j + 1]))
        .unwrap_or("");
    Summary {
        caught: extract_num(section, "caught"),
        missed: extract_num(section, "missed"),
        timeout: extract_num(section, "timeout"),
        unviable: extract_num(section, "unviable"),
        total: extract_num(section, "total"),
    }
}

fn extract_survivors(json: &str) -> Vec<Survivor> {
    let mut survivors = Vec::new();

    // Find the "survivors" array and parse each object.
    let Some(arr_start) = json.find("\"survivors\"") else {
        return survivors;
    };
    let rest = &json[arr_start..];
    let Some(bracket) = rest.find('[') else {
        return survivors;
    };
    let arr = &rest[bracket..];

    // Split on "},{" to find individual objects.
    // This is fragile but works for our known JSON structure.
    for chunk in arr.split('{').skip(1) {
        let obj = format!("{{{}", chunk);
        let name = extract_string(&obj, "name");
        let file = extract_string(&obj, "file");
        let function = extract_string(&obj, "function");
        let line = extract_num(&obj, "line");
        if !file.is_empty() {
            survivors.push(Survivor {
                name,
                file,
                function,
                line,
            });
        }
    }

    survivors
}

fn fetch_url(url: &str) -> Option<String> {
    let output = Command::new("curl")
        .args(["-sf", url])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        None
    }
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
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}
