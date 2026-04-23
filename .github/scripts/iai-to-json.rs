//! Convert iai-callgrind benchmark output to JSON for the CI site.
//!
//! Usage: iai-to-json <iai-output.txt> <sha>
//!
//! Parses iai-callgrind's output format which includes instruction
//! counts, cache misses, and branch mispredictions per benchmark.
//!
//! Compile: `rustc -O iai-to-json.rs -o iai-to-json`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::SystemTime;

struct BenchResult {
    name: String,
    instructions: Option<u64>,
    l1_misses: Option<u64>,
    l2_misses: Option<u64>,
    branch_misses: Option<u64>,
}

fn parse_output(contents: &str) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let mut current_name = String::new();
    let mut current = BenchResult {
        name: String::new(),
        instructions: None,
        l1_misses: None,
        l2_misses: None,
        branch_misses: None,
    };

    for line in contents.lines() {
        let line = line.trim();

        // Benchmark name lines end with ':'
        // e.g., "field::fp_mul" or "  fp_mul:"
        if line.ends_with(':') && !line.contains('|') && !line.starts_with("Instructions")
            && !line.starts_with("L1") && !line.starts_with("L2")
            && !line.starts_with("Ram") && !line.starts_with("Branch")
            && !line.starts_with("Total")
        {
            if !current.name.is_empty() {
                results.push(current);
            }
            current_name = line.trim_end_matches(':').trim().to_string();
            current = BenchResult {
                name: current_name.clone(),
                instructions: None,
                l1_misses: None,
                l2_misses: None,
                branch_misses: None,
            };
            continue;
        }

        // Parse metric lines.
        // Format varies but typically:
        //   Instructions:  1234|N/A (No previous results)
        //   L1 Hits:       1200|N/A
        //   L1 Misses:     34|N/A
        if let Some(val) = extract_metric(line, "Instructions:") {
            current.instructions = Some(val);
        } else if let Some(val) = extract_metric(line, "L1 Misses:") {
            current.l1_misses = Some(val);
        } else if let Some(val) = extract_metric(line, "L2 Misses:") {
            current.l2_misses = Some(val);
        } else if let Some(val) = extract_metric(line, "Branches Misses:") {
            current.branch_misses = Some(val);
        } else if let Some(val) = extract_metric(line, "Branch Misses:") {
            current.branch_misses = Some(val);
        }
    }

    if !current.name.is_empty() {
        results.push(current);
    }

    results
}

fn extract_metric(line: &str, prefix: &str) -> Option<u64> {
    if !line.starts_with(prefix) {
        return None;
    }
    let rest = line[prefix.len()..].trim();
    // Take digits before any '|' or space.
    let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    num_str.parse().ok()
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

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: iai-to-json <iai-output.txt> <sha>");
        std::process::exit(1);
    }

    let contents = fs::read_to_string(&args[1])?;
    let sha = &args[2];
    let results = parse_output(&contents);

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"total\": {},", results.len())?;
    writeln!(w, "  \"results\": [")?;

    for (i, r) in results.iter().enumerate() {
        write!(w, "    {{\"name\": {}", json_str(&r.name))?;
        if let Some(v) = r.instructions { write!(w, ", \"instructions\": {}", v)?; }
        if let Some(v) = r.l1_misses { write!(w, ", \"l1_misses\": {}", v)?; }
        if let Some(v) = r.l2_misses { write!(w, ", \"l2_misses\": {}", v)?; }
        if let Some(v) = r.branch_misses { write!(w, ", \"branch_misses\": {}", v)?; }
        write!(w, "}}")?;
        if i + 1 < results.len() { writeln!(w, ",")?; } else { writeln!(w)?; }
    }

    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    Ok(())
}
