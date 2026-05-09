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
    /// L1 cache misses, derived from `LL Hits + RAM Hits` (any access that
    /// failed L1 went on to hit LL or fall through to RAM).
    l1_misses: Option<u64>,
    /// LL cache misses, equivalent to `RAM Hits` (any access that failed
    /// the last-level cache went to RAM).
    l2_misses: Option<u64>,
    /// Branch mispredictions. Not collected by default in iai-callgrind
    /// (requires `--branch-sim=yes`); reported as null when absent.
    branch_misses: Option<u64>,
    /// Callgrind's estimated CPU cycles (instructions + memory penalties).
    estimated_cycles: Option<u64>,
}

/// Strip ANSI CSI escape sequences (`ESC [ ... letter`) from a line.
/// iai-callgrind colorizes output and `tee` preserves the codes, so the
/// captured `iai-output.txt` contains them and would otherwise foil
/// prefix matching.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for c2 in chars.by_ref() {
                if c2.is_ascii_alphabetic() { break; }
            }
        } else {
            out.push(c);
        }
    }
    out
}

const METRIC_PREFIXES: &[&str] = &[
    "Instructions:", "L1 Hits:", "LL Hits:", "RAM Hits:",
    "Total read+write:", "Estimated Cycles:",
    // Older / alternative names kept so we don't regress if the format swings back.
    "L1 Misses:", "L2 Misses:", "Branch Misses:", "Branches Misses:",
];

fn parse_output(contents: &str) -> Vec<BenchResult> {
    let mut results: Vec<BenchResult> = Vec::new();
    let mut current: Option<BenchResult> = None;
    let mut ll_hits: Option<u64> = None;
    let mut ram_hits: Option<u64> = None;

    let finalize = |bench: &mut BenchResult, ll: Option<u64>, ram: Option<u64>| {
        // L1 miss = anything that didn't hit L1 → went to LL or RAM.
        // LL miss = anything that didn't hit LL → went to RAM.
        if let (Some(l), Some(r)) = (ll, ram) {
            bench.l1_misses.get_or_insert(l + r);
            bench.l2_misses.get_or_insert(r);
        }
    };

    for raw_line in contents.lines() {
        let line = strip_ansi(raw_line);
        let trimmed = line.trim();

        // Benchmark name: a `<binary>::<group>::<bench>` path. iai-callgrind
        // 0.16 emits these without a trailing colon, e.g. `iai::field::fp_mul`.
        // Distinguish from metric lines (which start with one of METRIC_PREFIXES)
        // by requiring the `::` separator and the absence of any metric prefix.
        let is_metric = METRIC_PREFIXES.iter().any(|p| trimmed.starts_with(p));
        if !is_metric && trimmed.contains("::") && !trimmed.is_empty() {
            if let Some(mut bench) = current.take() {
                finalize(&mut bench, ll_hits, ram_hits);
                results.push(bench);
            }
            ll_hits = None;
            ram_hits = None;
            current = Some(BenchResult {
                name: trimmed.to_string(),
                instructions: None,
                l1_misses: None,
                l2_misses: None,
                branch_misses: None,
                estimated_cycles: None,
            });
            continue;
        }

        let Some(bench) = current.as_mut() else { continue };

        if let Some(v) = extract_metric(trimmed, "Instructions:") {
            bench.instructions = Some(v);
        } else if let Some(v) = extract_metric(trimmed, "LL Hits:") {
            ll_hits = Some(v);
        } else if let Some(v) = extract_metric(trimmed, "RAM Hits:") {
            ram_hits = Some(v);
        } else if let Some(v) = extract_metric(trimmed, "Estimated Cycles:") {
            bench.estimated_cycles = Some(v);
        } else if let Some(v) = extract_metric(trimmed, "L1 Misses:") {
            bench.l1_misses = Some(v);
        } else if let Some(v) = extract_metric(trimmed, "L2 Misses:") {
            bench.l2_misses = Some(v);
        } else if let Some(v) = extract_metric(trimmed, "Branch Misses:")
            .or_else(|| extract_metric(trimmed, "Branches Misses:"))
        {
            bench.branch_misses = Some(v);
        }
    }

    if let Some(mut bench) = current {
        finalize(&mut bench, ll_hits, ram_hits);
        results.push(bench);
    }

    results
}

fn extract_metric(line: &str, prefix: &str) -> Option<u64> {
    if !line.starts_with(prefix) {
        return None;
    }
    let rest = line[prefix.len()..].trim();
    // Take digits before any '|' or whitespace.
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
        if let Some(v) = r.estimated_cycles { write!(w, ", \"estimated_cycles\": {}", v)?; }
        write!(w, "}}")?;
        if i + 1 < results.len() { writeln!(w, ",")?; } else { writeln!(w)?; }
    }

    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    Ok(())
}
