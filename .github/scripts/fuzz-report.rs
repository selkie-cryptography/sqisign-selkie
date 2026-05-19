//! Collect and merge fuzz target results into JSON.
//!
//! Two modes:
//!   fuzz-report target <target-name> <sha> <fuzz-output.txt>
//!     → writes per-target result JSON to stdout
//!
//!   fuzz-report merge <sha> <results-dir>
//!     → finds all fuzz-result.json files, merges, writes to stdout
//!
//! Compile: `rustc -O fuzz-report.rs -o fuzz-report`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::SystemTime;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: fuzz-report target <name> <sha> <output.txt>");
        eprintln!("       fuzz-report merge <sha> <results-dir>");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "target" => {
            if args.len() != 5 {
                eprintln!("usage: fuzz-report target <name> <sha> <output.txt>");
                std::process::exit(1);
            }
            target_report(&args[2], &args[3], &args[4])
        }
        "merge" => {
            if args.len() != 4 {
                eprintln!("usage: fuzz-report merge <sha> <results-dir>");
                std::process::exit(1);
            }
            merge_report(&args[2], &args[3])
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            std::process::exit(1);
        }
    }
}

fn target_report(target: &str, _sha: &str, output_file: &str) -> io::Result<()> {
    let output = fs::read_to_string(output_file).unwrap_or_default();

    // Extract runs from "stat::number_of_executed_units: NNN"
    let runs: u64 = output.lines()
        .filter_map(|l| {
            l.find("stat::number_of_executed_units:").map(|i| {
                let rest = &l[i + 31..];
                rest.trim().parse().unwrap_or(0)
            })
        })
        .last()
        .unwrap_or(0);

    // Count crashes.
    let artifacts_dir = format!("fuzz/artifacts/{target}");
    let crashes = fs::read_dir(&artifacts_dir).ok()
        .map(|entries| entries.flatten().count() as u64)
        .unwrap_or(0);

    // Count corpus.
    let corpus_dir = format!("fuzz/corpus/{target}");
    let corpus = fs::read_dir(&corpus_dir).ok()
        .map(|entries| entries.flatten().count() as u64)
        .unwrap_or(0);

    let status = if crashes == 0 { "pass" } else { "fail" };

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    writeln!(w, "{{")?;
    writeln!(w, "  \"target\": {},", json_str(target))?;
    writeln!(w, "  \"status\": {},", json_str(status))?;
    writeln!(w, "  \"runs\": {},", runs)?;
    writeln!(w, "  \"crashes\": {},", crashes)?;
    writeln!(w, "  \"corpus_size\": {}", corpus)?;
    writeln!(w, "}}")?;
    Ok(())
}

fn merge_report(sha: &str, results_dir: &str) -> io::Result<()> {
    let mut targets = Vec::new();
    let mut pass = 0u32;
    let mut fail = 0u32;
    let mut total_runs = 0u64;
    let mut total_crashes = 0u64;
    let mut total_corpus = 0u64;

    // Find all fuzz-result.json files recursively.
    collect_results(Path::new(results_dir), &mut targets);

    for t in &targets {
        if t.status == "fail" { fail += 1; } else { pass += 1; }
        total_runs += t.runs;
        total_crashes += t.crashes;
        total_corpus += t.corpus_size;
    }

    let status = if fail == 0 { "pass" } else { "fail" };

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"status\": {},", json_str(status))?;
    writeln!(w, "  \"pass\": {},", pass)?;
    writeln!(w, "  \"fail\": {},", fail)?;
    writeln!(w, "  \"total_runs\": {},", total_runs)?;
    writeln!(w, "  \"total_crashes\": {},", total_crashes)?;
    writeln!(w, "  \"total_corpus\": {},", total_corpus)?;
    writeln!(w, "  \"targets\": [")?;
    for (i, t) in targets.iter().enumerate() {
        write!(w, "    {{\"target\":{},\"status\":{},\"runs\":{},\"crashes\":{},\"corpus_size\":{}}}",
            json_str(&t.target), json_str(&t.status), t.runs, t.crashes, t.corpus_size)?;
        if i + 1 < targets.len() { writeln!(w, ",")?; } else { writeln!(w)?; }
    }
    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    Ok(())
}

struct TargetResult {
    target: String,
    status: String,
    runs: u64,
    crashes: u64,
    corpus_size: u64,
}

fn collect_results(dir: &Path, results: &mut Vec<TargetResult>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_results(&path, results);
        } else if path.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("fuzz_") && n.ends_with(".json"))
        {
            // Strict prefix match so a stray sibling file (manifest,
            // checksum, status.json, …) in the results dir can't feed
            // the merge.
            if let Ok(contents) = fs::read_to_string(&path) {
                let target = extract_str(&contents, "target");
                let status = extract_str(&contents, "status");
                let runs = extract_num(&contents, "runs");
                let crashes = extract_num(&contents, "crashes");
                let corpus_size = extract_num(&contents, "corpus_size");
                if !target.is_empty() {
                    results.push(TargetResult { target, status, runs, crashes, corpus_size });
                }
            }
        }
    }
}

fn extract_str(json: &str, key: &str) -> String {
    let needle = format!("\"{}\"", key);
    json.find(&needle).and_then(|i| {
        let rest = &json[i + needle.len()..];
        rest.find(':').and_then(|c| {
            let after = rest[c + 1..].trim_start();
            if after.starts_with('"') {
                let mut end = 1;
                let bytes = after.as_bytes();
                while end < bytes.len() {
                    if bytes[end] == b'"' && (end == 1 || bytes[end - 1] != b'\\') { break }
                    end += 1;
                }
                Some(after[1..end].to_string())
            } else { None }
        })
    }).unwrap_or_default()
}

fn extract_num(json: &str, key: &str) -> u64 {
    let needle = format!("\"{}\"", key);
    json.find(&needle).and_then(|i| {
        let rest = &json[i + needle.len()..];
        rest.find(':').map(|c| {
            let after = rest[c + 1..].trim_start();
            let end = after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len());
            after[..end].parse().unwrap_or(0)
        })
    }).unwrap_or(0)
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len()+2); o.push('"');
    for c in s.chars() { match c { '"'=>o.push_str("\\\""), '\\'=>o.push_str("\\\\"),
        '\n'=>o.push_str("\\n"), '\r'=>o.push_str("\\r"), '\t'=>o.push_str("\\t"),
        c if (c as u32)<0x20 => o.push_str(&format!("\\u{:04x}",c as u32)), _=>o.push(c) } }
    o.push('"'); o
}

fn iso8601_now() -> String {
    let dur = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
    let s = dur.as_secs(); let (h,m,sc)=((s%86400)/3600,(s%3600)/60,s%60);
    let mut y=1970i64; let mut r=(s/86400) as i64;
    loop { let yd=if y%4==0&&(y%100!=0||y%400==0){366}else{365}; if r<yd{break} r-=yd; y+=1; }
    let lp=y%4==0&&(y%100!=0||y%400==0);
    let md=[31,if lp{29}else{28},31,30,31,30,31,31,30,31,30,31];
    let mut mo=0; for &d in &md { if r<d{break} r-=d; mo+=1; }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",y,mo+1,r+1,h,m,sc)
}
