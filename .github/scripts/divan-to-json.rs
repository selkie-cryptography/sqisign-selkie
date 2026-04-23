//! Parse divan benchmark output into github-action-benchmark's
//! `customSmallerIsBetter` JSON format.
//!
//! Usage: divan-to-json (reads stdin, writes stdout)
//!
//! Divan output has group headers like:
//!   bigint   fastest  │ slowest  │ median  │ mean  │ samples │ iters
//! followed by result lines like:
//!   ├─ mul   2.6 ns   │ 2.7 ns   │ 2.6 ns  │ 2.6 ns│ 100     │ 204800
//!
//! We extract group::name and the "mean" column (4th timing value).
//!
//! Compile: `rustc -O divan-to-json.rs -o divan-to-json`

use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    let mut group = String::new();
    let mut first = true;

    writeln!(w, "[")?;

    for line in stdin.lock().lines() {
        let line = line?;

        // Detect group headers: a word followed by "fastest".
        if let Some(first_word) = line.split_whitespace().next() {
            if first_word
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
                && line.contains("fastest")
            {
                group = first_word.to_string();
                continue;
            }
        }

        // Match result lines: tree characters followed by a name and │-separated columns.
        let trimmed = line
            .trim_start_matches(|c: char| "│├╰─ \t".contains(c) || !c.is_ascii_alphanumeric() && c != '_');
        if trimmed.is_empty() || !trimmed.chars().next().map_or(false, |c| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        if !line.contains('│') {
            continue;
        }

        // Extract name.
        let name = match trimmed.split_whitespace().next() {
            Some(n) => n,
            None => continue,
        };

        let full_name = if group.is_empty() {
            name.to_string()
        } else {
            format!("{}::{}", group, name)
        };

        // Split by │ and take the 4th column (mean).
        let columns: Vec<&str> = line.split('│').collect();
        if columns.len() < 4 {
            continue;
        }
        let mean_field = columns[3].trim();

        // Parse value and unit.
        let mut parts = mean_field.split_whitespace();
        let value_str = match parts.next() {
            Some(v) => v,
            None => continue,
        };
        let unit = match parts.next() {
            Some(u) => u,
            None => continue,
        };

        let value: f64 = match value_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Normalize to nanoseconds.
        let ns = match unit {
            "ps" => value * 0.001,
            "ns" => value,
            "µs" => value * 1_000.0,
            "ms" => value * 1_000_000.0,
            "s" => value * 1_000_000_000.0,
            _ => continue,
        };

        if !first {
            writeln!(w, ",")?;
        }
        first = false;

        write!(
            w,
            "  {{\"name\": \"{}\", \"unit\": \"ns/iter\", \"value\": {}}}",
            full_name, ns
        )?;
    }

    writeln!(w)?;
    writeln!(w, "]")?;

    Ok(())
}
