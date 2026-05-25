//! Parse divan benchmark output into a JSON array of per-benchmark records.
//!
//! Usage: divan-to-json (reads stdin, writes stdout)
//!
//! Divan prints a group header followed by one row per benchmark:
//!
//! ```text
//!   bigint   fastest  │ slowest  │ median  │ mean   │ samples │ iters
//!   ├─ mul   2.6 ns   │ 2.7 ns   │ 2.6 ns  │ 2.6 ns │ 100     │ 204800
//! ```
//!
//! We capture every timing column (not just the mean, as an earlier
//! version did) so the dashboard and PR comparison can show median, the
//! fast/slow spread, and sample counts. `value` is the median — more
//! robust to CI noise than the mean — and `range` is the half-spread,
//! so consumers that want a single number plus an error bar have both.
//!
//! Compile: `rustc -O divan-to-json.rs -o divan-to-json`

use std::io::{self, BufRead, Write};

/// One parsed benchmark row, with all timing columns in nanoseconds.
///
/// `mean` is the only always-present timing (every divan row has it);
/// the rest are `Option` because a malformed or truncated row may omit
/// later columns, and historical divan layouts vary.
struct Bench {
    /// `group::name`, e.g. `field::fp_mul`.
    full_name: String,
    /// Shortest observed sample (the `fastest` column).
    fastest: Option<f64>,
    /// Longest observed sample (the `slowest` column).
    slowest: Option<f64>,
    /// Median sample — the headline `value`.
    median: Option<f64>,
    /// Arithmetic mean of the samples.
    mean: f64,
    /// Number of samples divan collected.
    samples: Option<u64>,
    /// Total iterations across all samples.
    iters: Option<u64>,
}

impl Bench {
    /// Parses one divan result row under `group`, or `None` if `line`
    /// is not a `│`-delimited result row.
    ///
    /// The first column carries both the leaf name and the `fastest`
    /// timing (`├─ mul   2.6 ns`); the remaining columns are
    /// `slowest │ median │ mean │ samples │ iters`.
    fn from_row(group: &str, line: &str) -> Option<Bench> {
        if !line.contains('│') {
            return None;
        }

        let columns: Vec<&str> = line.split('│').collect();
        if columns.len() < 4 {
            return None;
        }

        // First column: strip tree glyphs / leading punctuation, then
        // the leaf name is the first token and `fastest` follows it.
        let head = columns[0].trim_start_matches(|c: char| {
            "│├╰─ \t".contains(c) || (!c.is_ascii_alphanumeric() && c != '_')
        });
        let mut head_tokens = head.split_whitespace();
        let name = head_tokens.next()?;
        if !name.chars().next()?.is_ascii_alphanumeric() && !name.starts_with('_') {
            return None;
        }

        let full_name = if group.is_empty() {
            name.to_string()
        } else {
            format!("{group}::{name}")
        };

        let fastest = parse_time(&head_tokens.collect::<Vec<_>>().join(" "));
        let slowest = parse_time(columns[1]);
        let median = parse_time(columns[2]);
        let mean = parse_time(columns[3])?;

        let samples = columns.get(4).and_then(|c| parse_count(c));
        let iters = columns.get(5).and_then(|c| parse_count(c));

        Some(Bench {
            full_name,
            fastest,
            slowest,
            median,
            mean,
            samples,
            iters,
        })
    }

    /// The headline value: median when divan reported it, else the mean.
    fn value(&self) -> f64 {
        self.median.unwrap_or(self.mean)
    }

    /// The half-spread `(slowest - fastest) / 2`, when both bounds exist.
    fn half_spread(&self) -> Option<f64> {
        match (self.fastest, self.slowest) {
            (Some(f), Some(s)) if s >= f => Some((s - f) / 2.0),
            _ => None,
        }
    }

    /// Writes this record as one object of the output JSON array.
    ///
    /// Always emits every key; absent timings/counts are `null` so the
    /// consumer schema is uniform.
    fn write_json(&self, w: &mut impl Write) -> io::Result<()> {
        let range = match self.half_spread() {
            Some(h) => format!("\"\\u00b1 {h} ns/iter\""),
            None => "null".to_string(),
        };

        write!(
            w,
            "  {{\"name\": \"{}\", \"unit\": \"ns/iter\", \"value\": {}, \"range\": {}, \
             \"median_ns\": {}, \"fastest_ns\": {}, \"slowest_ns\": {}, \"mean_ns\": {}, \
             \"samples\": {}, \"iters\": {}}}",
            self.full_name,
            self.value(),
            range,
            num_or_null(self.median),
            num_or_null(self.fastest),
            num_or_null(self.slowest),
            self.mean,
            count_or_null(self.samples),
            count_or_null(self.iters),
        )
    }
}

/// Parses a divan timing field (`"2.6 ns"`, `"1.2 µs"`) into nanoseconds.
fn parse_time(field: &str) -> Option<f64> {
    let mut parts = field.split_whitespace();
    let value: f64 = parts.next()?.parse().ok()?;
    let factor = match parts.next()? {
        "ps" => 0.001,
        "ns" => 1.0,
        "µs" | "us" => 1_000.0,
        "ms" => 1_000_000.0,
        "s" => 1_000_000_000.0,
        _ => return None,
    };

    Some(value * factor)
}

/// Parses a divan count field into an absolute count.
///
/// Divan abbreviates large iteration counts with a `K`/`M`/`G`/`T`
/// magnitude suffix that may be attached (`12.8M`) or space-separated
/// (`12.8 M`), and may use thousands separators (`204,800`). We scan the
/// numeric prefix and the first non-numeric character as the suffix,
/// wherever it falls, so both layouts parse.
fn parse_count(field: &str) -> Option<u64> {
    let mut digits = String::new();
    let mut suffix = None;

    for c in field.trim().chars() {
        if c.is_ascii_digit() || c == '.' {
            digits.push(c);
        } else if c == ',' || c.is_whitespace() {
            continue;
        } else {
            suffix = Some(c);
            break;
        }
    }

    let scale = match suffix {
        None => 1.0,
        Some('K' | 'k') => 1e3,
        Some('M') => 1e6,
        Some('G' | 'B') => 1e9,
        Some('T') => 1e12,
        Some(_) => return None,
    };

    let value: f64 = digits.parse().ok()?;

    Some((value * scale).round() as u64)
}

/// Renders an optional nanosecond timing as a JSON number or `null`.
fn num_or_null(value: Option<f64>) -> String {
    value.map_or_else(|| "null".to_string(), |v| v.to_string())
}

/// Renders an optional count as a JSON number or `null`.
fn count_or_null(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_string(), |v| v.to_string())
}

/// Reads divan output from stdin and writes the JSON array to stdout.
fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    let mut group = String::new();
    let mut first = true;

    writeln!(w, "[")?;

    for line in stdin.lock().lines() {
        let line = line?;

        // A group header is a bare word followed by the `fastest` column
        // label; it sets the group for the rows that follow.
        if let Some(word) = line.split_whitespace().next() {
            if word.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && line.contains("fastest")
            {
                group = word.to_string();
                continue;
            }
        }

        let Some(bench) = Bench::from_row(&group, &line) else {
            continue;
        };

        if !first {
            writeln!(w, ",")?;
        }
        first = false;

        bench.write_json(&mut w)?;
    }

    writeln!(w)?;
    writeln!(w, "]")?;

    Ok(())
}
