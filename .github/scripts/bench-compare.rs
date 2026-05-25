//! Compare two benchmark payloads and render a Markdown report for a PR
//! comment.
//!
//! Usage: `bench-compare <baseline.json> <current.json>` (writes Markdown
//! to stdout). Either file may be missing or empty — a missing baseline
//! yields a "no baseline yet" report rather than an error, so a PR's
//! first run still posts its own numbers.
//!
//! Both inputs are the grouped payload the dashboard stores:
//! `{ "groups": [ { "name", "benchmarks": [ { "name", "median_ns",
//! "fastest_ns", "slowest_ns", "samples", ... } ] } ] }`. We key each
//! benchmark by `group::name`, then tabulate the median delta. Wall-clock
//! medians from shared CI runners are noisy, so the report is explicitly
//! informational, not a gate.
//!
//! Compile: `rustc -O bench-compare.rs -o bench-compare`

use std::{collections::BTreeMap, env, fs};

/// Regression threshold for the ⚠️ marker, matching the historical 115%
/// alert level. Wall-clock noise means smaller deltas are not signal.
const REGRESSION_FACTOR: f64 = 1.15;

/// Hidden marker so the comment poster can find and update its own
/// previous comment instead of stacking new ones.
const MARKER: &str = "<!-- sqisign-bench-report -->";

/// A single benchmark's headline timing and spread, in nanoseconds.
struct Stats {
    /// Median sample time — the comparison key.
    median: f64,
    /// Half the fast/slow spread, when both bounds are known.
    half_spread: Option<f64>,
    /// Sample count divan collected.
    samples: Option<u64>,
}

/// Reads `path` and parses its grouped payload into a `group::name ->
/// Stats` map, or an empty map if the file is absent, empty, or malformed.
fn load(path: &str) -> BTreeMap<String, Stats> {
    let mut out = BTreeMap::new();

    let Ok(text) = fs::read_to_string(path) else {
        return out;
    };
    let Some(root) = Json::parse(&text) else {
        return out;
    };
    let Some(groups) = root.get("groups").and_then(Json::as_array) else {
        return out;
    };

    for group in groups {
        let Some(group_name) = group.get("name").and_then(Json::as_str) else {
            continue;
        };
        let Some(benches) = group.get("benchmarks").and_then(Json::as_array) else {
            continue;
        };

        for bench in benches {
            let Some(name) = bench.get("name").and_then(Json::as_str) else {
                continue;
            };
            // Older payloads only carry `time_ns`; new ones carry both.
            let Some(median) = bench
                .get("median_ns")
                .and_then(Json::as_f64)
                .or_else(|| bench.get("time_ns").and_then(Json::as_f64))
            else {
                continue;
            };

            let half_spread = match (
                bench.get("fastest_ns").and_then(Json::as_f64),
                bench.get("slowest_ns").and_then(Json::as_f64),
            ) {
                (Some(f), Some(s)) if s >= f => Some((s - f) / 2.0),
                _ => None,
            };
            let samples = bench
                .get("samples")
                .and_then(Json::as_f64)
                .map(|s| s as u64);

            out.insert(
                format!("{group_name}::{name}"),
                Stats {
                    median,
                    half_spread,
                    samples,
                },
            );
        }
    }

    out
}

/// Formats a nanosecond timing with an appropriate unit and 3 significant
/// decimals.
fn fmt_ns(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:.3} ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.3} µs", ns / 1_000.0)
    } else {
        format!("{ns:.3} ns")
    }
}

/// Renders the comparison table and writes it to stdout.
fn main() {
    let args: Vec<String> = env::args().collect();
    let baseline = args.get(1).map_or_else(BTreeMap::new, |p| load(p));
    let current = args.get(2).map_or_else(BTreeMap::new, |p| load(p));

    let mut out = String::new();
    out.push_str(MARKER);
    out.push('\n');
    out.push_str("## Benchmarks (PR vs `main`)\n\n");
    out.push_str(
        "_Wall-clock medians via [divan](https://github.com/nvzqz/divan) on \
         shared CI runners — noisy and informational, not a merge gate. \
         Deterministic instruction-count gating is tracked separately._\n\n",
    );

    if current.is_empty() {
        out.push_str("No benchmark data was produced for this run.\n");
        print!("{out}");
        return;
    }

    if baseline.is_empty() {
        out.push_str("No baseline on `main` yet — showing this PR's numbers only.\n\n");
    }

    out.push_str("| Benchmark | `main` median | PR median (± spread) | Δ | samples |\n");
    out.push_str("|---|--:|--:|--:|--:|\n");

    let mut regressions = 0usize;

    for (name, cur) in &current {
        let spread = cur
            .half_spread
            .map_or_else(String::new, |h| format!(" ± {}", fmt_ns(h)));
        let samples = cur
            .samples
            .map_or_else(|| "—".to_string(), |s| s.to_string());

        let (base_cell, delta_cell) = match baseline.get(name) {
            Some(base) if base.median > 0.0 => {
                let ratio = cur.median / base.median;
                let pct = (ratio - 1.0) * 100.0;
                let marker = if ratio >= REGRESSION_FACTOR {
                    regressions += 1;
                    " ⚠️"
                } else if ratio <= 1.0 / REGRESSION_FACTOR {
                    " ✅"
                } else {
                    ""
                };
                (fmt_ns(base.median), format!("{pct:+.1}%{marker}"))
            }
            _ => ("—".to_string(), "new".to_string()),
        };

        out.push_str(&format!(
            "| `{}` | {} | {}{} | {} | {} |\n",
            name,
            base_cell,
            fmt_ns(cur.median),
            spread,
            delta_cell,
            samples
        ));
    }

    out.push('\n');
    if regressions > 0 {
        out.push_str(&format!(
            "⚠️ {regressions} benchmark(s) slower than `main` by ≥15% — likely \
             runner noise, but worth a glance.\n",
        ));
    }

    print!("{out}");
}

/// A minimal JSON value, enough to read the benchmark payload without a
/// serde dependency (CI scripts are dependency-free, compiled with `rustc`).
enum Json {
    /// A JSON number.
    Num(f64),
    /// A JSON string (escapes already decoded loosely; we only read keys
    /// and numeric values, so exact unescaping is unnecessary).
    Str(String),
    /// A JSON array.
    Arr(Vec<Json>),
    /// A JSON object, as ordered key/value pairs.
    Obj(Vec<(String, Json)>),
    /// `null`, `true`, or `false` — kept as one variant since we never
    /// read boolean or null fields.
    Other,
}

impl Json {
    /// Parses a complete JSON document, or `None` on malformed input.
    fn parse(text: &str) -> Option<Json> {
        let bytes = text.as_bytes();
        let mut pos = 0;

        let value = Json::parse_value(bytes, &mut pos)?;
        skip_ws(bytes, &mut pos);

        Some(value)
    }

    /// Looks up `key` in an object, returning `None` for non-objects or
    /// absent keys.
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Returns the array elements, or `None` for non-arrays.
    fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(items) => Some(items),
            _ => None,
        }
    }

    /// Returns the string value, or `None` for non-strings.
    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the numeric value, or `None` for non-numbers.
    fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// Parses one value starting at `*pos`, advancing past it.
    fn parse_value(bytes: &[u8], pos: &mut usize) -> Option<Json> {
        skip_ws(bytes, pos);

        match bytes.get(*pos)? {
            b'{' => Json::parse_object(bytes, pos),
            b'[' => Json::parse_array(bytes, pos),
            b'"' => parse_string(bytes, pos).map(Json::Str),
            b't' | b'f' | b'n' => {
                while bytes.get(*pos).is_some_and(u8::is_ascii_alphabetic) {
                    *pos += 1;
                }
                Some(Json::Other)
            }
            _ => parse_number(bytes, pos).map(Json::Num),
        }
    }

    /// Parses an object body starting at the opening `{`.
    fn parse_object(bytes: &[u8], pos: &mut usize) -> Option<Json> {
        *pos += 1; // consume '{'
        let mut pairs = Vec::new();

        loop {
            skip_ws(bytes, pos);
            match bytes.get(*pos)? {
                b'}' => {
                    *pos += 1;
                    return Some(Json::Obj(pairs));
                }
                b',' => {
                    *pos += 1;
                    continue;
                }
                b'"' => {
                    let key = parse_string(bytes, pos)?;
                    skip_ws(bytes, pos);
                    if bytes.get(*pos)? != &b':' {
                        return None;
                    }
                    *pos += 1;
                    let value = Json::parse_value(bytes, pos)?;
                    pairs.push((key, value));
                }
                _ => return None,
            }
        }
    }

    /// Parses an array body starting at the opening `[`.
    fn parse_array(bytes: &[u8], pos: &mut usize) -> Option<Json> {
        *pos += 1; // consume '['
        let mut items = Vec::new();

        loop {
            skip_ws(bytes, pos);
            match bytes.get(*pos)? {
                b']' => {
                    *pos += 1;
                    return Some(Json::Arr(items));
                }
                b',' => {
                    *pos += 1;
                    continue;
                }
                _ => items.push(Json::parse_value(bytes, pos)?),
            }
        }
    }
}

/// Advances `*pos` past any JSON whitespace.
fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while let Some(b) = bytes.get(*pos) {
        if b.is_ascii_whitespace() {
            *pos += 1;
        } else {
            break;
        }
    }
}

/// Parses a JSON string starting at the opening quote, decoding `\"`,
/// `\\`, and `\uXXXX` loosely (enough for keys and our ASCII values).
fn parse_string(bytes: &[u8], pos: &mut usize) -> Option<String> {
    *pos += 1; // consume opening '"'
    let mut s = String::new();

    while let Some(&b) = bytes.get(*pos) {
        match b {
            b'"' => {
                *pos += 1;
                return Some(s);
            }
            b'\\' => {
                *pos += 1;
                let esc = *bytes.get(*pos)?;
                if esc == b'u' {
                    *pos += 4; // skip the 4 hex digits; we never key on them
                } else {
                    s.push(esc as char);
                }
                *pos += 1;
            }
            _ => {
                s.push(b as char);
                *pos += 1;
            }
        }
    }

    None
}

/// Parses a JSON number starting at `*pos`, advancing past it.
fn parse_number(bytes: &[u8], pos: &mut usize) -> Option<f64> {
    let start = *pos;

    while let Some(&b) = bytes.get(*pos) {
        if b.is_ascii_digit() || matches!(b, b'-' | b'+' | b'.' | b'e' | b'E') {
            *pos += 1;
        } else {
            break;
        }
    }

    std::str::from_utf8(&bytes[start..*pos]).ok()?.parse().ok()
}
