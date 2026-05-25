//! Render a Markdown CI report for one data kind, for a PR comment and
//! job summary.
//!
//! Usage: `ci-report <kind> <baseline.json> <current.json>` (writes
//! Markdown to stdout). Either file may be missing or empty — a missing
//! baseline yields a current-only report rather than an error.
//!
//! `kind` is one of `bench`, `iai`, `kat`, `dudect`, `tacet`, `mutants`,
//! `coverage` — the same kinds the dashboard stores per commit. The first
//! output line is a hidden marker (`<!-- ci-report:<kind> -->`) so the
//! comment poster can update its own previous comment in place, and the
//! footer deep-links the dashboard at the baseline commit.
//!
//! Compile: `rustc -O ci-report.rs -o ci-report`

use std::collections::BTreeMap;
use std::env;
use std::fs;

/// Dashboard / data host (same Fly app serves the static site and JSON).
const SITE: &str = "https://sqisign-selkie-ci.fly.dev";

/// Wall-clock regression threshold for the ⚠️ marker (matches the
/// historical 115% alert). Wall-clock noise means smaller deltas on
/// `bench` are not signal; `iai` instruction counts are deterministic and
/// use a much tighter threshold.
const WALLCLOCK_FACTOR: f64 = 1.15;

/// Renders the report for `kind` and writes it to stdout.
fn main() {
    let args: Vec<String> = env::args().collect();
    let kind = args.get(1).map(String::as_str).unwrap_or("");
    let baseline = args.get(2).and_then(|p| Json::from_file(p));
    let current = args.get(3).and_then(|p| Json::from_file(p));

    let base = baseline.as_ref();
    let cur = match current.as_ref() {
        Some(c) => c,
        None => {
            println!("<!-- ci-report:{kind} -->\nNo `{kind}` data was produced for this run.");
            return;
        }
    };

    let body = match kind {
        "bench" => render_bench(base, cur),
        "iai" => render_iai(base, cur),
        "kat" => render_kat(base, cur),
        "dudect" | "tacet" => render_ct(kind, base, cur),
        "mutants" => render_mutants(base, cur),
        "coverage" => render_coverage(base, cur),
        _ => format!("_No report renderer for `{kind}`._\n"),
    };

    // Link to the dashboard at the baseline (main) commit, which always
    // has data; PR-head commits are not uploaded.
    let footer = match base.and_then(|b| b.get("sha")).and_then(Json::as_str) {
        Some(sha) => format!(
            "\n[📊 Full dashboard (main @ {})]({SITE}/?sha={sha})\n",
            &sha[..sha.len().min(7)]
        ),
        None => format!("\n[📊 Full CI dashboard]({SITE}/)\n"),
    };

    print!("<!-- ci-report:{kind} -->\n{body}{footer}");
}

/// Renders the `bench` (divan wall-clock) report: a median comparison
/// table per benchmark.
fn render_bench(base: Option<&Json>, cur: &Json) -> String {
    let base_map = bench_map(base);
    let cur_map = bench_map(Some(cur));

    let mut out = String::from("### Benchmarks (PR vs `main`)\n\n");
    out.push_str(
        "_Wall-clock medians via [divan](https://github.com/nvzqz/divan) on \
         dedicated `perf-2x` Fly runners — lower-noise than shared CPUs but \
         still wall-clock, so informational, not a merge gate. Deterministic \
         instruction-count gating (iai) is separate._\n\n",
    );

    if cur_map.is_empty() {
        return out + "No benchmark data.\n";
    }
    if base_map.is_empty() {
        out.push_str("No baseline on `main` yet — showing this PR's numbers only.\n\n");
    }

    // Build the table body, counting regressions for the visible summary.
    let mut rows = String::new();
    let mut regressions = 0usize;

    for (name, (median, spread, samples)) in &cur_map {
        let spread_str = spread.map_or_else(String::new, |h| format!(" ± {}", fmt_ns(h)));
        let samples_str = samples.map_or_else(|| "—".to_string(), |s| s.to_string());

        let (base_cell, delta) = match base_map.get(name) {
            Some((b, _, _)) if *b > 0.0 => {
                if *median / *b >= WALLCLOCK_FACTOR {
                    regressions += 1;
                }
                (fmt_ns(*b), wallclock_delta(*median, *b))
            }
            _ => ("—".to_string(), "new".to_string()),
        };

        rows.push_str(&format!(
            "| `{name}` | {base_cell} | {}{spread_str} | {delta} | {samples_str} |\n",
            fmt_ns(*median)
        ));
    }

    if regressions > 0 {
        out.push_str(&format!(
            "⚠️ **{regressions} benchmark(s) ≥15% slower than `main`** — likely runner noise; expand the table to check.\n\n"
        ));
    }

    // Fold the ~80-row table so it doesn't dominate the PR conversation.
    out.push_str(&format!(
        "<details><summary>{} benchmarks</summary>\n\n\
         | Benchmark | `main` | PR (± spread) | Δ | samples |\n|---|--:|--:|--:|--:|\n{rows}\n</details>\n",
        cur_map.len()
    ));

    out
}

/// Builds a `group::name -> (median_ns, half_spread, samples)` map from a
/// `bench` payload.
fn bench_map(root: Option<&Json>) -> BTreeMap<String, (f64, Option<f64>, Option<u64>)> {
    let mut map = BTreeMap::new();

    let Some(groups) = root.and_then(|r| r.get("groups")).and_then(Json::as_array) else {
        return map;
    };

    for group in groups {
        let Some(gname) = group.get("name").and_then(Json::as_str) else {
            continue;
        };
        let Some(benches) = group.get("benchmarks").and_then(Json::as_array) else {
            continue;
        };

        for b in benches {
            let Some(name) = b.get("name").and_then(Json::as_str) else {
                continue;
            };
            let Some(median) = b
                .get("median_ns")
                .and_then(Json::as_f64)
                .or_else(|| b.get("time_ns").and_then(Json::as_f64))
            else {
                continue;
            };

            let spread = match (
                b.get("fastest_ns").and_then(Json::as_f64),
                b.get("slowest_ns").and_then(Json::as_f64),
            ) {
                (Some(f), Some(s)) if s >= f => Some((s - f) / 2.0),
                _ => None,
            };
            let samples = b.get("samples").and_then(Json::as_f64).map(|s| s as u64);

            map.insert(format!("{gname}::{name}"), (median, spread, samples));
        }
    }

    map
}

/// Renders the `iai` report: deterministic instruction-count deltas. Any
/// increase is real signal (no runner noise), so the threshold is tight.
fn render_iai(base: Option<&Json>, cur: &Json) -> String {
    let base_map = iai_map(base);
    let cur_map = iai_map(Some(cur));

    let mut out = String::from("### Instruction counts (PR vs `main`)\n\n");
    out.push_str(
        "_Deterministic instruction counts via \
         [iai-callgrind](https://github.com/iai-callgrind/iai-callgrind) \
         (Valgrind) — immune to runner noise, so any change is real._\n\n",
    );

    if cur_map.is_empty() {
        return out + "No instruction-count data.\n";
    }

    let cur_total: u64 = cur_map.values().sum();
    let base_total: u64 = base_map.values().sum();
    if base_total > 0 {
        out.push_str(&format!(
            "**Total: {} instructions ({} vs `main`).**\n\n",
            fmt_int(cur_total),
            signed_pct(cur_total as f64, base_total as f64)
        ));
    }

    let mut rows = String::new();
    let mut increased = 0usize;

    for (name, &cur_instr) in &cur_map {
        let (base_cell, delta) = match base_map.get(name) {
            Some(&b) if b > 0 => {
                let pct = (cur_instr as f64 / b as f64 - 1.0) * 100.0;
                let mark = if cur_instr > b {
                    increased += 1;
                    " ⚠️"
                } else if cur_instr < b {
                    " ✅"
                } else {
                    ""
                };
                (fmt_int(b), format!("{pct:+.2}%{mark}"))
            }
            _ => ("—".to_string(), "new".to_string()),
        };

        rows.push_str(&format!(
            "| `{name}` | {base_cell} | {} | {delta} |\n",
            fmt_int(cur_instr)
        ));
    }

    if increased > 0 {
        out.push_str(&format!(
            "⚠️ **{increased} benchmark(s) with more instructions than `main`** (deterministic — real, not noise).\n\n"
        ));
    }

    out.push_str(&format!(
        "<details><summary>{} benchmarks</summary>\n\n\
         | Benchmark | `main` | PR | Δ |\n|---|--:|--:|--:|\n{rows}\n</details>\n",
        cur_map.len()
    ));

    out
}

/// Builds a `name -> instructions` map from an `iai` payload.
fn iai_map(root: Option<&Json>) -> BTreeMap<String, u64> {
    let mut map = BTreeMap::new();

    let Some(results) = root.and_then(|r| r.get("results")).and_then(Json::as_array) else {
        return map;
    };

    for r in results {
        let Some(name) = r.get("name").and_then(Json::as_str) else {
            continue;
        };
        let Some(instr) = r.get("instructions").and_then(Json::as_f64) else {
            continue;
        };
        map.insert(name.to_string(), instr as u64);
    }

    map
}

/// Renders the `kat` report: known-answer-test pass/fail across suites.
fn render_kat(_base: Option<&Json>, cur: &Json) -> String {
    let Some(suites) = cur.get("suites").and_then(Json::as_array) else {
        return "### Known-answer tests\n\nNo KAT data.\n".to_string();
    };

    let mut pass = 0u64;
    let mut total = 0u64;
    let mut failing = Vec::new();

    for s in suites {
        let p = s.get("pass").and_then(Json::as_f64).unwrap_or(0.0) as u64;
        let t = s.get("total").and_then(Json::as_f64).unwrap_or(0.0) as u64;
        pass += p;
        total += t;
        if t > p {
            if let Some(name) = s.get("name").and_then(Json::as_str) {
                failing.push(format!("`{name}` ({}/{t})", p));
            }
        }
    }

    let head = if total > 0 && pass == total {
        format!("### Known-answer tests\n\n✅ **{pass}/{total} passing.**\n")
    } else {
        format!(
            "### Known-answer tests\n\n❌ **{}/{total} failing.**\n",
            total - pass
        )
    };

    if failing.is_empty() {
        head
    } else {
        format!("{head}\nFailing suites: {}\n", failing.join(", "))
    }
}

/// Renders the `dudect` / `tacet` constant-time report: pass/fail counts
/// and the worst statistic, flagging new failures against the baseline.
fn render_ct(kind: &str, base: Option<&Json>, cur: &Json) -> String {
    let metric = if kind == "dudect" { "max_t" } else { "leak_prob" };
    let label = if kind == "dudect" {
        "DudeCT (Welch t)"
    } else {
        "Tacet (leak probability)"
    };

    let pass = cur.get("pass_count").and_then(Json::as_f64).unwrap_or(0.0) as u64;
    let fail = cur.get("fail_count").and_then(Json::as_f64).unwrap_or(0.0) as u64;
    let base_fail = base
        .and_then(|b| b.get("fail_count"))
        .and_then(Json::as_f64)
        .map(|f| f as u64);

    let mut worst = 0.0f64;
    let mut fails = Vec::new();
    if let Some(results) = cur.get("results").and_then(Json::as_array) {
        for r in results {
            let v = r.get(metric).and_then(Json::as_f64).unwrap_or(0.0).abs();
            worst = worst.max(v);
            if r.get("status").and_then(Json::as_str) == Some("fail") {
                if let Some(name) = r.get("name").and_then(Json::as_str) {
                    fails.push(format!("`{name}` ({metric} {v:.3})"));
                }
            }
        }
    }

    let icon = if fail == 0 { "✅" } else { "❌" };
    let regressed = match base_fail {
        Some(bf) if fail > bf => format!(" — **{} new vs `main`**", fail - bf),
        _ => String::new(),
    };

    let mut out = format!(
        "### Constant-time: {label}\n\n{icon} **{pass} pass / {fail} fail** \
         (worst {metric} {worst:.3}){regressed}.\n",
    );
    if !fails.is_empty() {
        out.push_str(&format!("\nFailing: {}\n", fails.join(", ")));
    }

    out
}

/// Renders the `mutants` report: mutation kill rate and survivor count.
fn render_mutants(base: Option<&Json>, cur: &Json) -> String {
    let rate = |root: Option<&Json>| -> Option<(u64, u64)> {
        let s = root?.get("summary")?;
        let caught = s.get("caught").and_then(Json::as_f64)? as u64;
        let total = s.get("total").and_then(Json::as_f64)? as u64;
        Some((caught, total))
    };

    let Some((caught, total)) = rate(Some(cur)) else {
        return "### Mutation testing\n\nNo mutants data.\n".to_string();
    };
    let pct = if total > 0 {
        caught as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let missed = total - caught;

    let delta = match rate(base) {
        Some((bc, bt)) if bt > 0 => {
            let bp = bc as f64 / bt as f64 * 100.0;
            format!(" ({:+.1} pp vs `main`)", pct - bp)
        }
        _ => String::new(),
    };

    format!(
        "### Mutation testing\n\n**{pct:.1}% caught** ({caught}/{total}){delta}. \
         {missed} survivor(s).\n"
    )
}

/// Renders the `coverage` report: line coverage percentage and delta.
fn render_coverage(base: Option<&Json>, cur: &Json) -> String {
    let pct = |root: Option<&Json>| -> Option<f64> {
        root?.get("total")?.get("percent").and_then(Json::as_f64)
    };

    let Some(p) = pct(Some(cur)) else {
        return "### Coverage\n\nNo coverage data.\n".to_string();
    };
    let lines = cur
        .get("total")
        .and_then(|t| Some((t.get("covered")?.as_f64()? as u64, t.get("total")?.as_f64()? as u64)));
    let lines_str = lines.map_or_else(String::new, |(c, t)| format!(" ({c}/{t} lines)"));

    let delta = match pct(base) {
        Some(bp) => format!(" ({:+.2} pp vs `main`)", p - bp),
        None => String::new(),
    };

    format!("### Coverage\n\n**{p:.2}%**{lines_str}{delta}.\n")
}

/// Formats a wall-clock delta with a ⚠️/✅ marker at the 15% threshold.
fn wallclock_delta(cur: f64, base: f64) -> String {
    let ratio = cur / base;
    let pct = (ratio - 1.0) * 100.0;
    let mark = if ratio >= WALLCLOCK_FACTOR {
        " ⚠️"
    } else if ratio <= 1.0 / WALLCLOCK_FACTOR {
        " ✅"
    } else {
        ""
    };
    format!("{pct:+.1}%{mark}")
}

/// Formats a signed percentage change of `cur` relative to `base`.
fn signed_pct(cur: f64, base: f64) -> String {
    format!("{:+.2}%", (cur / base - 1.0) * 100.0)
}

/// Formats a nanosecond timing with an appropriate unit and 3 decimals.
fn fmt_ns(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:.3} ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.3} µs", ns / 1_000.0)
    } else {
        format!("{ns:.3} ns")
    }
}

/// Formats an integer with `,` thousands separators.
fn fmt_int(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();

    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }

    out
}

/// A minimal JSON value, enough to read CI payloads without a serde
/// dependency (CI scripts are dependency-free, compiled with `rustc`).
enum Json {
    /// A JSON number.
    Num(f64),
    /// A JSON string.
    Str(String),
    /// A JSON array.
    Arr(Vec<Json>),
    /// A JSON object, as ordered key/value pairs.
    Obj(Vec<(String, Json)>),
    /// `null`, `true`, or `false` (never read individually).
    Other,
}

impl Json {
    /// Parses the file at `path`, or `None` if absent/empty/malformed.
    fn from_file(path: &str) -> Option<Json> {
        let text = fs::read_to_string(path).ok()?;
        let bytes = text.as_bytes();
        let mut pos = 0;

        Json::parse_value(bytes, &mut pos)
    }

    /// Looks up `key` in an object, returning `None` otherwise.
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

    /// Parses one value at `*pos`, advancing past it.
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

    /// Parses an object body starting at `{`.
    fn parse_object(bytes: &[u8], pos: &mut usize) -> Option<Json> {
        *pos += 1;
        let mut pairs = Vec::new();

        loop {
            skip_ws(bytes, pos);
            match bytes.get(*pos)? {
                b'}' => {
                    *pos += 1;
                    return Some(Json::Obj(pairs));
                }
                b',' => *pos += 1,
                b'"' => {
                    let key = parse_string(bytes, pos)?;
                    skip_ws(bytes, pos);
                    if bytes.get(*pos)? != &b':' {
                        return None;
                    }
                    *pos += 1;
                    pairs.push((key, Json::parse_value(bytes, pos)?));
                }
                _ => return None,
            }
        }
    }

    /// Parses an array body starting at `[`.
    fn parse_array(bytes: &[u8], pos: &mut usize) -> Option<Json> {
        *pos += 1;
        let mut items = Vec::new();

        loop {
            skip_ws(bytes, pos);
            match bytes.get(*pos)? {
                b']' => {
                    *pos += 1;
                    return Some(Json::Arr(items));
                }
                b',' => *pos += 1,
                _ => items.push(Json::parse_value(bytes, pos)?),
            }
        }
    }
}

/// Advances `*pos` past JSON whitespace.
fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while bytes.get(*pos).is_some_and(u8::is_ascii_whitespace) {
        *pos += 1;
    }
}

/// Parses a JSON string starting at the opening quote, decoding escapes
/// loosely (enough for keys and ASCII values).
fn parse_string(bytes: &[u8], pos: &mut usize) -> Option<String> {
    *pos += 1;
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
                    *pos += 4;
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

/// Parses a JSON number at `*pos`, advancing past it.
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
