//! Render a Markdown CI report for one data kind, for a PR comment and
//! job summary.
//!
//! Usage: `ci-report <kind> <baseline.json> <current.json>` (writes
//! Markdown to stdout). Either file may be missing or empty — a missing
//! baseline yields a current-only report rather than an error.
//!
//! `kind` is one of `bench`, `instructions`, `kat`, `dudect`, `tacet`, `mutants`,
//! `coverage` — the same kinds the dashboard stores per commit. The first
//! output line is a hidden marker (`<!-- ci-report:<kind> -->`) so the
//! comment poster can update its own previous comment in place, and the
//! footer deep-links the dashboard at the baseline commit.
//!
//! Compile: `rustc -O ci-report.rs -o ci-report`

use std::{collections::BTreeMap, env, fs};

/// Dashboard / data host (same Fly app serves the static site and JSON).
const SITE: &str = "https://sqisign-selkie-ci.fly.dev";

/// Wall-clock regression threshold for the ⚠️ marker (matches the
/// historical 115% alert). Wall-clock noise means smaller deltas on
/// `bench` are not signal; `instructions` counts are deterministic and
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
        "instructions" => {
            let alloc_pr = args.get(4).and_then(|p| Json::from_file(p));
            let alloc_latest = args.get(5).and_then(|p| Json::from_file(p));
            let stack_pr = args.get(6).and_then(|p| Json::from_file(p));
            let stack_latest = args.get(7).and_then(|p| Json::from_file(p));
            render_instructions(
                base,
                cur,
                alloc_pr.as_ref(),
                alloc_latest.as_ref(),
                stack_pr.as_ref(),
                stack_latest.as_ref(),
            )
        }
        "kat" => render_kat(base, cur),
        "dudect" | "tacet" => render_ct(kind, base, cur),
        "mutants" => render_mutants(base, cur),
        "coverage" => render_coverage(base, cur),
        _ => format!("_No report renderer for `{kind}`._\n"),
    };

    // Deep-link the dashboard at a specific commit, jumping to this kind's
    // section. Prefer the main baseline (which has uploaded data); fall back
    // to the commit this run reported (e.g. a PR head, or a kind with no main
    // baseline yet) so the link is always sha-specific. dudect/tacet share
    // the constant-time section.
    let anchor = match kind {
        "bench" => "#bench-section",
        "instructions" => "#instructions-section",
        "kat" => "#kat-section",
        "dudect" | "tacet" => "#dudect-section",
        "mutants" => "#mutants-section",
        "coverage" => "#coverage-section",
        _ => "",
    };
    let link_sha = base
        .and_then(|b| b.get("sha"))
        .and_then(Json::as_str)
        .map(|sha| ("main @ ", sha))
        .or_else(|| cur.get("sha").and_then(Json::as_str).map(|sha| ("", sha)));
    let footer = match link_sha {
        Some((label, sha)) => format!(
            "\n[📊 Full dashboard ({label}{})]({SITE}/?sha={sha}{anchor})\n",
            &sha[..sha.len().min(7)]
        ),
        None => format!("\n[📊 Full CI dashboard]({SITE}/{anchor})\n"),
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
         instruction-count gating (instructions) is separate._\n\n",
    );

    if cur_map.is_empty() {
        return out + "No benchmark data.\n";
    }
    if base_map.is_empty() {
        out.push_str("No baseline on `main` yet — showing this PR's numbers only.\n\n");
    }

    // Top-level library benches (the `sqisign` group — keygen/sign/verify)
    // are the headline numbers, so list them first; the rest alphabetical.
    let mut entries: Vec<_> = cur_map.iter().collect();
    entries.sort_by(|(a, _), (b, _)| {
        let sqi = |n: &String| n.starts_with("sqisign::");
        sqi(b).cmp(&sqi(a)).then_with(|| a.cmp(b))
    });

    // Build the table body, counting regressions for the visible summary.
    let mut rows = String::new();
    let mut regressions = 0usize;

    for (name, (median, spread, samples)) in entries {
        let spread_str = spread.map_or_else(String::new, |h| format!(" ± {}", fmt_ns(h)));
        let samples_str = samples.map_or_else(|| "—".to_string(), |s| s.to_string());

        let (base_cell, delta) = match base_map.get(name) {
            Some((b, ..)) if *b > 0.0 => {
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

    // Headline Δ% for the public benches as a ```diff block: GitHub
    // colors `-` rows red (slower) and `+` rows green (faster), with a
    // █ bar proportional to the magnitude. Renders everywhere; no Mermaid.
    let mut diff_rows = String::new();
    for b in ["keygen", "sign", "verify"] {
        let key = format!("sqisign::{b}");
        if let (Some((cur, ..)), Some((base, ..))) = (cur_map.get(&key), base_map.get(&key)) {
            if *base > 0.0 {
                let pct = (*cur / *base - 1.0) * 100.0;
                let bar = "█".repeat(((pct.abs() / 3.0).ceil() as usize).clamp(1, 8));
                let prefix = if pct > 0.5 {
                    "-"
                } else if pct < -0.5 {
                    "+"
                } else {
                    " "
                };
                let pct_str = format!("{pct:+.1}%");
                diff_rows.push_str(&format!("{prefix} {b:<7} {pct_str:>6}  {bar}\n"));
            }
        }
    }
    if !diff_rows.is_empty() {
        out.push_str("```diff\n@@ sqisign median vs main  (- slower / + faster) @@\n");
        out.push_str(&diff_rows);
        out.push_str("```\n\n");
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

/// Renders the `instructions` report: deterministic instruction-count deltas.
/// Any increase is real signal (no runner noise), so the threshold is tight.
fn render_instructions(
    base: Option<&Json>,
    cur: &Json,
    alloc_pr: Option<&Json>,
    alloc_latest: Option<&Json>,
    stack_pr: Option<&Json>,
    stack_latest: Option<&Json>,
) -> String {
    let base_map = instructions_map(base);
    let cur_map = instructions_map(Some(cur));

    let mut out = String::from("### Instruction counts (PR vs `main`)\n\n");
    out.push_str(
        "_Deterministic instruction counts via \
         [gungraun](https://github.com/gungraun/gungraun) \
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

    // Headline Δ% for the sqisign top-level ops as a ```diff block, mirroring
    // the bench section: GitHub colors `-` rows red (more instructions =
    // slower) and `+` rows green (fewer = faster), with a █ bar proportional
    // to magnitude. Threshold matches bench (0.5%) for visual consistency.
    let mut diff_rows = String::new();
    for b in ["keygen", "sign", "verify"] {
        let key = format!("sqisign::kat_{b}");
        if let (Some(&cur), Some(&base)) = (cur_map.get(&key), base_map.get(&key)) {
            if base > 0 {
                let pct = (cur as f64 / base as f64 - 1.0) * 100.0;
                let bar = "█".repeat(((pct.abs() / 3.0).ceil() as usize).clamp(1, 8));
                let prefix = if pct > 0.5 {
                    "-"
                } else if pct < -0.5 {
                    "+"
                } else {
                    " "
                };
                let pct_str = format!("{pct:+.1}%");
                diff_rows.push_str(&format!("{prefix} {b:<7} {pct_str:>6}  {bar}\n"));
            }
        }
    }
    if !diff_rows.is_empty() {
        out.push_str("```diff\n@@ sqisign instructions vs main  (- slower / + faster) @@\n");
        out.push_str(&diff_rows);
        out.push_str("```\n\n");
    }

    if let Some(line) = resources_block(alloc_pr, alloc_latest, stack_pr, stack_latest) {
        out.push_str(&line);
        out.push_str("\n\n");
    }

    out.push_str(&format!(
        "<details><summary>{} benchmarks</summary>\n\n\
         | Benchmark | `main` | PR | Δ |\n|---|--:|--:|--:|\n{rows}\n</details>\n",
        cur_map.len()
    ));

    let flamegraphs = instructions_flamegraphs(Some(cur));
    if !flamegraphs.is_empty() {
        out.push_str("\n#### Flamegraphs (`Ir`)\n\n");

        // Embed the heaviest benchmark's flamegraph inline (most interesting
        // call tree); link the rest. SVGs are hosted on the CI site and
        // render via GitHub's image proxy, like SVG badges.
        let featured = cur_map
            .iter()
            .filter(|(name, _)| flamegraphs.contains_key(*name))
            .max_by_key(|(_, &ir)| ir)
            .map(|(name, _)| name.clone())
            .or_else(|| flamegraphs.keys().next().cloned());

        if let Some(url) = featured.as_ref().and_then(|name| flamegraphs.get(name)) {
            let name = featured.as_deref().unwrap_or("");
            out.push_str(&format!("**`{name}`**\n\n![{name} flamegraph]({url})\n\n"));
        }

        out.push_str("<details><summary>All flamegraphs</summary>\n\n");
        for (name, url) in &flamegraphs {
            out.push_str(&format!("- [`{name}`]({url})\n"));
        }
        out.push_str("\n</details>\n");
    }

    out
}

/// Builds a `name -> instructions` map from an `instructions` payload.
fn instructions_map(root: Option<&Json>) -> BTreeMap<String, u64> {
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

/// Builds a `name -> flamegraph URL` map from an `instructions` payload.
fn instructions_flamegraphs(root: Option<&Json>) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();

    let Some(results) = root.and_then(|r| r.get("results")).and_then(Json::as_array) else {
        return map;
    };

    for r in results {
        if let (Some(name), Some(url)) = (
            r.get("name").and_then(Json::as_str),
            r.get("flamegraph").and_then(Json::as_str),
        ) {
            map.insert(name.to_string(), url.to_string());
        }
    }

    map
}

/// Builds a one-line **Resources** block summarising peak alloc per top-level
/// op (sign/keygen/verify) and peak stack, with PR-vs-main deltas when both
/// values exist. Falls back to main's latest as bare context if the PR didn't
/// run perf-metrics. Returns `None` when no resource data is present at all.
fn resources_block(
    alloc_pr: Option<&Json>,
    alloc_latest: Option<&Json>,
    stack_pr: Option<&Json>,
    stack_latest: Option<&Json>,
) -> Option<String> {
    fn op_bytes(root: Option<&Json>, name: &str) -> Option<u64> {
        let ops = root?.get("operations")?.as_array()?;
        for o in ops {
            if o.get("name").and_then(Json::as_str) == Some(name) {
                return o.get("bytes").and_then(Json::as_f64).map(|f| f as u64);
            }
        }
        None
    }
    fn stack_bytes(root: Option<&Json>) -> Option<u64> {
        root?.get("peak_stack_bytes")
            .and_then(Json::as_f64)
            .map(|f| f as u64)
    }
    fn delta(pr: Option<u64>, main: Option<u64>) -> String {
        match (pr, main) {
            (Some(p), Some(m)) if p != m => {
                let d = p as i64 - m as i64;
                let sign = if d > 0 { "+" } else { "-" };
                format!(" ({sign}{})", fmt_bytes(d.unsigned_abs()))
            }
            _ => String::new(),
        }
    }
    // Prefer the PR value; fall back to main's latest as context when absent.
    fn pick(pr: Option<u64>, main: Option<u64>) -> Option<(u64, String)> {
        pr.or(main).map(|b| (b, delta(pr, main)))
    }

    let ops: Vec<(&str, Option<(u64, String)>)> = ["sign", "keygen", "verify"]
        .iter()
        .map(|n| (*n, pick(op_bytes(alloc_pr, n), op_bytes(alloc_latest, n))))
        .collect();
    let stack = pick(stack_bytes(stack_pr), stack_bytes(stack_latest));

    let any_op = ops.iter().any(|(_, v)| v.is_some());
    if !any_op && stack.is_none() {
        return None;
    }

    let mut tokens: Vec<String> = Vec::new();
    for (name, v) in ops {
        if let Some((b, d)) = v {
            tokens.push(format!("{name} {}{d}", fmt_bytes(b)));
        }
    }
    if let Some((b, d)) = stack {
        tokens.push(format!("stack {}{d}", fmt_bytes(b)));
    }

    Some(format!("**Resources** · {}", tokens.join(" · ")))
}

/// Humanizes a byte count as `X.Y MB` / `X.Y KB` / `N B`.
fn fmt_bytes(b: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    let bf = b as f64;
    if bf >= MB {
        format!("{:.1} MB", bf / MB)
    } else if bf >= KB {
        format!("{:.1} KB", bf / KB)
    } else {
        format!("{b} B")
    }
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
    let metric = if kind == "dudect" {
        "max_t"
    } else {
        "leak_prob"
    };
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
    let lines = cur.get("total").and_then(|t| {
        Some((
            t.get("covered")?.as_f64()? as u64,
            t.get("total")?.as_f64()? as u64,
        ))
    });
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
