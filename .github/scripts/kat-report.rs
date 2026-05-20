//! Build the dashboard's `kat.json` from the gate's nextest run.
//!
//! Usage: kat-report <sha> <junit.xml>
//!
//! Reads a JUnit XML file produced by nextest (enabled via the
//! `[profile.default.junit]` section in `.config/nextest.toml`),
//! filters per-test results into two suites — `kat` (lib unit tests
//! whose name contains `kat_`) and `wycheproof` (integration tests
//! in the `wycheproof` binary) — and emits the structured JSON the
//! CI dashboard consumes.
//!
//! Replaces the earlier implementation that re-ran `cargo test
//! --lib -- kat_` and `cargo test --test wycheproof` independently
//! of the gate. The gate now runs the full nextest suite under
//! `--cargo-profile release-checked`; this script just shapes its
//! output.
//!
//! Compile: `rustc -O kat-report.rs -o kat-report`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::SystemTime;

#[derive(Clone)]
struct TestResult {
    name: String,
    status: String, // "pass", "fail", "ignored"
    detail: String,
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: kat-report <sha> <junit.xml>");
        std::process::exit(1);
    }
    let sha = &args[1];
    let junit_path = &args[2];

    let junit_xml = fs::read_to_string(junit_path).unwrap_or_else(|e| {
        eprintln!("[kat-report] cannot read {junit_path}: {e}");
        std::process::exit(1);
    });

    let (kat_results, wyche_results) = parse_junit(&junit_xml);

    let kat_vectors = count_kat_vectors();
    let wycheproof_vectors = count_wycheproof_vectors();
    let wycheproof_files = parse_wycheproof_files();

    let all_results: Vec<(&str, &[TestResult], u64)> = vec![
        ("kat", &kat_results[..], kat_vectors),
        ("wycheproof", &wyche_results[..], wycheproof_vectors),
    ];

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;

    // Per-suite summary and results.
    writeln!(w, "  \"suites\": [")?;
    for (si, &(suite_name, results, vectors)) in all_results.iter().enumerate() {
        let pass = results.iter().filter(|r| r.status == "pass").count();
        let fail = results.iter().filter(|r| r.status == "fail").count();
        let ignored = results
            .iter()
            .filter(|r| r.status == "ignored" || r.status == "skip")
            .count();

        writeln!(w, "    {{")?;
        writeln!(w, "      \"name\": {},", json_str(suite_name))?;
        writeln!(w, "      \"pass\": {pass},")?;
        writeln!(w, "      \"fail\": {fail},")?;
        writeln!(w, "      \"ignored\": {ignored},")?;
        writeln!(w, "      \"skip\": {ignored},")?;
        writeln!(w, "      \"total\": {},", results.len())?;
        writeln!(w, "      \"vectors\": {vectors},")?;
        writeln!(w, "      \"tests\": [")?;

        for (i, r) in results.iter().enumerate() {
            write!(
                w,
                "        {{\"name\": {}, \"status\": {}",
                json_str(&r.name),
                json_str(&r.status)
            )?;
            if !r.detail.is_empty() {
                write!(w, ", \"detail\": {}", json_str(&r.detail))?;
            }
            write!(w, "}}")?;
            if i + 1 < results.len() {
                writeln!(w, ",")?;
            } else {
                writeln!(w)?;
            }
        }

        // wycheproof suite gets a vector_files breakdown.
        if suite_name == "wycheproof" && !wycheproof_files.is_empty() {
            writeln!(w, "      ],")?;
            writeln!(w, "      \"vector_files\": [")?;
            for (fi, vf) in wycheproof_files.iter().enumerate() {
                write!(w, "        {vf}")?;
                if fi + 1 < wycheproof_files.len() {
                    writeln!(w, ",")?;
                } else {
                    writeln!(w)?;
                }
            }
            write!(w, "      ]\n    }}")?;
        } else {
            write!(w, "      ]\n    }}")?;
        }
        if si + 1 < all_results.len() {
            writeln!(w, ",")?;
        } else {
            writeln!(w)?;
        }
    }
    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;

    Ok(())
}

/// Parse nextest's JUnit XML into (kat-suite results, wycheproof-suite results).
///
/// nextest emits one `<testcase classname="…" name="…" time="…"/>`
/// per test, optionally wrapping a `<failure>` or `<skipped/>` child
/// for non-passing outcomes. This walks the file line by line; we
/// don't need a full XML parser because the format is regular and
/// each `<testcase>` opens on its own line.
fn parse_junit(xml: &str) -> (Vec<TestResult>, Vec<TestResult>) {
    let mut kat = Vec::new();
    let mut wyche = Vec::new();

    let mut current: Option<(String, String, TestResult)> = None;

    for line in xml.lines() {
        let l = line.trim();

        if l.starts_with("<testcase ") {
            let classname = extract_xml_attr(l, "classname");
            let name = extract_xml_attr(l, "name");
            let r = TestResult {
                name: name.clone(),
                status: "pass".to_string(),
                detail: String::new(),
            };
            if l.ends_with("/>") {
                push_into_suite(classname, name, r, &mut kat, &mut wyche);
            } else {
                current = Some((classname, name, r));
            }
        } else if l.starts_with("<failure") {
            if let Some((_, _, r)) = current.as_mut() {
                r.status = "fail".to_string();
                if r.detail.is_empty() {
                    let msg = extract_xml_attr(l, "message");
                    if !msg.is_empty() {
                        r.detail = msg;
                    }
                }
            }
        } else if l.starts_with("<skipped") {
            if let Some((_, _, r)) = current.as_mut() {
                r.status = "ignored".to_string();
            }
        } else if l.starts_with("</testcase>") {
            if let Some((classname, name, r)) = current.take() {
                push_into_suite(classname, name, r, &mut kat, &mut wyche);
            }
        }
    }

    (kat, wyche)
}

/// Routes a test result into the kat or wycheproof bucket based on
/// classname/name. Tests that match neither (most of the lib + the
/// other integration tests) are dropped — the gate is the source of
/// truth for whether they passed; kat.json is just a per-suite view
/// for the dashboard.
fn push_into_suite(
    classname: String,
    name: String,
    r: TestResult,
    kat: &mut Vec<TestResult>,
    wyche: &mut Vec<TestResult>,
) {
    // nextest's classname for integration tests typically encodes
    // the binary name (e.g. ends with `::wycheproof`). For lib unit
    // tests, classname is `<crate>::<module path>`.
    if classname.ends_with("::wycheproof") || classname == "wycheproof" {
        wyche.push(r);
    } else if name.contains("kat_") {
        kat.push(r);
    }
}

/// Pulls an XML attribute value out of a single `<tag …>` line.
/// Doesn't handle escaped quotes inside attribute values — fine for
/// nextest's output, which doesn't emit those for our test names.
fn extract_xml_attr(line: &str, key: &str) -> String {
    let needle = format!(" {key}=\"");
    let Some(start) = line.find(&needle) else {
        return String::new();
    };
    let rest = &line[start + needle.len()..];
    let end = rest.find('"').unwrap_or(rest.len());
    rest[..end].to_string()
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
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn iso8601_now() -> String {
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Quick ISO 8601 formatter without chrono.
    let (y, mo, d, h, mi, s) = epoch_to_ymdhms(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

fn epoch_to_ymdhms(secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let mut days = secs / 86400;

    let mut year = 1970u64;
    loop {
        let dy = if is_leap(year) { 366 } else { 365 };
        if days < dy {
            break;
        }
        days -= dy;
        year += 1;
    }
    let mdays: [u64; 12] = [31, if is_leap(year) { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 0u64;
    while month < 12 && days >= mdays[month as usize] {
        days -= mdays[month as usize];
        month += 1;
    }
    (year, month + 1, days + 1, h, m, s)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Counts individual KAT vectors from `src/keys/kat_data.rs`.
/// Each vector is a tuple starting with `(` on its own indented line.
fn count_kat_vectors() -> u64 {
    let content = fs::read_to_string("src/keys/kat_data.rs").unwrap_or_default();
    content
        .lines()
        .filter(|line| {
            let t = line.trim();
            t == "("
        })
        .count() as u64
}

/// Counts individual Wycheproof test vectors by summing
/// `numberOfTests` across all JSON files in `tests/vectors/`.
fn count_wycheproof_vectors() -> u64 {
    let mut total = 0u64;
    let dir = match fs::read_dir("tests/vectors") {
        Ok(d) => d,
        Err(_) => return 0,
    };
    for entry in dir.flatten() {
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "json") {
            continue;
        }
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        total += extract_num_u64(&content, "numberOfTests");
    }
    total
}

/// Parses each wycheproof JSON file into a compact JSON object for
/// the dashboard: file name, algorithm, per-vector tcId/comment/result.
fn parse_wycheproof_files() -> Vec<String> {
    let dir = match fs::read_dir("tests/vectors") {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let mut files = Vec::new();
    for entry in dir.flatten() {
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "json") {
            continue;
        }
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let filename = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        let algorithm = extract_string_val(&content, "algorithm");
        let num_tests = extract_num_u64(&content, "numberOfTests");

        // Extract vectors: walk testGroups[].tests[].
        let mut vectors = Vec::new();
        let mut pos = 0;
        while let Some(tc_start) = content[pos..].find("\"tcId\"") {
            let abs = pos + tc_start;
            let tc_id = extract_num_u64(&content[abs..], "tcId");
            let comment = extract_string_val(&content[abs..], "comment");
            let result = extract_string_val(&content[abs..], "result");
            vectors.push(format!(
                "{{\"tcId\":{},\"comment\":{},\"result\":{}}}",
                tc_id,
                json_str(&comment),
                json_str(&result)
            ));
            // Advance past this tcId to find the next one.
            pos = abs + 6;
        }

        // Count by result type.
        let valid = vectors.iter().filter(|v| v.contains("\"valid\"")).count();
        let invalid = vectors.iter().filter(|v| v.contains("\"invalid\"")).count();

        let mut out = String::from("{\n");
        out.push_str(&format!("          \"file\": {},\n", json_str(&filename)));
        out.push_str(&format!(
            "          \"algorithm\": {},\n",
            json_str(&algorithm)
        ));
        out.push_str(&format!("          \"total\": {num_tests},\n"));
        out.push_str(&format!("          \"valid\": {valid},\n"));
        out.push_str(&format!("          \"invalid\": {invalid},\n"));
        out.push_str("          \"vectors\": [\n");
        for (i, v) in vectors.iter().enumerate() {
            out.push_str("            ");
            out.push_str(v);
            if i + 1 < vectors.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("          ]\n        }");
        files.push(out);
    }
    files
}

/// Extracts a JSON string value for a key (simple, non-nested).
fn extract_string_val(json: &str, key: &str) -> String {
    let needle = format!("\"{key}\"");
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
    let mut end = 1;
    let bytes = after.as_bytes();
    while end < bytes.len() {
        if bytes[end] == b'"' && bytes[end - 1] != b'\\' {
            break;
        }
        end += 1;
    }
    after[1..end].to_string()
}

fn extract_num_u64(json: &str, key: &str) -> u64 {
    let needle = format!("\"{key}\"");
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

// Self-test against a baked-in JUnit fixture. Build and run with
// `rustc --test -O kat-report.rs && ./kat-report`. Fixture matches
// nextest 0.9.133's actual emitted format (the version pinned in
// ci.yml's gate). If a future nextest changes the JUnit shape
// enough that the parser misroutes tests, this test fails before
// the parser's misbehavior reaches the dashboard.
#[cfg(test)]
mod tests {
    use super::*;

    /// Fixture: two passing kat lib tests, one failing kat lib test,
    /// one ignored kat lib test, two wycheproof integration tests
    /// (one pass, one failing), and a non-kat lib test that should
    /// route into neither suite.
    const FIXTURE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="6" failures="2" errors="0" uuid="aaa" timestamp="2026-05-19T23:08:50.565-04:00" time="0.083">
    <testsuite name="sqisign-selkie" tests="4" disabled="0" errors="0" failures="1">
        <testcase name="keys::tests::verify_kat_001" classname="sqisign-selkie" timestamp="2026-05-19T23:08:50.565-04:00" time="0.080">
        </testcase>
        <testcase name="keys::tests::keygen_kat_005" classname="sqisign-selkie" timestamp="2026-05-19T23:08:50.565-04:00" time="0.080">
        </testcase>
        <testcase name="keys::signing::tests::sign_kat_derand_042" classname="sqisign-selkie" timestamp="2026-05-19T23:08:50.565-04:00" time="0.080">
            <failure type="test failure" message="assertion failed: foo">trace text</failure>
        </testcase>
        <testcase name="keys::tests::keygen_kat_007" classname="sqisign-selkie" timestamp="2026-05-19T23:08:50.565-04:00" time="0">
            <skipped/>
        </testcase>
        <testcase name="curves::tests::ladder_smoke" classname="sqisign-selkie" timestamp="2026-05-19T23:08:50.565-04:00" time="0.001">
        </testcase>
    </testsuite>
    <testsuite name="sqisign-selkie::wycheproof" tests="2" disabled="0" errors="0" failures="1">
        <testcase name="sqisign_sign_vectors" classname="sqisign-selkie::wycheproof" timestamp="2026-05-19T23:08:50.565-04:00" time="0.072">
        </testcase>
        <testcase name="sqisign_verify_vectors" classname="sqisign-selkie::wycheproof" timestamp="2026-05-19T23:08:50.565-04:00" time="0.072">
            <failure type="test failure" message="bad vector">trace</failure>
        </testcase>
    </testsuite>
</testsuites>
"#;

    #[test]
    fn parse_junit_routes_and_statuses() {
        let (kat, wyche) = parse_junit(FIXTURE);

        // Routing: kat suite gets the four `kat_`-named lib tests
        // (classname="sqisign-selkie"); wycheproof suite gets the
        // integration binary's tests (classname ends with
        // "::wycheproof"); the non-kat lib test (`ladder_smoke`)
        // appears in neither.
        let kat_names: Vec<&str> = kat.iter().map(|t| t.name.as_str()).collect();
        let wyche_names: Vec<&str> = wyche.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(kat_names.len(), 4, "kat suite test count");
        assert_eq!(
            wyche_names,
            vec!["sqisign_sign_vectors", "sqisign_verify_vectors"],
            "wycheproof suite tests + order"
        );
        assert!(!kat_names.iter().any(|n| *n == "curves::tests::ladder_smoke"));
        assert!(!wyche_names.iter().any(|n| *n == "curves::tests::ladder_smoke"));

        // Status detection from `<failure …>` / `<skipped/>` children.
        assert_eq!(kat.iter().filter(|t| t.status == "pass").count(), 2);
        assert_eq!(kat.iter().filter(|t| t.status == "fail").count(), 1);
        assert_eq!(kat.iter().filter(|t| t.status == "ignored").count(), 1);
        assert_eq!(wyche.iter().filter(|t| t.status == "fail").count(), 1);

        // Failure message captured from the `message=` attribute.
        let failing = kat.iter().find(|t| t.status == "fail").unwrap();
        assert!(
            failing.detail.contains("assertion failed: foo"),
            "expected failure message captured, got {:?}",
            failing.detail
        );
    }
}
