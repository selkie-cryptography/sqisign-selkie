//! Upload CI data (coverage, bench, mutants, dudect) to the Fly.io CI site.
//!
//! Usage: ci-upload <coverage|bench|mutants|dudect> <json-file> <sha>
//!
//! - Stores per-commit data at /data/<kind>/<sha>.json
//! - Updates /data/<kind>/latest.json
//! - Maintains /data/<kind>/index.json (last 50 summaries)
//! - Prunes per-commit files beyond 30 entries
//! - Writes status.json before/after for the site's indicator
//!
//! Compile: `rustc -O ci-upload.rs -o ci-upload`

use std::env;
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

const APP: &str = "sqisign-selkie-ci";
const SITE: &str = "https://sqisign-selkie-ci.fly.dev";
const MAX_INDEX: usize = 50;
const MAX_FILES: usize = 30;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: ci-upload <coverage|bench|mutants|dudect> <json-file> <sha>");
        std::process::exit(1);
    }

    let kind = &args[1];
    let json_path = &args[2];
    let sha = &args[3];
    let dir = format!("/data/{kind}");

    // Read the JSON to extract fields for the index.
    let json_contents = fs::read_to_string(json_path)
        .unwrap_or_else(|e| { eprintln!("cannot read {json_path}: {e}"); std::process::exit(1); });

    // Wake the app.
    eprintln!("[ci-upload] restarting {APP}...");
    run("flyctl", &["apps", "restart", APP, "--skip-health-checks"]);
    thread::sleep(Duration::from_secs(5));

    // Signal "running".
    let status_running = format!("{{\"state\":\"running\",\"sha\":{}}}", json_str(sha));
    write_tmp("status.json", &status_running);
    sftp_put("/tmp/status.json", &format!("{dir}/status.json"));
    eprintln!("[ci-upload] signaled running");

    // Upload per-commit and latest.
    ssh_cmd(&format!("rm -f {dir}/latest.json {dir}/{sha}.json"));
    sftp_put(json_path, &format!("{dir}/{sha}.json"));
    sftp_put(json_path, &format!("{dir}/latest.json"));
    eprintln!("[ci-upload] uploaded {sha}.json and latest.json");

    // Update index.
    let existing_index = fetch_url(&format!("{SITE}/{kind}/index.json"))
        .unwrap_or_else(|| "[]".to_string());
    let new_index = build_index(kind, sha, &json_contents, &existing_index);
    write_tmp("index-new.json", &new_index);
    ssh_cmd(&format!("rm -f {dir}/index.json"));
    sftp_put("/tmp/index-new.json", &format!("{dir}/index.json"));
    eprintln!("[ci-upload] updated index");

    // Prune old per-commit files.
    let keep_shas = extract_index_shas(&new_index, MAX_FILES);
    let files = ssh_ls(&dir);
    let mut pruned = 0;
    for f in &files {
        if !f.ends_with(".json") { continue }
        match f.as_str() {
            "latest.json" | "index.json" | "status.json" => continue,
            _ => {}
        }
        let file_sha = f.trim_end_matches(".json");
        if !keep_shas.contains(&file_sha.to_string()) {
            ssh_cmd(&format!("rm -f {dir}/{f}"));
            pruned += 1;
        }
    }
    if pruned > 0 { eprintln!("[ci-upload] pruned {pruned} old files"); }

    // Signal "done".
    let status_done = format!("{{\"state\":\"done\",\"sha\":{}}}", json_str(sha));
    write_tmp("status.json", &status_done);
    ssh_cmd(&format!("rm -f {dir}/status.json"));
    sftp_put("/tmp/status.json", &format!("{dir}/status.json"));
    eprintln!("[ci-upload] done");

    // Update the global manifest so the dashboard can poll one file.
    update_manifest(kind, sha);
}

/// Updates `/data/manifest.json` — a single object mapping each data
/// kind to its latest SHA and timestamp. The dashboard polls this one
/// file instead of 20+ individual files.
fn update_manifest(kind: &str, sha: &str) {
    let manifest_url = format!("{SITE}/manifest.json");
    let existing = fetch_url(&manifest_url).unwrap_or_else(|| "{}".to_string());

    // Parse existing entries (simple key extraction).
    let all_kinds = [
        "coverage", "bench", "mutants", "dudect", "tacet", "deny",
        "unsafe", "size", "docs", "msrv", "panic", "fuzz", "iai",
        "alloc", "platform", "stack", "ctgrind", "api", "kat", "zeroize",
    ];

    let mut entries = Vec::new();
    for k in &all_kinds {
        if *k == kind {
            // Replace with the new SHA.
            entries.push(format!("  {}: {}", json_str(k), json_str(sha)));
        } else {
            // Preserve the existing value.
            let existing_sha = extract_string(&existing, k);
            if !existing_sha.is_empty() {
                entries.push(format!("  {}: {}", json_str(k), json_str(&existing_sha)));
            }
        }
    }

    let manifest = format!("{{\n{}\n}}", entries.join(",\n"));
    write_tmp("manifest.json", &manifest);
    ssh_cmd("rm -f /data/manifest.json");
    sftp_put("/tmp/manifest.json", "/data/manifest.json");
    eprintln!("[ci-upload] updated manifest");
}

/// Build a new index JSON array by prepending this commit's entry
/// and capping at MAX_INDEX entries.
fn build_index(kind: &str, sha: &str, json: &str, existing: &str) -> String {
    let updated_at = extract_string(json, "updated_at");

    // Build the new entry based on kind.
    let entry = match kind {
        "coverage" => {
            let pct = extract_num_f64(json, "percent");
            format!("{{\"sha\":{},\"percent\":{:.4},\"updated_at\":{}}}",
                json_str(sha), pct, json_str(&updated_at))
        }
        "mutants" => {
            let caught = extract_num_in_section(json, "summary", "caught");
            let missed = extract_num_in_section(json, "summary", "missed");
            let timeout = extract_num_in_section(json, "summary", "timeout");
            format!("{{\"sha\":{},\"caught\":{},\"missed\":{},\"timeout\":{},\"updated_at\":{}}}",
                json_str(sha), caught, missed, timeout, json_str(&updated_at))
        }
        "dudect" => {
            let pass = extract_num_u64(json, "pass_count");
            let fail = extract_num_u64(json, "fail_count");
            format!("{{\"sha\":{},\"pass_count\":{},\"fail_count\":{},\"updated_at\":{}}}",
                json_str(sha), pass, fail, json_str(&updated_at))
        }
        _ => {
            format!("{{\"sha\":{},\"updated_at\":{}}}", json_str(sha), json_str(&updated_at))
        }
    };

    // Parse existing entries (just extract sha values and raw objects).
    let mut entries = Vec::new();
    entries.push(entry);

    // Collect existing entries, skipping any with the same sha.
    for chunk in existing.split('{').skip(1) {
        let obj = format!("{{{chunk}");
        let obj_sha = extract_string(&obj, "sha");
        if obj_sha == sha || obj_sha.is_empty() { continue }
        // Find the closing brace for this object.
        if let Some(end) = obj.find('}') {
            entries.push(obj[..end + 1].to_string());
        }
    }

    // Cap at MAX_INDEX.
    entries.truncate(MAX_INDEX);

    // Serialize.
    let mut out = String::from("[\n");
    for (i, e) in entries.iter().enumerate() {
        out.push_str("  ");
        out.push_str(e);
        if i + 1 < entries.len() { out.push(',') }
        out.push('\n');
    }
    out.push(']');
    out
}

/// Extract sha values from the first `n` index entries.
fn extract_index_shas(index_json: &str, n: usize) -> Vec<String> {
    let mut shas = Vec::new();
    for chunk in index_json.split('{').skip(1) {
        let obj = format!("{{{chunk}");
        let s = extract_string(&obj, "sha");
        if !s.is_empty() { shas.push(s) }
        if shas.len() >= n { break }
    }
    shas
}

// --- Shell helpers ---

fn run(cmd: &str, args: &[&str]) {
    let status = Command::new(cmd).args(args)
        .status()
        .unwrap_or_else(|e| { eprintln!("failed to run {cmd}: {e}"); std::process::exit(1); });
    if !status.success() {
        eprintln!("warning: {cmd} {:?} exited with {status}", args);
    }
}

fn ssh_cmd(cmd: &str) {
    run("flyctl", &["ssh", "console", "-a", APP, "-C", cmd]);
}

fn ssh_ls(dir: &str) -> Vec<String> {
    let output = Command::new("flyctl")
        .args(["ssh", "console", "-a", APP, "-C", &format!("ls {dir}")])
        .stdout(Stdio::piped()).stderr(Stdio::piped())
        .output().unwrap_or_else(|e| { eprintln!("ssh ls failed: {e}"); std::process::exit(1); });
    String::from_utf8_lossy(&output.stdout)
        .lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
}

fn sftp_put(local: &str, remote: &str) {
    let input = format!("put {local} {remote}\n");
    let mut child = Command::new("flyctl")
        .args(["ssh", "sftp", "shell", "-a", APP])
        .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped())
        .spawn().unwrap_or_else(|e| { eprintln!("sftp spawn failed: {e}"); std::process::exit(1); });
    child.stdin.as_mut().unwrap().write_all(input.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    if !out.status.success() {
        eprintln!("warning: sftp put {local} → {remote} failed: {}",
            String::from_utf8_lossy(&out.stderr));
    }
}

fn fetch_url(url: &str) -> Option<String> {
    let output = Command::new("curl").args(["-sf", url])
        .stdout(Stdio::piped()).stderr(Stdio::null())
        .output().ok()?;
    if output.status.success() { Some(String::from_utf8_lossy(&output.stdout).to_string()) }
    else { None }
}

fn write_tmp(name: &str, contents: &str) {
    let path = format!("/tmp/{name}");
    fs::write(&path, contents).unwrap_or_else(|e| { eprintln!("write {path}: {e}"); std::process::exit(1); });
}

// --- JSON helpers ---

fn extract_string(json: &str, key: &str) -> String {
    let needle = format!("\"{}\"", key);
    let Some(idx) = json.find(&needle) else { return String::new() };
    let rest = &json[idx + needle.len()..];
    let Some(colon) = rest.find(':') else { return String::new() };
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('"') { return String::new() }
    let mut end = 1;
    let bytes = after.as_bytes();
    while end < bytes.len() {
        if bytes[end] == b'"' && bytes[end - 1] != b'\\' { break }
        end += 1;
    }
    after[1..end].to_string()
}

fn extract_num_u64(json: &str, key: &str) -> u64 {
    let needle = format!("\"{}\"", key);
    let Some(idx) = json.find(&needle) else { return 0 };
    let rest = &json[idx + needle.len()..];
    let Some(colon) = rest.find(':') else { return 0 };
    let after = rest[colon + 1..].trim_start();
    let end = after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len());
    after[..end].parse().unwrap_or(0)
}

fn extract_num_f64(json: &str, key: &str) -> f64 {
    let needle = format!("\"{}\"", key);
    let Some(idx) = json.find(&needle) else { return 0.0 };
    let rest = &json[idx + needle.len()..];
    let Some(colon) = rest.find(':') else { return 0.0 };
    let after = rest[colon + 1..].trim_start();
    let end = after.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(after.len());
    after[..end].parse().unwrap_or(0.0)
}

fn extract_num_in_section(json: &str, section: &str, key: &str) -> u64 {
    let needle = format!("\"{}\"", section);
    let Some(idx) = json.find(&needle) else { return 0 };
    let rest = &json[idx..];
    let Some(brace) = rest.find('{') else { return 0 };
    let Some(end) = rest[brace..].find('}') else { return 0 };
    extract_num_u64(&rest[brace..brace + end], key)
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
