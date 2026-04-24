//! Signal build state (running/done) to the CI dashboard.
//!
//! Usage: ci-signal <kind> <state> <sha>
//!
//! Writes `{"state":"<state>","sha":"<sha>"}` to `/data/<kind>/status.json`
//! on the Fly.io CI site. Used at the start of CI jobs to signal "running"
//! so the dashboard can animate the favicon.
//!
//! Compile: `rustc -O ci-signal.rs -o ci-signal`

use std::env;
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

const APP: &str = "sqisign-selkie-ci";

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: ci-signal <kind> <state> <sha>");
        std::process::exit(1);
    }

    let kind = &args[1];
    let state = &args[2];
    let sha = &args[3];
    let dir = format!("/data/{kind}");

    // Wake the app.
    run("flyctl", &["apps", "restart", APP, "--skip-health-checks"]);
    thread::sleep(Duration::from_secs(5));

    // Write status.
    let status = format!("{{\"state\":{},\"sha\":{}}}", json_str(state), json_str(sha));
    let path = "/tmp/ci-status.json";
    fs::write(path, &status).expect("failed to write status file");

    // Upload.
    ssh_cmd(&format!("rm -f {dir}/status.json"));
    sftp_put(path, &format!("{dir}/status.json"));
    eprintln!("[ci-signal] {kind}: {state} ({sha})");
}

fn run(cmd: &str, args: &[&str]) {
    let status = Command::new(cmd)
        .args(args)
        .status()
        .unwrap_or_else(|e| {
            eprintln!("failed to run {cmd}: {e}");
            std::process::exit(1);
        });
    if !status.success() {
        eprintln!("warning: {cmd} {:?} exited with {status}", args);
    }
}

fn ssh_cmd(cmd: &str) {
    run("flyctl", &["ssh", "console", "-a", APP, "-C", cmd]);
}

fn sftp_put(local: &str, remote: &str) {
    let input = format!("put {local} {remote}\n");
    let mut child = Command::new("flyctl")
        .args(["ssh", "sftp", "shell", "-a", APP])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| {
            eprintln!("sftp spawn failed: {e}");
            std::process::exit(1);
        });
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    if !out.status.success() {
        eprintln!(
            "warning: sftp put {local} → {remote} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
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
