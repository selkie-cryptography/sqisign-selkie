//! Stage CI intermediate artifacts (fuzz shard results, mutants shard
//! outcomes) in the Tigris bucket used as cross-job scratch storage.
//!
//! Usage:
//!   ci-scratch put   <kind> <sha> <key> <local-file>
//!   ci-scratch get   <kind> <sha> <out-dir>
//!   ci-scratch clean <kind> <sha>
//!
//! - `put`   uploads <local-file> to s3://$BUCKET/<kind>/<sha>/<key>.json.
//! - `get`   downloads every object under s3://$BUCKET/<kind>/<sha>/ into
//!           <out-dir>, flattening to <out-dir>/<key>.json (filenames are
//!           preserved verbatim from the bucket).
//! - `clean` removes every object under s3://$BUCKET/<kind>/<sha>/.
//!
//! Reads credentials from env:
//!   TIGRIS_ACCESS_KEY_ID, TIGRIS_SECRET_ACCESS_KEY.
//! The bucket name is the `BUCKET` const below — config, not a secret.
//!
//! A bucket lifecycle rule auto-expires objects after 7 days, so a
//! failed `clean` doesn't leak indefinitely — the rule is the durable
//! safety net; per-job cleanup is just hygiene.
//!
//! Companion to ci-upload.rs, which publishes final dashboard payloads.
//! Keep the two responsibilities split: ci-upload writes durable
//! /<kind>/<sha>.json + index/manifest into the dashboard volume;
//! ci-scratch shuttles ephemeral inter-job fragments via Tigris.
//!
//! Compile: `rustc -O ci-scratch.rs -o ci-scratch`

use std::env;
use std::process::{Command, Stdio};

const BUCKET: &str = "selkie-ci-scratch";
const ENDPOINT: &str = "https://fly.storage.tigris.dev";

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("put") if args.len() == 6 => put(&args[2], &args[3], &args[4], &args[5]),
        Some("get") if args.len() == 5 => get(&args[2], &args[3], &args[4]),
        Some("clean") if args.len() == 4 => clean(&args[2], &args[3]),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  \
         ci-scratch put   <kind> <sha> <key> <local-file>\n  \
         ci-scratch get   <kind> <sha> <out-dir>\n  \
         ci-scratch clean <kind> <sha>"
    );
    std::process::exit(1);
}

fn put(kind: &str, sha: &str, key: &str, local: &str) {
    let dst = format!("s3://{BUCKET}/{kind}/{sha}/{key}.json");
    eprintln!("[ci-scratch] put {local} → {dst}");
    aws(&["s3", "cp", local, &dst]);
}

fn get(kind: &str, sha: &str, out_dir: &str) {
    let prefix = format!("s3://{BUCKET}/{kind}/{sha}/");
    eprintln!("[ci-scratch] get {prefix} → {out_dir}/");
    std::fs::create_dir_all(out_dir).unwrap_or_else(|e| {
        eprintln!("::error::ci-scratch: cannot create {out_dir}: {e}");
        std::process::exit(1);
    });
    aws(&["s3", "cp", "--recursive", &prefix, out_dir]);
}

fn clean(kind: &str, sha: &str) {
    let prefix = format!("s3://{BUCKET}/{kind}/{sha}/");
    eprintln!("[ci-scratch] clean {prefix}");
    aws(&["s3", "rm", "--recursive", &prefix]);
}

/// Run `aws --endpoint-url <ENDPOINT> <args>`, mapping `TIGRIS_*`
/// credentials into the `AWS_*` names awscli expects. Inherits
/// stdout/stderr so aws's own diagnostics surface in the workflow log.
fn aws(args: &[&str]) {
    let access_key = require_env("TIGRIS_ACCESS_KEY_ID");
    let secret_key = require_env("TIGRIS_SECRET_ACCESS_KEY");
    let status = Command::new("aws")
        .env("AWS_ACCESS_KEY_ID", access_key)
        .env("AWS_SECRET_ACCESS_KEY", secret_key)
        .env("AWS_REGION", "auto")
        .arg("--endpoint-url")
        .arg(ENDPOINT)
        .args(args)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .unwrap_or_else(|e| {
            eprintln!("::error::ci-scratch: failed to spawn aws: {e}");
            std::process::exit(1);
        });
    if !status.success() {
        eprintln!("::error::ci-scratch: aws {args:?} exited with {status}");
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn require_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| {
        eprintln!("::error::ci-scratch: {name} env var not set");
        std::process::exit(1);
    })
}
