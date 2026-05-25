//! Verify an OCI image manifest's total uncompressed layer size is
//! under a given GiB limit. Used by `infra-ci.yml` to catch runner
//! images that would silently fail to unpack on Fly Machines
//! (Fly's image-unpack ceiling sits below the documented 8 GiB
//! limit empirically — a 6.77 GiB image failed to unpack in
//! production, causing every spawned Machine to abort before its
//! actions-runner registered and runs to hang in "queued" forever).
//!
//! Usage:
//!   check-image-size <image-ref> <max-gib>
//!
//! - `<image-ref>` is any docker-style reference, e.g.
//!   `registry.fly.io/sqisign-infra-runners:base`
//!   `registry.fly.io/sqisign-infra-runners@sha256:...`
//!   A tag is resolved to its current manifest digest first; a
//!   `@sha256:` reference is used as-is. Single-platform manifests
//!   only.
//! - `<max-gib>` is the inclusive upper bound (e.g. `7`). The
//!   image must be strictly smaller.
//!
//! Reads `FLY_API_TOKEN` (or `FLY_API_TOKEN_RUNNERS`) from the env
//! for Basic-auth against `registry.fly.io`. Shells out to `curl`
//! for HTTP and `zstd -d` for per-layer decompression — both are
//! preinstalled in the runner image. Posts a `::error::` workflow
//! annotation on overage so the failure surfaces in the PR/run UI.
//!
//! Compile: `rustc -O check-image-size.rs -o check-image-size`

use std::env;
use std::io::Read;
use std::process::{Command, Stdio};

const HEAD_ACCEPT: &str = "application/vnd.oci.image.manifest.v1+json,\
                           application/vnd.docker.distribution.manifest.v2+json";

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        die("usage: check-image-size <image-ref> <max-gib>");
    }
    let image_ref = &args[1];
    let max_gib: u64 = args[2]
        .parse()
        .unwrap_or_else(|e| die(&format!("max-gib must be a positive integer ({e})")));
    let max_bytes = max_gib
        .checked_mul(1 << 30)
        .unwrap_or_else(|| die("max-gib overflows u64 bytes"));

    let token = env::var("FLY_API_TOKEN")
        .or_else(|_| env::var("FLY_API_TOKEN_RUNNERS"))
        .unwrap_or_else(|_| die("FLY_API_TOKEN (or FLY_API_TOKEN_RUNNERS) must be set"));

    let (host, path) = image_ref
        .split_once('/')
        .unwrap_or_else(|| die(&format!("image-ref missing '/' separator: {image_ref}")));
    let (repo, reference) = parse_reference(path);

    // Resolve tag -> digest if needed so the size we measure is the
    // exact manifest the registry will serve, not whatever the tag
    // happens to point at later.
    let digest = if reference.starts_with("sha256:") {
        reference.to_string()
    } else {
        resolve_digest(host, repo, reference, &token)
    };

    eprintln!("Inspecting {host}/{repo}@{digest}");

    let manifest = curl_get_text(
        &format!("https://{host}/v2/{repo}/manifests/{digest}"),
        &token,
        Some(HEAD_ACCEPT),
    );
    let layer_digests = parse_layer_digests(&manifest);
    if layer_digests.is_empty() {
        die("manifest has no layers (multi-arch index? this script handles single-platform manifests only)");
    }

    let mut total: u64 = 0;
    for (i, layer) in layer_digests.iter().enumerate() {
        let n = measure_layer(host, repo, layer, &token);
        eprintln!(
            "  layer {:>2}/{}  uncompressed={:>7.1} MiB",
            i + 1,
            layer_digests.len(),
            n as f64 / (1u64 << 20) as f64,
        );
        total += n;
    }

    let gib = total as f64 / (1u64 << 30) as f64;
    eprintln!("Total uncompressed: {gib:.2} GiB  (limit: {max_gib} GiB)");

    if total >= max_bytes {
        eprintln!(
            "::error::image {image_ref} is {gib:.2} GiB uncompressed (>= {max_gib} GiB). \
             Trim Dockerfile.base / Dockerfile.runtime before merging."
        );
        std::process::exit(1);
    }
}

/// Splits "name[:tag|@digest]" into ("name", "tag|digest").
/// Defaults to "latest" if neither tag nor digest is present.
/// `rsplit_once` so a port number in `name` (`host:5000/foo:tag`)
/// doesn't get misread as a tag.
fn parse_reference(path: &str) -> (&str, &str) {
    if let Some((repo, digest)) = path.rsplit_once('@') {
        (repo, digest)
    } else if let Some((repo, tag)) = path.rsplit_once(':') {
        (repo, tag)
    } else {
        (path, "latest")
    }
}

/// HEAD the manifest endpoint and return the `Docker-Content-Digest`
/// header. The Accept header constrains the response to a single-
/// platform manifest; a multi-arch index would return an index
/// digest that wouldn't match any Machine's actual platform digest.
fn resolve_digest(host: &str, repo: &str, tag: &str, token: &str) -> String {
    let url = format!("https://{host}/v2/{repo}/manifests/{tag}");
    let out = run(
        "curl",
        &[
            "-fsSL", "-I", "-u", &format!("x:{token}"), "-H", &format!("Accept: {HEAD_ACCEPT}"),
            &url,
        ],
    );
    for line in out.lines() {
        // Header names are case-insensitive per RFC 7230.
        if let Some(rest) = line
            .strip_prefix("docker-content-digest:")
            .or_else(|| line.strip_prefix("Docker-Content-Digest:"))
        {
            return rest.trim().to_string();
        }
    }
    die(&format!("could not resolve {tag} digest from {url}"))
}

/// GET the manifest JSON.
fn curl_get_text(url: &str, token: &str, accept: Option<&str>) -> String {
    let mut args = vec!["-fsSL", "-u"];
    let auth = format!("x:{token}");
    args.push(&auth);
    let accept_hdr;
    if let Some(a) = accept {
        accept_hdr = format!("Accept: {a}");
        args.extend_from_slice(&["-H", &accept_hdr]);
    }
    args.push(url);
    run("curl", &args)
}

/// Pull layer digests out of the manifest JSON. The OCI/Docker
/// manifest layout is regular enough that scanning for `"digest":`
/// after `"layers":` is robust; avoids pulling a JSON crate into a
/// single-file `rustc` script.
fn parse_layer_digests(manifest: &str) -> Vec<String> {
    let Some(after_layers) = manifest.find("\"layers\"").map(|i| &manifest[i..]) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let needle = "\"digest\"";
    let mut rest = after_layers;
    while let Some(i) = rest.find(needle) {
        rest = &rest[i + needle.len()..];
        // Skip ':' + whitespace, then expect '"sha256:...".
        let Some(start) = rest.find('"') else { break };
        let value_start = start + 1;
        let Some(end) = rest[value_start..].find('"') else { break };
        out.push(rest[value_start..value_start + end].to_string());
        rest = &rest[value_start + end..];
    }
    out
}

/// Pipe `curl <blob>` -> `zstd -d` -> stdout, counting bytes.
fn measure_layer(host: &str, repo: &str, layer_digest: &str, token: &str) -> u64 {
    let url = format!("https://{host}/v2/{repo}/blobs/{layer_digest}");
    let mut curl = Command::new("curl")
        .args(["-fsSL", "-u", &format!("x:{token}"), &url])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| die(&format!("spawn curl: {e}")));
    let curl_out = curl.stdout.take().expect("curl stdout");

    let mut zstd = Command::new("zstd")
        .args(["-d", "-c"])
        .stdin(Stdio::from(curl_out))
        .stdout(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| die(&format!("spawn zstd: {e}")));
    let mut zstd_out = zstd.stdout.take().expect("zstd stdout");

    let mut total: u64 = 0;
    let mut buf = [0u8; 64 * 1024];
    loop {
        match zstd_out.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => total += n as u64,
            Err(e) => die(&format!("read zstd stdout: {e}")),
        }
    }

    let curl_status = curl.wait().unwrap_or_else(|e| die(&format!("curl wait: {e}")));
    let zstd_status = zstd.wait().unwrap_or_else(|e| die(&format!("zstd wait: {e}")));
    if !curl_status.success() {
        die(&format!("curl exited {curl_status} for {url}"));
    }
    if !zstd_status.success() {
        die(&format!("zstd exited {zstd_status} for {url}"));
    }
    total
}

fn run(prog: &str, args: &[&str]) -> String {
    let out = Command::new(prog)
        .args(args)
        .output()
        .unwrap_or_else(|e| die(&format!("spawn {prog}: {e}")));
    if !out.status.success() {
        die(&format!(
            "{prog} {} exited {}: {}",
            args.join(" "),
            out.status,
            String::from_utf8_lossy(&out.stderr),
        ));
    }
    String::from_utf8(out.stdout).unwrap_or_else(|e| die(&format!("{prog} stdout not utf-8: {e}")))
}

fn die(msg: &str) -> ! {
    eprintln!("[check-image-size] {msg}");
    std::process::exit(1);
}
