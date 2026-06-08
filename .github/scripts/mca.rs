//! Static per-kernel pipeline estimate for the hot x86_64 field asm.
//!
//! Cross-compiles the lib to x86_64 asm with the `mca` feature (which
//! exposes `#[no_mangle] mca_*` shims around the otherwise-inlined hot
//! kernels), slices each shim's body out of the `.s`, and runs
//! `llvm-mca` over it.  No execution: `llvm-mca` statically models port
//! pressure and throughput, so this runs on any host (including Apple
//! Silicon) with no x86 VM or emulator.
//!
//! Env:
//!   MCA_BIN    path to llvm-mca (default: brew llvm if present, else PATH)
//!   MCA_CPU    `-mcpu` target   (default: znver4; CI passes `native`)
//!   MCA_ITERS  llvm-mca iterations (default: 100)
//!   MCA_CARGO  cargo invocation (default: `cargo +nightly`; CI: `cargo`)
//! Positional args restrict the run to the named kernels.
//!
//! Compile + run: `rustc -O mca.rs -o /tmp/mca && /tmp/mca`.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, exit};

/// Cross-compilation target whose asm we analyze.
const TARGET: &str = "x86_64-unknown-linux-gnu";

/// Kernels analyzed by default, in report order.
const ALL_KERNELS: &[&str] = &["fp64_mul", "fp64_mul_wide", "fp64_square", "fp51_square"];

fn main() {
    let kernels: Vec<String> = {
        let args: Vec<String> = env::args().skip(1).collect();
        if args.is_empty() {
            ALL_KERNELS.iter().map(|s| (*s).to_string()).collect()
        } else {
            args
        }
    };

    let repo_root = repo_root();
    env::set_current_dir(&repo_root).expect("cd to repo root");

    let mca_bin = env::var("MCA_BIN").unwrap_or_else(|_| {
        let brew = "/opt/homebrew/opt/llvm/bin/llvm-mca";
        if Path::new(brew).is_file() {
            brew.to_string()
        } else {
            "llvm-mca".to_string()
        }
    });
    let mca_cpu = env::var("MCA_CPU").unwrap_or_else(|_| "znver4".to_string());
    let mca_iters = env::var("MCA_ITERS").unwrap_or_else(|_| "100".to_string());
    let cargo_cmd = env::var("MCA_CARGO").unwrap_or_else(|_| "cargo +nightly".to_string());

    build_asm(&cargo_cmd);

    let asm_path = newest_asm().unwrap_or_else(|| {
        eprintln!("::error::no emitted .s under target/{TARGET}/release/deps/");
        exit(1);
    });
    eprintln!("asm: {}", asm_path.display());
    let asm = fs::read_to_string(&asm_path).expect("read emitted .s");

    let mut status = 0;
    for k in &kernels {
        let sym = format!("mca_{k}");

        let Some(snippet) = extract(&asm, &sym) else {
            // fp64/fp2 shims need bmi2+adx; absent if the build CPU lacks them.
            eprintln!("skip: {sym} not in asm (needs bmi2+adx?)");
            continue;
        };

        match run_mca(&mca_bin, &mca_cpu, &mca_iters, &snippet) {
            Ok(report) => print_summary(&sym, &report, &mca_iters),
            Err(e) => {
                eprintln!("::warning::llvm-mca failed on {sym}: {e}");
                status = 1;
            }
        }
    }
    exit(status);
}

/// Repo root via `git rev-parse --show-toplevel`.
fn repo_root() -> PathBuf {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .expect("run git rev-parse");
    let s = String::from_utf8(out.stdout).expect("git output utf8");
    PathBuf::from(s.trim())
}

/// Cross-compiles the lib to x86_64 asm with the `mca` feature.
fn build_asm(cargo_cmd: &str) {
    let parts: Vec<&str> = cargo_cmd.split_whitespace().collect();
    let (prog, lead) = parts.split_first().expect("MCA_CARGO is empty");

    eprintln!("Building {TARGET} asm with --features mca ...");
    let status = Command::new(prog)
        .args(lead)
        .args([
            "rustc",
            "--target",
            TARGET,
            "--release",
            "--lib",
            "--features",
            "mca",
            "--",
            "--emit",
            "asm",
            "-Cllvm-args=--x86-asm-syntax=intel",
        ])
        .status()
        .expect("spawn cargo rustc");
    if !status.success() {
        eprintln!("::error::cargo rustc --emit asm failed");
        exit(1);
    }
}

/// Newest `sqisign_selkie-*.s` under the target deps dir.
fn newest_asm() -> Option<PathBuf> {
    let dir = PathBuf::from(format!("target/{TARGET}/release/deps"));
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(&dir).ok()?.flatten() {
        let path = entry.path();
        let name = path.file_name()?.to_string_lossy().into_owned();
        if !(name.starts_with("sqisign_selkie-") && name.ends_with(".s")) {
            continue;
        }
        let mtime = entry.metadata().ok()?.modified().ok()?;
        if best.as_ref().is_none_or(|(t, _)| mtime > *t) {
            best = Some((mtime, path));
        }
    }
    best.map(|(_, p)| p)
}

/// Slices one shim's instruction stream out of the `.s`: from `^<sym>:`
/// to the next `.cfi_endproc`, dropping the label, assembler directives,
/// `#APP`/`#NO_APP` inline-asm markers, and blank lines.  Returns `None`
/// if the symbol is absent.  Prepends `.intel_syntax noprefix` so
/// llvm-mca parses Intel syntax.
fn extract(asm: &str, sym: &str) -> Option<String> {
    let label = format!("{sym}:");
    let mut grabbing = false;
    let mut out = String::from(".intel_syntax noprefix\n");
    let mut found = false;

    for line in asm.lines() {
        if !grabbing {
            if line.starts_with(&label) {
                grabbing = true;
                found = true;
            }
            continue;
        }
        if line.contains(".cfi_endproc") {
            break;
        }
        let t = line.trim_start();
        if t.is_empty() || t.starts_with('.') || t.starts_with("#APP") || t.starts_with("#NO_APP") {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }

    found.then_some(out)
}

/// Runs llvm-mca over a snippet (fed on stdin) and returns its report.
/// `-mtriple` pins the x86_64 backend so a non-x86 host's llvm-mca still
/// recognizes the `-mcpu` and models the right pipeline.
fn run_mca(bin: &str, cpu: &str, iters: &str, snippet: &str) -> Result<String, String> {
    let mut child = Command::new(bin)
        .args([
            "-mtriple",
            TARGET,
            &format!("-mcpu={cpu}"),
            &format!("-iterations={iters}"),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn {bin}: {e}"))?;

    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(snippet.as_bytes())
        .map_err(|e| format!("write stdin: {e}"))?;

    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Prints the headline numbers and busiest five ports as Markdown.
fn print_summary(sym: &str, report: &str, iters: &str) {
    let field = |label: &str| -> String {
        report
            .lines()
            .find(|l| l.contains(label))
            .and_then(|l| l.split(':').nth(1))
            .map(|v| v.replace(' ', ""))
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "n/a".to_string())
    };

    println!("### {sym}");
    println!(
        "- Block RThroughput: {} cycles/iter",
        field("Block RThroughput")
    );
    println!("- Total Cycles ({iters} iters): {}", field("Total Cycles"));
    println!("- uOps/cycle: {}", field("uOps Per Cycle"));
    println!("- Top port pressure (cycles/iter):");
    for (port, press) in top_ports(report, 5) {
        println!("    - {port}: {press:.2}");
    }
    println!();
}

/// Maps resource indices to port names and returns the busiest `n` ports
/// from the per-iteration pressure row, descending.
fn top_ports(report: &str, n: usize) -> Vec<(String, f64)> {
    let mut names: BTreeMap<String, String> = BTreeMap::new();
    let mut header: Vec<String> = Vec::new();
    let mut pressure: BTreeMap<String, f64> = BTreeMap::new();

    let mut in_resources = false;
    let mut want_header = false;
    let mut want_row = false;

    let strip = |tok: &str| tok.trim_matches(['[', ']']).to_string();

    for line in report.lines() {
        if line.starts_with("Resources:") {
            in_resources = true;
            continue;
        }
        if in_resources {
            if line.trim().is_empty() {
                in_resources = false;
            } else if let Some(rest) = line.trim().strip_prefix('[') {
                // `[3]   - Zn4ALU0`
                let idx = rest.split(']').next().unwrap_or("").to_string();
                if let Some(name) = line.split_whitespace().nth(2) {
                    names.insert(idx, name.to_string());
                }
                continue;
            }
        }

        if line.starts_with("Resource pressure per iteration:") {
            want_header = true;
            continue;
        }
        if want_header {
            header = line.split_whitespace().map(strip).collect();
            want_header = false;
            want_row = true;
            continue;
        }
        if want_row {
            for (col, tok) in line.split_whitespace().enumerate() {
                if tok == "-" {
                    continue;
                }
                if let (Some(idx), Ok(v)) = (header.get(col), tok.parse::<f64>()) {
                    if v > 0.0 {
                        pressure.insert(idx.clone(), v);
                    }
                }
            }
            want_row = false;
        }
    }

    let mut pairs: Vec<(String, f64)> = pressure
        .into_iter()
        .map(|(idx, v)| (names.get(&idx).cloned().unwrap_or(format!("[{idx}]")), v))
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(n);
    pairs
}
