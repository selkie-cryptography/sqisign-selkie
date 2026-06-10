//! Local profiling driver for the `profile_sign` workload on macOS
//! (Apple Silicon). Thin wrapper over cargo-instruments / xctrace, dhat,
//! and samply, all pointed at the deterministic `examples/profile_sign.rs`
//! keygen/sign loop built under the `profiling` profile.
//!
//! These are WALL-CLOCK / ALLOCATION profilers -- the only kind available
//! on arm64 macOS. The deterministic instruction counts the perf
//! decisions are graded on come from gungraun (valgrind/callgrind) on x86
//! CI; valgrind has no Apple Silicon port. Use this to find hot frames
//! and the direction of a change; confirm committed wins on CI.
//!
//! Usage:
//!   profile time     [keygen|sign|verify|all] [iters]   Instruments Time Profiler (flamegraph)
//!   profile counters [keygen|sign|verify|all] [iters]   Apple PMU incl. instructions-retired
//!   profile samply   [keygen|sign|verify|all] [iters]   samply -> Firefox Profiler
//!   profile dhat     [keygen|sign|verify|all] [iters]   heap profile -> dhat-heap.json
//!   profile build                                       just build the example
//!
//! `time`/`counters` use cargo-instruments when present, else raw xctrace.
//! `dhat` with no mode profiles keygen + sign + verify together (`all`).
//!
//! Compile + run:
//!   rustc -O scripts/profile.rs -o /tmp/profile && /tmp/profile time sign 50

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, exit};
use std::time::{SystemTime, UNIX_EPOCH};

const PROFILE: &str = "profiling";
const EXAMPLE: &str = "profile_sign";

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let rest: Vec<String> = args.iter().skip(1).cloned().collect();
    let root = repo_root();
    let bin = root
        .join("target")
        .join(PROFILE)
        .join("examples")
        .join(EXAMPLE);

    match cmd {
        "time" | "counters" => {
            let template = if cmd == "counters" {
                "CPU Counters"
            } else {
                "Time Profiler"
            };
            if have("cargo-instruments") {
                // cargo-instruments builds, records, and opens Instruments.
                let mut c = cargo(&root);
                c.args(["instruments", "-t", template, "--profile", PROFILE, "--example", EXAMPLE]);
                c.arg("--").args(&rest);
                run(c);
            } else {
                build(&root);
                let out = format!("{EXAMPLE}-{cmd}-{}.trace", stamp());
                let mut c = Command::new("xctrace");
                c.current_dir(&root)
                    .args(["record", "--template", template, "--output", &out, "--launch", "--"])
                    .arg(&bin)
                    .args(&rest);
                run(c);
                eprintln!("wrote {out}");
                let mut opener = Command::new("open");
                opener.current_dir(&root).arg(&out);
                run(opener);
            }
        }
        "samply" => {
            if !have("samply") {
                eprintln!("samply not found: cargo install samply");
                exit(1);
            }
            build(&root);
            let mut c = Command::new("samply");
            c.current_dir(&root).arg("record").arg(&bin).args(&rest);
            run(c);
        }
        "dhat" => {
            // No mode -> profile the full path (keygen + sign + verify).
            let passthrough = if rest.is_empty() {
                vec!["all".to_string()]
            } else {
                rest
            };
            let mut c = cargo(&root);
            c.args(["run", "--profile", PROFILE, "--features", "dhat-heap", "--example", EXAMPLE, "--"]);
            c.args(&passthrough);
            run(c);
            eprintln!("wrote {}/dhat-heap.json", root.display());
            eprintln!("view at https://nnethercote.github.io/dh_view/dh_view.html (Load the JSON)");
        }
        "build" => {
            build(&root);
            eprintln!("built {}", bin.display());
        }
        "" | "-h" | "--help" | "help" => usage(),
        other => {
            eprintln!("unknown command: {other}\n");
            usage();
            exit(2);
        }
    }
}

/// Walk up from the current directory to the crate root (the dir whose
/// `Cargo.toml` names this package), so the driver works from any subdir.
fn repo_root() -> PathBuf {
    let mut dir = env::current_dir().expect("cwd");
    loop {
        if dir.join("Cargo.toml").is_file() && dir.join("src").join("lib.rs").is_file() {
            return dir;
        }
        if !dir.pop() {
            eprintln!("not inside the sqisign-selkie crate (no Cargo.toml + src/lib.rs found above cwd)");
            exit(1);
        }
    }
}

fn cargo(root: &Path) -> Command {
    let mut c = Command::new("cargo");
    c.current_dir(root);
    c
}

fn build(root: &Path) {
    let mut c = cargo(root);
    c.args(["build", "--profile", PROFILE, "--example", EXAMPLE]);
    run(c);
}

/// Run a command inheriting stdio (so profilers stay interactive and
/// build output streams); abort on spawn failure or nonzero exit.
fn run(mut c: Command) {
    let what = format!("{c:?}");
    match c.status() {
        Ok(s) if s.success() => {}
        Ok(s) => {
            eprintln!("command failed ({s}): {what}");
            exit(s.code().unwrap_or(1));
        }
        Err(e) => {
            eprintln!("could not run {what}: {e}");
            exit(1);
        }
    }
}

/// Is `cmd` on PATH? Spawns it with `--help` and checks it launched at
/// all (exit status is irrelevant; a spawn error means absent).
fn have(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn stamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn usage() {
    eprint!(
        "{}",
        "profile -- local flamegraph / heap profiling for sqisign-selkie\n\
         \n\
         usage: profile <command> [keygen|sign|verify|all] [iters]\n\
         \n\
           time      Instruments Time Profiler -> flamegraph (auto-opens)\n\
           counters  Apple PMU incl. instructions-retired (closest local Ir proxy)\n\
           samply    samply -> Firefox Profiler\n\
           dhat      heap profile -> dhat-heap.json (no mode -> all)\n\
           build     just build the profiling example\n\
         \n\
         compile: rustc -O scripts/profile.rs -o /tmp/profile\n\
         example: /tmp/profile time sign 50\n"
    );
}
