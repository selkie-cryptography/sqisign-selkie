//! Profile-guided optimization (PGO) build driver for sqisign-selkie.
//!
//! Runs the three-phase pipeline -- instrument, train, optimize -- for the
//! HOST target. PGO recovers the inlining and branch-layout that static
//! heuristics miss in the branchy lattice / quaternion / Deuring frames
//! (hand-tuned asm kernels gain nothing); measured ~8-9% on keygen and
//! sign, ~2% on verify, byte-identical (KAT-verified).
//!
//! # The profile is regenerated, never committed
//!
//! A `.profdata` is ~20 MB and keyed to the exact rustc/LLVM version (it
//! goes stale on every toolchain bump), so committing it would bloat
//! history with an opaque artifact that rots. It is instead fully
//! reproducible from a deterministic corpus, so CI regenerates it,
//! cache-keyed on `(target triple, rustc version, source hash)`. The
//! output lives under `target/` (already git-ignored).
//!
//! # Corpus
//!
//! Training drives `examples/profile_sign all <seeds>`, which perturbs one
//! seed byte per iteration -- a varied, deterministic, self-contained
//! keygen / sign / verify workload (no C-ref oracle needed). Varied
//! trajectories are what make the profile generalize instead of
//! overfitting one path. Seeds past ~100 are diminishing returns (the hot
//! frames saturate early, only the rejection-sampling tail keeps growing);
//! the count is a parameter so the plateau is easy to confirm with `bench`.
//!
//! # Features
//!
//! Training and the profile-use release build share the production main
//! track ([`TRAIN_FEATURES`]); `profile_sign` uses only the public API, so
//! it needs no `expose-internals`. They MUST match for the profile to
//! apply. The A/B `bench` adds `expose-internals` ([`BENCH_FEATURES`])
//! because the `sqisign` bench target requires it; the shared
//! keygen/sign/verify functions the bench calls receive the trained
//! profile.
//!
//! # Flags
//!
//! Host rustflags are read from `.cargo/config.toml` so the instrumented
//! and optimized builds match production codegen (target-cpu,
//! target-feature, the aes backend cfg); the `-Cprofile-{generate,use}`
//! flag is appended, not substituted (a bare `RUSTFLAGS` env would drop
//! the config flags, since cargo does not merge the two).
//!
//! # Usage
//!
//!   pgo-build generate [seeds]   force instrument + train + merge -> target/pgo/<triple>.profdata
//!   pgo-build build    [seeds]   reuse-or-generate, then `cargo build --release` with -Cprofile-use
//!   pgo-build bench    [seeds]   reuse-or-generate, then A/B baseline vs PGO (keygen/sign/verify)
//!   pgo-build flags              print the resolved host triple, rustflags, and profile path
//!
//! # Compile and run
//!
//!   rustc -O .github/scripts/pgo-build.rs -o /tmp/pgo-build && /tmp/pgo-build bench 100

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};

/// Example binary that drives the perturbed-seed keygen/sign/verify loop.
const EXAMPLE: &str = "profile_sign";

/// Features for the training workload and the profile-use release build:
/// the production main track. `profile_sign` uses only the public API,
/// so it needs no `expose-internals`. Training and the release build
/// must share this set for the profile to match; set it to the shipped
/// configuration.
const TRAIN_FEATURES: &str = "";

/// Features for the A/B `bench`: the `sqisign` bench target requires
/// `expose-internals`. The trained profile still applies to the shared
/// keygen/sign/verify functions the bench calls.
const BENCH_FEATURES: &str = "expose-internals";

/// Seeds when none is given. ~100 varied trajectories saturate the hot
/// frames; see the module docs on diminishing returns.
const DEFAULT_SEEDS: usize = 100;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let seeds: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SEEDS);

    let root = repo_root();
    let triple = host_triple();
    let flags = config_rustflags(&root, &triple);
    let profile = root.join("target").join("pgo").join(format!("{triple}.profdata"));

    match cmd {
        "generate" => generate(&root, &triple, &flags, &profile, seeds),
        "build" => {
            ensure(&root, &triple, &flags, &profile, seeds);
            build_release(&root, &flags, &profile);
        }
        "bench" => {
            ensure(&root, &triple, &flags, &profile, seeds);
            bench(&root, &flags, &profile);
        }
        "flags" => {
            println!("triple:    {triple}");
            println!("rustflags: {}", flags.join(" "));
            println!("profile:   {}", profile.display());
            println!("toolchain: {}", rustc_version());
        }
        "" | "-h" | "--help" | "help" => usage(),
        other => {
            eprintln!("unknown command: {other}\n");
            usage();
            exit(2);
        }
    }
}

/// Prints usage and exits zero.
fn usage() {
    eprintln!(
        "pgo-build <generate|build|bench|flags> [seeds]\n\
         \n  generate [seeds]  force instrument + train + merge -> target/pgo/<triple>.profdata\
         \n  build    [seeds]  reuse-or-generate, then cargo build --release with -Cprofile-use\
         \n  bench    [seeds]  reuse-or-generate, then A/B baseline vs PGO\
         \n  flags             print resolved host triple, rustflags, profile path"
    );
}

/// Regenerates the profile only when it is absent, so a cached profile is
/// reused. `generate` forces a rebuild (needed after a toolchain bump,
/// since the profile is LLVM-version-keyed).
fn ensure(root: &Path, triple: &str, flags: &[String], profile: &Path, seeds: usize) {
    if profile.is_file() {
        eprintln!("[pgo] reusing {}", profile.display());
    } else {
        generate(root, triple, flags, profile, seeds);
    }
}

/// Instruments, trains on `seeds` perturbed-seed trajectories, and merges
/// the raw counters into `profile`, writing a sidecar describing how it
/// was made. Overwrites any existing profile (it is toolchain-keyed).
fn generate(root: &Path, triple: &str, flags: &[String], profile: &Path, seeds: usize) {
    let raw = root.join("target").join("pgo").join("raw");
    let _ = fs::remove_dir_all(&raw);
    fs::create_dir_all(&raw).expect("create target/pgo/raw");

    // Instrumented build + run in one step: cargo run rebuilds the example
    // with -Cprofile-generate, then executing it writes the .profraw.
    let mut gen_flags = flags.to_vec();
    gen_flags.push("-Cprofile-generate=".to_string() + raw.to_str().expect("utf-8 path"));

    eprintln!("[pgo] instrument + train: {EXAMPLE} all {seeds} ({triple})");
    let mut c = cargo(root, &gen_flags);
    c.args(["run", "--release", "--example", EXAMPLE, "--features", TRAIN_FEATURES, "--"]);
    c.args(["all", &seeds.to_string()]);
    run(c);

    fs::create_dir_all(profile.parent().expect("profile dir")).expect("create target/pgo");
    eprintln!("[pgo] merge -> {}", profile.display());
    let mut m = Command::new(llvm_profdata(triple));
    m.args(["merge", "-o"]).arg(profile).arg(&raw);
    run(m);

    let meta = profile.with_extension("profdata.meta");
    let body = format!(
        "triple = {triple}\nseeds = {seeds}\nfeatures = {TRAIN_FEATURES}\n\
         rustflags = {}\ntoolchain = {}\n",
        flags.join(" "),
        rustc_version(),
    );
    fs::write(&meta, body).expect("write profile meta");
    eprintln!("[pgo] wrote {} ({})", profile.display(), meta.display());
}

/// Builds the release artifacts with `-Cprofile-use` -- the production PGO
/// build. Shares [`TRAIN_FEATURES`] with the profile so functions match.
fn build_release(root: &Path, flags: &[String], profile: &Path) {
    let mut pgo_flags = flags.to_vec();
    pgo_flags.push("-Cprofile-use=".to_string() + profile.to_str().expect("utf-8 path"));

    eprintln!("[pgo] release build with -Cprofile-use ({TRAIN_FEATURES})");
    let mut c = cargo(root, &pgo_flags);
    c.args(["build", "--release", "--features", TRAIN_FEATURES]);
    run(c);
}

/// Builds and runs the keygen/sign/verify bench twice -- baseline flags,
/// then with `-Cprofile-use` -- so the PGO delta is read directly.
fn bench(root: &Path, flags: &[String], profile: &Path) {
    let filter = "keygen_derand|sign_derand|verify";

    eprintln!("\n[pgo] === BASELINE ===");
    let mut base = cargo(root, flags);
    base.args(["bench", "--bench", "sqisign", "--features", BENCH_FEATURES, "--", filter]);
    run(base);

    eprintln!("\n[pgo] === PGO ({}) ===", profile.display());
    let mut pgo_flags = flags.to_vec();
    pgo_flags.push("-Cprofile-use=".to_string() + profile.to_str().expect("utf-8 path"));
    let mut pgo = cargo(root, &pgo_flags);
    pgo.args(["bench", "--bench", "sqisign", "--features", BENCH_FEATURES, "--", filter]);
    run(pgo);
}

/// Walks up from the cwd to the crate root (the dir whose `Cargo.toml`
/// names this package), so the driver works from any subdirectory.
fn repo_root() -> PathBuf {
    let mut dir = env::current_dir().expect("cwd");
    loop {
        if dir.join("Cargo.toml").is_file() && dir.join("src").join("lib.rs").is_file() {
            return dir;
        }
        if !dir.pop() {
            eprintln!("not inside the sqisign-selkie crate");
            exit(1);
        }
    }
}

/// Returns the host target triple from `rustc -vV` (the `host:` line).
fn host_triple() -> String {
    let out = Command::new("rustc").arg("-vV").output().expect("run rustc -vV");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("host: "))
        .expect("rustc -vV host line")
        .trim()
        .to_string()
}

/// Returns the `rustc -V` version string (the profile is keyed to it).
fn rustc_version() -> String {
    let out = Command::new("rustc").arg("-V").output().expect("run rustc -V");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// Path to the toolchain's bundled `llvm-profdata` (the `llvm-tools`
/// rustup component; CI must install it).
fn llvm_profdata(triple: &str) -> PathBuf {
    let out = Command::new("rustc").args(["--print", "sysroot"]).output().expect("rustc sysroot");
    let sysroot = String::from_utf8_lossy(&out.stdout).trim().to_string();
    PathBuf::from(sysroot)
        .join("lib")
        .join("rustlib")
        .join(triple)
        .join("bin")
        .join("llvm-profdata")
}

/// Extracts `rustflags` for `[target.<triple>]` from `.cargo/config.toml`
/// so PGO builds match production codegen. Returns the quoted tokens in
/// the array (empty if the section or key is absent).
fn config_rustflags(root: &Path, triple: &str) -> Vec<String> {
    let text = match fs::read_to_string(root.join(".cargo").join("config.toml")) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    // Slice from the target section header to the next section header.
    let header = format!("[target.{triple}]");
    let Some(start) = text.find(&header) else {
        eprintln!("[pgo] warning: no {header} in .cargo/config.toml; using bare rustflags");
        return Vec::new();
    };
    let section = &text[start + header.len()..];
    let section = section.split("\n[").next().unwrap_or(section);

    // Capture the rustflags = [ ... ] array body (it may span lines).
    let Some(key) = section.find("rustflags") else {
        return Vec::new();
    };
    let after = &section[key..];
    let (Some(lb), Some(rb)) = (after.find('['), after.find(']')) else {
        return Vec::new();
    };
    let body = &after[lb + 1..rb];

    // Collect the double-quoted tokens.
    let mut tokens = Vec::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        if c == '"' {
            let mut tok = String::new();
            for d in chars.by_ref() {
                if d == '"' {
                    break;
                }
                tok.push(d);
            }
            tokens.push(tok);
        }
    }
    tokens
}

/// Builds a `cargo` invocation in `root` with `RUSTFLAGS` set to `flags`
/// (joined), so config flags plus the appended profile flag both apply.
fn cargo(root: &Path, flags: &[String]) -> Command {
    let mut c = Command::new("cargo");
    c.current_dir(root);
    c.env("RUSTFLAGS", flags.join(" "));
    c
}

/// Runs a command inheriting stdio; aborts on spawn failure or nonzero
/// exit so a failed phase never silently produces a stale profile.
fn run(mut c: Command) {
    let what = format!("{c:?}");
    match c.status() {
        Ok(s) if s.success() => {}
        Ok(s) => {
            eprintln!("command failed ({s}): {what}");
            exit(1);
        }
        Err(e) => {
            eprintln!("spawn failed ({e}): {what}");
            exit(1);
        }
    }
}
