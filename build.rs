//! Compile-time CPU detection for `Fp` arch selection.
//!
//! Emits `cfg(sqisign_selkie_arch = "neon")` or `"avx2"` when the target
//! is a CPU where the corresponding vectorised `Fp` implementation is
//! expected to win.  The arch dispatcher in
//! `src/fields/fp/arch/mod.rs` consumes the cfg to `pub use` the
//! matching backend's `Fp` type.
//!
//! ## `"neon"` (aarch64)
//!
//! - **Set** on aarch64 with `target_cpu` not in the known-loss list and a host
//!   check (`sysctl machdep.cpu.brand_string`) that doesn't say the chip is
//!   wider-pipe Apple Silicon (M2 and later).
//! - **Not set** on `aarch64-apple-darwin` whose detected host chip is Apple M2
//!   / M3 / M4 (or any future `Apple M{N>=2}` spelling), and on any target
//!   whose `target_cpu` matches a known-loss entry.
//!
//! The list reflects measured / projected outcomes for the 2026/394
//! NEON Fp port:
//!
//! - Apple M4: measured -1.80x.
//! - Apple M3 / M2: projected loss (wider scalar pipe than M1).
//! - Apple M1: paper-confirmed 1.22x win.
//! - Cortex-A76 / Neoverse N1 / Neoverse V1 (Graviton 2 / 3, Ampere Altra,
//!   Raspberry Pi 5): paper / projected 1.3-1.5x win.
//! - Neoverse V2 (Graviton 4, Google Axion): uncertain; pending bench.
//!
//! ## `"avx2"` (x86_64)
//!
//! - **Set** on `target_arch = "x86_64"` when `CARGO_CFG_TARGET_FEATURE`
//!   contains `avx2`.  AVX2 is not part of base x86_64; users opt in via
//!   `RUSTFLAGS="-C target-cpu=..."` or `-C target-feature=+avx2`.
//!
//! The `arch::x86_64::avx2` backend doesn't exist yet — this hook is
//! infrastructure that lights up when the AVX2 `Fp` lands.

fn main() {
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_cpu = std::env::var("CARGO_CFG_TARGET_CPU").unwrap_or_default();
    let target_features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();

    println!("cargo::rustc-check-cfg=cfg(sqisign_selkie_arch, values(\"neon\", \"avx2\"))");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_OS");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_CPU");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");

    match target_arch.as_str() {
        "aarch64" => detect_aarch64_neon(&target_os, &target_cpu),
        "x86_64" => detect_x86_64_avx2(&target_features),
        _ => {}
    }
}

fn detect_aarch64_neon(target_os: &str, target_cpu: &str) {
    // CPUs whose scalar u64-mul throughput beats the NEON Fp29x4 path.
    let known_loss_cpus = ["apple-m2", "apple-m3", "apple-m4"];
    if known_loss_cpus.contains(&target_cpu) {
        return;
    }

    // Native macOS builds default `target_cpu` to "apple-a14" or "generic"
    // even when the host is M2+; read sysctl to disambiguate before
    // assuming the NEON path is a win.
    let native_macos = cfg!(target_os = "macos") && target_os == "macos";
    if native_macos && host_apple_chip_is_loss() {
        return;
    }

    println!("cargo::rustc-cfg=sqisign_selkie_arch=\"neon\"");
}

fn detect_x86_64_avx2(target_features: &str) {
    // `CARGO_CFG_TARGET_FEATURE` is a comma-separated list of enabled
    // target features (e.g. "fxsr,sse,sse2,sse3,ssse3,avx,avx2,...").
    let has_avx2 = target_features.split(',').any(|f| f == "avx2");
    if has_avx2 {
        println!("cargo::rustc-cfg=sqisign_selkie_arch=\"avx2\"");
    }
}

/// Returns `true` when the host is an Apple Silicon chip wider than M1
/// (M2, M3, M4, or any future `Apple M{N>=2}` spelling).
///
/// Reads `sysctl -n machdep.cpu.brand_string` which returns strings of
/// the form `"Apple M4 Pro"` / `"Apple M2 Max"` / `"Apple M1"`.  Returns
/// `false` on failure so the surrounding logic falls through to its
/// default.
fn host_apple_chip_is_loss() -> bool {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output();
    let brand = match output {
        Ok(o) if o.status.success() => match String::from_utf8(o.stdout) {
            Ok(s) => s,
            Err(_) => return false,
        },
        _ => return false,
    };
    let brand = brand.trim();
    brand.starts_with("Apple M")
        && brand
            .strip_prefix("Apple M")
            .and_then(|rest| rest.chars().next())
            .is_some_and(|c| matches!(c, '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'))
}
