//! Compile-time CPU detection for arch-specific routing.
//!
//! Emits the `neon_fp_default` cfg when the target is an aarch64 core that
//! benefits from the NEON-vectorised Fp29x4 path.  Currently:
//!
//! - **Set** on aarch64 with `target_cpu` not in the known-loss list, and on
//!   native macOS builds whose host chip is M1 or earlier (M-series cores wider
//!   than M1 lose to scalar Fp51).
//! - **Not set** on `aarch64-apple-darwin` whose detected host chip is Apple M2
//!   / M3 / M4 (or any future model spelled `Apple M{N≥2}`), and on any target
//!   whose `target_cpu` matches a known-loss entry.
//!
//! The list reflects measured / projected outcomes for the 2026/394
//! NEON Fp port:
//!
//! - Apple M4: measured -1.80x on this codebase.
//! - Apple M3 / M2: projected loss (broader scalar pipe than M1).
//! - Apple M1: paper-confirmed 1.22x win.
//! - Cortex-A76 / Neoverse N1 / Neoverse V1 (Graviton 2 / 3, Ampere Altra,
//!   Raspberry Pi 5): paper / projected 1.3-1.5x win.
//! - Neoverse V2 (Graviton 4, Google Axion): uncertain; pending bench.
//!
//! The cfg is currently a hook with no production effect — `Fp::mul`
//! still routes to the scalar Rust radix-51 implementation.  Future
//! commits that wire `Fp29x4` through `Fp::mul` (or higher-level
//! callers) consume this cfg to gate the NEON path.

fn main() {
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_cpu = std::env::var("CARGO_CFG_TARGET_CPU").unwrap_or_default();

    println!("cargo::rustc-check-cfg=cfg(neon_fp_default)");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_OS");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_CPU");

    if target_arch != "aarch64" {
        return;
    }

    // CPUs whose scalar u64-mul throughput beats the NEON Fp29x4 path.
    // Extend as per-target bench data accumulates.
    let known_loss_cpus = ["apple-m2", "apple-m3", "apple-m4"];
    if known_loss_cpus.contains(&target_cpu.as_str()) {
        return;
    }

    // On native macOS builds, `target_cpu` is often "apple-a14" or "generic"
    // even when the host is M2+; read sysctl to disambiguate before
    // assuming the NEON path is a win.
    let native_macos = cfg!(target_os = "macos") && target_os == "macos";
    if native_macos && host_apple_chip_is_loss() {
        return;
    }

    println!("cargo::rustc-cfg=neon_fp_default");
}

/// Returns `true` when the host is an Apple Silicon chip wider than M1
/// (M2, M3, M4, or any future spelling that matches `Apple M{N≥2}`).
///
/// Reads `sysctl -n machdep.cpu.brand_string` which returns strings of the
/// form `"Apple M4 Pro"` / `"Apple M2 Max"` / `"Apple M1"`.  Returns `false`
/// on failure (sysctl unavailable, unparseable output) so the surrounding
/// logic falls through to its default.
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
    // Match "Apple M2", "Apple M2 Pro", "Apple M2 Max", ..., "Apple M3 ...",
    // "Apple M4 ...".  Do not match "Apple M1" or "Apple M1 Pro".
    brand.starts_with("Apple M")
        && brand
            .strip_prefix("Apple M")
            .and_then(|rest| rest.chars().next())
            .is_some_and(|c| matches!(c, '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'))
}
