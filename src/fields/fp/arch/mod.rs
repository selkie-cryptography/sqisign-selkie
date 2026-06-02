//! Architecture-specific backends for [`Fp`][super::Fp] arithmetic.
//!
//! Skeleton module that re-exports the active backend.  The
//! backend is selected at compile time by the target architecture:
//!
//! - `aarch64`: see file `arch/aarch64.rs`.  Future home for NEON vectorised
//!   `Fp` arithmetic per De Feo, Jian, Wang, Yang (ePrint 2026/394): radix-29
//!   packed limbs in NEON 32-bit lanes, Karatsuba over `Fp²` already in place
//!   at the higher level, lazy reduction across operation chains.  Measured win
//!   on Apple M1: roughly 1.22× total signing speedup.
//! - any other target: see file `arch/fallback.rs`.  Current scalar Rust
//!   radix-51 representation with interleaved schoolbook multiplication and
//!   Montgomery reduction via the `P4 = 5·2^44` constant, exploiting `p =
//!   5·2^248 − 1`'s Montgomery-friendly structure.
//!
//! # No x86_64-specific path
//!
//! The biggest remaining x86_64 perf lever is **ADX bigint** (on
//! the quaternion side), not vectorised `Fp`.  AVX-512-IFMA52
//! would be the analogous vectorised-Fp lever but is excluded from
//! sqisign-selkie's target set — only a narrow subset of Intel
//! server SKUs support it, AMD omitted it from Zen 4, and no major
//! crypto library (BoringSSL, libcrux, rust-openssl,
//! curve25519-dalek) currently targets it.  Plain AVX2 over the
//! existing radix-51 layout gives a smaller win than NEON-on-ARM
//! and is not on the immediate roadmap.
//!
//! # Current state
//!
//! Both backends are empty — they exist only to anchor the
//! `cfg`-dispatch pattern.  Future PRs filling in NEON intrinsics
//! add their bodies to `arch/aarch64.rs` behind
//! `cfg(target_feature = "neon")` (always-on for `aarch64`).

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)] // empty until NEON implementations land
pub(super) mod aarch64;

#[cfg(not(target_arch = "aarch64"))]
#[allow(dead_code)] // empty until called from non-aarch64 builds
pub(super) mod fallback;
