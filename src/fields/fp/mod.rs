//! Field arithmetic modulo [p = 5 · 2²⁴⁸ − 1][§2.1].
//!
//! `Fp` is a concrete type re-exported from one of the architecture-
//! specific implementations under [`arch`].  All higher-level code in
//! the crate (curves, points, isogenies, Fp², quaternions, signing)
//! interacts with `Fp` through this re-export and is backend-agnostic;
//! swapping in a vectorised `Fp` later doesn't ripple beyond the
//! `arch` module.
//!
//! The generic backend (`arch::generic`) is the always-available
//! radix-51 Montgomery `[u64; 5]` implementation matching the SQIsign
//! C reference.  `arch::aarch64::neon` (radix-29 NEON-vectorised) and
//! `arch::x86_64::avx2` (radix-26 AVX2-vectorised) provide alternate
//! storage layouts selected via `cfg(sqisign_selkie_arch)` emitted by
//! `build.rs`.  On x86_64 with `target_feature = "bmi2"` and
//! `target_feature = "adx"`, the generic backend's hot leaves dispatch
//! to `arch::x86_64::mulx_adx`'s MULX + dual ADCX/ADOX asm without
//! changing the radix-51 storage.  All backends
//! use Montgomery form per `p = β · 2^α − 1`'s structure (β = 5, α = 248).
//!
//! [§2.1]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.1
//! [§8.1]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.1

// `arch` visibility tracks two orthogonal axes:
//
//   - `expose-internals` feature → fully `pub` (for benches and external
//     differential-test crates that need to reach `Fp29` / `Fp29x4`).
//   - `sqisign_selkie_arch = "neon" | "avx2"` cfg (set by `build.rs` when the
//     corresponding vectorised `Fp` backend is expected to win) → `pub(crate)`
//     so higher-level callers inside this crate can opt in without requiring
//     the feature.
//   - Otherwise → private, since the alternate-radix layouts are implementation
//     detail and nothing currently routes through them.
#[cfg(all(
    not(feature = "expose-internals"),
    any(sqisign_selkie_arch = "neon", sqisign_selkie_arch = "avx2")
))]
pub(crate) mod arch;

#[cfg(all(
    not(feature = "expose-internals"),
    not(any(sqisign_selkie_arch = "neon", sqisign_selkie_arch = "avx2"))
))]
mod arch;

#[cfg(feature = "expose-internals")]
pub mod arch;

#[cfg(test)]
mod tests;

/// Number of bytes in a canonical encoding of an element of F_p.
pub const FP_ENCODED_BYTES: usize = 32;

// `Fp` re-export: scalar backend picked by `cfg(target_feature)`
// directly (no `build.rs` indirection).  The active backend is the
// best scalar `Fp` available on the target -- `Fp64` MULX+ADX asm
// on x86_64+bmi2+adx, else the always-available `Fp51`.
//
// Each non-`Fp51` backend has a `from_limbs([u64; 5])` const-bridge
// that accepts `Fp51`'s radix-51 Montgomery limbs and const-converts
// at compile time, so precomputed-constant tables in `params.rs`
// and `deuring/precomputed.rs` are signature-compatible across all
// backends.
//
// Batch helpers (`Fp26x4` / `Fp29x4`) compile in independently
// when their ISA is available; see [`batch`] below.  They are not
// the active `Fp` -- per-op single-lane SIMD loses to scalar `Fp64`
// asm on Sapphire and loses to scalar `Fp51` on M4.  Call sites
// that need 4-Fp-at-once batches reach for the batch type
// explicitly.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx",
)))]
pub use arch::generic::Fp51 as Fp;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx",
))]
pub use arch::x86_64::mulx_adx::Fp64 as Fp;

/// Batch helpers for call sites that want to process 4 `Fp`
/// values in parallel.  Compiled in when the corresponding SIMD ISA
/// is available; not the active scalar `Fp`.
///
/// Each batch type has `from_active_fps([Fp; 4]) -> Self` and
/// `into_active_fps(self) -> [Fp; 4]` conversion at the boundary
/// (paid once per batch entry/exit, amortized over the batch ops).
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
))]
pub mod batch {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[allow(unused_imports)] // no in-crate callers yet; persistent-batch refactor
    pub use super::arch::aarch64::neon::Fp29x4;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[allow(unused_imports)] // no in-crate callers yet; persistent-batch refactor
    pub use super::arch::x86_64::avx2::Fp26x4;
}
