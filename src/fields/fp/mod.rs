//! Field arithmetic modulo [p = 5 · 2²⁴⁸ − 1][§2.1].
//!
//! `Fp` is a concrete type re-exported from one of the architecture-
//! specific implementations under [`arch`].  All higher-level code in
//! the crate (curves, points, isogenies, Fp², quaternions, signing)
//! interacts with `Fp` through this re-export and is backend-agnostic;
//! swapping in a vectorised `Fp` later doesn't ripple beyond the
//! `arch` module.
//!
//! The portable backend (`arch::portable`) is the always-available
//! radix-51 Montgomery `[u64; 5]` implementation matching the SQIsign
//! C reference.  Future commits add `arch::aarch64::neon` (radix-29
//! NEON-vectorised) and `arch::x86_64::avx2` (radix-26 AVX2-vectorised)
//! complete-`Fp` backends selectable via `cfg(sqisign_selkie_arch)`
//! emitted by `build.rs`.  All backends use Montgomery form per
//! `p = β · 2^α − 1`'s structure (β = 5, α = 248).
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

// `Fp` re-export: today always the portable backend.  The future
// `cfg(sqisign_selkie_arch = "neon")` / `"avx2"` arms light up once
// the corresponding `arch::aarch64::neon::Fp` / `arch::x86_64::avx2::Fp`
// types exist (currently the aarch64/x86_64 submodules hold primitives
// and stubs, not a full `Fp`).
pub use arch::portable::Fp;
