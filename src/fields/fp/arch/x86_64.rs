//! x86_64 implementations of [`Fp`][super::super::Fp] arithmetic.
//!
//! Each submodule is a complete `Fp` backend specialised for an x86_64
//! instruction subset.  The arch dispatcher in [`super`] picks one of
//! these via `cfg(sqisign_selkie_arch = "...")` set by `build.rs`.
//!
//! Currently no backend is filled in — [`avx2`] is the planned target
//! and is gated on `cfg(target_feature = "avx2")` (Haswell 2013+, Zen
//! and later AMD).
//!
//! AVX-512-IFMA52 is the analogous vectorised-Fp lever but sits behind
//! a narrow Intel-server SKU subset and is omitted from AMD Zen 4;
//! sqisign-selkie does not target it.  AVX2 over radix-26 (52-bit pairs)
//! is the realistic x86_64 vectorised-Fp target.

#[cfg(target_feature = "avx2")]
#[allow(dead_code)] // module is a stub until the AVX2 Fp lands
pub mod avx2;
