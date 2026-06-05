//! x86_64 implementations of [`Fp`][super::super::Fp] arithmetic.
//!
//! Two submodules with different roles:
//!
//! - [`avx2`] — radix-26 `Fp26` backed by AVX2 lane-packed primitives via
//!   `_mm256_mul_epu32`.  Standalone alternate-storage `Fp` backend. File-level
//!   gated on `cfg(target_feature = "avx2")`; activated as the crate-wide `Fp`
//!   by `cfg(sqisign_selkie_arch = "avx2")` in the dispatcher.
//! - [`mulx_adx`] — MULX + dual ADCX/ADOX asm leaves for the radix-51 storage
//!   in [`super::generic`].  Not a standalone backend; `generic`'s hot leaves
//!   call into these under `cfg(all(target_feature = "bmi2", target_feature =
//!   "adx"))`. Matches the structural layering of the SQIsign C reference's
//!   `gf/broadwell/lvl1/fp_asm.S` + the surrounding `fp.c` C wrappers.
//!
//! AVX-512-IFMA52 is the analogous vectorised-Fp lever but sits behind
//! a narrow Intel-server SKU subset and is omitted from AMD Zen 4;
//! sqisign-selkie does not target it.  AVX2 over radix-26 (52-bit pairs)
//! is the realistic x86_64 vectorised-Fp target.

#[allow(dead_code)] // dispatcher activation gated on cfg(sqisign_selkie_arch = "avx2")
pub mod avx2;

#[cfg(all(target_feature = "bmi2", target_feature = "adx"))]
pub mod mulx_adx;
