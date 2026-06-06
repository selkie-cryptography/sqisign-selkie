//! x86_64 backends for [`Fp`][super::super::Fp] arithmetic.
//!
//! Two submodules, each with a distinct storage layout chosen to fit
//! a different x86_64 instruction subset:
//!
//! - `mulx_adx`: radix-2^64 `Fp64([u64; 4])` with MULX + dual ADCX/ADOX asm.
//! - [`avx2`]: radix-2^26 `Fp26x4` 4-wide batch with `_mm256_mul_epu32`.
//!
//! `mulx_adx` matches the C ref's `gf/broadwell/lvl1/gf5248.c` storage
//! exactly.  Library code only at present -- the dispatcher in
//! `super::super` does NOT activate Fp64 as the crate-wide `Fp` yet;
//! higher-level isogeny tests surfaced correctness issues that need
//! debugging on x86_64 before the dispatcher can flip.  Module is
//! cfg-gated on `target_feature = "bmi2"` + `target_feature = "adx"`;
//! not in scope on doc builds without them.
//!
//! `avx2` is not a standalone active `Fp` backend; exposed as a batch
//! helper for call sites that explicitly opt into 4-Fp-at-once storage.
//! File-level gated on `cfg(target_feature = "avx2")`.
//!
//! AVX-512-IFMA52 is the analogous vectorised-Fp lever but sits behind
//! a narrow Intel-server SKU subset and is omitted from AMD Zen 4;
//! sqisign-selkie does not target it.  AVX2 over radix-2^26 (52-bit
//! pairs via VPMULUDQ) is the realistic x86_64 target.

#[allow(dead_code)] // dispatcher activation gated on cfg(sqisign_selkie_arch = "avx2")
pub mod avx2;

#[cfg(all(target_feature = "bmi2", target_feature = "adx"))]
pub mod mulx_adx;
