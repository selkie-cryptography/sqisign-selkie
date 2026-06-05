//! Architecture-specific implementations of [`Fp`][super::Fp].
//!
//! Each subdirectory holds a complete `Fp` backend specialised for a
//! target CPU architecture and instruction subset.  The active backend
//! is selected by `cfg(sqisign_selkie_arch)`, emitted by `build.rs`:
//!
//! - `generic` — radix-51 Montgomery `[u64; 5]` scalar `Fp`.  Always compiles;
//!   the fallback for any target without a specialised cfg arm set.  Hot leaves
//!   (`mul`, `square`, `sum_of_2_products`, `difference_of_2_products`)
//!   dispatch to `x86_64::mulx_adx` when `target_feature = "bmi2"` and
//!   `target_feature = "adx"` are both set.
//! - `aarch64::neon` — radix-29 `Fp` backed by `Fp29` / `Fp29x4` NEON
//!   primitives.  Activated by `cfg(sqisign_selkie_arch = "neon")`.
//! - `x86_64::avx2` — radix-26 `Fp` backed by AVX2 lane-packed primitives via
//!   `_mm256_mul_epu32`.  Activated by `cfg(sqisign_selkie_arch = "avx2")`.
//! - `x86_64::mulx_adx` — MULX + dual ADCX/ADOX asm leaves for the radix-51
//!   storage in `generic`; called from `generic`'s hot leaves under
//!   `cfg(all(target_feature = "bmi2", target_feature = "adx"))`.  Not a
//!   standalone backend.
//!
//! # No AVX-512 / IFMA52 path
//!
//! AVX-512-IFMA52 is the analogous "vectorised Fp" lever for x86_64 but
//! sits behind a narrow Intel-server CPU subset, is omitted from AMD
//! Zen 4, and isn't targeted by any major crypto library (BoringSSL,
//! libcrux, rust-openssl, curve25519-dalek).  AVX2 over a radix-26
//! layout is the realistic x86_64 vectorised-Fp target.

pub mod generic;

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)] // not yet routed; activation needs persistent Fp29 storage
pub mod aarch64;

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // submodule stubs; AVX2 Fp implementation is future
pub mod x86_64;
