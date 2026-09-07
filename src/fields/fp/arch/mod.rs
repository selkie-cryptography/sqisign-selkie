//! Architecture-specific implementations of [`Fp`][super::Fp].
//!
//! Each subdirectory holds a complete `Fp` backend specialised for a
//! target CPU architecture and instruction subset.  The active backend
//! is selected by `cfg(sqisign_selkie_arch)`, emitted by `build.rs`:
//!
//! - `generic` -- radix-2^55 Montgomery `[u64; 6]` scalar `Fp55`.  Always
//!   compiles; the fallback for any target without a specialised cfg arm set.
//!   Source of truth for `pub const FOO: Fp = Fp::from_limbs([...])`
//!   precomputed tables; every other backend's `from_limbs([u64; 6])`
//!   const-bridge accepts `Fp55`'s radix-55 limbs and const-converts.
//! - `x86_64::mulx_adx` -- radix-2^64 packed `Fp64([u64; 6])`.  Standalone
//!   scalar `Fp` backend with MULX + dual-chain ADCX/ADOX asm leaves.  Active
//!   on x86_64 builds with `target_feature = "bmi2"` + `target_feature = "adx"`
//!   (Broadwell 2014+).  Matches the C ref's `gf/broadwell/p324_3` storage.
//! - `aarch64::neon` -- radix-2^29 `Fp29` (scalar, 12 limbs) + `Fp29x4` (4-wide
//!   NEON batch).  Activated by `cfg(sqisign_selkie_arch = "neon")`.
//! - `x86_64::avx2` -- radix-2^26 `Fp26` (scalar, 13 limbs) + `Fp26x4` (4-wide
//!   AVX2 batch).  Activated by `cfg(sqisign_selkie_arch = "avx2")`.
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
