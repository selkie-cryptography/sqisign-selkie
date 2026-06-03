//! Architecture-specific implementations of [`Fp`][super::Fp].
//!
//! Each subdirectory holds one or more complete `Fp` implementations
//! specialised for a target CPU architecture and instruction set:
//!
//! - [`aarch64::neon`] — radix-29 `Fp` backed by `Fp29` / `Fp29x4` NEON
//!   primitives.  Active on aarch64 hosts that are not wider-scalar-pipe Apple
//!   Silicon (M2 and later), where the build script sets
//!   `cfg(sqisign_selkie_arch = "neon")`.  Currently only the primitives land
//!   here; the full `Fp` wrapper (invert, sqrt, etc) follows in later commits.
//! - `x86_64::avx2` — *future*, radix-26 `Fp` backed by AVX2 lane-packed
//!   primitives via `_mm256_mul_epu32`.  Gated on `cfg(all(target_arch =
//!   "x86_64", target_feature = "avx2"))` and selected by
//!   `cfg(sqisign_selkie_arch = "avx2")`.
//!
//! The remaining cross-architecture portable scalar `Fp` (radix-51,
//! `[u64; 5]`) still lives in the parent [`super`] module; it will move
//! to `arch/portable.rs` once the higher-level callers are demonstrated
//! to be backend-agnostic.
//!
//! # No AVX-512 / IFMA52 path
//!
//! AVX-512-IFMA52 is the analogous "vectorised Fp" lever for x86_64 but
//! sits behind a narrow Intel-server CPU subset, is omitted from AMD
//! Zen 4, and isn't targeted by any major crypto library (BoringSSL,
//! libcrux, rust-openssl, curve25519-dalek).  AVX2 over a radix-26
//! layout is the realistic x86_64 vectorised-Fp target.

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)] // not yet routed; activation needs persistent Fp29 storage
pub mod aarch64;
