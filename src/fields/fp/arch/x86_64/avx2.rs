//! AVX2-vectorised `Fp` implementation, future.
//!
//! Planned design: radix-26 packed `[u32; 10]` (or `[u32; 8]` in two
//! `__m256i` halves) where each u32 lane holds 26 bits of the
//! 248-bit modulus.  Multiplication uses `_mm256_mul_epu32` (8-lane
//! `u32 * u32 -> u64`, throughput-1 on Haswell+) over a Karatsuba-
//! decomposed schoolbook, mirroring the structure of
//! [`arch::aarch64::neon::Fp29x4`].
//!
//! Activated when `build.rs` emits `cfg(sqisign_selkie_arch = "avx2")`
//! and the target carries `cfg(target_feature = "avx2")`.  Both gates
//! are belt-and-suspenders: the file-level cfg in [`super`] prevents
//! the AVX2 intrinsics from being instantiated on non-AVX2 builds,
//! and the dispatcher's `sqisign_selkie_arch = "avx2"` arm guards the
//! `pub use` of the resulting `Fp`.
//!
//! Currently empty; the full Fp implementation lands in a future
//! commit.  See [`crate::fields::fp::arch`] for the parallel
//! `aarch64::neon` work that this module's structure mirrors.
