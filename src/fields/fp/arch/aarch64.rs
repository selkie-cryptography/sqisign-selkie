//! aarch64 NEON backend for [`Fp`][super::super::Fp] arithmetic.
//!
//! Future home for the vectorised `Fp` implementation of De Feo,
//! Jian, Wang, Yang (ePrint 2026/394, CHES 2026).  Their design,
//! adapted from the SQIsign C reference's NEON port:
//!
//! - **Limb layout**: radix-29 unsaturated limbs packed into NEON 32-bit lanes,
//!   four `Fp` elements per 128-bit vector.  Nine 29-bit limbs cover the
//!   248-bit modulus with two bits of per-limb carry headroom for chains of
//!   additions before normalisation.
//! - **Multiplication**: schoolbook `Fp × Fp` with `vmlal_u32`
//!   (multiply-accumulate widening to 64-bit lanes), interleaved with
//!   Montgomery-style reduction via the `p = 5·2^248 − 1` structure.
//! - **Karatsuba `Fp²`**: composed at the [`crate::fields::fp2`] level over
//!   vectorised `Fp` muls; already 3M+5A and stays so.
//! - **Lazy reduction**: limbs are normalised only at boundaries where
//!   downstream code requires it (e.g. before `to_bytes`), not after every
//!   add/sub.
//!
//! Reported speedup on Apple M1: 1.22× total signing.  On
//! Cortex-A76 the same code gets 1.48–1.52× because more of the
//! workload is multiplier-bound on the in-order core.
//!
//! # Constant-time
//!
//! All NEON arithmetic is data-flow only; no data-dependent
//! branches or memory access patterns.  Per-microarchitecture CT
//! audit is required because NEON multiplier latencies vary
//! between Apple Silicon revisions (M1 / M2 / M3 / M4) and across
//! Cortex-A cores.
