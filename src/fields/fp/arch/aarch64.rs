//! aarch64 NEON backend for [`Fp`][super::super::Fp] arithmetic.
//!
//! Future home for the vectorised `Fp` implementation of De Feo,
//! Jian, Wang, Yang ([ePrint 2026/394][2026-394], CHES 2026), adapted
//! from the SQIsign C reference's NEON port.
//!
//! - **Limb layout**: [`Fp29`] holds nine 29-bit unsaturated limbs in 32-bit
//!   lanes.  Nine limbs cover the 248-bit modulus with 13 bits of per-limb
//!   carry headroom, enough for a chain of additions before normalisation.
//! - **SIMD packing**: 4 elements share a 9-vector of `uint32x4_t`, one limb
//!   per lane.  One `vmlal_u32` schoolbook step computes the same limb position
//!   for four independent products.
//! - **Multiplication**: schoolbook `Fp × Fp` with `vmlal_u32` (multiply-
//!   accumulate widening to 64-bit lanes), interleaved with Montgomery-style
//!   reduction via the `p = 5·2^248 − 1` structure.
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
//! # Why nine 29-bit limbs rather than eight 31-bit limbs
//!
//! NEON's widening multiply-accumulate is `u32 × u32 → u64`.  With 31-bit
//! limbs the accumulator only has 2 bits of headroom before the upper
//! `u64` lane overflows, leaving no slack for the Montgomery cross-terms
//! that fold `P4 = 5·2^44` into limb positions.  With 29-bit limbs the
//! accumulator has 6 bits of headroom: enough for the schoolbook column
//! and the two `× P4` cross-terms without an interleaved normalisation.
//!
//! # Constant-time
//!
//! All NEON arithmetic is data-flow only; no data-dependent
//! branches or memory access patterns.  Per-microarchitecture CT
//! audit is required because NEON multiplier latencies vary
//! between Apple Silicon revisions (M1 / M2 / M3 / M4) and across
//! Cortex-A cores.
//!
//! [2026-394]: https://eprint.iacr.org/2026/394.pdf

/// Bits per limb in the radix-29 representation.
pub(super) const RADIX_29: u32 = 29;

/// Mask for a single radix-29 limb.
pub(super) const MASK_29: u32 = (1u32 << RADIX_29) - 1;

/// Number of limbs in the radix-29 representation.
///
/// Nine limbs of 29 bits each cover 261 bits, with 13 bits of headroom
/// above the 248-bit modulus.
pub(super) const LIMBS_29: usize = 9;

/// Field element in radix-29 unsaturated limb form.
///
/// Parallel representation to [`super::super::Fp`]'s radix-51 layout,
/// laid out for NEON 32-bit-lane packing.  Limbs are little-endian:
/// `limbs[0]` is the least significant 29 bits.  Unsaturated: each
/// limb may briefly carry more than `2^29` while a chain of operations
/// is in flight; normalisation happens at boundaries that require it
/// (`to_bytes`, equality, square-root, …).
///
/// # Invariants
///
/// - In *normalised* form, each `limbs[i] < 2^29`.  The high limb `limbs[8]`
///   satisfies the additional bound implied by the modulus `p = 5 · 2^248 − 1`.
/// - In *unsaturated* form, each `limbs[i] < 2^29 + ε` where `ε` is bounded by
///   the depth of the operation chain since last normalisation.
///
/// The arithmetic methods that consume and produce `Fp29` document
/// which form they accept and produce.  Public conversion via
/// [`super::super::Fp`] always normalises.
///
/// # Why this is not yet wired into [`super::super::Fp`]
///
/// The NEON arithmetic methods on `Fp29` are unwritten.  Until they
/// exist, switching the production path from radix-51 to scalar
/// radix-29 would be a regression: 81 `u32 × u32` muls beats 25
/// `u64 × u64` muls only when four `Fp29` products run in parallel
/// across NEON lanes.
#[derive(Clone, Copy, Debug)]
pub(super) struct Fp29 {
    /// Nine 29-bit limbs, little-endian.
    pub(super) limbs: [u32; LIMBS_29],
}

impl Fp29 {
    /// Additive identity, in normalised form.
    pub(super) const ZERO: Self = Self {
        limbs: [0; LIMBS_29],
    };
}

const _: () = {
    assert!(MASK_29 == (1u32 << RADIX_29) - 1);
    assert!(RADIX_29 as usize * LIMBS_29 >= 248);
};
