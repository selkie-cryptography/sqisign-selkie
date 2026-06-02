//! aarch64 NEON backend for [`Fp`] arithmetic.
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

use subtle::{Choice, ConditionallySelectable};

use super::super::Fp;

#[cfg(test)]
mod tests;

/// Bits per limb in the radix-29 representation.
pub(super) const RADIX_29: u32 = 29;

/// Mask for a single radix-29 limb.
pub(super) const MASK_29: u32 = (1u32 << RADIX_29) - 1;

/// Number of limbs in the radix-29 representation.
///
/// Nine limbs of 29 bits each cover 261 bits, with 13 bits of headroom
/// above the 248-bit modulus.
pub(super) const LIMBS_29: usize = 9;

/// Montgomery fold multiplier: `5 · 2^16`.
///
/// At the boundary where the schoolbook column index `i ≥ 8`, the CIOS
/// Montgomery reduction adds `v[i-8] · P4_29` to the accumulator.  This is
/// equivalent (mod p) to adding `v[i-8] · 5 · 2^248` because `5 · 2^248 ≡ 1
/// (mod p)`, and within limb 8 the offset is `248 − 8·29 = 16`.
const P4_29: u32 = 5 << 16;

/// `p` in radix-29 form.
///
/// Used by [`Fp29::final_sub`] to subtract the modulus from an unreduced
/// result.  Computed from `p = 5 · 2^248 − 1`:
/// limbs 0..7 are `2^29 − 1`, limb 8 is `0x4FFFF`.
const P_LIMBS: [u32; LIMBS_29] = [
    0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF,
    0x0004FFFF,
];

/// `R²_29 mod p` where `R_29 = 2^261`.
///
/// Precomputed via `pow(2, 522, p)` and packed into 9 × 29-bit limbs.
/// Used by [`From<Fp>`] to enter Fp29 Montgomery form:
/// `canonical_value · R²_29 · R⁻¹ = canonical_value · R`.
const R2_29: Fp29 = Fp29 {
    limbs: [
        0x0CF5_C28F,
        0x0666_6666,
        0x1333_3333,
        0x1999_9999,
        0x0CCC_CCCC,
        0x0666_6666,
        0x1333_3333,
        0x1999_9999,
        0x0001_CCCC,
    ],
};

/// `1` in non-Montgomery form, used to exit Montgomery form via
/// [`Fp29::mul`]: `mont · 1 · R⁻¹ = mont / R = canonical`.
const ONE_RAW: Fp29 = Fp29 {
    limbs: [1, 0, 0, 0, 0, 0, 0, 0, 0],
};

/// Field element in radix-29 limb form, in Montgomery representation.
///
/// Parallel representation to [`Fp`]'s radix-51 layout,
/// laid out for NEON 32-bit-lane packing.  Limbs are little-endian:
/// `limbs[0]` is the least significant 29 bits.  The stored value is
/// `value · R_29 mod p` where `R_29 = 2^261`; multiplication is
/// Montgomery-style ([`Fp29::mul`] returns `a · b · R⁻¹`).
///
/// # Invariants
///
/// - After [`Fp29::mul`] or [`From<Fp>`], `limbs[i] < 2^29` for `i < 8` and
///   `limbs[8] < 2^20` (sub-`2p` bound).
/// - [`Fp29::from_bytes_le`] / [`Fp29::to_bytes_le`] operate on canonical
///   (non-Montgomery) limbs; they're the byte boundary, before/after the
///   Montgomery scaling.
///
/// # Why this is not yet wired into [`Fp`]
///
/// Switching the production path from radix-51 to scalar radix-29 is a
/// regression: 81 `u32 × u32` muls beat 25 `u64 × u64` muls only when four
/// `Fp29` products run in parallel across NEON lanes.  The current scalar
/// implementation exists to anchor cross-impl tests against `Fp`; the NEON
/// vectorised version replaces these method bodies in a follow-on commit.
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

    /// Decodes 32 bytes (little-endian) into a normalised radix-29 element.
    ///
    /// Mirrors [`Fp::from_bytes`] but stays out of Montgomery
    /// form: the limbs hold the canonical integer value, not `value · R mod p`.
    /// The input must encode a value less than `p`; out-of-range bits in
    /// `bytes[31]` simply flow into the high limb without canonicalisation.
    pub(super) fn from_bytes_le(bytes: &[u8; 32]) -> Self {
        let mut limbs = [0u32; LIMBS_29];
        let mut acc: u64 = 0;
        let mut bits: u32 = 0;
        let mut limb_idx = 0;

        for &byte in bytes.iter() {
            acc |= (byte as u64) << bits;
            bits += 8;
            if bits >= RADIX_29 && limb_idx < LIMBS_29 - 1 {
                limbs[limb_idx] = (acc as u32) & MASK_29;
                acc >>= RADIX_29;
                bits -= RADIX_29;
                limb_idx += 1;
            }
        }
        limbs[limb_idx] = acc as u32;

        Self { limbs }
    }

    /// Encodes a normalised radix-29 element as 32 bytes, little-endian.
    ///
    /// Each limb must be `< 2^29`; if the value is unsaturated the encoded
    /// bytes will overflow into adjacent positions.
    pub(super) fn to_bytes_le(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        let mut acc: u64 = 0;
        let mut bits: u32 = 0;
        let mut pos = 0;

        for &limb in self.limbs.iter() {
            acc |= (limb as u64) << bits;
            bits += RADIX_29;
            while bits >= 8 && pos < 32 {
                out[pos] = acc as u8;
                acc >>= 8;
                bits -= 8;
                pos += 1;
            }
        }
        if pos < 32 {
            out[pos] = acc as u8;
        }

        out
    }

    /// Montgomery multiplication: returns `a · b · R⁻¹ mod p`.
    ///
    /// Implements CIOS Montgomery reduction interleaved with schoolbook
    /// product over the 17 column positions of a 9×9 multiplication.  The
    /// fold step at column `i ≥ 8` adds `v[i-8] · P4_29` to absorb the
    /// previously-computed low column `v[i-8]` into the high columns,
    /// exploiting `5 · 2^248 ≡ 1 (mod p)`.
    ///
    /// Output limbs satisfy `limbs[i] < 2^29` for `i < 8` and `limbs[8] < 2^20`
    /// (so the result is in `[0, 2p)`).  Use [`Fp29::final_sub`] to
    /// canonicalise to `[0, p)`.
    pub(super) fn mul(&self, rhs: &Fp29) -> Fp29 {
        let a = &self.limbs;
        let b = &rhs.limbs;
        let mut t: u64 = 0;
        let mut v = [0u32; LIMBS_29];
        let mut c = [0u32; LIMBS_29];

        for i in 0..(2 * LIMBS_29 - 1) {
            let j_lo = if i >= LIMBS_29 { i - LIMBS_29 + 1 } else { 0 };
            let j_hi = i.min(LIMBS_29 - 1);
            for j in j_lo..=j_hi {
                t = t.wrapping_add((a[j] as u64).wrapping_mul(b[i - j] as u64));
            }

            if i >= LIMBS_29 - 1 {
                let fold_idx = i - (LIMBS_29 - 1);
                t = t.wrapping_add((v[fold_idx] as u64).wrapping_mul(P4_29 as u64));
            }

            let limb = (t as u32) & MASK_29;
            if i < LIMBS_29 {
                v[i] = limb;
            } else {
                c[i - LIMBS_29] = limb;
            }
            t >>= RADIX_29;
        }

        c[LIMBS_29 - 1] = t as u32;

        Self { limbs: c }
    }

    /// Conditionally subtracts `p` to canonicalise an in-range result.
    ///
    /// Assumes `self < 2p` with each limb already `< 2^29`.  Returns the
    /// representative in `[0, p)`.  Constant-time via
    /// [`subtle::ConditionallySelectable`].
    pub(super) fn final_sub(self) -> Self {
        let mut diff = [0u32; LIMBS_29];
        let mut borrow: u32 = 0;

        for i in 0..LIMBS_29 {
            let d = (self.limbs[i] as i64) - (P_LIMBS[i] as i64) - (borrow as i64);
            diff[i] = (d as u32) & MASK_29;
            borrow = ((d as u64) >> 63) as u32 & 1;
        }

        // borrow == 0 ⇒ subtraction succeeded (self ≥ p), use diff.
        // borrow == 1 ⇒ self < p, keep self.
        let take_diff = Choice::from((1 - borrow) as u8);
        let mut out = [0u32; LIMBS_29];
        for i in 0..LIMBS_29 {
            out[i] = u32::conditional_select(&self.limbs[i], &diff[i], take_diff);
        }

        Self { limbs: out }
    }

    /// Exits Montgomery form: `mont → mont / R = canonical`.
    ///
    /// Multiplies by `1` in non-Montgomery form ([`ONE_RAW`]); the Montgomery
    /// product is `mont · 1 · R⁻¹ = mont / R`.  Then canonicalises via
    /// [`Self::final_sub`].
    pub(super) fn reduce_montgomery(self) -> Self {
        self.mul(&ONE_RAW).final_sub()
    }
}

impl From<Fp> for Fp29 {
    /// Converts radix-51 Montgomery form to radix-29 Montgomery form.
    ///
    /// Routes through canonical bytes: [`Fp::to_bytes`] exits
    /// the radix-51 Montgomery scaling, [`Fp29::from_bytes_le`] repacks the
    /// integer value at radix-29, then multiplication by [`R2_29`] enters the
    /// radix-29 Montgomery form (`canonical · R²_29 · R⁻¹ = canonical · R`).
    /// Expensive (two Montgomery reductions); intended for test boundaries.
    fn from(fp: Fp) -> Self {
        Self::from_bytes_le(&fp.to_bytes()).mul(&R2_29)
    }
}

impl From<Fp29> for Fp {
    /// Converts radix-29 Montgomery form back to radix-51 Montgomery form.
    ///
    /// Symmetric to [`From<Fp> for Fp29`]: drops the radix-29
    /// Montgomery scaling via [`Fp29::reduce_montgomery`], emits canonical
    /// bytes, then runs [`Fp::from_bytes`] to enter the radix-51
    /// Montgomery form.
    fn from(fp29: Fp29) -> Self {
        Self::from_bytes(&fp29.reduce_montgomery().to_bytes_le())
    }
}

const _: () = {
    assert!(MASK_29 == (1u32 << RADIX_29) - 1);
    assert!(RADIX_29 as usize * LIMBS_29 >= 248);
};
