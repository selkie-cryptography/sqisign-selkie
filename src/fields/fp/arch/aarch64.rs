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
//! NEON's widening multiply-accumulate is `u32 * u32 -> u64`.  With 31-bit
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

use core::{
    arch::aarch64::{
        uint32x4_t, uint64x2_t, vaddq_s64, vaddq_u32, vaddq_u64, vandq_u32, vdupq_n_u32,
        vdupq_n_u64, vget_high_s32, vget_low_s32, vget_low_u32, vld1q_u32, vmlal_high_u32,
        vmlal_u32, vmovl_s32, vmovn_high_u64, vmovn_u64, vreinterpretq_s32_u32,
        vreinterpretq_u32_s32, vreinterpretq_u64_s64, vshrq_n_s32, vshrq_n_s64, vshrq_n_u64,
        vst1q_u32, vsubq_u32, vsubq_u64,
    },
    ops::{Add, Sub},
};

use subtle::{Choice, ConditionallySelectable};

use super::super::Fp;

#[cfg(test)]
mod tests;

/// Bits per limb in the radix-29 representation.
pub const RADIX_29: u32 = 29;

/// Mask for a single radix-29 limb.
pub const MASK_29: u32 = (1u32 << RADIX_29) - 1;

/// Number of limbs in the radix-29 representation.
///
/// Nine limbs of 29 bits each cover 261 bits, with 13 bits of headroom
/// above the 248-bit modulus.
pub const LIMBS_29: usize = 9;

/// Montgomery fold multiplier: `5 · 2^16`.
///
/// At the boundary where the schoolbook column index `i ≥ 8`, the interleaved
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
pub struct Fp29 {
    /// Nine 29-bit limbs, little-endian.
    pub limbs: [u32; LIMBS_29],
}

impl Fp29 {
    /// Additive identity, in normalised form.
    pub const ZERO: Self = Self {
        limbs: [0; LIMBS_29],
    };

    /// Decodes 32 bytes (little-endian) into a normalised radix-29 element.
    ///
    /// Mirrors [`Fp::from_bytes`] but stays out of Montgomery
    /// form: the limbs hold the canonical integer value, not `value · R mod p`.
    /// The input must encode a value less than `p`; out-of-range bits in
    /// `bytes[31]` simply flow into the high limb without canonicalisation.
    pub fn from_bytes_le(bytes: &[u8; 32]) -> Self {
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
    pub fn to_bytes_le(self) -> [u8; 32] {
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
    /// 9×9 schoolbook product with Montgomery reduction interleaved column-
    /// by-column over the 17 output positions.  The fold step at column
    /// `i ≥ 8` adds `v[i-8] · P4_29`, exploiting `5 · 2^248 ≡ 1 (mod p)`
    /// to absorb the previously-computed low column into the high columns.
    ///
    /// Output limbs satisfy `limbs[i] < 2^29` for `i < 8` and `limbs[8] < 2^20`
    /// (so the result is in `[0, 2p)`).  Use [`Fp29::final_sub`] to
    /// canonicalise to `[0, p)`.
    pub fn mul(&self, rhs: &Fp29) -> Fp29 {
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

    /// Squares this element via [`Fp29::mul`].
    ///
    /// The optimised radix-29 square (symmetric cross-terms, `2 · a[i] · a[j]`)
    /// is deferred to the NEON-intrinsics commit, where the symmetry
    /// translates to fewer vectorised products.
    pub fn square(&self) -> Fp29 {
        self.mul(self)
    }

    /// Propagates carries through the limbs, returning a sign mask:
    /// `0` if the final accumulator was non-negative, `0xFFFFFFFF` if it
    /// was negative (indicating a borrow occurred upstream).
    ///
    /// Mirrors [`Fp::prop`]: arithmetic right-shift on an `i64` carry
    /// preserves the sign, and the high bit of `limbs[LIMBS_29 - 1]` after
    /// the final wrapping add encodes whether the cumulative value
    /// overflowed (borrowed).
    ///
    /// The cast chain `u32 -> i32 -> i64` is load-bearing: `u32 -> i64`
    /// zero-extends and would lose the borrow sign, while `u32 -> i32`
    /// preserves bits (same width) and `i32 -> i64` then sign-extends.
    fn prop(&mut self) -> u32 {
        let mut carry = (self.limbs[0] as i32) as i64;
        carry >>= RADIX_29;
        self.limbs[0] &= MASK_29;
        for i in 1..LIMBS_29 - 1 {
            carry += (self.limbs[i] as i32) as i64;
            self.limbs[i] = (carry as u32) & MASK_29;
            carry >>= RADIX_29;
        }
        self.limbs[LIMBS_29 - 1] = self.limbs[LIMBS_29 - 1].wrapping_add(carry as u32);
        let sign = (self.limbs[LIMBS_29 - 1] >> 1) >> 30;
        sign.wrapping_neg()
    }

    /// Conditionally subtracts `p` to canonicalise an in-range result.
    ///
    /// Assumes `self < 2p` with each limb already `< 2^29`.  Returns the
    /// representative in `[0, p)`.  Constant-time via
    /// [`subtle::ConditionallySelectable`].
    pub fn final_sub(self) -> Self {
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

    /// Exits Montgomery form: `mont -> mont / R = canonical`.
    ///
    /// Multiplies by `1` in non-Montgomery form ([`ONE_RAW`]); the Montgomery
    /// product is `mont · 1 · R⁻¹ = mont / R`.  Then canonicalises via
    /// [`Self::final_sub`].
    pub fn reduce_montgomery(self) -> Self {
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

/// Four [`Fp29`] elements packed into NEON 32-bit-lane Structure-of-Arrays
/// (SoA) layout: the four elements are interleaved lane-wise so one
/// `vmlal_u32` computes the same schoolbook column across all four products
/// simultaneously.  Each of the nine `uint32x4_t` vectors holds the `i`-th
/// limb of four independent field elements at lanes 0..3.
///
/// Per [ePrint 2026/394][2026-394], this layout amortises the 81 `u32 * u32`
/// scalar muls of one `Fp29` product down to ~20 NEON ops on Cortex-A76 /
/// Neoverse N1 cores.
///
/// # Safety
///
/// Public methods are safe.  Internal `unsafe` blocks wrap aarch64 NEON
/// intrinsics, which are `unsafe fn` purely as a platform gate; NEON is
/// part of the aarch64 base ISA, so `cfg(target_arch = "aarch64")` already
/// satisfies `target_feature = "neon"` with no runtime detection.  Per-block
/// `SAFETY:` comments cover data-flow obligations.
///
/// [2026-394]: https://eprint.iacr.org/2026/394.pdf
#[derive(Clone, Copy)]
pub struct Fp29x4 {
    /// Nine NEON 4-lane vectors.  Lane `j` of `limbs[i]` is the `i`-th radix-29
    /// limb of the `j`-th field element of the batch.
    pub limbs: [uint32x4_t; LIMBS_29],
}

impl Fp29x4 {
    /// Packs four scalar [`Fp29`] elements into the SoA layout via a
    /// per-limb gather: `limbs[i]` ends up holding
    /// `[elements[0].limbs[i], ..., elements[3].limbs[i]]`.
    pub fn from_scalars(elements: &[Fp29; 4]) -> Self {
        // SAFETY: vdupq_n_u32 and vld1q_u32 require the aarch64+neon target
        // feature, which is part of the aarch64 base ISA and therefore always
        // available where this `cfg(target_arch = "aarch64")` module compiles.
        // The lane buffer is a stack-local `[u32; 4]` whose pointer is
        // guaranteed properly aligned for the NEON load.
        let zero = unsafe { vdupq_n_u32(0) };
        let mut limbs = [zero; LIMBS_29];
        for (i, slot) in limbs.iter_mut().enumerate() {
            let lane = [
                elements[0].limbs[i],
                elements[1].limbs[i],
                elements[2].limbs[i],
                elements[3].limbs[i],
            ];
            *slot = unsafe { vld1q_u32(lane.as_ptr()) };
        }
        Self { limbs }
    }

    /// Unpacks the SoA layout back into four scalar [`Fp29`] elements.
    /// Inverse of [`Fp29x4::from_scalars`].
    pub fn to_scalars(self) -> [Fp29; 4] {
        // SAFETY: see `from_scalars`.  The lane buffer is a stack-local
        // `[u32; 4]` properly aligned for the NEON store.
        let mut out = [Fp29::ZERO; 4];
        let mut lane = [0u32; 4];
        for (i, &limb) in self.limbs.iter().enumerate() {
            unsafe { vst1q_u32(lane.as_mut_ptr(), limb) };
            for (j, &v) in lane.iter().enumerate() {
                out[j].limbs[i] = v;
            }
        }
        out
    }

    /// Vectorised Montgomery multiplication: returns
    /// `[a[0] * b[0] * R^-1, ..., a[3] * b[3] * R^-1]` packed in SoA form.
    ///
    /// Karatsuba-decomposed: splits each 9-limb input as
    /// `a = a_lo + a_hi * 2^(5 * 29)` with `a_lo = a[0..5]` (5 limbs) and
    /// `a_hi = a[5..9]` (4 limbs), computes three sub-products
    ///
    /// - `P0 = a_lo * b_lo` (9 columns, plain 5x5).
    /// - `P1 = a_hi * b_hi` (7 columns, plain 4x4).
    /// - `Q  = (a_lo + a_hi) * (b_lo + b_hi)` (9 columns, plain 5x5).
    ///
    /// then assembles the 17-column polynomial
    /// `P0 + (Q - P0 - P1) * x^5 + P1 * x^10` and Montgomery-reduces it in a
    /// single pass.
    ///
    /// 75 multiply-accumulates total (25 + 16 + 25 sub-product plus 9 fold)
    /// against 90 for the straight 9x9 schoolbook.  The three sub-products
    /// run with independent carry chains and pipeline across NEON execution
    /// ports, where the schoolbook serialises on one chain.
    /// On Cortex-A76 / Neoverse N1 hardware (matched in the 2026/394 paper
    /// numbers) this path is ~1.4x faster than four scalar `Fp::mul`s.  On
    /// wider scalar cores (Apple M2+) the radix-29 mul-count tax exceeds the
    /// NEON parallelism win and scalar `Fp` remains faster; `build.rs` gates
    /// production routing accordingly.
    pub fn mul(&self, rhs: &Fp29x4) -> Fp29x4 {
        let a = &self.limbs;
        let b = &rhs.limbs;

        // SAFETY: register-width NEON ops; covered by the type-level Safety
        // note.  Lane buffer ranges are within `LIMBS_29 = 9`.
        unsafe {
            let zero_u32 = vdupq_n_u32(0);
            let zero_u64 = vdupq_n_u64(0);
            let mask_u32 = vdupq_n_u32(MASK_29);
            let p4_vec = vdupq_n_u32(P4_29);

            let a_lo = [a[0], a[1], a[2], a[3], a[4]];
            let b_lo = [b[0], b[1], b[2], b[3], b[4]];
            let a_hi = [a[5], a[6], a[7], a[8]];
            let b_hi = [b[5], b[6], b[7], b[8]];

            // a_sum / b_sum: a_lo + a_hi padded to 5 limbs.  Each sum-limb
            // remains under 2^30 (< 2^29 + < 2^29).
            let a_sum = [
                vaddq_u32(a_lo[0], a_hi[0]),
                vaddq_u32(a_lo[1], a_hi[1]),
                vaddq_u32(a_lo[2], a_hi[2]),
                vaddq_u32(a_lo[3], a_hi[3]),
                a_lo[4],
            ];
            let b_sum = [
                vaddq_u32(b_lo[0], b_hi[0]),
                vaddq_u32(b_lo[1], b_hi[1]),
                vaddq_u32(b_lo[2], b_hi[2]),
                vaddq_u32(b_lo[3], b_hi[3]),
                b_lo[4],
            ];

            let p0 = Self::polynomial_5x5(&a_lo, &b_lo);
            let p1 = Self::polynomial_4x4(&a_hi, &b_hi);
            let q = Self::polynomial_5x5(&a_sum, &b_sum);

            // mid[i] = q[i] - p0[i] - p1[i] (p1 zero-padded above index 6).
            // No underflow by Karatsuba algebra: q >= p0 + p1.
            let mut mid: [(uint64x2_t, uint64x2_t); 9] = [(zero_u64, zero_u64); 9];
            for i in 0..9 {
                let (p1_lo, p1_hi) = if i < 7 { p1[i] } else { (zero_u64, zero_u64) };
                mid[i].0 = vsubq_u64(vsubq_u64(q[i].0, p0[i].0), p1_lo);
                mid[i].1 = vsubq_u64(vsubq_u64(q[i].1, p0[i].1), p1_hi);
            }

            // 17-column polynomial: P0 at offset 0, mid at offset 5, P1 at offset 10.
            let mut full: [(uint64x2_t, uint64x2_t); 17] = [(zero_u64, zero_u64); 17];
            full[..9].copy_from_slice(&p0);
            for i in 0..9 {
                full[5 + i].0 = vaddq_u64(full[5 + i].0, mid[i].0);
                full[5 + i].1 = vaddq_u64(full[5 + i].1, mid[i].1);
            }
            for i in 0..7 {
                full[10 + i].0 = vaddq_u64(full[10 + i].0, p1[i].0);
                full[10 + i].1 = vaddq_u64(full[10 + i].1, p1[i].1);
            }

            // Montgomery-reduce the 17-column polynomial: for i in 0..9 extract
            // v[i] = low 29 bits and fold v[i] * P4_29 into column i+8; for
            // i in 9..17 extract output limbs c[0..7].  Then c[8] = the carry
            // out of column 16 (mirrors the schoolbook's post-loop `c[8] = t`).
            let mut out = [zero_u32; LIMBS_29];
            for i in 0..17 {
                let limb_pair = vmovn_u64(full[i].0);
                let limb_full = vmovn_high_u64(limb_pair, full[i].1);
                let v = vandq_u32(limb_full, mask_u32);

                if i < 9 {
                    full[i + 8].0 = vmlal_u32(full[i + 8].0, vget_low_u32(v), vget_low_u32(p4_vec));
                    full[i + 8].1 = vmlal_high_u32(full[i + 8].1, v, p4_vec);
                }

                if i + 1 < 17 {
                    full[i + 1].0 = vaddq_u64(full[i + 1].0, vshrq_n_u64::<29>(full[i].0));
                    full[i + 1].1 = vaddq_u64(full[i + 1].1, vshrq_n_u64::<29>(full[i].1));
                }

                if (9..17).contains(&i) {
                    out[i - 9] = v;
                }
            }

            // c[8]: the carry that would have propagated to column 17.
            let final_carry_lo = vshrq_n_u64::<29>(full[16].0);
            let final_carry_hi = vshrq_n_u64::<29>(full[16].1);
            let final_pair = vmovn_u64(final_carry_lo);
            out[LIMBS_29 - 1] = vmovn_high_u64(final_pair, final_carry_hi);

            Fp29x4 { limbs: out }
        }
    }

    /// Propagates carries across the nine limbs in vectorised SoA form.
    /// Returns a per-lane mask: `0` if the cumulative sum at that lane was
    /// non-negative, all-ones if negative (indicating an upstream borrow).
    ///
    /// Lane-parallel version of [`Fp29::prop`].  The `u32 -> i32 -> i64`
    /// sign-extension cast chain becomes `vreinterpretq_s32_u32` followed by
    /// `vmovl_s32` on each half of the lane vector; arithmetic right shift
    /// (`vshrq_n_s64::<29>`) preserves the sign of the carry.
    fn prop(&mut self) -> uint32x4_t {
        // SAFETY: register-width NEON ops; covered by the type-level Safety note.
        unsafe {
            let mask_u32 = vdupq_n_u32(MASK_29);

            // Initialise carry from limb 0, then mask limb 0 to 29 bits.
            let limb0_s32 = vreinterpretq_s32_u32(self.limbs[0]);
            let mut carry_lo = vmovl_s32(vget_low_s32(limb0_s32));
            let mut carry_hi = vmovl_s32(vget_high_s32(limb0_s32));
            carry_lo = vshrq_n_s64::<29>(carry_lo);
            carry_hi = vshrq_n_s64::<29>(carry_hi);
            self.limbs[0] = vandq_u32(self.limbs[0], mask_u32);

            // Propagate through limbs 1..8.
            for i in 1..LIMBS_29 - 1 {
                let limb_s32 = vreinterpretq_s32_u32(self.limbs[i]);
                carry_lo = vaddq_s64(carry_lo, vmovl_s32(vget_low_s32(limb_s32)));
                carry_hi = vaddq_s64(carry_hi, vmovl_s32(vget_high_s32(limb_s32)));

                let carry_u_lo = vreinterpretq_u64_s64(carry_lo);
                let carry_u_hi = vreinterpretq_u64_s64(carry_hi);
                let pair_lo = vmovn_u64(carry_u_lo);
                let combined = vmovn_high_u64(pair_lo, carry_u_hi);
                self.limbs[i] = vandq_u32(combined, mask_u32);

                carry_lo = vshrq_n_s64::<29>(carry_lo);
                carry_hi = vshrq_n_s64::<29>(carry_hi);
            }

            // Fold final carry into the high limb (unmasked).
            let carry_u_lo = vreinterpretq_u64_s64(carry_lo);
            let carry_u_hi = vreinterpretq_u64_s64(carry_hi);
            let pair_lo = vmovn_u64(carry_u_lo);
            let carry_u32 = vmovn_high_u64(pair_lo, carry_u_hi);
            self.limbs[LIMBS_29 - 1] = vaddq_u32(self.limbs[LIMBS_29 - 1], carry_u32);

            // Sign mask from the top bit of limb 8: arithmetic right shift by
            // 31 fills the lane with the sign bit.
            let signed = vreinterpretq_s32_u32(self.limbs[LIMBS_29 - 1]);
            vreinterpretq_u32_s32(vshrq_n_s32::<31>(signed))
        }
    }

    /// Plain 5x5 polynomial multiplication on NEON SoA layout, used by
    /// [`Fp29x4::mul`] for the three sub-products.
    ///
    /// Returns a 9-column unreduced polynomial product as
    /// `[(uint64x2_t, uint64x2_t); 9]`, one `(lo, hi)` pair per column.
    /// Inputs may carry up to 2 bits of excess above `2^29` (e.g. from the
    /// `a_lo + a_hi` step); the accumulator stays within `u64` per lane
    /// because each column accumulates at most five `u30 * u30 = u60`
    /// products and `5 * 2^60 < 2^63`.
    #[inline]
    fn polynomial_5x5(a: &[uint32x4_t; 5], b: &[uint32x4_t; 5]) -> [(uint64x2_t, uint64x2_t); 9] {
        // SAFETY: register-width NEON ops; covered by the type-level Safety note.
        unsafe {
            let zero = vdupq_n_u64(0);
            let mut out: [(uint64x2_t, uint64x2_t); 9] = [(zero, zero); 9];
            for (i, &a_i) in a.iter().enumerate() {
                for (j, &b_j) in b.iter().enumerate() {
                    let col = i + j;
                    out[col].0 = vmlal_u32(out[col].0, vget_low_u32(a_i), vget_low_u32(b_j));
                    out[col].1 = vmlal_high_u32(out[col].1, a_i, b_j);
                }
            }
            out
        }
    }

    /// Plain 4x4 polynomial multiplication on NEON SoA layout.  Used by
    /// [`Fp29x4::mul`] for the `P1 = a_hi * b_hi` sub-product.
    /// Returns a 7-column unreduced polynomial product.
    #[inline]
    fn polynomial_4x4(a: &[uint32x4_t; 4], b: &[uint32x4_t; 4]) -> [(uint64x2_t, uint64x2_t); 7] {
        // SAFETY: see [`Fp29x4::polynomial_5x5`].
        unsafe {
            let zero = vdupq_n_u64(0);
            let mut out: [(uint64x2_t, uint64x2_t); 7] = [(zero, zero); 7];
            for (i, &a_i) in a.iter().enumerate() {
                for (j, &b_j) in b.iter().enumerate() {
                    let col = i + j;
                    out[col].0 = vmlal_u32(out[col].0, vget_low_u32(a_i), vget_low_u32(b_j));
                    out[col].1 = vmlal_high_u32(out[col].1, a_i, b_j);
                }
            }
            out
        }
    }
}

impl Add<Fp29x4> for Fp29x4 {
    type Output = Fp29x4;

    /// Vectorised modular addition over four `Fp29` elements in parallel,
    /// each result reduced to `[0, 2p)`.  Mirrors [`Fp29::add`] structurally
    /// at the lane level: limbwise vector add, subtract `2p` via the
    /// add-2-to-limb-0 / subtract-`2·P4_29`-from-limb-8 trick, propagate
    /// carries, then conditionally add `2p` back per lane on borrow.
    fn add(self, rhs: Fp29x4) -> Fp29x4 {
        // SAFETY: register-width NEON ops; covered by the type-level Safety note.
        unsafe {
            let mut n = Fp29x4 {
                limbs: [
                    vaddq_u32(self.limbs[0], rhs.limbs[0]),
                    vaddq_u32(self.limbs[1], rhs.limbs[1]),
                    vaddq_u32(self.limbs[2], rhs.limbs[2]),
                    vaddq_u32(self.limbs[3], rhs.limbs[3]),
                    vaddq_u32(self.limbs[4], rhs.limbs[4]),
                    vaddq_u32(self.limbs[5], rhs.limbs[5]),
                    vaddq_u32(self.limbs[6], rhs.limbs[6]),
                    vaddq_u32(self.limbs[7], rhs.limbs[7]),
                    vaddq_u32(self.limbs[8], rhs.limbs[8]),
                ],
            };
            let two = vdupq_n_u32(2);
            let two_p4 = vdupq_n_u32(2 * P4_29);
            n.limbs[0] = vaddq_u32(n.limbs[0], two);
            n.limbs[LIMBS_29 - 1] = vsubq_u32(n.limbs[LIMBS_29 - 1], two_p4);
            let borrow = n.prop();
            n.limbs[0] = vsubq_u32(n.limbs[0], vandq_u32(two, borrow));
            n.limbs[LIMBS_29 - 1] = vaddq_u32(n.limbs[LIMBS_29 - 1], vandq_u32(two_p4, borrow));
            n.prop();
            n
        }
    }
}

impl Sub<Fp29x4> for Fp29x4 {
    type Output = Fp29x4;

    /// Vectorised modular subtraction over four `Fp29` elements in parallel,
    /// each result reduced to `[0, 2p)`.  Lane-wise wrapping subtract, then
    /// conditionally adds `2p` per lane on borrow.  Mirrors [`Fp29::sub`].
    fn sub(self, rhs: Fp29x4) -> Fp29x4 {
        // SAFETY: register-width NEON ops; covered by the type-level Safety note.
        unsafe {
            let mut n = Fp29x4 {
                limbs: [
                    vsubq_u32(self.limbs[0], rhs.limbs[0]),
                    vsubq_u32(self.limbs[1], rhs.limbs[1]),
                    vsubq_u32(self.limbs[2], rhs.limbs[2]),
                    vsubq_u32(self.limbs[3], rhs.limbs[3]),
                    vsubq_u32(self.limbs[4], rhs.limbs[4]),
                    vsubq_u32(self.limbs[5], rhs.limbs[5]),
                    vsubq_u32(self.limbs[6], rhs.limbs[6]),
                    vsubq_u32(self.limbs[7], rhs.limbs[7]),
                    vsubq_u32(self.limbs[8], rhs.limbs[8]),
                ],
            };
            let two = vdupq_n_u32(2);
            let two_p4 = vdupq_n_u32(2 * P4_29);
            let borrow = n.prop();
            n.limbs[0] = vsubq_u32(n.limbs[0], vandq_u32(two, borrow));
            n.limbs[LIMBS_29 - 1] = vaddq_u32(n.limbs[LIMBS_29 - 1], vandq_u32(two_p4, borrow));
            n.prop();
            n
        }
    }
}

impl Add<Fp29> for Fp29 {
    type Output = Fp29;

    /// Modular addition, reduced to `[0, 2p)`.
    ///
    /// Adds limbwise, subtracts `2p` (via add-2-to-limb-0 / subtract-`2·P4_29`-
    /// from-limb-8), propagates carries, then conditionally adds `2p` back if
    /// the propagation detected a borrow.  Mirrors `Fp::add` structurally.
    fn add(self, rhs: Fp29) -> Fp29 {
        let mut n = Fp29 {
            limbs: [
                self.limbs[0] + rhs.limbs[0],
                self.limbs[1] + rhs.limbs[1],
                self.limbs[2] + rhs.limbs[2],
                self.limbs[3] + rhs.limbs[3],
                self.limbs[4] + rhs.limbs[4],
                self.limbs[5] + rhs.limbs[5],
                self.limbs[6] + rhs.limbs[6],
                self.limbs[7] + rhs.limbs[7],
                self.limbs[8] + rhs.limbs[8],
            ],
        };
        n.limbs[0] = n.limbs[0].wrapping_add(2);
        n.limbs[LIMBS_29 - 1] = n.limbs[LIMBS_29 - 1].wrapping_sub(2 * P4_29);
        let carry = n.prop();
        n.limbs[0] = n.limbs[0].wrapping_sub(2u32 & carry);
        n.limbs[LIMBS_29 - 1] = n.limbs[LIMBS_29 - 1].wrapping_add((2 * P4_29) & carry);
        n.prop();
        n
    }
}

impl Sub<Fp29> for Fp29 {
    type Output = Fp29;

    /// Modular subtraction, reduced to `[0, 2p)`.
    ///
    /// Limbwise wrapping-subtract; if the propagation detects a borrow,
    /// adds `2p` back.  Mirrors `Fp::sub` structurally.
    fn sub(self, rhs: Fp29) -> Fp29 {
        let mut n = Fp29 {
            limbs: [
                self.limbs[0].wrapping_sub(rhs.limbs[0]),
                self.limbs[1].wrapping_sub(rhs.limbs[1]),
                self.limbs[2].wrapping_sub(rhs.limbs[2]),
                self.limbs[3].wrapping_sub(rhs.limbs[3]),
                self.limbs[4].wrapping_sub(rhs.limbs[4]),
                self.limbs[5].wrapping_sub(rhs.limbs[5]),
                self.limbs[6].wrapping_sub(rhs.limbs[6]),
                self.limbs[7].wrapping_sub(rhs.limbs[7]),
                self.limbs[8].wrapping_sub(rhs.limbs[8]),
            ],
        };
        let carry = n.prop();
        n.limbs[0] = n.limbs[0].wrapping_sub(2u32 & carry);
        n.limbs[LIMBS_29 - 1] = n.limbs[LIMBS_29 - 1].wrapping_add((2 * P4_29) & carry);
        n.prop();
        n
    }
}

const _: () = {
    assert!(MASK_29 == (1u32 << RADIX_29) - 1);
    assert!(RADIX_29 as usize * LIMBS_29 >= 248);
};
