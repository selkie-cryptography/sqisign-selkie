//! x86_64 AVX2 backend for `Fp` arithmetic.
//!
//! Future home of the AVX2-vectorised `Fp` implementation analogous to
//! `crate::fields::fp::arch::aarch64::neon`. Targets Haswell-and-later
//! Intel and Zen-and-later AMD CPUs (AVX2 is base ISA from 2013 on
//! Intel, 2017 on AMD).
//!
//! - **Limb layout**: [`Fp26`] holds ten 26-bit unsaturated limbs in 32-bit
//!   lanes. Ten limbs cover the 248-bit modulus with 12 bits of per-limb carry
//!   headroom.
//! - **SIMD packing** (planned): the 10 * `u32` limbs pack into two `__m256i`
//!   halves (8 + 2 lanes); a column of the schoolbook product uses
//!   `_mm256_mul_epu32` (`VPMULUDQ`) for the 4-lane `u32 * u32 -> u64` widening
//!   multiply that AVX2 provides.
//! - **Multiplication** (planned): schoolbook `Fp * Fp` with `VPMULUDQ` over
//!   the 100-column product, interleaved with Montgomery-style reduction via
//!   the `p = 5 * 2^248 - 1` structure (`P4_26 = 5 * 2^14`).
//!
//! # Why radix-26 over radix-25.5 / radix-30 / radix-31
//!
//! AVX2's widening multiply is `u32 * u32 -> u64`, leaving 8 free bits in
//! the 64-bit accumulator for the per-column schoolbook plus the ` * P4_26`
//! Montgomery cross-terms. Radix-26 keeps each limb safely under 32 bits
//! while leaving the accumulator with 12 bits of slack for the two
//! cross-term folds before a normalisation pass.
//!
//! # Why no AVX-512-IFMA52
//!
//! AVX-512-IFMA52 is the analogous "vectorised Fp" lever for x86_64 but
//! sits behind a narrow Intel-server CPU subset, is omitted from AMD
//! Zen 4, and isn't targeted by any major crypto library. AVX2 over a
//! radix-26 layout is the realistic x86_64 vectorised-Fp target.
//!
//! # Status
//!
//! This commit lands **scaffolding only**: the limb layout, Montgomery
//! parameters, byte (de)serialisation, and the
//! [`Fp26::from_limbs`] const constructor that lets the existing
//! `pub const FOO: Fp = Fp::from_limbs([u64; 5])` precomputed-constant
//! tables embed identically under cfg-avx2. Runtime arithmetic
//! (`Mul`, `Add`, `Sub`, `square`, `invert`, `sqrt`, ...) lands in
//! follow-up commits. The arch dispatcher's cfg-avx2 arm doesn't
//! activate until those land.

#[cfg(target_feature = "avx2")]
use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_blendv_epi8, _mm256_cmpgt_epi64,
    _mm256_loadu_si256, _mm256_mul_epu32, _mm256_or_si256, _mm256_set1_epi64x,
    _mm256_setzero_si256, _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256,
    _mm256_sub_epi64,
};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

/// `1` in non-Montgomery form, used to exit Montgomery form via the `Mul`
/// trait impl: `mont * 1 * R^-1 = mont / R = canonical`.
const ONE_RAW: Fp26 = Fp26 {
    limbs: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
};

/// Bits per limb in the radix-26 representation.
pub const RADIX_26: u32 = 26;

/// Mask for a single radix-26 limb.
pub const MASK_26: u32 = (1u32 << RADIX_26) - 1;

/// Number of limbs in the radix-26 representation.
///
/// Ten limbs of 26 bits each cover 260 bits, with 12 bits of headroom
/// above the 248-bit modulus.
pub const LIMBS_26: usize = 10;

/// Montgomery fold multiplier: `5 * 2^14`.
///
/// At the boundary where the schoolbook column index `i >= LIMBS_26 - 1`,
/// the interleaved Montgomery reduction adds `v[i - (LIMBS_26 - 1)] * P4_26`
/// to the accumulator. This is equivalent (mod p) to adding
/// `v[i - 9] * 5 * 2^248` because `5 * 2^248 == 1 (mod p)`, and within
/// limb 9 the offset is `248 - 9 * 26 = 14`.
const P4_26: u32 = 5 << 14;

/// `p` in radix-26 form.
///
/// Used by `Fp26::final_sub` (future) to subtract the modulus from an
/// unreduced result. Computed from `p = 5 * 2^248 - 1`: limbs 0..8 are
/// `2^26 - 1`, limb 9 holds the top 14 bits.
const P_LIMBS_26: [u32; LIMBS_26] = [
    0x3FFFFFF, 0x3FFFFFF, 0x3FFFFFF, 0x3FFFFFF, 0x3FFFFFF, 0x3FFFFFF, 0x3FFFFFF, 0x3FFFFFF,
    0x3FFFFFF, 0x0013FFF,
];

/// `R^2_26 mod p` where `R_26 = 2^260`.
///
/// Precomputed via `pow(2, 520, p)` and packed into 10 * 26-bit limbs.
/// Used by `from_bytes` (future) to enter Fp26 Montgomery form:
/// `canonical_value * R^2_26 * R^-1 = canonical_value * R`.
#[allow(dead_code)] // used by future Mul-based from_bytes
const R2_26: Fp26 = Fp26 {
    limbs: [
        0x33D70A3, 0x0CCCCCC, 0x3333333, 0x0CCCCCC, 0x3333333, 0x0CCCCCC, 0x3333333, 0x0CCCCCC,
        0x3333333, 0x0010CCC,
    ],
};

/// Bridge constant for the portable backend's radix-51 Montgomery
/// limbs `[u64; 5]` to Fp26 Montgomery form. Computed as
/// `K = 2^(2 * R_26_exp - R_portable_exp) mod p = 2^265 mod p`, where
/// R_portable = 2^255 and R_26 = 2^260; the Montgomery multiplication
/// `mont_mul_26(portable_mont, K)` lands the value in Fp26's
/// `value * R_26 mod p` form: `(value * 2^255) * 2^265 * 2^(-260) = value *
/// 2^260`.
const K_PORT_TO_26: [u32; LIMBS_26] = [0x0006666, 0, 0, 0, 0, 0, 0, 0, 0, 0x0008000];

/// Field element in radix-26 limb form, in Montgomery representation.
///
/// Parallel representation to [`crate::fields::fp::arch::portable::Fp`]'s
/// radix-51 layout, laid out for AVX2 32-bit-lane packing. Limbs are
/// little-endian: `limbs[0]` is the least significant 26 bits. The stored
/// value is `value * R_26 mod p` where `R_26 = 2^260`; multiplication
/// (future) returns `a * b * R^-1`.
///
/// # Invariants
///
/// - After the (future) `Mul` impl or `From<Fp>`, `limbs[i] < 2^26` for `i < 9`
///   and `limbs[9] < 2^17` (sub-`2p` bound).
/// - [`Fp26::from_bytes_le`] / [`Fp26::to_bytes_le`] operate on canonical
///   (non-Montgomery) limbs; they're the byte boundary, before/after the
///   Montgomery scaling.
#[derive(Clone, Copy, Debug)]
pub struct Fp26 {
    /// Ten 26-bit limbs, little-endian.
    pub limbs: [u32; LIMBS_26],
}

impl Fp26 {
    /// Additive identity (zero), in radix-26 Montgomery form.
    pub const ZERO: Self = Self {
        limbs: [0; LIMBS_26],
    };

    /// Multiplicative identity in radix-26 Montgomery form: `1 * R_26 mod p`,
    /// precomputed via `python -c 'pow(2, 260, 5*2**248 - 1)'` then packed
    /// into 10 * 26-bit limbs.
    pub const ONE: Self = Self {
        limbs: [0x333, 0, 0, 0, 0, 0, 0, 0, 0, 0x4000],
    };

    /// Two in radix-26 Montgomery form: `2 * R_26 mod p`.
    pub const TWO: Self = Self {
        limbs: [0x666, 0, 0, 0, 0, 0, 0, 0, 0, 0x8000],
    };

    /// Four in radix-26 Montgomery form: `4 * R_26 mod p`.
    pub const FOUR: Self = Self {
        limbs: [0xCCC, 0, 0, 0, 0, 0, 0, 0, 0, 0x10000],
    };

    /// `-1 mod p` in radix-26 Montgomery form: `(p - 1) * R_26 mod p`.
    pub const MINUS_ONE: Self = Self {
        limbs: [
            0x3FFFCCC, 0x3FFFFFF, 0x3FFFFFF, 0x3FFFFFF, 0x3FFFFFF, 0x3FFFFFF, 0x3FFFFFF, 0x3FFFFFF,
            0x3FFFFFF, 0x000FFFF,
        ],
    };

    /// Decodes 32 bytes (little-endian) into a normalised radix-26 element.
    ///
    /// Mirrors `Fp::from_bytes` at the canonical-form level: the input
    /// must encode a value less than `p`; out-of-range bits in `bytes[31]`
    /// simply flow into the high limb without canonicalisation. The
    /// result is *not* in Montgomery form; combine with multiplication
    /// by [`R2_26`] (future) to enter Fp26's Montgomery scaling.
    pub fn from_bytes_le(bytes: &[u8; 32]) -> Self {
        let mut limbs = [0u32; LIMBS_26];
        let mut acc: u64 = 0;
        let mut bits: u32 = 0;
        let mut limb_idx = 0;

        for &byte in bytes.iter() {
            acc |= (byte as u64) << bits;
            bits += 8;

            if bits >= RADIX_26 && limb_idx < LIMBS_26 - 1 {
                limbs[limb_idx] = (acc as u32) & MASK_26;
                acc >>= RADIX_26;
                bits -= RADIX_26;
                limb_idx += 1;
            }
        }

        limbs[limb_idx] = acc as u32;

        Self { limbs }
    }

    /// Encodes a normalised radix-26 element as 32 bytes, little-endian.
    ///
    /// Each limb must be `< 2^26`; if the value is unsaturated the encoded
    /// bytes will overflow into adjacent positions.
    pub fn to_bytes_le(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        let mut acc: u64 = 0;
        let mut bits: u32 = 0;
        let mut pos = 0;

        for &limb in self.limbs.iter() {
            acc |= (limb as u64) << bits;
            bits += RADIX_26;

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

    /// Constructs from radix-51 portable-Montgomery limbs.
    ///
    /// Signature-compatible with the portable backend's `Fp::from_limbs`, so
    /// the crate's precomputed-constant tables (`params.rs`,
    /// `deuring/precomputed.rs`, `curves/montgomery`) embed identically under
    /// either backend selection. The input limbs encode the field element in
    /// radix-51 Montgomery form (`value * 2^255 mod p`); this constructor
    /// repacks them at radix-26 and Montgomery-multiplies by the const
    /// `K = 2^265 mod p`, landing the value in this backend's
    /// `value * 2^260 mod p` form.
    ///
    /// `const fn` so the constants stay `pub const`.
    pub const fn from_limbs(portable_mont: [u64; 5]) -> Self {
        let radix26 = Self::repack_51_to_26(portable_mont);
        Self {
            limbs: Self::mont_mul_const(radix26, K_PORT_TO_26),
        }
    }

    /// Repacks a 5-limb radix-51 little-endian value as 10-limb radix-26 LE.
    ///
    /// Pure bit redistribution: the integer value is unchanged. Input fits in
    /// 255 bits (5 * 51); output uses 260 bits (10 * 26), so the top 5 bits of
    /// `out[9]` are always zero.
    const fn repack_51_to_26(src: [u64; 5]) -> [u32; LIMBS_26] {
        let mut out = [0u32; LIMBS_26];
        let mut acc: u128 = 0;
        let mut bits: u32 = 0;
        let mut src_idx = 0;
        let mut i = 0;

        while i < LIMBS_26 {
            while bits < RADIX_26 && src_idx < 5 {
                acc |= (src[src_idx] as u128) << bits;
                bits += 51;
                src_idx += 1;
            }

            out[i] = (acc as u32) & MASK_26;
            acc >>= RADIX_26;
            bits = bits.saturating_sub(RADIX_26);
            i += 1;
        }

        out
    }

    /// Const-fn Montgomery multiplication on radix-26 limbs.
    ///
    /// Identical CIOS structure to `super::super::aarch64::neon::Fp29`'s
    /// const Mont mul, retuned for radix-26 / 10 limbs / `P4_26 = 5 * 2^14`.
    /// Used by [`Self::from_limbs`] to enter Fp26 Montgomery form at compile
    /// time from the portable backend's Montgomery limbs. A SIMD-vectorised
    /// runtime version follows in a later commit.
    const fn mont_mul_const(a: [u32; LIMBS_26], b: [u32; LIMBS_26]) -> [u32; LIMBS_26] {
        let mut t: u64 = 0;
        let mut v = [0u32; LIMBS_26];
        let mut c = [0u32; LIMBS_26];

        let mut i = 0;

        while i < 2 * LIMBS_26 - 1 {
            let j_lo = if i >= LIMBS_26 { i - LIMBS_26 + 1 } else { 0 };
            let j_hi = if i < LIMBS_26 - 1 { i } else { LIMBS_26 - 1 };
            let mut j = j_lo;

            while j <= j_hi {
                t = t.wrapping_add((a[j] as u64).wrapping_mul(b[i - j] as u64));
                j += 1;
            }

            if i >= LIMBS_26 - 1 {
                let fold_idx = i - (LIMBS_26 - 1);
                t = t.wrapping_add((v[fold_idx] as u64).wrapping_mul(P4_26 as u64));
            }

            let limb = (t as u32) & MASK_26;

            if i < LIMBS_26 {
                v[i] = limb;
            } else {
                c[i - LIMBS_26] = limb;
            }

            t >>= RADIX_26;
            i += 1;
        }

        c[LIMBS_26 - 1] = t as u32;
        c
    }

    /// Squares this element via the [`Mul`] impl.
    ///
    /// A symmetric-cross-term optimization (which uses fewer u32 * u32
    /// products) lands alongside the AVX2 intrinsics commit, where the
    /// savings translate to fewer `VPMULUDQ` ops.
    pub fn square(&self) -> Self {
        self * self
    }

    /// Propagates carries through the limbs, returning a sign mask:
    /// `0` if the final accumulator was non-negative, `0xFFFFFFFF` if it
    /// was negative (indicating a borrow occurred upstream).
    ///
    /// Mirrors `super::super::aarch64::neon::Fp29::prop`: arithmetic
    /// right-shift on an `i64` carry preserves the sign, and the high bit
    /// of `limbs[LIMBS_26 - 1]` after the final wrapping add encodes
    /// whether the cumulative value overflowed (borrowed).
    ///
    /// The cast chain `u32 -> i32 -> i64` is load-bearing: `u32 -> i64`
    /// zero-extends and would lose the borrow sign, while `u32 -> i32`
    /// preserves bits (same width) and `i32 -> i64` then sign-extends.
    fn prop(&mut self) -> u32 {
        let mut carry = (self.limbs[0] as i32) as i64;
        carry >>= RADIX_26;
        self.limbs[0] &= MASK_26;

        for i in 1..LIMBS_26 - 1 {
            carry += (self.limbs[i] as i32) as i64;
            self.limbs[i] = (carry as u32) & MASK_26;
            carry >>= RADIX_26;
        }

        self.limbs[LIMBS_26 - 1] = self.limbs[LIMBS_26 - 1].wrapping_add(carry as u32);

        let sign = (self.limbs[LIMBS_26 - 1] >> 1) >> 30;
        sign.wrapping_neg()
    }

    /// Conditionally subtracts `p` to canonicalise an in-range result.
    ///
    /// Assumes `self < 2p` with each limb already `< 2^26`. Returns the
    /// representative in `[0, p)`. Constant-time via
    /// [`subtle::ConditionallySelectable`].
    pub fn final_sub(self) -> Self {
        let mut diff = [0u32; LIMBS_26];
        let mut borrow: u32 = 0;

        for i in 0..LIMBS_26 {
            let d = (self.limbs[i] as i64) - (P_LIMBS_26[i] as i64) - (borrow as i64);
            diff[i] = (d as u32) & MASK_26;
            borrow = ((d as u64) >> 63) as u32 & 1;
        }

        // borrow == 0: subtraction succeeded (self >= p), use diff.
        // borrow == 1: self < p, keep self.
        let take_diff = Choice::from((1 - borrow) as u8);
        let mut out = [0u32; LIMBS_26];

        for i in 0..LIMBS_26 {
            out[i] = u32::conditional_select(&self.limbs[i], &diff[i], take_diff);
        }

        Self { limbs: out }
    }

    /// Exits Montgomery form: `mont -> mont / R = canonical`.
    ///
    /// Multiplies by `1` in non-Montgomery form ([`ONE_RAW`]); the Montgomery
    /// product is `mont * 1 * R^-1 = mont / R`. Then canonicalises via
    /// [`Self::final_sub`].
    pub fn reduce_montgomery(self) -> Self {
        (&self * &ONE_RAW).final_sub()
    }

    /// Decodes canonical 32-byte little-endian into a Montgomery-form `Fp26`.
    ///
    /// Mirrors `Fp::from_bytes` at the API level: unpacks the bytes as a
    /// canonical integer, then enters this backend's Montgomery form via
    /// multiplication by [`R2_26`].
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        &Self::from_bytes_le(bytes) * &R2_26
    }

    /// Encodes a Montgomery-form `Fp26` as canonical 32-byte little-endian.
    ///
    /// Mirrors `Fp::to_bytes`: exits Montgomery form via
    /// [`Self::reduce_montgomery`], then packs the canonical limbs into 32
    /// bytes.
    pub fn to_bytes(self) -> [u8; 32] {
        self.reduce_montgomery().to_bytes_le()
    }

    /// Constructs a field element from a small integer.
    ///
    /// Mirrors `Fp::from_small`: places the canonical integer value in the
    /// low limbs (radix-26 splits a `u32` across `limbs[0]` and `limbs[1]`)
    /// and enters Montgomery form via the precomputed [`R2_26`] constant.
    pub fn from_small(x: u32) -> Self {
        let mut canonical = Self {
            limbs: [0; LIMBS_26],
        };
        canonical.limbs[0] = x & MASK_26;
        canonical.limbs[1] = x >> RADIX_26;

        &canonical * &R2_26
    }

    /// Squares this element `n` times. Mirrors `Fp::pow2k`.
    #[must_use]
    pub fn pow2k(&self, n: u32) -> Self {
        let mut r = *self;

        for _ in 0..n {
            r = r.square();
        }

        r
    }

    /// Computes `self^((p-3)/4)`. Same addition chain as `Fp::pow_p3div4`;
    /// the prime is identical across backends so the chain transfers
    /// unchanged, just running over this backend's Montgomery-form
    /// multiplication.
    #[must_use]
    pub(crate) fn pow_p3div4(&self) -> Self {
        let x = *self;
        let z = x.square();
        let t0 = &x * &z;
        let z = t0.square();
        let z = &x * &z;
        let t1 = z.square();
        let t3 = t1.square();
        let t2 = t3.square();
        let t4 = t2.pow2k(3);
        let t2 = &t2 * &t4;
        let t4 = t2.pow2k(6);
        let t2 = &t2 * &t4;
        let t4 = t2.pow2k(2);
        let t3 = &t3 * &t4;
        let t3 = t3.pow2k(13);
        let t2 = &t2 * &t3;
        let t3 = t2.pow2k(27);
        let t2 = &t2 * &t3;
        let z = &z * &t2;
        let t2 = z.pow2k(4);
        let t1 = &t1 * &t2;
        let t0 = &t0 * &t1;
        let t1 = &t1 * &t0;
        let t0 = &t1 * &t0;
        let t2 = &t0 * &t1;
        let t0 = &t0 * &t2;
        let t1 = &t1 * &t0;
        let t1 = t1.pow2k(63);
        let t1 = &t0 * &t1;
        let t1 = t1.pow2k(64);
        let t0 = &t0 * &t1;
        let t0 = t0.pow2k(57);

        &z * &t0
    }

    /// Computes the multiplicative inverse: `self^(p-2)`. Mirrors
    /// `Fp::invert`.
    #[must_use]
    pub fn invert(&self) -> Self {
        let t = self.pow_p3div4();
        let t = t.pow2k(2);

        self * &t
    }

    /// Tests whether this element is a quadratic residue in F_p.
    /// Mirrors `Fp::is_square`.
    pub fn is_square(&self) -> Choice {
        let r = self.pow_p3div4();
        let r = r.square();
        let r = &r * self;

        r.ct_eq(&Fp26::ONE) | self.ct_eq(&Fp26::ZERO)
    }

    /// Computes the square root (when `self` is a QR). Mirrors `Fp::sqrt`;
    /// result meaningful only when [`Self::is_square`] is true.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        let y = self.pow_p3div4();

        &y * self
    }
}

impl Add<Fp26> for Fp26 {
    type Output = Fp26;

    /// Modular addition, reduced to `[0, 2p)`.
    ///
    /// Adds limbwise, subtracts `2p` (via add-2-to-limb-0 / subtract-`2 *
    /// P4_26`-from-limb-9), propagates carries, then conditionally adds `2p`
    /// back if the propagation detected a borrow. Mirrors `Fp::add`
    /// structurally.
    fn add(self, rhs: Fp26) -> Fp26 {
        let mut n = Fp26 {
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
                self.limbs[9] + rhs.limbs[9],
            ],
        };
        n.limbs[0] = n.limbs[0].wrapping_add(2);
        n.limbs[LIMBS_26 - 1] = n.limbs[LIMBS_26 - 1].wrapping_sub(2 * P4_26);

        let carry = n.prop();
        n.limbs[0] = n.limbs[0].wrapping_sub(2u32 & carry);
        n.limbs[LIMBS_26 - 1] = n.limbs[LIMBS_26 - 1].wrapping_add((2 * P4_26) & carry);
        n.prop();

        n
    }
}

impl Sub<Fp26> for Fp26 {
    type Output = Fp26;

    /// Modular subtraction, reduced to `[0, 2p)`.
    ///
    /// Limbwise wrapping-subtract; if the propagation detects a borrow,
    /// adds `2p` back. Mirrors `Fp::sub` structurally.
    fn sub(self, rhs: Fp26) -> Fp26 {
        let mut n = Fp26 {
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
                self.limbs[9].wrapping_sub(rhs.limbs[9]),
            ],
        };

        let carry = n.prop();
        n.limbs[0] = n.limbs[0].wrapping_sub(2u32 & carry);
        n.limbs[LIMBS_26 - 1] = n.limbs[LIMBS_26 - 1].wrapping_add((2 * P4_26) & carry);
        n.prop();

        n
    }
}

impl<'b> Mul<&'b Fp26> for &Fp26 {
    type Output = Fp26;

    /// Montgomery multiplication: returns `a * b * R^-1 mod p`.
    ///
    /// 10x10 schoolbook product with Montgomery reduction interleaved column-
    /// by-column over the 19 output positions. The fold step at column
    /// `i >= 9` adds `v[i-9] * P4_26`, exploiting `5 * 2^248 == 1 (mod p)`
    /// to absorb the previously-computed low column into the high columns.
    ///
    /// Output limbs satisfy `limbs[i] < 2^26` for `i < 9` and
    /// `limbs[9] < 2^17` (so the result is in `[0, 2p)`). Use
    /// `Fp26::final_sub` to canonicalise to `[0, p)`.
    ///
    /// Delegates to `Fp26::mont_mul_const` (outer-product CIOS,
    /// const-fn so the constants stay `pub const`).  An earlier AVX2
    /// single-Fp operand-scanning variant (`mont_mul_avx2` via
    /// `_mm256_mul_epu32`) was measured 4.6x slower than the portable
    /// radix-51 + MULX path on Fly perf-2x x86_64-v3 (PR #223
    /// e58050c) and dropped 2026-06-03 — see `build.rs::detect_x86_64_avx2`
    /// for the structural reasoning.  `Fp26x4` SoA AVX2 paths are
    /// the production AVX2 surface; this single-Fp body stays scalar.
    fn mul(self, rhs: &'b Fp26) -> Fp26 {
        let limbs = Fp26::mont_mul_const(self.limbs, rhs.limbs);

        Fp26 { limbs }
    }
}

impl Mul<Fp26> for Fp26 {
    type Output = Fp26;
    fn mul(self, rhs: Fp26) -> Fp26 {
        &self * &rhs
    }
}

impl<'b> Add<&'b Fp26> for &Fp26 {
    type Output = Fp26;
    fn add(self, rhs: &'b Fp26) -> Fp26 {
        *self + *rhs
    }
}

impl<'b> Sub<&'b Fp26> for &Fp26 {
    type Output = Fp26;
    fn sub(self, rhs: &'b Fp26) -> Fp26 {
        *self - *rhs
    }
}

impl Neg for &Fp26 {
    type Output = Fp26;
    fn neg(self) -> Fp26 {
        Fp26::ZERO - *self
    }
}

impl Neg for Fp26 {
    type Output = Fp26;
    fn neg(self) -> Fp26 {
        -&self
    }
}

impl AddAssign<&Fp26> for Fp26 {
    fn add_assign(&mut self, rhs: &Fp26) {
        *self = *self + *rhs;
    }
}

impl AddAssign for Fp26 {
    fn add_assign(&mut self, rhs: Fp26) {
        *self = *self + rhs;
    }
}

impl SubAssign<&Fp26> for Fp26 {
    fn sub_assign(&mut self, rhs: &Fp26) {
        *self = *self - *rhs;
    }
}

impl SubAssign for Fp26 {
    fn sub_assign(&mut self, rhs: Fp26) {
        *self = *self - rhs;
    }
}

impl MulAssign<&Fp26> for Fp26 {
    fn mul_assign(&mut self, rhs: &Fp26) {
        *self = &*self * rhs;
    }
}

impl MulAssign for Fp26 {
    fn mul_assign(&mut self, rhs: Fp26) {
        *self = &*self * &rhs;
    }
}

impl ConstantTimeEq for Fp26 {
    fn ct_eq(&self, other: &Fp26) -> Choice {
        self.to_bytes().ct_eq(&other.to_bytes())
    }
}

impl ConditionallySelectable for Fp26 {
    fn conditional_select(a: &Fp26, b: &Fp26, choice: Choice) -> Fp26 {
        let mut limbs = [0u32; LIMBS_26];

        for (i, slot) in limbs.iter_mut().enumerate() {
            *slot = u32::conditional_select(&a.limbs[i], &b.limbs[i], choice);
        }

        Fp26 { limbs }
    }
}

impl Eq for Fp26 {}

impl PartialEq for Fp26 {
    fn eq(&self, other: &Fp26) -> bool {
        self.ct_eq(other).into()
    }
}

impl Fp26 {
    /// Returns `a1 * b1 + a2 * b2 mod p`.
    ///
    /// Backend-portable baseline: two separate Montgomery multiplications
    /// and one addition.  A radix-26 fused-column variant (analog of the
    /// portable backend's Longa sum-of-products, ePrint 2022/367) would
    /// trade more partial products for one fewer reduction; whether that
    /// pays off at radix-26's 10 limbs needs measurement.  Defaulting to
    /// the delegate keeps the surface matched to
    /// `arch::portable::Fp::sum_of_products`.
    #[must_use]
    pub fn sum_of_products(a1: &Fp26, b1: &Fp26, a2: &Fp26, b2: &Fp26) -> Fp26 {
        &(a1 * b1) + &(a2 * b2)
    }

    /// Returns `a1 * b1 - a2 * b2 mod p`.  Backend-portable baseline; see
    /// [`Fp26::sum_of_products`] for the optimisation note.
    #[must_use]
    pub fn difference_of_products(a1: &Fp26, b1: &Fp26, a2: &Fp26, b2: &Fp26) -> Fp26 {
        &(a1 * b1) - &(a2 * b2)
    }

    /// Returns the sum of four base-field products `pairs[i].0 * pairs[i].1`.
    ///
    /// Backend-portable baseline: four separate Montgomery multiplications
    /// and three additions.  Surface parity with
    /// [`arch::portable::Fp::sum_of_products_4`][p], whose Longa-fused
    /// implementation saves three reductions vs the four-mul path.
    /// A radix-26 fused variant is future work.
    ///
    /// [p]: crate::fields::fp::arch::portable::Fp::sum_of_products_4
    #[must_use]
    pub fn sum_of_products_4(pairs: [(&Fp26, &Fp26); 4]) -> Fp26 {
        let [(a1, b1), (a2, b2), (a3, b3), (a4, b4)] = pairs;
        &(&(&(a1 * b1) + &(a2 * b2)) + &(a3 * b3)) + &(a4 * b4)
    }

    /// Returns the t=4 sum-of-products with the last pair subtracted.
    ///
    /// Backend-portable baseline; see [`Fp26::sum_of_products_4`] for
    /// the optimisation note.
    #[must_use]
    pub fn difference_of_products_4(pairs: [(&Fp26, &Fp26); 4]) -> Fp26 {
        let [(a1, b1), (a2, b2), (a3, b3), (a4, b4)] = pairs;
        &(&(&(a1 * b1) + &(a2 * b2)) + &(a3 * b3)) - &(a4 * b4)
    }
}

/// Four [`Fp26`] elements packed into AVX2 Structure-of-Arrays (SoA)
/// layout: the four elements are interleaved lane-wise so one
/// `_mm256_mul_epu32` computes the same schoolbook column across all
/// four products simultaneously.  Each of the ten `__m256i` vectors
/// holds the `i`-th radix-26 limb of four independent field elements
/// at u64 lanes 0..3.  Using u64 lanes (rather than u32) gives the
/// Montgomery multiplication accumulator native room for the
/// `u32 * u32 -> u64` partial products without widening shuffles.
///
/// Parallel to `crate::fields::fp::arch::aarch64::neon::Fp29x4`'s
/// NEON SoA layout but at AVX2's 256-bit register width and radix-26.
/// The eventual `mul` / `add` / `sub` / `square` methods land in
/// follow-up commits; this commit ships the layout + transpose
/// primitives.
///
/// # Safety
///
/// Public methods are safe.  Internal `unsafe` blocks wrap x86_64
/// AVX2 intrinsics; the type is gated on `cfg(target_feature = "avx2")`
/// so the intrinsics never enter non-AVX2 builds.
#[cfg(target_feature = "avx2")]
#[derive(Clone, Copy)]
pub struct Fp26x4 {
    /// Ten AVX2 4-lane u64 vectors.  Lane `j` of `limbs[i]` is the
    /// `i`-th radix-26 limb of element `j` (zero-extended to u64).
    pub limbs: [__m256i; LIMBS_26],
}

#[cfg(target_feature = "avx2")]
impl Fp26x4 {
    /// Packs four scalar [`Fp26`] elements into the SoA layout via a
    /// per-limb gather: `limbs[i]` ends up holding
    /// `[elements[0].limbs[i], ..., elements[3].limbs[i]]` (each u32
    /// limb zero-extended to a u64 lane).
    pub fn from_scalars(elements: &[Fp26; 4]) -> Self {
        // SAFETY: AVX2 intrinsics are unconditionally callable here
        // because the type is gated on cfg(target_feature = "avx2").
        // The lane buffer is a stack-local `[u64; 4]` whose pointer is
        // properly aligned for the AVX2 256-bit load.
        let zero = unsafe { _mm256_setzero_si256() };
        let mut limbs = [zero; LIMBS_26];

        for (i, slot) in limbs.iter_mut().enumerate() {
            let lane: [u64; 4] = [
                elements[0].limbs[i] as u64,
                elements[1].limbs[i] as u64,
                elements[2].limbs[i] as u64,
                elements[3].limbs[i] as u64,
            ];
            *slot = unsafe { _mm256_loadu_si256(lane.as_ptr() as *const __m256i) };
        }

        Self { limbs }
    }

    /// Unpacks the SoA layout back into four scalar [`Fp26`] elements.
    /// Inverse of [`Fp26x4::from_scalars`].
    pub fn to_scalars(self) -> [Fp26; 4] {
        // SAFETY: see `from_scalars`.  The lane buffer is a stack-local
        // `[u64; 4]` properly aligned for the AVX2 256-bit store.  Each
        // lane is < 2^26 by Fp26 invariant, so the u64 -> u32 narrowing
        // is lossless.
        let mut out = [Fp26::ZERO; 4];
        let mut lane = [0u64; 4];

        for (i, &limb) in self.limbs.iter().enumerate() {
            unsafe { _mm256_storeu_si256(lane.as_mut_ptr() as *mut __m256i, limb) };

            for (j, &v) in lane.iter().enumerate() {
                out[j].limbs[i] = v as u32;
            }
        }

        out
    }

    /// Conditionally subtracts `p` per lane to canonicalise.
    ///
    /// Assumes each lane is `< 2p` with each limb already `< 2^26`.
    /// Returns the representative in `[0, p)` per lane.  Constant-time
    /// via `_mm256_blendv_epi8`-driven lane select.  Mirrors
    /// `Fp26::final_sub`.
    ///
    /// AVX2 has no `_mm256_srai_epi64`, so borrow tracking goes through
    /// `_mm256_cmpgt_epi64` (per-lane all-0s / all-1s mask) instead of
    /// the i64-arithmetic-shift trick that the NEON `Fp29x4::prop` uses.
    pub fn final_sub(self) -> Self {
        // SAFETY: register-width AVX2 ops; the type's
        // cfg(target_feature = "avx2") gate makes these always available
        // where Fp26x4 compiles.
        unsafe {
            let mask_26 = _mm256_set1_epi64x(MASK_26 as i64);
            let zero = _mm256_setzero_si256();
            let one = _mm256_set1_epi64x(1);

            let mut diff = [zero; LIMBS_26];
            let mut borrow = zero;

            for (i, &limb_i) in self.limbs.iter().enumerate() {
                // d = limb_i - p[i] - borrow.
                let p_i = _mm256_set1_epi64x(P_LIMBS_26[i] as i64);
                let d = _mm256_sub_epi64(limb_i, p_i);
                let d = _mm256_sub_epi64(d, borrow);

                // diff[i] = d & MASK_26 (low 26 bits per lane).
                diff[i] = _mm256_and_si256(d, mask_26);

                // borrow_next = (d as i64 < 0) per lane.  _mm256_cmpgt_epi64
                // returns all-1s per lane where 0 > d (i.e. d's high bit is
                // set as i64).  Mask down to 0/1 per lane for the next sub.
                let neg_mask = _mm256_cmpgt_epi64(zero, d);
                borrow = _mm256_and_si256(neg_mask, one);
            }

            // After all 10 limbs, `borrow` per lane is 1 iff self < p
            // (subtraction underflowed).  Build a per-lane mask that's
            // all-1s where we should keep `self` (borrow == 1), all-0s
            // where we should use `diff` (borrow == 0).
            let take_self = _mm256_cmpgt_epi64(borrow, zero);

            let mut out = [zero; LIMBS_26];

            for (i, slot) in out.iter_mut().enumerate() {
                // blendv: per-byte, mask high bit picks second operand.
                // Lane-wide all-1s/all-0s makes this a clean lane select.
                *slot = _mm256_blendv_epi8(diff[i], self.limbs[i], take_self);
            }

            Self { limbs: out }
        }
    }

    /// Vectorised carry propagation across the 10 limbs per lane.
    /// Returns a per-lane mask: all-1s if the cumulative value
    /// underflowed (interpreted as i64 lanes had the high bit set
    /// at the end), all-0s otherwise.
    ///
    /// Mirrors [`Fp26::prop`]'s sign-aware carry chain, but AVX2 has
    /// no `_mm256_srai_epi64`: each per-limb shift sign-extends
    /// manually via `_mm256_cmpgt_epi64` (detect sign) +
    /// `_mm256_srli_epi64` (logical body) +
    /// `_mm256_slli_epi64(_, 64 - 26) | _` (sign fill).
    ///
    /// # Safety
    ///
    /// Caller must satisfy AVX2 (the type is `cfg(target_feature =
    /// "avx2")`).
    unsafe fn prop(&mut self) -> __m256i {
        let zero = _mm256_setzero_si256();
        let mask_26 = _mm256_set1_epi64x(MASK_26 as i64);

        // arith-shift-right by RADIX_26 on lane 0.
        let val = self.limbs[0];
        let neg_mask = _mm256_cmpgt_epi64(zero, val);
        let logical = _mm256_srli_epi64::<{ RADIX_26 as i32 }>(val);
        let sign_fill = _mm256_slli_epi64::<{ 64 - RADIX_26 as i32 }>(neg_mask);
        let mut carry = _mm256_or_si256(logical, sign_fill);

        self.limbs[0] = _mm256_and_si256(self.limbs[0], mask_26);

        for i in 1..LIMBS_26 - 1 {
            let v = _mm256_add_epi64(carry, self.limbs[i]);
            self.limbs[i] = _mm256_and_si256(v, mask_26);

            let neg_mask = _mm256_cmpgt_epi64(zero, v);
            let logical = _mm256_srli_epi64::<{ RADIX_26 as i32 }>(v);
            let sign_fill = _mm256_slli_epi64::<{ 64 - RADIX_26 as i32 }>(neg_mask);
            carry = _mm256_or_si256(logical, sign_fill);
        }

        self.limbs[LIMBS_26 - 1] = _mm256_add_epi64(self.limbs[LIMBS_26 - 1], carry);

        // Per-lane sign mask: scalar Fp26::prop uses bit 31 of limb 9
        // (a u32).  For Fp26x4's u64 lanes, the same bit position
        // indicates "borrow occurred upstream" — extract bit 31 and
        // negate to all-0s / all-1s per lane.
        let bit_31 = _mm256_srli_epi64::<31>(self.limbs[LIMBS_26 - 1]);
        let bit_31_isolated = _mm256_and_si256(bit_31, _mm256_set1_epi64x(1));

        _mm256_sub_epi64(zero, bit_31_isolated)
    }
}

#[cfg(target_feature = "avx2")]
impl Add<Fp26x4> for Fp26x4 {
    type Output = Fp26x4;

    /// Vectorised modular addition over four `Fp26` elements per lane,
    /// each result reduced to `[0, 2p)`.  Lane-wise add, then subtract
    /// `2p` (add 2 to limb 0, subtract `2 * P4_26` from limb 9),
    /// propagate carries, conditionally add `2p` back per lane on
    /// borrow.  Mirrors [`Fp26::add`] structurally.
    fn add(self, rhs: Fp26x4) -> Fp26x4 {
        // SAFETY: register-width AVX2 ops; type-level cfg gate covers.
        unsafe {
            let mut n = Fp26x4 {
                limbs: [
                    _mm256_add_epi64(self.limbs[0], rhs.limbs[0]),
                    _mm256_add_epi64(self.limbs[1], rhs.limbs[1]),
                    _mm256_add_epi64(self.limbs[2], rhs.limbs[2]),
                    _mm256_add_epi64(self.limbs[3], rhs.limbs[3]),
                    _mm256_add_epi64(self.limbs[4], rhs.limbs[4]),
                    _mm256_add_epi64(self.limbs[5], rhs.limbs[5]),
                    _mm256_add_epi64(self.limbs[6], rhs.limbs[6]),
                    _mm256_add_epi64(self.limbs[7], rhs.limbs[7]),
                    _mm256_add_epi64(self.limbs[8], rhs.limbs[8]),
                    _mm256_add_epi64(self.limbs[9], rhs.limbs[9]),
                ],
            };

            let two = _mm256_set1_epi64x(2);
            let two_p4 = _mm256_set1_epi64x((2 * P4_26) as i64);

            n.limbs[0] = _mm256_add_epi64(n.limbs[0], two);
            n.limbs[LIMBS_26 - 1] = _mm256_sub_epi64(n.limbs[LIMBS_26 - 1], two_p4);

            let borrow = n.prop();
            n.limbs[0] = _mm256_sub_epi64(n.limbs[0], _mm256_and_si256(two, borrow));
            n.limbs[LIMBS_26 - 1] =
                _mm256_add_epi64(n.limbs[LIMBS_26 - 1], _mm256_and_si256(two_p4, borrow));
            n.prop();

            n
        }
    }
}

#[cfg(target_feature = "avx2")]
impl Sub<Fp26x4> for Fp26x4 {
    type Output = Fp26x4;

    /// Vectorised modular subtraction over four `Fp26` elements per
    /// lane, each result reduced to `[0, 2p)`.  Lane-wise wrapping-sub;
    /// if the per-lane prop detects a borrow, adds `2p` back per lane.
    /// Mirrors [`Fp26::sub`] structurally.
    fn sub(self, rhs: Fp26x4) -> Fp26x4 {
        // SAFETY: register-width AVX2 ops; type-level cfg gate covers.
        unsafe {
            let mut n = Fp26x4 {
                limbs: [
                    _mm256_sub_epi64(self.limbs[0], rhs.limbs[0]),
                    _mm256_sub_epi64(self.limbs[1], rhs.limbs[1]),
                    _mm256_sub_epi64(self.limbs[2], rhs.limbs[2]),
                    _mm256_sub_epi64(self.limbs[3], rhs.limbs[3]),
                    _mm256_sub_epi64(self.limbs[4], rhs.limbs[4]),
                    _mm256_sub_epi64(self.limbs[5], rhs.limbs[5]),
                    _mm256_sub_epi64(self.limbs[6], rhs.limbs[6]),
                    _mm256_sub_epi64(self.limbs[7], rhs.limbs[7]),
                    _mm256_sub_epi64(self.limbs[8], rhs.limbs[8]),
                    _mm256_sub_epi64(self.limbs[9], rhs.limbs[9]),
                ],
            };

            let two = _mm256_set1_epi64x(2);
            let two_p4 = _mm256_set1_epi64x((2 * P4_26) as i64);

            let borrow = n.prop();
            n.limbs[0] = _mm256_sub_epi64(n.limbs[0], _mm256_and_si256(two, borrow));
            n.limbs[LIMBS_26 - 1] =
                _mm256_add_epi64(n.limbs[LIMBS_26 - 1], _mm256_and_si256(two_p4, borrow));
            n.prop();

            n
        }
    }
}

#[cfg(target_feature = "avx2")]
impl Fp26x4 {
    /// Vectorised Montgomery multiplication: returns
    /// `[a[0] * b[0] * R^-1, ..., a[3] * b[3] * R^-1]` packed in SoA form.
    ///
    /// 10x10 schoolbook outer-product CIOS, identical algorithm to
    /// `Fp26::mont_mul_const` but lane-parallel over 4 independent
    /// `Fp26` products via `_mm256_mul_epu32` (VPMULUDQ: u32 * u32 -> u64
    /// across 4 lanes).
    ///
    /// 19 column iterations.  Each column accumulates the partial
    /// products `a[j] * b[i-j]` for `j` in the valid range, plus (when
    /// `i >= 9`) the Montgomery fold term `v[i-9] * P4_26`.  The low
    /// 26 bits of the running u64 lane become the column's output
    /// limb; the high bits carry to the next column via
    /// `_mm256_srli_epi64::<26>`.
    ///
    /// Output limbs satisfy `limbs[i] < 2^26` per lane for `i < 9` and
    /// `limbs[9] < 2^20` per lane (result in `[0, 2p)`).  Use
    /// [`Fp26x4::final_sub`] to canonicalise.
    ///
    /// A Karatsuba 5+5 decomposition (3 sub-products of 5x5 + assembly)
    /// is the natural performance optimisation; it lands in a follow-up
    /// commit.  The schoolbook here is the correctness baseline.
    pub fn mul(&self, rhs: &Fp26x4) -> Fp26x4 {
        // SAFETY: register-width AVX2 ops; the type's
        // cfg(target_feature = "avx2") gate makes the intrinsics
        // unconditionally callable here.
        unsafe {
            let a = &self.limbs;
            let b = &rhs.limbs;

            let zero = _mm256_setzero_si256();
            let mask_26 = _mm256_set1_epi64x(MASK_26 as i64);
            let p4 = _mm256_set1_epi64x(P4_26 as i64);

            let mut t = zero;
            let mut v = [zero; LIMBS_26];
            let mut c = [zero; LIMBS_26];

            for i in 0..2 * LIMBS_26 - 1 {
                let j_lo = if i >= LIMBS_26 { i - LIMBS_26 + 1 } else { 0 };
                let j_hi = i.min(LIMBS_26 - 1);

                for j in j_lo..=j_hi {
                    let prod = _mm256_mul_epu32(a[j], b[i - j]);
                    t = _mm256_add_epi64(t, prod);
                }

                if i >= LIMBS_26 - 1 {
                    let fold_idx = i - (LIMBS_26 - 1);
                    let fold_prod = _mm256_mul_epu32(v[fold_idx], p4);
                    t = _mm256_add_epi64(t, fold_prod);
                }

                let limb = _mm256_and_si256(t, mask_26);

                if i < LIMBS_26 {
                    v[i] = limb;
                } else {
                    c[i - LIMBS_26] = limb;
                }

                t = _mm256_srli_epi64::<{ RADIX_26 as i32 }>(t);
            }

            c[LIMBS_26 - 1] = t;

            Self { limbs: c }
        }
    }

    /// Squares the four packed elements lane-wise.
    ///
    /// Symmetric-cross-term schoolbook over `_mm256_mul_epu32`: for
    /// `a = self`, column `k` of `a*a` accumulates `Σ a[j] * a[k-j]`
    /// over the valid `j` range.  Pairs `(j, k-j)` with `j < k-j` are
    /// distinct off-diagonal products: each `a[j] * a[k-j]` appears
    /// twice in the asymmetric schoolbook (once for `(j, k-j)` and
    /// once for `(k-j, j)`), so the symmetric form computes the
    /// product once and doubles it via `_mm256_slli_epi64::<1>`.
    /// Diagonal pairs `j = k-j` (only when `k` is even) contribute
    /// once.  Montgomery interleaving is identical to
    /// [`Fp26x4::mul`].
    ///
    /// Multiply count: 55 `_mm256_mul_epu32` calls vs the 100 of the
    /// asymmetric schoolbook (`a * a` going through `mul`).  The
    /// doubling adds a cheap shift per off-diagonal column.  A
    /// follow-up Karatsuba decomposition can squeeze the constant
    /// further.
    pub fn square(&self) -> Fp26x4 {
        // SAFETY: register-width AVX2 ops; the type's cfg(target_feature
        // = "avx2") gate makes the intrinsics unconditionally callable.
        unsafe {
            let a = &self.limbs;

            let zero = _mm256_setzero_si256();
            let mask_26 = _mm256_set1_epi64x(MASK_26 as i64);
            let p4 = _mm256_set1_epi64x(P4_26 as i64);

            let mut t = zero;
            let mut v = [zero; LIMBS_26];
            let mut c = [zero; LIMBS_26];

            for i in 0..2 * LIMBS_26 - 1 {
                let j_lo = if i >= LIMBS_26 { i - LIMBS_26 + 1 } else { 0 };
                let j_hi = i.min(LIMBS_26 - 1);
                // Pair-iteration upper bound: j <= k-j means j <= i/2.
                // Even `i` contributes the diagonal `a[i/2] * a[i/2]`;
                // odd `i` has no diagonal.
                let pair_hi = (i / 2).min(j_hi);

                for j in j_lo..=pair_hi {
                    let other = i - j;
                    let prod = _mm256_mul_epu32(a[j], a[other]);

                    if j == other {
                        t = _mm256_add_epi64(t, prod);
                    } else {
                        let doubled = _mm256_slli_epi64::<1>(prod);
                        t = _mm256_add_epi64(t, doubled);
                    }
                }

                if i >= LIMBS_26 - 1 {
                    let fold_idx = i - (LIMBS_26 - 1);
                    let fold_prod = _mm256_mul_epu32(v[fold_idx], p4);
                    t = _mm256_add_epi64(t, fold_prod);
                }

                let limb = _mm256_and_si256(t, mask_26);

                if i < LIMBS_26 {
                    v[i] = limb;
                } else {
                    c[i - LIMBS_26] = limb;
                }

                t = _mm256_srli_epi64::<{ RADIX_26 as i32 }>(t);
            }

            c[LIMBS_26 - 1] = t;

            Self { limbs: c }
        }
    }
}

// Cross-impl test submodule is gated on cfg(not(sqisign_selkie_arch =
// "avx2")) for the same reason neon's is: under cfg-avx2 `Fp = Fp26` and
// the cross-bridge tests become tautological. Tests fire on x86_64 hosts
// under default features.
#[cfg(all(test, not(sqisign_selkie_arch = "avx2")))]
mod tests;

// Compile-time correctness checks: from_limbs must agree with the portable
// backend's Mont layout for the canonical small constants. These run in
// the const evaluator, so failure breaks the build before any test executes.
const _: () = {
    assert!(MASK_26 == (1u32 << RADIX_26) - 1);
    assert!(RADIX_26 as usize * LIMBS_26 >= 248);

    let from_portable_zero = Fp26::from_limbs(super::super::portable::Fp::ZERO.0);
    let mut i = 0;

    while i < LIMBS_26 {
        assert!(from_portable_zero.limbs[i] == Fp26::ZERO.limbs[i]);
        i += 1;
    }

    let from_portable_one = Fp26::from_limbs(super::super::portable::Fp::ONE.0);
    let mut i = 0;

    while i < LIMBS_26 {
        assert!(from_portable_one.limbs[i] == Fp26::ONE.limbs[i]);
        i += 1;
    }

    let from_portable_two = Fp26::from_limbs(super::super::portable::Fp::TWO.0);
    let mut i = 0;

    while i < LIMBS_26 {
        assert!(from_portable_two.limbs[i] == Fp26::TWO.limbs[i]);
        i += 1;
    }

    let from_portable_four = Fp26::from_limbs(super::super::portable::Fp::FOUR.0);
    let mut i = 0;

    while i < LIMBS_26 {
        assert!(from_portable_four.limbs[i] == Fp26::FOUR.limbs[i]);
        i += 1;
    }

    let from_portable_minus_one = Fp26::from_limbs(super::super::portable::Fp::MINUS_ONE.0);
    let mut i = 0;

    while i < LIMBS_26 {
        assert!(from_portable_minus_one.limbs[i] == Fp26::MINUS_ONE.limbs[i]);
        i += 1;
    }
};
