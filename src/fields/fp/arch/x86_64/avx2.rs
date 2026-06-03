//! x86_64 AVX2 backend for [`Fp`] arithmetic.
//!
//! Future home of the AVX2-vectorised `Fp` implementation analogous to
//! [`crate::fields::fp::arch::aarch64::neon`]. Targets Haswell-and-later
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

use subtle::ConditionallySelectable;

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
/// Used by [`Fp26::final_sub`] (future) to subtract the modulus from an
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
    /// Identical CIOS structure to [`super::super::aarch64::neon::Fp29`]'s
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
}

impl ConditionallySelectable for Fp26 {
    fn conditional_select(a: &Fp26, b: &Fp26, choice: subtle::Choice) -> Fp26 {
        let mut limbs = [0u32; LIMBS_26];

        for (i, slot) in limbs.iter_mut().enumerate() {
            *slot = u32::conditional_select(&a.limbs[i], &b.limbs[i], choice);
        }

        Fp26 { limbs }
    }
}

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
