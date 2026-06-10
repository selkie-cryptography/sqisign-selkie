//! Byte-exactness tests for [`DoublePlusExponent::from_bigint`].
//!
//! The optimized top-limb implementation must produce a bit-identical
//! mantissa and exponent to the original `abs`/shift/`to_f64_trunc`
//! formula, which itself byte-matches mini-GMP's `mini_mpz_get_d_2exp`
//! (the C reference's `dpe_set_z` path). A 1-ULP divergence would change
//! the LLL size-reduction rounding and the Lovasz test, and hence the
//! produced signature bytes.

use super::*;
use crate::quaternions::bigint::BigInt;

/// The original `from_bigint` formula, kept verbatim as the reference
/// the optimized version must match bit-for-bit.
fn from_bigint_ref<const N: usize>(v: &BigInt<N>) -> DoublePlusExponent {
    if bool::from(v.is_zero()) {
        return DoublePlusExponent::ZERO;
    }
    let bits = v.bitsize() as i64;
    let abs = v.abs();
    let shifted = if bits > f64::MAX_EXP as i64 {
        abs >> (bits - f64::MAX_EXP as i64) as u32
    } else {
        abs
    };
    let raw = shifted.to_f64_trunc();
    let mantissa = frexp(raw).0;
    let signed = if bool::from(v.is_negative()) {
        -mantissa
    } else {
        mantissa
    };
    DoublePlusExponent { m: signed, e: bits }
}

/// Asserts the optimized conversion matches the reference bit-for-bit.
fn assert_same<const N: usize>(v: &BigInt<N>) {
    let got = DoublePlusExponent::from_bigint(v);
    let want = from_bigint_ref(v);
    assert_eq!(got.e, want.e, "exponent mismatch for {v:?}");
    assert_eq!(
        got.m.to_bits(),
        want.m.to_bits(),
        "mantissa bits mismatch for {v:?}",
    );
}

fn from_limbs<const N: usize>(sign: u64, limbs: [u64; N]) -> BigInt<N> {
    BigInt::from_sign_and_limbs(sign, limbs)
}

#[test]
fn from_bigint_matches_reference_on_edges() {
    // Zero, and zero stored with a sign bit.
    assert_same(&BigInt::<8>::ZERO);
    assert_same(&from_limbs::<8>(1, [0; 8]));

    // Single-limb values exercising every clz (m < 0, m == 0, m > 0).
    for shift in 0..64u32 {
        let top = 1u64 << shift; // clz = 63 - shift
        assert_same(&from_limbs::<8>(0, [top, 0, 0, 0, 0, 0, 0, 0]));
        assert_same(&from_limbs::<8>(1, [top, 0, 0, 0, 0, 0, 0, 0]));
        assert_same(&from_limbs::<8>(
            0,
            [top | (top >> 1) | 1, 0, 0, 0, 0, 0, 0, 0],
        ));
    }

    // u64::MAX top limb (clz 0, m == -11: mask low 11 bits).
    assert_same(&from_limbs::<8>(0, [u64::MAX, 0, 0, 0, 0, 0, 0, 0]));

    // Top limb with few significant bits, forcing a pull from the next
    // limb (m > 0): top_idx = 2, top = small, low = full.
    for top in [1u64, 2, 3, 0x1F, 0xFFF, (1u64 << 11) - 1] {
        assert_same(&from_limbs::<8>(
            0,
            [0xDEAD_BEEF_CAFE_F00D, top, 0, 0, 0, 0, 0, 0],
        ));
        assert_same(&from_limbs::<8>(1, [u64::MAX, top, 0, 0, 0, 0, 0, 0]));
    }

    // Wide value (bits > 1024) at N = 20: top limbs full.
    let mut wide = [0u64; 20];
    wide[19] = 0x8000_0000_0000_0001;
    wide[18] = 0x0123_4567_89AB_CDEF;
    wide[0] = 0xFFFF_FFFF_FFFF_FFFF;
    assert_same(&from_limbs::<20>(0, wide));
    assert_same(&from_limbs::<20>(1, wide));
}

#[test]
fn from_bigint_matches_reference_random_sweep() {
    // Deterministic xorshift so the sweep is reproducible.
    let mut s: u64 = 0x1234_5678_9ABC_DEF1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    for _ in 0..20_000 {
        const N: usize = 12;
        let top_idx = 1 + (next() as usize % N);
        let mut limbs = [0u64; N];
        for limb in limbs.iter_mut().take(top_idx) {
            *limb = next();
        }
        // Vary the top limb's clz so all of m < 0 / == 0 / > 0 are hit.
        let shift = (next() % 64) as u32;
        limbs[top_idx - 1] >>= shift;
        if limbs[top_idx - 1] == 0 {
            limbs[top_idx - 1] = 1;
        }
        let sign = next() & 1;

        assert_same(&from_limbs::<N>(sign, limbs));
    }
}
