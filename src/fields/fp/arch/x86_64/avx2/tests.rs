//! Cross-implementation tests for the radix-26 [`Fp26`] mirror.
//!
//! Compares Fp26's runtime arithmetic against the portable backend's
//! `Fp` via canonical-byte equivalence: the two backends are different
//! representations of the same field, so the canonical-byte image of
//! any arithmetic must agree.  Tests only fire on `target_arch =
//! "x86_64"` (the enclosing module gate) so they're invisible on the
//! aarch64-host development loop; CI runs them on x86_64 runners.

use proptest::prelude::*;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use super::{super::super::portable::Fp as PortableFp, Fp26, LIMBS_26, MASK_26, RADIX_26};

/// Builds a [`PortableFp`] from arbitrary 32-byte inputs.  The portable
/// backend's `from_bytes` is the cross-impl reference for canonical-byte
/// equivalence.
fn arb_fp() -> impl Strategy<Value = PortableFp> {
    any::<[u8; 32]>().prop_map(|b| PortableFp::from_bytes(&b))
}

/// Builds a byte string guaranteed to encode a value below `2^248 < p`,
/// so [`PortableFp::from_bytes`] round-trips it exactly.
fn arb_canonical_bytes() -> impl Strategy<Value = [u8; 32]> {
    any::<[u8; 32]>().prop_map(|mut b| {
        b[31] = 0;
        b
    })
}

/// Brings a [`PortableFp`] into Fp26 form via canonical bytes.  The
/// bridge is byte-identical to going through `from_limbs` on the
/// portable Mont limbs.
fn into_fp26(fp: PortableFp) -> Fp26 {
    Fp26::from_bytes(&fp.to_bytes())
}

proptest! {
    /// Bit-packing round-trips through 10 * 26-bit limbs without loss for
    /// any input below `2^248`.
    #[test]
    fn bytes_round_trip(canonical in arb_canonical_bytes()) {
        let fp26 = Fp26::from_bytes_le(&canonical);
        prop_assert_eq!(fp26.to_bytes_le(), canonical);
    }

    /// `Fp26::from_bytes` (Montgomery enter) then `Fp26::to_bytes`
    /// (Montgomery exit) is the identity on canonical bytes.
    #[test]
    fn from_bytes_to_bytes_round_trip(fp in arb_fp()) {
        let bytes = fp.to_bytes();
        let fp26 = Fp26::from_bytes(&bytes);
        prop_assert_eq!(fp26.to_bytes(), bytes);
    }

    /// `Fp26::from_limbs(portable.0)` agrees with `Fp26::from_bytes(&
    /// portable.to_bytes())` on canonical-byte output.  Both convert the
    /// same canonical value into Fp26 Montgomery form.
    #[test]
    fn from_limbs_matches_from_bytes(fp in arb_fp()) {
        let via_limbs = Fp26::from_limbs(fp.0);
        let via_bytes = Fp26::from_bytes(&fp.to_bytes());
        prop_assert_eq!(via_limbs.to_bytes(), via_bytes.to_bytes());
    }

    /// Cross-impl multiplication: Fp26's Mont mul must agree with the
    /// portable Fp's Mont mul after canonicalising both sides.
    #[test]
    fn fp26_mul_matches_fp_mul(a in arb_fp(), b in arb_fp()) {
        let a26 = into_fp26(a);
        let b26 = into_fp26(b);
        let prod26 = &a26 * &b26;
        let expected = (&a * &b).to_bytes();
        prop_assert_eq!(prod26.to_bytes(), expected);
    }

    /// Cross-impl addition.
    #[test]
    fn fp26_add_matches_fp_add(a in arb_fp(), b in arb_fp()) {
        let sum26 = into_fp26(a) + into_fp26(b);
        let expected = (&a + &b).to_bytes();
        prop_assert_eq!(sum26.to_bytes(), expected);
    }

    /// Cross-impl subtraction.
    #[test]
    fn fp26_sub_matches_fp_sub(a in arb_fp(), b in arb_fp()) {
        let diff26 = into_fp26(a) - into_fp26(b);
        let expected = (&a - &b).to_bytes();
        prop_assert_eq!(diff26.to_bytes(), expected);
    }

    /// Cross-impl negation.
    #[test]
    fn fp26_neg_matches_fp_neg(a in arb_fp()) {
        let neg26 = -into_fp26(a);
        let expected = (-a).to_bytes();
        prop_assert_eq!(neg26.to_bytes(), expected);
    }

    /// Cross-impl squaring.
    #[test]
    fn fp26_square_matches_fp_square(a in arb_fp()) {
        let sq26 = into_fp26(a).square();
        let expected = a.square().to_bytes();
        prop_assert_eq!(sq26.to_bytes(), expected);
    }

    /// `ConstantTimeEq` on Fp26 agrees with canonical-byte equality.
    #[test]
    fn fp26_ct_eq_matches_byte_equality(a in arb_fp(), b in arb_fp()) {
        let a26 = into_fp26(a);
        let b26 = into_fp26(b);
        let ct: bool = a26.ct_eq(&b26).into();
        prop_assert_eq!(ct, a.to_bytes() == b.to_bytes());
    }

    /// `ConditionallySelectable::conditional_select` picks `b` on
    /// `Choice(1)` and `a` on `Choice(0)`.
    #[test]
    fn fp26_conditional_select_picks_branch(a in arb_fp(), b in arb_fp()) {
        let a26 = into_fp26(a);
        let b26 = into_fp26(b);
        let pick_b = Fp26::conditional_select(&a26, &b26, Choice::from(1));
        let pick_a = Fp26::conditional_select(&a26, &b26, Choice::from(0));
        prop_assert_eq!(pick_b.to_bytes(), b.to_bytes());
        prop_assert_eq!(pick_a.to_bytes(), a.to_bytes());
    }

    /// `Fp26::from_small` agrees with `Fp::from_small` for every `u32`,
    /// including values larger than `2^26` that need the limb-1 spill.
    #[test]
    fn fp26_from_small_matches_fp_from_small(x in any::<u32>()) {
        let actual = Fp26::from_small(x).to_bytes();
        let expected = PortableFp::from_small(x).to_bytes();
        prop_assert_eq!(actual, expected);
    }

    /// `Fp26::pow2k` agrees with `Fp::pow2k` for `n` up to a small bound.
    #[test]
    fn fp26_pow2k_matches_fp_pow2k(a in arb_fp(), n in 0u32..16) {
        let actual = into_fp26(a).pow2k(n).to_bytes();
        let expected = a.pow2k(n).to_bytes();
        prop_assert_eq!(actual, expected);
    }

    /// `Fp26::invert` agrees with `Fp::invert`. Zero is excluded —
    /// invert(0) is undefined in both backends.
    #[test]
    fn fp26_invert_matches_fp_invert(a in arb_fp()) {
        prop_assume!(a.to_bytes() != PortableFp::ZERO.to_bytes());
        let actual = into_fp26(a).invert().to_bytes();
        let expected = a.invert().to_bytes();
        prop_assert_eq!(actual, expected);
    }

    /// `Fp26::is_square` agrees with `Fp::is_square` on the boolean outcome.
    #[test]
    fn fp26_is_square_matches_fp_is_square(a in arb_fp()) {
        let actual: bool = into_fp26(a).is_square().into();
        let expected: bool = a.is_square().into();
        prop_assert_eq!(actual, expected);
    }

    /// `Fp26::sqrt` agrees with `Fp::sqrt` for quadratic residues. Both
    /// backends return `+/- r`; compare via the square to dodge the sign
    /// ambiguity.
    #[test]
    fn fp26_sqrt_squares_to_input_when_qr(a in arb_fp()) {
        prop_assume!(bool::from(a.is_square()));
        let root26 = into_fp26(a).sqrt();
        let recovered = root26.square().to_bytes();
        prop_assert_eq!(recovered, a.to_bytes());
    }
}

#[test]
fn zero_round_trips() {
    let fp26 = Fp26::from_limbs(PortableFp::ZERO.0);
    assert_eq!(fp26.limbs, Fp26::ZERO.limbs);
    assert_eq!(fp26.to_bytes(), PortableFp::ZERO.to_bytes());
}

#[test]
fn one_round_trips() {
    let fp26 = Fp26::from_limbs(PortableFp::ONE.0);
    assert_eq!(fp26.to_bytes(), PortableFp::ONE.to_bytes());
}

#[test]
fn fp26_one_matches_fp_one() {
    let recovered = Fp26::from_bytes(&PortableFp::ONE.to_bytes());
    assert_eq!(recovered.to_bytes(), PortableFp::ONE.to_bytes());
}

#[test]
fn fp26_minus_one_matches_fp_minus_one() {
    let recovered = Fp26::from_bytes(&PortableFp::MINUS_ONE.to_bytes());
    assert_eq!(recovered.to_bytes(), PortableFp::MINUS_ONE.to_bytes());
}

#[test]
fn limb_layout_invariants() {
    assert_eq!(LIMBS_26, 10);
    assert_eq!(RADIX_26, 26);
    assert_eq!(MASK_26, (1u32 << RADIX_26) - 1);
}
