//! Cross-implementation tests for the radix-29 [`Fp29`] mirror.
//!
//! Until the NEON arithmetic methods land, the only externally observable
//! behaviour of `Fp29` is its byte layout and its round-trip with [`Fp`].
//! These tests pin both, and `fp29_mul_matches_fp_mul` proves the scalar
//! radix-29 Montgomery multiplication is byte-equivalent to `Fp::mul`.

use proptest::prelude::*;

use super::{super::super::Fp, Fp29, LIMBS_29, MASK_29, RADIX_29};

/// Builds an `Fp` from arbitrary 32-byte inputs, matching the convention used
/// in the parent `Fp` test module.
fn arb_fp() -> impl Strategy<Value = Fp> {
    any::<[u8; 32]>().prop_map(|b| Fp::from_bytes(&b))
}

/// Builds a byte string guaranteed to encode a value below `2^248 < p`,
/// so `Fp::from_bytes` round-trips it exactly.
fn arb_canonical_bytes() -> impl Strategy<Value = [u8; 32]> {
    any::<[u8; 32]>().prop_map(|mut b| {
        b[31] = 0;
        b
    })
}

proptest! {
    /// Bit-packing round-trips through 9 × 29-bit limbs without loss for any
    /// input below `2^248`.
    #[test]
    fn bytes_round_trip(canonical in arb_canonical_bytes()) {
        let fp29 = Fp29::from_bytes_le(&canonical);
        prop_assert_eq!(fp29.to_bytes_le(), canonical);
    }

    /// `Fp::to_bytes ∘ Fp29::from ∘ ... ∘ Fp::from` is the identity on
    /// canonical bytes, confirming the cross-representation conversion
    /// preserves the field element value.
    #[test]
    fn fp_to_fp29_to_fp_preserves_value(fp in arb_fp()) {
        let fp29 = Fp29::from(fp);
        let fp_back = Fp::from(fp29);
        prop_assert_eq!(fp.to_bytes(), fp_back.to_bytes());
    }

    /// `Fp29::from_bytes_le` agrees with `Fp::from_bytes ∘ Fp::to_bytes`
    /// for canonical inputs: both must materialise the same integer value
    /// and re-emit the same bytes.
    #[test]
    fn fp_and_fp29_emit_same_canonical_bytes(canonical in arb_canonical_bytes()) {
        let fp = Fp::from_bytes(&canonical);
        let fp29 = Fp29::from_bytes_le(&canonical);
        prop_assert_eq!(fp.to_bytes(), fp29.to_bytes_le());
    }

    /// Cross-impl multiplication: `Fp29::mul` must agree with `Fp::mul`
    /// after converting both inputs and converting the product back.
    /// This is the gold-standard correctness check for the radix-29
    /// CIOS Montgomery implementation.
    #[test]
    fn fp29_mul_matches_fp_mul(a in arb_fp(), b in arb_fp()) {
        let a29 = Fp29::from(a);
        let b29 = Fp29::from(b);
        let product29 = a29.mul(&b29);
        let product_back = Fp::from(product29);
        let expected = &a * &b;
        prop_assert_eq!(product_back.to_bytes(), expected.to_bytes());
    }

    /// Cross-impl addition: limbwise add, then reduce-to-`[0, 2p)`, must
    /// agree with `Fp::add` after the canonical-bytes round-trip.
    #[test]
    fn fp29_add_matches_fp_add(a in arb_fp(), b in arb_fp()) {
        let sum29 = Fp29::from(a).add(Fp29::from(b));
        let sum_back = Fp::from(sum29);
        let expected = &a + &b;
        prop_assert_eq!(sum_back.to_bytes(), expected.to_bytes());
    }

    /// Cross-impl subtraction: limbwise wrapping-sub, then add-`2p`-on-borrow,
    /// must agree with `Fp::sub`.
    #[test]
    fn fp29_sub_matches_fp_sub(a in arb_fp(), b in arb_fp()) {
        let diff29 = Fp29::from(a).sub(Fp29::from(b));
        let diff_back = Fp::from(diff29);
        let expected = &a - &b;
        prop_assert_eq!(diff_back.to_bytes(), expected.to_bytes());
    }

    /// Cross-impl squaring: currently delegates to `mul(self, self)`, so the
    /// test mainly pins the boundary; an optimised symmetric square lands
    /// alongside the NEON intrinsic commit.
    #[test]
    fn fp29_square_matches_fp_square(a in arb_fp()) {
        let sq29 = Fp29::from(a).square();
        let sq_back = Fp::from(sq29);
        let expected = a.square();
        prop_assert_eq!(sq_back.to_bytes(), expected.to_bytes());
    }
}

#[test]
fn zero_round_trips() {
    let fp29 = Fp29::from(Fp::ZERO);
    assert_eq!(fp29.limbs, Fp29::ZERO.limbs);
    assert_eq!(Fp::from(fp29).to_bytes(), Fp::ZERO.to_bytes());
}

#[test]
fn one_round_trips() {
    let fp29 = Fp29::from(Fp::ONE);
    let fp_back = Fp::from(fp29);
    assert_eq!(fp_back.to_bytes(), Fp::ONE.to_bytes());
}

#[test]
fn minus_one_round_trips() {
    let fp29 = Fp29::from(Fp::MINUS_ONE);
    let fp_back = Fp::from(fp29);
    assert_eq!(fp_back.to_bytes(), Fp::MINUS_ONE.to_bytes());
}

#[test]
fn limb_layout_invariants() {
    assert_eq!(LIMBS_29, 9);
    assert_eq!(RADIX_29, 29);
    assert_eq!(MASK_29, (1u32 << RADIX_29) - 1);
}
