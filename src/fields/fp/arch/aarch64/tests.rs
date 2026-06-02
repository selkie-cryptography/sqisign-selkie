//! Cross-implementation tests for the radix-29 [`Fp29`] mirror.
//!
//! Until the NEON arithmetic methods land, the only externally observable
//! behaviour of `Fp29` is its byte layout and its round-trip with
//! [`super::super::super::Fp`].  These tests pin both.

use proptest::prelude::*;

use super::{super::super::Fp, Fp29, LIMBS_29, MASK_29, RADIX_29};

/// Builds an `Fp` from arbitrary 32-byte inputs, matching the convention used
/// in [`super::super::super::tests`].
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
