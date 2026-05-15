//! Property-based tests for curve types.

use proptest::prelude::*;

use super::{
    TorsionExponent,
    isogeny::IsogenyDegree,
    montgomery::{Coefficient, Curve},
};
use crate::fields::{fp::Fp, fp2::Fp2};

fn arb_fp2() -> impl Strategy<Value = Fp2> {
    (any::<[u8; 32]>(), any::<[u8; 32]>())
        .prop_map(|(a, b)| Fp2::new(Fp::from_bytes(&a), Fp::from_bytes(&b)))
}

fn arb_curve() -> impl Strategy<Value = Curve> {
    arb_fp2().prop_map(|a| Curve::from(Coefficient::from(a)))
}

proptest! {
    #[test]
    fn torsion_exponent_checked_sub_none_on_underflow(
        a in 0u32..=248,
        b in 0u32..=248,
    ) {
        let a_te = TorsionExponent::try_from(a).unwrap();
        let result = a_te.checked_sub(b);
        if b > a {
            prop_assert!(result.is_none(), "checked_sub should return None when b > a");
        } else {
            let v = result.unwrap();
            prop_assert_eq!(u32::from(v), a - b);
        }
    }

    #[test]
    fn prop_torsion_exponent_roundtrip(e in 0u32..=248) {
        let te = TorsionExponent::try_from(e).unwrap();
        prop_assert_eq!(u32::from(te), e);
    }

    #[test]
    fn prop_torsion_exponent_rejects_over_248(e in 249u32..=1000) {
        prop_assert!(TorsionExponent::try_from(e).is_err());
    }
}

proptest! {
    /// Coefficient roundtrip: Curve → coefficient → Curve preserves j.
    #[test]
    fn prop_curve_coefficient_roundtrip(a in arb_fp2()) {
        let c1 = Curve::from(Coefficient::from(a));
        let c2 = Curve::from(*c1.coefficient());
        prop_assert_eq!(c1.j_invariant(), c2.j_invariant());
    }

    /// Doubling constants roundtrip preserves j.
    #[test]
    fn prop_curve_doubling_constants_roundtrip(a in arb_fp2()) {
        let c1 = Curve::from(Coefficient::from(a));
        let dc = *c1.doubling_constants();
        let c2 = Curve::from(dc);
        prop_assert_eq!(c1.j_invariant(), c2.j_invariant());
    }
}

proptest! {
    #[test]
    fn prop_coefficient_serialization_roundtrip(a in arb_fp2()) {
        let coeff = Coefficient::from(a);
        let bytes = coeff.to_bytes();
        let coeff2 = Coefficient::from_bytes(&bytes);
        // Coefficients are equal if they produce the same j.
        let j1 = Curve::from(coeff).j_invariant();
        let j2 = Curve::from(coeff2).j_invariant();
        prop_assert_eq!(j1, j2);
    }
}

proptest! {
    /// new_odd rejects even values.
    #[test]
    fn prop_isogeny_degree_rejects_even(limb in any::<u64>()) {
        let even = limb & !1; // clear LSB
        prop_assert!(IsogenyDegree::new_odd([even, 0, 0, 0]).is_none());
    }

    /// new_odd rejects zero.
    #[test]
    fn prop_isogeny_degree_rejects_zero(_dummy in 0u8..1) {
        prop_assert!(IsogenyDegree::new_odd([0, 0, 0, 0]).is_none());
    }

    /// new_odd accepts odd values.
    #[test]
    fn prop_isogeny_degree_accepts_odd(limb in any::<u64>()) {
        let odd = limb | 1; // set LSB
        prop_assert!(IsogenyDegree::new_odd([odd, 0, 0, 0]).is_some());
    }

    /// Ordering is consistent with magnitude.
    #[test]
    fn prop_isogeny_degree_ordering(a in 1u64..10000, b in 1u64..10000) {
        let a_odd = a | 1;
        let b_odd = b | 1;
        let da = IsogenyDegree::new_odd([a_odd, 0, 0, 0]).unwrap();
        let db = IsogenyDegree::new_odd([b_odd, 0, 0, 0]).unwrap();
        prop_assert_eq!(da.cmp(&db), a_odd.cmp(&b_odd));
    }
}
