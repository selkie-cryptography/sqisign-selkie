//! Property-based tests for curve types.

use proptest::prelude::*;

use super::TorsionExponent;

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
    fn torsion_exponent_roundtrip(e in 0u32..=248) {
        let te = TorsionExponent::try_from(e).unwrap();
        prop_assert_eq!(u32::from(te), e);
    }

    #[test]
    fn torsion_exponent_rejects_over_248(e in 249u32..=1000) {
        prop_assert!(TorsionExponent::try_from(e).is_err());
    }
}
