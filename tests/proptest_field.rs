//! Property-based tests for field arithmetic algebraic invariants.
//!
//! Run with: `cargo test --test proptest_field --features expose-internals`

use proptest::prelude::*;
use sqisign_selkie::fields::{fp::Fp, fp2::Fp2};

// --- Fp properties ---

fn arb_fp() -> impl Strategy<Value = Fp> {
    any::<[u8; 32]>().prop_map(|b| Fp::from_bytes(&b))
}

proptest! {
    #[test]
    fn fp_add_commutative(a in arb_fp(), b in arb_fp()) {
        prop_assert_eq!(a + b, b + a);
    }

    #[test]
    fn fp_add_associative(a in arb_fp(), b in arb_fp(), c in arb_fp()) {
        prop_assert_eq!((a + b) + c, a + (b + c));
    }

    #[test]
    fn fp_add_identity(a in arb_fp()) {
        prop_assert_eq!(a + Fp::ZERO, a);
        prop_assert_eq!(Fp::ZERO + a, a);
    }

    #[test]
    fn fp_add_inverse(a in arb_fp()) {
        prop_assert_eq!(a + (-a), Fp::ZERO);
        prop_assert_eq!((-a) + a, Fp::ZERO);
    }

    #[test]
    fn fp_mul_commutative(a in arb_fp(), b in arb_fp()) {
        prop_assert_eq!(a * b, b * a);
    }

    #[test]
    fn fp_mul_associative(a in arb_fp(), b in arb_fp(), c in arb_fp()) {
        prop_assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn fp_mul_identity(a in arb_fp()) {
        prop_assert_eq!(a * Fp::ONE, a);
        prop_assert_eq!(Fp::ONE * a, a);
    }

    #[test]
    fn fp_mul_zero(a in arb_fp()) {
        prop_assert_eq!(a * Fp::ZERO, Fp::ZERO);
    }

    #[test]
    fn fp_distributive(a in arb_fp(), b in arb_fp(), c in arb_fp()) {
        prop_assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn fp_sub_is_add_neg(a in arb_fp(), b in arb_fp()) {
        prop_assert_eq!(a - b, a + (-b));
    }

    #[test]
    fn fp_double_neg(a in arb_fp()) {
        prop_assert_eq!(-(-a), a);
    }

    #[test]
    fn fp_serialization_roundtrip(a in arb_fp()) {
        let bytes = a.to_bytes();
        let b = Fp::from_bytes(&bytes);
        prop_assert_eq!(a, b);
    }
}

// --- Fp2 properties ---

fn arb_fp2() -> impl Strategy<Value = Fp2> {
    (arb_fp(), arb_fp()).prop_map(|(a, b)| Fp2::new(a, b))
}

proptest! {
    #[test]
    fn fp2_add_commutative(a in arb_fp2(), b in arb_fp2()) {
        prop_assert_eq!(a + b, b + a);
    }

    #[test]
    fn fp2_add_associative(a in arb_fp2(), b in arb_fp2(), c in arb_fp2()) {
        prop_assert_eq!((a + b) + c, a + (b + c));
    }

    #[test]
    fn fp2_add_identity(a in arb_fp2()) {
        prop_assert_eq!(a + Fp2::ZERO, a);
    }

    #[test]
    fn fp2_mul_commutative(a in arb_fp2(), b in arb_fp2()) {
        prop_assert_eq!(a * b, b * a);
    }

    #[test]
    fn fp2_mul_associative(a in arb_fp2(), b in arb_fp2(), c in arb_fp2()) {
        prop_assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn fp2_mul_identity(a in arb_fp2()) {
        prop_assert_eq!(a * Fp2::ONE, a);
    }

    #[test]
    fn fp2_distributive(a in arb_fp2(), b in arb_fp2(), c in arb_fp2()) {
        prop_assert_eq!(a * (b + c), a * b + a * c);
    }
}
