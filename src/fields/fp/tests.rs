use proptest::prelude::*;
use subtle::ConstantTimeEq;

use super::*;

fn arb_fp() -> impl Strategy<Value = Fp> {
    any::<[u8; 32]>().prop_map(|b| Fp::from_bytes(&b))
}

#[test]
fn zero_is_additive_identity() {
    let a = Fp::from_small(42);
    assert_eq!(a, &a + &Fp::ZERO);
    assert_eq!(a, &Fp::ZERO + &a);
}

#[test]
fn one_is_multiplicative_identity() {
    let a = Fp::from_small(42);
    assert_eq!(a, &a * &Fp::ONE);
    assert_eq!(a, &Fp::ONE * &a);
}

#[test]
fn subtraction_is_inverse_of_addition() {
    let a = Fp::from_small(100);
    let b = Fp::from_small(42);
    let c = &a + &b;
    assert_eq!(a, &c - &b);
}

#[test]
fn negation() {
    let a = Fp::from_small(42);
    let neg_a = -&a;
    assert_eq!(Fp::ZERO, &a + &neg_a);
}

#[test]
fn multiplication_distributes() {
    let a = Fp::from_small(3);
    let b = Fp::from_small(7);
    let c = Fp::from_small(11);
    // a * (b + c) == a*b + a*c
    let lhs = &a * &(&b + &c);
    let rhs = &(&a * &b) + &(&a * &c);
    assert_eq!(lhs, rhs);
}

#[test]
fn inversion() {
    let a = Fp::from_small(42);
    let a_inv = a.invert();
    assert_eq!(Fp::ONE, &a * &a_inv);
}

#[test]
fn square_equals_mul() {
    let a = Fp::from_small(17);
    assert_eq!(a.square(), &a * &a);
}

#[test]
fn roundtrip_bytes() {
    let a = Fp::from_small(12345);
    let bytes = a.to_bytes();
    let b = Fp::from_bytes(&bytes);
    assert_eq!(a, b);
}

#[test]
fn zero_encoding() {
    let bytes = Fp::ZERO.to_bytes();
    assert_eq!(bytes, [0u8; 32]);
}

#[test]
fn sqrt_of_square() {
    let a = Fp::from_small(7);
    let a2 = a.square();
    assert!(bool::from(a2.is_square()));
    let r = a2.sqrt();
    // sqrt may return either root
    assert!(r == a || r == -a);
}

proptest! {
    #[test]
    fn prop_fp_add_commutative(a in arb_fp(), b in arb_fp()) {
        prop_assert_eq!(a + b, b + a);
    }

    #[test]
    fn prop_fp_add_associative(a in arb_fp(), b in arb_fp(), c in arb_fp()) {
        prop_assert_eq!((a + b) + c, a + (b + c));
    }

    #[test]
    fn prop_fp_add_identity(a in arb_fp()) {
        prop_assert_eq!(a + Fp::ZERO, a);
        prop_assert_eq!(Fp::ZERO + a, a);
    }

    #[test]
    fn prop_fp_add_inverse(a in arb_fp()) {
        prop_assert_eq!(a + (-a), Fp::ZERO);
        prop_assert_eq!((-a) + a, Fp::ZERO);
    }

    #[test]
    fn prop_fp_mul_commutative(a in arb_fp(), b in arb_fp()) {
        prop_assert_eq!(a * b, b * a);
    }

    #[test]
    fn prop_fp_mul_associative(a in arb_fp(), b in arb_fp(), c in arb_fp()) {
        prop_assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn prop_fp_mul_identity(a in arb_fp()) {
        prop_assert_eq!(a * Fp::ONE, a);
        prop_assert_eq!(Fp::ONE * a, a);
    }

    #[test]
    fn prop_fp_mul_zero(a in arb_fp()) {
        prop_assert_eq!(a * Fp::ZERO, Fp::ZERO);
    }

    #[test]
    fn prop_fp_distributive(a in arb_fp(), b in arb_fp(), c in arb_fp()) {
        prop_assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn prop_fp_sub_is_add_neg(a in arb_fp(), b in arb_fp()) {
        prop_assert_eq!(a - b, a + (-b));
    }

    #[test]
    fn prop_fp_double_neg(a in arb_fp()) {
        prop_assert_eq!(-(-a), a);
    }

    #[test]
    fn prop_fp_serialization_roundtrip(a in arb_fp()) {
        let bytes = a.to_bytes();
        let b = Fp::from_bytes(&bytes);
        prop_assert_eq!(a, b);
    }

    #[test]
    fn prop_fp_square_equals_mul(a in arb_fp()) {
        prop_assert_eq!(a.square(), a * a);
    }

    #[test]
    fn prop_fp_inversion(a in arb_fp()) {
        // Skip zero (not invertible).
        prop_assume!(!bool::from(a.ct_eq(&Fp::ZERO)));
        prop_assert_eq!(a * a.invert(), Fp::ONE);
    }

    #[test]
    fn prop_fp_sqrt_of_square(a in arb_fp()) {
        let a2 = a.square();
        prop_assert!(bool::from(a2.is_square()));
        let s = a2.sqrt();
        // sqrt returns either a or -a.
        prop_assert!(s == a || s == -a);
    }

    #[test]
    fn prop_fp_non_square_detected(a in arb_fp()) {
        // If a is a square, a * non_square should be a non-square
        // (product of QR x QNR = QNR). Use -1 as the QNR since
        // p = 3 (mod 4) implies -1 is not a quadratic residue.
        let neg_a = -(a.square());
        // -a^2 is a non-square unless a = 0.
        if !bool::from(a.ct_eq(&Fp::ZERO)) {
            prop_assert!(!bool::from(neg_a.is_square()));
        }
    }

    #[test]
    fn prop_fp_sum_of_products_matches_naive(
        a1 in arb_fp(), b1 in arb_fp(),
        a2 in arb_fp(), b2 in arb_fp(),
    ) {
        let fused = Fp::sum_of_products(&a1, &b1, &a2, &b2);
        let naive = &(&a1 * &b1) + &(&a2 * &b2);
        prop_assert_eq!(fused, naive);
    }

    #[test]
    fn prop_fp_difference_of_products_matches_naive(
        a1 in arb_fp(), b1 in arb_fp(),
        a2 in arb_fp(), b2 in arb_fp(),
    ) {
        let fused = Fp::difference_of_products(&a1, &b1, &a2, &b2);
        let naive = &(&a1 * &b1) - &(&a2 * &b2);
        prop_assert_eq!(fused, naive);
    }

    #[test]
    fn prop_fp_sum_of_products_4_matches_naive(
        a1 in arb_fp(), b1 in arb_fp(),
        a2 in arb_fp(), b2 in arb_fp(),
        a3 in arb_fp(), b3 in arb_fp(),
        a4 in arb_fp(), b4 in arb_fp(),
    ) {
        let fused = Fp::sum_of_products_4([
            (&a1, &b1), (&a2, &b2), (&a3, &b3), (&a4, &b4),
        ]);
        let naive = &(&(&(&a1 * &b1) + &(&a2 * &b2)) + &(&a3 * &b3)) + &(&a4 * &b4);
        prop_assert_eq!(fused, naive);
    }

    #[test]
    fn prop_fp_difference_of_products_4_matches_naive(
        a1 in arb_fp(), b1 in arb_fp(),
        a2 in arb_fp(), b2 in arb_fp(),
        a3 in arb_fp(), b3 in arb_fp(),
        a4 in arb_fp(), b4 in arb_fp(),
    ) {
        let fused = Fp::difference_of_products_4([
            (&a1, &b1), (&a2, &b2), (&a3, &b3), (&a4, &b4),
        ]);
        let naive = &(&(&(&a1 * &b1) + &(&a2 * &b2)) + &(&a3 * &b3)) - &(&a4 * &b4);
        prop_assert_eq!(fused, naive);
    }
}
