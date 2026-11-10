use proptest::prelude::*;
use subtle::ConstantTimeEq;

use super::*;

fn arb_fp() -> impl Strategy<Value = Fp> {
    any::<[u8; FP_ENCODED_BYTES]>().prop_map(|mut b| {
        // Keep the value below p: clear the top byte's high bits so the
        // decoder's canonical-input contract holds (p's top byte is 0x2f).
        b[FP_ENCODED_BYTES - 1] &= 0x0F;
        Fp::from_bytes(&b)
    })
}

/// Decodes a lowercase little-endian hex string of `FP_ENCODED_BYTES` bytes.
fn fp_from_hex(hex: &str) -> Fp {
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();
    Fp::from_bytes(bytes.as_slice().try_into().unwrap())
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
fn constants_are_consistent() {
    assert_eq!(Fp::from_small(1), Fp::ONE);
    assert_eq!(Fp::from_small(2), Fp::TWO);
    assert_eq!(Fp::from_small(4), Fp::FOUR);
    assert_eq!(-Fp::ONE, Fp::MINUS_ONE);
    assert_eq!(&Fp::TWO_INV * &Fp::TWO, Fp::ONE);
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
    assert_eq!(bytes, [0u8; FP_ENCODED_BYTES]);
}

#[test]
fn minus_one_encodes_as_p_minus_one() {
    let mut expected = [0xFFu8; FP_ENCODED_BYTES];
    expected[0] = 0xFE;
    expected[FP_ENCODED_BYTES - 1] = 0x2F;
    assert_eq!(Fp::MINUS_ONE.to_bytes(), expected);
}

/// `2^((p-3)/4)`, `2^-1`, and `sqrt(2)` computed with Python's `pow`
/// on `p = 3 * 2^324 - 1`.
#[test]
fn exponent_chain_matches_reference_vectors() {
    let two = Fp::from_small(2);
    assert_eq!(
        two.pow_p3div4(),
        fp_from_hex(
            "5c296b5259d483f7f3dc6640972ca79e6924e592ff30e915160da1e75c1b913dfc5a6b651d5253ed02"
        )
    );
    assert_eq!(
        two.invert(),
        fp_from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000000000000000000018"
        )
    );
    assert!(bool::from(two.is_square()));
    assert_eq!(
        two.sqrt(),
        fp_from_hex(
            "b852d6a4b2a807efe7b9cd802e594e3dd348ca25ff61d22b2c1a42cfb936227bf8b5d6ca3aa4a6da05"
        )
    );
}

#[test]
fn sqrt_of_square() {
    let a = Fp::from_small(7);
    let a2 = a.square();
    assert!(bool::from(a2.is_square()));
    let r = a2.sqrt();
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
        prop_assume!(!bool::from(a.ct_eq(&Fp::ZERO)));
        prop_assert_eq!(a * a.invert(), Fp::ONE);
    }

    #[test]
    fn prop_fp_sqrt_of_square(a in arb_fp()) {
        let a2 = a.square();
        prop_assert!(bool::from(a2.is_square()));
        let s = a2.sqrt();
        prop_assert!(s == a || s == -a);
    }

    #[test]
    fn prop_fp_non_square_detected(a in arb_fp()) {
        // p = 3 (mod 4), so -1 is a non-residue and -a^2 is a
        // non-square unless a = 0.
        let neg_a = -(a.square());
        if !bool::from(a.ct_eq(&Fp::ZERO)) {
            prop_assert!(!bool::from(neg_a.is_square()));
        }
    }

    #[test]
    fn prop_fp_sum_of_2_products_matches_naive(
        a1 in arb_fp(), b1 in arb_fp(),
        a2 in arb_fp(), b2 in arb_fp(),
    ) {
        let fused = Fp::sum_of_2_products(&a1, &b1, &a2, &b2);
        let naive = &(&a1 * &b1) + &(&a2 * &b2);
        prop_assert_eq!(fused, naive);
    }

    #[test]
    fn prop_fp_difference_of_2_products_matches_naive(
        a1 in arb_fp(), b1 in arb_fp(),
        a2 in arb_fp(), b2 in arb_fp(),
    ) {
        let fused = Fp::difference_of_2_products(&a1, &b1, &a2, &b2);
        let naive = &(&a1 * &b1) - &(&a2 * &b2);
        prop_assert_eq!(fused, naive);
    }
}
