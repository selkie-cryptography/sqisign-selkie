//! Property-based tests for quaternion arithmetic algebraic invariants.
//!
//! Covers `BigInt<4>`, `Element<4>`, and `Vector<4>`.
//!
//! Run with: `cargo test --test proptest_quaternion --features expose-internals`

use proptest::prelude::*;
use sqisign_selkie::quaternions::{
    algebra::Element,
    bigint::BigInt,
    linear::Vector,
};

// ---------------------------------------------------------------------------
// Arbitrary strategies
// ---------------------------------------------------------------------------

/// Generates a random `BigInt<4>` from a sign bit and four u64 limbs.
fn arb_bigint4() -> impl Strategy<Value = BigInt<4>> {
    (any::<bool>(), any::<[u64; 4]>()).prop_map(|(neg, limbs)| {
        BigInt::from_sign_and_limbs(if neg { 1 } else { 0 }, limbs)
    })
}

/// Generates a small `BigInt<4>` (fits in i64) for tests where overflow
/// in multiplication would wrap and obscure the algebraic property.
fn arb_small_bigint4() -> impl Strategy<Value = BigInt<4>> {
    any::<i32>().prop_map(|v| BigInt::from_i64(v as i64))
}

/// Generates a random `Element<4>` from four small i64 coordinates.
/// Uses i8-range values so that chained multiplications (3 deep) and
/// norm computations stay well within BigInt<4>'s 256-bit budget.
fn arb_element4() -> impl Strategy<Value = Element<4>> {
    (any::<i8>(), any::<i8>(), any::<i8>(), any::<i8>()).prop_map(|(a, b, c, d)| {
        Element::from_i64(a as i64, b as i64, c as i64, d as i64)
    })
}

/// Generates a random `Vector<4>` from four BigInt<4> values.
fn arb_vector4() -> impl Strategy<Value = Vector<4>> {
    (arb_small_bigint4(), arb_small_bigint4(), arb_small_bigint4(), arb_small_bigint4())
        .prop_map(|(a, b, c, d)| Vector::new(a, b, c, d))
}

// ---------------------------------------------------------------------------
// BigInt<4> properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn bigint_add_commutative(a in arb_bigint4(), b in arb_bigint4()) {
        prop_assert_eq!(a + b, b + a);
    }

    // Uses small values to avoid overflow — BigInt<4> addition wraps on
    // 256-bit overflow, breaking associativity for full-range inputs.
    #[test]
    fn bigint_add_associative(a in arb_small_bigint4(), b in arb_small_bigint4(), c in arb_small_bigint4()) {
        prop_assert_eq!((a + b) + c, a + (b + c));
    }

    #[test]
    fn bigint_add_identity(a in arb_bigint4()) {
        prop_assert_eq!(a + BigInt::ZERO, a);
        prop_assert_eq!(BigInt::ZERO + a, a);
    }

    #[test]
    fn bigint_add_inverse(a in arb_bigint4()) {
        prop_assert_eq!(a + (-a), BigInt::ZERO);
        prop_assert_eq!((-a) + a, BigInt::ZERO);
    }

    #[test]
    fn bigint_sub_is_add_neg(a in arb_bigint4(), b in arb_bigint4()) {
        prop_assert_eq!(a - b, a + (-b));
    }

    #[test]
    fn bigint_double_neg(a in arb_bigint4()) {
        prop_assert_eq!(-(-a), a);
    }

    #[test]
    fn bigint_mul_commutative(a in arb_small_bigint4(), b in arb_small_bigint4()) {
        prop_assert_eq!(a * b, b * a);
    }

    #[test]
    fn bigint_mul_identity(a in arb_bigint4()) {
        prop_assert_eq!(a * BigInt::ONE, a);
        prop_assert_eq!(BigInt::ONE * a, a);
    }

    #[test]
    fn bigint_mul_zero(a in arb_bigint4()) {
        prop_assert_eq!(a * BigInt::ZERO, BigInt::ZERO);
    }

    #[test]
    fn bigint_mul_minus_one(a in arb_bigint4()) {
        prop_assert_eq!(a * BigInt::MINUS_ONE, -a);
    }

    #[test]
    fn bigint_distributive(a in arb_small_bigint4(), b in arb_small_bigint4(), c in arb_small_bigint4()) {
        prop_assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn bigint_mul_associative(a in arb_small_bigint4(), b in arb_small_bigint4(), c in arb_small_bigint4()) {
        prop_assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn bigint_abs_nonnegative(a in arb_bigint4()) {
        let abs_a = a.abs();
        // abs(a) is non-negative (sign == 0) unless a is zero.
        prop_assert!(!bool::from(abs_a.is_negative()) || bool::from(abs_a.is_zero()));
    }

    #[test]
    fn bigint_abs_idempotent(a in arb_bigint4()) {
        prop_assert_eq!(a.abs().abs(), a.abs());
    }
}

// ---------------------------------------------------------------------------
// Element<4> (quaternion) properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn element_add_commutative(a in arb_element4(), b in arb_element4()) {
        prop_assert_eq!(a.add(&b), b.add(&a));
    }

    #[test]
    fn element_add_associative(a in arb_element4(), b in arb_element4(), c in arb_element4()) {
        prop_assert_eq!(a.add(&b).add(&c), a.add(&b.add(&c)));
    }

    #[test]
    fn element_add_identity(a in arb_element4()) {
        prop_assert_eq!(a.add(&Element::ZERO), a);
    }

    #[test]
    fn element_sub_is_add_neg(a in arb_element4(), b in arb_element4()) {
        // a - b should equal a + (-b).
        // Element doesn't have Neg, so negate by scalar_mul(-1).
        let neg_b = b.scalar_mul(&BigInt::MINUS_ONE);
        prop_assert_eq!(a.sub(&b), a.add(&neg_b));
    }

    // Quaternion multiplication at width 4 involves the prime p (~248
    // bits), so products always overflow BigInt<4> even for tiny
    // coordinates. Multiplication properties (associativity,
    // distributivity, norm multiplicativity, conjugate anti-automorphism)
    // are tested in the unit tests with specific small values that are
    // known to fit. Property tests cover the additive structure and
    // conjugation involution.

    #[test]
    fn element_conjugate_involution(a in arb_element4()) {
        // conj(conj(a)) == a
        prop_assert_eq!(a.conjugate().conjugate(), a);
    }

    #[test]
    fn element_norm_of_conjugate(a in arb_element4()) {
        // nrd(conj(a)) == nrd(a)  — uses widening norm() → BigInt<8>.
        prop_assert_eq!(a.norm(), a.conjugate().norm());
    }
}

// ---------------------------------------------------------------------------
// Vector<4> properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn vector_add_commutative(a in arb_vector4(), b in arb_vector4()) {
        prop_assert_eq!(a + b, b + a);
    }

    #[test]
    fn vector_add_associative(a in arb_vector4(), b in arb_vector4(), c in arb_vector4()) {
        prop_assert_eq!((a + b) + c, a + (b + c));
    }

    #[test]
    fn vector_sub_is_add_neg(a in arb_vector4(), b in arb_vector4()) {
        prop_assert_eq!(a - b, a + (-b));
    }

    #[test]
    fn vector_double_neg(a in arb_vector4()) {
        prop_assert_eq!(-(-a), a);
    }
}
