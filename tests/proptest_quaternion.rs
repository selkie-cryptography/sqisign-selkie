//! Property-based tests for quaternion arithmetic algebraic invariants.
//!
//! Covers `BigInt<4>`, `Element<4>`, and `Vector<4>`.
//!
//! Run with: `cargo test --test proptest_quaternion --features
//! expose-internals`

use proptest::prelude::*;
use sqisign_selkie::quaternions::{
    algebra::Element,
    bigint::BigInt,
    linear::{Matrix, Vector},
};

// ---------------------------------------------------------------------------
// Arbitrary strategies
// ---------------------------------------------------------------------------

/// Generates a random `BigInt<4>` from a sign bit and four u64 limbs.
fn arb_bigint4() -> impl Strategy<Value = BigInt<4>> {
    (any::<bool>(), any::<[u64; 4]>())
        .prop_map(|(neg, limbs)| BigInt::from_sign_and_limbs(if neg { 1 } else { 0 }, limbs))
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
    (any::<i8>(), any::<i8>(), any::<i8>(), any::<i8>())
        .prop_map(|(a, b, c, d)| Element::from_i64(a as i64, b as i64, c as i64, d as i64))
}

/// Generates a random `Vector<4>` from four BigInt<4> values.
fn arb_vector4() -> impl Strategy<Value = Vector<4>> {
    (
        arb_small_bigint4(),
        arb_small_bigint4(),
        arb_small_bigint4(),
        arb_small_bigint4(),
    )
        .prop_map(|(a, b, c, d)| Vector::new(a, b, c, d))
}

/// Generates a random `Matrix<4>` from four row vectors.
/// Uses i8-range entries so that determinant (~i8^4 * 4 terms ≈ 2^29)
/// and adjugate (~i8^3 * 6 terms ≈ 2^24) stay within BigInt<4>.
fn arb_matrix4() -> impl Strategy<Value = Matrix<4>> {
    (arb_vector4(), arb_vector4(), arb_vector4(), arb_vector4())
        .prop_map(|(r0, r1, r2, r3)| Matrix::from_rows(r0, r1, r2, r3))
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
// BigInt<4> division and GCD properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn bigint_div_rem_identity(a in arb_small_bigint4(), d in arb_small_bigint4()) {
        // a = q * d + r, with 0 <= r < |d|.
        prop_assume!(!bool::from(d.is_zero()));
        let (q, r) = a.div_rem(&d);
        prop_assert_eq!(q * d + r, a);
    }

    #[test]
    fn bigint_div_rem_remainder_nonnegative(a in arb_small_bigint4(), d in arb_small_bigint4()) {
        prop_assume!(!bool::from(d.is_zero()));
        let (_, r) = a.div_rem(&d);
        prop_assert!(!bool::from(r.is_negative()));
    }

    #[test]
    fn bigint_gcd_commutative(a in arb_small_bigint4(), b in arb_small_bigint4()) {
        prop_assert_eq!(a.gcd(&b), b.gcd(&a));
    }

    #[test]
    fn bigint_gcd_divides_both(a in arb_small_bigint4(), b in arb_small_bigint4()) {
        let g = a.gcd(&b);
        if !bool::from(g.is_zero()) {
            let (_, ra) = a.div_rem(&g);
            let (_, rb) = b.div_rem(&g);
            prop_assert!(bool::from(ra.is_zero()), "gcd does not divide a");
            prop_assert!(bool::from(rb.is_zero()), "gcd does not divide b");
        }
    }

    #[test]
    fn bigint_gcd_with_zero(a in arb_small_bigint4()) {
        // gcd(a, 0) = |a|.
        prop_assert_eq!(a.gcd(&BigInt::ZERO), a.abs());
    }

    #[test]
    fn bigint_gcd_idempotent(a in arb_small_bigint4()) {
        // gcd(a, a) = |a|.
        prop_assert_eq!(a.gcd(&a), a.abs());
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

    #[test]
    fn vector_dot_commutative(a in arb_vector4(), b in arb_vector4()) {
        prop_assert_eq!(a.dot(&b), b.dot(&a));
    }

    #[test]
    fn vector_dot_zero(a in arb_vector4()) {
        prop_assert!(bool::from(a.dot(&Vector::ZERO).is_zero()));
    }
}

// ---------------------------------------------------------------------------
// Matrix<4> properties
// ---------------------------------------------------------------------------

/// Helper: builds a scalar matrix `s * I`.
fn scalar_matrix(s: BigInt<4>) -> Matrix<4> {
    Matrix::from_rows(
        Vector::new(s, BigInt::ZERO, BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::ZERO, s, BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, s, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ZERO, s),
    )
}

proptest! {
    #[test]
    fn matrix_transpose_involution(a in arb_matrix4()) {
        prop_assert_eq!(a.transpose().transpose(), a);
    }

    #[test]
    fn matrix_mul_identity(a in arb_matrix4()) {
        prop_assert_eq!(a.mat_mul(&Matrix::IDENTITY), a);
        prop_assert_eq!(Matrix::IDENTITY.mat_mul(&a), a);
    }

    #[test]
    fn matrix_mul_associative(a in arb_matrix4(), b in arb_matrix4(), c in arb_matrix4()) {
        prop_assert_eq!(a.mat_mul(&b).mat_mul(&c), a.mat_mul(&b.mat_mul(&c)));
    }

    #[test]
    fn matrix_eval_identity(v in arb_vector4()) {
        prop_assert_eq!(Matrix::<4>::IDENTITY.eval(&v), v);
    }

    #[test]
    fn matrix_eval_linearity(m in arb_matrix4(), u in arb_vector4(), v in arb_vector4()) {
        // M(u + v) == M(u) + M(v)
        prop_assert_eq!(m.eval(&(u + v)), m.eval(&u) + m.eval(&v));
    }

    #[test]
    fn matrix_eval_composition(a in arb_matrix4(), b in arb_matrix4(), v in arb_vector4()) {
        // (A * B)(v) == A(B(v))
        prop_assert_eq!(a.mat_mul(&b).eval(&v), a.eval(&b.eval(&v)));
    }

    #[test]
    fn matrix_det_of_transpose(a in arb_matrix4()) {
        // det(A^T) == det(A)
        prop_assert_eq!(a.transpose().det(), a.det());
    }

    #[test]
    fn matrix_adjugate_identity(a in arb_matrix4()) {
        // A * adj(A) == det(A) * I
        let product = a.mat_mul(&a.adjugate());
        let det_i = scalar_matrix(a.det());
        prop_assert_eq!(product, det_i);
    }

    #[test]
    fn matrix_transpose_of_product(a in arb_matrix4(), b in arb_matrix4()) {
        // (A * B)^T == B^T * A^T
        prop_assert_eq!(
            a.mat_mul(&b).transpose(),
            b.transpose().mat_mul(&a.transpose())
        );
    }

    // --- Vector::PartialEq ---

    #[test]
    fn vector_eq_reflexive(a in arb_vector4()) {
        prop_assert_eq!(a, a);
    }

    #[test]
    fn vector_ne_different(a in arb_small_bigint4(), b in arb_small_bigint4()) {
        // Two vectors that differ in one component must not be equal.
        let v1 = Vector::new(a, BigInt::ZERO, BigInt::ZERO, BigInt::ZERO);
        let v2 = Vector::new(b, BigInt::ZERO, BigInt::ZERO, BigInt::ZERO);
        if a != b {
            prop_assert_ne!(v1, v2);
        }
    }
}
