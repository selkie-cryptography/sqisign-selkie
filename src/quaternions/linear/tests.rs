use proptest::prelude::*;

use super::*;

type I = BigInt<4>;
type V = Vector<4>;
type M = Matrix<4>;

fn i(v: i64) -> I {
    I::from(v)
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Generates a small `BigInt<4>` (fits in i64) for tests where overflow
/// in multiplication would wrap and obscure the algebraic property.
fn arb_small_bigint4() -> impl Strategy<Value = BigInt<4>> {
    any::<i32>().prop_map(|v| BigInt::from_i64(v as i64))
}

/// Generates a random `Vector<4>` from four `BigInt<4>` values.
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
/// Uses i8-range entries so that determinant (~i8^4 * 4 terms ~ 2^29)
/// and adjugate (~i8^3 * 6 terms ~ 2^24) stay within `BigInt<4>`.
fn arb_matrix4() -> impl Strategy<Value = Matrix<4>> {
    (arb_vector4(), arb_vector4(), arb_vector4(), arb_vector4())
        .prop_map(|(r0, r1, r2, r3)| Matrix::from_rows(r0, r1, r2, r3))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[test]
fn vec_add() {
    let a = V::new(i(1), i(2), i(3), i(4));
    let b = V::new(i(10), i(20), i(30), i(40));
    let c = a + b;
    assert_eq!(c[0], i(11));
    assert_eq!(c[1], i(22));
    assert_eq!(c[2], i(33));
    assert_eq!(c[3], i(44));
}

#[test]
fn vec_dot() {
    let a = V::new(i(1), i(2), i(3), i(4));
    let b = V::new(i(5), i(6), i(7), i(8));
    assert_eq!(a.dot(&b), i(70)); // 5+12+21+32
}

#[test]
fn mat_identity() {
    let v = V::new(i(1), i(2), i(3), i(4));
    let result = M::IDENTITY * v;
    assert_eq!(result, v);
}

#[test]
fn mat_eval() {
    let mut m = M::ZERO;
    m[0][0] = i(2);
    m[1][1] = i(3);
    m[2][2] = i(4);
    m[3][3] = i(5);
    let v = V::new(i(1), i(1), i(1), i(1));
    let r = m * v;
    assert_eq!(r[0], i(2));
    assert_eq!(r[1], i(3));
    assert_eq!(r[2], i(4));
    assert_eq!(r[3], i(5));
}

#[test]
fn mat_transpose() {
    let m = M::from_rows(
        V::new(i(1), i(2), i(3), i(4)),
        V::new(i(5), i(6), i(7), i(8)),
        V::new(i(9), i(10), i(11), i(12)),
        V::new(i(13), i(14), i(15), i(16)),
    );
    let t = m.transpose();
    assert_eq!(t[0][0], i(1));
    assert_eq!(t[0][1], i(5));
    assert_eq!(t[0][2], i(9));
    assert_eq!(t[0][3], i(13));
    assert_eq!(t[1][0], i(2));
    assert_eq!(t[3][3], i(16));
}

#[test]
fn mat_mul() {
    let a = M::IDENTITY;
    let b = M::from_rows(
        V::new(i(1), i(2), i(3), i(4)),
        V::new(i(5), i(6), i(7), i(8)),
        V::new(i(9), i(10), i(11), i(12)),
        V::new(i(13), i(14), i(15), i(16)),
    );
    assert_eq!(a * b, b);
}

#[test]
fn mat_scalar_div() {
    let m = M::from_rows(
        V::new(i(6), i(12), i(18), i(24)),
        V::new(i(3), i(9), i(15), i(21)),
        V::new(i(0), i(0), i(0), i(0)),
        V::new(i(30), i(60), i(90), i(120)),
    );
    let result = m.scalar_div(&i(3)).expect("all divisible by 3");
    assert_eq!(result[0][0], i(2));
    assert_eq!(result[0][1], i(4));
    assert_eq!(result[3][3], i(40));
}

#[test]
fn mat_scalar_div_fails() {
    let m = M::from_rows(V::new(i(6), i(7), i(0), i(0)), V::ZERO, V::ZERO, V::ZERO);
    assert!(m.scalar_div(&i(3)).is_none());
}

#[test]
fn hnf_identity() {
    assert_eq!(M::IDENTITY.hnf(), M::IDENTITY);
}

#[test]
fn hnf_diagonal() {
    let m = M::from_rows(
        V::new(i(2), i(0), i(0), i(0)),
        V::new(i(0), i(3), i(0), i(0)),
        V::new(i(0), i(0), i(5), i(0)),
        V::new(i(0), i(0), i(0), i(7)),
    );
    assert_eq!(m.hnf(), m);
}

#[test]
fn hnf_upper_triangular_reduction() {
    let m = M::from_rows(
        V::new(i(2), i(5), i(0), i(0)),
        V::new(i(0), i(3), i(0), i(0)),
        V::new(i(0), i(0), i(1), i(0)),
        V::new(i(0), i(0), i(0), i(1)),
    );
    let h = m.hnf();
    assert_eq!(h[0][0], i(2));
    assert_eq!(h[0][1], i(1)); // 5 mod 2 = 1
    assert_eq!(h[1][1], i(3));
    assert!(bool::from(h[0][0].is_positive()));
    assert!(bool::from(h[1][1].is_positive()));
}

#[test]
fn hnf_is_upper_triangular() {
    let m = M::from_rows(
        V::new(i(6), i(4), i(2), i(1)),
        V::new(i(0), i(3), i(1), i(0)),
        V::new(i(0), i(0), i(5), i(2)),
        V::new(i(0), i(0), i(0), i(7)),
    );
    let h = m.hnf();
    // Lower triangle zero.
    assert!(bool::from(h[1][0].is_zero()));
    assert!(bool::from(h[2][0].is_zero()));
    assert!(bool::from(h[2][1].is_zero()));
    assert!(bool::from(h[3][0].is_zero()));
    assert!(bool::from(h[3][1].is_zero()));
    assert!(bool::from(h[3][2].is_zero()));
    // Pivots positive.
    assert!(bool::from(h[0][0].is_positive()));
    assert!(bool::from(h[1][1].is_positive()));
    assert!(bool::from(h[2][2].is_positive()));
    assert!(bool::from(h[3][3].is_positive()));
}

#[test]
fn hnf_from_8_columns() {
    let cols = [
        V::new(i(1), i(0), i(0), i(0)),
        V::new(i(0), i(1), i(0), i(0)),
        V::new(i(0), i(0), i(1), i(0)),
        V::new(i(0), i(0), i(0), i(1)),
        V::new(i(1), i(0), i(0), i(0)),
        V::new(i(0), i(1), i(0), i(0)),
        V::new(i(0), i(0), i(1), i(0)),
        V::new(i(0), i(0), i(0), i(1)),
    ];
    let h = M::from_hnf_columns(&cols);
    assert_eq!(h, M::IDENTITY);
}

// ---------------------------------------------------------------------------
// Property-based tests — Vector<4>
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

// ---------------------------------------------------------------------------
// Property-based tests — Matrix<4>
// ---------------------------------------------------------------------------

/// Builds a scalar matrix `s * I`.
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
}
