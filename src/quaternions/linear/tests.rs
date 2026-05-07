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

// Cross-check `from_hnf_columns_mod` against the classical
// `from_hnf_columns` on a non-trivial diagonal-ish input. Modular HNF
// of `cols` with modulus `D` must equal the classical HNF of
// `cols ∪ D·I_4` (the explicit lattice ⟨cols⟩ + D·Z^4).
#[test]
fn hnf_mod_matches_classical_on_extended_cols() {
    // Diagonal cols + a non-identity off-diagonal column at row 0.
    // Classical HNF of these alone has pivots (2, 2, 2, 5), but the
    // 5 should reduce to gcd(5, 8) = 1 with modulus = 8.
    let cols = [
        V::new(i(2), i(0), i(0), i(0)),
        V::new(i(0), i(2), i(0), i(0)),
        V::new(i(0), i(0), i(2), i(0)),
        V::new(i(0), i(0), i(0), i(5)),
    ];
    let modulus = i(8);
    let h_mod = M::from_hnf_columns_mod::<4>(&cols, &modulus);

    // Reference: same lattice, classical HNF on cols + 8·I_4.
    let extended = [
        cols[0],
        cols[1],
        cols[2],
        cols[3],
        V::new(i(8), i(0), i(0), i(0)),
        V::new(i(0), i(8), i(0), i(0)),
        V::new(i(0), i(0), i(8), i(0)),
        V::new(i(0), i(0), i(0), i(8)),
    ];
    let h_classical = M::from_hnf_columns(&extended);

    assert_eq!(
        h_mod, h_classical,
        "mod-HNF must agree with classical HNF on the extended (cols ∪ D·I_4) input.\n\
         h_mod = {h_mod:?}\n\
         h_classical = {h_classical:?}"
    );
}

// Test our `from_hnf_columns_mod` against C ref's exact KAT 29 first-FINDUV
// inputs. C ref's `quat_lattice_alg_elem_mul` produces these
// post-multiplication columns and passes them to mod-HNF with modulus = |det|.
// Both spans the canonical lattice with covolume 64·N^4·k². The expected
// canonical HNF has diagonal (2k·2N, 2k·2N, 2N, 2N) pre-reduce-denom, where 2k
// = 0x2c1d9c..., 2N = 0x1398f6a... .
//
// Captured from `cref_kat29_iso.log` (first FINDUV_MUL block).
#[test]
fn hnf_mod_kat29_cref_first_finduv() {
    fn h(s: &str) -> BigInt<60> {
        let neg = s.starts_with('-');
        let hex = if neg { &s[1..] } else { s };
        let trimmed = hex.trim_start_matches("0x");
        let mut even = String::new();
        if trimmed.len() % 2 != 0 {
            even.push('0');
        }
        even.push_str(trimmed);
        let mut bytes_be: Vec<u8> = (0..even.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&even[i..i + 2], 16).unwrap())
            .collect();
        // little-endian
        bytes_be.reverse();
        let val = BigInt::<60>::from_bytes_le_unsigned(&bytes_be);
        if neg { val.wrapping_neg() } else { val }
    }
    type V60 = Vector<60>;
    let cols = [
        V60::new(
            h("3608eab7d8d6d9ca588157440bf4ac3330b1e5e9b59f442fc84025a4c5758c8318"),
            h("0"),
            h("0"),
            h("0"),
        ),
        V60::new(
            h("0"),
            h("-3608eab7d8d6d9ca588157440bf4ac3330b1e5e9b59f442fc84025a4c5758c8318"),
            h("0"),
            h("0"),
        ),
        V60::new(
            h("-1439ed37bf01f5f6ae4b1f17f5c4cff4a7caeabf541dce91b291b4478c8bb98c48"),
            h("18d71b690f3f0d5ed5a0e96cc64ad3e8bd31a3f5a6bbd1b020637b5a093a459c8a"),
            h("1398f6a0001a1e0db940394daddff8be32ea"),
            h("0"),
        ),
        V60::new(
            h("18d71b690f3f0d5ed5a0e96cc64ad3e8bd31a3f5a6bbd1b020637b5a093a459c8a"),
            h("1439ed37bf01f5f6ae4b1f17f5c4cff4a7caeabf541dce91b291b4478c8bb98c48"),
            h("0"),
            h("-1398f6a0001a1e0db940394daddff8be32ea"),
        ),
    ];
    // modulus = 64 · N^4 · k² = 0x111c5b8...
    let modulus = h(
        "111c5b8d70a0b3a01863b8da08c070b0bae11274203ec1452181dadff3972c8c274a2c1b776a9e3f9ab56043cb8c0308cc829042b4583090047a2dd8b16730d600dcfd6ca98478f3f15c80ed188a8b82416fd94689a49100e23f671e90cd84bc8f18bf8100",
    );

    let h_out = Matrix::<60>::from_hnf_columns_mod::<60>(&cols, &modulus);

    // Expected canonical HNF (pre-reduce-denom) from C ref's reduced_id.basis
    // multiplied by g = 2N = 0x1398f6a0001a1e0db940394daddff8be32ea on each entry.
    // Diagonal pivots: (2k · 2N, 2k · 2N, 2N, 2N).
    let two_n = h("1398f6a0001a1e0db940394daddff8be32ea");
    let two_k = h("2c1d9c251a519ad987c008d9d1dbddc");
    let pivot_01 = two_k.ct_mul(&two_n); // 2k · 2N

    // Check diagonals
    assert_eq!(
        h_out[0][0], pivot_01,
        "row 0 pivot wrong:\n  got = {:?}\n  exp = {:?}",
        h_out[0][0], pivot_01
    );
    assert_eq!(h_out[1][1], pivot_01, "row 1 pivot wrong");
    assert_eq!(h_out[2][2], two_n, "row 2 pivot wrong");
    assert_eq!(h_out[3][3], two_n, "row 3 pivot wrong");
}

// Same canonical lattice as `hnf_mod_kat29_cref_first_finduv` but with OUR
// generator set (post-`mul_direct` of our HNF basis × conj_delta, not C ref's
// L²-reduced basis × conj_delta). |det(our new_cols)| = modulus, so canonical
// HNF must give the same canonical pivots as the C ref test.
#[test]
fn hnf_mod_kat29_our_first_finduv() {
    fn h(s: &str) -> BigInt<60> {
        let neg = s.starts_with('-');
        let hex = if neg { &s[1..] } else { s };
        let trimmed = hex.trim_start_matches("0x");
        let mut even = String::new();
        if trimmed.len() % 2 != 0 {
            even.push('0');
        }
        even.push_str(trimmed);
        let mut bytes_be: Vec<u8> = (0..even.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&even[i..i + 2], 16).unwrap())
            .collect();
        bytes_be.reverse();
        let val = BigInt::<60>::from_bytes_le_unsigned(&bytes_be);
        if neg { val.wrapping_neg() } else { val }
    }
    type V60 = Vector<60>;
    // Captured from a release `MULTIORDER_TRACE=1 cargo test --lib --release
    // keygen_kat_029 -- --ignored` run, first MULTIORDER_INPUT block.
    let cols = [
        V60::new(
            h("b65b87db26d1fa3044273b1312e9d6f5cd119b4ae2e598df35b8c3a4fc2b90f82c6"),
            h("-169ca9e69539ce8cadec9331d604ff4fab2fcddddcb9a45d8d7707f536a7dee63ec"),
            h("-24bece6c0030f859bb586b71a603f2649f76c"),
            h("-34ab16ce004630c4e1dc9a00c349ec7f28d4e"),
        ),
        V60::new(
            h("169ca9e69539ce8cadec9331d604ff4fab2fcddddcb9a45d8d7707f536a7dee63ec"),
            h("b65b87db26d1fa3044273b1312e9d6f5cd119b4ae2e598df35b8c3a4fc2b90f82c6"),
            h("34ab16ce004630c4e1dc9a00c349ec7f28d4e"),
            h("-24bece6c0030f859bb586b71a603f2649f76c"),
        ),
        V60::new(
            h("3b7149b9e34b3680833dc0b7c86ef7fe1476aef24116a13ea77a31c740a97172b10"),
            h("788b444dbbce35678a6b21aeaa975ba2dcd2ee49a3d6baca9e863ab7cbe86a962c"),
            h("-7595c7c0009cb452578157d2133fd475317c"),
            h("-1398f6a0001a1e0db940394daddff8be32ea0"),
        ),
        V60::new(
            h("aed2d3964b1516d9cb8088f82840613b9f446c6648a82d328bd05ff97f6d0a4ec9a"),
            h("24d49fd34e1167f3d5512d85f269f8ae6946e114645cfce11a0329d20a01928c724"),
            h("-1125d7cc0016da4c02183223f823f9a66c8cc"),
            h("-3c04734a004ffc0a0754af7de47de9c67beca"),
        ),
    ];
    let modulus = h(
        "111c5b8d70a0b3a01863b8da08c070b0bae11274203ec1452181dadff3972c8c274a2c1b776a9e3f9ab56043cb8c0308cc829042b4583090047a2dd8b16730d600dcfd6ca98478f3f15c80ed188a8b82416fd94689a49100e23f671e90cd84bc8f18bf8100",
    );

    let h_out = Matrix::<60>::from_hnf_columns_mod::<60>(&cols, &modulus);

    // Same canonical pivots expected as the C-ref-input test:
    //   diagonal (2k · 2N, 2k · 2N, 2N, 2N)
    let two_n = h("1398f6a0001a1e0db940394daddff8be32ea");
    let two_k = h("2c1d9c251a519ad987c008d9d1dbddc");
    let pivot_01 = two_k.ct_mul(&two_n);

    // Print the diagonals if they don't match (so we see what we got).
    eprintln!("h_out[0][0] = {:?}", h_out[0][0]);
    eprintln!("h_out[1][1] = {:?}", h_out[1][1]);
    eprintln!("h_out[2][2] = {:?}", h_out[2][2]);
    eprintln!("h_out[3][3] = {:?}", h_out[3][3]);
    eprintln!("expected pivot_01 = {:?}", pivot_01);
    eprintln!("expected pivot_23 = 2N = {:?}", two_n);

    assert_eq!(h_out[0][0], pivot_01, "row 0 pivot wrong");
    assert_eq!(h_out[1][1], pivot_01, "row 1 pivot wrong");
    assert_eq!(h_out[2][2], two_n, "row 2 pivot wrong");
    assert_eq!(h_out[3][3], two_n, "row 3 pivot wrong");
}

// Regression: the modular HNF must fold the implicit `modulus · I_4`
// generators into the gcd at every pivot, otherwise the row pivot equals
// `gcd(input row entries)` instead of the canonical
// `gcd(input row entries, modulus)`.
//
// Construction: cols = (e_0, e_1, e_2, 5·e_3), modulus D = 8. Then
// gcd(5, 8) = 1, so the lattice generated by the cols plus D·I_4 is all
// of Z^4 — canonical HNF is the identity. A buggy algorithm that only
// gcds the four input cols at row 3 picks pivot = 5 (since 5 mod 8 = 5)
// instead of 1.
#[test]
fn hnf_mod_folds_in_modulus() {
    let cols = [
        V::new(i(1), i(0), i(0), i(0)),
        V::new(i(0), i(1), i(0), i(0)),
        V::new(i(0), i(0), i(1), i(0)),
        V::new(i(0), i(0), i(0), i(5)),
    ];
    let modulus = i(8);
    let h = M::from_hnf_columns_mod::<4>(&cols, &modulus);
    // Lattice generated by (e_0, e_1, e_2, 5·e_3) ∪ 8·I_4 = Z^4
    // since gcd(5, 8) = 1. Canonical HNF is the identity.
    assert_eq!(
        h,
        M::IDENTITY,
        "from_hnf_columns_mod must fold modulus into the row gcd; \
         got {h:?} but expected identity"
    );
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
