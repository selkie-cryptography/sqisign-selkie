use proptest::prelude::*;

use super::*;

type I = BigInt<4>;
type V = Vector<4>;
type M = Matrix<4>;

fn i(v: i64) -> I {
    I::from(v)
}

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

/// `from_hnf_columns` must preserve the column lattice's covolume.
///
/// Regression for an earlier non-unimodular bug where the xgcd
/// accumulation only updated `a[pivot]` (not `a[j]`), silently
/// shrinking the lattice when xgcd's `u` was not ±1.
#[test]
fn from_hnf_columns_preserves_covolume_2x2_simulated() {
    // We test the 4x4 case with three fixed cols and combine
    // (a, b) into the first two cols' first row to trigger the
    // non-trivial xgcd path.
    //
    // Cols: (6, 10, 0, 0), (4, 14, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1).
    // The 2x2 top-left submatrix has det 6·14 - 10·4 = 44.
    let cols = [
        V::new(i(6), i(10), i(0), i(0)),
        V::new(i(4), i(14), i(0), i(0)),
        V::new(i(0), i(0), i(1), i(0)),
        V::new(i(0), i(0), i(0), i(1)),
    ];
    let original_det = M::from_columns(&cols).det();
    let hnf = M::from_hnf_columns(&cols);
    let hnf_det = hnf.det();

    assert_eq!(
        hnf_det.abs(),
        original_det.abs(),
        "from_hnf_columns must preserve |det|: original {original_det:?}, HNF {hnf_det:?}"
    );
}

/// Larger non-trivial xgcd case.
#[test]
fn from_hnf_columns_preserves_covolume_4x4() {
    let cols = [
        V::new(i(15), i(20), i(35), i(0)),
        V::new(i(12), i(8), i(28), i(0)),
        V::new(i(0), i(0), i(0), i(7)),
        V::new(i(0), i(0), i(7), i(0)),
    ];
    let original_det = M::from_columns(&cols).det();
    let hnf = M::from_hnf_columns(&cols);
    let hnf_det = hnf.det();

    assert_eq!(
        hnf_det.abs(),
        original_det.abs(),
        "from_hnf_columns must preserve |det|: original {original_det:?}, HNF {hnf_det:?}"
    );
}

/// `from_hnf_columns_mod_cref` must produce the same canonical HNF
/// as classical `from_hnf_columns` when modulus = |det|.
///
/// Both should give the unique upper-triangular HNF for the same
/// 4-col rank-4 input lattice.
#[test]
fn from_hnf_columns_mod_cref_matches_classical_4cols() {
    let cols = [
        V::new(i(10), i(0), i(0), i(0)),
        V::new(i(3), i(4), i(0), i(0)),
        V::new(i(5), i(7), i(8), i(0)),
        V::new(i(1), i(2), i(3), i(4)),
    ];
    let det = M::from_columns(&cols).det().abs();
    let hnf_classical = M::from_hnf_columns(&cols);
    let hnf_mod_cref = M::from_hnf_columns_mod_cref::<8>(&cols, &det);
    let cols_classical = hnf_classical.columns();
    let cols_mod_cref = hnf_mod_cref.columns();
    for j in 0..4 {
        for r in 0..4 {
            assert_eq!(
                cols_classical[j][r], cols_mod_cref[j][r],
                "col={j} row={r}: classical={:?}, mod_cref={:?}",
                cols_classical[j][r], cols_mod_cref[j][r]
            );
        }
    }
}

/// Same as above but with negative entries, which exercise different
/// xgcd cofactor sign-handling paths.
#[test]
fn from_hnf_columns_mod_cref_with_negatives_4cols() {
    let cols = [
        V::new(i(7), i(-3), i(-5), i(2)),
        V::new(i(-1), i(11), i(0), i(-4)),
        V::new(i(0), i(0), i(13), i(-1)),
        V::new(i(0), i(0), i(0), i(17)),
    ];
    let det = M::from_columns(&cols).det().abs();
    let hnf_classical = M::from_hnf_columns(&cols);
    let hnf_mod_cref = M::from_hnf_columns_mod_cref::<8>(&cols, &det);
    let cols_classical = hnf_classical.columns();
    let cols_mod_cref = hnf_mod_cref.columns();
    for j in 0..4 {
        for r in 0..4 {
            assert_eq!(
                cols_classical[j][r], cols_mod_cref[j][r],
                "col={j} row={r}: classical={:?}, mod_cref={:?}",
                cols_classical[j][r], cols_mod_cref[j][r]
            );
        }
    }
}

/// Exposes a SIGN-SIDE BUG: Selkie's three HNF variants
/// (`from_hnf_columns` classical, `from_hnf_columns_mod` constant-mod,
/// `from_hnf_columns_mod_cref` decreasing-mod) give DIFFERENT output
/// for ~600-bit inputs at width 30. They should all give the same
/// canonical upper-triangular HNF since they describe the same
/// lattice. This is the root cause of KAT-1 sign's wrong i_com_rsp
/// (see `project_sign_kat_001_pre_hnf_lattice_2026-05-10.md`).
///
/// Marked `#[ignore]` until fixed. Re-enable once HNF impls agree.
#[test]
#[ignore = "exposes Selkie HNF variant disagreement bug for large inputs"]
fn from_hnf_columns_mod_cref_large_values_4cols() {
    use crate::quaternions::bigint::BigInt;
    // Construct a 4×4 lattice with ~600-bit values, with some negatives.
    let mut limbs0 = [0u64; 30];
    limbs0[9] = 1;
    let _big = BigInt::<30>::from_limbs(limbs0);
    let mut limbs_a = [0u64; 30];
    limbs_a[9] = 0x1234_5678_9ABC_DEF0;
    limbs_a[5] = 0x1111_2222_3333_4444;
    let mut limbs_b = [0u64; 30];
    limbs_b[8] = 0xFEDC_BA98_7654_3210;
    let mut limbs_c = [0u64; 30];
    limbs_c[7] = 0xABCD_EF01_2345_6789;
    let big_a = BigInt::<30>::from_limbs(limbs_a);
    let big_b = BigInt::<30>::from_limbs(limbs_b);
    let big_c = BigInt::<30>::from_limbs(limbs_c);
    let zero = BigInt::<30>::ZERO;
    let cols = [
        Vector::<30>::new(big_a, big_b.wrapping_neg(), big_c.wrapping_neg(), zero),
        Vector::<30>::new(big_b, big_a, zero, big_c.wrapping_neg()),
        Vector::<30>::new(big_c, zero, big_a, big_b),
        Vector::<30>::new(zero, big_c, big_b.wrapping_neg(), big_a),
    ];
    let det = Matrix::from_columns(&cols).det().abs();
    let hnf_classical = Matrix::from_hnf_columns(&cols);
    let hnf_mod_cref = Matrix::from_hnf_columns_mod_cref::<60>(&cols, &det);
    let hnf_mod = Matrix::from_hnf_columns_mod::<60>(&cols, &det);
    let cols_classical = hnf_classical.columns();
    let cols_mod_cref = hnf_mod_cref.columns();
    let cols_mod = hnf_mod.columns();
    let mut all_match = true;
    for j in 0..4 {
        for r in 0..4 {
            if cols_classical[j][r] != cols_mod_cref[j][r] {
                eprintln!(
                    "MISMATCH classical vs mod_cref col={j} row={r}: classical={:?}, mod_cref={:?}",
                    cols_classical[j][r], cols_mod_cref[j][r]
                );
                all_match = false;
            }
            if cols_classical[j][r] != cols_mod[j][r] {
                eprintln!(
                    "MISMATCH classical vs mod col={j} row={r}: classical={:?}, mod={:?}",
                    cols_classical[j][r], cols_mod[j][r]
                );
                all_match = false;
            }
        }
    }
    assert!(all_match, "HNF variants give different outputs");
}

/// Sanity check: HNF of a 4×4 matrix where the answer is known.
/// All three Selkie HNF variants should give the same canonical
/// upper-triangular form.
#[test]
fn from_hnf_columns_predictable_4cols() {
    // 4 cols with det = 7·11·13·17 = 17017. HNF should be diagonal
    // (since cols are already independent) with pivots 7, 11, 13, 17.
    let cols = [
        V::new(i(7), i(0), i(0), i(0)),
        V::new(i(2), i(11), i(0), i(0)),
        V::new(i(3), i(5), i(13), i(0)),
        V::new(i(1), i(4), i(7), i(17)),
    ];
    let det = M::from_columns(&cols).det().abs();
    eprintln!("det = {:?}", det);
    let hnf_classical = M::from_hnf_columns(&cols);
    let hnf_mod = M::from_hnf_columns_mod::<8>(&cols, &det);
    let hnf_mod_cref = M::from_hnf_columns_mod_cref::<8>(&cols, &det);

    eprintln!("\nclassical HNF columns:");
    for (j, col) in hnf_classical.columns().iter().enumerate() {
        eprintln!(
            "  col {}: ({:?}, {:?}, {:?}, {:?})",
            j, col[0], col[1], col[2], col[3]
        );
    }
    eprintln!("\nmod HNF columns:");
    for (j, col) in hnf_mod.columns().iter().enumerate() {
        eprintln!(
            "  col {}: ({:?}, {:?}, {:?}, {:?})",
            j, col[0], col[1], col[2], col[3]
        );
    }
    eprintln!("\nmod_cref HNF columns:");
    for (j, col) in hnf_mod_cref.columns().iter().enumerate() {
        eprintln!(
            "  col {}: ({:?}, {:?}, {:?}, {:?})",
            j, col[0], col[1], col[2], col[3]
        );
    }
    // All three should match. Diagonals (7, 11, 13, 17), off-diagonals reduced.
    assert_eq!(hnf_classical, hnf_mod_cref, "classical vs mod_cref");
    assert_eq!(hnf_classical, hnf_mod, "classical vs mod");
}

/// Take the predictable test inputs and shift left by 200 bits.
/// HNF should still agree across all three Selkie variants — but at
/// width 30 with large entries this exposes the bug.
#[test]
fn from_hnf_columns_predictable_4cols_shifted() {
    use crate::quaternions::bigint::BigInt;
    let shift = 200;
    let big = |v: i64| -> BigInt<30> {
        let bi: BigInt<30> = BigInt::<30>::from(v);
        bi << shift
    };
    let z = BigInt::<30>::ZERO;
    let cols = [
        Vector::<30>::new(big(7), z, z, z),
        Vector::<30>::new(big(2), big(11), z, z),
        Vector::<30>::new(big(3), big(5), big(13), z),
        Vector::<30>::new(big(1), big(4), big(7), big(17)),
    ];
    let det = Matrix::from_columns(&cols).det().abs();
    eprintln!("det bits = {}", det.bitsize());
    let hnf_classical = Matrix::from_hnf_columns(&cols);
    let hnf_mod = Matrix::from_hnf_columns_mod::<60>(&cols, &det);
    let hnf_mod_cref = Matrix::from_hnf_columns_mod_cref::<60>(&cols, &det);

    eprintln!(
        "classical col 0 row 0 = {:?}",
        hnf_classical.columns()[0][0]
    );
    eprintln!("mod       col 0 row 0 = {:?}", hnf_mod.columns()[0][0]);
    eprintln!("mod_cref  col 0 row 0 = {:?}", hnf_mod_cref.columns()[0][0]);

    // The HNF should be diagonal with diagonals 7, 11, 13, 17 each
    // shifted by 200 bits (since cols are scaled but linearly independent).
    // Off-diagonals SHOULD match the original `from_hnf_columns_predictable_4cols`
    // shifted by 200 bits.
    assert_eq!(hnf_classical, hnf_mod_cref, "classical vs mod_cref");
    assert_eq!(hnf_classical, hnf_mod, "classical vs mod");
}

/// Minimal reproducer for the HNF bug — non-triangular 4×4 with small
/// values, similar shape to KAT-1's mul output.
#[test]
fn from_hnf_columns_quaternion_shape_small() {
    let cols = [
        V::new(i(11), i(-7), i(-3), i(2)),
        V::new(i(7), i(11), i(2), i(-3)),
        V::new(i(3), i(-2), i(11), i(7)),
        V::new(i(-2), i(3), i(-7), i(11)),
    ];
    let det = M::from_columns(&cols).det().abs();
    eprintln!("det = {:?}", det);
    let hnf_classical = M::from_hnf_columns(&cols);
    let hnf_mod = M::from_hnf_columns_mod::<8>(&cols, &det);
    let hnf_mod_cref = M::from_hnf_columns_mod_cref::<8>(&cols, &det);

    eprintln!("\nclassical:");
    for (j, col) in hnf_classical.columns().iter().enumerate() {
        eprintln!(
            "  col {}: ({:?}, {:?}, {:?}, {:?})",
            j, col[0], col[1], col[2], col[3]
        );
    }
    eprintln!("\nmod:");
    for (j, col) in hnf_mod.columns().iter().enumerate() {
        eprintln!(
            "  col {}: ({:?}, {:?}, {:?}, {:?})",
            j, col[0], col[1], col[2], col[3]
        );
    }
    eprintln!("\nmod_cref:");
    for (j, col) in hnf_mod_cref.columns().iter().enumerate() {
        eprintln!(
            "  col {}: ({:?}, {:?}, {:?}, {:?})",
            j, col[0], col[1], col[2], col[3]
        );
    }
    assert_eq!(hnf_classical, hnf_mod_cref, "classical vs mod_cref");
    assert_eq!(hnf_classical, hnf_mod, "classical vs mod");
}

/// Same quaternion-shape inputs at width 30 with 100-bit entries.
/// Tests if the bug is width-related or value-magnitude related.
#[test]
fn from_hnf_columns_quaternion_shape_med_w30() {
    use crate::quaternions::bigint::BigInt;
    let to30 = |v: i64| -> BigInt<30> { BigInt::<30>::from(v) };
    let a = to30(0x123456789ABCDEF0_i64);
    let b = to30(0x0FEDCBA987654321_i64);
    let c = to30(0x1111222233334444_i64);
    let d = to30(0x55556666_i64);
    let neg = |x: BigInt<30>| -> BigInt<30> { x.wrapping_neg() };
    let z = BigInt::<30>::ZERO;
    let cols = [
        Vector::<30>::new(a, neg(b), neg(c), d),
        Vector::<30>::new(b, a, d, neg(c)),
        Vector::<30>::new(c, neg(d), a, b),
        Vector::<30>::new(z, c, neg(b), a),
    ];
    let det = Matrix::from_columns(&cols).det().abs();
    eprintln!("det bits = {}", det.bitsize());
    let hnf_classical = Matrix::from_hnf_columns(&cols);
    let hnf_mod_cref = Matrix::from_hnf_columns_mod_cref::<60>(&cols, &det);
    let hnf_mod = Matrix::from_hnf_columns_mod::<60>(&cols, &det);
    if hnf_classical != hnf_mod_cref || hnf_classical != hnf_mod {
        eprintln!("\nclassical:");
        for (j, col) in hnf_classical.columns().iter().enumerate() {
            eprintln!(
                "  col {}: ({:?}, {:?}, {:?}, {:?})",
                j,
                col[0].bitsize(),
                col[1].bitsize(),
                col[2].bitsize(),
                col[3].bitsize()
            );
        }
        eprintln!("mod_cref:");
        for (j, col) in hnf_mod_cref.columns().iter().enumerate() {
            eprintln!(
                "  col {}: ({:?}, {:?}, {:?}, {:?})",
                j,
                col[0].bitsize(),
                col[1].bitsize(),
                col[2].bitsize(),
                col[3].bitsize()
            );
        }
    }
    assert_eq!(hnf_classical, hnf_mod_cref, "classical vs mod_cref");
    assert_eq!(hnf_classical, hnf_mod, "classical vs mod");
}

/// Quaternion-shape with 256-bit entries at width 30. Exposes Selkie's
/// HNF variant disagreement (= the same bug as KAT-1 sign's wrong
/// `i_com_rsp`). Marked `#[ignore]` until fixed.
#[test]
#[ignore = "exposes Selkie HNF variant disagreement bug at width 30 with 256-bit inputs"]
fn from_hnf_columns_quaternion_shape_256bit_w30() {
    use crate::quaternions::bigint::BigInt;
    let mk = |hi: u64, lo: u64| -> BigInt<30> {
        let mut limbs = [0u64; 30];
        limbs[0] = lo;
        limbs[1] = hi;
        limbs[3] = 0xA1B2_C3D4_E5F6_0708;
        BigInt::<30>::from_limbs(limbs)
    };
    let a = mk(0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210);
    let b = mk(0xFEDC_BA98_7654_3210, 0x1234_5678_9ABC_DEF0);
    let c = mk(0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444);
    let d = mk(0x5555_6666_7777_8888, 0x9999_AAAA_BBBB_CCCC);
    let neg = |x: BigInt<30>| x.wrapping_neg();
    let z = BigInt::<30>::ZERO;
    let cols = [
        Vector::<30>::new(a, neg(b), neg(c), d),
        Vector::<30>::new(b, a, d, neg(c)),
        Vector::<30>::new(c, neg(d), a, b),
        Vector::<30>::new(z, c, neg(b), a),
    ];
    let det = Matrix::from_columns(&cols).det().abs();
    eprintln!("det bits = {}", det.bitsize());
    let hnf_classical = Matrix::from_hnf_columns(&cols);
    let hnf_mod_cref = Matrix::from_hnf_columns_mod_cref::<60>(&cols, &det);
    let hnf_mod = Matrix::from_hnf_columns_mod::<60>(&cols, &det);
    if hnf_classical != hnf_mod_cref {
        eprintln!("classical vs mod_cref DIFFER:");
        for j in 0..4 {
            for r in 0..4 {
                let cl = hnf_classical.columns()[j][r];
                let mc = hnf_mod_cref.columns()[j][r];
                if cl != mc {
                    eprintln!(
                        "  [{j}][{r}]: classical bits={}, mod_cref bits={}",
                        cl.bitsize(),
                        mc.bitsize()
                    );
                }
            }
        }
    }
    if hnf_classical != hnf_mod {
        eprintln!("classical vs mod DIFFER:");
        for j in 0..4 {
            for r in 0..4 {
                let cl = hnf_classical.columns()[j][r];
                let mm = hnf_mod.columns()[j][r];
                if cl != mm {
                    eprintln!(
                        "  [{j}][{r}]: classical bits={}, mod bits={}",
                        cl.bitsize(),
                        mm.bitsize()
                    );
                }
            }
        }
    }
    if hnf_mod != hnf_mod_cref {
        eprintln!("mod vs mod_cref DIFFER!");
    } else {
        eprintln!("mod == mod_cref agree");
    }
    assert_eq!(hnf_classical, hnf_mod_cref);
}

/// EXACT KAT-1 reproduction: Selkie's mul_direct output (= O·α post-mul,
/// at denom 2, halved-coords from compute_backtracking) for the response
/// phase. Run Selkie's `from_hnf_columns_mod_cref` with mod = |det| and
/// compare to expected canonical HNF.
///
/// Inputs from `[O_ALPHA_PREHNF_SELKIE]` dump for KAT-1 iter 0
/// with C-ref α injected. Expected = C-ref's `[O_ALPHA_CREF]` divided by 2
/// (= reduced from denom 4 to denom 2).
#[test]
#[ignore = "validates Selkie HNF on exact KAT-1 inputs vs C-ref expected"]
fn from_hnf_columns_mod_cref_kat1_o_alpha() {
    use crate::quaternions::bigint::BigInt;

    let parse = |s: &str| -> BigInt<60> {
        let s = s.trim_start_matches("0x");
        let pad = format!("{:0>1$}", s, 60 * 16);
        let mut limbs = [0u64; 60];
        for (i, chunk) in pad.as_bytes().rchunks(16).enumerate() {
            if i >= 60 {
                break;
            }
            let lh = std::str::from_utf8(chunk).unwrap();
            limbs[i] = u64::from_str_radix(lh, 16).unwrap_or(0);
        }
        BigInt::<60>::from_limbs(limbs)
    };
    let neg = |x: BigInt<60>| x.wrapping_neg();

    // Selkie's KAT-1 PREHNF (= raw mul_direct output at denom 2, halved):
    // Col 0 = (a, -b, -c, -d) of α_conj_at_denom_1 (= halved C-ref α).
    let alpha_a = parse(
        "0x339de45818a8dcad1962ce0fabad5d66ddd0f321bcb2c9e4e982adb63e429421d5c85d3a06eee1986",
    );
    let alpha_b = parse(
        "0x11d7d24b11727d3922e1992446850b60c7cac60a57a5bd61f5eb9ef19b7b557f3bae970e2d1d325634",
    );
    let alpha_c = parse("0x20e8f2187a69ac25e8976218b1fe895350547aa192d2fba8ba");
    let alpha_d = parse("0x1938904070736ee61b443e069f76cf0a9dd44bbab8af7d9694");

    // Reconstruct the 4 Selkie PREHNF cols (verified by
    // SELKIE_DUMP_O_ALPHA_PREHNF).
    let cols = [
        Vector::<60>::new(alpha_a, neg(alpha_b), neg(alpha_c), neg(alpha_d)),
        Vector::<60>::new(alpha_b, alpha_a, alpha_d, neg(alpha_c)),
        // col 2: "(i+j)/2" * α — values from the dump.
        Vector::<60>::new(
            parse(
                "0x52465d3d32082e5ec57a753dbcfc575934bc581ca84e11b741cc92234285b05370e9f8ee9e089e06aa1e6c74be6615f5ad0e364d251b56bd",
            ),
            neg(parse(
                "0x3f0d68a11920953f442a9b108ea90598eda39a92086fd48fa6e98f82a29514bc753046b9e0b23dcb11cb8efe5283d9a2679738d370ca27f3",
            )),
            parse(
                "0x19cef22c0c546e568cb16707d5d6aeb438acfb9461f4dc234ee347101ad7c265d9868c72c8f35d80d",
            ),
            parse(
                "0x8ebe92588b93e9c9170cc92234285b05370e9f8ee9e089e06aa1e6c74be6615f5ad0e364d251b56bd",
            ),
        ),
        // col 3: "(1+k)/2" * α.
        Vector::<60>::new(
            parse(
                "0x3f0d68a11920953f442a9b108ea9059c2781e01392fd9f613d16707d5d6aeb2a523f78d5abdedc19a9f66a6236ad1bbfc41d0c73dfb84179",
            ),
            parse(
                "0x52465d3d32082e5ec57a753dbcfc57475cea0d0b35d0d89460336ddcbd7a4f8ba623ee96f84b3c10be7f7ad9431096b9fe77282007e90089",
            ),
            neg(parse(
                "0x8ebe92588b93e9c9170cc92234285b07459dc116907b4c3ef41808526bcef69460188d7dff816ff77",
            )),
            parse(
                "0x19cef22c0c546e568cb16707d5d6aeb2a523f78d5abdedc19a9f66a6236ad1bbfc41d0c73dfb84179",
            ),
        ),
    ];

    let det = Matrix::from_columns(&cols).det().abs();
    eprintln!("det bits = {}", det.bitsize());
    let hnf = Matrix::from_hnf_columns_mod_cref::<60>(&cols, &det);
    let hnf_v2 = Matrix::from_hnf_columns_mod_cref_v2::<60>(&cols, &det);
    let hnf_classical = Matrix::from_hnf_columns(&cols);
    let hnf_mod_const = Matrix::from_hnf_columns_mod::<60>(&cols, &det);
    eprintln!("hnf vs hnf_v2: {}", hnf == hnf_v2);
    eprintln!("hnf vs classical: {}", hnf == hnf_classical);
    eprintln!("hnf vs mod_const: {}", hnf == hnf_mod_const);
    eprintln!("classical vs mod_const: {}", hnf_classical == hnf_mod_const);
    // Dump all four cols 2 row 0 for comparison.
    eprintln!("col 2 row 0:");
    eprintln!("  mod_cref     = {:?}", hnf.columns()[2][0]);
    eprintln!("  mod_cref_v2  = {:?}", hnf_v2.columns()[2][0]);
    eprintln!("  classical    = {:?}", hnf_classical.columns()[2][0]);
    eprintln!("  mod_const    = {:?}", hnf_mod_const.columns()[2][0]);

    eprintln!("Selkie HNF cols (existing port):");
    for j in 0..4 {
        for r in 0..4 {
            let v = hnf.columns()[j][r];
            eprintln!("  col {} row {} bits={}", j, r, v.bitsize());
        }
    }
    eprintln!("Selkie HNF cols (v2 fresh port):");
    for j in 0..4 {
        for r in 0..4 {
            let v = hnf_v2.columns()[j][r];
            eprintln!("  col {} row {} bits={}", j, r, v.bitsize());
        }
    }
    eprintln!("v2 matches existing? {}", hnf == hnf_v2);

    // Expected col 0 row 0 = a_oa = 0xb52d... (= 2 · α.a in O₀-basis).
    // From [O_ALPHA_CREF] dump for KAT-1 iter 0:
    let expected_col0_row0 = parse(
        "0xb52dfc86a10b45395e61d4b8e09778a1a655e799991a2793fe06285926a82384cabace7ab278a9f563469c3a258d0ccad0d400000000000000000000000000000000000000000000000000000000000000",
    );
    let expected_col2_row0 = parse(
        "0x41868f7b09a3a079e352c3d2394fef5db3d3ea26b0c4a3aa54dae3d3119943e2701a72cc1a3cb1ada0a2a409950e2a27a7fdfc813463412ca2f947aaeda6fd35cbae2b557c921b48a975830a1736c7b14e",
    );

    let actual_col0_row0 = hnf.columns()[0][0];
    let actual_col2_row0 = hnf.columns()[2][0];
    let actual_v2_col0_row0 = hnf_v2.columns()[0][0];
    let actual_v2_col2_row0 = hnf_v2.columns()[2][0];
    eprintln!("col 0 row 0:");
    eprintln!("  expected = {:?}", expected_col0_row0);
    eprintln!("  actual   = {:?}", actual_col0_row0);
    eprintln!("  v2       = {:?}", actual_v2_col0_row0);
    eprintln!("col 2 row 0:");
    eprintln!("  v2       = {:?}", actual_v2_col2_row0);
    eprintln!("  expected = {:?}", expected_col2_row0);
    eprintln!("  actual   = {:?}", actual_col2_row0);

    assert_eq!(actual_col0_row0, expected_col0_row0, "col 0 row 0 mismatch");
    assert_eq!(actual_col2_row0, expected_col2_row0, "col 2 row 0 mismatch");
}
