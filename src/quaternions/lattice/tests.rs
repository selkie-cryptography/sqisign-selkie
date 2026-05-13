use super::*;

type I = BigInt<4>;
type L = Lattice<4>;
type H = HnfLattice<4>;
type V = Vector<4>;

fn i(v: i64) -> I {
    I::from(v)
}

#[test]
fn lattice_construction() {
    let lat = L::from_matrix(Matrix::IDENTITY);
    assert_eq!(*lat.denom(), I::ONE);
}

#[test]
fn lattice_basis_elem() {
    let lat = L::from_matrix(Matrix::IDENTITY);
    let e0 = lat.basis_elem(0);
    assert_eq!(e0, Element::<4>::from_i64(1, 0, 0, 0));
    let e1 = lat.basis_elem(1);
    assert_eq!(e1, Element::<4>::from_i64(0, 1, 0, 0));
}

#[test]
fn lattice_with_denominator() {
    // O_0 basis: (1, i, (i+j)/2, (1+ij)/2)
    // Columns with denom=2: [2,0,0,0], [0,2,0,0], [0,1,1,0], [1,0,0,1]
    let basis = Matrix::from_rows(
        V::new(i(2), i(0), i(0), i(1)),
        V::new(i(0), i(2), i(1), i(0)),
        V::new(i(0), i(0), i(1), i(0)),
        V::new(i(0), i(0), i(0), i(1)),
    );
    let lat = L::new(basis, i(2));
    let e2 = lat.basis_elem(2);
    assert_eq!(BigInt::<4>::from(e2.denom), i(2));
    assert_eq!(*e2.b.as_bigint(), i(1));
    assert_eq!(*e2.c.as_bigint(), i(1));
}

#[test]
fn lattice_hnf() {
    let basis = Matrix::from_rows(
        V::new(i(2), i(5), i(0), i(0)),
        V::new(i(0), i(3), i(0), i(0)),
        V::new(i(0), i(0), i(1), i(0)),
        V::new(i(0), i(0), i(0), i(1)),
    );
    let h = H::from(L::from_matrix(basis));
    assert_eq!(h.basis()[0][0], i(2));
    assert_eq!(h.basis()[0][1], i(1)); // 5 mod 2 = 1
}

#[test]
fn hnf_lattice_equality_same_denom() {
    let a = H::from(L::from_matrix(Matrix::IDENTITY));
    let b = H::from(L::from_matrix(Matrix::IDENTITY));
    assert_eq!(a, b);
}

#[test]
fn hnf_lattice_equality_different_denom() {
    let a = H::from(L::from_matrix(Matrix::IDENTITY));
    let b: H = L::new(
        Matrix::from_rows(
            V::new(i(2), i(0), i(0), i(0)),
            V::new(i(0), i(2), i(0), i(0)),
            V::new(i(0), i(0), i(2), i(0)),
            V::new(i(0), i(0), i(0), i(2)),
        ),
        i(2),
    )
    .into();
    assert_eq!(a, b);
}

#[test]
fn lattice_sum() {
    // Z^4 + 2*Z^4 = Z^4.
    let a = L::from_matrix(Matrix::IDENTITY);
    let b = L::from_matrix(Matrix::from_rows(
        V::new(i(2), i(0), i(0), i(0)),
        V::new(i(0), i(2), i(0), i(0)),
        V::new(i(0), i(0), i(2), i(0)),
        V::new(i(0), i(0), i(0), i(2)),
    ));
    let s = &a + &b;
    assert_eq!(s, H::from(L::from_matrix(Matrix::IDENTITY)));
}

#[test]
fn contains_basis_element() {
    let h = H::from(L::from_matrix(Matrix::IDENTITY));
    let elem = Element::<4>::from_i64(1, 0, 0, 0);
    let coords = h.contains(&elem);
    assert!(coords.is_some());
    let c = coords.unwrap();
    assert_eq!(c[0], i(1));
    assert_eq!(c[1], i(0));
}

#[test]
fn contains_linear_combination() {
    let h = H::from(L::from_matrix(Matrix::IDENTITY));
    let elem = Element::<4>::from_i64(3, 7, -2, 5);
    let coords = h.contains(&elem).expect("should be contained");
    assert_eq!(coords[0], i(3));
    assert_eq!(coords[1], i(7));
    assert_eq!(coords[2], i(-2));
    assert_eq!(coords[3], i(5));
}

#[test]
fn does_not_contain() {
    // 2*Z^4 does not contain (1, 0, 0, 0).
    let h: H = L::from_matrix(Matrix::from_rows(
        V::new(i(2), i(0), i(0), i(0)),
        V::new(i(0), i(2), i(0), i(0)),
        V::new(i(0), i(0), i(2), i(0)),
        V::new(i(0), i(0), i(0), i(2)),
    ))
    .into();
    let elem = Element::<4>::from_i64(1, 0, 0, 0);
    assert!(h.contains(&elem).is_none());
}

#[test]
fn contains_with_lattice_denominator() {
    // Lattice 2*Z^4 / 2 = Z^4 should contain (1, 0, 0, 0) / 1.
    let h: H = L::new(
        Matrix::from_rows(
            V::new(i(2), i(0), i(0), i(0)),
            V::new(i(0), i(2), i(0), i(0)),
            V::new(i(0), i(0), i(2), i(0)),
            V::new(i(0), i(0), i(0), i(2)),
        ),
        i(2),
    )
    .into();
    let elem = Element::<4>::from_i64(1, 0, 0, 0);
    assert!(h.contains(&elem).is_some());
}

#[test]
fn left_ideal_construction() {
    let order = Order::from_lattice_unchecked(L::from_matrix(Matrix::IDENTITY));
    let ideal_lat = H::from(L::from_matrix(Matrix::IDENTITY));
    let ideal = LeftIdeal::from_parts(ideal_lat, i(1), order);
    assert_eq!(*ideal.norm(), i(1));
}

#[test]
fn extremal_order_construction() {
    let order = L::from_matrix(Matrix::IDENTITY);
    let z = Element::<4>::from_i64(0, 1, 0, 0);
    let t = Element::<4>::from_i64(0, 0, 1, 0);
    let ext = ExtremalOrder::new(order, z, t, 1);
    assert_eq!(ext.q(), 1);
}

#[test]
fn from_hnf_roundtrip() {
    let lat = L::from_matrix(Matrix::IDENTITY);
    let h: H = lat.into();
    let back: L = h.into();
    assert_eq!(*back.basis(), Matrix::IDENTITY);
}

#[test]
fn matrix_det_identity() {
    assert_eq!(Matrix::<4>::IDENTITY.det(), i(1));
}

#[test]
fn matrix_det_diagonal() {
    let m = Matrix::from_rows(
        V::new(i(2), i(0), i(0), i(0)),
        V::new(i(0), i(3), i(0), i(0)),
        V::new(i(0), i(0), i(5), i(0)),
        V::new(i(0), i(0), i(0), i(7)),
    );
    assert_eq!(m.det(), i(210)); // 2*3*5*7
}

#[test]
fn matrix_adjugate_identity() {
    let adj = Matrix::<4>::IDENTITY.adjugate();
    assert_eq!(adj, Matrix::IDENTITY);
}

#[test]
fn matrix_adjugate_times_original_is_det_times_identity() {
    let m = Matrix::from_rows(
        V::new(i(2), i(1), i(0), i(0)),
        V::new(i(0), i(3), i(0), i(0)),
        V::new(i(0), i(0), i(5), i(1)),
        V::new(i(0), i(0), i(0), i(7)),
    );
    let adj = m.adjugate();
    let product = m.mat_mul(&adj);
    let det = m.det();
    for row in 0..4 {
        for col in 0..4 {
            let expected = if row == col { det } else { i(0) };
            assert_eq!(
                product[row][col], expected,
                "M*adj(M) != det*I at [{row}][{col}]"
            );
        }
    }
}

#[test]
fn lattice_intersection() {
    // Z^4 ∩ 2Z^4 = 2Z^4.
    let a = L::from_matrix(Matrix::IDENTITY);
    let b = L::from_matrix(Matrix::from_rows(
        V::new(i(2), i(0), i(0), i(0)),
        V::new(i(0), i(2), i(0), i(0)),
        V::new(i(0), i(0), i(2), i(0)),
        V::new(i(0), i(0), i(0), i(2)),
    ));
    let inter = a.intersection(&b);
    let expected: H = b.into();
    assert_eq!(inter, expected);
}

#[test]
fn ideal_creation() {
    // Create O₀⟨1, 1⟩ which should equal O₀ itself.
    let order = Order::from_lattice_unchecked(L::new(
        Matrix::from_rows(
            V::new(i(2), i(0), i(0), i(1)),
            V::new(i(0), i(2), i(1), i(0)),
            V::new(i(0), i(0), i(1), i(0)),
            V::new(i(0), i(0), i(0), i(1)),
        ),
        i(2),
    ));
    let one = Element::<4>::from_i64(1, 0, 0, 0);
    let _p = i(3);
    let ideal = LeftIdeal::new(&one, &i(1), &order);
    assert_eq!(*ideal.norm(), i(1));
}

#[test]
fn ideal_generator() {
    // Create an ideal I = O₀⟨i, 2⟩ with p = 3.
    // The generator should be an element γ with gcd(nrd(γ)/2, 2) = 1.
    let order = Order::from_lattice_unchecked(L::new(
        Matrix::from_rows(
            V::new(i(2), i(0), i(0), i(1)),
            V::new(i(0), i(2), i(1), i(0)),
            V::new(i(0), i(0), i(1), i(0)),
            V::new(i(0), i(0), i(0), i(1)),
        ),
        i(2),
    ));
    let alpha = Element::<4>::from_i64(0, 1, 0, 0); // i
    let _p = i(3);
    let ideal = LeftIdeal::new(&alpha, &i(2), &order);

    let gamma = ideal.generator().expect("generator should be found");
    // Verify: γ is nonzero.
    assert!(!gamma.is_zero());

    // Verify: nrd(γ) / N_I is coprime to N_I.
    let (nrd_num, nrd_den) = gamma.norm();
    let n_i: BigInt<8> = (*ideal.norm()).into();
    let (q, rem) = nrd_num.div_rem(&nrd_den.ct_mul(&n_i));
    assert!(bool::from(rem.is_zero()), "nrd(γ) not divisible by N_I");
    assert_eq!(q.gcd(&n_i), BigInt::<8>::ONE, "gcd(nrd(γ)/N_I, N_I) != 1");
}

// L2 reduction tests use BigInt<8> for arithmetic headroom.
type I8 = BigInt<8>;
type V8 = Vector<8>;

fn i8(v: i64) -> I8 {
    I8::from(v)
}

/// Compute the Gram matrix for the quaternion bilinear form
/// ⟨α, β⟩ = tr(αβ̄) with Gram diag(2, 2, 2p, 2p) on {1,i,j,k}.
fn quat_gram(basis: &[V8; 4], p: &I8) -> Matrix<8> {
    let two = i8(2);
    let two_p = two.ct_mul(p);
    let diag = [two, two, two_p, two_p];

    let mut g = Matrix::<8>::ZERO;
    for row in 0..4 {
        for col in 0..4 {
            let mut acc = I8::ZERO;
            for k in 0..4 {
                acc = acc.ct_add(&basis[row][k].ct_mul(&diag[k]).ct_mul(&basis[col][k]));
            }
            g[row][col] = acc;
        }
    }
    g
}

#[test]
fn l2_identity_basis() {
    let p = i8(3);
    let basis = [
        V8::new(i8(1), i8(0), i8(0), i8(0)),
        V8::new(i8(0), i8(1), i8(0), i8(0)),
        V8::new(i8(0), i8(0), i8(1), i8(0)),
        V8::new(i8(0), i8(0), i8(0), i8(1)),
    ];
    let gram = quat_gram(&basis, &p);

    let nrd = NrdBasis::from_cols_and_gram(basis, gram).l2_reduce();

    // Diagonal should be non-decreasing (short vectors first).
    for idx in 1..4 {
        assert!(
            nrd.gram()[idx][idx] >= nrd.gram()[idx - 1][idx - 1],
            "Gram diagonal not non-decreasing at position {idx}"
        );
    }
}

#[test]
fn l2_reduces_bad_basis() {
    let p = i8(3);
    let basis = [
        V8::new(i8(1), i8(0), i8(0), i8(0)),
        V8::new(i8(100), i8(1), i8(0), i8(0)),
        V8::new(i8(0), i8(0), i8(1), i8(0)),
        V8::new(i8(0), i8(0), i8(0), i8(1)),
    ];
    let gram = quat_gram(&basis, &p);
    let original_g00 = gram[0][0];

    let nrd = NrdBasis::from_cols_and_gram(basis, gram).l2_reduce();

    assert!(
        nrd.gram()[0][0] <= original_g00,
        "first vector got longer after reduction"
    );
}

#[test]
fn l2_gram_stays_symmetric() {
    let p = i8(3);
    let basis = [
        V8::new(i8(3), i8(1), i8(0), i8(0)),
        V8::new(i8(1), i8(2), i8(0), i8(0)),
        V8::new(i8(0), i8(0), i8(1), i8(1)),
        V8::new(i8(0), i8(0), i8(2), i8(1)),
    ];
    let gram = quat_gram(&basis, &p);

    let nrd = NrdBasis::from_cols_and_gram(basis, gram).l2_reduce();

    for row in 0..4 {
        for col in 0..4 {
            assert_eq!(
                nrd.gram()[row][col],
                nrd.gram()[col][row],
                "Gram not symmetric at [{row}][{col}]"
            );
        }
    }
}

#[test]
fn nrd_basis_identity_gram() {
    // Standard basis {1, i, j, k} has nrd gram diag(1, 1, p, p).
    let cols: [Vector<8>; 4] = [
        V8::new(i8(1), i8(0), i8(0), i8(0)),
        V8::new(i8(0), i8(1), i8(0), i8(0)),
        V8::new(i8(0), i8(0), i8(1), i8(0)),
        V8::new(i8(0), i8(0), i8(0), i8(1)),
    ];
    let nrd = NrdBasis::new(cols);
    let gram = nrd.gram();
    let p: I8 = crate::quaternions::precomputed::P_WIDE;

    assert_eq!(gram[0][0], I8::ONE, "nrd(1) = 1");
    assert_eq!(gram[1][1], I8::ONE, "nrd(i) = 1");
    assert_eq!(gram[2][2], p, "nrd(j) = p");
    assert_eq!(gram[3][3], p, "nrd(k) = p");

    for row in 0..4 {
        for col in 0..4 {
            if row != col {
                assert!(
                    bool::from(gram[row][col].is_zero()),
                    "G[{row}][{col}] should be zero for orthogonal basis"
                );
            }
        }
    }
}

#[test]
fn nrd_basis_gram_symmetric() {
    // Non-trivial basis: columns are not orthogonal.
    let cols: [Vector<8>; 4] = [
        V8::new(i8(3), i8(1), i8(0), i8(0)),
        V8::new(i8(1), i8(2), i8(0), i8(0)),
        V8::new(i8(0), i8(0), i8(1), i8(1)),
        V8::new(i8(0), i8(0), i8(2), i8(1)),
    ];
    let nrd = NrdBasis::new(cols);
    let gram = nrd.gram();

    for row in 0..4 {
        for col in 0..4 {
            assert_eq!(
                gram[row][col], gram[col][row],
                "Gram not symmetric at [{row}][{col}]"
            );
        }
    }
}

#[test]
fn nrd_basis_eval_quadratic_form() {
    // For identity basis with gram diag(1, 1, p, p),
    // c^T G c = c0² + c1² + p(c2² + c3²).
    let cols: [Vector<8>; 4] = [
        V8::new(i8(1), i8(0), i8(0), i8(0)),
        V8::new(i8(0), i8(1), i8(0), i8(0)),
        V8::new(i8(0), i8(0), i8(1), i8(0)),
        V8::new(i8(0), i8(0), i8(0), i8(1)),
    ];
    let nrd = NrdBasis::new(cols);
    let p: I8 = crate::quaternions::precomputed::P_WIDE;

    let c = [i8(2), i8(3), i8(0), i8(0)];
    let qf = nrd.eval_quadratic_form(&c);
    // 2² + 3² = 13
    assert_eq!(qf, i8(13));

    let c2 = [i8(0), i8(0), i8(1), i8(1)];
    let qf2 = nrd.eval_quadratic_form(&c2);
    // p(1² + 1²) = 2p
    assert_eq!(qf2, p.ct_mul(&i8(2)));
}

#[test]
fn canonicalize_already_canonical() {
    // A canonical HNF should be unchanged by canonicalize.
    let basis = Matrix::from_columns(&[
        V::new(i(6), i(0), i(0), i(0)),
        V::new(i(0), i(3), i(0), i(0)),
        V::new(i(2), i(1), i(2), i(0)),
        V::new(i(1), i(0), i(1), i(1)),
    ]);
    let lat = Lattice::new(basis, I::ONE);
    let hnf = HnfLattice::from(lat);
    let canonical = hnf.canonicalize();

    assert_eq!(*hnf.basis(), *canonical.basis());
}

#[test]
fn canonicalize_reduces_off_diagonals() {
    // Construct an HNF where off-diagonal entries exceed the
    // diagonal pivot. canonicalize should reduce them modulo the
    // pivot while preserving the lattice.
    //
    // Start with a canonical HNF, then add multiples of pivot
    // columns to inflate off-diagonal entries.
    let mut cols = [
        V::new(i(6), i(0), i(0), i(0)),
        V::new(i(0), i(3), i(0), i(0)),
        V::new(i(2), i(1), i(2), i(0)),
        V::new(i(1), i(0), i(1), i(1)),
    ];

    // Inflate: col[2] += 5 * col[1] (makes col[2][1] = 1 + 5*3 = 16,
    // which should reduce to 16 mod 3 = 1).
    // Also col[3] += 3 * col[2] (inflates col[3] entries).
    let col1_snap = cols[1];
    for row in 0..4 {
        cols[2][row] = cols[2][row].ct_add(&i(5).ct_mul(&col1_snap[row]));
    }
    let col2_snap = cols[2];
    for row in 0..4 {
        cols[3][row] = cols[3][row].ct_add(&i(3).ct_mul(&col2_snap[row]));
    }

    let inflated_basis = Matrix::from_columns(&cols);
    // Construct HnfLattice directly (the inflated matrix is still
    // a valid HNF — upper-triangular with positive pivots — just
    // not canonical).
    let inflated = HnfLattice::from(Lattice::new(inflated_basis, I::ONE));
    let canonical = inflated.canonicalize();

    // Off-diagonal entries should now be in [0, pivot).
    let h = canonical.basis();
    for col in 1..4 {
        let pivot = h[col][col];
        for row in 0..col {
            let entry = h[col][row];
            assert!(
                entry >= I::ZERO && entry < pivot,
                "h[{col}][{row}] = {entry:?} not in [0, {pivot:?})"
            );
        }
    }

    // The lattice should be unchanged: same HNF after
    // re-reducing from scratch.
    let original = HnfLattice::from(Lattice::new(
        Matrix::from_columns(&[
            V::new(i(6), i(0), i(0), i(0)),
            V::new(i(0), i(3), i(0), i(0)),
            V::new(i(2), i(1), i(2), i(0)),
            V::new(i(1), i(0), i(1), i(1)),
        ]),
        I::ONE,
    ));
    assert_eq!(
        canonical, original,
        "canonicalized lattice should equal original"
    );
}

#[test]
fn canonicalize_preserves_denom() {
    let basis = Matrix::from_columns(&[
        V::new(i(4), i(0), i(0), i(0)),
        V::new(i(0), i(2), i(0), i(0)),
        V::new(i(1), i(1), i(1), i(0)),
        V::new(i(0), i(0), i(0), i(1)),
    ]);
    let denom = i(3);
    let lat = Lattice::new(basis, denom);
    let hnf = HnfLattice::from(lat);
    let canonical = hnf.canonicalize();

    assert_eq!(*canonical.denom(), denom);
}

/// Intersection of two diagonal lattices with coprime scalars.
///
/// L1 = 3·Z^4 with denom=1; L2 = 5·Z^4 with denom=1. Intersection
/// must equal 15·Z^4 (covolume 15^4 = 50625).
#[test]
fn intersection_via_kernel_coprime_diagonal() {
    let basis_3 = Matrix::from_columns(&[
        V::new(i(3), i(0), i(0), i(0)),
        V::new(i(0), i(3), i(0), i(0)),
        V::new(i(0), i(0), i(3), i(0)),
        V::new(i(0), i(0), i(0), i(3)),
    ]);
    let basis_5 = Matrix::from_columns(&[
        V::new(i(5), i(0), i(0), i(0)),
        V::new(i(0), i(5), i(0), i(0)),
        V::new(i(0), i(0), i(5), i(0)),
        V::new(i(0), i(0), i(0), i(5)),
    ]);
    let l1 = Lattice::<4>::new(basis_3, i(1));
    let l2 = Lattice::<4>::new(basis_5, i(1));

    let inter = l1
        .intersection_via_kernel::<8>(&l2)
        .expect("intersection must succeed");
    let inter_lat: Lattice<4> = Lattice::from(inter);

    // Determinant of (15·I) = 15^4 = 50625; with any denom d the
    // covolume is 50625/d^4. Compare cross-multiplied integers.
    let det = inter_lat.basis().det().abs();
    let denom = *inter_lat.denom();
    let denom4 = {
        let d2 = denom.ct_mul(&denom);
        d2.ct_mul(&d2)
    };
    let expected = i(50625);
    let expected_scaled = expected.ct_mul(&denom4);

    assert_eq!(
        det, expected_scaled,
        "L1 ∩ L2: expected covolume 15^4 / denom^4 (det={det:?}, denom={denom:?})"
    );
}

/// Intersection of O0 with itself is O0.
///
/// O0 has basis with denom 2 (the standard maximal order). Self-
/// intersection must produce a lattice with the same covolume as O0.
/// Stresses the denom-tracking path that pure-integer diagonal
/// tests don't exercise.
#[test]
fn intersection_via_kernel_o0_self() {
    use crate::quaternions::precomputed::EXTREMAL_ORDERS;
    let o0 = *EXTREMAL_ORDERS[0].order();
    let o0_lat: Lattice<4> = Lattice::from(o0);

    let inter = o0_lat
        .intersection_via_kernel::<10>(&o0_lat)
        .expect("self-intersection must succeed");
    let inter_lat: Lattice<4> = Lattice::from(inter);

    // Cross-multiplied covolume equality: |det_o0| * denom_inter^4
    // = |det_inter| * denom_o0^4.
    let det_o0 = o0_lat.basis().det().abs();
    let denom_o0 = *o0_lat.denom();
    let det_inter = inter_lat.basis().det().abs();
    let denom_inter = *inter_lat.denom();
    let dn_o0_4 = {
        let d2 = denom_o0.ct_mul(&denom_o0);
        d2.ct_mul(&d2)
    };
    let dn_in_4 = {
        let d2 = denom_inter.ct_mul(&denom_inter);
        d2.ct_mul(&d2)
    };
    let lhs = det_o0.ct_mul(&dn_in_4);
    let rhs = det_inter.ct_mul(&dn_o0_4);

    assert_eq!(
        lhs, rhs,
        "O0 self-intersection covolume mismatch: \
         O0 det={det_o0:?} denom={denom_o0:?}, inter det={det_inter:?} denom={denom_inter:?}"
    );
}

/// Intersection of `2·O0 ∩ 3·O0 = 6·O0`.
///
/// Stress test: inputs share the O0 basis with denom 2 but
/// scaled differently. Covolume of `n·O0` equals `n^4 · cov(O0)`,
/// so `cov(6·O0) = 6^4 · cov(O0) = 1296 · cov(O0)`.
#[test]
fn intersection_via_kernel_scaled_o0() {
    use crate::quaternions::precomputed::EXTREMAL_ORDERS;
    let o0 = *EXTREMAL_ORDERS[0].order();
    let o0_lat: Lattice<4> = Lattice::from(o0);

    // n·O0: multiply each basis entry by n, keep denom. Net effect:
    // each basis vector is scaled by n in algebra coords.
    let scale_basis = |s: I| -> Matrix<4> {
        let mut m = Matrix::<4>::ZERO;
        for r in 0..4 {
            for c in 0..4 {
                m[r][c] = o0_lat.basis()[r][c].ct_mul(&s);
            }
        }
        m
    };
    let l2 = Lattice::<4>::new(scale_basis(i(2)), *o0_lat.denom());
    let l3 = Lattice::<4>::new(scale_basis(i(3)), *o0_lat.denom());

    let inter = l2
        .intersection_via_kernel::<10>(&l3)
        .expect("intersection must succeed");
    let inter_lat: Lattice<4> = Lattice::from(inter);

    // Expected: 6·O0. Covolume = 6^4 · cov(O0) = 1296 · cov(O0).
    let det_o0 = o0_lat.basis().det().abs();
    let denom_o0 = *o0_lat.denom();
    let det_inter = inter_lat.basis().det().abs();
    let denom_inter = *inter_lat.denom();
    let dn_o0_4 = {
        let d2 = denom_o0.ct_mul(&denom_o0);
        d2.ct_mul(&d2)
    };
    let dn_in_4 = {
        let d2 = denom_inter.ct_mul(&denom_inter);
        d2.ct_mul(&d2)
    };
    let factor_1296 = i(1296);

    // det_o0 / dn_o0_4 * 1296 = det_inter / dn_in_4
    // → det_o0 * dn_in_4 * 1296 = det_inter * dn_o0_4
    let lhs = det_o0.ct_mul(&dn_in_4).ct_mul(&factor_1296);
    let rhs = det_inter.ct_mul(&dn_o0_4);

    assert_eq!(
        lhs, rhs,
        "2·O0 ∩ 3·O0 expected 6·O0 (covolume 6^4 · cov(O0)): \
         det_inter={det_inter:?} denom_inter={denom_inter:?}"
    );
}

/// `intersection_via_kernel` output must satisfy
/// `refresh_norm`'s perfect-square covolume invariant.
///
/// Sanity check tying together intersection + refresh_norm. For the
/// intersection of O0 with `n·O0` (= `n·O0`), the resulting LeftIdeal
/// must have a well-defined integer norm.
#[test]
fn intersection_then_refresh_norm_o0_with_scaled() {
    use crate::quaternions::precomputed::EXTREMAL_ORDERS;
    let o0_ext = EXTREMAL_ORDERS[0];
    let o0 = *o0_ext.order();
    let o0_lat: Lattice<4> = Lattice::from(o0);

    let scale_basis = |s: I| -> Matrix<4> {
        let mut m = Matrix::<4>::ZERO;
        for r in 0..4 {
            for c in 0..4 {
                m[r][c] = o0_lat.basis()[r][c].ct_mul(&s);
            }
        }
        m
    };
    let l_scaled = Lattice::<4>::new(scale_basis(i(7)), *o0_lat.denom());

    let inter = o0_lat
        .intersection_via_kernel::<10>(&l_scaled)
        .expect("intersection must succeed");

    let mut ideal = LeftIdeal::<4>::from_parts(inter, I::ZERO, o0);
    let r = ideal.refresh_norm::<10>();

    assert!(
        r.is_some(),
        "refresh_norm on (O0 ∩ 7·O0) failed — covolume not a perfect square"
    );
    // 7·O0 ⊆ O0, so intersection = 7·O0. N(7·O0) = 7² = 49.
    assert_eq!(*ideal.norm(), i(49), "expected N(7·O0) = 49");
}

/// Intersection where one operand is `2^60 · O0`.
///
/// Mirrors the structure of the sign-time intersection
/// `I_chl ∩ I_sk` (entries at the scale of large powers of 2)
/// without exceeding the BigInt<4> 256-bit ceiling for the
/// resulting norm `(2^60)^2 = 2^120`.
#[test]
fn intersection_via_kernel_large_power_of_two() {
    use crate::quaternions::precomputed::EXTREMAL_ORDERS;
    let o0_ext = EXTREMAL_ORDERS[0];
    let o0 = *o0_ext.order();
    let o0_lat: Lattice<4> = Lattice::from(o0);

    let two_to_60 = I::ONE.shl(60);
    let mut basis_big = Matrix::<4>::ZERO;
    for r in 0..4 {
        for c in 0..4 {
            basis_big[r][c] = o0_lat.basis()[r][c].ct_mul(&two_to_60);
        }
    }
    let l_big = Lattice::<4>::new(basis_big, *o0_lat.denom());

    let inter = l_big
        .intersection_via_kernel::<10>(&o0_lat)
        .expect("intersection at scale 2^60 must succeed");

    let mut ideal = LeftIdeal::<4>::from_parts(inter, I::ZERO, o0);
    let r = ideal.refresh_norm::<10>();

    assert!(
        r.is_some(),
        "refresh_norm on (2^60·O0) ∩ O0 failed — covolume not a perfect square"
    );

    let expected = I::ONE.shl(120); // (2^60)^2 = 2^120
    assert_eq!(*ideal.norm(), expected, "expected N(2^60·O0) = 2^120");
}

/// `intersection_via_dual_sum_dual` parity with `intersection_via_kernel`
/// on simple inputs.
///
/// For diagonal lattices the two methods must produce the same
/// lattice (up to HNF basis equivalence; covolume is the
/// equivalence-class invariant we check).
#[test]
fn intersection_via_dual_sum_dual_diagonal_coprime() {
    let basis_3 = Matrix::from_columns(&[
        V::new(i(3), i(0), i(0), i(0)),
        V::new(i(0), i(3), i(0), i(0)),
        V::new(i(0), i(0), i(3), i(0)),
        V::new(i(0), i(0), i(0), i(3)),
    ]);
    let basis_5 = Matrix::from_columns(&[
        V::new(i(5), i(0), i(0), i(0)),
        V::new(i(0), i(5), i(0), i(0)),
        V::new(i(0), i(0), i(5), i(0)),
        V::new(i(0), i(0), i(0), i(5)),
    ]);
    let l1 = Lattice::<4>::new(basis_3, i(1));
    let l2 = Lattice::<4>::new(basis_5, i(1));

    let inter = l1
        .intersection_via_dual_sum_dual::<8>(&l2)
        .expect("dual-sum-dual must succeed");
    let inter_lat: Lattice<4> = Lattice::from(inter);

    let det = inter_lat.basis().det().abs();
    let denom = *inter_lat.denom();
    let denom4 = {
        let d2 = denom.ct_mul(&denom);
        d2.ct_mul(&d2)
    };
    let expected = i(50625); // 15^4
    let expected_scaled = expected.ct_mul(&denom4);

    assert_eq!(
        det, expected_scaled,
        "L1 ∩ L2 via dual-sum-dual: expected covolume 15^4 (det={det:?}, denom={denom:?})"
    );
}

/// Intersection of two diagonal lattices with non-coprime scalars.
///
/// L1 = 4·Z^4, L2 = 6·Z^4. lcm = 12, so intersection = 12·Z^4
/// (covolume 12^4 = 20736).
#[test]
fn intersection_via_kernel_lcm_diagonal() {
    let basis_4 = Matrix::from_columns(&[
        V::new(i(4), i(0), i(0), i(0)),
        V::new(i(0), i(4), i(0), i(0)),
        V::new(i(0), i(0), i(4), i(0)),
        V::new(i(0), i(0), i(0), i(4)),
    ]);
    let basis_6 = Matrix::from_columns(&[
        V::new(i(6), i(0), i(0), i(0)),
        V::new(i(0), i(6), i(0), i(0)),
        V::new(i(0), i(0), i(6), i(0)),
        V::new(i(0), i(0), i(0), i(6)),
    ]);
    let l1 = Lattice::<4>::new(basis_4, i(1));
    let l2 = Lattice::<4>::new(basis_6, i(1));

    let inter = l1
        .intersection_via_kernel::<8>(&l2)
        .expect("intersection must succeed");
    let inter_lat: Lattice<4> = Lattice::from(inter);

    let det = inter_lat.basis().det().abs();
    let denom = *inter_lat.denom();
    let denom4 = {
        let d2 = denom.ct_mul(&denom);
        d2.ct_mul(&d2)
    };
    let expected = i(20736); // 12^4
    let expected_scaled = expected.ct_mul(&denom4);

    assert_eq!(
        det, expected_scaled,
        "L1 ∩ L2: expected covolume 12^4 / denom^4 (det={det:?}, denom={denom:?})"
    );
}
