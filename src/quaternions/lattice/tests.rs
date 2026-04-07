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
    let mut basis = [
        V8::new(i8(1), i8(0), i8(0), i8(0)),
        V8::new(i8(0), i8(1), i8(0), i8(0)),
        V8::new(i8(0), i8(0), i8(1), i8(0)),
        V8::new(i8(0), i8(0), i8(0), i8(1)),
    ];
    let mut gram = quat_gram(&basis, &p);

    l2_reduce(&mut basis, &mut gram);

    // Diagonal should be non-decreasing (short vectors first).
    for idx in 1..4 {
        assert!(
            gram[idx][idx] >= gram[idx - 1][idx - 1],
            "Gram diagonal not non-decreasing at position {idx}"
        );
    }
}

#[test]
fn l2_reduces_bad_basis() {
    let p = i8(3);
    let mut basis = [
        V8::new(i8(1), i8(0), i8(0), i8(0)),
        V8::new(i8(100), i8(1), i8(0), i8(0)),
        V8::new(i8(0), i8(0), i8(1), i8(0)),
        V8::new(i8(0), i8(0), i8(0), i8(1)),
    ];
    let mut gram = quat_gram(&basis, &p);
    let original_g00 = gram[0][0];

    l2_reduce(&mut basis, &mut gram);

    assert!(
        gram[0][0] <= original_g00,
        "first vector got longer after reduction"
    );
}

#[test]
fn l2_gram_stays_symmetric() {
    let p = i8(3);
    let mut basis = [
        V8::new(i8(3), i8(1), i8(0), i8(0)),
        V8::new(i8(1), i8(2), i8(0), i8(0)),
        V8::new(i8(0), i8(0), i8(1), i8(1)),
        V8::new(i8(0), i8(0), i8(2), i8(1)),
    ];
    let mut gram = quat_gram(&basis, &p);

    l2_reduce(&mut basis, &mut gram);

    for row in 0..4 {
        for col in 0..4 {
            assert_eq!(
                gram[row][col], gram[col][row],
                "Gram not symmetric at [{row}][{col}]"
            );
        }
    }
}
