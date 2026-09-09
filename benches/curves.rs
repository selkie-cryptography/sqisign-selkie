#![allow(non_snake_case)]

use sqisign_selkie::{
    curves::{
        BasisHint, TorsionBasis, TorsionExponent,
        isogeny::Kernel,
        montgomery::{Curve, ProjectiveXOnlyPoint},
        scalar::Scalar,
    },
    params::{BASIS_E0_P_X, BASIS_E0_PMQ_X, BASIS_E0_Q_X, TORSION_EVEN_POWER},
};

fn main() {
    divan::main();
}

fn e0_basis() -> TorsionBasis {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
    let Q = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_Q_X, &curve);
    TorsionBasis::from((P, Q))
}

fn sample_scalar() -> Scalar {
    Scalar::from_limbs([
        0xDEAD_BEEF_CAFE_BABE,
        0x1234_5678_9ABC_DEF0,
        0xFEDC_BA98_7654_3210,
        0x0000_0000_0000_0001,
    ])
}

// --- Point operations ---

#[divan::bench]
fn xdbl(bencher: divan::Bencher) {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
    bencher.bench(|| divan::black_box(&P).double());
}

#[divan::bench]
fn xadd(bencher: divan::Bencher) {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
    let Q = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_Q_X, &curve);
    let PmQ = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_PMQ_X, &curve);
    bencher.bench(|| {
        divan::black_box(&P).differential_add(divan::black_box(&Q), divan::black_box(&PmQ))
    });
}

#[divan::bench]
fn projective_difference(bencher: divan::Bencher) {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
    let Q = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_Q_X, &curve);
    bencher.bench(|| divan::black_box(&P).projective_difference(divan::black_box(&Q)));
}

// --- Scalar multiplication ---

#[divan::bench]
fn scalar_mul(bencher: divan::Bencher) {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
    let n = sample_scalar();
    bencher.bench(|| divan::black_box(&P).scalar_mul(divan::black_box(&n)));
}

#[divan::bench]
fn ladder3pt(bencher: divan::Bencher) {
    let basis = e0_basis();
    let m = sample_scalar();
    bencher.bench(|| divan::black_box(&basis).scalar_mul_add(divan::black_box(&m)));
}

#[divan::bench]
fn ladder_biscalar(bencher: divan::Bencher) {
    let basis = e0_basis();
    let m = sample_scalar();
    let n = Scalar::from_limbs([0x42, 0, 0, 0]);
    let e = TorsionExponent::try_from(TORSION_EVEN_POWER).unwrap();
    bencher.bench(|| {
        divan::black_box(&basis).biscalar_mul(divan::black_box(&m), divan::black_box(&n), e)
    });
}

// --- Isogeny chains ---

#[divan::bench(args = [10, 50, 122, 248])]
fn isogeny_chain(bencher: divan::Bencher, e_val: u32) {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
    let Q = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_Q_X, &curve);
    let n = sample_scalar();
    let kernel_pt = P.scalar_mul(&n);

    let e = TorsionExponent::try_from(e_val).unwrap();
    let cofactor_doublings = TORSION_EVEN_POWER - e_val;
    let mut ker = kernel_pt;
    for _ in 0..cofactor_doublings {
        ker = ker.double();
    }
    let kernel = Kernel::new(ker);
    bencher.bench(|| divan::black_box(&kernel).isogeny(e, divan::black_box(&[Q])));
}

// --- Curve operations ---

#[divan::bench]
fn j_invariant(bencher: divan::Bencher) {
    let curve = Curve::E0;
    bencher.bench(|| divan::black_box(&curve).j_invariant());
}

#[divan::bench]
fn clear_cofactor(bencher: divan::Bencher) {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
    bencher.bench(|| divan::black_box(&P).clear_cofactor());
}

#[divan::bench]
fn isomorphism(bencher: divan::Bencher) {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
    let mut ker = P.scalar_mul(&sample_scalar());
    for _ in 0..(TORSION_EVEN_POWER - 2) {
        ker = ker.double();
    }
    let e = TorsionExponent::try_from(2u32).unwrap();
    let (target, _) = Kernel::new(ker).isogeny(e, &[]);
    bencher.bench(|| divan::black_box(&curve).isomorphism(divan::black_box(&target)));
}

// --- Torsion basis hint encode / decode ---
//
// `from_hint` is the verify-side entry: rebuild the basis on a curve
// from its hint byte. `to_hint` is the keygen-side inverse: derive the
// canonical basis and emit its hint. Measured on a non-E₀ curve so
// the search path runs rather than the precomputed E₀ shortcut.

fn non_e0_curve() -> Curve {
    let P = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &Curve::E0);
    let mut ker = P.scalar_mul(&sample_scalar());
    for _ in 0..(TORSION_EVEN_POWER - 8) {
        ker = ker.double();
    }
    let e = TorsionExponent::try_from(8u32).unwrap();
    let (target, _) = Kernel::new(ker).isogeny(e, &[]);
    target
}

#[divan::bench]
fn torsion_basis_from_hint(bencher: divan::Bencher) {
    let curve = non_e0_curve();
    let (_, hint) = TorsionBasis::to_hint(&curve).expect("honest curve has a hint");
    let hint = BasisHint::from_byte(hint.to_byte());
    bencher.bench(|| TorsionBasis::from_hint(divan::black_box(&curve), divan::black_box(hint)));
}

#[divan::bench]
fn torsion_basis_to_hint(bencher: divan::Bencher) {
    let curve = non_e0_curve();
    bencher.bench(|| TorsionBasis::to_hint(divan::black_box(&curve)));
}
