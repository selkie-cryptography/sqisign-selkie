#![allow(non_snake_case)]

mod common;

use sqisign_selkie::{
    curves::{
        BasisHint, ChangeOfBasisMatrix, TorsionBasis, TorsionExponent,
        isogeny::Kernel,
        montgomery::{Curve, ProjectiveXOnlyPoint},
        scalar::Scalar,
    },
    params::{BASIS_E0_P_X, BASIS_E0_PMQ_X, BASIS_E0_Q_X, TORSION_EVEN_POWER},
};

fn main() {
    divan::main();
}

/// Setup for the change-of-basis / pairing benches: pull the KAT[0]
/// public curve, derive its canonical 2^f torsion basis from the
/// hint, then double down to a reduced basis at 2^(f − 8) torsion so
/// `cross_pairings` and `from_bases` see a real (canonical, reduced)
/// pair rather than two co-equal bases at the same order.
fn kat0_basis_pair() -> (Curve, TorsionBasis, TorsionBasis, TorsionExponent) {
    let (vk, ..) = common::kat0_vk_sig_msg();
    let curve = *vk.curve();
    let hint = BasisHint::from_byte(vk.as_bytes()[64]);
    let canonical = TorsionBasis::from_hint(&curve, hint);

    let reduce_e = TORSION_EVEN_POWER - 8;
    let mut r = canonical.R;
    let mut s = canonical.S;
    let mut rs = canonical.RS;
    for _ in 0..(TORSION_EVEN_POWER - reduce_e) {
        r = r.double();
        s = s.double();
        rs = rs.double();
    }
    let reduced = TorsionBasis::from_propagated(r, rs, s);
    let e = TorsionExponent::try_from(reduce_e).unwrap();
    (curve, canonical, reduced, e)
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
// `from_hint` is the verify-side entry: rebuild the basis on E_pk
// from the verifying-key hint byte. `to_hint` is the keygen-side
// inverse: derive the canonical basis on a freshly-computed E_pk
// and emit its hint for serialization.

#[divan::bench]
fn torsion_basis_from_hint(bencher: divan::Bencher) {
    let (vk, ..) = common::kat0_vk_sig_msg();
    let curve = *vk.curve();
    let hint = BasisHint::from_byte(vk.as_bytes()[64]);
    bencher.bench(|| TorsionBasis::from_hint(divan::black_box(&curve), divan::black_box(hint)));
}

#[divan::bench]
fn torsion_basis_to_hint(bencher: divan::Bencher) {
    let (vk, ..) = common::kat0_vk_sig_msg();
    let curve = *vk.curve();
    bencher.bench(|| TorsionBasis::to_hint(divan::black_box(&curve)));
}

// --- Tate cross-pairings + change-of-basis matrix ---

#[divan::bench(sample_count = 30)]
fn cross_pairings(bencher: divan::Bencher) {
    let (_, canonical, reduced, e) = kat0_basis_pair();
    bencher.bench(|| {
        divan::black_box(&canonical).cross_pairings(divan::black_box(&reduced), e)
    });
}

#[divan::bench(sample_count = 30)]
fn change_of_basis_from_bases(bencher: divan::Bencher) {
    let (_, canonical, reduced, e) = kat0_basis_pair();
    bencher.bench(|| {
        ChangeOfBasisMatrix::from_bases(
            divan::black_box(&canonical),
            divan::black_box(&reduced),
            e,
        )
    });
}

#[divan::bench(sample_count = 30)]
fn change_of_basis_from_bases_invert(bencher: divan::Bencher) {
    let (_, canonical, reduced, e) = kat0_basis_pair();
    bencher.bench(|| {
        ChangeOfBasisMatrix::from_bases_invert(
            divan::black_box(&canonical),
            divan::black_box(&reduced),
            e,
        )
    });
}

#[divan::bench(sample_count = 30)]
fn change_of_basis_mul(bencher: divan::Bencher) {
    let (_, canonical, reduced, e) = kat0_basis_pair();
    let m = ChangeOfBasisMatrix::from_bases(&canonical, &reduced, e)
        .expect("KAT[0] basis pair admits a change-of-basis matrix");
    bencher.bench(|| divan::black_box(&m).mul(divan::black_box(&canonical)));
}
