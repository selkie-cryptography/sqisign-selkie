#![allow(non_snake_case)]

use sqisign_selkie::{
    curves::{
        TorsionBasis, TorsionExponent,
        montgomery::{Curve, ProjectiveXOnlyPoint},
        scalar::Scalar,
    },
    params::{BASIS_E0_PMQ_X, BASIS_E0_P_X, BASIS_E0_Q_X, TORSION_EVEN_POWER},
    surfaces::{EllipticProduct, Kernel},
};

fn main() {
    divan::main();
}

fn e0_product_kernel(e: u32) -> Option<(Kernel, TorsionExponent)> {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
    let Q = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_Q_X, &curve);
    let PmQ = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_PMQ_X, &curve);

    let doublings = TORSION_EVEN_POWER - e - 2;
    let mut P1 = P;
    let mut Q1 = Q;
    let mut PmQ1 = PmQ;
    for _ in 0..doublings {
        P1 = P1.double();
        Q1 = Q1.double();
        PmQ1 = PmQ1.double();
    }

    let s = Scalar::from_limbs([3, 0, 0, 0]);
    let mut P2 = P.scalar_mul(&s);
    let mut Q2 = Q.scalar_mul(&s);
    let mut PmQ2 = PmQ.scalar_mul(&s);
    for _ in 0..doublings {
        P2 = P2.double();
        Q2 = Q2.double();
        PmQ2 = PmQ2.double();
    }

    let product = EllipticProduct::new(curve, curve);
    let te = TorsionExponent::try_from(e).ok()?;
    let kernel = Kernel::from_montgomery(product, (P1, P2), (Q1, Q2), (PmQ1, PmQ2))?;
    Some((kernel, te))
}

#[divan::bench]
fn from_montgomery(bencher: divan::Bencher) {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
    let Q = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_Q_X, &curve);
    let PmQ = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_PMQ_X, &curve);
    let product = EllipticProduct::new(curve, curve);

    bencher.bench(|| {
        Kernel::from_montgomery(
            divan::black_box(product),
            divan::black_box((P, P)),
            divan::black_box((Q, Q)),
            divan::black_box((PmQ, PmQ)),
        )
    });
}

#[divan::bench]
fn lift(bencher: divan::Bencher) {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
    let Q = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_Q_X, &curve);
    let basis = TorsionBasis::from((P, Q));
    bencher.bench(|| divan::black_box(&basis).lift(divan::black_box(&curve)));
}

#[divan::bench(args = [10, 50, 122])]
fn isogeny_22_chain(bencher: divan::Bencher, e_val: u32) {
    if let Some((kernel, te)) = e0_product_kernel(e_val) {
        bencher.bench(|| divan::black_box(&kernel).isogeny(te, divan::black_box(&[])));
    }
}
