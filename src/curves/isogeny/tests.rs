use super::*;
use crate::{curves::montgomery::Coefficient, fields::fp::Fp};

#[test]
fn kernel_maps_to_identity() {
    let curve = Curve::E0;
    // (i, 0) is a 2-torsion point on E₀.
    let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::I, &curve);

    let (_, images) = Kernel::new(P).isogeny(TorsionExponent::try_from(1).unwrap(), &[P]);
    assert!(bool::from(images[0].is_identity()));
}

#[test]
fn non_kernel_survives() {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::I, &curve);
    let Q = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(5)), &curve);

    let (_, images) = Kernel::new(P).isogeny(TorsionExponent::try_from(1).unwrap(), &[Q]);
    assert!(!bool::from(images[0].is_identity()));
}

#[test]
fn codomain_has_valid_j_invariant() {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::I, &curve);

    let (codomain, _) = Kernel::new(P).isogeny(TorsionExponent::try_from(1).unwrap(), &[]);
    let _j = codomain.j_invariant();
}

/// Test that Isomorphism correctly maps points between two curves
/// with the same j-invariant but different projective representations.
///
/// Strategy: compute an isogeny from E₀ to get codomain E₁ with
/// unnormalized (A:C). Then construct E₁' from E₁'s affine A (C=1).
/// Both have the same j-invariant. The isomorphism E₁ → E₁' should
/// map a point Q₁ on E₁ to a point Q₁' on E₁' with the same
/// affine x-coordinate.
#[test]
fn isomorphism_preserves_affine_x() {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, &curve);

    // Compute a longer chain to get a codomain with non-trivial (A:C).
    let Q = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_Q_X, &curve);
    let (codomain, images) = Kernel::new(P).isogeny(TorsionExponent::try_from(10).unwrap(), &[Q]);

    // codomain has unnormalized doubling constants from the isogeny chain.
    let q1 = &images[0];

    // Reconstruct same curve from affine A (forces C=1).
    let codomain_affine = Curve::from(*codomain.coefficient());

    // Same j-invariant.
    assert_eq!(codomain.j_invariant(), codomain_affine.j_invariant());

    // Compute isomorphism.
    let iso = codomain
        .isomorphism(&codomain_affine)
        .expect("isomorphism should exist for same j-invariant");
    let q1_mapped = iso.eval(q1);

    // The mapped point should have the same affine x as the original.
    let x_orig = q1.to_affine_x();
    let x_mapped = q1_mapped.to_affine_x();
    assert_eq!(
        x_orig, x_mapped,
        "isomorphism between same curve (different projective rep) should preserve affine x"
    );
}

/// Test isomorphism maps on-curve points to on-curve points.
///
/// Uses a single 2-isogeny to produce a codomain with non-trivial
/// (A:C), then isomorphizes to the affine normalization and verifies
/// the mapped point satisfies y² = x³ + A'x² + x on the target.
#[test]
fn isomorphism_maps_on_curve() {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, &curve);
    let Q = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_Q_X, &curve);

    // Single 2-isogeny.
    let (cod1, imgs1) = Kernel::new(P).isogeny(TorsionExponent::try_from(1).unwrap(), &[Q]);
    let q_on_cod1 = &imgs1[0];

    // Verify source point is on cod1.
    assert!(
        cod1.recover_y(&q_on_cod1.to_affine_x()).is_some(),
        "source point should be on source curve"
    );

    // Reconstruct same curve from affine A (forces C=1).
    let cod2 = Curve::from(*cod1.coefficient());
    assert_eq!(cod1.j_invariant(), cod2.j_invariant());

    // Isomorphism cod1 → cod2.
    let iso = cod1.isomorphism(&cod2).expect("same j-invariant");
    let q_mapped = iso.eval(q_on_cod1);

    // Verify mapped point is on cod2.
    assert!(
        cod2.recover_y(&q_mapped.to_affine_x()).is_some(),
        "mapped point should be on target curve"
    );
}

#[test]
fn isogeny_e1_matches_direct_two_isogeny() {
    let curve = Curve::E0;
    let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::I, &curve);
    let Q = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(5)), &curve);

    // Via Kernel::isogeny.
    let (_, chain_imgs) = Kernel::new(P).isogeny(TorsionExponent::try_from(1).unwrap(), &[Q]);

    // Via direct TwoIsogeny.
    let phi = TwoIsogeny::from_kernel(&P);
    let direct_Q = phi.eval(&Q);

    assert_eq!(chain_imgs[0], direct_Q);
}

// --- Property tests ---

use proptest::prelude::*;

fn arb_curve() -> impl Strategy<Value = Curve> {
    (any::<[u8; 32]>(), any::<[u8; 32]>()).prop_map(|(a, b)| {
        Curve::from(Coefficient::from(Fp2::new(
            Fp::from_bytes(&a),
            Fp::from_bytes(&b),
        )))
    })
}

proptest! {
    /// c0 and c1 must both be nonzero for any curve.
    #[test]
    fn two_isogeny_singular_constants_nonzero(curve in arb_curve()) {
        let iso = TwoIsogenySingular::from_curve(&curve);
        prop_assert!(!bool::from(iso.c0.ct_eq(&Fp2::ZERO)));
        prop_assert!(!bool::from(iso.c1.ct_eq(&Fp2::ZERO)));
    }

    /// The codomain j-invariant must differ from the domain's.
    #[test]
    fn two_isogeny_singular_changes_j(curve in arb_curve()) {
        let iso = TwoIsogenySingular::from_curve(&curve);
        prop_assert_ne!(curve.j_invariant(), iso.codomain.j_invariant());
    }
}

/// Singular 2-isogeny: structural invariants that any mutation
/// in from_curve's arithmetic would break.
#[test]
fn two_isogeny_singular_structural() {
    use subtle::ConstantTimeEq;

    // Use a curve built from a known affine coefficient so the
    // doubling constants are normalized.
    let curve = Curve::from(Coefficient::from(Fp2::from_fp(Fp::from_small(6))));
    let iso = TwoIsogenySingular::from_curve(&curve);

    // Invariant 1: c0^2 = c1^2 - 4 (from the definition).
    let four = Fp2::from_fp(Fp::from_small(4));
    let c1_sq = iso.c1.square();
    let c0_sq = iso.c0.square();
    assert_eq!(c0_sq, c1_sq - four, "c0^2 must equal c1^2 - 4");

    // Invariant 2: c0 and c1 are nonzero.
    assert!(!bool::from(iso.c0.ct_eq(&Fp2::ZERO)), "c0 nonzero");
    assert!(!bool::from(iso.c1.ct_eq(&Fp2::ZERO)), "c1 nonzero");

    // Invariant 3: codomain j differs from domain j.
    assert_ne!(
        curve.j_invariant(),
        iso.codomain.j_invariant(),
        "isogeny must change j-invariant"
    );

    // Invariant 4: evaluating a point produces a valid output.
    let P = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, &curve);
    let Q = iso.eval(&P);
    assert!(!bool::from(Q.Z.ct_eq(&Fp2::ZERO)), "image Z nonzero");

    // Invariant 5: evaluating a DIFFERENT point gives a
    // DIFFERENT output (the isogeny is not constant).
    let R = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_Q_X, &curve);
    let S = iso.eval(&R);
    assert_ne!(Q, S, "different inputs produce different outputs");
}
