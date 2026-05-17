use super::{super::TorsionBasis, *};

#[test]
fn doubling_identity_is_identity() {
    let id = ProjectiveXOnlyPoint::identity(&Curve::E0);
    let dbl = id.double();
    assert!(bool::from(dbl.is_identity()));
}

#[test]
fn mul_by_one() {
    use crate::{curves::scalar::Scalar, fields::fp::Fp};

    let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &Curve::E0);
    assert_eq!(P.scalar_mul(&Scalar::from_u64(1)), P);
}

#[test]
fn mul_by_two_equals_doubling() {
    use crate::{curves::scalar::Scalar, fields::fp::Fp};

    let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &Curve::E0);
    assert_eq!(P.scalar_mul(&Scalar::from_u64(2)), P.double());
}

#[test]
fn mul_by_three_equals_double_plus_add() {
    use crate::{curves::scalar::Scalar, fields::fp::Fp};

    let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &Curve::E0);
    let triple = P.scalar_mul(&Scalar::from_u64(3));
    let dbl = P.double();
    let triple_add = dbl.differential_add(&P, &P);
    assert_eq!(triple, triple_add);
}

#[test]
fn mul_by_zero_is_identity() {
    use crate::{curves::scalar::Scalar, fields::fp::Fp};

    let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &Curve::E0);
    assert!(bool::from(P.scalar_mul(&Scalar::from_u64(0)).is_identity()));
}

#[test]
fn j_invariant_of_e0() {
    use crate::fields::fp::Fp;

    // j(E₀) = 256(0 − 3)³ / (0 − 4) = 1728
    let j = Curve::E0.j_invariant();
    assert_eq!(j, Fp2::from_fp(Fp::from_small(1728)));
}

#[test]
fn torsion_basis_holds_points() {
    use crate::{curves::TorsionBasis, fields::fp::Fp};

    let curve = Curve::E0;
    let R = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &curve);
    let S = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(7)), &curve);
    let RS = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(11)), &curve);

    let basis = TorsionBasis::from_propagated(R, S, RS);
    assert_eq!(basis.P, R);
    assert_eq!(basis.PmQ, S);
    assert_eq!(basis.Q, RS);
}

/// Verify that Jacobian doubling produces the same affine x
/// as Montgomery x-only doubling.
#[test]
fn jacobian_double_matches_montgomery() {
    let curve = Curve::E0;
    let p = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, &curve);

    // Montgomery double.
    let p2_mont = p.double();
    let p2_x_mont = &p2_mont.X * &p2_mont.Z.invert();

    // Jacobian: lift, double, convert, check.
    let x_aff = AffineX::from(&p.X * &p.Z.invert());
    let y = curve.recover_y(&x_aff).expect("P₀ should be on E₀");
    let p_jac = JacobianPoint::new(*x_aff.as_fp2(), y, Fp2::ONE, &curve);
    let p2_jac = p_jac.double();

    // Convert Jacobian to affine: x_aff = x / z².
    let z2_inv = p2_jac.Z.square().invert();
    let p2_x_jac = &p2_jac.X * &z2_inv;

    assert_eq!(
        p2_x_mont, p2_x_jac,
        "Jacobian double should match Montgomery double (affine x)"
    );
}

/// Verify that TorsionBasis::lift produces valid Jacobian points
/// that convert back to the correct Montgomery x-coordinates.
#[test]
fn lift_basis_round_trip() {
    let curve = Curve::E0;
    let p = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, &curve);
    let q = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_Q_X, &curve);
    let pmq = p.projective_difference(&q);

    let (p_jac, q_jac) = TorsionBasis::from_propagated(p, q, pmq)
        .lift(&curve)
        .expect("lift should succeed on E₀");

    // Check P: jac_to_xz(P_jac) should have same affine x as P.
    let p_back: ProjectiveXOnlyPoint = p_jac.into();
    let p_x = &p_back.X * &p_back.Z.invert();
    let p_orig_x = &p.X * &p.Z.invert();
    assert_eq!(
        p_x, p_orig_x,
        "P: lift + jac_to_xz should preserve affine x"
    );

    // Check Q: jac_to_xz(Q_jac) should have same affine x as Q.
    let q_back: ProjectiveXOnlyPoint = q_jac.into();
    let q_x = &q_back.X * &q_back.Z.invert();
    let q_orig_x = &q.X * &q.Z.invert();
    assert_eq!(
        q_x, q_orig_x,
        "Q: lift + jac_to_xz should preserve affine x"
    );

    // Check P_jac is on curve.
    let A = *curve.coefficient().as_fp2();
    let z_inv = p_jac.Z.invert();
    let xa = &p_jac.X * &z_inv.square();
    let ya = &p_jac.Y * &(&z_inv.square() * &z_inv);
    let lhs = ya.square();
    let xa2 = xa.square();
    let rhs = &(&(&xa2 * &xa) + &(&A * &xa2)) + &xa;
    assert_eq!(lhs, rhs, "P_jac should be on curve");

    // Check Q_jac is on curve.
    let z_inv = q_jac.Z.invert();
    let xa = &q_jac.X * &z_inv.square();
    let ya = &q_jac.Y * &(&z_inv.square() * &z_inv);
    let lhs = ya.square();
    let xa2 = xa.square();
    let rhs = &(&(&xa2 * &xa) + &(&A * &xa2)) + &xa;
    assert_eq!(lhs, rhs, "Q_jac should be on curve");
}

/// `Coefficient::is_singular` flags exactly `A == ±2` — the two
/// Montgomery coefficients where the discriminant `Δ = 4(A² − 4)`
/// vanishes. Anything else (including the start-curve `A = 0` and
/// arbitrary non-singular values) must be reported non-singular.
#[test]
fn coefficient_is_singular_flags_plus_minus_two() {
    use crate::fields::fp::Fp;

    let two = Fp2::from_fp(Fp::from_small(2));
    let neg_two = -&two;

    assert!(
        Coefficient::from(two).is_singular(),
        "A = 2 must be singular"
    );
    assert!(
        Coefficient::from(neg_two).is_singular(),
        "A = -2 must be singular"
    );

    assert!(
        !Coefficient::ZERO.is_singular(),
        "A = 0 (E_0) is non-singular"
    );
    let three = Fp2::from_fp(Fp::from_small(3));
    assert!(
        !Coefficient::from(three).is_singular(),
        "A = 3 is non-singular"
    );
}

/// Verify that jac_to_xz (From<JacobianPoint>) round-trips correctly.
#[test]
fn jac_to_xz_round_trip() {
    let curve = Curve::E0;
    let p = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, &curve);

    let x_aff = AffineX::from(&p.X * &p.Z.invert());
    let y = curve.recover_y(&x_aff).expect("P₀ should be on E₀");
    let p_jac = JacobianPoint::new(*x_aff.as_fp2(), y, Fp2::ONE, &curve);

    // jac_to_xz: (x, z) → (x, z²). For z=1, this is (x, 1).
    let p_mont: ProjectiveXOnlyPoint = p_jac.into();
    let p_x = &p_mont.X * &p_mont.Z.invert();
    let orig_x = &p.X * &p.Z.invert();
    assert_eq!(p_x, orig_x, "jac_to_xz should preserve affine x");
}
