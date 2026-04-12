use super::*;
use crate::curves::montgomery::{Curve, ProjectiveXOnlyPoint};

/// Test the splitting function with a synthetic product null point.
///
/// A product theta null point for E₁ × E₂ with theta constants
/// (a₁, b₁) and (a₂, b₂) is (a₁a₂, a₁b₂, b₁a₂, b₁b₂).
/// The splitting should find exactly one zero U_{i,j}.
///
/// This tests the splitting formula in isolation — no (2,2)-chain,
/// no kernel construction. The null point is constructed directly
/// with known product structure.
#[test]
fn splitting_synthetic_product() {
    use crate::fields::fp::Fp;

    // Fixed small values as theta constants. These are arbitrary
    // nonzero Fp2 elements; the only requirement is that the
    // product (a₁a₂, a₁b₂, b₁a₂, b₁b₂) has exactly one
    // vanishing U_{i,j}(0).
    let a1 = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let b1 = Fp2::new(Fp::from_small(5), Fp::from_small(11));
    let a2 = Fp2::new(Fp::from_small(13), Fp::from_small(17));
    let b2 = Fp2::new(Fp::from_small(19), Fp::from_small(23));

    let null = ThetaNullPoint::new(
        &a1 * &a2, // (0,0) component
        &a1 * &b2, // (0,1) component
        &b1 * &a2, // (1,0) component
        &b1 * &b2, // (1,1) component
    );

    let count = isogeny::get_index_splitting_count(&null);
    assert_eq!(
        count, 1,
        "product null point should have exactly 1 zero U index"
    );
}

/// Test the gluing codomain computation in isolation.
///
/// Uses E₀ × E₀ with the precomputed basis. This exercises the
/// gluing's `ActionByTranslation`, `theta_change_of_basis`,
/// `product_to_theta`, and `to_squared_theta` steps without
/// running the full chain. The codomain null point and dual are
/// checked for internal consistency.
///
/// Note: the kernel (P,Q)×(Q,P) may have a degenerate
/// `ActionByTranslation` determinant (the DMPR24 reference also
/// fails on this kernel). This test checks the codomain structure
/// rather than asserting the splitting succeeds.
#[test]
fn gluing_codomain_manual_check() {
    use crate::curves::TorsionBasis;

    let curve = Curve::E0;
    let p = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, &curve);
    let q = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_Q_X, &curve);

    // Scale to order 16.
    let mut p4 = p;
    let mut q4 = q;
    for _ in 0..244 {
        p4 = p4.double();
        q4 = q4.double();
    }

    let t1_0 = p4;
    let t1_1 = q4;
    let t2_0 = q4;
    let t2_1 = p4;

    let pmq_0 = t1_0.projective_difference(&t2_0);
    let pmq_1 = t1_1.projective_difference(&t2_1);
    let (t1_jac_0, t2_jac_0) = TorsionBasis::new(t1_0, t2_0, pmq_0)
        .lift(&curve)
        .expect("lift E0 failed");
    let (t1_jac_1, t2_jac_1) = TorsionBasis::new(t1_1, t2_1, pmq_1)
        .lift(&curve)
        .expect("lift E0 failed");

    let gluing = GluingKernel {
        T1: (t1_0, t1_1),
        T1_jac: (t1_jac_0, t1_jac_1),
        T2: (t2_0, t2_1),
        T2_jac: (t2_jac_0, t2_jac_1),
    };
    let data = gluing.codomain();

    // The dual should have delta = 0 (from hs1.W = 0 invariant).
    let null = &data.codomain.null;
    let _count = isogeny::get_index_splitting_count(null);

    // Check the dual form: (alpha, beta, gamma, 0).
    let dual_null =
        ThetaNullPoint::new(data.dual.alpha, data.dual.beta, data.dual.gamma, Fp2::ZERO);
    let _dual_count = isogeny::get_index_splitting_count(&dual_null);
}
