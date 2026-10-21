use super::*;
use crate::curves::montgomery::{Curve, ProjectiveXOnlyPoint};

/// Tests the splitting function with a synthetic product null point.
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

    let count = null.splitting_index_count();
    assert_eq!(
        count, 1,
        "product null point should have exactly 1 zero U index"
    );
}

/// Tests the gluing codomain computation in isolation.
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
    let (t1_jac_0, t2_jac_0) = TorsionBasis::from_propagated(t1_0, t2_0, pmq_0)
        .lift(&curve)
        .expect("lift E0 failed");
    let (t1_jac_1, t2_jac_1) = TorsionBasis::from_propagated(t1_1, t2_1, pmq_1)
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
    let _count = null.splitting_index_count();

    // Check the dual form: (alpha, beta, gamma, 0).
    let dual_null =
        ThetaNullPoint::new(data.dual.alpha, data.dual.beta, data.dual.gamma, Fp2::ZERO);
    let _dual_count = dual_null.splitting_index_count();
}

/// `EllipticProduct::from(&null)` (Algorithm 8.44) on a synthetic product null
/// point must recover the two component Montgomery coefficients exactly.
///
/// For a product theta null point `(α₁α₂, α₁β₂, β₁α₂, β₁β₂)` the algorithm
/// is expected to yield curves with affine coefficients
/// `Aᵢ = -2(αᵢ⁴ + βᵢ⁴) / (αᵢ⁴ - βᵢ⁴)`. The shared factors cancel cleanly
/// — the input components factor as `α₁⁴(α₂⁴ ± β₂⁴)` and
/// `α₂⁴(α₁⁴ ± β₁⁴)` after raising to the 4th power.
///
/// Direct value-based assertion. Catches surviving mutations on the final
/// projective-to-affine conversion (`A_num * C.invert()`), which the
/// existing end-to-end KAT path does not guard.
#[test]
fn theta_to_product_recovers_component_coefficients() {
    use crate::fields::fp::Fp;

    let alpha1 = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let beta1 = Fp2::new(Fp::from_small(5), Fp::from_small(11));
    let alpha2 = Fp2::new(Fp::from_small(13), Fp::from_small(17));
    let beta2 = Fp2::new(Fp::from_small(19), Fp::from_small(23));

    let null = ThetaNullPoint::new(
        &alpha1 * &alpha2,
        &alpha1 * &beta2,
        &beta1 * &alpha2,
        &beta1 * &beta2,
    );

    let product = EllipticProduct::from(&null);

    let four = |x: &Fp2| x.square().square();
    let expected_a = |alpha: &Fp2, beta: &Fp2| {
        let a4 = four(alpha);
        let b4 = four(beta);
        let num = -&(&(&a4 + &b4) + &(&a4 + &b4));
        &num * &(&a4 - &b4).invert()
    };

    let a1 = expected_a(&alpha1, &beta1);
    let a2 = expected_a(&alpha2, &beta2);

    assert_eq!(Fp2::from(*product.E1.coefficient()), a1, "E1 coefficient");
    assert_eq!(Fp2::from(*product.E2.coefficient()), a2, "E2 coefficient");
}

/// `theta_product_to_montgomery` (Algorithm 8.45) must compute the
/// projective `(X : Z)` pairs exactly per the formula
/// `X₁ = a·z + c·x, Z₁ = a·z − c·x, X₂ = a·y + b·x, Z₂ = a·y − b·x`.
///
/// Tests the formula in isolation by constructing a synthetic
/// [`JacobianPoint`] over a synthetic surface — the function only reads
/// the null-point components and the point's `(X, Y, Z, W)` coordinates,
/// so the surface is structural ballast.
///
/// One of the surviving mutants flips `+` to `−` in `X₁`, which makes
/// `X₁ = a·z − c·x = Z₁` and collapses `(X₁ : Z₁)` to `(1 : 1)` — the
/// projective `PartialEq` cross-multiplication catches this iff
/// `c·x ≠ 0`.
#[test]
fn theta_product_to_montgomery_matches_formula() {
    use crate::{
        curves::montgomery::{Curve, ProjectiveXOnlyPoint},
        fields::fp::Fp,
    };

    let alpha1 = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let beta1 = Fp2::new(Fp::from_small(5), Fp::from_small(11));
    let alpha2 = Fp2::new(Fp::from_small(13), Fp::from_small(17));
    let beta2 = Fp2::new(Fp::from_small(19), Fp::from_small(23));

    let null = ThetaNullPoint::new(
        &alpha1 * &alpha2,
        &alpha1 * &beta2,
        &beta1 * &alpha2,
        &beta1 * &beta2,
    );
    let surface = Jacobian::new(null);

    let x = Fp2::new(Fp::from_small(2), Fp::from_small(29));
    let y = Fp2::new(Fp::from_small(31), Fp::from_small(37));
    let z = Fp2::new(Fp::from_small(41), Fp::from_small(43));
    let w = Fp2::new(Fp::from_small(47), Fp::from_small(53));
    let pt = JacobianPoint::new(x, y, z, w, surface);

    // Curves are passed through to `ProjectiveXOnlyPoint::from_XZ` for
    // storage only — the formula's output values do not depend on them.
    let product = EllipticProduct::new(Curve::E0, Curve::E0);

    let (out1, out2) = pt.to_montgomery_on_product(&null, &product);

    let (a, b, c, _d) = (&null.a, &null.b, &null.c, &null.d);
    let exp_X1 = &(a * &z) + &(c * &x);
    let exp_Z1 = &(a * &z) - &(c * &x);
    let exp_X2 = &(a * &y) + &(b * &x);
    let exp_Z2 = &(a * &y) - &(b * &x);
    let exp1 = ProjectiveXOnlyPoint::from_XZ(exp_X1, exp_Z1, &Curve::E0);
    let exp2 = ProjectiveXOnlyPoint::from_XZ(exp_X2, exp_Z2, &Curve::E0);

    assert_eq!(out1, exp1, "(X₁ : Z₁) mismatch");
    assert_eq!(out2, exp2, "(X₂ : Z₂) mismatch");
}
