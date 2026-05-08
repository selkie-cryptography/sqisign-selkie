//! (2,2)-isogeny kernel types and their isogeny computations.
//!
//! Each step of a (2,2)-isogeny chain is defined by its kernel data.
//! The kernel type determines the domain, codomain, and algorithm:
//!
//! | Kernel | Domain → Codomain | Data |
//! |---|---|---|
//! | [`GluingKernel`] | [`EllipticProduct`] → [`Jacobian`] | Two 8-torsion `ProductPoint` |
//! | [`GenericKernel8`] | [`Jacobian`] → [`Jacobian`] | Two 8-torsion `JacobianPoint` |
//! | [`GenericKernel4`] | [`Jacobian`] → [`Jacobian`] | One 4-torsion `JacobianPoint` |
//! | [`GenericKernel2`] | [`Jacobian`] → [`Jacobian`] | Domain Jacobian only |
//! | [`SplittingKernel`] | [`Jacobian`] → [`EllipticProduct`] | Domain Jacobian only |
//!
//! The top-level [`Kernel::isogeny`] orchestrates a full chain by
//! constructing the appropriate kernel type at each step.
//!
//! See [§2.4.1] and [§8.5.3] through [§8.5.8].
//!
//! [§2.4.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.4.1
//! [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
//! [§8.5.8]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.8
//!
//! [`Kernel::isogeny`]: super::Kernel::isogeny

use crate::{
    curves::montgomery::{
        Coefficient, Curve, JacobianPoint as CurveJacobianPoint, ProjectiveXOnlyPoint,
    },
    fields::fp2::Fp2,
    surfaces::{
        DualThetaNullPoint, EllipticProduct, GluingMatrix, Jacobian, JacobianPoint, ProductPoint,
        ThetaNullPoint, hadamard4, precomputed::NORMALIZATION_TRANSFORMS,
    },
};

// ---------------------------------------------------------------------------
// Gluing: EllipticProduct → Jacobian (§8.5.5, §8.5.6)
// ---------------------------------------------------------------------------

/// Kernel of a gluing (2,2)-isogeny Φ₁ : E₁ × E₂ → A₁.
///
/// Defined by two 8-torsion points on the domain product, stored as pairs of
/// Montgomery [`ProjectiveXOnlyPoint`]s (one on each component curve) in
/// projective coordinates. The gluing internally converts these to product
/// points in alternative projective coordinates, _theta coordinates of level
/// 2_, and then points on the Jacobian in theta coordinates.
///
/// See [§8.5].
///
/// [§8.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
pub(crate) struct GluingKernel {
    /// T₁'' = (T₁''₁, T₁''₂) ∈ E₁ × E₂.
    /// Montgomery (X:Z) coordinates — used for product_to_theta and codomain.
    pub T1: (ProjectiveXOnlyPoint, ProjectiveXOnlyPoint),
    /// Jacobian (x,y,z) coordinates — used for gluing eval (needs y).
    pub T1_jac: (CurveJacobianPoint, CurveJacobianPoint),
    /// T₂'' = (T₂''₁, T₂''₂) ∈ E₁ × E₂, in Montgomery coordinates.
    pub T2: (ProjectiveXOnlyPoint, ProjectiveXOnlyPoint),
    /// Jacobian coordinates for T₂''.
    pub T2_jac: (CurveJacobianPoint, CurveJacobianPoint),
}

/// Data produced by the gluing codomain computation, needed for
/// evaluating points through the gluing isogeny.
pub(crate) struct GluingData {
    /// Dual isogenous theta null point (α : β : γ : 0).
    pub dual: DualThetaNullPoint,
    /// Codomain Jacobian.
    pub codomain: Jacobian,
    /// Image of T₁'' under Φ, in Jacobian coordinates on A
    /// (re-used for evaluation).
    pub J: JacobianPoint,
    /// Change-of-basis matrix N (4×4 over F_{p²}).
    pub N: GluingMatrix,
}

impl GluingKernel {
    /// Compute the gluing (2,2)-isogeny codomain and evaluation data.
    ///
    /// Implements `GluingCodomain` ([§8.5.5], Algorithm 8.38).
    ///
    /// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.5
    pub(crate) fn codomain(&self) -> GluingData {
        // Algorithm 8.38:
        // 1. T₁' ← [2](T₁'')    T₂' ← [2](T₂'')
        //
        // Double in JACOBIAN, then convert to Montgomery via jac_to_xz.
        // The C reference does this at gluing_compute:434-437 to ensure
        // the correct projective representative (x, z²) for the
        // theta change-of-basis computation.
        // Use double_for_theta (standard Z'=2yz Jacobian) to match
        // the C ref's projective representative for jac_to_xz.
        let T1_prime: (ProjectiveXOnlyPoint, ProjectiveXOnlyPoint) = (
            ProjectiveXOnlyPoint::from(self.T1_jac.0.double_for_theta()),
            ProjectiveXOnlyPoint::from(self.T1_jac.1.double_for_theta()),
        );
        let T2_prime: (ProjectiveXOnlyPoint, ProjectiveXOnlyPoint) = (
            ProjectiveXOnlyPoint::from(self.T2_jac.0.double_for_theta()),
            ProjectiveXOnlyPoint::from(self.T2_jac.1.double_for_theta()),
        );

        // 3. N ← ThetaChangeOfBasis(T₁', T₂')
        let N = theta_change_of_basis(&T1_prime, &T2_prime);

        // 4. [P₁, P₂] ← ProductToTheta([T₁'', T₂''], N)
        //
        // Apply N to the 8-torsion kernel points in Montgomery (X:Z).
        //
        // The C reference converts K1_8 via jac_to_xz before base_change.
        // Both 4-torsion (for N) and 8-torsion (for product_to_theta)
        // use the same (x, z²) representative from jac_to_xz.
        //
        // Our GluingKernel.T1/T2 are set by the chain to the
        // jac_to_xz-converted Montgomery points (from From<JacobianPoint>).
        let theta_pts = product_to_theta(&[self.T1, self.T2], &N);

        let P1 = &theta_pts[0];
        let P2 = &theta_pts[1];

        // 7–8. (X₁,Y₁,Z₁,W₁) ← H ∘ S(P₁), (X₂,Y₂,Z₂,W₂) ← H ∘ S(P₂)
        let hs1 = squared_hadamard4(&P1.0, &P1.1, &P1.2, &P1.3);
        let hs2 = squared_hadamard4(&P2.0, &P2.1, &P2.2, &P2.3);

        // 13–15. Recover α, β, γ from the cross-products.
        //
        // The C reference (`gluing_compute`, theta_isogenies.c:480-496):
        //   codomain = (X₁·X₂, Y₁·X₂, X₁·Z₂, 0)
        //   precomp  = (Y₁·Z₂, X₁·Z₂, Y₁·X₂, 0)  ← projective inverse
        //   imageK1_8 = (x, y) where x = X₁·Y₁·Z₂, y = Z₁·X₁·Z₂ (unneeded here)
        let alpha = &hs1.0 * &hs2.0; // X₁ · X₂
        let beta = &hs1.1 * &hs2.0; // Y₁ · X₂
        let gamma = &hs1.0 * &hs2.2; // X₁ · Z₂

        // Projective inverse: (α⁻¹, β⁻¹, γ⁻¹) = (Y₁·Z₂, X₁·Z₂, Y₁·X₂).
        // No field inversion needed — these are cross-products.
        let alpha_inv = &hs1.1 * &hs2.2; // Y₁ · Z₂
        let beta_inv = &hs1.0 * &hs2.2; // X₁ · Z₂
        let gamma_inv = &hs1.1 * &hs2.0; // Y₁ · X₂

        let dual = DualThetaNullPoint {
            alpha,
            beta,
            gamma,
            delta: Fp2::ZERO,
            alpha_inv,
            beta_inv,
            gamma_inv,
            delta_inv: Fp2::ZERO,
        };

        // 19. imageK1_8 = (x : x : y : y).
        //
        // C reference (theta_isogenies.c:478-480):
        //   imageK1_8.x = TT1.x * precomp.x = X₁ · (Y₁·Z₂)
        //   imageK1_8.y = TT1.z * precomp.z = Z₁ · (Y₁·X₂)
        //
        // where precomp.x = Y₁Z₂ = alpha_inv, and
        //       precomp.z = codomain.y = Y₁X₂ = beta.
        let x = &hs1.0 * &alpha_inv; // X₁ · (Y₁·Z₂)
        let y = &hs1.2 * &beta; // Z₁ · (Y₁·X₂)

        // Apply Hadamard: H(α, β, γ, 0). This gives standard form
        // with all four coordinates nonzero, which is needed for the
        // precomputation (d=0 makes three precomp values zero,
        // killing coordinates during theta doubling).
        //
        // Day 15 note: this is an internal representation choice.
        // C ref's chain stores codomain with t=0 directly and uses
        // a different (cross-product) precomputation that handles
        // the degeneracy. Selkie's chain runs in an H-shifted
        // representation throughout. Removing the Hadamard here
        // breaks Selkie's chain entirely (splitting:zeros=10) — the
        // downstream chain operations assume non-degenerate null.
        // The shifted representation is internally consistent and
        // produces correct codomains, just with different intermediate
        // bytes than C ref. The 6-KAT regression with the (P, Q) FDI
        // fix is from a different convention mismatch, not from this H.
        let (a2, b2, c2, d2) = hadamard4(&alpha, &beta, &gamma, &Fp2::ZERO);
        let null = ThetaNullPoint::new(a2, b2, c2, d2);
        let codomain = Jacobian::new(null);

        let J = JacobianPoint::new(x, x, y, y, codomain.clone());

        GluingData {
            dual,
            codomain,
            J,
            N,
        }
    }

    /// Evaluate the gluing at a point P ∈ E₁ × E₂.
    ///
    /// Implements `GluingEval` ([§8.5.6], Algorithm 8.39).
    ///
    /// [§8.5.6]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.6
    /// Evaluate the gluing at a general point P ∈ E₁ × E₂.
    ///
    /// Implements `GluingEval` ([§8.5.6], Algorithm 8.39).
    /// Uses `ADDComponents` to compute the addition and subtraction
    /// of P with the kernel generator T₁'', then combines via
    /// the change-of-basis matrix and Hadamard.
    ///
    /// [§8.5.6]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.6
    pub(crate) fn eval(
        P: &(CurveJacobianPoint, CurveJacobianPoint),
        T1: &(CurveJacobianPoint, CurveJacobianPoint),
        A1: &Coefficient,
        A2: &Coefficient,
        data: &GluingData,
    ) -> JacobianPoint {
        // Algorithm 8.39 (GluingEval):
        // Uses Jacobian (x,y,z) points for the cross-addition components,
        // which require y-coordinates to distinguish P+Q from P-Q.
        //
        // J = imageK1_8 has the pattern (x:x:y:y). J.x = J.y = x,
        // J.z = J.w = y. Scaling uses (y,y,x,x) = projective inverse.
        let (x_val, y_val) = (&data.J.X, &data.J.Z);

        // 4–5. ADDComponents for each component curve (Jacobian).
        let (u1, v1, w1) = add_sub_components_jac(&P.0, &T1.0, A1);
        let (u2, v2, w2) = add_sub_components_jac(&P.1, &T1.1, A2);

        // 6. U ← (u₁·u₂ + v₁·v₂, u₁·w₂, w₁·u₂, w₁·w₂)
        //
        // NOTE: The C reference (`gluing_eval_point`, lines 512-517)
        // uses cross-component products: u₁·w₂ (not u₁·w₁) and
        // v₁·w₂ (not v₁·w₁). Component 1 = (u₁,v₁,w₁) from curve 1,
        // component 2 = (u₂,v₂,w₂) from curve 2.
        let U = (&(&u1 * &u2) + &(&v1 * &v2), &u1 * &w2, &w1 * &u2, &w1 * &w2);

        // 7. V ← (v₁·u₂ + u₁·v₂, v₁·w₂, w₁·v₂, 0)
        let V = (&(&v1 * &u2) + &(&u1 * &v2), &v1 * &w2, &w1 * &v2, Fp2::ZERO);

        // 8–9. U ← N · U,  V ← N · V
        let U = &data.N * &U;
        let V = &data.N * &V;

        // 10–11. U ← S(U),  V ← S(V)
        let U = (U.0.square(), U.1.square(), U.2.square(), U.3.square());
        let V = (V.0.square(), V.1.square(), V.2.square(), V.3.square());

        // 12. (X±, Y±, Z±, W±) ← H(U − V)
        let diff = (&U.0 - &V.0, &U.1 - &V.1, &U.2 - &V.2, &U.3 - &V.3);
        let (Xpm, Ypm, Zpm, Wpm) = hadamard4(&diff.0, &diff.1, &diff.2, &diff.3);

        // 13. Scale by (y, y, x, x) — the projective inverse of (x,x,y,y).
        let Xout = &Xpm * y_val;
        let Yout = &Ypm * y_val;
        let Zout = &Zpm * x_val;
        let Wout = &Wpm * x_val;

        let (xr, yr, zr, wr) = hadamard4(&Xout, &Yout, &Zout, &Wout);
        JacobianPoint::new(xr, yr, zr, wr, data.codomain.clone())
    }

    /// Evaluate at a special point (P₁, 0) or (0, P₂).
    ///
    /// Implements `GluingEvalSpecial` ([§8.5.6], Algorithm 8.40).
    ///
    /// [§8.5.6]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.6
    pub(crate) fn eval_special(
        P: &(ProjectiveXOnlyPoint, ProjectiveXOnlyPoint),
        data: &GluingData,
    ) -> JacobianPoint {
        // Convert via ProductToTheta then apply the dual inverse.
        let theta = product_to_theta(&[*P], &data.N);
        let (x, y, z, w) = (&theta[0].0, &theta[0].1, &theta[0].2, &theta[0].3);

        // H ∘ S, then scale by inverse, then H again.
        let (X, Y, Z, _W) = squared_hadamard4(x, y, z, w);
        let Xp = &X * &data.dual.alpha_inv;
        let Yp = &Y * &data.dual.beta_inv;
        let Zp = &Z * &data.dual.gamma_inv;
        // W component is zero for gluing (δ = 0).

        let (xp, yp, zp, wp) = hadamard4(&Xp, &Yp, &Zp, &Fp2::ZERO);
        JacobianPoint::new(xp, yp, zp, wp, data.codomain.clone())
    }

    /// Compute the gluing and push all points through.
    pub(crate) fn isogeny(
        &self,
        pts: &[(ProjectiveXOnlyPoint, ProjectiveXOnlyPoint)],
    ) -> (GluingData, Vec<JacobianPoint>) {
        let data = self.codomain();
        let images = pts
            .iter()
            .map(|p| GluingKernel::eval_special(p, &data))
            .collect();
        (data, images)
    }
}

// ---------------------------------------------------------------------------
// Gluing helpers
// ---------------------------------------------------------------------------

/// Intermediate products from `ActionByTranslation` ([§8.5.5],
/// Algorithm 8.35) before inversion.
///
/// Each 4-torsion point P' produces: WX, WZ, UX, UZ, δ = WX − UZ,
/// and the Z coordinate of P'. The inversions of δ and Z are deferred
/// for batching.
///
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.5
struct TranslationData {
    WX: Fp2,
    WZ: Fp2,
    UX: Fp2,
    UZ: Fp2,
    delta: Fp2,
    X: Fp2,
    Z: Fp2,
}

/// Compute the pre-inversion data for `ActionByTranslation`
/// ([§8.5.5], Algorithm 8.35).
///
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.5
fn translation_pre_invert(P_prime: &ProjectiveXOnlyPoint) -> TranslationData {
    let P = P_prime.double();
    let (X, Z) = (P_prime.X, P_prime.Z);
    let (U, W) = (P.X, P.Z);
    let WX = &W * &X;
    let WZ = &W * &Z;
    let UX = &U * &X;
    let UZ = &U * &Z;
    let delta = &WX - &UZ;
    TranslationData {
        WX,
        WZ,
        UX,
        UZ,
        delta,
        X,
        Z,
    }
}

/// Complete `ActionByTranslation` ([§8.5.5], Algorithm 8.35)
/// given batched inverses of δ and Z.
///
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.5
fn translation_finish(d: &TranslationData, delta_inv: &Fp2, Z_inv: &Fp2) -> [[Fp2; 2]; 2] {
    let m00 = &(-&d.UZ) * delta_inv;
    let m01 = &(-&d.WZ) * delta_inv;
    let m10 = &(&d.UX * delta_inv) - &(&d.X * Z_inv);
    let m11 = -&m00;
    [[m00, m01], [m10, m11]]
}

/// Batch-invert k elements using [Montgomery's trick][mont].
///
/// Given \[a₁, ..., aₖ\], computes \[a₁⁻¹, ..., aₖ⁻¹\] using only
/// 1 inversion and 3(k−1) multiplications.
///
/// [mont]: https://en.wikipedia.org/wiki/Montgomery%27s_trick
fn batch_invert(elems: &[Fp2]) -> Vec<Fp2> {
    let n = elems.len();
    if n == 0 {
        return vec![];
    }

    // Forward pass: compute prefix products.
    let mut prefix = Vec::with_capacity(n);
    prefix.push(elems[0]);
    for i in 1..n {
        prefix.push(&prefix[i - 1] * &elems[i]);
    }

    // Single inversion of the total product.
    let mut inv = prefix[n - 1].invert();

    // Backward pass: peel off each element.
    let mut result = vec![Fp2::ZERO; n];
    for i in (1..n).rev() {
        result[i] = &inv * &prefix[i - 1];
        inv = &inv * &elems[i];
    }
    result[0] = inv;
    result
}

/// `ThetaChangeOfBasis` ([§8.5.5], Algorithm 8.36).
///
/// Computes the 4×4 change-of-basis matrix N from 4-torsion points
/// T₁ = (T₁₁, T₁₂) and T₂ = (T₂₁, T₂₂) on E₁ × E₂.
///
/// Uses batched inversions: 4 calls to `ActionByTranslation` need
/// 8 inversions (δ⁻¹ and Z⁻¹ each), batched into 1 inversion + 21
/// multiplications.
///
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.5
fn theta_change_of_basis(
    T1: &(ProjectiveXOnlyPoint, ProjectiveXOnlyPoint),
    T2: &(ProjectiveXOnlyPoint, ProjectiveXOnlyPoint),
) -> GluingMatrix {
    // Pre-inversion data for all four components.
    let d_G = translation_pre_invert(&T1.0);
    let d_Gp = translation_pre_invert(&T1.1);
    let d_H = translation_pre_invert(&T2.0);
    let d_Hp = translation_pre_invert(&T2.1);

    // Batch-invert all 8 elements: [δ_G, Z_G, δ_G', Z_G', δ_H, Z_H, δ_H', Z_H']
    let to_invert = [
        d_G.delta, d_G.Z, d_Gp.delta, d_Gp.Z, d_H.delta, d_H.Z, d_Hp.delta, d_Hp.Z,
    ];
    let invs = batch_invert(&to_invert);

    #[cfg(test)]
    {
        let zero_count = [&d_G.delta, &d_Gp.delta, &d_H.delta, &d_Hp.delta]
            .iter()
            .filter(|d| **d == &Fp2::ZERO)
            .count();
        if zero_count > 0 {
            crate::selkie_trace!(
                "GLUING: {zero_count}/4 delta(s) are ZERO (degenerate ActionByTranslation)"
            );
        }
    }

    // Complete each ActionByTranslation with the batched inverses.
    let G = translation_finish(&d_G, &invs[0], &invs[1]);
    let Gp = translation_finish(&d_Gp, &invs[2], &invs[3]);
    let H = translation_finish(&d_H, &invs[4], &invs[5]);
    let Hp = translation_finish(&d_Hp, &invs[6], &invs[7]);

    // Lines 4–7: intermediate products.
    let t1 = &G[0][0] * &H[0][0] + &G[0][1] * &H[1][0];
    let t2 = &G[1][0] * &H[0][0] + &G[1][1] * &H[1][0];
    let t3 = &Gp[0][0] * &Hp[0][0] + &Gp[0][1] * &Hp[1][0];
    let t4 = &Gp[1][0] * &Hp[0][0] + &Gp[1][1] * &Hp[1][0];

    // Lines 8–23: build the 4×4 matrix N.
    let one = Fp2::ONE;

    let gg00 = &G[0][0] * &Gp[0][0];
    let hh00 = &H[0][0] * &Hp[0][0];
    let t1t3 = &t1 * &t3;
    let N00 = &(&(&gg00 + &hh00) + &t1t3) + &one;

    let gg01 = &G[0][0] * &Gp[1][0];
    let hh01 = &H[0][0] * &Hp[1][0];
    let t1t4 = &t1 * &t4;
    let N01 = &(&gg01 + &hh01) + &t1t4;

    let gg10 = &G[1][0] * &Gp[0][0];
    let hh10 = &H[1][0] * &Hp[0][0];
    let t2t3 = &t2 * &t3;
    let N02 = &(&gg10 + &hh10) + &t2t3;

    let gg11 = &G[1][0] * &Gp[1][0];
    let hh11 = &H[1][0] * &Hp[1][0];
    let t2t4 = &t2 * &t4;
    let N03 = &(&gg11 + &hh11) + &t2t4;

    // Rows 1–3 reference N₀,ⱼ from row 0.
    let N10 = &(&Hp[0][0] * &N00) + &(&Hp[0][1] * &N01);
    let N11 = &(&Hp[1][0] * &N00) + &(&Hp[1][1] * &N01);
    let N12 = &(&Hp[0][0] * &N02) + &(&Hp[0][1] * &N03);
    let N13 = &(&Hp[1][0] * &N02) + &(&Hp[1][1] * &N03);

    let N20 = &(&G[0][0] * &N00) + &(&G[0][1] * &N02);
    let N21 = &(&G[0][0] * &N01) + &(&G[0][1] * &N03);
    let N22 = &(&G[1][0] * &N00) + &(&G[1][1] * &N02);
    let N23 = &(&G[1][0] * &N01) + &(&G[1][1] * &N03);

    let N30 = &(&G[0][0] * &N10) + &(&G[0][1] * &N12);
    let N31 = &(&G[0][0] * &N11) + &(&G[0][1] * &N13);
    let N32 = &(&G[1][0] * &N10) + &(&G[1][1] * &N12);
    let N33 = &(&G[1][0] * &N11) + &(&G[1][1] * &N13);

    GluingMatrix([
        [N00, N01, N02, N03],
        [N10, N11, N12, N13],
        [N20, N21, N22, N23],
        [N30, N31, N32, N33],
    ])
}

/// `ProductToTheta` ([§8.5.5], Algorithm 8.37).
///
/// Converts pairs of Montgomery points to product theta coordinates
/// using the change-of-basis matrix N. For each point P = (P₁, P₂)
/// where Pᵢ = (θ₀,ᵢ : θ₁,ᵢ) in dimension-1 theta coordinates:
///
/// ```text
/// x ← θ₀,₁ · θ₀,₂     y ← θ₀,₁ · θ₁,₂
/// z ← θ₁,₁ · θ₀,₂     w ← θ₁,₁ · θ₁,₂
/// P' ← N · (x : y : z : w)
/// ```
///
/// The dimension-1 theta coordinates (θ₀ : θ₁) are obtained from
/// Montgomery (X : Z) via `MontgomeryToTheta` (Algorithm 8.27):
/// θ₀ = a·(X−Z), θ₁ = b·(X+Z), where (a : b) is the theta null
/// point of the component curve.
///
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.5
fn product_to_theta(
    pts: &[(ProjectiveXOnlyPoint, ProjectiveXOnlyPoint)],
    N: &GluingMatrix,
) -> Vec<(Fp2, Fp2, Fp2, Fp2)> {
    pts.iter()
        .map(|(P1, P2)| {
            // Product theta coordinates from raw projective (X:Z) pairs.
            //
            // The C reference (`base_change` in theta_isogenies.c:144-148)
            // uses (X, Z) directly — NOT (X−Z, X+Z). The change-of-basis
            // matrix N was computed assuming this raw product structure.
            let x = &P1.X * &P2.X; // X₁ · X₂
            let y = &P1.X * &P2.Z; // X₁ · Z₂
            let z = &P1.Z * &P2.X; // Z₁ · X₂
            let w = &P1.Z * &P2.Z; // Z₁ · Z₂

            // Apply change-of-basis matrix N.
            N * &(x, y, z, w)
        })
        .collect()
}

/// Decompose the sum and difference of two projective points into
/// shared components.
///
/// Given two Jacobian points P = (x_P, y_P, z_P) and Q = (x_Q, y_Q, z_Q)
/// on the same Montgomery curve E_A, returns (u, v, w) such that:
/// - x(P + Q) = (u − v : w)
/// - x(P − Q) = (u + v : w)
///
/// This uses the full Jacobian addition formula with y-coordinates,
/// which is needed for the (2,2)-isogeny gluing step to correctly
/// distinguish P+Q from P-Q.
///
/// Implements `jac_to_xz_add_components` from the C reference
/// (`ec_jac.c:305`). The Montgomery x-only version was incorrect
/// for the gluing — see BUG 10 in project_theta_bugs.md.
///
/// [§8.2.4]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.2.4
fn add_sub_components_jac(
    P: &CurveJacobianPoint,
    Q: &CurveJacobianPoint,
    A: &Coefficient,
) -> (Fp2, Fp2, Fp2) {
    // C reference (ec_jac.c:305-335):
    let a = A.as_fp2();
    let t0 = P.Z.square(); // z1²
    let t1 = Q.Z.square(); // z2²
    let t2 = &P.X * &t1; // x1·z2²
    let t3 = &t0 * &Q.X; // z1²·x2
    let mut t4 = &P.Y * &Q.Z; // y1·z2
    t4 = &t4 * &t1; // y1·z2³
    let mut t5 = &P.Z * &Q.Y; // z1·y2
    t5 = &t5 * &t0; // z1³·y2
    let t0 = &t0 * &t1; // (z1·z2)²
    let t6 = &t4 * &t5; // (z1·z2)³·y1·y2
    let v = &t6 + &t6; // 2·(z1·z2)³·y1·y2
    let t4_sq = t4.square(); // y1²·z2⁶
    let t5_sq = t5.square(); // z1⁶·y2²
    let sum_y2 = &t4_sq + &t5_sq; // y1²·z2⁶ + z1⁶·y2²
    let sum_x = &t2 + &t3; // x1·z2² + z1²·x2
    let lambda = &t2 - &t3; // x1·z2² - z1²·x2
    let lambda_sq = lambda.square();
    let a_t0 = a * &t0;
    let gamma = &(&sum_x + &a_t0) * &lambda_sq; // (sum_x + A·(z1z2)²)·λ²
    let u = &sum_y2 - &gamma;
    let w = &lambda_sq * &t0; // (z1·z2)²·λ²
    (u, v, w)
}

/// H ∘ S on four coordinates: square each, then Hadamard.
fn squared_hadamard4(x: &Fp2, y: &Fp2, z: &Fp2, w: &Fp2) -> (Fp2, Fp2, Fp2, Fp2) {
    hadamard4(&x.square(), &y.square(), &z.square(), &w.square())
}

// ---------------------------------------------------------------------------
// Generic with 8-torsion: Jacobian → Jacobian (§8.5.3)
// ---------------------------------------------------------------------------

/// Kernel of a generic (2,2)-isogeny specified by two 8-torsion
/// points on a Jacobian.
///
/// The most efficient variant — used for all interior chain steps
/// where 8-torsion data is available.
///
/// See [§8.5.3], Algorithm 8.30.
///
/// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
pub(crate) struct GenericKernel8 {
    /// 8-torsion point T₁'' on the domain Jacobian.
    pub T1: JacobianPoint,
    /// 8-torsion point T₂'' on the domain Jacobian.
    pub T2: JacobianPoint,
}

impl GenericKernel8 {
    /// Compute the generic (2,2)-isogeny and push points through.
    ///
    /// Returns the codomain Jacobian and the images of `pts`.
    ///
    /// Implements `GenericCodomainWith8Torsion` + `GenericEval`
    /// ([§8.5.3], Algorithm 8.30; [§8.5.4], Algorithm 8.34).
    ///
    /// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
    /// [§8.5.4]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.4
    pub(crate) fn isogeny(&self, pts: &[JacobianPoint]) -> (Jacobian, Vec<JacobianPoint>) {
        let (dual, codomain) = codomain_8torsion(&self.T1, &self.T2);
        let images = pts.iter().map(|p| eval(p, &dual, &codomain)).collect();
        (codomain, images)
    }
}

// ---------------------------------------------------------------------------
// Generic with 4-torsion: Jacobian → Jacobian (§8.5.3)
// ---------------------------------------------------------------------------

/// Kernel of a generic (2,2)-isogeny specified by a single 4-torsion
/// point on a Jacobian, plus the domain's theta null point.
///
/// Used at the penultimate step of a chain when only 4-torsion data
/// remains. Requires two square roots.
///
/// See [§8.5.3], Algorithm 8.32.
///
/// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
pub(crate) struct GenericKernel4 {
    /// 4-torsion point T₁' on the domain Jacobian.
    pub T1: JacobianPoint,
}

impl GenericKernel4 {
    /// Compute the generic (2,2)-isogeny and push points through.
    ///
    /// Implements `GenericCodomainWith4Torsion` + `GenericEval`
    /// ([§8.5.3], Algorithm 8.32).
    ///
    /// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
    pub(crate) fn isogeny(
        &self,
        domain: &Jacobian,
        pts: &[JacobianPoint],
    ) -> (Jacobian, Vec<JacobianPoint>) {
        let (dual, codomain) = codomain_4torsion(&self.T1, domain);
        let images = pts.iter().map(|p| eval(p, &dual, &codomain)).collect();
        (codomain, images)
    }

    /// Penultimate-step (2,2)-isogeny for an `extra_torsion=false`
    /// chain tail.
    ///
    /// Uses [`codomain_4torsion_no_hadamard`] (flag `(0, 0)`) and
    /// [`eval_no_outer_hadamard`]. The codomain is left in dual
    /// form for the matching ultimate step or splitting consumer.
    ///
    /// C reference: `theta_isogeny_compute_4(..., 0, 0)` at
    /// `theta_isogenies.c:1258`.
    pub(crate) fn isogeny_penultimate(
        &self,
        domain: &Jacobian,
        pts: &[JacobianPoint],
    ) -> (Jacobian, Vec<JacobianPoint>) {
        let (dual, codomain) = codomain_4torsion_no_hadamard(&self.T1, domain);
        let images = pts
            .iter()
            .map(|p| eval_no_outer_hadamard(p, &dual, &codomain))
            .collect();
        (codomain, images)
    }
}

// ---------------------------------------------------------------------------
// Generic with 2-torsion: Jacobian → Jacobian (§8.5.3)
// ---------------------------------------------------------------------------

/// Kernel of a generic (2,2)-isogeny specified only by the domain
/// Jacobian's theta null point (2-torsion kernel generators).
///
/// Used at the final generic step of a chain. Requires three square
/// roots.
///
/// See [§8.5.3], Algorithm 8.33.
///
/// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
pub(crate) struct GenericKernel2;

impl GenericKernel2 {
    /// Compute the generic (2,2)-isogeny and push points through.
    ///
    /// Implements `GenericCodomain` + `GenericEval`
    /// ([§8.5.3], Algorithm 8.33).
    ///
    /// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
    pub(crate) fn isogeny(
        domain: &Jacobian,
        pts: &[JacobianPoint],
    ) -> (Jacobian, Vec<JacobianPoint>) {
        let (dual, codomain) = codomain_from_null(domain);
        let images = pts.iter().map(|p| eval(p, &dual, &codomain)).collect();
        (codomain, images)
    }

    /// Ultimate-step (2,2)-isogeny for an `extra_torsion=false`
    /// chain tail.
    ///
    /// Uses [`codomain_2torsion_ultimate`] (flag `(1, 0)`) and
    /// [`eval_ultimate`]. The codomain is left in dual form for the
    /// splitting consumer.
    ///
    /// C reference: `theta_isogeny_compute_2(..., 1, 0)` at
    /// `theta_isogenies.c:1266`.
    pub(crate) fn isogeny_ultimate(
        domain: &Jacobian,
        pts: &[JacobianPoint],
    ) -> (Jacobian, Vec<JacobianPoint>) {
        let (dual, codomain) = codomain_2torsion_ultimate(domain);
        let images = pts
            .iter()
            .map(|p| eval_ultimate(p, &dual, &codomain))
            .collect();
        (codomain, images)
    }
}

// ---------------------------------------------------------------------------
// Splitting: Jacobian → EllipticProduct (§8.5.7)
// ---------------------------------------------------------------------------

/// Kernel of a splitting (2,2)-isogeny Φₑ : Aₑ₋₁ → E₃ × E₄.
///
/// The final step of a (2,2)-isogeny chain. The domain Jacobian's
/// theta null point has product structure, allowing recovery of the
/// component curves E₃, E₄.
///
/// See [§8.5.7].
///
/// [§8.5.7]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.7
pub(crate) struct SplittingKernel {
    /// The domain Jacobian (whose null point has product structure).
    pub domain: Jacobian,
}

impl SplittingKernel {
    /// Compute the splitting and push points through.
    ///
    /// Returns the codomain product E₃ × E₄ and the images of `pts`
    /// converted back to Montgomery coordinates, or [`None`] if the
    /// chain's final theta null does not admit a product splitting
    /// (i.e., [`count_splitting_indices`] is not exactly 1).
    ///
    /// Implements `SplittingIsomorphism` + `ThetaToProduct` +
    /// `ThetaProductPointToMontgomery` ([§8.5.7]).
    ///
    /// # Divergences
    ///
    /// The spec's Algorithm 8.42 says to "find the unique index
    /// such that `U_{i,j}(0) = 0`" without stating that exactly
    /// one must vanish. In practice the splitting-index count is
    /// a load-bearing invariant: a malformed input (e.g., an
    /// upstream chain that was fed a kernel short on torsion)
    /// produces a terminal theta null where either 0 or 10 of the
    /// ten `U_{i,j}(0)` coordinates vanish, and the splitting
    /// machinery then silently picks a wrong branch and emits
    /// bad curves. Treating `count != 1` as an explicit error
    /// is the difference between "signing key is wrong but
    /// keygen looks successful" and "keygen retries with a fresh
    /// random ideal." The SQIsign v2 spec review flags this as
    /// a recommended spec clarification (§\textsc{SplittingIsomorphism}
    /// must handle malformed input).
    ///
    /// [§8.5.7]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.7
    pub(crate) fn isogeny(
        &self,
        pts: &[JacobianPoint],
        randomize: Option<&mut dyn rand_core::RngCore>,
    ) -> Option<(EllipticProduct, Vec<ProductPoint>)> {
        // Exactly one of the 10 `U_{i,j}(0)` coordinates must
        // vanish for the chain's terminal theta null to
        // correspond to a product of elliptic curves. Anything
        // else is a malformed chain output (see Divergences
        // above).
        let split_count = count_splitting_indices(&self.domain.null);
        if split_count != 1 {
            #[cfg(test)]
            crate::selkie_trace!("    [chain] splitting: zeros={split_count}");
            return None;
        }

        // 1. SplittingIsomorphism: find the matrix M (Algorithm 8.42).
        let mut M = splitting_isomorphism(&self.domain.null);

        // Apply a random level-2 normalization matrix when the caller
        // requested randomization. See `NORMALIZATION_TRANSFORMS` for
        // the underlying construction. This does not change the
        // *abstract* product surface — both `M` and
        // `NORMALIZATION_TRANSFORMS[idx] · M` send the level-2 theta
        // null to a product theta null on the same `E₁ × E₂` — but it
        // does randomize the specific projective representative the
        // caller observes. Mirrors C reference's
        // `splitting_compute(...)`'s `randomize=true` branch
        // (`theta_isogenies.c:1043-1059`).
        if let Some(rng) = randomize {
            let idx = sample_normalization_index(rng) as usize;
            M = &NORMALIZATION_TRANSFORMS[idx] * &M;
        }

        // 2. Apply M to the null point to get product theta structure.
        let product_null = M.apply_null(&self.domain.null);

        // 3. ThetaToProduct: recover (A₁:C₁), (A₂:C₂) (Algorithm 8.44).
        let product = theta_to_product(&product_null);

        // 4. ThetaProductPointToMontgomery for each point (Algorithm 8.45).
        let images = pts
            .iter()
            .map(|p| {
                // Apply M to point, then convert to Montgomery.
                let mp = M.apply_point(p);
                theta_product_to_montgomery(&mp, &product_null, &product)
            })
            .collect();

        Some((product, images))
    }
}

/// Sample a uniform index in `[0, 6)` for selecting one of the six
/// [`NORMALIZATION_TRANSFORMS`] matrices.
///
/// Reads four bytes from `rng`, parses them as a little-endian
/// `u32`, and rejects-and-resamples any value `≥ 6 · ⌊2³² / 6⌋ =
/// 4_294_967_292` to obtain an unbiased `mod 6` reduction.
///
/// Mirrors the C reference's `sample_random_index`
/// (`theta_isogenies.c:980`) — same byte-stream consumption pattern
/// (4 bytes, little-endian, rejection-sample threshold `4_294_967_292`)
/// so a deterministic DRBG seeded identically on both
/// implementations selects the same index. The C reference also
/// uses a constant-time `mod 6` trick (Granlund–Möller); we use the
/// plain `% 6` since this code path is variable-time on public data
/// (the chain output is public; the secret signing key has already
/// been fully consumed by the kernel).
///
/// [`NORMALIZATION_TRANSFORMS`]: super::precomputed::NORMALIZATION_TRANSFORMS
fn sample_normalization_index<R: rand_core::RngCore + ?Sized>(rng: &mut R) -> u8 {
    loop {
        let mut buf = [0u8; 4];
        rng.fill_bytes(&mut buf);
        let seed = u32::from_le_bytes(buf);
        if seed < 4_294_967_292u32 {
            return (seed % 6) as u8;
        }
        // Resample on the rare seed in `[4_294_967_292, 2³²)` —
        // approximately `4 / 2³² ≈ 10⁻⁹` chance per draw.
    }
}

// ---------------------------------------------------------------------------
// Internal computations
// ---------------------------------------------------------------------------

/// Codomain from 8-torsion (Algorithm 8.30).
pub(crate) fn codomain_8torsion(
    T1: &JacobianPoint,
    T2: &JacobianPoint,
) -> (DualThetaNullPoint, Jacobian) {
    let hs1 = T1.squared().hadamard();
    let hs2 = T2.squared().hadamard();

    #[cfg(test)]
    dump_step_internal_inputs(T1, T2, &hs1, &hs2);

    let xawb = &hs1.X * &hs2.Y;
    let zaxb = &hs2.X * &hs1.Y;

    let alpha = &hs2.X * &xawb;
    let beta = &hs2.Y * &zaxb;
    let gamma = &hs2.Z * &xawb;
    let delta = &hs2.W * &zaxb;

    #[cfg(test)]
    dump_step_internal_pre_h(&alpha, &beta, &gamma, &delta);

    #[cfg(test)]
    {
        if alpha == gamma {
            crate::selkie_trace!("codomain_8torsion: alpha==gamma → after H: c=d=0!");
        }
    }

    let zgwd = &hs2.Z * &hs2.W;
    let alpha_inv = &hs1.Y * &zgwd;
    let beta_inv = &hs1.X * &zgwd;
    let gamma_inv = delta;
    let delta_inv = gamma;

    let dual = DualThetaNullPoint {
        alpha,
        beta,
        gamma,
        delta,
        alpha_inv,
        beta_inv,
        gamma_inv,
        delta_inv,
    };
    let null_B = hadamard_null(&dual);
    (dual, Jacobian::new(null_B))
}

/// Codomain from 4-torsion (Algorithm 8.32).
fn codomain_4torsion(T1: &JacobianPoint, domain: &Jacobian) -> (DualThetaNullPoint, Jacobian) {
    // Line 1: (xαβ, _, xγδ, _) ← H ∘ S(T₁')
    let hs = T1.squared().hadamard();

    // Line 2: (α², β², γ², δ²) ← H ∘ S(0_A)
    let (a2, b2, g2, d2) = hadamard4(
        &domain.null.a.square(),
        &domain.null.b.square(),
        &domain.null.c.square(),
        &domain.null.d.square(),
    );

    // Lines 3–4: square roots.
    let ab = (&a2 * &b2).sqrt();
    let ag = (&a2 * &g2).sqrt();

    // Lines 5–8: recover (α, β, γ, δ).
    let beta = &(&ab * &ag) * &hs.Z;
    let delta_inv = &beta * &hs.X;
    let beta_mul = &beta * &hs.X;
    let xgd_ab_a2 = &(&hs.Z * &ab) * &a2;
    let _delta = &xgd_ab_a2 * &(&ab * &a2);
    let alpha = &(&hs.X * &ab) * &a2;
    let gamma = &alpha * &g2;
    let delta_final = &alpha * &d2;

    // Projective inverses.
    let alpha_inv = &hs.X * &d2;
    let beta_inv = &alpha * &b2;
    let gamma_inv_val = &delta_inv * &b2;

    let dual = DualThetaNullPoint {
        alpha,
        beta: beta_mul,
        gamma,
        delta: delta_final,
        alpha_inv,
        beta_inv,
        gamma_inv: gamma_inv_val,
        delta_inv,
    };
    let null_B = hadamard_null(&dual);
    (dual, Jacobian::new(null_B))
}

/// Codomain from null point only (Algorithm 8.33).
fn codomain_from_null(domain: &Jacobian) -> (DualThetaNullPoint, Jacobian) {
    let (a2, b2, g2, d2) = hadamard4(
        &domain.null.a.square(),
        &domain.null.b.square(),
        &domain.null.c.square(),
        &domain.null.d.square(),
    );

    let alpha = a2;
    let beta = (&a2 * &b2).sqrt();
    let gamma = (&a2 * &g2).sqrt();
    let delta = (&a2 * &d2).sqrt();

    let ab = &alpha * &beta;
    let gd = &gamma * &delta;
    let alpha_inv = &ab * &d2;
    let beta_inv = &ab * &g2;
    let gamma_inv = &gd * &b2;
    let delta_inv = &gd * &a2;

    let dual = DualThetaNullPoint {
        alpha,
        beta,
        gamma,
        delta,
        alpha_inv,
        beta_inv,
        gamma_inv,
        delta_inv,
    };
    let null_B = hadamard_null(&dual);
    (dual, Jacobian::new(null_B))
}

/// Codomain from 8-torsion WITHOUT final Hadamard (dual form).
///
/// Same as [`codomain_8torsion`] but omits the Hadamard transform on
/// the codomain null point, corresponding to `hadamard_bool_2=0` in
/// the C reference. Used for the penultimate and ultimate chain steps
/// so the splitting step receives the codomain in dual form.
pub(crate) fn codomain_8torsion_no_hadamard(
    T1: &JacobianPoint,
    T2: &JacobianPoint,
) -> (DualThetaNullPoint, Jacobian) {
    let hs1 = T1.squared().hadamard();
    let hs2 = T2.squared().hadamard();

    let xawb = &hs1.X * &hs2.Y;
    let zaxb = &hs2.X * &hs1.Y;

    let alpha = &hs2.X * &xawb;
    let beta = &hs2.Y * &zaxb;
    let gamma = &hs2.Z * &xawb;
    let delta = &hs2.W * &zaxb;

    let zgwd = &hs2.Z * &hs2.W;
    let alpha_inv = &hs1.Y * &zgwd;
    let beta_inv = &hs1.X * &zgwd;
    let gamma_inv = delta;
    let delta_inv = gamma;

    let dual = DualThetaNullPoint {
        alpha,
        beta,
        gamma,
        delta,
        alpha_inv,
        beta_inv,
        gamma_inv,
        delta_inv,
    };
    // NO hadamard_null here — codomain stays in dual form.
    let null = ThetaNullPoint::new(dual.alpha, dual.beta, dual.gamma, dual.delta);
    (dual, Jacobian::new(null))
}

/// Evaluate: normal interior step (`hadamard_bool_1=0, hadamard_bool_2=1`).
///
/// The C reference's `theta_isogeny_eval` with bool_1=0, bool_2=1
/// computes: `H(precomp · to_squared_theta(P))` where
/// `to_squared_theta(P) = H(P²)`.
///
/// However, the `precomp` (alpha_inv etc.) stored in our
/// `DualThetaNullPoint` already incorporates the coordinate
/// relationships such that the eval formula is simply
/// `H(precomp · P²)` — the inner Hadamard is absorbed into how
/// the precomputation relates to the codomain.
pub(crate) fn eval(
    P: &JacobianPoint,
    dual: &DualThetaNullPoint,
    codomain: &Jacobian,
) -> JacobianPoint {
    let t = P
        .squared()
        .hadamard()
        .scale(
            &dual.alpha_inv,
            &dual.beta_inv,
            &dual.gamma_inv,
            &dual.delta_inv,
        )
        .hadamard();
    JacobianPoint::new(t.X, t.Y, t.Z, t.W, codomain.clone())
}

/// Evaluate: penultimate step (`hadamard_bool_1=0, hadamard_bool_2=0`).
///
/// Formula: `precomp · H(P²)` — no outer Hadamard.
pub(crate) fn eval_no_outer_hadamard(
    P: &JacobianPoint,
    dual: &DualThetaNullPoint,
    codomain: &Jacobian,
) -> JacobianPoint {
    let t = P.squared().hadamard().scale(
        &dual.alpha_inv,
        &dual.beta_inv,
        &dual.gamma_inv,
        &dual.delta_inv,
    );
    JacobianPoint::new(t.X, t.Y, t.Z, t.W, codomain.clone())
}

/// Evaluate: ultimate step (`hadamard_bool_1=1, hadamard_bool_2=0`).
///
/// Formula: `precomp · H(H(P)²)` — extra Hadamard on input, no outer.
pub(crate) fn eval_ultimate(
    P: &JacobianPoint,
    dual: &DualThetaNullPoint,
    codomain: &Jacobian,
) -> JacobianPoint {
    let t = P.hadamard().squared().hadamard().scale(
        &dual.alpha_inv,
        &dual.beta_inv,
        &dual.gamma_inv,
        &dual.delta_inv,
    );
    JacobianPoint::new(t.X, t.Y, t.Z, t.W, codomain.clone())
}

/// Codomain from 8-torsion: ultimate step (`hadamard_bool_1=1,
/// hadamard_bool_2=0`).
///
/// Same cross-product formulas as the normal 8-torsion codomain, but
/// applies Hadamard to each kernel point BEFORE `to_squared_theta`,
/// and omits the final Hadamard on the codomain. This compensates for
/// the penultimate step having produced a dual-form codomain.
///
/// C reference: `theta_isogeny_compute` with `hadamard_bool_1=1,
/// hadamard_bool_2=0` (theta_isogenies.c:636-644, 692-694).
pub(crate) fn codomain_8torsion_ultimate(
    T1: &JacobianPoint,
    T2: &JacobianPoint,
) -> (DualThetaNullPoint, Jacobian) {
    // bool_1=1: Hadamard before to_squared_theta
    let hs1 = T1.hadamard().squared().hadamard();
    let hs2 = T2.hadamard().squared().hadamard();

    let xawb = &hs1.X * &hs2.Y;
    let zaxb = &hs2.X * &hs1.Y;

    let alpha = &hs2.X * &xawb;
    let beta = &hs2.Y * &zaxb;
    let gamma = &hs2.Z * &xawb;
    let delta = &hs2.W * &zaxb;

    let zgwd = &hs2.Z * &hs2.W;
    let alpha_inv = &hs1.Y * &zgwd;
    let beta_inv = &hs1.X * &zgwd;
    let gamma_inv = delta;
    let delta_inv = gamma;

    let dual = DualThetaNullPoint {
        alpha,
        beta,
        gamma,
        delta,
        alpha_inv,
        beta_inv,
        gamma_inv,
        delta_inv,
    };
    // bool_2=0: NO final Hadamard — codomain stays in dual form.
    let null = ThetaNullPoint::new(dual.alpha, dual.beta, dual.gamma, dual.delta);
    (dual, Jacobian::new(null))
}

/// Codomain from 4-torsion: penultimate step (`hadamard_bool_1=0,
/// hadamard_bool_2=0`).
///
/// Same as [`codomain_4torsion`] (Algorithm 8.32) but omits the final
/// Hadamard on the codomain null point. The codomain stays in dual
/// form, matching the C reference's `theta_isogeny_compute_4(..., 0, 0)`
/// call at `theta_isogenies.c:1258` — the dedicated penultimate step
/// in an `extra_torsion=false` chain.
///
/// Pair with [`eval_no_outer_hadamard`] for the matching evaluator.
pub(crate) fn codomain_4torsion_no_hadamard(
    T1: &JacobianPoint,
    domain: &Jacobian,
) -> (DualThetaNullPoint, Jacobian) {
    // TT1 = H(S(T1_4)). For a dual-form 4-torsion point, only the X
    // and Z components are nonzero (Y = W = 0), but we don't depend
    // on that explicitly here.
    let tt1 = T1.squared().hadamard();
    // tt1.X, tt1.Y, tt1.Z, tt1.W ↔ C ref's TT1.x, TT1.y, TT1.z, TT1.t.

    // TT2 = H(S(domain.null)). Maps to C ref's TT2.{x,y,z,t}.
    let (tt2_x, tt2_y, tt2_z, tt2_t) = hadamard4(
        &domain.null.a.square(),
        &domain.null.b.square(),
        &domain.null.c.square(),
        &domain.null.d.square(),
    );

    let sqaabb = (&tt2_x * &tt2_y).sqrt();
    let sqaacc = (&tt2_x * &tt2_z).sqrt();

    // Codomain null point — direct transcription of C ref's
    // `theta_isogeny_compute_4(0, 0)` body
    // (`theta_isogenies.c:789–802`):
    //
    //   null.x = TT1.x · TT2.x · sqaacc
    //   null.y = TT1.x · sqaabb · sqaacc
    //   null.z = TT1.x · TT2.x · TT2.z
    //   null.t = TT1.z · sqaabb · TT2.x
    let null_x = &(&tt1.X * &tt2_x) * &sqaacc;
    let null_y = &(&tt1.X * &sqaabb) * &sqaacc;
    let null_z = &(&tt1.X * &tt2_x) * &tt2_z;
    let null_t = &(&tt1.Z * &sqaabb) * &tt2_x;

    // Precomputation for evaluation
    // (`theta_isogenies.c:790, 804–810`):
    //
    //   precomp.x = TT1.x · TT2.t · TT2.z · TT2.y
    //   precomp.y = TT1.x · TT2.t · TT2.z · sqaabb
    //   precomp.z = TT1.x · TT2.t · TT2.y · sqaacc
    //   precomp.t = sqaabb · sqaacc · TT1.z · TT2.y
    let xt = &tt1.X * &tt2_t;
    let xtz = &xt * &tt2_z;
    let xty = &xt * &tt2_y;
    let precomp_x = &xtz * &tt2_y;
    let precomp_y = &xtz * &sqaabb;
    let precomp_z = &xty * &sqaacc;
    let sab_sac_z = &(&sqaabb * &sqaacc) * &tt1.Z;
    let precomp_t = &sab_sac_z * &tt2_y;

    let dual = DualThetaNullPoint {
        alpha: null_x,
        beta: null_y,
        gamma: null_z,
        delta: null_t,
        alpha_inv: precomp_x,
        beta_inv: precomp_y,
        gamma_inv: precomp_z,
        delta_inv: precomp_t,
    };
    // hadamard_bool_2 = 0 → codomain stays in dual form (no final
    // Hadamard on the null point).
    let null = ThetaNullPoint::new(dual.alpha, dual.beta, dual.gamma, dual.delta);
    (dual, Jacobian::new(null))
}

/// Codomain from 2-torsion (null only): ultimate step
/// (`hadamard_bool_1=1, hadamard_bool_2=0`).
///
/// Same shape as [`codomain_from_null`] (Algorithm 8.33) but applies
/// Hadamard to the domain null before squaring (compensating for the
/// previous step's dual-form output) and omits the final Hadamard on
/// the codomain.
///
/// C reference: `theta_isogeny_compute_2(..., 1, 0)` at
/// `theta_isogenies.c:1266` — the dedicated ultimate step in an
/// `extra_torsion=false` chain.
///
/// Pair with [`eval_ultimate`] for the matching evaluator.
pub(crate) fn codomain_2torsion_ultimate(domain: &Jacobian) -> (DualThetaNullPoint, Jacobian) {
    // bool_1 = 1: TT2 = H(S(H(domain.null))) = to_squared_theta of
    // hadamarded null. Maps to C ref's TT2.{x,y,z,t} = AA, BB, CC, DD.
    let (na, nb, nc, nd) = hadamard4(
        &domain.null.a,
        &domain.null.b,
        &domain.null.c,
        &domain.null.d,
    );
    let (tt2_x, tt2_y, tt2_z, tt2_t) =
        hadamard4(&na.square(), &nb.square(), &nc.square(), &nd.square());

    // Codomain null — direct from C ref's
    // `theta_isogeny_compute_2(1, 0)` body
    // (`theta_isogenies.c:860–867`):
    //
    //   null.x = TT2.x                        // AA
    //   null.y = sqrt(TT2.x · TT2.y)          // sqrt(AA·BB)
    //   null.z = sqrt(TT2.x · TT2.z)          // sqrt(AA·CC)
    //   null.t = sqrt(TT2.x · TT2.t)          // sqrt(AA·DD)
    let alpha = tt2_x;
    let beta = (&tt2_x * &tt2_y).sqrt();
    let gamma = (&tt2_x * &tt2_z).sqrt();
    let delta = (&tt2_x * &tt2_t).sqrt();

    // Precomputation (`theta_isogenies.c:869–877`):
    //
    //   precomp.x = TT2.y · TT2.z · TT2.t          // BB·CC·DD
    //   precomp.y = TT2.z · TT2.t · null.y         // CC·DD·sqrt(AA·BB)
    //   precomp.z = TT2.y · TT2.t · null.z         // BB·DD·sqrt(AA·CC)
    //   precomp.t = TT2.y · TT2.z · null.t         // BB·CC·sqrt(AA·DD)
    let zt = &tt2_z * &tt2_t;
    let yt = &tt2_y * &tt2_t;
    let yz = &tt2_y * &tt2_z;

    let alpha_inv = &yz * &tt2_t; // BB·CC·DD
    let beta_inv = &zt * &beta; // CC·DD·sqrt(AA·BB)
    let gamma_inv = &yt * &gamma; // BB·DD·sqrt(AA·CC)
    let delta_inv = &yz * &delta; // BB·CC·sqrt(AA·DD)

    let dual = DualThetaNullPoint {
        alpha,
        beta,
        gamma,
        delta,
        alpha_inv,
        beta_inv,
        gamma_inv,
        delta_inv,
    };
    // bool_2=0: codomain stays in dual form (no final Hadamard).
    let null = ThetaNullPoint::new(dual.alpha, dual.beta, dual.gamma, dual.delta);
    (dual, Jacobian::new(null))
}

/// Codomain theta null point from dual via Hadamard.
fn hadamard_null(dual: &DualThetaNullPoint) -> ThetaNullPoint {
    let (a, b, c, d) = hadamard4(&dual.alpha, &dual.beta, &dual.gamma, &dual.delta);
    ThetaNullPoint::new(a, b, c, d)
}

// ---------------------------------------------------------------------------
// Splitting helpers (§8.5.7)
// ---------------------------------------------------------------------------

/// The splitting index (i, j) identifying which coordinate of the
/// theta null point vanishes under the U_{i,j} map.
///
/// This is an exhaustive enumeration of the 10 valid indices from
/// `GetIndexSplitting` ([§8.5.7], Algorithm 8.41). The match in
/// [`splitting_isomorphism`] covers all variants, so adding a new
/// variant requires adding its matrix — enforced at compile time.
///
/// [§8.5.7]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.7
#[derive(Copy, Clone, Debug)]
enum SplittingIndex {
    /// (i, j) = (0, 0)
    I00,
    /// (i, j) = (0, 1)
    I01,
    /// (i, j) = (0, 2)
    I02,
    /// (i, j) = (0, 3)
    I03,
    /// (i, j) = (1, 0)
    I10,
    /// (i, j) = (1, 2)
    I12,
    /// (i, j) = (2, 0)
    I20,
    /// (i, j) = (2, 1)
    I21,
    /// (i, j) = (3, 0)
    I30,
    /// (i, j) = (3, 3)
    I33,
}

/// All valid splitting indices, in the order checked by Algorithm 8.41.
const SPLITTING_INDICES: [(usize, usize, SplittingIndex); 10] = [
    (0, 0, SplittingIndex::I00),
    (0, 1, SplittingIndex::I01),
    (0, 2, SplittingIndex::I02),
    (0, 3, SplittingIndex::I03),
    (1, 0, SplittingIndex::I10),
    (1, 2, SplittingIndex::I12),
    (2, 0, SplittingIndex::I20),
    (2, 1, SplittingIndex::I21),
    (3, 0, SplittingIndex::I30),
    (3, 3, SplittingIndex::I33),
];

/// χ function for `GetIndexSplitting` (Algorithm 8.41).
fn chi(i: usize, j: usize) -> i8 {
    match (i, j) {
        (0, 0) | (0, 1) | (0, 2) | (0, 3) | (1, 0) | (1, 2) | (2, 0) | (2, 1) | (3, 0) | (3, 3) => {
            1
        }
        (1, 1) | (1, 3) | (2, 2) | (2, 3) | (3, 1) | (3, 2) => -1,
        _ => 0,
    }
}

/// Counts how many of the 10 `U_{i,j}(0)` coordinates vanish at this
/// theta null point.
///
/// For a chain that ends at a product of elliptic curves, exactly one
/// `U_{i,j}(0)` is zero (it identifies which product decomposition
/// applies). Any other count — particularly 0 — signals that the
/// codomain is not a product and the splitting machinery will produce
/// bad output if applied.
pub(crate) fn count_splitting_indices(null: &ThetaNullPoint) -> u32 {
    let coords = [&null.a, &null.b, &null.c, &null.d];
    let mut count = 0u32;
    for &(i, j, _idx) in &SPLITTING_INDICES {
        let mut U = Fp2::ZERO;
        for t in 0..4 {
            let chi_val = chi(i, t) as i64;
            if chi_val != 0 {
                let term = coords[j ^ t] * coords[t];
                if chi_val > 0 {
                    U = &U + &term;
                } else {
                    U = &U - &term;
                }
            }
        }
        if U == Fp2::ZERO {
            count += 1;
        }
    }
    count
}

/// Test alias for [`count_splitting_indices`], kept so diagnostic
/// callers in `surfaces::mod` continue to compile with the original
/// name.
#[cfg(test)]
pub(crate) fn get_index_splitting_count(null: &ThetaNullPoint) -> u32 {
    count_splitting_indices(null)
}

/// Find the splitting index such that U_{i,j}(0) = 0
/// (Algorithm 8.41).
fn get_index_splitting(null: &ThetaNullPoint) -> SplittingIndex {
    let coords = [&null.a, &null.b, &null.c, &null.d];

    let mut count = 0;
    let mut result = SplittingIndex::I00;
    for &(i, j, idx) in &SPLITTING_INDICES {
        let mut U = Fp2::ZERO;
        for t in 0..4 {
            let chi_val = chi(i, t) as i64;
            if chi_val != 0 {
                let term = coords[j ^ t] * coords[t];
                if chi_val > 0 {
                    U = &U + &term;
                } else {
                    U = &U - &term;
                }
            }
        }
        if U == Fp2::ZERO {
            count += 1;
            result = idx;
        }
    }
    debug_assert!(
        count == 1,
        "GetIndexSplitting: expected exactly one zero index, found {count}"
    );
    result
}

/// Compute `SplittingIsomorphism` (Algorithm 8.42).
///
/// Returns the 4×4 matrix M whose action on the null point recovers
/// the product theta structure.
fn splitting_isomorphism(null: &ThetaNullPoint) -> GluingMatrix {
    let idx = get_index_splitting(null);
    let one = Fp2::ONE;
    let neg = -&one;
    let zero = Fp2::ZERO;

    // The matrices for each (i,j) case come from Algorithm 8.42.
    // For simplicity, only implement the cases that arise in
    // Isogeny22Chain (the spec guarantees (i,j) = (0,0) or (1,1)
    // for SQIsign's chain via Algorithm 8.47).
    use SplittingIndex::*;
    GluingMatrix(match idx {
        I00 => {
            // C reference: SPLITTING_TRANSFORMS[0] for (i,j) = (0,0).
            // Uses i = sqrt(-1) in Fp2.
            let i_val = Fp2::I;
            let neg_i = -&i_val;
            [
                [one, i_val, one, i_val],
                [one, neg_i, neg, i_val],
                [one, i_val, neg, neg_i],
                [neg, i_val, neg, i_val],
            ]
        }
        I10 => [
            [one, one, one, one],
            [one, neg, neg, one],
            [one, one, neg, neg],
            [neg, one, neg, one],
        ],
        I20 => [
            [one, one, one, one],
            [one, neg, one, neg],
            [one, neg, neg, one],
            [neg, neg, one, one],
        ],
        I30 => [
            [one, one, one, one],
            [one, neg, one, neg],
            [one, one, neg, neg],
            [neg, one, one, neg],
        ],
        I01 => [
            [one, zero, zero, zero],
            [zero, zero, zero, one],
            [zero, zero, one, zero],
            [zero, neg, zero, zero],
        ],
        I21 => [
            [one, one, one, one],
            [one, neg, one, neg],
            [one, neg, neg, one],
            [one, one, neg, neg],
        ],
        I02 => [
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, zero, one],
            [zero, zero, neg, zero],
        ],
        I12 => [
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, zero, one],
            [zero, zero, one, zero],
        ],
        I03 => [
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, one, zero],
            [zero, zero, zero, neg],
        ],
        I33 => [
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, one, zero],
            [zero, zero, zero, one],
        ],
    })
}

/// Recover Montgomery coefficients from a product theta null point
/// (Algorithm 8.44).
///
/// Constructs each component curve via
/// `Curve::from(ProjectiveCoefficient)`, preserving the un-reduced
/// `(A : C)` form that comes out of the formulas (`A = -2(x⁴ + z⁴)`,
/// `C = x⁴ − z⁴`). The cached `DoublingConstants` is still normalized
/// to `(A₂₄/C₂₄ : 1)`, so `.double()` produces the normalized
/// `xDBL_A24` representative used in most downstream code paths.
///
/// The original `(A : C)` is still readable via `curve.projective`.
/// Code paths that need to byte-match C ref's un-normalized `xDBL`
/// (e.g., the outer-chain prep doublings in `to_isogeny`, where C ref
/// skips `ec_curve_normalize_A24`) use
/// [`ProjectiveXOnlyPoint::double_unnormalized`] instead.
pub(crate) fn theta_to_product(null: &ThetaNullPoint) -> EllipticProduct {
    use crate::curves::montgomery::ProjectiveCoefficient;

    let (a, b, c, d) = (&null.a, &null.b, &null.c, &null.d);

    // Check product structure: ad == bc.
    debug_assert!(
        &(a * d) == &(b * c),
        "ThetaToProduct: not a product theta structure"
    );

    let x = a.square().square(); // a⁴
    let y = b.square().square(); // b⁴
    let z = c.square().square(); // c⁴

    // (A₂ : C₂) for E₂: A₂ = -2(x + y), C₂ = x - y
    let pc2 = ProjectiveCoefficient {
        A: -&(&(&x + &y) + &(&x + &y)),
        C: &x - &y,
    };

    // (A₁ : C₁) for E₁: A₁ = -2(x + z), C₁ = x - z
    let pc1 = ProjectiveCoefficient {
        A: -&(&(&x + &z) + &(&x + &z)),
        C: &x - &z,
    };

    EllipticProduct::new(Curve::from(pc1), Curve::from(pc2))
}

/// Convert a theta point with product structure to Montgomery
/// coordinates on each component (Algorithm 8.45).
pub(crate) fn theta_product_to_montgomery(
    P: &JacobianPoint,
    null: &ThetaNullPoint,
    product: &EllipticProduct,
) -> ProductPoint {
    let (a, b, c, _d) = (&null.a, &null.b, &null.c, &null.d);
    #[allow(unused_variables)]
    let (x, y, z, w) = (&P.X, &P.Y, &P.Z, &P.W);

    // Algorithm 8.45:
    // X₁ = a·z + c·x,  Z₁ = a·z − c·x
    // X₂ = a·y + b·x,  Z₂ = a·y − b·x
    let X1 = &(a * z) + &(c * x);
    let Z1 = &(a * z) - &(c * x);
    let X2 = &(a * y) + &(b * x);
    let Z2 = &(a * y) - &(b * x);

    (
        ProjectiveXOnlyPoint::from_XZ(X1, Z1, &product.E1),
        ProjectiveXOnlyPoint::from_XZ(X2, Z2, &product.E2),
    )
}

/// Format an `Fp2` element as `(re, im)` little-endian hex strings.
///
/// Used by the `dump_step_internal_*` byte-interop diagnostics to emit
/// the same line format the C reference's `[CHAIN_DUMP]` macros use,
/// so a single `diff` run aligns lines between the two implementations.
#[cfg(test)]
fn dump_fp2_hex(value: &Fp2) -> (String, String) {
    let bytes = value.to_bytes();
    let re: String = bytes[..32]
        .iter()
        .rev()
        .map(|b| format!("{b:02x}"))
        .collect();
    let im: String = bytes[32..]
        .iter()
        .rev()
        .map(|b| format!("{b:02x}"))
        .collect();
    (re, im)
}

/// Emit `[CHAIN_DUMP] step=internal` lines for the four 8-torsion step
/// inputs and their squared-Hadamard images, byte-formatted to match
/// the C reference's `theta_isogeny_compute` instrumentation.
///
/// `T1`/`T2` are the kernel inputs (mapped `X→x`, `Y→y`, `Z→z`, `W→t`
/// to align with the C reference's `theta_point_t` field names).
/// `hs1`/`hs2` are `H ∘ S` of `T1`/`T2`, i.e. the C ref's `TT1`/`TT2`
/// when `hadamard_bool_1 == 0`.
///
/// Reader correlates dump groups with the surrounding
/// `[MODA] main {step}` / `[CHAIN_DUMP] step=main_<i>` markers emitted
/// by the chain main loop after the step returns.
#[cfg(test)]
fn dump_step_internal_inputs(
    T1: &JacobianPoint,
    T2: &JacobianPoint,
    hs1: &JacobianPoint,
    hs2: &JacobianPoint,
) {
    let emit = |label: &str, value: &Fp2| {
        let (re, im) = dump_fp2_hex(value);
        crate::selkie_trace!("[CHAIN_DUMP] step=internal {label}.re=0x{re}");
        crate::selkie_trace!("[CHAIN_DUMP] step=internal {label}.im=0x{im}");
    };
    emit("T1.x", &T1.X);
    emit("T1.y", &T1.Y);
    emit("T1.z", &T1.Z);
    emit("T1.t", &T1.W);
    emit("T2.x", &T2.X);
    emit("T2.y", &T2.Y);
    emit("T2.z", &T2.Z);
    emit("T2.t", &T2.W);
    emit("TT1.x", &hs1.X);
    emit("TT1.y", &hs1.Y);
    emit("TT1.z", &hs1.Z);
    emit("TT1.t", &hs1.W);
    emit("TT2.x", &hs2.X);
    emit("TT2.y", &hs2.Y);
    emit("TT2.z", &hs2.Z);
    emit("TT2.t", &hs2.W);
}

/// Emit `[CHAIN_DUMP] step=internal pre_H.null.{a..d}` lines for the
/// codomain null point before the final Hadamard.
///
/// In Selkie's [`codomain_8torsion`], `(α, β, γ, δ)` are the
/// pre-Hadamard codomain coordinates: they correspond exactly to the
/// C reference's `out->codomain.null_point.{x,y,z,t}` after the four
/// `fp2_mul`s but before the `hadamard_bool_2` branch. Mapped to
/// `{a,b,c,d}` so the labels match the post-Hadamard null dumps.
#[cfg(test)]
fn dump_step_internal_pre_h(alpha: &Fp2, beta: &Fp2, gamma: &Fp2, delta: &Fp2) {
    let emit = |label: &str, value: &Fp2| {
        let (re, im) = dump_fp2_hex(value);
        crate::selkie_trace!("[CHAIN_DUMP] step=internal {label}.re=0x{re}");
        crate::selkie_trace!("[CHAIN_DUMP] step=internal {label}.im=0x{im}");
    };
    emit("pre_H.null.a", alpha);
    emit("pre_H.null.b", beta);
    emit("pre_H.null.c", gamma);
    emit("pre_H.null.d", delta);
}
