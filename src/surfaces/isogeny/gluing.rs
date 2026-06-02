//! `GluingKernel`: the first (2,2)-isogeny step in a chain, taking
//! `EllipticProduct → Jacobian`.
//!
//! Defined by two 8-torsion points on the product `E₁ × E₂`. The
//! gluing converts these to product points in alternative projective
//! coordinates, theta coordinates of level 2, and finally to a
//! Jacobian in theta coordinates. See [§8.5].
//!
//! [§8.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5

use crate::{
    curves::montgomery::{Coefficient, JacobianPoint as CurveJacobianPoint, ProjectiveXOnlyPoint},
    fields::fp2::Fp2,
    surfaces::{
        DualThetaNullPoint, GluingMatrix, Jacobian, JacobianPoint, ThetaNullPoint, hadamard4,
    },
};

/// Kernel data for the gluing (2,2)-isogeny `E₁ × E₂ → A`.
///
/// Holds two 8-torsion points `T₁'', T₂''` on the elliptic product in
/// both Montgomery `(X:Z)` form (for `product_to_theta` and codomain)
/// and Jacobian `(x, y, z)` form (for gluing evaluation, which needs `y`).
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
    /// Computes the gluing (2,2)-isogeny codomain and evaluation data.
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

    /// Computes the gluing and pushes all points through.
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

/// Intermediate products from `ActionByTranslation` ([§8.5.5],
/// Algorithm 8.35) before inversion.
///
/// Each 4-torsion point P' produces: WX, WZ, UX, UZ, δ = WX − UZ,
/// and the Z coordinate of P'. The inversions of δ and Z are deferred
/// for batching.
///
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.5
struct TranslationData {
    /// `W · Z`, pre-inversion.
    WZ: Fp2,
    /// `U · X`, pre-inversion.
    UX: Fp2,
    /// `U · Z`, pre-inversion.
    UZ: Fp2,
    /// `δ = WX − UZ`; its inverse is needed for the action.
    delta: Fp2,
    /// `X` coordinate of `P'`.
    X: Fp2,
    /// `Z` coordinate of `P'`.
    Z: Fp2,
}

/// Computes the pre-inversion data for `ActionByTranslation`
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
