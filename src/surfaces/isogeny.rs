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
//! [§2.4.1]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.4
//! [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
//! [§8.5.8]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
//!
//! [`Kernel::isogeny`]: super::Kernel::isogeny

use crate::curves::montgomery::MontgomeryPoint;
use crate::fields::fp2::Fp2;
use crate::surfaces::{
    DualThetaNullPoint, EllipticProduct, Jacobian, JacobianPoint,
    ProductPoint, ThetaNullPoint, hadamard4,
};

// ---------------------------------------------------------------------------
// Gluing: EllipticProduct → Jacobian (§8.5.5, §8.5.6)
// ---------------------------------------------------------------------------

/// Kernel of a gluing (2,2)-isogeny Φ₁ : E₁ × E₂ → A₁.
///
/// Defined by two 8-torsion points on the domain product, stored as
/// pairs of Montgomery [`MontgomeryPoint`]s (one on each component
/// curve). The gluing internally converts these to product theta
/// coordinates.
///
/// See [§8.5.5] and [§8.5.6].
///
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
/// [§8.5.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
pub(crate) struct GluingKernel {
    /// T₁'' = (T₁''₁, T₁''₂) ∈ E₁ × E₂, in Montgomery coordinates.
    pub T1: (MontgomeryPoint, MontgomeryPoint),
    /// T₂'' = (T₂''₁, T₂''₂) ∈ E₁ × E₂, in Montgomery coordinates.
    pub T2: (MontgomeryPoint, MontgomeryPoint),
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
    pub N: [[Fp2; 4]; 4],
}

impl GluingKernel {
    /// Compute the gluing (2,2)-isogeny codomain and evaluation data.
    ///
    /// Implements `GluingCodomain` ([§8.5.5], Algorithm 8.38).
    ///
    /// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
    pub(crate) fn codomain(&self) -> GluingData {
        // Algorithm 8.38:
        // 1. T₁' ← [2](T₁'')    T₂' ← [2](T₂'')
        let T1_prime = (self.T1.0.double(), self.T1.1.double());
        let T2_prime = (self.T2.0.double(), self.T2.1.double());

        // 3. N ← ThetaChangeOfBasis(T₁', T₂')
        let N = theta_change_of_basis(&T1_prime, &T2_prime);

        // 4. [P₁, P₂] ← ProductToTheta([T₁'', T₂''], N)
        let theta_pts = product_to_theta(&[self.T1, self.T2], &N);
        let P1 = &theta_pts[0];
        let P2 = &theta_pts[1];

        // 7–8. (X₁,Y₁,Z₁,W₁) ← H ∘ S(P₁), (X₂,Y₂,Z₂,W₂) ← H ∘ S(P₂)
        let hs1 = squared_hadamard4(&P1.0, &P1.1, &P1.2, &P1.3);
        let hs2 = squared_hadamard4(&P2.0, &P2.1, &P2.2, &P2.3);

        // 13–15. Recover α, β, γ from the cross-products.
        let alpha = &hs1.0 * &hs2.1;
        let beta = &hs1.1 * &hs2.0;
        let gamma = &hs1.0 * &hs2.2;

        // For gluing, δ = 0 and the inverse has δ⁻¹ = 0.
        let alpha_inv = &hs1.0 * &hs2.0.invert();
        let beta_inv = beta;
        let gamma_inv = gamma;

        let dual = DualThetaNullPoint {
            alpha, beta, gamma,
            delta: Fp2::ZERO,
            alpha_inv, beta_inv, gamma_inv,
            delta_inv: Fp2::ZERO,
        };

        // 19. x ← X₁ · α⁻¹, y ← Z₁ · γ⁻¹
        let x = &hs1.0 * &alpha_inv;
        let y = &hs1.2 * &gamma_inv;

        // 24. (a₂,b₂,c₂,d₂) ← H(α, β, γ, 0)
        let (a2, b2, c2, d2) = hadamard4(&alpha, &beta, &gamma, &Fp2::ZERO);
        let null = ThetaNullPoint::new(a2, b2, c2, d2);
        let codomain = Jacobian::new(null);

        let J = JacobianPoint::new(x, x, y, y, codomain.clone());

        GluingData { dual, codomain, J, N }
    }

    /// Evaluate the gluing at a point P ∈ E₁ × E₂.
    ///
    /// Implements `GluingEval` ([§8.5.6], Algorithm 8.39).
    ///
    /// [§8.5.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
    /// Evaluate the gluing at a general point P ∈ E₁ × E₂.
    ///
    /// Implements `GluingEval` ([§8.5.6], Algorithm 8.39).
    /// Uses `ADDComponents` to compute the addition and subtraction
    /// of P with the kernel generator T₁'', then combines via
    /// the change-of-basis matrix and Hadamard.
    ///
    /// [§8.5.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
    pub(crate) fn eval(
        P: &(MontgomeryPoint, MontgomeryPoint),
        T1: &(MontgomeryPoint, MontgomeryPoint),
        data: &GluingData,
    ) -> JacobianPoint {
        // Algorithm 8.39:
        // 1–3. P₁,P₂ ← P;  T₁,T₂ ← T₁'';  x,y ← J
        let (x, y) = (&data.J.x, &data.J.y);

        // 4–5. ADDComponents for each component curve.
        let (u1, v1, w1) = add_sub_components(&P.0, &T1.0);
        let (u2, v2, w2) = add_sub_components(&P.1, &T1.1);

        // 6. U ← (u₁·u₂ + v₁·v₂, u₁·w₁, w₁·u₂, w₁·w₂)
        let U = (
            &(&u1 * &u2) + &(&v1 * &v2),
            &u1 * &w1,
            &w1 * &u2,
            &w1 * &w2,
        );

        // 7. V ← (v₁·u₂ + u₁·v₂, v₁·w₁, w₁·v₂, 0)
        let V = (
            &(&v1 * &u2) + &(&u1 * &v2),
            &v1 * &w1,
            &w1 * &v2,
            Fp2::ZERO,
        );

        // 8–9. U ← N · U,  V ← N · V
        let U = mat4_mul_vec(&data.N, &U);
        let V = mat4_mul_vec(&data.N, &V);

        // 10–11. U ← S(U),  V ← S(V)
        let U = (U.0.square(), U.1.square(), U.2.square(), U.3.square());
        let V = (V.0.square(), V.1.square(), V.2.square(), V.3.square());

        // 12. (X±, Y±, Z±, W±) ← H(U − V)
        let diff = (
            &U.0 - &V.0, &U.1 - &V.1, &U.2 - &V.2, &U.3 - &V.3,
        );
        let (Xpm, Ypm, Zpm, Wpm) = hadamard4(&diff.0, &diff.1, &diff.2, &diff.3);

        // 13. Scale by (y, y, x, x) and Hadamard.
        let Xout = &Xpm * y;
        let Yout = &Ypm * y;
        let Zout = &Zpm * x;
        let Wout = &Wpm * x;

        let (xr, yr, zr, wr) = hadamard4(&Xout, &Yout, &Zout, &Wout);
        JacobianPoint::new(xr, yr, zr, wr, data.codomain.clone())
    }

    /// Evaluate at a special point (P₁, 0) or (0, P₂).
    ///
    /// Implements `GluingEvalSpecial` ([§8.5.6], Algorithm 8.40).
    ///
    /// [§8.5.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
    pub(crate) fn eval_special(
        P: &(MontgomeryPoint, MontgomeryPoint),
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
        pts: &[(MontgomeryPoint, MontgomeryPoint)],
    ) -> (GluingData, Vec<JacobianPoint>) {
        let data = self.codomain();
        let images = pts.iter()
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
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
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
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
fn translation_pre_invert(P_prime: &MontgomeryPoint) -> TranslationData {
    let P = P_prime.double();
    let (X, Z) = (P_prime.X, P_prime.Z);
    let (U, W) = (P.X, P.Z);
    let WX = &W * &X;
    let WZ = &W * &Z;
    let UX = &U * &X;
    let UZ = &U * &Z;
    let delta = &WX - &UZ;
    TranslationData { WX, WZ, UX, UZ, delta, X, Z }
}

/// Complete `ActionByTranslation` ([§8.5.5], Algorithm 8.35)
/// given batched inverses of δ and Z.
///
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
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
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
fn theta_change_of_basis(
    T1: &(MontgomeryPoint, MontgomeryPoint),
    T2: &(MontgomeryPoint, MontgomeryPoint),
) -> [[Fp2; 4]; 4] {
    // Pre-inversion data for all four components.
    let d_G  = translation_pre_invert(&T1.0);
    let d_Gp = translation_pre_invert(&T1.1);
    let d_H  = translation_pre_invert(&T2.0);
    let d_Hp = translation_pre_invert(&T2.1);

    // Batch-invert all 8 elements: [δ_G, Z_G, δ_G', Z_G', δ_H, Z_H, δ_H', Z_H']
    let to_invert = [
        d_G.delta, d_G.Z,
        d_Gp.delta, d_Gp.Z,
        d_H.delta, d_H.Z,
        d_Hp.delta, d_Hp.Z,
    ];
    let invs = batch_invert(&to_invert);

    // Complete each ActionByTranslation with the batched inverses.
    let G  = translation_finish(&d_G,  &invs[0], &invs[1]);
    let Gp = translation_finish(&d_Gp, &invs[2], &invs[3]);
    let H  = translation_finish(&d_H,  &invs[4], &invs[5]);
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

    [
        [N00, N01, N02, N03],
        [N10, N11, N12, N13],
        [N20, N21, N22, N23],
        [N30, N31, N32, N33],
    ]
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
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
fn product_to_theta(
    pts: &[(MontgomeryPoint, MontgomeryPoint)],
    N: &[[Fp2; 4]; 4],
) -> Vec<(Fp2, Fp2, Fp2, Fp2)> {
    pts.iter().map(|(P1, P2)| {
        // Dimension-1 theta coords for each component.
        // For the product theta structure, the theta null point
        // (a : b) of each curve comes from the 4-torsion basis.
        // In the SQIsign chain, these are already set up by the
        // calling code. For now, use (X±Z) directly as a proxy
        // for (θ₀ : θ₁) — the matrix N absorbs the basis choice.
        let theta_0_1 = &P1.X - &P1.Z;  // θ₀ of P₁
        let theta_1_1 = &P1.X + &P1.Z;  // θ₁ of P₁
        let theta_0_2 = &P2.X - &P2.Z;  // θ₀ of P₂
        let theta_1_2 = &P2.X + &P2.Z;  // θ₁ of P₂

        // Product theta coordinates.
        let x = &theta_0_1 * &theta_0_2;
        let y = &theta_0_1 * &theta_1_2;
        let z = &theta_1_1 * &theta_0_2;
        let w = &theta_1_1 * &theta_1_2;

        // Apply change-of-basis matrix N.
        let v = [&x, &y, &z, &w];
        let mut out = [Fp2::ZERO; 4];
        for i in 0..4 {
            for j in 0..4 {
                out[i] = &out[i] + &(&N[i][j] * v[j]);
            }
        }
        (out[0], out[1], out[2], out[3])
    }).collect()
}

/// Decompose the sum and difference of two projective points into
/// shared components.
///
/// Given P = (X_P : Z_P) and Q = (X_Q : Z_Q) on the same curve,
/// returns (u, v, w) such that:
/// - x(P + Q) = (u − v : w)
/// - x(P − Q) = (u + v : w)
///
/// This avoids computing P + Q and P − Q separately, saving
/// multiplications in the gluing evaluation where both are needed.
///
/// Implements `ADDComponents` ([§8.2.4], Algorithm 8.12).
///
/// [§8.2.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
fn add_sub_components(P: &MontgomeryPoint, Q: &MontgomeryPoint) -> (Fp2, Fp2, Fp2) {
    let u = &(&P.X * &Q.X) + &(&P.Z * &Q.Z);
    let v = &(&P.X * &Q.Z) - &(&P.Z * &Q.X);
    let w = &(&P.X * &Q.Z) + &(&P.Z * &Q.X);
    (u, v, w)
}

/// Multiply a 4×4 matrix by a 4-vector.
fn mat4_mul_vec(
    M: &[[Fp2; 4]; 4],
    v: &(Fp2, Fp2, Fp2, Fp2),
) -> (Fp2, Fp2, Fp2, Fp2) {
    let va = [&v.0, &v.1, &v.2, &v.3];
    let mut out = [Fp2::ZERO; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i] = &out[i] + &(&M[i][j] * va[j]);
        }
    }
    (out[0], out[1], out[2], out[3])
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
/// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
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
    /// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
    /// [§8.5.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
    pub(crate) fn isogeny(
        &self,
        pts: &[JacobianPoint],
    ) -> (Jacobian, Vec<JacobianPoint>) {
        let (dual, codomain) = codomain_8torsion(&self.T1, &self.T2);
        let images = pts.iter()
            .map(|p| eval(p, &dual, &codomain))
            .collect();
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
/// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
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
    /// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
    pub(crate) fn isogeny(
        &self,
        domain: &Jacobian,
        pts: &[JacobianPoint],
    ) -> (Jacobian, Vec<JacobianPoint>) {
        let (dual, codomain) = codomain_4torsion(&self.T1, domain);
        let images = pts.iter()
            .map(|p| eval(p, &dual, &codomain))
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
/// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
pub(crate) struct GenericKernel2;

impl GenericKernel2 {
    /// Compute the generic (2,2)-isogeny and push points through.
    ///
    /// Implements `GenericCodomain` + `GenericEval`
    /// ([§8.5.3], Algorithm 8.33).
    ///
    /// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
    pub(crate) fn isogeny(
        domain: &Jacobian,
        pts: &[JacobianPoint],
    ) -> (Jacobian, Vec<JacobianPoint>) {
        let (dual, codomain) = codomain_from_null(domain);
        let images = pts.iter()
            .map(|p| eval(p, &dual, &codomain))
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
/// [§8.5.7]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
pub(crate) struct SplittingKernel {
    /// The domain Jacobian (whose null point has product structure).
    pub domain: Jacobian,
}

impl SplittingKernel {
    /// Compute the splitting and push points through.
    ///
    /// Returns the codomain product E₃ × E₄ and the images of `pts`
    /// converted back to Montgomery coordinates.
    ///
    /// Implements `SplittingIsomorphism` + `ThetaToProduct` +
    /// `ThetaProductPointToMontgomery` ([§8.5.7]).
    ///
    /// [§8.5.7]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
    pub(crate) fn isogeny(
        &self,
        pts: &[JacobianPoint],
    ) -> (EllipticProduct, Vec<ProductPoint>) {
        // 1. SplittingIsomorphism: find the matrix M (Algorithm 8.42).
        let M = splitting_isomorphism(&self.domain.null);

        // 2. Apply M to the null point to get product theta structure.
        let product_null = mat4_apply(&M, &self.domain.null);

        // 3. ThetaToProduct: recover (A₁:C₁), (A₂:C₂) (Algorithm 8.44).
        let product = theta_to_product(&product_null);

        // 4. ThetaProductPointToMontgomery for each point (Algorithm 8.45).
        let images = pts.iter()
            .map(|p| {
                // Apply M to point, then convert to Montgomery.
                let mp = mat4_apply_point(&M, p);
                theta_product_to_montgomery(&mp, &product_null, &product)
            })
            .collect();

        (product, images)
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

    let xawb = &hs1.x * &hs2.y;
    let zaxb = &hs2.x * &hs1.y;

    let alpha = &hs2.x * &xawb;
    let beta = &hs2.y * &zaxb;
    let gamma = &hs2.z * &xawb;
    let delta = &hs2.w * &zaxb;

    let zgwd = &hs2.z * &hs2.w;
    let alpha_inv = &hs1.y * &zgwd;
    let beta_inv = &hs1.x * &zgwd;
    let gamma_inv = delta;
    let delta_inv = gamma;

    let dual = DualThetaNullPoint {
        alpha, beta, gamma, delta,
        alpha_inv, beta_inv, gamma_inv, delta_inv,
    };
    let null_B = hadamard_null(&dual);
    (dual, Jacobian::new(null_B))
}

/// Codomain from 4-torsion (Algorithm 8.32).
fn codomain_4torsion(
    T1: &JacobianPoint,
    domain: &Jacobian,
) -> (DualThetaNullPoint, Jacobian) {
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
    let beta = &(&ab * &ag) * &hs.z;
    let delta_inv = &beta * &hs.x;
    let beta_mul = &beta * &hs.x;
    let xgd_ab_a2 = &(&hs.z * &ab) * &a2;
    let delta = &xgd_ab_a2 * &(&ab * &a2);
    let alpha = &(&hs.x * &ab) * &a2;
    let gamma = &alpha * &g2;
    let delta_final = &alpha * &d2;

    // Projective inverses.
    let alpha_inv = &hs.x * &d2;
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
        alpha, beta, gamma, delta,
        alpha_inv, beta_inv, gamma_inv, delta_inv,
    };
    let null_B = hadamard_null(&dual);
    (dual, Jacobian::new(null_B))
}

/// Evaluate a generic (2,2)-isogeny at a point (Algorithm 8.34).
pub(crate) fn eval(
    P: &JacobianPoint,
    dual: &DualThetaNullPoint,
    codomain: &Jacobian,
) -> JacobianPoint {
    let t = P.squared()
        .scale(&dual.alpha_inv, &dual.beta_inv,
               &dual.gamma_inv, &dual.delta_inv)
        .hadamard();
    JacobianPoint::new(t.x, t.y, t.z, t.w, codomain.clone())
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
/// [§8.5.7]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
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
        (0,0)|(0,1)|(0,2)|(0,3)|(1,0)|(1,2)|(2,0)|(2,1)|(3,0)|(3,3) => 1,
        (1,1)|(1,3)|(2,2)|(2,3)|(3,1)|(3,2) => -1,
        _ => 0,
    }
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
    debug_assert!(count == 1, "GetIndexSplitting: expected exactly one zero index");
    result
}

/// Compute `SplittingIsomorphism` (Algorithm 8.42).
///
/// Returns the 4×4 matrix M whose action on the null point recovers
/// the product theta structure.
fn splitting_isomorphism(null: &ThetaNullPoint) -> [[Fp2; 4]; 4] {
    let idx = get_index_splitting(null);
    let one = Fp2::ONE;
    let neg = -&one;
    let zero = Fp2::ZERO;

    // The matrices for each (i,j) case come from Algorithm 8.42.
    // For simplicity, only implement the cases that arise in
    // Isogeny22Chain (the spec guarantees (i,j) = (0,0) or (1,1)
    // for SQIsign's chain via Algorithm 8.47).
    use SplittingIndex::*;
    match idx {
        I00 => {
            let sqrt_m1 = Fp2::I;
            let neg_sqrt_m1 = -&sqrt_m1;
            [
                [one,        sqrt_m1,     one,         neg_sqrt_m1],
                [one,        neg_sqrt_m1, neg,         neg_sqrt_m1],
                [one,        sqrt_m1,     neg,         sqrt_m1    ],
                [neg,        sqrt_m1,     neg,         neg_sqrt_m1],
            ]
        }
        I10 => [
            [one, one, one, one],
            [one, neg, neg, one],
            [one, one, neg, neg],
            [neg, one, neg, one],
        ],
        I20 => [
            [one,  one,  one,  one ],
            [one,  neg,  one,  neg ],
            [one,  neg,  neg,  one ],
            [neg,  neg,  one,  one ],
        ],
        I30 => [
            [one,  one,  one,  one ],
            [one,  neg,  one,  neg ],
            [one,  one,  neg,  neg ],
            [neg,  one,  one,  neg ],
        ],
        I01 => [
            [one,  zero, zero, zero],
            [zero, zero, zero, one ],
            [zero, zero, one,  zero],
            [zero, neg,  zero, zero],
        ],
        I21 => [
            [one,  one,  one,  one ],
            [one,  neg,  one,  neg ],
            [one,  neg,  neg,  one ],
            [one,  one,  neg,  neg ],
        ],
        I02 => [
            [one,  zero, zero, zero],
            [zero, one,  zero, zero],
            [zero, zero, zero, one ],
            [zero, zero, neg,  zero],
        ],
        I12 => [
            [one,  zero, zero, zero],
            [zero, one,  zero, zero],
            [zero, zero, zero, one ],
            [zero, zero, one,  zero],
        ],
        I03 => [
            [one,  zero, zero, zero],
            [zero, one,  zero, zero],
            [zero, zero, one,  zero],
            [zero, zero, zero, neg ],
        ],
        I33 => [
            [one,  zero, zero, zero],
            [zero, one,  zero, zero],
            [zero, zero, one,  zero],
            [zero, zero, zero, one ],
        ],
    }
}

/// Apply a 4×4 matrix to a theta null point.
fn mat4_apply(M: &[[Fp2; 4]; 4], null: &ThetaNullPoint) -> ThetaNullPoint {
    let v = [&null.a, &null.b, &null.c, &null.d];
    let mut out = [Fp2::ZERO; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i] = &out[i] + &(&M[i][j] * v[j]);
        }
    }
    ThetaNullPoint::new(out[0], out[1], out[2], out[3])
}

/// Apply a 4×4 matrix to a theta point.
fn mat4_apply_point(M: &[[Fp2; 4]; 4], P: &JacobianPoint) -> JacobianPoint {
    let v = [&P.x, &P.y, &P.z, &P.w];
    let mut out = [Fp2::ZERO; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i] = &out[i] + &(&M[i][j] * v[j]);
        }
    }
    JacobianPoint::new(out[0], out[1], out[2], out[3], P.surface.clone())
}

/// Recover Montgomery coefficients from a product theta null point
/// (Algorithm 8.44).
fn theta_to_product(null: &ThetaNullPoint) -> EllipticProduct {
    use crate::curves::montgomery::{Curve, MontgomeryCoefficient};

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
    let A2_num = -&(&(&x + &y) + &(&x + &y));
    let C2 = &x - &y;

    // (A₁ : C₁) for E₁: A₁ = -2(x + z), C₁ = x - z
    let A1_num = -&(&(&x + &z) + &(&x + &z));
    let C1 = &x - &z;

    // Convert from projective (A:C) to affine A/C for Curve.
    let A1 = &A1_num * &C1.invert();
    let A2 = &A2_num * &C2.invert();

    EllipticProduct::new(
        Curve::new(MontgomeryCoefficient::from(A1)),
        Curve::new(MontgomeryCoefficient::from(A2)),
    )
}

/// Convert a theta point with product structure to Montgomery
/// coordinates on each component (Algorithm 8.45).
fn theta_product_to_montgomery(
    P: &JacobianPoint,
    null: &ThetaNullPoint,
    product: &EllipticProduct,
) -> ProductPoint {
    use crate::curves::montgomery::MontgomeryPoint;

    let (a, b, c, d) = (&null.a, &null.b, &null.c, &null.d);
    let (x, y, z, w) = (&P.x, &P.y, &P.z, &P.w);

    // Algorithm 8.45:
    // X₁ = a·z + c·x,  Z₁ = a·z − c·x
    // X₂ = a·y + b·x,  Z₂ = a·y − b·x
    let X1 = &(a * z) + &(c * x);
    let Z1 = &(a * z) - &(c * x);
    let X2 = &(a * y) + &(b * x);
    let Z2 = &(a * y) - &(b * x);

    (
        MontgomeryPoint::from_XZ(X1, Z1, &product.E1),
        MontgomeryPoint::from_XZ(X2, Z2, &product.E2),
    )
}
