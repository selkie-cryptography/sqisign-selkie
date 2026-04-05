//! Principally polarized abelian surfaces and (2,2)-isogenies between
//! them, computed using theta coordinates (of level 2).
//!
//! A principally polarized abelian surface is isomorphic to either:
//! 1. A product of elliptic curves E₁ × E₂ ([`EllipticProduct`])
//! 2. The Jacobian Jac(C) of a genus-2 hyperelliptic curve C ([`Jacobian`])
//!
//! At the API boundary, points on products are represented as pairs of
//! Montgomery [`ProjectiveXOnlyPoint`]s — one on each component curve. Theta
//! coordinates (of level 2) are used internally during the (2,2)-isogeny
//! chain but are not exposed.
//!
//! A (2,2)-isogeny chain Φ : E₁ × E₂ → E₃ × E₄ passes through:
//!
//! ```text
//! Montgomery pairs  ──gluing──▶  theta on Jacobians  ──splitting──▶  Montgomery pairs
//! (EllipticProduct)            (Jacobian)                           (EllipticProduct)
//! ```
//!
//! See [§2.4] and [§8.5] of the SQIsign spec.
//!
//! [§2.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.4
//! [§8.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5

pub(crate) mod isogeny;

use core::ops::Mul;

use crate::{
    curves::montgomery::{Curve, ProjectiveXOnlyPoint, lift_basis},
    fields::fp2::Fp2,
    surfaces::isogeny::{GluingKernel, SplittingKernel},
};

// ---------------------------------------------------------------------------
// Theta internals (pub(crate))
// ---------------------------------------------------------------------------

/// The theta null point 0_A = (a : b : c : d) of a principally
/// polarized abelian surface A.
///
/// Internal to the (2,2)-isogeny computation.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ThetaNullPoint {
    pub(crate) a: Fp2,
    pub(crate) b: Fp2,
    pub(crate) c: Fp2,
    pub(crate) d: Fp2,
}

impl ThetaNullPoint {
    pub(crate) fn new(a: Fp2, b: Fp2, c: Fp2, d: Fp2) -> ThetaNullPoint {
        ThetaNullPoint { a, b, c, d }
    }

    /// Precompute doubling constants.
    ///
    /// Implements `ThetaPrecomp` ([§8.5.2], Algorithm 8.28).
    ///
    /// [§8.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
    pub(crate) fn precompute(&self) -> ThetaPrecomp {
        let (a, b, c, d) = (&self.a, &self.b, &self.c, &self.d);
        let h = hadamard4(&a.square(), &b.square(), &c.square(), &d.square());
        let t1 = &h.0 * &h.1;
        let t2 = &h.2 * &h.3;
        let t1_lo = a * b;
        let t2_lo = c * d;
        ThetaPrecomp {
            c1: &t1_lo * c,
            c2: &t1_lo * d,
            c3: &t2_lo * a,
            c4: &t2_lo * b,
            c5: &t1 * &h.2,
            c6: &t1 * &h.3,
            c7: &t2 * &h.0,
            c8: &t2 * &h.1,
        }
    }
}

/// The dual isogenous theta null point and its projective inverse.
#[derive(Copy, Clone, Debug)]
pub(crate) struct DualThetaNullPoint {
    pub(crate) alpha: Fp2,
    pub(crate) beta: Fp2,
    pub(crate) gamma: Fp2,
    pub(crate) delta: Fp2,
    pub(crate) alpha_inv: Fp2,
    pub(crate) beta_inv: Fp2,
    pub(crate) gamma_inv: Fp2,
    pub(crate) delta_inv: Fp2,
}

/// Precomputed constants for theta doubling.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ThetaPrecomp {
    pub(crate) c1: Fp2,
    pub(crate) c2: Fp2,
    pub(crate) c3: Fp2,
    pub(crate) c4: Fp2,
    pub(crate) c5: Fp2,
    pub(crate) c6: Fp2,
    pub(crate) c7: Fp2,
    pub(crate) c8: Fp2,
}

/// Hadamard transform on four F_{p²} elements.
pub(crate) fn hadamard4(x: &Fp2, y: &Fp2, z: &Fp2, w: &Fp2) -> (Fp2, Fp2, Fp2, Fp2) {
    (
        &(&(x + y) + z) + w,
        &(&(x - y) + z) - w,
        &(&(x + y) - z) - w,
        &(&(x - y) - z) + w,
    )
}

/// A 4×4 matrix over F_{p²}, used for change-of-basis transformations
/// in theta coordinates (of level 2).
///
/// Used by `ThetaChangeOfBasis` ([§8.5.5], Algorithm 8.36),
/// `SplittingIsomorphism` ([§8.5.7], Algorithm 8.42), and
/// `ProductToTheta` ([§8.5.5], Algorithm 8.37).
///
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
/// [§8.5.7]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
#[derive(Copy, Clone, Debug)]
pub(crate) struct GluingMatrix(pub(crate) [[Fp2; 4]; 4]);

impl GluingMatrix {
    /// Apply this matrix to a theta null point.
    pub(crate) fn apply_null(&self, null: &ThetaNullPoint) -> ThetaNullPoint {
        let v = [&null.a, &null.b, &null.c, &null.d];
        let mut out = [Fp2::ZERO; 4];
        for (i, out_elem) in out.iter_mut().enumerate() {
            for (j, vj) in v.iter().enumerate() {
                *out_elem = &*out_elem + &(&self.0[i][j] * *vj);
            }
        }
        ThetaNullPoint::new(out[0], out[1], out[2], out[3])
    }

    /// Apply this matrix to a point in theta coordinates.
    pub(crate) fn apply_point(&self, P: &JacobianPoint) -> JacobianPoint {
        let v = [&P.X, &P.Y, &P.Z, &P.W];
        let mut out = [Fp2::ZERO; 4];
        for (i, out_elem) in out.iter_mut().enumerate() {
            for (j, vj) in v.iter().enumerate() {
                *out_elem = &*out_elem + &(&self.0[i][j] * *vj);
            }
        }
        JacobianPoint::new(out[0], out[1], out[2], out[3], P.surface.clone())
    }
}

/// Matrix-vector multiplication: M · v over F_{p²}⁴.
impl Mul<&(Fp2, Fp2, Fp2, Fp2)> for &GluingMatrix {
    type Output = (Fp2, Fp2, Fp2, Fp2);

    fn mul(self, v: &(Fp2, Fp2, Fp2, Fp2)) -> (Fp2, Fp2, Fp2, Fp2) {
        let va = [&v.0, &v.1, &v.2, &v.3];
        let mut out = [Fp2::ZERO; 4];
        for (i, out_elem) in out.iter_mut().enumerate() {
            for (j, vaj) in va.iter().enumerate() {
                *out_elem = &*out_elem + &(&self.0[i][j] * *vaj);
            }
        }
        (out[0], out[1], out[2], out[3])
    }
}

/// Matrix-matrix multiplication: self · rhs over F_{p²}⁴ˣ⁴.
impl Mul<&GluingMatrix> for &GluingMatrix {
    type Output = GluingMatrix;

    /// Matrix multiplication: self · rhs.
    fn mul(self, rhs: &GluingMatrix) -> GluingMatrix {
        let mut out = [[Fp2::ZERO; 4]; 4];
        for (i, out_row) in out.iter_mut().enumerate() {
            for (j, out_elem) in out_row.iter_mut().enumerate() {
                for k in 0..4 {
                    *out_elem = &*out_elem + &(&self.0[i][k] * &rhs.0[k][j]);
                }
            }
        }
        GluingMatrix(out)
    }
}

impl Mul<GluingMatrix> for GluingMatrix {
    type Output = GluingMatrix;
    fn mul(self, rhs: GluingMatrix) -> GluingMatrix {
        &self * &rhs
    }
}

// ---------------------------------------------------------------------------
// Surface types
// ---------------------------------------------------------------------------

/// A product of two Montgomery elliptic curves, E₁ × E₂.
///
/// Points on the product are pairs of Montgomery [`ProjectiveXOnlyPoint`]s.
///
/// See [§2.4].
///
/// [§2.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.4
#[derive(Copy, Clone, Debug)]
pub struct EllipticProduct {
    /// The first component curve E₁.
    pub E1: Curve,
    /// The second component curve E₂.
    pub E2: Curve,
}

impl EllipticProduct {
    /// Construct from two curves.
    pub fn new(E1: Curve, E2: Curve) -> EllipticProduct {
        EllipticProduct { E1, E2 }
    }
}

/// A pair of Montgomery points, one on each component of an
/// [`EllipticProduct`].
pub type ProductPoint = (ProjectiveXOnlyPoint, ProjectiveXOnlyPoint);

/// The Jacobian Jac(C) of a genus-2 hyperelliptic curve C.
///
/// Represented internally in theta coordinates (of level 2).
/// The intermediate objects in a (2,2)-isogeny chain.
///
/// See [§2.4] and [§8.5.1].
///
/// [§2.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.4
/// [§8.5.1]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
#[derive(Clone, Debug)]
pub(crate) struct Jacobian {
    pub(crate) null: ThetaNullPoint,
    pub(crate) precomp: ThetaPrecomp,
}

impl Jacobian {
    pub(crate) fn new(null: ThetaNullPoint) -> Jacobian {
        let precomp = null.precompute();
        Jacobian { null, precomp }
    }
}

// ---------------------------------------------------------------------------
// Points on Jacobians (internal only)
// ---------------------------------------------------------------------------

/// A point on a [`Jacobian`] in theta coordinates (x : y : z : w).
///
/// Internal to the (2,2)-isogeny chain. Never exposed at the API
/// boundary — inputs and outputs are [`ProductPoint`]s in Montgomery
/// coordinates.
#[derive(Clone, Debug)]
pub(crate) struct JacobianPoint {
    pub(crate) X: Fp2,
    pub(crate) Y: Fp2,
    pub(crate) Z: Fp2,
    pub(crate) W: Fp2,
    pub(crate) surface: Jacobian,
}

impl JacobianPoint {
    pub(crate) fn new(X: Fp2, Y: Fp2, Z: Fp2, W: Fp2, surface: Jacobian) -> JacobianPoint {
        JacobianPoint {
            X,
            Y,
            Z,
            W,
            surface,
        }
    }

    /// Hadamard transform.
    #[must_use]
    pub(crate) fn hadamard(&self) -> JacobianPoint {
        let (x, y, z, w) = hadamard4(&self.X, &self.Y, &self.Z, &self.W);
        JacobianPoint {
            X: x,
            Y: y,
            Z: z,
            W: w,
            surface: self.surface.clone(),
        }
    }

    /// Square each coordinate.
    #[must_use]
    pub(crate) fn squared(&self) -> JacobianPoint {
        JacobianPoint {
            X: self.X.square(),
            Y: self.Y.square(),
            Z: self.Z.square(),
            W: self.W.square(),
            surface: self.surface.clone(),
        }
    }

    /// Scale each coordinate.
    #[must_use]
    pub(crate) fn scale(&self, cx: &Fp2, cy: &Fp2, cz: &Fp2, cw: &Fp2) -> JacobianPoint {
        JacobianPoint {
            X: &self.X * cx,
            Y: &self.Y * cy,
            Z: &self.Z * cz,
            W: &self.W * cw,
            surface: self.surface.clone(),
        }
    }

    /// Compute \[2\]P in theta coordinates.
    ///
    /// Implements `ThetaDBL` ([§8.5.2], Algorithm 8.29).
    ///
    /// [§8.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
    #[must_use]
    pub(crate) fn double(&self) -> JacobianPoint {
        let c = &self.surface.precomp;
        let t = self.squared().hadamard().squared();
        let t = t.scale(&c.c8, &c.c7, &c.c6, &c.c5);
        let t = t.hadamard();
        t.scale(&c.c4, &c.c3, &c.c2, &c.c1)
    }
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// The kernel of a (2,2)-isogeny Φ : E₁ × E₂ → E₃ × E₄, defined by
/// two isotropic generators P, Q on E₁ × E₂.
///
/// Points are pairs of Montgomery [`ProjectiveXOnlyPoint`]s — one on each
/// component curve of the domain [`EllipticProduct`].
///
/// See [§2.4].
///
/// [§2.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.4
/// A pair of Jacobian points, one on each curve of a product.
pub type JacobianProductPoint = (
    crate::curves::montgomery::JacobianPoint,
    crate::curves::montgomery::JacobianPoint,
);

/// The kernel of a (2,2)-isogeny on E₁ × E₂, defined by two
/// isotropic generators in Jacobian coordinates.
///
/// Construct via [`from_montgomery`](Self::from_montgomery) (lifts
/// internally) or [`from_jacobian`](Self::from_jacobian) (pre-lifted).
pub struct Kernel {
    /// The domain product.
    pub domain: EllipticProduct,
    /// First kernel generator (P₁, P₂) ∈ E₁ × E₂ in Jacobian coords.
    pub P: JacobianProductPoint,
    /// Second kernel generator (Q₁, Q₂) ∈ E₁ × E₂ in Jacobian coords.
    pub Q: JacobianProductPoint,
}

impl Kernel {
    /// Construct from pre-lifted Jacobian points.
    ///
    /// Use this when the caller has already lifted from Montgomery
    /// to Jacobian (e.g., because they need to do Jacobian doubling
    /// for scaling before passing to the chain).
    pub fn from_jacobian(
        domain: EllipticProduct,
        P: JacobianProductPoint,
        Q: JacobianProductPoint,
    ) -> Kernel {
        Kernel { domain, P, Q }
    }

    /// Construct from Montgomery x-only points by lifting to Jacobian.
    ///
    /// Recovers y-coordinates via [`lift_basis`] (Okeya-Sakurai).
    /// Requires the component-wise difference P−Q for y-recovery.
    ///
    /// Returns `None` if the y-recovery fails (points not on curve).
    ///
    /// [`lift_basis`]: crate::curves::montgomery::lift_basis
    pub fn from_montgomery(
        domain: EllipticProduct,
        P: ProductPoint,
        Q: ProductPoint,
        PmQ: ProductPoint,
    ) -> Option<Kernel> {
        let (p1, q1) = lift_basis(&P.0, &Q.0, &PmQ.0, &domain.E1)?;
        let (p2, q2) = lift_basis(&P.1, &Q.1, &PmQ.1, &domain.E2)?;
        Some(Kernel {
            domain,
            P: (p1, p2),
            Q: (q1, q2),
        })
    }

    /// Compute the (2^e, 2^e)-isogeny defined by this kernel via a
    /// chain of (2,2)-isogenies, and push points through it.
    ///
    /// The kernel generators P, Q must have order 2^(e+2). The extra
    /// 2 bits of torsion allow all interior steps to use
    /// `GenericCodomainWith8Torsion` ([§8.5.3], Algorithm 8.30),
    /// avoiding square roots.
    ///
    /// Implements `Isogeny22ChainWithTorsion` ([§8.5.8], Algorithm 8.47).
    ///
    /// # NIST-I parameter set
    ///
    /// For NIST-I (p = 5·2²⁴⁸ − 1), v₂(p²−1) = 249, so
    /// 2^(e+2)-torsion is defined over F_{p²} for all e ≤ 247.
    /// This covers all call sites in SQIsign: verification
    /// ([§4.5], Algorithm 4.9), signing via `IdealToIsogeny`
    /// ([§3.2.3]), and `SplitAuxiliaryIsogeny` ([§4.4]).
    ///
    /// # Security
    ///
    /// The kernel generators must have order exactly 2^(e+2) and
    /// must be isotropic (the 2^(e+2)-Weil pairing
    /// e_{2^(e+2)}(P, Q) = 1; see [§2.2.5] and [§2.4]).
    /// These properties are assumed by construction.
    ///
    /// [§2.2.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
    /// [§2.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.4
    /// [§3.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.3.2
    /// [§4.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.4
    /// [§4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
    /// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
    /// [§8.5.8]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
    pub fn isogeny(
        &self,
        e: crate::curves::TorsionExponent,
        pts: &[ProductPoint],
    ) -> (EllipticProduct, Vec<ProductPoint>) {
        // Algorithm 8.47 (Isogeny22ChainWithTorsion):
        // https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
        let e = e.value();
        assert!(e >= 2, "chain requires e >= 2");

        // --- Phase 1: balanced strategy (lines 1–15) ---
        //
        // Kernel points arrive in Jacobian coordinates (either lifted
        // by from_montgomery or provided directly by from_jacobian).
        // All strategy doublings are in Jacobian to preserve the
        // correct projective representative (x, z²) for the gluing.
        type JacPair = JacobianProductPoint;

        let mut strat_pts: Vec<(JacPair, JacPair)> = vec![(self.P, self.Q)];
        let mut orders: Vec<u32> = vec![e];
        let mut k: usize = 0;

        while orders[k] != 1 {
            k += 1;
            let n = if orders[k - 1] >= 16 {
                orders[k - 1] / 2
            } else {
                orders[k - 1] - 1
            };
            let (mut R, mut S) = strat_pts[k - 1];
            for _ in 0..n {
                R = (R.0.double(), R.1.double());
                S = (S.0.double(), S.1.double());
            }
            strat_pts.push((R, S));
            orders.push(orders[k - 1] - n);
        }

        // --- Phase 2: gluing (lines 16–20) ---
        //
        // Convert bottom strategy point from Jacobian to Montgomery
        // via jac_to_xz (the From impl). The codomain computation
        // uses these Montgomery points.
        let A1 = *self.domain.E1.coefficient().as_fp2();
        let A2 = *self.domain.E2.coefficient().as_fp2();

        let gluing_T1_jac = strat_pts[k].0;
        let gluing_T2_jac = strat_pts[k].1;
        let gluing_T1_mont: ProductPoint = (
            ProjectiveXOnlyPoint::from(&gluing_T1_jac.0),
            ProjectiveXOnlyPoint::from(&gluing_T1_jac.1),
        );
        let gluing_T2_mont: ProductPoint = (
            ProjectiveXOnlyPoint::from(&gluing_T2_jac.0),
            ProjectiveXOnlyPoint::from(&gluing_T2_jac.1),
        );

        let gluing = GluingKernel {
            T1: gluing_T1_mont,
            T1_jac: gluing_T1_jac,
            T2: gluing_T2_mont,
            T2_jac: gluing_T2_jac,
        };
        #[cfg(test)]
        {
            let fp2_hex = |fp2val: &Fp2| {
                let bytes = fp2val.to_bytes();
                let re: String = bytes[..32]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                let im: String = bytes[32..]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                format!("0x{re} + i*0x{im}")
            };
            eprintln!("GLUE_IN T1.0.X={}", fp2_hex(&gluing.T1.0.X));
            eprintln!("GLUE_IN T1.0.Z={}", fp2_hex(&gluing.T1.0.Z));
            eprintln!("GLUE_IN T1.1.X={}", fp2_hex(&gluing.T1.1.X));
            eprintln!("GLUE_IN T1.1.Z={}", fp2_hex(&gluing.T1.1.Z));
        }

        let (gluing_data, _) = gluing.isogeny(&[]);

        #[cfg(test)]
        {
            let null = &gluing_data.codomain.null;
            let fp2_hex = |fp2val: &Fp2| {
                let bytes = fp2val.to_bytes();
                let re: String = bytes[..32]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                let im: String = bytes[32..]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                (re, im)
            };
            let (ar, ai) = fp2_hex(&null.a);
            let (br, bi) = fp2_hex(&null.b);
            eprintln!("GLUE null_a_re=0x{ar}");
            eprintln!("GLUE null_a_im=0x{ai}");
            eprintln!("GLUE null_b_re=0x{br}");
            eprintln!("GLUE null_b_im=0x{bi}");
        }

        // Push passenger points through the gluing.
        let mut theta_pts: Vec<JacobianPoint> = pts
            .iter()
            .map(|p| GluingKernel::eval_special(p, &gluing_data))
            .collect();

        // Push remaining strategy points through the gluing.
        //
        // GluingKernel::eval needs Jacobian points. For the strategy
        // points, we need to lift them too. However, we don't have
        // PmQ at each strategy level. Instead, we can compute PmQ
        // from the strategy points themselves.
        //
        // TODO: The proper fix is to maintain Jacobian points during
        // the balanced strategy (Phase 1), like the C ref does with
        // `double_couple_jac_point_iter`. For now, lift each strategy
        // point using the bottom-level PmQ (which is an approximation
        // that may be incorrect for intermediate strategy levels).
        // Push remaining strategy points through the gluing eval.
        // The strategy points are already in Jacobian — pass directly.
        let mut theta_strat: Vec<(JacobianPoint, JacobianPoint)> = Vec::new();
        for &(ri_jac, si_jac) in strat_pts.iter().take(k) {
            let R = GluingKernel::eval(&ri_jac, &gluing.T1_jac, &A1, &A2, &gluing_data);
            let S = GluingKernel::eval(&si_jac, &gluing.T1_jac, &A1, &A2, &gluing_data);
            theta_strat.push((R, S));
        }
        let mut orders: Vec<u32> = orders[..k].iter().map(|o| o - 1).collect();
        k = k.saturating_sub(1);

        let mut current_jacobian = gluing_data.codomain.clone();

        // --- Phase 3: generic loop (lines 22–38) ---
        //
        // The C reference (theta_isogenies.c:1158-1198) uses three
        // different hadamard_bool configurations across the chain:
        //
        //   Normal steps (i < n-2):   bool_1=0, bool_2=1
        //     Codomain: H(dual) (standard form)
        //     Eval:     H(precomp · H(P²))
        //
        //   Penultimate (i == n-2):   bool_1=0, bool_2=0
        //     Codomain: dual (NOT Hadamard-transformed)
        //     Eval:     precomp · H(P²)  (no outer Hadamard)
        //
        //   Ultimate (i == n-1):      bool_1=1, bool_2=0
        //     Codomain: dual (NOT Hadamard-transformed)
        //     Eval:     precomp · H(H(P)²)  (extra H on input)
        //
        // The splitting step expects the codomain in dual form (what
        // bool_2=0 produces). Using bool_2=1 for all steps—as we
        // originally did—feeds the splitting a Hadamard-transformed
        // null point, which has no zero U_{i,j}(0) entry.
        //
        // The total number of generic steps is e - 1 (the gluing
        // consumed one step from the chain of length e). The C ref
        // (theta_isogenies.c:1173-1178) uses `n - 2` and `n - 1`
        // to identify the penultimate and ultimate steps, where
        // n = e - 1 = number of generic steps.
        // The number of generic isogeny steps is e - 1 (the gluing
        // consumed the first step). The C ref uses step indices
        // 1..n where n = e - 1, with penultimate = n-2 and
        // ultimate = n-1 (0-indexed: n-3 and n-2).
        // Track how many generic steps remain. The total is e - 1
        // (the gluing consumed one step). We decrement each iteration
        // to identify the penultimate (remaining == 2) and ultimate
        // (remaining == 1) steps for hadamard_bool handling.
        // The total number of generic isogeny steps is e - 1 (the
        // gluing consumed the first step from the chain of degree 2^e).
        // Each loop iteration computes one (2,2)-isogeny from the
        // bottom of the strategy tree and pushes all other points through.
        let mut steps_remaining = (e as i32) - 1;

        let mut _step_index = 0u32;
        while !orders.is_empty() && (k > 0 || orders[0] != 0) {
            // Push down with ThetaDBL until order = 1.
            while orders[k] != 1 {
                k += 1;
                let n = orders[k - 1] / 2;
                let (mut R, mut S) = theta_strat[k - 1].clone();
                for _ in 0..n {
                    R = R.double();
                    S = S.double();
                }
                if k >= theta_strat.len() {
                    theta_strat.push((R, S));
                    orders.push(orders[k - 1] - n);
                }
            }

            // The C reference uses three different hadamard_bool
            // settings across the chain (theta_isogenies.c:1172-1178):
            //
            //   Normal (steps_remaining > 2):
            //     codomain: H(dual)            eval: H(precomp · H(P²))
            //   Penultimate (steps_remaining == 2):
            //     codomain: dual (no H)        eval: precomp · H(P²)
            //   Ultimate (steps_remaining == 1):
            //     codomain: dual (no H)        eval: precomp · H(H(P)²)
            //
            // The splitting step expects the codomain in dual form
            // (without the final Hadamard), which is what bool_2=0
            // produces.
            let (dual, new_jac) = if steps_remaining == 1 {
                // Ultimate: bool_1=1, bool_2=0
                isogeny::codomain_8torsion_ultimate(&theta_strat[k].0, &theta_strat[k].1)
            } else if steps_remaining == 2 {
                // Penultimate: bool_1=0, bool_2=0
                isogeny::codomain_8torsion_no_hadamard(&theta_strat[k].0, &theta_strat[k].1)
            } else {
                // Normal: bool_1=0, bool_2=1
                isogeny::codomain_8torsion(&theta_strat[k].0, &theta_strat[k].1)
            };

            // Evaluate: use matching bool_1/bool_2 for point evaluation.
            let eval_fn = |pt: &JacobianPoint| -> JacobianPoint {
                if steps_remaining == 1 {
                    isogeny::eval_ultimate(pt, &dual, &new_jac)
                } else if steps_remaining == 2 {
                    isogeny::eval_no_outer_hadamard(pt, &dual, &new_jac)
                } else {
                    isogeny::eval(pt, &dual, &new_jac)
                }
            };

            for pt in theta_pts.iter_mut() {
                *pt = eval_fn(pt);
            }

            for i in 0..k {
                theta_strat[i].0 = eval_fn(&theta_strat[i].0);
                theta_strat[i].1 = eval_fn(&theta_strat[i].1);
                orders[i] -= 1;
            }

            theta_strat.truncate(k);
            orders.truncate(k);
            k = k.saturating_sub(1);
            current_jacobian = new_jac;

            #[cfg(test)]
            if _step_index < 3 || steps_remaining <= 2 {
                eprintln!(
                    "step {_step_index}: k={k}, orders={orders:?}, remaining={steps_remaining}"
                );
            }

            _step_index += 1;
            steps_remaining -= 1;
        }

        // --- Phase 4: splitting (lines 39–45) ---
        //
        // The penultimate and ultimate steps used bool_2=0, so the
        // codomain is in dual form (not Hadamard-transformed). The
        // splitting step expects this form.
        #[cfg(test)]
        {
            let count = isogeny::get_index_splitting_count(&current_jacobian.null);
            eprintln!("splitting: zeros={count}");
            // Print null point in same hex format as C ref for comparison.
            let null = &current_jacobian.null;
            let fp2_hex = |fp2val: &Fp2| {
                let bytes = fp2val.to_bytes();
                // First 32 bytes = real, next 32 = imag (little-endian each)
                let re_hex: String = bytes[..32]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                let im_hex: String = bytes[32..]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                (re_hex, im_hex)
            };
            for (name, fp2val) in [
                ("a", &null.a),
                ("b", &null.b),
                ("c", &null.c),
                ("d", &null.d),
            ] {
                let (re, im) = fp2_hex(fp2val);
                eprintln!("SPLIT null_{name}_re=0x{re}");
                eprintln!("SPLIT null_{name}_im=0x{im}");
            }
        }

        let splitter = SplittingKernel {
            domain: current_jacobian,
        };
        splitter.isogeny(&theta_pts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curves::{
        TorsionExponent,
        montgomery::{Curve, ProjectiveXOnlyPoint},
    };

    /// Minimal (2,2)-chain test: e=2 on E₁ × E₂ where E₁ ≠ E₂.
    ///
    /// This exercises only the gluing + splitting (no generic steps).
    /// We compute E₁ as a 2-isogeny from E₀ to get a different curve.
    #[test]
    fn chain_e2_different_curves() {
        use crate::curves::isogeny::Kernel as CurveKernel;

        let e0 = Curve::E0;
        let p = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, &e0);
        let q = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_Q_X, &e0);

        // Compute E₁ via a 2-isogeny from E₀ with kernel [2^247]P.
        let mut k = p;
        for _ in 0..247 {
            k = k.double();
        }
        // k has order 2. Compute 2-isogeny and push P, Q through.
        let (e1, images) =
            CurveKernel::new(k).isogeny(TorsionExponent::try_from(1).unwrap(), &[p, q]);
        let p1 = images[0]; // P on E₁, order 2^247
        let q1 = images[1]; // Q on E₁, order 2^247

        // Scale both bases to order 2^4 = 16.
        // E₀ basis: double 244 times from order 2^248.
        let mut p0_4 = p;
        let mut q0_4 = q;
        for _ in 0..244 {
            p0_4 = p0_4.double();
            q0_4 = q0_4.double();
        }

        // E₁ basis: double 243 times from order 2^247.
        let mut p1_4 = p1;
        let mut q1_4 = q1;
        for _ in 0..243 {
            p1_4 = p1_4.double();
            q1_4 = q1_4.double();
        }

        // Kernel on E₀ × E₁: K₁ = (P₀₄, P₁₄), K₂ = (Q₀₄, Q₁₄).
        let product = EllipticProduct::new(e0, e1);
        let pmq0 = p0_4.projective_difference(&q0_4);
        let pmq1 = p1_4.projective_difference(&q1_4);
        let kernel = Kernel::from_montgomery(product, (p0_4, p1_4), (q0_4, q1_4), (pmq0, pmq1))
            .expect("kernel lift");

        let (codomain, _) = kernel.isogeny(TorsionExponent::try_from(2).unwrap(), &[]);

        eprintln!("chain_e2: E₀ j = {:?}", e0.j_invariant());
        eprintln!("chain_e2: E₁ j = {:?}", e1.j_invariant());
        eprintln!("chain_e2: codomain E1 j = {:?}", codomain.E1.j_invariant());
        eprintln!("chain_e2: codomain E2 j = {:?}", codomain.E2.j_invariant());
    }

    /// Test that the gluing codomain has zero U_{i,j} BEFORE our
    /// chain runs — i.e., compute the gluing codomain manually using
    /// the C reference's exact formula and check if it splits.
    ///
    /// This isolates whether the issue is in gluing codomain vs
    /// the splitting function itself.
    #[test]
    fn gluing_codomain_manual_check() {
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

        // The gluing kernel is T1 = (P4, Q4), T2 = (Q4, P4).
        // The 4-torsion points are T1' = [2]T1, T2' = [2]T2.
        let t1_0 = p4; // first component of T1 on E1
        let t1_1 = q4; // second component of T1 on E2
        let t2_0 = q4; // first component of T2 on E1
        let t2_1 = p4; // second component of T2 on E2

        // 2. N = ThetaChangeOfBasis(T1', T2')
        // 3. Apply N to T1, T2 via product_to_theta
        // 4. to_squared_theta on the results
        // 5. Cross-products for codomain
        // 6. Hadamard for final codomain

        // Lift to Jacobian for gluing eval.
        use crate::curves::montgomery::lift_basis;
        let pmq_0 = t1_0.projective_difference(&t2_0);
        let pmq_1 = t1_1.projective_difference(&t2_1);
        let (t1_jac_0, t2_jac_0) =
            lift_basis(&t1_0, &t2_0, &pmq_0, &curve).expect("lift E0 failed");
        let (t1_jac_1, t2_jac_1) =
            lift_basis(&t1_1, &t2_1, &pmq_1, &curve).expect("lift E0 failed");

        let gluing = GluingKernel {
            T1: (t1_0, t1_1),
            T1_jac: (t1_jac_0, t1_jac_1),
            T2: (t2_0, t2_1),
            T2_jac: (t2_jac_0, t2_jac_1),
        };
        let data = gluing.codomain();

        // The codomain null point is data.codomain.null.
        let null = &data.codomain.null;
        let count = isogeny::get_index_splitting_count(null);
        eprintln!("manual gluing: codomain zeros = {count}");
        eprintln!(
            "manual gluing: null = ({:?}, {:?}, {:?}, {:?})",
            null.a, null.b, null.c, null.d
        );

        // Also check: the codomain should have the structure of a
        // product of two elliptic curves. For a DIAGONAL kernel on
        // E₀ × E₀, the codomain should be E₀ × E₀ (the isogeny
        // is essentially the identity on each factor).
        // For an ANTI-DIAGONAL kernel, it should produce E₁ × E₂
        // where E₁ ≅ E₂ ≅ E₀/(⟨P₂⟩) (quotient by 2-torsion).

        // Check the DUAL form too (pre-Hadamard = alpha, beta, gamma, 0).
        let dual_null =
            ThetaNullPoint::new(data.dual.alpha, data.dual.beta, data.dual.gamma, Fp2::ZERO);
        let dual_count = isogeny::get_index_splitting_count(&dual_null);
        eprintln!("manual gluing: dual zeros = {dual_count}");
    }

    /// Test the gluing with REAL kernel data from KAT vector 0.
    ///
    /// Parse the KAT signature, run verification up to the chain,
    /// capture the kernel points, and test just the gluing (e=2).
    #[test]
    fn gluing_from_kat_data() {
        use crate::{
            curves::{BasisHint, TorsionExponent as TE, isogeny::Kernel as CurveKernel},
            keys::{SIGNATURE_BYTES, Signature, VerifyingKey},
        };

        let pk_hex = "07CCD21425136F6E865E497D2D4D208F0054AD81372066E817480787AAF7B2029550C89E892D618CE3230F23510BFBE68FCCDDAEA51DB1436B462ADFAF008A010B";
        let sm_hex = "84228651F271B0F39F2F19F2E8718F31ED3365AC9E5CB303AFE663D0CFC11F0455D891B0CA6C7E653F9BA2667730BB77BEFE1B1A31828404284AF8FD7BAACC010001D974B5CA671FF65708D8B462A5A84A1443EE9B5FED7218767C9D85CEED04DB0A69A2F6EC3BE835B3B2624B9A0DF68837AD00BCACC27D1EC806A44840267471D86EFF3447018ADB0A6551EE8322AB30010202D81C4D8D734FCBFBEADE3D3F8A039FAA2A2C9957E835AD55B22E75BF57BB556AC8";

        let pk_bytes = hex::decode(pk_hex).unwrap();
        let sm_bytes = hex::decode(sm_hex).unwrap();
        let sig_bytes: &[u8; SIGNATURE_BYTES] = sm_bytes[..SIGNATURE_BYTES].try_into().unwrap();

        let vk = VerifyingKey::from_bytes(pk_bytes.as_slice().try_into().unwrap()).unwrap();
        let sig = Signature::from_bytes(sig_bytes).unwrap();

        // Reproduce verification steps up to the (2,2)-chain.
        let f = crate::params::TORSION_EVEN_POWER;
        let e_rsp = crate::params::E_RSP;
        let e_rsp_prime = e_rsp - sig.n_bt.value() - sig.r_rsp.value();

        // Challenge isogeny.
        let basis_pk = crate::curves::TorsionBasis::from_hint(
            vk.curve(),
            BasisHint::from_byte(u8::from(vk.hint)),
        );
        let kernel_gen = basis_pk.scalar_mul_add(sig.chl.as_ref());
        let mut K_chl = kernel_gen;
        for _ in 0..sig.n_bt.value() {
            K_chl = K_chl.double();
        }
        let (curve_chl, _) =
            CurveKernel::new(K_chl).isogeny(TE::try_from(f - sig.n_bt.value()).unwrap(), &[]);

        // Bases.
        let basis_aux = crate::curves::TorsionBasis::from_hint(
            &sig.curve_aux,
            BasisHint::from_byte(u8::from(sig.hint_aux)),
        );
        let basis_chl = crate::curves::TorsionBasis::from_hint(
            &curve_chl,
            BasisHint::from_byte(u8::from(sig.hint_chl)),
        );

        // Scale.
        let mut P_aux = basis_aux.R;
        let mut Q_aux = basis_aux.S;
        for _ in 0..(f - e_rsp_prime - 2) {
            P_aux = P_aux.double();
            Q_aux = Q_aux.double();
        }
        let mut P_chl = basis_chl.R;
        let mut Q_chl = basis_chl.S;
        for _ in 0..(f - e_rsp_prime - sig.r_rsp.value() - 2) {
            P_chl = P_chl.double();
            Q_chl = Q_chl.double();
        }

        // Apply M_chl.
        let basis_chl_scaled: crate::curves::TorsionBasis = (P_chl, Q_chl).into();
        let basis_chl_transformed = &sig.M_chl * &basis_chl_scaled;
        let (mut P_chl, mut Q_chl) = (basis_chl_transformed.R, basis_chl_transformed.S);

        // Even response isogeny.
        if sig.r_rsp.value() > 0 {
            let kernel_pt = if sig.M_chl.first_column_even() {
                Q_chl
            } else {
                P_chl
            };
            let mut K = kernel_pt;
            for _ in 0..(e_rsp_prime + 2) {
                K = K.double();
            }
            let (new_curve, images) = CurveKernel::new(K)
                .isogeny_small(
                    TE::try_from(sig.r_rsp.value()).unwrap(),
                    &[P_chl, Q_chl],
                    false,
                )
                .expect("even response should succeed");
            let _curve_chl = new_curve;
            P_chl = images[0];
            Q_chl = images[1];
        }

        // NOW we have the actual kernel points for the (2,2)-chain.
        // P_chl, Q_chl on E_chl and P_aux, Q_aux on E_aux.
        // These are guaranteed correct by the C reference KAT.

        eprintln!("KAT chain: e_rsp_prime={e_rsp_prime}");
        eprintln!("KAT chain: P_chl.X = {:?}", P_chl.X);
        eprintln!("KAT chain: P_aux.X = {:?}", P_aux.X);

        // Test just the gluing (e=2) by scaling down to order 16.
        let scale = e_rsp_prime; // double this many more times to get order 4
        let mut p_chl_4 = P_chl;
        let mut q_chl_4 = Q_chl;
        let mut p_aux_4 = P_aux;
        let mut q_aux_4 = Q_aux;
        for _ in 0..scale {
            p_chl_4 = p_chl_4.double();
            q_chl_4 = q_chl_4.double();
            p_aux_4 = p_aux_4.double();
            q_aux_4 = q_aux_4.double();
        }

        // These should have order 4 (= 2^{e_rsp_prime+2 - scale} = 2^2 = 4).
        // For e=2 chain, we need order 2^{2+2} = 16, so scale less.
        // Actually for e=2: points need order 2^4 = 16.
        // We have order 2^{e_rsp_prime+2}. Scale by 2^{e_rsp_prime-2}.
        let mut p_chl_16 = P_chl;
        let mut q_chl_16 = Q_chl;
        let mut p_aux_16 = P_aux;
        let mut q_aux_16 = Q_aux;
        for _ in 0..(e_rsp_prime - 2) {
            p_chl_16 = p_chl_16.double();
            q_chl_16 = q_chl_16.double();
            p_aux_16 = p_aux_16.double();
            q_aux_16 = q_aux_16.double();
        }

        let product = EllipticProduct::new(curve_chl, sig.curve_aux);
        let pmq_chl = p_chl_16.projective_difference(&q_chl_16);
        let pmq_aux = p_aux_16.projective_difference(&q_aux_16);
        let kernel = Kernel::from_montgomery(
            product,
            (p_chl_16, p_aux_16),
            (q_chl_16, q_aux_16),
            (pmq_chl, pmq_aux),
        )
        .expect("kernel lift");

        // This should work if the gluing is correct.
        let (codomain, _) = kernel.isogeny(TE::try_from(2).unwrap(), &[]);
        eprintln!(
            "KAT e=2: j(E1)={:?}, j(E2)={:?}",
            codomain.E1.j_invariant(),
            codomain.E2.j_invariant()
        );
    }

    /// Test the splitting function with a synthetic product null point.
    ///
    /// A product theta null point for E₁ × E₂ with theta constants
    /// (a₁, b₁) and (a₂, b₂) is (a₁a₂, a₁b₂, b₁a₂, b₁b₂).
    /// The splitting should find exactly one zero U_{i,j}.
    #[test]
    fn splitting_synthetic_product() {
        use crate::fields::fp::Fp;

        // Use random-looking but fixed values as theta constants.
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
        eprintln!("synthetic product: zeros = {count}");
        assert_eq!(
            count, 1,
            "product null point should have exactly 1 zero U index"
        );
    }

    /// Chain test at e=3: gluing + 1 generic step + splitting.
    #[test]
    fn chain_e3_one_generic_step() {
        let curve = Curve::E0;
        let p = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, &curve);
        let q = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_Q_X, &curve);

        // Scale to order 2^5 = 32: double 248 - 5 = 243 times.
        let mut p5 = p;
        let mut q5 = q;
        for _ in 0..243 {
            p5 = p5.double();
            q5 = q5.double();
        }

        let product = EllipticProduct::new(curve, curve);
        let pmq1 = p5.projective_difference(&q5);
        let pmq2 = q5.projective_difference(&p5);
        let kernel = Kernel::from_montgomery(product, (p5, q5), (q5, p5), (pmq1, pmq2))
            .expect("kernel lift");

        let (codomain, _) = kernel.isogeny(TorsionExponent::try_from(3).unwrap(), &[]);

        eprintln!("chain_e3: codomain E1 j = {:?}", codomain.E1.j_invariant());
        eprintln!("chain_e3: codomain E2 j = {:?}", codomain.E2.j_invariant());
    }
}
