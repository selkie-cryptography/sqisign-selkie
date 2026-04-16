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
//! # Divergences from spec / C reference
//!
//! - **Jacobian gluing.** The spec only specifies x-only Montgomery arithmetic,
//!   but gluing needs full (x,y,z) to distinguish P+Q from P−Q. We follow the C
//!   reference's nonstandard Jacobian doubling z₃ = 2y·z² so that `jac_to_xz:
//!   (x,y,z) ↦ (x, z²)` agrees with `product_to_theta`.
//! - **Squared theta.** `product_to_theta` uses X·Z products, not X² and Z²
//!   separately.
//!
//! [§2.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.4
//! [§8.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5

pub(crate) mod isogeny;

#[cfg(test)]
mod tests;

use core::ops::Mul;

use crate::{
    curves::{
        TorsionBasis,
        montgomery::{Curve, JacobianPoint as CurveJacobianPoint, ProjectiveXOnlyPoint},
    },
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
    /// [§8.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.2
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
/// [§8.5.5]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.5
/// [§8.5.7]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.7
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
/// [§8.5.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.1
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
    /// [§8.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.2
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
/// A pair of curve Jacobian points, one on each curve of a product.
pub type JacobianProductPoint = (CurveJacobianPoint, CurveJacobianPoint);

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
    pub fn from_montgomery(
        domain: EllipticProduct,
        P: ProductPoint,
        Q: ProductPoint,
        PmQ: ProductPoint,
    ) -> Option<Kernel> {
        let (p1, q1) = match TorsionBasis::from_propagated(P.0, Q.0, PmQ.0).lift(&domain.E1) {
            Some(r) => r,
            None => {
                eprintln!("    from_montgomery: lift E1 failed");
                return None;
            }
        };
        let (p2, q2) = match TorsionBasis::from_propagated(P.1, Q.1, PmQ.1).lift(&domain.E2) {
            Some(r) => r,
            None => {
                eprintln!("    from_montgomery: lift E2 failed");
                return None;
            }
        };
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
    /// [§2.2.5]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.2.5
    /// [§2.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.4
    /// [§3.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.3.2.3
    /// [§4.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.4
    /// [§4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
    /// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
    /// [§8.5.8]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.8
    /// Compute the chain with standard torsion (verification path).
    ///
    /// The kernel has order 2^(e+2). The last two isogeny steps
    /// use special hadamard_bool settings (penultimate/ultimate)
    /// to produce dual-form output for the splitting step.
    pub fn isogeny(
        &self,
        e: crate::curves::TorsionExponent,
        pts: &[ProductPoint],
    ) -> (EllipticProduct, Vec<ProductPoint>) {
        self.isogeny_inner(e, pts, false)
    }

    /// Compute the chain with extra torsion (signing path).
    ///
    /// The kernel has order 2^(e+2) where the extra 2 bits are
    /// HD_extra_torsion from the C ref. ALL isogeny steps use
    /// normal hadamard_bool (bool_1=0, bool_2=1), matching the
    /// C ref's behavior when `extra_torsion=true`.
    ///
    /// # Divergences
    ///
    /// The spec does not distinguish these two modes. The C ref
    /// (theta_isogenies.c) uses `extra_torsion` to control
    /// hadamard_bool settings in the chain's final steps.
    pub fn isogeny_extra_torsion(
        &self,
        e: crate::curves::TorsionExponent,
        pts: &[ProductPoint],
    ) -> (EllipticProduct, Vec<ProductPoint>) {
        self.isogeny_inner(e, pts, true)
    }

    fn isogeny_inner(
        &self,
        e: crate::curves::TorsionExponent,
        pts: &[ProductPoint],
        _extra_torsion: bool,
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
                // Use double_for_theta (standard Z'=2yz Jacobian)
                // to match the C ref's projective representative.
                R = (R.0.double_for_theta(), R.1.double_for_theta());
                S = (S.0.double_for_theta(), S.1.double_for_theta());
            }
            strat_pts.push((R, S));
            orders.push(orders[k - 1] - n);
        }

        // --- Phase 2: gluing (lines 16–20) ---
        //
        // Convert bottom strategy point from Jacobian to Montgomery
        // via jac_to_xz (the From impl). The codomain computation
        // uses these Montgomery points.
        let A1 = *self.domain.E1.coefficient();
        let A2 = *self.domain.E2.coefficient();

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

        // Check the gluing codomain's null point for zero components
        #[cfg(test)]
        {
            let null = &gluing_data.codomain.null;
            let comps = [
                ("a", &null.a),
                ("b", &null.b),
                ("c", &null.c),
                ("d", &null.d),
            ];
            let zeros: Vec<&str> = comps
                .iter()
                .filter(|(_, v)| **v == Fp2::ZERO)
                .map(|(n, _)| *n)
                .collect();
            if !zeros.is_empty() {
                eprintln!("GLUE CODOMAIN: zero components: {zeros:?}");
            }
            let pc = null.precompute();
            let pc_zeros: Vec<&str> = [
                ("c1", &pc.c1),
                ("c2", &pc.c2),
                ("c3", &pc.c3),
                ("c4", &pc.c4),
                ("c5", &pc.c5),
                ("c6", &pc.c6),
                ("c7", &pc.c7),
                ("c8", &pc.c8),
            ]
            .iter()
            .filter(|(_, v)| **v == Fp2::ZERO)
            .map(|(n, _)| *n)
            .collect();
            if !pc_zeros.is_empty() {
                eprintln!("GLUE PRECOMP: zero values: {pc_zeros:?}");
            }
        }

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
        // Push remaining strategy points through the gluing eval.
        // The C ref (theta_isogenies.c:1174-1178) pushes levels
        // 0..current-1 through the gluing, then decrements current.
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
                #[cfg(test)]
                {
                    let any_zero = |p: &JacobianPoint| {
                        p.X == Fp2::ZERO && p.Y == Fp2::ZERO && p.Z == Fp2::ZERO && p.W == Fp2::ZERO
                    };
                    if any_zero(&R) || any_zero(&S) {
                        eprintln!(
                            "CHAIN pushdown: after {n} doublings from level {}, R or S is ZERO",
                            k - 1
                        );
                    }
                    let hs = S.squared().hadamard();
                    if hs.X == hs.Z {
                        eprintln!(
                            "CHAIN pushdown: after {n} dbls from lvl {}, S has H(S²).X==H(S²).Z",
                            k - 1
                        );
                    }
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
            #[cfg(test)]
            {
                let t1 = &theta_strat[k].0;
                let t2 = &theta_strat[k].1;
                let any_zero = |p: &JacobianPoint| {
                    p.X == Fp2::ZERO && p.Y == Fp2::ZERO && p.Z == Fp2::ZERO && p.W == Fp2::ZERO
                };
                if any_zero(t1) || any_zero(t2) {
                    eprintln!("CHAIN step {_step_index}: 8-torsion input is ZERO (k={k})");
                }
                if _step_index == 0 {
                    let fp2_hex = |v: &Fp2| -> String {
                        let b = v.to_bytes();
                        let r: String =
                            b[..32].iter().rev().map(|x| format!("{:02x}", x)).collect();
                        format!("0x{r}")
                    };
                    eprintln!(
                        "STEP0 T2.X={} T2.Y={} T2.Z={} T2.W={}",
                        fp2_hex(&t2.X),
                        fp2_hex(&t2.Y),
                        fp2_hex(&t2.Z),
                        fp2_hex(&t2.W)
                    );
                    // Check if T2 has Z == 0 or W == 0 component
                    if t2.Z == Fp2::ZERO {
                        eprintln!("STEP0: T2.Z is ZERO!");
                    }
                    if t2.W == Fp2::ZERO {
                        eprintln!("STEP0: T2.W is ZERO!");
                    }
                    // Check for X == Z relationship (which causes alpha==gamma)
                    let hs = t2.squared().hadamard();
                    if hs.X == hs.Z {
                        eprintln!("STEP0: H(T2²).X == H(T2²).Z → will cause alpha==gamma");
                    }
                }
                // Check ALL strategy points before eval.
                for (si, sp) in theta_strat.iter().enumerate() {
                    if any_zero(&sp.0) {
                        eprintln!("CHAIN step {_step_index}: strat[{si}].0 is ZERO");
                    }
                    if any_zero(&sp.1) {
                        eprintln!("CHAIN step {_step_index}: strat[{si}].1 is ZERO");
                    }
                }
                // Check null point.
                let null = &current_jacobian.null;
                let null_zero = null.a == Fp2::ZERO && null.b == Fp2::ZERO;
                if null_zero {
                    eprintln!("CHAIN step {_step_index}: codomain null is ZERO");
                }
            }

            // The C ref (theta_isogenies.c:1221-1226) uses three
            // hadamard_bool configurations, keyed on the step index:
            //   Penultimate (i == n-2): bool_1=0, bool_2=0
            //   Ultimate (i == n-1):    bool_1=1, bool_2=0
            //   All other steps:        bool_1=0, bool_2=1
            // These apply unconditionally for both extra_torsion
            // true and false. The penultimate/ultimate produce dual
            // form directly, which the splitting step expects.
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

                #[cfg(test)]
                {
                    let any_zero = |p: &JacobianPoint| {
                        p.X == Fp2::ZERO && p.Y == Fp2::ZERO && p.Z == Fp2::ZERO && p.W == Fp2::ZERO
                    };
                    if any_zero(&theta_strat[i].0) || any_zero(&theta_strat[i].1) {
                        eprintln!("CHAIN step {_step_index}: strat[{i}] became ZERO AFTER eval");
                    }
                }
            }

            theta_strat.truncate(k);
            orders.truncate(k);
            k = k.saturating_sub(1);
            current_jacobian = new_jac;

            #[cfg(test)]
            {
                let null = &current_jacobian.null;
                let zero_components: Vec<&str> = [
                    ("a", &null.a),
                    ("b", &null.b),
                    ("c", &null.c),
                    ("d", &null.d),
                ]
                .iter()
                .filter(|(_, v)| **v == Fp2::ZERO)
                .map(|(n, _)| *n)
                .collect();
                if !zero_components.is_empty() {
                    eprintln!(
                        "CHAIN step {_step_index}: null has zero components: {zero_components:?}"
                    );
                }
            }

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
        // The splitting step expects the codomain in dual form
        // (without the final Hadamard transform on the null point).
        // The penultimate and ultimate steps use bool_2=0, producing
        // dual form directly — no post-chain Hadamard needed.
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
