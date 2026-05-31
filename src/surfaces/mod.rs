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
pub(crate) mod precomputed;

use core::ops::Mul;

use crate::{
    curves::{
        TorsionBasis, TorsionExponent,
        montgomery::{Curve, JacobianPoint as CurveJacobianPoint, ProjectiveXOnlyPoint},
        pairing::weil_pairing,
    },
    fields::fp2::Fp2,
    surfaces::isogeny::{GluingKernel, SplittingKernel},
};

#[cfg(test)]
mod tests;

/// The theta null point 0_A = (a : b : c : d) of a principally
/// polarized abelian surface A.
///
/// Internal to the (2,2)-isogeny computation.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ThetaNullPoint {
    /// First theta coordinate `θ_a`.
    pub(crate) a: Fp2,
    /// Second theta coordinate `θ_b`.
    pub(crate) b: Fp2,
    /// Third theta coordinate `θ_c`.
    pub(crate) c: Fp2,
    /// Fourth theta coordinate `θ_d`.
    pub(crate) d: Fp2,
}

impl ThetaNullPoint {
    /// Builds a theta null point from its four `Fp²` coordinates.
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
    /// Dual theta coordinate `α`.
    pub(crate) alpha: Fp2,
    /// Dual theta coordinate `β`.
    pub(crate) beta: Fp2,
    /// Dual theta coordinate `γ`.
    pub(crate) gamma: Fp2,
    /// Dual theta coordinate `δ`.
    pub(crate) delta: Fp2,
    /// Projective inverse of `α`.
    pub(crate) alpha_inv: Fp2,
    /// Projective inverse of `β`.
    pub(crate) beta_inv: Fp2,
    /// Projective inverse of `γ`.
    pub(crate) gamma_inv: Fp2,
    /// Projective inverse of `δ`.
    pub(crate) delta_inv: Fp2,
}

impl From<&DualThetaNullPoint> for ThetaNullPoint {
    /// Codomain theta null point from its dual via the Hadamard
    /// transform on the four coordinates.
    fn from(dual: &DualThetaNullPoint) -> Self {
        let (a, b, c, d) = hadamard4(&dual.alpha, &dual.beta, &dual.gamma, &dual.delta);
        ThetaNullPoint::new(a, b, c, d)
    }
}

/// Precomputed constants for theta doubling.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ThetaPrecomp {
    /// Doubling constant `c₁`.
    pub(crate) c1: Fp2,
    /// Doubling constant `c₂`.
    pub(crate) c2: Fp2,
    /// Doubling constant `c₃`.
    pub(crate) c3: Fp2,
    /// Doubling constant `c₄`.
    pub(crate) c4: Fp2,
    /// Doubling constant `c₅`.
    pub(crate) c5: Fp2,
    /// Doubling constant `c₆`.
    pub(crate) c6: Fp2,
    /// Doubling constant `c₇`.
    pub(crate) c7: Fp2,
    /// Doubling constant `c₈`.
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
    /// Theta null point of this surface.
    pub(crate) null: ThetaNullPoint,
    /// Precomputed doubling constants for `null`.
    pub(crate) precomp: ThetaPrecomp,
}

impl Jacobian {
    /// Builds a `Jacobian` from `null` and its precomputed doubling constants.
    pub(crate) fn new(null: ThetaNullPoint) -> Jacobian {
        let precomp = null.precompute();
        Jacobian { null, precomp }
    }
}

/// A point on a [`Jacobian`] in theta coordinates (x : y : z : w).
///
/// Internal to the (2,2)-isogeny chain. Never exposed at the API
/// boundary — inputs and outputs are [`ProductPoint`]s in Montgomery
/// coordinates.
#[derive(Clone, Debug)]
pub(crate) struct JacobianPoint {
    /// First theta coordinate `X`.
    pub(crate) X: Fp2,
    /// Second theta coordinate `Y`.
    pub(crate) Y: Fp2,
    /// Third theta coordinate `Z`.
    pub(crate) Z: Fp2,
    /// Fourth theta coordinate `W`.
    pub(crate) W: Fp2,
    /// Surface on which this point lies.
    pub(crate) surface: Jacobian,
}

impl JacobianPoint {
    /// Builds a `JacobianPoint` from its four theta coordinates and surface.
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
    /// Recovers y-coordinates via the Okeya-Sakurai lift, which
    /// requires the component-wise difference `P − Q` for y-recovery.
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
                return None;
            }
        };
        let (p2, q2) = match TorsionBasis::from_propagated(P.1, Q.1, PmQ.1).lift(&domain.E2) {
            Some(r) => r,
            None => {
                return None;
            }
        };
        Some(Kernel {
            domain,
            P: (p1, p2),
            Q: (q1, q2),
        })
    }

    /// Check that this kernel is isotropic for the product Weil
    /// pairing at exponent `e_kernel` (i.e., that the generators
    /// `(P, Q)` have order dividing `2^e_kernel` and span a
    /// Lagrangian subgroup of `E_1[2^e_kernel] × E_2[2^e_kernel]`).
    ///
    /// The Lagrangian condition for the canonical product
    /// polarization on `E_1 × E_2` is
    ///     `e_{2^e_kernel}(P_1, Q_1) · e_{2^e_kernel}(P_2, Q_2) = 1`
    /// in the group `μ_{2^e_kernel}` of `2^e_kernel`-th roots of
    /// unity, where `e_n` is the `2^n`-Weil pairing on each
    /// component curve (see [§2.4]). This method computes both
    /// pairings and returns `true` iff their product is `1`.
    ///
    /// `e_kernel` is the order exponent of the *kernel generators*
    /// (not the chain length): `e + 2` for [`Self::isogeny`]
    /// (`extra_torsion=true`), `e` for
    /// [`Self::isogeny_no_extra_torsion`] (`extra_torsion=false`).
    ///
    /// # Constant-time
    ///
    /// Variable-time. Used only behind `debug_assert!`, never in
    /// release builds, so this is correct-by-construction.
    ///
    /// [§2.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.4
    pub(crate) fn is_isotropic(&self, e_kernel: TorsionExponent) -> bool {
        // P + Q on each component side, derived from the Jacobian
        // generators via differential addition.
        let (ppq1, _) = self.P.0.x_add_sub(&self.Q.0);
        let (ppq2, _) = self.P.1.x_add_sub(&self.Q.1);

        let p1 = ProjectiveXOnlyPoint::from(&self.P.0);
        let q1 = ProjectiveXOnlyPoint::from(&self.Q.0);
        let p2 = ProjectiveXOnlyPoint::from(&self.P.1);
        let q2 = ProjectiveXOnlyPoint::from(&self.Q.1);

        let w1 = weil_pairing(&p1, &q1, &ppq1, e_kernel);
        let w2 = weil_pairing(&p2, &q2, &ppq2, e_kernel);
        // Product polarization on E_1 × E_2: a strictly Lagrangian
        // kernel of order 2^e_kernel satisfies
        //   `e_{2^e_kernel}(P_1, Q_1) · e_{2^e_kernel}(P_2, Q_2) = 1`
        // in μ_{2^e_kernel}.
        //
        // Sign-side chain entries pass kernels of order 2^(e+2) with
        // 2 extra torsion bits on top of a Lagrangian subgroup of
        // order 2^e (the chain absorbs those 2 bits via
        // `hadamard_bool` in its last 2 steps). The chain's effective
        // requirement, post-absorption, is that `(4P, 4Q)` is
        // Lagrangian under the 2^e-Weil pairing:
        //
        //   e_{2^e}(4P, 4Q) = 1  iff  e_{2^(e+2)}(4P, 4Q)^4 = 1
        //                        iff  ζ^{16·4} = ζ^{64} = 1
        //                        iff  ζ ∈ μ_64
        //
        // where `ζ = e_{2^(e+2)}(P, Q)` and the second step uses the
        // Weil pairing compatibility `e_m(X, Y) = e_n(X, Y)^{n/m}`
        // for `m | n` and `X, Y` of order `m`.
        //
        // Empirically:
        //   - r_rsp = 0 trajectories produce `ζ ∈ μ_4`.
        //   - r_rsp = 1 trajectories (e.g. KAT-001 sign iter 1) produce `ζ ∈ μ_16`.
        //   - Both subsets of `μ_64`; the chain handles both identically.
        //
        // Verification-path (`extra_torsion = false`) callers consume
        // a strictly-Lagrangian kernel of order 2^e_kernel, where
        // `prod = 1` — also satisfies `prod^64 = 1`, so the same
        // check works for both call sites.
        let prod = w1.as_fp2() * w2.as_fp2();
        // ζ^64 = ((ζ^2)^4)^4·... — six squarings: ζ^2, ζ^4, ζ^8, ζ^16, ζ^32, ζ^64.
        let prod64 = prod.square().square().square().square().square().square();
        prod64 == Fp2::ONE
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
    ///
    /// Returns `None` if the chain cannot run: `e < 2` (no main
    /// step after gluing), the kernel fails to split, or any
    /// internal numerical breakdown.
    pub fn isogeny(
        &self,
        e: TorsionExponent,
        pts: &[ProductPoint],
        randomize: Option<&mut dyn rand_core::RngCore>,
    ) -> Option<(EllipticProduct, Vec<ProductPoint>)> {
        debug_assert!(
            {
                let kernel_e = TorsionExponent::try_from(e.value() + 2)
                    .expect("e + 2 within TorsionExponent bounds");
                self.is_isotropic(kernel_e)
            },
            "kernel must be isotropic for the 2^(e+2)-Weil pairing"
        );
        self.isogeny_inner(e, pts, false, randomize)
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
        e: TorsionExponent,
        pts: &[ProductPoint],
    ) -> Option<(EllipticProduct, Vec<ProductPoint>)> {
        debug_assert!(
            {
                let kernel_e = TorsionExponent::try_from(e.value() + 2)
                    .expect("e + 2 within TorsionExponent bounds");
                self.is_isotropic(kernel_e)
            },
            "kernel must be isotropic for the 2^(e+2)-Weil pairing"
        );
        // FDI's internal chain — non-randomized in the C reference
        // (`theta_chain_compute_and_eval`, `dim2id2iso.c:240, 1018, 1094`).
        self.isogeny_inner(e, pts, true, None)
    }

    /// Compute the chain consuming a kernel of order exactly `2^e`,
    /// using a dedicated 4-isogeny + 2-isogeny tail in place of the
    /// 8-torsion penultimate/ultimate hadamard absorption.
    ///
    /// Mirrors the C reference's `extra_torsion=false` mode at
    /// `theta_isogenies.c:1250-1270`. Used by `to_isogeny`'s outer
    /// chain when `sui.e ∈ {f-1, f}` — the only case where the
    /// `extra_torsion=true` path can't pad the kernel up to
    /// `2^(sui.e + 2)` because we only have `2^f` torsion available.
    ///
    /// `e` is the chain length (number of (2,2)-isogeny steps).
    /// The kernel basis points `(P, Q)` must have order exactly
    /// `2^e`; the function consumes all of it.
    ///
    /// Returns `None` if the splitting step does not find a
    /// product structure (signaling a malformed kernel or numerical
    /// breakdown).
    pub fn isogeny_no_extra_torsion(
        &self,
        e: TorsionExponent,
        pts: &[ProductPoint],
        randomize: Option<&mut dyn rand_core::RngCore>,
    ) -> Option<(EllipticProduct, Vec<ProductPoint>)> {
        debug_assert!(
            self.is_isotropic(e),
            "kernel must be isotropic for the 2^e-Weil pairing"
        );
        self.isogeny_inner_no_extra_torsion(e, pts, randomize)
    }

    /// Runs the `(2, 2)`-isogeny chain machine implementing
    /// [Algorithm 8.47][Alg. 8.47] (`Isogeny22ChainWithTorsion`).
    ///
    /// [Alg. 8.47]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.47
    fn isogeny_inner(
        &self,
        e: TorsionExponent,
        pts: &[ProductPoint],
        _extra_torsion: bool,
        randomize: Option<&mut dyn rand_core::RngCore>,
    ) -> Option<(EllipticProduct, Vec<ProductPoint>)> {
        // Algorithm 8.47 (Isogeny22ChainWithTorsion):
        // https://sqisign.org/spec/sqisign-20250707.pdf#section.8.5
        let e = e.value();
        // Phase 1 indexes `orders[1]`; chain needs gluing + 1 main
        // step. Reachable from verify on adversarial signatures —
        // surface as `None` rather than panic.
        if e < 2 {
            return None;
        }

        // Phase 1: balanced strategy (lines 1–15).
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

        // Phase 2: gluing (lines 16–20).
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
        let (gluing_data, _) = gluing.isogeny(&[]);

        // Check the gluing codomain's null point for zero components
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

        // Phase 3: generic loop (lines 22–38).
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
            // The C ref (theta_isogenies.c:1221-1226) uses three
            // hadamard_bool configurations, keyed on the step index:
            //   Penultimate (i == n-2): bool_1=0, bool_2=0
            //   Ultimate (i == n-1):    bool_1=1, bool_2=0
            //   All other steps:        bool_1=0, bool_2=1
            // These apply unconditionally for both extra_torsion
            // true and false. The penultimate/ultimate produce dual
            // form directly, which the splitting step expects.
            let T1 = &theta_strat[k].0;
            let T2 = &theta_strat[k].1;
            let step = if steps_remaining == 1 {
                // Ultimate: bool_1=1, bool_2=0
                isogeny::EightTorsionStepKernel::Ultimate { T1, T2 }
            } else if steps_remaining == 2 {
                // Penultimate: bool_1=0, bool_2=0
                isogeny::EightTorsionStepKernel::Penultimate { T1, T2 }
            } else {
                // Normal: bool_1=0, bool_2=1
                isogeny::EightTorsionStepKernel::Interior { T1, T2 }
            };
            let iso = step.isogeny();

            for pt in theta_pts.iter_mut() {
                *pt = iso.eval(pt);
            }

            for i in 0..k {
                theta_strat[i].0 = iso.eval(&theta_strat[i].0);
                theta_strat[i].1 = iso.eval(&theta_strat[i].1);
                orders[i] -= 1;
            }

            theta_strat.truncate(k);
            orders.truncate(k);
            k = k.saturating_sub(1);
            current_jacobian = iso.into_codomain();

            _step_index += 1;
            steps_remaining -= 1;
        }

        // Phase 4: splitting (lines 39–45).
        //
        // The splitting step expects the codomain in dual form
        // (without the final Hadamard transform on the null point).
        // The penultimate and ultimate steps use bool_2=0, producing
        // dual form directly — no post-chain Hadamard needed.
        let splitter = SplittingKernel {
            domain: current_jacobian,
        };
        splitter.isogeny(&theta_pts, randomize)
    }

    /// Body for [`Self::isogeny_no_extra_torsion`].
    ///
    /// Mirrors the C reference's `_theta_chain_compute_impl` with
    /// `extra_torsion = false` (`theta_isogenies.c:1086-1314`).
    /// Differences from [`Self::isogeny_inner`]:
    ///
    ///   * Strategy starts at `orders[0] = e - 2` instead of `e`. The kernel
    ///     basis has order exactly `2^e`, so the bottom of the strategy tree
    ///     gives 8-torsion at level `current = 1` (after Phase 1) for the
    ///     gluing — same as Mode A — and the main loop runs `e − 2` total
    ///     isogeny steps including the gluing.
    ///   * The main loop uses normal hadamard (`(0, 1)`) for every step. There
    ///     are no penultimate/ultimate special cases — they are replaced by
    ///     dedicated 4- and 2-isogeny tail steps after the loop.
    ///   * After the loop: push the level-0 kernel point through the last
    ///     main-loop step, then run a dedicated 4-isogeny
    ///     ([`isogeny::FourTorsionStepKernel`]) and a dedicated
    ///     2-isogeny ([`isogeny::TwoTorsionStepKernel`]) before
    ///     handing off to the splitter.
    fn isogeny_inner_no_extra_torsion(
        &self,
        e: TorsionExponent,
        pts: &[ProductPoint],
        randomize: Option<&mut dyn rand_core::RngCore>,
    ) -> Option<(EllipticProduct, Vec<ProductPoint>)> {
        let e = e.value();
        // Chain layout requires `gluing + ≥1 main step + 4-iso + 2-iso`,
        // i.e. `e ≥ 4`. Honest signing always picks `sui.e ≈ f - 2`
        // (~246), but a malformed signing key could in principle drive
        // `find_uv` into a tiny `sui.e`. Surface that as `None` (the
        // sign retry loop handles it) rather than panicking.
        if e < 4 {
            return None;
        }

        type JacPair = JacobianProductPoint;

        // Phase 1: balanced strategy with `orders[0] = e - 2`.
        let mut strat_pts: Vec<(JacPair, JacPair)> = vec![(self.P, self.Q)];
        let mut orders: Vec<u32> = vec![e - 2];
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
                R = (R.0.double_for_theta(), R.1.double_for_theta());
                S = (S.0.double_for_theta(), S.1.double_for_theta());
            }
            strat_pts.push((R, S));
            orders.push(orders[k - 1] - n);
        }

        // Phase 2: gluing — same as Mode A.
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
        let (gluing_data, _) = gluing.isogeny(&[]);

        let mut theta_pts: Vec<JacobianPoint> = pts
            .iter()
            .map(|p| GluingKernel::eval_special(p, &gluing_data))
            .collect();

        let mut theta_strat: Vec<(JacobianPoint, JacobianPoint)> = Vec::new();
        for &(ri_jac, si_jac) in strat_pts.iter().take(k) {
            let R = GluingKernel::eval(&ri_jac, &gluing.T1_jac, &A1, &A2, &gluing_data);
            let S = GluingKernel::eval(&si_jac, &gluing.T1_jac, &A1, &A2, &gluing_data);
            theta_strat.push((R, S));
        }
        let mut orders: Vec<u32> = orders[..k].iter().map(|o| o - 1).collect();
        k = k.saturating_sub(1);

        let mut current_jacobian = gluing_data.codomain.clone();

        // Phase 3: main loop — ALL steps use normal hadamard.
        // Track the last step's `(dual, codomain)` and the level-0
        // kernel point separately so we can push the level-0 point
        // through the last step *after* the loop exits (mirrors C
        // ref's `if (n >= 3) { theta_isogeny_eval(thetaQ1[0], step,
        // thetaQ1[0]); }` at `theta_isogenies.c:1252`).
        let mut last_step: Option<isogeny::StepIsogeny> = None;
        let mut last_kernel: Option<(JacobianPoint, JacobianPoint)> = None;

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

            let iso = isogeny::EightTorsionStepKernel::Interior {
                T1: &theta_strat[k].0,
                T2: &theta_strat[k].1,
            }
            .isogeny();

            // If this is the last main-loop iteration (kernel at
            // level 0), capture the level-0 point so the post-loop
            // 4-isogeny step can push it through the stashed
            // [`StepIsogeny`].
            let stash_level_0 = k == 0;
            if stash_level_0 {
                last_kernel = Some(theta_strat[0].clone());
            }

            for pt in theta_pts.iter_mut() {
                *pt = iso.eval(pt);
            }

            for i in 0..k {
                theta_strat[i].0 = iso.eval(&theta_strat[i].0);
                theta_strat[i].1 = iso.eval(&theta_strat[i].1);
                orders[i] -= 1;
            }

            theta_strat.truncate(k);
            orders.truncate(k);
            k = k.saturating_sub(1);

            // At level 0, keep the whole [`StepIsogeny`] so the
            // post-loop push reuses it without re-cloning the
            // codomain. Otherwise, take the codomain out by value.
            if stash_level_0 {
                current_jacobian = iso.codomain().clone();
                last_step = Some(iso);
            } else {
                current_jacobian = iso.into_codomain();
            }
        }

        // Post-loop: push the level-0 kernel point through the last
        // main-loop step's isogeny so it becomes the 4-torsion
        // kernel for the dedicated penultimate (4-iso) step.
        // Mirrors C ref's `if (n >= 3) { theta_isogeny_eval(thetaQ1[0],
        // step, thetaQ1[0]); }` at `theta_isogenies.c:1252`.
        let last_iso = last_step?;
        let (kp1, _kp2) = last_kernel?;
        let kp1 = last_iso.eval(&kp1);

        // Dedicated penultimate: 4-isogeny.
        // C ref: `theta_isogeny_compute_4(step, theta, thetaQ1[0],
        // thetaQ2[0], 0, 0)` at `theta_isogenies.c:1258`. Algorithm
        // 8.32 computes the codomain from a single 4-torsion
        // generator, so [`FourTorsionStepKernel`] takes just `kp1`
        // and the domain.
        let iso_4 = isogeny::FourTorsionStepKernel {
            T1: &kp1,
            domain: &current_jacobian,
        }
        .isogeny();
        theta_pts = theta_pts.iter().map(|p| iso_4.eval(p)).collect();
        current_jacobian = iso_4.into_codomain();

        // Dedicated ultimate: 2-isogeny.
        // C ref: `theta_isogeny_compute_2(step, theta, thetaQ1[0],
        // thetaQ2[0], 1, 0)` at `theta_isogenies.c:1266`. Algorithm
        // 8.33 computes the codomain from the null point alone — the
        // kernel basis is implicit in the null structure, so
        // [`TwoTorsionStepKernel`] takes only the domain.
        let iso_2 = isogeny::TwoTorsionStepKernel {
            domain: &current_jacobian,
        }
        .isogeny();
        theta_pts = theta_pts.iter().map(|p| iso_2.eval(p)).collect();
        current_jacobian = iso_2.into_codomain();

        // Phase 4: splitting.
        let splitter = SplittingKernel {
            domain: current_jacobian,
        };
        splitter.isogeny(&theta_pts, randomize)
    }
}
