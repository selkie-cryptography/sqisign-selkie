//! Principally polarized abelian surfaces and (2,2)-isogenies between
//! them, computed using theta coordinates (of level 2).
//!
//! A principally polarized abelian surface is isomorphic to either:
//! 1. A product of elliptic curves E₁ × E₂ ([`EllipticProduct`])
//! 2. The Jacobian Jac(C) of a genus-2 hyperelliptic curve C ([`Jacobian`])
//!
//! At the API boundary, points on products are represented as pairs of
//! Montgomery [`MontgomeryPoint`]s — one on each component curve. Theta
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

use crate::curves::montgomery::{Curve, MontgomeryPoint};
use crate::fields::fp2::Fp2;

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
            c1: &t1_lo * c, c2: &t1_lo * d,
            c3: &t2_lo * a, c4: &t2_lo * b,
            c5: &t1 * &h.2, c6: &t1 * &h.3,
            c7: &t2 * &h.0, c8: &t2 * &h.1,
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
    pub(crate) c1: Fp2, pub(crate) c2: Fp2,
    pub(crate) c3: Fp2, pub(crate) c4: Fp2,
    pub(crate) c5: Fp2, pub(crate) c6: Fp2,
    pub(crate) c7: Fp2, pub(crate) c8: Fp2,
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

// ---------------------------------------------------------------------------
// Surface types
// ---------------------------------------------------------------------------

/// A product of two Montgomery elliptic curves, E₁ × E₂.
///
/// Points on the product are pairs of Montgomery [`MontgomeryPoint`]s.
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
pub type ProductPoint = (MontgomeryPoint, MontgomeryPoint);

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
    pub(crate) x: Fp2,
    pub(crate) y: Fp2,
    pub(crate) z: Fp2,
    pub(crate) w: Fp2,
    pub(crate) surface: Jacobian,
}

impl JacobianPoint {
    pub(crate) fn new(x: Fp2, y: Fp2, z: Fp2, w: Fp2, surface: Jacobian) -> JacobianPoint {
        JacobianPoint { x, y, z, w, surface }
    }

    /// Hadamard transform.
    #[must_use]
    pub(crate) fn hadamard(&self) -> JacobianPoint {
        let (x, y, z, w) = hadamard4(&self.x, &self.y, &self.z, &self.w);
        JacobianPoint { x, y, z, w, surface: self.surface.clone() }
    }

    /// Square each coordinate.
    #[must_use]
    pub(crate) fn squared(&self) -> JacobianPoint {
        JacobianPoint {
            x: self.x.square(), y: self.y.square(),
            z: self.z.square(), w: self.w.square(),
            surface: self.surface.clone(),
        }
    }

    /// Scale each coordinate.
    #[must_use]
    pub(crate) fn scale(&self, cx: &Fp2, cy: &Fp2, cz: &Fp2, cw: &Fp2) -> JacobianPoint {
        JacobianPoint {
            x: &self.x * cx, y: &self.y * cy,
            z: &self.z * cz, w: &self.w * cw,
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
/// Points are pairs of Montgomery [`MontgomeryPoint`]s — one on each
/// component curve of the domain [`EllipticProduct`].
///
/// See [§2.4].
///
/// [§2.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.4
#[derive(Clone, Debug)]
pub struct Kernel {
    /// The domain product.
    pub domain: EllipticProduct,
    /// First kernel generator (P₁, P₂) ∈ E₁ × E₂.
    pub P: ProductPoint,
    /// Second kernel generator (Q₁, Q₂) ∈ E₁ × E₂.
    pub Q: ProductPoint,
}

impl Kernel {
    /// Construct from a domain product and two isotropic generators.
    pub fn new(domain: EllipticProduct, P: ProductPoint, Q: ProductPoint) -> Kernel {
        Kernel { domain, P, Q }
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
        use crate::surfaces::isogeny::{
            GluingKernel, GenericKernel8, SplittingKernel,
        };

        let e = e.value();
        assert!(e >= 2, "chain requires e >= 2");

        // --- Phase 1: balanced strategy (lines 1–15) ---
        // Points are Montgomery pairs on E₁ × E₂. Doubling uses
        // xDBL on each component curve.
        //
        // strat_pts[i] = ((R₁,R₂), (S₁,S₂)) at decreasing orders.
        let mut strat_pts: Vec<(ProductPoint, ProductPoint)> = vec![(self.P, self.Q)];
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
        // Bottom strategy point → GluingKernel → Φ₁ : E₁×E₂ → A₁
        let gluing = GluingKernel {
            T1: strat_pts[k].0,
            T2: strat_pts[k].1,
        };
        let (gluing_data, _) = gluing.isogeny(&[]);

        // Push passenger points through the gluing.
        let mut theta_pts: Vec<JacobianPoint> = pts.iter()
            .map(|p| GluingKernel::eval_special(p, &gluing_data))
            .collect();

        // Push remaining strategy points through the gluing.
        let mut theta_strat: Vec<(JacobianPoint, JacobianPoint)> = Vec::new();
        for i in 0..k {
            let R = GluingKernel::eval(&strat_pts[i].0, &strat_pts[i].0, &gluing_data);
            let S = GluingKernel::eval(&strat_pts[i].1, &strat_pts[i].1, &gluing_data);
            theta_strat.push((R, S));
        }
        let mut orders: Vec<u32> = orders[..k].iter().map(|o| o - 1).collect();
        k = k.saturating_sub(1);

        let mut current_jacobian = gluing_data.codomain.clone();

        // --- Phase 3: generic loop (lines 22–38) ---
        // All steps use GenericKernel8 (8-torsion, no square roots).
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

            // GenericKernel8 from bottom strat point.
            let kernel = GenericKernel8 {
                T1: theta_strat[k].0.clone(),
                T2: theta_strat[k].1.clone(),
            };
            let (new_jac, _) = kernel.isogeny(&[]);

            // Re-derive dual for manual eval of strat/passenger pts.
            let (dual, _) = isogeny::codomain_8torsion(
                &theta_strat[k].0, &theta_strat[k].1,
            );

            // Eval passenger points.
            for pt in theta_pts.iter_mut() {
                *pt = isogeny::eval(pt, &dual, &new_jac);
            }

            // Eval remaining strategy points.
            for i in 0..k {
                theta_strat[i].0 = isogeny::eval(&theta_strat[i].0, &dual, &new_jac);
                theta_strat[i].1 = isogeny::eval(&theta_strat[i].1, &dual, &new_jac);
                orders[i] -= 1;
            }

            theta_strat.truncate(k);
            orders.truncate(k);
            k = k.saturating_sub(1);
            current_jacobian = new_jac;
        }

        // --- Phase 4: splitting (lines 39–45) ---
        // Jacobian → EllipticProduct + Montgomery pairs.
        let splitter = SplittingKernel { domain: current_jacobian };
        splitter.isogeny(&theta_pts)
    }
}
