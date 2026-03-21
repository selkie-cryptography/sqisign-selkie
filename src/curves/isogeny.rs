//! Isogenies between elliptic curves.
//!
//! An isogeny φ : E₁ → E₂ is defined by its kernel: a point P ∈ E₁
//! of order 2^e generates the cyclic subgroup ⟨P⟩, which uniquely
//! determines the isogeny.
//!
//! The [`Kernel`] type wraps a [`MontgomeryPoint`] that generates
//! the kernel subgroup. Computing the isogeny and pushing points
//! through is done via [`Kernel::isogeny`].
//!
//! Internally, isogenies of degree 2^e are decomposed into chains
//! of 4-isogenies (with an optional trailing 2-isogeny if e is odd).
//!
//! See [§2.3] and [§8.4].
//!
//! [§2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.3
//! [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4

use subtle::ConstantTimeEq;

use crate::curves::TorsionExponent;
use crate::curves::montgomery::{Curve, MontgomeryPoint};
use crate::fields::fp2::Fp2;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// The kernel of a 2^e-isogeny, defined by a generator point P of
/// order 2^e on an elliptic curve.
///
/// The curve is carried by the point itself (as with all
/// [`MontgomeryPoint`]s). Computing the isogeny and evaluating
/// points through it is done via [`Kernel::isogeny`] or
/// [`Kernel::isogeny_small`].
///
/// See [§2.3].
///
/// [§2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.3
#[derive(Copy, Clone, Debug)]
pub struct Kernel(MontgomeryPoint);

impl Kernel {
    /// Construct from a generator point.
    pub fn new(generator: MontgomeryPoint) -> Kernel {
        Kernel(generator)
    }

    /// The generator point.
    pub fn generator(&self) -> &MontgomeryPoint {
        &self.0
    }

    /// Compute the 2^e-isogeny defined by this kernel and push
    /// points through it.
    ///
    /// Uses a balanced strategy: a chain of ⌊e/2⌋ 4-isogenies
    /// followed by an optional 2-isogeny if e is odd.
    ///
    /// Returns the codomain curve and the images of `pts`.
    ///
    /// Implements `TwoIsogenyChain` ([§8.4], Algorithm 8.25).
    ///
    /// # Security
    ///
    /// The generator must have order exactly 2^e. This is assumed
    /// by construction and not verified at runtime.
    ///
    /// [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4
    pub fn isogeny(
        &self,
        e: TorsionExponent,
        pts: &[MontgomeryPoint],
    ) -> (Curve, Vec<MontgomeryPoint>) {
        let e = e.value();
        let mut curve = *self.0.curve();
        let mut pts: Vec<MontgomeryPoint> = pts.to_vec();

        let mut strat_pts: Vec<MontgomeryPoint> = vec![self.0];
        let mut orders: Vec<u32> = vec![e];
        let mut k: usize = 0;

        // Chain of 4-isogenies.
        for j in 0..(e / 2) as usize {
            while orders[k] != 2 {
                k += 1;
                let n = (orders[k - 1] / 4) * 2 + (orders[k - 1] % 2);
                let new_order = orders[k - 1] - n;
                orders.push(new_order);
                let mut pt = strat_pts[k - 1];
                for _ in 0..n {
                    pt = pt.double();
                }
                strat_pts.push(pt);
            }

            if j == 0 {
                let K = strat_pts[k];
                debug_assert!(
                    !bool::from(K.double().X.ct_eq(&Fp2::ZERO)),
                    "unexpected singular 4-isogeny kernel"
                );
            }

            let phi = FourIsogeny::from_kernel(&strat_pts[k]);
            curve = phi.codomain;

            for i in 0..k {
                strat_pts[i] = phi.eval(&strat_pts[i]);
            }
            for i in 0..k {
                orders[i] -= 2;
            }
            orders.truncate(k);
            strat_pts.truncate(k);
            k = k.saturating_sub(1);

            for pt in pts.iter_mut() {
                *pt = phi.eval(pt);
            }
        }

        // Final 2-isogeny if e is odd.
        if e % 2 == 1 {
            if let Some(&K) = strat_pts.first() {
                debug_assert!(
                    !bool::from(K.X.ct_eq(&Fp2::ZERO)),
                    "unexpected singular 2-isogeny kernel"
                );
                let phi = TwoIsogeny::from_kernel(&K);
                curve = phi.codomain;
                for pt in pts.iter_mut() {
                    *pt = phi.eval(pt);
                }
            }
        }

        (curve, pts)
    }

    /// Compute a small 2^e-isogeny chain naively.
    ///
    /// For small exponents. Handles the singular kernel case P = (0 : 1)
    /// when `is_signing` is true.
    ///
    /// Implements `TwoIsogenyChainSmall` ([§8.4], Algorithm 8.26).
    ///
    /// # Timing
    ///
    /// This algorithm branches on whether each intermediate kernel point
    /// is singular (X = 0). This is safe because during verification,
    /// singular kernels are rejected (public outcome), and during signing,
    /// the signer already knows the kernel structure.
    ///
    /// [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4
    pub fn isogeny_small(
        &self,
        e: TorsionExponent,
        pts: &[MontgomeryPoint],
        is_signing: bool,
    ) -> Result<(Curve, Vec<MontgomeryPoint>), &'static str> {
        let e = e.value();
        let mut curve = *self.0.curve();
        let mut P = self.0;
        let mut pts: Vec<MontgomeryPoint> = pts.to_vec();

        for i in 0..e {
            let mut K = P;
            for _ in 0..(e - i - 1) {
                K = K.double();
            }
            P = P.double();

            if i == 0 && !bool::from(K.double().is_identity()) {
                return Err("wrong point order");
            }

            if bool::from(K.X.ct_eq(&Fp2::ZERO)) {
                if !is_signing {
                    return Err("unexpected singular 2-isogeny");
                }
                let phi = TwoIsogenySingular::from_curve(&curve);
                P = phi.eval(&P);
                curve = phi.codomain;
                for pt in pts.iter_mut() {
                    *pt = phi.eval(pt);
                }
            } else {
                let phi = TwoIsogeny::from_kernel(&K);
                P = phi.eval(&P);
                curve = phi.codomain;
                for pt in pts.iter_mut() {
                    *pt = phi.eval(pt);
                }
            }
        }

        Ok((curve, pts))
    }
}

// ---------------------------------------------------------------------------
// Individual isogeny steps (internal)
// ---------------------------------------------------------------------------

/// A non-singular 2-isogeny φ : E → E'.
///
/// See [§8.4], Algorithms 8.19–8.20.
///
/// [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4
#[derive(Copy, Clone, Debug)]
pub(crate) struct TwoIsogeny {
    pub(crate) codomain: Curve,
    kernel: MontgomeryPoint,
}

impl TwoIsogeny {
    pub(crate) fn from_kernel(P: &MontgomeryPoint) -> TwoIsogeny {
        let xp_sq = P.X.square();
        let zp_sq = P.Z.square();
        let A24 = &zp_sq - &xp_sq;
        let C24 = zp_sq;
        TwoIsogeny {
            codomain: Curve::from_projective(A24, C24),
            kernel: *P,
        }
    }

    pub(crate) fn eval(&self, Q: &MontgomeryPoint) -> MontgomeryPoint {
        let t0 = &self.kernel.X + &self.kernel.Z;
        let t1 = &self.kernel.X - &self.kernel.Z;
        let t2 = &Q.X + &Q.Z;
        let t3 = &Q.X - &Q.Z;
        let t0 = &t0 * &t3;
        let t1 = &t1 * &t2;
        let t2 = &t0 + &t1;
        let t3 = &t0 - &t1;
        MontgomeryPoint::from_XZ(&Q.X * &t2, &Q.Z * &t3, &self.codomain)
    }
}

/// A singular 2-isogeny with kernel at (0, 0).
///
/// See [§8.4], Algorithms 8.21–8.22.
///
/// [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4
#[derive(Copy, Clone, Debug)]
pub(crate) struct TwoIsogenySingular {
    pub(crate) codomain: Curve,
    c0: Fp2,
    c1: Fp2,
}

impl TwoIsogenySingular {
    pub(crate) fn from_curve(curve: &Curve) -> TwoIsogenySingular {
        let (A24, C24) = curve.projective_constants();
        let t0 = &(A24 + A24) - C24;
        let t0 = &t0 + &t0;
        let t1 = C24.invert();
        let t0 = &t0 * &t1;
        let c1 = t0;
        let A24_prime = &t0 + &t0;
        let t0 = t0.square();
        let four = Fp2::from_fp(crate::fields::fp::Fp::from_small(4));
        let t0 = &t0 - &four;
        let t0 = t0.sqrt();
        let c0 = -&t0;
        let C24_prime = &t0 + &t0;
        let A24_prime = &A24_prime + &C24_prime;
        let C24_prime = &C24_prime + &C24_prime;
        TwoIsogenySingular {
            codomain: Curve::from_projective(A24_prime, C24_prime),
            c0, c1,
        }
    }

    pub(crate) fn eval(&self, Q: &MontgomeryPoint) -> MontgomeryPoint {
        let t0 = &Q.X * &Q.Z;
        let t1 = &Q.X + &(&self.c0 * &Q.Z);
        let t1 = &t1 * &Q.X;
        let XQ = &Q.Z.square() + &t1;
        let ZQ = &t0 * &self.c1;
        MontgomeryPoint::from_XZ(XQ, ZQ, &self.codomain)
    }
}

/// A 4-isogeny φ : E → E'.
///
/// See [§8.4], Algorithms 8.23–8.24.
///
/// [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4
#[derive(Copy, Clone, Debug)]
pub(crate) struct FourIsogeny {
    pub(crate) codomain: Curve,
    c0: Fp2,
    c1: Fp2,
    c2: Fp2,
}

impl FourIsogeny {
    pub(crate) fn from_kernel(P: &MontgomeryPoint) -> FourIsogeny {
        let zp_sq = P.Z.square();
        let c1 = &P.X - &P.Z;
        let c2 = &P.X + &P.Z;
        let xp_sq = P.X.square();
        let t3 = &zp_sq + &xp_sq;
        let t4 = &zp_sq - &xp_sq;
        let A24 = &t3 * &t4;
        let C24 = zp_sq.square();
        let c0 = { let d = &zp_sq + &zp_sq; &d + &d };
        FourIsogeny {
            codomain: Curve::from_projective(A24, C24),
            c0, c1, c2,
        }
    }

    pub(crate) fn eval(&self, Q: &MontgomeryPoint) -> MontgomeryPoint {
        let t0 = &Q.X + &Q.Z;
        let t1 = &Q.X - &Q.Z;
        let xq = &t0 * &self.c1;
        let zq = &t1 * &self.c2;
        let t0 = &(&t0 * &t1) * &self.c0;
        let t1 = (&xq + &zq).square();
        let zq = (&xq - &zq).square();
        let xq = &(&t0 + &t1) * &t1;
        let zq = &zq * &(&t0 - &zq);
        MontgomeryPoint::from_XZ(xq, zq, &self.codomain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::fp::Fp;

    #[test]
    fn kernel_maps_to_identity() {
        let curve = Curve::E0;
        // (i, 0) is a 2-torsion point on E₀.
        let P = MontgomeryPoint::from_affine_x(Fp2::I, &curve);

        let (_, images) = Kernel::new(P).isogeny(
            TorsionExponent::new(1),
            &[P],
        );
        assert!(bool::from(images[0].is_identity()));
    }

    #[test]
    fn non_kernel_survives() {
        let curve = Curve::E0;
        let P = MontgomeryPoint::from_affine_x(Fp2::I, &curve);
        let Q = MontgomeryPoint::from_affine_x(
            Fp2::from_fp(Fp::from_small(5)),
            &curve,
        );

        let (_, images) = Kernel::new(P).isogeny(
            TorsionExponent::new(1),
            &[Q],
        );
        assert!(!bool::from(images[0].is_identity()));
    }

    #[test]
    fn codomain_has_valid_j_invariant() {
        let curve = Curve::E0;
        let P = MontgomeryPoint::from_affine_x(Fp2::I, &curve);

        let (codomain, _) = Kernel::new(P).isogeny(
            TorsionExponent::new(1),
            &[],
        );
        let _j = codomain.j_invariant();
    }

    #[test]
    fn isogeny_e1_matches_direct_two_isogeny() {
        let curve = Curve::E0;
        let P = MontgomeryPoint::from_affine_x(Fp2::I, &curve);
        let Q = MontgomeryPoint::from_affine_x(
            Fp2::from_fp(Fp::from_small(5)),
            &curve,
        );

        // Via Kernel::isogeny.
        let (_, chain_imgs) = Kernel::new(P).isogeny(
            TorsionExponent::new(1),
            &[Q],
        );

        // Via direct TwoIsogeny.
        let phi = TwoIsogeny::from_kernel(&P);
        let direct_Q = phi.eval(&Q);

        assert_eq!(chain_imgs[0], direct_Q);
    }
}
