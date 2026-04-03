//! Isogenies between elliptic curves.
//!
//! An isogeny φ : E₁ → E₂ is defined by its kernel: a point P ∈ E₁
//! of order 2^e generates the cyclic subgroup ⟨P⟩, which uniquely
//! determines the isogeny.
//!
//! The [`Kernel`] type wraps a [`ProjectiveXOnlyPoint`] that generates
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

use crate::curves::{
    montgomery::{Curve, ProjectiveXOnlyPoint},
    scalar::Scalar,
    TorsionExponent,
};
use crate::fields::fp2::Fp2;

// ---------------------------------------------------------------------------
// IsogenyDegree
// ---------------------------------------------------------------------------

/// A positive odd integer representing the degree of a separable isogeny.
///
/// Stored as an unsigned 256-bit integer (`[u64; 4]`, little-endian).
/// For NIST-I, isogeny degrees are bounded by 2^{f−2} = 2^{246},
/// so 256 bits is always sufficient (see [Algorithm 3.15][Alg. 3.15]
/// and Kim et al., ePrint 2025/1649, Table 2).
///
/// Positive by construction (no sign bit). Always odd — enforced by
/// [`IsogenyDegree::new_odd`].
///
/// [Alg. 3.15]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.15
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IsogenyDegree([u64; 4]);

impl IsogenyDegree {
    /// Construct from a positive odd limb array, or `None` if zero or even.
    pub fn new_odd(limbs: [u64; 4]) -> Option<Self> {
        if limbs[0] & 1 == 0 {
            return None; // even or zero
        }
        Some(Self(limbs))
    }

    /// The underlying limbs (little-endian).
    #[inline]
    pub fn limbs(&self) -> &[u64; 4] {
        &self.0
    }

    /// Bit length (position of highest set bit).
    pub fn bit_length(&self) -> u32 {
        for i in (0..4).rev() {
            if self.0[i] != 0 {
                return (i as u32) * 64 + (64 - self.0[i].leading_zeros());
            }
        }
        0
    }

    /// Convert to a [`Scalar`] for elliptic curve point multiplication.
    #[inline]
    pub fn to_scalar(&self) -> Scalar {
        Scalar::from_limbs(self.0)
    }

    /// Widen to [`BigInt<8>`](crate::quaternions::bigint::BigInt) for
    /// quaternion arithmetic.
    pub fn to_bigint_wide(&self) -> crate::quaternions::bigint::BigInt<8> {
        crate::quaternions::bigint::BigInt::<8>::from_sign_and_limbs(
            0,
            [self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0],
        )
    }
}

impl From<IsogenyDegree> for Scalar {
    #[inline]
    fn from(d: IsogenyDegree) -> Scalar {
        Scalar::from_limbs(d.0)
    }
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// The kernel of a 2^e-isogeny, defined by a generator point P of
/// order 2^e on an elliptic curve.
///
/// The curve is carried by the point itself (as with all
/// [`ProjectiveXOnlyPoint`]s). Computing the isogeny and evaluating
/// points through it is done via [`Kernel::isogeny`] or
/// [`Kernel::isogeny_small`].
///
/// See [§2.3].
///
/// [§2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.3
#[derive(Copy, Clone, Debug)]
pub struct Kernel(ProjectiveXOnlyPoint);

impl Kernel {
    /// Construct from a generator point.
    pub fn new(generator: ProjectiveXOnlyPoint) -> Kernel {
        Kernel(generator)
    }

    /// The generator point.
    pub fn generator(&self) -> &ProjectiveXOnlyPoint {
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
        pts: &[ProjectiveXOnlyPoint],
    ) -> (Curve, Vec<ProjectiveXOnlyPoint>) {
        let e = e.value();
        let mut curve = *self.0.curve();
        let mut pts: Vec<ProjectiveXOnlyPoint> = pts.to_vec();

        let mut strat_pts: Vec<ProjectiveXOnlyPoint> = vec![self.0];
        let mut orders: Vec<u32> = vec![e];
        let mut k: usize = 0;

        // Chain of 4-isogenies.
        for _j in 0..(e / 2) as usize {
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

            // Note: singular 4-isogeny kernels ([2]K at x=0) can occur
            // legitimately during verification. The 4-isogeny formulas
            // handle this case correctly.

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
    /// Implements `TwoIsogenyChainSmall` ([§8.4], [Algorithm 8.26][Alg. 8.26]).
    ///
    /// # Singular kernels
    ///
    /// Intermediate kernel points may land on (0 : 1), producing a
    /// singular 2-isogeny. The spec says this "can include the special
    /// case of a kernel generator P = (0 : 1) only during signing"
    /// (page 65), but in practice singular kernels also arise during
    /// verification when the even response isogeny passes through
    /// (0, 0). We handle both cases uniformly.
    ///
    /// [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4
    /// [Alg. 8.26]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.26
    pub fn isogeny_small(
        &self,
        e: TorsionExponent,
        pts: &[ProjectiveXOnlyPoint],
        _is_signing: bool,
    ) -> Result<(Curve, Vec<ProjectiveXOnlyPoint>), &'static str> {
        let e = e.value();
        let mut curve = *self.0.curve();
        let mut P = self.0;
        let mut pts: Vec<ProjectiveXOnlyPoint> = pts.to_vec();

        for i in 0..e {
            // Compute the 2-torsion kernel for this step by
            // doubling P down to order 2.
            let mut K = P;
            for _ in 0..(e - i - 1) {
                K = K.double();
            }

            if i == 0 && !bool::from(K.double().is_identity()) {
                return Err("wrong point order");
            }

            if bool::from(K.X.ct_eq(&Fp2::ZERO)) {
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
    kernel: ProjectiveXOnlyPoint,
}

impl TwoIsogeny {
    pub(crate) fn from_kernel(P: &ProjectiveXOnlyPoint) -> TwoIsogeny {
        let xp_sq = P.X.square();
        let zp_sq = P.Z.square();
        let A24 = &zp_sq - &xp_sq;
        let C24 = zp_sq;
        TwoIsogeny {
            codomain: Curve::from_projective(A24, C24),
            kernel: *P,
        }
    }

    pub(crate) fn eval(&self, Q: &ProjectiveXOnlyPoint) -> ProjectiveXOnlyPoint {
        let t0 = &self.kernel.X + &self.kernel.Z;
        let t1 = &self.kernel.X - &self.kernel.Z;
        let t2 = &Q.X + &Q.Z;
        let t3 = &Q.X - &Q.Z;
        let t0 = &t0 * &t3;
        let t1 = &t1 * &t2;
        let t2 = &t0 + &t1;
        let t3 = &t0 - &t1;
        ProjectiveXOnlyPoint::from_XZ(&Q.X * &t2, &Q.Z * &t3, &self.codomain)
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
            c0,
            c1,
        }
    }

    pub(crate) fn eval(&self, Q: &ProjectiveXOnlyPoint) -> ProjectiveXOnlyPoint {
        let t0 = &Q.X * &Q.Z;
        let t1 = &Q.X + &(&self.c0 * &Q.Z);
        let t1 = &t1 * &Q.X;
        let XQ = &Q.Z.square() + &t1;
        let ZQ = &t0 * &self.c1;
        ProjectiveXOnlyPoint::from_XZ(XQ, ZQ, &self.codomain)
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
    pub(crate) fn from_kernel(P: &ProjectiveXOnlyPoint) -> FourIsogeny {
        let zp_sq = P.Z.square();
        let c1 = &P.X - &P.Z;
        let c2 = &P.X + &P.Z;
        let xp_sq = P.X.square();
        let t3 = &zp_sq + &xp_sq;
        let t4 = &zp_sq - &xp_sq;
        let A24 = &t3 * &t4;
        let C24 = zp_sq.square();
        let c0 = {
            let d = &zp_sq + &zp_sq;
            &d + &d
        };
        FourIsogeny {
            codomain: Curve::from_projective(A24, C24),
            c0,
            c1,
            c2,
        }
    }

    pub(crate) fn eval(&self, Q: &ProjectiveXOnlyPoint) -> ProjectiveXOnlyPoint {
        let t0 = &Q.X + &Q.Z;
        let t1 = &Q.X - &Q.Z;
        let xq = &t0 * &self.c1;
        let zq = &t1 * &self.c2;
        let t0 = &(&t0 * &t1) * &self.c0;
        let t1 = (&xq + &zq).square();
        let zq = (&xq - &zq).square();
        let xq = &(&t0 + &t1) * &t1;
        let zq = &zq * &(&t0 - &zq);
        ProjectiveXOnlyPoint::from_XZ(xq, zq, &self.codomain)
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
        let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::I, &curve);

        let (_, images) = Kernel::new(P).isogeny(TorsionExponent::new(1), &[P]);
        assert!(bool::from(images[0].is_identity()));
    }

    #[test]
    fn non_kernel_survives() {
        let curve = Curve::E0;
        let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::I, &curve);
        let Q = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(5)), &curve);

        let (_, images) = Kernel::new(P).isogeny(TorsionExponent::new(1), &[Q]);
        assert!(!bool::from(images[0].is_identity()));
    }

    #[test]
    fn codomain_has_valid_j_invariant() {
        let curve = Curve::E0;
        let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::I, &curve);

        let (codomain, _) = Kernel::new(P).isogeny(TorsionExponent::new(1), &[]);
        let _j = codomain.j_invariant();
    }

    #[test]
    fn isogeny_e1_matches_direct_two_isogeny() {
        let curve = Curve::E0;
        let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::I, &curve);
        let Q = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(5)), &curve);

        // Via Kernel::isogeny.
        let (_, chain_imgs) = Kernel::new(P).isogeny(TorsionExponent::new(1), &[Q]);

        // Via direct TwoIsogeny.
        let phi = TwoIsogeny::from_kernel(&P);
        let direct_Q = phi.eval(&Q);

        assert_eq!(chain_imgs[0], direct_Q);
    }
}
