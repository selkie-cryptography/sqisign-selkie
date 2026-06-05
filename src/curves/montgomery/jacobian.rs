//! Jacobian points on Montgomery curves (for the (2,2)-isogeny gluing step).

use core::ops::Neg;

use super::{Curve, point::ProjectiveXOnlyPoint};
use crate::fields::fp2::Fp2;

/// A point on a Montgomery curve in Jacobian coordinates (x, y, z).
///
/// Represents the affine point (x/z², y/z³) on E_A : y² = x³ + Ax² + x.
///
/// Jacobian coordinates are needed for the gluing step of the
/// (2,2)-isogeny chain, where the y-coordinate is required to
/// compute the cross-addition components (`jac_to_xz_add_components`
/// in the C reference). Montgomery x-only arithmetic is insufficient
/// because it cannot distinguish P+Q from P−Q.
///
/// **Naming:** This is a dim-1 elliptic curve point, NOT the dim-2
/// theta-coordinate `JacobianPoint` in [`crate::surfaces`]. The name
/// collision is unfortunate; we keep both because they serve different
/// layers (curves vs surfaces).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JacobianPoint {
    /// Jacobian `X` coordinate.
    pub(crate) X: Fp2,
    /// Jacobian `Y` coordinate.
    pub(crate) Y: Fp2,
    /// Jacobian `Z` coordinate.
    pub(crate) Z: Fp2,
    /// Curve on which this point lies.
    curve: Curve,
}

impl JacobianPoint {
    /// Creates from coordinates and a curve.
    pub fn new(X: Fp2, Y: Fp2, Z: Fp2, curve: &Curve) -> Self {
        Self {
            X,
            Y,
            Z,
            curve: *curve,
        }
    }

    /// The curve this point lives on.
    pub fn curve(&self) -> &Curve {
        &self.curve
    }

    /// Doubles this Jacobian point on y² = x³ + Ax² + x.
    ///
    /// **IMPORTANT:** This uses the C reference's unified add-or-double
    /// formula (`ec_add_jac_v2` in ec_jac.c:228-297), NOT standard
    /// Jacobian doubling. The C ref's formula produces z₃ = 2y·z²
    /// (instead of standard z₃ = 2y·z), giving a different projective
    /// representative after `jac_to_xz`. This matters because the
    /// gluing's `product_to_theta` is sensitive to the projective
    /// representative, not just the affine ratio.
    ///
    /// The SQIsign spec does NOT describe Jacobian doubling — only
    /// x-only Montgomery arithmetic (§8.2). These formulas come
    /// entirely from the C reference.
    ///
    /// The C ref computes for P = Q (doubling case):
    /// ```text
    /// dx = 2y₁           (tangent denominator)
    /// dy = z₁·M           (tangent numerator, M = 3x₁² + z₁²(2Ax₁ + z₁²))
    /// u₁ = x₁·z₁²         (= u₂ since P = Q)
    /// v₁ = y₁·z₁³
    /// t0 = z₁²            (= z₁·z₂ since P = Q)
    ///
    /// x₃ = dy² − dx²·(A·z₁⁴ + 2·x₁·z₁²)
    /// y₃ = dy·(u₁·dx² − x₃) − v₁·dx³
    /// z₃ = dx·z₁² = 2y₁·z₁²
    /// ```
    #[must_use]
    pub fn double(&self) -> JacobianPoint {
        let A = *self.curve.coefficient().as_fp2();

        let zz = self.Z.square(); // z₁²
        let zzzz = zz.square(); // z₁⁴
        let xx = self.X.square(); // x₁²

        // M = 3x₁² + z₁²·(2A·x₁ + z₁²)
        let two_a = &A + &A;
        let two_a_x = &two_a * &self.X;
        let inner = &two_a_x + &zz;
        let m_term = &inner * &zz; // z₁²·(2Ax₁ + z₁²)
        let three_xx = &(&xx + &xx) + &xx;
        let m = &three_xx + &m_term; // M = 3x₁² + z₁²(2Ax₁ + z₁²)

        // dx = 2y₁, dy = z₁·M
        let dx = &self.Y + &self.Y; // 2y₁
        let dy = &self.Z * &m; // z₁·M

        // Precomputations
        let dx_sq = dx.square(); // 4y₁²
        let dy_sq = dy.square(); // z₁²·M²
        let u1 = &self.X * &zz; // x₁·z₁²
        let v1 = &self.Y * &(&zz * &self.Z); // y₁·z₁³

        // x₃ = dy² − dx²·(A·z₁⁴ + u₁ + u₁)
        let x3 = {
            let a_zzzz = &A * &zzzz; // A·z₁⁴
            let sum = &(&a_zzzz + &u1) + &u1; // A·z₁⁴ + 2·x₁·z₁²
            &dy_sq - &(&dx_sq * &sum)
        };

        // y₃ = dy·(u₁·dx² − x₃) − v₁·dx³
        let y3 = {
            let u1_dx_sq = &u1 * &dx_sq;
            let dx_cubed = &dx_sq * &dx;
            &(&dy * &(&u1_dx_sq - &x3)) - &(&v1 * &dx_cubed)
        };

        // z₃ = dx·z₁² = 2y₁·z₁²
        //
        // NOTE: This is NOT the standard Jacobian Z' = 2yz. It
        // uses Z' = 2yz², which changes the projective
        // representative of jac_to_xz. The theta gluing accounts
        // for this difference via `double_for_theta` which uses
        // the standard formula.
        let z3 = &dx * &zz;

        JacobianPoint {
            X: x3,
            Y: y3,
            Z: z3,
            curve: self.curve,
        }
    }

    /// Jacobian doubling matching the C ref's `DBL` (ec_jac.c:110).
    ///
    /// Uses `Z' = 2yz` (standard Jacobian), unlike `double` which
    /// uses `Z' = 2yz²`. The theta change-of-basis requires the
    /// standard convention so that `jac_to_xz` produces the same
    /// projective representative as the C ref.
    ///
    /// TODO: reconcile with `double` — one doubling formula for
    /// all uses, matching the C ref's convention throughout.
    #[must_use]
    pub fn double_for_theta(&self) -> JacobianPoint {
        let A = *self.curve.coefficient().as_fp2();

        // Tangent slope numerator alpha = 3x² + z²(z² + 2Ax).
        let xx = self.X.square();
        let three_xx = &(&xx + &xx) + &xx;
        let zz = self.Z.square();
        let two_ax = &(&A * &self.X) + &(&A * &self.X);
        let alpha = &three_xx + &(&zz * &(&zz + &two_ax));

        // Output Z and supporting cross-terms.
        let z3 = &(&self.Y * &self.Z) + &(&self.Y * &self.Z); // 2yz
        let z3_sq = z3.square(); // 4y²z²
        let yy = self.Y.square();
        let two_yy = &yy + &yy;
        let four_xyy = &two_yy * &(&self.X + &self.X);

        // Output X and Y.
        let x3 = &(&alpha.square() - &(&A * &z3_sq)) - &(&four_xyy + &four_xyy);
        let four_yyyy = two_yy.square();
        let y3 = &(&alpha * &(&four_xyy - &x3)) - &(&four_yyyy + &four_yyyy);

        JacobianPoint {
            X: x3,
            Y: y3,
            Z: z3,
            curve: self.curve,
        }
    }

    /// Computes the x-only Montgomery projective coordinates of P + Q
    /// and P − Q from two Jacobian points.
    ///
    /// Returns `(x(P+Q), x(P-Q))` as `ProjectiveXOnlyPoint`s.
    ///
    /// Uses the full Jacobian addition formula (`ec_jac.c:305`) to
    /// deterministically distinguish P+Q from P−Q (impossible with
    /// x-only arithmetic alone).
    ///
    /// This is used by the change-of-basis matrix computation in
    /// [`crate::curves::pairing`] to compute the cross-pairing sum points.
    #[must_use]
    pub fn x_add_sub(&self, other: &Self) -> (ProjectiveXOnlyPoint, ProjectiveXOnlyPoint) {
        let a = *self.curve.coefficient().as_fp2();

        let t0 = self.Z.square(); // z1²
        let t1 = other.Z.square(); // z2²
        let t2 = &self.X * &t1; // x1·z2²
        let t3 = &t0 * &other.X; // z1²·x2
        let mut t4 = &self.Y * &other.Z; // y1·z2
        t4 = &t4 * &t1; // y1·z2³
        let mut t5 = &self.Z * &other.Y; // z1·y2
        t5 = &t5 * &t0; // z1³·y2
        let t0 = &t0 * &t1; // (z1·z2)²
        let t6 = &t4 * &t5; // (z1·z2)³·y1·y2
        let v = &t6 + &t6; // 2·(z1·z2)³·y1·y2

        // Fuse `t4² + t5²` into one Fp²-level sum-of-products (2 Mont
        // reductions vs 4 for two separate squares plus an add).  t4_sq
        // and t5_sq aren't used elsewhere.
        let sum_y2 = Fp2::sum_of_2_products(&t4, &t4, &t5, &t5);
        let sum_x = &t2 + &t3;
        let lambda = &t2 - &t3;
        let lambda_sq = lambda.square();
        let a_t0 = &a * &t0;
        let gamma = &(&sum_x + &a_t0) * &lambda_sq;

        let u = &sum_y2 - &gamma;
        let w = &lambda_sq * &t0;

        // x(P+Q) = (u + v) : w,  x(P-Q) = (u - v) : w
        let x_add = ProjectiveXOnlyPoint::from_XZ(&u + &v, w, &self.curve);
        let x_sub = ProjectiveXOnlyPoint::from_XZ(&u - &v, w, &self.curve);
        (x_add, x_sub)
    }
}

impl Neg for JacobianPoint {
    type Output = Self;
    /// −(x, y, z) = (x, −y, z).
    fn neg(self) -> Self {
        Self {
            X: self.X,
            Y: -&self.Y,
            Z: self.Z,
            curve: self.curve,
        }
    }
}

impl Neg for &JacobianPoint {
    type Output = JacobianPoint;
    fn neg(self) -> JacobianPoint {
        JacobianPoint {
            X: self.X,
            Y: -&self.Y,
            Z: self.Z,
            curve: self.curve,
        }
    }
}

/// Converts a Jacobian point to Montgomery projective (X:Z) = (x : z²).
///
/// This is the C reference's `jac_to_xz` (`ec_jac.c:34`). The
/// projective representative `(x, z²)` is NOT the same as `(X, Z)`
/// from Montgomery doubling — the balanced strategy must use Jacobian
/// doubling to produce the correct representative for the gluing's
/// `product_to_theta` computation.
impl From<JacobianPoint> for ProjectiveXOnlyPoint {
    fn from(jac: JacobianPoint) -> ProjectiveXOnlyPoint {
        let z_sq = jac.Z.square();
        ProjectiveXOnlyPoint::from_XZ(jac.X, z_sq, &jac.curve)
    }
}

impl From<&JacobianPoint> for ProjectiveXOnlyPoint {
    fn from(jac: &JacobianPoint) -> ProjectiveXOnlyPoint {
        let z_sq = jac.Z.square();
        ProjectiveXOnlyPoint::from_XZ(jac.X, z_sq, &jac.curve)
    }
}
