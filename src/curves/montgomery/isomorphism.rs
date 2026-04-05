//! Isomorphisms between Montgomery curves (§2.2.1.1, §8.2.2).

use super::{Curve, point::ProjectiveXOnlyPoint};
use crate::fields::fp2::Fp2;

/// An isomorphism between two Montgomery curves with the same j-invariant.
///
/// Implements [IsomorphismMontgomeryCurves][Alg. 8.9] from the spec.
/// Works entirely in projective coordinates — no field inversions.
///
/// Constructed via [`Curve::isomorphism`]. Precomputes projective
/// constants so that each point evaluation ([`eval`](Self::eval))
/// costs 7M + 2a.
///
/// [Alg. 8.9]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.9
#[derive(Copy, Clone, Debug)]
pub struct Isomorphism {
    /// λ_x = (2A'³ − 9A'C'²)(3C³ − A²C).
    pub(super) lambda_x: Fp2,
    /// λ_z = (2A³ − 9AC²)(3C'³ − A'²C').
    pub(super) lambda_z: Fp2,
    /// Precomputed: 3CC'.
    pub(super) three_cc_prime: Fp2,
    /// AC' (source A · target C).
    pub(super) ac_prime: Fp2,
    /// A'C (target A · source C).
    pub(super) a_prime_c: Fp2,
    pub(super) target: Curve,
}

impl Isomorphism {
    /// Apply this isomorphism to a projective x-only point.
    ///
    /// Implements lines 5–6 of [Algorithm 8.9][Alg. 8.9]:
    /// ```text
    /// X' ← λ_x(3X·CC' + AC'·Z) − λ_z·A'C·Z
    /// Z' ← 3λ_z·CC'·Z
    /// ```
    ///
    /// [Alg. 8.9]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.9
    #[must_use]
    pub fn eval(&self, p: &ProjectiveXOnlyPoint) -> ProjectiveXOnlyPoint {
        // X' = λ_x·(3CC'·X + AC'·Z) − λ_z·A'C·Z
        let term = &(&self.three_cc_prime * &p.X) + &(&self.ac_prime * &p.Z);
        let new_x = &(&self.lambda_x * &term) - &(&self.lambda_z * &(&self.a_prime_c * &p.Z));
        // Z' = 3·λ_z·CC'·Z
        let new_z = &self.lambda_z * &(&self.three_cc_prime * &p.Z);
        ProjectiveXOnlyPoint::from_XZ(new_x, new_z, &self.target)
    }
}
