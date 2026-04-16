//! Endomorphism action on torsion bases.
//!
//! Bridges the quaternion algebra and elliptic curve worlds by
//! representing how quaternion order elements act on the torsion
//! basis E_t[2^f]. Each action is a 2×2 matrix over Z/2^f Z.
//!
//! See [§3.2.1.1] of the SQIsign specification.
//!
//! [§3.2.1.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.2.1.1

use crate::{
    curves::{TorsionBasis, TorsionExponent, montgomery::ProjectiveXOnlyPoint, scalar::Scalar},
    quaternions::bigint::BigInt,
};

/// A 2×2 matrix over Z/2^f Z representing the action of an
/// endomorphism on a torsion basis (P, Q) of E[2^f].
///
/// Given an endomorphism α and basis (P, Q), the matrix M_α
/// satisfies: α(P) = [M[0][0]]P + [M[1][0]]Q and
/// α(Q) = [M[0][1]]P + [M[1][1]]Q.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActionMatrix {
    /// Entries stored row-major: [[a, b], [c, d]].
    entries: [[Scalar; 2]; 2],
}

impl ActionMatrix {
    /// The zero matrix.
    pub const ZERO: Self = Self {
        entries: [[Scalar::ZERO; 2]; 2],
    };

    /// The identity matrix.
    pub const IDENTITY: Self = Self {
        entries: [[Scalar::ONE, Scalar::ZERO], [Scalar::ZERO, Scalar::ONE]],
    };

    /// Creates a matrix from four entries (row-major).
    pub const fn new(a: Scalar, b: Scalar, c: Scalar, d: Scalar) -> Self {
        Self {
            entries: [[a, b], [c, d]],
        }
    }

    /// Creates a matrix from four little-endian limb arrays.
    pub const fn from_limbs(a: [u64; 4], b: [u64; 4], c: [u64; 4], d: [u64; 4]) -> Self {
        Self {
            entries: [
                [Scalar::from_limbs(a), Scalar::from_limbs(b)],
                [Scalar::from_limbs(c), Scalar::from_limbs(d)],
            ],
        }
    }

    /// Returns entry at (row, col).
    pub const fn entry(&self, row: usize, col: usize) -> &Scalar {
        &self.entries[row][col]
    }

    /// Add scalar * other to self, mod 2^f: self += scalar * other.
    ///
    /// The `scalar` argument is a `BigInt<4>` because callers in the
    /// Deuring correspondence pass quaternion coordinates, which are
    /// `BigInt<4>`.
    pub fn add_scaled_mod(&self, scalar: &BigInt<4>, other: &Self, f: u32) -> Self {
        let s = Scalar::from(*scalar);
        Self {
            entries: [
                [
                    self.entries[0][0].add_mod2k(&s.mul_mod2k(&other.entries[0][0], f), f),
                    self.entries[0][1].add_mod2k(&s.mul_mod2k(&other.entries[0][1], f), f),
                ],
                [
                    self.entries[1][0].add_mod2k(&s.mul_mod2k(&other.entries[1][0], f), f),
                    self.entries[1][1].add_mod2k(&s.mul_mod2k(&other.entries[1][1], f), f),
                ],
            ],
        }
    }

    /// Matrix-vector multiplication mod 2^f: M · [c1, c2]^T.
    ///
    /// Returns [M[0][0]*c1 + M[0][1]*c2, M[1][0]*c1 + M[1][1]*c2] mod 2^f.
    ///
    /// The arguments are `BigInt<4>` because callers pass quaternion
    /// coordinates.
    pub fn eval_mod(&self, c1: &BigInt<4>, c2: &BigInt<4>, f: u32) -> (BigInt<4>, BigInt<4>) {
        let s1 = Scalar::from(*c1);
        let s2 = Scalar::from(*c2);
        let r0 = self.entries[0][0]
            .mul_mod2k(&s1, f)
            .add_mod2k(&self.entries[0][1].mul_mod2k(&s2, f), f);
        let r1 = self.entries[1][0]
            .mul_mod2k(&s1, f)
            .add_mod2k(&self.entries[1][1].mul_mod2k(&s2, f), f);
        (BigInt::from(r0), BigInt::from(r1))
    }

    /// Apply this matrix, scaled by a scalar, to a pair of points.
    ///
    /// Computes P' = [s·m₀₀]P + [s·m₁₀]Q and
    ///          Q' = [s·m₀₁]P + [s·m₁₁]Q, all mod 2^f.
    pub fn apply_scaled(
        &self,
        scalar: &BigInt<4>,
        p: ProjectiveXOnlyPoint,
        q: ProjectiveXOnlyPoint,
        f: TorsionExponent,
    ) -> (ProjectiveXOnlyPoint, ProjectiveXOnlyPoint) {
        let s = Scalar::from(*scalar);
        let fv = f.value();

        let s00 = s.mul_mod2k(self.entry(0, 0), fv);
        let s01 = s.mul_mod2k(self.entry(0, 1), fv);
        let s10 = s.mul_mod2k(self.entry(1, 0), fv);
        let s11 = s.mul_mod2k(self.entry(1, 1), fv);

        let basis = TorsionBasis::from((p, q));
        let p_prime = basis.eval_decomposition(&s00, &s10);
        let q_prime = basis.eval_decomposition(&s01, &s11);
        (p_prime, q_prime)
    }

    /// Apply this action matrix to a torsion basis, returning a new
    /// basis for the endomorphism image.
    ///
    /// Computes `θ(P) = [m00]P + [m10]Q` and `θ(Q) = [m01]P + [m11]Q`,
    /// then derives `θ(P) − θ(Q)` via `projective_difference` to
    /// ensure the Okeya-Sakurai y-recovery produces consistent
    /// Jacobian coordinates. The resulting basis is safe to pass to
    /// `TorsionBasis::lift`.
    #[must_use]
    pub fn apply_to_basis(&self, basis: &TorsionBasis, f: TorsionExponent) -> TorsionBasis {
        let p_prime = basis.biscalar_mul(self.entry(0, 0), self.entry(1, 0), f);
        let q_prime = basis.biscalar_mul(self.entry(0, 1), self.entry(1, 1), f);
        TorsionBasis::from((p_prime, q_prime))
    }

    /// Determinant mod 2^f: `ad − bc`.
    pub fn det_mod(&self, f: u32) -> Scalar {
        self.entries[0][0]
            .mul_mod2k(&self.entries[1][1], f)
            .sub_mod2k(
                &self.entries[0][1].mul_mod2k(&self.entries[1][0], f),
                f,
            )
    }

    /// Classical adjugate mod 2^f: `[[d, −b], [−c, a]]`.
    pub fn adjugate_mod(&self, f: u32) -> Self {
        let zero = Scalar::ZERO;
        Self {
            entries: [
                [
                    self.entries[1][1],
                    zero.sub_mod2k(&self.entries[0][1], f),
                ],
                [
                    zero.sub_mod2k(&self.entries[1][0], f),
                    self.entries[0][0],
                ],
            ],
        }
    }

    /// Matrix-matrix multiplication mod 2^f.
    pub fn mat_mul_mod(&self, rhs: &Self, f: u32) -> Self {
        Self {
            entries: [
                [
                    self.entries[0][0]
                        .mul_mod2k(&rhs.entries[0][0], f)
                        .add_mod2k(&self.entries[0][1].mul_mod2k(&rhs.entries[1][0], f), f),
                    self.entries[0][0]
                        .mul_mod2k(&rhs.entries[0][1], f)
                        .add_mod2k(&self.entries[0][1].mul_mod2k(&rhs.entries[1][1], f), f),
                ],
                [
                    self.entries[1][0]
                        .mul_mod2k(&rhs.entries[0][0], f)
                        .add_mod2k(&self.entries[1][1].mul_mod2k(&rhs.entries[1][0], f), f),
                    self.entries[1][0]
                        .mul_mod2k(&rhs.entries[0][1], f)
                        .add_mod2k(&self.entries[1][1].mul_mod2k(&rhs.entries[1][1], f), f),
                ],
            ],
        }
    }
}

/// Precomputed data for one extremal order's curve: the curve,
/// its torsion basis, and the action matrices for the order's
/// basis elements.
///
/// For each order O_t with basis (b_{t,1}, ..., b_{t,4}), the
/// matrices M_{t,u} represent the action of b_{t,u} on E_t[2^f]
/// with respect to the basis (P_t, Q_t).
///
/// See [§3.2.1.1] of the spec.
///
/// [§3.2.1.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.2.1.1
pub struct CurveEndomorphisms {
    /// Action matrices for the four basis elements of the order.
    /// M[u] represents the action of the u-th basis element.
    pub action: [ActionMatrix; 4],
    // TODO: Add curve (ec_curve_t) and torsion basis (P_t, Q_t)
    // once we define the bridge between quaternion and curve types.
    // These require Fp2 coordinates which live in the curves module.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_matrix_identity() {
        let id = ActionMatrix::IDENTITY;
        let c1 = BigInt::<4>::from(7i64);
        let c2 = BigInt::<4>::from(11i64);
        let (r0, r1) = id.eval_mod(&c1, &c2, 248);
        assert_eq!(r0, c1);
        assert_eq!(r1, c2);
    }

    #[test]
    fn action_matrix_mul() {
        let a = ActionMatrix::new(
            Scalar::from_u64(1),
            Scalar::from_u64(2),
            Scalar::from_u64(3),
            Scalar::from_u64(4),
        );
        let b = ActionMatrix::new(
            Scalar::from_u64(5),
            Scalar::from_u64(6),
            Scalar::from_u64(7),
            Scalar::from_u64(8),
        );
        let c = a.mat_mul_mod(&b, 248);
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(*c.entry(0, 0), Scalar::from_u64(19));
        assert_eq!(*c.entry(0, 1), Scalar::from_u64(22));
        assert_eq!(*c.entry(1, 0), Scalar::from_u64(43));
        assert_eq!(*c.entry(1, 1), Scalar::from_u64(50));
    }
}
