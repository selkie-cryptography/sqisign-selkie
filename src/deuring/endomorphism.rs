//! Endomorphism action on torsion bases.
//!
//! Bridges the quaternion algebra and elliptic curve worlds by
//! representing how quaternion order elements act on the torsion
//! basis E_t[2^f]. Each action is a 2×2 matrix over Z/2^f Z.
//!
//! See [§3.2.1.1] of the SQIsign specification.
//!
//! [§3.2.1.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.2.1.1

use crate::curves::montgomery::ProjectiveXOnlyPoint;
use crate::curves::scalar::Scalar;
use crate::curves::{TorsionBasis, TorsionExponent};
use crate::quaternions::bigint::BigInt;

/// A 2×2 matrix over Z/2^f Z representing the action of an
/// endomorphism on a torsion basis (P, Q) of E[2^f].
///
/// Given an endomorphism α and basis (P, Q), the matrix M_α
/// satisfies: α(P) = [M[0][0]]P + [M[1][0]]Q and
/// α(Q) = [M[0][1]]P + [M[1][1]]Q.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActionMatrix {
    /// Entries stored row-major: [[a, b], [c, d]].
    entries: [[BigInt<4>; 2]; 2],
}

impl ActionMatrix {
    /// The zero matrix.
    pub const ZERO: Self = Self {
        entries: [[BigInt::ZERO; 2]; 2],
    };

    /// The identity matrix.
    pub const IDENTITY: Self = Self {
        entries: [[BigInt::ONE, BigInt::ZERO], [BigInt::ZERO, BigInt::ONE]],
    };

    /// Creates a matrix from four entries (row-major).
    pub const fn new(a: BigInt<4>, b: BigInt<4>, c: BigInt<4>, d: BigInt<4>) -> Self {
        Self {
            entries: [[a, b], [c, d]],
        }
    }

    /// Creates a matrix from four little-endian limb arrays.
    pub const fn from_limbs(a: [u64; 4], b: [u64; 4], c: [u64; 4], d: [u64; 4]) -> Self {
        Self {
            entries: [
                [BigInt::from_limbs(a), BigInt::from_limbs(b)],
                [BigInt::from_limbs(c), BigInt::from_limbs(d)],
            ],
        }
    }

    /// Returns entry at (row, col).
    pub const fn entry(&self, row: usize, col: usize) -> &BigInt<4> {
        &self.entries[row][col]
    }

    /// Add scalar * other to self, mod 2^f: self += scalar * other.
    pub fn add_scaled_mod(&self, scalar: &BigInt<4>, other: &Self, f: u32) -> Self {
        let modulus = BigInt::<4>::ONE.shl(f);
        Self {
            entries: [
                [
                    self.entries[0][0]
                        .ct_add(&scalar.ct_mul(&other.entries[0][0]))
                        .ct_mod(&modulus),
                    self.entries[0][1]
                        .ct_add(&scalar.ct_mul(&other.entries[0][1]))
                        .ct_mod(&modulus),
                ],
                [
                    self.entries[1][0]
                        .ct_add(&scalar.ct_mul(&other.entries[1][0]))
                        .ct_mod(&modulus),
                    self.entries[1][1]
                        .ct_add(&scalar.ct_mul(&other.entries[1][1]))
                        .ct_mod(&modulus),
                ],
            ],
        }
    }

    /// Matrix-vector multiplication mod 2^f: M · [c1, c2]^T.
    ///
    /// Returns [M[0][0]*c1 + M[0][1]*c2, M[1][0]*c1 + M[1][1]*c2] mod 2^f.
    pub fn eval_mod(&self, c1: &BigInt<4>, c2: &BigInt<4>, f: u32) -> (BigInt<4>, BigInt<4>) {
        let modulus = BigInt::<4>::ONE.shl(f);
        let r0 = self.entries[0][0]
            .ct_mul(c1)
            .ct_add(&self.entries[0][1].ct_mul(c2))
            .ct_mod(&modulus);
        let r1 = self.entries[1][0]
            .ct_mul(c1)
            .ct_add(&self.entries[1][1].ct_mul(c2))
            .ct_mod(&modulus);
        (r0, r1)
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
        let modulus = BigInt::<4>::ONE.shl(f.value());

        let s00 = Scalar::from_limbs(*scalar.ct_mul(self.entry(0, 0)).ct_mod(&modulus).as_limbs());
        let s01 = Scalar::from_limbs(*scalar.ct_mul(self.entry(0, 1)).ct_mod(&modulus).as_limbs());
        let s10 = Scalar::from_limbs(*scalar.ct_mul(self.entry(1, 0)).ct_mod(&modulus).as_limbs());
        let s11 = Scalar::from_limbs(*scalar.ct_mul(self.entry(1, 1)).ct_mod(&modulus).as_limbs());

        let basis = TorsionBasis::new(p, q, p.projective_difference(&q));
        let p_prime = basis.eval_decomposition(&s00, &s10);
        let q_prime = basis.eval_decomposition(&s01, &s11);
        (p_prime, q_prime)
    }

    /// Matrix-matrix multiplication mod 2^f.
    pub fn mat_mul_mod(&self, rhs: &Self, f: u32) -> Self {
        let modulus = BigInt::<4>::ONE.shl(f);
        let mul_mod = |a: &BigInt<4>, b: &BigInt<4>| a.ct_mul(b).ct_mod(&modulus);
        let add_mod = |a: BigInt<4>, b: BigInt<4>| a.ct_add(&b).ct_mod(&modulus);

        Self {
            entries: [
                [
                    add_mod(
                        mul_mod(&self.entries[0][0], &rhs.entries[0][0]),
                        mul_mod(&self.entries[0][1], &rhs.entries[1][0]),
                    ),
                    add_mod(
                        mul_mod(&self.entries[0][0], &rhs.entries[0][1]),
                        mul_mod(&self.entries[0][1], &rhs.entries[1][1]),
                    ),
                ],
                [
                    add_mod(
                        mul_mod(&self.entries[1][0], &rhs.entries[0][0]),
                        mul_mod(&self.entries[1][1], &rhs.entries[1][0]),
                    ),
                    add_mod(
                        mul_mod(&self.entries[1][0], &rhs.entries[0][1]),
                        mul_mod(&self.entries[1][1], &rhs.entries[1][1]),
                    ),
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
            BigInt::from(1i64),
            BigInt::from(2i64),
            BigInt::from(3i64),
            BigInt::from(4i64),
        );
        let b = ActionMatrix::new(
            BigInt::from(5i64),
            BigInt::from(6i64),
            BigInt::from(7i64),
            BigInt::from(8i64),
        );
        let c = a.mat_mul_mod(&b, 248);
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(*c.entry(0, 0), BigInt::from(19i64));
        assert_eq!(*c.entry(0, 1), BigInt::from(22i64));
        assert_eq!(*c.entry(1, 0), BigInt::from(43i64));
        assert_eq!(*c.entry(1, 1), BigInt::from(50i64));
    }
}
