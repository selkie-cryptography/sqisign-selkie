//! 4×4 integer matrices over [`BigInt<N>`][super::super::bigint::BigInt].
//!
//! Used for quaternion lattice basis matrices, Gram matrices, and HNF
//! representations. The Hermite-Normal-Form constructors live in the
//! [`hnf`] submodule.

use core::{
    fmt,
    ops::{Index, IndexMut, Mul},
};

use super::{super::bigint::BigInt, Vector};

mod hnf;

/// A 4×4 matrix of [`BigInt<N>`] values, stored row-major.
///
/// Used for quaternion lattice basis matrices, Gram matrices,
/// and HNF representations.
///
/// Indexing: `mat[i]` returns row `i` as a [`Vector`].
#[derive(Clone)]
pub struct Matrix<const N: usize>([Vector<N>; 4]);

impl<const N: usize> Matrix<N> {
    /// The zero matrix.
    pub const ZERO: Self = Self([Vector::ZERO; 4]);

    /// The 4×4 identity matrix.
    pub const IDENTITY: Self = Self([
        Vector::new(BigInt::ONE, BigInt::ZERO, BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ONE, BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ONE, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ZERO, BigInt::ONE),
    ]);

    /// Creates a matrix from four row vectors.
    #[inline]
    pub const fn from_rows(r0: Vector<N>, r1: Vector<N>, r2: Vector<N>, r3: Vector<N>) -> Self {
        Self([r0, r1, r2, r3])
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let s = &self.0;
        Self([
            Vector::new(s[0][0], s[1][0], s[2][0], s[3][0]),
            Vector::new(s[0][1], s[1][1], s[2][1], s[3][1]),
            Vector::new(s[0][2], s[1][2], s[2][2], s[3][2]),
            Vector::new(s[0][3], s[1][3], s[2][3], s[3][3]),
        ])
    }

    /// Matrix-vector multiplication: `self * v`.
    pub fn eval(&self, v: &Vector<N>) -> Vector<N> {
        Vector::new(
            self.0[0].dot(v),
            self.0[1].dot(v),
            self.0[2].dot(v),
            self.0[3].dot(v),
        )
    }

    /// Vector-matrix multiplication: `v^T * self`.
    pub fn eval_left(&self, v: &Vector<N>) -> Vector<N> {
        self.transpose().eval(v)
    }

    /// Matrix-matrix multiplication: `self * rhs`.
    pub fn mat_mul(&self, rhs: &Self) -> Self {
        let rt = rhs.transpose();
        Self([
            Vector::new(
                self.0[0].dot(&rt.0[0]),
                self.0[0].dot(&rt.0[1]),
                self.0[0].dot(&rt.0[2]),
                self.0[0].dot(&rt.0[3]),
            ),
            Vector::new(
                self.0[1].dot(&rt.0[0]),
                self.0[1].dot(&rt.0[1]),
                self.0[1].dot(&rt.0[2]),
                self.0[1].dot(&rt.0[3]),
            ),
            Vector::new(
                self.0[2].dot(&rt.0[0]),
                self.0[2].dot(&rt.0[1]),
                self.0[2].dot(&rt.0[2]),
                self.0[2].dot(&rt.0[3]),
            ),
            Vector::new(
                self.0[3].dot(&rt.0[0]),
                self.0[3].dot(&rt.0[1]),
                self.0[3].dot(&rt.0[2]),
                self.0[3].dot(&rt.0[3]),
            ),
        ])
    }

    /// Determinant of a 4×4 matrix via Laplace expansion.
    pub fn det(&self) -> BigInt<N> {
        let m = &self.0;
        // Expand along first row.
        let minor =
            |r0: usize, r1: usize, r2: usize, c0: usize, c1: usize, c2: usize| -> BigInt<N> {
                // 3×3 determinant of rows r0,r1,r2 and cols c0,c1,c2.
                m[r0][c0]
                    .ct_mul(
                        &m[r1][c1]
                            .ct_mul(&m[r2][c2])
                            .ct_sub(&m[r1][c2].ct_mul(&m[r2][c1])),
                    )
                    .ct_sub(
                        &m[r0][c1].ct_mul(
                            &m[r1][c0]
                                .ct_mul(&m[r2][c2])
                                .ct_sub(&m[r1][c2].ct_mul(&m[r2][c0])),
                        ),
                    )
                    .ct_add(
                        &m[r0][c2].ct_mul(
                            &m[r1][c0]
                                .ct_mul(&m[r2][c1])
                                .ct_sub(&m[r1][c1].ct_mul(&m[r2][c0])),
                        ),
                    )
            };
        m[0][0]
            .ct_mul(&minor(1, 2, 3, 1, 2, 3))
            .ct_sub(&m[0][1].ct_mul(&minor(1, 2, 3, 0, 2, 3)))
            .ct_add(&m[0][2].ct_mul(&minor(1, 2, 3, 0, 1, 3)))
            .ct_sub(&m[0][3].ct_mul(&minor(1, 2, 3, 0, 1, 2)))
    }

    /// Adjugate (classical adjoint) of a 4×4 matrix.
    ///
    /// `self * self.adjugate() == det(self) * I`.
    pub fn adjugate(&self) -> Self {
        let m = &self.0;
        let minor3 =
            |r0: usize, r1: usize, r2: usize, c0: usize, c1: usize, c2: usize| -> BigInt<N> {
                m[r0][c0]
                    .ct_mul(
                        &m[r1][c1]
                            .ct_mul(&m[r2][c2])
                            .ct_sub(&m[r1][c2].ct_mul(&m[r2][c1])),
                    )
                    .ct_sub(
                        &m[r0][c1].ct_mul(
                            &m[r1][c0]
                                .ct_mul(&m[r2][c2])
                                .ct_sub(&m[r1][c2].ct_mul(&m[r2][c0])),
                        ),
                    )
                    .ct_add(
                        &m[r0][c2].ct_mul(
                            &m[r1][c0]
                                .ct_mul(&m[r2][c1])
                                .ct_sub(&m[r1][c1].ct_mul(&m[r2][c0])),
                        ),
                    )
            };

        // Cofactor C[i][j] = (-1)^(i+j) * minor(rows without i, cols without j).
        // Adjugate = transpose of cofactor matrix.
        let rows = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
        let mut result = Self::ZERO;
        for i in 0..4 {
            for j in 0..4 {
                let m3 = minor3(
                    rows[i][0], rows[i][1], rows[i][2], rows[j][0], rows[j][1], rows[j][2],
                );
                // Adjugate is transposed: result[j][i] = cofactor[i][j]
                result.0[j][i] = if (i + j) % 2 == 0 {
                    m3
                } else {
                    m3.wrapping_neg()
                };
            }
        }
        result
    }

    /// Scalar division: divides every entry by `scalar`.
    ///
    /// Returns `None` if any entry is not evenly divisible.
    pub fn scalar_div(&self, scalar: &BigInt<N>) -> Option<Self> {
        let mut result = Self::ZERO;
        let mut i = 0;
        while i < 4 {
            let mut j = 0;
            while j < 4 {
                let (q, r) = self.0[i][j].div_rem(scalar);
                if !bool::from(r.is_zero()) {
                    return None;
                }
                result.0[i][j] = q;
                j += 1;
            }
            i += 1;
        }
        Some(result)
    }

    /// Returns the columns of this matrix as an array of [`Vector`].
    pub fn columns(&self) -> [Vector<N>; 4] {
        let s = &self.0;
        [
            Vector::new(s[0][0], s[1][0], s[2][0], s[3][0]),
            Vector::new(s[0][1], s[1][1], s[2][1], s[3][1]),
            Vector::new(s[0][2], s[1][2], s[2][2], s[3][2]),
            Vector::new(s[0][3], s[1][3], s[2][3], s[3][3]),
        ]
    }

    /// Creates a matrix from column vectors.
    pub fn from_columns(cols: &[Vector<N>; 4]) -> Self {
        Self::from_rows(
            Vector::new(cols[0][0], cols[1][0], cols[2][0], cols[3][0]),
            Vector::new(cols[0][1], cols[1][1], cols[2][1], cols[3][1]),
            Vector::new(cols[0][2], cols[1][2], cols[2][2], cols[3][2]),
            Vector::new(cols[0][3], cols[1][3], cols[2][3], cols[3][3]),
        )
    }
}

impl<const N: usize> Copy for Matrix<N> where BigInt<N>: Copy {}

impl<const N: usize> Index<usize> for Matrix<N> {
    type Output = Vector<N>;
    #[inline]
    fn index(&self, idx: usize) -> &Vector<N> {
        &self.0[idx]
    }
}

impl<const N: usize> IndexMut<usize> for Matrix<N> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut Vector<N> {
        &mut self.0[idx]
    }
}

impl<const N: usize> Mul<Vector<N>> for Matrix<N> {
    type Output = Vector<N>;
    #[inline]
    fn mul(self, rhs: Vector<N>) -> Vector<N> {
        self.eval(&rhs)
    }
}

impl<const N: usize> Mul for Matrix<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mat_mul(&rhs)
    }
}

impl<const N: usize> PartialEq for Matrix<N> {
    fn eq(&self, other: &Self) -> bool {
        self.0[0] == other.0[0]
            && self.0[1] == other.0[1]
            && self.0[2] == other.0[2]
            && self.0[3] == other.0[3]
    }
}

impl<const N: usize> Eq for Matrix<N> {}

impl<const N: usize> fmt::Debug for Matrix<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Matrix([{:?}, {:?}, {:?}, {:?}])",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}
