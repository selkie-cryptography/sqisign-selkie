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
                    .vt_mul(
                        &m[r1][c1]
                            .vt_mul(&m[r2][c2])
                            .ct_sub(&m[r1][c2].vt_mul(&m[r2][c1])),
                    )
                    .ct_sub(
                        &m[r0][c1].vt_mul(
                            &m[r1][c0]
                                .vt_mul(&m[r2][c2])
                                .ct_sub(&m[r1][c2].vt_mul(&m[r2][c0])),
                        ),
                    )
                    .ct_add(
                        &m[r0][c2].vt_mul(
                            &m[r1][c0]
                                .vt_mul(&m[r2][c1])
                                .ct_sub(&m[r1][c1].vt_mul(&m[r2][c0])),
                        ),
                    )
            };
        m[0][0]
            .vt_mul(&minor(1, 2, 3, 1, 2, 3))
            .ct_sub(&m[0][1].vt_mul(&minor(1, 2, 3, 0, 2, 3)))
            .ct_add(&m[0][2].vt_mul(&minor(1, 2, 3, 0, 1, 3)))
            .ct_sub(&m[0][3].vt_mul(&minor(1, 2, 3, 0, 1, 2)))
    }

    /// Determinant of the 3x3 minor on rows `r` and columns `c`.
    ///
    /// Used by [`Self::adjugate`].
    fn minor3(&self, r: [usize; 3], c: [usize; 3]) -> BigInt<N> {
        let m = &self.0;

        m[r[0]][c[0]]
            .vt_mul(
                &m[r[1]][c[1]]
                    .vt_mul(&m[r[2]][c[2]])
                    .ct_sub(&m[r[1]][c[2]].vt_mul(&m[r[2]][c[1]])),
            )
            .ct_sub(
                &m[r[0]][c[1]].vt_mul(
                    &m[r[1]][c[0]]
                        .vt_mul(&m[r[2]][c[2]])
                        .ct_sub(&m[r[1]][c[2]].vt_mul(&m[r[2]][c[0]])),
                ),
            )
            .ct_add(
                &m[r[0]][c[2]].vt_mul(
                    &m[r[1]][c[0]]
                        .vt_mul(&m[r[2]][c[1]])
                        .ct_sub(&m[r[1]][c[1]].vt_mul(&m[r[2]][c[0]])),
                ),
            )
    }

    /// Adjugate (classical adjoint) of a 4×4 matrix.
    ///
    /// `self * self.adjugate() == det(self) * I`.  Equals
    /// `transpose(self.cofactor())`, built transposed in place.
    pub fn adjugate(&self) -> Self {
        // Cofactor C[i][j] = (-1)^(i+j) * minor(rows without i, cols without j).
        // Adjugate = transpose of cofactor matrix.
        let rows = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

        let mut result = Self::ZERO;
        for i in 0..4 {
            for j in 0..4 {
                let m3 = self.minor3(rows[i], rows[j]);

                // Adjugate is transposed: result[j][i] = cofactor[i][j].
                result.0[j][i] = if (i + j) % 2 == 0 {
                    m3
                } else {
                    m3.wrapping_neg()
                };
            }
        }

        result
    }

    /// Cofactor matrix `C[i][j] = (-1)^(i+j) * minor(i, j)` together with
    /// the determinant, sharing the twelve `2x2` row-pair minors.
    ///
    /// The cofactor matrix equals `transpose(self.adjugate())`. Forming
    /// it by Laplace expansion along complementary `2x2` blocks (the six
    /// minors of rows `(0,1)` and the six of rows `(2,3)`, each over a
    /// column pair) issues ~66 limb-multiplies versus the ~180 of
    /// sixteen independent `3x3` minors plus a separate [`Self::det`].
    /// The determinant is the same complementary-block sum, so
    /// `Lattice::dual` gets both from one pass.
    pub fn cofactor_and_det(&self) -> (Self, BigInt<N>) {
        let m = &self.0;

        // 2x2 minors of rows (0,1) over each column pair.
        let s01 = m[0][0].vt_mul(&m[1][1]).ct_sub(&m[0][1].vt_mul(&m[1][0]));
        let s02 = m[0][0].vt_mul(&m[1][2]).ct_sub(&m[0][2].vt_mul(&m[1][0]));
        let s03 = m[0][0].vt_mul(&m[1][3]).ct_sub(&m[0][3].vt_mul(&m[1][0]));
        let s12 = m[0][1].vt_mul(&m[1][2]).ct_sub(&m[0][2].vt_mul(&m[1][1]));
        let s13 = m[0][1].vt_mul(&m[1][3]).ct_sub(&m[0][3].vt_mul(&m[1][1]));
        let s23 = m[0][2].vt_mul(&m[1][3]).ct_sub(&m[0][3].vt_mul(&m[1][2]));

        // 2x2 minors of rows (2,3) over each column pair.
        let t01 = m[2][0].vt_mul(&m[3][1]).ct_sub(&m[2][1].vt_mul(&m[3][0]));
        let t02 = m[2][0].vt_mul(&m[3][2]).ct_sub(&m[2][2].vt_mul(&m[3][0]));
        let t03 = m[2][0].vt_mul(&m[3][3]).ct_sub(&m[2][3].vt_mul(&m[3][0]));
        let t12 = m[2][1].vt_mul(&m[3][2]).ct_sub(&m[2][2].vt_mul(&m[3][1]));
        let t13 = m[2][1].vt_mul(&m[3][3]).ct_sub(&m[2][3].vt_mul(&m[3][1]));
        let t23 = m[2][2].vt_mul(&m[3][3]).ct_sub(&m[2][3].vt_mul(&m[3][2]));

        // Cofactors of rows 0,1 from the rows-(2,3) minors `t`.
        let c00 = m[1][1]
            .vt_mul(&t23)
            .ct_sub(&m[1][2].vt_mul(&t13))
            .ct_add(&m[1][3].vt_mul(&t12));
        let c01 = m[1][2]
            .vt_mul(&t03)
            .ct_sub(&m[1][0].vt_mul(&t23))
            .ct_sub(&m[1][3].vt_mul(&t02));
        let c02 = m[1][0]
            .vt_mul(&t13)
            .ct_sub(&m[1][1].vt_mul(&t03))
            .ct_add(&m[1][3].vt_mul(&t01));
        let c03 = m[1][1]
            .vt_mul(&t02)
            .ct_sub(&m[1][0].vt_mul(&t12))
            .ct_sub(&m[1][2].vt_mul(&t01));

        let c10 = m[0][2]
            .vt_mul(&t13)
            .ct_sub(&m[0][1].vt_mul(&t23))
            .ct_sub(&m[0][3].vt_mul(&t12));
        let c11 = m[0][0]
            .vt_mul(&t23)
            .ct_sub(&m[0][2].vt_mul(&t03))
            .ct_add(&m[0][3].vt_mul(&t02));
        let c12 = m[0][1]
            .vt_mul(&t03)
            .ct_sub(&m[0][0].vt_mul(&t13))
            .ct_sub(&m[0][3].vt_mul(&t01));
        let c13 = m[0][0]
            .vt_mul(&t12)
            .ct_sub(&m[0][1].vt_mul(&t02))
            .ct_add(&m[0][2].vt_mul(&t01));

        // Cofactors of rows 2,3 from the rows-(0,1) minors `s`.
        let c20 = m[3][1]
            .vt_mul(&s23)
            .ct_sub(&m[3][2].vt_mul(&s13))
            .ct_add(&m[3][3].vt_mul(&s12));
        let c21 = m[3][2]
            .vt_mul(&s03)
            .ct_sub(&m[3][0].vt_mul(&s23))
            .ct_sub(&m[3][3].vt_mul(&s02));
        let c22 = m[3][0]
            .vt_mul(&s13)
            .ct_sub(&m[3][1].vt_mul(&s03))
            .ct_add(&m[3][3].vt_mul(&s01));
        let c23 = m[3][1]
            .vt_mul(&s02)
            .ct_sub(&m[3][0].vt_mul(&s12))
            .ct_sub(&m[3][2].vt_mul(&s01));

        let c30 = m[2][2]
            .vt_mul(&s13)
            .ct_sub(&m[2][1].vt_mul(&s23))
            .ct_sub(&m[2][3].vt_mul(&s12));
        let c31 = m[2][0]
            .vt_mul(&s23)
            .ct_sub(&m[2][2].vt_mul(&s03))
            .ct_add(&m[2][3].vt_mul(&s02));
        let c32 = m[2][1]
            .vt_mul(&s03)
            .ct_sub(&m[2][0].vt_mul(&s13))
            .ct_sub(&m[2][3].vt_mul(&s01));
        let c33 = m[2][0]
            .vt_mul(&s12)
            .ct_sub(&m[2][1].vt_mul(&s02))
            .ct_add(&m[2][2].vt_mul(&s01));

        // det = sum over complementary column-pair blocks.
        let det = s01
            .vt_mul(&t23)
            .ct_sub(&s02.vt_mul(&t13))
            .ct_add(&s03.vt_mul(&t12))
            .ct_add(&s12.vt_mul(&t03))
            .ct_sub(&s13.vt_mul(&t02))
            .ct_add(&s23.vt_mul(&t01));

        let cofactor = Self::from_rows(
            Vector::new(c00, c01, c02, c03),
            Vector::new(c10, c11, c12, c13),
            Vector::new(c20, c21, c22, c23),
            Vector::new(c30, c31, c32, c33),
        );

        (cofactor, det)
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

    /// Returns column `c` as a [`Vector`].
    ///
    /// Building a single column directly avoids the four-`Vector`
    /// [`Self::columns`] array when only one column is needed.
    pub fn column(&self, c: usize) -> Vector<N> {
        let s = &self.0;

        Vector::new(s[0][c], s[1][c], s[2][c], s[3][c])
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
