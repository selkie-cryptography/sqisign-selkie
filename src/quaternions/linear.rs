//! 4-element integer vectors and 4×4 integer matrices over [`BigInt`].
//!
//! These types represent the fixed-dimension vectors and matrices used
//! in quaternion lattice arithmetic: basis matrices, coordinate vectors,
//! and Gram matrices. All dimensions are 4, matching the rank of the
//! quaternion algebra B_{p,∞}.

use core::{
    fmt,
    ops::{Add, Index, IndexMut, Mul, Neg, Sub},
};

use super::bigint::BigInt;

// ---------------------------------------------------------------------------
// Vector<N>: 4-element vector of BigInt<N>
// ---------------------------------------------------------------------------

/// A 4-element vector of [`BigInt<N>`] values.
///
/// Used for quaternion algebra element coordinates in the basis
/// `{1, i, j, ij}` and for lattice basis column vectors.
#[derive(Clone)]
pub struct Vector<const N: usize>([BigInt<N>; 4]);

impl<const N: usize> Vector<N> {
    /// The zero vector.
    pub const ZERO: Self = Self([BigInt::ZERO; 4]);

    /// Creates a vector from four elements.
    #[inline]
    pub const fn new(a: BigInt<N>, b: BigInt<N>, c: BigInt<N>, d: BigInt<N>) -> Self {
        Self([a, b, c, d])
    }

    /// Dot product: `sum_i self[i]*other[i]`.
    pub fn dot(&self, other: &Self) -> BigInt<N> {
        let mut acc = self.0[0].ct_mul(&other.0[0]);
        acc = acc.ct_add(&self.0[1].ct_mul(&other.0[1]));
        acc = acc.ct_add(&self.0[2].ct_mul(&other.0[2]));
        acc = acc.ct_add(&self.0[3].ct_mul(&other.0[3]));
        acc
    }
}

impl<const N: usize> Copy for Vector<N> where BigInt<N>: Copy {}

impl<const N: usize> Index<usize> for Vector<N> {
    type Output = BigInt<N>;
    #[inline]
    fn index(&self, idx: usize) -> &BigInt<N> {
        &self.0[idx]
    }
}

impl<const N: usize> IndexMut<usize> for Vector<N> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut BigInt<N> {
        &mut self.0[idx]
    }
}

impl<const N: usize> Add for Vector<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0].ct_add(&rhs.0[0]),
            self.0[1].ct_add(&rhs.0[1]),
            self.0[2].ct_add(&rhs.0[2]),
            self.0[3].ct_add(&rhs.0[3]),
        ])
    }
}

impl<const N: usize> Sub for Vector<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0].ct_sub(&rhs.0[0]),
            self.0[1].ct_sub(&rhs.0[1]),
            self.0[2].ct_sub(&rhs.0[2]),
            self.0[3].ct_sub(&rhs.0[3]),
        ])
    }
}

impl<const N: usize> Neg for Vector<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self([
            self.0[0].wrapping_neg(),
            self.0[1].wrapping_neg(),
            self.0[2].wrapping_neg(),
            self.0[3].wrapping_neg(),
        ])
    }
}

impl<const N: usize> PartialEq for Vector<N> {
    fn eq(&self, other: &Self) -> bool {
        self.0[0] == other.0[0]
            && self.0[1] == other.0[1]
            && self.0[2] == other.0[2]
            && self.0[3] == other.0[3]
    }
}

impl<const N: usize> Eq for Vector<N> {}

/// Widen: zero-extend a four-element vector to eight-limb elements.
impl From<Vector<4>> for Vector<8> {
    fn from(v: Vector<4>) -> Self {
        Self::new(v[0].into(), v[1].into(), v[2].into(), v[3].into())
    }
}

impl<const N: usize> fmt::Debug for Vector<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Vector([{}, {}, {}, {}])",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

// ---------------------------------------------------------------------------
// Matrix<N>: 4×4 matrix of BigInt<N>
// ---------------------------------------------------------------------------

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

    /// Compute the column-style Hermite Normal Form of this matrix.
    ///
    /// The result is an upper-triangular matrix with positive pivots,
    /// elements to the left of pivots are zero, and elements to the
    /// right of pivots are in `[0, pivot)`.
    ///
    /// [Alg. 3.2] of the SQIsign specification.
    ///
    /// [Alg. 3.2]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.2
    pub fn hnf(&self) -> Self {
        Self::from_hnf_columns(&self.columns())
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

    /// Modular Hermite Normal Form (mod-HNF) constructor for
    /// fixed-precision arithmetic, computed with internal widening
    /// to `BigInt<W>`.
    ///
    /// # Why this exists: HNF coefficient blow-up
    ///
    /// Classical integer HNF (as implemented by [`Matrix::from_hnf_columns`]
    /// and as written in Algorithm 3.2 of the SQIsign v2
    /// specification) produces intermediate column entries whose
    /// magnitude can grow substantially during the xgcd /
    /// elimination phases. The worst-case bound is exponential in
    /// the rank — for our rank-4 ideal lattices this can reach
    /// $2^{3000}$ bits or more, well beyond any reasonable fixed
    /// storage width.
    ///
    /// At our commitment-ideal working width of `N = 30` (1920
    /// bits), classical HNF applied to the 8-column concatenation
    /// `[O_0 \cdot \gamma \mid O_0 \cdot N]` silently truncates:
    /// intermediate xgcd products exceed the storage budget and
    /// wrap, and the resulting "HNF" basis collapses to contain
    /// elements of the ambient order $O_0$ rather than representing
    /// the ideal $I = O_0\langle\gamma, N\rangle$. Every downstream
    /// step then operates on a lattice unrelated to the intended
    /// ideal.
    ///
    /// # Algorithm
    ///
    /// This is the standard Domich--Kannan--Trotter modular HNF
    /// (see also Cohen's *A Course in Computational Algebraic
    /// Number Theory*, §2.4.2, and Storjohann's subsequent
    /// refinements). The key observation is: if `D` is any
    /// positive multiple of the lattice determinant $\det(L)$ in
    /// the rank-$n$ case $L \subseteq \mathbb{Z}^n$, then
    /// $L \supseteq D \cdot \mathbb{Z}^n$. Consequently, adding
    /// $D \cdot e_i$ to any column preserves the lattice, and so
    /// we may reduce every intermediate entry modulo `D` after
    /// every arithmetic update without changing the resulting
    /// HNF. Entries are then bounded by `D` rather than by the
    /// exponential classical bound.
    ///
    /// The working width `W` must be at least large enough to
    /// hold a product of two `D`-sized values before reduction:
    /// `W * 64 >= 2 * bits(D) + slack`. A compile-time assertion
    /// checks `W >= N`; callers are responsible for sizing `W`
    /// against their specific modulus.
    ///
    /// # Divergences from the spec and the C reference
    ///
    /// - **Spec (Algorithm 3.2):** describes classical HNF over
    ///   arbitrary-precision integers. It does not discuss fixed-precision
    ///   adaptations or the coefficient-size bounds necessary for a safe
    ///   fixed-width implementation. This is a spec gap at the "implementation
    ///   guidance" level; documented in `latex/spec-review.tex`.
    /// - **C reference:** uses GMP (`ibz_t`), so coefficient growth is absorbed
    ///   by arbitrary-precision arithmetic and `quat_lattice_hnf` follows the
    ///   classical algorithm directly. Our fixed-precision constraint forces
    ///   the modular variant; this is an implementation-level advance over the
    ///   C reference, not a mathematical one.
    /// - **Output semantics:** identical to classical HNF in $\mathbb{Z}$,
    ///   provided `modulus` is a valid multiple of `det(L(cols))`. Callers that
    ///   pass a non-multiple produce an HNF of a different (possibly larger)
    ///   lattice.
    ///
    /// # Parameters
    ///
    /// * `cols` - input generator columns. May have more than 4 entries; the
    ///   HNF reduction produces a 4-column output.
    /// * `modulus` - a positive multiple of the lattice determinant
    ///   $\det(L(\mathtt{cols}))$. For a left $O_0$-ideal of norm $N$
    ///   constructed as $O_0\gamma + O_0 N$, the integer-column covolume is
    ///   $d^4 \cdot N^2 \cdot p / 4$; the caller typically passes $4 d^4 N^2 p$
    ///   or any larger positive multiple.
    ///
    /// # Width requirement
    ///
    /// The `const W` working width must satisfy
    /// `64 * W >= 2 * bits(modulus) + O(1)`. For `modulus ≈
    /// 2^1282` (NIST-I commitment ideals) this needs `W >= 42`
    /// with headroom.
    ///
    /// # Panics
    ///
    /// Panics if `cols.len() < 4` (need at least rank-4
    /// generators), or at compile time if `W < N`.
    ///
    /// WARNING: Not constant-time. The xgcd / elimination
    /// sequence is data-dependent. `TODO(ct)`: make constant-time
    /// before production use — this path is on the secret-derived
    /// signing ideal.
    pub fn from_hnf_columns_mod<const W: usize>(cols: &[Vector<N>], modulus: &BigInt<N>) -> Self {
        const {
            assert!(
                W >= N,
                "Matrix::from_hnf_columns_mod: working width W must be >= storage width N"
            )
        };
        let c = cols.len();
        assert!(c >= 4, "need at least 4 columns for rank-4 HNF");
        let d = 4usize;

        let modulus_w: BigInt<W> = modulus.widen();

        // Positive reduction mod D: returns r in [0, D).
        let reduce = |x: &BigInt<W>| -> BigInt<W> {
            let r = x.ct_mod(&modulus_w);
            if bool::from(r.is_negative()) {
                r.ct_add(&modulus_w)
            } else {
                r
            }
        };

        // Widen every input column into the working width W and
        // immediately reduce mod D.
        let mut a: Vec<[BigInt<W>; 4]> = cols
            .iter()
            .map(|v| {
                [
                    reduce(&v[0].widen::<W>()),
                    reduce(&v[1].widen::<W>()),
                    reduce(&v[2].widen::<W>()),
                    reduce(&v[3].widen::<W>()),
                ]
            })
            .collect();

        // Classical HNF, reducing every updated entry mod D.
        let mut pivot = d;
        while pivot > 0 {
            pivot -= 1;

            if pivot > 0 {
                let mut j = pivot;
                while j > 0 {
                    j -= 1;
                    let val_i = a[pivot][pivot];
                    let val_j = a[j][pivot];
                    if !(bool::from(val_i.is_zero()) && bool::from(val_j.is_zero())) {
                        let (_g, u, v) = val_i.xgcd(&val_j);
                        let old_i = a[pivot];
                        let old_j = a[j];
                        for r in 0..d {
                            let prod1 = u.ct_mul(&old_i[r]);
                            let prod2 = v.ct_mul(&old_j[r]);
                            a[pivot][r] = reduce(&prod1.ct_add(&prod2));
                        }
                    }
                }
            }

            {
                let mut j = d;
                while j < c {
                    let val_i = a[pivot][pivot];
                    let val_j = a[j][pivot];
                    if !(bool::from(val_i.is_zero()) && bool::from(val_j.is_zero())) {
                        let (_g, u, v) = val_i.xgcd(&val_j);
                        let old_i = a[pivot];
                        let old_j = a[j];
                        for r in 0..d {
                            let prod1 = u.ct_mul(&old_i[r]);
                            let prod2 = v.ct_mul(&old_j[r]);
                            a[pivot][r] = reduce(&prod1.ct_add(&prod2));
                        }
                    }
                    j += 1;
                }
            }

            let piv = a[pivot][pivot];
            if bool::from(piv.is_zero()) {
                continue;
            }

            // Eliminate a[j][pivot] for j < pivot.
            {
                let mut j = 0;
                while j < pivot {
                    let (g, _) = a[j][pivot].div_rem(&piv);
                    if !bool::from(g.is_zero()) {
                        let col_piv = a[pivot];
                        for (r, col_piv_r) in col_piv.iter().enumerate().take(d) {
                            let sub = g.ct_mul(col_piv_r);
                            a[j][r] = reduce(&a[j][r].ct_sub(&sub));
                        }
                    }
                    j += 1;
                }
            }

            // Reduce a[j][pivot] for j > pivot into [0, piv).
            {
                let mut j = pivot + 1;
                while j < c {
                    let entry = a[j][pivot];
                    let r = entry.ct_mod(&piv);
                    let (g, _) = entry.ct_sub(&r).div_rem(&piv);
                    if !bool::from(g.is_zero()) {
                        let col_piv = a[pivot];
                        for (row, col_piv_row) in col_piv.iter().enumerate().take(d) {
                            let sub = g.ct_mul(col_piv_row);
                            a[j][row] = reduce(&a[j][row].ct_sub(&sub));
                        }
                    }
                    j += 1;
                }
            }
        }

        // Final narrow from W back to N. HNF entries are bounded
        // by `modulus`, which fits in N by the caller's contract.
        let narrow = |x: &BigInt<W>| -> BigInt<N> {
            x.narrow_to::<N>()
                .expect("mod-HNF output fits in N: entries are bounded by modulus < 2^(64 N)")
        };
        Self::from_columns(&[
            Vector::new(
                narrow(&a[0][0]),
                narrow(&a[0][1]),
                narrow(&a[0][2]),
                narrow(&a[0][3]),
            ),
            Vector::new(
                narrow(&a[1][0]),
                narrow(&a[1][1]),
                narrow(&a[1][2]),
                narrow(&a[1][3]),
            ),
            Vector::new(
                narrow(&a[2][0]),
                narrow(&a[2][1]),
                narrow(&a[2][2]),
                narrow(&a[2][3]),
            ),
            Vector::new(
                narrow(&a[3][0]),
                narrow(&a[3][1]),
                narrow(&a[3][2]),
                narrow(&a[3][3]),
            ),
        ])
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

// ---------------------------------------------------------------------------
// Hermite Normal Form (HNF)
// ---------------------------------------------------------------------------

impl<const N: usize> Matrix<N> {
    /// Compute the column-style Hermite Normal Form from a set of
    /// 4-element column vectors (generators).
    ///
    /// The input can have more than 4 columns (e.g., 8 columns when
    /// computing the sum of two lattices); the HNF reduction produces
    /// 4 independent columns.
    ///
    /// Implements [Alg. 3.2].
    ///
    /// [Alg. 3.2]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.2
    pub fn from_hnf_columns(cols: &[Vector<N>]) -> Self {
        let c = cols.len();
        assert!(c >= 4, "need at least 4 columns for rank-4 HNF");
        let d = 4usize;

        // Work with a mutable array of columns. a[col][row].
        let mut a: Vec<[BigInt<N>; 4]> = cols.iter().map(|v| [v[0], v[1], v[2], v[3]]).collect();

        // [Alg. 3.2] — 0-based indexing.
        // Spec's 1-based "i" maps to 0-based "pivot".
        //
        // [Alg. 3.2]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.2
        let mut pivot = d;

        while pivot > 0 {
            pivot -= 1;

            // Lines 1-7: Accumulate gcd into a[pivot][pivot] by combining
            // column pivot with columns j < pivot (spec Algorithm 3.2).
            if pivot > 0 {
                let mut j = pivot;
                while j > 0 {
                    j -= 1;
                    let val_i = a[pivot][pivot];
                    let val_j = a[j][pivot];
                    if !(bool::from(val_i.is_zero()) && bool::from(val_j.is_zero())) {
                        let (_g, u, v) = val_i.xgcd(&val_j);
                        let old_i = a[pivot];
                        let old_j = a[j];
                        for r in 0..d {
                            a[pivot][r] = u.ct_mul(&old_i[r]).ct_add(&v.ct_mul(&old_j[r]));
                        }
                    }
                }
            }

            // For non-square input (c > d): also XGCD with extra columns
            // j >= d to fold their contributions into the pivot.
            {
                let mut j = d;
                while j < c {
                    let val_i = a[pivot][pivot];
                    let val_j = a[j][pivot];
                    if !(bool::from(val_i.is_zero()) && bool::from(val_j.is_zero())) {
                        let (_g, u, v) = val_i.xgcd(&val_j);
                        let old_i = a[pivot];
                        let old_j = a[j];
                        for r in 0..d {
                            a[pivot][r] = u.ct_mul(&old_i[r]).ct_add(&v.ct_mul(&old_j[r]));
                        }
                    }
                    j += 1;
                }
            }

            // Ensure pivot is positive.
            if bool::from(a[pivot][pivot].is_negative()) {
                for elem in &mut a[pivot] {
                    *elem = elem.wrapping_neg();
                }
            }
            let piv = a[pivot][pivot];
            if bool::from(piv.is_zero()) {
                continue;
            }

            // Lines 8-10: Eliminate a[j][pivot] for j < pivot.
            // g ← a[j][pivot] / piv (exact integer division).
            // col_j ← col_j - g * col_pivot.
            {
                let mut j = 0;
                while j < pivot {
                    let (g, _) = a[j][pivot].div_rem(&piv);
                    if !bool::from(g.is_zero()) {
                        let col_piv = a[pivot];
                        for (r, col_piv_r) in col_piv.iter().enumerate().take(d) {
                            a[j][r] = a[j][r].ct_sub(&g.ct_mul(col_piv_r));
                        }
                    }
                    j += 1;
                }
            }

            // Lines 11-14: Reduce a[j][pivot] for j > pivot to [0, piv).
            // r ← a[j][pivot] mod piv.
            // g ← (a[j][pivot] - r) / piv.
            // col_j ← col_j - g * col_pivot.
            {
                let mut j = pivot + 1;
                while j < c {
                    let entry = a[j][pivot];
                    let r = entry.ct_mod(&piv);
                    let (g, _) = entry.ct_sub(&r).div_rem(&piv);
                    if !bool::from(g.is_zero()) {
                        let col_piv = a[pivot];
                        for (row, col_piv_row) in col_piv.iter().enumerate().take(d) {
                            a[j][row] = a[j][row].ct_sub(&g.ct_mul(col_piv_row));
                        }
                    }
                    j += 1;
                }
            }
        }

        // The first d columns contain the HNF.
        Self::from_columns(&[
            Vector::new(a[0][0], a[0][1], a[0][2], a[0][3]),
            Vector::new(a[1][0], a[1][1], a[1][2], a[1][3]),
            Vector::new(a[2][0], a[2][1], a[2][2], a[2][3]),
            Vector::new(a[3][0], a[3][1], a[3][2], a[3][3]),
        ])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    type I = BigInt<4>;
    type V = Vector<4>;
    type M = Matrix<4>;

    fn i(v: i64) -> I {
        I::from(v)
    }

    #[test]
    fn vec_add() {
        let a = V::new(i(1), i(2), i(3), i(4));
        let b = V::new(i(10), i(20), i(30), i(40));
        let c = a + b;
        assert_eq!(c[0], i(11));
        assert_eq!(c[1], i(22));
        assert_eq!(c[2], i(33));
        assert_eq!(c[3], i(44));
    }

    #[test]
    fn vec_dot() {
        let a = V::new(i(1), i(2), i(3), i(4));
        let b = V::new(i(5), i(6), i(7), i(8));
        assert_eq!(a.dot(&b), i(70)); // 5+12+21+32
    }

    #[test]
    fn mat_identity() {
        let v = V::new(i(1), i(2), i(3), i(4));
        let result = M::IDENTITY * v;
        assert_eq!(result, v);
    }

    #[test]
    fn mat_eval() {
        let mut m = M::ZERO;
        m[0][0] = i(2);
        m[1][1] = i(3);
        m[2][2] = i(4);
        m[3][3] = i(5);
        let v = V::new(i(1), i(1), i(1), i(1));
        let r = m * v;
        assert_eq!(r[0], i(2));
        assert_eq!(r[1], i(3));
        assert_eq!(r[2], i(4));
        assert_eq!(r[3], i(5));
    }

    #[test]
    fn mat_transpose() {
        let m = M::from_rows(
            V::new(i(1), i(2), i(3), i(4)),
            V::new(i(5), i(6), i(7), i(8)),
            V::new(i(9), i(10), i(11), i(12)),
            V::new(i(13), i(14), i(15), i(16)),
        );
        let t = m.transpose();
        assert_eq!(t[0][0], i(1));
        assert_eq!(t[0][1], i(5));
        assert_eq!(t[0][2], i(9));
        assert_eq!(t[0][3], i(13));
        assert_eq!(t[1][0], i(2));
        assert_eq!(t[3][3], i(16));
    }

    #[test]
    fn mat_mul() {
        let a = M::IDENTITY;
        let b = M::from_rows(
            V::new(i(1), i(2), i(3), i(4)),
            V::new(i(5), i(6), i(7), i(8)),
            V::new(i(9), i(10), i(11), i(12)),
            V::new(i(13), i(14), i(15), i(16)),
        );
        assert_eq!(a * b, b);
    }

    #[test]
    fn mat_scalar_div() {
        let m = M::from_rows(
            V::new(i(6), i(12), i(18), i(24)),
            V::new(i(3), i(9), i(15), i(21)),
            V::new(i(0), i(0), i(0), i(0)),
            V::new(i(30), i(60), i(90), i(120)),
        );
        let result = m.scalar_div(&i(3)).expect("all divisible by 3");
        assert_eq!(result[0][0], i(2));
        assert_eq!(result[0][1], i(4));
        assert_eq!(result[3][3], i(40));
    }

    #[test]
    fn mat_scalar_div_fails() {
        let m = M::from_rows(V::new(i(6), i(7), i(0), i(0)), V::ZERO, V::ZERO, V::ZERO);
        assert!(m.scalar_div(&i(3)).is_none());
    }

    #[test]
    fn hnf_identity() {
        assert_eq!(M::IDENTITY.hnf(), M::IDENTITY);
    }

    #[test]
    fn hnf_diagonal() {
        let m = M::from_rows(
            V::new(i(2), i(0), i(0), i(0)),
            V::new(i(0), i(3), i(0), i(0)),
            V::new(i(0), i(0), i(5), i(0)),
            V::new(i(0), i(0), i(0), i(7)),
        );
        assert_eq!(m.hnf(), m);
    }

    #[test]
    fn hnf_upper_triangular_reduction() {
        let m = M::from_rows(
            V::new(i(2), i(5), i(0), i(0)),
            V::new(i(0), i(3), i(0), i(0)),
            V::new(i(0), i(0), i(1), i(0)),
            V::new(i(0), i(0), i(0), i(1)),
        );
        let h = m.hnf();
        assert_eq!(h[0][0], i(2));
        assert_eq!(h[0][1], i(1)); // 5 mod 2 = 1
        assert_eq!(h[1][1], i(3));
        assert!(bool::from(h[0][0].is_positive()));
        assert!(bool::from(h[1][1].is_positive()));
    }

    #[test]
    fn hnf_is_upper_triangular() {
        let m = M::from_rows(
            V::new(i(6), i(4), i(2), i(1)),
            V::new(i(0), i(3), i(1), i(0)),
            V::new(i(0), i(0), i(5), i(2)),
            V::new(i(0), i(0), i(0), i(7)),
        );
        let h = m.hnf();
        // Lower triangle zero.
        assert!(bool::from(h[1][0].is_zero()));
        assert!(bool::from(h[2][0].is_zero()));
        assert!(bool::from(h[2][1].is_zero()));
        assert!(bool::from(h[3][0].is_zero()));
        assert!(bool::from(h[3][1].is_zero()));
        assert!(bool::from(h[3][2].is_zero()));
        // Pivots positive.
        assert!(bool::from(h[0][0].is_positive()));
        assert!(bool::from(h[1][1].is_positive()));
        assert!(bool::from(h[2][2].is_positive()));
        assert!(bool::from(h[3][3].is_positive()));
    }

    #[test]
    fn hnf_from_8_columns() {
        let cols = [
            V::new(i(1), i(0), i(0), i(0)),
            V::new(i(0), i(1), i(0), i(0)),
            V::new(i(0), i(0), i(1), i(0)),
            V::new(i(0), i(0), i(0), i(1)),
            V::new(i(1), i(0), i(0), i(0)),
            V::new(i(0), i(1), i(0), i(0)),
            V::new(i(0), i(0), i(1), i(0)),
            V::new(i(0), i(0), i(0), i(1)),
        ];
        let h = M::from_hnf_columns(&cols);
        assert_eq!(h, M::IDENTITY);
    }
}
