//! 4-element integer vectors and 4×4 integer matrices over [`BigInt`].
//!
//! These types represent the fixed-dimension vectors and matrices used
//! in quaternion lattice arithmetic: basis matrices, coordinate vectors,
//! and Gram matrices. All dimensions are 4, matching the rank of the
//! quaternion algebra B_{p,∞}.

use core::{
    array, fmt,
    ops::{Add, Index, IndexMut, Mul, Neg, Sub},
};

use super::bigint::BigInt;

#[cfg(test)]
mod tests;

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

    /// Widen each component from `BigInt<N>` to `BigInt<W>`.
    ///
    /// Sign-extends per [`BigInt::widen`]. Requires `W ≥ N`.
    #[inline]
    pub fn widen<const W: usize>(self) -> Vector<W> {
        Vector::new(
            self.0[0].widen::<W>(),
            self.0[1].widen::<W>(),
            self.0[2].widen::<W>(),
            self.0[3].widen::<W>(),
        )
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
    #[cfg_attr(test, mutants::skip)] // formatting, not correctness
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
    /// This is the standard modular HNF, originally due to
    /// Domich–Kannan–Trotter and treated in detail as Algorithm
    /// 2.4.8 in Cohen, *A Course in Computational Algebraic
    /// Number Theory*, GTM 138, Springer (1993),
    /// [doi:10.1007/978-3-662-02945-9][cohen1993]. The same
    /// algorithm name appears in the C reference's
    /// `ibz_mat_4xn_hnf_mod_core` (`quaternion/ref/generic/hnf/hnf.c`).
    ///
    /// The key observation: if `D` is any positive multiple of the
    /// lattice determinant $\det(L)$ in the rank-$n$ case
    /// $L \subseteq \mathbb{Z}^n$, then
    /// $L \supseteq D \cdot \mathbb{Z}^n$. We exploit this in two
    /// ways:
    ///
    /// 1. We append the modulus columns $D \cdot e_i$ ($i = 0, \dots, n-1$) to
    ///    the input generator set, so the row-pivot gcd accumulation produces
    ///    the canonical pivot $d_i = \gcd(\text{row-}i \text{ entries}, D)$
    ///    rather than the smaller $\gcd(\text{row-}i \text{ entries})$, which
    ///    is what Cohen 2.4.8 achieves via its inner $\mathrm{xgcd}(a_{k,i},
    ///    m)$ step with decreasing $m$.
    /// 2. We reduce every intermediate entry modulo $D$ after every arithmetic
    ///    update. Adding any multiple of $D \cdot e_i \in L$ to a column does
    ///    not change the lattice, so this leaves the HNF unchanged while
    ///    bounding every entry by $D$ rather than by the exponential classical
    ///    bound.
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
    ///   guidance" level.
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
    ///
    /// [cohen1993]: https://doi.org/10.1007/978-3-662-02945-9
    pub fn from_hnf_columns_mod<const W: usize>(cols: &[Vector<N>], modulus: &BigInt<N>) -> Self {
        const {
            assert!(
                W >= N,
                "Matrix::from_hnf_columns_mod: working width W must be >= storage width N"
            )
        };
        let c_orig = cols.len();
        assert!(c_orig >= 4, "need at least 4 columns for rank-4 HNF");
        let d = 4usize;

        let modulus_w: BigInt<W> = modulus.widen();

        // Positive Euclidean reduction mod D: returns r in [0, D).
        let reduce = |x: &BigInt<W>| -> BigInt<W> {
            let r = x.ct_mod(&modulus_w);
            if bool::from(r.is_negative()) {
                r.ct_add(&modulus_w)
            } else {
                r
            }
        };

        // Convert a value in [0, D] to the centered representative in
        // (-D/2, D/2] when in [0, D), or leave it as D when exactly D.
        //
        // Used only for `xgcd` inputs so that
        // `xgcd(a[k][i], a[j][i])` computes the true integer gcd of
        // their *signed* values rather than the much larger gcd of
        // their wrapped (near-D positive) representations.
        //
        // The "leave D as D" carve-out matters for the modulus columns
        // (`D · e_i`) we append: their pivot-row entry is the literal D.
        // Centering would map D → 0, dropping the modulus from the gcd
        // and giving non-canonical pivots for negatively-signed inputs.
        //
        // The cofactors `u, v` returned by xgcd satisfy
        // `u·signed(a[k][i]) + v·signed(a[j][i]) = g`. Applied to the
        // unsigned (Euclidean) values, the linear combination
        // `u·a[k][r] + v·a[j][r]` differs from the signed version only
        // by a multiple of D in each coordinate, so reducing mod D
        // recovers the canonical value.
        let to_centered = |x: BigInt<W>| -> BigInt<W> {
            let two_x = x.ct_add(&x);
            // 2x > D  ⇔  x > D/2.
            let above_half = bool::from(two_x.ct_sub(&modulus_w).is_positive());
            // x < D  (so x ∈ [0, D)). Excludes the literal modulus value.
            let below_d = bool::from(modulus_w.ct_sub(&x).is_positive());
            if above_half && below_d {
                x.ct_sub(&modulus_w)
            } else {
                x
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

        // Append the implicit modulus columns D·e_i (i = 0..d) so the
        // row-pivot gcd accumulation produces the canonical pivot
        // gcd(row entries, D) instead of just gcd(row entries).
        //
        // The lattice we want to HNF-reduce is L = ⟨input cols⟩ + D·Z^d
        // (the caller's contract: D is a positive multiple of the
        // intended covolume, which means D·Z^d ⊆ L). Without these
        // explicit generators, our column-style accumulation only sees
        // the input columns: combining (5·e_3) and modulus 8 in the row
        // gcd yields 5 instead of the canonical gcd(5, 8) = 1.
        //
        // Cohen 2.4.8 / `ibz_mat_4xn_hnf_mod_core` (C reference) reaches
        // the same canonical answer via an explicit `xgcd(a[k][i], m)`
        // step at each pivot, with `m` decreasing as gcds peel off.
        // Adding 4 explicit un-reduced columns is mathematically
        // equivalent and slots into our existing column-style loop
        // without restructuring the algorithm.
        for i in 0..d {
            let mut extra = [BigInt::<W>::ZERO; 4];
            extra[i] = modulus_w;
            a.push(extra);
        }
        let c = c_orig + d;

        // Classical HNF, reducing every updated entry mod D.
        // Helper for the gcd-combine pair update. Mirrors Cohen 2.4.8 /
        // C ref's `ibz_mat_4xn_hnf_mod_core` lines:
        //
        //     c = u·a[k] + v·a[j]                       (new gcd col)
        //     a[j] = (a[k][i]/g)·a[j] - (a[j][i]/g)·a[k]  (orthogonal col, row-i zero)
        //     a[k] = c mod m
        //
        // Updating only `a[k]` (the gcd col) without simultaneously
        // updating `a[j]` makes the pair-transformation non-unimodular,
        // shrinking the lattice. With our previous one-sided update the
        // canonical HNF covolume came out as `(product of nontrivial
        // pivots) × (some integer factor)` — fine on inputs where the
        // gcd-combine never actually combines two nonzero values, but
        // wrong on real inputs (KAT 29).
        let combine_pair =
            |a: &mut Vec<[BigInt<W>; 4]>,
             k: usize,
             j: usize,
             reduce: &dyn Fn(&BigInt<W>) -> BigInt<W>,
             to_centered: &dyn Fn(BigInt<W>) -> BigInt<W>| {
                let val_k_eu = a[k][k];
                let val_j_eu = a[j][k];
                // Mirror C ref's `if (!ibz_is_zero(&(a[j][i])))` guard. With
                // val_j = 0, xgcd(val_k, 0) = (val_k, ±1, 0) and the
                // pair-update degenerates to a[k] := ±a[k], a[j] := ±a[j]
                // — a sign flip that, while unimodular, breaks the
                // canonical-HNF output convention (entries should stay
                // sign-stable across pivot iterations).
                if bool::from(val_j_eu.is_zero()) {
                    return;
                }
                // xgcd on centered representatives so the gcd is the true
                // integer gcd of the signed values (not the wrapped ones).
                let val_k = to_centered(val_k_eu);
                let val_j = to_centered(val_j_eu);
                let (g, u, v) = val_k.xgcd(&val_j);
                // coeff_k = a[k][k]_signed / g, coeff_j = a[j][k]_signed / g.
                // Both are exact (g divides both signed values).
                let (coeff_k, _) = val_k.div_rem(&g);
                let (coeff_j, _) = val_j.div_rem(&g);
                let old_k = a[k];
                let old_j = a[j];
                for r in 0..4 {
                    // u·a[k] + v·a[j] gives the new gcd col.
                    let new_k_r = reduce(&u.ct_mul(&old_k[r]).ct_add(&v.ct_mul(&old_j[r])));
                    // coeff_k·a[j] - coeff_j·a[k] gives the orthogonal col
                    // (row-k entry zero).
                    let new_j_r =
                        reduce(&coeff_k.ct_mul(&old_j[r]).ct_sub(&coeff_j.ct_mul(&old_k[r])));
                    a[k][r] = new_k_r;
                    a[j][r] = new_j_r;
                }
            };

        let mut pivot = d;
        while pivot > 0 {
            pivot -= 1;

            if pivot > 0 {
                let mut j = pivot;
                while j > 0 {
                    j -= 1;
                    combine_pair(&mut a, pivot, j, &reduce, &to_centered);
                }
            }

            {
                let mut j = d;
                while j < c {
                    combine_pair(&mut a, pivot, j, &reduce, &to_centered);
                    j += 1;
                }
            }

            // Edge case: every input row-pivot entry is a multiple of D.
            // After folding in the D·e_pivot extra, xgcd(0, D) = (D, 0, 1),
            // so a[pivot] becomes D·e_pivot, which then reduces to all
            // zeros mod D. The canonical pivot in this case is D itself
            // (the lattice's row-pivot projection is D·Z), so set the
            // pivot column to modulus·e_pivot directly. Mirrors the
            // `if (ibz_is_zero(&w[i][i])) ibz_copy(&w[i][i], &m)` step
            // in C ref's `ibz_mat_4xn_hnf_mod_core`.
            if bool::from(a[pivot][pivot].is_zero()) {
                a[pivot] = [BigInt::<W>::ZERO; 4];
                a[pivot][pivot] = modulus_w;
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

impl<const N: usize> Matrix<N> {
    /// Modular HNF mirroring C-ref's `ibz_mat_4xn_hnf_mod_core`
    /// (`hnf.c:116`) byte-for-byte.
    ///
    /// Differs from [`Matrix::from_hnf_columns_mod`] in convention:
    ///
    /// - **Decreasing modulus**: `m /= d` after each pivot, peeling off the
    ///   gcd. The original `from_hnf_columns_mod` uses a constant modulus
    ///   throughout.
    /// - **Separate output array**: `w[0..3]` accumulates output columns,
    ///   distinct from the input/working `a[]`.
    /// - **Pivot col tracking**: `k = n-1, n-2, ..., n-4` (the last 4 cols
    ///   become output), separate from the row pivot `i = 3, 2, 1, 0`.
    ///
    /// Used by [`crate::quaternions::lattice::Lattice::sum_mod`]
    /// (sign-side response phase via `from_generator_mod_hnf`).
    /// The original `from_hnf_columns_mod` is still used by keygen's
    /// `build_equiv_from_delta`, where its convention coincides with
    /// canonical HNF for keygen-shaped inputs.
    pub fn from_hnf_columns_mod_cref<const W: usize>(
        cols: &[Vector<N>],
        modulus: &BigInt<N>,
    ) -> Self {
        const {
            assert!(
                W >= N,
                "Matrix::from_hnf_columns_mod_cref: working width W must be >= storage width N"
            )
        };
        let n = cols.len();
        assert!(n >= 4, "need at least 4 columns for rank-4 HNF");

        let mut a: Vec<[BigInt<W>; 4]> = cols
            .iter()
            .map(|v| {
                [
                    v[0].widen::<W>(),
                    v[1].widen::<W>(),
                    v[2].widen::<W>(),
                    v[3].widen::<W>(),
                ]
            })
            .collect();

        let mut m: BigInt<W> = modulus.widen();
        let mut w: [[BigInt<W>; 4]; 4] = [[BigInt::<W>::ZERO; 4]; 4];

        let lin_comb = |c1: &BigInt<W>,
                        v1: &[BigInt<W>; 4],
                        c2: &BigInt<W>,
                        v2: &[BigInt<W>; 4]|
         -> [BigInt<W>; 4] {
            [
                c1.ct_mul(&v1[0]).ct_add(&c2.ct_mul(&v2[0])),
                c1.ct_mul(&v1[1]).ct_add(&c2.ct_mul(&v2[1])),
                c1.ct_mul(&v1[2]).ct_add(&c2.ct_mul(&v2[2])),
                c1.ct_mul(&v1[3]).ct_add(&c2.ct_mul(&v2[3])),
            ]
        };

        // Two mod variants matching C-ref:
        //
        // - `centered_mod`: result in `(-m/2, m/2]`. Used in inner gcd-combine loop
        //   (`ibz_vec_4_linear_combination_mod` → `ibz_centered_mod` in `hnf.c`).
        // - `positive_mod`: result in `[0, |m|)`. Used in output store
        //   (`ibz_vec_4_scalar_mul_mod` → `ibz_mod`).
        let centered_mod = |x: &BigInt<W>, m: &BigInt<W>| -> BigInt<W> {
            let mut r = x.ct_mod(m);
            if bool::from(r.is_negative()) {
                r = r.ct_add(m);
            }
            let two_r = r.ct_add(&r);
            if bool::from(two_r.ct_sub(m).is_positive()) {
                r.ct_sub(m)
            } else {
                r
            }
        };
        let positive_mod = |x: &BigInt<W>, m: &BigInt<W>| -> BigInt<W> {
            let r = x.ct_mod(m);
            if bool::from(r.is_negative()) {
                r.ct_add(m)
            } else {
                r
            }
        };
        let vec_centered_mod_m = |v: &[BigInt<W>; 4], m: &BigInt<W>| -> [BigInt<W>; 4] {
            [
                centered_mod(&v[0], m),
                centered_mod(&v[1], m),
                centered_mod(&v[2], m),
                centered_mod(&v[3], m),
            ]
        };
        let vec_positive_mod_m = |v: &[BigInt<W>; 4], m: &BigInt<W>| -> [BigInt<W>; 4] {
            [
                positive_mod(&v[0], m),
                positive_mod(&v[1], m),
                positive_mod(&v[2], m),
                positive_mod(&v[3], m),
            ]
        };

        // Truncated division matching C `mpz_tdiv_qr` / `ibz_div`:
        // quotient rounds toward zero, remainder takes sign of dividend.
        // Selkie's `BigInt::div_rem` is Euclidean (floor for positive
        // divisor, with positive remainder), which differs from C-ref
        // on negative dividends. The HNF algorithm passes negative
        // intermediate values to `ibz_div` in multiple places
        // (coeff_1 = a[k][i]/d, coeff_2 = a[j][i]/d, and inside
        // `ibz_xgcd_with_u_not_0`), so matching `mpz_tdiv_qr` semantics
        // is required to produce the same canonical HNF as C-ref.
        let trunc_div_rem = |a: &BigInt<W>, b: &BigInt<W>| -> (BigInt<W>, BigInt<W>) {
            let (q_eu, r_eu) = a.div_rem(b);
            if bool::from(a.is_negative()) && !bool::from(r_eu.is_zero()) {
                // Euclidean→truncated conversion for a<0 with nonzero remainder:
                //   trunc rounds toward 0 → |q_trunc| = |q_eu| - 1.
                //   sign(q_trunc) = sign(a)⊕sign(b) = sign(q_eu).
                //   So q_trunc = q_eu + sign(b) (when sign(q_eu) = −sign(b),
                //   moves q_eu one step toward 0).
                // Examples: (−7, 3) Eu=(−3, 2), trunc=(−2, −1); add +1=sign(3).
                //           (−7, −3) Eu=(3, 2), trunc=(2, −1); add −1=sign(−3).
                let b_abs = b.abs();
                let r_trunc = r_eu.ct_sub(&b_abs);
                let sign_b = if bool::from(b.is_negative()) {
                    BigInt::<W>::ONE.wrapping_neg()
                } else {
                    BigInt::<W>::ONE
                };
                let q_trunc = q_eu.ct_add(&sign_b);
                (q_trunc, r_trunc)
            } else {
                (q_eu, r_eu)
            }
        };

        // Helper: xgcd with u != 0 guarantee AND `u·x > 0`, mirroring
        // C-ref's `ibz_xgcd_with_u_not_0`. The "u·x > 0" loop
        // (`hnf_internal.c:90-112`) is critical for the downstream HNF
        // gcd-combine step to produce canonical off-diagonal entries.
        // Without it, Selkie's HNF mod gives a "valid HNF" of the same
        // lattice but with different off-diagonal values than C-ref.
        // Euclidean xgcd giving CANONICAL cofactors matching GMP's
        // `mpz_gcdext` semantics (|u| ≤ y/(2·gcd)). Used in place of
        // Selkie's Stein-based `BigInt::xgcd`, whose cofactors may be
        // far outside the canonical range — even after the C-ref
        // "u·x > 0" loop, the LOOP-NORMALIZED u depends on the initial
        // magnitude, and Stein's output isn't guaranteed close enough
        // to the canonical to land at the same value. Matching
        // `mpz_gcdext` byte-for-byte is the surest way to make the
        // HNF mod algorithm produce canonical upper-triangular form
        // matching C-ref's `ibz_mat_4xn_hnf_mod_core`.
        let euclidean_xgcd = |x: &BigInt<W>, y: &BigInt<W>| -> (BigInt<W>, BigInt<W>, BigInt<W>) {
            // Returns (gcd, u, v) with u·x + v·y = gcd, gcd ≥ 0, and
            // |u| ≤ |y|/(2·gcd), matching mpz_gcdext.
            if bool::from(x.is_zero()) && bool::from(y.is_zero()) {
                return (BigInt::<W>::ZERO, BigInt::<W>::ZERO, BigInt::<W>::ZERO);
            }
            // Handle signed inputs by running on absolute values and
            // adjusting cofactors at the end.
            let x_neg = bool::from(x.is_negative());
            let y_neg = bool::from(y.is_negative());
            let mut a = x.abs();
            let mut b = y.abs();
            // (a, b) initial. Track Bezout coefficients (u_a, v_a) and
            // (u_b, v_b) such that a = u_a·|x| + v_a·|y|,
            // b = u_b·|x| + v_b·|y|. Start: a = |x|, b = |y|.
            let mut u_a = BigInt::<W>::ONE;
            let mut v_a = BigInt::<W>::ZERO;
            let mut u_b = BigInt::<W>::ZERO;
            let mut v_b = BigInt::<W>::ONE;
            // Euclidean: while b != 0, (a, b) = (b, a mod b). Carry
            // coefficients along.
            while !bool::from(b.is_zero()) {
                let (q, r) = a.div_rem(&b);
                a = b;
                b = r;
                let new_u_a = u_b;
                let new_v_a = v_b;
                let new_u_b = u_a.ct_sub(&q.ct_mul(&u_b));
                let new_v_b = v_a.ct_sub(&q.ct_mul(&v_b));
                u_a = new_u_a;
                v_a = new_v_a;
                u_b = new_u_b;
                v_b = new_v_b;
            }
            // gcd = a, with a = u_a·|x| + v_a·|y|.
            // Adjust cofactor signs to match the SIGNED inputs.
            let u_final = if x_neg { u_a.wrapping_neg() } else { u_a };
            let v_final = if y_neg { v_a.wrapping_neg() } else { v_a };
            (a, u_final, v_final)
        };

        let xgcd_with_u_not_0 =
            |x: &BigInt<W>, y: &BigInt<W>| -> (BigInt<W>, BigInt<W>, BigInt<W>) {
                // Special case: both zero.
                if bool::from(x.is_zero()) && bool::from(y.is_zero()) {
                    return (BigInt::<W>::ONE, BigInt::<W>::ONE, BigInt::<W>::ZERO);
                }
                let (d, mut u, mut v) = euclidean_xgcd(x, y);

                // Step 1: ensure u != 0. If u == 0 (= y divides x), shift
                // to u = 1 by setting v -= x/y.
                if bool::from(u.is_zero()) && !bool::from(x.is_zero()) {
                    let y_use = if bool::from(y.is_zero()) {
                        BigInt::<W>::ONE
                    } else {
                        *y
                    };
                    // C-ref uses `ibz_div` (truncated). For negative x,
                    // Selkie's Euclidean `div_rem` would differ.
                    let (q, _) = trunc_div_rem(x, &y_use);
                    v = v.ct_sub(&q);
                    u = BigInt::<W>::ONE;
                }

                // Step 2: ensure u·x > 0 (and as small as possible),
                // matching C-ref `hnf_internal.c:90-112`. Each step
                // adds ±y/d to u and ∓x/d to v, preserving the Bezout
                // identity `u·x + v·y = d`.
                if !bool::from(x.is_zero()) {
                    let xy = x.ct_mul(y);
                    let neg = bool::from(xy.is_negative());
                    let (q_y_d_sgn, _) = trunc_div_rem(y, &d);
                    let q_y_d = if neg {
                        q_y_d_sgn.wrapping_neg()
                    } else {
                        q_y_d_sgn
                    };
                    let (q_x_d_sgn, _) = trunc_div_rem(x, &d);
                    let q_x_d = if neg {
                        q_x_d_sgn.wrapping_neg()
                    } else {
                        q_x_d_sgn
                    };
                    // First, run C-ref's "while u·x ≤ 0" loop to ensure
                    // u·x > 0. Each step: u += sign·y/d, v -= sign·x/d.
                    let mut ux = x.ct_mul(&u);
                    while !bool::from(ux.is_positive()) {
                        u = u.ct_add(&q_y_d);
                        v = v.ct_sub(&q_x_d);
                        ux = x.ct_mul(&u);
                    }
                    // Then minimize |u|: while subtracting one offset
                    // (u -= sign·y/d) keeps ux > 0 AND reduces |u|, do
                    // so. Matches GMP `mpz_gcdext`'s |u| ≤ |y|/(2g)
                    // bound. Without this, Selkie's Stein-based xgcd
                    // (whose initial |u| can be far from canonical)
                    // leaves us with non-canonical cofactors after
                    // the upward "u·x > 0" loop, producing a
                    // different HNF mod result than C-ref.
                    loop {
                        let try_u = u.ct_sub(&q_y_d);
                        let try_ux = x.ct_mul(&try_u);
                        if !bool::from(try_ux.is_positive()) {
                            break;
                        }
                        if !bool::from(u.abs().ct_sub(&try_u.abs()).is_positive()) {
                            break;
                        }
                        u = try_u;
                        v = v.ct_add(&q_x_d);
                    }
                }
                (d, u, v)
            };

        let mut i: i32 = 3;
        let mut k: usize = n - 1;
        let mut j: usize = n - 1;

        #[cfg(test)]
        let trace = std::env::var("SELKIE_TRACE_HNF").is_ok();
        #[cfg(not(test))]
        let trace = false;
        #[cfg(test)]
        let dump_vec = |label: &str, v: &[BigInt<W>; 4]| {
            if trace {
                eprintln!(
                    "  {}: ({} bits, sign={}; {} bits; {} bits; {} bits)",
                    label,
                    v[0].bitsize(),
                    if bool::from(v[0].is_negative()) {
                        '-'
                    } else {
                        '+'
                    },
                    v[1].bitsize(),
                    v[2].bitsize(),
                    v[3].bitsize()
                );
            }
        };
        #[cfg(not(test))]
        let dump_vec = |_label: &str, _v: &[BigInt<W>; 4]| {};
        if trace {
            eprintln!(
                "[HNF_TRACE] entering outer loop, n={n}, modulus.bits={}",
                m.bitsize()
            );
        }

        while i != -1 {
            if trace {
                eprintln!(
                    "[HNF_TRACE] === outer i={i}, k={k}, m.bits={} ===",
                    m.bitsize()
                );
                dump_vec("a[k] (before inner)", &a[k]);
            }
            // Inner loop: accumulate gcd of row-i entries into a[k][i].
            while j != 0 {
                j -= 1;
                if !bool::from(a[j][i as usize].is_zero()) {
                    let val_k = a[k][i as usize];
                    let val_j = a[j][i as usize];
                    if trace {
                        eprintln!(
                            "[HNF_TRACE]  inner j={j}: val_k.bits={} (sign {}), val_j.bits={} (sign {})",
                            val_k.bitsize(),
                            if bool::from(val_k.is_negative()) {
                                '-'
                            } else {
                                '+'
                            },
                            val_j.bitsize(),
                            if bool::from(val_j.is_negative()) {
                                '-'
                            } else {
                                '+'
                            },
                        );
                    }
                    let (d, u, v) = xgcd_with_u_not_0(&val_k, &val_j);
                    if trace {
                        eprintln!(
                            "[HNF_TRACE]    xgcd: d.bits={}, u.bits={} (sign {}), v.bits={} (sign {})",
                            d.bitsize(),
                            u.bitsize(),
                            if bool::from(u.is_negative()) {
                                '-'
                            } else {
                                '+'
                            },
                            v.bitsize(),
                            if bool::from(v.is_negative()) {
                                '-'
                            } else {
                                '+'
                            },
                        );
                    }
                    let c = lin_comb(&u, &a[k], &v, &a[j]);
                    // C-ref uses `ibz_div` (truncated) for coeff_1 and coeff_2.
                    // a[k][i] or a[j][i] may be negative (post centered_mod),
                    // so truncated vs Euclidean div gives different coeffs.
                    let (coeff_1, _) = trunc_div_rem(&val_k, &d);
                    let (coeff_2_pos, _) = trunc_div_rem(&val_j, &d);
                    let coeff_2 = coeff_2_pos.wrapping_neg();
                    let new_j = lin_comb(&coeff_1, &a[j], &coeff_2, &a[k]);
                    a[j] = vec_centered_mod_m(&new_j, &m);
                    a[k] = vec_centered_mod_m(&c, &m);
                    if trace {
                        dump_vec("    new a[k]", &a[k]);
                        dump_vec("    new a[j]", &a[j]);
                    }
                }
            }

            // xgcd col-k's pivot entry with modulus → final pivot.
            let val_k_i = a[k][i as usize];
            let (d, u, _v) = xgcd_with_u_not_0(&val_k_i, &m);
            if trace {
                eprintln!(
                    "[HNF_TRACE]  pivot xgcd: a[k][i].bits={}, d.bits={}, u.bits={} (sign {})",
                    val_k_i.bitsize(),
                    d.bitsize(),
                    u.bitsize(),
                    if bool::from(u.is_negative()) {
                        '-'
                    } else {
                        '+'
                    },
                );
            }

            // Output: positive mod (matches C-ref's `ibz_vec_4_scalar_mul_mod`).
            let mul_k_u: [BigInt<W>; 4] = array::from_fn(|r| u.ct_mul(&a[k][r]));
            w[i as usize] = vec_positive_mod_m(&mul_k_u, &m);

            if bool::from(w[i as usize][i as usize].is_zero()) {
                w[i as usize][i as usize] = m;
            }
            if trace {
                dump_vec("  w[i] (after set)", &w[i as usize]);
            }

            let pivot = w[i as usize][i as usize];
            for h in (i as usize + 1)..4 {
                // Floor division (per C-ref `ibz_div_floor`). Selkie's
                // `div_rem` is truncated; using Euclidean (positive)
                // remainder gives floor q for negative entries.
                let entry = w[h][i as usize];
                let r = positive_mod(&entry, &pivot);
                let (q, _) = entry.ct_sub(&r).div_rem(&pivot);
                let neg_q = q.wrapping_neg();
                let w_i = w[i as usize];
                let updated = lin_comb(&BigInt::<W>::ONE, &w[h], &neg_q, &w_i);
                w[h] = updated;
            }

            let (new_m, _r) = m.div_rem(&d);
            m = new_m;

            if i != 0 {
                k -= 1;
                i -= 1;
                j = k;
                if bool::from(a[k][i as usize].is_zero()) {
                    a[k][i as usize] = m;
                }
            } else {
                break;
            }
        }

        let narrow = |x: &BigInt<W>| -> BigInt<N> {
            x.narrow_to::<N>()
                .expect("mod-HNF output fits in N: entries are bounded by modulus < 2^(64 N)")
        };
        Self::from_columns(&[
            Vector::new(
                narrow(&w[0][0]),
                narrow(&w[0][1]),
                narrow(&w[0][2]),
                narrow(&w[0][3]),
            ),
            Vector::new(
                narrow(&w[1][0]),
                narrow(&w[1][1]),
                narrow(&w[1][2]),
                narrow(&w[1][3]),
            ),
            Vector::new(
                narrow(&w[2][0]),
                narrow(&w[2][1]),
                narrow(&w[2][2]),
                narrow(&w[2][3]),
            ),
            Vector::new(
                narrow(&w[3][0]),
                narrow(&w[3][1]),
                narrow(&w[3][2]),
                narrow(&w[3][3]),
            ),
        ])
    }

    /// V2 fresh literal port of C-ref's `ibz_mat_4xn_hnf_mod_core`.
    /// Used for cross-validation against `from_hnf_columns_mod_cref`
    /// (which has a known bug for KAT-1 sign's inputs). Calls Selkie's
    /// `BigInt::xgcd` directly (= Stein binary) rather than a custom
    /// Euclidean, and uses literal truncated-division and centered-mod
    /// translations from the C source.
    pub fn from_hnf_columns_mod_cref_v2<const W: usize>(
        cols: &[Vector<N>],
        modulus: &BigInt<N>,
    ) -> Self {
        const {
            assert!(
                W >= N,
                "from_hnf_columns_mod_cref_v2: working width W must be >= storage width N"
            )
        };

        // Truncated division (mpz_tdiv_qr / ibz_div): trunc toward 0,
        // remainder has sign of dividend.
        let t_div_rem = |a: &BigInt<W>, b: &BigInt<W>| -> (BigInt<W>, BigInt<W>) {
            let abs_a = a.abs();
            let abs_b = b.abs();
            let (q_mag, r_mag) = abs_a.div_rem(&abs_b);
            let a_neg = bool::from(a.is_negative());
            let b_neg = bool::from(b.is_negative());
            let q_sign_negative = a_neg ^ b_neg;
            let q = if q_sign_negative {
                q_mag.wrapping_neg()
            } else {
                q_mag
            };
            let r = if a_neg { r_mag.wrapping_neg() } else { r_mag };
            (q, r)
        };
        // Floor division (mpz_fdiv_qr / ibz_div_floor): only used on
        // positive divisor in HNF, where Selkie's Euclidean div_rem
        // gives the floor result directly.
        let f_div_rem = |a: &BigInt<W>, b: &BigInt<W>| -> (BigInt<W>, BigInt<W>) { a.div_rem(b) };
        // Centered mod (hnf_internal.c:21-36).
        let centered_mod = |a: &BigInt<W>, m: &BigInt<W>| -> BigInt<W> {
            let tmp = a.ct_mod(m);
            let tmp = if bool::from(tmp.is_zero()) { *m } else { tmp };
            let two = BigInt::<W>::ONE.ct_add(&BigInt::<W>::ONE);
            let (d, _) = f_div_rem(m, &two);
            let cmp = tmp.ct_sub(&d);
            if bool::from(cmp.is_positive()) {
                tmp.ct_sub(m)
            } else {
                tmp
            }
        };
        let vec4_lin_comb = |ca: &BigInt<W>,
                             va: &[BigInt<W>; 4],
                             cb: &BigInt<W>,
                             vb: &[BigInt<W>; 4]|
         -> [BigInt<W>; 4] {
            let mut out = [BigInt::<W>::ZERO; 4];
            for i in 0..4 {
                out[i] = ca.ct_mul(&va[i]).ct_add(&cb.ct_mul(&vb[i]));
            }
            out
        };
        let vec4_lin_comb_mod = |ca: &BigInt<W>,
                                 va: &[BigInt<W>; 4],
                                 cb: &BigInt<W>,
                                 vb: &[BigInt<W>; 4],
                                 m: &BigInt<W>|
         -> [BigInt<W>; 4] {
            let mut sums = [BigInt::<W>::ZERO; 4];
            for i in 0..4 {
                let s = ca.ct_mul(&va[i]).ct_add(&cb.ct_mul(&vb[i]));
                sums[i] = centered_mod(&s, m);
            }
            sums
        };
        let vec4_copy_mod = |v: &[BigInt<W>; 4], m: &BigInt<W>| -> [BigInt<W>; 4] {
            let mut out = [BigInt::<W>::ZERO; 4];
            for i in 0..4 {
                out[i] = centered_mod(&v[i], m);
            }
            out
        };
        let vec4_scalar_mul_mod =
            |s: &BigInt<W>, v: &[BigInt<W>; 4], m: &BigInt<W>| -> [BigInt<W>; 4] {
                let mut out = [BigInt::<W>::ZERO; 4];
                for i in 0..4 {
                    out[i] = v[i].ct_mul(s).ct_mod(m);
                }
                out
            };
        let ibz_xgcd_with_u_not_0 =
            |x: &BigInt<W>, y: &BigInt<W>| -> (BigInt<W>, BigInt<W>, BigInt<W>) {
                if bool::from(x.is_zero()) && bool::from(y.is_zero()) {
                    return (BigInt::<W>::ONE, BigInt::<W>::ONE, BigInt::<W>::ZERO);
                }
                let x1 = *x;
                let y1 = *y;
                let (d, mut u, mut v) = x1.xgcd(&y1);
                if bool::from(u.is_zero()) {
                    if !bool::from(x1.is_zero()) {
                        let y_use = if bool::from(y1.is_zero()) {
                            BigInt::<W>::ONE
                        } else {
                            y1
                        };
                        let (q, _r) = t_div_rem(&x1, &y_use);
                        v = v.ct_sub(&q);
                    }
                    u = BigInt::<W>::ONE;
                }
                if !bool::from(x1.is_zero()) {
                    let r = x1.ct_mul(&y1);
                    let neg = bool::from(r.is_negative());
                    let mut q = x1.ct_mul(&u);
                    while !bool::from(q.is_positive()) {
                        let (mut q_y, _r_y) = t_div_rem(&y1, &d);
                        if neg {
                            q_y = q_y.wrapping_neg();
                        }
                        u = u.ct_add(&q_y);
                        let (mut q_x, _r_x) = t_div_rem(&x1, &d);
                        if neg {
                            q_x = q_x.wrapping_neg();
                        }
                        v = v.ct_sub(&q_x);
                        q = x1.ct_mul(&u);
                    }
                }
                (d, u, v)
            };

        let n = cols.len();
        assert!(n > 3, "generator_number must be > 3");
        let mut i: i32 = 3;
        let mut j: usize = n - 1;
        let mut k: usize = n - 1;
        let mut a: Vec<[BigInt<W>; 4]> = cols
            .iter()
            .map(|c| {
                [
                    c[0].widen::<W>(),
                    c[1].widen::<W>(),
                    c[2].widen::<W>(),
                    c[3].widen::<W>(),
                ]
            })
            .collect();
        let mut w: [[BigInt<W>; 4]; 4] = [[BigInt::<W>::ZERO; 4]; 4];
        assert!(bool::from(modulus.is_positive()), "modulus must be > 0");
        let mut m: BigInt<W> = modulus.widen::<W>();

        while i != -1 {
            while j != 0 {
                j -= 1;
                if !bool::from(a[j][i as usize].is_zero()) {
                    let val_k_i = a[k][i as usize];
                    let val_j_i = a[j][i as usize];
                    let (d, u, v) = ibz_xgcd_with_u_not_0(&val_k_i, &val_j_i);
                    let c = vec4_lin_comb(&u, &a[k], &v, &a[j]);
                    let (coeff_1, _r1) = t_div_rem(&val_k_i, &d);
                    let (coeff_2_pre, _r2) = t_div_rem(&val_j_i, &d);
                    let coeff_2 = coeff_2_pre.wrapping_neg();
                    let new_a_j = vec4_lin_comb_mod(&coeff_1, &a[j], &coeff_2, &a[k], &m);
                    a[j] = new_a_j;
                    a[k] = vec4_copy_mod(&c, &m);
                }
            }
            let val_k_i = a[k][i as usize];
            let (d, u, _v) = ibz_xgcd_with_u_not_0(&val_k_i, &m);
            w[i as usize] = vec4_scalar_mul_mod(&u, &a[k], &m);
            if bool::from(w[i as usize][i as usize].is_zero()) {
                w[i as usize][i as usize] = m;
            }
            let pivot = w[i as usize][i as usize];
            for h in ((i as usize) + 1)..4 {
                let w_h_i = w[h][i as usize];
                let (q_pre, _r) = f_div_rem(&w_h_i, &pivot);
                let q = q_pre.wrapping_neg();
                let w_h = w[h];
                let w_i = w[i as usize];
                w[h] = vec4_lin_comb(&BigInt::<W>::ONE, &w_h, &q, &w_i);
            }
            let (new_m, _r) = t_div_rem(&m, &d);
            m = new_m;
            if i != 0 {
                k -= 1;
                i -= 1;
                j = k;
                if bool::from(a[k][i as usize].is_zero()) {
                    a[k][i as usize] = m;
                }
            } else {
                i -= 1;
            }
        }
        let narrow =
            |x: &BigInt<W>| -> BigInt<N> { x.narrow_to::<N>().expect("HNF output fits in N") };
        Self::from_columns(&[
            Vector::new(
                narrow(&w[0][0]),
                narrow(&w[0][1]),
                narrow(&w[0][2]),
                narrow(&w[0][3]),
            ),
            Vector::new(
                narrow(&w[1][0]),
                narrow(&w[1][1]),
                narrow(&w[1][2]),
                narrow(&w[1][3]),
            ),
            Vector::new(
                narrow(&w[2][0]),
                narrow(&w[2][1]),
                narrow(&w[2][2]),
                narrow(&w[2][3]),
            ),
            Vector::new(
                narrow(&w[3][0]),
                narrow(&w[3][1]),
                narrow(&w[3][2]),
                narrow(&w[3][3]),
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
            //
            // Unimodular two-column xgcd: the transformation
            //   [a[pivot]_new]   [u,         v       ] [a[pivot]_old]
            //   [a[j]_new    ] = [-val_j/g,  val_i/g ] [a[j]_old    ]
            // has determinant `(u·val_i + v·val_j)/g = g/g = 1`, so
            // the column lattice spanned is preserved. Earlier this
            // code only updated `a[pivot]` (leaving `a[j]` at its
            // old data), which is NOT unimodular and changes the
            // lattice covolume — silent corruption of the Z-span
            // when xgcd's `u` is not ±1.
            if pivot > 0 {
                let mut j = pivot;
                while j > 0 {
                    j -= 1;
                    let val_i = a[pivot][pivot];
                    let val_j = a[j][pivot];
                    if !(bool::from(val_i.is_zero()) && bool::from(val_j.is_zero())) {
                        let (g, u, v) = val_i.xgcd(&val_j);
                        let (val_i_over_g, _) = val_i.div_rem(&g);
                        let (val_j_over_g, _) = val_j.div_rem(&g);
                        let old_i = a[pivot];
                        let old_j = a[j];
                        for r in 0..d {
                            a[pivot][r] = u.ct_mul(&old_i[r]).ct_add(&v.ct_mul(&old_j[r]));
                            a[j][r] = val_i_over_g
                                .ct_mul(&old_j[r])
                                .ct_sub(&val_j_over_g.ct_mul(&old_i[r]));
                        }
                    }
                }
            }

            // For non-square input (c > d): also XGCD with extra columns
            // j >= d to fold their contributions into the pivot.
            // Same unimodular tracking as above.
            {
                let mut j = d;
                while j < c {
                    let val_i = a[pivot][pivot];
                    let val_j = a[j][pivot];
                    if !(bool::from(val_i.is_zero()) && bool::from(val_j.is_zero())) {
                        let (g, u, v) = val_i.xgcd(&val_j);
                        let (val_i_over_g, _) = val_i.div_rem(&g);
                        let (val_j_over_g, _) = val_j.div_rem(&g);
                        let old_i = a[pivot];
                        let old_j = a[j];
                        for r in 0..d {
                            a[pivot][r] = u.ct_mul(&old_i[r]).ct_add(&v.ct_mul(&old_j[r]));
                            a[j][r] = val_i_over_g
                                .ct_mul(&old_j[r])
                                .ct_sub(&val_j_over_g.ct_mul(&old_i[r]));
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
