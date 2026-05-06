//! Quaternion lattices, orders, and ideals.
//!
//! A quaternion lattice is a full-rank Z-submodule of B_{p,∞},
//! represented by a 4×4 integer basis matrix with a common denominator.
//! Orders and ideals are lattices with additional algebraic structure.
//!
//! Lattices that have been reduced to Hermite Normal Form are
//! represented by [`HnfLattice`], a distinct type from [`Lattice`].
//! This ensures at compile time that operations requiring canonical
//! form (equality, containment) receive properly reduced input.
//!
//! # Divergences from spec / C reference
//!
//! - **Fixed-width integers**: the C ref uses GMP; we use `BigInt<N>` with `N`
//!   chosen per Kim et al. (ePrint 2025/1649). The commitment ideal uses
//!   `LeftIdeal<9>` for D_MIX = 2^512+75 (513 bits); signing will use
//!   `LeftIdeal<110>` for the full 7,026-bit worst case.
//! - **`random_prime_norm_wide`**: constructs the ideal lattice O₀⟨γ,N⟩ via
//!   direct quaternion multiplication with `BigInt<9>`, bypassing `Element`
//!   (whose `Coordinate` would need widening). The C ref uses `Element`
//!   throughout since GMP has no width limit.
//! - **`reduce_to_prime_norm`**: generic over N (was `LeftIdeal<8>` only).
//!
//! See [§3.1.5.2] and [§3.1.6] of the SQIsign specification.
//!
//! [§3.1.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.2
//! [§3.1.6]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.3.1.6

use core::ops::Add;

use rand_core::{OsRng, RngCore};

use super::{
    algebra::{Coordinate, Denominator, Element},
    bigint::BigInt,
    linear::{Matrix, Vector},
};

#[cfg(not(feature = "expose-internals"))]
pub(crate) mod dpe;
#[cfg(feature = "expose-internals")]
pub mod dpe;
use dpe::DoublePlusExponent;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Lattice<N>: quaternion lattice (not necessarily in HNF)
// ---------------------------------------------------------------------------

/// A rank-4 lattice in B_{p,∞}, represented by a 4×4 integer basis
/// matrix and a common denominator.
///
/// By convention ([§3.1.5.2]), columns of the matrix are basis vectors,
/// so that `(α₁ α₂ α₃ α₄) = (1 i j k) · L / denom`.
///
/// This type does not guarantee any canonical form. Convert to
/// [`HnfLattice`] via `Into` when canonical form is needed.
///
/// [§3.1.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.2
#[derive(Clone)]
pub struct Lattice<const N: usize> {
    /// 4×4 basis matrix (columns are basis vectors in the {1,i,j,k} basis).
    basis: Matrix<N>,
    /// Common denominator for the basis.
    denom: BigInt<N>,
}

impl<const N: usize> Lattice<N> {
    /// Creates a lattice from a basis matrix and denominator.
    #[inline]
    pub const fn new(basis: Matrix<N>, denom: BigInt<N>) -> Self {
        Self { basis, denom }
    }

    /// Creates a lattice from an integer basis matrix (denominator = 1).
    #[inline]
    pub const fn from_matrix(basis: Matrix<N>) -> Self {
        Self {
            basis,
            denom: BigInt::ONE,
        }
    }

    /// Returns the basis matrix.
    #[inline]
    pub const fn basis(&self) -> &Matrix<N> {
        &self.basis
    }

    /// Returns a mutable reference to the basis matrix.
    #[inline]
    pub fn basis_mut(&mut self) -> &mut Matrix<N> {
        &mut self.basis
    }

    /// Returns the denominator.
    #[inline]
    pub const fn denom(&self) -> &BigInt<N> {
        &self.denom
    }

    /// Reduce this lattice to Hermite Normal Form.
    ///
    /// Prefer using `HnfLattice::from(lattice)` or `lattice.into()`.
    ///
    /// [Alg. 3.2] of the SQIsign specification.
    ///
    /// [Alg. 3.2]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.2
    fn hnf(self) -> HnfLattice<N> {
        HnfLattice {
            basis: self.basis.hnf(),
            denom: self.denom,
        }
    }

    /// Dual lattice: Λ* = {f ∈ B*_{p,∞} | f(x) ∈ Z ∀x ∈ Λ}.
    ///
    /// Computed as: dual basis = denom · adj(basis)ᵀ, dual denom = det(basis),
    /// where adj(basis)ᵀ = cofactor(basis) = (basis⁻¹ · det(basis))ᵀ.
    /// The adjugate avoids division, keeping everything in integers.
    ///
    /// See [§3.1.5.2] of the spec.
    ///
    /// [§3.1.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.2
    fn dual(&self) -> Self {
        // L^{-1} = adj(L) / det(L), so L^{-T} = adj(L)^T / det(L).
        // Dual basis = denom · adj(basis)^T, with new denom = det(basis).
        let adj_t = self.basis.adjugate().transpose();
        let det = self.basis.det();

        let mut dual_basis = Matrix::ZERO;
        for row in 0..4 {
            for col in 0..4 {
                dual_basis[row][col] = self.denom.ct_mul(&adj_t[row][col]);
            }
        }

        Self {
            basis: dual_basis,
            denom: det,
        }
    }

    /// Conjugate lattice: negate the i, j, k coordinates of each
    /// basis vector, leaving the scalar coordinate unchanged.
    ///
    /// For a lattice with basis columns (α₁, ..., α₄), the conjugate
    /// lattice has basis columns (ᾱ₁, ..., ᾱ₄). In the {1,i,j,k}
    /// basis, conjugation negates rows 1, 2, 3 of the basis matrix.
    ///
    /// See [§3.1.6.1] (Ideal inverse) of the spec.
    ///
    /// [§3.1.6.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.6.1
    pub fn conjugate(&self) -> Self {
        let mut conj_basis = self.basis;
        for col in 0..4 {
            // Negate rows 1, 2, 3 (the i, j, k components).
            conj_basis[1][col] = conj_basis[1][col].wrapping_neg();
            conj_basis[2][col] = conj_basis[2][col].wrapping_neg();
            conj_basis[3][col] = conj_basis[3][col].wrapping_neg();
        }
        Self {
            basis: conj_basis,
            denom: self.denom,
        }
    }

    /// Lattice intersection: `self ∩ other`.
    ///
    /// Computed via the identity L₁ ∩ L₂ = dual(dual(L₁) + dual(L₂)).
    ///
    /// WARNING: the dual computation cubes the entry size through
    /// 3×3 subdeterminants. For wide lattices (entries > ~600 bits),
    /// use [`intersection_via_kernel`](Self::intersection_via_kernel)
    /// instead.
    ///
    /// See [§3.1.5.2] (Intersection) of the spec.
    ///
    /// [§3.1.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.2
    pub fn intersection(&self, other: &Self) -> HnfLattice<N> {
        let dual1 = self.dual();
        let dual2 = other.dual();
        let dual_sum: Lattice<N> = dual1.sum(&dual2).into();
        let result = dual_sum.dual();
        HnfLattice::from(result)
    }

    /// Lattice intersection via the stacked-kernel method.
    ///
    /// Computes L₁ ∩ L₂ by finding the integer kernel of the 4×8
    /// constraint matrix `[d₂·B₁ | −d₁·B₂]` and mapping kernel
    /// vectors back through B₁. This avoids the dual computation
    /// (which cubes entry sizes) at the cost of working at a wider
    /// intermediate width `W`.
    ///
    /// The intermediate width `W` must be large enough for the
    /// column-reduction operations on the constraint matrix.
    /// Entries start at `bits(d) + bits(B)` and can grow during
    /// xgcd elimination. `W` should provide margin above the
    /// initial entry size; the Hadamard bound is
    /// `W >= 4 * (bits(d) + bits(B)) / 64` but smaller values
    /// may work in practice.
    ///
    /// Returns `None` when the resulting HNF basis or denominator
    /// does not narrow into `BigInt<N>`. Callers should either
    /// widen `W` and retry, or continue.
    ///
    /// # Divergences
    ///
    /// The spec and C reference use `dual → sum → dual`
    /// (with GMP for arbitrary precision). We use the kernel
    /// method to avoid the cubic entry-size blow-up that makes
    /// the dual approach incompatible with fixed-width arithmetic.
    #[allow(clippy::needless_range_loop)]
    pub fn intersection_via_kernel<const W: usize>(&self, other: &Self) -> Option<HnfLattice<N>> {
        const { assert!(W >= N, "intersection_via_kernel: W must be >= N") };

        let d1: BigInt<W> = self.denom.widen();
        let d2: BigInt<W> = other.denom.widen();
        let b1 = self.basis;
        let b2 = other.basis;

        // Form the 4×8 constraint matrix M = [d₂·B₁ | −d₁·B₂].
        // Augment with an 8×8 identity below to track column
        // operations: the full 12×8 matrix is [M; I₈].
        //
        // We represent each column as a 12-element array of
        // BigInt<W>. After column reduction on the top 4 rows,
        // columns with all-zero top blocks have kernel vectors
        // in their bottom 8 entries.

        // Build 8 columns of length 12.
        let mut cols: [[BigInt<W>; 12]; 8] = [[BigInt::<W>::ZERO; 12]; 8];

        // First 4 columns: d₂ · B₁ (top), identity cols 0-3 (bottom)
        for col in 0..4 {
            for row in 0..4 {
                cols[col][row] = d2.ct_mul(&b1[row][col].widen::<W>());
            }
            cols[col][4 + col] = BigInt::<W>::ONE;
        }

        // Last 4 columns: −d₁ · B₂ (top), identity cols 4-7 (bottom)
        for col in 0..4 {
            for row in 0..4 {
                cols[4 + col][row] = d1.ct_mul(&b2[row][col].widen::<W>()).wrapping_neg();
            }
            cols[4 + col][4 + 4 + col] = BigInt::<W>::ONE;
        }

        // Column HNF on the top 4 rows. Process each row from
        // 0 to 3: accumulate all nonzero entries into a single
        // pivot column via repeated xgcd, then reduce remaining
        // entries modulo the pivot. This is the same as our
        // `from_hnf_columns` but operating on 8 columns of
        // length 12 (with tracking in the bottom 8 rows).
        let mut pivot_col_for_row = [usize::MAX; 4];

        for pivot_row in 0..4 {
            // Phase 1: accumulate GCD. Pick first nonzero column
            // as initial pivot, then xgcd with all others.
            let mut pc = usize::MAX;
            for col in 0..8 {
                // Skip columns already used as pivots for earlier rows.
                if pivot_col_for_row[..pivot_row].contains(&col) {
                    continue;
                }
                if bool::from(cols[col][pivot_row].is_zero()) {
                    continue;
                }
                if pc == usize::MAX {
                    pc = col;
                    continue;
                }
                let piv = cols[pc][pivot_row];
                let entry = cols[col][pivot_row];
                let (g, u, v) = piv.xgcd(&entry);
                let (piv_over_g, _) = piv.div_rem(&g);
                let (entry_over_g, _) = entry.div_rem(&g);
                let old_pc: [BigInt<W>; 12] = cols[pc];
                let old_col: [BigInt<W>; 12] = cols[col];
                for r in 0..12 {
                    cols[pc][r] = u.ct_mul(&old_pc[r]).ct_add(&v.ct_mul(&old_col[r]));
                    cols[col][r] = piv_over_g
                        .ct_mul(&old_col[r])
                        .ct_sub(&entry_over_g.ct_mul(&old_pc[r]));
                }
            }
            if pc == usize::MAX {
                continue;
            }
            pivot_col_for_row[pivot_row] = pc;

            // Make pivot positive.
            if bool::from(cols[pc][pivot_row].is_negative()) {
                for r in 0..12 {
                    cols[pc][r] = cols[pc][r].wrapping_neg();
                }
            }

            // Phase 2: reduce all other columns modulo the pivot
            // in this row (ensures entries are in [0, pivot)).
            let piv = cols[pc][pivot_row];
            for col in 0..8 {
                if col == pc {
                    continue;
                }
                let entry = cols[col][pivot_row];
                if bool::from(entry.is_zero()) {
                    continue;
                }
                let (q, _) = entry.div_rem(&piv);
                if !bool::from(q.is_zero()) {
                    let snap = cols[pc];
                    for r in 0..12 {
                        cols[col][r] = cols[col][r].ct_sub(&q.ct_mul(&snap[r]));
                    }
                }
            }
        }

        // Extract kernel vectors: columns whose top 4 entries are
        // all zero. There should be exactly 4 such columns.
        let mut kernel_vecs = [[BigInt::<W>::ZERO; 8]; 4];
        let mut n_kernel = 0;
        for col in 0..8 {
            let top_zero = (0..4).all(|r| bool::from(cols[col][r].is_zero()));
            if top_zero && n_kernel < 4 {
                for r in 0..8 {
                    kernel_vecs[n_kernel][r] = cols[col][4 + r];
                }
                n_kernel += 1;
            }
        }
        // Intersection basis: for each kernel vector [a; b]
        // (where a is the first 4 entries), compute B₁ · a.
        // The result has denominator d₁.
        let mut inter_cols = [Vector::<W>::ZERO; 4];
        for (i, kv) in kernel_vecs.iter().enumerate() {
            let a = [kv[0], kv[1], kv[2], kv[3]];
            let mut v = [BigInt::<W>::ZERO; 4];
            for row in 0..4 {
                for k in 0..4 {
                    v[row] = v[row].ct_add(&b1[row][k].widen::<W>().ct_mul(&a[k]));
                }
            }
            inter_cols[i] = Vector::new(v[0], v[1], v[2], v[3]);
        }

        // Reduce to HNF at width W, then narrow to N.
        let inter_basis_w = Matrix::<W>::from_hnf_columns(&inter_cols);
        let denom_w = d1;

        // Narrow back to BigInt<N>.
        let mut basis_n = Matrix::<N>::ZERO;
        for row in 0..4 {
            for col in 0..4 {
                basis_n[row][col] = inter_basis_w[row][col].narrow_to::<N>()?;
            }
        }
        let denom_n: BigInt<N> = denom_w.narrow_to()?;

        Some(HnfLattice {
            basis: basis_n,
            denom: denom_n,
        })
    }

    // Lattice product: `self · other`.
    //
    // The product of two lattices L₁L₂ is the lattice generated by
    // all products α·β where α ∈ L₁ and β ∈ L₂. Computed by
    // multiplying each pair of basis elements and taking the HNF
    // of all 16 resulting column vectors.
    //
    // This requires quaternion multiplication, so it's only available
    // when both lattices are `Lattice<4>` (to access `basis_elem`).
    //
    // See [§3.1.5.2] (Multiplication) of the spec.
    //
    // [§3.1.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.2
    //
    // TODO: Implement lattice product. Requires Element::mul on each
    // pair of basis elements, producing 16 columns for HNF.
    // This is needed for ideal multiplication in id2iso.

    /// Lattice sum: `self + other`.
    ///
    /// The sum of two lattices is the lattice generated by the union of
    /// their bases. Computed by concatenating the columns of both basis
    /// matrices and taking the HNF.
    ///
    /// See [§3.1.5.2] (Sum) of the spec.
    ///
    /// [§3.1.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.2
    fn sum(&self, other: &Self) -> HnfLattice<N> {
        if self.denom == other.denom {
            let cols_a = self.basis.columns();
            let cols_b = other.basis.columns();
            let all_cols = [
                cols_a[0], cols_a[1], cols_a[2], cols_a[3], cols_b[0], cols_b[1], cols_b[2],
                cols_b[3],
            ];
            HnfLattice {
                basis: Matrix::from_hnf_columns(&all_cols),
                denom: self.denom,
            }
        } else {
            // Scale to common denominator.
            let common_denom = self.denom.ct_mul(&other.denom);
            let scale_a = other.denom;
            let scale_b = self.denom;

            let scale_cols = |basis: &Matrix<N>, s: BigInt<N>| -> [Vector<N>; 4] {
                let cols = basis.columns();
                core::array::from_fn(|idx| {
                    Vector::new(
                        cols[idx][0].ct_mul(&s),
                        cols[idx][1].ct_mul(&s),
                        cols[idx][2].ct_mul(&s),
                        cols[idx][3].ct_mul(&s),
                    )
                })
            };

            let cols_a = scale_cols(&self.basis, scale_a);
            let cols_b = scale_cols(&other.basis, scale_b);
            let all_cols = [
                cols_a[0], cols_a[1], cols_a[2], cols_a[3], cols_b[0], cols_b[1], cols_b[2],
                cols_b[3],
            ];
            HnfLattice {
                basis: Matrix::from_hnf_columns(&all_cols),
                denom: common_denom,
            }
        }
    }

    /// Lattice sum computed with modular HNF at intermediate
    /// width `W`, using `modulus` as the bounding modulus.
    ///
    /// Semantically identical to [`Lattice::sum`]: the result is
    /// the Z-lattice generated by the columns of `self` and
    /// `other`, in HNF. The difference is that the final HNF
    /// reduction goes through [`Matrix::from_hnf_columns_mod`] instead of the
    /// classical [`Matrix::hnf`], which avoids coefficient
    /// blow-up at fixed precision.
    ///
    /// `modulus` must be a positive multiple of the integer-
    /// column covolume of the combined lattice. The working
    /// width `W` must be large enough to hold a product of two
    /// `modulus`-sized values before reduction; see the rustdoc
    /// on [`Matrix::from_hnf_columns_mod`] for the precise width requirement.
    ///
    /// The two input lattices must share the same denominator.
    /// Unlike [`Lattice::sum`], this method does not rescale
    /// mismatched denominators because the modulus bound is
    /// chosen to fit the shared-denominator case: rescaling
    /// would multiply column magnitudes by the other lattice's
    /// denom and could push intermediates outside the budget.
    /// Callers that need to combine lattices with different
    /// denominators should rescale them beforehand and choose a
    /// modulus that accommodates the scaled magnitudes.
    ///
    /// Returns `None` if the denominators differ. In release
    /// builds this is the sole failure mode; debug builds
    /// additionally assert the shared-denom precondition so
    /// caller bugs surface loudly during development.
    pub fn sum_mod<const W: usize>(
        &self,
        other: &Self,
        modulus: &BigInt<N>,
    ) -> Option<HnfLattice<N>> {
        debug_assert_eq!(
            self.denom, other.denom,
            "Lattice::sum_mod: denominators must match",
        );
        if self.denom != other.denom {
            return None;
        }
        let cols_a = self.basis.columns();
        let cols_b = other.basis.columns();
        let all_cols = [
            cols_a[0], cols_a[1], cols_a[2], cols_a[3], cols_b[0], cols_b[1], cols_b[2], cols_b[3],
        ];
        Some(HnfLattice {
            basis: Matrix::<N>::from_hnf_columns_mod::<W>(&all_cols, modulus),
            denom: self.denom,
        })
    }
}

impl<const N: usize> Copy for Lattice<N> where BigInt<N>: Copy {}

impl<const N: usize> Lattice<N> {
    /// Returns the j-th basis vector (column j) as a quaternion element.
    pub fn basis_elem(&self, j: usize) -> Element<N> {
        Element::<N>::new(
            Coordinate::from_bigint(self.basis[0][j]),
            Coordinate::from_bigint(self.basis[1][j]),
            Coordinate::from_bigint(self.basis[2][j]),
            Coordinate::from_bigint(self.basis[3][j]),
            Denominator::from_bigint_unchecked(self.denom),
        )
    }
}

impl<const N: usize> Lattice<N> {
    /// Decompose an element into coordinates in this lattice's column basis.
    ///
    /// Given α ∈ L, finds (c₀, c₁, c₂, c₃) such that
    /// α = c₀·b₀ + c₁·b₁ + c₂·b₂ + c₃·b₃ where bⱼ are the column
    /// basis elements of L/denom.
    ///
    /// Returns `None` if α is not in the lattice (non-integer solution).
    ///
    /// Uses the adjugate: x = adj(B)·v / det(B), avoiding field inversion.
    pub fn decompose(&self, elem: &Element<N>) -> Option<[BigInt<N>; 4]> {
        let ed = BigInt::<N>::from(elem.denom);

        // Scale to common denominator: target = α_coords · lattice_denom / α_denom.
        let elem_coords = [
            *elem.a.as_bigint(),
            *elem.b.as_bigint(),
            *elem.c.as_bigint(),
            *elem.d.as_bigint(),
        ];
        let mut rhs = [BigInt::<N>::ZERO; 4];
        for i in 0..4 {
            let scaled = elem_coords[i].ct_mul(&self.denom);
            let (q, r) = scaled.div_rem(&ed);
            if !bool::from(r.is_zero()) {
                return None;
            }
            rhs[i] = q;
        }

        // x = adj(B) · rhs / det(B)
        let adj = self.basis.adjugate();
        let det = self.basis.det();
        if bool::from(det.is_zero()) {
            return None;
        }

        let mut result = [BigInt::<N>::ZERO; 4];
        for i in 0..4 {
            let mut val = BigInt::<N>::ZERO;
            for (j, rhs_j) in rhs.iter().enumerate() {
                val = val.ct_add(&adj[i][j].ct_mul(rhs_j));
            }
            let (q, r) = val.div_rem(&det);
            if !bool::from(r.is_zero()) {
                return None;
            }
            result[i] = q;
        }

        Some(result)
    }

    /// Lattice product: `self · other`.
    ///
    /// Multiplies each pair of basis elements (4×4 = 16 products),
    /// producing 16 column vectors, then takes the HNF to get
    /// a 4×4 basis for the product lattice.
    ///
    /// Uses [`Element::mul_direct`] at width N — coordinates must
    /// use at most N/2 limbs to avoid overflow.
    ///
    /// See [§3.1.5.2] (Multiplication) of the spec.
    ///
    /// [§3.1.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.2
    pub fn product(&self, other: &Self) -> HnfLattice<N> {
        let mut all_cols = Vec::new();

        for i in 0..4 {
            let alpha = self.basis_elem(i);
            for j in 0..4 {
                let beta = other.basis_elem(j);
                let product = alpha.mul_direct(&beta);
                // The product has denom = self.denom * other.denom * product.denom.
                // We need to express it in a common denominator for HNF.
                all_cols.push(Vector::new(
                    *product.a.as_bigint(),
                    *product.b.as_bigint(),
                    *product.c.as_bigint(),
                    *product.d.as_bigint(),
                ));
            }
        }

        // The common denominator for all columns.
        // Each product has denom = (self.denom/1) * (other.denom/1) * element_denom.
        // Since basis_elem includes the lattice denom, and mul produces a
        // normalized result, the product columns share a common denom.
        // For now, use the product of the two lattice denoms as the HNF denom.
        let result_basis = Matrix::from_hnf_columns(&all_cols);
        HnfLattice {
            basis: result_basis,
            // The product denominator needs careful handling of the
            // element denominators. For now, use the product.
            denom: self.denom.ct_mul(&other.denom),
        }
    }

    /// Sample a random element from this lattice whose reduced norm
    /// is less than `radius`.
    ///
    /// Uses rejection sampling from a bounding parallelogram of an
    /// L2-reduced dual lattice, following the C reference's
    /// `quat_lattice_sample_from_ball` (lat_ball.c:58).
    ///
    /// The const generic `W` is the intermediate width for products.
    /// It must satisfy `W >= 2 * N` to avoid overflow in Gram matrix
    /// entries (products of N-wide values). For example:
    /// - `Lattice<4>::sample_from_ball::<8>` (verification)
    /// - `Lattice<9>::sample_from_ball::<18>` (commitment)
    /// - `Lattice<110>::sample_from_ball::<220>` (signing)
    ///
    /// # Algorithm
    ///
    /// 1. Compute the Gram matrix G (reduced norm quadratic form)
    /// 2. Compute the dual Gram matrix G* = adj(G)/det(G)
    /// 3. LLL-reduce G* to get tight per-coordinate bounds
    /// 4. Rejection sample: pick uniform coords in the bounding box, map
    ///    through the LLL transformation, check norm ≤ radius
    /// 5. Convert integer coords back to a quaternion element
    ///
    /// Implements [LatticeSampling][Alg. 3.3] as used by
    /// [RandomEquivalentQuaternion][Alg. 4.3].
    ///
    /// WARNING: Not constant-time (rejection loop, LLL).
    ///
    /// TODO(ct): Make constant-time before production use.
    ///
    /// [Alg. 3.3]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.3
    /// [Alg. 4.3]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.3
    pub fn sample_from_ball<const W: usize, R: RngCore>(
        &self,
        radius: &BigInt<N>,
        rng: &mut R,
    ) -> Option<Element<N>> {
        // Algorithm (matches C ref `quat_lattice_sample_from_ball` in
        // `lat_ball.c`):
        //
        //   1. G  = primal Gram of the lattice basis.
        //   2. dualG = adj(G); det_G = det(G); so G⁻¹ = dualG / det_G.
        //   3. LLL-reduce dualG, tracking the unimodular transform U: after reduction,
        //      `dualG_red = U^T · dualG · U` and U is integer with det(U) = ±1.
        //   4. box[i] = √(dualG_red[i][i] · rad / det_G).
        //   5. U_inv = inv(U) (= adj(U) · sign(det U), since |det U| = 1).
        //   6. Repeat: sample y[i] uniform in [−box[i], box[i]]; x = U_inv^T · y; norm
        //      = x^T · G · x; accept if 0 < norm ≤ rad.
        //   7. α = Σ x[i] · basis_col[i].
        //
        // The dual-LLL bound is asymptotically tighter than the per-axis
        // primal-Gram diagonals: the bounding parallelogram's volume
        // shrinks toward the ellipsoid's, raising the acceptance rate
        // closer to the Hermite-constant optimum and (importantly) into
        // the regime where the ~10⁴-iter inner budget converges quickly
        // for the typical signing radius.
        let cols_n = self.basis.columns();
        let cols_w: [Vector<W>; 4] = core::array::from_fn(|i| {
            Vector::new(
                cols_n[i][0].widen::<W>(),
                cols_n[i][1].widen::<W>(),
                cols_n[i][2].widen::<W>(),
                cols_n[i][3].widen::<W>(),
            )
        });
        // Primal Gram (no LLL reduction here — we LLL the dual below).
        let g_w = *NrdBasis::new(cols_w).gram();

        // Adjust radius: rad = radius · denom² · 2.
        // (Gram corresponds to twice the reduced norm; the radius
        // squared by denom matches the field-reduced rep.)
        let denom_wide: BigInt<W> = self.denom.widen();
        let rad: BigInt<W> = radius
            .widen::<W>()
            .ct_mul(&denom_wide)
            .ct_mul(&denom_wide)
            .ct_mul(&BigInt::<W>::from_u64(2));

        // dualG = adj(G); det_g = det(G).
        let det_g = g_w.det();
        let dual_g = g_w.adjugate();

        // LLL-reduce dualG with unimodular tracking. Trick: feed
        // identity columns + the dual gram into `NrdBasis`. The
        // existing `l2_reduce` updates both:
        //   - cols (initially identity) → cols of U;
        //   - gram (initially dualG) → reduced dualG = U^T · dualG · U.
        // The algorithm only consults `gram` for GSO/decisions, so the
        // mismatch between `cols` and `compute_gram(cols)` is harmless;
        // we only need `from_cols_and_gram` to bypass the constructor's
        // gram recomputation.
        let identity_cols: [Vector<W>; 4] = [
            Vector::new(BigInt::ONE, BigInt::ZERO, BigInt::ZERO, BigInt::ZERO),
            Vector::new(BigInt::ZERO, BigInt::ONE, BigInt::ZERO, BigInt::ZERO),
            Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ONE, BigInt::ZERO),
            Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ZERO, BigInt::ONE),
        ];
        let nrd = NrdBasis::from_cols_and_gram(identity_cols, dual_g).l2_reduce();
        let reduced_dual = *nrd.gram();
        let u_lll = Matrix::from_columns(nrd.cols());

        // box[i] = √(reduced_dual[i][i] · rad / det_g).
        let mut bounds = [BigInt::<W>::ZERO; 4];
        let mut all_zero = true;
        for (i, bound) in bounds.iter_mut().enumerate() {
            let diag = reduced_dual[i][i];
            if bool::from(diag.is_zero()) || bool::from(det_g.is_zero()) {
                continue;
            }
            let prod = diag.ct_mul(&rad);
            let (quot, _) = prod.div_rem(&det_g);
            *bound = quot.sqrt_floor();
            if !bool::from(bound.is_zero()) {
                all_zero = false;
            }
        }
        if all_zero {
            #[cfg(test)]
            eprintln!(
                "[sample_from_ball] ball too small: rad bits={}, dualG_red diag bits=[{}, {}, {}, {}], det_G bits={}",
                rad.bitsize(),
                reduced_dual[0][0].bitsize(),
                reduced_dual[1][1].bitsize(),
                reduced_dual[2][2].bitsize(),
                reduced_dual[3][3].bitsize(),
                det_g.bitsize(),
            );
            return None;
        }

        // U_inv = inv(U_lll) = adj(U_lll) · sign(det U_lll).
        // Since U_lll is unimodular, det = ±1 and adj is integer.
        let det_u = u_lll.det();
        let adj_u = u_lll.adjugate();
        let u_inv = if bool::from(det_u.is_negative()) {
            let mut m = adj_u;
            for i in 0..4 {
                for j in 0..4 {
                    m[i][j] = m[i][j].wrapping_neg();
                }
            }
            m
        } else {
            adj_u
        };

        // Rejection sampling.
        let byte_cap = W * 8;
        #[cfg(test)]
        let mut _n_pos = 0u64;
        #[cfg(test)]
        let mut _n_fit = 0u64;
        #[cfg(test)]
        let mut _best_nrd_over_rad_bits: i64 = 0;
        for _ in 0..200_000 {
            // y[i] uniform in [−bounds[i], bounds[i]].
            let mut y = [BigInt::<W>::ZERO; 4];
            for i in 0..4 {
                if bool::from(bounds[i].is_zero()) {
                    continue;
                }
                let two_b = bounds[i].ct_add(&bounds[i]);
                let bitlen = two_b.bitsize();
                loop {
                    let mut bytes = vec![0u8; byte_cap];
                    let needed = (bitlen as usize).div_ceil(8);
                    rng.fill_bytes(&mut bytes[..needed]);
                    let val = BigInt::<W>::from_bytes_le_unsigned(&bytes[..needed]).abs();
                    if val.bitsize() <= bitlen {
                        let diff = val.ct_sub(&two_b);
                        if bool::from(diff.is_negative()) || bool::from(diff.is_zero()) {
                            y[i] = val.ct_sub(&bounds[i]);
                            break;
                        }
                    }
                }
            }

            // x = U_inv^T · y, i.e., coords in the original lattice basis.
            let y_vec = Vector::new(y[0], y[1], y[2], y[3]);
            let x_vec = u_inv.eval_left(&y_vec);
            let x = [x_vec[0], x_vec[1], x_vec[2], x_vec[3]];

            // Evaluate primal quadratic form: nrd = x^T · G · x.
            let mut nrd = BigInt::<W>::ZERO;
            for i in 0..4 {
                for j in 0..4 {
                    nrd = nrd.ct_add(&x[i].ct_mul(&x[j]).ct_mul(&g_w[i][j]));
                }
            }

            // Reject if zero or exceeds radius.
            if bool::from(nrd.is_zero()) {
                continue;
            }
            // nrd must be positive and ≤ rad.
            if bool::from(nrd.is_negative()) {
                continue; // negative — shouldn't happen for a PD form
            }
            #[cfg(test)]
            {
                _n_pos += 1;
                let nrd_bits = nrd.bitsize() as i64;
                let rad_bits = rad.bitsize() as i64;
                let delta = nrd_bits - rad_bits;
                if _best_nrd_over_rad_bits == 0 || delta < _best_nrd_over_rad_bits {
                    _best_nrd_over_rad_bits = delta;
                }
            }
            let diff = nrd.ct_sub(&rad);
            if !bool::from(diff.is_negative()) && !bool::from(diff.is_zero()) {
                continue; // nrd > rad
            }
            #[cfg(test)]
            {
                _n_fit += 1;
            }

            // Convert to quaternion element:
            //   α = Σ x[i] · col_i (using ORIGINAL basis cols, not the
            //   LLL-reduced ones — `x` is already in original coords).
            let mut coords = [BigInt::<W>::ZERO; 4];
            for i in 0..4 {
                for (k, coord) in coords.iter_mut().enumerate() {
                    *coord = coord.ct_add(&x[i].ct_mul(&cols_w[i][k]));
                }
            }

            let narrow = |v: BigInt<W>| -> BigInt<N> {
                v.narrow_to::<N>()
                    .expect("sampled element fits in BigInt<N>")
            };

            return Some(Element::<N>::new(
                Coordinate::from(narrow(coords[0])),
                Coordinate::from(narrow(coords[1])),
                Coordinate::from(narrow(coords[2])),
                Coordinate::from(narrow(coords[3])),
                Denominator::from_bigint_unchecked(self.denom),
            ));
        }
        #[cfg(test)]
        eprintln!(
            "[sample_from_ball] exhausted 200k attempts: rad bits={}, dualG_red diag bits=[{}, {}, {}, {}], bounds bits=[{}, {}, {}, {}], n_pos={}, n_fit={}, best nrd-rad bits={}",
            rad.bitsize(),
            reduced_dual[0][0].bitsize(),
            reduced_dual[1][1].bitsize(),
            reduced_dual[2][2].bitsize(),
            reduced_dual[3][3].bitsize(),
            bounds[0].bitsize(),
            bounds[1].bitsize(),
            bounds[2].bitsize(),
            bounds[3].bitsize(),
            _n_pos,
            _n_fit,
            _best_nrd_over_rad_bits,
        );
        None
    }
}

/// Lattice sum: `L₁ + L₂` produces an [`HnfLattice`].
impl<const N: usize> Add for Lattice<N> {
    type Output = HnfLattice<N>;
    fn add(self, rhs: Self) -> HnfLattice<N> {
        self.sum(&rhs)
    }
}

/// Lattice sum by reference.
impl<const N: usize> Add<&Lattice<N>> for &Lattice<N> {
    type Output = HnfLattice<N>;
    fn add(self, rhs: &Lattice<N>) -> HnfLattice<N> {
        self.sum(rhs)
    }
}

impl<const N: usize> From<Lattice<N>> for HnfLattice<N> {
    fn from(lat: Lattice<N>) -> Self {
        lat.hnf()
    }
}

impl<const N: usize> core::fmt::Debug for Lattice<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Lattice({:?} / {})", self.basis, self.denom)
    }
}

// ---------------------------------------------------------------------------
// HnfLattice<N>: lattice in Hermite Normal Form
// ---------------------------------------------------------------------------

/// A rank-4 lattice in Hermite Normal Form.
///
/// This type guarantees that the basis matrix is in column-style HNF:
/// upper-triangular with positive pivots, elements to the left of
/// pivots are zero, elements to the right are in `[0, pivot)`.
///
/// Can only be constructed via `From<Lattice>` (which computes the
/// HNF) or the `+` operator on two [`Lattice`] values. Two
/// `HnfLattice` values can be compared directly for lattice equality.
// TODO: Consider making `Lattice` generic over a marker type (e.g.,
// `Lattice<N, Form>` with `Form ∈ {General, Hnf}`) so that shared
// operations (conjugate, intersection, product, dual, sum) are
// implemented once, and HNF-specific operations (contains,
// back-substitution) are available only when `Form = Hnf`.
#[derive(Clone)]
pub struct HnfLattice<const N: usize> {
    basis: Matrix<N>,
    denom: BigInt<N>,
}

impl<const N: usize> HnfLattice<N> {
    /// Returns the HNF basis matrix.
    #[inline]
    pub const fn basis(&self) -> &Matrix<N> {
        &self.basis
    }

    /// Returns the denominator.
    #[inline]
    pub const fn denom(&self) -> &BigInt<N> {
        &self.denom
    }

    /// Canonicalize HNF entries: reduce off-diagonal entries modulo
    /// the diagonal pivot in each row.
    ///
    /// After mod-HNF ([`Matrix::from_hnf_columns_mod`]), the basis
    /// is in valid upper-triangular HNF but off-diagonal entries may
    /// be as large as the modulus (~2^1282 for commitment ideals).
    /// A canonical HNF has `0 ≤ h[col][pivot] < h[pivot][pivot]`
    /// for columns `col > pivot`. This method enforces that,
    /// shrinking entries to be bounded by the diagonal pivots.
    ///
    /// This is the "lines 11–14" reduction step of [Algorithm 3.2]
    /// applied to an already-triangulated matrix.
    ///
    /// [Algorithm 3.2]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.2
    #[must_use]
    pub fn canonicalize(&self) -> Self {
        let mut h = self.basis;
        // h[col][row]: column-major storage.
        // For each pivot row, reduce columns to the right.
        for pivot in 0..4 {
            let piv = h[pivot][pivot];
            if bool::from(piv.is_zero()) {
                continue;
            }
            // Columns to the right: col > pivot.
            for col in (pivot + 1)..4 {
                let entry = h[col][pivot];
                if bool::from(entry.is_zero()) {
                    continue;
                }
                // Reduce entry into [0, piv).
                let r = entry.ct_mod(&piv);
                let (g, _) = entry.ct_sub(&r).div_rem(&piv);
                if bool::from(g.is_zero()) {
                    continue;
                }
                let col_piv = h[pivot];
                for row in 0..4 {
                    h[col][row] = h[col][row].ct_sub(&g.ct_mul(&col_piv[row]));
                }
            }
        }
        Self {
            basis: h,
            denom: self.denom,
        }
    }
}

impl<const N: usize> HnfLattice<N> {
    /// Checks if a quaternion element is contained in this lattice.
    ///
    /// An element α is in the lattice L/d if the system
    /// `L · x = α_coords * d / α_denom` has an integer solution x.
    ///
    /// Returns `Some(coords)` if contained, `None` otherwise.
    ///
    /// Requires HNF form for back-substitution to work correctly.
    ///
    /// See [§3.1.5.2] (Containment) of the spec.
    ///
    /// [§3.1.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.2
    pub fn contains(&self, elem: &Element<N>) -> Option<Vector<N>> {
        let coords: [BigInt<N>; 4] = [
            *elem.a.as_bigint(),
            *elem.b.as_bigint(),
            *elem.c.as_bigint(),
            *elem.d.as_bigint(),
        ];
        let ed: BigInt<N> = *elem.denom.as_bigint();

        // rhs = elem.coord * self.denom / elem.denom
        let rhs = [
            coords[0].ct_mul(&self.denom),
            coords[1].ct_mul(&self.denom),
            coords[2].ct_mul(&self.denom),
            coords[3].ct_mul(&self.denom),
        ];
        let mut target = [BigInt::ZERO; 4];
        for idx in 0..4 {
            let (q, r) = rhs[idx].div_rem(&ed);
            if !bool::from(r.is_zero()) {
                return None;
            }
            target[idx] = q;
        }

        // Back-substitution on upper-triangular HNF.
        let h = &self.basis;
        let mut x = [BigInt::ZERO; 4];
        let mut row = 4;
        while row > 0 {
            row -= 1;
            let pivot = h[row][row];
            if bool::from(pivot.is_zero()) {
                if !bool::from(target[row].is_zero()) {
                    return None;
                }
                continue;
            }
            let mut val = target[row];
            let mut col = row + 1;
            while col < 4 {
                val = val.ct_sub(&h[row][col].ct_mul(&x[col]));
                col += 1;
            }
            let (q, r) = val.div_rem(&pivot);
            if !bool::from(r.is_zero()) {
                return None;
            }
            x[row] = q;
        }

        Some(Vector::new(x[0], x[1], x[2], x[3]))
    }
    /// Conjugate this lattice (negate the i, j, k coordinates).
    ///
    /// Result is re-reduced to HNF since negation breaks the form.
    ///
    /// See [§3.1.6.1] (Ideal inverse) of the spec.
    ///
    /// [§3.1.6.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.6.1
    #[must_use]
    pub fn conjugate(&self) -> Self {
        let mut conj_basis = self.basis;
        for col in 0..4 {
            conj_basis[1][col] = conj_basis[1][col].wrapping_neg();
            conj_basis[2][col] = conj_basis[2][col].wrapping_neg();
            conj_basis[3][col] = conj_basis[3][col].wrapping_neg();
        }
        Self::from(Lattice {
            basis: conj_basis,
            denom: self.denom,
        })
    }

    /// Lattice intersection: `self ∩ other`.
    ///
    /// Computed via `L₁ ∩ L₂ = dual(dual(L₁) + dual(L₂))`.
    ///
    /// See [§3.1.5.2] (Intersection) of the spec.
    ///
    /// [§3.1.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.2
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        let lat_a = Lattice::<N>::from(*self);
        let lat_b = Lattice::<N>::from(*other);
        lat_a.intersection(&lat_b)
    }

    /// Lattice product: `self · other`.
    ///
    /// Multiplies each pair of basis elements (4×4 = 16 products),
    /// then takes HNF. Delegates to [`Lattice::product`] which uses
    /// [`Element::mul_direct`] at width N.
    ///
    /// See [§3.1.5.2] (Multiplication) of the spec.
    ///
    /// [§3.1.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.2
    #[must_use]
    pub fn product(&self, other: &Self) -> Self {
        let lat_a = Lattice::<N>::from(*self);
        let lat_b = Lattice::<N>::from(*other);
        lat_a.product(&lat_b)
    }
}

impl<const N: usize> Copy for HnfLattice<N> where BigInt<N>: Copy {}

impl<const N: usize> From<HnfLattice<N>> for Lattice<N> {
    fn from(hnf: HnfLattice<N>) -> Self {
        Self {
            basis: hnf.basis,
            denom: hnf.denom,
        }
    }
}

impl<const N: usize> PartialEq for HnfLattice<N> {
    fn eq(&self, other: &Self) -> bool {
        if self.denom == other.denom {
            self.basis == other.basis
        } else {
            // Scale to common denominator and compare.
            let g = self.denom.gcd(&other.denom);
            let scale_a = {
                let (q, _) = other.denom.div_rem(&g);
                q
            };
            let scale_b = {
                let (q, _) = self.denom.div_rem(&g);
                q
            };
            let mut equal = true;
            for row in 0..4 {
                for col in 0..4 {
                    let lhs = self.basis[row][col].ct_mul(&scale_a);
                    let rhs = other.basis[row][col].ct_mul(&scale_b);
                    if lhs != rhs {
                        equal = false;
                    }
                }
            }
            equal
        }
    }
}

impl<const N: usize> Eq for HnfLattice<N> {}

impl<const N: usize> core::fmt::Debug for HnfLattice<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "HnfLattice({:?} / {})", self.basis, self.denom)
    }
}

// ---------------------------------------------------------------------------
// Order<N>: maximal order in B_{p,∞}
// ---------------------------------------------------------------------------

/// A maximal order in B_{p,∞}.
///
/// An order is a lattice that is also a subring of B_{p,∞} (closed under
/// multiplication, contains 1). This newtype over [`Lattice`] enforces
/// the order invariant at the type level: values are only constructed by
/// operations that guarantee the result is an order:
///
/// - [`ExtremalOrder::order`] — precomputed extremal orders
/// - [`LeftIdeal::right_order`] — O_R(I) = I⁻¹ · I
/// - [`Order::from_lattice_unchecked`] — internal use when the lattice is known
///   to be an order (e.g., narrowing after `reduce_to_prime_norm`)
///
/// Implements [`Deref<Target = Lattice<N>>`](core::ops::Deref) so all
/// lattice methods are available transparently. Use `From<Order<N>>` to
/// unwrap into the underlying [`Lattice`].
///
/// See [§3.1.5.1] of the SQIsign specification.
///
/// [§3.1.5.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.1
#[derive(Clone)]
pub struct Order<const N: usize>(Lattice<N>);

impl<const N: usize> Order<N> {
    /// Constructs an order from a lattice that is known to be an order.
    ///
    /// # Safety (logical)
    ///
    /// The caller must ensure the lattice is actually a maximal order
    /// (closed under multiplication, contains 1). This is not checked.
    pub(crate) const fn from_lattice_unchecked(lattice: Lattice<N>) -> Self {
        Self(lattice)
    }

    /// Returns the underlying lattice.
    #[inline]
    pub const fn lattice(&self) -> &Lattice<N> {
        &self.0
    }
}

impl<const N: usize> core::ops::Deref for Order<N> {
    type Target = Lattice<N>;

    #[inline]
    fn deref(&self) -> &Lattice<N> {
        &self.0
    }
}

impl<const N: usize> Copy for Order<N> where BigInt<N>: Copy {}

/// Unwrap an order into its underlying lattice.
impl<const N: usize> From<Order<N>> for Lattice<N> {
    fn from(order: Order<N>) -> Self {
        order.0
    }
}

impl<const N: usize> core::fmt::Debug for Order<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Order({:?})", self.0)
    }
}

// ---------------------------------------------------------------------------
// LeftIdeal<N>: left ideal of a maximal order
// ---------------------------------------------------------------------------

/// A left ideal of a maximal order in B_{p,∞}.
///
/// An ideal I = O⟨α, N⟩ is represented by its lattice (in HNF), its
/// norm nrd(I), and the parent (left) order O_L(I).
///
/// See [§3.1.6] of the SQIsign specification.
///
/// [§3.1.6]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.3.1.6
#[derive(Clone)]
pub struct LeftIdeal<const N: usize> {
    /// The lattice representation of this ideal, in HNF.
    lattice: HnfLattice<N>,
    /// The norm of the ideal: nrd(I) = gcd of norms of elements.
    norm: BigInt<N>,
    /// The parent (left) order.
    parent_order: Order<N>,
}

impl<const N: usize> LeftIdeal<N> {
    /// Assemble a left ideal from pre-built components.
    ///
    /// The caller is responsible for ensuring the lattice is the
    /// correct HNF representation of the ideal.
    #[inline]
    pub const fn from_parts(
        lattice: HnfLattice<N>,
        norm: BigInt<N>,
        parent_order: Order<N>,
    ) -> Self {
        Self {
            lattice,
            norm,
            parent_order,
        }
    }

    /// Returns the lattice representation (in HNF).
    #[inline]
    pub const fn lattice(&self) -> &HnfLattice<N> {
        &self.lattice
    }

    /// Returns a mutable reference to the lattice.
    #[inline]
    pub fn lattice_mut(&mut self) -> &mut HnfLattice<N> {
        &mut self.lattice
    }

    /// Returns the norm of the ideal.
    #[inline]
    pub const fn norm(&self) -> &BigInt<N> {
        &self.norm
    }

    /// Returns the parent (left) order.
    #[inline]
    pub const fn parent_order(&self) -> &Order<N> {
        &self.parent_order
    }

    /// Recomputes `self.norm` from the lattice covolume ratio.
    ///
    /// For a left ideal `I` of a maximal order `O`, the reduced norm
    /// satisfies `n(I)² = [O : I]` as `Z`-module index. The index
    /// is derived from the basis determinants:
    ///
    /// ```text
    /// [O : I] = (O.denom)^4 · det(I.basis) / ((I.denom)^4 · det(O.basis))
    /// ```
    ///
    /// The C reference applies this in `quat_lideal_norm` after
    /// every ideal-construction operation. Our fixed-width
    /// constructors (`LeftIdeal::new`, `from_generator_mod_hnf`)
    /// instead store the caller-supplied target `N` verbatim —
    /// which is correct only when `α` and `N` satisfy
    /// `gcd(nrd(α)/N, N) = 1`. When they do not, `self.norm` drifts
    /// from the true covolume-derived norm and downstream
    /// operations (notably `smallest_equiv_narrow`) reject valid
    /// δ. Calling `refresh_norm` after construction restores
    /// agreement with the lattice.
    ///
    /// `W` is the working width for the determinants; pick it so
    /// that `W · 64 ≥ 4 · bits(max basis entry) + 5`. Returns
    /// `None` if the computed index is not a perfect square (a
    /// bug in construction), or if the index or its square root
    /// fails to narrow back to `BigInt<N>`.
    pub fn refresh_norm<const W: usize>(&mut self) -> Option<()> {
        const { assert!(W >= N, "refresh_norm: W must be >= N") };

        let widen_mat = |m: &Matrix<N>| -> Matrix<W> {
            let mut out = Matrix::<W>::ZERO;
            for r in 0..4 {
                for c in 0..4 {
                    out[r][c] = m[r][c].widen::<W>();
                }
            }
            out
        };

        let i_basis_w = widen_mat(self.lattice.basis());
        let i_denom_w: BigInt<W> = self.lattice.denom().widen();
        let o_basis_w = widen_mat(self.parent_order.basis());
        let o_denom_w: BigInt<W> = self.parent_order.denom().widen();

        let det_i = i_basis_w.det();
        let det_o = o_basis_w.det();

        let o_denom_sq = o_denom_w.ct_mul(&o_denom_w);
        let o_denom_4 = o_denom_sq.ct_mul(&o_denom_sq);
        let i_denom_sq = i_denom_w.ct_mul(&i_denom_w);
        let i_denom_4 = i_denom_sq.ct_mul(&i_denom_sq);

        // Signed `num` and `den` may swap sign based on lattice
        // basis orientation. `[O:I]` is an absolute-value quantity,
        // so take absolute values before the divisibility check.
        let num = o_denom_4.ct_mul(&det_i).abs();
        let den = i_denom_4.ct_mul(&det_o).abs();

        if bool::from(den.is_zero()) {
            #[cfg(test)]
            eprintln!(
                "[refresh_norm] denominator zero: i_denom bits={}, det_o bits={}, o_denom bits={}, det_i bits={}",
                i_denom_w.bitsize(),
                det_o.bitsize(),
                o_denom_w.bitsize(),
                det_i.bitsize(),
            );
            return None;
        }
        let (index, rem) = num.div_rem(&den);
        if !bool::from(rem.is_zero()) {
            #[cfg(test)]
            eprintln!(
                "[refresh_norm] num/den has remainder: num bits={}, den bits={}, rem bits={}",
                num.bitsize(),
                den.bitsize(),
                rem.bitsize(),
            );
            return None;
        }

        // A degenerate (zero-covolume) lattice is not a valid ideal.
        // Reject explicitly so callers don't inherit `self.norm = 0`
        // and panic on downstream `div_rem` by the stored norm.
        if bool::from(index.is_zero()) {
            #[cfg(test)]
            eprintln!("[refresh_norm] degenerate lattice: [O:I] = 0");
            return None;
        }

        let n_sqrt = index.sqrt_floor();
        // Verify perfect square: n_sqrt² == index.
        if n_sqrt.ct_mul(&n_sqrt) != index {
            #[cfg(test)]
            {
                let sqr = n_sqrt.ct_mul(&n_sqrt);
                let diff = index.ct_sub(&sqr);
                eprintln!(
                    "[refresh_norm] index is not a perfect square: index bits={}, sqrt_floor bits={}, index-sqrt² bits={}, det_i bits={}, det_o bits={}, i_denom bits={}, o_denom bits={}",
                    index.bitsize(),
                    n_sqrt.bitsize(),
                    diff.bitsize(),
                    det_i.bitsize(),
                    det_o.bitsize(),
                    i_denom_w.bitsize(),
                    o_denom_w.bitsize(),
                );
            }
            return None;
        }

        self.norm = n_sqrt.narrow_to::<N>()?;
        Some(())
    }

    /// Compute the inverse ideal I⁻¹ = (1/nrd(I)) · Ī.
    ///
    /// Returns the conjugate lattice scaled by 1/nrd(I). Used for
    /// pushforward: `[J]_* I = J⁻¹(J ∩ I)`.
    ///
    /// See [§3.1.6.1] (Ideal inverse) of the spec.
    ///
    /// [§3.1.6.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.6.1
    pub fn inverse(&self) -> HnfLattice<N> {
        let mut conj = self.lattice.conjugate();
        // Scale by 1/nrd(I) — multiply the denominator by nrd(I).
        conj.denom = conj.denom.ct_mul(&self.norm);
        conj
    }

    /// Compute the right order O_R(I) = (1/nrd(I)) · Ī · I.
    ///
    /// The right order of a left ideal I is the set {α ∈ B : Iα ⊆ I}.
    /// For a left O-ideal, O_R(I) is a maximal order isomorphic to
    /// End(E_I) under the Deuring correspondence.
    ///
    /// See [§3.1.5.1] (Right order) of the spec.
    ///
    /// [§3.1.5.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.1
    #[must_use]
    pub fn right_order(&self) -> Order<N> {
        // O_R(I) = I⁻¹ · I = (1/nrd(I)) · Ī · I
        let i_inv = self.inverse();
        let product = i_inv.product(self.lattice());
        Order::from_lattice_unchecked(Lattice::from(product))
    }

    /// Pushforward of an ideal: `[J]_* I = J⁻¹(J ∩ I)`.
    ///
    /// Given `self = J` and `other = I` (with coprime norms),
    /// computes the pushforward ideal. The result has norm `nrd(I)`
    /// and left order `O_R(J)` (provided by `right_order_j`).
    ///
    /// See [§3.1.6.1] (Pushforward and pullback of ideals) of the spec.
    ///
    /// [§3.1.6.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.6.1
    pub fn pushforward(&self, other: &Self, right_order_j: &Order<N>) -> Self {
        let j_inter_i = self.lattice.intersection(&other.lattice);
        let j_inv = self.inverse();
        let result_lattice = j_inv.product(&j_inter_i);

        Self {
            lattice: result_lattice,
            norm: *other.norm(),
            parent_order: *right_order_j,
        }
    }
}

impl<const N: usize> Copy for LeftIdeal<N> where BigInt<N>: Copy {}

// Methods requiring Element<4>::mul() / norm() (widen to BigInt<8>).
impl<const N: usize> LeftIdeal<N> {
    /// Create the left ideal I = O⟨α, N⟩ = Oα + ON.
    ///
    /// Uses [`Element::mul_direct`] at width `N`, which performs the
    /// quaternion multiplication without widening. This is correct
    /// iff the intermediate products fit in `BigInt<N>` — roughly,
    /// `2 * bits(max coordinate of α or order basis) + bits(p) ≤ 64 * N`.
    /// For the NIST-I response phase at `N = 22`, coordinates are
    /// bounded by ~2^575, products are ~2^1150, and the
    /// `p · (c² + d²)` term tops out near ~2^1400 — just within
    /// `BigInt<22>` (1408 bits).
    ///
    /// For callers working at narrow widths on small moduli, the
    /// `LeftIdeal<4>::new` inherent version (which widens internally
    /// to `BigInt<8>`) remains available.
    ///
    /// See [§3.1.6.1] of the spec.
    ///
    /// [§3.1.6.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.6.1
    pub fn from_generator(alpha: &Element<N>, norm: &BigInt<N>, order: &Order<N>) -> Self {
        // Compute Oα: multiply each basis element of O by α.
        let mut o_alpha_cols = [Vector::<N>::ZERO; 4];
        for (j, o_alpha_col) in o_alpha_cols.iter_mut().enumerate() {
            let basis_j = order.basis_elem(j);
            let product = basis_j.mul_direct(alpha);
            *o_alpha_col = Vector::new(
                *product.a.as_bigint(),
                *product.b.as_bigint(),
                *product.c.as_bigint(),
                *product.d.as_bigint(),
            );
        }
        let o_alpha_denom = order.denom().ct_mul(alpha.denom.as_bigint());
        let o_alpha = Lattice::new(Matrix::from_columns(&o_alpha_cols), o_alpha_denom);

        // Compute ON: scale each basis vector of O by N.
        let mut o_n_cols = order.basis().columns();
        for col in &mut o_n_cols {
            for row in 0..4 {
                col[row] = col[row].ct_mul(norm);
            }
        }
        let o_n = Lattice::new(Matrix::from_columns(&o_n_cols), *order.denom());

        Self {
            lattice: o_alpha.sum(&o_n),
            norm: *norm,
            parent_order: *order,
        }
    }
}

impl LeftIdeal<30> {
    /// Construct a left ideal `I = O⟨α, N⟩` at storage width
    /// `N = 30` via modular HNF, avoiding the classical HNF
    /// coefficient blow-up that corrupts the generic
    /// [`from_generator`][Self::from_generator] path at this
    /// width.
    ///
    /// This is the response-phase analogue of the construction
    /// path used by [`random_prime_norm_wide`][Self::random_prime_norm_wide]
    /// for the commitment ideal. At `N = 30` the incoming
    /// generator `α_rsp` from the sampling step has coordinates
    /// up to ≈ 2^1400 bits; the classical HNF inside
    /// [`Lattice::sum`] will silently overflow on products of
    /// these entries, whereas [`Lattice::sum_mod`] bounds every
    /// intermediate by the per-call modulus
    /// `D = 4 · d⁴ · norm² · p`.
    ///
    /// Because the modulus depends on `norm` (which varies per
    /// signature — it is `q_rsp · D_MIX` with `q_rsp` sampled
    /// each iteration), it is computed at call time rather than
    /// precomputed as a const.
    ///
    /// # Returns
    ///
    /// `None` if `α.denom` differs from `order.denom()` (which
    /// would put the two sub-lattices at mismatched denominators
    /// and force a rescale that is incompatible with the chosen
    /// `sum_mod` width budget). The sole current caller
    /// (`sign_derand`'s response phase) always passes `α` with
    /// denom `1`, and the order is `O₀` with denom `2`, so the
    /// denoms match by construction. The check is defensive.
    pub fn from_generator_mod_hnf(
        alpha: &Element<30>,
        norm: &BigInt<30>,
        order: &Order<30>,
    ) -> Option<Self> {
        // Compute Oα: multiply each basis element of O by α.
        let mut o_alpha_cols = [Vector::<30>::ZERO; 4];
        for (j, o_alpha_col) in o_alpha_cols.iter_mut().enumerate() {
            let basis_j = order.basis_elem(j);
            let product = basis_j.mul_direct(alpha);
            *o_alpha_col = Vector::new(
                *product.a.as_bigint(),
                *product.b.as_bigint(),
                *product.c.as_bigint(),
                *product.d.as_bigint(),
            );
        }
        let o_alpha_denom = order.denom().ct_mul(alpha.denom.as_bigint());
        let o_alpha = Lattice::new(Matrix::from_columns(&o_alpha_cols), o_alpha_denom);

        // Compute ON: scale each basis vector of O by N.
        // Use the same denominator as Oα (= order_denom * α_denom)
        // so that sum_mod can combine them. Scaling ON's integer
        // basis by α_denom preserves the lattice: ON/d_O =
        // ON·α_d / (d_O·α_d) = ON·α_d / o_alpha_denom.
        let alpha_d = *alpha.denom.as_bigint();
        let mut o_n_cols = order.basis().columns();
        for col in &mut o_n_cols {
            for row in 0..4 {
                col[row] = col[row].ct_mul(norm).ct_mul(&alpha_d);
            }
        }
        let o_n = Lattice::new(Matrix::from_columns(&o_n_cols), o_alpha_denom);

        // Mod-HNF bounding modulus `D = 4 · d⁴ · norm² · p`.
        //
        // For the sign response phase, `norm = q_rsp · D_MIX`
        // with `q_rsp ≤ 2^126` and `D_MIX ≈ 2^513`, so
        // `norm ≲ 2^640` and `D ≲ 2^(2 + 4 + 1280 + 256) =
        // 2^1542`. This fits comfortably in `BigInt<30>` (1920
        // bits).
        let p_wide: BigInt<30> = {
            let p8: BigInt<8> = crate::quaternions::precomputed::P_WIDE;
            let mut limbs = [0u64; 30];
            limbs[..8].copy_from_slice(p8.as_limbs());
            BigInt::from_sign_and_limbs(0, limbs)
        };
        // With the shared denom = order_denom · α_denom, the
        // modulus needs to account for both: D = 4·d_total⁴·norm²·p.
        let d_total = o_alpha_denom;
        let d_sq = d_total.ct_mul(&d_total);
        let d_fourth = d_sq.ct_mul(&d_sq);
        let norm_sq = norm.ct_mul(norm);
        let four = BigInt::<30>::from_u64(4);
        let modulus = four.ct_mul(&d_fourth).ct_mul(&norm_sq).ct_mul(&p_wide);

        let lattice = o_alpha.sum_mod::<60>(&o_n, &modulus)?;

        Some(Self {
            lattice,
            norm: *norm,
            parent_order: *order,
        })
    }
}

impl LeftIdeal<4> {
    /// Create the left ideal I = O⟨α, N⟩ = Oα + ON at width 4.
    ///
    /// Uses [`Element<4>::mul`] which widens to `BigInt<8>`
    /// internally, so this is the safe choice for small moduli
    /// (up to ~128 bits) where direct multiplication would overflow.
    /// For wider widths, use
    /// [`LeftIdeal::from_generator`](LeftIdeal::from_generator).
    ///
    /// See [§3.1.6.1] of the spec.
    ///
    /// [§3.1.6.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.6.1
    pub fn new(alpha: &Element<4>, norm: &BigInt<4>, order: &Order<4>) -> Self {
        // Compute Oα: multiply each basis element of O by α, at
        // `BigInt<8>` throughout.
        //
        // `mul_direct` (no GCD normalization) is used instead of
        // `mul` because the post-sum HNF needs every column to be
        // expressed at the SAME `denom`. `Element::mul` normalizes
        // each product independently — if different products reduce
        // by different GCDs, the stored numerators end up at
        // inconsistent scales while `o_alpha_denom` is computed
        // uniformly as `order.denom · alpha.denom`. The mismatch
        // silently corrupts the lattice. `mul_direct` keeps every
        // product's denom at exactly `order.denom · alpha.denom`
        // with numerators scaled accordingly.
        //
        // Width safety: product coordinates reach `|p · c · d| ≤
        // 2^250 · 2^N · 2^N` for α of coord magnitude ~2^N. For
        // narrow-path callers (`N < 2^127`), the product fits in
        // `BigInt<8>` (512 bits) with margin: 2^(250+127+127) =
        // 2^504 < 2^512. `BigInt<4>` (256 bits) would overflow
        // already at N > 2^3.
        // Widen α and the order's basis to BigInt<12> (= 768 bits).
        //
        // Width-12 is required for KAT-shaped α: every NIST-I KAT
        // secret-ideal generator has |coord| in [2^134, 2^140] (see
        // `survey_kat_secret_ideal_coord_magnitudes`). The largest
        // intermediate is `|p · c · d| ≈ 2^(250 + 140 + 140) = 2^530`
        // and the modulus `64 · N² · p · denom²` ≈ 2^538 — both
        // overflow BigInt<8> (= 512 bits). Width 12 (= 768 bits)
        // gives ~230 bits of headroom and absorbs the post-HNF
        // canonicalization without truncation.
        let alpha_w = Element::<12>::new(
            Coordinate::from_bigint(alpha.a.as_bigint().widen::<12>()),
            Coordinate::from_bigint(alpha.b.as_bigint().widen::<12>()),
            Coordinate::from_bigint(alpha.c.as_bigint().widen::<12>()),
            Coordinate::from_bigint(alpha.d.as_bigint().widen::<12>()),
            Denominator::from_bigint_unchecked(alpha.denom.as_bigint().widen::<12>()),
        );
        let order_basis_cols_4 = order.basis().columns();
        let widen_col_4_to_w = |col: &Vector<4>| -> Vector<12> {
            Vector::new(
                col[0].widen::<12>(),
                col[1].widen::<12>(),
                col[2].widen::<12>(),
                col[3].widen::<12>(),
            )
        };
        let order_denom_w: BigInt<12> = order.denom().widen();

        let mut o_alpha_cols_w = [Vector::<12>::ZERO; 4];
        for (j, col) in o_alpha_cols_w.iter_mut().enumerate() {
            let basis_col_w = widen_col_4_to_w(&order_basis_cols_4[j]);
            let basis_j_w = Element::<12>::new(
                Coordinate::from_bigint(basis_col_w[0]),
                Coordinate::from_bigint(basis_col_w[1]),
                Coordinate::from_bigint(basis_col_w[2]),
                Coordinate::from_bigint(basis_col_w[3]),
                Denominator::from_bigint_unchecked(order_denom_w),
            );
            let product = basis_j_w.mul_direct(&alpha_w);
            *col = Vector::new(
                *product.a.as_bigint(),
                *product.b.as_bigint(),
                *product.c.as_bigint(),
                *product.d.as_bigint(),
            );
        }
        let o_alpha_denom_w = order_denom_w.ct_mul(&alpha.denom.as_bigint().widen::<12>());

        // Compute ON: scale each basis vector of O by N, at `BigInt<12>`.
        let norm_w: BigInt<12> = norm.widen();
        let mut o_n_cols_w: [Vector<12>; 4] =
            core::array::from_fn(|j| widen_col_4_to_w(&order_basis_cols_4[j]));
        for col in &mut o_n_cols_w {
            for row in 0..4 {
                col[row] = col[row].ct_mul(&norm_w);
            }
        }
        let o_n_denom_w = order_denom_w;

        // Use [`Lattice::sum_mod`] at `BigInt<12>` to avoid
        // coefficient blow-up inside the XGCD pivot reduction of
        // classical HNF. The wider variant
        // [`LeftIdeal::random_prime_norm_wide`] uses `sum_mod::<44>`
        // at much larger ideal sizes; the narrow path needs only
        // enough headroom to absorb the modulus and its squared
        // products.
        //
        // `sum_mod` requires the two lattices to share a denom.
        // `o_alpha_denom = order.denom · alpha.denom` (may be
        // larger than 1 for alpha.denom > 1), while
        // `o_n_denom = order.denom`. When they're equal (the
        // common case for alpha.denom = 1), sum directly. When
        // different, scale to a common denom.
        let scale_cols_in_place_w = |cols: &mut [Vector<12>; 4], s: &BigInt<12>| {
            for col in cols.iter_mut() {
                for row in 0..4 {
                    col[row] = col[row].ct_mul(s);
                }
            }
        };
        let (common_denom_w, o_alpha_cols_w, o_n_cols_w) = if o_alpha_denom_w == o_n_denom_w {
            (o_alpha_denom_w, o_alpha_cols_w, o_n_cols_w)
        } else {
            let mut o_a = o_alpha_cols_w;
            let mut o_b = o_n_cols_w;
            scale_cols_in_place_w(&mut o_a, &o_n_denom_w);
            scale_cols_in_place_w(&mut o_b, &o_alpha_denom_w);
            (o_alpha_denom_w.ct_mul(&o_n_denom_w), o_a, o_b)
        };
        let o_alpha_w = Lattice::<12>::new(Matrix::from_columns(&o_alpha_cols_w), common_denom_w);
        let o_n_w = Lattice::<12>::new(Matrix::from_columns(&o_n_cols_w), common_denom_w);
        // Modulus `64 · N² · p · common_denom²` ≈ 2^538 for KAT-shaped
        // α (norm ≈ 2^140, p ≈ 2^250, denom ≤ 2). Width 12 holds it
        // with margin; sum_mod's working width 24 holds the squared
        // intermediates xgcd produces during HNF reduction.
        let modulus_w: BigInt<12> = {
            let n_w: BigInt<12> = norm.widen();
            let n_sq = n_w.ct_mul(&n_w);
            let p_w: BigInt<12> = crate::quaternions::precomputed::P_WIDE.widen::<12>();
            let denom_sq = common_denom_w.ct_mul(&common_denom_w);
            BigInt::<12>::from_u64(64)
                .ct_mul(&n_sq)
                .ct_mul(&p_w)
                .ct_mul(&denom_sq)
        };
        let lattice_w = o_alpha_w
            .sum_mod::<24>(&o_n_w, &modulus_w)
            .expect("denoms share common_denom_w by construction");

        // Canonicalize: compute the GCD of every basis entry and
        // the denom, then divide through. The common-denom
        // rescaling above multiplied the denom by
        // `o_n_denom_8 = order.denom`, which leaves a factor of 2
        // (or similar) shared across every basis column and the
        // denom. Without this GCD step, the lattice is represented
        // at 2× its minimal denom, making `basis[0] / denom` come
        // out as `7/2` instead of `7/1` (the latter being an
        // actual O_0 element) and the HNF diagonal encoding the
        // same lattice at a coarser grain. See the analogous step
        // in `smallest_equiv` (lattice.rs, shortly after `hnf_8`).
        let basis_cols_w = lattice_w.basis().columns();
        let denom_w = lattice_w.denom();
        let mut g: BigInt<12> = denom_w.abs();
        for col in &basis_cols_w {
            for row in 0..4 {
                if !bool::from(col[row].is_zero()) {
                    g = g.gcd(&col[row].abs());
                }
            }
        }
        // Narrow back to `HnfLattice<4>`. The HNF entries of a
        // proper ideal with `nrd ≤ N^2` fit in `BigInt<4>` for
        // `N < 2^128`; `narrow_to` returns `None` if this fails, which
        // indicates either a miscomputation upstream or a caller
        // passing α with coordinates outside `[−N, N)`.
        let basis_4 = {
            let mut m = Matrix::<4>::ZERO;
            for (j, col) in basis_cols_w.iter().enumerate() {
                for row in 0..4 {
                    let (q, _) = col[row].div_rem(&g);
                    m[row][j] = q
                        .narrow_to::<4>()
                        .expect("LeftIdeal<4>::new basis entry does not fit in BigInt<4>");
                }
            }
            m
        };
        let denom_4 = {
            let (q, _) = denom_w.div_rem(&g);
            q.narrow_to::<4>()
                .expect("LeftIdeal<4>::new denom does not fit in BigInt<4>")
        };
        let lattice = HnfLattice::from(Lattice::new(basis_4, denom_4));

        #[cfg(test)]
        if std::env::var("LEFTIDEAL_NEW_TRACE").is_ok() {
            eprintln!(
                "[LIN4] α coord bits: a={}, b={}, c={}, d={} | norm bits={}",
                alpha.a.as_bigint().abs().bitsize(),
                alpha.b.as_bigint().abs().bitsize(),
                alpha.c.as_bigint().abs().bitsize(),
                alpha.d.as_bigint().abs().bitsize(),
                norm.bitsize(),
            );
            eprintln!(
                "[LIN4] o_alpha col0 (Oα · 1) bits: a={}, b={}, c={}, d={}",
                o_alpha_cols_w[0][0].abs().bitsize(),
                o_alpha_cols_w[0][1].abs().bitsize(),
                o_alpha_cols_w[0][2].abs().bitsize(),
                o_alpha_cols_w[0][3].abs().bitsize(),
            );
            eprintln!(
                "[LIN4] common_denom_w bits={}, modulus_w bits={}, sum_lattice denom_w bits={}",
                common_denom_w.bitsize(),
                modulus_w.bitsize(),
                lattice_w.denom().bitsize(),
            );
            eprintln!("[LIN4] gcd g (final divisor) bits = {}", g.bitsize());
            for (i, col) in lattice.basis().columns().iter().enumerate() {
                eprintln!(
                    "[LIN4] HNF col{i} bits: [{}, {}, {}, {}]",
                    col[0].abs().bitsize(),
                    col[1].abs().bitsize(),
                    col[2].abs().bitsize(),
                    col[3].abs().bitsize(),
                );
            }
            eprintln!("[LIN4] HNF denom = {}", denom_4);
        }

        Self {
            lattice,
            norm: *norm,
            parent_order: *order,
        }
    }

    /// Construct a random left ideal of a given prime norm.
    ///
    /// [Alg. 3.10] from the spec (prime case).
    ///
    /// WARNING: Not constant-time — brute-force search with
    /// data-dependent Legendre symbol and modular sqrt.
    ///
    /// TODO(ct): Make constant-time before production use. The norm
    /// argument may be secret-derived during signing (Algorithm 4.2
    /// line 23, where the norm depends on α_rsp).
    ///
    /// [Alg. 3.10]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.10
    pub fn random_prime_norm(n: &BigInt<4>, order: &ExtremalOrder<4>) -> Option<Self> {
        // Algorithm 3.10, prime case: sample γ = g₁i + g₂j + g₃ij
        // with g₁, g₂, g₃ uniform in [0, N-1], check Legendre
        // symbol, then adjust with modular sqrt.
        let n_bits = n.bitsize() as usize;
        let n_bytes = n_bits.div_ceil(8);

        for _ in 0..10_000 {
            // Sample g₁, g₂, g₃ uniform in [0, N-1] via rejection.
            let sample_mod_n = || -> BigInt<4> {
                loop {
                    let mut bytes = [0u8; 32];
                    OsRng.fill_bytes(&mut bytes[..n_bytes]);
                    // Mask top byte to avoid bias.
                    if n_bits % 8 != 0 {
                        bytes[n_bytes - 1] &= (1u8 << (n_bits % 8)) - 1;
                    }
                    let val = BigInt::<4>::from_bytes_le_unsigned(&bytes[..n_bytes]);
                    // Reject if val >= N.
                    if val.bitsize() <= n.bitsize() && val.ct_mod(n) == val {
                        return val; // val < N
                    }
                }
            };

            let g1 = sample_mod_n();
            let g2 = sample_mod_n();
            let g3 = sample_mod_n();

            // γ = g₁i + g₂j + g₃ij  (a = 0, denom = 1)
            let gamma = Element::<4>::new(
                Coordinate::ZERO,
                Coordinate::from_bigint(g1),
                Coordinate::from_bigint(g2),
                Coordinate::from_bigint(g3),
                Denominator::ONE,
            );
            let (nrd_num, nrd_den) = gamma.norm();

            // Narrow norm to BigInt<4>.
            let nrd_num_4: subtle::CtOption<BigInt<4>> = nrd_num.into();
            let nrd_den_4: subtle::CtOption<BigInt<4>> = nrd_den.into();
            if !bool::from(nrd_num_4.is_some()) || !bool::from(nrd_den_4.is_some()) {
                continue;
            }
            let nrd_num_4 = nrd_num_4.unwrap();
            let nrd_den_4 = nrd_den_4.unwrap();
            let (nrd_val, rem) = nrd_num_4.div_rem(&nrd_den_4);
            if !bool::from(rem.is_zero()) {
                continue;
            }

            // Check Legendre(-nrd(γ), N) = 1.
            let neg_nrd = n.ct_sub(&nrd_val.ct_mod(n));
            if BigInt::<4>::legendre(&neg_nrd, n) != 1 {
                continue;
            }

            // γ ← γ + √(-nrd(γ)) mod N
            let sqrt = match BigInt::<4>::modular_sqrt(&neg_nrd, n) {
                Some(s) => s,
                None => continue,
            };
            let gamma_adjusted = Element::<4>::new(
                Coordinate::from_bigint(sqrt),
                Coordinate::from_bigint(g1),
                Coordinate::from_bigint(g2),
                Coordinate::from_bigint(g3),
                Denominator::ONE,
            );

            return Some(Self::new(&gamma_adjusted, n, order.order()));
        }

        None
    }

    /// Construct a random left ideal of a given (not necessarily prime) norm.
    ///
    /// [Alg. 3.10][Alg. 3.10] from the spec (non-prime case).
    /// Uses [`ExtremalOrder::represent_integer`] to find γ with
    /// nrd(γ) = m·N, then samples random β with gcd(nrd(β), N) = 1.
    ///
    /// # Divergences
    ///
    /// The `gcd(nrd(β), N) = 1` check runs at `BigInt<8>` rather
    /// than narrowing down to `BigInt<4>` first. An earlier version
    /// of this function narrowed `nrd(β) ≈ p·N² ≈ 2^505` (for
    /// response-phase `N ≈ 2^126`) into `BigInt<4>`, which always
    /// failed silently and rejected every one of the 10,000
    /// samples. `random_norm` then always returned `None`, signing
    /// quietly ran out of its 1000-iteration retry budget, and
    /// failed as `SigningFailed`. See the paper's `§Bugs from Fixed
    /// Width Arithmetic -> random_norm narrows nrd(β) into
    /// BigInt<4>` entry.
    ///
    /// WARNING: Not constant-time.
    ///
    /// TODO(ct): Make constant-time before production use.
    ///
    /// [Alg. 3.10]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.10
    pub fn random_norm<R: RngCore>(
        n: &BigInt<4>,
        order: &ExtremalOrder<4>,
        rng: &mut R,
    ) -> Option<Self> {
        // m = QUAT_prime_cofactor (precomputed prime ≈ p)
        let m4 = crate::params::QUAT_PRIME_COFACTOR;
        let m = BigInt::<8>::from_limbs({
            let mut limbs = [0u64; 8];
            limbs[..4].copy_from_slice(m4.as_limbs());
            limbs
        });

        // Line 10: γ ← GeneralizedRepresentInteger(mN, i, O₀, false)
        let n_wide = BigInt::<8>::from_limbs({
            let mut limbs = [0u64; 8];
            limbs[..4].copy_from_slice(n.as_limbs());
            limbs
        });
        let mn = m.ct_mul(&n_wide);
        let order_wide = ExtremalOrder::<8>::from(*order);
        let gamma = order_wide.represent_integer(&mn, false, rng)?;

        // Lines 11-14: sample β = x + yi + zj + wij with gcd(nrd(β), N) = 1
        let n_bits = n.bitsize() as usize;
        let n_bytes = n_bits.div_ceil(8);

        let mut sample_in_range = || -> BigInt<4> {
            loop {
                let mut bytes = [0u8; 32];
                rng.fill_bytes(&mut bytes[..n_bytes]);
                if n_bits % 8 != 0 {
                    bytes[n_bytes - 1] &= (1u8 << (n_bits % 8)) - 1;
                }
                let val = BigInt::<4>::from_bytes_le_unsigned(&bytes[..n_bytes]);
                // Ensure val in [1, N]: reject 0 and val >= N.
                if bool::from(val.is_zero()) || val.ct_mod(n) != val {
                    continue;
                }
                return val;
            }
        };

        for _ in 0..10_000 {
            let x = sample_in_range();
            let y = sample_in_range();
            let z = sample_in_range();
            let w = sample_in_range();

            let beta = Element::<4>::new(
                Coordinate::from_bigint(x),
                Coordinate::from_bigint(y),
                Coordinate::from_bigint(z),
                Coordinate::from_bigint(w),
                Denominator::ONE,
            );

            // Check gcd(nrd(β), N) = 1. `nrd(β) = x² + y² + p(z² +
            // w²)` is about `p · N² ≈ 2^505` for N ~ 2^126, so it
            // doesn't narrow to `BigInt<4>`. Compute the gcd at
            // `BigInt<8>` against a widened N, then narrow the
            // (always small) gcd back to check for 1. An earlier
            // version of this function narrowed nrd before the
            // gcd check, which silently rejected every sample for
            // any N > ~2^64 and made `random_norm` return `None`
            // after 10_000 futile iterations.
            let (nrd_num, nrd_den) = beta.norm();
            let (nrd_val_wide, rem) = nrd_num.div_rem(&nrd_den);
            if !bool::from(rem.is_zero()) {
                continue;
            }
            let n_wide: BigInt<8> = n.widen();
            let gcd_wide = nrd_val_wide.gcd(&n_wide);
            if gcd_wide != BigInt::<8>::ONE {
                continue;
            }

            // Line 15: J' ← ideal generated by γβ and N
            //
            // `Element<4>::mul` would silently truncate: γ has
            // coords ~√(m·N) ≈ 2^129, β has coords < N ≈ 2^8, and
            // the quaternion product components reach ~p · 2^137 ≈
            // 2^388 — far beyond `BigInt<4>`'s 256-bit budget. In
            // release builds `from_wide` just drops the top limbs,
            // producing an `α` with wrong nrd and hence an invalid
            // O_0-ideal.
            //
            // Fix: compute γ·β at `Element<8>`, then reduce each
            // coordinate modulo `N`. The ideal `O·α + O·N` is
            // unchanged by `α → α mod N` because `N · Z<1,i,j,k> ⊂
            // N · O_0 = O · N` (since `O_0 ⊃ Z<1,i,j,k>`), so any
            // coordinate shift by a multiple of N lives in `O·N`
            // and is absorbed by the sum. The reduced α has
            // coords `< N`, trivially fitting in `BigInt<4>`.
            let widen_elem_4_to_8 = |e: &Element<4>| -> Element<8> {
                Element::<8>::new(
                    Coordinate::from_bigint(e.a.as_bigint().widen::<8>()),
                    Coordinate::from_bigint(e.b.as_bigint().widen::<8>()),
                    Coordinate::from_bigint(e.c.as_bigint().widen::<8>()),
                    Coordinate::from_bigint(e.d.as_bigint().widen::<8>()),
                    Denominator::from_bigint_unchecked(e.denom.as_bigint().widen::<8>()),
                )
            };
            let gamma_8 = widen_elem_4_to_8(&gamma);
            let beta_8 = widen_elem_4_to_8(&beta);
            let gamma_beta_8 = gamma_8.mul_direct(&beta_8);

            // Reduce each numerator coord mod `N · denom`. For
            // α = (a, b, c, d) / denom, subtracting `k · N · denom`
            // from `a` changes α by `k · N`, which lives in
            // `N · Z<1,i,j,k> ⊂ N · O_0 = O · N` and is absorbed
            // by the sum `O · α + O · N`. Reducing mod `N` alone
            // (without the `· denom` factor) would leave a
            // half-integer residue for `denom = 2` and push α out
            // of O_0 entirely.
            let n_times_denom_8 = n.widen::<8>().ct_mul(gamma_beta_8.denom.as_bigint());
            let reduce_coord = |c: &BigInt<8>| -> BigInt<4> {
                let r = c.ct_mod(&n_times_denom_8);
                r.narrow_to::<4>()
                    .expect("coord reduced mod N·denom fits in BigInt<4>")
            };
            let denom_4 = gamma_beta_8
                .denom
                .as_bigint()
                .narrow_to::<4>()
                .expect("product denom = γ.denom·β.denom fits in BigInt<4>");
            let gamma_beta = Element::<4>::new(
                Coordinate::from_bigint(reduce_coord(gamma_beta_8.a.as_bigint())),
                Coordinate::from_bigint(reduce_coord(gamma_beta_8.b.as_bigint())),
                Coordinate::from_bigint(reduce_coord(gamma_beta_8.c.as_bigint())),
                Coordinate::from_bigint(reduce_coord(gamma_beta_8.d.as_bigint())),
                Denominator::from_bigint_unchecked(denom_4),
            );

            // Re-check the ideal-norm coprimality post-reduction.
            //
            // `γβ` had `nrd = m · N · nrd(β)` with `gcd(m · nrd(β),
            // N) = 1` by the earlier check, so `gcd(nrd(γβ)/N, N) =
            // 1` held for the *un-reduced* product. Reducing each
            // coordinate mod `N·denom` preserves `N | nrd(α)` but
            // shifts `nrd(α)/N` by an arbitrary integer `M` —
            // `new_nrd/N = old_nrd/N + M`. For composite `N`, `M`
            // has ~`Σ 1/p_i` probability of landing `new_nrd/N`
            // into a residue with a common factor with some `p_i |
            // N`. When that happens the constructed lattice is a
            // valid ideal of norm `N / gcd`, not `N`, and the
            // stored norm `self.norm = N` is wrong. Verify the
            // invariant holds before committing; otherwise
            // resample.
            let (nrd_num_4, nrd_den_4) = gamma_beta.norm();
            let (nrd_val_4, rem_nrd) = nrd_num_4.div_rem(&nrd_den_4);
            if !bool::from(rem_nrd.is_zero()) {
                continue;
            }
            // `nrd(α) ≤ 4·(N·denom)² + p·... ≲ 2^{260}`, so `nrd/N ≤
            // 2^{256}` — within `BigInt<8>`'s 512-bit budget.
            let (nrd_over_n, rem_n) = nrd_val_4.div_rem(&n_wide);
            if !bool::from(rem_n.is_zero()) {
                // Should not occur — `mod-reduction` preserves `N |
                // nrd`. Skip defensively.
                continue;
            }
            let coprime_check: BigInt<8> = nrd_over_n.gcd(&n_wide);
            if coprime_check != BigInt::<8>::ONE {
                continue;
            }

            // Verify the constructed lattice really is an
            // `O_0`-ideal of norm `N`: every basis column of the
            // HNF must have `nrd` divisible by `N · denom²`. This
            // is a stronger invariant than the coprimality check
            // above — certain `α` pass `gcd(nrd(α)/N, N) = 1` yet
            // produce an HNF whose (1,1)-block or similar row
            // slots a lattice element outside `O_0·α + O_0·N`
            // (probably due to width/denom handling in
            // `LeftIdeal::<4>::new`'s `sum_mod<16>`). Until the
            // underlying construction is fully bulletproof for
            // every `α`, re-verify each sample and resample on
            // failure.
            let candidate = Self::new(&gamma_beta, n, order.order());
            let cand_lat: Lattice<4> = (*candidate.lattice()).into();
            let cand_denom = *cand_lat.denom();
            let cand_denom_sq = cand_denom.ct_mul(&cand_denom);
            let n_times_denom_sq = n.ct_mul(&cand_denom_sq);
            let n_times_denom_sq_8: BigInt<8> = n_times_denom_sq.widen();
            let p4 = crate::quaternions::precomputed::P;
            let mut valid = true;
            for j in 0..4 {
                let col = cand_lat.basis().columns()[j];
                let nrd_col_4 = col[0]
                    .ct_mul(&col[0])
                    .ct_add(&col[1].ct_mul(&col[1]))
                    .ct_add(&p4.ct_mul(&col[2].ct_mul(&col[2]).ct_add(&col[3].ct_mul(&col[3]))));
                let nrd_col_8: BigInt<8> = nrd_col_4.widen();
                let (_, rem_col) = nrd_col_8.div_rem(&n_times_denom_sq_8);
                if !bool::from(rem_col.is_zero()) {
                    valid = false;
                    break;
                }
            }
            if !valid {
                continue;
            }

            return Some(candidate);
        }

        None
    }

    /// Find a primitive generator γ of this ideal.
    ///
    /// [Alg. 3.8] from the spec.
    ///
    /// WARNING: Not constant-time — bounded brute-force search with
    /// data-dependent norm checks and GCD.
    ///
    /// TODO(ct): Make constant-time before production use. Called on
    /// secret-derived ideals via IdealToKernel during signing.
    ///
    /// [Alg. 3.8]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.8
    pub fn generator(&self) -> Option<Element<4>> {
        let basis = self.lattice.basis();
        let n_i = &self.norm;

        const MAX_NORM: i64 = 1000;
        let mut n: i64 = 0;
        while n < MAX_NORM {
            n += 1;
            let mut a = -n;
            while a <= n {
                let rem_a = n - a.abs();
                let mut b = -rem_a;
                while b <= rem_a {
                    let rem_b = rem_a - b.abs();
                    let mut c = -rem_b;
                    while c <= rem_b {
                        let d = rem_b - c.abs();
                        for &d_val in &[d, -d] {
                            if a.abs() + b.abs() + c.abs() + d_val.abs() != n {
                                continue;
                            }

                            let a_big = BigInt::<4>::from_i64(a);
                            let b_big = BigInt::<4>::from_i64(b);
                            let c_big = BigInt::<4>::from_i64(c);
                            let d_big = BigInt::<4>::from_i64(d_val);

                            let g = a_big
                                .abs()
                                .gcd(&b_big.abs())
                                .gcd(&c_big.abs())
                                .gcd(&d_big.abs());
                            if g != BigInt::ONE {
                                continue;
                            }

                            let mut gamma_coords = [BigInt::<4>::ZERO; 4];
                            for row in 0..4 {
                                gamma_coords[row] = a_big
                                    .ct_mul(&basis[row][0])
                                    .ct_add(&b_big.ct_mul(&basis[row][1]))
                                    .ct_add(&c_big.ct_mul(&basis[row][2]))
                                    .ct_add(&d_big.ct_mul(&basis[row][3]));
                            }
                            let gamma = Element::<4>::new(
                                Coordinate::from_bigint(gamma_coords[0]),
                                Coordinate::from_bigint(gamma_coords[1]),
                                Coordinate::from_bigint(gamma_coords[2]),
                                Coordinate::from_bigint(gamma_coords[3]),
                                Denominator::from_bigint_unchecked(*self.lattice.denom()),
                            );

                            let (nrd_num, nrd_den) = gamma.norm();
                            // Widen n_i to BigInt<8> for division.
                            let n_i_wide: BigInt<8> = (*n_i).into();
                            let (q, rem) = nrd_num.div_rem(&nrd_den.ct_mul(&n_i_wide));
                            if !bool::from(rem.is_zero()) {
                                continue;
                            }
                            if q.gcd(&n_i_wide) == BigInt::<8>::ONE {
                                return Some(gamma);
                            }
                        }
                        c += 1;
                    }
                    b += 1;
                }
                a += 1;
            }
        }
        None
    }

    // KernelToIdeal (Algorithm 3.17) is defined as
    // TorsionBasis::kernel_to_ideal() in curves/mod.rs.
}

impl LeftIdeal<30> {
    /// Construct a random left ideal of a given prime norm (wide version).
    ///
    /// For the commitment phase (Algorithm 4.2 line 4), the norm D_MIX
    /// = 2^512 + 75 is 513 bits. This method stores the resulting
    /// ideal at `BigInt<30>` (1920 bits) so that:
    /// - Column entries `p·g_i ≈ 2^769` fit without truncation.
    /// - The downstream [`reduce_to_prime_norm`] gram computation `c^T·G·c ≈
    ///   2^1806` fits without overflow.
    ///
    /// After construction, call [`reduce_to_prime_norm`] to get a
    /// small prime norm, then [`narrow`] to convert to `LeftIdeal<4>`
    /// for [`to_isogeny`].
    ///
    /// [Alg. 3.10][Alg. 3.10] from the spec (prime case).
    ///
    /// WARNING: Not constant-time.
    ///
    /// [Alg. 3.10]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.10
    pub fn random_prime_norm_wide<R: RngCore>(
        n: &BigInt<30>,
        order: &ExtremalOrder<4>,
        rng: &mut R,
    ) -> Option<Self> {
        let p_wide: BigInt<30> = {
            let p8: BigInt<8> = crate::quaternions::precomputed::P_WIDE;
            let mut limbs = [0u64; 30];
            limbs[..8].copy_from_slice(p8.as_limbs());
            BigInt::from_sign_and_limbs(0, limbs)
        };

        let zero_big = BigInt::<30>::ZERO;
        let one_big = BigInt::<30>::ONE;
        let n_minus_one = n.ct_sub(&one_big);

        for _ in 0..10_000 {
            // Phase A: trace-zero quaternion γ = a + g₁·i + g₂·j +
            // g₃·k with nrd(γ) ≡ 0 (mod N). Sample (g₁, g₂, g₃) ∈
            // [0, N − 1] via [`BigInt::rand_interval`] (matches
            // C ref's `ibz_rand_interval(0, n−1)` byte-for-byte),
            // compute disc = −nrd mod N, and recover a via
            // `sqrt mod N` after a Legendre check.
            let g1 = BigInt::<30>::rand_interval(rng, &zero_big, &n_minus_one);
            let g2 = BigInt::<30>::rand_interval(rng, &zero_big, &n_minus_one);
            let g3 = BigInt::<30>::rand_interval(rng, &zero_big, &n_minus_one);

            // nrd(γ) = g₁² + p(g₂² + g₃²) for γ = g₁i + g₂j + g₃ij
            // in the quaternion algebra B_{p,∞} = (-1, -p). With
            // g_i < 2^513 and p ≈ 2^256 the result is ≈ 2^1282 bits,
            // well within `BigInt<30>` (1920 bits).
            let g1_sq = g1.ct_mul(&g1);
            let g2_sq = g2.ct_mul(&g2);
            let g3_sq = g3.ct_mul(&g3);
            let nrd = g1_sq.ct_add(&p_wide.ct_mul(&g2_sq.ct_add(&g3_sq)));

            let nrd_mod = nrd.ct_mod(n);
            let neg_nrd = n.ct_sub(&nrd_mod);

            // Check Legendre(-nrd(γ), N) = 1. The `_w::<30>` variants
            // keep the primality/sqrt arithmetic at the storage width
            // (well above the 1026-bit `pow_mod` requirement).
            if BigInt::<30>::legendre_w::<30>(&neg_nrd, n) != 1 {
                continue;
            }

            // a = √(-nrd(γ)) mod N.
            let a = match BigInt::<30>::modular_sqrt_w::<30>(&neg_nrd, n) {
                Some(s) => s,
                None => continue,
            };

            #[cfg(test)]
            if std::env::var("SAMPID_TRACE").is_ok() {
                eprintln!("[SAMPID] phaseA gen.coord=[{}, {}, {}, {}]", a, g1, g2, g3);
            }

            // Phase B: rerandomize the principal ideal class by
            // sampling δ = (d₀, d₁, d₂, d₃) with gcd(nrd(δ), N) = 1
            // and replacing γ ← γ · δ. Mirrors C ref's
            // `quat_sampling_random_ideal_O0_given_norm`
            // (`normeq.c:297-384`). Without this step the resulting
            // ideal lattice differs from C ref's by a multiplicative
            // δ-twist, so the downstream `reduce_to_prime_norm`
            // basis (and every byte after it) diverges.
            let delta_coords: Option<[BigInt<30>; 4]> = (0..1000).find_map(|_| {
                let d0 = BigInt::<30>::rand_interval(rng, &one_big, n);
                let d1 = BigInt::<30>::rand_interval(rng, &one_big, n);
                let d2 = BigInt::<30>::rand_interval(rng, &one_big, n);
                let d3 = BigInt::<30>::rand_interval(rng, &one_big, n);
                let nrd_d = d0
                    .ct_mul(&d0)
                    .ct_add(&d1.ct_mul(&d1))
                    .ct_add(&p_wide.ct_mul(&d2.ct_mul(&d2).ct_add(&d3.ct_mul(&d3))));
                let nrd_d_mod = nrd_d.ct_mod(n);
                if nrd_d_mod.gcd(n) == BigInt::<30>::ONE {
                    Some([d0, d1, d2, d3])
                } else {
                    None
                }
            });
            let Some([d0, d1, d2, d3]) = delta_coords else {
                continue;
            };

            #[cfg(test)]
            if std::env::var("SAMPID_TRACE").is_ok() {
                eprintln!(
                    "[SAMPID] phaseB rerand.coord=[{}, {}, {}, {}]",
                    d0, d1, d2, d3
                );
            }

            // γ · δ via [`Element::mul_direct`] at width 30.
            // Inputs use ≤ 9 limbs (513 bits); products fit the
            // N/2 = 15-limb precondition. Output coords reach
            // ~2^1285 (still well within 30 limbs).
            let gamma_elem = Element::<30>::new(
                Coordinate::from_bigint(a),
                Coordinate::from_bigint(g1),
                Coordinate::from_bigint(g2),
                Coordinate::from_bigint(g3),
                Denominator::from_bigint_unchecked(one_big),
            );
            let delta_elem = Element::<30>::new(
                Coordinate::from_bigint(d0),
                Coordinate::from_bigint(d1),
                Coordinate::from_bigint(d2),
                Coordinate::from_bigint(d3),
                Denominator::from_bigint_unchecked(one_big),
            );
            let new_gen = gamma_elem.mul_direct(&delta_elem);
            let a = *new_gen.a.as_bigint();
            let g1 = *new_gen.b.as_bigint();
            let g2 = *new_gen.c.as_bigint();
            let g3 = *new_gen.d.as_bigint();

            #[cfg(test)]
            if std::env::var("SAMPID_TRACE").is_ok() {
                eprintln!(
                    "[SAMPID] mult.coord=[{}, {}, {}, {}] denom={}",
                    a,
                    g1,
                    g2,
                    g3,
                    new_gen.denom.as_bigint()
                );
            }

            // Construct I = O₀⟨γ, N⟩ as a lattice.
            //
            // Precompute the 4 products of basis quaternions with γ:
            //   1·γ = ( a,    g₁,   g₂,   g₃)
            //   i·γ = (-g₁,   a,   -g₃,   g₂)
            //   j·γ = (-pg₂,  pg₃,  a,   -g₁)
            //   k·γ = (-pg₃, -pg₂,  g₁,   a )
            //
            // For B_{p,∞} = (-1,-p): i²=-1, j²=-p, k=ij. These
            // `pg_i` products reach ≈ 2^769 and required the 1920-bit
            // storage width.
            let pg2 = p_wide.ct_mul(&g2);
            let pg3 = p_wide.ct_mul(&g3);
            let prod_1 = [a, g1, g2, g3];
            let prod_i = [g1.wrapping_neg(), a, g3.wrapping_neg(), g2];
            let prod_j = [pg2.wrapping_neg(), pg3, a, g1.wrapping_neg()];
            let prod_k = [pg3.wrapping_neg(), pg2.wrapping_neg(), g1, a];

            // For each order basis element e = (e₀,e₁,e₂,e₃)/denom,
            // compute e·γ = (e₀·(1·γ) + e₁·(i·γ) + e₂·(j·γ) + e₃·(k·γ))/denom.
            let order_wide = ExtremalOrder::<30>::from(*order);
            let order_lat = order_wide.order();
            let order_denom = *order_lat.denom();

            let mut o_alpha_cols = [Vector::<30>::ZERO; 4];
            for (col, o_alpha_col) in o_alpha_cols.iter_mut().enumerate() {
                let e = [
                    order_lat.basis()[0][col],
                    order_lat.basis()[1][col],
                    order_lat.basis()[2][col],
                    order_lat.basis()[3][col],
                ];
                for row in 0..4 {
                    o_alpha_col[row] = e[0]
                        .ct_mul(&prod_1[row])
                        .ct_add(&e[1].ct_mul(&prod_i[row]))
                        .ct_add(&e[2].ct_mul(&prod_j[row]))
                        .ct_add(&e[3].ct_mul(&prod_k[row]));
                }
            }

            // O₀·N: scale each order basis column by N.
            let mut o_n_cols = [Vector::<30>::ZERO; 4];
            for (col, o_n_col) in o_n_cols.iter_mut().enumerate() {
                for row in 0..4 {
                    o_n_col[row] = order_lat.basis()[row][col].ct_mul(n);
                }
            }

            // I = O₀·γ + O₀·N, as a Z-lattice sum.
            //
            // We use [`Lattice::sum_mod`] rather than the classical
            // [`Lattice::sum`] because at `BigInt<30>` the
            // classical HNF path inside `sum` suffers from
            // coefficient blow-up: intermediate xgcd products
            // exceed the 1920-bit storage budget and silently
            // truncate, collapsing the basis to something unrelated
            // to the intended ideal. Modular HNF bounds every
            // intermediate by the precomputed NIST-I constant
            // [`D_HNF_MODULUS_COMMITMENT`][crate::quaternions::precomputed::D_HNF_MODULUS_COMMITMENT]
            // = 4 · d⁴ · D_MIX² · p, keeping everything within the
            // `W = 44` working width. See the rustdoc on
            // [`Matrix::from_hnf_columns_mod`] and the "Fixed-Precision HNF"
            // rationale (spec gap: Algorithm 3.2 doesn't discuss
            // fixed-precision adaptations).
            let o_alpha = Lattice::new(Matrix::from_columns(&o_alpha_cols), order_denom);
            let o_n = Lattice::new(Matrix::from_columns(&o_n_cols), order_denom);

            // `o_alpha` and `o_n` share `order_denom`, so
            // `sum_mod` returns `Some` by construction here; the
            // `?` is defensive against any future refactor that
            // changes one of the two denominators.
            let lattice = o_alpha.sum_mod::<44>(
                &o_n,
                &crate::quaternions::precomputed::D_HNF_MODULUS_COMMITMENT,
            )?;
            return Some(LeftIdeal {
                lattice,
                norm: *n,
                parent_order: *order_lat,
            });
        }

        None
    }
}

impl<const N: usize> LeftIdeal<N> {
    /// Narrow a wide `LeftIdeal<N>` to `LeftIdeal<4>` after norm reduction.
    ///
    /// After [`reduce_to_prime_norm`], the norm is a small prime and
    /// the HNF basis entries are bounded. This converts the wide
    /// representation to the narrow one needed by [`to_isogeny`].
    ///
    /// Returns `None` if any entry doesn't fit in `BigInt<4>`.
    pub fn narrow(&self) -> Option<LeftIdeal<4>> {
        self.narrow_to::<4>()
    }

    /// Generic narrow: convert a wide `LeftIdeal<N>` to `LeftIdeal<M>`
    /// when all coordinates fit in `BigInt<M>`. Returns `None`
    /// otherwise.
    pub fn narrow_to<const M: usize>(&self) -> Option<LeftIdeal<M>> {
        let narrow_int = |v: &BigInt<N>| -> Option<BigInt<M>> { v.narrow_to::<M>() };

        let mut basis_m = [[BigInt::<M>::ZERO; 4]; 4];
        let basis_n = self.lattice().basis();
        for row in 0..4 {
            for col in 0..4 {
                basis_m[row][col] = narrow_int(&basis_n[row][col])?;
            }
        }

        let norm_m = narrow_int(self.norm())?;
        let denom_m = narrow_int(self.lattice().denom())?;

        // Narrow the parent order.
        let order_n = self.parent_order();
        let mut order_basis_m = [[BigInt::<M>::ZERO; 4]; 4];
        let order_basis_n = order_n.basis();
        for row in 0..4 {
            for col in 0..4 {
                order_basis_m[row][col] = narrow_int(&order_basis_n[row][col])?;
            }
        }
        let order_denom_m = narrow_int(order_n.denom())?;

        let to_matrix = |rows: [[BigInt<M>; 4]; 4]| -> Matrix<M> {
            Matrix::from_rows(
                Vector::new(rows[0][0], rows[0][1], rows[0][2], rows[0][3]),
                Vector::new(rows[1][0], rows[1][1], rows[1][2], rows[1][3]),
                Vector::new(rows[2][0], rows[2][1], rows[2][2], rows[2][3]),
                Vector::new(rows[3][0], rows[3][1], rows[3][2], rows[3][3]),
            )
        };

        Some(LeftIdeal {
            lattice: HnfLattice {
                basis: to_matrix(basis_m),
                denom: denom_m,
            },
            norm: norm_m,
            parent_order: Order::from_lattice_unchecked(Lattice {
                basis: to_matrix(order_basis_m),
                denom: order_denom_m,
            }),
        })
    }

    /// Widen this ideal to `LeftIdeal<M>` by zero-extending all
    /// `BigInt<N>` limbs in the lattice basis, norm, denominator, and
    /// parent order.
    ///
    /// `M` must be `>= N`, enforced at compile time.
    ///
    /// Used in signing to lift `LeftIdeal<4>` (typical secret/challenge
    /// ideals) and `LeftIdeal<9>` (the commitment ideal at D_mix width)
    /// to a common wide type for the response-phase lattice arithmetic.
    #[must_use]
    pub fn widen<const M: usize>(&self) -> LeftIdeal<M> {
        const { assert!(M >= N, "LeftIdeal::widen: M must be >= N") };

        // Widen the HNF lattice.
        let basis_n = self.lattice().basis();
        let mut basis_m = Matrix::<M>::ZERO;
        for row in 0..4 {
            for col in 0..4 {
                basis_m[row][col] = basis_n[row][col].widen::<M>();
            }
        }
        let lat_wide = HnfLattice {
            basis: basis_m,
            denom: self.lattice().denom().widen::<M>(),
        };

        // Widen the parent order.
        let order_n = self.parent_order();
        let order_basis_n = order_n.basis();
        let mut order_basis_m = Matrix::<M>::ZERO;
        for row in 0..4 {
            for col in 0..4 {
                order_basis_m[row][col] = order_basis_n[row][col].widen::<M>();
            }
        }
        let order_wide = Order::from_lattice_unchecked(Lattice::new(
            order_basis_m,
            order_n.denom().widen::<M>(),
        ));

        LeftIdeal {
            lattice: lat_wide,
            norm: self.norm().widen::<M>(),
            parent_order: order_wide,
        }
    }
}

impl<const N: usize> LeftIdeal<N>
where
    [u64; N]: Default,
{
    /// Replace this ideal with an equivalent one of prime norm.
    ///
    /// Samples random short elements α in the ideal's L2-reduced
    /// lattice until nrd(α) / N(I) is prime, then sets
    /// I ← I · ᾱ / N(I).
    ///
    /// Implements [RandomEquivalentPrimeIdeal][Alg. 3.9].
    ///
    /// # Width requirement
    ///
    /// The const generic `PRIME_W` is the working width for the
    /// Miller-Rabin primality check on the candidate norm. For
    /// correctness, `64 * PRIME_W >= 2 * bits(norm)` — otherwise
    /// [`BigInt::pow_mod`] inside Miller-Rabin silently truncates
    /// and rejects genuine primes. Callers should pass `PRIME_W >=
    /// ceil(bits(norm) / 32)`.
    ///
    /// For the SQIsign v2 commitment phase (`LeftIdeal<9>`, norms
    /// up to ~2^513), pass `PRIME_W = 18`.
    ///
    /// WARNING: Not constant-time.
    ///
    /// TODO(ct): Make constant-time before production use.
    ///
    /// [Alg. 3.9]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.9
    pub fn reduce_to_prime_norm<const PRIME_W: usize, R: RngCore>(&mut self, rng: &mut R) -> bool {
        const {
            assert!(
                PRIME_W >= N,
                "reduce_to_prime_norm: PRIME_W must be >= storage width N"
            )
        };
        let bound = crate::params::EQUIV_BOUND_COEFF;
        let primality_rounds = crate::params::PRIMALITY_NUM_ITER;
        // Step 1: Compute the *class gram* and L2-reduce.
        //
        // The C reference (`quat_lideal_class_gram`) divides the
        // raw reduced-norm bilinear form by `d² · N(I)` before
        // LLL. This is critical for fixed-precision: raw gram
        // entries can reach ~2^2820 bits for our commitment
        // ideals (quadratic in the ~2^1282-bit HNF entries, plus
        // a factor of p), overflowing `BigInt<30>`. The class
        // gram entries are bounded by `det_class^{1/2}` after
        // LLL, which is ~2^126 for NIST-I commitment ideals —
        // well within budget.
        //
        // After dividing, `c^T · G_class · c` directly equals
        // the equivalent-ideal norm `m = nrd(α) / (d² · N(I))`,
        // eliminating the two separate divisibility checks in the
        // sampling loop.
        //
        // # Divergences
        //
        // The SQIsign spec (Algorithm 3.9) describes sampling
        // with the raw reduced-norm form. The class-gram
        // transformation is a C-reference implementation detail
        // that we adopt for the same reason: it keeps
        // intermediates bounded for fixed-precision arithmetic.
        //
        // Canonicalize the HNF first: mod-HNF produces valid HNF
        // but with off-diagonal entries up to the modulus (~2^1282).
        // Classical HNF canonicalization reduces off-diagonals modulo
        // the diagonal pivots, bringing entries to ~2^127 — the same
        // range GMP's HNF produces for the C reference. Without
        // this, gram entries reach ~2^514 and the DPE GSO's 53-bit
        // mantissa can't handle the cancellation.
        let canonical = self.lattice.canonicalize();
        let basis = canonical.basis();
        let cols = basis.columns();
        let nrd = NrdBasis::new(cols);

        let denom = self.lattice.denom();
        let denom_sq = denom.ct_mul(denom);
        let class_divisor = denom_sq.ct_mul(&self.norm);

        // The C reference's `quat_lattice_gram` computes the
        // *trace* bilinear form T(b_i, b_j) = 2·nrd_bilinear,
        // which is exactly divisible by d²·N(I) for every pair
        // of ideal-lattice columns. Our nrd Gram computes the
        // reduced-norm bilinear form (no factor of 2), so we
        // multiply by 2 before dividing to get the class gram.
        let two = BigInt::<N>::from_u64(2);
        let mut class_gram = Matrix::<N>::ZERO;
        for i in 0..4 {
            for j in 0..4 {
                let traced = nrd.gram()[i][j].ct_mul(&two);
                let (q, _rem) = traced.div_rem(&class_divisor);
                class_gram[i][j] = q;
            }
        }

        let class_basis = NrdBasis::from_cols_and_gram(*nrd.cols(), class_gram).l2_reduce();

        // Step 2: sample random short vectors until m is prime.
        //
        // After LLL on the class gram, `c^T · G_class · c`
        // directly equals `m = nrd(α_elt) / N(I)` — the
        // equivalent-ideal norm. No further division needed.
        let limit = (2 * i64::from(bound) + 1).pow(4);
        for _ in 0..limit {
            let c: [BigInt<N>; 4] = [
                BigInt::from_i64(Self::rand_interval(rng, bound)),
                BigInt::from_i64(Self::rand_interval(rng, bound)),
                BigInt::from_i64(Self::rand_interval(rng, bound)),
                BigInt::from_i64(Self::rand_interval(rng, bound)),
            ];

            // Evaluate class quadratic form.
            // G_class = 2·nrd_bilinear / (d²·N), so
            // c^T·G_class·c = 2·nrd(α_int) / (d²·N) = 2·m.
            // Divide by 2 to get m.
            let m = class_basis.eval_quadratic_form(&c).shr(1);

            if m.is_probable_prime_w::<PRIME_W>(primality_rounds) {
                // Reconstruct α = Σ c_i · col_i in the reduced basis.
                let mut alpha = [BigInt::<N>::ZERO; 4];
                for (i, c_i) in c.iter().enumerate() {
                    for (k, alpha_k) in alpha.iter_mut().enumerate() {
                        *alpha_k = alpha_k.ct_add(&c_i.ct_mul(&class_basis.cols()[i][k]));
                    }
                }

                // Conjugate α: negate the i, j, k coordinates.
                alpha[1] = alpha[1].wrapping_neg();
                alpha[2] = alpha[2].wrapping_neg();
                alpha[3] = alpha[3].wrapping_neg();

                // Quaternion mul in B_{p,∞} = (-1, -p) at BigInt<N> width.
                let p_n: BigInt<N> = {
                    let p8: BigInt<8> = crate::quaternions::precomputed::P_WIDE;
                    let mut limbs = [0u64; N];
                    let src = p8.as_limbs();
                    let len = src.len().min(N);
                    limbs[..len].copy_from_slice(&src[..len]);
                    BigInt::from_sign_and_limbs(0, limbs)
                };
                let qmul = |a: &[BigInt<N>; 4], b: &[BigInt<N>; 4]| -> [BigInt<N>; 4] {
                    let (a0, a1, a2, a3) = (&a[0], &a[1], &a[2], &a[3]);
                    let (b0, b1, b2, b3) = (&b[0], &b[1], &b[2], &b[3]);
                    [
                        a0.ct_mul(b0)
                            .ct_sub(&a1.ct_mul(b1))
                            .ct_sub(&p_n.ct_mul(&a2.ct_mul(b2).ct_add(&a3.ct_mul(b3)))),
                        a0.ct_mul(b1)
                            .ct_add(&a1.ct_mul(b0))
                            .ct_add(&p_n.ct_mul(&a2.ct_mul(b3).ct_sub(&a3.ct_mul(b2)))),
                        a0.ct_mul(b2)
                            .ct_add(&a2.ct_mul(b0))
                            .ct_sub(&a1.ct_mul(b3))
                            .ct_add(&a3.ct_mul(b1)),
                        a0.ct_mul(b3)
                            .ct_add(&a3.ct_mul(b0))
                            .ct_add(&a1.ct_mul(b2))
                            .ct_sub(&a2.ct_mul(b1)),
                    ]
                };

                let old_basis = self.lattice.basis();
                let old_cols = old_basis.columns();
                let mut new_cols = [Vector::<N>::ZERO; 4];
                for col_idx in 0..4 {
                    let col = [
                        old_cols[col_idx][0],
                        old_cols[col_idx][1],
                        old_cols[col_idx][2],
                        old_cols[col_idx][3],
                    ];
                    let r = qmul(&col, &alpha);
                    new_cols[col_idx] = Vector::new(r[0], r[1], r[2], r[3]);
                }

                let new_norm = m;

                // Build J = O₀⟨ᾱ, m⟩ directly rather than
                // computing I · ᾱ / N(I) by multiplying the old
                // basis. The old-basis-times-ᾱ approach produces
                // integer columns with denom `d² · N(I)` and
                // entries up to ~2^3200 bits — far too large for
                // `BigInt<N>` storage. Constructing O₀⟨ᾱ, m⟩
                // instead gives entries bounded by
                // `p · max(ᾱ) ≈ 2^1543` (≈ 25 limbs) with
                // denom `d²` (≈ 4), fitting in `BigInt<N>`.
                //
                // The two ideals are the same: both are the unique
                // left O₀-ideal equivalent to I with norm m.

                // Compute O₀·ᾱ: multiply each basis element by ᾱ.
                let order = &self.parent_order;
                let alpha_denom = *denom;
                let mut o_alpha_cols = [Vector::<N>::ZERO; 4];
                for (j, o_col) in o_alpha_cols.iter_mut().enumerate() {
                    let e = [
                        order.basis()[0][j],
                        order.basis()[1][j],
                        order.basis()[2][j],
                        order.basis()[3][j],
                    ];
                    let r = qmul(&e, &alpha);
                    *o_col = Vector::new(r[0], r[1], r[2], r[3]);
                }
                let o_alpha_denom = order.denom().ct_mul(&alpha_denom);

                // Compute O₀·m: scale each basis column by m.
                let mut o_m_cols = order.basis().columns();
                for col in &mut o_m_cols {
                    for row in 0..4 {
                        col[row] = col[row].ct_mul(&new_norm);
                    }
                }
                // o_m's natural denom is `order.denom()`, but we
                // rescale to match `o_alpha_denom` below.

                // Rescale O₀·m to the common denom `d · alpha_denom`
                // (= `o_alpha_denom`). Scale factor = `alpha_denom`.
                for col in &mut o_m_cols {
                    for row in 0..4 {
                        col[row] = col[row].ct_mul(&alpha_denom);
                    }
                }

                // Mod-HNF with modulus `4 · d⁴ · m² · p` (a
                // multiple of the integer-column covolume for the
                // O₀-ideal of norm m with denom `d²`).
                let d4 = {
                    let d2 = o_alpha_denom.ct_mul(&o_alpha_denom);
                    d2.ct_mul(&d2)
                };
                let m_sq = new_norm.ct_mul(&new_norm);
                let four = BigInt::<N>::from_u64(4);
                let modulus = four.ct_mul(&d4).ct_mul(&m_sq).ct_mul(&p_n);

                let all_cols = [
                    o_alpha_cols[0],
                    o_alpha_cols[1],
                    o_alpha_cols[2],
                    o_alpha_cols[3],
                    o_m_cols[0],
                    o_m_cols[1],
                    o_m_cols[2],
                    o_m_cols[3],
                ];
                // Working width for mod-HNF: needs ≥ 2·bits(modulus).
                // Modulus ≈ 4·d⁴·m²·p; for m ≤ 2^520 this is
                // ≈ 2^1304. At N=30 (1920 bits), we use W=44
                // (2816 bits) — the same intermediate width as the
                // commitment path. At N=4, W=44 is generous but
                // harmless.
                let hnf_basis = Matrix::<N>::from_hnf_columns_mod::<44>(&all_cols, &modulus);

                // Canonicalize: divide out gcd(basis_entries, denom),
                // mirroring C ref's `quat_lattice_reduce_denom`. The
                // construction above gives `denom = order.denom · alpha.denom`
                // (typically 4), with every HNF basis entry sharing a factor
                // of 2 — leaving the lattice at twice its minimal denom.
                // Without this step, [`LeftIdeal::generator`] returns an
                // [`Element`] with denom 4 (not 2), and serialized `gen.coord`
                // come out at 2× C ref's KAT secret-key generator bytes
                // (since both impls write `coord` directly without dividing
                // by `denom`). Same pattern as [`LeftIdeal::new`] (just
                // above), kept inline here to avoid widening the basis to
                // a temporary lattice.
                let mut g_denom = o_alpha_denom.abs();
                for row in 0..4 {
                    for col in 0..4 {
                        let entry = hnf_basis[row][col];
                        if !bool::from(entry.is_zero()) {
                            g_denom = g_denom.gcd(&entry.abs());
                        }
                    }
                }
                let mut canonical_basis = hnf_basis;
                for row in 0..4 {
                    for col in 0..4 {
                        let (q, _) = canonical_basis[row][col].div_rem(&g_denom);
                        canonical_basis[row][col] = q;
                    }
                }
                let (canonical_denom, _) = o_alpha_denom.div_rem(&g_denom);

                self.lattice = HnfLattice {
                    basis: canonical_basis,
                    denom: canonical_denom,
                };
                self.norm = new_norm;

                return true;
            }
        }
        false
    }

    /// Sample a uniform random integer in \[−m, m\] via rejection
    /// sampling, mirroring C ref's `ibz_rand_interval_minm_m`
    /// byte-for-byte (`intbig.c:475-552`).
    ///
    /// C ref calls `ibz_rand_interval(rand, 0, 2m)` then subtracts
    /// `m`. With `m = 64` (the only caller's choice), `bmina = 128`
    /// has `len_bits = 8`, so each try draws **1 byte** and rejects
    /// when `val > 128` (~50% rejection, ~2 bytes amortized). An
    /// earlier version used `next_u32()` (4 bytes/call, lower
    /// rejection) which doubled per-iteration byte consumption and
    /// desynced from C ref's DRBG byte stream.
    ///
    /// WARNING: Not constant-time (rejection loop). The bound `m`
    /// is public, so this is acceptable for SQIsign.
    fn rand_interval<R: RngCore>(rng: &mut R, m: u32) -> i64 {
        let zero = BigInt::<4>::ZERO;
        let bound = BigInt::<4>::from_u64(2 * u64::from(m));
        let val = BigInt::<4>::rand_interval(rng, &zero, &bound);
        // `val ∈ [0, 2m]`, subtract `m` to get `[−m, m]`.
        (val.as_limbs()[0] as i64) - i64::from(m)
    }
}

impl<const N: usize> core::fmt::Debug for LeftIdeal<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "LeftIdeal(norm={}, lattice={:?})",
            self.norm, self.lattice
        )
    }
}

// ---------------------------------------------------------------------------
// ExtremalOrder<N>: p-extremal maximal order
// ---------------------------------------------------------------------------

/// A p-extremal maximal order in B_{p,∞}.
///
/// These are maximal orders containing j and a distinguished quadratic
/// subring Z[ω] of small discriminant, such that j and Z[ω] are
/// orthogonal. The element z with z² = -q generates the quadratic
/// subring, and t is an element of norm p orthogonal to z.
///
/// See [§3.1.7.2] of the SQIsign specification.
///
/// [§3.1.7.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.7.2
#[derive(Clone)]
pub struct ExtremalOrder<const N: usize> {
    /// The order.
    order: Order<N>,
    /// Element z with z² = -q (small discriminant).
    z: Element<4>,
    /// Element t with nrd(t) = p, orthogonal to z.
    t: Element<4>,
    /// The absolute value |z²| (a small positive integer).
    q: u32,
}

impl<const N: usize> ExtremalOrder<N> {
    /// Creates an extremal order from typed components.
    ///
    /// The lattice must be a maximal order (closed under multiplication,
    /// contains 1). This is not checked — the lattice is wrapped in
    /// [`Order`] unconditionally. Prefer
    /// [`from_raw_limbs`](ExtremalOrder::from_raw_limbs) for constructing from
    /// precomputed raw data.
    #[inline]
    pub const fn new(order: Lattice<N>, z: Element<4>, t: Element<4>, q: u32) -> Self {
        Self {
            order: Order::from_lattice_unchecked(order),
            z,
            t,
            q,
        }
    }

    /// Returns the maximal order as an [`Order`].
    #[inline]
    pub const fn order(&self) -> &Order<N> {
        &self.order
    }

    /// Returns the element z (z² = -q).
    #[inline]
    pub const fn z(&self) -> &Element<4> {
        &self.z
    }

    /// Returns the element t (nrd(t) = p).
    #[inline]
    pub const fn t(&self) -> &Element<4> {
        &self.t
    }

    /// Returns q = |z²|.
    #[inline]
    pub const fn q(&self) -> u32 {
        self.q
    }
}

impl ExtremalOrder<4> {
    /// Construct from raw sign+limbs data, matching the C reference's
    /// `quat_p_extremal_maximal_order_t` layout.
    ///
    /// # Data format
    ///
    /// All integer values are `(sign, [u64; 4])` where `sign = 0` means
    /// non-negative and `sign = 1` means negative. The `[u64; 4]` array
    /// holds the absolute value in little-endian 64-bit limbs. This
    /// matches GMP's `_mp_size` (sign) + `_mp_d` (limbs) representation
    /// used by the C reference's `quaternion_data.c`.
    ///
    /// - `basis`: 4×4 matrix of the order's lattice basis in HNF, expressed in
    ///   the `{1, i, j, k}` basis. Columns divided by the lattice denominator
    ///   give elements of B_{p,∞}.
    /// - `z`: the element z with z² = −q, as four coordinates `[a, b, c, d]` in
    ///   the `{1, i, j, k}` basis.
    /// - `q`: the absolute value |z²|.
    ///
    /// Both the lattice denominator and the z denominator are deduced
    /// from `basis[0][0]` (the top-left HNF entry), which equals both
    /// for all NIST-I extremal orders. All orders have `t = j`.
    ///
    /// # Divergence from internal `Element` representation
    ///
    /// The z data here stores all four coordinates explicitly, matching
    /// the C reference's `quat_alg_elem_t` (which always has four
    /// coordinates + denominator). In practice, for all NIST-I extremal
    /// orders, z has the form `(0, b, 0, d)/denom` — the `a` and `c`
    /// coordinates are zero. However, this constructor does not assume
    /// that: it passes all four coordinates through to `Element::new`,
    /// so the Sage precomputation script can output z in the same
    /// format as the C reference without special-casing.
    pub const fn from_raw_limbs(
        basis: [[(u64, [u64; 4]); 4]; 4],
        z: [(u64, [u64; 4]); 4],
        q: u32,
    ) -> Self {
        const fn bi(sl: (u64, [u64; 4])) -> BigInt<4> {
            BigInt::from_sign_and_limbs(sl.0, sl.1)
        }
        // Both lattice denom and z denom = basis[0][0] (all 7 NIST-I orders).
        let denom = bi(basis[0][0]);
        Self::new(
            Lattice::new(
                Matrix::from_rows(
                    Vector::new(
                        bi(basis[0][0]),
                        bi(basis[0][1]),
                        bi(basis[0][2]),
                        bi(basis[0][3]),
                    ),
                    Vector::new(
                        bi(basis[1][0]),
                        bi(basis[1][1]),
                        bi(basis[1][2]),
                        bi(basis[1][3]),
                    ),
                    Vector::new(
                        bi(basis[2][0]),
                        bi(basis[2][1]),
                        bi(basis[2][2]),
                        bi(basis[2][3]),
                    ),
                    Vector::new(
                        bi(basis[3][0]),
                        bi(basis[3][1]),
                        bi(basis[3][2]),
                        bi(basis[3][3]),
                    ),
                ),
                denom,
            ),
            Element::new(
                Coordinate::from_bigint(bi(z[0])),
                Coordinate::from_bigint(bi(z[1])),
                Coordinate::from_bigint(bi(z[2])),
                Coordinate::from_bigint(bi(z[3])),
                Denominator::from_bigint_unchecked(denom),
            ),
            Element::J,
            q,
        )
    }
}

impl<const N: usize> Copy for ExtremalOrder<N> where BigInt<N>: Copy {}

impl ExtremalOrder<4> {
    /// Widen to `ExtremalOrder<M>` by zero-extending all `BigInt<4>`
    /// limbs in the lattice basis and denominator to `BigInt<M>`.
    ///
    /// The z and t elements remain at width 4 (they are always small).
    /// Used when lattice arithmetic needs wider intermediates (e.g.,
    /// `LeftIdeal<8>` for `represent_integer`, `LeftIdeal<9>` for
    /// D_MIX commitment).
    #[must_use]
    pub fn widen<const M: usize>(&self) -> ExtremalOrder<M> {
        let basis4 = self.order().basis();
        let denom4 = self.order().denom();

        let mut basis_m = Matrix::<M>::ZERO;
        for row in 0..4 {
            for col in 0..4 {
                basis_m[row][col] = basis4[row][col].widen::<M>();
            }
        }
        let order_lat = Lattice::new(basis_m, denom4.widen::<M>());

        ExtremalOrder::new(order_lat, *self.z(), *self.t(), self.q())
    }
}

impl From<ExtremalOrder<4>> for ExtremalOrder<8> {
    fn from(order: ExtremalOrder<4>) -> Self {
        order.widen()
    }
}

impl From<ExtremalOrder<4>> for ExtremalOrder<9> {
    fn from(order: ExtremalOrder<4>) -> Self {
        order.widen()
    }
}

impl From<ExtremalOrder<4>> for ExtremalOrder<30> {
    fn from(order: ExtremalOrder<4>) -> Self {
        order.widen()
    }
}

impl<const N: usize> core::fmt::Debug for ExtremalOrder<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ExtremalOrder(q={}, z={:?}, t={:?})",
            self.q, self.z, self.t
        )
    }
}

// ---------------------------------------------------------------------------
// NrdBasis: quaternion lattice basis with its reduced-norm Gram matrix
// ---------------------------------------------------------------------------

/// Dimension of quaternion lattices.
const D: usize = 4;

/// A rank-4 quaternion lattice basis paired with its reduced-norm
/// Gram matrix.
///
/// The Gram matrix `G[i][j] = nrd_bilinear(b_i, b_j)` is the
/// inner product of basis vectors under the reduced-norm bilinear
/// form of B_{p,∞} = (−1, −p)_Q:
///
///   `G[i][j] = a_i·a_j + b_i·b_j + p·(c_i·c_j + d_i·d_j)`
///
/// where `(a, b, c, d)` are the `{1, i, j, k}` coordinates.
///
/// The Gram matrix is derived from — and kept in sync with — the
/// columns. It is computed once at construction and updated
/// incrementally during L² reduction.
pub struct NrdBasis<const N: usize> {
    cols: [Vector<N>; D],
    gram: Matrix<N>,
}

impl<const N: usize> NrdBasis<N> {
    /// Construct from column vectors, computing the reduced-norm
    /// Gram matrix.
    pub fn new(cols: [Vector<N>; D]) -> Self {
        let gram = Self::compute_gram(&cols);
        Self { cols, gram }
    }

    /// Compute the reduced-norm Gram matrix for column vectors in
    /// B_{p,∞} = (−1, −p)_Q.
    fn compute_gram(cols: &[Vector<N>; D]) -> Matrix<N> {
        let p: BigInt<N> = {
            let p8: BigInt<8> = crate::quaternions::precomputed::P_WIDE;
            let mut limbs = [0u64; N];
            let src = p8.as_limbs();
            let len = src.len().min(N);
            limbs[..len].copy_from_slice(&src[..len]);
            BigInt::from_sign_and_limbs(0, limbs)
        };
        let mut gram = Matrix::<N>::ZERO;
        for i in 0..D {
            for j in i..D {
                let scalar = cols[i][0]
                    .ct_mul(&cols[j][0])
                    .ct_add(&cols[i][1].ct_mul(&cols[j][1]));
                let jk = cols[i][2]
                    .ct_mul(&cols[j][2])
                    .ct_add(&cols[i][3].ct_mul(&cols[j][3]));
                let val = scalar.ct_add(&p.ct_mul(&jk));
                gram[i][j] = val;
                if i != j {
                    gram[j][i] = val;
                }
            }
        }
        gram
    }

    /// Construct from columns and a precomputed Gram matrix.
    ///
    /// The caller is responsible for ensuring `gram` is the correct
    /// inner-product matrix for `cols`. This exists for callers that
    /// need a non-standard inner product (e.g., the class gram after
    /// division by `d²·N(I)`, or the dual gram).
    pub fn from_cols_and_gram(cols: [Vector<N>; D], gram: Matrix<N>) -> Self {
        Self { cols, gram }
    }

    /// Returns the basis columns.
    #[inline]
    pub fn cols(&self) -> &[Vector<N>; D] {
        &self.cols
    }

    /// Returns the Gram matrix.
    #[inline]
    pub fn gram(&self) -> &Matrix<N> {
        &self.gram
    }

    /// Evaluate the quadratic form `c^T · G · c`.
    #[must_use]
    pub fn eval_quadratic_form(&self, c: &[BigInt<N>; D]) -> BigInt<N> {
        let mut result = BigInt::<N>::ZERO;
        for i in 0..D {
            for j in 0..D {
                result = result.ct_add(&c[i].ct_mul(&c[j]).ct_mul(&self.gram[i][j]));
            }
        }
        result
    }

    /// L² reduction with DPE-based GSO ([Alg. 3.3]).
    ///
    /// Reduces the basis in place, keeping the Gram matrix in sync.
    /// Uses [`DoublePlusExponent`](dpe::DoublePlusExponent) (double-precision with extended
    /// exponent) for the Gram-Schmidt coefficients, matching the C
    /// reference's approach. The basis and Gram updates remain exact
    /// (integer). Size-reduction rounding uses
    /// [`DoublePlusExponent::to_bigint`](dpe::DoublePlusExponent::to_bigint) to convert the float
    /// μ back to an integer coefficient, which handles values that
    /// exceed `i64` range (e.g., μ[3][0] ≈ 2^260 before first
    /// reduction).
    ///
    /// # Precision requirement
    ///
    /// DPE has a 53-bit mantissa. This is sufficient when the Gram
    /// entries are ≲ 2^127 (i.e., after
    /// [`HnfLattice::canonicalize`] shrinks mod-HNF entries). For
    /// Gram entries ≳ 2^200, the GSO subtraction chain loses too
    /// many bits and size-reduction oscillates. Callers operating
    /// on wide lattices must canonicalize first.
    ///
    /// # Divergences
    ///
    /// The spec ([Alg. 3.3]) describes L² with deep insertion. We
    /// implement deep insertion following the spec's [Alg. 3.6].
    /// The C reference uses the same approach with `dpe_t`.
    ///
    /// [Alg. 3.3]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.3
    /// [Alg. 3.6]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.6
    #[must_use]
    pub fn l2_reduce(mut self) -> Self {
        /// L² reduction parameter η (size-reduction threshold).
        /// Following the spec: η = 0.51 (any value in (1/2, 1) works).
        const ETA: f64 = 0.51;

        /// L² reduction parameter δ.
        const DELTA: f64 = 0.99;

        let eta_bar = (ETA + 0.5) / 2.0;
        let delta_bar = (DELTA + 1.0) / 2.0;

        fn extend_gso_family<const N: usize>(
            gram: &Matrix<N>,
            k: usize,
            r: &mut [[DoublePlusExponent; D]; D],
            mu: &mut [[DoublePlusExponent; D]; D],
        ) {
            for j in 0..=k {
                r[k][j] = DoublePlusExponent::from_bigint(&gram[k][j]);
                for l in 0..j {
                    r[k][j] -= r[k][l] * mu[j][l];
                }
                if j < k {
                    mu[k][j] = r[k][j] / r[j][j];
                }
            }
        }

        fn size_reduce<const N: usize>(
            basis: &mut [Vector<N>; D],
            gram: &mut Matrix<N>,
            k: usize,
            r: &mut [[DoublePlusExponent; D]; D],
            mu: &mut [[DoublePlusExponent; D]; D],
            eta_bar: f64,
        ) {
            loop {
                extend_gso_family(gram, k, r, mu);

                let mut done = true;
                let mut ii = k;
                while ii > 0 {
                    ii -= 1;
                    if mu[k][ii].abs().to_f64() > eta_bar {
                        done = false;
                        let x_big: BigInt<N> = mu[k][ii].to_bigint();

                        if bool::from(x_big.is_zero()) {
                            continue;
                        }

                        // b_k ← b_k - x · b_ii
                        let old_bi = basis[ii];
                        for row in 0..D {
                            basis[k][row] = basis[k][row].ct_sub(&x_big.ct_mul(&old_bi[row]));
                        }

                        // Update Gram matrix symmetrically.
                        for j in 0..D {
                            let update = x_big.ct_mul(&gram[ii][j]);
                            gram[k][j] = gram[k][j].ct_sub(&update);
                        }
                        for j in 0..D {
                            let update = x_big.ct_mul(&gram[j][ii]);
                            gram[j][k] = gram[j][k].ct_sub(&update);
                        }

                        // Update μ incrementally.
                        let x_dpe = DoublePlusExponent::from_bigint(&x_big);
                        let mu_ii = mu[ii];
                        for l in 0..ii {
                            mu[k][l] -= x_dpe * mu_ii[l];
                        }
                        mu[k][ii] -= x_dpe;

                        // Update r[k][ii] from the updated Gram.
                        r[k][ii] = DoublePlusExponent::from_bigint(&gram[k][ii]);
                        for l in 0..ii {
                            r[k][ii] -= r[k][l] * mu[ii][l];
                        }
                    }
                }

                if done {
                    break;
                }
            }
        }

        fn insert_before<const N: usize>(
            basis: &mut [Vector<N>; D],
            gram: &mut Matrix<N>,
            k: usize,
            s: usize,
            r: &mut [[DoublePlusExponent; D]; D],
            mu: &mut [[DoublePlusExponent; D]; D],
        ) {
            let mut j = k;
            while j > s {
                basis.swap(j, j - 1);

                for row in 0..D {
                    let tmp = gram[row][j];
                    gram[row][j] = gram[row][j - 1];
                    gram[row][j - 1] = tmp;
                }
                for col in 0..D {
                    let tmp = gram[j][col];
                    gram[j][col] = gram[j - 1][col];
                    gram[j - 1][col] = tmp;
                }

                j -= 1;
            }

            r[s][s] = DoublePlusExponent::from_bigint(&gram[s][s]);
            for i in 0..s {
                mu[s][i] = mu[k][i];
                r[s][i] = r[k][i];
                r[s][s] -= mu[s][i] * r[s][i];
            }
        }

        let mut r = [[DoublePlusExponent::ZERO; D]; D];
        let mut mu = [[DoublePlusExponent::ZERO; D]; D];

        r[0][0] = DoublePlusExponent::from_bigint(&self.gram[0][0]);
        mu[0][0] = DoublePlusExponent::from_f64(1.0);

        let mut t = [DoublePlusExponent::ZERO; D];

        let mut k = 1usize;
        while k < D {
            size_reduce(&mut self.cols, &mut self.gram, k, &mut r, &mut mu, eta_bar);

            t[0] = DoublePlusExponent::from_bigint(&self.gram[k][k]);
            for i in 1..=k {
                t[i] = t[i - 1] - mu[k][i - 1] * r[k][i - 1];
            }

            // Deep insertion (C ref's `quat_lll_core` ordering, l2.c:110): iterate
            // s from k down to 1, BREAK on first level where Lovász holds (i.e.,
            // δ̄ · r[s-1][s-1] >= t[s-1]). Resulting `s` is the deepest level
            // such that Lovász fails at all levels in (s, k]. In monotonic cases
            // this equals "smallest j where t[j] < δ̄·r[j][j]"; under DPE rounding
            // a non-monotonic pattern can occur and the iteration order chosen
            // here matches C ref's byte-for-byte.
            let delta_bar_dpe = DoublePlusExponent::from_f64(delta_bar);
            let mut s = k;
            while s > 0 {
                if !(t[s - 1] < delta_bar_dpe * r[s - 1][s - 1]) {
                    break;
                }
                s -= 1;
            }

            if k != s {
                insert_before(&mut self.cols, &mut self.gram, k, s, &mut r, &mut mu);
                k = s;
            }

            k += 1;
        }

        self
    }
}
