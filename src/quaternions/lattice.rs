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

use core::{array, ops::Add};

use rand_core::RngCore;

use super::{
    algebra::{Coordinate, Denominator, Element},
    bigint::BigInt,
    linear::{Matrix, Vector},
};

#[cfg(not(feature = "expose-internals"))]
pub(crate) mod dpe;
#[cfg(feature = "expose-internals")]
pub mod dpe;

mod order;
pub use order::{ExtremalOrder, Order};

mod left_ideal;
pub use left_ideal::LeftIdeal;

mod nrd_basis;
pub use nrd_basis::NrdBasis;

// reason: MLLL engine is a complete sketch but not yet called from
// `Lattice::product` / intersection (the integration step); exercised only by
// its own tests until then. Drop this allow when the callers are wired.
#[allow(dead_code)]
mod mlll;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod intersection_kat1_iter0_tests;

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

    /// Reduces this lattice to Hermite Normal Form.
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
    pub(crate) fn dual(&self) -> Self {
        // L^{-1} = adj(L) / det(L), so L^{-T} = adj(L)^T / det(L).
        // adj(basis)^T is the cofactor matrix, so the dual numerator
        // denom · adj(basis)^T equals denom · cofactor(basis). The
        // cofactor and det share their 2x2 minors, and the scale runs in
        // place, avoiding the separate adjugate and transpose copies (a
        // Matrix<W> is ~16 KB at the response-phase width W = 128).
        let (mut dual_basis, det) = self.basis.cofactor_and_det();

        for row in 0..4 {
            for col in 0..4 {
                let scaled = self.denom.ct_mul(&dual_basis[row][col]);
                dual_basis[row][col] = scaled;
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
    // reason: the body indexes into multiple parallel arrays (basis,
    // augmented identity, scaled cols) at matching positions; the
    // iterator-zip rewrite obscures the linear-algebra correspondence.
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

    /// Lattice intersection via the dual approach:
    /// `L1 ∩ L2 = (L1* + L2*)*`.
    ///
    /// This is the algorithm C-ref's `quat_lattice_intersect`
    /// uses (`lattice.c:127`). It avoids the column-op edge cases
    /// that bite [`Lattice::intersection_via_kernel`] for inputs
    /// with very imbalanced magnitudes (KAT-1 sign iter 2 with
    /// ~520-bit `i_chl_sk` vs ~140-bit `conj(I_com)`).
    ///
    /// The 8-column dual-sum is reduced by modular HNF — the
    /// blow-up-prone step, so `W` must be wide (entry sizes grow
    /// cubically through the two duals). For a max input entry size
    /// of `B` bits and max input denom `D` bits, intermediate
    /// entries reach ~`3·B + max(B+D, 4·B)` bits. Pick `W` to
    /// accommodate. [`Lattice::compact_intersection`] reduces the same
    /// dual-sum with MLLL, bounding the growth so a far smaller `W` suffices.
    ///
    /// Returns `None` if any of the intermediate basis entries or
    /// the final lattice does not narrow back to `BigInt<N>`.
    pub fn intersection_via_dual_sum_dual<const W: usize>(
        &self,
        other: &Self,
    ) -> Option<HnfLattice<N>> {
        // Mirror C-ref's `quat_lattice_add` recipe (`lattice.c:85`): modular
        // HNF mod `gcd(det of each scaled dual basis)` over the 8 columns. The
        // explicit modulus generators drive the canonical pivot gcds, which the
        // non-modular [`Matrix::from_hnf_columns`] does not — it produces a
        // strict superlattice on these inputs (the duals' basis entries
        // ~`denom · adj(B)^T` share large gcds with `det(B)`). See
        // `intersection_kat1_iter0_tests::dsd_step_by_step` for the regression.
        self.dual_sum_dual::<W, _>(other, |cols| {
            let first: [Vector<W>; 4] = [cols[0], cols[1], cols[2], cols[3]];
            let second: [Vector<W>; 4] = [cols[4], cols[5], cols[6], cols[7]];
            let modulus = Matrix::<W>::from_columns(&first)
                .det()
                .abs()
                .gcd(&Matrix::<W>::from_columns(&second).det().abs());
            Matrix::<W>::from_hnf_columns_mod::<W>(&cols, &modulus)
        })
    }

    /// Lattice intersection reducing the 8-column dual-sum with MLLL instead of
    /// HNF — `CompactLatticeIntersection` ([Alg. 3] of ePrint 2026/1031).
    ///
    /// Same `(L1* + L2*)*` identity and canonical [`HnfLattice`] output as
    /// [`Lattice::intersection_via_dual_sum_dual`], but the dominant dual-sum
    /// reduction uses [`mlll`]'s ML2, which keeps intermediate integers bounded
    /// by the largest input norm² ([Lemma 10]) rather than incurring HNF's
    /// coefficient blow-up; the final 4×4 canonicalization is unchanged. MLLL
    /// spans exactly the 8-column Z-module, so the result is identical to the
    /// modular-HNF path — at a far smaller working width `W`.
    ///
    /// Returns `None` if the result does not narrow back to `BigInt<N>`.
    ///
    /// [Alg. 3]: https://eprint.iacr.org/2026/1031.pdf#algorithm.3
    /// [Lemma 10]: https://eprint.iacr.org/2026/1031.pdf#lemma.1.10
    // reason: validated differentially against `intersection_via_dual_sum_dual`
    // but not yet wired into callers; the allow comes off at the migration.
    #[allow(dead_code)]
    pub(crate) fn compact_intersection<const W: usize>(
        &self,
        other: &Self,
    ) -> Option<HnfLattice<N>> {
        self.dual_sum_dual::<W, _>(other, |cols| {
            Matrix::<W>::from_columns(&mlll::Generators::<W, 8>::new(cols).mlll_reduce())
        })
    }

    /// Shared `(L1* + L2*)*` machinery, parameterized by how the 8-column
    /// dual-sum is reduced: modular HNF for
    /// [`intersection_via_dual_sum_dual`](Self::intersection_via_dual_sum_dual),
    /// MLLL for [`compact_intersection`](Self::compact_intersection).
    fn dual_sum_dual<const W: usize, F>(&self, other: &Self, reduce_sum: F) -> Option<HnfLattice<N>>
    where
        F: FnOnce([Vector<W>; 8]) -> Matrix<W>,
    {
        const { assert!(W >= N, "dual_sum_dual: W must be >= N") };

        // Widen self and other to working width.
        let widen_lat = |lat: &Self| -> Lattice<W> {
            let mut basis = Matrix::<W>::ZERO;
            for r in 0..4 {
                for c in 0..4 {
                    basis[r][c] = lat.basis[r][c].widen::<W>();
                }
            }
            Lattice::<W>::new(basis, lat.denom.widen::<W>())
        };

        let l1_w = widen_lat(self);
        let l2_w = widen_lat(other);

        // dual(L1), dual(L2).
        let d1 = l1_w.dual();
        let d2 = l2_w.dual();

        // Bring both duals to the common denominator `d1.denom · d2.denom` by
        // scaling each numerator basis by the other's denom, then hand the 8
        // columns to `reduce_sum`.
        let scale_basis = |basis: &Matrix<W>, s: BigInt<W>| -> Matrix<W> {
            let mut out = Matrix::<W>::ZERO;
            for r in 0..4 {
                for c in 0..4 {
                    out[r][c] = basis[r][c].ct_mul(&s);
                }
            }
            out
        };
        let tmp_a = scale_basis(d1.basis(), *d2.denom());
        let tmp_b = scale_basis(d2.basis(), *d1.denom());
        let all_cols = [
            tmp_a.column(0),
            tmp_a.column(1),
            tmp_a.column(2),
            tmp_a.column(3),
            tmp_b.column(0),
            tmp_b.column(1),
            tmp_b.column(2),
            tmp_b.column(3),
        ];
        let common_denom = d1.denom().ct_mul(d2.denom());
        let sum_basis = reduce_sum(all_cols);
        let sum_lat = Lattice::<W>::new(sum_basis, common_denom);

        // dual of the sum = L1 ∩ L2.
        let result_w = sum_lat.dual();

        // Reduce gcd of basis entries with denom. The double-dual
        // (mathematically self-inverse) leaves a `d^3` factor in basis
        // numerators and `d^4` in the denom; factor it out before narrowing,
        // otherwise the basis entries don't fit in `BigInt<N>`.
        let mut basis_w = result_w.basis;
        let denom_w = result_w.denom;
        let mut g = denom_w.abs();
        for r in 0..4 {
            for c in 0..4 {
                g = g.gcd(&basis_w[r][c].abs());
                if g == BigInt::<W>::ONE {
                    break;
                }
            }
            if g == BigInt::<W>::ONE {
                break;
            }
        }
        let denom_reduced = if g == BigInt::<W>::ONE {
            denom_w
        } else {
            for r in 0..4 {
                for c in 0..4 {
                    let (q, _) = basis_w[r][c].div_rem(&g);
                    basis_w[r][c] = q;
                }
            }
            let (q, _) = denom_w.div_rem(&g);
            q
        };

        let reduced_lat = Lattice::<W> {
            basis: basis_w,
            denom: denom_reduced,
        };
        let result_hnf = reduced_lat.hnf();
        let result_hnf = result_hnf.canonicalize();

        // Second-pass gcd reduction. The HNF and canonicalize
        // may leave a residual common factor between the (now
        // reduced) basis entries and denom that the first pass
        // didn't catch (e.g., when the dual-of-sum's redundant
        // factor distributes unevenly across cols).
        let mut basis2 = result_hnf.basis;
        let denom2 = *result_hnf.denom();
        let mut g2 = denom2.abs();
        for r in 0..4 {
            for c in 0..4 {
                g2 = g2.gcd(&basis2[r][c].abs());
                if g2 == BigInt::<W>::ONE {
                    break;
                }
            }
            if g2 == BigInt::<W>::ONE {
                break;
            }
        }
        let denom_final = if g2 == BigInt::<W>::ONE {
            denom2
        } else {
            for r in 0..4 {
                for c in 0..4 {
                    let (q, _) = basis2[r][c].div_rem(&g2);
                    basis2[r][c] = q;
                }
            }
            let (q, _) = denom2.div_rem(&g2);
            q
        };
        let result_hnf = HnfLattice::<W> {
            basis: basis2,
            denom: denom_final,
        };

        let mut basis_n = Matrix::<N>::ZERO;
        for r in 0..4 {
            for c in 0..4 {
                basis_n[r][c] = result_hnf.basis()[r][c].narrow_to::<N>()?;
            }
        }
        let denom_n: BigInt<N> = result_hnf.denom().narrow_to::<N>()?;

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
    pub(crate) fn sum(&self, other: &Self) -> HnfLattice<N> {
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
                array::from_fn(|idx| {
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

    /// Lattice sum mirroring C-ref's `quat_lattice_add` byte-for-byte.
    ///
    /// Differs from [`Lattice::sum_mod`]:
    ///
    /// 1. Each lattice's basis is scaled by the OTHER's denom before HNF (= the
    ///    cross-scaling C-ref does in `quat_lattice_add`), so the two operands
    ///    need NOT share a denom.
    /// 2. Output denom = `self.denom * other.denom` (NOT just `self.denom`).
    /// 3. Calls [`Matrix::from_hnf_columns_mod_cref`] (decreasing modulus
    ///    algorithm) instead of [`Matrix::from_hnf_columns_mod`] (constant
    ///    modulus). The decreasing-modulus algorithm produces a different
    ///    canonical HNF for inputs whose lattice covolume properly divides
    ///    `modulus`, which IS the typical case in sign's response phase.
    ///    Diagnosed via byte-diff of `i_com_rsp` HNF vs C-ref's
    ///    `lideal_com_resp` for KAT-1 iter 0.
    /// 4. Returns the HNF result without `reduce_denom` — the caller should
    ///    call `reduce_denom` if a canonical-denom representation is needed.
    pub fn sum_mod_cref<const W: usize>(
        &self,
        other: &Self,
        modulus: &BigInt<N>,
    ) -> Option<HnfLattice<N>> {
        // Scale each lattice's basis by the OTHER's denom (per Cref).
        let scale_cols = |basis: &Matrix<N>, s: &BigInt<N>| -> [Vector<N>; 4] {
            let cols = basis.columns();
            array::from_fn(|idx| {
                Vector::new(
                    cols[idx][0].ct_mul(s),
                    cols[idx][1].ct_mul(s),
                    cols[idx][2].ct_mul(s),
                    cols[idx][3].ct_mul(s),
                )
            })
        };

        // Match C-ref `quat_lattice_add` order:
        //   gen[0..4] = lat1.denom · lat2.basis   (= "self.denom · other.basis")
        //   gen[4..8] = lat2.denom · lat1.basis   (= "other.denom · self.basis")
        // C-ref's HNF uses the LAST 4 cols as output cols (per `k = n-1`
        // decrementing), so the second group's basis is the "primary"
        // input and ends up in the output. Matters for byte-equality
        // even though both orderings produce a valid canonical HNF.
        let cols_first = scale_cols(&other.basis, &self.denom);
        let cols_second = scale_cols(&self.basis, &other.denom);
        let all_cols = [
            cols_first[0],
            cols_first[1],
            cols_first[2],
            cols_first[3],
            cols_second[0],
            cols_second[1],
            cols_second[2],
            cols_second[3],
        ];
        Some(HnfLattice {
            basis: Matrix::<N>::from_hnf_columns_mod_cref::<W>(&all_cols, modulus),
            denom: self.denom.ct_mul(&other.denom),
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
    /// # HNF strategy
    ///
    /// Mirrors C ref's `quat_lattice_mul` (`quaternion/ref/generic/
    /// lattice.c:178-216`): form a 4×4 matrix from the first
    /// quaternion product (`alpha_0 · beta_*`, columns 0..4), take
    /// `|det|` as the modular-HNF bound, then run
    /// [`Matrix::from_hnf_columns_mod`] on the full 16 columns at
    /// working width `N`. Classical HNF on 16 columns of large
    /// entries (≈ 600 bits, the multi-order-enumeration `conj(I)·J_t`
    /// shape) overflows fixed-precision; the modular variant bounds
    /// every intermediate by `4·|det|` and stays within a working
    /// width around `4·bits(det)`. For typical SQIsign-shaped inputs
    /// (`bits(det) < N·64/4`) the modular path returns a canonical
    /// HNF; in the rank-deficient case `det = 0` we fall back to
    /// classical HNF (which is well-defined there).
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
                all_cols.push(Vector::new(
                    *product.a.as_bigint(),
                    *product.b.as_bigint(),
                    *product.c.as_bigint(),
                    *product.d.as_bigint(),
                ));
            }
        }

        let new_denom = self.denom.ct_mul(&other.denom);

        let first_block: [Vector<N>; 4] = [all_cols[0], all_cols[1], all_cols[2], all_cols[3]];
        let det_modulus = Matrix::<N>::from_columns(&first_block).det().abs();
        let result_basis = if bool::from(det_modulus.is_zero()) {
            Matrix::from_hnf_columns(&all_cols)
        } else {
            Matrix::from_hnf_columns_mod::<N>(&all_cols, &det_modulus)
        };
        HnfLattice {
            basis: result_basis,
            denom: new_denom,
        }
    }

    /// Lattice product with explicit covolume modulus.
    ///
    /// Same as [`Lattice::product`] but lets the caller supply the
    /// modular-HNF bound directly. Use this when the
    /// `det(first 4 cols)` heuristic in [`Lattice::product`] gives a
    /// modulus that's a *multiple* of the canonical covolume rather
    /// than equal to it: the result then has covolume = modulus
    /// instead of canonical covolume, embedding the lattice in a
    /// coarser sublattice. For ideal-class arithmetic
    /// `conj(reduced_id) · J_t` the canonical covolume formula is
    /// `denom_result^4 · N(result)² / 4`; passing that here returns
    /// the canonical (not coarsened) HNF.
    ///
    /// `modulus` must be a positive multiple of the integer-column
    /// covolume of the result lattice.
    pub fn product_with_modulus(&self, other: &Self, modulus: &BigInt<N>) -> HnfLattice<N> {
        let mut all_cols = Vec::new();

        for i in 0..4 {
            let alpha = self.basis_elem(i);
            for j in 0..4 {
                let beta = other.basis_elem(j);
                let product = alpha.mul_direct(&beta);
                all_cols.push(Vector::new(
                    *product.a.as_bigint(),
                    *product.b.as_bigint(),
                    *product.c.as_bigint(),
                    *product.d.as_bigint(),
                ));
            }
        }

        let new_denom = self.denom.ct_mul(&other.denom);
        let basis = if bool::from(modulus.is_zero()) {
            Matrix::from_hnf_columns(&all_cols)
        } else {
            Matrix::from_hnf_columns_mod::<N>(&all_cols, modulus)
        };
        HnfLattice {
            basis,
            denom: new_denom,
        }
    }

    /// Lattice product via MLLL instead of HNF — `CompactIdealMultiplication`
    /// ([Alg. 2] of ePrint 2026/1031).
    ///
    /// Forms the 16 pairwise products of the two bases and reduces them with
    /// [`mlll`]'s ML2 rather than taking an HNF, yielding an LLL-reduced basis
    /// of the *same* product lattice as [`Lattice::product`] while bounding
    /// intermediate integers by the largest input norm² instead of HNF's nrd⁴.
    /// Unlike `product`, the returned basis is reduced, not canonical HNF.
    ///
    /// [Alg. 2]: https://eprint.iacr.org/2026/1031.pdf#algorithm.2
    // reason: validated differentially against the exact product lattice but
    // not yet wired into callers; the allow comes off when the HNF-based
    // product / intersection call sites migrate to MLLL.
    #[allow(dead_code)]
    #[must_use]
    pub(crate) fn compact_product(&self, other: &Self) -> Self {
        let mut cols = [Vector::<N>::ZERO; 16];
        for i in 0..4 {
            let alpha = self.basis_elem(i);
            for j in 0..4 {
                let prod = alpha.mul_direct(&other.basis_elem(j));
                cols[i * 4 + j] = Vector::new(
                    *prod.a.as_bigint(),
                    *prod.b.as_bigint(),
                    *prod.c.as_bigint(),
                    *prod.d.as_bigint(),
                );
            }
        }

        let reduced = mlll::Generators::<N, 16>::new(cols).mlll_reduce();
        let new_denom = self.denom.ct_mul(&other.denom);

        Self::new(Matrix::from_columns(&reduced), new_denom)
    }

    /// Right-multiply this lattice by a single quaternion element.
    ///
    /// Each basis vector `b_j` of `self` is replaced by `b_j · elem`,
    /// the result is HNF-reduced via classical HNF, and the
    /// denominator becomes `self.denom * elem.denom`. Mirrors C ref's
    /// `quat_lattice_alg_elem_mul`
    /// (`quaternion/ref/generic/lattice.c`).
    ///
    /// The storage width `N` must be wide enough to hold the post-mul
    /// product entries plus classical-HNF intermediate growth. For
    /// the multi-order-enumeration `self · conj(δ)/N(I)` shape
    /// (entries ≈ 530 bits, intermediates ≈ 4×) callers should work
    /// at `N ≥ 50`.
    pub fn alg_elem_mul(&self, elem: &Element<N>) -> HnfLattice<N> {
        let new_cols: [Vector<N>; 4] = array::from_fn(|j| {
            let basis_j = self.basis_elem(j);
            let prod = basis_j.mul_direct(elem);
            Vector::new(
                *prod.a.as_bigint(),
                *prod.b.as_bigint(),
                *prod.c.as_bigint(),
                *prod.d.as_bigint(),
            )
        });
        let new_denom = self.denom.ct_mul(elem.denom.as_bigint());

        // Modular HNF with `|det(new_cols)|` as the bound, mirroring
        // the strategy in [`Lattice::product`]. Classical HNF on 4
        // wide columns (≈ 530 bits for the multi-order-enumeration
        // `conj(δ)/N(I)` shape) blows past the storage budget; the
        // modular variant bounds intermediates by `|det|` and stays
        // within working width `N` whenever `bits(det) < N·64/2`.
        let det_modulus = Matrix::<N>::from_columns(&new_cols).det().abs();
        let basis = if bool::from(det_modulus.is_zero()) {
            Matrix::from_hnf_columns(&new_cols)
        } else {
            Matrix::from_hnf_columns_mod::<N>(&new_cols, &det_modulus)
        };
        HnfLattice {
            basis,
            denom: new_denom,
        }
    }

    /// Like [`Lattice::alg_elem_mul`] but with explicit modular-HNF
    /// modulus. See [`Lattice::product_with_modulus`] for when this
    /// matters.
    pub fn alg_elem_mul_with_modulus(
        &self,
        elem: &Element<N>,
        modulus: &BigInt<N>,
    ) -> HnfLattice<N> {
        let new_cols: [Vector<N>; 4] = array::from_fn(|j| {
            let basis_j = self.basis_elem(j);
            let prod = basis_j.mul_direct(elem);
            Vector::new(
                *prod.a.as_bigint(),
                *prod.b.as_bigint(),
                *prod.c.as_bigint(),
                *prod.d.as_bigint(),
            )
        });
        let new_denom = self.denom.ct_mul(elem.denom.as_bigint());
        let basis = if bool::from(modulus.is_zero()) {
            Matrix::from_hnf_columns(&new_cols)
        } else {
            Matrix::from_hnf_columns_mod::<N>(&new_cols, modulus)
        };
        HnfLattice {
            basis,
            denom: new_denom,
        }
    }

    /// Samples a random element from this lattice whose reduced norm
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
        let cols_w: [Vector<W>; 4] = array::from_fn(|i| {
            Vector::new(
                cols_n[i][0].widen::<W>(),
                cols_n[i][1].widen::<W>(),
                cols_n[i][2].widen::<W>(),
                cols_n[i][3].widen::<W>(),
            )
        });
        // Primal Gram, scaled by 2 to match C ref's `quat_lattice_gram`
        // (`lattice.c:344`), which builds `G[i][j] = 2 · (a_i·a_j +
        // b_i·b_j + p · (c_i·c_j + d_i·d_j))` — the TRACE pairing, not
        // the reduced-norm pairing. Selkie's `NrdBasis::compute_gram`
        // omits the factor of 2 (it's a reduced-norm pairing matrix),
        // so for byte-equality with `quat_lattice_sample_from_ball` we
        // multiply through here. Without this, det_G and the LLL-reduced
        // dualG diagonal entries scale by a factor of 2⁴ = 16, producing
        // different `bounds[i]` after the `sqrt_floor` and different
        // sample byte-streams.
        let two = BigInt::<W>::from_u64(2);
        let g_w = {
            let mut g = *NrdBasis::new(cols_w).gram();
            for i in 0..4 {
                for j in 0..4 {
                    g[i][j] = g[i][j].ct_mul(&two);
                }
            }
            g
        };

        // Adjust radius: rad = radius · denom² · 2.
        // (Gram corresponds to twice the reduced norm; the radius
        // squared by denom matches the field-reduced rep.)
        let denom_wide: BigInt<W> = self.denom.widen();
        let rad: BigInt<W> = radius
            .widen::<W>()
            .ct_mul(&denom_wide)
            .ct_mul(&denom_wide)
            .ct_mul(&two);

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
        #[cfg(test)]
        {
            crate::l2_trace_active::set(true);
        }
        let nrd = NrdBasis::from_cols_and_gram(identity_cols, dual_g).l2_reduce();
        #[cfg(test)]
        {
            crate::l2_trace_active::set(false);
        }
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
            *bound = quot.sqrt_floor()?;
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
            //
            // Mirrors C ref's `quat_lattice_sample_from_ball`
            // (`lat_ball.c:97-106`): for each coordinate, draw via
            // `ibz_rand_interval(x, 0, 2·box[i])`, then subtract box[i].
            // Must use `BigInt::rand_interval` (not an inline byte read)
            // because that function applies the top-byte mask that
            // matches `ibz_rand_interval` byte-for-byte. An earlier
            // version read `bitlen.div_ceil(8)` bytes unmasked and
            // rejected samples whose bit-length exceeded `bitlen` —
            // every such rejection consumed bytes from the DRBG without
            // a matching consumption in C ref's masked path, breaking
            // the byte-stream parity used by the deterministic KAT
            // signatures.
            let mut y = [BigInt::<W>::ZERO; 4];
            let _ = byte_cap;
            for i in 0..4 {
                if bool::from(bounds[i].is_zero()) {
                    continue;
                }
                let two_b = bounds[i].ct_add(&bounds[i]);
                let raw = BigInt::<W>::rand_interval(rng, &BigInt::<W>::ZERO, &two_b);
                y[i] = raw.ct_sub(&bounds[i]);
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
    /// Column-style HNF basis matrix.
    basis: Matrix<N>,
    /// Common denominator shared by all basis entries.
    denom: BigInt<N>,
}

impl<const N: usize> HnfLattice<N> {
    /// Divides out `gcd(basis_entries, denom)`, mirroring C ref's
    /// `quat_lattice_reduce_denom` (`quaternion/ref/generic/lattice.c:41`).
    ///
    /// After lattice multiplication or alg-elem multiplication the
    /// `denom` accumulates factors from both operands, but the basis
    /// entries typically share a common gcd with the denom. Dividing
    /// both by that gcd produces the canonical representation
    /// (smallest denom for the given Z-lattice).
    #[must_use]
    pub fn reduce_denom(self) -> Self {
        let mut g = self.denom.abs();
        for row in 0..4 {
            for col in 0..4 {
                let entry = self.basis[row][col];
                if !bool::from(entry.is_zero()) {
                    g = g.gcd(&entry.abs());
                    if g == BigInt::<N>::ONE {
                        return self;
                    }
                }
            }
        }
        if g == BigInt::<N>::ONE {
            return self;
        }
        let mut basis = self.basis;
        for row in 0..4 {
            for col in 0..4 {
                let (q, _) = basis[row][col].div_rem(&g);
                basis[row][col] = q;
            }
        }
        let (denom, _) = self.denom.div_rem(&g);
        Self { basis, denom }
    }

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
