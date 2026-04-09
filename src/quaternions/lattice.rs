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
    ideal::gram_matrix_nrd,
    linear::{Matrix, Vector, hnf_from_columns},
};

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
                basis: hnf_from_columns(&all_cols),
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
                basis: hnf_from_columns(&all_cols),
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
        let result_basis = hnf_from_columns(&all_cols);
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
    pub fn sample_from_ball<const W: usize>(&self, radius: &BigInt<N>) -> Option<Element<N>> {
        // Widen columns to BigInt<W> for intermediate products.
        let cols_n = self.basis.columns();
        let cols_w: [Vector<W>; 4] = core::array::from_fn(|i| {
            Vector::new(
                cols_n[i][0].widen::<W>(),
                cols_n[i][1].widen::<W>(),
                cols_n[i][2].widen::<W>(),
                cols_n[i][3].widen::<W>(),
            )
        });
        let gram = gram_matrix_nrd(&cols_w);

        // Adjust radius: rad = radius * denom² * 2
        // (Gram matrix corresponds to twice the reduced norm)
        let denom_wide: BigInt<W> = self.denom.widen();
        let rad: BigInt<W> = radius
            .widen::<W>()
            .ct_mul(&denom_wide)
            .ct_mul(&denom_wide)
            .ct_mul(&BigInt::<W>::from_u64(2));

        // Step 2: Compute dual Gram matrix and LLL-reduce it.
        // G* = adj(G), with det(G) tracked separately.
        let det_g = gram.det();
        let dual_gram = gram.adjugate();

        // LLL-reduce the dual Gram to get tighter per-axis bounds.
        // U is the transformation matrix (inverse of the LLL reduction).
        let mut dual_cols = dual_gram.columns();
        let mut dual_gram_reduced = dual_gram;
        l2_reduce::<W>(&mut dual_cols, &mut dual_gram_reduced);

        // Reconstruct U: the LLL reduction implicitly applies U to the
        // columns. We need U^{-T} for mapping samples back. Since LLL
        // produces a unimodular U with det = ±1, U^{-1} = adj(U)/det(U)
        // and det(U) = ±1, so U^{-1} = ±adj(U).
        //
        // The C ref computes U explicitly during LLL then inverts it.
        // Our l2_reduce doesn't return U, so we recover it from the
        // relationship: reduced_cols = original_cols * U (column-wise).
        // For now, use a simpler approach: solve for U.
        //
        // Actually, the C ref's approach is:
        //   1. LLL-reduce dual_gram, getting U such that dual_gram_reduced = U^T ·
        //      dual_gram · U
        //   2. Invert U: U_inv = adj(U) * det(U) (det = ±1)
        //   3. Bounding box: box[i] = √(dual_gram_reduced[i][i] * radius / det_g)
        //   4. Sample x in [-box[i], box[i]]^4
        //   5. Map: x ← U_inv^T · x
        //   6. Check: x^T · G · x ≤ radius
        //
        // Our l2_reduce modifies the Gram matrix in place and the columns.
        // The columns after reduction ARE the transformed basis. We need
        // to express the sample in the ORIGINAL basis. Since
        //   reduced_col[i] = Σ_j U[j][i] · original_col[j]
        // and we want to go from reduced coords back to original coords,
        // we need to multiply by U (which we don't have directly).
        //
        // TODO: modify l2_reduce to return U, or compute U from the
        // relationship between original and reduced columns.
        // For now, use the reduced columns directly to compute the
        // bounding box, and sample in the original basis with those bounds.

        // Compute per-axis bounding box from the reduced diagonal.
        let mut bounds = [BigInt::<W>::ZERO; 4];
        let mut all_zero = true;
        for i in 0..4 {
            // box[i] = √(dual_gram_reduced[i][i] * radius / det_g)
            let num = dual_gram_reduced[i][i].ct_mul(&rad);
            let (bound_sq, _) = num.div_rem(&det_g);
            bounds[i] = bound_sq.sqrt_floor();
            if !bool::from(bounds[i].is_zero()) {
                all_zero = false;
            }
        }
        if all_zero {
            return None; // ball too small
        }

        // Step 3: Rejection sampling.
        // Byte buffer for random sampling — sized for BigInt<W>.
        let byte_cap = W * 8;
        for _ in 0..10_000 {
            // Sample uniform x[i] in [-bounds[i], bounds[i]].
            let mut x = [BigInt::<W>::ZERO; 4];
            for i in 0..4 {
                if bool::from(bounds[i].is_zero()) {
                    continue;
                }
                // Random in [0, 2*bounds[i]], then subtract bounds[i].
                let two_b = bounds[i].ct_add(&bounds[i]);
                // Simple rejection sampling for uniform in [0, 2b].
                let bitlen = two_b.bitsize();
                loop {
                    let mut bytes = vec![0u8; byte_cap];
                    let needed = (bitlen as usize).div_ceil(8);
                    OsRng.fill_bytes(&mut bytes[..needed]);
                    let val = BigInt::<W>::from_bytes_le_unsigned(&bytes[..needed]);
                    let val = val.abs(); // ensure positive
                    if val.bitsize() <= bitlen {
                        // Check val <= 2*bounds[i]
                        let diff = val.ct_sub(&two_b);
                        if bool::from(diff.is_negative()) || bool::from(diff.is_zero()) {
                            x[i] = val.ct_sub(&bounds[i]);
                            break;
                        }
                    }
                }
            }

            // TODO: map x through U_inv^T (currently skipped — using
            // original basis coords directly, which is less tight but
            // still correct for rejection sampling).

            // Evaluate quadratic form: nrd = x^T · G · x.
            let mut nrd = BigInt::<W>::ZERO;
            for i in 0..4 {
                for j in 0..4 {
                    nrd = nrd.ct_add(&x[i].ct_mul(&x[j]).ct_mul(&gram[i][j]));
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
            let diff = nrd.ct_sub(&rad);
            if !bool::from(diff.is_negative()) && !bool::from(diff.is_zero()) {
                continue; // nrd > rad
            }

            // Step 4: Convert to quaternion element.
            // result = Σ x[i] · col_i, with the lattice denominator.
            let mut coords = [BigInt::<W>::ZERO; 4];
            for i in 0..4 {
                for (k, coord) in coords.iter_mut().enumerate() {
                    *coord = coord.ct_add(&x[i].ct_mul(&cols_w[i][k]));
                }
            }

            // Narrow back to BigInt<N> (should fit after sampling).
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
        None // sampling failed after max attempts
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
        // Compute Oα: multiply each basis element of O by α.
        let mut o_alpha_cols = [Vector::ZERO; 4];
        for (j, o_alpha_col) in o_alpha_cols.iter_mut().enumerate() {
            let basis_j = order.basis_elem(j);
            let product = basis_j.mul(alpha);
            *o_alpha_col = Vector::new(
                *product.a.as_bigint(),
                *product.b.as_bigint(),
                *product.c.as_bigint(),
                *product.d.as_bigint(),
            );
        }
        let o_alpha_denom = order.denom().ct_mul(&BigInt::<4>::from(alpha.denom));
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
    /// Uses [`represent_integer`](crate::quaternions::ideal::represent_integer)
    /// to find γ with nrd(γ) = m·N, then samples random β with
    /// gcd(nrd(β), N) = 1.
    ///
    /// WARNING: Not constant-time.
    ///
    /// TODO(ct): Make constant-time before production use.
    ///
    /// [Alg. 3.10]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.10
    pub fn random_norm(n: &BigInt<4>, order: &ExtremalOrder<4>) -> Option<Self> {
        use crate::quaternions::ideal::represent_integer;

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
        let gamma = represent_integer(&mn, &order_wide, false)?;

        // Lines 11-14: sample β = x + yi + zj + wij with gcd(nrd(β), N) = 1
        let n_bits = n.bitsize() as usize;
        let n_bytes = n_bits.div_ceil(8);

        let sample_in_range = || -> BigInt<4> {
            loop {
                let mut bytes = [0u8; 32];
                OsRng.fill_bytes(&mut bytes[..n_bytes]);
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

            // Check gcd(nrd(β), N) = 1.
            let (nrd_num, nrd_den) = beta.norm();
            let nrd_num_4: subtle::CtOption<BigInt<4>> = nrd_num.into();
            let nrd_den_4: subtle::CtOption<BigInt<4>> = nrd_den.into();
            if !bool::from(nrd_num_4.is_some()) || !bool::from(nrd_den_4.is_some()) {
                continue;
            }
            let nrd_4 = nrd_num_4.unwrap();
            let den_4 = nrd_den_4.unwrap();
            let (nrd_val, rem) = nrd_4.div_rem(&den_4);
            if !bool::from(rem.is_zero()) {
                continue;
            }

            let gcd = nrd_val.gcd(n);
            if gcd != BigInt::<4>::ONE {
                continue;
            }

            // Line 15: J' ← ideal generated by γβ and N
            let gamma_beta = gamma.mul(&beta);
            return Some(Self::new(&gamma_beta, n, order.order()));
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
        let n_bits = n.bitsize() as usize;
        let n_bytes = n_bits.div_ceil(8);

        let sample_mod_n = |rng: &mut R| -> BigInt<30> {
            loop {
                let mut bytes = [0u8; 240]; // 30 × 8
                rng.fill_bytes(&mut bytes[..n_bytes]);
                if n_bits % 8 != 0 {
                    bytes[n_bytes - 1] &= (1u8 << (n_bits % 8)) - 1;
                }
                let val = BigInt::<30>::from_bytes_le_unsigned(&bytes[..n_bytes]);
                if val.ct_mod(n) == val {
                    return val;
                }
            }
        };

        for _ in 0..10_000 {
            let g1 = sample_mod_n(rng);
            let g2 = sample_mod_n(rng);
            let g3 = sample_mod_n(rng);

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
            // entry in `latex/spec-review.tex` for the full
            // rationale.
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
        let narrow_int = |v: &BigInt<N>| -> Option<BigInt<4>> {
            // Check that all limbs above 4 are zero (positive) or all-ones (negative sign).
            let limbs = v.as_limbs();
            for &limb in &limbs[4..] {
                if limb != 0 {
                    return None;
                }
            }
            let mut out = [0u64; 4];
            out.copy_from_slice(&limbs[..4]);
            let sign = if bool::from(v.is_negative()) { 1 } else { 0 };
            Some(BigInt::from_sign_and_limbs(sign, out))
        };

        let mut basis_4 = [[BigInt::<4>::ZERO; 4]; 4];
        let basis_8 = self.lattice().basis();
        for row in 0..4 {
            for col in 0..4 {
                basis_4[row][col] = narrow_int(&basis_8[row][col])?;
            }
        }

        let norm_4 = narrow_int(self.norm())?;
        let denom_4 = narrow_int(self.lattice().denom())?;

        // Narrow the parent order.
        let order_8 = self.parent_order();
        let mut order_basis_4 = [[BigInt::<4>::ZERO; 4]; 4];
        let order_basis_8 = order_8.basis();
        for row in 0..4 {
            for col in 0..4 {
                order_basis_4[row][col] = narrow_int(&order_basis_8[row][col])?;
            }
        }
        let order_denom_4 = narrow_int(order_8.denom())?;

        let to_matrix = |rows: [[BigInt<4>; 4]; 4]| -> Matrix<4> {
            Matrix::from_rows(
                Vector::new(rows[0][0], rows[0][1], rows[0][2], rows[0][3]),
                Vector::new(rows[1][0], rows[1][1], rows[1][2], rows[1][3]),
                Vector::new(rows[2][0], rows[2][1], rows[2][2], rows[2][3]),
                Vector::new(rows[3][0], rows[3][1], rows[3][2], rows[3][3]),
            )
        };

        Some(LeftIdeal {
            lattice: HnfLattice {
                basis: to_matrix(basis_4),
                denom: denom_4,
            },
            norm: norm_4,
            parent_order: Order::from_lattice_unchecked(Lattice {
                basis: to_matrix(order_basis_4),
                denom: order_denom_4,
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
        // Step 1: L2-reduce the basis and compute the Gram matrix.
        let basis = self.lattice.basis();
        let mut cols = basis.columns();
        let mut gram = gram_matrix_nrd::<N>(&cols);
        l2_reduce::<N>(&mut cols, &mut gram);

        let denom = self.lattice.denom();
        let denom_sq = denom.ct_mul(denom);

        // Step 2: sample random short vectors until norm is prime.
        let limit = (2 * bound as i64 + 1).pow(4);
        let mut _ok_div = 0u32;
        for _ in 0..limit {
            let c: [BigInt<N>; 4] = [
                BigInt::from_i64(Self::rand_interval(rng, bound)),
                BigInt::from_i64(Self::rand_interval(rng, bound)),
                BigInt::from_i64(Self::rand_interval(rng, bound)),
                BigInt::from_i64(Self::rand_interval(rng, bound)),
            ];

            // Evaluate quadratic form: nrd = c^T · G · c.
            let mut nrd = BigInt::<N>::ZERO;
            for i in 0..4 {
                for j in 0..4 {
                    nrd = nrd.ct_add(&c[i].ct_mul(&c[j]).ct_mul(&gram[i][j]));
                }
            }

            // `nrd = c^T·G·c` is nrd(α_int) where
            // α_int = Σ c_i · cols[i] is the integer-coordinate
            // quaternion. The actual element of I is
            // α_elt = α_int / d, so nrd(α_elt) = nrd / d².
            let (nrd_alpha, rem1) = nrd.div_rem(&denom_sq);
            if !bool::from(rem1.is_zero()) {
                continue;
            }
            // The equivalent ideal J = I · ᾱ_elt / N(I) has norm
            // m = nrd(α_elt) / N(I). Reject α unless N(I) divides
            // nrd(α_elt) exactly and m is prime.
            let (m, rem2) = nrd_alpha.div_rem(&self.norm);
            if !bool::from(rem2.is_zero()) {
                continue;
            }
            _ok_div += 1;

            if m.is_probable_prime_w::<PRIME_W>(primality_rounds) {
                eprintln!(
                    "  reduce: ok_div={_ok_div} found prime m_bits={}",
                    m.bitsize()
                );
                // Reconstruct α = Σ c_i · col_i in the reduced basis.
                let mut alpha = [BigInt::<N>::ZERO; 4];
                for i in 0..4 {
                    for (k, alpha_k) in alpha.iter_mut().enumerate() {
                        *alpha_k = alpha_k.ct_add(&c[i].ct_mul(&cols[i][k]));
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

                // J = I · ᾱ_elt / N(I), with α_elt = α_int / d.
                // A basis element of I is cols[i] / d, so a basis
                // element of J is
                //   (cols[i] / d) · (ᾱ_int / d) / N(I)
                //   = (cols[i] · ᾱ_int) / (d² · N(I)).
                // Stored as integer columns `new_cols = cols · ᾱ_int`
                // with denominator `d² · N(I)`.
                let new_denom = denom_sq.ct_mul(&self.norm);
                let new_norm = m;

                // Assemble new lattice and reduce to HNF.
                let new_basis = Matrix::from_columns(&new_cols);
                let new_lat = Lattice::new(new_basis, new_denom);
                let hnf = new_lat.hnf();

                self.lattice = hnf;
                self.norm = new_norm;

                return true;
            }
        }
        eprintln!("  reduce exhausted: ok_div={_ok_div}");
        false
    }

    /// Sample a uniform random integer in \[−m, m\] via rejection sampling.
    ///
    /// WARNING: Not constant-time (rejection loop). The bound `m` is
    /// public, so this is acceptable for SQIsign.
    fn rand_interval<R: RngCore>(rng: &mut R, m: i32) -> i64 {
        assert!(m >= 0);
        let range = 2 * (m as u32) + 1;
        let threshold = u32::MAX - (u32::MAX % range);
        loop {
            let val = rng.next_u32();
            if val < threshold {
                return (val % range) as i64 - m as i64;
            }
        }
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
// L2 lattice reduction
// ---------------------------------------------------------------------------

/// Dimension of quaternion lattices.
const D: usize = 4;

/// L2 reduction parameter η (size-reduction threshold).
/// Following the spec: η = 0.51 (any value in (1/2, 1) works).
const ETA: f64 = 0.51;

/// L2 reduction parameter δ (Lovász condition threshold).
/// Following the spec: δ = 0.99 (any value in (1/4, 1) works).
const DELTA: f64 = 0.99;

/// L2 lattice reduction for a dimension-four lattice.
///
/// Takes a basis (as an array of four column vectors) and its Gram
/// matrix, and returns an (η, δ)-reduced basis and Gram matrix.
/// The basis and Gram matrix are modified in-place.
///
/// WARNING: Not constant-time — the number of reduction steps
/// depends on the input basis.
///
/// TODO(ct): Make constant-time before production use. Called on
/// secret-derived ideal bases during signing (via SuitableIdeals
/// and RandomEquivalentQuaternion).
///
/// [Alg. 3.3] from the spec.
///
/// [Alg. 3.3]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.3
pub(crate) fn l2_reduce<const N: usize>(basis: &mut [Vector<N>; D], gram: &mut Matrix<N>) {
    /// Extend the GSO family from row k-1 to row k ([Alg. 3.4]).
    ///
    /// [Alg. 3.4]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.4
    fn extend_gso_family<const N: usize>(
        gram: &Matrix<N>,
        k: usize,
        r: &mut [[f64; D]; D],
        mu: &mut [[f64; D]; D],
    ) {
        for j in 0..=k {
            r[k][j] = gram[k][j].to_f64();
            for l in 0..j {
                r[k][j] -= r[k][l] * mu[j][l];
            }
            if j < k {
                mu[k][j] = r[k][j] / r[j][j];
            }
        }
    }

    /// Size-reduce the basis at index k ([Alg. 3.5]).
    ///
    /// [Alg. 3.5]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.5
    fn size_reduce<const N: usize>(
        basis: &mut [Vector<N>; D],
        gram: &mut Matrix<N>,
        k: usize,
        r: &mut [[f64; D]; D],
        mu: &mut [[f64; D]; D],
    ) {
        let eta_bar = (ETA + 0.5) / 2.0;

        loop {
            extend_gso_family(gram, k, r, mu);

            let mut done = true;
            let mut ii = k;
            while ii > 0 {
                ii -= 1;
                if mu[k][ii].abs() > eta_bar {
                    done = false;
                    let x = mu[k][ii].round() as i64;
                    let x_big = BigInt::<N>::from_i64(x);

                    // b_k ← b_k - X * b_i
                    let old_bi = basis[ii];
                    for row in 0..D {
                        basis[k][row] = basis[k][row].ct_sub(&x_big.ct_mul(&old_bi[row]));
                    }

                    // Update Gram matrix
                    for j in 0..D {
                        let update = x_big.ct_mul(&gram[ii][j]);
                        gram[k][j] = gram[k][j].ct_sub(&update);
                    }
                    for j in 0..D {
                        let update = x_big.ct_mul(&gram[j][ii]);
                        gram[j][k] = gram[j][k].ct_sub(&update);
                    }

                    // Update μ. Snapshot row ii first (it's `Copy`) so we
                    // can iterate row k mutably without aliasing.
                    let x_f = x as f64;
                    let mu_ii = mu[ii];
                    for (slot, &v) in mu[k].iter_mut().take(ii).zip(mu_ii.iter()) {
                        *slot -= x_f * v;
                    }
                    mu[k][ii] -= x_f;

                    // Update r
                    r[k][ii] = gram[k][ii].to_f64();
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

    /// Insert basis vector k before position s ([Alg. 3.6]).
    ///
    /// [Alg. 3.6]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.6
    fn insert_before<const N: usize>(
        basis: &mut [Vector<N>; D],
        gram: &mut Matrix<N>,
        k: usize,
        s: usize,
        r: &mut [[f64; D]; D],
        mu: &mut [[f64; D]; D],
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

        r[s][s] = gram[s][s].to_f64();
        for i in 0..s {
            mu[s][i] = mu[k][i];
            r[s][i] = r[k][i];
            r[s][s] -= mu[s][i] * r[s][i];
        }
    }

    let delta_bar = (DELTA + 1.0) / 2.0;

    let mut r = [[0.0f64; D]; D];
    let mut mu = [[0.0f64; D]; D];

    r[0][0] = gram[0][0].to_f64();
    mu[0][0] = 1.0;

    let mut t = [0.0f64; D];

    let mut k = 1usize;
    while k < D {
        size_reduce(basis, gram, k, &mut r, &mut mu);

        t[0] = gram[k][k].to_f64();
        for i in 1..=k {
            t[i] = t[i - 1] - mu[k][i - 1] * r[k][i - 1];
        }

        let mut s = k;
        for j in 0..k {
            if t[j] < delta_bar * r[j][j] {
                s = s.min(j);
            }
        }

        if k != s {
            insert_before(basis, gram, k, s, &mut r, &mut mu);
            k = s;
        }

        k += 1;
    }
}
