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
//! See [§3.1.5.2] and [§3.1.6] of the SQIsign specification.
//!
//! [§3.1.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.2
//! [§3.1.6]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.3.1.6

use core::ops::Add;

use super::algebra::{Coordinate, Denominator, Element};
use super::bigint::BigInt;
use super::linear::{hnf_from_columns, Matrix, Vector};

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
    /// [Algorithm 3.2] of the SQIsign specification.
    ///
    /// [Algorithm 3.2]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.2
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
}

impl<const N: usize> Copy for Lattice<N> where BigInt<N>: Copy {}

impl Lattice<4> {
    /// Returns the j-th basis vector (column j) as a quaternion element.
    pub fn basis_elem(&self, j: usize) -> Element {
        Element::new(
            Coordinate::from_bigint(self.basis[0][j]),
            Coordinate::from_bigint(self.basis[1][j]),
            Coordinate::from_bigint(self.basis[2][j]),
            Coordinate::from_bigint(self.basis[3][j]),
            Denominator::from_bigint_unchecked(self.denom),
        )
    }

    /// Decompose an element into coordinates in this lattice's column basis.
    ///
    /// Given α ∈ L, finds (c₀, c₁, c₂, c₃) such that
    /// α = c₀·b₀ + c₁·b₁ + c₂·b₂ + c₃·b₃ where bⱼ are the column
    /// basis elements of L/denom.
    ///
    /// Returns `None` if α is not in the lattice (non-integer solution).
    ///
    /// Uses the adjugate: x = adj(B)·v / det(B), avoiding field inversion.
    pub fn decompose(&self, elem: &Element) -> Option<[BigInt<4>; 4]> {
        let ed = BigInt::<4>::from(elem.denom);

        // Scale to common denominator: target = α_coords · lattice_denom / α_denom.
        let elem_coords = [
            *elem.a.as_bigint(),
            *elem.b.as_bigint(),
            *elem.c.as_bigint(),
            *elem.d.as_bigint(),
        ];
        let mut rhs = [BigInt::<4>::ZERO; 4];
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

        let mut result = [BigInt::<4>::ZERO; 4];
        for i in 0..4 {
            let mut val = BigInt::<4>::ZERO;
            for j in 0..4 {
                val = val.ct_add(&adj[i][j].ct_mul(&rhs[j]));
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
    /// See [§3.1.5.2] (Multiplication) of the spec.
    ///
    /// [§3.1.5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.2
    pub fn product(&self, other: &Self) -> HnfLattice<4> {
        let mut all_cols = Vec::new();

        for i in 0..4 {
            let alpha = self.basis_elem(i);
            for j in 0..4 {
                let beta = other.basis_elem(j);
                let product = alpha.mul(&beta);
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
    /// # Algorithm
    ///
    /// 1. Compute the Gram matrix G (reduced norm quadratic form)
    /// 2. Compute the dual Gram matrix G* = adj(G)/det(G)
    /// 3. LLL-reduce G* to get tight per-coordinate bounds
    /// 4. Rejection sample: pick uniform coords in the bounding box,
    ///    map through the LLL transformation, check norm ≤ radius
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
    pub fn sample_from_ball(&self, radius: &BigInt<4>) -> Option<Element> {
        // Widen to BigInt<8> for intermediate products.
        use super::ideal::gram_matrix_nrd;

        let cols_4 = self.basis.columns();
        let w = |v: BigInt<4>| -> BigInt<8> { v.into() };
        let cols_8: [Vector<8>; 4] = core::array::from_fn(|i| {
            Vector::new(
                w(cols_4[i][0]),
                w(cols_4[i][1]),
                w(cols_4[i][2]),
                w(cols_4[i][3]),
            )
        });
        let gram = gram_matrix_nrd(&cols_8);

        // Adjust radius: rad = radius * denom² * 2
        // (Gram matrix corresponds to twice the reduced norm)
        let denom_wide: BigInt<8> = self.denom.into();
        let rad: BigInt<8> = BigInt::<8>::from(*radius)
            .ct_mul(&denom_wide)
            .ct_mul(&denom_wide)
            .ct_mul(&BigInt::<8>::from_u64(2));

        // Step 2: Compute dual Gram matrix and LLL-reduce it.
        // G* = adj(G), with det(G) tracked separately.
        let det_g = gram.det();
        let dual_gram = gram.adjugate();

        // LLL-reduce the dual Gram to get tighter per-axis bounds.
        // U is the transformation matrix (inverse of the LLL reduction).
        let mut dual_cols = dual_gram.columns();
        let mut dual_gram_reduced = dual_gram;
        l2_reduce::<8>(&mut dual_cols, &mut dual_gram_reduced);

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
        //   1. LLL-reduce dual_gram, getting U such that dual_gram_reduced = U^T · dual_gram · U
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
        let mut bounds = [BigInt::<8>::ZERO; 4];
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
        use rand_core::{OsRng, RngCore};

        for _ in 0..10_000 {
            // Sample uniform x[i] in [-bounds[i], bounds[i]].
            let mut x = [BigInt::<8>::ZERO; 4];
            for i in 0..4 {
                if bool::from(bounds[i].is_zero()) {
                    continue;
                }
                // Random in [0, 2*bounds[i]], then subtract bounds[i].
                let two_b = bounds[i].ct_add(&bounds[i]);
                // Simple rejection sampling for uniform in [0, 2b].
                let bitlen = two_b.bitsize();
                loop {
                    let mut bytes = [0u8; 64]; // BigInt<8> = 512 bits
                    OsRng.fill_bytes(&mut bytes[..((bitlen as usize + 7) / 8)]);
                    let val =
                        BigInt::<8>::from_bytes_le_unsigned(&bytes[..((bitlen as usize + 7) / 8)]);
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
            let mut nrd = BigInt::<8>::ZERO;
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
            let mut coords = [BigInt::<8>::ZERO; 4];
            for i in 0..4 {
                for k in 0..4 {
                    coords[k] = coords[k].ct_add(&x[i].ct_mul(&cols_8[i][k]));
                }
            }

            // Narrow to BigInt<4> (should fit after sampling).
            let narrow = |v: BigInt<8>| -> BigInt<4> {
                v.narrow().expect("sampled element fits in BigInt<4>")
            };

            return Some(Element::new(
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
    pub fn contains(&self, elem: &Element) -> Option<Vector<N>> {
        // Widen element coordinates from BigInt<4> to BigInt<N>.
        let coords: [BigInt<N>; 4] = [
            elem.a.to_bigint(),
            elem.b.to_bigint(),
            elem.c.to_bigint(),
            elem.d.to_bigint(),
        ];
        let ed: BigInt<N> = elem.denom.to_bigint();

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
// LeftIdeal<N>: left ideal of a maximal order
// ---------------------------------------------------------------------------

/// A left ideal of a maximal order in B_{p,∞}.
///
/// An ideal I = O⟨α, N⟩ is represented by its lattice (in HNF), its
/// norm nrd(I), and the parent order O_L(I).
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
    /// The parent (left) order as a lattice.
    parent_order: Lattice<N>,
}

impl<const N: usize> LeftIdeal<N> {
    // Ideal creation and other Element-dependent methods are in
    // the impl LeftIdeal<4> block below (requires Lattice<4> for
    // basis_elem to produce Element values).

    /// Assemble a left ideal from pre-built components.
    ///
    /// The caller is responsible for ensuring the lattice is the
    /// correct HNF representation of the ideal.
    #[inline]
    pub const fn from_parts(
        lattice: HnfLattice<N>,
        norm: BigInt<N>,
        parent_order: Lattice<N>,
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
    pub const fn parent_order(&self) -> &Lattice<N> {
        &self.parent_order
    }
}

impl<const N: usize> Copy for LeftIdeal<N> where BigInt<N>: Copy {}

// Methods that bridge between Lattice<4> and concrete Element.
impl LeftIdeal<4> {
    /// Create the left ideal I = O⟨α, N⟩ = Oα + ON.
    ///
    /// See [§3.1.6.1] of the spec.
    ///
    /// [§3.1.6.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.6.1
    pub fn new(alpha: &Element, norm: &BigInt<4>, order: &Lattice<4>) -> Self {
        // Compute Oα: multiply each basis element of O by α.
        let mut o_alpha_cols = [Vector::ZERO; 4];
        for j in 0..4 {
            let basis_j = order.basis_elem(j);
            let product = basis_j.mul(alpha);
            o_alpha_cols[j] = Vector::new(
                *product.a.as_bigint(),
                *product.b.as_bigint(),
                *product.c.as_bigint(),
                *product.d.as_bigint(),
            );
        }
        let o_alpha_denom = order.denom.ct_mul(&BigInt::<4>::from(alpha.denom));
        let o_alpha = Lattice::new(Matrix::from_columns(&o_alpha_cols), o_alpha_denom);

        // Compute ON: scale each basis vector of O by N.
        let mut o_n_cols = order.basis.columns();
        for j in 0..4 {
            for row in 0..4 {
                o_n_cols[j][row] = o_n_cols[j][row].ct_mul(norm);
            }
        }
        let o_n = Lattice::new(Matrix::from_columns(&o_n_cols), order.denom);

        Self {
            lattice: o_alpha.sum(&o_n),
            norm: *norm,
            parent_order: *order,
        }
    }

    /// Construct a random left ideal of a given prime norm.
    ///
    /// [Algorithm 3.10] from the spec (prime case).
    ///
    /// WARNING: Not constant-time — brute-force search with
    /// data-dependent Legendre symbol and modular sqrt.
    ///
    /// TODO(ct): Make constant-time before production use. The norm
    /// argument may be secret-derived during signing (Algorithm 4.2
    /// line 23, where the norm depends on α_rsp).
    ///
    /// [Algorithm 3.10]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.10
    pub fn random_prime_norm(n: &BigInt<4>, order: &ExtremalOrder<4>) -> Option<Self> {
        let bound: u32 = 10000;
        let mut counter: u32 = 0;

        while counter < bound {
            counter += 1;

            let g1 = counter as i64 % 100;
            let g2 = (counter as i64 / 100) % 100;
            let g3 = (counter as i64 / 10000) % 100;

            let gamma = Element::from_i64(0, g1, g2, g3);
            let (nrd_num, nrd_den) = gamma.norm();

            // Narrow norm to BigInt<4> for Legendre/sqrt.
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

            let neg_nrd = n.ct_sub(&nrd_val.ct_mod(n));
            if BigInt::<4>::legendre(&neg_nrd, n) != 1 {
                continue;
            }

            let sqrt = match BigInt::<4>::modular_sqrt(&neg_nrd, n) {
                Some(s) => s,
                None => continue,
            };
            let gamma_adjusted = Element::new(
                Coordinate::from_bigint(sqrt),
                Coordinate::from_i64(g1),
                Coordinate::from_i64(g2),
                Coordinate::from_i64(g3),
                Denominator::ONE,
            );

            return Some(Self::new(&gamma_adjusted, n, order.order()));
        }

        None
    }

    /// Find a primitive generator γ of this ideal.
    ///
    /// [Algorithm 3.8] from the spec.
    ///
    /// WARNING: Not constant-time — bounded brute-force search with
    /// data-dependent norm checks and GCD.
    ///
    /// TODO(ct): Make constant-time before production use. Called on
    /// secret-derived ideals via IdealToKernel during signing.
    ///
    /// [Algorithm 3.8]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.8
    pub fn generator(&self) -> Option<Element> {
        let basis = self.lattice.basis();
        let n_i = &self.norm;

        const MAX_NORM: i64 = 1000;
        let mut n: i64 = 0;
        while n < MAX_NORM {
            n += 1;
            let mut a = -(n as i64);
            while a <= n as i64 {
                let rem_a = n as i64 - a.abs();
                let mut b = -rem_a;
                while b <= rem_a {
                    let rem_b = rem_a - b.abs();
                    let mut c = -rem_b;
                    while c <= rem_b {
                        let d = rem_b - c.abs();
                        for &d_val in &[d, -d] {
                            if a.abs() + b.abs() + c.abs() + d_val.abs() != n as i64 {
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
                            let gamma = Element::new(
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

impl LeftIdeal<8> {
    /// Replace this ideal with an equivalent one of prime norm.
    ///
    /// Samples random short elements α in the ideal's L2-reduced
    /// lattice until nrd(α) / N(I) is prime, then sets
    /// I ← I · ᾱ / N(I).
    ///
    /// Implements [RandomEquivalentPrimeIdeal][Alg. 3.9].
    /// Matches `quat_lideal_prime_norm_reduced_equivalent`
    /// (lll_applications.c:48).
    ///
    /// WARNING: Not constant-time.
    ///
    /// TODO(ct): Make constant-time before production use. Called on
    /// secret-derived ideals during signing (Algorithm 4.2 lines 6, 24).
    ///
    /// TODO: Implement. Requires:
    /// - Random sampling in \[−m, m\]
    /// - Ideal multiplication by element (`quat_lideal_mul`)
    ///
    /// [Alg. 3.9]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.9
    pub fn reduce_to_prime_norm(&mut self, bound: i32, primality_rounds: u32) -> bool {
        // Step 1: L2-reduce the basis and compute the Gram matrix.
        let basis = self.lattice.basis();
        let mut cols = basis.columns();
        let mut gram = crate::quaternions::ideal::gram_matrix_nrd(&cols);
        l2_reduce::<8>(&mut cols, &mut gram);

        let denom = self.lattice.denom();
        let denom_sq = denom.ct_mul(denom);

        // Step 2: sample random short vectors until norm is prime.
        let limit = (2 * bound as i64 + 1).pow(4);
        for _ in 0..limit {
            let c: [BigInt<8>; 4] = [
                BigInt::from_i64(Self::rand_interval(bound)),
                BigInt::from_i64(Self::rand_interval(bound)),
                BigInt::from_i64(Self::rand_interval(bound)),
                BigInt::from_i64(Self::rand_interval(bound)),
            ];

            // Evaluate quadratic form: nrd = c^T · G · c.
            let mut nrd = BigInt::<8>::ZERO;
            for i in 0..4 {
                for j in 0..4 {
                    nrd = nrd.ct_add(&c[i].ct_mul(&c[j]).ct_mul(&gram[i][j]));
                }
            }

            // Norm of equivalent ideal = nrd / denom².
            let (norm, _rem) = nrd.div_rem(&denom_sq);

            if norm.is_probable_prime(primality_rounds) {
                // Reconstruct α = Σ c_i · col_i in the reduced basis.
                let mut alpha = [BigInt::<8>::ZERO; 4];
                for i in 0..4 {
                    for k in 0..4 {
                        alpha[k] = alpha[k].ct_add(&c[i].ct_mul(&cols[i][k]));
                    }
                }

                // Conjugate α: negate the i, j, k coordinates.
                alpha[1] = alpha[1].wrapping_neg();
                alpha[2] = alpha[2].wrapping_neg();
                alpha[3] = alpha[3].wrapping_neg();

                // Multiply: new lattice = old lattice · ᾱ.
                // Wide quaternion mul in B_{p,∞} = (-1, -p): same
                // formula as Element::mul but at BigInt<8> width.
                let p_wide: BigInt<8> = crate::quaternions::precomputed::P_WIDE;
                let qmul = |a: &[BigInt<8>; 4], b: &[BigInt<8>; 4]| -> [BigInt<8>; 4] {
                    let (a0, a1, a2, a3) = (&a[0], &a[1], &a[2], &a[3]);
                    let (b0, b1, b2, b3) = (&b[0], &b[1], &b[2], &b[3]);
                    [
                        a0.ct_mul(b0)
                            .ct_sub(&a1.ct_mul(b1))
                            .ct_sub(&p_wide.ct_mul(&a2.ct_mul(b2).ct_add(&a3.ct_mul(b3)))),
                        a0.ct_mul(b1)
                            .ct_add(&a1.ct_mul(b0))
                            .ct_add(&p_wide.ct_mul(&a2.ct_mul(b3).ct_sub(&a3.ct_mul(b2)))),
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
                let mut new_cols = [Vector::<8>::ZERO; 4];
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

                // New denom = old_denom * alpha_denom.
                // α was built from lattice coords (integer), so
                // alpha_denom = lattice denom (already in self.lattice).
                let new_denom = denom.ct_mul(denom);

                // New norm = old_norm * nrd(α) / denom².
                // But nrd(α)/denom² = norm (which we already computed).
                let new_norm = self.norm.ct_mul(&norm);
                let (new_norm, _) = new_norm.div_rem(&denom_sq);

                // Assemble new lattice and reduce to HNF.
                let new_basis = Matrix::from_columns(&new_cols);
                let new_lat = Lattice::new(new_basis, new_denom);
                let hnf = new_lat.hnf();

                self.lattice = hnf;
                self.norm = new_norm;

                return true;
            }
        }
        false
    }

    /// Sample a uniform random integer in \[−m, m\] via rejection sampling.
    ///
    /// WARNING: Not constant-time (rejection loop). The bound `m` is
    /// public, so this is acceptable for SQIsign.
    fn rand_interval(m: i32) -> i64 {
        use rand_core::{OsRng, RngCore};

        assert!(m >= 0);
        let range = 2 * (m as u32) + 1;
        let threshold = u32::MAX - (u32::MAX % range);
        loop {
            let val = OsRng.next_u32();
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
    /// The order as a lattice.
    order: Lattice<N>,
    /// Element z with z² = -q (small discriminant).
    z: Element,
    /// Element t with nrd(t) = p, orthogonal to z.
    t: Element,
    /// The absolute value |z²| (a small positive integer).
    q: u32,
}

impl<const N: usize> ExtremalOrder<N> {
    /// Creates an extremal order from its components.
    #[inline]
    pub const fn new(order: Lattice<N>, z: Element, t: Element, q: u32) -> Self {
        Self { order, z, t, q }
    }

    /// Returns the order as a lattice.
    #[inline]
    pub const fn order(&self) -> &Lattice<N> {
        &self.order
    }

    /// Returns the element z (z² = -q).
    #[inline]
    pub const fn z(&self) -> &Element {
        &self.z
    }

    /// Returns the element t (nrd(t) = p).
    #[inline]
    pub const fn t(&self) -> &Element {
        &self.t
    }

    /// Returns q = |z²|.
    #[inline]
    pub const fn q(&self) -> u32 {
        self.q
    }
}

impl<const N: usize> Copy for ExtremalOrder<N> where BigInt<N>: Copy {}

/// Widen an `ExtremalOrder<4>` to `ExtremalOrder<8>` for lattice
/// arithmetic that needs wider matrix entries.
impl From<ExtremalOrder<4>> for ExtremalOrder<8> {
    fn from(order: ExtremalOrder<4>) -> Self {
        let basis4 = order.order().basis();
        let denom4 = order.order().denom();

        let mut basis8 = Matrix::<8>::ZERO;
        for row in 0..4 {
            for col in 0..4 {
                basis8[row][col] = basis4[row][col].into();
            }
        }
        let order_lat = Lattice::new(basis8, (*denom4).into());

        // Element is concrete — z and t copy directly.
        ExtremalOrder::new(order_lat, *order.z(), *order.t(), order.q())
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

/// Extend the GSO family from row k-1 to row k.
///
/// [Algorithm 3.4] from the spec.
///
/// [Algorithm 3.4]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.4
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

/// Size-reduce the basis at index k.
///
/// [Algorithm 3.5] from the spec.
///
/// [Algorithm 3.5]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.5
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

                // Update μ
                let x_f = x as f64;
                for j in 0..ii {
                    mu[k][j] -= x_f * mu[ii][j];
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

/// Insert basis vector k before position s, shifting others right.
///
/// [Algorithm 3.6] from the spec.
///
/// [Algorithm 3.6]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.6
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
        let tmp = basis[j];
        basis[j] = basis[j - 1];
        basis[j - 1] = tmp;

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
/// [Algorithm 3.3] from the spec.
///
/// [Algorithm 3.3]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.3
pub fn l2_reduce<const N: usize>(basis: &mut [Vector<N>; D], gram: &mut Matrix<N>) {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    type I = BigInt<4>;
    type L = Lattice<4>;
    type H = HnfLattice<4>;
    type V = Vector<4>;

    fn i(v: i64) -> I {
        I::from(v)
    }

    #[test]
    fn lattice_construction() {
        let lat = L::from_matrix(Matrix::IDENTITY);
        assert_eq!(*lat.denom(), I::ONE);
    }

    #[test]
    fn lattice_basis_elem() {
        let lat = L::from_matrix(Matrix::IDENTITY);
        let e0 = lat.basis_elem(0);
        assert_eq!(e0, Element::from_i64(1, 0, 0, 0));
        let e1 = lat.basis_elem(1);
        assert_eq!(e1, Element::from_i64(0, 1, 0, 0));
    }

    #[test]
    fn lattice_with_denominator() {
        // O_0 basis: (1, i, (i+j)/2, (1+ij)/2)
        // Columns with denom=2: [2,0,0,0], [0,2,0,0], [0,1,1,0], [1,0,0,1]
        let basis = Matrix::from_rows(
            V::new(i(2), i(0), i(0), i(1)),
            V::new(i(0), i(2), i(1), i(0)),
            V::new(i(0), i(0), i(1), i(0)),
            V::new(i(0), i(0), i(0), i(1)),
        );
        let lat = L::new(basis, i(2));
        let e2 = lat.basis_elem(2);
        assert_eq!(BigInt::<4>::from(e2.denom), i(2));
        assert_eq!(*e2.b.as_bigint(), i(1));
        assert_eq!(*e2.c.as_bigint(), i(1));
    }

    #[test]
    fn lattice_hnf() {
        let basis = Matrix::from_rows(
            V::new(i(2), i(5), i(0), i(0)),
            V::new(i(0), i(3), i(0), i(0)),
            V::new(i(0), i(0), i(1), i(0)),
            V::new(i(0), i(0), i(0), i(1)),
        );
        let h = H::from(L::from_matrix(basis));
        assert_eq!(h.basis()[0][0], i(2));
        assert_eq!(h.basis()[0][1], i(1)); // 5 mod 2 = 1
    }

    #[test]
    fn hnf_lattice_equality_same_denom() {
        let a = H::from(L::from_matrix(Matrix::IDENTITY));
        let b = H::from(L::from_matrix(Matrix::IDENTITY));
        assert_eq!(a, b);
    }

    #[test]
    fn hnf_lattice_equality_different_denom() {
        let a = H::from(L::from_matrix(Matrix::IDENTITY));
        let b: H = L::new(
            Matrix::from_rows(
                V::new(i(2), i(0), i(0), i(0)),
                V::new(i(0), i(2), i(0), i(0)),
                V::new(i(0), i(0), i(2), i(0)),
                V::new(i(0), i(0), i(0), i(2)),
            ),
            i(2),
        )
        .into();
        assert_eq!(a, b);
    }

    #[test]
    fn lattice_sum() {
        // Z^4 + 2*Z^4 = Z^4.
        let a = L::from_matrix(Matrix::IDENTITY);
        let b = L::from_matrix(Matrix::from_rows(
            V::new(i(2), i(0), i(0), i(0)),
            V::new(i(0), i(2), i(0), i(0)),
            V::new(i(0), i(0), i(2), i(0)),
            V::new(i(0), i(0), i(0), i(2)),
        ));
        let s = &a + &b;
        assert_eq!(s, H::from(L::from_matrix(Matrix::IDENTITY)));
    }

    #[test]
    fn contains_basis_element() {
        let h = H::from(L::from_matrix(Matrix::IDENTITY));
        let elem = Element::from_i64(1, 0, 0, 0);
        let coords = h.contains(&elem);
        assert!(coords.is_some());
        let c = coords.unwrap();
        assert_eq!(c[0], i(1));
        assert_eq!(c[1], i(0));
    }

    #[test]
    fn contains_linear_combination() {
        let h = H::from(L::from_matrix(Matrix::IDENTITY));
        let elem = Element::from_i64(3, 7, -2, 5);
        let coords = h.contains(&elem).expect("should be contained");
        assert_eq!(coords[0], i(3));
        assert_eq!(coords[1], i(7));
        assert_eq!(coords[2], i(-2));
        assert_eq!(coords[3], i(5));
    }

    #[test]
    fn does_not_contain() {
        // 2*Z^4 does not contain (1, 0, 0, 0).
        let h: H = L::from_matrix(Matrix::from_rows(
            V::new(i(2), i(0), i(0), i(0)),
            V::new(i(0), i(2), i(0), i(0)),
            V::new(i(0), i(0), i(2), i(0)),
            V::new(i(0), i(0), i(0), i(2)),
        ))
        .into();
        let elem = Element::from_i64(1, 0, 0, 0);
        assert!(h.contains(&elem).is_none());
    }

    #[test]
    fn contains_with_lattice_denominator() {
        // Lattice 2*Z^4 / 2 = Z^4 should contain (1, 0, 0, 0) / 1.
        let h: H = L::new(
            Matrix::from_rows(
                V::new(i(2), i(0), i(0), i(0)),
                V::new(i(0), i(2), i(0), i(0)),
                V::new(i(0), i(0), i(2), i(0)),
                V::new(i(0), i(0), i(0), i(2)),
            ),
            i(2),
        )
        .into();
        let elem = Element::from_i64(1, 0, 0, 0);
        assert!(h.contains(&elem).is_some());
    }

    #[test]
    fn left_ideal_construction() {
        let order = L::from_matrix(Matrix::IDENTITY);
        let ideal_lat = H::from(L::from_matrix(Matrix::IDENTITY));
        let ideal = LeftIdeal::from_parts(ideal_lat, i(1), order);
        assert_eq!(*ideal.norm(), i(1));
    }

    #[test]
    fn extremal_order_construction() {
        let order = L::from_matrix(Matrix::IDENTITY);
        let z = Element::from_i64(0, 1, 0, 0);
        let t = Element::from_i64(0, 0, 1, 0);
        let ext = ExtremalOrder::new(order, z, t, 1);
        assert_eq!(ext.q(), 1);
    }

    #[test]
    fn from_hnf_roundtrip() {
        let lat = L::from_matrix(Matrix::IDENTITY);
        let h: H = lat.into();
        let back: L = h.into();
        assert_eq!(*back.basis(), Matrix::IDENTITY);
    }

    #[test]
    fn matrix_det_identity() {
        assert_eq!(Matrix::<4>::IDENTITY.det(), i(1));
    }

    #[test]
    fn matrix_det_diagonal() {
        let m = Matrix::from_rows(
            V::new(i(2), i(0), i(0), i(0)),
            V::new(i(0), i(3), i(0), i(0)),
            V::new(i(0), i(0), i(5), i(0)),
            V::new(i(0), i(0), i(0), i(7)),
        );
        assert_eq!(m.det(), i(210)); // 2*3*5*7
    }

    #[test]
    fn matrix_adjugate_identity() {
        let adj = Matrix::<4>::IDENTITY.adjugate();
        assert_eq!(adj, Matrix::IDENTITY);
    }

    #[test]
    fn matrix_adjugate_times_original_is_det_times_identity() {
        let m = Matrix::from_rows(
            V::new(i(2), i(1), i(0), i(0)),
            V::new(i(0), i(3), i(0), i(0)),
            V::new(i(0), i(0), i(5), i(1)),
            V::new(i(0), i(0), i(0), i(7)),
        );
        let adj = m.adjugate();
        let product = m.mat_mul(&adj);
        let det = m.det();
        for row in 0..4 {
            for col in 0..4 {
                let expected = if row == col { det } else { i(0) };
                assert_eq!(
                    product[row][col], expected,
                    "M*adj(M) != det*I at [{row}][{col}]"
                );
            }
        }
    }

    #[test]
    fn lattice_intersection() {
        // Z^4 ∩ 2Z^4 = 2Z^4.
        let a = L::from_matrix(Matrix::IDENTITY);
        let b = L::from_matrix(Matrix::from_rows(
            V::new(i(2), i(0), i(0), i(0)),
            V::new(i(0), i(2), i(0), i(0)),
            V::new(i(0), i(0), i(2), i(0)),
            V::new(i(0), i(0), i(0), i(2)),
        ));
        let inter = a.intersection(&b);
        let expected: H = b.into();
        assert_eq!(inter, expected);
    }

    #[test]
    fn ideal_creation() {
        // Create O₀⟨1, 1⟩ which should equal O₀ itself.
        let order = L::new(
            Matrix::from_rows(
                V::new(i(2), i(0), i(0), i(1)),
                V::new(i(0), i(2), i(1), i(0)),
                V::new(i(0), i(0), i(1), i(0)),
                V::new(i(0), i(0), i(0), i(1)),
            ),
            i(2),
        );
        let one = Element::from_i64(1, 0, 0, 0);
        let p = i(3);
        let ideal = LeftIdeal::new(&one, &i(1), &order);
        assert_eq!(*ideal.norm(), i(1));
    }

    #[test]
    fn ideal_generator() {
        // Create an ideal I = O₀⟨i, 2⟩ with p = 3.
        // The generator should be an element γ with gcd(nrd(γ)/2, 2) = 1.
        let order = L::new(
            Matrix::from_rows(
                V::new(i(2), i(0), i(0), i(1)),
                V::new(i(0), i(2), i(1), i(0)),
                V::new(i(0), i(0), i(1), i(0)),
                V::new(i(0), i(0), i(0), i(1)),
            ),
            i(2),
        );
        let alpha = Element::from_i64(0, 1, 0, 0); // i
        let p = i(3);
        let ideal = LeftIdeal::new(&alpha, &i(2), &order);

        let gamma = ideal.generator().expect("generator should be found");
        // Verify: γ is nonzero.
        assert!(!gamma.is_zero());

        // Verify: nrd(γ) / N_I is coprime to N_I.
        let (nrd_num, nrd_den) = gamma.norm();
        let n_i: BigInt<8> = (*ideal.norm()).into();
        let (q, rem) = nrd_num.div_rem(&nrd_den.ct_mul(&n_i));
        assert!(bool::from(rem.is_zero()), "nrd(γ) not divisible by N_I");
        assert_eq!(q.gcd(&n_i), BigInt::<8>::ONE, "gcd(nrd(γ)/N_I, N_I) != 1");
    }

    // L2 reduction tests use BigInt<8> for arithmetic headroom.
    type I8 = BigInt<8>;
    type V8 = Vector<8>;

    fn i8(v: i64) -> I8 {
        I8::from(v)
    }

    /// Compute the Gram matrix for the quaternion bilinear form
    /// ⟨α, β⟩ = tr(αβ̄) with Gram diag(2, 2, 2p, 2p) on {1,i,j,k}.
    fn quat_gram(basis: &[V8; 4], p: &I8) -> Matrix<8> {
        let two = i8(2);
        let two_p = two.ct_mul(p);
        let diag = [two, two, two_p, two_p];

        let mut g = Matrix::<8>::ZERO;
        for row in 0..4 {
            for col in 0..4 {
                let mut acc = I8::ZERO;
                for k in 0..4 {
                    acc = acc.ct_add(&basis[row][k].ct_mul(&diag[k]).ct_mul(&basis[col][k]));
                }
                g[row][col] = acc;
            }
        }
        g
    }

    #[test]
    fn l2_identity_basis() {
        let p = i8(3);
        let mut basis = [
            V8::new(i8(1), i8(0), i8(0), i8(0)),
            V8::new(i8(0), i8(1), i8(0), i8(0)),
            V8::new(i8(0), i8(0), i8(1), i8(0)),
            V8::new(i8(0), i8(0), i8(0), i8(1)),
        ];
        let mut gram = quat_gram(&basis, &p);

        l2_reduce(&mut basis, &mut gram);

        // Diagonal should be non-decreasing (short vectors first).
        for idx in 1..4 {
            assert!(
                gram[idx][idx] >= gram[idx - 1][idx - 1],
                "Gram diagonal not non-decreasing at position {idx}"
            );
        }
    }

    #[test]
    fn l2_reduces_bad_basis() {
        let p = i8(3);
        let mut basis = [
            V8::new(i8(1), i8(0), i8(0), i8(0)),
            V8::new(i8(100), i8(1), i8(0), i8(0)),
            V8::new(i8(0), i8(0), i8(1), i8(0)),
            V8::new(i8(0), i8(0), i8(0), i8(1)),
        ];
        let mut gram = quat_gram(&basis, &p);
        let original_g00 = gram[0][0];

        l2_reduce(&mut basis, &mut gram);

        assert!(
            gram[0][0] <= original_g00,
            "first vector got longer after reduction"
        );
    }

    #[test]
    fn l2_gram_stays_symmetric() {
        let p = i8(3);
        let mut basis = [
            V8::new(i8(3), i8(1), i8(0), i8(0)),
            V8::new(i8(1), i8(2), i8(0), i8(0)),
            V8::new(i8(0), i8(0), i8(1), i8(1)),
            V8::new(i8(0), i8(0), i8(2), i8(1)),
        ];
        let mut gram = quat_gram(&basis, &p);

        l2_reduce(&mut basis, &mut gram);

        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(
                    gram[row][col], gram[col][row],
                    "Gram not symmetric at [{row}][{col}]"
                );
            }
        }
    }
}
