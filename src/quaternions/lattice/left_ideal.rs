//! Left ideals of maximal orders in `B_{p,∞}`.
//!
//! A left ideal `I = O⟨α, N⟩` of a maximal order `O ⊂ B_{p,∞}` is
//! represented by its lattice in Hermite Normal Form, its reduced norm
//! `nrd(I)`, and its parent (left) order `O_L(I)`.
//!
//! See [§3.1.6] of the SQIsign specification.
//!
//! [§3.1.6]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.3.1.6

use rand_core::RngCore;

use super::{
    super::{
        algebra::Element,
        bigint::BigInt,
        linear::{Matrix, Vector},
    },
    HnfLattice, Lattice, NrdBasis, Order,
};

mod n30;
mod n4;

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

        let n_sqrt = index.sqrt_floor()?;
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

impl<const N: usize> LeftIdeal<N> {
    /// Narrow a wide `LeftIdeal<N>` to `LeftIdeal<4>` after norm reduction.
    ///
    /// After `reduce_to_prime_norm`, the norm is a small prime and
    /// the HNF basis entries are bounded. This converts the wide
    /// representation to the narrow one needed by `to_isogeny`.
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
            let m = class_basis.eval_quadratic_form(&c) >> 1;

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

impl<const N: usize> Copy for LeftIdeal<N> where BigInt<N>: Copy {}

impl<const N: usize> core::fmt::Debug for LeftIdeal<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "LeftIdeal(norm={}, lattice={:?})",
            self.norm, self.lattice
        )
    }
}
