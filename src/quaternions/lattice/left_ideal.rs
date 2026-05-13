//! Left ideals of maximal orders in `B_{p,∞}`.
//!
//! A left ideal `I = O⟨α, N⟩` of a maximal order `O ⊂ B_{p,∞}` is
//! represented by its lattice in Hermite Normal Form, its reduced norm
//! `nrd(I)`, and its parent (left) order `O_L(I)`.
//!
//! See [§3.1.6] of the SQIsign specification.
//!
//! [§3.1.6]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.3.1.6

use core::array;

use rand_core::{OsRng, RngCore};

use super::{
    super::{
        algebra::{Coordinate, Denominator, Element},
        bigint::BigInt,
        linear::{Matrix, Vector},
    },
    ExtremalOrder, HnfLattice, Lattice, NrdBasis, Order,
};

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
        // Apply classical HNF to o_alpha (= 4 mul_direct cols),
        // mirroring C-ref's `quat_lattice_alg_elem_mul` which calls
        // `quat_lattice_hnf` after the multiplication. Without this,
        // o_alpha is the raw mul cols; the downstream `sum_mod_cref`
        // sees DIFFERENT inputs than C-ref's `quat_lattice_add` which
        // gets HNF-reduced o_alpha. Classical HNF at width 30
        // overflows for our shape; widen to W=60 to compute, then
        // narrow back.
        let o_alpha_cols_hnf: [Vector<30>; 4] = {
            let widened: [Vector<60>; 4] = array::from_fn(|i| {
                let v = &o_alpha_cols[i];
                Vector::<60>::new(
                    v[0].widen::<60>(),
                    v[1].widen::<60>(),
                    v[2].widen::<60>(),
                    v[3].widen::<60>(),
                )
            });
            // Use C-ref's `quat_lattice_hnf` recipe: modular HNF with
            // mod = |det| of the input matrix, NOT classical HNF.
            // Selkie's classical `Matrix::hnf()` produces a valid
            // upper-triangular HNF but with off-diagonal entries
            // (cols 2,3 rows 0,1) that differ from C-ref's modular-HNF
            // result, even though both bases describe the same lattice.
            // Tested on KAT-1: byte-mismatch with C-ref's `lideal_com_resp`
            // is gone once we use HNF mod with mod=|det|.
            let det_w = Matrix::from_columns(&widened).det().abs();
            // Use the constant-modulus variant (= old Selkie path)
            // — for 4-col input this should match canonical HNF.
            let hnf_w = Matrix::from_hnf_columns_mod::<60>(&widened, &det_w);
            let cols_w: [Vector<60>; 4] = hnf_w.columns();
            array::from_fn(|i| {
                Vector::<30>::new(
                    cols_w[i][0]
                        .narrow_to::<30>()
                        .expect("o_alpha hnf col 0 fits in 30"),
                    cols_w[i][1]
                        .narrow_to::<30>()
                        .expect("o_alpha hnf col 1 fits in 30"),
                    cols_w[i][2]
                        .narrow_to::<30>()
                        .expect("o_alpha hnf col 2 fits in 30"),
                    cols_w[i][3]
                        .narrow_to::<30>()
                        .expect("o_alpha hnf col 3 fits in 30"),
                )
            })
        };
        let o_alpha = Lattice::new(Matrix::from_columns(&o_alpha_cols_hnf), o_alpha_denom);

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
        // Mod-HNF modulus uses the spec/Selkie historic formula
        // `4 · d⁴ · norm² · p`.  An earlier diagnostic toggle compared
        // this against C-ref's `quat_lattice_add` `gcd(det1, det2)`
        // recipe; the two produced the same Z-module for the sign
        // path and the toggle has been retired.
        let modulus = four.ct_mul(&d_fourth).ct_mul(&norm_sq).ct_mul(&p_wide);

        // ON denom may differ from o_alpha denom — pre-scale ON
        // basis so both share `o_alpha_denom` for `sum_mod` (which
        // assumes equal denoms).
        let alpha_d = *alpha.denom.as_bigint();
        let mut o_n_for_sum = order.basis().columns();
        for col in &mut o_n_for_sum {
            for row in 0..4 {
                col[row] = col[row].ct_mul(norm).ct_mul(&alpha_d);
            }
        }
        let o_n_scaled = Lattice::new(Matrix::from_columns(&o_n_for_sum), o_alpha_denom);
        let lattice = o_alpha.sum_mod::<60>(&o_n_scaled, &modulus)?;

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
            array::from_fn(|j| widen_col_4_to_w(&order_basis_cols_4[j]));
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

        // Lines 11-14: sample β = x + yi + zj + wij with gcd(nrd(β), N) = 1.
        //
        // Match C-ref's `ibz_rand_interval(out, 1, N)` byte-for-byte
        // (`src/quaternion/ref/generic/intbig.c:413`). C-ref masks the
        // top limb to `ceil(log2(N - 1))` bits, rejects `tmp > N - 1`,
        // then returns `tmp + 1`, mapping accepted tmps in `[0, N-1]`
        // to results in `[1, N]`. The earlier Selkie code rejected
        // `val == 0` and returned `val ∈ [1, N-1]` — same DRBG bytes
        // but each accepted sample one less than C-ref's, which then
        // compounded through `γ·β` into a different `i_aux` lattice
        // (KAT 39 iter 0 byte-diff vs `[I_AUX_CREF]`).
        let n_minus_1 = n.ct_sub(&BigInt::<4>::ONE);
        let bmina_bits = n_minus_1.bitsize() as usize;
        let bmina_bytes = bmina_bits.div_ceil(8);
        let mut sample_in_range = || -> BigInt<4> {
            loop {
                let mut bytes = [0u8; 32];
                rng.fill_bytes(&mut bytes[..bmina_bytes]);
                if bmina_bits % 8 != 0 {
                    bytes[bmina_bytes - 1] &= (1u8 << (bmina_bits % 8)) - 1;
                }
                let tmp = BigInt::<4>::from_bytes_le_unsigned(&bytes[..bmina_bytes]);
                // Reject when `tmp > N - 1`, matching C-ref's
                // `mpz_cmp(tmp, bmina) <= 0` accept condition.
                if tmp > n_minus_1 {
                    continue;
                }
                return tmp.ct_add(&BigInt::<4>::ONE);
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
    /// - The downstream `reduce_to_prime_norm` gram computation `c^T·G·c ≈
    ///   2^1806` fits without overflow.
    ///
    /// After construction, call `reduce_to_prime_norm` to get a small
    /// prime norm, then `narrow_to::<4>()` to convert to `LeftIdeal<4>`
    /// for `to_isogeny`.
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
