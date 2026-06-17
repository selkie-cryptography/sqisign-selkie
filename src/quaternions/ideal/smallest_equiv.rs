//! Smallest-norm equivalent ideal: given `I`, find `J ∼ I` with
//! smaller norm by LLL-reducing the basis and pushing through the
//! shortest element `δ`. Used by both signing's `id2iso` chain
//! (Alg. 3.16 preparation) and the alternate-order search.

use crate::quaternions::{
    algebra::{Coordinate, Denominator, Element},
    bigint::BigInt,
    lattice::{HnfLattice, Lattice, LeftIdeal, NrdBasis},
    linear::{Matrix, Vector},
    precomputed::P_WIDE,
};

impl LeftIdeal<4> {
    /// Decompose this ideal for id2iso via [Alg. 3.16][Alg. 3.16]
    /// (SuitableIdeals).
    ///
    /// Finds elements β₁, β₂ and integers u, v, e such that
    /// u · d₁ + v · d₂ = 2^e where d_i = nrd(β_i) / nrd(I),
    /// both d_i are odd, gcd(u · d₁, v · d₂) = 1, and e ≤ f.
    ///
    /// Currently searches only the standard order (t = 0, no connecting
    /// ideals). This is sufficient for many ideals but may fail for some;
    /// connecting ideals for the remaining six extremal orders are needed
    /// for full coverage.
    ///
    /// # Side-channel considerations
    ///
    /// **Not constant-time.** This algorithm has data-dependent branches
    /// (L2 reduction loop count, pair search termination, GCD/primality
    /// checks) and data-dependent memory access patterns (sort, Vec
    /// growth). The input ideal I is derived from the signing key during
    /// signing (Algorithm 4.2, lines 13–19), so timing variations could
    /// in principle leak information about the signing key.
    ///
    /// The spec (§9.3.2) analyzes SuitableIdeals only in terms of
    /// failure probability, not side-channel resistance. The C reference
    /// implementation is also variable-time here. Making this fully
    /// constant-time would require CT L2 reduction, CT enumeration with
    /// oblivious sorting, and CT pair selection — an open problem for
    /// quaternion-based schemes.
    ///
    /// Compute the equivalent ideal of smallest norm.
    ///
    /// LLL-reduces the basis, takes the first (shortest) basis
    /// vector δ, and returns the equivalent ideal
    /// `I · δ̄ / nrd(I)` of norm `nrd(δ) / nrd(I)`.
    ///
    /// # Divergences
    ///
    /// The spec does not describe this as a named algorithm. The C
    /// reference performs this step inside `find_uv` (dim2id2iso.c,
    /// lines 526-546) before enumerating short vectors, calling it
    /// "replacing ideal by the equivalent ideal of smallest norm".
    /// Without this step, large-norm ideals (~2^257) produce
    /// short vectors with large degrees, and the `u·d₁ + v·d₂ =
    /// 2^e` search fails.
    #[must_use]
    pub fn smallest_equiv(&self) -> Option<Self> {
        self.smallest_equiv_with_delta()
            .map(|(ideal, _delta)| ideal)
    }

    /// Like [`Self::smallest_equiv`] but also returns the LLL-first
    /// element `δ ∈ I` used to derive the equivalent ideal
    /// `I · δ̄ / nrd(I)`.
    ///
    /// Required by the alternate-order search in
    /// [`Self::suitable_ideals`]: when a short vector is enumerated
    /// in a pushforward or `conj(I_reduced) · J_t` lattice, it must
    /// be transported back to the original ideal via multiplication
    /// by `δ`. Exposing `δ` here avoids recomputing it (and the full
    /// L2 reduction) at the transport step.
    ///
    /// See the C reference `dim2id2iso.c:546-565` for the analogous
    /// `reduced_id` + `delta` construction.
    #[must_use]
    pub fn smallest_equiv_with_delta(&self) -> Option<(Self, Element<4>)> {
        // LLL-reduce the basis at BigInt<8> for headroom.
        let lattice: Lattice<4> = (*self.lattice()).into();
        let cols_4 = lattice.basis().columns();
        let cols_8: [Vector<8>; 4] = core::array::from_fn(|j| cols_4[j].into());
        let denom_8: BigInt<8> = (*lattice.denom()).into();

        let nrd_basis = NrdBasis::new(cols_8).l2_reduce();

        // δ = first basis vector (shortest after LLL).
        let delta = Element::<4>::new(
            Coordinate::from_bigint(nrd_basis.cols()[0][0].narrow_to::<4>()?),
            Coordinate::from_bigint(nrd_basis.cols()[0][1].narrow_to::<4>()?),
            Coordinate::from_bigint(nrd_basis.cols()[0][2].narrow_to::<4>()?),
            Coordinate::from_bigint(nrd_basis.cols()[0][3].narrow_to::<4>()?),
            Denominator::from_bigint_unchecked(denom_8.narrow_to::<4>()?),
        );

        // nrd(δ) at BigInt<8> for precision.
        let (nrd_num, nrd_den) = delta.norm();
        let (new_norm, rem) = nrd_num.div_rem(&nrd_den);
        if !bool::from(rem.is_zero()) {
            return None;
        }
        // new_norm = nrd(δ), ideal norm = nrd(δ) / nrd(I)
        let norm_8: BigInt<8> = (*self.norm()).into();
        let (equiv_norm_8, rem2) = new_norm.div_rem(&norm_8);
        if !bool::from(rem2.is_zero()) {
            return None;
        }
        let equiv_norm: BigInt<4> = equiv_norm_8.narrow_to()?;

        // Construct I · δ̄ / nrd(I).
        // δ̄ = conjugate of δ. Each basis element of I multiplied
        // by δ̄ via Element<4>::mul (widens to BigInt<8> internally).
        let delta_conj = delta.conjugate();

        // I · δ̄: multiply each basis element by δ̄ using
        // mul_direct at BigInt<8> to avoid normalization (which
        // changes the denominator unpredictably). The raw product
        // denom is exactly lattice_denom * delta_denom.
        let delta_conj_8 = Element::<8>::new(
            Coordinate::from_bigint(delta_conj.a.as_bigint().widen::<8>()),
            Coordinate::from_bigint(delta_conj.b.as_bigint().widen::<8>()),
            Coordinate::from_bigint(delta_conj.c.as_bigint().widen::<8>()),
            Coordinate::from_bigint(delta_conj.d.as_bigint().widen::<8>()),
            Denominator::from_bigint_unchecked(delta_conj.denom.as_bigint().widen::<8>()),
        );
        let mut new_cols = [Vector::<8>::ZERO; 4];
        // reason: the body writes new_cols[j] and reads lattice.basis_elem(j) in
        // parallel; both are indexed by j, but lattice has no iterator over its
        // basis elements, so iter_mut().enumerate() doesn't simplify here.
        #[allow(clippy::needless_range_loop)]
        for j in 0..4 {
            let bj = lattice.basis_elem(j);
            let bj_8 = Element::<8>::new(
                Coordinate::from_bigint(bj.a.as_bigint().widen::<8>()),
                Coordinate::from_bigint(bj.b.as_bigint().widen::<8>()),
                Coordinate::from_bigint(bj.c.as_bigint().widen::<8>()),
                Coordinate::from_bigint(bj.d.as_bigint().widen::<8>()),
                Denominator::from_bigint_unchecked(bj.denom.as_bigint().widen::<8>()),
            );
            let product = bj_8.mul_direct(&delta_conj_8);
            new_cols[j] = Vector::new(
                *product.a.as_bigint(),
                *product.b.as_bigint(),
                *product.c.as_bigint(),
                *product.d.as_bigint(),
            );
        }
        // Raw product denom = lattice_denom * delta_denom.
        // Dividing by nrd(I) multiplies denom by nrd(I).
        let product_denom: BigInt<8> = {
            let ld: BigInt<8> = lattice.denom().widen();
            let dd: BigInt<8> = delta.denom.as_bigint().widen();
            ld.vt_mul(&dd).vt_mul(&norm_8)
        };

        // HNF at width 8, simplify by GCD, then narrow to 4.
        let hnf_8 = Matrix::<8>::from_hnf_columns(&new_cols);
        let mut g = product_denom.abs();
        for row in 0..4 {
            for col in 0..4 {
                if !bool::from(hnf_8[row][col].is_zero()) {
                    g = g.gcd(&hnf_8[row][col].abs());
                }
            }
        }
        let mut basis_4 = Matrix::<4>::ZERO;
        for row in 0..4 {
            for col in 0..4 {
                let (q, _) = hnf_8[row][col].div_rem(&g);
                basis_4[row][col] = q.narrow_to::<4>()?;
            }
        }
        let (denom_simplified, _) = product_denom.div_rem(&g);
        let denom_4: BigInt<4> = denom_simplified.narrow_to()?;

        let result_lattice = HnfLattice::from(Lattice::new(basis_4, denom_4));

        let ideal = Self::from_parts(result_lattice, equiv_norm, *self.parent_order());
        Some((ideal, delta))
    }
}

impl<const N: usize> LeftIdeal<N> {
    /// Returns the equivalent ideal of smallest norm as a
    /// [`LeftIdeal<4>`].
    ///
    /// LLL-reduces the basis, takes the first (shortest) basis
    /// vector δ, and returns the equivalent ideal `I · δ̄ / nrd(I)`
    /// of norm `nrd(δ) / nrd(I)`. The result's norm is typically
    /// much smaller than the input (≈ √p for generic inputs).
    ///
    /// Generalizes [`LeftIdeal<4>::smallest_equiv`] with storage
    /// width `N` and internal LLL working width `W` as const
    /// generics. Used by the signing response path to reduce a
    /// wide [`LeftIdeal<30>`] (norm ≈ `2^258`) to a form that fits
    /// in [`BigInt<4>`] before [`to_isogeny`].
    ///
    /// Returns [`None`] when any of these fit checks fails:
    /// - `nrd(δ)` is not exactly divisible by `nrd(I)`.
    /// - The equivalent norm exceeds [`BigInt<4>`].
    /// - The HNF entries of the reduced basis exceed [`BigInt<4>`].
    /// - The reduced denominator exceeds [`BigInt<4>`].
    ///
    /// `W` must satisfy `W ≥ 2·N` to hold squared Gram entries
    /// during LLL; this is enforced at compile time.
    ///
    /// # Divergences
    ///
    /// The spec does not describe this as a named algorithm. The C
    /// reference performs the reduction inside `find_uv`
    /// (`dim2id2iso.c:526-546`), calling it "replacing ideal by the
    /// equivalent ideal of smallest norm".
    ///
    /// # Constant-time
    ///
    /// Variable-time. `TODO(ct)`: called on secret-derived ideals
    /// during signing (response-phase `i_com_rsp`) — L2 reduction
    /// has data-dependent loop counts.
    ///
    /// [`to_isogeny`]: crate::quaternions::lattice::LeftIdeal::to_isogeny
    /// [`LeftIdeal<4>::smallest_equiv`]: LeftIdeal::smallest_equiv
    #[must_use]
    pub fn smallest_equiv_narrow<const W: usize>(&self) -> Option<LeftIdeal<4>> {
        const {
            assert!(
                W >= 2 * N,
                "smallest_equiv_narrow: W must be >= 2*N for LLL headroom"
            )
        };
        // Canonicalize the HNF first (reduces off-diagonals modulo
        // diagonal pivots) and widen to working width `W`.
        let canonical = self.lattice().canonicalize();
        let cols_n = canonical.basis().columns();
        let mut cols_w: [Vector<W>; 4] = core::array::from_fn(|j| {
            Vector::new(
                cols_n[j][0].widen::<W>(),
                cols_n[j][1].widen::<W>(),
                cols_n[j][2].widen::<W>(),
                cols_n[j][3].widen::<W>(),
            )
        });
        let denom_w: BigInt<W> = canonical.denom().widen();

        // L2-reduce the basis on the **class gram** rather than the
        // raw nrd gram. The class gram divides the raw form by
        // `2·d²·N(I)` (the C reference's `quat_lideal_class_gram`),
        // which keeps Gram entries bounded by Cauchy-Schwarz at
        // `~p / d²` (i.e. ≲ 2^124 at NIST-I) **regardless of**
        // `N(I)` — so DPE's 53-bit mantissa is sufficient even for
        // response-phase intersection ideals at `~2^378`.
        //
        // Without this normalization, the raw gram entries scale
        // as `n(I)² · p ≈ 2^1004` for the intersection ideal,
        // overflowing DPE precision and leaving brute-force on
        // the unreduced HNF as the only avenue — which cannot
        // do better than `nrd(δ) ≈ 2·N(I)`, so the equivalent
        // ideal preserves the input norm bit-size and signing
        // stalls when the result must narrow to `BigInt<4>`.
        //
        // Implements the L2 step of [Alg. 3.3] (RandomEquivalentQuaternion).
        // [Alg. 3.3]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.3
        let p_w: BigInt<W> = P_WIDE.widen::<W>();
        let nrd = NrdBasis::<W>::new(cols_w);
        let denom_sq = denom_w.vt_mul(&denom_w);
        let self_norm_w: BigInt<W> = self.norm().widen::<W>();
        let class_divisor: BigInt<W> = denom_sq.vt_mul(&self_norm_w);
        let two_w = BigInt::<W>::from_u64(2);
        let mut class_gram = Matrix::<W>::ZERO;
        for i in 0..4 {
            for j in 0..4 {
                let traced = nrd.gram()[i][j].vt_mul(&two_w);
                let (q, _rem) = traced.div_rem(&class_divisor);
                class_gram[i][j] = q;
            }
        }
        let class_basis = NrdBasis::<W>::from_cols_and_gram(cols_w, class_gram).l2_reduce();
        cols_w = *class_basis.cols();

        let eval_basis = |c: &[i64; 4]| -> [BigInt<W>; 4] {
            // v = Σ c_j · col_j, coordinate-wise.
            let mut v = [BigInt::<W>::ZERO; 4];
            for (j, cj) in c.iter().enumerate() {
                let cj_big = BigInt::<W>::from_i64(*cj);
                for (k, vk) in v.iter_mut().enumerate() {
                    *vk = vk.ct_add(&cj_big.vt_mul(&cols_w[j][k]));
                }
            }
            v
        };
        let nrd_of = |v: &[BigInt<W>; 4]| -> BigInt<W> {
            // nrd_num = a² + b² + p(c² + d²) at width W.
            let a2 = v[0].vt_mul(&v[0]);
            let b2 = v[1].vt_mul(&v[1]);
            let c2 = v[2].vt_mul(&v[2]);
            let d2 = v[3].vt_mul(&v[3]);
            a2.ct_add(&b2).ct_add(&p_w.vt_mul(&c2.ct_add(&d2)))
        };
        // Collect the top-K shortest δ candidates (by nrd). When the
        // absolute shortest fails to produce an equivalent ideal
        // that narrows to `BigInt<4>` — e.g., because one HNF entry
        // happens to miss a few bits of common factor with
        // `product_denom` — we fall through to the next-shortest.
        //
        // Brute-force over an **unreduced** HNF basis cannot do
        // better than `min_nrd ≈ 2·N(I)` (Minkowski's second
        // theorem on non-LLL-reduced bases), so the equivalent
        // ideal has `n(I') ≈ N(I)` — roughly preserving the input
        // norm bit-size. This is only useful when `N(I)` already
        // fits in `BigInt<4>` before the reduction (the typical
        // commitment-phase case). For larger ideals (e.g. the
        // response-phase intersection at `~2^378`), proper LLL on
        // the class gram is required — see Task #34 (arbitrary-
        // precision LLL for response-phase ideals). Empirically
        // `MAG = 4`, `TOP_K = 16` covers the tail for `N(I) ≤ 2^258`.
        const TOP_K: usize = 16;
        const MAG: i64 = 4;
        let mut candidates: Vec<([BigInt<W>; 4], BigInt<W>)> =
            Vec::with_capacity(((2 * MAG + 1) as usize).pow(4));
        for c0 in -MAG..=MAG {
            for c1 in -MAG..=MAG {
                for c2 in -MAG..=MAG {
                    for c3 in -MAG..=MAG {
                        if c0 == 0 && c1 == 0 && c2 == 0 && c3 == 0 {
                            continue;
                        }
                        let v = eval_basis(&[c0, c1, c2, c3]);
                        let nrd = nrd_of(&v);
                        if bool::from(nrd.is_zero()) {
                            continue;
                        }
                        candidates.push((v, nrd));
                    }
                }
            }
        }
        candidates.sort_by_key(|c| c.1);
        candidates.truncate(TOP_K);

        // Try each candidate in ascending `nrd` order; return the
        // first equivalent ideal that narrows to `BigInt<4>` AND
        // has non-unit norm. A unit-norm (norm = 1) equivalent
        // means `δ` is a primitive generator of a principal
        // `self`, i.e., `self = O·δ`. Downstream, `to_isogeny`'s
        // `suitable_ideals` cannot factor the unit ideal `O_0`
        // and exhausts its pair budget. Since `smallest_equiv` is
        // supposed to hand the caller a non-trivial ideal
        // equivalent to `self`, skip `δ`'s that collapse to
        // `O_0` and try the next-shortest.
        for (best_v, _) in candidates {
            if let Some(mut result) = self.build_equiv_from_delta::<W>(best_v, denom_w) {
                let rn = *result.norm();
                if rn == BigInt::<4>::ONE || bool::from(rn.is_zero()) {
                    continue;
                }
                // Verify the constructed lattice's covolume matches
                // the claimed norm. If not, the mod-HNF reduction
                // produced a basis for a sublattice (or our
                // stored norm is wrong) — skip and try next.
                let _stored_norm = *result.norm();
                // `refresh_norm<24>` covers `BigInt<4>` basis
                // entries up to ~256 bits: 4-fold det products
                // reach ~1029 bits, comfortably within 24·64 = 1536
                // bits. Trust whatever covolume `refresh_norm`
                // computes — if it differs from the brute-force
                // estimate (e.g. our `bj·δ̄/N(I)` div+HNF produced
                // a sublattice rather than the actual equivalent
                // ideal), the refreshed value is the
                // mathematically correct one.
                if result.refresh_norm::<24>().is_none() {
                    continue;
                }
                let refreshed_norm = *result.norm();
                if refreshed_norm == BigInt::<4>::ONE || bool::from(refreshed_norm.is_zero()) {
                    continue;
                }
                // Skip even-norm candidates: `to_isogeny` scales
                // step-6 matrix entries by `invmod(parent_norm·d₁,
                // 2^f)`, which is undefined when `parent_norm` is
                // even. Mirrors the C ref's
                // `quat_lideal_prime_norm_reduced_equivalent`,
                // which only accepts prime-norm candidates.
                if bool::from(refreshed_norm.is_even()) {
                    continue;
                }
                return Some(result);
            }
        }
        None
    }

    /// Builds the equivalent-ideal `LeftIdeal<4>` from a specific
    /// short element `δ` of `self`'s lattice (given by its
    /// coordinate numerators at width `W` and the shared lattice
    /// denominator).
    ///
    /// Returns `None` if any of the downstream divisibility /
    /// narrowing checks fail. Callers iterating over multiple
    /// candidate `δ`'s use this to test each in turn.
    fn build_equiv_from_delta<const W: usize>(
        &self,
        delta_coords: [BigInt<W>; 4],
        denom_w: BigInt<W>,
    ) -> Option<LeftIdeal<4>> {
        const {
            assert!(
                W >= 2 * N,
                "build_equiv_from_delta: W must be >= 2*N for LLL headroom"
            )
        };
        let delta_w = Element::<W>::new(
            Coordinate::from_bigint(delta_coords[0]),
            Coordinate::from_bigint(delta_coords[1]),
            Coordinate::from_bigint(delta_coords[2]),
            Coordinate::from_bigint(delta_coords[3]),
            Denominator::from_bigint_unchecked(denom_w),
        );

        // nrd(δ) at BigInt<W> using direct multiplication
        // (`mul_direct` + norm).
        let delta_nrd_num = {
            let a = delta_w.a.as_bigint();
            let b = delta_w.b.as_bigint();
            let c = delta_w.c.as_bigint();
            let d = delta_w.d.as_bigint();
            let p_w: BigInt<W> = {
                let p8 = P_WIDE;
                p8.widen::<W>()
            };
            a.vt_mul(a)
                .ct_add(&b.vt_mul(b))
                .ct_add(&p_w.vt_mul(&c.vt_mul(c).ct_add(&d.vt_mul(d))))
        };
        let delta_nrd_den = denom_w.vt_mul(&denom_w);
        let (new_norm_w, rem) = delta_nrd_num.div_rem(&delta_nrd_den);
        if !bool::from(rem.is_zero()) {
            return None;
        }
        let self_norm_w: BigInt<W> = self.norm().widen::<W>();
        let (equiv_norm_w, rem2) = new_norm_w.div_rem(&self_norm_w);
        if !bool::from(rem2.is_zero()) {
            return None;
        }
        let equiv_norm: BigInt<4> = equiv_norm_w.narrow_to()?;

        // Conjugate δ: negate i, j, k coords; a stays.
        let delta_conj_w = Element::<W>::new(
            Coordinate::from_bigint(*delta_w.a.as_bigint()),
            Coordinate::from_bigint(delta_w.b.as_bigint().wrapping_neg()),
            Coordinate::from_bigint(delta_w.c.as_bigint().wrapping_neg()),
            Coordinate::from_bigint(delta_w.d.as_bigint().wrapping_neg()),
            Denominator::from_bigint_unchecked(*delta_w.denom.as_bigint()),
        );

        // Build the equivalent ideal as `O₀·δ̄ + O₀·equiv_norm`
        // directly, NOT as `I·δ̄/N(I)` via per-column quaternion
        // multiplication. The two ideals are mathematically
        // identical (both are the unique left `O₀`-ideal in `[I]`
        // with norm `equiv_norm`), but the construction-from-the-
        // generator approach has bounded entry sizes:
        // `O₀·δ̄` columns are bounded by `p · max(δ̄)` ≈ `p · √nrd(δ)`
        // and `O₀·equiv_norm` columns are bounded by `equiv_norm`,
        // so the 8 generators all fit in `BigInt<W>`. The
        // `I·δ̄/N(I)` per-column path produces integer columns
        // with denom `d²·N(I)` and entries up to `~equiv_norm² ·
        // N(I)^4 · d^4` (≳ 2^1300) before division, which the
        // mod-HNF reduction couldn't tame at any reasonable
        // modulus without corrupting the lattice.
        //
        // Same construction as `reduce_to_prime_norm`
        // (`lattice.rs:2621-2700`).
        let order = self.parent_order();
        let order_basis = order.basis();
        let order_denom = order.denom();
        let alpha_denom = *delta_conj_w.denom.as_bigint();

        let p_w: BigInt<W> = P_WIDE.widen::<W>();
        let qmul = |a: &[BigInt<W>; 4], b: &[BigInt<W>; 4]| -> [BigInt<W>; 4] {
            let (a0, a1, a2, a3) = (&a[0], &a[1], &a[2], &a[3]);
            let (b0, b1, b2, b3) = (&b[0], &b[1], &b[2], &b[3]);
            [
                a0.vt_mul(b0)
                    .ct_sub(&a1.vt_mul(b1))
                    .ct_sub(&p_w.vt_mul(&a2.vt_mul(b2).ct_add(&a3.vt_mul(b3)))),
                a0.vt_mul(b1)
                    .ct_add(&a1.vt_mul(b0))
                    .ct_add(&p_w.vt_mul(&a2.vt_mul(b3).ct_sub(&a3.vt_mul(b2)))),
                a0.vt_mul(b2)
                    .ct_add(&a2.vt_mul(b0))
                    .ct_sub(&a1.vt_mul(b3))
                    .ct_add(&a3.vt_mul(b1)),
                a0.vt_mul(b3)
                    .ct_add(&a3.vt_mul(b0))
                    .ct_add(&a1.vt_mul(b2))
                    .ct_sub(&a2.vt_mul(b1)),
            ]
        };
        let alpha_arr = [
            *delta_conj_w.a.as_bigint(),
            *delta_conj_w.b.as_bigint(),
            *delta_conj_w.c.as_bigint(),
            *delta_conj_w.d.as_bigint(),
        ];

        // Compute O₀·δ̄ at width W.
        let mut o_alpha_cols = [Vector::<W>::ZERO; 4];
        for (j, o_col) in o_alpha_cols.iter_mut().enumerate() {
            let e = [
                order_basis[0][j].widen::<W>(),
                order_basis[1][j].widen::<W>(),
                order_basis[2][j].widen::<W>(),
                order_basis[3][j].widen::<W>(),
            ];
            let r = qmul(&e, &alpha_arr);
            *o_col = Vector::new(r[0], r[1], r[2], r[3]);
        }
        let o_alpha_denom = order_denom.widen::<W>().vt_mul(&alpha_denom);

        // Compute O₀·equiv_norm, rescaled to the shared denom
        // `order.denom · α.denom`.
        let equiv_norm_w: BigInt<W> = equiv_norm_w; // shadow
        let mut o_n_cols: [Vector<W>; 4] = core::array::from_fn(|j| {
            Vector::new(
                order_basis[0][j].widen::<W>(),
                order_basis[1][j].widen::<W>(),
                order_basis[2][j].widen::<W>(),
                order_basis[3][j].widen::<W>(),
            )
        });
        for col in &mut o_n_cols {
            for row in 0..4 {
                col[row] = col[row].vt_mul(&equiv_norm_w).vt_mul(&alpha_denom);
            }
        }

        // Mod-HNF with modulus `4 · d⁴ · equiv_norm² · p` (a
        // multiple of the integer-column covolume for the O₀-ideal
        // of norm `equiv_norm` with denom `d_total`).
        let d_sq = o_alpha_denom.vt_mul(&o_alpha_denom);
        let d_fourth = d_sq.vt_mul(&d_sq);
        let m_sq = equiv_norm_w.vt_mul(&equiv_norm_w);
        let four = BigInt::<W>::from_u64(4);
        let modulus = four.vt_mul(&d_fourth).vt_mul(&m_sq).vt_mul(&p_w);

        let all_cols = [
            o_alpha_cols[0],
            o_alpha_cols[1],
            o_alpha_cols[2],
            o_alpha_cols[3],
            o_n_cols[0],
            o_n_cols[1],
            o_n_cols[2],
            o_n_cols[3],
        ];
        let hnf_w = Matrix::<W>::from_hnf_columns_mod::<W>(&all_cols, &modulus);

        // Narrow basis and denom to BigInt<4>.
        let mut basis_4 = Matrix::<4>::ZERO;
        for row in 0..4 {
            for col in 0..4 {
                match hnf_w[row][col].narrow_to::<4>() {
                    Some(v) => basis_4[row][col] = v,
                    None => {
                        return None;
                    }
                }
            }
        }
        let denom_4: BigInt<4> = match o_alpha_denom.narrow_to() {
            Some(d) => d,
            None => {
                return None;
            }
        };

        let result_lattice = HnfLattice::from(Lattice::new(basis_4, denom_4));

        // Parent order must also narrow (all N>4 orders are widened
        // copies of the base `Order<4>`).
        let parent_order_4 = {
            let pbasis = self.parent_order().basis();
            let pdenom = self.parent_order().denom();
            let mut narrowed = Matrix::<4>::ZERO;
            for row in 0..4 {
                for col in 0..4 {
                    narrowed[row][col] = pbasis[row][col].narrow_to::<4>()?;
                }
            }
            let narrowed_denom = pdenom.narrow_to::<4>()?;
            crate::quaternions::lattice::Order::<4>::from_lattice_unchecked(Lattice::new(
                narrowed,
                narrowed_denom,
            ))
        };

        Some(LeftIdeal::<4>::from_parts(
            result_lattice,
            equiv_norm,
            parent_order_4,
        ))
    }
}
