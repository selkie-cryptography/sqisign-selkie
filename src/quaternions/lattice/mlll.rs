//! Modified LLL (MLLL) reduction of a redundant generating set.
//!
//! Reduces `G ≥ 4` generators of a rank-4 quaternion lattice to an
//! LLL-reduced basis, dropping the `G − 4` linearly dependent generators as
//! they collapse to zero. This is the [ML2 algorithm][ml2] (Nguyen–Stehlé
//! floating-point LLL, [Alg. 1]) over the [`DoublePlusExponent`] GSO, with the
//! [Pohst MLLL][pohst] rank-deficiency handling ([Alg. 1] lines 12–13) and the
//! lazy size-reduction subroutine ([Alg. 8]).
//!
//! It generalizes [`super::nrd_basis::NrdBasis::l2_reduce`] — the `G = 4`
//! special case — so it can replace the HNF step in quaternion ideal
//! multiplication and intersection per [Compact Quaternion Algorithms for
//! SQIsign][cqa]: feeding the 16 pairwise products of two ideal bases (or the
//! 8 columns of a dual-sum) directly into MLLL produces a basis of the *same*
//! lattice while keeping intermediate integers bounded by the largest input
//! norm² ([Lemma 8]), versus the `nrd⁴` blow-up HNF incurs.
//!
//! # Precision
//!
//! Uses a 53-bit-mantissa DPE GSO, like `l2_reduce`. The wide-input precision
//! probes in `super::tests` (`l2_precision_disguise_*`, `largenorm_512`)
//! confirm the DPE size-reduction converges and reduces correctly with Gram
//! entries up to ~2^2056 — far above the ~2^1027 scale of NIST-I ideal-product
//! inputs — so 53-bit precision is sufficient here.
//!
//! # Validation
//!
//! Validated by differential tests against the canonical HNF (the
//! `mlll_preserves_lattice_*` tests in `tests`): for redundant generating
//! sets, including the `G = 16` ideal-product count and a randomized sweep,
//! `HNF(mlll_reduce(gens)) == HNF(gens)`, confirming the reduced basis spans
//! the same lattice. DPE precision at the ~2^1027 NIST-I ideal-product scale is
//! covered by the wide-input probes in `super::tests`.
//!
//! [cqa]: https://eprint.iacr.org/2026/1031.pdf
//! [Alg. 1]: https://eprint.iacr.org/2026/1031.pdf#algorithm.1
//! [Alg. 8]: https://eprint.iacr.org/2026/1031.pdf#algorithm.8
//! [Lemma 8]: https://eprint.iacr.org/2026/1031.pdf#lemma.1.8
//! [ml2]: https://doi.org/10.1137/070705702
//! [pohst]: https://doi.org/10.1016/S0747-7171(87)80061-5

use core::cmp::Ordering;

use super::{
    super::{bigint::BigInt, linear::Vector},
    dpe::DoublePlusExponent,
};

#[cfg(test)]
mod tests;

/// L² size-reduction threshold η (any value in `(1/2, 1)`), matching
/// `l2_reduce`.
const ETA: f64 = 0.51;

/// L² Lovász parameter δ, matching `l2_reduce`.
const DELTA: f64 = 0.99;

/// A set of `G` quaternion generators paired with its `G×G` reduced-norm Gram
/// matrix.
///
/// The Gram entry `gram[i][j] = nrd_bilinear(cols[i], cols[j])` under the
/// reduced-norm form of `B_{p,∞} = (−1, −p)_Q`:
/// `aᵢ·aⱼ + bᵢ·bⱼ + p·(cᵢ·cⱼ + dᵢ·dⱼ)`.
pub struct Generators<const N: usize, const G: usize> {
    /// The `G` generator column vectors (each four `{1, i, j, k}` coordinates).
    cols: [Vector<N>; G],
    /// The `G×G` reduced-norm Gram matrix, kept in sync with `cols`.
    gram: [[BigInt<N>; G]; G],
}

impl<const N: usize, const G: usize> Generators<N, G> {
    /// Constructs from `G` generators, computing the reduced-norm Gram matrix.
    pub fn new(cols: [Vector<N>; G]) -> Self {
        let gram = Self::compute_gram(&cols);
        Self { cols, gram }
    }

    /// Returns `p` widened to `BigInt<N>`.
    fn p_wide() -> BigInt<N> {
        let p8: BigInt<8> = crate::quaternions::precomputed::P_WIDE;
        let mut limbs = [0u64; N];
        let src = p8.as_limbs();
        let len = src.len().min(N);
        limbs[..len].copy_from_slice(&src[..len]);
        BigInt::from_sign_and_limbs(0, limbs)
    }

    /// Computes the `G×G` reduced-norm Gram matrix for the given generators.
    fn compute_gram(cols: &[Vector<N>; G]) -> [[BigInt<N>; G]; G] {
        let p = Self::p_wide();
        let mut gram = [[BigInt::<N>::ZERO; G]; G];

        for i in 0..G {
            for j in i..G {
                let scalar = cols[i][0]
                    .ct_mul(&cols[j][0])
                    .ct_add(&cols[i][1].ct_mul(&cols[j][1]));
                let jk = cols[i][2]
                    .ct_mul(&cols[j][2])
                    .ct_add(&cols[i][3].ct_mul(&cols[j][3]));
                let val = scalar.ct_add(&p.ct_mul(&jk));

                gram[i][j] = val;
                gram[j][i] = val;
            }
        }

        gram
    }

    /// Reduces the generating set and returns the rank-4 LLL-reduced basis
    /// (the four trailing columns once the dependent generators have collapsed
    /// to the front).
    ///
    /// Implements ML2 ([Alg. 1]). The lattice spanned is unchanged; only the
    /// basis representation differs (all column ops are unimodular).
    ///
    /// # Constant-time
    ///
    /// Variable-time. `TODO(ct)`: the size-reduction branches and the
    /// iteration count depend on the generators, which are secret-derived
    /// on the signing response path (Algorithm 4.2, lines 13-19).
    ///
    /// [Alg. 1]: https://eprint.iacr.org/2026/1031.pdf#algorithm.1
    #[must_use]
    pub fn mlll_reduce(mut self) -> [Vector<N>; 4] {
        debug_assert!(
            G >= 4,
            "MLLL needs at least 4 generators for a rank-4 lattice"
        );

        let eta_bar = (ETA + 0.5) / 2.0;
        let delta_bar = DoublePlusExponent::from_f64((DELTA + 1.0) / 2.0);

        let mut r = [[DoublePlusExponent::ZERO; G]; G];
        let mut mu = [[DoublePlusExponent::ZERO; G]; G];

        r[0][0] = DoublePlusExponent::from_bigint(&self.gram[0][0]);

        // ζ counts leading generators that have collapsed to zero (dependent);
        // the live basis is always cols[ζ..G].
        let mut zeta: usize = 0;
        let mut kappa: usize = 1;

        while kappa < G {
            Self::lazy_size_reduce(
                &mut self.cols,
                &mut self.gram,
                zeta,
                kappa,
                &mut r,
                &mut mu,
                eta_bar,
            );

            // Projected squared norms tᵢ = ‖π_i(b_κ)‖² over the live prefix,
            // from the GSO `lazy_size_reduce` just finalized.
            let mut t = [DoublePlusExponent::ZERO; G];
            t[zeta] = DoublePlusExponent::from_bigint(&self.gram[kappa][kappa]);
            for i in zeta..kappa {
                t[i + 1] = t[i] - mu[kappa][i] * r[kappa][i];
            }

            // Deep-insertion Lovász descent: the lowest level (≥ ζ) where b_κ is
            // long enough to sit. Generalizes l2_reduce's `t[s-1] <
            // δ̄·r[s-1][s-1]` test from floor 0 to floor ζ.
            let mut s = kappa;
            while s > zeta {
                match t[s - 1].partial_cmp(&(delta_bar * r[s - 1][s - 1])) {
                    Some(Ordering::Less) => s -= 1,
                    _ => break,
                }
            }

            if kappa != s {
                Self::insert_before(
                    &mut self.cols,
                    &mut self.gram,
                    kappa,
                    s,
                    &mut r,
                    &mut mu,
                    zeta,
                );
                kappa = s;
            }

            // A dependent generator size-reduces to zero and descends to ζ;
            // absorb it into the dead block and re-seat the new live base.
            if Self::is_zero(&self.cols[kappa]) {
                zeta += 1;
                r[zeta][zeta] = DoublePlusExponent::from_bigint(&self.gram[zeta][zeta]);
                kappa = zeta + 1;
            } else {
                kappa += 1;
            }
        }

        debug_assert_eq!(
            zeta,
            G - 4,
            "rank-4 lattice should leave exactly 4 live generators"
        );

        [
            self.cols[G - 4],
            self.cols[G - 3],
            self.cols[G - 2],
            self.cols[G - 1],
        ]
    }

    /// Returns whether every coordinate of `v` is zero.
    fn is_zero(v: &Vector<N>) -> bool {
        (0..4).all(|row| bool::from(v[row].is_zero()))
    }

    /// Lazy size-reduction of `b_κ` against `b_{ζ..κ}` ([Alg. 8]).
    ///
    /// Recomputes the Cholesky GSO (`r`, `μ`) for `b_κ` from the Gram each
    /// pass, then size-reduces `b_κ` by the rounded `μ` coefficients, repeating
    /// until `max|μ_{κ,j}| ≤ η̄`. All generator and Gram updates are exact
    /// integer arithmetic; on return `r[κ][·]` and `μ[κ][·]` are final.
    ///
    /// [Alg. 8]: https://eprint.iacr.org/2026/1031.pdf#algorithm.8
    // reason: the explicit (r, μ) in/out parameters mirror Alg. 8's signature;
    // bundling them into a struct would re-couple the GSO scratch to the
    // generating set and obscure the mapping to the paper.
    #[allow(clippy::too_many_arguments)]
    fn lazy_size_reduce(
        cols: &mut [Vector<N>; G],
        gram: &mut [[BigInt<N>; G]; G],
        zeta: usize,
        kappa: usize,
        r: &mut [[DoublePlusExponent; G]; G],
        mu: &mut [[DoublePlusExponent; G]; G],
        eta_bar: f64,
    ) {
        let eta_bar_dpe = DoublePlusExponent::from_f64(eta_bar);
        let neg_eta_bar_dpe = DoublePlusExponent::from_f64(-eta_bar);

        loop {
            // Cholesky GSO of b_κ against the live prefix (Alg. 8 step 2).
            for j in zeta..=kappa {
                r[kappa][j] = DoublePlusExponent::from_bigint(&gram[kappa][j]);
                for l in zeta..j {
                    r[kappa][j] -= r[kappa][l] * mu[j][l];
                }
                if j < kappa {
                    mu[kappa][j] = r[kappa][j] / r[j][j];
                }
            }

            // Size-reduction (Alg. 8 step 3): subtract ⌊μ_{κ,i}⌉·b_i, high i
            // first, until every coefficient is within η̄.
            let mut done = true;
            let mut i = kappa;
            while i > zeta {
                i -= 1;
                if mu[kappa][i] > eta_bar_dpe || mu[kappa][i] < neg_eta_bar_dpe {
                    done = false;
                    let x: BigInt<N> = mu[kappa][i].to_bigint();

                    if bool::from(x.is_zero()) {
                        continue;
                    }

                    // i < kappa, so a disjoint split lets us read b_i and
                    // mutate b_kappa without copying the whole Vector<N>
                    // b_i each size-reduction step.
                    {
                        let (lo, hi) = cols.split_at_mut(kappa);
                        let old = &lo[i];
                        let tgt = &mut hi[0];
                        for row in 0..4 {
                            tgt[row] = tgt[row].ct_sub(&x.ct_mul(&old[row]));
                        }
                    }

                    // Symmetric rank-1 Gram update for b_κ -= x·b_i. G is
                    // symmetric, so off the diagonal the κ-row product
                    // x·G[i][m] equals the κ-column product x·G[m][i];
                    // compute each once and reuse for both, instead of
                    // multiplying twice.
                    let mut prods = [BigInt::<N>::ZERO; G];
                    for (p, src) in prods.iter_mut().zip(gram[i].iter()) {
                        *p = x.ct_mul(src);
                    }

                    for (dst, p) in gram[kappa].iter_mut().zip(prods.iter()) {
                        *dst = dst.ct_sub(p);
                    }

                    // The m == κ entry recomputes from the just-updated G[κ][i],
                    // which folds in the x²·G[i][i] term of ‖b_κ - x·b_i‖².
                    for (m, row) in gram.iter_mut().enumerate() {
                        let upd = if m == kappa {
                            x.ct_mul(&row[i])
                        } else {
                            prods[m]
                        };
                        row[kappa] = row[kappa].ct_sub(&upd);
                    }
                }
            }

            if done {
                break;
            }
        }
    }

    /// Inserts `b_κ` immediately before position `s` (`s < κ`): rotates the
    /// generator and its Gram row/column down via adjacent transpositions, then
    /// recomputes the GSO row for the new occupant of `s` from `b_κ`'s
    /// coefficients over the live prefix `[ζ, s)`.
    // reason: the (r, μ) GSO state is threaded explicitly to mirror
    // `l2_reduce::insert_before`; a wrapper struct would obscure that parallel.
    #[allow(clippy::too_many_arguments)]
    fn insert_before(
        cols: &mut [Vector<N>; G],
        gram: &mut [[BigInt<N>; G]; G],
        kappa: usize,
        s: usize,
        r: &mut [[DoublePlusExponent; G]; G],
        mu: &mut [[DoublePlusExponent; G]; G],
        zeta: usize,
    ) {
        let mut j = kappa;
        while j > s {
            cols.swap(j, j - 1);
            gram.swap(j, j - 1);
            for row in gram.iter_mut() {
                row.swap(j, j - 1);
            }
            j -= 1;
        }

        r[s][s] = DoublePlusExponent::from_bigint(&gram[s][s]);
        for i in zeta..s {
            mu[s][i] = mu[kappa][i];
            r[s][i] = r[kappa][i];
            let prod = mu[s][i] * r[s][i];
            r[s][s] -= prod;
        }
    }
}
