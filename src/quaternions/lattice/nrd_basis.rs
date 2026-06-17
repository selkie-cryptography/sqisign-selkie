//! Quaternion lattice basis with reduced-norm Gram matrix and L²
//! reduction.
//!
//! [`NrdBasis`] pairs a rank-4 quaternion lattice basis with its
//! reduced-norm Gram matrix, keeping the two in sync. The dominant
//! operation on this type is [`NrdBasis::l2_reduce`], the L²
//! lattice-reduction algorithm of Nguyen and Stehlé over the
//! `DoublePlusExponent` floating-point representation.
//!
//! See [§4.2.2] of the SQIsign specification for the role of L²
//! reduction in quaternion-side processing.
//!
//! [§4.2.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.4.2.2

use super::{
    super::{
        bigint::BigInt,
        linear::{Matrix, Vector},
    },
    dpe::DoublePlusExponent,
};

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
    /// Four basis column vectors over `{1, i, j, k}`.
    cols: [Vector<N>; D],
    /// Reduced-norm Gram matrix `G[i][j] = nrd_bilinear(b_i, b_j)`.
    gram: Matrix<N>,
}

impl<const N: usize> NrdBasis<N> {
    /// Constructs from column vectors, computing the reduced-norm
    /// Gram matrix.
    pub fn new(cols: [Vector<N>; D]) -> Self {
        let gram = Self::compute_gram(&cols);
        Self { cols, gram }
    }

    /// Computes the reduced-norm Gram matrix for column vectors in
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
                    .vt_mul(&cols[j][0])
                    .ct_add(&cols[i][1].vt_mul(&cols[j][1]));
                let jk = cols[i][2]
                    .vt_mul(&cols[j][2])
                    .ct_add(&cols[i][3].vt_mul(&cols[j][3]));
                let val = scalar.ct_add(&p.vt_mul(&jk));
                gram[i][j] = val;
                if i != j {
                    gram[j][i] = val;
                }
            }
        }
        gram
    }

    /// Constructs from columns and a precomputed Gram matrix.
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
        // Carry-save MAC: form the D*D signed product terms, then sum
        // them with a single sign-and-magnitude merge instead of one
        // `ct_add` per term. Byte-identical: the widened width holds the
        // exact sum. See `BigInt::mac_sum`.
        let mut terms = [BigInt::<N>::ZERO; D * D];
        for i in 0..D {
            for j in 0..D {
                terms[i * D + j] = c[i].vt_mul(&c[j]).vt_mul(&self.gram[i][j]);
            }
        }
        BigInt::mac_sum(&terms)
    }

    /// L² reduction with DPE-based GSO ([Alg. 3.3]).
    ///
    /// Reduces the basis in place, keeping the Gram matrix in sync.
    /// Uses [`DoublePlusExponent`] (double-precision
    /// with extended exponent) for the Gram-Schmidt coefficients, matching
    /// the C reference's approach. The basis and Gram updates remain exact
    /// (integer). Size-reduction rounding uses
    /// [`DoublePlusExponent::to_bigint`] to
    /// convert the float μ back to an integer coefficient, which handles
    /// values that exceed `i64` range (e.g., `μ[3][0] ≈ 2^260` before first
    /// reduction).
    ///
    /// # Precision requirement
    ///
    /// DPE has a 53-bit mantissa. This is sufficient when the Gram
    /// entries are ≲ 2^127 (i.e., after
    /// [`super::HnfLattice::canonicalize`] shrinks mod-HNF entries). For
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
        #[cfg(test)]
        let l2_my_call: u64 = {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            COUNTER.fetch_add(1, Ordering::SeqCst) + 1
        };
        #[cfg(test)]
        if std::env::var("L2_TRACE").is_ok()
            && (crate::l2_trace_active::get() || std::env::var_os("SELKIE_L2_TRACE_ALL").is_some())
        {
            eprintln!("[L2_SELKIE] === call #{l2_my_call} begin ===");
            for i in 0..4 {
                for j in 0..=i {
                    eprintln!(
                        "[L2_SELKIE] call={l2_my_call} G_in[{i}][{j}] = {}",
                        self.gram[i][j]
                    );
                }
            }
        }
        // Capture every L² input as a text file under
        // `$SELKIE_L2_DUMP_DIR/l2_in_{call}.txt`. Each file has 16
        // lines of the form `g_{i}_{j} = <signed-decimal>` plus 16
        // lines of the form `c_{j}_{i} = <signed-decimal>` for the
        // 4×4 basis columns. Pair with the equivalent C-ref dump to
        // get a minimal repro for L²-LLL byte-equality investigation
        // (memory entry `2026-05-11-late`).
        #[cfg(test)]
        if let Some(dir) = std::env::var_os("SELKIE_L2_DUMP_DIR") {
            use std::io::Write;
            let path = std::path::PathBuf::from(&dir).join(format!("l2_in_{l2_my_call:04}.txt"));
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(mut f) = std::fs::File::create(&path) {
                for i in 0..4 {
                    for j in 0..4 {
                        let _ = writeln!(f, "g_{i}_{j} = {}", self.gram[i][j]);
                    }
                }
                for j in 0..4 {
                    for i in 0..4 {
                        let _ = writeln!(f, "c_{j}_{i} = {}", self.cols[j][i]);
                    }
                }
            }
        }

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
            // DPE-native comparison thresholds. Matches C-ref's
            // `dpe_cmp_d(u, ETABAR) > 0 || dpe_cmp_d(u, -ETABAR) < 0`
            // exactly — comparing DPE-to-DPE preserves the full DPE
            // precision, whereas the earlier `mu.abs().to_f64() >
            // eta_bar` round-tripped through f64 and could flip the
            // decision for values within f64-precision of the
            // threshold. That f64 round-trip is the documented
            // borderline-decision divergence from C-ref's
            // `quat_lll_core` (memory 2026-05-11-late).
            let eta_bar_dpe = DoublePlusExponent::from_f64(eta_bar);
            let neg_eta_bar_dpe = DoublePlusExponent::from_f64(-eta_bar);
            loop {
                extend_gso_family(gram, k, r, mu);

                let mut done = true;
                let mut ii = k;
                while ii > 0 {
                    ii -= 1;
                    if mu[k][ii] > eta_bar_dpe || mu[k][ii] < neg_eta_bar_dpe {
                        done = false;
                        let x_big: BigInt<N> = mu[k][ii].to_bigint();

                        if bool::from(x_big.is_zero()) {
                            continue;
                        }

                        // b_k ← b_k - x · b_ii. ii < k, so a disjoint
                        // split lets us read b_ii and mutate b_k without
                        // copying the whole Vector<N> b_ii each step.
                        let (lo, hi) = basis.split_at_mut(k);
                        let old_bi = &lo[ii];
                        let tgt = &mut hi[0];
                        for row in 0..D {
                            tgt[row] = tgt[row].ct_sub(&x_big.vt_mul(&old_bi[row]));
                        }

                        // Update Gram matrix symmetrically.
                        for j in 0..D {
                            let update = x_big.vt_mul(&gram[ii][j]);
                            gram[k][j] = gram[k][j].ct_sub(&update);
                        }
                        for j in 0..D {
                            let update = x_big.vt_mul(&gram[j][ii]);
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

            #[cfg(test)]
            if std::env::var("L2_TRACE").is_ok()
                && (crate::l2_trace_active::get()
                    || std::env::var_os("SELKIE_L2_TRACE_ALL").is_some())
            {
                for (i, val) in r[k][..=k].iter().enumerate() {
                    eprintln!(
                        "[L2_SELKIE] call={l2_my_call} kappa={k} post-size-reduce r[{k}][{i}] mant={:.17} exp={}",
                        val.m, val.e
                    );
                }
                for (i, val) in mu[k][..k].iter().enumerate() {
                    eprintln!(
                        "[L2_SELKIE] call={l2_my_call} kappa={k} post-size-reduce u[{k}][{i}] mant={:.17} exp={}",
                        val.m, val.e
                    );
                }
                eprintln!(
                    "[L2_SELKIE] call={l2_my_call} kappa={k} basis col[{k}] row[0] = {}",
                    self.cols[k][0]
                );
                eprintln!(
                    "[L2_SELKIE] call={l2_my_call} kappa={k} basis col[{k}] row[1] = {}",
                    self.cols[k][1]
                );
                eprintln!(
                    "[L2_SELKIE] call={l2_my_call} kappa={k} basis col[{k}] row[2] = {}",
                    self.cols[k][2]
                );
                eprintln!(
                    "[L2_SELKIE] call={l2_my_call} kappa={k} basis col[{k}] row[3] = {}",
                    self.cols[k][3]
                );
            }

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
                // Lovász fails at level s-1 iff t[s-1] < δ̄·r[s-1][s-1].
                // Anything else (≥ or `DoublePlusExponent::partial_cmp`
                // returning `None`) is treated as a hold and breaks the
                // descent — matching C ref's `dpe_cmp` short-circuit.
                match t[s - 1].partial_cmp(&(delta_bar_dpe * r[s - 1][s - 1])) {
                    Some(core::cmp::Ordering::Less) => s -= 1,
                    _ => break,
                }
            }

            #[cfg(test)]
            if std::env::var("L2_TRACE").is_ok()
                && (crate::l2_trace_active::get()
                    || std::env::var_os("SELKIE_L2_TRACE_ALL").is_some())
            {
                eprintln!("[L2_SELKIE] call={l2_my_call} kappa={k} swap={s}");
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
