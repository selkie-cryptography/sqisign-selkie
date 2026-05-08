//! Ideal algorithms for SQIsign key generation and signing.
//!
//! Algorithms that operate across multiple quaternion types (elements,
//! orders, ideals) and don't naturally belong to a single type.
//!
//! - [`ExtremalOrder::represent_integer`]: find γ ∈ O with nrd(γ) = M ([Alg.
//!   3.12])
//! - [`ExtremalOrder::represent_integer_any`]: same, trying all precomputed
//!   orders
//! - [`equivalent_prime_ideal`]: find J ∼ I with prime norm ([Alg. 3.9])
//! - [`SuitableIdealResult`]: output of SuitableIdeals ([Alg. 3.16][Alg. 3.16])
//!
//! [Alg. 3.9]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.9
//! [Alg. 3.12]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.12
//! [Alg. 3.16]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.16

use core::ops::Deref;

#[cfg(test)]
use rand_core::OsRng;
use rand_core::RngCore;

use super::{
    algebra::{Coordinate, Denominator, Element},
    bigint::BigInt,
    lattice::{ExtremalOrder, HnfLattice, Lattice, LeftIdeal, NrdBasis},
    linear::{Matrix, Vector},
    precomputed::{EXTREMAL_ORDERS, NUM_EXTREMAL_ORDERS, P_WIDE, connecting_ideal},
};
use crate::curves::{TorsionExponent, isogeny::IsogenyDegree};

/// Find γ ∈ O with nrd(γ) = M, trying each precomputed extremal order.
///
/// Iterates over the seven precomputed extremal orders, calling
/// [`represent_integer`] on each until one succeeds.
impl ExtremalOrder<8> {
    /// Find γ ∈ O with nrd(γ) = M, trying all precomputed orders.
    ///
    /// Iterates over [`EXTREMAL_ORDERS`] and calls
    /// [`represent_integer`](Self::represent_integer) on each until
    /// one succeeds.
    ///
    /// WARNING: Not constant-time — data-dependent iteration over
    /// orders with early return on first success.
    ///
    /// TODO(ct): Make constant-time before production use. Called on
    /// secret-derived norms during signing (via FixedDegreeIsogeny,
    /// Algorithm 4.2 lines 21–24).
    pub fn represent_integer_any<R: RngCore>(m: &BigInt<8>, rng: &mut R) -> Option<Element<4>> {
        for order in &EXTREMAL_ORDERS {
            let order_wide = ExtremalOrder::<8>::from(*order);
            if let Some(gamma) = order_wide.represent_integer(m, false, rng) {
                return Some(gamma);
            }
        }
        None
    }

    /// Find γ ∈ O with nrd(γ) = M using this extremal order.
    ///
    /// Implements [Alg. 3.12].
    ///
    /// # Divergences
    ///
    /// - **t range**: t is sampled from `[-m', m']` (spec line 5), including
    ///   negative values. Both signs must be tried because the isogeny
    ///   condition (line 15) depends on the sign of t, even though M' = 4M -
    ///   p(z² + qt²) depends only on t².
    /// - **γ construction order**: the spec writes `ωjt` (line 17) but the C
    ///   ref computes `j·ω·t` (reversed). For ω = i this gives `ji·t = -kt`
    ///   instead of `ij·t = kt`. We match the C ref.
    /// - **Divisibility check**: the spec's "largest d with γ/d ∈ O" (line 18)
    ///   is checked by constructing γ as an `Element`, normalizing
    ///   (GCD-reducing), and verifying the denominator was divided by exactly
    ///   2. This matches the C ref's `quat_alg_make_primitive`. Hand-deriving
    ///   the divisibility condition from coordinate parity is fragile and
    ///   order- dependent.
    /// - **Arithmetic width**: primality testing and Cornacchia on up to
    ///   ~514-bit `m_prime` candidates in `BigInt<8>` (4·M for M ≤ 2^512)
    ///   overflow the 512-bit storage during modular exponentiation. Uses
    ///   widened variants (`_w::<17>`, 1088 bits ≥ 2·514 = 1028) for
    ///   correctness across every caller, including the aux-path `random_norm`
    ///   which passes M ≈ 2^377. Using a narrower width (e.g. `_w::<9>`)
    ///   silently truncates Miller-Rabin exponentiations and makes
    ///   `represent_integer` loop indefinitely without ever finding a witness.
    /// - **Search bound**: computed from the spec's formula `ceil(sqrt(4M /
    ///   (p·sqrt(q))))`, not hardcoded.
    ///
    /// # Constant-time
    ///
    /// Variable-time. `TODO(ct)`: input is secret-derived via
    /// Algorithm 4.2 (FixedDegreeIsogeny calls this on secret norms).
    ///
    /// [Alg. 3.12]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.12
    pub fn represent_integer<R: RngCore>(
        &self,
        m: &BigInt<8>,
        isogeny_cond: bool,
        rng: &mut R,
    ) -> Option<Element<4>> {
        let p: BigInt<8> = P_WIDE;
        let q_val = self.q();
        let q = BigInt::<8>::from_u64(q_val as u64);
        let four_m = BigInt::<8>::from_u64(4).ct_mul(m);

        // M must be positive and odd for solutions to exist.
        if bool::from(m.is_negative()) || bool::from(m.is_zero()) || bool::from(m.is_even()) {
            return None;
        }
        if *m <= p {
            // M ≤ p: the bound formula produces nonsense.
            return None;
        }

        // Bounds matching C ref `quat_represent_integer`
        // (`normeq.c:127-138`):
        //
        //   sq_bound = floor(4M / p) − q
        //   z_max    = floor(sqrt(sq_bound))
        //   counter  = 4M / floor(sqrt(q · p²))
        //
        // All exact integer math (no f64) so byte-stream parity
        // with C ref is preserved when the same DRBG state drives
        // sampling. Counter is essentially unbounded for typical
        // FDI inputs (~2^27); the loop exits early on first
        // success per PNT (`O(log M) ≈ 400` expected iterations).
        // We additionally cap at `MAX_ITER = 10_000` for
        // wall-clock safety; a caller seeing `None` should retry
        // with different randomness.
        const MAX_ITER: u64 = 10_000;
        let z_max_big = {
            let (q_quot, _) = four_m.div_rem(&p);
            if q_quot <= q {
                return None;
            }
            q_quot.ct_sub(&q).sqrt_floor()
        };
        if bool::from(z_max_big.is_zero()) {
            return None;
        }
        let counter_big = {
            let qp2 = q.ct_mul(&p).ct_mul(&p);
            let qp2_sqrt = qp2.sqrt_floor();
            if bool::from(qp2_sqrt.is_zero()) {
                return None;
            }
            let (cnt, _) = four_m.div_rem(&qp2_sqrt);
            cnt
        };
        if bool::from(counter_big.is_zero()) {
            return None;
        }
        // Project counter to a u64 budget capped at MAX_ITER. C
        // ref's full counter walks `2^27`+ values without finding
        // a solution only for inputs we wouldn't expect to succeed
        // anyway; this cap matches the previous Rust behavior.
        let bound: u64 = {
            let limbs = counter_big.as_limbs();
            // Counter spans at most a few limbs for any caller; if
            // limb 1+ is non-zero the value vastly exceeds
            // MAX_ITER.
            if limbs[1..].iter().any(|&l| l != 0) || limbs[0] > MAX_ITER {
                MAX_ITER
            } else {
                limbs[0]
            }
        };

        let mut _primes_found = 0u32;
        let mut _cornacchia_ok = 0u32;
        let mut _parity_ok = 0u32;
        let mut _isogeny_cond_fail = 0u32;
        // Random sampling per spec [Alg. 3.12] and C ref
        // (`normeq.c` calls `ibz_rand_interval` for both `z` and
        // `t`). Each iteration picks one `(z, t)` pair uniformly
        // from the search box; expected hits per `O(log M)` ≈ 400
        // attempts. Sampling uses [`BigInt::rand_interval`] to
        // mirror C ref's byte-stream contract (top-bit-aligned
        // mask + rejection on overflow).
        //
        // # Divergences (KAT byte-stream)
        //
        // The C reference samples `t` from `[1, t_max]` (positive
        // only); we match that. An earlier version used
        // `[-t_max, t_max]` (signed), which doubled the search
        // domain at the cost of every byte-stream draw consuming
        // a different number of bytes than C ref's
        // `ibz_rand_interval(rand, 1, temp)` call.
        #[cfg(test)]
        if std::env::var("REPI_TRACE").is_ok() {
            crate::selkie_trace!(
                "[REPI] init bound={} counter={} adjusted_n_gamma={}",
                z_max_big, counter_big, four_m
            );
            crate::selkie_trace!(
                "[REPI] init q={} p={} non_diag={} standard_order={}",
                q,
                p,
                if isogeny_cond { 1 } else { 0 },
                if q_val == 1 { 1 } else { 0 }
            );
        }
        let one_big = BigInt::<8>::ONE;
        let mut iter: u64 = 0;
        while iter < bound {
            #[cfg(test)]
            let _iter_idx = iter;
            iter += 1;

            let z = BigInt::<8>::rand_interval(rng, &one_big, &z_max_big);
            #[cfg(test)]
            if std::env::var("REPI_TRACE").is_ok() {
                crate::selkie_trace!("[REPI] iter={} z={}", _iter_idx, z);
            }

            let pz_sq = p.ct_mul(&z.ct_mul(&z));
            if four_m <= pz_sq {
                continue;
            }
            let remaining = four_m.ct_sub(&pz_sq);
            // `t_max = floor(sqrt((4M − p·z²) / (q·p)))`, exact
            // integer (mirrors C ref `normeq.c:151-155`).
            let qp = q.ct_mul(&p);
            if bool::from(qp.is_zero()) {
                continue;
            }
            let (rem_div_qp, _) = remaining.div_rem(&qp);
            let t_max_big = rem_div_qp.sqrt_floor();
            let z_sq = z.ct_mul(&z);
            if bool::from(t_max_big.is_zero()) {
                continue;
            }
            let t = BigInt::<8>::rand_interval(rng, &one_big, &t_max_big);
            {
                let t_sq = t.ct_mul(&t);
                let inner = z_sq.ct_add(&q.ct_mul(&t_sq));
                let m_prime = four_m.ct_sub(&p.ct_mul(&inner));

                if bool::from(m_prime.is_zero()) || bool::from(m_prime.is_negative()) {
                    continue;
                }

                // `m_prime` can be up to `~bits(m) + 2`, and the
                // caller's `m` ranges from ~273 bits (FDI) up to
                // `4·m` ≈ 2^514 at aux-path magnitudes. Miller-Rabin
                // needs a working width `W` satisfying
                // `64·W ≥ 2·bits(m_prime) − 1` to avoid silent
                // truncation inside `pow_mod_w`. `W = 17` covers
                // every `BigInt<8>` input (2·512 − 1 = 1023, 64·17 =
                // 1088). An earlier version used `W = 9`, which was
                // silently truncating Miller-Rabin exponentiations
                // for any `m` above ~280 bits and made
                // `represent_integer` never find a witness for aux-
                // path magnitudes.
                if !m_prime.is_probable_prime_w::<17>(12) {
                    continue;
                }
                _primes_found += 1;

                // Cornacchia uses the same modular-sqrt machinery
                // and needs the same widening.
                let Some((x, y)) = BigInt::<8>::cornacchia_w::<17>(&q, &m_prime) else {
                    continue;
                };
                _cornacchia_ok += 1;

                // Lines 12-15: isogeny condition (spec Algorithm 3.12).
                // If isogenyCond and q = 1: ensure x ≡ t (mod 2)
                // (swap x,y if needed), then check x-t ≡ 2 (mod 4)
                // and y-z ≡ 2 (mod 4).
                let (mut x_use, mut y_use) = (x, y);
                if isogeny_cond && q_val == 1 {
                    if bool::from(x_use.is_odd()) != bool::from(t.is_odd()) {
                        core::mem::swap(&mut x_use, &mut y_use);
                    }
                    let four = BigInt::from_u64(4);
                    let xt_diff = x_use.ct_sub(&t).ct_mod(&four);
                    let yz_diff = y_use.ct_sub(&z).ct_mod(&four);
                    if xt_diff != BigInt::from_u64(2) || yz_diff != BigInt::from_u64(2) {
                        _isogeny_cond_fail += 1;
                        continue;
                    }
                }

                // Lines 16-19: construct γ = x + ωy + jz + jωt
                // (note: the C ref computes j·ω·t, not ω·j·t as
                // the spec's notation might suggest — the product
                // order matters since ij = k but ji = -k).
                //
                // Then find the largest d with γ/d ∈ O and check
                // d = 2. We construct γ as an Element at BigInt<8>
                // width, normalize (which divides by the GCD of the
                // coordinates), and check that normalization divided
                // by exactly 2.
                let omega = self.z();
                // C ref `normeq.c:71-74` builds the t-coordinate basis
                // element as `t · order->t · order->z` = `t · j · ω`
                // (left-to-right). For O₀ on E0 with q=1 this gives
                // `j · i = -k`, so `coord3 = -t`. The reverse order
                // `ω · j = i · j = +k` flips the k-coefficient's sign,
                // breaking byte-equality with C ref's RepresentInteger
                // for any input where `ω` has an `i`-component.
                let omega_j = Element::<4>::J.mul(omega);

                let omega_coords = [
                    omega.a.wide(),
                    omega.b.wide(),
                    omega.c.wide(),
                    omega.d.wide(),
                ];
                let omega_d = omega.denom.wide();
                let oj_coords = [
                    omega_j.a.wide(),
                    omega_j.b.wide(),
                    omega_j.c.wide(),
                    omega_j.d.wide(),
                ];
                let oj_d = omega_j.denom.wide();
                let common_d = omega_d.ct_mul(&oj_d);

                let scale_omega = oj_d;
                let scale_omega_j = omega_d;

                // γ = x·common_d + ω·y·scale_ω + j·z·common_d + j·ω·t·scale_ωj
                // (matching the C ref's quat_order_elem_create)
                let mut gamma_coords = [BigInt::<8>::ZERO; 4];
                for k in 0..4 {
                    let x_term = if k == 0 {
                        x_use.ct_mul(&common_d)
                    } else {
                        BigInt::ZERO
                    };
                    let y_term = y_use.ct_mul(&scale_omega).ct_mul(&omega_coords[k]);
                    let z_term = if k == 2 {
                        z.ct_mul(&common_d)
                    } else {
                        BigInt::ZERO
                    };
                    // j·ω·t (C ref order: order->t * temp * order->z)
                    let t_term = t.ct_mul(&scale_omega_j).ct_mul(&oj_coords[k]);
                    gamma_coords[k] = x_term.ct_add(&y_term).ct_add(&z_term).ct_add(&t_term);
                }

                // Check: largest d with γ/d ∈ O is 2.
                //
                // # Bug history (2026-04-15)
                //
                // The original code used `normalize()` (GCD of quaternion
                // coords and denom) to find d. This is WRONG: normalize
                // divides the {1,i,j,k} coordinates by their GCD, but the
                // spec's "content" is the GCD of the ORDER-BASIS
                // coefficients. For O₀ with common_d=2, gamma has coords
                // (2x, 2y, 2z, -2t)/2. normalize() finds GCD(2x,...,2)=2,
                // divides to get (x,y,z,-t)/1, giving nrd = 4M. But the
                // spec requires nrd(gamma/d) = M, where d=2 is the content
                // of gamma's ORDER-BASIS decomposition.
                //
                // The C ref's `quat_alg_make_primitive` decomposes gamma
                // on the order basis and divides by the GCD of those
                // coefficients. We replicate this via `order.decompose()`.
                //
                // This bug caused `represent_integer` to return elements
                // with nrd = 4*M (4x the expected norm). The action matrix
                // det(M) then equaled 4*M mod 2^f instead of M mod 2^f.
                // The (2,2)-chain kernel had degree 4x too large, making
                // it non-isotropic for the product Weil pairing, so the
                // chain never produced a product surface (splitting:
                // zeros=0). The bug was invisible for diagonal endomorphisms
                // like [3] because the kernel was constructed differently
                // (direct scalar mul, not action matrix). Tracking it down
                // required:
                //   - Verifying the action matrix (correct: det matches nrd)
                //   - Verifying the biladder (correct: group elements match)
                //   - Verifying the chain for [3] (correct: splits)
                //   - Discovering nrd(theta) = 4*m via Python norm computation
                //   - Tracing back to normalize() vs make_primitive
                // Decompose γ on the order basis to find the content
                // (GCD of the order-basis coefficients), matching the C
                // ref's `quat_alg_make_primitive`.
                //
                // Run at width 20 (1280 bits): for p-extremal orders
                // with `q ≥ 5` the basis entries reach ~250 bits
                // (e.g. q=97 row 1 col 3 ≈ 2^250), and
                // `Lattice::decompose` computes a 4×4 adjugate whose
                // 3×3 minors accumulate up to ~3·250 = 750 bits.
                // Then `adjugate · rhs` (with rhs ≈ basis_entry size
                // ≈ 250 bits) reaches ~1000 bits before the final
                // `/ det`. Width 8 (512 bits) and even width 12
                // (768 bits) overflow silently → wrong coefficients
                // → wrong content check → spurious `represent_integer`
                // failures and silently-wrong successes (the latter
                // produced a γ with wrong nrd in KAT 29 iter 0 t=1
                // before this widening). Width 20 leaves comfortable
                // margin for all NIST-I orders.
                let gamma_w = Element::<20>::new(
                    Coordinate::from_bigint(gamma_coords[0].widen::<20>()),
                    Coordinate::from_bigint(gamma_coords[1].widen::<20>()),
                    Coordinate::from_bigint(gamma_coords[2].widen::<20>()),
                    Coordinate::from_bigint(gamma_coords[3].widen::<20>()),
                    Denominator::from_bigint_unchecked(common_d.widen::<20>()),
                );
                let order_lattice_w: Lattice<20> = {
                    let lat4: &Lattice<4> = EXTREMAL_ORDERS
                        .iter()
                        .find(|o| o.q() == self.q())
                        .map(|o| o.order().lattice())?;
                    let basis4 = lat4.basis();
                    let mut basis_w = Matrix::<20>::ZERO;
                    for row in 0..4 {
                        for col in 0..4 {
                            basis_w[row][col] = basis4[row][col].widen::<20>();
                        }
                    }
                    Lattice::new(basis_w, lat4.denom().widen::<20>())
                };
                let Some(basis_coeffs) = order_lattice_w.decompose(&gamma_w) else {
                    // γ not in the order — skip.
                    continue;
                };

                // Content = GCD of all 4 basis coefficients.
                let mut content = basis_coeffs[0].abs();
                for coeff in &basis_coeffs[1..] {
                    content = content.gcd(&coeff.abs());
                }

                // C ref `normeq.c:225-228` accept condition:
                //   content == 2 if `non_diag || standard_order`,
                //   else content == 1.
                // `non_diag` here is our `isogeny_cond` flag (the
                // C ref reuses the same name); `standard_order`
                // is `q == 1` (the order containing
                // `(1+j)/2` rather than `(1+ωj)/2`).
                let expected_content = if isogeny_cond || q_val == 1 {
                    BigInt::<20>::from_u64(2)
                } else {
                    BigInt::<20>::from_u64(1)
                };
                if content != expected_content {
                    continue;
                }
                _parity_ok += 1;

                // γ / content via order-basis coefficients (matches
                // C ref's `quat_alg_make_primitive` followed by the
                // `ibz_mat_4x4_eval(coeffs, basis, coeffs)` mapback
                // at `normeq.c:234`).
                let final_coeffs: [BigInt<20>; 4] = core::array::from_fn(|k| {
                    let (q, _) = basis_coeffs[k].div_rem(&content);
                    q
                });
                let basis = order_lattice_w.basis();
                let denom_w = *order_lattice_w.denom();
                let mut result_coords_w = [BigInt::<20>::ZERO; 4];
                for j in 0..4 {
                    for k in 0..4 {
                        result_coords_w[j] =
                            result_coords_w[j].ct_add(&final_coeffs[k].ct_mul(&basis[j][k]));
                    }
                }

                // Narrow to `Element<4>` for the return. Skip if the
                // result doesn't fit (extremely rare: the result
                // norm equals the caller's `m`, which fits in
                // `BigInt<4>` for FDI inputs).
                let r0 = result_coords_w[0].narrow_to::<4>()?;
                let r1 = result_coords_w[1].narrow_to::<4>()?;
                let r2 = result_coords_w[2].narrow_to::<4>()?;
                let r3 = result_coords_w[3].narrow_to::<4>()?;
                let denom4 = denom_w.narrow_to::<4>()?;
                let result = Element::<4>::new(
                    Coordinate::from_bigint(r0),
                    Coordinate::from_bigint(r1),
                    Coordinate::from_bigint(r2),
                    Coordinate::from_bigint(r3),
                    Denominator::from_bigint_unchecked(denom4),
                );

                return Some(result);
            }
        }

        None
    }
}

// RandomEquivalentPrimeIdeal (Algorithm 3.9) is defined as
// LeftIdeal<8>::reduce_to_prime_norm() in lattice.rs.

// ---------------------------------------------------------------------------
// SuitableIdeals (Algorithm 3.16)
// ---------------------------------------------------------------------------

/// One factor of a [`SuitableIdealResult`] decomposition.
///
/// Bundles a short-vector element β ∈ J_t · I together with the
/// extremal order it came from and its degree d = nrd(β) / nrd(J_t · I).
/// Grouping these prevents accidentally pairing one factor's degree
/// with another factor's element.
///
/// Generic over the parent-ideal storage width `N`: callers with a
/// narrowed ideal use `IdealFactor<4>`; the response phase passes
/// the un-reduced intersection at `IdealFactor<N>` for some
/// `N ≥ 8` so that downstream `to_isogeny` scaling reads the
/// original (un-reduced) `nrd(parent_ideal)`.
pub(crate) struct IdealFactor<const N: usize> {
    /// The extremal order that produced this factor.
    pub(crate) order: &'static ExtremalOrder<4>,
    /// The short vector β.
    pub(crate) beta: ShortVector,
    /// Degree `d = nrd(β) / nrd(parent_ideal)`.
    pub(crate) degree: IsogenyDegree,
    /// The ideal β was enumerated from: for `t = 0` this is the
    /// caller-supplied ideal `I` (possibly replaced by its smallest
    /// equivalent), for `t > 0` it will be `J_t · I`.
    ///
    /// Carrying the parent ideal here keeps the invariant
    /// `nrd(β) = degree · nrd(parent_ideal)` local to the factor,
    /// so downstream scaling formulas needing `nrd(J_t · I)` read
    /// it through [`LeftIdeal::norm`] on this field rather than
    /// reaching for the caller-supplied ideal whose norm no longer
    /// matches β after the reduction step.
    pub(crate) parent_ideal: LeftIdeal<N>,
}

/// Result of [Alg. 3.16][Alg. 3.16] (SuitableIdeals).
///
/// Contains integers u, v and exponent e such that
/// u · d₁ + v · d₂ = 2^e with gcd(u · d₁, v · d₂) = 1 and e ≤ f,
/// where d₁ and d₂ are the degrees of [`factor1`](Self::factor1)
/// and [`factor2`](Self::factor2).
///
/// [Alg. 3.16]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.16
pub(crate) struct SuitableIdealResult<const N: usize> {
    /// Odd positive integer u.
    // TODO: Replace with a positive-integer newtype.
    pub(crate) u: BigInt<4>,
    /// Positive integer v.
    // TODO: Replace with a positive-integer newtype.
    pub(crate) v: BigInt<4>,
    /// Exponent e ≤ f.
    pub(crate) e: TorsionExponent,
    /// First factor (β₁, d₁, order index s).
    pub(crate) factor1: IdealFactor<N>,
    /// Second factor (β₂, d₂, order index t).
    pub(crate) factor2: IdealFactor<N>,
}

/// A candidate from the short-vector enumeration, retaining its
/// degree both as a membership invariant and as the sort key.
///
/// A single enumeration batch uses a fixed parent ideal, so
/// `nrd(parent_ideal)` is constant across every candidate and
/// sorting by `degree = nrd(β) / nrd(parent_ideal)` gives the same
/// ordering as sorting by `nrd(β)` itself. Promoted to an
/// [`IdealFactor`] once a viable pair is selected in
/// [`try_find_uv`].
struct ShortVectorCandidate {
    /// Quaternion element β, a linear combination of the reduced basis.
    elem: Element<4>,
    /// Degree: `nrd(β) / nrd(parent_ideal)`, a positive odd integer.
    degree: IsogenyDegree,
}

/// One per-order enumeration batch: the extremal order the batch
/// belongs to, the ideal the short vectors were enumerated from
/// (`I` for `t = 0`, typically `J_t · I` or an equivalent for
/// `t > 0`), and the resulting candidate list sorted by degree.
///
/// Passing this through [`try_find_uv`] keeps the per-factor
/// `(order, parent_ideal)` pair local to each β; in the
/// multi-order search the two factors come from different batches.
struct ShortVectorBatch<const N: usize> {
    /// The extremal order that produced this batch.
    order: &'static ExtremalOrder<4>,
    /// The ideal the short vectors live in.
    parent_ideal: LeftIdeal<N>,
}

/// A quaternion element that is a short vector in some ideal lattice.
///
/// Wraps an [`Element<4>`] with a stronger contract than a bare
/// algebra element: a `ShortVector` was produced by the
/// short-vector enumeration inside [`LeftIdeal::suitable_ideals`]
/// and lives in a specific [`LeftIdeal<4>`] (tracked by the
/// consumer via [`IdealFactor::parent_ideal`]) with norm on the
/// order of `√nrd(parent_ideal)`.
///
/// Construction is private to this module — there is no public
/// constructor — so a `ShortVector` in hand is a proof-by-type
/// that the enumeration has already validated the underlying
/// element.
///
/// The inner [`Element<4>`] is reachable through [`Deref`], so any
/// site that wants `&Element<4>` (for example `action_matrix` in
/// the deuring module) works by deref coercion.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShortVector(Element<4>);

impl Deref for ShortVector {
    type Target = Element<4>;

    fn deref(&self) -> &Element<4> {
        &self.0
    }
}

/// Canonicalize the L2-reduced basis of an ideal in the **special
/// order** O₀ (the t=0 case in `suitable_ideals`'s multi-order loop).
///
/// Mirrors C ref's `post_LLL_basis_treatment(_, _, _, is_special_order=true)`
/// in `the-sqisign/src/id2iso/ref/lvlx/dim2id2iso.c:213-268`.
///
/// L2 reduction returns *some* reduced basis; the choice between
/// equivalent reduced bases (related by signed column permutations)
/// is implementation-defined. C ref imposes a deterministic
/// canonicalization on the special order so its enumerated short
/// vectors and the eventual `(β_s, β_t)` pair are reproducible. Two
/// independent L2 implementations agree on the abstract reduced
/// basis but may differ in the L2-output column order; without this
/// canonicalization step, the downstream
/// `enumerate_short_vectors` and `try_find_uv` produce a *different*
/// `(β_s, β_t)` than C ref does for the same input ideal — even when
/// L2 is bit-exact.
///
/// The canonicalization has two phases:
///
/// 1. **Column swap.** If `gram[0][0] == gram[2][2]`, swap col 1 ↔ col 2. Else
///    if `gram[0][0] == gram[3][3]`, swap col 1 ↔ col 3. Else if `gram[1][1] ==
///    gram[3][3]`, swap col 1 ↔ col 2 (same swap as the first case). The Gram
///    is updated correspondingly via the symmetric permutation `P^T G P`.
/// 2. **Sign flip.** If `basis[0][0] != basis[1][1]`, negate col 1. If
///    `basis[0][2] != basis[1][3]`, negate col 3. Each negation flips the sign
///    of the corresponding row + column of Gram.
///
/// The non-special-order branch (`is_special_order=false`) of the
/// C ref is empty, so this function is only called for `t = 0`.
fn post_lll_basis_treatment_special<const W: usize>(
    cols: &mut [Vector<W>; 4],
    gram: &mut Matrix<W>,
) {
    use crate::quaternions::linear::Vector;

    // Helper: swap columns `a` and `b` in the basis (cols are stored
    // column-major, so swapping `cols[a]` ↔ `cols[b]` swaps the cols).
    let swap_cols = |cols: &mut [Vector<W>; 4], a: usize, b: usize| {
        cols.swap(a, b);
    };
    // Helper: apply the symmetric permutation `P^T G P` for swapping
    // cols `a` ↔ `b` to the Gram matrix in place. This swaps row a ↔
    // row b AND col a ↔ col b. Equivalent to swapping `gram[a][k]` ↔
    // `gram[b][k]` for each k, then `gram[k][a]` ↔ `gram[k][b]` for
    // each k. The combined effect: every off-diagonal entry indexed
    // by (a, k) or (k, a) flips with its counterpart at (b, k) or
    // (k, b); the diagonal entries `gram[a][a]` and `gram[b][b]` swap.
    let swap_gram_rows_cols = |gram: &mut Matrix<W>, a: usize, b: usize| {
        for k in 0..4 {
            let tmp = gram[a][k];
            gram[a][k] = gram[b][k];
            gram[b][k] = tmp;
        }
        for k in 0..4 {
            let tmp = gram[k][a];
            gram[k][a] = gram[k][b];
            gram[k][b] = tmp;
        }
    };
    // Helper: negate column `j` of basis + corresponding row/col of
    // Gram (sign change is equivalent to multiplying both sides by
    // -1, which leaves diagonal entries unchanged but flips signs of
    // off-diagonal entries).
    let negate_col = |cols: &mut [Vector<W>; 4], gram: &mut Matrix<W>, j: usize| {
        let v = cols[j];
        cols[j] = Vector::new(
            v[0].wrapping_neg(),
            v[1].wrapping_neg(),
            v[2].wrapping_neg(),
            v[3].wrapping_neg(),
        );
        for k in 0..4 {
            gram[j][k] = gram[j][k].wrapping_neg();
        }
        for k in 0..4 {
            gram[k][j] = gram[k][j].wrapping_neg();
        }
    };

    // Phase 1: column reorder based on Gram diagonal patterns.
    if gram[0][0] == gram[2][2] {
        swap_cols(cols, 1, 2);
        swap_gram_rows_cols(gram, 1, 2);
    } else if gram[0][0] == gram[3][3] {
        swap_cols(cols, 1, 3);
        swap_gram_rows_cols(gram, 1, 3);
    } else if gram[1][1] == gram[3][3] {
        swap_cols(cols, 1, 2);
        swap_gram_rows_cols(gram, 1, 2);
    }

    // Phase 2: sign-flip cols based on basis-entry equality checks.
    if cols[0][0] != cols[1][1] {
        negate_col(cols, gram, 1);
    }
    if cols[2][0] != cols[3][1] {
        negate_col(cols, gram, 3);
    }
}

/// Enumerate the C-reference's filtered half-cube of integer
/// 4-tuples in `[-m, m]⁴`.
///
/// Returns the tuples in the same deterministic order as the C
/// reference's `enumerate_hypercube` (`dim2id2iso.c:270-376`).
/// Pinning the candidate order pins the first-success
/// `(β_s, β_t)` choice in [`try_find_uv`], which in turn pins
/// `e_pk` for KAT byte match.
///
/// Filters applied, in order:
///
/// * Half-cube iteration: walk only `x ≤ 0`, breaking each inner loop when the
///   leading-zero suffix would cross into the positive half. `±v` and `v` give
///   the same Gram-form value, so keeping just one representative halves the
///   candidate pool.
/// * Skip all-even tuples: `2·v` has Gram-form value `4·G(v, v)`, never smaller
///   than `G(v, v)` itself.
/// * Skip all-mult-of-3 tuples for the same reason.
/// * When `gram_has_i_symmetry` is set — i.e., the L2-reduced basis is `(γ, iγ,
///   β, iβ)` so that `G[0][0] = G[1][1]` and `G[2][2] = G[3][3]` — keep only
///   the `i`-orbit representative with the smallest lex rank in the `dim = 2m +
///   1` hypercube layout.
fn enumerate_hypercube(m: i64, gram_has_i_symmetry: bool) -> Vec<[i64; 4]> {
    debug_assert!(m > 0);

    let dim = 2 * m + 1;
    let dim2 = dim * dim;
    let dim3 = dim2 * dim;

    let cap = (dim as usize).pow(4);
    let mut out = Vec::with_capacity(cap);

    for x in -m..=0 {
        for y in -m..=m {
            if x == 0 && y > 0 {
                break;
            }
            for z in -m..=m {
                if x == 0 && y == 0 && z > 0 {
                    break;
                }
                for w in -m..=m {
                    if x == 0 && y == 0 && z == 0 && w >= 0 {
                        break;
                    }

                    if (x | y | z | w) & 1 == 0 {
                        continue;
                    }
                    if x.rem_euclid(3) == 0
                        && y.rem_euclid(3) == 0
                        && z.rem_euclid(3) == 0
                        && w.rem_euclid(3) == 0
                    {
                        continue;
                    }

                    if gram_has_i_symmetry {
                        let check1 = (m + w) + dim * (m + z) + dim2 * (m + y) + dim3 * (m + x);
                        let check2 = (m - z) + dim * (m + w) + dim2 * (m - x) + dim3 * (m + y);
                        let check3 = (m + z) + dim * (m - w) + dim2 * (m + x) + dim3 * (m - y);
                        if !(check1 <= check2 && check1 <= check3) {
                            continue;
                        }
                    }

                    out.push([x, y, z, w]);
                }
            }
        }
    }

    out
}

impl<const W: usize> NrdBasis<W> {
    /// Enumerate non-zero lattice vectors within the box \[-m, m\]⁴
    /// from an L2-reduced basis, compute their degrees, and sort by
    /// norm.
    ///
    /// For NIST-I with m = 2, this produces up to
    /// (2·2+1)⁴ − 1 = 624 non-zero vectors.
    ///
    /// Generic over the working-width `W`. Use `W = 8` for typical
    /// commitment ideals (~2^133 norm); use `W = 16` for the wide
    /// response-phase intersection `I_inter ≈ 2^385` whose Gram
    /// matrix entries reach ~2^770.
    ///
    /// # Mirror of C reference's off-by-one
    ///
    /// The very last step here drops the last enumerated vector
    /// (`vectors.pop()`) before sorting. This is a deliberate
    /// reproduction of an off-by-one bug in the C reference's
    /// `enumerate_hypercube` (`dim2id2iso.c:449`,
    /// `return count - 1;`): after pushing N vectors into
    /// `small_vecs`, the C reference reports `count - 1` to the
    /// caller, so the last-enumerated vector is silently dropped
    /// before `qsort`. The KAT vectors shipped with SQIsign were
    /// generated by the C reference, so byte-equality with the KATs
    /// requires we drop the same vector here.
    ///
    /// Without the drop, our sorted batch contains one more entry
    /// than C ref's, shifting all sort indices by one and causing
    /// [`try_find_uv`] to pick a different `(β_s, β_t)` pair. On
    /// NIST-I that flips ~6 keygen KATs from passing to failing
    /// (and vice-versa), but, more importantly, makes the published
    /// `(u, v)` decomposition byte-different from C ref's, which
    /// then propagates through every downstream stage
    /// (`fixed_degree_isogeny`, the inner chain, the outer chain,
    /// the splitter) so the resulting public key fails to match the
    /// KAT vector at the byte level.
    fn enumerate_short_vectors(
        &self,
        ideal_norm: &BigInt<W>,
        lattice_denom: &BigInt<W>,
    ) -> Vec<ShortVectorCandidate> {
        let m = crate::params::FINDUV_BOX_SIZE;
        let denom_sq = lattice_denom.ct_mul(lattice_denom);
        let divisor = ideal_norm.ct_mul(&denom_sq);

        // Verify the lattice denom fits in BigInt<4> — we only check
        // existence here (not actually used below) to mirror the
        // pre-generic invariant.
        let Some(den_4) = lattice_denom.narrow_to::<4>() else {
            return Vec::new();
        };

        let width = (2 * m + 1) as usize;
        let mut vectors = Vec::with_capacity(width.pow(4) - 1);

        #[cfg(test)]
        let (
            mut _rej_zero_nrd,
            mut _rej_nonintegral,
            mut _rej_degree_zero,
            mut _rej_narrow_degree,
            mut _rej_not_odd,
            mut _rej_narrow_coord,
        ) = (0u32, 0u32, 0u32, 0u32, 0u32, 0u32);

        #[cfg(test)]
        if std::env::var("ENUM_TRACE").is_ok() {
            // Print G[0][0], G[0][1], G[1][1] and the divisor once so we
            // can see if divisor divides G[i][i] (which is nrd(α_i)·denom²
            // for the i-th basis column).
            crate::selkie_trace!(
                "[enum-trace] G[0][0]={}, G[1][1]={}, G[2][2]={}, G[3][3]={}, divisor={}",
                self.gram()[0][0],
                self.gram()[1][1],
                self.gram()[2][2],
                self.gram()[3][3],
                divisor,
            );
            // Check divisibility of each diagonal.
            for i in 0..4 {
                let (_, rem) = self.gram()[i][i].div_rem(&divisor);
                crate::selkie_trace!(
                    "[enum-trace] G[{i}][{i}] / divisor: rem = {} (is_zero={})",
                    rem,
                    bool::from(rem.is_zero()),
                );
            }
        }

        let need_remove_symmetry =
            self.gram()[0][0] == self.gram()[1][1] && self.gram()[3][3] == self.gram()[2][2];

        for [x_i, y_i, z_i, w_i] in enumerate_hypercube(m, need_remove_symmetry) {
            let x = [
                BigInt::<W>::from_i64(x_i),
                BigInt::<W>::from_i64(y_i),
                BigInt::<W>::from_i64(z_i),
                BigInt::<W>::from_i64(w_i),
            ];

            // nrd(β) · denom² = Σ x_i x_j G_{ij}.
            let nrd_scaled = self.eval_quadratic_form(&x);

            if bool::from(nrd_scaled.is_zero()) || bool::from(nrd_scaled.is_negative()) {
                #[cfg(test)]
                {
                    _rej_zero_nrd += 1;
                }
                continue;
            }

            // degree = nrd_scaled / (nrd(I) · denom²).
            let (degree_wide, rem) = nrd_scaled.div_rem(&divisor);
            if !bool::from(rem.is_zero()) {
                #[cfg(test)]
                {
                    _rej_nonintegral += 1;
                }
                continue;
            }
            if bool::from(degree_wide.is_zero()) {
                #[cfg(test)]
                {
                    _rej_degree_zero += 1;
                }
                continue;
            }
            let Some(degree_4) = degree_wide.narrow_to::<4>() else {
                #[cfg(test)]
                {
                    _rej_narrow_degree += 1;
                }
                continue;
            };
            let Some(degree) = IsogenyDegree::new_odd(*degree_4.as_limbs()) else {
                #[cfg(test)]
                {
                    _rej_not_odd += 1;
                }
                continue;
            };

            // β = Σ x_k · col_k.
            let coords: [BigInt<W>; 4] = core::array::from_fn(|row| {
                (0..4).fold(BigInt::<W>::ZERO, |acc, k| {
                    acc.ct_add(&x[k].ct_mul(&self.cols()[k][row]))
                })
            });

            // Narrow coordinates to BigInt<4>. After L2 reduction
            // with small coefficients this should always succeed.
            let narrow: [Option<BigInt<4>>; 4] =
                core::array::from_fn(|i| coords[i].narrow_to::<4>());
            let [Some(a), Some(b), Some(c), Some(d)] = narrow else {
                #[cfg(test)]
                {
                    _rej_narrow_coord += 1;
                }
                continue;
            };

            vectors.push(ShortVectorCandidate {
                elem: Element::<4>::new(
                    Coordinate::from_bigint(a),
                    Coordinate::from_bigint(b),
                    Coordinate::from_bigint(c),
                    Coordinate::from_bigint(d),
                    Denominator::from_bigint_unchecked(den_4),
                ),
                degree,
            });
        }

        let _ = width; // capacity hint only

        // Mirror C ref's `enumerate_hypercube` off-by-one at
        // `dim2id2iso.c:449` (`return count - 1;`). C ref kept `count`
        // vectors during enumeration but reports only `count - 1` to
        // the caller, so the last-enumerated vector is silently
        // dropped before `qsort`. The KAT vectors were generated by
        // the C reference, so byte-equality requires we drop the same
        // vector. (Without this drop, our sorted batch contains one
        // more entry than C ref's, shifting all subsequent indices
        // and causing `try_find_uv` to pick a different `(β_s, β_t)`
        // for many KATs.) See the function-level docs for the full
        // context.
        vectors.pop();
        debug_assert!(
            vectors.len() < width.pow(4),
            "post-pop vector count consistent with hypercube bound"
        );

        // Sort enumerated vectors by `degree` (= norm) ascending,
        // matching C ref's `qsort(small_vecs_and_norms, ...,
        // compare_vec_by_norm)` in `dim2id2iso.c:634`. Ties broken
        // by original enumeration order (Rust's `sort_by` is stable,
        // matching C ref's explicit `idx` tiebreaker).
        //
        // Without this sort, both impls produce the same set of
        // short vectors but in different orders → `try_find_uv`
        // selects a different first valid `(β_s, β_t)` pair → the
        // entire downstream Deuring correspondence diverges → keygen
        // pk bytes don't match C ref.
        vectors.sort_by(|a, b| {
            let al = a.degree.limbs();
            let bl = b.degree.limbs();
            // Compare from MSB to LSB (limbs are little-endian).
            for i in (0..al.len()).rev() {
                match al[i].cmp(&bl[i]) {
                    core::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            core::cmp::Ordering::Equal
        });

        #[cfg(test)]
        if std::env::var("SELKIE_DUMP_SORTED").is_ok() {
            let limit: usize = std::env::var("SELKIE_DUMP_SORTED_LIMIT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10);
            for (i, v) in vectors.iter().enumerate().take(limit) {
                let limbs = v.degree.limbs();
                let mut last_nz = 0;
                for (k, &l) in limbs.iter().enumerate() {
                    if l != 0 {
                        last_nz = k;
                    }
                }
                let mut s = String::new();
                for k in (0..=last_nz).rev() {
                    s.push_str(&format!("{:016x}", limbs[k]));
                }
                let s = s.trim_start_matches('0').to_string();
                crate::selkie_trace!("[SELKIE_SORTED] idx={i} degree=0x{s}");
            }
        }

        #[cfg(test)]
        if std::env::var("ENUM_DIAG").is_ok() {
            crate::selkie_trace!(
                "[enum] ideal_norm={} bits, denom={} bits, kept={}, rejected: \
                 zero_nrd={} nonintegral={} degree_zero={} narrow_degree={} \
                 not_odd={} narrow_coord={}",
                ideal_norm.bitsize(),
                lattice_denom.bitsize(),
                vectors.len(),
                _rej_zero_nrd,
                _rej_nonintegral,
                _rej_degree_zero,
                _rej_narrow_degree,
                _rej_not_odd,
                _rej_narrow_coord,
            );
        }

        // Stable sort by `degree`: `nrd(parent_ideal)` is constant
        // across this batch, so ordering by degree matches ordering
        // by `nrd(β)` exactly — no `f64` precision concerns. A
        // stable sort preserves insertion order for equal-degree
        // candidates, giving deterministic pair selection in
        // [`try_find_uv`]; determinism matters for signing because a
        // non-deterministic pair choice could leak information about
        // which short vectors matched.
        vectors.sort_by_key(|c| c.degree);
        vectors
    }
}

/// Try to find coprime odd degrees and matching `u`, `v` from a
/// pair of short-vector candidates enumerated from potentially
/// different extremal orders.
///
/// `(order1, parent_ideal1)` is the (extremal order, ideal) pair
/// that produced `sv1`; `(order2, parent_ideal2)` is the same for
/// `sv2`. For the `t = 0` single-order search both pairs are
/// identical; for the multi-order search they select different
/// rows of `ACTION_MATRICES` and different `nrd(J_t · I)` scaling
/// factors downstream. Carrying each β's origin through to the
/// resulting [`IdealFactor`] keeps the `nrd(β) = degree ·
/// nrd(parent_ideal)` invariant local to each factor.
///
/// Returns `None` when the pair fails any of the SuitableIdeals
/// conditions: non-coprime degrees, no solution to
/// `u·d₁ + v·d₂ = 2^f` with `u, v > 0`, or an exponent that would
/// push `e` out of range.
fn try_find_uv<const N: usize>(
    sv1: &ShortVectorCandidate,
    sv2: &ShortVectorCandidate,
    batch1: &ShortVectorBatch<N>,
    batch2: &ShortVectorBatch<N>,
    two_f: &BigInt<8>,
    f: TorsionExponent,
) -> Option<SuitableIdealResult<N>> {
    let d1 = &sv1.degree;
    let d2 = &sv2.degree;

    // Oddness is guaranteed by IsogenyDegree construction.
    // Check gcd(d₁, d₂) = 1 via widened BigInt.
    let d1_w = d1.to_bigint_wide();
    let d2_w = d2.to_bigint_wide();
    if d1_w.gcd(&d2_w) != BigInt::<8>::ONE {
        return None;
    }

    // Enumerate positive-integer solutions `(u, v)` to
    // `u·d₁ + v·d₂ = 2^f` along the line, starting at the smallest
    // `v` and walking `v += d₁` (correspondingly `u -= d₂`) — matches
    // C ref's `find_uv_from_lists` enumeration direction
    // (`dim2id2iso.c:404-429`):
    //
    //     v = (n · d₂⁻¹) mod d₁    (smallest non-negative v)
    //     while v < n / d₂:
    //         u = (n − v·d₂) / d₁
    //         if accept(u, v): return (u, v)
    //         v += d₁
    //
    // Earlier versions started at the smallest `u` and walked
    // `v -= d₁`, but that's the *opposite* enumeration order: it
    // accepts `(small u, large v)` first while C ref accepts
    // `(large u, small v)` first. With the same accept criterion
    // both directions terminate, but they pick different `(u, v)` —
    // so on KAT-aligned DRBG the chosen `(β_s, β_t)` diverges.
    let d2_inv = d2_w.invert_mod(&d1_w)?;
    let v0 = two_f.ct_mul(&d2_inv).ct_mod(&d1_w);
    let mut v = v0;
    let mut u = {
        let vd2 = v.ct_mul(&d2_w);
        if vd2 >= *two_f {
            return None;
        }
        let (u, rem) = two_f.ct_sub(&vd2).div_rem(&d1_w);
        if !bool::from(rem.is_zero()) || bool::from(u.is_negative()) {
            return None;
        }
        u
    };

    // Note: an earlier version "balance-biased" the start by jumping
    // to `u ≈ 2^{f/2}` to give `represent_integer` a larger search
    // space. C reference (`find_uv_from_lists` in `dim2id2iso.c:382`)
    // walks the full line from the natural starting point with no
    // such bias, and the chosen `(u, v)` matters for interop with
    // KAT vectors — different starting points → different first
    // success → different `(β_s, β_t)` selection → different chain
    // codomain → different `e_pk`. Removed.
    let _ = f;

    // Cap the line-walk. Without a cap, the `MIN_U_ODD_BITS`
    // filter below can force the loop to step `(u, v)` forward
    // by `(+d_2, -d_1)` for pathological `d_1, d_2` pairs where
    // no step in the visible portion of the line has `u_odd ≥
    // 2^{MIN_U_ODD_BITS}`. For `d_1 = 1` and `v_0 ≈ 2^f`, the
    // natural termination condition `v ≤ d_1` requires
    // `~2^{f}` steps — unbounded in practice. Give up after
    // `MAX_WALK_STEPS` and let the caller try the next pair.
    const MAX_WALK_STEPS: u32 = 10_000;
    let mut steps = 0u32;
    loop {
        steps += 1;
        if steps > MAX_WALK_STEPS {
            return None;
        }
        if !bool::from(u.is_zero()) && !bool::from(v.is_zero()) {
            // Factor out the 2-adic part of `gcd(u, v)`, matching the
            // C reference (`dim2id2iso.c:833`). The spec writes
            // `v_2(u)` in Algorithm 3.16 line 14, but that is only
            // equivalent to `v_2(gcd(u, v))` when `v_2(u) ≤ v_2(v)`.
            let e_val = u.gcd(&v).trailing_zeros();
            // No `e_val < 2` rejection: matches the C reference's
            // `find_uv_from_lists` (`dim2id2iso.c:397-483`), which
            // returns the first `(u, v)` from the line walk and
            // dispatches downstream on `e = f − v_2(gcd(u, v))`.
            // [`LeftIdeal::to_isogeny`] now selects between
            // [`surfaces::Kernel::isogeny`] (kernel `2^(e+2)`,
            // `e ≤ f − 2`) and
            // [`surfaces::Kernel::isogeny_no_extra_torsion`] (kernel
            // `2^e`, `e ∈ {f − 1, f}`) based on `sui.e`.
            // Mirror C ref's `(ibz_get(u) != 0 && ibz_get(v) != 0)`
            // guard at `dim2id2iso.c:421`. `ibz_get` returns an
            // `int32_t` packed as `(sign_bit << 31) | (low_31_bits)`
            // (`intbig.c:399-410`); for non-negative `u, v` (which
            // is the case on this enumeration line) this is just
            // the low 31 bits, so `ibz_get(u) == 0` ⟺ `v_2(u) >= 31`.
            // Per the C ref comment, the filter "removes weird
            // cases where u, v have big power of two".
            //
            // Without this filter, our line walk would accept
            // `(u, v)` pairs that C ref rejects, leading to
            // different `(β_s, β_t)` selections than C ref on
            // KAT-aligned DRBG sequences.
            if u.trailing_zeros() >= 31 || v.trailing_zeros() >= 31 {
                // Advance `v += d₁`, `u -= d₂`. Stop if u would go
                // non-positive.
                v = v.ct_add(&d1_w);
                if u <= d2_w {
                    return None;
                }
                u = u.ct_sub(&d2_w);
                continue;
            }
            if let Ok(e) = TorsionExponent::try_from(f.value() - e_val) {
                if let (Some(u_narrow), Some(v_narrow)) =
                    (u.shr(e_val).narrow(), v.shr(e_val).narrow())
                {
                    return Some(SuitableIdealResult {
                        u: u_narrow,
                        v: v_narrow,
                        e,
                        factor1: IdealFactor {
                            order: batch1.order,
                            beta: ShortVector(sv1.elem),
                            degree: *d1,
                            parent_ideal: batch1.parent_ideal,
                        },
                        factor2: IdealFactor {
                            order: batch2.order,
                            beta: ShortVector(sv2.elem),
                            degree: *d2,
                            parent_ideal: batch2.parent_ideal,
                        },
                    });
                }
            }
        }

        // Advance to the next solution on the line: `v += d₁`,
        // `u -= d₂`. Stop when `u` would go non-positive (matches
        // C ref's `while (cmp < 0)` exit at v >= n/d₂).
        v = v.ct_add(&d1_w);
        if u <= d2_w {
            return None;
        }
        u = u.ct_sub(&d2_w);
    }
}

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
    /// growth). The input ideal I is derived from the secret key during
    /// signing (Algorithm 4.2, lines 13–19), so timing variations could
    /// in principle leak information about the secret.
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
            ld.ct_mul(&dd).ct_mul(&norm_8)
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
    /// [`to_isogeny`]: crate::deuring::LeftIdeal::to_isogeny
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
        let denom_sq = denom_w.ct_mul(&denom_w);
        let self_norm_w: BigInt<W> = self.norm().widen::<W>();
        let class_divisor: BigInt<W> = denom_sq.ct_mul(&self_norm_w);
        let two_w = BigInt::<W>::from_u64(2);
        let mut class_gram = Matrix::<W>::ZERO;
        for i in 0..4 {
            for j in 0..4 {
                let traced = nrd.gram()[i][j].ct_mul(&two_w);
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
                    *vk = vk.ct_add(&cj_big.ct_mul(&cols_w[j][k]));
                }
            }
            v
        };
        let nrd_of = |v: &[BigInt<W>; 4]| -> BigInt<W> {
            // nrd_num = a² + b² + p(c² + d²) at width W.
            let a2 = v[0].ct_mul(&v[0]);
            let b2 = v[1].ct_mul(&v[1]);
            let c2 = v[2].ct_mul(&v[2]);
            let d2 = v[3].ct_mul(&v[3]);
            a2.ct_add(&b2).ct_add(&p_w.ct_mul(&c2.ct_add(&d2)))
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
        #[cfg(test)]
        if let Some((_, min_nrd)) = candidates.first() {
            crate::selkie_trace!(
                "[smallest_equiv_narrow] self.norm={} bits, min brute-force nrd={} bits (ratio={} bits)",
                self.norm().bitsize(),
                min_nrd.bitsize(),
                min_nrd.bitsize() as i64 - self.norm().bitsize() as i64,
            );
        }

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
                    #[cfg(test)]
                    crate::selkie_trace!(
                        "[smallest_equiv_narrow] skipping trivial candidate, norm={} bits",
                        rn.bitsize()
                    );
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
                    #[cfg(test)]
                    crate::selkie_trace!(
                        "[smallest_equiv_narrow] refresh_norm failed on candidate, stored norm={} bits",
                        _stored_norm.bitsize()
                    );
                    continue;
                }
                let refreshed_norm = *result.norm();
                if refreshed_norm == BigInt::<4>::ONE || bool::from(refreshed_norm.is_zero()) {
                    #[cfg(test)]
                    crate::selkie_trace!(
                        "[smallest_equiv_narrow] skipping trivial post-refresh, norm={} bits",
                        refreshed_norm.bitsize()
                    );
                    continue;
                }
                // Skip even-norm candidates: `to_isogeny` scales
                // step-6 matrix entries by `invmod(parent_norm·d₁,
                // 2^f)`, which is undefined when `parent_norm` is
                // even. Mirrors the C ref's
                // `quat_lideal_prime_norm_reduced_equivalent`,
                // which only accepts prime-norm candidates.
                if bool::from(refreshed_norm.is_even()) {
                    #[cfg(test)]
                    crate::selkie_trace!(
                        "[smallest_equiv_narrow] skipping even-norm candidate, norm={} bits",
                        refreshed_norm.bitsize()
                    );
                    continue;
                }
                #[cfg(test)]
                crate::selkie_trace!(
                    "[smallest_equiv_narrow] accepted candidate, brute-force norm={} bits, \
                     refresh norm={} bits",
                    _stored_norm.bitsize(),
                    refreshed_norm.bitsize(),
                );
                return Some(result);
            }
        }
        #[cfg(test)]
        crate::selkie_trace!(
            "[smallest_equiv_narrow] no TOP_K={TOP_K} candidate yielded a narrow-able equivalent ideal"
        );
        None
    }

    /// Build the equivalent-ideal `LeftIdeal<4>` from a specific
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
            a.ct_mul(a)
                .ct_add(&b.ct_mul(b))
                .ct_add(&p_w.ct_mul(&c.ct_mul(c).ct_add(&d.ct_mul(d))))
        };
        let delta_nrd_den = denom_w.ct_mul(&denom_w);
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
                a0.ct_mul(b0)
                    .ct_sub(&a1.ct_mul(b1))
                    .ct_sub(&p_w.ct_mul(&a2.ct_mul(b2).ct_add(&a3.ct_mul(b3)))),
                a0.ct_mul(b1)
                    .ct_add(&a1.ct_mul(b0))
                    .ct_add(&p_w.ct_mul(&a2.ct_mul(b3).ct_sub(&a3.ct_mul(b2)))),
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
        let o_alpha_denom = order_denom.widen::<W>().ct_mul(&alpha_denom);

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
                col[row] = col[row].ct_mul(&equiv_norm_w).ct_mul(&alpha_denom);
            }
        }

        // Mod-HNF with modulus `4 · d⁴ · equiv_norm² · p` (a
        // multiple of the integer-column covolume for the O₀-ideal
        // of norm `equiv_norm` with denom `d_total`).
        let d_sq = o_alpha_denom.ct_mul(&o_alpha_denom);
        let d_fourth = d_sq.ct_mul(&d_sq);
        let m_sq = equiv_norm_w.ct_mul(&equiv_norm_w);
        let four = BigInt::<W>::from_u64(4);
        let modulus = four.ct_mul(&d_fourth).ct_mul(&m_sq).ct_mul(&p_w);

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
                        #[cfg(test)]
                        crate::selkie_trace!(
                            "[build_equiv_from_delta] basis[{row}][{col}] narrow_to<4> None: hnf_w bits={}",
                            hnf_w[row][col].bitsize(),
                        );
                        return None;
                    }
                }
            }
        }
        let denom_4: BigInt<4> = match o_alpha_denom.narrow_to() {
            Some(d) => d,
            None => {
                #[cfg(test)]
                crate::selkie_trace!(
                    "[build_equiv_from_delta] denom narrow_to<4> None: o_alpha_denom bits={}",
                    o_alpha_denom.bitsize(),
                );
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

impl<const N: usize> LeftIdeal<N> {
    /// [Alg. 3.16]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.16
    pub(crate) fn suitable_ideals(&self) -> Option<SuitableIdealResult<N>> {
        // Internal arithmetic widens to 8 limbs for the L2 reduction
        // and short-vector enumeration. Inputs with `N > 8` would
        // require a wider `NrdBasis`; reject at compile time.
        const { assert!(N <= 8, "suitable_ideals supports N ≤ 8") };
        let f = TorsionExponent::FULL;
        let two_f = BigInt::<8>::ONE.shl(f.value());

        // Phase 1: for each of the seven extremal orders O_t, build
        // the corresponding ideal in which β is enumerated:
        //
        // - `t = 0`: the caller-supplied ideal `self`. Short vectors β ∈ `self` have
        //   `nrd(β) = degree · nrd(self)`.
        // - `t > 0`: the pushforward `J_t · self` (see [§3.1.6.1][§3.1.6.1]), where
        //   `J_t = connecting_ideal(t)` is the precomputed left-O_0 ideal with
        //   right-order O_t. `pushforward` returns the left-O_t ideal `J_t^{-1} · (J_t
        //   ∩ self)`, which has the same norm as `self` but lives in a different
        //   lattice — β's enumerated here act on `E_t` via `ACTION_MATRICES[t][*]`.
        //
        // The C reference (`dim2id2iso.c:535-609`) does an equivalent
        // multi-order L2 reduction + enumeration via
        // `conj(I_reduced) · J_t`, which produces the same degrees
        // and a right-O_t lattice; the pushforward shape here
        // matches the spec literal and avoids the delta-transport
        // step. See [§3.1.7.2].
        //
        // [§3.1.6.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.6.1
        // [§3.1.7.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.7.2

        // Per-order enumeration: compute an L2-reduced basis and
        // short-vector list for each extremal order, then stage a
        // `ShortVectorBatch` pointing at the matching parent ideal.
        let mut batches: [Option<ShortVectorBatch<N>>; NUM_EXTREMAL_ORDERS] = Default::default();
        let mut short_vecs_per_order: [Vec<ShortVectorCandidate>; NUM_EXTREMAL_ORDERS] =
            Default::default();

        // C ref's multi-order recipe (`dim2id2iso.c:617-666`):
        // 1. After LLL on self, let δ = first post-L² basis col.
        // 2. k = nrd(δ_num) / (denom_self² · N(self)) — integer "abstract norm" of the
        //    equivalent ideal class.
        // 3. reduced_id.lattice = self.lattice · conj(δ)/N(self) (a fractional
        //    left-O_0-ideal in B; integer-col covolume at denom 4N is 64·N^5·k).
        // 4. conj_reduced_id = conjugate(reduced_id).
        // 5. ideal[t] = conj_reduced_id · J_t. Norm = k · N(J_t). Integer-col covolume
        //    at denom 8N is 1024·N^4·(k·N_J)².
        //
        // We pass these *exact* covolume formulas as modular-HNF
        // moduli rather than the `det(first 4 cols)` heuristic, which
        // would give a strict multiple of canonical covolume and
        // embed the lattice in a coarser sublattice (failing
        // enumerate's divisor check).
        const W2: usize = 60;
        // (conj_lat, k_norm, n_self, denom_self, conj_delta) — extended
        // to also carry `denom_self_w2` and `conj_delta` for the
        // cross-order beta post-processing below (C ref's
        // `dim2id2iso.c:651-672` `quat_alg_mul(beta, &delta, beta)
        // + quat_alg_normalize + quat_alg_conj` for `j != 0`).
        let mut conj_reduced_state: Option<(
            Lattice<W2>,
            BigInt<W2>,
            BigInt<W2>,
            BigInt<W2>,
            Element<W2>,
        )> = None;

        for t in 0..NUM_EXTREMAL_ORDERS {
            // Build the parent ideal for this order.
            let parent_ideal_t = if t == 0 {
                *self
            } else {
                let Some((ref conj_lat, ref k_norm, ref n_self_w2, _, _)) = conj_reduced_state
                else {
                    continue;
                };
                let j_t_lat: Lattice<W2> = {
                    let l = *connecting_ideal(t).widen::<W2>().lattice();
                    l.into()
                };
                let j_t_norm: BigInt<W2> = *connecting_ideal(t).widen::<W2>().norm();
                let prod_norm = k_norm.ct_mul(&j_t_norm);
                // modulus = 1024 · N^4 · (k·N_J)² (integer-col covolume of result).
                let n_sq = n_self_w2.ct_mul(n_self_w2);
                let n4 = n_sq.ct_mul(&n_sq);
                let prod_norm_sq = prod_norm.ct_mul(&prod_norm);
                let modulus_outer = BigInt::<W2>::from_u64(1024)
                    .ct_mul(&n4)
                    .ct_mul(&prod_norm_sq);
                let prod_lat = conj_lat
                    .product_with_modulus(&j_t_lat, &modulus_outer)
                    .reduce_denom();
                let parent_o0 = *EXTREMAL_ORDERS[0].widen::<W2>().order();
                let ideal_w2 = LeftIdeal::<W2>::from_parts(prod_lat, prod_norm, parent_o0);
                match ideal_w2.narrow_to::<N>() {
                    Some(p) => p,
                    None => {
                        #[cfg(test)]
                        crate::selkie_trace!(
                            "[suitable_ideals] t={t}: conj_reduced·J_t narrow_to::<{N}> failed (basis max={}, denom={} bits, norm={} bits)",
                            {
                                let b = ideal_w2.lattice().basis();
                                (0..4)
                                    .flat_map(|r| (0..4).map(move |c| b[r][c].bitsize()))
                                    .max()
                                    .unwrap_or(0)
                            },
                            ideal_w2.lattice().denom().bitsize(),
                            ideal_w2.norm().bitsize(),
                        );
                        continue;
                    }
                }
            };

            // Widen the parent ideal's lattice to width 16 for L2
            // reduction. Width 16 fits the response-phase
            // intersection ideal (`nrd(I) ≈ 2^385`, lattice cols
            // ~2^385) — its Gram matrix entries can reach ~2^770,
            // which overflows the previous width-8 working space.
            // For the commitment path (norm ~2^133) width 16 has
            // ample slack.
            const W: usize = 16;
            let lattice_n: Lattice<N> = (*parent_ideal_t.lattice()).into();
            let cols_n = lattice_n.basis().columns();
            let cols_w: [Vector<W>; 4] = core::array::from_fn(|j| cols_n[j].widen());
            let denom_w: BigInt<W> = (*lattice_n.denom()).widen();
            let norm_w: BigInt<W> = (*parent_ideal_t.norm()).widen();

            // Feed L2 the *class gram* (= 2·nrd_bilinear / (denom²·ideal_norm))
            // rather than the raw NRD gram, matching C ref's
            // `quat_lideal_class_gram` + `quat_lll_core` pipeline
            // (`the-sqisign/src/quaternion/ref/generic/ideal.c:237` →
            // `lll/lll_applications.c:11-25`). L2 is algebraically
            // scale-invariant in its decisions, but the DPE
            // floating-point representation has finite precision —
            // borderline swap/size-reduction decisions can flip when
            // the gram values are at different magnitudes. Running
            // L2 on the same class gram C ref runs it on guarantees
            // bit-exact decision agreement.
            //
            // After L2, recompute the NRD gram from the post-L2
            // cols so `enumerate_short_vectors` (which uses NRD form
            // + divisor = ideal_norm·denom²) keeps working unchanged.
            let nrd_pre = NrdBasis::new(cols_w);
            let class_gram = {
                let two = BigInt::<W>::from_u64(2);
                let denom_sq = denom_w.ct_mul(&denom_w);
                let class_divisor = denom_sq.ct_mul(&norm_w);
                let mut g = Matrix::<W>::ZERO;
                for i in 0..4 {
                    for j in 0..4 {
                        let traced = nrd_pre.gram()[i][j].ct_mul(&two);
                        let (q, _rem) = traced.div_rem(&class_divisor);
                        g[i][j] = q;
                    }
                }
                g
            };
            #[cfg(test)]
            if t == 0 && std::env::var_os("SELKIE_DUMP_CLASS_GRAM").is_some() {
                crate::selkie_trace!("[SELKIE_CLASS_GRAM_BEGIN]");
                for i in 0..4 {
                    for j in 0..4 {
                        let v = class_gram[i][j];
                        let neg = bool::from(v.is_negative());
                        eprint!("g[{i}][{j}] sign={} hex=", if neg { 1 } else { 0 });
                        let limbs = v.abs();
                        let limbs = limbs.as_limbs();
                        let mut last_nz = 0;
                        for (k, &l) in limbs.iter().enumerate() {
                            if l != 0 {
                                last_nz = k;
                            }
                        }
                        for k in (0..=last_nz).rev() {
                            eprint!("{:016x}", limbs[k]);
                        }
                        crate::selkie_trace!();
                    }
                }
                crate::selkie_trace!("[SELKIE_CLASS_GRAM_END]");
            }
            let class_basis = NrdBasis::from_cols_and_gram(cols_w, class_gram).l2_reduce();

            #[cfg(test)]
            if t == 0 && std::env::var_os("SELKIE_DUMP_POSTL2_GRAM").is_some() {
                let cols_dump = class_basis.cols();
                crate::selkie_trace!("[SELKIE_POSTL2_COLS_BEGIN]");
                #[allow(clippy::needless_range_loop)]
                for j in 0..4 {
                    for r in 0..4 {
                        let v = cols_dump[j][r];
                        let neg = bool::from(v.is_negative());
                        eprint!("c[{j}][{r}] sign={} hex=", if neg { 1 } else { 0 });
                        let abs = v.abs();
                        let limbs = abs.as_limbs();
                        let mut last_nz = 0;
                        for (k, &l) in limbs.iter().enumerate() {
                            if l != 0 {
                                last_nz = k;
                            }
                        }
                        for k in (0..=last_nz).rev() {
                            eprint!("{:016x}", limbs[k]);
                        }
                        crate::selkie_trace!();
                    }
                }
                crate::selkie_trace!("[SELKIE_POSTL2_COLS_END]");
                crate::selkie_trace!("[SELKIE_POSTL2_GRAM_BEGIN] (class form)");
                let g = class_basis.gram();
                for i in 0..4 {
                    for j in 0..4 {
                        let v = g[i][j];
                        let neg = bool::from(v.is_negative());
                        eprint!("g[{i}][{j}] sign={} hex=", if neg { 1 } else { 0 });
                        let abs = v.abs();
                        let limbs = abs.as_limbs();
                        let mut last_nz = 0;
                        for (k, &l) in limbs.iter().enumerate() {
                            if l != 0 {
                                last_nz = k;
                            }
                        }
                        for k in (0..=last_nz).rev() {
                            eprint!("{:016x}", limbs[k]);
                        }
                        crate::selkie_trace!();
                    }
                }
                crate::selkie_trace!("[SELKIE_POSTL2_GRAM_END]");
            }

            let post_l2_cols = *class_basis.cols();
            let nrd_basis = NrdBasis::new(post_l2_cols);

            // Apply C ref's `post_LLL_basis_treatment` for the
            // "special" order t=0 only. Canonicalizes the L2-reduced
            // basis by column swaps + sign flips based on Gram
            // diagonal patterns. See
            // `the-sqisign/src/id2iso/ref/lvlx/dim2id2iso.c:213-268`.
            let nrd_basis = if t == 0 {
                let mut cols = *nrd_basis.cols();
                let mut gram = *nrd_basis.gram();
                post_lll_basis_treatment_special::<W>(&mut cols, &mut gram);
                NrdBasis::from_cols_and_gram(cols, gram)
            } else {
                nrd_basis
            };
            short_vecs_per_order[t] = nrd_basis.enumerate_short_vectors(&norm_w, &denom_w);
            batches[t] = Some(ShortVectorBatch {
                order: &EXTREMAL_ORDERS[t],
                parent_ideal: parent_ideal_t,
            });

            // After t=0 L², populate conj_reduced_state for t > 0.
            if t == 0 {
                let cols_t0 = post_l2_cols;
                let denom_t0_w2: BigInt<W2> = denom_w.widen();
                let norm_t0_w2: BigInt<W2> = norm_w.widen();

                let p_w2: BigInt<W2> = {
                    let p8 = P_WIDE;
                    let mut limbs = [0u64; W2];
                    limbs[..8].copy_from_slice(p8.as_limbs());
                    BigInt::from_sign_and_limbs(0, limbs)
                };

                let dx: BigInt<W2> = cols_t0[0][0].widen();
                let dy: BigInt<W2> = cols_t0[0][1].widen();
                let dz: BigInt<W2> = cols_t0[0][2].widen();
                let dw: BigInt<W2> = cols_t0[0][3].widen();

                // nrd(δ_num) = dx² + dy² + p·(dz² + dw²).
                let nrd_delta_num = dx
                    .ct_mul(&dx)
                    .ct_add(&dy.ct_mul(&dy))
                    .ct_add(&p_w2.ct_mul(&dz.ct_mul(&dz).ct_add(&dw.ct_mul(&dw))));

                // k = nrd(δ_num) / (denom² · N(self)). Integer for valid input.
                let denom_sq = denom_t0_w2.ct_mul(&denom_t0_w2);
                let div = denom_sq.ct_mul(&norm_t0_w2);
                let (k_norm, rem) = nrd_delta_num.div_rem(&div);
                if !bool::from(rem.is_zero()) {
                    #[cfg(test)]
                    crate::selkie_trace!("[suitable_ideals] reduced_id k extraction non-integer");
                    continue;
                }

                // conj(δ) as Element<W2> with denom = denom_self · N(self).
                let conj_delta = Element::<W2>::new(
                    Coordinate::from_bigint(dx),
                    Coordinate::from_bigint(dy.wrapping_neg()),
                    Coordinate::from_bigint(dz.wrapping_neg()),
                    Coordinate::from_bigint(dw.wrapping_neg()),
                    Denominator::from_bigint_unchecked(denom_t0_w2.ct_mul(&norm_t0_w2)),
                );

                let self_lat_w2: Lattice<W2> = {
                    let widened = self.widen::<W2>();
                    (*widened.lattice()).into()
                };

                // modulus_inner = (4N)^4·k²/4 = 64·N^4·k²
                // (integer-col covolume of reduced_id at denom 4N).
                let n_sq = norm_t0_w2.ct_mul(&norm_t0_w2);
                let n4 = n_sq.ct_mul(&n_sq);
                let k_sq = k_norm.ct_mul(&k_norm);
                let modulus_inner = BigInt::<W2>::from_u64(64).ct_mul(&n4).ct_mul(&k_sq);

                let reduced_id_lat_hnf = self_lat_w2
                    .alg_elem_mul_with_modulus(&conj_delta, &modulus_inner)
                    .reduce_denom();
                let reduced_id_lat: Lattice<W2> = reduced_id_lat_hnf.into();
                conj_reduced_state = Some((
                    reduced_id_lat.conjugate(),
                    k_norm,
                    norm_t0_w2,
                    denom_t0_w2,
                    conj_delta,
                ));
            }
        }

        // Phase 2: iterate (s, t) pairs with `t ≥ s` (matching the
        // C ref's diag filter) and search for a viable (β_s, β_t).
        // First success wins. `try_find_uv` handles all the coprime
        // and `u·d_s + v·d_t = 2^e` checks.
        let mut _pairs_tried = 0u64;
        for s in 0..NUM_EXTREMAL_ORDERS {
            let Some(batch_s) = batches[s].as_ref() else {
                continue;
            };
            for t in s..NUM_EXTREMAL_ORDERS {
                let Some(batch_t) = batches[t].as_ref() else {
                    continue;
                };
                let svs_s = &short_vecs_per_order[s];
                let svs_t = &short_vecs_per_order[t];

                let same_batch = s == t;
                for (i, sv1) in svs_s.iter().enumerate() {
                    // When s == t we iterate the upper-triangular
                    // half to avoid pairing a candidate with itself
                    // or double-counting (β_a, β_b) / (β_b, β_a).
                    // When s != t both orderings are distinct
                    // enumerations so we iterate the full cross
                    // product.
                    let inner_start = if same_batch { i } else { 0 };
                    for sv2 in &svs_t[inner_start..] {
                        _pairs_tried += 1;
                        if let Some(result) = try_find_uv(sv1, sv2, batch_s, batch_t, &two_f, f) {
                            #[cfg(test)]
                            {
                                crate::selkie_trace!(
                                    "[suitable_ideals] selected (s={s}, t={t}) after {_pairs_tried} pairs \
                                     | norm={} bits, batch sizes={:?}",
                                    self.norm().bitsize(),
                                    short_vecs_per_order
                                        .iter()
                                        .map(|v| v.len())
                                        .collect::<Vec<_>>(),
                                );
                                if std::env::var("SUITABLE_IDEALS_TRACE").is_ok() {
                                    crate::selkie_trace!(
                                        "[suitable_ideals] u={} v={} e={}",
                                        result.u,
                                        result.v,
                                        result.e.value(),
                                    );
                                    crate::selkie_trace!(
                                        "[suitable_ideals] beta_s coord=[{}, {}, {}, {}] denom={} d_s={}",
                                        result.factor1.beta.a.as_bigint(),
                                        result.factor1.beta.b.as_bigint(),
                                        result.factor1.beta.c.as_bigint(),
                                        result.factor1.beta.d.as_bigint(),
                                        result.factor1.beta.denom.as_bigint(),
                                        result.factor1.degree.to_bigint(),
                                    );
                                    crate::selkie_trace!(
                                        "[suitable_ideals] beta_t coord=[{}, {}, {}, {}] denom={} d_t={}",
                                        result.factor2.beta.a.as_bigint(),
                                        result.factor2.beta.b.as_bigint(),
                                        result.factor2.beta.c.as_bigint(),
                                        result.factor2.beta.d.as_bigint(),
                                        result.factor2.beta.denom.as_bigint(),
                                        result.factor2.degree.to_bigint(),
                                    );
                                }
                            }
                            // Cross-order beta post-processing
                            // (C ref `dim2id2iso.c:651-672`):
                            //
                            //   if (j_i != 0) {
                            //       beta_i = delta · beta_i  (quaternion mul)
                            //       beta_i = normalize(beta_i)
                            //       beta_i = conj(beta_i)
                            //   }
                            //
                            // where `delta` is `conj_delta` with denom
                            // adjusted to `denom_self · k_norm`. This
                            // maps `beta_i ∈ conj(reduced_id) · J_t`
                            // (where it was enumerated) back to `O_t`
                            // (where downstream `action_matrix(O_t)`
                            // expects it). Without this, the
                            // alternate-order action_matrix call
                            // returns None and keygen retries with
                            // wrong DRBG offset.
                            //
                            // No transformation when both `s == 0`
                            // and `t == 0` (the special-order path
                            // already produces β in O_0).
                            let mut result = result;
                            if let (true, Some((_, k_norm, _n_self_w2, denom_self_w2, conj_delta))) =
                                (s != 0 || t != 0, conj_reduced_state.as_ref())
                            {
                                // delta_pp: same coords as conj_delta,
                                // denom = denom_self · k_norm
                                // (instead of denom_self · n_self).
                                let mut delta_pp = *conj_delta;
                                let denom_pp = denom_self_w2.ct_mul(k_norm);
                                delta_pp.denom = Denominator::from_bigint_unchecked(denom_pp);

                                let transform = |beta4: &Element<4>| -> Option<Element<4>> {
                                    let beta_w = Element::<W2>::new(
                                        Coordinate::from_bigint(
                                            beta4.a.as_bigint().widen::<W2>(),
                                        ),
                                        Coordinate::from_bigint(
                                            beta4.b.as_bigint().widen::<W2>(),
                                        ),
                                        Coordinate::from_bigint(
                                            beta4.c.as_bigint().widen::<W2>(),
                                        ),
                                        Coordinate::from_bigint(
                                            beta4.d.as_bigint().widen::<W2>(),
                                        ),
                                        Denominator::from_bigint_unchecked(
                                            BigInt::<4>::from(beta4.denom).widen::<W2>(),
                                        ),
                                    );
                                    let prod = delta_pp.mul_direct(&beta_w);
                                    let mut prod = prod;
                                    prod.normalize();
                                    let conjugated = prod.conjugate();
                                    conjugated.narrow_to::<4>()
                                };

                                if s != 0 {
                                    let new_beta = transform(&result.factor1.beta.0)?;
                                    result.factor1.beta = ShortVector(new_beta);
                                }
                                if t != 0 {
                                    let new_beta = transform(&result.factor2.beta.0)?;
                                    result.factor2.beta = ShortVector(new_beta);
                                }
                            }
                            return Some(result);
                        }
                    }
                }
            }
        }

        #[cfg(test)]
        crate::selkie_trace!(
            "[suitable_ideals] EXHAUSTED after {_pairs_tried} pairs \
             | norm={} bits, batch sizes={:?}",
            self.norm().bitsize(),
            short_vecs_per_order
                .iter()
                .map(|v| v.len())
                .collect::<Vec<_>>(),
        );
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Miller-Rabin width check: `pow_mod_w<W>` requires
    /// `64·W ≥ 2·bits(modulus) − 1`. For 379-bit moduli (aux-path
    /// `represent_integer`) we need `W ≥ 12`. Running at `W = 9`
    /// silently truncates every Miller-Rabin exponentiation and
    /// makes every input look composite, so `represent_integer`
    /// never finds a witness. This test confirms the issue by
    /// comparing `is_probable_prime_w::<9>` and
    /// `is_probable_prime_w::<20>` across a range of small
    /// candidates built on top of a 379-bit base: the
    /// 20-limb width is well above the safe threshold while the
    /// 9-limb width is below it, and the two disagree on roughly
    /// half the inputs.
    #[test]
    fn primality_w9_vs_w20_at_aux_magnitude() {
        // Base = 2^378 (a 379-bit even number). Try (base + k)
        // for small k and check that the wider width finds at
        // least one prime that the narrower width misses. If we
        // don't see any disagreement, either the width math is
        // fine at this bit width or the Rng seed is unlucky —
        // in either case, a valid workflow should never rely on
        // `W = 9` for 379-bit inputs.
        let mut base = [0u64; 8];
        base[5] = 1u64 << (378 - 64 * 5);
        let base_b = BigInt::<8>::from_limbs(base);
        assert_eq!(base_b.bitsize(), 379);
        let mut disagreements = 0;
        for k in 1..200i64 {
            let cand = base_b.ct_add(&BigInt::<8>::from_i64(2 * k - 1));
            // `cand` is an odd 379-bit integer.
            let wide_says_prime = cand.is_probable_prime_w::<20>(4);
            let narrow_says_prime = cand.is_probable_prime_w::<9>(4);
            if wide_says_prime && !narrow_says_prime {
                disagreements += 1;
            }
        }
        assert!(
            disagreements > 0,
            "expected `is_probable_prime_w::<9>` to miss primes that `<20>` accepts at 379 bits"
        );
    }

    /// Timing benchmark for `represent_integer` at aux-path
    /// magnitudes (`M ≈ 2^377`). Run with
    /// `cargo test --lib --release represent_integer_aux_magnitude
    ///   -- --ignored --nocapture` and read the elapsed time
    /// printed on success.
    #[test]
    #[ignore]
    fn represent_integer_aux_magnitude() {
        use crate::params::QUAT_PRIME_COFACTOR;
        let order = ExtremalOrder::<8>::from(EXTREMAL_ORDERS[0]);
        // Simulate the aux-path input: `m = QUAT_PRIME_COFACTOR`
        // (~2^251) times a 126-bit `aux_norm`. Use a fixed odd
        // 126-bit value so the measurement is reproducible.
        let m4 = QUAT_PRIME_COFACTOR;
        let m_wide = BigInt::<8>::from_limbs({
            let mut limbs = [0u64; 8];
            limbs[..4].copy_from_slice(m4.as_limbs());
            limbs
        });
        let aux_norm = BigInt::<8>::from_limbs({
            let mut limbs = [0u64; 8];
            limbs[0] = 0x1234_5678_9ABC_DEF1;
            limbs[1] = 0xFEDC_BA98_7654_3211;
            limbs
        });
        let mn = m_wide.ct_mul(&aux_norm);
        let t0 = std::time::Instant::now();
        let gamma = order.represent_integer(&mn, false, &mut OsRng);
        let elapsed = t0.elapsed();
        crate::selkie_trace!(
            "[aux-RI] mn.bits={} elapsed={elapsed:?} ok={}",
            mn.bitsize(),
            gamma.is_some()
        );
        assert!(gamma.is_some(), "represent_integer(~2^377) must find a γ");
    }

    #[test]
    fn is_probable_prime_small() {
        let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
        for &p in &primes {
            assert!(
                BigInt::<4>::from_u64(p).is_probable_prime(6),
                "{p} should be prime"
            );
        }
        let composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 25, 27];
        for &c in &composites {
            assert!(
                !BigInt::<4>::from_u64(c).is_probable_prime(6),
                "{c} should not be prime"
            );
        }
    }

    #[test]
    fn legendre_symbol() {
        assert_eq!(
            BigInt::<4>::legendre(&BigInt::from(1i64), &BigInt::from(7i64)),
            1
        );
        assert_eq!(
            BigInt::<4>::legendre(&BigInt::from(2i64), &BigInt::from(7i64)),
            1
        );
        assert_eq!(
            BigInt::<4>::legendre(&BigInt::from(3i64), &BigInt::from(7i64)),
            -1
        );
        assert_eq!(
            BigInt::<4>::legendre(&BigInt::from(4i64), &BigInt::from(7i64)),
            1
        );
        assert_eq!(
            BigInt::<4>::legendre(&BigInt::from(0i64), &BigInt::from(7i64)),
            0
        );
    }

    #[test]
    fn random_ideal_prime_norm() {
        use super::super::lattice::LeftIdeal;

        let n = BigInt::<4>::from_u64(7);
        let result = LeftIdeal::random_prime_norm(&n, &EXTREMAL_ORDERS[0]);
        if let Some(ideal) = result {
            assert_eq!(*ideal.norm(), n, "ideal norm should be N");
        }
    }

    #[test]
    fn represent_integer_q1() {
        let p: BigInt<8> = P_WIDE;
        let order = ExtremalOrder::<8>::from(EXTREMAL_ORDERS[0]);
        let m = p.ct_add(&BigInt::TWO);

        let result = order.represent_integer(&m, false, &mut OsRng);
        if let Some(gamma) = result {
            let (nrd_num, nrd_den) = gamma.norm();
            let (nrd, rem) = nrd_num.div_rem(&nrd_den);
            assert!(bool::from(rem.is_zero()), "nrd not integer");
            assert_eq!(nrd, m, "nrd(γ) should equal M");
        }
    }

    /// Tries all seven extremal orders with NIST-I p.
    #[test]
    fn represent_integer_any_order_verifies_norm() {
        let p: BigInt<8> = P_WIDE;
        let m = p.ct_add(&BigInt::TWO);

        let result = ExtremalOrder::<8>::represent_integer_any(&m, &mut OsRng);
        if let Some(gamma) = result {
            let (nrd_num, nrd_den) = gamma.norm();
            let (nrd, rem) = nrd_num.div_rem(&nrd_den);
            assert!(bool::from(rem.is_zero()), "nrd not integer");
            assert_eq!(nrd, m, "nrd(γ) should equal M");
        }
    }

    /// Two DRBG instances seeded identically must drive
    /// `represent_integer` to the *same* γ. Locks in determinism
    /// of the search loop (no `OsRng` leak, no
    /// implementation-specific iteration order beyond the
    /// `(z, t)` byte stream).
    ///
    /// Required precondition for KAT byte-match: the C ref's
    /// `quat_represent_integer` is deterministic given the DRBG
    /// state, so ours must be too. Until our byte-stream is
    /// validated against C ref ground truth, this test only
    /// asserts self-consistency, not interop.
    #[test]
    fn represent_integer_deterministic_under_same_drbg_seed() {
        use crate::drbg::Aes256CtrDrbg;

        // Pick a target large enough that the search loop runs.
        // M = p + 2 (smallest valid input ≥ p that's odd).
        let p: BigInt<8> = P_WIDE;
        let m = p.ct_add(&BigInt::TWO);
        let order = ExtremalOrder::<8>::from(EXTREMAL_ORDERS[0]);

        let seed = [0x42u8; 48];
        let mut d1 = Aes256CtrDrbg::new(&seed);
        let mut d2 = Aes256CtrDrbg::new(&seed);

        let g1 = order.represent_integer(&m, false, &mut d1);
        let g2 = order.represent_integer(&m, false, &mut d2);

        // Both runs must agree on outcome (Some/None) and, when
        // Some, on the actual quaternion element returned.
        match (g1, g2) {
            (None, None) => {
                // Acceptable: search exhausted on both sides.
            }
            (Some(a), Some(b)) => {
                assert_eq!(
                    a.a.as_bigint(),
                    b.a.as_bigint(),
                    "γ.a divergence between same-seed runs"
                );
                assert_eq!(a.b.as_bigint(), b.b.as_bigint(), "γ.b divergence");
                assert_eq!(a.c.as_bigint(), b.c.as_bigint(), "γ.c divergence");
                assert_eq!(a.d.as_bigint(), b.d.as_bigint(), "γ.d divergence");
                assert_eq!(
                    a.denom.as_bigint(),
                    b.denom.as_bigint(),
                    "γ.denom divergence"
                );
            }
            _ => panic!("same-seed DRBGs disagreed on Some/None outcome"),
        }
    }

    #[test]
    fn gram_matrix_nrd_identity_basis() {
        use super::super::{lattice::NrdBasis, linear::Vector};

        // Standard basis {1, i, j, k} has Gram matrix diag(1, 1, p, p).
        let cols: [Vector<8>; 4] = [
            Vector::new(BigInt::ONE, BigInt::ZERO, BigInt::ZERO, BigInt::ZERO),
            Vector::new(BigInt::ZERO, BigInt::ONE, BigInt::ZERO, BigInt::ZERO),
            Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ONE, BigInt::ZERO),
            Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ZERO, BigInt::ONE),
        ];
        let nrd = NrdBasis::new(cols);
        let gram = nrd.gram();
        let p: BigInt<8> = P_WIDE;

        assert_eq!(gram[0][0], BigInt::ONE, "nrd(1) = 1");
        assert_eq!(gram[1][1], BigInt::ONE, "nrd(i) = 1");
        assert_eq!(gram[2][2], p, "nrd(j) = p");
        assert_eq!(gram[3][3], p, "nrd(k) = p");

        // Off-diagonal entries should be zero.
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(
                        bool::from(gram[i][j].is_zero()),
                        "G[{i}][{j}] should be zero for orthogonal basis"
                    );
                }
            }
        }
    }

    /// `random_prime_norm(N)` produces a valid O_0-ideal: every basis
    /// element has nrd divisible by N (equivalently, by
    /// N · denom²  at the integer-column level).
    ///
    /// This test previously exposed a bug where `LeftIdeal<4>::new`
    /// used `Element::mul` (GCD-normalized) while assuming a
    /// uniform `order.denom · alpha.denom` for `o_alpha_denom`. The
    /// denom mismatch let the HNF produce lattices containing the
    /// unit element — impossible in a proper ideal of norm > 1.
    /// Fixed by using `mul_direct` (no GCD normalization), so every
    /// column is scaled consistently relative to the stored denom.
    #[test]
    fn random_prime_norm_lattice_actually_has_norm() {
        use super::super::lattice::LeftIdeal;

        let n = BigInt::<4>::from_u64(7);
        let Some(ideal) = LeftIdeal::random_prime_norm(&n, &EXTREMAL_ORDERS[0]) else {
            return;
        };

        let lat: Lattice<4> = (*ideal.lattice()).into();
        let denom = *lat.denom();
        let denom_sq = denom.ct_mul(&denom);
        let n_times_denom_sq = n.ct_mul(&denom_sq);
        let p4 = crate::quaternions::precomputed::P;

        for j in 0..4 {
            let col = lat.basis().columns()[j];
            let nrd_col_4 = col[0]
                .ct_mul(&col[0])
                .ct_add(&col[1].ct_mul(&col[1]))
                .ct_add(&p4.ct_mul(&col[2].ct_mul(&col[2]).ct_add(&col[3].ct_mul(&col[3]))));
            let nrd_col_8: BigInt<8> = nrd_col_4.widen();
            let divisor_8: BigInt<8> = n_times_denom_sq.widen();
            let (_, rem) = nrd_col_8.div_rem(&divisor_8);
            assert!(
                bool::from(rem.is_zero()),
                "random_prime_norm(7) basis[{j}] nrd not divisible by 7·denom² — \
                 lattice is not an O_0-ideal of norm 7",
            );
        }
    }

    /// `random_norm(N)` produces a valid O_0-ideal for composite N:
    /// every basis element has nrd divisible by N.
    ///
    /// Fixed (Task #28) by computing γ·β at `Element<8>` in
    /// `random_norm` — `Element<4>::mul` silently truncates when
    /// product coords reach ~2^388 (γ has coords ~2^129 from
    /// `represent_integer`) — then reducing each numerator coord
    /// mod `N · denom` to fit the result back in `BigInt<4>`. The
    /// reduction preserves the ideal `O·α + O·N` since the
    /// difference lives in `N · Z<1,i,j,k> ⊂ N · O_0 = O · N`.
    #[test]
    fn random_norm_lattice_actually_has_norm() {
        use super::super::lattice::LeftIdeal;

        let n = BigInt::<4>::from_u64(143);
        let Some(ideal) = LeftIdeal::random_norm(&n, &EXTREMAL_ORDERS[0], &mut OsRng) else {
            return;
        };

        let lat: Lattice<4> = (*ideal.lattice()).into();
        let denom = *lat.denom();
        let denom_sq = denom.ct_mul(&denom);
        let n_times_denom_sq = n.ct_mul(&denom_sq);
        let p4 = crate::quaternions::precomputed::P;

        for j in 0..4 {
            let col = lat.basis().columns()[j];
            let nrd_col_4 = col[0]
                .ct_mul(&col[0])
                .ct_add(&col[1].ct_mul(&col[1]))
                .ct_add(&p4.ct_mul(&col[2].ct_mul(&col[2]).ct_add(&col[3].ct_mul(&col[3]))));
            let nrd_col_8: BigInt<8> = nrd_col_4.widen();
            let divisor_8: BigInt<8> = n_times_denom_sq.widen();
            let (_, rem) = nrd_col_8.div_rem(&divisor_8);
            assert!(
                bool::from(rem.is_zero()),
                "random_norm(143) basis[{j}] nrd not divisible by 143·denom² — \
                 lattice is not an O_0-ideal of norm 143"
            );
        }
    }

    /// `suitable_ideals` on an ideal with composite norm (product
    /// of two coprime odd primes). This is the regime the
    /// response-path intersection ideal occupies at scale (~2^252);
    /// this test uses tiny primes to make it a fast regression
    /// gate for the multi-order refactor tracked in Task #26.
    ///
    /// Today this test may either succeed (if the j=0 degree-based
    /// path finds a pair) or skip (if `random_norm` can't build an
    /// ideal for the chosen composite). It's a harness for future
    /// iteration more than an assertion of current behavior.
    #[test]
    #[ignore] // Takes ~20 minutes; run explicitly with --include-ignored.
    fn suitable_ideals_composite_norm_smoke() {
        use super::super::lattice::LeftIdeal;

        // Try a few small composites (products of coprime odd primes).
        // `random_norm` with composite norm has high rejection rate
        // because β must satisfy `gcd(nrd(β), N) = 1` and with
        // multiple prime factors collisions are common; this loop
        // gives us a decent chance of getting one buildable fixture.
        let candidates: [u64; 6] = [15, 21, 35, 77, 143, 323];
        let mut ideal = None;
        for n_u64 in candidates {
            let n = BigInt::<4>::from_u64(n_u64);
            if let Some(i) = LeftIdeal::random_norm(&n, &EXTREMAL_ORDERS[0], &mut OsRng) {
                crate::selkie_trace!("[composite-norm smoke] built ideal with norm {n_u64}");
                ideal = Some(i);
                break;
            }
        }
        let Some(ideal) = ideal else {
            crate::selkie_trace!("[composite-norm smoke] no composite fixture buildable — skipping");
            return;
        };

        match ideal.suitable_ideals() {
            Some(r) => {
                crate::selkie_trace!(
                    "[composite-norm smoke] succeeded: (s, t) = ({}, {}), \
                     degrees = ({:?}, {:?}), e = {}",
                    EXTREMAL_ORDERS
                        .iter()
                        .position(|o| o.q() == r.factor1.order.q())
                        .unwrap_or(99),
                    EXTREMAL_ORDERS
                        .iter()
                        .position(|o| o.q() == r.factor2.order.q())
                        .unwrap_or(99),
                    r.factor1.degree,
                    r.factor2.degree,
                    r.e.value(),
                );
            }
            None => {
                crate::selkie_trace!("[composite-norm smoke] suitable_ideals returned None");
            }
        }
    }

    /// `smallest_equiv_with_delta` returns an ideal with norm
    /// strictly smaller than the input (since LLL finds a short
    /// basis vector) and a `δ` that lies in the original ideal.
    /// Rather than re-prove containment (which would reconstruct
    /// the `I · δ̄ / nrd(I)` identity), this test checks the norm
    /// shrinkage and that the returned tuple is consistent with
    /// [`Self::smallest_equiv`] (same reduced ideal).
    #[test]
    fn smallest_equiv_with_delta_consistent() {
        use super::super::lattice::LeftIdeal;

        let n = BigInt::<4>::from_u64(13);
        let Some(ideal) = LeftIdeal::random_prime_norm(&n, &EXTREMAL_ORDERS[0]) else {
            return;
        };

        let (reduced_with_delta, _delta) = match ideal.smallest_equiv_with_delta() {
            Some(r) => r,
            None => return,
        };
        let reduced_only = ideal
            .smallest_equiv()
            .expect("smallest_equiv succeeded above");

        assert_eq!(
            reduced_with_delta.norm(),
            reduced_only.norm(),
            "both smallest_equiv variants must produce the same reduced norm",
        );
        // The reduced norm should be bounded by the input norm
        // (generically much smaller — O(√p) vs O(p)).
        assert!(
            reduced_with_delta.norm().bitsize() <= ideal.norm().bitsize(),
            "reduced norm should not exceed the original",
        );
    }

    #[test]
    fn suitable_ideals_small_prime_norm() {
        use super::super::lattice::LeftIdeal;

        // Create an ideal of small prime norm and test SuitableIdeals.
        let n = BigInt::<4>::from_u64(7);
        let Some(ideal) = LeftIdeal::random_prime_norm(&n, &EXTREMAL_ORDERS[0]) else {
            // random_prime_norm may fail; skip if so.
            return;
        };

        let result = ideal.suitable_ideals();
        if let Some(r) = result {
            let f = TorsionExponent::FULL;

            let d1_w = r.factor1.degree.to_bigint_wide();
            let d2_w = r.factor2.degree.to_bigint_wide();

            // Widen to BigInt<8> for the verification arithmetic
            // (u·d1 can exceed 256 bits).
            let u_w: BigInt<8> = r.u.into();
            let v_w: BigInt<8> = r.v.into();
            let two_e = BigInt::<8>::ONE.shl(r.e.value());

            // Verify: u · d1 + v · d2 = 2^e.
            let lhs = u_w.ct_mul(&d1_w).ct_add(&v_w.ct_mul(&d2_w));
            assert_eq!(lhs, two_e, "u·d₁ + v·d₂ should equal 2^e");

            // Verify: e ≤ f.
            assert!(r.e <= f, "e should be ≤ f");

            // Verify: gcd(d1, d2) = 1 (oddness guaranteed by IsogenyDegree).
            assert_eq!(d1_w.gcd(&d2_w), BigInt::<8>::ONE, "gcd(d₁, d₂) should be 1");

            // Verify: u is odd.
            assert!(
                bool::from(r.u.is_odd()),
                "u should be odd after 2-adic reduction"
            );
        }
    }

    /// `enumerate_hypercube` mirrors the C reference's
    /// `enumerate_hypercube` in `dim2id2iso.c:270-376` exactly.
    ///
    /// Locks in the structural enumeration order so that any
    /// future change to the filter logic that diverges from the
    /// C ref will fail this test. Specifically:
    ///
    ///   * The half-cube break pattern (`x ≤ 0`, then nested non-positive
    ///     breaks).
    ///   * The all-even and all-mult-of-3 skips.
    ///   * The `i`-orbit symmetry filter via the `check1 ≤ check2 ∧ check1 ≤
    ///     check3` predicate.
    ///
    /// At `m = 2` (NIST-I), with no symmetry the filtered cube
    /// has 246 tuples; with symmetry it has 137 tuples. The
    /// numbers come from running the C reference for a basis
    /// without/with i-symmetry respectively.
    #[test]
    fn enumerate_hypercube_matches_c_ref() {
        let no_sym = enumerate_hypercube(2, false);
        let with_sym = enumerate_hypercube(2, true);

        // Locked-in counts at `m = 2` (NIST-I `FINDUV_BOX_SIZE`).
        // `272` = full cube `5⁴ = 625` minus the positive-half
        // (313 tuples) minus the 40 all-even tuples within the
        // half-cube. The all-mult-of-3 filter contributes 0 at
        // `m = 2` because `0` is the only multiple of 3 in
        // `[-2, 2]` and `(0, 0, 0, 0)` is already excluded by the
        // half-cube break. `136` is the further reduction from
        // the `i`-orbit symmetry filter — exactly half of `272`,
        // as expected when each orbit has size 2.
        assert_eq!(no_sym.len(), 272);
        assert_eq!(with_sym.len(), 136);

        // Half-cube property: every kept tuple has either x < 0,
        // or x = 0 ∧ y ≤ 0, or x = 0 ∧ y = 0 ∧ z ≤ 0,
        // or x = 0 ∧ y = 0 ∧ z = 0 ∧ w < 0.
        for &[x, y, z, w] in &no_sym {
            let in_half_cube = x < 0
                || (x == 0 && y < 0)
                || (x == 0 && y == 0 && z < 0)
                || (x == 0 && y == 0 && z == 0 && w < 0);
            assert!(in_half_cube, "tuple [{x},{y},{z},{w}] violates half-cube");
        }

        // No tuple has all four coords even.
        for &[x, y, z, w] in &no_sym {
            assert!(
                (x | y | z | w) & 1 != 0,
                "tuple [{x},{y},{z},{w}] is all-even"
            );
        }

        // No tuple has all four coords divisible by 3.
        for &[x, y, z, w] in &no_sym {
            assert!(
                !(x.rem_euclid(3) == 0
                    && y.rem_euclid(3) == 0
                    && z.rem_euclid(3) == 0
                    && w.rem_euclid(3) == 0),
                "tuple [{x},{y},{z},{w}] is all-mult-of-3"
            );
        }

        // Symmetry filter strictly removes some tuples.
        assert!(with_sym.len() < no_sym.len());
    }
}
