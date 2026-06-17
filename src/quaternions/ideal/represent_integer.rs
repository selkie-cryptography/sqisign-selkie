//! `RepresentInteger` ([Alg. 3.12]): find γ ∈ O with `nrd(γ) = M`.
//!
//! [Alg. 3.12]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.12

use rand_core::RngCore;

use crate::quaternions::{
    algebra::{Coordinate, Denominator, Element},
    bigint::BigInt,
    lattice::{ExtremalOrder, Lattice},
    linear::Matrix,
    precomputed::{EXTREMAL_ORDERS, P_WIDE},
};

/// Finds γ ∈ O with nrd(γ) = M, trying each precomputed extremal order.
///
/// Iterates over the seven precomputed extremal orders, calling
/// [`ExtremalOrder::represent_integer`] on each until one succeeds.
impl ExtremalOrder<8> {
    /// Finds γ ∈ O with nrd(γ) = M, trying all precomputed orders.
    ///
    /// Iterates over [`EXTREMAL_ORDERS`] and calls
    /// [`Self::represent_integer`] on each until one succeeds.
    ///
    /// WARNING: Not constant-time — data-dependent iteration over
    /// orders with early return on first success.
    ///
    /// TODO(ct): Make constant-time before production use. Called on
    /// secret-derived norms during signing (via FixedDegreeIsogeny,
    /// Algorithm 4.2 lines 21–24).
    pub fn represent_integer_any<R: RngCore>(m: &BigInt<8>, rng: &mut R) -> Option<Element<8>> {
        for order in &EXTREMAL_ORDERS {
            let order_wide = ExtremalOrder::<8>::from(*order);
            if let Some(gamma) = order_wide.represent_integer(m, false, rng) {
                return Some(gamma);
            }
        }
        None
    }

    /// Finds γ ∈ O with nrd(γ) = M using this extremal order.
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
    ) -> Option<Element<8>> {
        let p: BigInt<8> = P_WIDE;
        let q_val = self.q();
        let q = BigInt::<8>::from_u64(q_val as u64);
        let four_m = BigInt::<8>::from_u64(4).vt_mul(m);

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
            let (q_quot, _) = four_m.vt_div_rem(&p);
            if q_quot <= q {
                return None;
            }
            q_quot.vt_sub(&q).sqrt_floor()?
        };
        if bool::from(z_max_big.is_zero()) {
            return None;
        }
        let counter_big = {
            let qp2 = q.vt_mul(&p).vt_mul(&p);
            let qp2_sqrt = qp2.sqrt_floor()?;
            if bool::from(qp2_sqrt.is_zero()) {
                return None;
            }
            let (cnt, _) = four_m.vt_div_rem(&qp2_sqrt);
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
        let one_big = BigInt::<8>::ONE;
        let mut iter: u64 = 0;
        while iter < bound {
            iter += 1;

            let z = BigInt::<8>::rand_interval(rng, &one_big, &z_max_big);

            let pz_sq = p.vt_mul(&z.vt_mul(&z));
            if four_m <= pz_sq {
                continue;
            }
            let remaining = four_m.vt_sub(&pz_sq);
            // `t_max = floor(sqrt((4M − p·z²) / (q·p)))`, exact
            // integer (mirrors C ref `normeq.c:151-155`).
            let qp = q.vt_mul(&p);
            if bool::from(qp.is_zero()) {
                continue;
            }
            let (rem_div_qp, _) = remaining.vt_div_rem(&qp);
            let t_max_big = rem_div_qp.sqrt_floor()?;
            let z_sq = z.vt_mul(&z);
            if bool::from(t_max_big.is_zero()) {
                continue;
            }
            let t = BigInt::<8>::rand_interval(rng, &one_big, &t_max_big);
            {
                let t_sq = t.vt_mul(&t);
                let inner = z_sq.vt_add(&q.vt_mul(&t_sq));
                let m_prime = four_m.vt_sub(&p.vt_mul(&inner));

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
                if !m_prime.is_probable_prime_auto::<17>(12) {
                    continue;
                }
                _primes_found += 1;

                // Cornacchia shares the Montgomery modular-sqrt machinery,
                // correct at the candidate's own width; narrow from the
                // same WMAX = 17 ceiling as the primality test above.
                let Some((x, y)) = BigInt::<8>::cornacchia_auto::<17>(&q, &m_prime) else {
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
                    let xt_diff = x_use.vt_sub(&t).vt_mod(&four);
                    let yz_diff = y_use.vt_sub(&z).vt_mod(&four);
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
                let omega_j = Element::<4>::J.mul(omega)?;

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
                let common_d = omega_d.vt_mul(&oj_d);

                let scale_omega = oj_d;
                let scale_omega_j = omega_d;

                // γ = x·common_d + ω·y·scale_ω + j·z·common_d + j·ω·t·scale_ωj
                // (matching the C ref's quat_order_elem_create)
                let mut gamma_coords = [BigInt::<8>::ZERO; 4];
                for k in 0..4 {
                    let x_term = if k == 0 {
                        x_use.vt_mul(&common_d)
                    } else {
                        BigInt::ZERO
                    };
                    let y_term = y_use.vt_mul(&scale_omega).vt_mul(&omega_coords[k]);
                    let z_term = if k == 2 {
                        z.vt_mul(&common_d)
                    } else {
                        BigInt::ZERO
                    };
                    // j·ω·t (C ref order: order->t * temp * order->z)
                    let t_term = t.vt_mul(&scale_omega_j).vt_mul(&oj_coords[k]);
                    gamma_coords[k] = x_term.vt_add(&y_term).vt_add(&z_term).vt_add(&t_term);
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
                    let (q, _) = basis_coeffs[k].vt_div_rem(&content);
                    q
                });
                let basis = order_lattice_w.basis();
                let denom_w = *order_lattice_w.denom();
                let mut result_coords_w = [BigInt::<20>::ZERO; 4];
                for j in 0..4 {
                    for k in 0..4 {
                        result_coords_w[j] =
                            result_coords_w[j].vt_add(&final_coeffs[k].vt_mul(&basis[j][k]));
                    }
                }

                // Narrow to `Element<8>` for the return. A q ≥ 5 order
                // with `nrd(γ) = M ≈ 2^271` (commitment-path FDI on a
                // cross-order pick) produces γ whose `{1, i}` coordinates
                // reach ~2^259, past `BigInt<4>`'s 256-bit budget.
                // Narrowing to width 4 here returned `None` on a valid γ,
                // aborting the whole search and forcing a spurious sign
                // retry that diverged the signature bytes from the C
                // reference. Width 8 (512 bits) holds every NIST-I γ with
                // margin; the downstream `EndomorphismAction::apply` is
                // generic over the storage width and widens internally, so
                // a wider return is free.
                let r0 = result_coords_w[0].narrow_to::<8>()?;
                let r1 = result_coords_w[1].narrow_to::<8>()?;
                let r2 = result_coords_w[2].narrow_to::<8>()?;
                let r3 = result_coords_w[3].narrow_to::<8>()?;
                let denom8 = denom_w.narrow_to::<8>()?;
                let result = Element::<8>::new(
                    Coordinate::from_bigint(r0),
                    Coordinate::from_bigint(r1),
                    Coordinate::from_bigint(r2),
                    Coordinate::from_bigint(r3),
                    Denominator::from_bigint_unchecked(denom8),
                );

                return Some(result);
            }
        }

        None
    }
}
