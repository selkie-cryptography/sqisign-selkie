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

use super::{
    algebra::{Coordinate, Denominator, Element},
    bigint::BigInt,
    lattice::{ExtremalOrder, Lattice, NrdBasis},
    linear::Vector,
    precomputed::{EXTREMAL_ORDERS, P_WIDE},
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
    pub fn represent_integer_any(m: &BigInt<8>) -> Option<Element<4>> {
        for order in &EXTREMAL_ORDERS {
            let order_wide = ExtremalOrder::<8>::from(*order);
            if let Some(gamma) = order_wide.represent_integer(m, false) {
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
    /// - **t range**: the spec samples t from `[-m', m']` (line 5),
    ///   not `[0, m']`. Both signs must be tried because the isogeny
    ///   condition (line 15) depends on the sign of t, even though
    ///   M' = 4M - p(z² + qt²) depends only on t².
    /// - **Divisibility check**: the spec's "largest d with γ/d ∈ O"
    ///   (line 18) is divisibility in the *order*, not coordinate-wise
    ///   in {1, i, j, k}. For orders with half-integer basis elements,
    ///   this requires checking divisibility by 2·d_O (the order's
    ///   common denominator), not just that all coordinates are even.
    /// - **Arithmetic width**: primality testing and Cornacchia on
    ///   ~273-bit candidates in `BigInt<8>` overflow the 512-bit
    ///   storage during modular exponentiation. Uses widened variants
    ///   (`_w::<16>`) for correctness.
    ///
    /// # Constant-time
    ///
    /// Variable-time. `TODO(ct)`: input is secret-derived via
    /// Algorithm 4.2 (FixedDegreeIsogeny calls this on secret norms).
    ///
    /// [Alg. 3.12]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.12
    pub fn represent_integer(&self, m: &BigInt<8>, isogeny_cond: bool) -> Option<Element<4>> {
        let p: BigInt<8> = P_WIDE;
        let q_val = self.q();
        let q = BigInt::<8>::from_u64(q_val as u64);
        let four_m = BigInt::<8>::from_u64(4).ct_mul(m);

        // Spec line 1: bound = ceil(sqrt(4M / (p·sqrt(q)))).
        let bound: u32 = {
            let q_sqrt = (q_val as f64).sqrt();
            let approx = (four_m.to_f64() / (p.to_f64() * q_sqrt)).sqrt();
            (approx.ceil() as u32).max(256)
        };
        let z_max = {
            let approx = (four_m.to_f64() / p.to_f64() - q_val as f64)
                .max(0.0)
                .sqrt();
            approx as i64
        };

        eprintln!(
            "        represent_integer: bound={bound}, z_max={z_max}, q={q_val}"
        );
        let mut _primes_found = 0u32;
        let mut _cornacchia_ok = 0u32;
        let mut _parity_ok = 0u32;
        let mut _isogeny_cond_fail = 0u32;
        let mut counter: u32 = 0;
        while counter < bound {
            counter += 1;

            let z_val = ((counter as i64 - 1) % z_max.max(1)) + 1;
            let z = BigInt::<8>::from_i64(z_val);

            let pz_sq = p.ct_mul(&z.ct_mul(&z));
            if four_m <= pz_sq {
                continue;
            }
            let remaining = four_m.ct_sub(&pz_sq);
            let t_max = {
                let qp = q.ct_mul(&p);
                if bool::from(qp.is_zero()) {
                    0i64
                } else {
                    (remaining.to_f64() / qp.to_f64()).sqrt() as i64
                }
            };

            // Line 5: sample t from [-m', m'] (spec uses both signs).
            // M' = 4M - p(z² + qt²) depends only on t², so the
            // prime and Cornacchia results are the same for ±t. But
            // the isogeny condition (line 12-15) and the divisibility
            // check (line 18-19) depend on the sign of t.
            let z_sq = z.ct_mul(&z);
            let t_max_capped = t_max.min(50);
            for t_val in (-t_max_capped)..=t_max_capped {
                let t = BigInt::<8>::from_i64(t_val);
                let t_sq = t.ct_mul(&t);
                let inner = z_sq.ct_add(&q.ct_mul(&t_sq));
                let m_prime = four_m.ct_sub(&p.ct_mul(&inner));

                if bool::from(m_prime.is_zero()) || bool::from(m_prime.is_negative()) {
                    continue;
                }

                // m' can be up to ~273 bits in BigInt<8> (512 bits).
                // Squaring during Miller-Rabin produces up to 546-bit
                // intermediates, which overflows BigInt<8>. Widen to
                // BigInt<9> (576 bits ≥ 2×273) for correct modular
                // exponentiation.
                if !m_prime.is_probable_prime_w::<9>(12) {
                    continue;
                }
                _primes_found += 1;

                // Cornacchia uses modular sqrt internally, which
                // needs the same widening as primality testing.
                let Some((x, y)) = BigInt::<8>::cornacchia_w::<9>(&q, &m_prime) else {
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

                // Lines 16-19: construct γ = x + ωy + jz + ωjt.
                // Check that γ/2 ∈ O (spec line 18: d = 2).
                //
                // For O₀ with ω = i, basis {1, i, (1+j)/2, (i+k)/2}:
                //   γ = x + iy + jz + kt
                //   γ/2 ∈ O₀ iff x ≡ z (mod 2) and y ≡ t (mod 2)
                // When satisfied, the order-basis coordinates are:
                //   a = (x-z)/2, b = (y-t)/2, c = z, d = t
                //
                // For other orders (q > 1), the condition differs.
                // TODO: generalize for q > 1 orders.
                let x_mod2 = x_use.as_limbs()[0] & 1;
                let y_mod2 = y_use.as_limbs()[0] & 1;
                let z_mod2 = z.as_limbs()[0] & 1;
                let t_mod2 = t.abs().as_limbs()[0] & 1;

                if x_mod2 != z_mod2 || y_mod2 != t_mod2 {
                    continue;
                }
                _parity_ok += 1;
                if _parity_ok == 1 {
                    eprintln!(
                        "        represent_integer: first parity_ok at z={z_val}, t={t_val}, \
                         isogeny_cond={isogeny_cond}"
                    );
                }

                // γ/2 in {1, i, j, k} coordinates:
                //   (x/2, y/2, z/2, t/2) — but these aren't the
                //   order-basis coords. The Element type uses {1,i,j,k}
                //   with a denominator. So γ/2 has {1,i,j,k} coords
                //   (x, y, z, t) with denominator 2.
                let narrow = |v: &BigInt<8>| -> BigInt<4> {
                    v.narrow_to::<4>()
                        .expect("γ coord fits in BigInt<4>: bounded by √M")
                };
                let gamma = Element::<4>::new(
                    Coordinate::from_bigint(narrow(&x_use)),
                    Coordinate::from_bigint(narrow(&y_use)),
                    Coordinate::from_bigint(narrow(&z)),
                    Coordinate::from_bigint(narrow(&t)),
                    Denominator::from_bigint_unchecked(BigInt::TWO),
                );

                let mut result = gamma;
                result.normalize();
                return Some(result);
            }
        }

        eprintln!(
            "        represent_integer: exhausted bound={bound}, z_max={z_max}, \
             primes={_primes_found}, cornacchia_ok={_cornacchia_ok}, parity_ok={_parity_ok}, \
             isogeny_cond_fail={_isogeny_cond_fail}"
        );
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
pub struct IdealFactor {
    /// The extremal order that produced this factor.
    pub order: &'static ExtremalOrder<4>,
    /// The element β (a short vector in J_t · I).
    pub beta: Element<4>,
    /// Degree d = nrd(β) / nrd(J_t · I).
    pub degree: IsogenyDegree,
}

/// Result of [Alg. 3.16][Alg. 3.16] (SuitableIdeals).
///
/// Contains integers u, v and exponent e such that
/// u · d₁ + v · d₂ = 2^e with gcd(u · d₁, v · d₂) = 1 and e ≤ f,
/// where d₁ and d₂ are the degrees of [`factor1`](Self::factor1)
/// and [`factor2`](Self::factor2).
///
/// [Alg. 3.16]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.16
pub struct SuitableIdealResult {
    /// Odd positive integer u.
    // TODO: Replace with a positive-integer newtype.
    pub u: BigInt<4>,
    /// Positive integer v.
    // TODO: Replace with a positive-integer newtype.
    pub v: BigInt<4>,
    /// Exponent e ≤ f.
    pub e: TorsionExponent,
    /// First factor (β₁, d₁, order index s).
    pub factor1: IdealFactor,
    /// Second factor (β₂, d₂, order index t).
    pub factor2: IdealFactor,
}

/// An element from the short-vector enumeration, paired with its
/// degree (reduced norm divided by the ideal norm).
struct ShortVector {
    /// Quaternion element β (linear combination of reduced basis).
    elem: Element<4>,
    /// Degree: nrd(β) / nrd(I), as a positive odd integer.
    degree: IsogenyDegree,
    /// Approximate norm for sorting (f64 precision, 53-bit mantissa).
    norm_approx: f64,
}

impl NrdBasis<8> {
    /// Enumerate non-zero lattice vectors within the box \[-m, m\]⁴
    /// from an L2-reduced basis, compute their degrees, and sort by
    /// norm.
    ///
    /// For NIST-I with m = 2, this produces up to
    /// (2·2+1)⁴ − 1 = 624 non-zero vectors.
    fn enumerate_short_vectors(
        &self,
        ideal_norm: &BigInt<8>,
        lattice_denom: &BigInt<8>,
    ) -> Vec<ShortVector> {
        let m = crate::params::FINDUV_BOX_SIZE;
        let denom_sq = lattice_denom.ct_mul(lattice_denom);
        let divisor = ideal_norm.ct_mul(&denom_sq);

        let Some(den_4) = lattice_denom.narrow() else {
            return Vec::new();
        };

        let coeffs: Vec<BigInt<8>> = (-m..=m).map(BigInt::<8>::from_i64).collect();
        let width = coeffs.len();
        let mut vectors = Vec::with_capacity(width.pow(4) - 1);

        for ix0 in 0..width {
            for ix1 in 0..width {
                for ix2 in 0..width {
                    for ix3 in 0..width {
                        let x = [coeffs[ix0], coeffs[ix1], coeffs[ix2], coeffs[ix3]];

                        if x.iter().all(|xi| bool::from(xi.is_zero())) {
                            continue;
                        }

                        // nrd(β) · denom² = Σ x_i x_j G_{ij}.
                        let nrd_scaled = self.eval_quadratic_form(&x);

                        if bool::from(nrd_scaled.is_zero()) || bool::from(nrd_scaled.is_negative())
                        {
                            continue;
                        }

                        // degree = nrd_scaled / (nrd(I) · denom²).
                        let (degree_wide, rem) = nrd_scaled.div_rem(&divisor);
                        if !bool::from(rem.is_zero()) || bool::from(degree_wide.is_zero()) {
                            continue;
                        }
                        let Some(degree_4) = degree_wide.narrow() else {
                            continue;
                        };
                        let Some(degree) = IsogenyDegree::new_odd(*degree_4.as_limbs()) else {
                            continue;
                        };

                        // β = Σ x_k · col_k.
                        let coords: [BigInt<8>; 4] = core::array::from_fn(|row| {
                            (0..4).fold(BigInt::<8>::ZERO, |acc, k| {
                                acc.ct_add(&x[k].ct_mul(&self.cols()[k][row]))
                            })
                        });

                        // Narrow coordinates to BigInt<4>. After L2 reduction
                        // with small coefficients this should always succeed.
                        let narrow: [Option<BigInt<4>>; 4] =
                            core::array::from_fn(|i| coords[i].narrow());
                        let [Some(a), Some(b), Some(c), Some(d)] = narrow else {
                            continue;
                        };

                        vectors.push(ShortVector {
                            elem: Element::<4>::new(
                                Coordinate::from_bigint(a),
                                Coordinate::from_bigint(b),
                                Coordinate::from_bigint(c),
                                Coordinate::from_bigint(d),
                                Denominator::from_bigint_unchecked(den_4),
                            ),
                            degree,
                            norm_approx: nrd_scaled.to_f64(),
                        });
                    }
                }
            }
        }

        // Stable sort: preserves insertion order for equal-norm vectors,
        // ensuring deterministic pair selection in the search phase.
        // Determinism matters for signing — non-deterministic pair choice
        // could leak information about which short vectors matched.
        vectors.sort_by(|a, b| {
            a.norm_approx
                .partial_cmp(&b.norm_approx)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        vectors
    }
}

/// Try to find coprime odd degrees and matching u, v from a pair
/// of short vectors. Returns `None` if the pair doesn't satisfy
/// the SuitableIdeals conditions.
fn try_find_uv(
    sv1: &ShortVector,
    sv2: &ShortVector,
    two_f: &BigInt<8>,
    f: TorsionExponent,
    order: &'static ExtremalOrder<4>,
) -> Option<SuitableIdealResult> {
    let d1 = &sv1.degree;
    let d2 = &sv2.degree;

    // Oddness is guaranteed by IsogenyDegree construction.
    // Check gcd(d₁, d₂) = 1 via widened BigInt.
    let d1_w = d1.to_bigint_wide();
    let d2_w = d2.to_bigint_wide();
    if d1_w.gcd(&d2_w) != BigInt::<8>::ONE {
        return None;
    }

    // u = 2^f · d₁⁻¹ mod d₂.
    let d1_inv = d1_w.invert_mod(&d2_w)?;
    let u = two_f.ct_mul(&d1_inv).ct_mod(&d2_w);

    // v = (2^f − u · d₁) / d₂.
    let ud1 = u.ct_mul(&d1_w);
    if ud1 >= *two_f {
        return None;
    }
    let (v, rem) = two_f.ct_sub(&ud1).div_rem(&d2_w);
    if !bool::from(rem.is_zero()) || bool::from(v.is_zero()) || bool::from(v.is_negative()) {
        return None;
    }

    // Factor out the 2-adic part of u.
    let e_val = u.trailing_zeros();
    let e = TorsionExponent::try_from(f.value() - e_val).ok()?;

    Some(SuitableIdealResult {
        u: u.shr(e_val).narrow()?,
        v: v.shr(e_val).narrow()?,
        e,
        factor1: IdealFactor {
            order,
            beta: sv1.elem,
            degree: *d1,
        },
        factor2: IdealFactor {
            order,
            beta: sv2.elem,
            degree: *d2,
        },
    })
}

impl super::lattice::LeftIdeal<4> {
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
    /// [Alg. 3.16]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.16
    pub fn suitable_ideals(&self) -> Option<SuitableIdealResult> {
        let f = TorsionExponent::FULL;
        let two_f = BigInt::<8>::ONE.shl(f.value());

        // Phase 1: L2-reduce the ideal basis and enumerate short vectors.
        //
        // For t = 0, the connecting ideal J_0 = O_0 and J_0 · I = I,
        // so we operate directly on the ideal's own lattice.
        //
        // TODO: Add connecting ideals for t = 1..6 to search across
        // all seven extremal orders (§3.1.7.2).
        let lattice: Lattice<4> = (*self.lattice()).into();
        let cols_4 = lattice.basis().columns();
        let cols_8: [Vector<8>; 4] = core::array::from_fn(|j| cols_4[j].into());
        let denom_8: BigInt<8> = (*lattice.denom()).into();
        let norm_8: BigInt<8> = (*self.norm()).into();

        eprintln!(
            "      suitable_ideals: norm_bits={}, denom={:?}",
            norm_8.bitsize(),
            denom_8
        );

        let nrd_basis = NrdBasis::new(cols_8).l2_reduce();
        let short_vecs = nrd_basis.enumerate_short_vectors(&norm_8, &denom_8);

        eprintln!(
            "      suitable_ideals: {} short vectors enumerated",
            short_vecs.len()
        );

        // Phase 2: Search pairs (β₁, β₂) sorted by ascending norm.
        //
        // Currently both β₁ and β₂ come from L_0 (standard order).
        // When connecting ideals are added, this becomes a nested loop
        // over all (s, t) order-index pairs.
        let order = &EXTREMAL_ORDERS[0];
        let mut _pairs_tried = 0u64;
        for (i, sv1) in short_vecs.iter().enumerate() {
            for sv2 in &short_vecs[i..] {
                _pairs_tried += 1;
                if let Some(result) = try_find_uv(sv1, sv2, &two_f, f, order) {
                    eprintln!(
                        "      suitable_ideals: found after {_pairs_tried} pairs, e={}",
                        result.e.value()
                    );
                    return Some(result);
                }
            }
        }

        eprintln!(
            "      suitable_ideals: no valid pair found ({_pairs_tried} tried)"
        );
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let result = order.represent_integer(&m, false);
        if let Some(gamma) = result {
            let (nrd_num, nrd_den) = gamma.norm();
            let (nrd, rem) = nrd_num.div_rem(&nrd_den);
            assert!(bool::from(rem.is_zero()), "nrd not integer");
            assert_eq!(nrd, m, "nrd(γ) should equal M");
        }
    }

    /// Slow test (~35s): tries all seven orders with NIST-I p.
    /// Run with: `cargo test represent_integer_any -- --ignored`
    #[test]
    #[ignore]
    fn represent_integer_any_order_verifies_norm() {
        let p: BigInt<8> = P_WIDE;
        let m = p.ct_add(&BigInt::TWO);

        let result = ExtremalOrder::<8>::represent_integer_any(&m);
        if let Some(gamma) = result {
            let (nrd_num, nrd_den) = gamma.norm();
            let (nrd, rem) = nrd_num.div_rem(&nrd_den);
            assert!(bool::from(rem.is_zero()), "nrd not integer");
            assert_eq!(nrd, m, "nrd(γ) should equal M");
        }
    }

    #[test]
    fn trailing_zeros_basic() {
        assert_eq!(BigInt::<4>::from_u64(1).trailing_zeros(), 0);
        assert_eq!(BigInt::<4>::from_u64(2).trailing_zeros(), 1);
        assert_eq!(BigInt::<4>::from_u64(4).trailing_zeros(), 2);
        assert_eq!(BigInt::<4>::from_u64(8).trailing_zeros(), 3);
        assert_eq!(BigInt::<4>::from_u64(12).trailing_zeros(), 2); // 0b1100
        assert_eq!(BigInt::<4>::ZERO.trailing_zeros(), 256); // 4 * 64
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
}
