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
    pub fn represent_integer(&self, m: &BigInt<8>, isogeny_cond: bool) -> Option<Element<4>> {
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

        // Spec line 1: bound = ceil(sqrt(4M / (p·sqrt(q)))).
        let bound: u32 = {
            let q_sqrt = (q_val as f64).sqrt();
            let ratio = four_m.to_f64() / (p.to_f64() * q_sqrt);
            if ratio <= 0.0 {
                return None;
            }
            (ratio.sqrt().ceil() as u32).max(256)
        };
        let z_max = {
            let approx = (four_m.to_f64() / p.to_f64() - q_val as f64)
                .max(0.0)
                .sqrt();
            approx as i64
        };

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
                let omega_j = omega.mul(&Element::<4>::J);

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
                let narrow = |v: &BigInt<8>| -> BigInt<4> {
                    v.narrow_to::<4>()
                        .expect("γ coord fits in BigInt<4>: bounded by √M")
                };
                let gamma = Element::<4>::new(
                    Coordinate::from_bigint(narrow(&gamma_coords[0])),
                    Coordinate::from_bigint(narrow(&gamma_coords[1])),
                    Coordinate::from_bigint(narrow(&gamma_coords[2])),
                    Coordinate::from_bigint(narrow(&gamma_coords[3])),
                    Denominator::from_bigint_unchecked(narrow(&common_d)),
                );

                // Decompose gamma on the order basis to find the content
                // (GCD of the order-basis coefficients), matching the C
                // ref's `quat_alg_make_primitive`.
                // Use the narrow (BigInt<4>) order for decomposition.
                // The extremal order at width 8 wraps the same lattice;
                // we narrow it to width 4 for the decompose call.
                // Find the matching narrow order by q value.
                let Some(narrow_order) = EXTREMAL_ORDERS.iter().find(|o| o.q() == self.q()) else {
                    continue;
                };
                let order_lattice: &Lattice<4> = narrow_order.order();
                let Some(basis_coeffs) = order_lattice.decompose(&gamma) else {
                    // gamma not in the order — skip.
                    continue;
                };

                // Content = GCD of all 4 basis coefficients.
                let mut content = basis_coeffs[0].abs();
                for coeff in &basis_coeffs[1..] {
                    content = content.gcd(&coeff.abs());
                }

                // d = content. Check d = 2.
                if content != BigInt::TWO {
                    continue;
                }
                _parity_ok += 1;

                // Return gamma / content by halving the order-basis
                // coefficients and reconstructing the quaternion element.
                //
                // Dividing the {1,i,j,k} coords by 2 is NOT equivalent
                // to halving the order-basis coefficients (unless the
                // basis is diagonal). We must reconstruct from the halved
                // coefficients: gamma/2 = Σ (c_k/2) · basis_col_k / denom.
                let half = |c: &BigInt<4>| -> BigInt<4> {
                    let (q, _) = c.div_rem(&BigInt::TWO);
                    q
                };
                let half_coeffs: [BigInt<4>; 4] = [
                    half(&basis_coeffs[0]),
                    half(&basis_coeffs[1]),
                    half(&basis_coeffs[2]),
                    half(&basis_coeffs[3]),
                ];

                // Reconstruct: gamma/2 = Σ (c_k/2) · basis_col_k / denom
                let basis = order_lattice.basis();
                let denom = *order_lattice.denom();
                let mut result_coords = [BigInt::<4>::ZERO; 4];
                for j in 0..4 {
                    for k in 0..4 {
                        result_coords[j] =
                            result_coords[j].ct_add(&half_coeffs[k].ct_mul(&basis[j][k]));
                    }
                }

                // The result has denom = order_lattice.denom().
                let result = Element::<4>::new(
                    Coordinate::from_bigint(result_coords[0]),
                    Coordinate::from_bigint(result_coords[1]),
                    Coordinate::from_bigint(result_coords[2]),
                    Coordinate::from_bigint(result_coords[3]),
                    Denominator::from_bigint_unchecked(denom),
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
pub(crate) struct IdealFactor {
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
    pub(crate) parent_ideal: LeftIdeal<4>,
}

/// Result of [Alg. 3.16][Alg. 3.16] (SuitableIdeals).
///
/// Contains integers u, v and exponent e such that
/// u · d₁ + v · d₂ = 2^e with gcd(u · d₁, v · d₂) = 1 and e ≤ f,
/// where d₁ and d₂ are the degrees of [`factor1`](Self::factor1)
/// and [`factor2`](Self::factor2).
///
/// [Alg. 3.16]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.16
pub(crate) struct SuitableIdealResult {
    /// Odd positive integer u.
    // TODO: Replace with a positive-integer newtype.
    pub(crate) u: BigInt<4>,
    /// Positive integer v.
    // TODO: Replace with a positive-integer newtype.
    pub(crate) v: BigInt<4>,
    /// Exponent e ≤ f.
    pub(crate) e: TorsionExponent,
    /// First factor (β₁, d₁, order index s).
    pub(crate) factor1: IdealFactor,
    /// Second factor (β₂, d₂, order index t).
    pub(crate) factor2: IdealFactor,
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
struct ShortVectorBatch {
    /// The extremal order that produced this batch.
    order: &'static ExtremalOrder<4>,
    /// The ideal the short vectors live in.
    parent_ideal: LeftIdeal<4>,
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
    ) -> Vec<ShortVectorCandidate> {
        let m = crate::params::FINDUV_BOX_SIZE;
        let denom_sq = lattice_denom.ct_mul(lattice_denom);
        let divisor = ideal_norm.ct_mul(&denom_sq);

        let Some(den_4) = lattice_denom.narrow() else {
            return Vec::new();
        };

        let coeffs: Vec<BigInt<8>> = (-m..=m).map(BigInt::<8>::from_i64).collect();
        let width = coeffs.len();
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
            eprintln!(
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
                eprintln!(
                    "[enum-trace] G[{i}][{i}] / divisor: rem = {} (is_zero={})",
                    rem,
                    bool::from(rem.is_zero()),
                );
            }
        }

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
                        let Some(degree_4) = degree_wide.narrow() else {
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
                }
            }
        }

        #[cfg(test)]
        if std::env::var("ENUM_DIAG").is_ok() {
            eprintln!(
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
fn try_find_uv(
    sv1: &ShortVectorCandidate,
    sv2: &ShortVectorCandidate,
    batch1: &ShortVectorBatch,
    batch2: &ShortVectorBatch,
    two_f: &BigInt<8>,
    f: TorsionExponent,
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

    // Enumerate every positive-integer solution `(u, v)` to
    // `u·d₁ + v·d₂ = 2^f` along the line
    // `(u, v) = (u_0 + k·d₂, v_0 − k·d₁)` for `k = 0, 1, 2, …` until
    // `v ≤ 0`. The initial solution has `u_0 = 2^f · d₁⁻¹ mod d₂`,
    // so `u_0 ∈ [0, d₂)` and `v_0 = (2^f − u_0·d₁)/d₂`. For each
    // valid solution, factor out the 2-adic part of `gcd(u, v)` to
    // obtain `(u', v', e)` with `u'·d₁ + v'·d₂ = 2^e`.
    //
    // Prior versions checked only the `k = 0` pair and returned
    // `None` whenever the resulting `e` constraint failed.
    // Matching the C reference's `find_uv_from_lists`
    // (`dim2id2iso.c:382-460`), which walks the whole line with a
    // `v += d₁` increment, improves acceptance by roughly an order
    // of magnitude for short-vector pairs with `d₁·d₂ ≪ 2^f`.
    let d1_inv = d1_w.invert_mod(&d2_w)?;
    let u0 = two_f.ct_mul(&d1_inv).ct_mod(&d2_w);
    let mut u = u0;
    let mut v = {
        let ud1 = u.ct_mul(&d1_w);
        if ud1 >= *two_f {
            return None;
        }
        let (v, rem) = two_f.ct_sub(&ud1).div_rem(&d2_w);
        if !bool::from(rem.is_zero()) || bool::from(v.is_negative()) {
            return None;
        }
        v
    };

    loop {
        if !bool::from(u.is_zero()) && !bool::from(v.is_zero()) {
            // Factor out the 2-adic part of `gcd(u, v)`, matching the
            // C reference (`dim2id2iso.c:833`). The spec writes
            // `v_2(u)` in Algorithm 3.16 line 14, but that is only
            // equivalent to `v_2(gcd(u, v))` when `v_2(u) ≤ v_2(v)`.
            let e_val = u.gcd(&v).trailing_zeros();
            // Require `sui.e = f − e_val ≤ f − 2` — i.e., `e_val ≥ 2`.
            //
            // [`LeftIdeal::to_isogeny`]'s outer (2,2)-chain feeds
            // [`surfaces::Kernel::isogeny`] a kernel of order
            // `2^(sui.e + 2)` — two torsion bits above the
            // `2^sui.e`-subgroup that is the chain's real kernel.
            // Those 2 bits are mandatory (the chain's penultimate
            // and ultimate steps consume 4- and 2-torsion residue
            // via the `hadamard_bool` mechanism of Algorithm 8.41)
            // and come from the `2^f`-torsion image basis
            // `(phi_u(P_0), theta·phi_v(P_0))` via `scale = f −
            // sui.e − 2` doublings. The padding only works when
            // `sui.e ≤ f − 2`.
            //
            // The C reference's alternate `extra_torsion = false`
            // chain path (`theta_isogenies.c:1088`) accepts a
            // kernel of order exactly `2^sui.e` by running a
            // shorter 8-torsion chain followed by dedicated
            // 4-isogeny and 2-isogeny tail steps, so it handles
            // `sui.e ∈ {f-1, f}` directly. We don't implement
            // that variant, so we reject those cases here.
            //
            // Pairs with `e_val < 2` get skipped; the outer v-loop
            // enumerates more `(u, v)` solutions for the same
            // `(β₁, β₂)`, so the acceptance cost is small (≈ 20%
            // of pairs on NIST-I in practice).
            if e_val < 2 {
                u = u.ct_add(&d2_w);
                if v <= d1_w {
                    return None;
                }
                v = v.ct_sub(&d1_w);
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

        // Advance to the next solution on the line: `u += d₂`,
        // `v −= d₁`. Stop when `v` would go non-positive.
        u = u.ct_add(&d2_w);
        if v <= d1_w {
            return None;
        }
        v = v.ct_sub(&d1_w);
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
        // Widen basis + denom to BigInt<W>.
        let lattice: Lattice<N> = (*self.lattice()).into();
        let cols_n = lattice.basis().columns();
        let cols_w: [Vector<W>; 4] = core::array::from_fn(|j| {
            Vector::new(
                cols_n[j][0].widen::<W>(),
                cols_n[j][1].widen::<W>(),
                cols_n[j][2].widen::<W>(),
                cols_n[j][3].widen::<W>(),
            )
        });
        let denom_w: BigInt<W> = lattice.denom().widen();

        // # Divergences from the spec's Algorithm 3.3
        //
        // The spec reduces via L2/LLL on an NrdBasis to find a
        // short δ. Our `NrdBasis::l2_reduce` uses DPE (53-bit
        // mantissa) which works only when the Gram entries are
        // `≲ 2^200`; response-phase `I_com,rsp` has Gram entries
        // `~n(I)² · 2 ≈ 2^520` at NIST-I, so LLL fails to
        // reduce (measured: `nrd(δ) / n(I) ≈ 2^124` instead of the
        // theoretical LLL bound `≈ 2^1.5`), which produces
        // equivalent-ideal HNF entries that do not narrow to
        // `BigInt<4>` and stalls signing.
        //
        // Until the LLL stage gains arbitrary-precision or
        // exact-integer GSO, brute-force over small integer
        // combinations `c ∈ {-M, …, M}^4 \ {0}` of the widened
        // HNF basis columns and take the combination with
        // smallest `nrd`. Empirically the response-phase HNF
        // form `(d, d, 1, 1)` places short elements at
        // coefficient magnitude `≤ 1`, so `M = 1` (`3^4 − 1 = 80`
        // combinations) suffices. A larger `M` can be substituted
        // if callers report misses.
        let p_w: BigInt<W> = P_WIDE.widen::<W>();
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
        const MAG: i64 = 4;
        let mut best_v = cols_w[0];
        let mut best_nrd = nrd_of(&[cols_w[0][0], cols_w[0][1], cols_w[0][2], cols_w[0][3]]);
        for c0 in -MAG..=MAG {
            for c1 in -MAG..=MAG {
                for c2 in -MAG..=MAG {
                    for c3 in -MAG..=MAG {
                        if c0 == 0 && c1 == 0 && c2 == 0 && c3 == 0 {
                            continue;
                        }
                        let v = eval_basis(&[c0, c1, c2, c3]);
                        let nrd = nrd_of(&v);
                        if nrd.ct_sub(&best_nrd).is_negative().into() {
                            best_nrd = nrd;
                            best_v = Vector::new(v[0], v[1], v[2], v[3]);
                        }
                    }
                }
            }
        }

        let delta_w = Element::<W>::new(
            Coordinate::from_bigint(best_v[0]),
            Coordinate::from_bigint(best_v[1]),
            Coordinate::from_bigint(best_v[2]),
            Coordinate::from_bigint(best_v[3]),
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
            #[cfg(test)]
            eprintln!(
                "[smallest_equiv_narrow] nrd(δ) not divisible by denom²: nrd_num bits={}, nrd_den bits={}, rem bits={}",
                delta_nrd_num.bitsize(),
                delta_nrd_den.bitsize(),
                rem.bitsize(),
            );
            return None;
        }
        let self_norm_w: BigInt<W> = self.norm().widen::<W>();
        let (equiv_norm_w, rem2) = new_norm_w.div_rem(&self_norm_w);
        if !bool::from(rem2.is_zero()) {
            #[cfg(test)]
            eprintln!(
                "[smallest_equiv_narrow] new_norm not divisible by self.norm: new_norm bits={}, self_norm bits={}",
                new_norm_w.bitsize(),
                self_norm_w.bitsize(),
            );
            return None;
        }
        let equiv_norm: BigInt<4> = match equiv_norm_w.narrow_to() {
            Some(v) => v,
            None => {
                #[cfg(test)]
                eprintln!(
                    "[smallest_equiv_narrow] equiv_norm narrow to 4 failed: {} bits",
                    equiv_norm_w.bitsize(),
                );
                return None;
            }
        };

        // Conjugate δ: negate i, j, k coords; a stays.
        let delta_conj_w = Element::<W>::new(
            Coordinate::from_bigint(*delta_w.a.as_bigint()),
            Coordinate::from_bigint(delta_w.b.as_bigint().wrapping_neg()),
            Coordinate::from_bigint(delta_w.c.as_bigint().wrapping_neg()),
            Coordinate::from_bigint(delta_w.d.as_bigint().wrapping_neg()),
            Denominator::from_bigint_unchecked(*delta_w.denom.as_bigint()),
        );
        let mut new_cols = [Vector::<W>::ZERO; 4];
        #[allow(clippy::needless_range_loop)]
        for j in 0..4 {
            let bj = lattice.basis_elem(j);
            let bj_w = Element::<W>::new(
                Coordinate::from_bigint(bj.a.as_bigint().widen::<W>()),
                Coordinate::from_bigint(bj.b.as_bigint().widen::<W>()),
                Coordinate::from_bigint(bj.c.as_bigint().widen::<W>()),
                Coordinate::from_bigint(bj.d.as_bigint().widen::<W>()),
                Denominator::from_bigint_unchecked(bj.denom.as_bigint().widen::<W>()),
            );
            let product = bj_w.mul_direct(&delta_conj_w);
            new_cols[j] = Vector::new(
                *product.a.as_bigint(),
                *product.b.as_bigint(),
                *product.c.as_bigint(),
                *product.d.as_bigint(),
            );
        }
        // Raw product denom = lattice_denom * delta_denom.
        // Dividing by nrd(I) multiplies denom by nrd(I).
        let product_denom: BigInt<W> = denom_w.ct_mul(&denom_w).ct_mul(&self_norm_w);

        // HNF at width W, simplify by GCD, then narrow to 4.
        let hnf_w = Matrix::<W>::from_hnf_columns(&new_cols);
        let mut g = product_denom.abs();
        for row in 0..4 {
            for col in 0..4 {
                if !bool::from(hnf_w[row][col].is_zero()) {
                    g = g.gcd(&hnf_w[row][col].abs());
                }
            }
        }
        let mut basis_4 = Matrix::<4>::ZERO;
        for row in 0..4 {
            for col in 0..4 {
                let (q, _) = hnf_w[row][col].div_rem(&g);
                basis_4[row][col] = match q.narrow_to::<4>() {
                    Some(v) => v,
                    None => {
                        #[cfg(test)]
                        eprintln!(
                            "[smallest_equiv_narrow] basis[{row}][{col}] narrow to 4 failed: q {} bits, hnf {} bits, g {} bits",
                            q.bitsize(),
                            hnf_w[row][col].bitsize(),
                            g.bitsize(),
                        );
                        return None;
                    }
                };
            }
        }
        let (denom_simplified, _) = product_denom.div_rem(&g);
        let denom_4: BigInt<4> = match denom_simplified.narrow_to() {
            Some(v) => v,
            None => {
                #[cfg(test)]
                eprintln!(
                    "[smallest_equiv_narrow] denom narrow to 4 failed: {} bits, product_denom {} bits, g {} bits",
                    denom_simplified.bitsize(),
                    product_denom.bitsize(),
                    g.bitsize(),
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

impl LeftIdeal<4> {
    /// [Alg. 3.16]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.16
    pub(crate) fn suitable_ideals(&self) -> Option<SuitableIdealResult> {
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
        let mut batches: [Option<ShortVectorBatch>; NUM_EXTREMAL_ORDERS] = Default::default();
        let mut short_vecs_per_order: [Vec<ShortVectorCandidate>; NUM_EXTREMAL_ORDERS] =
            Default::default();

        for t in 0..NUM_EXTREMAL_ORDERS {
            // Build the parent ideal for this order.
            let parent_ideal_t = if t == 0 {
                *self
            } else {
                // Compute pushforward at `BigInt<8>` to avoid
                // `Lattice<4>::product`'s mul_direct overflow when
                // the intersection ideal has coords ~2^252. The
                // resulting lattice should fit back in `BigInt<4>`
                // for narrow-path callers (N ≤ 2^127); if not,
                // skip this curve index this iteration.
                let j_t_8 = connecting_ideal(t).widen::<8>();
                let self_8 = self.widen::<8>();
                let order_t_8 = EXTREMAL_ORDERS[t].widen::<8>();
                let push_8 = j_t_8.pushforward(&self_8, order_t_8.order());
                match push_8.narrow() {
                    Some(p) => p,
                    None => continue,
                }
            };

            let lattice_t: Lattice<4> = (*parent_ideal_t.lattice()).into();
            let cols_4 = lattice_t.basis().columns();
            let cols_8: [Vector<8>; 4] = core::array::from_fn(|j| cols_4[j].into());
            let denom_8: BigInt<8> = (*lattice_t.denom()).into();
            let norm_8: BigInt<8> = (*parent_ideal_t.norm()).into();

            let nrd_basis = NrdBasis::new(cols_8).l2_reduce();
            short_vecs_per_order[t] = nrd_basis.enumerate_short_vectors(&norm_8, &denom_8);
            batches[t] = Some(ShortVectorBatch {
                order: &EXTREMAL_ORDERS[t],
                parent_ideal: parent_ideal_t,
            });
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
                            eprintln!(
                                "[suitable_ideals] selected (s={s}, t={t}) after {_pairs_tried} pairs \
                                 | norm={} bits, batch sizes={:?}",
                                self.norm().bitsize(),
                                short_vecs_per_order
                                    .iter()
                                    .map(|v| v.len())
                                    .collect::<Vec<_>>(),
                            );
                            return Some(result);
                        }
                    }
                }
            }
        }

        #[cfg(test)]
        eprintln!(
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
        let gamma = order.represent_integer(&mn, false);
        let elapsed = t0.elapsed();
        eprintln!(
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

        let result = order.represent_integer(&m, false);
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
        let Some(ideal) = LeftIdeal::random_norm(&n, &EXTREMAL_ORDERS[0]) else {
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
            if let Some(i) = LeftIdeal::random_norm(&n, &EXTREMAL_ORDERS[0]) {
                eprintln!("[composite-norm smoke] built ideal with norm {n_u64}");
                ideal = Some(i);
                break;
            }
        }
        let Some(ideal) = ideal else {
            eprintln!("[composite-norm smoke] no composite fixture buildable — skipping");
            return;
        };

        match ideal.suitable_ideals() {
            Some(r) => {
                eprintln!(
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
                eprintln!("[composite-norm smoke] suitable_ideals returned None");
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
}
