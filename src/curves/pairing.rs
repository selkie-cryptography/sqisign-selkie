//! Tate pairing computation via cubical arithmetic.
//!
//! Computes the reduced Tate pairing t_{2^e}(P, Q) for points on a
//! Montgomery curve E : y² = x³ + Ax² + x over F_{p²}.
//!
//! Uses the cubical arithmetic approach from [§8.3.2] of the spec,
//! which works entirely with x-only projective coordinates. The
//! cubical ladder (Algorithm 8.15) replaces Miller's loop, and the
//! final exponentiation uses the Frobenius endomorphism.
//!
//! [§8.3.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.3.2

use core::ops::{Div, Mul};

use subtle::{Choice, ConditionallySelectable};

use crate::{
    curves::{TorsionBasis, TorsionExponent, montgomery::ProjectiveXOnlyPoint, scalar::Scalar},
    fields::{fp::Fp, fp2::Fp2},
    quaternions::bigint::BigInt,
};

// ---------------------------------------------------------------------------
// Reduced Tate pairing value
// ---------------------------------------------------------------------------

/// An element of μ_{2^e}, the group of 2^e-th roots of unity in F_{p²}*.
///
/// Produced by the reduced Tate pairing ([`tate_pairing`]) and
/// consumed by [`NormalizedDlog`](RootOfUnity::dlog) to solve
/// discrete logarithms in 2-power order subgroups.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RootOfUnity(Fp2);

impl RootOfUnity {
    /// The identity element (1 ∈ μ_{2^e}).
    pub const ONE: Self = Self(Fp2::ONE);

    /// The underlying F_{p²} element.
    pub fn as_fp2(&self) -> &Fp2 {
        &self.0
    }

    /// Compute ζ^{2^n} by repeated squaring.
    #[must_use]
    pub fn square_n(&self, n: u32) -> Self {
        let mut result = self.0;
        for _ in 0..n {
            result = result.square();
        }
        Self(result)
    }

    /// Compute ζ^k for a scalar k.
    #[must_use]
    pub fn pow(&self, k: u32) -> Self {
        if k == 0 {
            return Self::ONE;
        }
        let mut result = Fp2::ONE;
        let mut base = self.0;
        let mut exp = k;
        while exp > 0 {
            if exp & 1 == 1 {
                result = &result * &base;
            }
            base = base.square();
            exp >>= 1;
        }
        Self(result)
    }

    /// Compute ζ^k for a [`Scalar`]-sized exponent.
    ///
    /// Iterates over the scalar's little-endian byte representation
    /// and performs a standard square-and-multiply loop.
    /// Variable-time in `k` — only use on public values such as the
    /// codomain-disambiguation exponent `d₁·u²` in
    /// `IdealToIsogeny`, which is derived from the public
    /// decomposition `(u, d₁)` of the caller-supplied ideal.
    #[must_use]
    pub fn pow_scalar(&self, k: &Scalar) -> Self {
        let bytes = k.to_le_bytes();
        let mut result = Fp2::ONE;
        let mut base = self.0;
        for byte in bytes.iter() {
            let mut b = *byte;
            for _ in 0..8 {
                if b & 1 == 1 {
                    result = &result * &base;
                }
                base = base.square();
                b >>= 1;
            }
        }
        Self(result)
    }

    /// Compute the discrete log k ∈ \[0, 2^e) such that target = self^k.
    ///
    /// Uses the Pohlig-Hellman algorithm for 2-power order groups.
    /// SQIsign only applies this to pairing outputs.
    ///
    /// Implements [NormalizedDlog][Alg. 2.4].
    ///
    /// [Alg. 2.4]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.2.4
    pub fn dlog(&self, target: &Self, e: TorsionExponent) -> Scalar {
        if e.value() == 0 {
            return Scalar::ZERO;
        }
        if e.value() == 1 {
            return if *target == Self::ONE {
                Scalar::ZERO
            } else {
                Scalar::ONE
            };
        }

        // e' = ⌊e/2⌋
        let e_prime = e.halve();

        // ζ'₀ = ζ₀^{2^{e-e'}},  ζ'₁ = ζ₁^{2^{e-e'}}
        let diff = e - e_prime;
        let z0_prime = self.square_n(diff.value());
        let z1_prime = target.square_n(diff.value());

        // k' = NormalizedDlog(ζ'₀, ζ'₁) — low bits
        let k_prime = z0_prime.dlog(&z1_prime, e_prime);

        // ζ''₀ = ζ₀^{2^{e'}},  ζ''₁ = ζ₁ / ζ₀^{k'}
        //
        // `k'` is a `Scalar` and may exceed `u32::MAX`: the recursion
        // bounds `k' < 2^{e'}`, so for e ≥ 64 the low 32 bits are not
        // enough. `pow_scalar` walks all 256 bits of the scalar.
        // (The `M_chl` / `M_sk` matrix entries reach `2^126`, so a
        // truncating `pow(u32)` here corrupts every entry whose dlog
        // exceeds 2^32 — they all collapse to a fixed root of unity
        // and the recovered matrix has the same value in every slot.)
        //
        // TODO(ct): `pow_scalar` is variable-time in `k_prime`. When
        // this dlog is called from `from_bases` with secret-derived
        // bases (M_sk in Algorithm 4.1, M_chl in Algorithm 4.8), the
        // pow leaks bits of the dlog. Replace with a constant-time
        // square-and-multiply that processes a fixed number of bits.
        let z0_double_prime = self.square_n(e_prime.value());
        let z1_double_prime = target / &self.pow_scalar(&k_prime);

        // k'' = NormalizedDlog(ζ''₀, ζ''₁) — high bits
        let k_double_prime = z0_double_prime.dlog(&z1_double_prime, diff);

        // k = k' + 2^{e'} · k''
        let k_prime_big = BigInt::<4>::from(k_prime);
        let k_double_prime_big = BigInt::<4>::from(k_double_prime);
        let k_high = k_double_prime_big.shl(e_prime.value());
        let k = k_prime_big.ct_add(&k_high);
        Scalar::from_limbs(*k.as_limbs())
    }
}

impl<'b> Mul<&'b RootOfUnity> for &RootOfUnity {
    type Output = RootOfUnity;
    fn mul(self, rhs: &'b RootOfUnity) -> RootOfUnity {
        RootOfUnity(&self.0 * &rhs.0)
    }
}

impl<'b> Div<&'b RootOfUnity> for &RootOfUnity {
    type Output = RootOfUnity;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: &'b RootOfUnity) -> RootOfUnity {
        RootOfUnity(&self.0 * &rhs.0.invert())
    }
}

impl Mul for RootOfUnity {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        &self * &rhs
    }
}

impl Div for RootOfUnity {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        &self / &rhs
    }
}

impl ConditionallySelectable for RootOfUnity {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self(Fp2::conditional_select(&a.0, &b.0, choice))
    }
}

// ---------------------------------------------------------------------------
// Cubical point (internal)
// ---------------------------------------------------------------------------

/// A cubical point: projective (X : Z) in F_{p²}.
///
/// Same representation as [`ProjectiveXOnlyPoint`] but used in the
/// cubical arithmetic context where the formulas differ.
#[derive(Copy, Clone, Debug)]
struct CubicalPoint {
    X: Fp2,
    Z: Fp2,
}

impl CubicalPoint {
    /// The point at infinity (1 : 0) in cubical coordinates.
    fn infinity() -> Self {
        Self {
            X: Fp2::ONE,
            Z: Fp2::ZERO,
        }
    }

    /// Construct from an affine x-coordinate: (x : 1).
    fn from_affine(x: Fp2) -> Self {
        Self { X: x, Z: Fp2::ONE }
    }

    /// Cubical doubling ([§8.3.2], Algorithm 8.13).
    fn double(&self, a24: &Fp2) -> Self {
        let sum = &self.X + &self.Z;
        let diff = &self.X - &self.Z;
        let a_sq = sum.square();
        let b_sq = diff.square();
        let c = &a_sq - &b_sq;
        let x2 = &a_sq * &b_sq;
        let z2 = &c * &(&b_sq + &(a24 * &c));
        Self { X: x2, Z: z2 }
    }

    /// Cubical differential addition ([§8.3.2], Algorithm 8.14).
    ///
    /// # Why this *must* divide X by `x(P-Q)`, not multiply Z by `x(P-Q)`
    ///
    /// The spec writes line 7 as `X_2 ← X_2 / x(P-Q)`. A naïve
    /// "optimization" replaces it with `Z_2 ← Z_2 * x(P-Q)` — the
    /// two produce points with the same affine `X/Z` ratio and skip
    /// one `Fp2::invert()` per ladder step. **That optimization is
    /// wrong.** The cubical Tate pairing (Alg 8.18) is not
    /// projectively invariant: downstream `CubicalTranslate`
    /// (Alg 8.16, `X ← X(T)·X(P) − Z(T)·Z(P)`) is bilinear in
    /// `(X, Z)`, not in the ratio `X/Z`. Multiplying Z by a factor
    /// that the spec attaches to X scales the two coordinates
    /// asymmetrically, and the translate then produces a different
    /// cubical point. The error compounds across ladder iterations,
    /// `Translate`, and `Ratio` (Alg 8.17), and surfaces as a
    /// non-bilinear "Tate pairing" — the output is still a 2^e-th
    /// root of unity, but `T([2]P, Q) ≠ T(P, Q)^2`. That alone breaks
    /// every downstream pairing-based check (Weil-pairing codomain
    /// disambiguation in `LeftIdeal::to_isogeny`,
    /// dlog-based change-of-basis recovery, etc).
    fn differential_add(&self, other: &Self, x_diff: &Fp2) -> Self {
        let a = &self.X + &self.Z;
        let b = &self.X - &self.Z;
        let c = &other.X + &other.Z;
        let d = &other.X - &other.Z;
        let x2 = (&a * &d + &b * &c).square();
        let z2 = (&a * &d - &b * &c).square();
        // Spec line 7: X_2 ← X_2 / x(P-Q). See doc comment above for
        // why we cannot move the factor onto Z_2 instead.
        let x2 = &x2 * &x_diff.invert();
        Self { X: x2, Z: z2 }
    }

    /// Cubical translation by a 2-torsion point ([§8.3.2], Algorithm 8.16).
    fn translate(&self, t: &Self) -> Self {
        let x = &(&t.X * &self.X) - &(&t.Z * &self.Z);
        let z = &(&t.Z * &self.X) - &(&t.X * &self.Z);
        let z = if t.Z == Fp2::ZERO { -&z } else { z };
        let x = if t.X == Fp2::ZERO { -&x } else { x };
        Self { X: x, Z: z }
    }

    /// Cubical ratio ([§8.3.2], Algorithm 8.17).
    fn ratio(&self, other: &Self) -> Fp2 {
        if self.X == Fp2::ZERO {
            &other.Z * &self.Z.invert()
        } else {
            &other.X * &self.X.invert()
        }
    }
}

// ---------------------------------------------------------------------------
// Tate pairing (Algorithm 8.18)
// ---------------------------------------------------------------------------

/// Compute the reduced Tate pairing t_{2^e}(P, Q).
///
/// Takes three projective x-only points P, Q, P+Q on the same curve
/// and the torsion exponent e (where 2^e · P = O_E). Internally
/// normalizes to affine x-coordinates for the cubical arithmetic.
///
/// Implements [Tate][Alg. 8.18] from the spec.
///
/// # Panics
///
/// Debug-asserts that all three points are on the same curve.
///
/// [Alg. 8.18]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.18
pub(crate) fn tate_pairing(
    p: &ProjectiveXOnlyPoint,
    q: &ProjectiveXOnlyPoint,
    pq: &ProjectiveXOnlyPoint,
    e: TorsionExponent,
) -> RootOfUnity {
    let e = e.value();
    debug_assert!(p.curve() == q.curve(), "P and Q must be on the same curve");
    debug_assert!(
        q.curve() == pq.curve(),
        "Q and P+Q must be on the same curve"
    );

    let curve = p.curve();
    let a = *curve.coefficient().as_fp2();
    let two = Fp2::from_fp(Fp::from_small(2));
    let four = Fp2::from_fp(Fp::from_small(4));
    let a24 = &(&a + &two) * &four.invert();

    // Normalize to affine x-coordinates for cubical arithmetic.
    let x_p = p.to_affine_x();
    let x_q = q.to_affine_x();
    let x_pq = pq.to_affine_x();

    // Step 1: (nP, nPQ) ← CubicalLadder(E, e-1, (x(P+Q),1), (x(P),1), x(Q))
    let mut np = CubicalPoint::from_affine(*x_p.as_fp2());
    let mut npq = CubicalPoint::from_affine(*x_pq.as_fp2());
    let xq = x_q.as_fp2();
    for _ in 0..(e - 1) {
        npq = npq.differential_add(&np, xq);
        np = np.double(&a24);
    }

    // Step 2: O ← CubicalTranslate(nP, nP)
    let o = np.translate(&np);

    // Step 3: Q' ← CubicalTranslate(nPQ, nP)
    let q_prime = npq.translate(&np);

    // Step 4: λ ← CubicalRatio(Q, Q') / CubicalRatio((1,0), O)
    let q_tilde = CubicalPoint::from_affine(*xq);
    let lambda = &q_tilde.ratio(&q_prime) * &CubicalPoint::infinity().ratio(&o).invert();

    // Step 5: λ^((p²-1)/2^e)
    //
    // Factor: (p²-1)/2^e = (p-1) · (p+1)/2^e.
    //
    // λ^(p-1) = conj(λ)/λ  (Frobenius: x^p = conj(x) in Fp2).
    let lambda_p_minus_1 = &lambda.conjugate() * &lambda.invert();

    // λ^((p+1)/2^e): p+1 = 5·2^248, so (p+1)/2^e = 5·2^(248-e).
    // Compute as λ^5 then square (248-e) times.
    let l2 = lambda_p_minus_1.square();
    let l4 = l2.square();
    let l5 = &l4 * &lambda_p_minus_1;
    let mut result = l5;
    for _ in 0..248u32.saturating_sub(e) {
        result = result.square();
    }

    RootOfUnity(result)
}

impl TorsionBasis {
    /// Five Tate cross-pairings of `self` (full-order canonical) and
    /// `reduced` (order 2^e), batched per the C reference's
    /// `tate_dlog_partial` (`ec/ref/lvlx/biextension.c:621`).
    ///
    /// Returns `[w0, w1, w2_inv, w3, w4_inv]` ∈ μ_{2^e}^5 where:
    /// - `w0`     = `t(P, Q)`           (canonical reference)
    /// - `w1`     = `t(R, P)`
    /// - `w2_inv` = `1 / t(R, Q)`
    /// - `w3`     = `t(S, P)`
    /// - `w4_inv` = `1 / t(S, Q)`
    ///
    /// `(P, Q, P−Q) = self` must be at the curve's full
    /// 2^TORSION_EVEN_POWER torsion; `(R, S, R−S) = reduced` at order
    /// 2^e. The asymmetric ladder runs `TORSION_EVEN_POWER − 1`
    /// doublings on the full-order side and `e − 1` on the reduced
    /// side, so each side reaches its 2-torsion independently. With
    /// both at 2-torsion the cubical translates and monodromy ratios
    /// are well-defined and the post-Frobenius `clear_cofac` +
    /// `2^e_diff` squarings produce primitive 2^e-th roots of unity.
    ///
    /// The X/Z swap on `w2_inv` and `w4_inv` is the C reference's
    /// idiomatic free inversion: the final `(X/Z)^(p−1)` step picks
    /// up the inverse without an explicit Fp2 division.
    ///
    /// Used by [`ChangeOfBasisMatrix::from_bases`] to build
    /// `M_chl` / `M_sk` matrices that survive the dlog-precision
    /// check; the symmetric reduced-input version of `tate_pairing`
    /// produces ord(ζ) = 2^(2e − TORSION_EVEN_POWER) on bases
    /// pre-doubled to order 2^e, which collapses any matrix at
    /// 2e ≤ TORSION_EVEN_POWER.
    ///
    /// # Constant-time
    ///
    /// Variable-time. Used in change-of-basis recovery for
    /// `M_chl` / `M_sk`. `TODO(ct)`: secret-derived in Algorithm 4.8 /
    /// Algorithm 4.1 — leaks matrix entries through cubical-ladder
    /// timing.
    ///
    /// # Errors
    ///
    /// Returns `None` if either basis fails to lift to Jacobian.
    ///
    /// [`ChangeOfBasisMatrix::from_bases`]:
    ///     crate::curves::ChangeOfBasisMatrix::from_bases
    pub fn cross_pairings(
        &self,
        reduced: &TorsionBasis,
        e: TorsionExponent,
    ) -> Option<[RootOfUnity; 5]> {
        let curve = *self.R.curve();
        debug_assert_eq!(
            curve,
            *reduced.R.curve(),
            "self and `reduced` must share a curve"
        );

        // Lift to Jacobian for the four difference-x's needed to seed
        // the cubical accumulators (mirrors `compute_difference_points`
        // at `biextension.c:447`).
        let (p_jac, q_jac) = self.lift(&curve)?;
        let (r_jac, s_jac) = reduced.lift(&curve)?;
        let (_, x_pmr) = p_jac.x_add_sub(&r_jac);
        let (_, x_pms) = p_jac.x_add_sub(&s_jac);
        let (_, x_rmq) = r_jac.x_add_sub(&q_jac);
        let (_, x_smq) = s_jac.x_add_sub(&q_jac);

        // a24 = (A + 2C) / (4C). Affine curve form (C = 1).
        let a = *curve.coefficient().as_fp2();
        let two = Fp2::from_fp(Fp::from_small(2));
        let four = Fp2::from_fp(Fp::from_small(4));
        let a24 = &(&a + &two) * &four.invert();

        // Basis points → (x : 1). Diff points stay in raw projective
        // (X : Z) form: their cubical structure depends on the
        // post-jac representative, not the affine x alone.
        let xp = *self.R.to_affine_x().as_fp2();
        let xq = *self.S.to_affine_x().as_fp2();
        let xpmq = *self.RS.to_affine_x().as_fp2();
        let xr = *reduced.R.to_affine_x().as_fp2();
        let xs = *reduced.S.to_affine_x().as_fp2();

        let mut np = CubicalPoint::from_affine(xp);
        let mut npq = CubicalPoint::from_affine(xpmq);
        let mut nr = CubicalPoint::from_affine(xr);
        let mut ns = CubicalPoint::from_affine(xs);
        let mut pnr = CubicalPoint {
            X: x_pmr.X,
            Z: x_pmr.Z,
        };
        let mut pns = CubicalPoint {
            X: x_pms.X,
            Z: x_pms.Z,
        };
        let mut nrq = CubicalPoint {
            X: x_rmq.X,
            Z: x_rmq.Z,
        };
        let mut nsq = CubicalPoint {
            X: x_smq.X,
            Z: x_smq.Z,
        };

        let e_full = TorsionExponent::FULL.value();
        let e_red = e.value();

        // Loop 1: full-order ladder on (P, Q). Runs `e_full − 1` iters
        // → np at 2-torsion of full order.
        for _ in 0..(e_full - 1) {
            npq = npq.differential_add(&np, &xq);
            np = np.double(&a24);
        }

        // Loop 2: reduced-order accumulators for (R, S). Runs
        // `e_red − 1` iters → nr, ns at their reduced 2-torsion.
        // PnR/PnS/nRQ/nSQ accumulate the cross-pairing structure.
        for _ in 0..(e_red - 1) {
            pnr = pnr.differential_add(&nr, &xp);
            nrq = nrq.differential_add(&nr, &xq);
            nr = nr.double(&a24);

            pns = pns.differential_add(&ns, &xp);
            nsq = nsq.differential_add(&ns, &xq);
            ns = ns.double(&a24);
        }

        // Cubical translates by 2-torsion.
        let npq_t = npq.translate(&np);
        let pnr_t = pnr.translate(&nr);
        let nrq_t = nrq.translate(&nr);
        let pns_t = pns.translate(&ns);
        let nsq_t = nsq.translate(&ns);
        let np_t = np.translate(&np);
        let nr_t = nr.translate(&nr);
        let ns_t = ns.translate(&ns);

        // Five (X, Z) cubical monodromy ratios per `point_ratio`
        // (`biextension.c:95`):
        //   R.x = nQ_t.x · P_arg.x,   R.z = PnQ_t.x
        // `w2_inv` and `w4_inv` swap (X, Z) — that's the C reference's
        // free inversion: (X/Z)^(p−1) → (Z/X)^(p−1) = ((X/Z)^(p−1))^{−1}.
        let raw_xz: [(Fp2, Fp2); 5] = [
            (&np_t.X * &xq, npq_t.X),
            (&nr_t.X * &xp, pnr_t.X),
            (nrq_t.X, &nr_t.X * &xq),
            (&ns_t.X * &xp, pns_t.X),
            (nsq_t.X, &ns_t.X * &xq),
        ];

        // Final exponentiation:
        //   1. (X/Z)^(p−1) via Frobenius (conjugation in Fp2 with p ≡ 3 mod 4, which
        //      holds for p = 5·2^248 − 1).
        //   2. clear_cofac → ^5 (since (p+1)/2^TORSION_EVEN_POWER = 5).
        //   3. 2^e_diff squarings → land in μ_{2^e}.
        let e_diff = e_full - e_red;
        let mut out = [RootOfUnity::ONE; 5];
        for (idx, (w_x, w_z)) in raw_xz.into_iter().enumerate() {
            // (X/Z)^(p−1) = (X^p · Z) / (X · Z^p).
            let x_p = w_x.conjugate();
            let z_p = w_z.conjugate();
            let num = &w_z * &x_p;
            let den = &w_x * &z_p;
            let frac = &num * &den.invert();

            let f2 = frac.square();
            let f4 = f2.square();
            let f5 = &f4 * &frac;

            let mut result = f5;
            for _ in 0..e_diff {
                result = result.square();
            }
            out[idx] = RootOfUnity(result);
        }

        Some(out)
    }
}

/// Compute the Weil pairing e_{2^e}(P, Q).
///
/// Defined as `e(P, Q) = T(P, Q) / T(Q, P)` where `T` is the
/// reduced Tate pairing. Takes the same `(P, Q, P+Q)` triple as
/// [`tate_pairing`].
///
/// Used in [`LeftIdeal::to_isogeny`] to disambiguate the two
/// codomain components of the (2,2)-chain on `E_u × E_v`.
///
/// [`LeftIdeal::to_isogeny`]: crate::quaternions::lattice::LeftIdeal::to_isogeny
pub(crate) fn weil_pairing(
    p: &ProjectiveXOnlyPoint,
    q: &ProjectiveXOnlyPoint,
    pq: &ProjectiveXOnlyPoint,
    e: TorsionExponent,
) -> RootOfUnity {
    let t_pq = tate_pairing(p, q, pq, e);
    let t_qp = tate_pairing(q, p, pq, e);
    let result = t_pq.as_fp2() * &t_qp.as_fp2().invert();
    RootOfUnity(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        curves::{
            BasisHint, ChangeOfBasisMatrix, TorsionBasis,
            montgomery::{Coefficient, Curve},
        },
        deuring::precomputed::torsion_basis::ExtremalCurve,
        fields::fp2::Fp2,
        params,
    };

    /// Build the E₀ torsion basis from params.
    fn e0_basis() -> TorsionBasis {
        let curve = Curve::E0;
        let p = ProjectiveXOnlyPoint::from_affine_x(params::BASIS_E0_P_X, &curve);
        let q = ProjectiveXOnlyPoint::from_affine_x(params::BASIS_E0_Q_X, &curve);
        let pmq = ProjectiveXOnlyPoint::from_affine_x(params::BASIS_E0_PMQ_X, &curve);
        TorsionBasis::from_propagated(p, q, pmq)
    }

    #[test]
    fn tate_pairing_is_root_of_unity() {
        let basis = e0_basis();
        let e = TorsionExponent::FULL; // 248

        let zeta = tate_pairing(&basis.R, &basis.S, &basis.RS, e);

        // ζ ≠ 1 (non-degenerate pairing on a basis).
        assert_ne!(
            zeta,
            RootOfUnity::ONE,
            "pairing of basis should be non-trivial"
        );

        // ζ^{2^e} = 1 (it's a 2^e-th root of unity).
        let should_be_one = zeta.square_n(e.value());
        assert_eq!(should_be_one, RootOfUnity::ONE, "ζ^(2^e) should equal 1");
    }

    /// On the full E_0 basis (order 2^TORSION_EVEN_POWER), ord(ζ) is
    /// the full 2^e exponent — i.e., ζ is primitive. Sister test to
    /// [`tate_pairing_primitive_on_reduced_basis`] documenting the
    /// contract our existing test only loosely checks.
    #[test]
    fn tate_pairing_primitive_on_full_basis() {
        let basis = e0_basis();
        let e = TorsionExponent::FULL;
        let zeta = tate_pairing(&basis.R, &basis.S, &basis.RS, e);

        let ord = (0..=e.value() + 4).find(|&k| zeta.square_n(k) == RootOfUnity::ONE);
        assert_eq!(ord, Some(e.value()), "ord(ζ) must equal 2^{}", e.value());
    }

    /// `TorsionBasis::cross_pairings`'s canonical reference `w[0]`
    /// must be a primitive 2^e-th root of unity. The other four
    /// outputs only have full order if all four basis-coefficient
    /// dlogs are odd, which conflicts with the `(R, S)` basis
    /// constraint `det = ad − bc` being odd — so at least one of
    /// `w[1..4]` will land in a proper subgroup of μ_{2^e}. The
    /// `cross_pairings_dlog_roundtrip` test below verifies the
    /// reduced-precision dlogs still reconstruct the input matrix
    /// correctly.
    #[test]
    fn cross_pairings_canonical_primitive() {
        let pq_full = e0_basis();
        let e_red: u32 = 128;
        let scale = TorsionExponent::FULL.value() - e_red;
        let scale_scalar = Scalar::from_limbs(*BigInt::<4>::ONE.shl(scale).as_limbs());

        let one = Scalar::from_u64(1);
        let two = Scalar::from_u64(2);
        let r_full = pq_full.biscalar_mul(&two, &one, TorsionExponent::FULL);
        let s_full = pq_full.biscalar_mul(&one, &one, TorsionExponent::FULL);
        // R − S = (2P + Q) − (P + Q) = P.
        let reduced = TorsionBasis::from_propagated(
            &scale_scalar * &r_full,
            &scale_scalar * &s_full,
            &scale_scalar * &pq_full.R,
        );

        let e = TorsionExponent::try_from(e_red).expect("128 valid");
        let ws = pq_full
            .cross_pairings(&reduced, e)
            .expect("cross_pairings must succeed on E_0 bases");

        assert_ne!(ws[0], RootOfUnity::ONE, "w[0] must not be trivial");
        let ord = (0..=e.value() + 4).find(|&k| ws[0].square_n(k) == RootOfUnity::ONE);
        assert_eq!(
            ord,
            Some(e.value()),
            "w[0] must be primitive 2^{}",
            e.value()
        );
    }

    /// `cross_pairings` followed by `RootOfUnity::dlog` must reconstruct
    /// the original matrix coefficients of the reduced basis when
    /// expressed in the full-order canonical basis.
    ///
    /// Setup: `R = 3P + 5Q`, `S = 3P + 4Q` at full order, then doubled
    /// `e_diff` times to land at order 2^e. `R − S = Q` falls out by
    /// construction. Expected dlogs (per the C ref's `r1·P + r2·Q = R`
    /// contract): `(r1, r2) = (3, 5)`, `(s1, s2) = (3, 4)`.
    #[test]
    fn cross_pairings_dlog_roundtrip() {
        let pq_full = e0_basis();
        let e_red: u32 = 128;
        let scale = TorsionExponent::FULL.value() - e_red;
        let scale_scalar = Scalar::from_limbs(*BigInt::<4>::ONE.shl(scale).as_limbs());

        let alpha = Scalar::from_u64(3);
        let beta = Scalar::from_u64(5);
        let gamma = Scalar::from_u64(3);
        let delta = Scalar::from_u64(4);
        let r_full = pq_full.biscalar_mul(&alpha, &beta, TorsionExponent::FULL);
        let s_full = pq_full.biscalar_mul(&gamma, &delta, TorsionExponent::FULL);
        // R − S = (3P + 5Q) − (3P + 4Q) = Q.
        let reduced = TorsionBasis::from_propagated(
            &scale_scalar * &r_full,
            &scale_scalar * &s_full,
            &scale_scalar * &pq_full.S,
        );

        let e = TorsionExponent::try_from(e_red).expect("128 valid");
        let ws = pq_full
            .cross_pairings(&reduced, e)
            .expect("cross_pairings must succeed on E_0 bases");

        // Per C ref's `tate_dlog_partial` (biextension.c:720):
        //   r2 = dlog_w0(w[1]),  r1 = dlog_w0(w[2]),
        //   s2 = dlog_w0(w[3]),  s1 = dlog_w0(w[4]).
        let r2 = ws[0].dlog(&ws[1], e);
        let r1 = ws[0].dlog(&ws[2], e);
        let s2 = ws[0].dlog(&ws[3], e);
        let s1 = ws[0].dlog(&ws[4], e);

        assert_eq!(r1, alpha, "r1 must equal α = 3");
        assert_eq!(r2, beta, "r2 must equal β = 5");
        assert_eq!(s1, gamma, "s1 must equal γ = 3");
        assert_eq!(s2, delta, "s2 must equal δ = 4");
    }

    /// Documents a known limitation of [`tate_pairing`]: its symmetric
    /// cubical-ladder formulation produces ord(ζ) =
    /// 2^(2·e − TORSION_EVEN_POWER) when both inputs are pre-reduced
    /// to order 2^e (rather than primitively used at the curve's
    /// 2^TORSION_EVEN_POWER torsion). For e=128, e_full=248 this is
    /// 2^8, which collapses any change-of-basis matrix at e_cob ≤ 124.
    ///
    /// Sign and verify route around this by using
    /// [`TorsionBasis::cross_pairings`] (asymmetric ladder, full-order
    /// canonical × reduced-order target) which produces a primitive
    /// 2^e-th root. See [`from_bases`] for the change-of-basis call
    /// site.
    ///
    /// Fixing the symmetric path would require restructuring the
    /// cubical ladder to track the path-dependent `xq^k` factor that
    /// differential_add accumulates — an open task tracked separately.
    /// This test asserts the *current* (broken) behavior so a
    /// downstream fix to [`tate_pairing`] flags as a regression.
    ///
    /// [`from_bases`]: crate::curves::ChangeOfBasisMatrix::from_bases
    #[test]
    fn tate_pairing_subprimitive_on_reduced_basis() {
        let basis = e0_basis();
        let e_full = TorsionExponent::FULL.value();
        let e_red: u32 = 128;
        let scale = e_full - e_red;
        let scale_scalar = Scalar::from_limbs(*BigInt::<4>::ONE.shl(scale).as_limbs());

        let r = &scale_scalar * &basis.R;
        let s = &scale_scalar * &basis.S;
        let rs = &scale_scalar * &basis.RS;
        let reduced = TorsionBasis::from_propagated(r, s, rs);

        let e = TorsionExponent::try_from(e_red).expect("128 is a valid TorsionExponent");
        let zeta = tate_pairing(&reduced.R, &reduced.S, &reduced.RS, e);
        assert_ne!(zeta, RootOfUnity::ONE);

        let ord = (0..=e.value() + 4).find(|&k| zeta.square_n(k) == RootOfUnity::ONE);
        let expected = 2 * e_red - e_full;
        assert_eq!(
            ord,
            Some(expected),
            "current symmetric tate_pairing on reduced bases gives \
             ord(ζ) = 2^(2·e − e_full) = 2^{expected}; if this assertion \
             fires, the symmetric path was fixed and `cross_pairings` may \
             no longer be necessary"
        );
    }

    #[test]
    fn dlog_round_trip_large() {
        let basis = e0_basis();
        let e = TorsionExponent::FULL;
        let zeta = tate_pairing(&basis.R, &basis.S, &basis.RS, e);

        // dlog with full exponent: ζ^42 should round-trip.
        let zeta42 = zeta.pow(42);
        let k = zeta.dlog(&zeta42, e);
        assert_eq!(k, Scalar::from_u64(42), "dlog(ζ^42) should be 42");
    }

    /// Dlog must round-trip for exponents that exceed `u32::MAX`.
    ///
    /// `M_chl` / `M_sk` entries reach `2^126`, so the dlog must
    /// preserve all 126 bits. An earlier implementation truncated
    /// `k'` (the low half of the dlog recursion) to `u32` before
    /// calling `pow`, which produced a fixed root of unity for any
    /// `k' ≥ 2^32` — every cross-pairing dlog collapsed to the same
    /// value and `from_bases` returned a constant matrix.
    #[test]
    fn dlog_round_trip_above_u32() {
        let basis = e0_basis();
        let e = TorsionExponent::FULL; // 248
        let zeta = tate_pairing(&basis.R, &basis.S, &basis.RS, e);

        // Pick a value with bits set above 2^32 so any `as u32` cast
        // would lose information.
        let k_in = Scalar::from_limbs([0x1234_5678_9ABC_DEF0, 0x55, 0, 0]);
        let zeta_k = zeta.pow_scalar(&k_in);
        let k_out = zeta.dlog(&zeta_k, e);
        assert_eq!(
            k_out, k_in,
            "dlog(ζ^k) must round-trip for k with bits above 2^64"
        );
    }

    /// Verify Tate pairing bilinearity (P+Q convention).
    ///
    /// `T([2]P, Q, [2]P+Q) == T(P, Q, P+Q)^2`. The third argument
    /// is the SUM, computed via `differential_add(P, Q, P-Q)`.
    ///
    /// Currently fails: our cubical Tate implementation does not
    /// satisfy this relation. The output is always a 2^e-th root of
    /// unity (verified by `tate_pairing_is_root_of_unity`), so the
    /// algorithm produces "a pairing-like value" — but it does not
    /// scale linearly under doubling of the first argument. This
    /// blocks Weil-pairing-based codomain disambiguation in
    /// `LeftIdeal::to_isogeny` (which the C reference uses to pick
    /// `codomain.E1` vs `codomain.E2` in `dim2id2iso.c:1148-1178`).
    #[test]
    fn tate_bilinear_in_first_arg_with_sum() {
        let basis = e0_basis();
        let e = TorsionExponent::FULL;

        // Compute P+Q from (P, Q, P-Q): differential_add(P, Q, P-Q) = P + Q.
        let ppq = basis.R.differential_add(&basis.S, &basis.RS);
        let t_pq = tate_pairing(&basis.R, &basis.S, &ppq, e);
        let t_pq_squared = t_pq.square_n(1);

        let p2 = basis.R.double();
        // [2]P + Q via differential_add([2]P, Q, [2]P-Q).
        // [2]P-Q from differential_add(P, P-Q, Q).
        let two_p_minus_q = basis.R.differential_add(&basis.RS, &basis.S);
        let two_p_plus_q = p2.differential_add(&basis.S, &two_p_minus_q);

        let t_2p_q = tate_pairing(&p2, &basis.S, &two_p_plus_q, e);
        assert_eq!(
            t_2p_q, t_pq_squared,
            "Tate bilinearity (P+Q form): T([2]P, Q, [2]P+Q) should equal T(P, Q, P+Q)^2"
        );
    }

    /// Verify Tate pairing bilinearity (P-Q convention).
    ///
    /// `T([2]P, Q, [2]P-Q) == T(P, Q, P-Q)^2`. Same root cause as
    /// [`tate_bilinear_in_first_arg_with_sum`] — fails for both
    /// sum and difference conventions of the third argument.
    #[test]
    fn tate_bilinear_in_first_arg_with_diff() {
        let basis = e0_basis();
        let e = TorsionExponent::FULL;

        let t_pq = tate_pairing(&basis.R, &basis.S, &basis.RS, e);
        let t_pq_squared = t_pq.square_n(1);

        let p2 = basis.R.double();
        // [2]P - Q via differential_add(P, P-Q, Q).
        let two_p_minus_q = basis.R.differential_add(&basis.RS, &basis.S);

        let t_2p_q = tate_pairing(&p2, &basis.S, &two_p_minus_q, e);
        assert_eq!(
            t_2p_q, t_pq_squared,
            "Tate bilinearity (P-Q form): T([2]P, Q, [2]P-Q) should equal T(P, Q, P-Q)^2"
        );
    }

    /// Verify Tate pairing antisymmetry — the property
    /// `from_bases` relies on for the cross-pairing dlog.
    ///
    /// Specifically: `t(P, Q) · t(Q, P) == 1` so that
    /// `ζ_2 = 1/t(target.R, full.R)` correctly recovers
    /// `ζ^{coefficient of P in target.R}`.
    #[test]
    fn tate_antisymmetric() {
        let basis = e0_basis();
        let e = TorsionExponent::FULL;

        // P+Q for the third arg per `tate_pairing`'s convention.
        let ppq = basis.R.differential_add(&basis.S, &basis.RS);

        let t_pq = tate_pairing(&basis.R, &basis.S, &ppq, e);
        let t_qp = tate_pairing(&basis.S, &basis.R, &ppq, e);
        let product = t_pq.as_fp2() * t_qp.as_fp2();
        assert_eq!(
            product,
            Fp2::ONE,
            "Tate antisymmetry: t(P, Q) · t(Q, P) should equal 1"
        );
    }

    /// Verify Weil pairing antisymmetry: `W(P, Q) * W(Q, P) == 1`.
    #[test]
    fn weil_antisymmetric() {
        let basis = e0_basis();
        let e = TorsionExponent::FULL;

        // Use P+Q form for Weil since Tate's bilinearity probably holds there.
        let ppq = basis.R.differential_add(&basis.S, &basis.RS);
        let w_pq = weil_pairing(&basis.R, &basis.S, &ppq, e);
        let w_qp = weil_pairing(&basis.S, &basis.R, &ppq, e);
        let product = w_pq.as_fp2() * w_qp.as_fp2();
        assert_eq!(
            product,
            Fp2::ONE,
            "Weil antisymmetry: W(P, Q) * W(Q, P) should equal 1"
        );
    }

    #[test]
    fn dlog_round_trip() {
        let basis = e0_basis();

        // Use the full-order pairing which is guaranteed primitive.
        let e = TorsionExponent::FULL; // 248
        let zeta = tate_pairing(&basis.R, &basis.S, &basis.RS, e);
        assert_ne!(zeta, RootOfUnity::ONE);

        // Use a small exponent for the dlog test by squaring down.
        // ζ' = ζ^{2^{248-10}} is a primitive 2^10-th root.
        let e_small = TorsionExponent::try_from(10).unwrap();
        let zeta_small = zeta.square_n(248 - 10);
        assert_ne!(zeta_small, RootOfUnity::ONE);
        assert_eq!(zeta_small.square_n(10), RootOfUnity::ONE);

        // ζ'^7 should dlog back to 7.
        let zeta7 = zeta_small.pow(7);
        let k = zeta_small.dlog(&zeta7, e_small);
        assert_eq!(k, Scalar::from_u64(7), "dlog(ζ'^7) should be 7");
    }

    /// `mul(from_bases(A, B), A)` must equal `B` in x-only sense
    /// even when `B` is constructed via a unimodular matrix on `A`.
    ///
    /// This guards against breaking the weaker (downstream-relevant)
    /// invariant when fixing the stronger `from_bases_independent_bases`
    /// invariant. The chain consumer of `M_chl` only needs x-only
    /// equality.
    #[test]
    fn from_bases_x_only_roundtrip_unimodular() {
        let basis_a = e0_basis();
        let e = TorsionExponent::FULL;
        let m_ab = [
            [Scalar::from_u64(2), Scalar::from_u64(1)],
            [Scalar::from_u64(1), Scalar::from_u64(1)],
        ];
        let p_b = basis_a.biscalar_mul(&m_ab[0][0], &m_ab[1][0], e);
        let q_b = basis_a.biscalar_mul(&m_ab[0][1], &m_ab[1][1], e);
        let k = e.value();
        let pmq_a_scalar = m_ab[0][0].sub_mod2k(&m_ab[0][1], k);
        let pmq_c_scalar = m_ab[1][0].sub_mod2k(&m_ab[1][1], k);
        let pmq_b = basis_a.biscalar_mul(&pmq_a_scalar, &pmq_c_scalar, e);
        let basis_b = TorsionBasis::from_propagated(p_b, q_b, pmq_b);

        let recovered =
            ChangeOfBasisMatrix::from_bases(&basis_a, &basis_b, e).expect("dlog should succeed");
        let applied = recovered.mul(&basis_a);
        assert_eq!(
            applied.R, basis_b.R,
            "x-only: applied.R must equal basis_b.R"
        );
        assert_eq!(
            applied.S, basis_b.S,
            "x-only: applied.S must equal basis_b.S"
        );
        assert_eq!(
            applied.RS, basis_b.RS,
            "x-only: applied.RS must equal basis_b.RS"
        );
    }

    /// `from_bases(A, B) · A` must equal `B` even when `A` and `B`
    /// are unrelated bases on the same curve (i.e., `B` was not
    /// constructed by applying a matrix to `A`).
    ///
    /// In the actual signing pipeline, `from_bases` is called with
    /// `det_chl_scaled` (a hint-derived basis) and `transformed`
    /// (a basis built by applying `m1` to a propagated basis) — two
    /// genuinely independent bases on the same curve. The simpler
    /// `from_bases_mul_roundtrip` test exercises only the
    /// "target = M·source" case, which can succeed even when
    /// `from_bases` produces an x-only-equivalent but projectively
    /// different matrix. This stronger test mirrors the signing
    /// usage by deriving `B` from `A` via a unimodular matrix, then
    /// asserts the recovered matrix faithfully reproduces `B`.
    ///
    /// Currently failing: for unimodular target bases the recovered
    /// dlogs come out as `−k mod 2^e` rather than `+k`. The weaker
    /// x-only round-trip via `mul` still works (per
    /// `from_bases_x_only_roundtrip_unimodular`), but the matrix
    /// entries are off by a sign. Tracked as part of task #39.
    #[test]
    #[ignore]
    fn from_bases_independent_bases() {
        let basis_a = e0_basis();
        let e = TorsionExponent::FULL;

        // basis_b = (some transform) · basis_a, but constructed so it
        // looks like an "independent" basis (any two non-degenerate
        // 2^e-torsion bases are related by an invertible matrix).
        // Unimodular matrix (det = 2 - 1 = 1 ⇒ invertible mod 2).
        let m_ab = [
            [Scalar::from_u64(2), Scalar::from_u64(1)],
            [Scalar::from_u64(1), Scalar::from_u64(1)],
        ];
        let p_b = basis_a.biscalar_mul(&m_ab[0][0], &m_ab[1][0], e);
        let q_b = basis_a.biscalar_mul(&m_ab[0][1], &m_ab[1][1], e);
        let k = e.value();
        let pmq_a_scalar = m_ab[0][0].sub_mod2k(&m_ab[0][1], k);
        let pmq_c_scalar = m_ab[1][0].sub_mod2k(&m_ab[1][1], k);
        let pmq_b = basis_a.biscalar_mul(&pmq_a_scalar, &pmq_c_scalar, e);
        let basis_b = TorsionBasis::from_propagated(p_b, q_b, pmq_b);

        // Recover m_ab and check entries.
        let recovered =
            ChangeOfBasisMatrix::from_bases(&basis_a, &basis_b, e).expect("dlog should succeed");

        // First check the weaker x-only round-trip — this is what
        // the chain downstream consumes.
        let applied = recovered.mul(&basis_a);
        assert_eq!(
            applied.R, basis_b.R,
            "x-only round-trip: applied.R must equal basis_b.R"
        );
        assert_eq!(
            applied.S, basis_b.S,
            "x-only round-trip: applied.S must equal basis_b.S"
        );
        assert_eq!(
            applied.RS, basis_b.RS,
            "x-only round-trip: applied.RS must equal basis_b.RS"
        );

        // Stricter check: actual entry values match the constructed
        // matrix. Currently failing — see test docstring.
        assert_eq!(
            recovered.entries[0][0], m_ab[0][0],
            "[0][0]: got {:?}, expected {:?}",
            recovered.entries[0][0], m_ab[0][0]
        );
        assert_eq!(
            recovered.entries[0][1], m_ab[0][1],
            "[0][1]: got {:?}, expected {:?}",
            recovered.entries[0][1], m_ab[0][1]
        );
        assert_eq!(
            recovered.entries[1][0], m_ab[1][0],
            "[1][0]: got {:?}, expected {:?}",
            recovered.entries[1][0], m_ab[1][0]
        );
        assert_eq!(
            recovered.entries[1][1], m_ab[1][1],
            "[1][1]: got {:?}, expected {:?}",
            recovered.entries[1][1], m_ab[1][1]
        );
    }

    /// Scaling by `2^k` via `Scalar * Point` matches scaling via
    /// `k` repeated doublings — in x-only sense.
    ///
    /// Signing-side reduces basis via `Scalar(2^k) * point`. Verify
    /// reduces basis via a doubling loop. Both paths must produce
    /// the same affine x for the post-scaling basis to match across
    /// signing and verify.
    #[test]
    fn scalar_mul_pow2_matches_doubling() {
        let basis = e0_basis();
        let target = basis.R;
        let k_bits = 60u32; // arbitrary, exercise the multi-bit ladder

        let scale_scalar = Scalar::from_limbs(*BigInt::<4>::ONE.shl(k_bits).as_limbs());
        let via_mul = &scale_scalar * &target;

        let mut via_double = target;
        for _ in 0..k_bits {
            via_double = via_double.double();
        }

        assert_eq!(
            via_mul, via_double,
            "[2^k]P via scalar mul must equal [2^k]P via repeated doubling (x-only)"
        );
    }

    /// `biscalar_mul(k, l).x == biscalar_mul(-k, -l).x`.
    ///
    /// Underlies a lot of x-only reasoning: scalars `k` and `-k`
    /// (= `2^e − k`) produce points that differ only in y-sign, so
    /// the affine x must be identical. If this property fails, the
    /// "x-only equivalence" arguments throughout `from_bases` and
    /// the bench/verify path break down.
    #[test]
    fn biscalar_mul_negation_x_only() {
        let basis = e0_basis();
        let e = TorsionExponent::FULL;

        let k = Scalar::from_u64(13);
        let l = Scalar::from_u64(7);
        let neg_k = Scalar::ZERO.sub_mod2k(&k, e.value());
        let neg_l = Scalar::ZERO.sub_mod2k(&l, e.value());

        let p_pos = basis.biscalar_mul(&k, &l, e);
        let p_neg = basis.biscalar_mul(&neg_k, &neg_l, e);

        assert_eq!(
            p_pos, p_neg,
            "biscalar_mul([k, l]) and biscalar_mul([-k, -l]) must give same x-only point"
        );
    }

    /// `from_hint(curve, to_hint(curve).hint)` must reproduce
    /// `to_hint(curve).basis` exactly.
    ///
    /// In the signing pipeline, signing computes
    /// `(basis, hint) = to_hint(e_chl_final)` and embeds `hint` in
    /// the signature. Verify reconstructs the canonical basis via
    /// `from_hint(curve_chl, hint)`. For the matrix `M_chl` that
    /// signing computes (against signing's `basis`) to apply
    /// correctly on verify's reconstructed basis, the two bases must
    /// be identical.
    ///
    /// This test asserts the round-trip on `E_0`.  If it fails,
    /// signing's `det_chl_scaled` and verify's `basis_chl_scaled`
    /// would disagree, and the matrix `M_chl` — even if computed
    /// correctly relative to signing's basis — would map to a
    /// different basis when applied on verify, producing the
    /// observed `transformed.R != post-M_chl.R` mismatch.
    #[test]
    fn to_hint_from_hint_roundtrip_e0() {
        // E_0: A = 0 by NIST-I convention.
        let curve = Curve::from(Coefficient::ZERO);

        let (basis_via_to, hint) = TorsionBasis::to_hint(&curve);
        let basis_via_from = TorsionBasis::from_hint(&curve, BasisHint::from_byte(hint.to_byte()));

        assert_eq!(
            basis_via_to.R, basis_via_from.R,
            "to_hint/from_hint round-trip must match on E_0: R differs"
        );
        assert_eq!(
            basis_via_to.S, basis_via_from.S,
            "to_hint/from_hint round-trip must match on E_0: S differs"
        );
        assert_eq!(
            basis_via_to.RS, basis_via_from.RS,
            "to_hint/from_hint round-trip must match on E_0: RS differs"
        );
    }

    /// `to_hint` / `from_hint` round-trip on a non-`E_0` curve with
    /// non-zero `A`.
    ///
    /// `from_hint` has separate code paths for `A == 0` (use the
    /// precomputed `BASIS_E0_*` constants) and `A != 0` (recover
    /// the basis via the hint's `(h_A, h)` payload). The `E_0` test
    /// only exercises the first path. This test exercises the second
    /// by using one of the alternate extremal curves (which have
    /// non-zero `A` by construction).
    #[test]
    fn to_hint_from_hint_roundtrip_alternate_curve() {
        // E1 is an alternate extremal-order curve with non-zero A.
        let (_, _, _, a) = ExtremalCurve::E1.basis();
        assert_ne!(a, Fp2::ZERO, "alternate curve must have non-zero A");
        let curve = Curve::from(Coefficient::from(a));

        let (basis_via_to, hint) = TorsionBasis::to_hint(&curve);
        let basis_via_from = TorsionBasis::from_hint(&curve, BasisHint::from_byte(hint.to_byte()));

        assert_eq!(
            basis_via_to.R, basis_via_from.R,
            "to_hint/from_hint round-trip on alternate curve: R differs"
        );
        assert_eq!(
            basis_via_to.S, basis_via_from.S,
            "to_hint/from_hint round-trip on alternate curve: S differs"
        );
        assert_eq!(
            basis_via_to.RS, basis_via_from.RS,
            "to_hint/from_hint round-trip on alternate curve: RS differs"
        );
    }

    /// `to_hint` / `from_hint` round-trip on **every** alternate
    /// extremal curve, asserting all three basis points (R, S, RS).
    ///
    /// Regression test for the Bug 3 fix: `to_hint` previously did not
    /// call `curve.normalize()` while `from_hint` did. On an
    /// unnormalized curve, `clear_cofactor`'s doublings produced a
    /// projectively different `(X : Z)` for P and Q in `to_hint` than
    /// `from_hint` recomputed for the same affine x. The downstream
    /// `projective_difference(P, Q)` (which contains a square root) is
    /// sensitive to the projective rep and picks different sqrt
    /// branches, so `S = P − Q` ends up as a *different abstract
    /// point* in `to_hint(c)` vs `from_hint(c, hint)` — same curve,
    /// same hint.
    ///
    /// The pre-existing `to_hint_from_hint_roundtrip_alternate_curve`
    /// test only exercised `ExtremalCurve::E1`, which happened to
    /// land on the same sqrt branch on both sides. Iterating all
    /// non-zero-A alternates raises the chance of hitting a
    /// branch-divergent curve, and post-fix the round-trip must
    /// agree on every component for every curve.
    #[test]
    fn to_hint_from_hint_roundtrip_all_alternate_curves() {
        for ec in ExtremalCurve::ALL.iter().copied() {
            if ec == ExtremalCurve::E0 {
                // E0 (A = 0) is exercised by the dedicated E0 test.
                continue;
            }
            let (_, _, _, a) = ec.basis();
            assert_ne!(a, Fp2::ZERO, "alternate curve {ec:?} must have non-zero A");
            let curve = Curve::from(Coefficient::from(a));

            let (basis_via_to, hint) = TorsionBasis::to_hint(&curve);
            let basis_via_from =
                TorsionBasis::from_hint(&curve, BasisHint::from_byte(hint.to_byte()));

            assert_eq!(
                basis_via_to.R, basis_via_from.R,
                "{ec:?}: to_hint/from_hint round-trip — R differs"
            );
            assert_eq!(
                basis_via_to.S, basis_via_from.S,
                "{ec:?}: to_hint/from_hint round-trip — S (= P−Q) differs"
            );
            assert_eq!(
                basis_via_to.RS, basis_via_from.RS,
                "{ec:?}: to_hint/from_hint round-trip — RS differs"
            );
        }
    }

    /// `from_bases` must recover the *exact* matrix entries used to
    /// build the target — not just an x-only-equivalent.
    ///
    /// `mul(M, source).x == mul(-M, source).x` for x-only points, so a
    /// matrix-level round-trip via `mul` catches transposes but
    /// silently accepts negated entries. Verify-side downstream
    /// consumers (the `(2,2)`-isogeny chain) need the actual scalar
    /// values, not just x-only equivalents — so this test asserts
    /// `recovered.entries == m_known` directly.
    #[test]
    fn from_bases_recovers_known_entries() {
        let source = e0_basis();
        let e = TorsionExponent::FULL;

        let m_known = [
            [Scalar::from_u64(3), Scalar::from_u64(5)],
            [Scalar::from_u64(7), Scalar::from_u64(11)],
        ];
        // Target with matrix M applied via the convention `from_bases`
        // expects (column-major: column j of M = coefficients of
        // target.basis[j] in source).
        let p_target = source.biscalar_mul(&m_known[0][0], &m_known[1][0], e);
        let q_target = source.biscalar_mul(&m_known[0][1], &m_known[1][1], e);
        let k = e.value();
        let pmq_a = m_known[0][0].sub_mod2k(&m_known[0][1], k);
        let pmq_c = m_known[1][0].sub_mod2k(&m_known[1][1], k);
        let pmq_target = source.biscalar_mul(&pmq_a, &pmq_c, e);
        let target = TorsionBasis::from_propagated(p_target, q_target, pmq_target);

        let recovered =
            ChangeOfBasisMatrix::from_bases(&source, &target, e).expect("dlog should succeed");

        assert_eq!(
            recovered.entries[0][0], m_known[0][0],
            "entries[0][0]: got {:?}, expected {:?}",
            recovered.entries[0][0], m_known[0][0]
        );
        assert_eq!(
            recovered.entries[0][1], m_known[0][1],
            "entries[0][1]: got {:?}, expected {:?}",
            recovered.entries[0][1], m_known[0][1]
        );
        assert_eq!(
            recovered.entries[1][0], m_known[1][0],
            "entries[1][0]: got {:?}, expected {:?}",
            recovered.entries[1][0], m_known[1][0]
        );
        assert_eq!(
            recovered.entries[1][1], m_known[1][1],
            "entries[1][1]: got {:?}, expected {:?}",
            recovered.entries[1][1], m_known[1][1]
        );
    }

    /// Round-trip: `from_bases(A, B) · A == B`.
    ///
    /// Catches transposes between the column-major storage produced by
    /// [`ChangeOfBasisMatrix::from_bases`] and the column-applied
    /// semantics of `mul`. Without this, `from_bases` and `mul` can be
    /// internally inconsistent and only fail in the full sign + verify
    /// round-trip — which costs minutes per attempt.
    #[test]
    fn from_bases_mul_roundtrip() {
        let source = e0_basis();
        let e = TorsionExponent::FULL;

        // Build a target = M · source for a known M with small entries
        // (still applied modulo 2^e via biscalar).
        let m_known = [
            [Scalar::from_u64(3), Scalar::from_u64(5)],
            [Scalar::from_u64(7), Scalar::from_u64(11)],
        ];
        let p_target = source.biscalar_mul(&m_known[0][0], &m_known[1][0], e);
        let q_target = source.biscalar_mul(&m_known[0][1], &m_known[1][1], e);
        // Difference via biscalar to avoid a sqrt branch flip.
        let k = e.value();
        let pmq_a = m_known[0][0].sub_mod2k(&m_known[0][1], k);
        let pmq_c = m_known[1][0].sub_mod2k(&m_known[1][1], k);
        let pmq_target = source.biscalar_mul(&pmq_a, &pmq_c, e);
        let target = TorsionBasis::from_propagated(p_target, q_target, pmq_target);

        // Recover M via from_bases, then check that mul reproduces target.
        let recovered =
            ChangeOfBasisMatrix::from_bases(&source, &target, e).expect("dlog should succeed");
        let applied = recovered.mul(&source);

        assert_eq!(
            applied.R, target.R,
            "from_bases + mul round-trip must reproduce target.R"
        );
        assert_eq!(
            applied.S, target.S,
            "from_bases + mul round-trip must reproduce target.S"
        );
    }

    /// `from_bases` round-trip with matrix entries that exceed `2^32`.
    ///
    /// `m_known` is chosen so that the recovered dlogs straddle the
    /// 32-bit boundary in the recursion's intermediate `k'` value.
    /// Catches the historic bug where `dlog` cast the recursive
    /// `k'` to `u32` and silently truncated for `k' > u32::MAX`,
    /// collapsing every `M_chl` cross-pairing to a fixed root of
    /// unity (\S\ref{sec:dlog-truncation} in the bug catalog).
    #[test]
    fn from_bases_mul_roundtrip_above_u32() {
        let source = e0_basis();
        let e = TorsionExponent::FULL;

        // Entries straddling 2^32 in different limbs to force the
        // dlog recursion's `k'` past u32::MAX at multiple levels.
        let m_known = [
            [
                Scalar::from_limbs([0xDEAD_BEEF_1234_5678, 0x1234_5678_ABCD_EF01, 0, 0]),
                Scalar::from_limbs([0xFEDC_BA98_7654_3210, 0xCAFE_0000_0000_0001, 0, 0]),
            ],
            [
                Scalar::from_limbs([0x0123_4567_89AB_CDEF, 0xFACE_FACE_FACE_FACE, 0, 0]),
                Scalar::from_limbs([0xA5A5_A5A5_A5A5_A5A5, 0x5A5A_5A5A_5A5A_5A5A, 0, 0]),
            ],
        ];

        let p_target = source.biscalar_mul(&m_known[0][0], &m_known[1][0], e);
        let q_target = source.biscalar_mul(&m_known[0][1], &m_known[1][1], e);
        let k = e.value();
        let pmq_a = m_known[0][0].sub_mod2k(&m_known[0][1], k);
        let pmq_c = m_known[1][0].sub_mod2k(&m_known[1][1], k);
        let pmq_target = source.biscalar_mul(&pmq_a, &pmq_c, e);
        let target = TorsionBasis::from_propagated(p_target, q_target, pmq_target);

        let recovered =
            ChangeOfBasisMatrix::from_bases(&source, &target, e).expect("dlog should succeed");
        let applied = recovered.mul(&source);

        assert_eq!(
            applied.R, target.R,
            "from_bases + mul round-trip with k > u32 must reproduce target.R"
        );
        assert_eq!(
            applied.S, target.S,
            "from_bases + mul round-trip with k > u32 must reproduce target.S"
        );
    }

    /// Exercise the exact `from_bases_invert → mul → from_bases` chain
    /// `keys::signing` uses to build `m1` and `m_chl`. Catches breakage
    /// in the inverse-direction matrix or in the `transformed = m1·b`
    /// → `from_bases(canonical, transformed)` second hop, both of
    /// which the simpler `from_bases_mul_roundtrip` test misses.
    ///
    /// Setup mirrors signing:
    /// 1. `canonical_a` = E_0 basis at full order (proxy for `det_aux`).
    /// 2. `reduced_a` = M_known applied to `canonical_a` via biscalar at
    ///    `e_cob` (proxy for `basis_aux`). Order 2^e_cob.
    /// 3. `m1 = from_bases_invert(canonical_a, reduced_a, e_cob)` — expects
    ///    `m1·reduced_a = canonical_a` at 2^e_cob precision.
    /// 4. Apply `m1` to a different "shared" reduced basis via `mul`; the
    ///    result is `transformed`.
    /// 5. `m_chl = from_bases(canonical_a, transformed, e_cob)` should succeed
    ///    (all lifts work) and reproduce a known relationship.
    ///
    /// Step 5 is the load-bearing assertion: if `transformed` (= the
    /// mul output) has lift-incompatible (R, S, RS), the second
    /// `from_bases` returns `None` and signing drops. This is the
    /// failure mode observed when running `sign_kat_zero_only`.
    #[test]
    fn from_bases_invert_mul_then_from_bases_chain() {
        let canonical = e0_basis();
        let e_red: u32 = 128;
        let scale = TorsionExponent::FULL.value() - e_red;
        let scale_scalar = Scalar::from_limbs(*BigInt::<4>::ONE.shl(scale).as_limbs());
        let e = TorsionExponent::try_from(e_red).expect("128 valid");

        // m_known: det = 3·5 − 4·1 = 11 (odd) → invertible mod 2^128.
        let m_known = [
            [Scalar::from_u64(3), Scalar::from_u64(4)],
            [Scalar::from_u64(1), Scalar::from_u64(5)],
        ];
        // Build reduced basis at order 2^e_cob: biscalar at FULL order,
        // then scale down by 2^scale.
        let p_full = canonical.biscalar_mul(&m_known[0][0], &m_known[1][0], TorsionExponent::FULL);
        let q_full = canonical.biscalar_mul(&m_known[0][1], &m_known[1][1], TorsionExponent::FULL);
        let k = TorsionExponent::FULL.value();
        let pmq_a = m_known[0][0].sub_mod2k(&m_known[0][1], k);
        let pmq_c = m_known[1][0].sub_mod2k(&m_known[1][1], k);
        let pmq_full = canonical.biscalar_mul(&pmq_a, &pmq_c, TorsionExponent::FULL);
        let reduced = TorsionBasis::from_propagated(
            &scale_scalar * &p_full,
            &scale_scalar * &q_full,
            &scale_scalar * &pmq_full,
        );

        // Step 3: m1 = inverse direction. m1·reduced ≡ canonical (at 2^e_cob).
        let m1 = ChangeOfBasisMatrix::from_bases_invert(&canonical, &reduced, e)
            .expect("from_bases_invert must succeed on full + reduced E_0 bases");

        // Step 4: apply m1 to a reduced-order basis. Sign uses
        // `basis_chl` here (response-phase output, at order 2^e_cob).
        // We proxy with `reduced` (also order 2^e_cob) — the matrix
        // application then yields `transformed` at order 2^e_cob,
        // matching the sign-side call shape that m_chl consumes.
        let transformed = m1.mul(&reduced);

        // Step 5: second from_bases. This is the call sign currently
        // sees fail with `m_chl None`. transformed ≡ m1·reduced
        // ≡ m1·M·canonical = M^(-1)·M·canonical = canonical (mod 2^e).
        // So m_chl reconstructs the identity matrix at 2^e.
        let m_chl = ChangeOfBasisMatrix::from_bases(&canonical, &transformed, e)
            .expect("from_bases on transformed must succeed (lift consistency)");

        // m_chl·canonical = transformed = canonical → m_chl == identity.
        assert_eq!(
            m_chl.entries[0][0],
            Scalar::from_u64(1),
            "m_chl[0][0] must be 1"
        );
        assert_eq!(m_chl.entries[0][1], Scalar::ZERO, "m_chl[0][1] must be 0");
        assert_eq!(m_chl.entries[1][0], Scalar::ZERO, "m_chl[1][0] must be 0");
        assert_eq!(
            m_chl.entries[1][1],
            Scalar::from_u64(1),
            "m_chl[1][1] must be 1"
        );
    }
}
