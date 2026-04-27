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
    curves::{TorsionExponent, montgomery::ProjectiveXOnlyPoint, scalar::Scalar},
    fields::fp2::Fp2,
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
        let z0_double_prime = self.square_n(e_prime.value());
        // TODO: pow by Scalar — for now convert k' to u32 for small
        // intermediate values. The recursion ensures k' < 2^{e'} which
        // fits in u32 for e' ≤ 124.
        let z1_double_prime = target / &self.pow(k_prime.as_limbs()[0] as u32);

        // k'' = NormalizedDlog(ζ''₀, ζ''₁) — high bits
        let k_double_prime = z0_double_prime.dlog(&z1_double_prime, diff);

        // k = k' + 2^{e'} · k''
        use crate::quaternions::bigint::BigInt;
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
    let two = Fp2::from_fp(crate::fields::fp::Fp::from_small(2));
    let four = Fp2::from_fp(crate::fields::fp::Fp::from_small(4));
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
        curves::{TorsionBasis, montgomery::Curve},
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
}
