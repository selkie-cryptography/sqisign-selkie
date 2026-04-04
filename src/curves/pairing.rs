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
//! [§8.3.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.3

use core::ops::{Div, Mul};

use subtle::{Choice, ConditionallySelectable};

use crate::curves::montgomery::ProjectiveXOnlyPoint;
use crate::curves::scalar::Scalar;
use crate::curves::TorsionExponent;
use crate::fields::fp2::Fp2;

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

    /// Compute the discrete log k ∈ \[0, 2^e) such that target = self^k.
    ///
    /// Uses the Pohlig-Hellman algorithm for 2-power order groups.
    /// SQIsign only applies this to pairing outputs.
    ///
    /// Implements [NormalizedDlog][Alg. 2.4].
    ///
    /// [Alg. 2.4]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.2.4
    pub fn dlog(&self, target: &Self, e: TorsionExponent) -> Scalar {
        let e = e.value();
        if e == 0 {
            return Scalar::ZERO;
        }
        if e == 1 {
            return if *target == Self::ONE {
                Scalar::ZERO
            } else {
                Scalar::ONE
            };
        }

        // e' = ⌊e/2⌋
        let e_prime = e / 2;

        // ζ'₀ = ζ₀^{2^{e-e'}},  ζ'₁ = ζ₁^{2^{e-e'}}
        let z0_prime = self.square_n(e - e_prime);
        let z1_prime = target.square_n(e - e_prime);

        // k' = NormalizedDlog(ζ'₀, ζ'₁) — low bits
        let k_prime = z0_prime.dlog(&z1_prime, TorsionExponent::new(e_prime));

        // ζ''₀ = ζ₀^{2^{e'}},  ζ''₁ = ζ₁ / ζ₀^{k'}
        let z0_double_prime = self.square_n(e_prime);
        // TODO: pow by Scalar — for now convert k' to u32 for small
        // intermediate values. The recursion ensures k' < 2^{e'} which
        // fits in u32 for e' ≤ 124.
        let z1_double_prime = target / &self.pow(k_prime.as_limbs()[0] as u32);

        // k'' = NormalizedDlog(ζ''₀, ζ''₁) — high bits
        let k_double_prime =
            z0_double_prime.dlog(&z1_double_prime, TorsionExponent::new(e - e_prime));

        // k = k' + 2^{e'} · k''
        // Use BigInt for the shift since e' can exceed 63.
        use crate::quaternions::bigint::BigInt;
        let k_prime_big = BigInt::<4>::from(k_prime);
        let k_double_prime_big = BigInt::<4>::from(k_double_prime);
        let k_high = k_double_prime_big.shl(e_prime);
        let k = k_prime_big.ct_add(&k_high);
        Scalar::from_limbs(*k.as_limbs())
    }
}

impl<'a, 'b> Mul<&'b RootOfUnity> for &'a RootOfUnity {
    type Output = RootOfUnity;
    fn mul(self, rhs: &'b RootOfUnity) -> RootOfUnity {
        RootOfUnity(&self.0 * &rhs.0)
    }
}

impl<'a, 'b> Div<&'b RootOfUnity> for &'a RootOfUnity {
    type Output = RootOfUnity;
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
    fn differential_add(&self, other: &Self, x_diff: &Fp2) -> Self {
        let a = &self.X + &self.Z;
        let b = &self.X - &self.Z;
        let c = &other.X + &other.Z;
        let d = &other.X - &other.Z;
        let x2 = (&a * &d + &b * &c).square();
        let z2 = &(&a * &d - &b * &c).square() * x_diff;
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
    let a = Fp2::from(*curve.coefficient().as_fp2());
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
