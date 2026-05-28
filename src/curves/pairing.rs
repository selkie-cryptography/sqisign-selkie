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
    curves::{TorsionBasis, TorsionExponent, scalar::Scalar},
    fields::{fp::Fp, fp2::Fp2},
    quaternions::bigint::BigInt,
};

#[cfg(test)]
mod tests;

/// An element of μ_{2^e}, the group of 2^e-th roots of unity in F_{p²}*.
///
/// Produced by the reduced Tate pairing ([`TorsionBasis::tate`])
/// and consumed by [`NormalizedDlog`](RootOfUnity::dlog) to solve
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

    /// Computes ζ^{2^n} by repeated squaring.
    #[must_use]
    pub fn square_n(&self, n: u32) -> Self {
        let mut result = self.0;
        for _ in 0..n {
            result = result.square();
        }
        Self(result)
    }

    /// Computes ζ^k for a scalar k.
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

    /// Computes ζ^k for a [`Scalar`]-sized exponent.
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

    /// Computes the discrete log k ∈ \[0, 2^e) such that target = self^k.
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
        let k_high = k_double_prime_big << e_prime.value();
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
    // reason: in a multiplicative group, division IS multiplication by the
    // inverse; clippy's heuristic only sees the literal `*` and flags
    // `Div::div` containing `Mul::mul`.
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

/// A cubical point: projective (X : Z) in F_{p²}.
///
/// Same representation as
/// [`ProjectiveXOnlyPoint`](crate::curves::montgomery::ProjectiveXOnlyPoint)
/// but used in the cubical arithmetic context where the formulas
/// differ.
#[derive(Copy, Clone, Debug)]
struct CubicalPoint {
    /// Projective `X` coordinate.
    X: Fp2,
    /// Projective `Z` coordinate.
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

    /// Constructs from an affine x-coordinate: (x : 1).
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

impl TorsionBasis {
    /// Reduced Tate pairing t_{2^e}(P, Q) on this basis, using
    /// `PmQ` as the cubical-ladder differential point.
    ///
    /// `e` is the torsion exponent (where 2^e · P = O_E). Internally
    /// normalizes to affine x-coordinates for the cubical arithmetic.
    ///
    /// Implements [Tate][Alg. 8.18] from the spec.
    ///
    /// [Alg. 8.18]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.18
    pub(crate) fn tate(&self, e: TorsionExponent) -> RootOfUnity {
        let e = e.value();

        let curve = self.P.curve();
        let a = *curve.coefficient().as_fp2();
        let two = Fp2::from_fp(Fp::from_small(2));
        let four = Fp2::from_fp(Fp::from_small(4));
        let a24 = &(&a + &two) * &four.invert();

        // Normalize to affine x-coordinates for cubical arithmetic.
        let x_p = self.P.to_affine_x();
        let x_q = self.Q.to_affine_x();
        let x_pmq = self.PmQ.to_affine_x();

        // Step 1: (nP, nPQ) ← CubicalLadder(E, e-1, (x(P+Q),1), (x(P),1), x(Q))
        //
        // `x(P-Q)` works equivalently to `x(P+Q)` here: the cubical
        // ladder's differential-add requirement is on x(npq - np) = x(Q),
        // which holds whether npq starts at P+Q or P-Q (x is sign-symmetric).
        let mut np = CubicalPoint::from_affine(*x_p.as_fp2());
        let mut npq = CubicalPoint::from_affine(*x_pmq.as_fp2());
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
    /// check; the symmetric reduced-input version of [`Self::tate`]
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
        let curve = *self.P.curve();
        debug_assert_eq!(
            curve,
            *reduced.P.curve(),
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
        //
        // # Field-to-ladder-name mapping
        //
        // The pairing ladder operates on a basis pair `(p_ladder,
        // q_ladder)` with precomputed difference `x_PmQ_ladder`. Under
        // Selkie's spec-permuted storage layout (see [`TorsionBasis`])
        // those map to the fields as:
        //
        //   p_ladder       ↔ self.P     (first basis point)
        //   q_ladder       ↔ self.PmQ   (= our P − Q; spec's permuted "S")
        //   x_PmQ_ladder   ↔ self.Q     (= our Q; spec's permuted "RS")
        //
        // i.e., the pairing computation reads the *positional* slot
        // semantics, not the field-name semantics. C ref's
        // `ec_dlog_2_tate` does the same: it consumes (B.P, B.Q, B.PmQ)
        // positionally with the same permutation applied at basis
        // construction time.
        let xp = *self.P.to_affine_x().as_fp2();
        let xq = *self.PmQ.to_affine_x().as_fp2();
        let xpmq = *self.Q.to_affine_x().as_fp2();
        let xr = *reduced.P.to_affine_x().as_fp2();
        let xs = *reduced.PmQ.to_affine_x().as_fp2();

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
        // Two-pass with Montgomery's trick on the five `den` inversions:
        // one `Fp2::invert` + 12 muls instead of five `Fp2::invert`s.
        let mut nums: [Fp2; 5] = [Fp2::ZERO; 5];
        let mut dens: [Fp2; 5] = [Fp2::ZERO; 5];
        for (idx, (w_x, w_z)) in raw_xz.iter().enumerate() {
            // (X/Z)^(p−1) = (X^p · Z) / (X · Z^p).
            let x_p = w_x.conjugate();
            let z_p = w_z.conjugate();
            nums[idx] = w_z * &x_p;
            dens[idx] = w_x * &z_p;
        }

        // Forward prefix products, one inversion, backward peel.
        let p1 = &dens[0] * &dens[1];
        let p2 = &p1 * &dens[2];
        let p3 = &p2 * &dens[3];
        let p4 = &p3 * &dens[4];
        let mut inv = p4.invert();
        let mut den_invs: [Fp2; 5] = [Fp2::ZERO; 5];
        den_invs[4] = &inv * &p3;
        inv = &inv * &dens[4];
        den_invs[3] = &inv * &p2;
        inv = &inv * &dens[3];
        den_invs[2] = &inv * &p1;
        inv = &inv * &dens[2];
        den_invs[1] = &inv * &dens[0];
        inv = &inv * &dens[1];
        den_invs[0] = inv;

        let mut out = [RootOfUnity::ONE; 5];
        for idx in 0..5 {
            let frac = &nums[idx] * &den_invs[idx];

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

impl TorsionBasis {
    /// Weil pairing e_{2^e}(P, Q) on this basis.
    ///
    /// Defined as `e(P, Q) = t(P, Q) / t(Q, P)` where `t` is the
    /// reduced Tate pairing. The `PmQ` differential is sign-symmetric
    /// in `(P, Q)` on the x-line, so the same basis serves both Tate
    /// directions.
    ///
    /// Used in [`LeftIdeal::to_isogeny`] to disambiguate the two
    /// codomain components of the (2,2)-chain on `E_u × E_v`.
    ///
    /// [`LeftIdeal::to_isogeny`]: crate::quaternions::lattice::LeftIdeal::to_isogeny
    pub(crate) fn weil(&self, e: TorsionExponent) -> RootOfUnity {
        let t_pq = self.tate(e);
        let swapped = TorsionBasis {
            P: self.Q,
            PmQ: self.PmQ,
            Q: self.P,
        };
        let t_qp = swapped.tate(e);
        let result = t_pq.as_fp2() * &t_qp.as_fp2().invert();
        RootOfUnity(result)
    }
}
