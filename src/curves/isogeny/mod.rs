//! Isogenies between elliptic curves.
//!
//! An isogeny φ : E₁ → E₂ is defined by its kernel: a point P ∈ E₁
//! of order 2^e generates the cyclic subgroup ⟨P⟩, which uniquely
//! determines the isogeny.
//!
//! The [`Kernel`] type wraps a [`ProjectiveXOnlyPoint`] that generates
//! the kernel subgroup. Computing the isogeny and pushing points
//! through is done via [`Kernel::isogeny`].
//!
//! Internally, isogenies of degree 2^e are decomposed into chains
//! of 4-isogenies (with an optional trailing 2-isogeny if e is odd).
//!
//! See [§2.3] and [§8.4].
//!
//! [§2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.3
//! [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4

use core::cmp::Ordering;

use subtle::{Choice, ConstantTimeEq, ConstantTimeGreater};

use crate::{
    curves::{
        TorsionExponent,
        montgomery::{Curve, DoublingConstants, ProjectiveXOnlyPoint},
        scalar::Scalar,
    },
    fields::fp2::Fp2,
    quaternions::bigint::BigInt,
};

/// Cost units for the cost-optimal 4-isogeny chain strategy
/// ([Algorithm 8.25 / §8.4][§8.4]): `P_COST_4ISO` is the cost of one
/// "step-down" doubling in 4-iso-step terms (two x-only Montgomery
/// point doublings, ~4M + 2S each with `S ≈ 0.8 M`); `Q_COST_4ISO`
/// is the cost of one 4-isogeny pushforward (~6M + 2S). Only the
/// ratio drives the strategy DP, and DFJP is robust to small
/// mis-estimation.
///
/// [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4
const P_COST_4ISO: u32 = 11;
const Q_COST_4ISO: u32 = 8;

/// Maximum supported subtree size for the precomputed 4-isogeny chain
/// strategy. NIST-I uses chains of bit length `E_RSP = 126`, decomposed
/// into ⌊126/2⌋ = 63 4-isogenies; 128 leaves headroom for shorter
/// chains to share the same array.
const STRATEGY_4ISO_MAX_LEN: usize = 128;

/// Returns the cost-optimal first-split array for an isogeny chain of
/// size up to `N - 1`, computed by dynamic programming on the cost
/// function `cost[h] = j·p + (h − j)·q + cost[j] + cost[h − j]` with
/// `cost[1] = 0`. `split[h]` (for `1 < h < N`) is the optimal `j` for
/// a subtree of size `h`; `split[0]` and `split[1]` are unused —
/// a size-1 subtree is a single kernel, no doublings needed.
const fn compute_optimal_strategy_splits<const N: usize>(p: u32, q: u32) -> [u32; N] {
    let mut cost = [0u64; N];
    let mut split = [0u32; N];
    let mut h = 2usize;
    while h < N {
        let mut best_cost = u64::MAX;
        let mut best_split = 1u32;
        let mut j = 1u32;
        while (j as usize) < h {
            let c = (j as u64) * (p as u64)
                + ((h - j as usize) as u64) * (q as u64)
                + cost[j as usize]
                + cost[h - j as usize];
            if c < best_cost {
                best_cost = c;
                best_split = j;
            }
            j += 1;
        }
        cost[h] = best_cost;
        split[h] = best_split;
        h += 1;
    }
    split
}

/// Precomputed cost-optimal splits for the 4-isogeny chain at NIST-I
/// cost parameters. Used by [`Kernel::isogeny`].
const STRATEGY_4ISO: [u32; STRATEGY_4ISO_MAX_LEN] =
    compute_optimal_strategy_splits::<STRATEGY_4ISO_MAX_LEN>(P_COST_4ISO, Q_COST_4ISO);

/// A positive odd integer representing the degree of a separable isogeny.
///
/// Stored as an unsigned 256-bit integer (`[u64; 4]`, little-endian).
/// For NIST-I, isogeny degrees are bounded by 2^{f−2} = 2^{246},
/// so 256 bits is always sufficient (see [Alg. 3.15][Alg. 3.15]
/// and Kim et al., ePrint 2025/1649, Table 2).
///
/// Positive by construction (no sign bit). Always odd — enforced by
/// [`IsogenyDegree::new_odd`].
///
/// [Alg. 3.15]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.15
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IsogenyDegree([u64; 4]);

impl IsogenyDegree {
    /// Construct from a positive odd limb array, or `None` if zero or even.
    pub fn new_odd(limbs: [u64; 4]) -> Option<Self> {
        if limbs[0] & 1 == 0 {
            return None; // even or zero
        }
        Some(Self(limbs))
    }

    /// The underlying limbs (little-endian).
    #[inline]
    pub fn limbs(&self) -> &[u64; 4] {
        &self.0
    }

    /// Bit length (position of highest set bit).
    pub fn bit_length(&self) -> u32 {
        for i in (0..4).rev() {
            if self.0[i] != 0 {
                return (i as u32) * 64 + (64 - self.0[i].leading_zeros());
            }
        }
        0
    }

    /// Convert to a [`Scalar`] for elliptic curve point multiplication.
    #[inline]
    pub fn to_scalar(self) -> Scalar {
        Scalar::from_limbs(self.0)
    }

    /// Widens to [`BigInt<8>`](crate::quaternions::bigint::BigInt) for
    /// quaternion arithmetic.
    pub(crate) fn to_bigint_wide(self) -> BigInt<8> {
        BigInt::<8>::from_sign_and_limbs(
            0,
            [self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0],
        )
    }
}

impl IsogenyDegree {
    /// Constructs from a positive odd 256-bit value encoded as 32
    /// little-endian bytes, or `None` if the value is zero or even.
    pub(crate) fn from_bytes_le(bytes: &[u8; 32]) -> Option<Self> {
        let b = BigInt::<4>::from_bytes_le_unsigned(bytes);
        Self::new_odd(*b.as_limbs())
    }

    /// Converts to a [`BigInt<4>`](crate::quaternions::bigint::BigInt)
    /// for quaternion arithmetic.
    #[inline]
    pub(crate) fn to_bigint(self) -> BigInt<4> {
        BigInt::<4>::from_limbs(self.0)
    }
}

impl From<IsogenyDegree> for BigInt<4> {
    #[inline]
    fn from(d: IsogenyDegree) -> BigInt<4> {
        d.to_bigint()
    }
}

impl From<IsogenyDegree> for Scalar {
    #[inline]
    fn from(d: IsogenyDegree) -> Scalar {
        Scalar::from_limbs(d.0)
    }
}

impl Ord for IsogenyDegree {
    /// Numeric comparison, most-significant limb first.
    ///
    /// A derived `Ord` on `[u64; 4]` would compare lexicographically
    /// starting at index 0 — the *least*-significant limb in our
    /// little-endian encoding — giving the wrong answer whenever
    /// two degrees differ above the bottom 64 bits.
    ///
    /// `IsogenyDegree` is non-negative and non-zero by construction
    /// (via [`IsogenyDegree::new_odd`]), so this is a magnitude
    /// compare with no sign or zero special cases.
    ///
    /// # Constant-time
    ///
    /// Each comparison is data-flow-uniform: the loop visits every
    /// limb, and per-limb tests use [`ConstantTimeGreater`] on
    /// `u64` from the `subtle` crate. A single comparison therefore
    /// does not leak the operands through its memory access pattern
    /// or through branches on the limb values.
    ///
    /// The final return lowers the CT accumulator [`Choice`] values
    /// into an [`Ordering`], and sort algorithms branch on that
    /// result — so `sort_by_key` / `sort_by` on secret-derived
    /// degrees remains variable-time overall. CT callers must avoid
    /// those sorts in favour of a sorting network built on
    /// `ConditionallySelectable::conditional_swap`.
    fn cmp(&self, other: &Self) -> Ordering {
        // Walk MSB → LSB; `undecided` gates further updates so the
        // first differing limb wins without a data-dependent branch.
        let mut gt = Choice::from(0u8);
        let mut lt = Choice::from(0u8);
        for i in (0..4).rev() {
            let undecided = !(gt | lt);
            gt |= undecided & self.0[i].ct_gt(&other.0[i]);
            lt |= undecided & other.0[i].ct_gt(&self.0[i]);
        }
        if bool::from(gt) {
            Ordering::Greater
        } else if bool::from(lt) {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    }
}

impl PartialOrd for IsogenyDegree {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The kernel of a 2^e-isogeny, defined by a generator point P of
/// order 2^e on an elliptic curve.
///
/// The curve is carried by the point itself (as with all
/// [`ProjectiveXOnlyPoint`]s). Computing the isogeny and evaluating
/// points through it is done via [`Kernel::isogeny`] or
/// [`Kernel::isogeny_small`].
///
/// See [§2.3].
///
/// [§2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.3
#[derive(Copy, Clone, Debug)]
pub struct Kernel(ProjectiveXOnlyPoint);

impl Kernel {
    /// Construct from a generator point.
    pub fn new(generator: ProjectiveXOnlyPoint) -> Kernel {
        Kernel(generator)
    }

    /// Compute the 2^e-isogeny defined by this kernel and push
    /// points through it.
    ///
    /// Uses a cost-optimal DFJP strategy precomputed by
    /// [`compute_optimal_strategy_splits`] at compile time: a chain of
    /// ⌊e/2⌋ 4-isogenies followed by an optional 2-isogeny if `e` is
    /// odd.
    ///
    /// Returns the codomain curve and the images of `pts`.
    ///
    /// Implements `TwoIsogenyChain` ([§8.4], Algorithm 8.25).
    ///
    /// # Security
    ///
    /// The generator must have order exactly 2^e. This is assumed
    /// by construction and not verified at runtime.
    ///
    /// [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4
    pub fn isogeny(
        &self,
        e: TorsionExponent,
        pts: &[ProjectiveXOnlyPoint],
    ) -> (Curve, Vec<ProjectiveXOnlyPoint>) {
        let e = e.value();
        let mut curve = *self.0.curve();
        let mut pts: Vec<ProjectiveXOnlyPoint> = pts.to_vec();

        let mut strat_pts: Vec<ProjectiveXOnlyPoint> = vec![self.0];
        let mut orders: Vec<u32> = vec![e];
        let mut k: usize = 0;

        // Chain of 4-isogenies.
        for _j in 0..(e / 2) as usize {
            while orders[k] != 2 {
                k += 1;
                let o = orders[k - 1];
                // Subtree size in 4-iso-step terms (one 4-iso ≡ two
                // log-order steps); the odd-parity `extra` doubling
                // keeps the inner loop converging on `orders[k] == 2`
                // for odd `e`.
                let h = (o / 2) as usize;
                let extra = o % 2;
                let n = STRATEGY_4ISO[h] * 2 + extra;
                let new_order = o - n;
                orders.push(new_order);
                let mut pt = strat_pts[k - 1];
                for _ in 0..n {
                    pt = pt.double();
                }
                strat_pts.push(pt);
            }

            // Note: singular 4-isogeny kernels ([2]K at x=0) can occur
            // legitimately during verification. The 4-isogeny formulas
            // handle this case correctly.

            let phi = FourIsogeny::from_kernel(&strat_pts[k]);
            curve = phi.codomain;

            for pt in strat_pts.iter_mut().take(k) {
                *pt = phi.eval(pt);
            }
            for order in orders.iter_mut().take(k) {
                *order -= 2;
            }
            orders.truncate(k);
            strat_pts.truncate(k);
            k = k.saturating_sub(1);

            for pt in pts.iter_mut() {
                *pt = phi.eval(pt);
            }
        }

        // Final 2-isogeny if e is odd.
        if e % 2 == 1 {
            if let Some(&K) = strat_pts.first() {
                debug_assert!(
                    !bool::from(K.X.ct_eq(&Fp2::ZERO)),
                    "unexpected singular 2-isogeny kernel"
                );
                let phi = TwoIsogeny::from_kernel(&K);
                curve = phi.codomain;
                for pt in pts.iter_mut() {
                    *pt = phi.eval(pt);
                }
            }
        }

        (curve, pts)
    }

    /// Compute a small 2^e-isogeny chain naively.
    ///
    /// For small exponents. Handles the singular kernel case P = (0 : 1)
    /// when `is_signing` is true.
    ///
    /// Implements `TwoIsogenyChainSmall` ([§8.4], [Alg. 8.26][Alg. 8.26]).
    ///
    /// # Singular kernels
    ///
    /// Intermediate kernel points may land on (0 : 1), producing a
    /// singular 2-isogeny. The spec says this "can include the special
    /// case of a kernel generator P = (0 : 1) only during signing"
    /// (page 65), but in practice singular kernels also arise during
    /// verification when the even response isogeny passes through
    /// (0, 0). We handle both cases uniformly.
    ///
    /// [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4
    /// [Alg. 8.26]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.26
    pub fn isogeny_small(
        &self,
        e: TorsionExponent,
        pts: &[ProjectiveXOnlyPoint],
        _is_signing: bool,
    ) -> Result<(Curve, Vec<ProjectiveXOnlyPoint>), &'static str> {
        let e = e.value();
        let mut curve = *self.0.curve();
        let mut P = self.0;
        let mut pts: Vec<ProjectiveXOnlyPoint> = pts.to_vec();

        for i in 0..e {
            // Compute the 2-torsion kernel for this step by
            // doubling P down to order 2.
            let mut K = P;
            for _ in 0..(e - i - 1) {
                K = K.double();
            }

            if i == 0 && !bool::from(K.double().is_identity()) {
                return Err("wrong point order");
            }

            if bool::from(K.X.ct_eq(&Fp2::ZERO)) {
                let phi = TwoIsogenySingular::from_curve(&curve);
                P = phi.eval(&P);
                curve = phi.codomain;
                for pt in pts.iter_mut() {
                    *pt = phi.eval(pt);
                }
            } else {
                let phi = TwoIsogeny::from_kernel(&K);
                P = phi.eval(&P);
                curve = phi.codomain;
                for pt in pts.iter_mut() {
                    *pt = phi.eval(pt);
                }
            }
        }

        Ok((curve, pts))
    }
}

/// A non-singular 2-isogeny φ : E → E'.
///
/// See [§8.4], Algorithms 8.19–8.20.
///
/// [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4
#[derive(Copy, Clone, Debug)]
pub(crate) struct TwoIsogeny {
    pub(crate) codomain: Curve,
    kernel: ProjectiveXOnlyPoint,
}

impl TwoIsogeny {
    pub(crate) fn from_kernel(P: &ProjectiveXOnlyPoint) -> TwoIsogeny {
        let xp_sq = P.X.square();
        let zp_sq = P.Z.square();
        let A24 = &zp_sq - &xp_sq;
        let C24 = zp_sq;
        TwoIsogeny {
            codomain: Curve::from(DoublingConstants { A24, C24 }),
            kernel: *P,
        }
    }

    pub(crate) fn eval(&self, Q: &ProjectiveXOnlyPoint) -> ProjectiveXOnlyPoint {
        let t0 = &self.kernel.X + &self.kernel.Z;
        let t1 = &self.kernel.X - &self.kernel.Z;
        let t2 = &Q.X + &Q.Z;
        let t3 = &Q.X - &Q.Z;
        let t0 = &t0 * &t3;
        let t1 = &t1 * &t2;
        let t2 = &t0 + &t1;
        let t3 = &t0 - &t1;
        ProjectiveXOnlyPoint::from_XZ(&Q.X * &t2, &Q.Z * &t3, &self.codomain)
    }
}

/// A singular 2-isogeny with kernel at (0, 0).
///
/// See [§8.4], Algorithms 8.21–8.22.
///
/// [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4
#[derive(Copy, Clone, Debug)]
pub(crate) struct TwoIsogenySingular {
    pub(crate) codomain: Curve,
    c0: Fp2,
    c1: Fp2,
}

impl TwoIsogenySingular {
    pub(crate) fn from_curve(curve: &Curve) -> TwoIsogenySingular {
        let dc = curve.doubling_constants();
        let (A24, C24) = (&dc.A24, &dc.C24);
        let t0 = &(A24 + A24) - C24;
        let t0 = &t0 + &t0;
        let t1 = C24.invert();
        let t0 = &t0 * &t1;
        let c1 = t0;
        let A24_prime = &t0 + &t0;
        let t0 = t0.square();
        let four = Fp2::from_fp(crate::fields::fp::Fp::from_small(4));
        let t0 = &t0 - &four;
        let t0 = t0.sqrt();
        let c0 = -&t0;
        let C24_prime = &t0 + &t0;
        let A24_prime = &A24_prime + &C24_prime;
        let C24_prime = &C24_prime + &C24_prime;
        TwoIsogenySingular {
            codomain: Curve::from(DoublingConstants {
                A24: A24_prime,
                C24: C24_prime,
            }),
            c0,
            c1,
        }
    }

    pub(crate) fn eval(&self, Q: &ProjectiveXOnlyPoint) -> ProjectiveXOnlyPoint {
        // Mirrors C ref's `xeval_2_singular` (`xeval.c:25`):
        //   R.x = Q.z² + K.x · Q.x · Q.z + Q.x²
        //   R.z = Q.x · Q.z · K.z
        //
        // where C ref's K.x = A (affine A coefficient of the domain) and
        // K.z = −√(A² − 4) — see `xisog_2_singular` (`xisog.c:20`).
        //
        // Selkie's `from_curve` stores c1 = A and c0 = −√(A² − 4) (the
        // structural invariant `c0² = c1² − 4` is checked by
        // `two_isogeny_singular_structural`), so c1 corresponds to C ref's
        // K.x and c0 to K.z. Use them accordingly here.
        let t0 = &Q.X * &Q.Z;
        let t1 = &Q.X + &(&self.c1 * &Q.Z);
        let t1 = &t1 * &Q.X;
        let XQ = &Q.Z.square() + &t1;
        let ZQ = &t0 * &self.c0;
        ProjectiveXOnlyPoint::from_XZ(XQ, ZQ, &self.codomain)
    }
}

/// A 4-isogeny φ : E → E'.
///
/// See [§8.4], Algorithms 8.23–8.24.
///
/// [§8.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.4
#[derive(Copy, Clone, Debug)]
pub(crate) struct FourIsogeny {
    pub(crate) codomain: Curve,
    c0: Fp2,
    c1: Fp2,
    c2: Fp2,
}

impl FourIsogeny {
    pub(crate) fn from_kernel(P: &ProjectiveXOnlyPoint) -> FourIsogeny {
        let zp_sq = P.Z.square();
        let c1 = &P.X - &P.Z;
        let c2 = &P.X + &P.Z;
        let xp_sq = P.X.square();
        let t3 = &zp_sq + &xp_sq;
        let t4 = &zp_sq - &xp_sq;
        let A24 = &t3 * &t4;
        let C24 = zp_sq.square();
        let c0 = {
            let d = &zp_sq + &zp_sq;
            &d + &d
        };
        FourIsogeny {
            codomain: Curve::from(DoublingConstants { A24, C24 }),
            c0,
            c1,
            c2,
        }
    }

    pub(crate) fn eval(&self, Q: &ProjectiveXOnlyPoint) -> ProjectiveXOnlyPoint {
        let t0 = &Q.X + &Q.Z;
        let t1 = &Q.X - &Q.Z;
        let xq = &t0 * &self.c1;
        let zq = &t1 * &self.c2;
        let t0 = &(&t0 * &t1) * &self.c0;
        let t1 = (&xq + &zq).square();
        let zq = (&xq - &zq).square();
        let xq = &(&t0 + &t1) * &t1;
        let zq = &zq * &(&t0 - &zq);
        ProjectiveXOnlyPoint::from_XZ(xq, zq, &self.codomain)
    }
}

#[cfg(test)]
mod tests;
