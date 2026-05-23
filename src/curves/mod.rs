//! Elliptic curves, points, and isogenies between them.
//!
//! This module provides:
//! - [`montgomery`]: Montgomery curves and x-only projective point arithmetic
//! - [`TorsionBasis`]: generators of torsion subgroups, used to define isogeny
//!   kernels
//! - `two_isogeny`, `four_isogeny`: individual isogeny steps
//! - `chain`: chains of isogenies of degree 2^e
//! - Torsion basis hints and ladders ([§2.2.3], [§8.2])
//!
//! # Divergences from spec / C reference
//!
//! - **Basis slot convention**: the C reference stores `(P, P−Q, Q)` not `(P,
//!   Q, P−Q)`, so [`scalar_mul_add`](TorsionBasis::scalar_mul_add) computes `P
//!   + [m](P−Q)`. We match this convention. See the comment on
//!   [`TorsionBasis::from_hint`].
//! - **P−Q never recomputed**: after scaling, matrix application, or isogeny
//!   evaluation, the third basis point P−Q is propagated, not recomputed via
//!   [`projective_difference`](montgomery::ProjectiveXOnlyPoint::projective_difference).
//!   Recomputing invokes Fp2 sqrt which may pick a different branch.
//!
//! [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.2.3
//! [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2

pub mod isogeny;
pub mod montgomery;
pub(crate) mod pairing;
pub mod scalar;

use core::ops::Sub;

pub use scalar::Scalar;
use subtle::ConditionallySelectable;

#[cfg(test)]
mod tests;

use crate::{
    curves::montgomery::{
        AffineX, Curve, JacobianPoint, ProjectiveXOnlyPoint, differential_add_and_double,
    },
    deuring::precomputed::ENDOMORPHISM_MATRICES,
    fields::{fp::Fp, fp2::Fp2},
    params::TORSION_EVEN_POWER,
    quaternions::{
        algebra::{Coordinate, Denominator, Element},
        bigint::BigInt,
        lattice::LeftIdeal,
        precomputed::EXTREMAL_ORDERS,
    },
};

/// An exponent e such that 2^e divides the torsion group order.
///
/// Always satisfies 0 ≤ e ≤ [`TORSION_EVEN_POWER`]. Used to specify
/// the degree 2^e of isogeny chains and torsion subgroups.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct TorsionExponent(u32);

impl TorsionExponent {
    /// The full torsion exponent f = [`TORSION_EVEN_POWER`].
    pub const FULL: TorsionExponent = TorsionExponent(TORSION_EVEN_POWER);

    /// The raw exponent value.
    pub fn value(self) -> u32 {
        self.0
    }

    /// Subtract, returning `None` if the result would be negative
    /// or exceeds f.
    pub fn checked_sub(self, rhs: u32) -> Option<TorsionExponent> {
        self.0.checked_sub(rhs).and_then(|e| e.try_into().ok())
    }

    /// Floor-divide by 2: ⌊e/2⌋. Always valid since ⌊e/2⌋ ≤ e.
    #[must_use]
    pub fn halve(self) -> TorsionExponent {
        TorsionExponent(self.0 / 2)
    }
}

impl Sub for TorsionExponent {
    type Output = Self;
    /// Subtract two exponents. The result is always ≤ self, so always valid.
    ///
    /// # Panics
    ///
    /// Debug-panics if `rhs > self`.
    fn sub(self, rhs: Self) -> Self {
        debug_assert!(rhs.0 <= self.0);
        Self(self.0 - rhs.0)
    }
}

impl From<TorsionExponent> for u32 {
    fn from(e: TorsionExponent) -> u32 {
        e.0
    }
}

impl TryFrom<u32> for TorsionExponent {
    type Error = ();
    fn try_from(e: u32) -> Result<Self, ()> {
        if e <= TORSION_EVEN_POWER {
            Ok(TorsionExponent(e))
        } else {
            Err(())
        }
    }
}

/// A 1-byte hint for deterministic torsion basis reconstruction.
///
/// Encodes a pair (h_A, h) where h_A is a quadratic residuosity flag
/// (1 bit, stored in the LSB) and h is a 7-bit index used to find a
/// suitable x-coordinate for the first basis point.
///
/// See [§2.2.3], [§4.6].
///
/// [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.2.3
/// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BasisHint(u8);

impl BasisHint {
    /// The quadratic residuosity flag h_A (0 or 1).
    fn h_A(self) -> u8 {
        self.0 & 1
    }

    /// The 7-bit index h.
    fn h(self) -> u8 {
        self.0 >> 1
    }

    /// Construct from the (h_A, h) pair.
    fn new(h_A: u8, h: u8) -> BasisHint {
        debug_assert!(h_A <= 1);
        debug_assert!(h < 128);
        BasisHint((h << 1) | (h_A & 1))
    }

    /// The raw byte representation.
    pub fn to_byte(self) -> u8 {
        self.0
    }

    /// Construct from a raw byte.
    pub fn from_byte(b: u8) -> BasisHint {
        BasisHint(b)
    }
}

/// Hint for the verifying key torsion basis on E_pk.
///
/// Serialized as part of the [verifying key][§4.6] (1 byte).
///
/// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct VerifyingKeyHint(BasisHint);

impl From<u8> for VerifyingKeyHint {
    fn from(b: u8) -> Self {
        VerifyingKeyHint(BasisHint::from_byte(b))
    }
}

impl From<VerifyingKeyHint> for u8 {
    fn from(h: VerifyingKeyHint) -> u8 {
        h.0.to_byte()
    }
}

/// Hint for the auxiliary curve torsion basis on E_aux.
///
/// Serialized as part of the [signature][§4.6] (1 byte).
///
/// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AuxiliaryHint(BasisHint);

impl From<u8> for AuxiliaryHint {
    fn from(b: u8) -> Self {
        AuxiliaryHint(BasisHint::from_byte(b))
    }
}

impl From<AuxiliaryHint> for u8 {
    fn from(h: AuxiliaryHint) -> u8 {
        h.0.to_byte()
    }
}

/// Hint for the challenge curve torsion basis on E_chl.
///
/// Serialized as part of the [signature][§4.6] (1 byte).
///
/// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ChallengeHint(BasisHint);

impl From<u8> for ChallengeHint {
    fn from(b: u8) -> Self {
        ChallengeHint(BasisHint::from_byte(b))
    }
}

impl From<ChallengeHint> for u8 {
    fn from(h: ChallengeHint) -> u8 {
        h.0.to_byte()
    }
}

/// An x-only basis of a torsion subgroup E\[m\] stored in the
/// **spec-permuted layout** of [§2.2.3]: a triple `(x_P, x_{P−Q}, x_Q)`
/// where `(P, Q)` is the basis pair and `Q` is constructed so the
/// 2-torsion point at `(0, 0)` lies in this slot.
///
/// # Layout — why the middle slot holds the difference
///
/// The spec (§2.2.3) computes a basis `(R, S)` and then **permutes**
/// the triple `(x_R, x_S, x_RS)` to `(x_R, x_RS, x_S)` so that the
/// final "second basis point" is `R + S`, which is above `(0, 0)`.
/// In our field naming this maps to:
/// - [`P`](Self::P): the first basis point (spec's `R`).
/// - [`PmQ`](Self::PmQ): `x(P − Q)` — the differential-addition precomputation
///   (spec's permuted `x_S`).
/// - [`Q`](Self::Q): the second basis point, above `(0, 0)` for non-`E₀` curves
///   (spec's permuted `x_RS`).
///
/// The "above `(0, 0)`" property of `Q` enables the special fast doubling
/// `xDBL_E0`; downstream algorithms assume it.
///
/// `PmQ` is required for differential addition, the only way to
/// compute `P + Q` in x-only Montgomery arithmetic. `LadderBiscalar`
/// reads `(P, Q, PmQ)` and computes `[m]P + [n]·(P − Q)` under the
/// spec-permuted application (see [`biscalar_mul`](Self::biscalar_mul)).
/// The matrix `M_chl` recorded in the signature is encoded in this
/// same layout, so signing and verify must agree.
///
/// # Constructors
///
/// Prefer `From<(P, Q)>` which computes `P − Q` automatically via
/// `projective_difference`. Use [`from_propagated`](Self::from_propagated)
/// when `P − Q` is already known from a prior computation (e.g.,
/// propagated through an isogeny evaluation). The `PmQ` argument to
/// `from_propagated` MUST be the actual `projective_difference(P, Q)`
/// — NOT an independently computed point with the same affine x —
/// since the projective representative affects the Okeya-Sakurai
/// y-recovery in [`lift`](Self::lift).
///
/// [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.2.3
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TorsionBasis {
    /// First basis point P.
    pub P: ProjectiveXOnlyPoint,
    /// Differential `x(P − Q)`, precomputed for differential addition.
    ///
    /// Must be the actual `projective_difference(P, Q)` — not an
    /// independently computed point with the same affine x. The
    /// projective representative affects the Okeya-Sakurai y-recovery
    /// in [`lift`](Self::lift) and propagates through isogeny chain
    /// evaluations.
    pub PmQ: ProjectiveXOnlyPoint,
    /// Second basis point Q. For non-`E₀` curves this is constructed
    /// so it lies above the 2-torsion point at `(0, 0)`, enabling
    /// the `xDBL_E0` fast doubling path on doublings of `Q`.
    pub Q: ProjectiveXOnlyPoint,
}

/// Construct a [`TorsionBasis`] from two [`ProjectiveXOnlyPoint`]s,
/// computing `P − Q` via [`ProjectiveXOnlyPoint::projective_difference`].
///
/// # Security
///
/// This does **not** verify that `P` and `Q` actually generate the
/// full n-torsion subgroup E\[n\]. The caller must ensure the points
/// are linearly independent and of the correct order. In SQIsign, this
/// is guaranteed by construction from `TorsionBasisFromHint` or
/// from the `ChallengeMatrix` transformation.
impl From<(ProjectiveXOnlyPoint, ProjectiveXOnlyPoint)> for TorsionBasis {
    /// Build a torsion basis from the pair `(P, Q)`, computing the
    /// differential `P − Q` internally.
    ///
    /// Each named field holds its semantic content:
    /// - `P` — first basis point
    /// - `PmQ` — `x(P − Q)`, the differential
    /// - `Q` — second basis point
    fn from((P, Q): (ProjectiveXOnlyPoint, ProjectiveXOnlyPoint)) -> TorsionBasis {
        let PmQ = P.projective_difference(&Q);
        TorsionBasis { P, PmQ, Q }
    }
}

impl TorsionBasis {
    /// Construct a basis from pre-propagated components.
    ///
    /// Arguments correspond directly to fields:
    /// - `P`: first basis point.
    /// - `PmQ`: `x(P − Q)`, propagated from a prior computation.
    /// - `Q`: second basis point (above `(0, 0)` for non-`E₀` curves — see the
    ///   [`TorsionBasis`] struct doc on the spec permutation).
    ///
    /// # Provenance of `PmQ`
    ///
    /// `PmQ` must have been obtained from one of:
    /// - a precomputed constant (e.g., `BASIS_E0_PMQ_X`)
    /// - propagation through a group homomorphism alongside `P` and `Q` (scalar
    ///   multiplication, isogeny evaluation, doubling)
    /// - rearranging an existing `TorsionBasis`'s fields
    ///
    /// Do NOT pass a point computed by a separate biladder call or
    /// any other independent computation — even if it has the
    /// correct affine x-coordinate, its projective representative
    /// will be inconsistent, causing `lift` to recover the wrong
    /// Jacobian y-sign. Use `From<(P, Q)>` instead.
    pub fn from_propagated(
        P: ProjectiveXOnlyPoint,
        PmQ: ProjectiveXOnlyPoint,
        Q: ProjectiveXOnlyPoint,
    ) -> TorsionBasis {
        TorsionBasis { P, PmQ, Q }
    }

    /// Lift this x-only basis to Jacobian coordinates on the
    /// given curve.
    ///
    /// Normalizes `P` internally and uses the Okeya-Sakurai
    /// algorithm to recover `Q`'s y-coordinate from `P`'s
    /// y-coordinate and the difference point `PmQ` (= `x(P − Q)`).
    ///
    /// Returns `None` if y-recovery fails (point not on curve).
    ///
    /// Corresponds to `lift_basis_normalized` in the C reference
    /// (`basis.c:79`).
    #[must_use]
    pub fn lift(&self, curve: &Curve) -> Option<(JacobianPoint, JacobianPoint)> {
        let A = *curve.coefficient().as_fp2();

        // Normalize P: compute affine x_P = X_P / Z_P.
        let z_inv = self.P.Z.invert();
        let x_p = &self.P.X * &z_inv;

        // Recover y_P via Curve::recover_y.
        let y_p = curve.recover_y(&AffineX::from(x_p))?;

        let p_jac = JacobianPoint::new(x_p, y_p, Fp2::ONE, curve);

        // Okeya-Sakurai: recover y of the "second basis point" from
        // x_P, y_P, the PmQ field, and the Q field.
        // C reference: basis.c:91-116.
        let v1 = &x_p * &self.PmQ.Z;
        let v2 = &self.PmQ.X + &v1;
        let v3 = {
            let diff = &self.PmQ.X - &v1;
            let diff_sq = diff.square();
            &diff_sq * &self.Q.X
        };
        let two_a = &A + &A;
        let v1_new = &two_a * &self.PmQ.Z;
        let v2 = &v2 + &v1_new;
        let v4 = &(&x_p * &self.PmQ.X) + &self.PmQ.Z;
        let v2 = &v2 * &v4;
        let v1_new = &v1_new * &self.PmQ.Z;
        let v2 = &v2 - &v1_new;
        let v2 = &v2 * &self.Q.Z;
        let y_s_num = &v3 - &v2;
        let two_yp = &y_p + &y_p;
        let v1 = &(&two_yp * &self.PmQ.Z) * &self.Q.Z;

        // S in Jacobian: (X_S·v1·Z_S : y_s_num·(Z_S·v1)² : Z_S·v1)
        let x_s_tmp = &self.PmQ.X * &v1;
        let z_s_jac = &self.PmQ.Z * &v1;
        let z_s_jac_sq = z_s_jac.square();
        let y_s_jac = &y_s_num * &z_s_jac_sq;
        let x_s_jac = &x_s_tmp * &z_s_jac;

        let s_jac = JacobianPoint::new(x_s_jac, y_s_jac, z_s_jac, curve);

        Some((p_jac, s_jac))
    }

    /// Convert kernel scalars on E₀\[2^f\] to the corresponding
    /// left O₀-ideal.
    ///
    /// Given scalars (c₁, c₂) such that the kernel generator is
    /// \[c₁\]P₀ + \[c₂\]Q₀ on E₀\[2^f\] (the canonical torsion
    /// basis), computes I = O₀⟨α, 2^f⟩ where
    /// α = a + b·(j + (1+k)/2) − i.
    ///
    /// Uses the precomputed E₀ action matrices (M_i, M_j, M_{gen4})
    /// internally. Only valid for the NIST-I starting curve E₀ and
    /// its canonical basis.
    ///
    /// Implements [KernelToIdeal][Alg. 3.17].
    ///
    /// [Alg. 3.17]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.17
    // TODO: c1/c2 are BigInt<4> (signed) but semantically unsigned
    // scalars mod 2^f. Investigating types beyond BigInt — we need
    // unsigned modular arithmetic mod 2^k on Scalar, with add, sub,
    // mul, and invert_mod. For now, convert at boundaries.
    pub fn kernel_to_ideal(
        c1: &BigInt<4>,
        c2: &BigInt<4>,
        f: TorsionExponent,
    ) -> Option<LeftIdeal<4>> {
        // Action matrices for E₀: [i, j, k, gen2, gen3, gen4].
        let m_i = &ENDOMORPHISM_MATRICES[0][0];
        let m_j = &ENDOMORPHISM_MATRICES[0][1];
        let m_gen4 = &ENDOMORPHISM_MATRICES[0][5];

        let modulus = BigInt::<4>::ONE << f.value();

        // Step 1: [d1, d2]^T = M_θ · [c1, c2]^T mod 2^f.
        // θ = j + (1+k)/2, so M_θ = M_j + M_gen4.
        let (jc1, jc2) = m_j.eval_mod(c1, c2, f.value());
        let (gc1, gc2) = m_gen4.eval_mod(c1, c2, f.value());
        let d1 = jc1.ct_add(&gc1).ct_mod(&modulus);
        let d2 = jc2.ct_add(&gc2).ct_mod(&modulus);

        // Step 2–3: [a, b]^T = M^{-1} · M_i · [c1, c2]^T mod 2^f.
        let (e1, e2) = m_i.eval_mod(c1, c2, f.value());
        let det = c1.ct_mul(&d2).ct_sub(&d1.ct_mul(c2)).ct_mod(&modulus);
        let det_inv = det.invert_mod(&modulus)?;
        let a = det_inv
            .ct_mul(&d2.ct_mul(&e1).ct_sub(&d1.ct_mul(&e2)))
            .ct_mod(&modulus);
        let b = det_inv
            .ct_mul(&c1.ct_mul(&e2).ct_sub(&c2.ct_mul(&e1)))
            .ct_mod(&modulus);

        // Step 4: α = a + b·(j + (1+k)/2) − i.
        // In {1, i, j, k} with denom 2: (2a+b, −2, 2b, b)/2.
        // Matches C ref `id2iso_kernel_dlogs_to_ideal_even` (id2iso.c:247-254).
        let two_a = a.ct_add(&a);
        let two_b = b.ct_add(&b);
        let alpha = Element {
            a: Coordinate::from(two_a.ct_add(&b)),
            b: Coordinate::from(-2i64),
            c: Coordinate::from(two_b),
            d: Coordinate::from(b),
            denom: Denominator::TWO,
        };

        Some(LeftIdeal::new(&alpha, &modulus, EXTREMAL_ORDERS[0].order()))
    }

    /// Compute `P + [m]·(P − Q)` via the three-point Montgomery ladder.
    ///
    /// The scalar `m` is a [`Scalar`] (256-bit unsigned integer in four
    /// u64 limbs); the ladder always processes exactly 256 bits, in
    /// little-endian bit order from LSB to MSB. Constant-time in the
    /// scalar value.
    ///
    /// # What this computes (not `P + [m]Q`)
    ///
    /// The 3-pt Montgomery ladder takes a basis triple
    /// `(start, increment, increment − start)` and computes
    /// `start + [m]·increment`. We pass `(P, PmQ, Q)` from this struct
    /// (which under our layout means start = `P`,
    /// increment = `x(P − Q)`, difference = `x(Q)`), so the result is
    /// `P + [m]·(P − Q)`, NOT `P + [m]·Q`.
    ///
    /// This is the spec's permuted convention (§2.2.3): the kernel
    /// generator for the challenge isogeny is encoded as a basis
    /// transformation `(1, chl)` applied to `(P, P − Q)`, and the
    /// downstream isogeny chain compensates for the permutation when
    /// it consumes the kernel. Signing and verifying must agree on
    /// the convention; both ends are correctly aligned to this one.
    ///
    /// Implements `Ladder3pt` ([§8.2], [Alg. 8.7][Alg. 8.7]).
    ///
    /// [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
    /// [Alg. 8.7]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.7
    pub fn scalar_mul_add(&self, m: &Scalar) -> ProjectiveXOnlyPoint {
        // Three-point Montgomery ladder. Ladder seeds (matching
        // C ref's `ec_ladder3pt` convention):
        //   x0 = ladder "increment"  ← self.PmQ (= our P − Q)
        //   x1 = ladder "start"      ← self.P
        //   x2 = ladder "P − Q"      ← self.Q (the difference of x1 and x0
        //                                      under our field naming:
        //                                      P − (P − Q) = Q)
        //
        // After 256 iterations the result is x1 = P + [m]·(P − Q), i.e.,
        // the ladder accumulates [m] times the PmQ field into P.
        //
        // Processes 4 limbs × 64 bits = 256 bits from LSB to MSB.
        let mut x0 = self.PmQ;
        let mut x1 = self.P;
        let mut x2 = self.Q;

        for limb in m.as_limbs() {
            for bit_pos in 0..64u32 {
                let bit = ((limb >> bit_pos) & 1) as u8;
                // C ref: cswap when bit == 0
                let mask = subtle::Choice::from(bit ^ 1);
                ProjectiveXOnlyPoint::conditional_swap(&mut x1, &mut x2, mask);
                differential_add_and_double(&mut x0, &mut x1, &x2);
                ProjectiveXOnlyPoint::conditional_swap(&mut x1, &mut x2, mask);
            }
        }
        x1
    }

    /// Evaluate a [`KernelDecomposition`][crate::deuring::KernelDecomposition]
    /// against this basis: computes `[a]·P + [b]·PmQ` via the biscalar
    /// Montgomery ladder.
    ///
    /// The scalars come from the kernel decomposition produced by the
    /// Deuring correspondence. The result is `[a]·P + [b]·(P − Q)`
    /// under our spec-permuted layout — see [`scalar_mul_add`] for
    /// why this is `(P − Q)` rather than `Q`.
    ///
    /// [`scalar_mul_add`]: Self::scalar_mul_add
    pub fn eval_decomposition(&self, a: &Scalar, b: &Scalar) -> ProjectiveXOnlyPoint {
        self.biscalar_mul(a, b, TorsionExponent::FULL)
    }

    /// Compute `[m]·P + [n]·(P − Q)` from this basis.
    ///
    /// **Spec-permuted convention.** This function multiplies the
    /// *differential* `P − Q` (the [`PmQ`](Self::PmQ) field), not the
    /// second basis point `Q`. C-ref's `ec_biscalar_mul` operates on
    /// the same convention positionally (see `basis.c:422-425`:
    /// C-ref's `B.Q` slot stores `P − Q`). Selkie's [`ENDOMORPHISM_MATRICES`]
    /// table is byte-imported from C-ref and is therefore encoded
    /// against this convention — for an endomorphism `θ` with matrix
    /// `M`, applying `biscalar_mul(M[0][0], M[1][0])` yields
    /// `θ(P)`'s spec-permuted decomposition, which is what every
    /// downstream consumer of `ENDOMORPHISM_MATRICES` expects.
    ///
    /// Both scalars are [`Scalar`]s reduced mod 2^e, where `e` is the
    /// torsion exponent of the basis. Constant-time in the scalar values.
    ///
    /// Implements [LadderBiscalar][Alg. 8.8] ([Alg. 8.8][Alg. 8.8]).
    ///
    /// [`ENDOMORPHISM_MATRICES`]: crate::deuring::precomputed::ENDOMORPHISM_MATRICES
    /// [Alg. 8.8]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.8
    pub fn biscalar_mul(&self, m: &Scalar, n: &Scalar, e: TorsionExponent) -> ProjectiveXOnlyPoint {
        let kbits = e.value() as usize;
        // Local-name aliases for the ladder roles, NOT the field semantics:
        //  - `P`   = first basis point P (the [m]-multiplied operand).
        //  - `Q`   = the [n]-multiplied operand. Here that is `P − Q`, because this
        //    function computes `[m]·P + [n]·(P − Q)`.
        //  - `PmQ` = the precomputed `x(P − Q_local)`-differential the ladder needs for
        //    differential addition. With `Q_local = P − Q`, that differential is
        //    `x(Q)`.
        let P = &self.P;
        let Q = &self.PmQ;
        let PmQ = &self.Q;
        let curve = P.curve();

        // Convert to bytes for the recoding stage.
        let m_bytes = m.to_le_bytes();
        let n_bytes = n.to_le_bytes();

        // Recoding stage.
        // Determine sigma based on parity of m and n.
        let bit_m0 = m_bytes[0] & 1;
        let bit_n0 = n_bytes[0] & 1;
        let mask_m: u8 = 0u8.wrapping_sub(bit_m0);
        let mask_n: u8 = 0u8.wrapping_sub(bit_n0);

        // sigma = (0,1) if both same parity, else the even one gets sigma=1
        let evens = (bit_m0 ^ 1) + (bit_n0 ^ 1);
        let m_evens: u8 = 0u8.wrapping_sub(evens & 1);
        let mut sigma0: u8 = (bit_m0 ^ 1) & m_evens;
        let mut sigma1: u8 = ((bit_n0 ^ 1) & m_evens) | (1 & !m_evens);

        // Convert even scalars to odd (subtract 1).
        let mut m_t = [0u8; 32];
        let mut n_t = [0u8; 32];
        m_t.copy_from_slice(&m_bytes);
        n_t.copy_from_slice(&n_bytes);

        // Subtract 1 from even scalars (constant-time).
        sub_one_ct(&mut m_t, mask_m ^ 0xFF); // subtract if m was even
        sub_one_ct(&mut n_t, mask_n ^ 0xFF); // subtract if n was even

        // Compute recoding bits r[2i] and r[2i+1].
        let mut r = vec![0u8; 2 * kbits];
        let mut pre_sigma = 0u8;
        for i in 0..kbits {
            // Swap m_t and n_t if sigma changed.
            let swap_mask = 0u8.wrapping_sub(sigma0 ^ pre_sigma);
            swap_bytes_ct(&mut m_t, &mut n_t, swap_mask);

            let bs1_ip1: u8;
            let bs2_ip1: u8;
            if i == kbits - 1 {
                bs1_ip1 = 0;
                bs2_ip1 = 0;
            } else {
                bs1_ip1 = shr1_ct(&mut m_t);
                bs2_ip1 = shr1_ct(&mut n_t);
            }
            let bs1_i = m_t[0] & 1;
            let bs2_i = n_t[0] & 1;

            r[2 * i] = bs1_i ^ bs1_ip1;
            r[2 * i + 1] = bs2_i ^ bs2_ip1;

            // Update sigma if r[2i+1] = 1.
            pre_sigma = sigma0;
            let flip = 0u8.wrapping_sub(r[2 * i + 1]);
            let tmp = (sigma0 & !flip) | (sigma1 & flip);
            sigma1 = (sigma1 & !flip) | (sigma0 & flip);
            sigma0 = tmp;
        }

        // Evaluation stage.
        let mut R0 = ProjectiveXOnlyPoint::identity(curve);
        let sigma0_choice = subtle::Choice::from(sigma0 & 1);
        let mut R1 = ProjectiveXOnlyPoint::conditional_select(P, Q, sigma0_choice);
        let mut R2 = ProjectiveXOnlyPoint::conditional_select(Q, P, sigma0_choice);

        let mut D1 = R1;
        let mut D2 = R2;

        // R2 ← xADD(R1, R2, P−Q)
        R2 = R1.differential_add(&R2, PmQ);

        let mut F1 = R2;
        let mut F2 = *PmQ;

        // The C reference's biladder (`xDBLMUL`, `ec.c:485`) branches on
        // `A == 0` and uses the specialized `xDBL_E0` formula when `A = 0`,
        // but the general `xDBL_A24` formula otherwise. Mirror that here so
        // the biladder output's projective `(X : Z)` representative matches
        // the C reference byte-for-byte: `xDBL_E0` produces `2 · xDBL_A24`,
        // and that scaling is what propagates through the biladder's
        // doubling chain into the output rep.
        let a_is_zero = *curve.coefficient().as_fp2() == Fp2::ZERO;

        // Main loop: process bits from MSB to LSB.
        for i in (0..kbits).rev() {
            let h = r[2 * i] + r[2 * i + 1]; // h ∈ {0, 1, 2}

            // T0 ← R_{⌊h/2⌋}, then double it.
            let h_bit0 = subtle::Choice::from(h & 1);
            let h_bit1 = subtle::Choice::from((h >> 1) & 1);
            let mut T0 = ProjectiveXOnlyPoint::conditional_select(&R0, &R1, h_bit0);
            T0 = ProjectiveXOnlyPoint::conditional_select(&T0, &R2, h_bit1);
            T0 = if a_is_zero {
                T0.double_e0()
            } else {
                T0.double()
            };

            // T1 and T2 depend on r[2i+1].
            let r_bit = subtle::Choice::from(r[2 * i + 1] & 1);
            let T1_a = ProjectiveXOnlyPoint::conditional_select(&R0, &R1, r_bit);
            let T1_b = ProjectiveXOnlyPoint::conditional_select(&R1, &R2, r_bit);

            // Swap DIFF1a/DIFF1b based on r[2i+1].
            ProjectiveXOnlyPoint::conditional_swap(&mut D1, &mut D2, r_bit);
            let T1 = T1_a.differential_add(&T1_b, &D1);
            let T2 = R0.differential_add(&R2, &F1);

            // Swap DIFF2a/DIFF2b if h is odd.
            ProjectiveXOnlyPoint::conditional_swap(&mut F1, &mut F2, h_bit0);

            R0 = T0;
            R1 = T1;
            R2 = T2;
        }

        // Output: select based on parity of original scalars.
        let mut result =
            ProjectiveXOnlyPoint::conditional_select(&R0, &R1, subtle::Choice::from(m_evens & 1));
        let both_odd = subtle::Choice::from(bit_m0 & bit_n0);
        result = ProjectiveXOnlyPoint::conditional_select(&result, &R2, both_odd);

        result
    }

    /// Deterministically generate a torsion basis for E_A\[2^e\] from a
    /// curve and a hint, where e = [`TORSION_EVEN_POWER`].
    ///
    /// The hint encodes (h_A, h) where h_A indicates the quadratic
    /// residuosity of A, and h is used to quickly find a valid
    /// x-coordinate for the first basis point.
    ///
    /// Implements `TorsionBasisFromHint` ([§2.2.3], Algorithm 2.2).
    ///
    /// [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.2.3
    /// [`TORSION_EVEN_POWER`]: crate::params::TORSION_EVEN_POWER
    pub fn from_hint(curve: &Curve, hint: BasisHint) -> Option<TorsionBasis> {
        let _e = TORSION_EVEN_POWER;
        // Normalize the curve's A24/C24 constants so the Montgomery
        // ladder produces the same projective representative as the
        // C reference (which calls ec_normalize_curve_and_A24 here).
        let mut curve = *curve;
        curve.normalize();
        let curve = &curve;
        let A = *curve.coefficient().as_fp2();

        // Special case: A = 0 (the starting curve E₀).
        // Use precomputed basis points and compute the difference.
        if A == Fp2::ZERO {
            let P = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, curve);
            let Q = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_Q_X, curve);
            let PmQ = P.projective_difference(&Q);
            return Some(TorsionBasis { P, PmQ, Q });
        }

        let h_A = hint.h_A();
        let h = hint.h();

        // Compute x(P) from the hint. The `h == 0` branch performs a
        // bounded search; on an adversarial curve where no `n` in the
        // search window produces a valid x-coordinate, the search
        // returns `None` and we propagate that as a `from_hint`
        // failure rather than spinning forever (fuzz crash
        // `find_na_x_coord (crash-f13d2a...)` covered by the
        // wycheproof verify suite at `tcId = 49`).
        let x_P = if h == 0 {
            // Rare fallback: hint didn't fit in 7 bits.
            // Must search from scratch (starting at 128).
            if h_A == 0 {
                // A is NQR: search for n*A on the curve.
                find_na_x_coord(&A, curve, 128)?
            } else {
                // A is QR: search for -A/(1+i*b) on the curve.
                find_nqr_factor(&A, curve, 128)?
            }
        } else if h_A == 0 {
            // A is NQR: x(P) = h * A
            &A * &Fp2::from_fp(Fp::from_small(u32::from(h)))
        } else {
            // A is QR: x(P) = -A / (1 + i*h)
            let z = Fp2::new(Fp::ONE, Fp::from_small(u32::from(h)));
            &(-&A) * &z.invert()
        };

        let x_Q = -&(&A + &x_P); // x(Q) = -x(P) - A

        let mut P = ProjectiveXOnlyPoint::from_affine_x(x_P, curve);
        let mut Q = ProjectiveXOnlyPoint::from_affine_x(x_Q, curve);

        #[cfg(test)]
        if std::env::var("SELKIE_DUMP_FROM_HINT").is_ok() {
            let fmt = |x: &Fp2| {
                let bytes = x.to_bytes();
                let mut a = bytes[..32].to_vec();
                a.reverse();
                let mut b = bytes[32..].to_vec();
                b.reverse();
                format!("0x{} + i*0x{}", hex::encode(a), hex::encode(b))
            };
            eprintln!("[FROM_HINT_SELKIE] hint_A={h_A} hint_P={h}");
            eprintln!("[FROM_HINT_SELKIE] curve.A_aff = {}", fmt(&A));
            eprintln!("[FROM_HINT_SELKIE] x_P (pre-cofactor) = {}", fmt(&x_P));
            eprintln!("[FROM_HINT_SELKIE] x_Q (pre-cofactor) = {}", fmt(&x_Q));
        }

        // Clear odd cofactor to get points of order 2^e.
        // Multiply by (p+1)/2^e = cofactor.
        P = P.clear_cofactor();
        Q = Q.clear_cofactor();

        #[cfg(test)]
        if std::env::var("SELKIE_DUMP_FROM_HINT").is_ok() {
            let fmt = |x: &Fp2| {
                let bytes = x.to_bytes();
                let mut a = bytes[..32].to_vec();
                a.reverse();
                let mut b = bytes[32..].to_vec();
                b.reverse();
                format!("0x{} + i*0x{}", hex::encode(a), hex::encode(b))
            };
            eprintln!(
                "[FROM_HINT_SELKIE] P.x (post-cofactor) = {}",
                fmt(P.to_affine_x().as_fp2())
            );
            eprintln!(
                "[FROM_HINT_SELKIE] Q.x (post-cofactor) = {}",
                fmt(Q.to_affine_x().as_fp2())
            );
        }

        let PmQ = P.projective_difference(&Q);

        // The spec (§2.2.3) constructs Q so that Q lies above the
        // 2-torsion point at `(0, 0)`, and stores the triple in the
        // permuted layout `(x_P, x_{P−Q}, x_Q)`. C ref's
        // `ec_curve_to_basis_2f_from_hint` (`basis.c:403-406`)
        // implements the same permutation:
        //
        //   PQ2->P   = P                          // x_P
        //   PQ2->Q   = difference_point(P, Q)     // x_{P − Q}
        //   PQ2->PmQ = Q                          // x_Q (above (0, 0))
        //
        // Selkie's struct field names match this layout directly:
        // `P` ↔ `B.P`, `PmQ` ↔ `B.Q`, `Q` ↔ `B.PmQ`. With these
        // assignments, `scalar_mul_add` (= ladder3pt) computes
        // `P + [m]·(P − Q)`, NOT `P + [m]·Q` — the spec relies on
        // this to encode the challenge kernel; see [§2.2.3] and
        // [`scalar_mul_add`](Self::scalar_mul_add).
        //
        // [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.2.3
        Some(TorsionBasis { P, PmQ, Q })
    }

    /// Generates a torsion basis for E_A\[2^e\] and its associated hint,
    /// where e = [`TORSION_EVEN_POWER`].
    ///
    /// Implements `TorsionBasisToHint` ([§2.2.3], Algorithm 2.1).
    ///
    /// Returns `None` when the bounded x-coordinate search exhausts
    /// without finding a valid representative — mirroring
    /// [`Self::from_hint`]'s `None` on adversarial curves. Honest sign
    /// and keygen flows hit a valid `n` within the first 7 bits;
    /// callers funnel the `None` into their retry loop alongside the
    /// other isogeny-failure cases.
    ///
    /// [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.2.3
    /// [`TORSION_EVEN_POWER`]: crate::params::TORSION_EVEN_POWER
    pub fn to_hint(curve: &Curve) -> Option<(TorsionBasis, BasisHint)> {
        let _e = TORSION_EVEN_POWER;
        // Normalize the curve's A24/C24 constants so the Montgomery
        // ladder produces the same projective representative as
        // `from_hint` (which also calls `curve.normalize()`). Without
        // this, `clear_cofactor`'s doubling formulae use unnormalized
        // (A24:C24) and produce a projectively different (X:Z) for
        // P and Q than `from_hint` recomputes — same affine x, but
        // different (X:Z). `projective_difference` (which contains a
        // square root) is sensitive to the projective representation
        // and picks a different sqrt branch on each side, so the
        // recovered P − Q is a *different abstract point* in sign vs
        // verify. That breaks the verify-side chain kernel and
        // surfaces as the (2,2)-chain `splitting=0` rejection on
        // every signature.
        let mut curve = *curve;
        curve.normalize();
        let curve = &curve;
        let A = *curve.coefficient().as_fp2();

        if A == Fp2::ZERO {
            // E₀ has no hint — the basis is precomputed.
            let P = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, curve);
            let Q = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_Q_X, curve);
            let PmQ = P.projective_difference(&Q);
            let basis = TorsionBasis { P, PmQ, Q };
            return Some((basis, BasisHint::from_byte(0)));
        }

        let h_A = bool::from(A.is_square());

        let (x_P, h) = if !h_A {
            // A is NQR: find n such that n*A is on the curve.
            find_na_x_coord_with_hint(&A, curve)?
        } else {
            // A is QR: find b such that -A/(1+i*b) is on the curve.
            find_nqr_factor_with_hint(&A, curve)?
        };

        let x_Q = -&(&A + &x_P);

        let mut P = ProjectiveXOnlyPoint::from_affine_x(x_P, curve);
        let mut Q = ProjectiveXOnlyPoint::from_affine_x(x_Q, curve);

        P = P.clear_cofactor();
        Q = Q.clear_cofactor();

        let PmQ = P.projective_difference(&Q);

        let basis = TorsionBasis { P, PmQ, Q };

        let hint_byte = BasisHint::new(h_A as u8, h);
        Some((basis, hint_byte))
    }
}

/// A 2×2 change-of-basis matrix over Z/2^e Z.
///
/// Encodes the relationship `Q₁ = [x₁]P₁ + [x₂]P₂`,
/// `Q₂ = [x₃]P₁ + [x₄]P₂`. Entries are reduced mod 2^e where `e`
/// is the torsion exponent.
///
/// # Storage
///
/// **Column-major**, matching the C reference:
/// - `entries[0][0] = x₁`, `entries[1][0] = x₂` (column 0 = coeffs of Q₁)
/// - `entries[0][1] = x₃`, `entries[1][1] = x₄` (column 1 = coeffs of Q₂)
///
/// `entries[i][j]` is the coefficient of source basis element `i` in
/// target basis element `j`. The wire format writes the four entries
/// in row-major byte order — `[0][0], [0][1], [1][0], [1][1]` — which
/// is what both this implementation and the C reference produce.
///
/// No `PartialEq`/`Eq`: M_sk in the signing key is secret.
#[derive(Copy, Clone, Debug)]
pub struct ChangeOfBasisMatrix {
    /// Column-major matrix entries.
    ///
    /// `entries[i][j]` = coefficient of source basis element `i`
    /// in target basis element `j`. Column 0 yields target.P,
    /// column 1 yields target.Q when applied via `mul`.
    pub entries: [[Scalar; 2]; 2],
    /// Torsion exponent: entries are reduced mod 2^e.
    pub e: TorsionExponent,
}

impl ChangeOfBasisMatrix {
    /// `true` iff every entry is strictly less than `2^k`.
    ///
    /// `Signature::from_bytes` uses this to enforce the spec bound on
    /// M_chl entries: each must be a positive integer
    /// `< 2^(e'_rsp + r_rsp + 2)` ([§4.5][§4.5], Algorithm 4.9
    /// step 5). Variable-time on the entries — fine since matrix
    /// entries are public.
    ///
    /// [§4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
    pub fn entries_below_pow2(&self, k: u32) -> bool {
        self.entries
            .iter()
            .flatten()
            .all(|entry| entry.bit_length() <= k)
    }

    /// Compute the change-of-basis matrix expressing `reduced` (a basis
    /// of E[2^e]) in terms of `canonical` (a basis at the curve's full
    /// 2^TORSION_EVEN_POWER torsion).
    ///
    /// The relationship encoded is:
    /// ```text
    ///   reduced.P = [r1]canonical.P + [r2]canonical.Q
    ///   reduced.Q = [s1]canonical.P + [s2]canonical.Q
    /// ```
    /// where `(r1, r2, s1, s2)` are the matrix entries reduced mod `2^e`.
    ///
    /// # Storage
    ///
    /// Entries are stored **column-major**, matching the C reference:
    ///   `entries[i][j]` = coefficient of `canonical.basis[i]` in
    ///   `reduced.basis[j]`.
    ///
    /// Concretely:
    /// - `entries[0][0] = r1`, `entries[1][0] = r2` (column 0 = coeffs of
    ///   reduced.P)
    /// - `entries[0][1] = s1`, `entries[1][1] = s2` (column 1 = coeffs of
    ///   reduced.Q)
    ///
    /// `mul` consumes this layout by applying columns: column 0 yields
    /// the new first basis element, column 1 yields the new second.
    ///
    /// Implements [ChangeOfBasis][Alg. 2.5] ([Alg. 2.5][Alg. 2.5]) via
    /// the C reference's `_change_of_basis_matrix_tate` structure
    /// (`id2iso/ref/lvlx/id2iso.c:330`), which delegates the five
    /// cross-pairings to [`TorsionBasis::cross_pairings`].
    ///
    /// `canonical` must be at full 2^TORSION_EVEN_POWER torsion;
    /// `reduced` at order 2^e. Pre-reducing both bases to 2^e produces
    /// a non-primitive `ζ` and collapses the matrix to a sub-precision
    /// rank-1 form (regression test
    /// `tate_pairing_primitive_on_reduced_basis`).
    ///
    /// Returns `None` if either basis fails to lift to Jacobian
    /// coordinates (e.g. a recomputed `P − Q` whose sqrt branch is
    /// inconsistent with `P` and `Q`'s y-coordinates).
    ///
    /// # X-only sign ambiguity
    ///
    /// On x-only inputs the dlogs `(r1, r2, s1, s2)` are only
    /// determined up to a per-column sign: the matrices `M` and
    /// `−M` produce x-only-equivalent images. The sqrt branch of
    /// `lift` resolves this implicitly. Chain consumers go through
    /// [`Self::mul`], which is itself x-only, so the resulting
    /// images agree byte-for-byte with C-ref. Code that inspects
    /// `entries` directly must not assume a textbook sign.
    ///
    /// [Alg. 2.5]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.2.5
    pub fn from_bases(
        canonical: &TorsionBasis,
        reduced: &TorsionBasis,
        e: TorsionExponent,
    ) -> Option<Self> {
        let ws = canonical.cross_pairings(reduced, e)?;

        // Per the C reference (`tate_dlog_partial`, biextension.c:720):
        //   r2 = dlog_w0(w[1]),  r1 = dlog_w0(w[2]),
        //   s2 = dlog_w0(w[3]),  s1 = dlog_w0(w[4]).
        // The X/Z swap on w[2] and w[4] folds in the inversion of
        // t(R, Q) and t(S, Q) the matrix-entry derivation requires.
        let r2 = ws[0].dlog(&ws[1], e);
        let r1 = ws[0].dlog(&ws[2], e);
        let s2 = ws[0].dlog(&ws[3], e);
        let s1 = ws[0].dlog(&ws[4], e);

        Some(Self {
            entries: [[r1, s1], [r2, s2]],
            e,
        })
    }

    /// Compute the change-of-basis matrix expressing `canonical` (a
    /// basis at full 2^TORSION_EVEN_POWER torsion) in terms of
    /// `reduced` (a basis of E[2^e]) — the inverse direction of
    /// [`from_bases`].
    ///
    /// Mirrors the C reference's `change_of_basis_matrix_tate_invert`
    /// (`id2iso/ref/lvlx/id2iso.c:392`): it computes
    /// [`from_bases(canonical, reduced, e)`] (which gives "coords of
    /// reduced in canonical") and inverts the resulting 2×2 matrix
    /// modulo 2^e.
    ///
    /// Returns `None` if either basis fails to lift, or if the
    /// underlying matrix is not invertible mod 2^e (det even, i.e.,
    /// the input bases don't span E[2^e]).
    ///
    /// [`from_bases`]: ChangeOfBasisMatrix::from_bases
    pub fn from_bases_invert(
        canonical: &TorsionBasis,
        reduced: &TorsionBasis,
        e: TorsionExponent,
    ) -> Option<Self> {
        let forward = Self::from_bases(canonical, reduced, e)?;
        let k = e.value();
        let a = &forward.entries[0][0];
        let b = &forward.entries[0][1];
        let c = &forward.entries[1][0];
        let d = &forward.entries[1][1];

        // det = ad − bc (mod 2^e). For a basis pair, det must be odd.
        let ad = a.mul_mod2k(d, k);
        let bc = b.mul_mod2k(c, k);
        let det = ad.sub_mod2k(&bc, k);
        let det_inv = det.inv_mod2k(k)?;

        // Adjugate / det. The signs flip via 0 − x mod 2^k.
        let zero = Scalar::ZERO;
        let neg_b = zero.sub_mod2k(b, k);
        let neg_c = zero.sub_mod2k(c, k);
        let inv_a = d.mul_mod2k(&det_inv, k);
        let inv_b = neg_b.mul_mod2k(&det_inv, k);
        let inv_c = neg_c.mul_mod2k(&det_inv, k);
        let inv_d = a.mul_mod2k(&det_inv, k);

        Some(Self {
            entries: [[inv_a, inv_b], [inv_c, inv_d]],
            e,
        })
    }

    /// Multiply this matrix by a [`TorsionBasis`]: `(P', Q') = M · (P, Q)`.
    ///
    /// Applies the matrix by **columns** (matching the C reference):
    /// - `R' = [a]P + [c]Q` where `a = M[0][0]`, `c = M[1][0]` (column 0)
    /// - `S' = [b]P + [d]Q` where `b = M[0][1]`, `d = M[1][1]` (column 1)
    /// - `R'-S'` via a third biscalar call (avoids sqrt branch instability)
    ///
    /// # Divergence from spec
    ///
    /// The C reference applies by columns, not rows. See the comment
    /// on `ChallengeMatrix` in `keys/mod.rs`.
    pub fn mul(&self, basis: &TorsionBasis) -> TorsionBasis {
        let a = &self.entries[0][0];
        let b = &self.entries[0][1];
        let c = &self.entries[1][0];
        let d = &self.entries[1][1];

        let p_prime = basis.biscalar_mul(a, c, self.e);
        let q_prime = basis.biscalar_mul(b, d, self.e);

        // R'-S' = [(a-b)]P + [(c-d)]Q
        let k = self.e.value();
        let a_minus_b = a.sub_mod2k(b, k);
        let c_minus_d = c.sub_mod2k(d, k);
        let pmq_prime = basis.biscalar_mul(&a_minus_b, &c_minus_d, self.e);

        TorsionBasis::from_propagated(p_prime, q_prime, pmq_prime)
    }

    // SetChangeOfBasisMatrix (Algorithm 4.8) will be inlined into
    // SigningKey::sign() since it's only called there to assemble
    // the signature's M_chl, hint_aux, and hint_chl.
}

/// Subtract 1 from a little-endian byte array, conditionally.
/// `mask` is 0xff to subtract, 0x00 to skip.
fn sub_one_ct(a: &mut [u8], mask: u8) {
    let mut borrow: u16 = (mask & 1) as u16;
    for byte in a.iter_mut() {
        let diff = (*byte as u16).wrapping_sub(borrow);
        *byte = diff as u8;
        borrow = (diff >> 8) & 1;
    }
}

/// Shift a little-endian byte array right by 1. Returns the shifted-out LSB.
fn shr1_ct(a: &mut [u8]) -> u8 {
    let lsb = a[0] & 1;
    let len = a.len();
    for i in 0..len - 1 {
        a[i] = (a[i] >> 1) | (a[i + 1] << 7);
    }
    a[len - 1] >>= 1;
    lsb
}

/// Conditionally swap two byte arrays. `mask` is 0xff to swap, 0x00 to skip.
fn swap_bytes_ct(a: &mut [u8], b: &mut [u8], mask: u8) {
    for (ai, bi) in a.iter_mut().zip(b.iter_mut()) {
        let diff = (*ai ^ *bi) & mask;
        *ai ^= diff;
        *bi ^= diff;
    }
}

/// Check if x³ + Ax² + x is a square in F_{p²} (i.e., (x, ·) is on E_A).
fn is_on_curve(x: &Fp2, A: &Fp2) -> bool {
    let t = &(x + A) * x; // x² + Ax
    let t = &(&t + &Fp2::ONE) * x; // x³ + Ax² + x
    bool::from(t.is_square())
}

/// Maximum number of `n` values to try in the fallback x-coordinate
/// searches.
///
/// Honest signing satisfies the predicate inside the first ~50
/// tries (max 37 across a 100k-sample sweep over random Fp²-valued
/// A; extremal-order curves are similar). The bound caps the
/// verify-side slow path: an attacker-supplied curve with `A ∈ Fp*`
/// and `A⁴ - 4A² + 2 = 0` (e.g. `A = √2 mod p`) makes the predicate
/// structurally unsatisfiable — `V_n = (1+n²)² + n²·A²·(A²-4)`
/// becomes a perfect square in Fp for every n, while `1+n²` toggles
/// squareness independently, and the search exhausts no matter how
/// many tries it gets.
const FIND_X_COORD_MAX_TRIES: u32 = 1 << 10;

/// Find `n` such that `n*A` is a valid x-coordinate on E_A. Returns
/// `Some(x(P))` on success or `None` if no `n` in
/// `[start, start + FIND_X_COORD_MAX_TRIES)` satisfies the predicate
/// (which only happens for adversarial / fuzz-crafted curves).
fn find_na_x_coord(A: &Fp2, _curve: &Curve, start: u8) -> Option<Fp2> {
    let mut x = &Fp2::from_fp(Fp::from_small(u32::from(start))) * A;
    for _ in 0..FIND_X_COORD_MAX_TRIES {
        if is_on_curve(&x, A) && !bool::from(x.is_square()) {
            return Some(x);
        }
        x = &x + A;
    }
    None
}

/// Find `n*A` and return `(x, hint)`. Used in signing where `A` is
/// honestly generated and the predicate is satisfied for some small
/// `n`; the bound is here for defense in depth.
fn find_na_x_coord_with_hint(A: &Fp2, _curve: &Curve) -> Option<(Fp2, u8)> {
    let mut x = *A;
    for n in 1u32..=FIND_X_COORD_MAX_TRIES {
        if is_on_curve(&x, A) && !bool::from(x.is_square()) {
            let hint = if n < 128 { n as u8 } else { 0 };
            return Some((x, hint));
        }
        x = &x + A;
    }
    None
}

/// Find `b` such that `-A/(1+i*b)` is a valid NQR x-coordinate on E_A.
/// Returns `None` if no `b` in `[start, start + FIND_X_COORD_MAX_TRIES)`
/// satisfies the predicate.
fn find_nqr_factor(A: &Fp2, _curve: &Curve, start: u8) -> Option<Fp2> {
    for offset in 0..FIND_X_COORD_MAX_TRIES {
        let n = u32::from(start).wrapping_add(offset);
        let z = Fp2::new(Fp::ONE, Fp::from_small(n));
        let x = &(-A) * &z.invert();
        if is_on_curve(&x, A) && !bool::from(x.is_square()) {
            return Some(x);
        }
    }
    None
}

/// Find `-A/(1+i*b)` and return `(x, hint)`.
fn find_nqr_factor_with_hint(A: &Fp2, _curve: &Curve) -> Option<(Fp2, u8)> {
    for n in 1u32..=FIND_X_COORD_MAX_TRIES {
        let z = Fp2::new(Fp::ONE, Fp::from_small(n));
        let x = &(-A) * &z.invert();
        if is_on_curve(&x, A) && !bool::from(x.is_square()) {
            let hint = if n < 128 { n as u8 } else { 0 };
            return Some((x, hint));
        }
    }
    None
}
