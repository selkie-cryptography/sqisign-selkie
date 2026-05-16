//! Endomorphism action on torsion bases.
//!
//! Bridges the quaternion algebra and elliptic curve worlds by
//! representing how quaternion order elements act on the torsion
//! basis E_t[2^f]. Each action is a 2×2 matrix over Z/2^f Z.
//!
//! In representation-theoretic language, [`EndomorphismAction`] *is*
//! the ring homomorphism ρ_t: O_t → M_2(Z/2^f Z) — the Deuring
//! representation of End(E_t) ≅ O_t on the 2^f-torsion.
//!
//! See [§3.2.1.1] of the SQIsign specification.
//!
//! [§3.2.1.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.2.1.1

use subtle::{Choice, ConditionallySelectable};

use crate::{
    curves::{TorsionBasis, TorsionExponent, montgomery::ProjectiveXOnlyPoint, scalar::Scalar},
    quaternions::{
        algebra::{Coordinate, Denominator, Element},
        bigint::BigInt,
        lattice::Lattice,
    },
};

/// A 2×2 matrix over Z/2^f Z representing the action of an
/// endomorphism on a torsion basis (P, Q) of E[2^f].
///
/// Given an endomorphism α and basis (P, Q), the matrix `M_α`
/// satisfies: `α(P) = [M[0][0]]P + [M[1][0]]Q` and
/// `α(Q) = [M[0][1]]P + [M[1][1]]Q`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndomorphismMatrix {
    /// Entries stored row-major: [[a, b], [c, d]].
    entries: [[Scalar; 2]; 2],
}

// reason: the matrix-algebra surface (`ZERO`, `det_mod`,
// `apply_scaled`, `apply_to_basis`) is part of the public type but
// not currently reached from any production path — keygen/sign route
// through `EndomorphismAction::apply` and the propagated-PmQ
// `apply_scaled_basis` variant. Kept as a stable surface for
// downstream consumers (`expose-internals`).
#[allow(dead_code)]
impl EndomorphismMatrix {
    /// The zero matrix.
    pub const ZERO: Self = Self {
        entries: [[Scalar::ZERO; 2]; 2],
    };

    /// Creates a matrix from four entries (row-major).
    pub const fn new(a: Scalar, b: Scalar, c: Scalar, d: Scalar) -> Self {
        Self {
            entries: [[a, b], [c, d]],
        }
    }

    /// Creates a matrix from four little-endian limb arrays.
    pub const fn from_limbs(a: [u64; 4], b: [u64; 4], c: [u64; 4], d: [u64; 4]) -> Self {
        Self {
            entries: [
                [Scalar::from_limbs(a), Scalar::from_limbs(b)],
                [Scalar::from_limbs(c), Scalar::from_limbs(d)],
            ],
        }
    }

    /// Returns entry at (row, col).
    pub const fn entry(&self, row: usize, col: usize) -> &Scalar {
        &self.entries[row][col]
    }

    /// Matrix-vector multiplication mod `2^f`: `M · [c1, c2]^T`.
    ///
    /// Returns `[M[0][0]*c1 + M[0][1]*c2, M[1][0]*c1 + M[1][1]*c2]` mod `2^f`.
    ///
    /// The arguments are `BigInt<4>` because callers pass quaternion
    /// coordinates.
    pub fn eval_mod(&self, c1: &BigInt<4>, c2: &BigInt<4>, f: u32) -> (BigInt<4>, BigInt<4>) {
        let s1 = Scalar::from(*c1);
        let s2 = Scalar::from(*c2);
        let r0 = self.entries[0][0]
            .mul_mod2k(&s1, f)
            .add_mod2k(&self.entries[0][1].mul_mod2k(&s2, f), f);
        let r1 = self.entries[1][0]
            .mul_mod2k(&s1, f)
            .add_mod2k(&self.entries[1][1].mul_mod2k(&s2, f), f);
        (BigInt::from(r0), BigInt::from(r1))
    }

    /// Apply this matrix, scaled by a scalar, to a pair of points.
    ///
    /// Computes P' = [s·m₀₀]P + [s·m₁₀]Q and
    ///          Q' = [s·m₀₁]P + [s·m₁₁]Q, all mod 2^f.
    ///
    /// The input difference `P − Q` is recovered via
    /// `projective_difference`. Callers that already carry a
    /// propagated `PmQ` should use [`Self::apply_scaled_basis`]
    /// instead; otherwise the sqrt-branch of `projective_difference`
    /// will yield a `PmQ' = projective_difference(P', Q')` whose
    /// projective rep is inconsistent with downstream consumers
    /// like `Kernel::from_montgomery`.
    pub fn apply_scaled(
        &self,
        scalar: &BigInt<4>,
        p: ProjectiveXOnlyPoint,
        q: ProjectiveXOnlyPoint,
        f: TorsionExponent,
    ) -> (ProjectiveXOnlyPoint, ProjectiveXOnlyPoint) {
        let s = Scalar::from(*scalar);
        let fv = f.value();

        let s00 = s.mul_mod2k(self.entry(0, 0), fv);
        let s01 = s.mul_mod2k(self.entry(0, 1), fv);
        let s10 = s.mul_mod2k(self.entry(1, 0), fv);
        let s11 = s.mul_mod2k(self.entry(1, 1), fv);

        let basis = TorsionBasis::from((p, q));
        let p_prime = basis.eval_decomposition(&s00, &s10);
        let q_prime = basis.eval_decomposition(&s01, &s11);
        (p_prime, q_prime)
    }

    /// Apply this matrix, scaled by a scalar, to a propagated basis.
    ///
    /// Computes
    ///   P'   = [s·m₀₀]P + [s·m₁₀]Q
    ///   Q'   = [s·m₀₁]P + [s·m₁₁]Q
    ///   PmQ' = [s·(m₀₀ − m₀₁)]P + [s·(m₁₀ − m₁₁)]Q
    /// all mod 2^f. All three are computed via biscalar
    /// multiplication using the input basis's tracked `PmQ`, so the
    /// output `PmQ'` is propagated — never recovered through
    /// `projective_difference(P', Q')`.
    ///
    /// Required when the output basis will be fed to a chain whose
    /// `lift_basis` (Okeya-Sakurai) expects a `PmQ` projective rep
    /// consistent with `P` and `Q`'s computation history. The
    /// sqrt-branch of `projective_difference` is not aligned with
    /// such a history, and a downstream `Kernel::from_montgomery →
    /// kernel.isogeny` chain that depends on it can produce a
    /// terminal theta null with `splitting_index_count() = 0`.
    pub fn apply_scaled_basis(
        &self,
        scalar: &BigInt<4>,
        basis: &TorsionBasis,
        f: TorsionExponent,
    ) -> (
        ProjectiveXOnlyPoint,
        ProjectiveXOnlyPoint,
        ProjectiveXOnlyPoint,
    ) {
        let s = Scalar::from(*scalar);
        let fv = f.value();

        let s00 = s.mul_mod2k(self.entry(0, 0), fv);
        let s01 = s.mul_mod2k(self.entry(0, 1), fv);
        let s10 = s.mul_mod2k(self.entry(1, 0), fv);
        let s11 = s.mul_mod2k(self.entry(1, 1), fv);

        let p_prime = basis.eval_decomposition(&s00, &s10);
        let q_prime = basis.eval_decomposition(&s01, &s11);
        let pmq_prime =
            basis.eval_decomposition(&s00.sub_mod2k(&s01, fv), &s10.sub_mod2k(&s11, fv));
        (p_prime, q_prime, pmq_prime)
    }

    /// Apply this action matrix to a torsion basis, returning a new
    /// basis for the endomorphism image.
    ///
    /// Computes `θ(P) = [m00]P + [m10]Q` and `θ(Q) = [m01]P + [m11]Q`,
    /// then derives `θ(P) − θ(Q)` via `projective_difference` to
    /// ensure the Okeya-Sakurai y-recovery produces consistent
    /// Jacobian coordinates. The resulting basis is safe to pass to
    /// `TorsionBasis::lift`.
    #[must_use]
    pub fn apply_to_basis(&self, basis: &TorsionBasis, f: TorsionExponent) -> TorsionBasis {
        let p_prime = basis.biscalar_mul(self.entry(0, 0), self.entry(1, 0), f);
        let q_prime = basis.biscalar_mul(self.entry(0, 1), self.entry(1, 1), f);
        TorsionBasis::from((p_prime, q_prime))
    }

    /// Determinant mod 2^f: `ad − bc`.
    pub fn det_mod(&self, f: u32) -> Scalar {
        self.entries[0][0]
            .mul_mod2k(&self.entries[1][1], f)
            .sub_mod2k(&self.entries[0][1].mul_mod2k(&self.entries[1][0], f), f)
    }

    /// Classical adjugate mod 2^f: `[[d, −b], [−c, a]]`.
    pub fn adjugate_mod(&self, f: u32) -> Self {
        let zero = Scalar::ZERO;
        Self {
            entries: [
                [self.entries[1][1], zero.sub_mod2k(&self.entries[0][1], f)],
                [zero.sub_mod2k(&self.entries[1][0], f), self.entries[0][0]],
            ],
        }
    }

    /// Matrix-matrix multiplication mod 2^f.
    pub fn mat_mul_mod(&self, rhs: &Self, f: u32) -> Self {
        Self {
            entries: [
                [
                    self.entries[0][0]
                        .mul_mod2k(&rhs.entries[0][0], f)
                        .add_mod2k(&self.entries[0][1].mul_mod2k(&rhs.entries[1][0], f), f),
                    self.entries[0][0]
                        .mul_mod2k(&rhs.entries[0][1], f)
                        .add_mod2k(&self.entries[0][1].mul_mod2k(&rhs.entries[1][1], f), f),
                ],
                [
                    self.entries[1][0]
                        .mul_mod2k(&rhs.entries[0][0], f)
                        .add_mod2k(&self.entries[1][1].mul_mod2k(&rhs.entries[1][0], f), f),
                    self.entries[1][0]
                        .mul_mod2k(&rhs.entries[0][1], f)
                        .add_mod2k(&self.entries[1][1].mul_mod2k(&rhs.entries[1][1], f), f),
                ],
            ],
        }
    }
}

impl ConditionallySelectable for EndomorphismMatrix {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self {
            entries: [
                [
                    Scalar::conditional_select(&a.entries[0][0], &b.entries[0][0], choice),
                    Scalar::conditional_select(&a.entries[0][1], &b.entries[0][1], choice),
                ],
                [
                    Scalar::conditional_select(&a.entries[1][0], &b.entries[1][0], choice),
                    Scalar::conditional_select(&a.entries[1][1], &b.entries[1][1], choice),
                ],
            ],
        }
    }
}

/// The action of a maximal order `O_t ≅ End(E_t)` on the
/// 2^f-torsion of `E_t`.
///
/// Carries the data needed to evaluate the Deuring representation
/// ρ_t: O_t → M_2(Z/2^f Z) on any element α ∈ O_t:
///
/// - [`order`](Self::order): the maximal order `O_t`, used to decompose α from
///   the `{1, i, j, k}` basis of `B_{p,∞}` onto `O_t`'s column basis.
/// - [`generators`](Self::generators): the action matrices of the three
///   non-identity generators (`gen2`, `gen3`, `gen4` — i.e., the 2nd, 3rd, 4th
///   order-basis elements). The identity for the 1st basis element is
///   synthesized at evaluation time.
///
/// Evaluate via [`Self::apply`].
///
/// See [§3.2.1.1] of the spec.
///
/// [§3.2.1.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.2.1.1
pub(crate) struct EndomorphismAction {
    /// The maximal order `O_t ≅ End(E_t)`. The element to be acted
    /// on is decomposed onto this order's column basis.
    pub(crate) order: &'static Lattice<4>,
    /// Action matrices for the three non-identity order generators.
    pub(crate) generators: [EndomorphismMatrix; 3],
}

impl EndomorphismAction {
    /// Apply this representation to `alpha ∈ O_t ≅ End(E_t)`,
    /// returning the action matrix on `E_t[2^f]`.
    ///
    /// Decomposes `alpha` from the `{1, i, j, k}` basis into the
    /// order's column basis `(c₀, c₁, c₂, c₃)`, then assembles:
    ///
    /// `ρ_t(alpha) = c₀·I + c₁·M_{gen2} + c₂·M_{gen3} + c₃·M_{gen4}  (mod 2^f)`
    ///
    /// Returns `None` if `alpha ∉ O_t` (decomposition fails).
    ///
    /// Implements the matrix-assembly half of
    /// [IdealToKernel][Alg. 3.14]; the surrounding column-pick step
    /// (used inside [`super::compute_even_response`]) is inlined at
    /// the call site.
    ///
    /// [Alg. 3.14]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.14
    pub(crate) fn apply(
        &self,
        alpha: &Element<4>,
        f: TorsionExponent,
    ) -> Option<EndomorphismMatrix> {
        // Decompose at width 20 — for p-extremal orders with `q ≥ 5`
        // the basis entries reach ~250 bits (e.g. q=97 row 1 col 3 ≈
        // 2^250), and `Lattice::decompose` computes a 4×4 adjugate
        // whose 3×3 minors accumulate up to ~3·250 = 750 bits. Then
        // `adjugate · rhs` (rhs ≈ basis_entry ≈ 250 bits) reaches
        // ~1000 bits before `/ det`. Width 8 silently overflows;
        // width 20 is comfortable margin for all NIST-I orders. Same
        // width pitfall as `ExtremalOrder::represent_integer`.
        let elem_w = Element::<20>::new(
            Coordinate::from_bigint(alpha.a.as_bigint().widen::<20>()),
            Coordinate::from_bigint(alpha.b.as_bigint().widen::<20>()),
            Coordinate::from_bigint(alpha.c.as_bigint().widen::<20>()),
            Coordinate::from_bigint(alpha.d.as_bigint().widen::<20>()),
            Denominator::from_bigint_unchecked(BigInt::<4>::from(alpha.denom).widen::<20>()),
        );
        let order_w: Lattice<20> = {
            let basis4 = self.order.basis();
            let mut basis_w = crate::quaternions::linear::Matrix::<20>::ZERO;
            for row in 0..4 {
                for col in 0..4 {
                    basis_w[row][col] = basis4[row][col].widen::<20>();
                }
            }
            Lattice::new(basis_w, self.order.denom().widen::<20>())
        };
        let coords_w = order_w.decompose(&elem_w)?;
        let coords: [BigInt<4>; 4] = [
            coords_w[0].narrow_to::<4>()?,
            coords_w[1].narrow_to::<4>()?,
            coords_w[2].narrow_to::<4>()?,
            coords_w[3].narrow_to::<4>()?,
        ];
        #[cfg(test)]
        {
            crate::selkie_trace!("CREF_FDI coeffs[0]={}", coords[0]);
            crate::selkie_trace!("CREF_FDI coeffs[1]={}", coords[1]);
            crate::selkie_trace!("CREF_FDI coeffs[2]={}", coords[2]);
            crate::selkie_trace!("CREF_FDI coeffs[3]={}", coords[3]);
        }

        // Reduce all coefficients mod 2^f. For negative coefficients,
        // ct_mod returns a negative remainder (truncated division),
        // so add the modulus to get the canonical representative in
        // [0, 2^f).
        let modulus = BigInt::<4>::ONE << f.value();
        let reduce = |c: &BigInt<4>| -> Scalar {
            let r = c.ct_mod(&modulus);
            if bool::from(r.is_negative()) {
                Scalar::from(r.ct_add(&modulus))
            } else {
                Scalar::from(r)
            }
        };

        let c0_scalar = reduce(&coords[0]);
        let mut result = EndomorphismMatrix::new(c0_scalar, Scalar::ZERO, Scalar::ZERO, c0_scalar);

        for k in 0..3 {
            let s = reduce(&coords[k + 1]);
            let other = &self.generators[k];
            let fv = f.value();
            result = EndomorphismMatrix::new(
                result
                    .entry(0, 0)
                    .add_mod2k(&s.mul_mod2k(other.entry(0, 0), fv), fv),
                result
                    .entry(0, 1)
                    .add_mod2k(&s.mul_mod2k(other.entry(0, 1), fv), fv),
                result
                    .entry(1, 0)
                    .add_mod2k(&s.mul_mod2k(other.entry(1, 0), fv), fv),
                result
                    .entry(1, 1)
                    .add_mod2k(&s.mul_mod2k(other.entry(1, 1), fv), fv),
            );
        }

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_matrix_mul() {
        let a = EndomorphismMatrix::new(
            Scalar::from_u64(1),
            Scalar::from_u64(2),
            Scalar::from_u64(3),
            Scalar::from_u64(4),
        );
        let b = EndomorphismMatrix::new(
            Scalar::from_u64(5),
            Scalar::from_u64(6),
            Scalar::from_u64(7),
            Scalar::from_u64(8),
        );
        let c = a.mat_mul_mod(&b, 248);
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(*c.entry(0, 0), Scalar::from_u64(19));
        assert_eq!(*c.entry(0, 1), Scalar::from_u64(22));
        assert_eq!(*c.entry(1, 0), Scalar::from_u64(43));
        assert_eq!(*c.entry(1, 1), Scalar::from_u64(50));
    }
}
