//! Projective x-only points on Montgomery curves.

use core::ops::Mul;

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use super::{AffineX, Curve};
use crate::{curves::scalar::Scalar, fields::fp2::Fp2};

/// A point on a Montgomery curve E_A : y² = x³ + Ax² + x over F_{p²}.
///
/// Stored in x-only projective coordinates (X : Z), where the affine
/// x-coordinate is x = X/Z when Z ≠ 0. The point at infinity
/// (identity) is represented as (∗ : 0). The y-coordinate is
/// discarded since SQIsign uses only the x-coordinate line.
///
/// Each point carries a reference to its [`Curve`], so the domain
/// is always known.
///
/// See [§8.2.1] for the projective coordinate conventions.
///
/// [§8.2.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.2.1
#[derive(Copy, Clone, Debug)]
pub struct ProjectiveXOnlyPoint {
    /// Projective X coordinate.
    pub X: Fp2,
    /// Projective Z coordinate (Z = 0 for the point at infinity).
    pub Z: Fp2,
    /// The curve this point lives on.
    curve: Curve,
}

impl ProjectiveXOnlyPoint {
    /// Construct the identity (point at infinity) on the given curve.
    pub fn identity(curve: &Curve) -> ProjectiveXOnlyPoint {
        ProjectiveXOnlyPoint {
            X: Fp2::ONE,
            Z: Fp2::ZERO,
            curve: *curve,
        }
    }

    /// Construct from projective coordinates on the given curve.
    pub fn from_XZ(X: Fp2, Z: Fp2, curve: &Curve) -> ProjectiveXOnlyPoint {
        ProjectiveXOnlyPoint {
            X,
            Z,
            curve: *curve,
        }
    }

    /// Compute the affine x-coordinate x = X/Z.
    #[must_use]
    pub fn to_affine_x(self) -> AffineX {
        AffineX::from(&self.X * &self.Z.invert())
    }

    /// Construct from an affine x-coordinate on the given curve.
    pub fn from_affine_x(x: Fp2, curve: &Curve) -> ProjectiveXOnlyPoint {
        ProjectiveXOnlyPoint {
            X: x,
            Z: Fp2::ONE,
            curve: *curve,
        }
    }

    /// The curve this point lives on.
    pub fn curve(&self) -> &Curve {
        &self.curve
    }

    /// Check if this is the point at infinity (Z == 0).
    pub fn is_identity(&self) -> Choice {
        self.Z.ct_eq(&Fp2::ZERO)
    }

    /// Compute \[2\]self.
    ///
    /// See [§8.2], Algorithm 8.3 (`xDBL`).
    ///
    /// [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
    #[must_use]
    pub fn double(&self) -> ProjectiveXOnlyPoint {
        let t0 = (&self.X + &self.Z).square();
        let t1 = (&self.X - &self.Z).square();
        let t2 = &t0 - &t1;
        let t1_c24 = &t1 * &self.curve.doubling.C24;
        let X2 = &t0 * &t1_c24;
        let Z2 = &t2 * &(&(&t2 * &self.curve.doubling.A24) + &t1_c24);
        ProjectiveXOnlyPoint {
            X: X2,
            Z: Z2,
            curve: self.curve,
        }
    }

    /// Compute \[2\]self using **un-normalized** projective curve
    /// constants `(A + 2C : 4C)` derived from `self.curve.projective`.
    ///
    /// Mirrors C ref's `xDBL` (`ec.c:234`): the formula uses
    /// `(A + 2C, 4C)` directly, never dividing by `4C`. Each call
    /// scales the output `(X : Z)` by a projective factor of `4C`
    /// per doubling relative to [`double`] (which always divides by
    /// `4C` because curves are constructed with normalized
    /// [`DoublingConstants`]).
    ///
    /// Use this for kernel-prep doublings on a curve that came out
    /// of `EllipticProduct::from(&ThetaNullPoint)` (and whose
    /// downstream consumers expect
    /// the un-normalized rep, like C ref's
    /// `double_couple_point_iter` on a freshly returned
    /// `Fu_codomain.E1`). Required for byte-equality with C ref on
    /// outer-chain prep when `scale > 0`.
    ///
    /// # Design note
    ///
    /// The current single-`Curve` design carries one cached
    /// (always normalized) [`DoublingConstants`] and lets specific
    /// call sites opt into un-normalized doubling via this method.
    /// C ref takes the dual approach: a per-curve runtime flag
    /// `is_A24_computed_and_normalized` that `ec_dbl` reads to
    /// dispatch to `xDBL` or `xDBL_A24`, and lazily flips the flag
    /// in `ec_curve_normalize_A24` when a long doubling chain
    /// (`n > 50`) makes normalization profitable.
    ///
    /// We could mirror C ref's flag-based design, or further split
    /// `Curve` into a normalized/un-normalized type pair that makes
    /// the choice unrepresentable at compile time. The latter is
    /// more in line with our compile-time-invariants style but is a
    /// substantive refactor (every `Curve` parameter and every
    /// `ProjectiveXOnlyPoint::curve` deref would need to think
    /// about which form it expects). The current escape-hatch is
    /// the smallest change that gives byte-equality at the one
    /// known interop boundary; promote to an in-type representation
    /// if more boundaries appear.
    ///
    /// [`double`]: Self::double
    /// [`DoublingConstants`]: crate::curves::montgomery::DoublingConstants
    #[must_use]
    pub fn double_unnormalized(&self) -> ProjectiveXOnlyPoint {
        let pc = &self.curve.projective;
        let two_c = &pc.C + &pc.C;
        let a24 = &pc.A + &two_c;
        let c24 = &two_c + &two_c;
        let t0 = (&self.X + &self.Z).square();
        let t1 = (&self.X - &self.Z).square();
        let t2 = &t0 - &t1;
        let t1_c24 = &t1 * &c24;
        let X2 = &t0 * &t1_c24;
        let Z2 = &t2 * &(&(&t2 * &a24) + &t1_c24);
        ProjectiveXOnlyPoint {
            X: X2,
            Z: Z2,
            curve: self.curve,
        }
    }

    /// Compute \[2\]self specialized for `A = 0` (the curve `E_0`).
    ///
    /// Implements the C reference's `xDBL_E0` (`ec.c:215-231`):
    ///
    /// - `X' = (X + Z)² · 2(X - Z)²`
    /// - `Z' = 4XZ · (2(X - Z)² + 4XZ)`
    ///
    /// This is **not** projectively equal to [`double`] in the
    /// `(X : Z)` representative — the output is exactly `2 ·
    /// double(self)` per coordinate. The C reference uses this
    /// specialized variant inside its biladder (`xDBLMUL`,
    /// `ec.c:485`) but uses the normalized `xDBL_A24` variant
    /// elsewhere (`ec_dbl_iter`, `ec.c:586`). To produce
    /// byte-equal projective representatives against the C
    /// reference's biladder output, the biladder MUST use this
    /// `xDBL_E0` form when `A = 0`.
    ///
    /// The caller is responsible for ensuring `A = 0` (i.e. the
    /// curve is `E_0`); the formula is correct only on that curve.
    ///
    /// [`double`]: Self::double
    #[must_use]
    pub fn double_e0(&self) -> ProjectiveXOnlyPoint {
        let t0 = (&self.X + &self.Z).square();
        let t1 = (&self.X - &self.Z).square();
        let t2 = &t0 - &t1;
        let t1_doubled = &t1 + &t1;
        let X2 = &t0 * &t1_doubled;
        let Z2 = &t2 * &(&t1_doubled + &t2);
        ProjectiveXOnlyPoint {
            X: X2,
            Z: Z2,
            curve: self.curve,
        }
    }

    /// Compute self + other, given self − other.
    ///
    /// See [§8.2], Algorithm 8.4 (`xADD`).
    ///
    /// [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
    #[must_use]
    pub fn differential_add(
        &self,
        other: &ProjectiveXOnlyPoint,
        difference: &ProjectiveXOnlyPoint,
    ) -> ProjectiveXOnlyPoint {
        let t0 = &(&self.X + &self.Z) * &(&other.X - &other.Z);
        let t1 = &(&self.X - &self.Z) * &(&other.X + &other.Z);
        let sum = (&t0 + &t1).square();
        let diff = (&t0 - &t1).square();
        ProjectiveXOnlyPoint {
            X: &difference.Z * &sum,
            Z: &difference.X * &diff,
            curve: self.curve,
        }
    }

    /// Compute the x-coordinate of self − other (or self + other)
    /// deterministically in projective coordinates.
    ///
    /// Given two points P = (X_P : Z_P) and Q = (X_Q : Z_Q) on the
    /// same curve with Montgomery coefficient (A : C), computes
    /// x_{P−Q} = (X_{PQ} : Z_{PQ}) using a projectively invariant
    /// square root.
    ///
    /// The choice of sign (P − Q vs P + Q) is deterministic and
    /// projectively invariant, ensuring that signing and verification
    /// agree on the same basis.
    ///
    /// Implements `ProjectiveDifference` ([§8.2.3], Algorithm 8.10).
    ///
    /// [§8.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.2.3
    #[must_use]
    pub fn projective_difference(&self, other: &ProjectiveXOnlyPoint) -> ProjectiveXOnlyPoint {
        let (X_P, Z_P) = (&self.X, &self.Z);
        let (X_Q, Z_Q) = (&other.X, &other.Z);
        let A = self.curve.coefficient().as_fp2();

        // Our curves are normalized with C = 1, so the (A : C) pair
        // in the algorithm reduces to just A.
        //
        // B_XX = (X_P · X_Q − Z_P · Z_Q)²
        let B_XX = (&(X_P * X_Q) - &(Z_P * Z_Q)).square();

        // B_XZ = (X_P · X_Q + Z_P · Z_Q)(X_P · Z_Q + Z_P · X_Q)
        //        + 2A · X_P · X_Q · Z_P · Z_Q
        let xpxq = X_P * X_Q;
        let zpzq = Z_P * Z_Q;
        let xpzq = X_P * Z_Q;
        let zpxq = Z_P * X_Q;
        let B_XZ = &(&(&xpxq + &zpzq) * &(&xpzq + &zpxq)) + &(&(A + A) * &(&xpxq * &zpzq));

        // B_ZZ = (X_P · Z_Q − Z_P · X_Q)²
        let B_ZZ = (&xpzq - &zpxq).square();

        // Normalize so the discriminant is a fourth power in Fp, making
        // the Fp2 square root deterministic. With C = 1 the C reference's
        // C·conj(C)²·conj(Z_P)²·conj(Z_Q)² factor reduces to
        // conj(Z_P)²·conj(Z_Q)². The conjugates are essential — γ = (Z_P·Z_Q)²
        // does not work because conj(z)² ≠ z² in Fp2.
        let gamma = &Z_P.conjugate().square() * &Z_Q.conjugate().square();

        // Scale: B_XX *= γ, B_XZ *= γ, B_ZZ *= γ
        let B_XX = &gamma * &B_XX;
        let B_XZ = &gamma * &B_XZ;
        let B_ZZ = &gamma * &B_ZZ;

        // δ = SquareRoot(B_XZ² − B_XX · B_ZZ)
        let discriminant = &B_XZ.square() - &(&B_XX * &B_ZZ);
        let delta = discriminant.sqrt();

        // x_{PQ} = (δ + B_XZ : B_ZZ)
        ProjectiveXOnlyPoint {
            X: &delta + &B_XZ,
            Z: B_ZZ,
            curve: self.curve,
        }
    }

    /// Clear the odd cofactor: computes `[c]P` where `c = 5 = (p+1)/2^f`.
    ///
    /// Projects a point onto the 2^f-torsion subgroup. Uses two
    /// doublings and one differential addition (much cheaper than
    /// a full scalar multiplication).
    pub fn clear_cofactor(&self) -> ProjectiveXOnlyPoint {
        // Multiply by 5 (the odd cofactor for p = 5·2^248 − 1) using
        // a 3-bit Montgomery ladder matching the C reference's
        // `xMUL(P, 5, 3, curve)`.
        //
        // This produces the same projective representative as the
        // C ref, which is critical: subsequent operations (scalar_mul_add,
        // projective_difference) are sensitive to the projective
        // representative, not just the affine x-coordinate.
        //
        // The cofactor 5 is public data, so variable-time is fine.
        let mut r0 = ProjectiveXOnlyPoint::identity(&self.curve);
        let mut r1 = *self;

        // 5 = 0b101, 3 bits. Process MSB to LSB with differential swap.
        let bits = [1u8, 0, 1]; // bits[0] = MSB (bit 2), bits[2] = LSB (bit 0)
        let mut prev_bit = 0u8;
        for &bit in &bits {
            let swap = Choice::from((bit ^ prev_bit) & 1);
            prev_bit = bit;
            ProjectiveXOnlyPoint::conditional_swap(&mut r0, &mut r1, swap);
            differential_add_and_double(&mut r0, &mut r1, self);
        }
        // Final swap
        let swap = Choice::from(prev_bit & 1);
        ProjectiveXOnlyPoint::conditional_swap(&mut r0, &mut r1, swap);
        r0
    }

    /// Scalar multiplication via the Montgomery ladder.
    ///
    /// Computes \[n\]self, constant-time in the scalar value.
    /// Always performs 256 ladder steps regardless of the scalar's
    /// magnitude to prevent timing side channels.
    ///
    /// Implements `Ladder` ([§8.2], Algorithm 8.6).
    ///
    /// [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
    pub fn scalar_mul(&self, n: &Scalar) -> ProjectiveXOnlyPoint {
        let mut x0 = ProjectiveXOnlyPoint::identity(&self.curve);
        let mut x1 = *self;

        let mut prev_bit = false;
        for cur_bit in n.bits_be(Scalar::BITS) {
            let swap: u8 = (prev_bit ^ cur_bit) as u8;
            ProjectiveXOnlyPoint::conditional_swap(&mut x0, &mut x1, swap.into());
            differential_add_and_double(&mut x0, &mut x1, self);
            prev_bit = cur_bit;
        }
        ProjectiveXOnlyPoint::conditional_swap(&mut x0, &mut x1, Choice::from(prev_bit as u8));
        x0
    }
}

/// Simultaneous doubling and differential addition.
///
/// Sets P ← \[2\]P and Q ← P + Q, given the difference P − Q.
///
/// See [§8.2], Algorithm 8.5 (`xDBLADD`).
///
/// [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
#[rustfmt::skip]
pub(crate) fn differential_add_and_double(
    P: &mut ProjectiveXOnlyPoint,
    Q: &mut ProjectiveXOnlyPoint,
    PmQ: &ProjectiveXOnlyPoint,
) {
    let sum_P  = &P.X + &P.Z;
    let diff_P = &P.X - &P.Z;

    // xDBL
    let t0     = sum_P.square();
    let t1     = diff_P.square();
    let t2     = &t0 - &t1;
    let t1_c24 = &t1 * &P.curve.doubling.C24;
    let dbl_X  = &t0 * &t1_c24;
    let dbl_Z  = &t2 * &(&(&t2 * &P.curve.doubling.A24) + &t1_c24);

    // xADD
    let u     = &sum_P * &(&Q.X - &Q.Z);
    let v     = &diff_P * &(&Q.X + &Q.Z);
    let add_X = &PmQ.Z * &(&u + &v).square();
    let add_Z = &PmQ.X * &(&u - &v).square();

    P.X = dbl_X;
    P.Z = dbl_Z;
    Q.X = add_X;
    Q.Z = add_Z;
}

/// Scalar multiplication `[n]P`. Constant-time in the scalar value.
impl Mul<&ProjectiveXOnlyPoint> for &Scalar {
    type Output = ProjectiveXOnlyPoint;

    fn mul(self, point: &ProjectiveXOnlyPoint) -> ProjectiveXOnlyPoint {
        point.scalar_mul(self)
    }
}

impl Mul<ProjectiveXOnlyPoint> for Scalar {
    type Output = ProjectiveXOnlyPoint;

    fn mul(self, point: ProjectiveXOnlyPoint) -> ProjectiveXOnlyPoint {
        point.scalar_mul(&self)
    }
}

impl Mul<&ProjectiveXOnlyPoint> for Scalar {
    type Output = ProjectiveXOnlyPoint;

    fn mul(self, point: &ProjectiveXOnlyPoint) -> ProjectiveXOnlyPoint {
        point.scalar_mul(&self)
    }
}

impl Mul<ProjectiveXOnlyPoint> for &Scalar {
    type Output = ProjectiveXOnlyPoint;

    fn mul(self, point: ProjectiveXOnlyPoint) -> ProjectiveXOnlyPoint {
        point.scalar_mul(self)
    }
}

impl ConditionallySelectable for ProjectiveXOnlyPoint {
    fn conditional_select(
        a: &ProjectiveXOnlyPoint,
        b: &ProjectiveXOnlyPoint,
        choice: Choice,
    ) -> ProjectiveXOnlyPoint {
        ProjectiveXOnlyPoint {
            X: Fp2::conditional_select(&a.X, &b.X, choice),
            Z: Fp2::conditional_select(&a.Z, &b.Z, choice),
            curve: a.curve, // assumed to be on the same curve
        }
    }
}

impl ConstantTimeEq for ProjectiveXOnlyPoint {
    /// Two projective points are equal iff X₁Z₂ = X₂Z₁.
    fn ct_eq(&self, other: &ProjectiveXOnlyPoint) -> Choice {
        (&self.X * &other.Z).ct_eq(&(&other.X * &self.Z))
    }
}

impl PartialEq for ProjectiveXOnlyPoint {
    fn eq(&self, other: &ProjectiveXOnlyPoint) -> bool {
        self.ct_eq(other).into()
    }
}

impl Eq for ProjectiveXOnlyPoint {}
