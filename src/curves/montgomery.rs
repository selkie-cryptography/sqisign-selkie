//! Elliptic curves in Montgomery form and their x-only projective arithmetic.
//!
//! SQIsign operates on supersingular Montgomery curves E_{A,B} over F_{p²}:
//!
//!   By² = x³ + Ax² + x
//!
//! where B = 1 for most operations (we track only A). Points are represented
//! in x-only projective coordinates (X : Z) since SQIsign only needs the
//! x-coordinate line.
//!
//! Torsion bases are represented as triplets (x_R, x_S, x_{R−S}) of affine
//! x-coordinates, which is the minimum information needed for differential
//! addition. These are never serialized — on the wire, bases are compressed
//! to 1-byte hints and reconstructed via `TorsionBasisFromHint` ([§2.2.3]).
//!
//! [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
//!
//! See also [§2.2] (elliptic curves) and [§8.2] (curve arithmetic).
//!
//! [§2.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
//! [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2

use core::ops::{Mul, MulAssign};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::fields::fp::Fp;
use crate::fields::fp2::Fp2;

/// The Montgomery coefficient A of a curve E_A : y² = x³ + Ax² + x.
///
/// This is the canonical representation of a Montgomery curve: a single
/// element of F_{p²}. On the wire, curves are encoded as their
/// Montgomery coefficient (64 bytes for NIST-I).
///
/// See [§2.2.1] (Montgomery curves) and [§4.6] (binary format).
///
/// [§2.2.1]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
/// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MontgomeryCoefficient(Fp2);

impl MontgomeryCoefficient {
    /// A = 0, the coefficient of the starting curve E₀.
    pub const ZERO: MontgomeryCoefficient = MontgomeryCoefficient(Fp2::ZERO);

    /// The underlying F_{p²} element.
    pub fn as_fp2(&self) -> &Fp2 {
        &self.0
    }

    /// Encode as bytes (delegates to F_{p²} encoding).
    pub fn to_bytes(&self) -> [u8; crate::params::CURVE_ENCODED_BYTES] {
        self.0.to_bytes()
    }

    /// Decode from bytes.
    pub fn from_bytes(bytes: &[u8; crate::params::CURVE_ENCODED_BYTES]) -> MontgomeryCoefficient {
        MontgomeryCoefficient(Fp2::from_bytes(bytes))
    }
}

impl From<Fp2> for MontgomeryCoefficient {
    fn from(a: Fp2) -> Self {
        MontgomeryCoefficient(a)
    }
}

impl From<MontgomeryCoefficient> for Fp2 {
    fn from(a: MontgomeryCoefficient) -> Fp2 {
        a.0
    }
}

/// A Montgomery curve E_A : y² = x³ + Ax² + x over F_{p²}.
///
/// Stores the [`MontgomeryCoefficient`] A together with precomputed
/// projective doubling constants (A₂₄, C₂₄) = (A + 2, 4).
#[derive(Copy, Clone, Debug)]
pub struct Curve {
    A: MontgomeryCoefficient,
    A24: Fp2,
    C24: Fp2,
}

impl Curve {
    /// The starting curve E₀ : y² = x³ + x (A = 0).
    ///
    /// This is the supersingular curve used as the base in SQIsign.
    pub const E0: Curve = Curve {
        A: MontgomeryCoefficient::ZERO,
        A24: Fp2::new(Fp::TWO, Fp::ZERO),
        C24: Fp2::new(Fp::FOUR, Fp::ZERO),
    };

    /// Construct a curve from its Montgomery coefficient.
    pub fn new(A: MontgomeryCoefficient) -> Curve {
        let a = A.as_fp2();
        let two = Fp2::from_fp(Fp::from_small(2));
        let four = Fp2::from_fp(Fp::from_small(4));
        Curve {
            A,
            A24: a + &two,
            C24: four,
        }
    }

    /// Construct from projective doubling constants (A₂₄ : C₂₄).
    ///
    /// This avoids field inversions when the curve is produced by an
    /// isogeny codomain computation, which naturally outputs projective
    /// constants. The affine coefficient A is recovered as
    /// A = 4·A₂₄/C₂₄ − 2.
    pub fn from_projective(A24: Fp2, C24: Fp2) -> Curve {
        let two = Fp2::from_fp(Fp::from_small(2));
        let four = Fp2::from_fp(Fp::from_small(4));
        let A = &(&(&four * &A24) * &C24.invert()) - &two;
        Curve {
            A: MontgomeryCoefficient(A),
            A24,
            C24,
        }
    }

    /// The Montgomery coefficient A.
    pub fn coefficient(&self) -> &MontgomeryCoefficient {
        &self.A
    }

    /// The projective doubling constants (A₂₄, C₂₄).
    pub fn projective_constants(&self) -> (&Fp2, &Fp2) {
        (&self.A24, &self.C24)
    }

    /// Compute the [j-invariant] j(E_A).
    ///
    /// j(E) = 256(A² − 3)³ / (A² − 4)
    ///
    /// [j-invariant]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
    pub fn j_invariant(&self) -> Fp2 {
        let A2 = self.A.as_fp2().square();
        let three = Fp2::from_fp(Fp::from_small(3));
        let four = Fp2::from_fp(Fp::from_small(4));
        let t = &A2 - &three;
        let t_sq = t.square();
        let t3 = &t_sq * &t;
        let denom = (&A2 - &four).invert();
        let c256 = Fp2::from_fp(Fp::from_small(256));
        let num = &c256 * &t3;
        &num * &denom
    }
}

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
/// [§8.2.1]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
#[derive(Copy, Clone, Debug)]
pub struct MontgomeryPoint {
    /// Projective X coordinate.
    pub X: Fp2,
    /// Projective Z coordinate (Z = 0 for the point at infinity).
    pub Z: Fp2,
    /// The curve this point lives on.
    curve: Curve,
}

impl MontgomeryPoint {
    /// Construct the identity (point at infinity) on the given curve.
    pub fn identity(curve: &Curve) -> MontgomeryPoint {
        MontgomeryPoint {
            X: Fp2::ONE,
            Z: Fp2::ZERO,
            curve: *curve,
        }
    }

    /// Construct from projective coordinates on the given curve.
    pub fn from_XZ(X: Fp2, Z: Fp2, curve: &Curve) -> MontgomeryPoint {
        MontgomeryPoint {
            X,
            Z,
            curve: *curve,
        }
    }

    /// Construct from an affine x-coordinate on the given curve.
    pub fn from_affine_x(x: Fp2, curve: &Curve) -> MontgomeryPoint {
        MontgomeryPoint {
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
    pub fn double(&self) -> MontgomeryPoint {
        let t0 = (&self.X + &self.Z).square();
        let t1 = (&self.X - &self.Z).square();
        let t2 = &t0 - &t1;
        let t1_c24 = &t1 * &self.curve.C24;
        let X2 = &t0 * &t1_c24;
        let Z2 = &t2 * &(&(&t2 * &self.curve.A24) + &t1_c24);
        MontgomeryPoint { X: X2, Z: Z2, curve: self.curve }
    }

    /// Compute self + other, given self − other.
    ///
    /// See [§8.2], Algorithm 8.4 (`xADD`).
    ///
    /// [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
    #[must_use]
    pub fn differential_add(&self, other: &MontgomeryPoint, difference: &MontgomeryPoint) -> MontgomeryPoint {
        let t0 = &(&self.X + &self.Z) * &(&other.X - &other.Z);
        let t1 = &(&self.X - &self.Z) * &(&other.X + &other.Z);
        let sum = (&t0 + &t1).square();
        let diff = (&t0 - &t1).square();
        MontgomeryPoint {
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
    /// [§8.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
    #[must_use]
    pub fn projective_difference(&self, other: &MontgomeryPoint) -> MontgomeryPoint {
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
        let B_XZ = &(&(&xpxq + &zpzq) * &(&xpzq + &zpxq))
            + &(&(A + A) * &(&xpxq * &zpzq));

        // B_ZZ = (X_P · Z_Q − Z_P · X_Q)²
        let B_ZZ = (&xpzq - &zpxq).square();

        // γ = (Z_P · Z_Q)²  (projective normalization factor)
        let gamma = zpzq.square();

        // Scale: B_XX *= γ, B_XZ *= γ, B_ZZ *= γ
        let B_XX = &gamma * &B_XX;
        let B_XZ = &gamma * &B_XZ;
        let B_ZZ = &gamma * &B_ZZ;

        // δ = SquareRoot(B_XZ² − B_XX · B_ZZ)
        let discriminant = &B_XZ.square() - &(&B_XX * &B_ZZ);
        let delta = discriminant.sqrt();

        // x_{PQ} = (δ + B_XZ : B_ZZ)
        MontgomeryPoint {
            X: &delta + &B_XZ,
            Z: B_ZZ,
            curve: self.curve,
        }
    }

    /// Scalar multiplication via the Montgomery ladder.
    ///
    /// Given a big-endian bit iterator for scalar n, computes \[n\]self.
    /// Constant-time in the value of n (the number of iterations is
    /// determined by the iterator length, which must be fixed and public).
    ///
    /// Implements `Ladder` ([§8.2], Algorithm 8.6).
    ///
    /// [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
    fn scalar_mul(&self, bits: impl Iterator<Item = bool>) -> MontgomeryPoint {
        let mut x0 = MontgomeryPoint::identity(&self.curve);
        let mut x1 = *self;

        let mut prev_bit = false;
        for cur_bit in bits {
            let swap: u8 = (prev_bit ^ cur_bit) as u8;
            MontgomeryPoint::conditional_swap(&mut x0, &mut x1, swap.into());
            differential_add_and_double(&mut x0, &mut x1, self);
            prev_bit = cur_bit;
        }
        MontgomeryPoint::conditional_swap(&mut x0, &mut x1, Choice::from(prev_bit as u8));
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
    P: &mut MontgomeryPoint,
    Q: &mut MontgomeryPoint,
    PmQ: &MontgomeryPoint,
) {
    let sum_P  = &P.X + &P.Z;
    let diff_P = &P.X - &P.Z;

    // xDBL
    let t0     = sum_P.square();
    let t1     = diff_P.square();
    let t2     = &t0 - &t1;
    let t1_c24 = &t1 * &P.curve.C24;
    let dbl_X  = &t0 * &t1_c24;
    let dbl_Z  = &t2 * &(&(&t2 * &P.curve.A24) + &t1_c24);

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

// ---------------------------------------------------------------------------
// Scalar multiplication via Mul trait
// ---------------------------------------------------------------------------

/// Multiply a point by a `u64` scalar. Constant-time in the scalar value.
impl Mul<u64> for &MontgomeryPoint {
    type Output = MontgomeryPoint;

    fn mul(self, scalar: u64) -> MontgomeryPoint {
        // Always iterate over all 64 bits for constant-time behavior.
        self.scalar_mul((0..64u32).rev().map(|i| (scalar >> i) & 1 == 1))
    }
}

impl Mul<u64> for MontgomeryPoint {
    type Output = MontgomeryPoint;

    fn mul(self, scalar: u64) -> MontgomeryPoint {
        &self * scalar
    }
}

impl MulAssign<u64> for MontgomeryPoint {
    fn mul_assign(&mut self, scalar: u64) {
        *self = &*self * scalar;
    }
}

// ---------------------------------------------------------------------------
// Constant-time traits
// ---------------------------------------------------------------------------

impl ConditionallySelectable for MontgomeryPoint {
    fn conditional_select(a: &MontgomeryPoint, b: &MontgomeryPoint, choice: Choice) -> MontgomeryPoint {
        MontgomeryPoint {
            X: Fp2::conditional_select(&a.X, &b.X, choice),
            Z: Fp2::conditional_select(&a.Z, &b.Z, choice),
            curve: a.curve, // assumed to be on the same curve
        }
    }
}

impl ConstantTimeEq for MontgomeryPoint {
    /// Two projective points are equal iff X₁Z₂ = X₂Z₁.
    fn ct_eq(&self, other: &MontgomeryPoint) -> Choice {
        (&self.X * &other.Z).ct_eq(&(&other.X * &self.Z))
    }
}

impl PartialEq for MontgomeryPoint {
    fn eq(&self, other: &MontgomeryPoint) -> bool {
        self.ct_eq(other).into()
    }
}

impl Eq for MontgomeryPoint {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubling_identity_is_identity() {
        let id = MontgomeryPoint::identity(&Curve::E0);
        let dbl = id.double();
        assert!(bool::from(dbl.is_identity()));
    }

    #[test]
    fn mul_by_one() {
        let P = MontgomeryPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &Curve::E0);
        assert_eq!(&P * 1u64, P);
    }

    #[test]
    fn mul_by_two_equals_doubling() {
        let P = MontgomeryPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &Curve::E0);
        assert_eq!(&P * 2u64, P.double());
    }

    #[test]
    fn mul_by_three_equals_double_plus_add() {
        let P = MontgomeryPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &Curve::E0);
        let triple = &P * 3u64;
        let dbl = P.double();
        let triple_add = dbl.differential_add(&P, &P);
        assert_eq!(triple, triple_add);
    }

    #[test]
    fn mul_by_zero_is_identity() {
        let P = MontgomeryPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &Curve::E0);
        assert!(bool::from((&P * 0u64).is_identity()));
    }

    #[test]
    fn j_invariant_of_e0() {
        // j(E₀) = 256(0 − 3)³ / (0 − 4) = 1728
        let j = Curve::E0.j_invariant();
        assert_eq!(j, Fp2::from_fp(Fp::from_small(1728)));
    }

    #[test]
    fn torsion_basis_holds_points() {
        use crate::curves::TorsionBasis;

        let curve = Curve::E0;
        let R = MontgomeryPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &curve);
        let S = MontgomeryPoint::from_affine_x(Fp2::from_fp(Fp::from_small(7)), &curve);
        let RS = MontgomeryPoint::from_affine_x(Fp2::from_fp(Fp::from_small(11)), &curve);

        let basis = TorsionBasis::new(R, S, RS);
        assert_eq!(basis.R, R);
        assert_eq!(basis.S, S);
        assert_eq!(basis.RS, RS);
    }
}
