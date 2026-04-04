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

use core::ops::{Mul, Neg};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use super::scalar::Scalar;
use crate::fields::{fp::Fp, fp2::Fp2};

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
pub struct Coefficient(Fp2);

/// An affine x-coordinate on a Montgomery curve, i.e. x = X/Z ∈ F_{p²}.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AffineX(Fp2);

impl AffineX {
    /// The underlying F_{p²} element.
    pub fn as_fp2(&self) -> &Fp2 {
        &self.0
    }
}

impl From<Fp2> for AffineX {
    fn from(x: Fp2) -> Self {
        Self(x)
    }
}

impl ConditionallySelectable for AffineX {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self(Fp2::conditional_select(&a.0, &b.0, choice))
    }
}

impl ConditionallySelectable for Coefficient {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self(Fp2::conditional_select(&a.0, &b.0, choice))
    }
}

impl Coefficient {
    /// A = 0, the coefficient of the starting curve E₀.
    pub const ZERO: Coefficient = Coefficient(Fp2::ZERO);

    /// The underlying F_{p²} element.
    pub fn as_fp2(&self) -> &Fp2 {
        &self.0
    }

    /// Encode as bytes (delegates to F_{p²} encoding).
    pub fn to_bytes(&self) -> [u8; crate::params::CURVE_ENCODED_BYTES] {
        self.0.to_bytes()
    }

    /// Decode from bytes.
    pub fn from_bytes(bytes: &[u8; crate::params::CURVE_ENCODED_BYTES]) -> Coefficient {
        Coefficient(Fp2::from_bytes(bytes))
    }
}

impl From<Fp2> for Coefficient {
    fn from(a: Fp2) -> Self {
        Coefficient(a)
    }
}

impl From<Coefficient> for Fp2 {
    fn from(a: Coefficient) -> Fp2 {
        a.0
    }
}

/// A Montgomery curve E_A : y² = x³ + Ax² + x over F_{p²}.
///
/// Stores the [`Coefficient`] A together with precomputed
/// projective doubling constants (A₂₄, C₂₄) = (A + 2C, 4C), and the
/// projective Montgomery coefficient (A : C).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Curve {
    /// Affine Montgomery coefficient A (= Aproj/Cproj).
    A: Coefficient,
    /// Projective Montgomery coefficient A (numerator).
    Aproj: Fp2,
    /// Projective Montgomery coefficient C (denominator).
    Cproj: Fp2,
    /// Doubling constant A₂₄ = A + 2C.
    A24: Fp2,
    /// Doubling constant C₂₄ = 4C.
    C24: Fp2,
}

impl Curve {
    /// The starting curve E₀ : y² = x³ + x (A = 0).
    ///
    /// This is the supersingular curve used as the base in SQIsign.
    pub const E0: Curve = Curve {
        A: Coefficient::ZERO,
        Aproj: Fp2::ZERO,
        Cproj: Fp2::ONE,
        A24: Fp2::new(Fp::TWO, Fp::ZERO),
        C24: Fp2::new(Fp::FOUR, Fp::ZERO),
    };

    /// Construct from the affine Montgomery coefficient A (C = 1).
    pub fn from_affine(A: Coefficient) -> Curve {
        let a = *A.as_fp2();
        let two = Fp2::from_fp(Fp::from_small(2));
        let four = Fp2::from_fp(Fp::from_small(4));
        Curve {
            A,
            Aproj: a,
            Cproj: Fp2::ONE,
            A24: &a + &two,
            C24: four,
        }
    }

    /// Construct from projective Montgomery coefficient (A : C).
    ///
    /// The affine coefficient is A/C (requires one inversion).
    pub fn from_projective_coeff(Aproj: Fp2, Cproj: Fp2) -> Curve {
        let c_inv = Cproj.invert();
        let a_affine = &Aproj * &c_inv;
        let two = Fp2::from_fp(Fp::from_small(2));
        let four = Fp2::from_fp(Fp::from_small(4));
        let two_c = &Cproj + &Cproj;
        Curve {
            A: Coefficient(a_affine),
            Aproj,
            Cproj,
            A24: &Aproj + &two_c,
            C24: &four * &Cproj,
        }
    }

    /// Construct from projective doubling constants (A₂₄ : C₂₄).
    ///
    /// Used when the curve is produced by an isogeny codomain
    /// computation, which naturally outputs doubling constants.
    /// The affine coefficient is recovered as A = 4·A₂₄/C₂₄ − 2.
    pub fn from_doubling_constants(A24: Fp2, C24: Fp2) -> Curve {
        let two = Fp2::from_fp(Fp::from_small(2));
        let four = Fp2::from_fp(Fp::from_small(4));
        let A = &(&(&four * &A24) * &C24.invert()) - &two;
        // (A : C) from (A24 : C24): A24 = A+2C, C24 = 4C
        // so (A : C) = (4·A24 − 2·C24 : C24).
        let two_c24 = &C24 + &C24;
        let four_a24 = {
            let t = &A24 + &A24;
            &t + &t
        };
        Curve {
            A: Coefficient(A),
            Aproj: &four_a24 - &two_c24,
            Cproj: C24,
            A24,
            C24,
        }
    }
}

impl From<Coefficient> for Curve {
    fn from(A: Coefficient) -> Self {
        Self::from_affine(A)
    }
}

impl Curve {
    /// Normalize the doubling constants to (A₂₄/C₂₄ : 1).
    ///
    /// The C reference (`ec_normalize_curve_and_A24`) normalizes
    /// the projective constants before torsion basis generation.
    /// This ensures that the Montgomery ladder produces the same
    /// projective representative as the C reference, which matters
    /// for differential addition consistency in subsequent operations.
    pub fn normalize(&mut self) {
        if self.C24 != Fp2::ONE {
            let inv = self.C24.invert();
            self.A24 = &self.A24 * &inv;
            self.C24 = Fp2::ONE;
        }
    }

    /// Check if the doubling constants are normalized (C₂₄ = 1).
    pub fn is_normalized(&self) -> bool {
        self.C24 == Fp2::ONE
    }

    /// The Montgomery coefficient A.
    pub fn coefficient(&self) -> &Coefficient {
        &self.A
    }

    /// The projective doubling constants (A₂₄, C₂₄).
    pub fn projective_constants(&self) -> (&Fp2, &Fp2) {
        (&self.A24, &self.C24)
    }

    /// The projective Montgomery coefficient (A : C).
    ///
    /// Isogeny codomains produce curves with C ≠ 1; use this for
    /// projective arithmetic (e.g., [`Isomorphism`]) instead of
    /// [`coefficient`](Self::coefficient).
    pub fn projective_coefficient(&self) -> (&Fp2, &Fp2) {
        (&self.Aproj, &self.Cproj)
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
    pub fn to_affine_x(&self) -> AffineX {
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
        let t1_c24 = &t1 * &self.curve.C24;
        let X2 = &t0 * &t1_c24;
        let Z2 = &t2 * &(&(&t2 * &self.curve.A24) + &t1_c24);
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
    /// [§8.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
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

        // Normalize so the discriminant is a fourth power in Fp,
        // making the Fp2 square root deterministic. The C reference
        // (`difference_point`, basis.c:48-64) uses the factor
        // C · conj(C)² · conj(Z_P)² · conj(Z_Q)². With C = 1 this
        // reduces to conj(Z_P)² · conj(Z_Q)².
        //
        // NOTE: We previously used γ = (Z_P · Z_Q)², which is WRONG
        // because conj(z)² ≠ z² for complex z. The conjugate
        // normalization ensures B_XZ² − B_XX·B_ZZ lies in Fp (up to
        // a fourth-power factor), so the square root is well-defined.
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

    /// Clear the odd cofactor: computes [c]P where c = 5 = (p+1)/2^f.
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
        // C ref, which is critical: subsequent operations (ladder3pt,
        // difference_point) are sensitive to the projective
        // representative, not just the affine x-coordinate.
        //
        // The cofactor 5 is public data, so variable-time is fine.
        let mut r0 = ProjectiveXOnlyPoint::identity(&self.curve);
        let mut r1 = *self;

        // 5 = 0b101, kbits = 3. Process MSB to LSB with differential swap.
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

/// Multiply a point by a [`Scalar`]. Constant-time in the scalar value.
impl Mul<&Scalar> for &ProjectiveXOnlyPoint {
    type Output = ProjectiveXOnlyPoint;

    fn mul(self, scalar: &Scalar) -> ProjectiveXOnlyPoint {
        self.scalar_mul(scalar)
    }
}

// ---------------------------------------------------------------------------
// Constant-time traits
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubling_identity_is_identity() {
        let id = ProjectiveXOnlyPoint::identity(&Curve::E0);
        let dbl = id.double();
        assert!(bool::from(dbl.is_identity()));
    }

    #[test]
    fn mul_by_one() {
        let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &Curve::E0);
        assert_eq!(P.scalar_mul(&Scalar::from_u64(1)), P);
    }

    #[test]
    fn mul_by_two_equals_doubling() {
        let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &Curve::E0);
        assert_eq!(P.scalar_mul(&Scalar::from_u64(2)), P.double());
    }

    #[test]
    fn mul_by_three_equals_double_plus_add() {
        let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &Curve::E0);
        let triple = P.scalar_mul(&Scalar::from_u64(3));
        let dbl = P.double();
        let triple_add = dbl.differential_add(&P, &P);
        assert_eq!(triple, triple_add);
    }

    #[test]
    fn mul_by_zero_is_identity() {
        let P = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &Curve::E0);
        assert!(bool::from(P.scalar_mul(&Scalar::from_u64(0)).is_identity()));
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
        let R = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(3)), &curve);
        let S = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(7)), &curve);
        let RS = ProjectiveXOnlyPoint::from_affine_x(Fp2::from_fp(Fp::from_small(11)), &curve);

        let basis = TorsionBasis::new(R, S, RS);
        assert_eq!(basis.R, R);
        assert_eq!(basis.S, S);
        assert_eq!(basis.RS, RS);
    }

    /// Verify that Jacobian doubling produces the same affine x
    /// as Montgomery x-only doubling.
    #[test]
    fn jacobian_double_matches_montgomery() {
        let curve = Curve::E0;
        let p = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, &curve);

        // Montgomery double.
        let p2_mont = p.double();
        let p2_x_mont = &p2_mont.X * &p2_mont.Z.invert();

        // Jacobian: lift, double, convert, check.
        let A = Fp2::from(*curve.coefficient().as_fp2());
        let y = recover_y(&(&p.X * &p.Z.invert()), &A).expect("P₀ should be on E₀");
        let p_jac = JacobianPoint::new(&p.X * &p.Z.invert(), y, Fp2::ONE, &curve);
        let p2_jac = p_jac.double();

        // Convert Jacobian to affine: x_aff = x / z².
        let z2_inv = p2_jac.Z.square().invert();
        let p2_x_jac = &p2_jac.X * &z2_inv;

        assert_eq!(
            p2_x_mont, p2_x_jac,
            "Jacobian double should match Montgomery double (affine x)"
        );
    }

    /// Verify that lift_basis produces valid Jacobian points that
    /// convert back to the correct Montgomery x-coordinates.
    #[test]
    fn lift_basis_round_trip() {
        let curve = Curve::E0;
        let p = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, &curve);
        let q = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_Q_X, &curve);
        let pmq = p.projective_difference(&q);

        let (p_jac, q_jac) =
            lift_basis(&p, &q, &pmq, &curve).expect("lift_basis should succeed on E₀");

        // Check P: jac_to_xz(P_jac) should have same affine x as P.
        let p_back: ProjectiveXOnlyPoint = p_jac.into();
        let p_x = &p_back.X * &p_back.Z.invert();
        let p_orig_x = &p.X * &p.Z.invert();
        assert_eq!(
            p_x, p_orig_x,
            "P: lift + jac_to_xz should preserve affine x"
        );

        // Check Q: jac_to_xz(Q_jac) should have same affine x as Q.
        let q_back: ProjectiveXOnlyPoint = q_jac.into();
        let q_x = &q_back.X * &q_back.Z.invert();
        let q_orig_x = &q.X * &q.Z.invert();
        assert_eq!(
            q_x, q_orig_x,
            "Q: lift + jac_to_xz should preserve affine x"
        );

        // Check P_jac is on curve.
        let A = Fp2::from(*curve.coefficient().as_fp2());
        let z_inv = p_jac.Z.invert();
        let xa = &p_jac.X * &z_inv.square();
        let ya = &p_jac.Y * &(&z_inv.square() * &z_inv);
        let lhs = ya.square();
        let xa2 = xa.square();
        let rhs = &(&(&xa2 * &xa) + &(&A * &xa2)) + &xa;
        assert_eq!(lhs, rhs, "P_jac should be on curve");

        // Check Q_jac is on curve.
        let z_inv = q_jac.Z.invert();
        let xa = &q_jac.X * &z_inv.square();
        let ya = &q_jac.Y * &(&z_inv.square() * &z_inv);
        let lhs = ya.square();
        let xa2 = xa.square();
        let rhs = &(&(&xa2 * &xa) + &(&A * &xa2)) + &xa;
        assert_eq!(lhs, rhs, "Q_jac should be on curve");
    }

    /// Verify that jac_to_xz (From<JacobianPoint>) round-trips correctly.
    #[test]
    fn jac_to_xz_round_trip() {
        let curve = Curve::E0;
        let p = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, &curve);

        let A = Fp2::from(*curve.coefficient().as_fp2());
        let y = recover_y(&(&p.X * &p.Z.invert()), &A).expect("P₀ should be on E₀");
        let p_jac = JacobianPoint::new(&p.X * &p.Z.invert(), y, Fp2::ONE, &curve);

        // jac_to_xz: (x, z) → (x, z²). For z=1, this is (x, 1).
        let p_mont: ProjectiveXOnlyPoint = p_jac.into();
        let p_x = &p_mont.X * &p_mont.Z.invert();
        let orig_x = &p.X * &p.Z.invert();
        assert_eq!(p_x, orig_x, "jac_to_xz should preserve affine x");
    }
}

// ---------------------------------------------------------------------------
// Isomorphisms between Montgomery curves (§2.2.1.1, §8.2.2)
// ---------------------------------------------------------------------------

/// An isomorphism between two Montgomery curves with the same j-invariant.
///
/// Implements [IsomorphismMontgomeryCurves][Alg. 8.9] from the spec.
/// Works entirely in projective coordinates — no field inversions.
///
/// The isomorphism maps x-only projective points via precomputed
/// projective constants (λ_x, λ_z, A, C, A', C') so that each
/// point evaluation is 4M + 1a.
///
/// [Alg. 8.9]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.9
#[derive(Copy, Clone, Debug)]
pub struct Isomorphism {
    /// λ_x = (2A'³ − 9A'C'²)(3C³ − A²C).
    lambda_x: Fp2,
    /// λ_z = (2A³ − 9AC²)(3C'³ − A'²C').
    lambda_z: Fp2,
    /// Precomputed: 3CC'.
    three_cc_prime: Fp2,
    /// AC' (source A · target C).
    ac_prime: Fp2,
    /// A'C (target A · source C).
    a_prime_c: Fp2,
    target: Curve,
}

impl Isomorphism {
    /// Compute the isomorphism from `source` to `target`.
    ///
    /// Both curves must have the same j-invariant. Returns `None` if
    /// λ_x = 0 or λ_z = 0 (degenerate case, see Remark 1 in the spec).
    ///
    /// Implements lines 1–3 of [Algorithm 8.9][Alg. 8.9].
    ///
    /// [Alg. 8.9]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.9
    #[must_use]
    pub fn new(source: &Curve, target: &Curve) -> Option<Self> {
        let (a_ref, c_ref) = source.projective_coefficient();
        let (a, c) = (*a_ref, *c_ref);
        let (ap_ref, cp_ref) = target.projective_coefficient();
        let (a_prime, c_prime) = (*ap_ref, *cp_ref);

        // Line 1: λ_x ← (2A'³ − 9A'C'²)(3C³ − A²C)
        let lambda_x = {
            let a_prime_c_prime_sq = &a_prime * &c_prime.square();
            let nine_a_prime_c_prime_sq = {
                let t = &a_prime_c_prime_sq + &a_prime_c_prime_sq;
                let t4 = &t + &t;
                let t8 = &t4 + &t4;
                &t8 + &a_prime_c_prime_sq
            };
            let two_a_prime_cubed = {
                let t = &a_prime.square() * &a_prime;
                &t + &t
            };
            let left = &two_a_prime_cubed - &nine_a_prime_c_prime_sq;

            let c_cubed = &c.square() * &c;
            let three_c_cubed = {
                let t = &c_cubed + &c_cubed;
                &t + &c_cubed
            };
            let a_sq_c = &a.square() * &c;
            let right = &three_c_cubed - &a_sq_c;

            &left * &right
        };

        // Line 2: λ_z ← (2A³ − 9AC²)(3C'³ − A'²C')
        let lambda_z = {
            let a_c_sq = &a * &c.square();
            let nine_a_c_sq = {
                let t = &a_c_sq + &a_c_sq;
                let t4 = &t + &t;
                let t8 = &t4 + &t4;
                &t8 + &a_c_sq
            };
            let two_a_cubed = {
                let t = &a.square() * &a;
                &t + &t
            };
            let left = &two_a_cubed - &nine_a_c_sq;

            let c_prime_cubed = &c_prime.square() * &c_prime;
            let three_c_prime_cubed = {
                let t = &c_prime_cubed + &c_prime_cubed;
                &t + &c_prime_cubed
            };
            let a_prime_sq_c_prime = &a_prime.square() * &c_prime;
            let right = &three_c_prime_cubed - &a_prime_sq_c_prime;

            &left * &right
        };

        // Line 3: degeneracy check
        if lambda_x == Fp2::ZERO || lambda_z == Fp2::ZERO {
            return None;
        }

        // Precompute constants for eval (lines 5–8).
        let cc_prime = &c * &c_prime;
        let three_cc_prime = {
            let t = &cc_prime + &cc_prime;
            &t + &cc_prime
        };

        Some(Self {
            lambda_x,
            lambda_z,
            three_cc_prime,
            ac_prime: &a * &c_prime,
            a_prime_c: &a_prime * &c,
            target: *target,
        })
    }

    /// Apply this isomorphism to a projective x-only point.
    ///
    /// Implements lines 5–6 of [Algorithm 8.9][Alg. 8.9]:
    /// ```text
    /// X' ← λ_x(3X·CC' + AC'·Z) − λ_z·A'C·Z
    /// Z' ← 3λ_z·CC'·Z
    /// ```
    ///
    /// [Alg. 8.9]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.9
    #[must_use]
    pub fn eval(&self, p: &ProjectiveXOnlyPoint) -> ProjectiveXOnlyPoint {
        // X' = λ_x·(3CC'·X + AC'·Z) − λ_z·A'C·Z
        let term = &(&self.three_cc_prime * &p.X) + &(&self.ac_prime * &p.Z);
        let new_x = &(&self.lambda_x * &term) - &(&self.lambda_z * &(&self.a_prime_c * &p.Z));
        // Z' = 3·λ_z·CC'·Z
        let new_z = &self.lambda_z * &(&self.three_cc_prime * &p.Z);
        ProjectiveXOnlyPoint::from_XZ(new_x, new_z, &self.target)
    }
}

// ---------------------------------------------------------------------------
// Jacobian points (for the (2,2)-isogeny gluing step)
// ---------------------------------------------------------------------------

/// A point on a Montgomery curve in Jacobian coordinates (x, y, z).
///
/// Represents the affine point (x/z², y/z³) on E_A : y² = x³ + Ax² + x.
///
/// Jacobian coordinates are needed for the gluing step of the
/// (2,2)-isogeny chain, where the y-coordinate is required to
/// compute the cross-addition components ([`jac_to_xz_add_components`]
/// in the C reference). Montgomery x-only arithmetic is insufficient
/// because it cannot distinguish P+Q from P−Q.
///
/// **Naming:** This is a dim-1 elliptic curve point, NOT the dim-2
/// theta-coordinate `JacobianPoint` in [`crate::surfaces`]. The name
/// collision is unfortunate; we keep both because they serve different
/// layers (curves vs surfaces).
///
/// # Coordinate representation TODOs
///
/// TODO: The `x`, `y` fields are bare `Fp2` — they should eventually
/// be newtypes (`AffineX`, `AffineY`) to prevent mixing with the
/// Montgomery coefficient `A` or other `Fp2` values. Similarly, the
/// Montgomery coefficient `A` should be a newtype distinct from field
/// elements. This is tracked as a future type-safety improvement.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JacobianPoint {
    pub(crate) X: Fp2,
    pub(crate) Y: Fp2,
    pub(crate) Z: Fp2,
    curve: Curve,
}

impl JacobianPoint {
    /// Create from coordinates and a curve.
    pub fn new(X: Fp2, Y: Fp2, Z: Fp2, curve: &Curve) -> Self {
        Self {
            X,
            Y,
            Z,
            curve: *curve,
        }
    }

    /// The curve this point lives on.
    pub fn curve(&self) -> &Curve {
        &self.curve
    }

    /// Double this Jacobian point on y² = x³ + Ax² + x.
    ///
    /// **IMPORTANT:** This uses the C reference's unified add-or-double
    /// formula (`ec_add_jac_v2` in ec_jac.c:228-297), NOT standard
    /// Jacobian doubling. The C ref's formula produces z₃ = 2y·z²
    /// (instead of standard z₃ = 2y·z), giving a different projective
    /// representative after `jac_to_xz`. This matters because the
    /// gluing's `product_to_theta` is sensitive to the projective
    /// representative, not just the affine ratio.
    ///
    /// The SQIsign spec does NOT describe Jacobian doubling — only
    /// x-only Montgomery arithmetic (§8.2). These formulas come
    /// entirely from the C reference.
    ///
    /// The C ref computes for P = Q (doubling case):
    /// ```text
    /// dx = 2y₁           (tangent denominator)
    /// dy = z₁·M           (tangent numerator, M = 3x₁² + z₁²(2Ax₁ + z₁²))
    /// u₁ = x₁·z₁²         (= u₂ since P = Q)
    /// v₁ = y₁·z₁³
    /// t0 = z₁²            (= z₁·z₂ since P = Q)
    ///
    /// x₃ = dy² − dx²·(A·z₁⁴ + 2·x₁·z₁²)
    /// y₃ = dy·(u₁·dx² − x₃) − v₁·dx³
    /// z₃ = dx·z₁² = 2y₁·z₁²
    /// ```
    #[must_use]
    pub fn double(&self) -> JacobianPoint {
        let A = *self.curve.coefficient().as_fp2();

        let zz = self.Z.square(); // z₁²
        let zzzz = zz.square(); // z₁⁴
        let xx = self.X.square(); // x₁²

        // M = 3x₁² + z₁²·(2A·x₁ + z₁²)
        let two_a = &A + &A;
        let two_a_x = &two_a * &self.X;
        let inner = &two_a_x + &zz;
        let m_term = &inner * &zz; // z₁²·(2Ax₁ + z₁²)
        let three_xx = &(&xx + &xx) + &xx;
        let m = &three_xx + &m_term; // M = 3x₁² + z₁²(2Ax₁ + z₁²)

        // dx = 2y₁, dy = z₁·M
        let dx = &self.Y + &self.Y; // 2y₁
        let dy = &self.Z * &m; // z₁·M

        // Precomputations
        let dx_sq = dx.square(); // 4y₁²
        let dy_sq = dy.square(); // z₁²·M²
        let u1 = &self.X * &zz; // x₁·z₁²
        let v1 = &self.Y * &(&zz * &self.Z); // y₁·z₁³

        // x₃ = dy² − dx²·(A·z₁⁴ + u₁ + u₁)
        let x3 = {
            let a_zzzz = &A * &zzzz; // A·z₁⁴
            let sum = &(&a_zzzz + &u1) + &u1; // A·z₁⁴ + 2·x₁·z₁²
            &dy_sq - &(&dx_sq * &sum)
        };

        // y₃ = dy·(u₁·dx² − x₃) − v₁·dx³
        let y3 = {
            let u1_dx_sq = &u1 * &dx_sq;
            let dx_cubed = &dx_sq * &dx;
            &(&dy * &(&u1_dx_sq - &x3)) - &(&v1 * &dx_cubed)
        };

        // z₃ = dx·z₁² = 2y₁·z₁²  (NOT 2y₁·z₁ like standard Jacobian!)
        let z3 = &dx * &zz;

        JacobianPoint {
            X: x3,
            Y: y3,
            Z: z3,
            curve: self.curve,
        }
    }
}

impl JacobianPoint {
    /// Compute the x-only Montgomery projective coordinates of P + Q
    /// and P − Q from two Jacobian points.
    ///
    /// Returns `(x(P+Q), x(P-Q))` as `ProjectiveXOnlyPoint`s.
    ///
    /// Uses the full Jacobian addition formula (`ec_jac.c:305`) to
    /// deterministically distinguish P+Q from P−Q (impossible with
    /// x-only arithmetic alone).
    ///
    /// This is used by
    /// [`ChangeOfBasis`](crate::curves::pairing::change_of_basis)
    /// to compute the cross-pairing sum points.
    #[must_use]
    pub fn x_add_sub(&self, other: &Self) -> (ProjectiveXOnlyPoint, ProjectiveXOnlyPoint) {
        let a = *self.curve.coefficient().as_fp2();

        let t0 = self.Z.square(); // z1²
        let t1 = other.Z.square(); // z2²
        let t2 = &self.X * &t1; // x1·z2²
        let t3 = &t0 * &other.X; // z1²·x2
        let mut t4 = &self.Y * &other.Z; // y1·z2
        t4 = &t4 * &t1; // y1·z2³
        let mut t5 = &self.Z * &other.Y; // z1·y2
        t5 = &t5 * &t0; // z1³·y2
        let t0 = &t0 * &t1; // (z1·z2)²
        let t6 = &t4 * &t5; // (z1·z2)³·y1·y2
        let v = &t6 + &t6; // 2·(z1·z2)³·y1·y2

        let t4_sq = t4.square();
        let t5_sq = t5.square();
        let sum_y2 = &t4_sq + &t5_sq;
        let sum_x = &t2 + &t3;
        let lambda = &t2 - &t3;
        let lambda_sq = lambda.square();
        let a_t0 = &a * &t0;
        let gamma = &(&sum_x + &a_t0) * &lambda_sq;

        let u = &sum_y2 - &gamma;
        let w = &lambda_sq * &t0;

        // x(P+Q) = (u + v) : w,  x(P-Q) = (u - v) : w
        let x_add = ProjectiveXOnlyPoint::from_XZ(&u + &v, w, &self.curve);
        let x_sub = ProjectiveXOnlyPoint::from_XZ(&u - &v, w, &self.curve);
        (x_add, x_sub)
    }
}

impl Neg for JacobianPoint {
    type Output = Self;
    /// −(x, y, z) = (x, −y, z).
    fn neg(self) -> Self {
        Self {
            X: self.X,
            Y: -&self.Y,
            Z: self.Z,
            curve: self.curve,
        }
    }
}

impl Neg for &JacobianPoint {
    type Output = JacobianPoint;
    fn neg(self) -> JacobianPoint {
        JacobianPoint {
            X: self.X,
            Y: -&self.Y,
            Z: self.Z,
            curve: self.curve,
        }
    }
}

/// Convert a Jacobian point to Montgomery projective (X:Z) = (x : z²).
///
/// This is the C reference's `jac_to_xz` (`ec_jac.c:34`). The
/// projective representative `(x, z²)` is NOT the same as `(X, Z)`
/// from Montgomery doubling — the balanced strategy must use Jacobian
/// doubling to produce the correct representative for the gluing's
/// `product_to_theta` computation.
impl From<JacobianPoint> for ProjectiveXOnlyPoint {
    fn from(jac: JacobianPoint) -> ProjectiveXOnlyPoint {
        let z_sq = jac.Z.square();
        ProjectiveXOnlyPoint::from_XZ(jac.X, z_sq, &jac.curve)
    }
}

impl From<&JacobianPoint> for ProjectiveXOnlyPoint {
    fn from(jac: &JacobianPoint) -> ProjectiveXOnlyPoint {
        let z_sq = jac.Z.square();
        ProjectiveXOnlyPoint::from_XZ(jac.X, z_sq, &jac.curve)
    }
}

/// Recover the y-coordinate of a point on E_A : y² = x³ + Ax² + x.
///
/// Given the affine x-coordinate, computes y = √(x³ + Ax² + x).
/// Returns `None` if x³ + Ax² + x is not a square in Fp2.
///
/// Corresponds to `ec_recover_y` in the C reference (`basis.c:7`).
pub fn recover_y(x: &Fp2, A: &Fp2) -> Option<Fp2> {
    let x2 = x.square();
    let rhs = &(&(&x2 * x) + &(&x2 * A)) + x; // x³ + Ax² + x
    if bool::from(rhs.is_square()) {
        Some(rhs.sqrt())
    } else {
        None
    }
}

/// Lift a Montgomery basis (P, Q, P−Q) to Jacobian coordinates.
///
/// Given P = (X_P : Z_P), Q = (X_Q : Z_Q), and PmQ = (X_{P-Q} : Z_{P-Q})
/// on a Montgomery curve E_A, computes (P_jac, Q_jac) with full (x,y,z)
/// coordinates.
///
/// P is normalized internally. Uses the Okeya-Sakurai algorithm
/// to recover Q's y-coordinate from P's y-coordinate and the difference
/// point.
///
/// Corresponds to `lift_basis_normalized` in the C reference (`basis.c:79`).
///
/// Returns `None` if y-recovery fails (x not on curve).
pub fn lift_basis(
    P: &ProjectiveXOnlyPoint,
    Q: &ProjectiveXOnlyPoint,
    PmQ: &ProjectiveXOnlyPoint,
    curve: &Curve,
) -> Option<(JacobianPoint, JacobianPoint)> {
    let A = *curve.coefficient().as_fp2();

    // Normalize P: compute affine x_P = X_P / Z_P.
    let z_inv = P.Z.invert();
    let x_P = &P.X * &z_inv;

    // Recover y_P = sqrt(x_P³ + A·x_P² + x_P).
    let y_P = recover_y(&x_P, &A)?;

    let P_jac = JacobianPoint::new(x_P, y_P, Fp2::ONE, curve);

    // Okeya-Sakurai: recover y_Q from x_P, y_P, x_Q, z_Q, x_{P-Q}, z_{P-Q}.
    // C reference: basis.c:91-116.
    let v1 = &x_P * &Q.Z;
    let v2 = &Q.X + &v1;
    let v3 = {
        let diff = &Q.X - &v1;
        let diff_sq = diff.square();
        &diff_sq * &PmQ.X
    };
    let two_A = &A + &A;
    let v1_new = &two_A * &Q.Z;
    let v2 = &v2 + &v1_new;
    let v4 = &(&x_P * &Q.X) + &Q.Z;
    let v2 = &v2 * &v4;
    let v1_new = &v1_new * &Q.Z;
    let v2 = &v2 - &v1_new;
    let v2 = &v2 * &PmQ.Z;
    let y_Q_num = &v3 - &v2;
    let two_yP = &y_P + &y_P;
    let v1 = &(&two_yP * &Q.Z) * &PmQ.Z;

    // Q in Jacobian: (x_Q·v1·z_Q : y_Q_num·(z_Q·v1)² : z_Q·v1)
    //
    // The C reference (basis.c:110-116) squares Q->z (= Z_Q·v1)
    // to compute y, NOT the original v1. This gives:
    //   z = Z_Q · v1
    //   y = y_num · z²  (where z = Z_Q · v1)
    //   x = (X_Q · v1) · z
    let x_Q_tmp = &Q.X * &v1;
    let z_Q_jac = &Q.Z * &v1;
    let z_Q_jac_sq = z_Q_jac.square();
    let y_Q_jac = &y_Q_num * &z_Q_jac_sq;
    let x_Q_jac = &x_Q_tmp * &z_Q_jac;

    let Q_jac = JacobianPoint::new(x_Q_jac, y_Q_jac, z_Q_jac, curve);

    Some((P_jac, Q_jac))
}
