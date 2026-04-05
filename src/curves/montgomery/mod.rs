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

mod isomorphism;
mod jacobian;
mod point;

#[cfg(test)]
mod tests;

pub use isomorphism::Isomorphism;
pub use jacobian::{JacobianPoint, lift_basis, recover_y};
pub use point::ProjectiveXOnlyPoint;
pub(crate) use point::differential_add_and_double;
use subtle::{Choice, ConditionallySelectable};

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

impl ConditionallySelectable for Coefficient {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self(Fp2::conditional_select(&a.0, &b.0, choice))
    }
}

/// Projective Montgomery coefficient `(A : C)` where `Cy² = x³ + Ax² + x`.
///
/// This is the general representation of a Montgomery curve. Isogeny
/// codomains produce curves with `C ≠ 1`. The affine coefficient is
/// `A/C` but computing it requires a field inversion.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ProjectiveCoefficient {
    /// Numerator A.
    pub A: Fp2,
    /// Denominator C.
    pub C: Fp2,
}

impl ProjectiveCoefficient {
    /// The underlying `(A, C)` pair.
    pub fn as_pair(&self) -> (&Fp2, &Fp2) {
        (&self.A, &self.C)
    }
}

impl From<Coefficient> for ProjectiveCoefficient {
    /// Affine → projective: `(A, 1)`.
    fn from(a: Coefficient) -> Self {
        Self {
            A: *a.as_fp2(),
            C: Fp2::ONE,
        }
    }
}

/// Projective doubling constants `(A₂₄ : C₂₄) = (A + 2C : 4C)`.
///
/// Isogeny codomain computations ([`TwoIsogeny`], [`FourIsogeny`])
/// naturally produce these. They are consumed by the Montgomery
/// ladder's doubling and differential addition formulas.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DoublingConstants {
    /// A₂₄ = A + 2C.
    pub A24: Fp2,
    /// C₂₄ = 4C.
    pub C24: Fp2,
}

impl DoublingConstants {
    /// The underlying `(A₂₄, C₂₄)` pair.
    pub fn as_pair(&self) -> (&Fp2, &Fp2) {
        (&self.A24, &self.C24)
    }

    /// Normalize to `(A₂₄/C₂₄ : 1)`.
    pub fn normalize(&mut self) {
        if self.C24 != Fp2::ONE {
            let inv = self.C24.invert();
            self.A24 = &self.A24 * &inv;
            self.C24 = Fp2::ONE;
        }
    }

    /// Check if normalized (C₂₄ = 1).
    pub fn is_normalized(&self) -> bool {
        self.C24 == Fp2::ONE
    }
}

impl From<ProjectiveCoefficient> for DoublingConstants {
    /// `(A : C) → (A + 2C : 4C)`.
    fn from(pc: ProjectiveCoefficient) -> Self {
        let two_c = &pc.C + &pc.C;
        let four = Fp2::from_fp(Fp::from_small(4));
        Self {
            A24: &pc.A + &two_c,
            C24: &four * &pc.C,
        }
    }
}

impl From<DoublingConstants> for ProjectiveCoefficient {
    /// `(A₂₄ : C₂₄) → (4·A₂₄ − 2·C₂₄ : C₂₄)`, same projective class as `(A :
    /// C)`.
    fn from(dc: DoublingConstants) -> Self {
        let two_c24 = &dc.C24 + &dc.C24;
        let four_a24 = {
            let t = &dc.A24 + &dc.A24;
            &t + &t
        };
        Self {
            A: &four_a24 - &two_c24,
            C: dc.C24,
        }
    }
}

/// A Montgomery curve E_A : y² = x³ + Ax² + x over F_{p²}.
///
/// Caches three representations of the curve coefficient:
/// - [`Coefficient`]: affine A (for serialization, j-invariant)
/// - [`ProjectiveCoefficient`]: projective (A : C) (for isomorphisms)
/// - [`DoublingConstants`]: (A₂₄, C₂₄) = (A+2C, 4C) (for point arithmetic)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Curve {
    affine: Coefficient,
    projective: ProjectiveCoefficient,
    pub(crate) doubling: DoublingConstants,
}

impl From<Coefficient> for Curve {
    /// Construct from affine A (C = 1).
    fn from(a: Coefficient) -> Self {
        let pc = ProjectiveCoefficient::from(a);
        let dc = DoublingConstants::from(pc);
        Self {
            affine: a,
            projective: pc,
            doubling: dc,
        }
    }
}

impl From<ProjectiveCoefficient> for Curve {
    /// Construct from projective (A : C). Requires one inversion for affine A.
    fn from(pc: ProjectiveCoefficient) -> Self {
        let a_affine = &pc.A * &pc.C.invert();
        let dc = DoublingConstants::from(pc);
        Self {
            affine: Coefficient(a_affine),
            projective: pc,
            doubling: dc,
        }
    }
}

impl From<DoublingConstants> for Curve {
    /// Construct from doubling constants (A₂₄ : C₂₄). Requires one inversion
    /// for affine A.
    fn from(dc: DoublingConstants) -> Self {
        let pc = ProjectiveCoefficient::from(dc);
        let two = Fp2::from_fp(Fp::from_small(2));
        let four = Fp2::from_fp(Fp::from_small(4));
        let a_affine = &(&(&four * &dc.A24) * &dc.C24.invert()) - &two;
        Self {
            affine: Coefficient(a_affine),
            projective: pc,
            doubling: dc,
        }
    }
}

impl Curve {
    /// The starting curve E₀ : y² = x³ + x (A = 0).
    ///
    /// This is the supersingular curve used as the base in SQIsign.
    pub const E0: Curve = Curve {
        affine: Coefficient::ZERO,
        projective: ProjectiveCoefficient {
            A: Fp2::ZERO,
            C: Fp2::ONE,
        },
        doubling: DoublingConstants {
            A24: Fp2::new(Fp::TWO, Fp::ZERO),
            C24: Fp2::new(Fp::FOUR, Fp::ZERO),
        },
    };

    /// Normalize the doubling constants to (A₂₄/C₂₄ : 1).
    ///
    /// The C reference (`ec_normalize_curve_and_A24`) normalizes
    /// the projective constants before torsion basis generation.
    /// This ensures that the Montgomery ladder produces the same
    /// projective representative as the C reference, which matters
    /// for differential addition consistency in subsequent operations.
    pub fn normalize(&mut self) {
        self.doubling.normalize();
    }

    /// Check if the doubling constants are normalized (C₂₄ = 1).
    pub fn is_normalized(&self) -> bool {
        self.doubling.is_normalized()
    }

    /// The affine Montgomery coefficient A.
    pub fn coefficient(&self) -> &Coefficient {
        &self.affine
    }

    /// The projective Montgomery coefficient (A : C).
    pub fn projective_coefficient(&self) -> &ProjectiveCoefficient {
        &self.projective
    }

    /// The projective doubling constants (A₂₄, C₂₄).
    pub fn doubling_constants(&self) -> &DoublingConstants {
        &self.doubling
    }

    /// Compute the [j-invariant] j(E_A).
    ///
    /// j(E) = 256(A² − 3)³ / (A² − 4)
    ///
    /// [j-invariant]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
    pub fn j_invariant(&self) -> Fp2 {
        let A2 = self.affine.as_fp2().square();
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

    /// Compute the isomorphism from `self` to `target`.
    ///
    /// Both curves must have the same j-invariant. Returns `None` if
    /// λ_x = 0 or λ_z = 0 (degenerate case, see Remark 1 in the spec).
    ///
    /// Implements lines 1–3 of [Algorithm 8.9][Alg. 8.9].
    ///
    /// [Alg. 8.9]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.9
    #[must_use]
    pub fn isomorphism(&self, target: &Curve) -> Option<Isomorphism> {
        let pc = self.projective_coefficient();
        let (a, c) = (pc.A, pc.C);
        let pc_target = target.projective_coefficient();
        let (a_prime, c_prime) = (pc_target.A, pc_target.C);

        // Hoist squares used by both λ_x and λ_z.
        let a_sq = a.square();
        let c_sq = c.square();
        let a_prime_sq = a_prime.square();
        let c_prime_sq = c_prime.square();

        // Line 1: λ_x ← (2A'³ − 9A'C'²)(3C³ − A²C)
        let lambda_x = {
            let nine_a_prime_c_prime_sq = {
                let t = &a_prime * &c_prime_sq;
                let t2 = &t + &t;
                let t4 = &t2 + &t2;
                let t8 = &t4 + &t4;
                &t8 + &t
            };
            let two_a_prime_cubed = {
                let t = &a_prime_sq * &a_prime;
                &t + &t
            };
            let left = &two_a_prime_cubed - &nine_a_prime_c_prime_sq;

            let three_c_cubed = {
                let t = &c_sq * &c;
                let t2 = &t + &t;
                &t2 + &t
            };
            let right = &three_c_cubed - &(&a_sq * &c);

            &left * &right
        };

        // Line 2: λ_z ← (2A³ − 9AC²)(3C'³ − A'²C')
        let lambda_z = {
            let nine_a_c_sq = {
                let t = &a * &c_sq;
                let t2 = &t + &t;
                let t4 = &t2 + &t2;
                let t8 = &t4 + &t4;
                &t8 + &t
            };
            let two_a_cubed = {
                let t = &a_sq * &a;
                &t + &t
            };
            let left = &two_a_cubed - &nine_a_c_sq;

            let three_c_prime_cubed = {
                let t = &c_prime_sq * &c_prime;
                let t2 = &t + &t;
                &t2 + &t
            };
            let right = &three_c_prime_cubed - &(&a_prime_sq * &c_prime);

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

        Some(Isomorphism {
            lambda_x,
            lambda_z,
            three_cc_prime,
            ac_prime: &a * &c_prime,
            a_prime_c: &a_prime * &c,
            target: *target,
        })
    }
}
