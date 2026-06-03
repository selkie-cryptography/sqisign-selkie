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
//! [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.2.3
//!
//! See also [§2.2] (elliptic curves) and [§8.2] (curve arithmetic).
//!
//! # Divergences from spec / C reference
//!
//! - **Three coefficient representations**: [`Curve`] caches affine
//!   [`Coefficient`], projective [`ProjectiveCoefficient`] `(A:C)`, and
//!   [`DoublingConstants`] `(A₂₄, C₂₄)`. The spec uses `(A:C)`; the C ref uses
//!   `(A₂₄, C₂₄)` internally. We store all three to avoid recomputation and
//!   inversions.
//! - **Isomorphism**: [`Isomorphism`] implements [Alg. 8.9][Alg. 8.9] directly
//!   from the spec with projective `(A:C)` coefficients — no inversions. The C
//!   ref converts through Short Weierstrass.
//! - **Curve normalization**: the C ref normalizes `(A₂₄/C₂₄ : 1)` before
//!   torsion basis generation. We match this via [`Curve::normalize`].
//!
//! [Alg. 8.9]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.9
//! [§2.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
//! [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2

mod isomorphism;
mod jacobian;
mod point;

#[cfg(test)]
mod tests;

pub use isomorphism::Isomorphism;
pub use jacobian::JacobianPoint;
pub use point::ProjectiveXOnlyPoint;
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
/// [§2.2.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.2.1
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

    /// Encodes as bytes (delegates to F_{p²} encoding).
    pub fn to_bytes(self) -> [u8; crate::params::CURVE_ENCODED_BYTES] {
        self.0.to_bytes()
    }

    /// Decodes from bytes.
    pub fn from_bytes(bytes: &[u8; crate::params::CURVE_ENCODED_BYTES]) -> Coefficient {
        Coefficient(Fp2::from_bytes(bytes))
    }

    /// `true` iff the Montgomery model `y² = x³ + Ax² + x` is
    /// singular. Singularity occurs exactly when the discriminant
    /// `Δ = 4(A² − 4)` vanishes, i.e., when `A == 2` or `A == −2`.
    /// Such coefficients are not valid Montgomery curves; verify's
    /// parse path rejects them, matching the C reference's
    /// `ec_curve_verify_A` (`ec.c:169`).
    pub fn is_singular(&self) -> bool {
        let two = Fp2::from_fp(Fp::from_small(2));
        self.0 == two || self.0 == -&two
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
#[derive(Copy, Clone, Debug)]
pub struct ProjectiveCoefficient {
    /// Numerator A.
    pub A: Fp2,
    /// Denominator C.
    pub C: Fp2,
}

impl PartialEq for ProjectiveCoefficient {
    /// Projective equality via cross-multiplication: `(A₁ : C₁) ==
    /// (A₂ : C₂)` iff `A₁·C₂ == A₂·C₁`. Same convention as
    /// [`ProjectiveXOnlyPoint`]. Avoids the inversion that comparing
    /// `A/C` directly would require, and recognizes `(A : C)` and
    /// `(kA : kC)` as the same projective coordinate without
    /// normalizing first.
    fn eq(&self, other: &Self) -> bool {
        &self.A * &other.C == &other.A * &self.C
    }
}

impl Eq for ProjectiveCoefficient {}

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
/// Isogeny codomain computations (`TwoIsogeny`, `FourIsogeny`)
/// naturally produce these. They are consumed by the Montgomery
/// ladder's doubling and differential addition formulas.
// No `PartialEq`/`Eq` on `DoublingConstants`: nothing in the crate
// compares two `DoublingConstants` values directly (only field reads
// `.A24`/`.C24` for the Montgomery ladder), and `Curve::eq` is defined
// in terms of `affine` rather than the cached `doubling` field. If a
// future caller does want equality, the right semantic is cross-multiply
// on `(A24 : C24)` (the type is a projective representative just like
// `ProjectiveCoefficient`); add it then with a deliberate choice rather
// than inheriting an untested derive.
#[derive(Copy, Clone, Debug)]
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

    /// Normalizes to `(A₂₄/C₂₄ : 1)`.
    pub fn normalize(&mut self) {
        if self.C24 != Fp2::ONE {
            let inv = self.C24.invert();
            self.A24 = &self.A24 * &inv;
            self.C24 = Fp2::ONE;
        }
    }

    /// Checks if normalized (C₂₄ = 1).
    pub fn is_normalized(&self) -> bool {
        self.C24 == Fp2::ONE
    }
}

impl From<ProjectiveCoefficient> for DoublingConstants {
    /// `(A : C) → ((A + 2C)/(4C) : 1)`, the normalized affine
    /// representative.
    ///
    /// The normalized form is required so Selkie's `xDBL` produces the
    /// same projective `(X : Z)` representative as the C reference's
    /// `xDBL_A24` (which always operates on a normalized
    /// `(A₂₄/(4C) : 1)` after `ec_curve_normalize_A24`,
    /// `dim2id2iso.c:30`). Without this normalization, every doubling
    /// scales the output by an extra factor of `4C`, producing a
    /// projectively-equivalent but byte-different `(X : Z)` that the
    /// downstream chain reads (and the gluing site dumps).
    ///
    /// Uses one inversion per curve construction. Negligible cost
    /// because curves are constructed rarely.
    fn from(pc: ProjectiveCoefficient) -> Self {
        let two_c = &pc.C + &pc.C;
        let four_c = &two_c + &two_c;
        let four_c_inv = four_c.invert();
        Self {
            A24: &(&pc.A + &two_c) * &four_c_inv,
            C24: Fp2::ONE,
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
#[derive(Copy, Clone, Debug)]
pub struct Curve {
    /// Affine Montgomery coefficient `A`.
    affine: Coefficient,
    /// Projective representative `(A : C)`.
    projective: ProjectiveCoefficient,
    /// Doubling constants `(A₂₄, C₂₄) = (A + 2C, 4C)`.
    pub(crate) doubling: DoublingConstants,
}

impl PartialEq for Curve {
    /// Two curves are equal iff they share the same Montgomery
    /// coefficient `A` (i.e. they describe the same elliptic curve
    /// `y² = x³ + Ax² + x`). Differences in the cached `(A : C)`
    /// projective representative or the `(A₂₄ : C₂₄)` doubling
    /// constants don't make the underlying curves different — those
    /// fields are alternate encodings of the same affine `A`.
    ///
    /// # Trusts the constructor invariant
    ///
    /// We compare only the affine field on the assumption that every
    /// `Curve` is built via one of the `From<…>` constructors, which
    /// derive all three fields from a single source and keep them
    /// mutually consistent. The fields are private (and `pub(crate)`
    /// for `doubling`), so no external code can produce a desynced
    /// `Curve`. If a future internal refactor ever populates the
    /// fields independently — e.g. via struct-literal syntax in a
    /// crate-internal helper — this `eq` could return `true` for
    /// curves that disagree at the projective/doubling level. Add a
    /// stricter cross-multiply comparison on those fields here if
    /// that risk ever materializes; the cost is ~8 extra `Fp²`
    /// mults, fired only in the handful of asserts that compare
    /// curves.
    fn eq(&self, other: &Self) -> bool {
        self.affine == other.affine
    }
}

impl Eq for Curve {}

impl From<Coefficient> for Curve {
    /// Constructs from affine A (C = 1).
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
    /// Constructs from projective (A : C). Requires one inversion for affine A.
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
    /// Constructs from doubling constants (A₂₄ : C₂₄). Requires one inversion
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
        // Normalized doubling constants `(A₂₄/(4C) : 1) = (1/2 : 1)`
        // for E0 (A = 0, C = 1). See `DoublingConstants::from` for
        // why the normalized form is required.
        //
        // `1/2` in Montgomery radix-2⁵¹ form was computed via
        // `(Fp::ONE + Fp::ONE).invert()` (see `print_one_half_limbs`
        // in `surfaces/tests.rs`).
        doubling: DoublingConstants {
            A24: Fp2::new(
                Fp::from_limbs([
                    0x000000000000000C,
                    0x0000000000000000,
                    0x0000000000000000,
                    0x0000000000000000,
                    0x0000400000000000,
                ]),
                Fp::ZERO,
            ),
            C24: Fp2::ONE,
        },
    };

    /// Normalizes the doubling constants to (A₂₄/C₂₄ : 1).
    ///
    /// The C reference (`ec_normalize_curve_and_A24`) normalizes
    /// the projective constants before torsion basis generation.
    /// This ensures that the Montgomery ladder produces the same
    /// projective representative as the C reference, which matters
    /// for differential addition consistency in subsequent operations.
    pub fn normalize(&mut self) {
        self.doubling.normalize();
    }

    /// Checks if the doubling constants are normalized (C₂₄ = 1).
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

    /// Computes the [j-invariant] j(E_A).
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

    /// Recover the y-coordinate for an affine x on this curve.
    ///
    /// Computes y = √(x³ + Ax² + x) on the Montgomery curve
    /// By² = x³ + Ax² + x (with B = 1). Returns `None` if x
    /// is not on the curve.
    #[must_use]
    pub fn recover_y(&self, x: &AffineX) -> Option<Fp2> {
        let a = self.affine.as_fp2();
        let x = x.as_fp2();
        let x2 = x.square();
        let rhs = &(&(&x2 * x) + &(&x2 * a)) + x;
        if bool::from(rhs.is_square()) {
            Some(rhs.sqrt())
        } else {
            None
        }
    }

    /// Computes the isomorphism from `self` to `target`.
    ///
    /// Both curves must have the same j-invariant. Returns `None` if
    /// λ_x = 0 or λ_z = 0 (degenerate case, see Remark 1 in the spec).
    ///
    /// Implements lines 1–3 of [Alg. 8.9][Alg. 8.9].
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
