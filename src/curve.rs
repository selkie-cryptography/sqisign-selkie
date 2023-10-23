//! Elliptic curves and their rational points defined over finite fields.

use std::{collections::hash_set::HashSet, iter::Sum, ops::*};

use crate::{field::FiniteField, isogeny::Isogeny};

pub mod montgomery;
pub mod short_weierstrass;

// P434 requires 55 bytes of storage
// `Scalar` values are always in
// Montgomery form; i.e., Scalar(a) = aR mod q,
// with R = 2^512.
pub struct Scalar([u8; 64]);

impl FiniteField for Scalar {}

/// The j-invariant of an elliptic curve defined over a finite field.
///
/// For a short Weierstraß curve:
///
/// j(E) = 1728(4A³ / 4A³ + 27B²)
///
/// It turns out that for all elliptic curves in the same isomorphism class
/// (regardless of the curve equation), they all have the same j-invariant. This
/// is useful for both distinguishing when curves are ~the same, but also to
/// generate a common value or canonical value amongst isomorphic curves.
pub struct JInvariant<F>(F)
where
    F: FiniteField;

/// Trait for general curve instantiation operations.
pub trait Curve: Sized {
    type FiniteField: FiniteField;

    fn j_invariant(&self) -> JInvariant<Self::FiniteField>;

    fn from_j_invariant(j: JInvariant<Self::FiniteField>) -> Self;

    // fn isogeny<P: Point<Self, F>>(&self, kernel: SubGroup<P, Self, F>) ->
    // Isogeny<P, Self, F>;
}

/// Trait modeling generic point operations on a particular curve instantiation.
// TODO: is the finite field F sufficient, or do we need to refine to a
// particular Scalar subtype? We can perhaps just constrain F to impl Scalar or
// something
pub trait Point<C, F>:
    Add<Self, Output = Self>
    + AddAssign<Self, Output = Self>
    + Clone
    + Copy
    + Default
    + Eq
    + Mul<Self::Scalar, Output = Self>
    + MulAssign<Self::Scalar, Output = Self>
    + Neg<Output = Self>
    + Sized
    + Sub<Self, Output = Self>
    + SubAssign<Self, Output = Self>
where
    C: Curve<FiniteField = F>,
    F: FiniteField,
{
    type Scalar: FiniteField;

    /// Generate a random point on this curve.
    fn random() -> Self;
}
