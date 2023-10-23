//! Elliptic curves in the short Weierstraß model and their rational points
//! defined over finite fields.

use std::{collections::hash_set::HashSet, iter::Sum, ops::*};

use crate::{
    curve::{Curve, JInvariant, Point, Scalar},
    field::FiniteField,
    isogeny::Isogeny,
};

/// An  elliptic curve in affine [short Weierstraß] form defined over a finite
/// field.
///
/// y² = x³ + ax + b
///
/// [short Weierstraß]:
pub struct ShortWeierstrass<F>
where
    F: FiniteField,
{
    /// The a coefficient associated with the degree-1 term of the short
    /// Weierstraß curve equation.
    a: F,
    /// The b coefficient associated with the degree-0 term of the short
    /// Weierstraß curve equation.
    b: F,
}

impl<F> Curve for ShortWeierstrass<F>
where
    F: FiniteField,
{
    type FiniteField = F;

    fn from_j_invariant(j: JInvariant<F>) -> Self {}

    fn j_invariant(&self) -> JInvariant<F> {}
}

/// A point in affine coordinates of a short Weierstraß curve.
pub struct Affine<F>
where
    F: FiniteField,
{
    x: F,
    y: F,
}

impl<F> Point<ShortWeierstrass<F>, F> for Affine<F> where F: FiniteField {}

/// A point on the projective (Kummer) line of a short Weierstraß curve.
pub struct Projective<F>
where
    F: FiniteField,
{
    X: F,
    Y: F,
    Z: F,
}

impl<F> Point<ShortWeierstrass<F>, F> for Projective<F> where F: FiniteField {}
