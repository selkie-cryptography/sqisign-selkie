//! Elliptic curves in the Montgomery model and their rational points defined
//! over finite fields.

use std::{collections::hash_set::HashSet, iter::Sum, ops::*};

use crate::{
    curve::{Curve, JInvariant, Point, Scalar},
    field::FiniteField,
    isogeny::Kernel,
};

/// An elliptic curve in projective Montgomery form defined over a finite field.
///
/// By² = Cx³ + Ax² + Cx
///
/// In the Montgomery form in affine coordinates:
///
/// by² = x³ + ax² + x
///
/// when a = A/C and b = B/C.
pub struct Montgomery<F>
where
    F: FiniteField,
{
    /// The degree-2 term coefficient.
    A: F,

    /// The ...other? degree-2 term coefficient?
    B: F,

    /// The degree-3 and degree-1 term coeffcients and denominator for the
    /// projective coordinates.
    C: F,
}

impl<F> Curve for Montgomery<F>
where
    F: FiniteField,
{
    type FiniteField = F;

    fn from_j_invariant(j: JInvariant<F>) -> Self {}

    fn j_invariant(&self) -> JInvariant<F> {}
}

/// A point in affine coordinates of a Montgomery curve.
#[derive(Clone, Copy, Default, Eq)]
pub struct Affine<F>
where
    F: FiniteField,
{
    x: F,
    y: F,
}

impl<F> Point<Montgomery<F>, F> for Affine<F> where F: FiniteField {}

impl<F> Kernel<F> for Affine<F> where F: FiniteField {}

/// A point on the projective (Kummer) line of a Montgomery curve.
pub struct Projective<F>
where
    F: FiniteField,
{
    X: F,
    Z: F,
}

impl<F> Point<Montgomery<F>, F> for Projective<F> where F: FiniteField {}
