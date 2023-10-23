//! Quadratic extension field 𝔽ₚ²

// TODO: Use https://eprint.iacr.org/2022/367.pdf for quadraic arithmetic for ~30% speedup and no
// need for double-wide types to hold intermediate products

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use subtle::{Choice, ConstantTimeEq};
use zeroize::DefaultIsZeroes;

use crate::field::*;

/// An element of a quadratic extension field 𝔽ₚ², where 𝔽ₚ(𝑖) as `x = a + b𝑖`,
/// (`i² = -1`), for a and b in the base field 𝔽ₚ.
#[derive(Copy, Clone, Default)]
pub struct QuadraticExtensionElement<F: FiniteField> {
    a: F,
    b: F,
}

impl<F: FiniteField> QuadraticExtensionElement<F> {}

impl<F> Add for QuadraticExtensionElement<F>
where
    F: FiniteField + Add<Output = F>,
{
    type Output = QuadraticExtensionElement<F>;

    fn add(self, rhs: Self) -> QuadraticExtensionElement<F> {
        &self + &rhs
    }
}

impl<F> Add<&QuadraticExtensionElement<F>> for QuadraticExtensionElement<F>
where
    F: FiniteField + Add<Output = F>,
{
    type Output = QuadraticExtensionElement<F>;

    fn add(self, rhs: &QuadraticExtensionElement<F>) -> QuadraticExtensionElement<F> {
        &self + rhs
    }
}

impl<F> Add<&QuadraticExtensionElement<F>> for &QuadraticExtensionElement<F>
where
    F: FiniteField + Add<Output = F>,
{
    type Output = QuadraticExtensionElement<F>;

    fn add(self, rhs: &QuadraticExtensionElement<F>) -> QuadraticExtensionElement<F> {
        QuadraticExtensionElement {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl<F> AddAssign for QuadraticExtensionElement<F>
where
    F: FiniteField + Add<Output = F>,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F> AddAssign<&QuadraticExtensionElement<F>> for QuadraticExtensionElement<F>
where
    F: FiniteField + Add<Output = F>,
{
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + *rhs;
    }
}

impl<F> ConstantTimeEq for QuadraticExtensionElement<F>
where
    F: FiniteField + ConstantTimeEq,
{
    /// Check whether two `QuadraticExtensionElement`s are equal, runtime
    /// independent of the value of the value.
    fn ct_eq(&self, other: &Self) -> Choice {
        self.a.ct_eq(&other.a) & self.b.ct_eq(&other.b)
    }
}

// impl<F: FiniteField> Debug for QuadraticExtensionElement<F> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         f.debug_struct("QuadraticExtensionElement<F>")
//             .field("a", hex::encode(&self.to_bytes()))
//             .field("b", hex::encode(&self.to_bytes()))
//             .finish()
//     }
// }

impl<F: FiniteField> DefaultIsZeroes for QuadraticExtensionElement<F> {}

impl<F: FiniteField> Eq for QuadraticExtensionElement<F> where F: ConstantTimeEq {}

impl<F> FiniteField for QuadraticExtensionElement<F>
where
    F: FiniteField + ConstantTimeEq + Add<Output = F> + Mul<Output = F> + Sub<Output = F>,
{
    fn zero() -> Self {
        Self::default()
    }

    fn one() -> Self {
        // TODO
        unimplemented!()
    }

    fn double(&self) -> Self {
        *self + *self
    }

    fn square(&self) -> Self {
        *self * *self
    }

    fn invert(&self) -> Self {
        // TODO
        unimplemented!()
    }
}

impl<F> Mul for QuadraticExtensionElement<F>
where
    F: FiniteField + Mul<Output = F>,
{
    type Output = QuadraticExtensionElement<F>;

    fn mul(self, rhs: Self) -> QuadraticExtensionElement<F> {
        self * &rhs
    }
}

impl<F> Mul<&QuadraticExtensionElement<F>> for QuadraticExtensionElement<F>
where
    F: FiniteField + Mul<Output = F>,
{
    type Output = QuadraticExtensionElement<F>;

    fn mul(self, rhs: &QuadraticExtensionElement<F>) -> QuadraticExtensionElement<F> {
        self * rhs
    }
}

impl<F> Mul<&QuadraticExtensionElement<F>> for &QuadraticExtensionElement<F>
where
    F: FiniteField + Add<Output = F> + Sub<Output = F> + Mul<Output = F>,
{
    type Output = QuadraticExtensionElement<F>;

    // Uses Karatsuba's trick to require 3 mul's instead of the
    // schoolbook 4, and 5 add's / sub's rather than 2.
    //
    // As long as 1 mul is more expensive than 3 add's/sub's, this is faster.

    fn mul(self, rhs: &QuadraticExtensionElement<F>) -> QuadraticExtensionElement<F> {
        let (a, b) = (self.a, self.b);
        let (c, d) = (rhs.a, rhs.b);

        // Using Karatsuba's trick:
        //
        // (a+b𝑖)(c+d𝑖) = ac + bc𝑖 + ad𝑖 - bd
        //               = ac + (bc + ad)𝑖 - bd
        //               = ac + ((a+c)(b+d) - bd - ac)𝑖 - bd
        //               = (ac - bd) + ((a+c)(b+d) - bd - ac)𝑖
        //
        // Which only needs 3 total multiplications, when reusing products.

        let ac = a * c;
        let bd = b * d;

        QuadraticExtensionElement {
            a: ac - bd,
            b: ((a + c) * (b + d)) - bd - ac,
        }
    }
}

impl<F: FiniteField> MulAssign for QuadraticExtensionElement<F>
where
    F: Mul<Output = F>,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: FiniteField> Neg for QuadraticExtensionElement<F>
where
    F: ConstantTimeEq + Add<Output = F> + Mul<Output = F> + Sub<Output = F>,
{
    type Output = QuadraticExtensionElement<F>;

    fn neg(self) -> QuadraticExtensionElement<F> {
        &QuadraticExtensionElement::<F>::zero() - &self
    }
}

impl<F: FiniteField> PartialEq for QuadraticExtensionElement<F>
where
    F: ConstantTimeEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: FiniteField> Sub for QuadraticExtensionElement<F>
where
    F: Sub<Output = F>,
{
    type Output = QuadraticExtensionElement<F>;

    fn sub(self, rhs: Self) -> Self {
        &self - &rhs
    }
}

impl<F: FiniteField> Sub<&QuadraticExtensionElement<F>> for QuadraticExtensionElement<F>
where
    F: Sub<Output = F>,
{
    type Output = QuadraticExtensionElement<F>;

    fn sub(self, rhs: &QuadraticExtensionElement<F>) -> QuadraticExtensionElement<F> {
        &self - rhs
    }
}

impl<F: FiniteField> Sub<&QuadraticExtensionElement<F>> for &QuadraticExtensionElement<F>
where
    F: Sub<Output = F>,
{
    type Output = QuadraticExtensionElement<F>;

    fn sub(self, rhs: &QuadraticExtensionElement<F>) -> QuadraticExtensionElement<F> {
        QuadraticExtensionElement {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl<F> SubAssign for QuadraticExtensionElement<F>
where
    F: FiniteField + Sub<Output = F>,
{
    fn sub_assign(&mut self, rhs: QuadraticExtensionElement<F>) {
        *self -= &rhs;
    }
}

impl<F: FiniteField> SubAssign<&QuadraticExtensionElement<F>> for QuadraticExtensionElement<F>
where
    F: Sub<Output = F>,
{
    fn sub_assign(&mut self, rhs: &QuadraticExtensionElement<F>) {
        *self = *self - *rhs;
    }
}
