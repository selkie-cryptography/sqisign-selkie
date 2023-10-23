//! Finite field implementations with cryptographically-secure supersingular
//! curves and isogenies between them in mind.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use zeroize::DefaultIsZeroes;

pub mod limb;
pub mod prime;
pub mod quadratic;

pub trait FiniteField:
    Add
    + for<'a> Add<&'a Self, Output = Self>
    + AddAssign
    + for<'a> AddAssign<&'a Self>
    + Copy
    + Default
    + DefaultIsZeroes
    + Eq
    + Mul
    + for<'a> Mul<&'a Self, Output = Self>
    + MulAssign
    + Neg
    + Sub
    + for<'a> Sub<&'a Self, Output = Self>
    + SubAssign
    + for<'a> SubAssign<&'a Self>
{
    /// Returns zero in the field, the additive identity.
    fn zero() -> Self;

    /// Returns one in the field, the multiplicative identity.
    fn one() -> Self;

    /// Returns `self + self`
    fn double(&self) -> Self;

    /// Returns `self * self`
    fn square(&self) -> Self;

    /// Returns the multiplicative inverse of `self`, unless `self` is zero.
    fn invert(&self) -> Self;
}

// TODO(dconnolly): make these generic over Limbs? Limb::WORDs?
// - [ ] addc
// - [ ] subc
// - [ ] mul
// - [ ] mac

/// Add and carry (`a + b + carry`), returning the result and the new carry.
#[inline(always)]
const fn addc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let (a, b, carry) = (a as u128, b as u128, carry as u128);
    let out = a + b + carry;
    (out as u64, (out >> 64) as u64)
}

/// Subtract and borrow (`a - (b + borrow)`), returning the result and the new
/// borrow.
#[inline(always)]
const fn subc(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let (a, b, borrow) = (a as u128, b as u128, (borrow >> 63) as u128);
    let out = a.wrapping_sub(b + borrow);
    (out as u64, (out >> 64) as u64)
}

/// Multiply, returning the double-wide result.
#[inline(always)]
const fn mul(a: u64, b: u64) -> (u64, u64) {
    mac(0, a, b, 0)
}

/// Multiply, add, and carry (`a + (b * c) + carry`), returning the
/// result and the new carry.
#[inline(always)]
const fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let (a, b, c, carry) = (a as u128, b as u128, c as u128, carry as u128);
    let out = a + (b * c) + carry;
    (out as u64, (out >> 64) as u64)
}
