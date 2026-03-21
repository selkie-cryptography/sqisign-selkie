//! Quadratic extension field F_{p²} = F_p(i), where i² = −1.
//!
//! Elements are represented as a + b·i with a, b ∈ F_p.
//! Since p ≡ 3 (mod 4), −1 is a quadratic non-residue in F_p,
//! so F_p(i) is a valid degree-2 extension.
//!
//! See [§2.1.2] of the SQIsign spec.
//!
//! [§2.1.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.1

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::fields::fp::{Fp, FP_ENCODED_BYTES};

/// Number of bytes in a canonical encoding of an element of F_{p²}.
pub const FP2_ENCODED_BYTES: usize = 2 * FP_ENCODED_BYTES;

/// An element of the quadratic extension field F_{p²} = F_p(i).
///
/// Represented as `a + b·i` where `a` is the "real" part and `b` the
/// "imaginary" part, with i² = −1.
#[derive(Copy, Clone, Debug)]
pub struct Fp2 {
    /// Real part.
    pub(crate) a: Fp,
    /// Imaginary part.
    pub(crate) b: Fp,
}

impl Fp2 {
    /// The additive identity in F_{p²}.
    pub const ZERO: Fp2 = Fp2 {
        a: Fp::ZERO,
        b: Fp::ZERO,
    };

    /// The multiplicative identity in F_{p²}.
    pub const ONE: Fp2 = Fp2 {
        a: Fp::ONE,
        b: Fp::ZERO,
    };

    /// The element i (the generator of the extension).
    pub const I: Fp2 = Fp2 {
        a: Fp::ZERO,
        b: Fp::ONE,
    };

    /// Construct an element from its real and imaginary parts.
    pub const fn new(a: Fp, b: Fp) -> Fp2 {
        Fp2 { a, b }
    }

    /// Construct from a base field element (imaginary part is zero).
    pub const fn from_fp(a: Fp) -> Fp2 {
        Fp2 { a, b: Fp::ZERO }
    }

    /// Square this element.
    ///
    /// Uses the identity (a + bi)² = (a² − b²) + 2ab·i, optimized as:
    ///   real = (a + b)(a − b)
    ///   imag = 2ab
    #[must_use]
    pub fn square(&self) -> Fp2 {
        let a_plus_b = &self.a + &self.b;
        let a_minus_b = &self.a - &self.b;
        let two_ab = &self.a * &self.b;
        Fp2 {
            a: &a_plus_b * &a_minus_b,
            b: &two_ab + &two_ab,
        }
    }

    /// Compute the norm: N(a + bi) = a² + b².
    #[must_use]
    pub fn norm(&self) -> Fp {
        &self.a.square() + &self.b.square()
    }

    /// Compute the conjugate: conj(a + bi) = a − bi.
    #[must_use]
    pub fn conjugate(&self) -> Fp2 {
        Fp2 {
            a: self.a,
            b: -&self.b,
        }
    }

    /// Compute the multiplicative inverse.
    ///
    /// (a + bi)⁻¹ = (a − bi) / (a² + b²)
    #[must_use]
    pub fn invert(&self) -> Fp2 {
        let n_inv = self.norm().invert();
        Fp2 {
            a: &self.a * &n_inv,
            b: &(-&self.b) * &n_inv,
        }
    }

    /// Test whether this element is a square in F_{p²}.
    ///
    /// An element a ∈ F_{p²} is a square iff a^{p+1} is a square in F_p,
    /// since a^{p+1} = N(a) = a² + b².
    pub fn is_square(&self) -> Choice {
        self.norm().is_square() | self.ct_eq(&Fp2::ZERO)
    }

    /// Compute the square root in F_{p²}.
    ///
    /// Uses the algorithm from [SQIsign spec, §2.1.2, Algorithm 8.2].
    /// The result is only meaningful when `self.is_square()` is true.
    #[must_use]
    pub fn sqrt(&self) -> Fp2 {
        // For a = a0 + a1·i:
        // δ = (a0² + a1²)^((p+1)/4)
        // x0 = a0 + δ
        // t0 = 2·x0
        // x1 = t0^((p-3)/4)
        // x0 = x0 · x1
        // x1 = a1 · x1
        // t1 = (2·x0)²
        // if t1 == t0: return x0 + x1·i
        // else:        return x1 − x0·i
        let delta = self.norm().sqrt();
        let x0 = &self.a + &delta;
        let t0 = &x0 + &x0;
        // t0^((p-3)/4)
        let x1 = t0.pow_p3div4();
        let x0 = &x0 * &x1;
        let x1 = &self.b * &x1;
        let t1 = (&x0 + &x0).square();

        // Constant-time select
        let is_eq = t1.ct_eq(&t0);
        Fp2 {
            a: Fp::conditional_select(&x1, &x0, is_eq),
            b: Fp::conditional_select(&(-&x0), &x1, is_eq),
        }
    }

    /// Encode this element as 64 bytes (real part ‖ imaginary part).
    pub fn to_bytes(&self) -> [u8; FP2_ENCODED_BYTES] {
        let mut out = [0u8; FP2_ENCODED_BYTES];
        out[..FP_ENCODED_BYTES].copy_from_slice(&self.a.to_bytes());
        out[FP_ENCODED_BYTES..].copy_from_slice(&self.b.to_bytes());
        out
    }

    /// Decode 64 bytes into an F_{p²} element.
    pub fn from_bytes(bytes: &[u8; FP2_ENCODED_BYTES]) -> Fp2 {
        let a = Fp::from_bytes(bytes[..FP_ENCODED_BYTES].try_into().expect("slice length"));
        let b = Fp::from_bytes(bytes[FP_ENCODED_BYTES..].try_into().expect("slice length"));
        Fp2 { a, b }
    }
}


// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

impl<'a, 'b> Add<&'b Fp2> for &'a Fp2 {
    type Output = Fp2;
    fn add(self, rhs: &'b Fp2) -> Fp2 {
        Fp2 {
            a: &self.a + &rhs.a,
            b: &self.b + &rhs.b,
        }
    }
}

impl<'a, 'b> Sub<&'b Fp2> for &'a Fp2 {
    type Output = Fp2;
    fn sub(self, rhs: &'b Fp2) -> Fp2 {
        Fp2 {
            a: &self.a - &rhs.a,
            b: &self.b - &rhs.b,
        }
    }
}

impl<'a> Neg for &'a Fp2 {
    type Output = Fp2;
    fn neg(self) -> Fp2 {
        Fp2 {
            a: -&self.a,
            b: -&self.b,
        }
    }
}

impl<'a, 'b> Mul<&'b Fp2> for &'a Fp2 {
    type Output = Fp2;

    /// Karatsuba multiplication: 3M + 5A instead of 4M + 2A.
    fn mul(self, rhs: &'b Fp2) -> Fp2 {
        let ac = &self.a * &rhs.a;
        let bd = &self.b * &rhs.b;
        Fp2 {
            a: &ac - &bd,
            b: &(&(&self.a + &self.b) * &(&rhs.a + &rhs.b)) - &(&ac + &bd),
        }
    }
}

/// Multiply an F_{p²} element by a base field element.
impl<'a, 'b> Mul<&'b Fp> for &'a Fp2 {
    type Output = Fp2;
    fn mul(self, rhs: &'b Fp) -> Fp2 {
        Fp2 {
            a: &self.a * rhs,
            b: &self.b * rhs,
        }
    }
}

// Owned convenience impls

impl Add for Fp2 {
    type Output = Fp2;
    fn add(self, rhs: Fp2) -> Fp2 { &self + &rhs }
}

impl Sub for Fp2 {
    type Output = Fp2;
    fn sub(self, rhs: Fp2) -> Fp2 { &self - &rhs }
}

impl Mul for Fp2 {
    type Output = Fp2;
    fn mul(self, rhs: Fp2) -> Fp2 { &self * &rhs }
}

impl Neg for Fp2 {
    type Output = Fp2;
    fn neg(self) -> Fp2 { -&self }
}

impl AddAssign<&Fp2> for Fp2 {
    fn add_assign(&mut self, rhs: &Fp2) { *self = &*self + rhs; }
}

impl SubAssign<&Fp2> for Fp2 {
    fn sub_assign(&mut self, rhs: &Fp2) { *self = &*self - rhs; }
}

impl MulAssign<&Fp2> for Fp2 {
    fn mul_assign(&mut self, rhs: &Fp2) { *self = &*self * rhs; }
}

impl AddAssign for Fp2 {
    fn add_assign(&mut self, rhs: Fp2) { *self += &rhs; }
}

impl SubAssign for Fp2 {
    fn sub_assign(&mut self, rhs: Fp2) { *self -= &rhs; }
}

impl MulAssign for Fp2 {
    fn mul_assign(&mut self, rhs: Fp2) { *self *= &rhs; }
}

// ---------------------------------------------------------------------------
// Constant-time traits
// ---------------------------------------------------------------------------

impl ConstantTimeEq for Fp2 {
    fn ct_eq(&self, other: &Fp2) -> Choice {
        self.a.ct_eq(&other.a) & self.b.ct_eq(&other.b)
    }
}

impl ConditionallySelectable for Fp2 {
    fn conditional_select(a: &Fp2, b: &Fp2, choice: Choice) -> Fp2 {
        Fp2 {
            a: Fp::conditional_select(&a.a, &b.a, choice),
            b: Fp::conditional_select(&a.b, &b.b, choice),
        }
    }
}

impl Eq for Fp2 {}

impl PartialEq for Fp2 {
    fn eq(&self, other: &Fp2) -> bool {
        self.ct_eq(other).into()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i_squared_is_minus_one() {
        let i2 = Fp2::I.square();
        assert_eq!(i2, -Fp2::ONE);
    }

    #[test]
    fn conjugate_mul_is_norm() {
        let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
        let n = &a * &a.conjugate();
        // Should be a real number equal to the norm
        assert_eq!(n.b, Fp::ZERO);
        assert_eq!(n.a, a.norm());
    }

    #[test]
    fn inversion() {
        let a = Fp2::new(Fp::from_small(5), Fp::from_small(13));
        let a_inv = a.invert();
        assert_eq!(&a * &a_inv, Fp2::ONE);
    }

    #[test]
    fn karatsuba_matches_schoolbook() {
        let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
        let b = Fp2::new(Fp::from_small(11), Fp::from_small(5));
        // (3 + 7i)(11 + 5i) = 33 + 15i + 77i + 35i² = 33 - 35 + (15+77)i = -2 + 92i
        let c = &a * &b;
        // Check via from_small arithmetic
        let expected_real = &Fp::from_small(33) - &Fp::from_small(35);
        let expected_imag = &Fp::from_small(15) + &Fp::from_small(77);
        assert_eq!(c.a, expected_real);
        assert_eq!(c.b, expected_imag);
    }

    #[test]
    fn roundtrip_bytes() {
        let a = Fp2::new(Fp::from_small(42), Fp::from_small(99));
        let bytes = a.to_bytes();
        let b = Fp2::from_bytes(&bytes);
        assert_eq!(a, b);
    }
}
