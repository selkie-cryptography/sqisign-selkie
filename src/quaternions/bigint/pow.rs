//! Integer exponentiation on [`BigInt<N>`][super::BigInt].

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Integer exponentiation: `self^exp`.
    ///
    /// Uses a simple square-and-multiply. The exponent is public (not
    /// constant-time w.r.t. the exponent value).
    pub fn pow(&self, exp: u32) -> Self {
        if exp == 0 {
            return Self::ONE;
        }
        let mut result = Self::ONE;
        let mut base = *self;
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result = result.ct_mul(&base);
            }
            base = base.ct_mul(&base);
            e >>= 1;
        }
        result
    }
}
