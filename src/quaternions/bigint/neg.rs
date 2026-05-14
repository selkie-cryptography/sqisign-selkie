//! Negation on [`BigInt<N>`][super::BigInt]: the
//! [`wrapping_neg`][BigInt::wrapping_neg] method and the corresponding
//! [`Neg`] trait impls.

use core::ops::Neg;

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Negation. Flips the sign bit.
    ///
    /// Table 1 (§3.1) from [Kouider et al.][ct-bigint]: `c_sign = 1 XOR
    /// a_sign`.
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    #[inline]
    pub fn wrapping_neg(&self) -> Self {
        Self {
            sign: self.sign ^ 1,
            limbs: self.limbs,
        }
    }
}

impl<const N: usize> Neg for BigInt<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}

impl<const N: usize> Neg for &BigInt<N> {
    type Output = BigInt<N>;
    #[inline]
    fn neg(self) -> BigInt<N> {
        self.wrapping_neg()
    }
}
