//! Subtraction on [`BigInt<N>`][super::BigInt]: the constant-time
//! [`ct_sub`][BigInt::ct_sub] entry point, the private
//! [`mag_sub`](BigInt::mag_sub) limb-level helper, and the
//! corresponding [`Sub`] trait impls.

use core::ops::Sub;

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Constant-time unsigned subtraction of magnitudes. Returns `(limbs,
    /// borrow)`. Borrow is 1 if `a < b` (unsigned).
    ///
    /// Same `b1 | b2` simplification as in [`Self::mag_add`] for tighter
    /// `sbcs` chain codegen.
    #[inline(always)]
    pub(super) const fn mag_sub(a: &[u64; N], b: &[u64; N]) -> ([u64; N], u64) {
        let mut result = [0u64; N];
        let mut borrow: u64 = 0;
        let mut i = 0;
        while i < N {
            let (d1, b1) = a[i].overflowing_sub(b[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            result[i] = d2;
            borrow = (b1 | b2) as u64;
            i += 1;
        }
        (result, borrow)
    }

    /// Two's-complement negation of a magnitude: `2^(64*N) - x mod
    /// 2^(64*N)`, discarding the top borrow.  For `x != 0` this is the
    /// magnitude of `-x` reduced mod `2^(64*N)`; equivalently the result
    /// of borrowing past the top limb.  Constant-time.
    #[inline(always)]
    pub(super) const fn mag_negate(x: &[u64; N]) -> [u64; N] {
        let zero = [0u64; N];
        let (neg, _) = Self::mag_sub(&zero, x);
        neg
    }

    /// Constant-time signed subtraction: `self - rhs`.
    #[inline]
    pub fn ct_sub(&self, rhs: &Self) -> Self {
        let neg_rhs = rhs.wrapping_neg();
        self.ct_add(&neg_rhs)
    }

    /// In-place constant-time signed subtraction: `self -= rhs`.
    ///
    /// Same merge as [`Self::ct_sub`] (negate `rhs`, add) without the
    /// return-value copy; CT-neutral.
    #[inline]
    pub fn ct_sub_assign(&mut self, rhs: &Self) {
        let neg_rhs = rhs.wrapping_neg();
        self.ct_add_assign(&neg_rhs);
    }
}

impl<const N: usize> core::ops::SubAssign<&BigInt<N>> for BigInt<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: &BigInt<N>) {
        self.ct_sub_assign(rhs);
    }
}

impl<const N: usize> Sub for BigInt<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.ct_sub(&rhs)
    }
}

impl<const N: usize> Sub<&BigInt<N>> for BigInt<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &Self) -> Self {
        self.ct_sub(rhs)
    }
}

impl<const N: usize> Sub<&BigInt<N>> for &BigInt<N> {
    type Output = BigInt<N>;
    #[inline]
    fn sub(self, rhs: &BigInt<N>) -> BigInt<N> {
        self.ct_sub(rhs)
    }
}
