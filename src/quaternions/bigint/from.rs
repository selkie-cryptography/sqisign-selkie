//! Construction of [`BigInt<N>`][super::BigInt] from primitive integer
//! types and raw limb arrays, plus the corresponding `From<i*/u*>`
//! impls.

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Creates a `BigInt` from a sign bit and little-endian limb array.
    #[inline]
    pub const fn from_sign_and_limbs(sign: u64, limbs: [u64; N]) -> Self {
        Self { sign, limbs }
    }

    /// Creates a non-negative `BigInt` from `N` little-endian `u64` limbs.
    ///
    /// Equivalent to `from_sign_and_limbs(0, limbs)`. Used heavily in
    /// precomputed constant tables where the sign is always positive.
    #[inline]
    pub const fn from_limbs(limbs: [u64; N]) -> Self {
        Self { sign: 0, limbs }
    }

    /// Creates a negative `BigInt` from `N` little-endian `u64` limbs
    /// representing the absolute value.
    ///
    /// Equivalent to `from_sign_and_limbs(1, limbs)`.
    #[inline]
    pub const fn from_limbs_neg(limbs: [u64; N]) -> Self {
        Self { sign: 1, limbs }
    }

    /// Creates a `BigInt` from an `i64`.
    #[inline]
    pub const fn from_i64(val: i64) -> Self {
        let sign = (val < 0) as u64;
        let abs = val.unsigned_abs();
        let mut limbs = [0u64; N];
        limbs[0] = abs;
        Self { sign, limbs }
    }

    /// Creates a `BigInt` from a `u64` (non-negative).
    #[inline]
    pub const fn from_u64(val: u64) -> Self {
        let mut limbs = [0u64; N];
        limbs[0] = val;
        Self { sign: 0, limbs }
    }
}

impl<const N: usize> From<i64> for BigInt<N> {
    fn from(val: i64) -> Self {
        Self::from_i64(val)
    }
}

impl<const N: usize> From<u64> for BigInt<N> {
    fn from(val: u64) -> Self {
        Self::from_u64(val)
    }
}

impl<const N: usize> From<i32> for BigInt<N> {
    fn from(val: i32) -> Self {
        Self::from_i64(val as i64)
    }
}
