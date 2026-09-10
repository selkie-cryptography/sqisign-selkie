//! Bit shifts on [`BigInt<N>`][super::BigInt]: the [`Shl<u32>`] and
//! [`Shr<u32>`] trait impls (the right shift is arithmetic, rounding
//! toward negative infinity), plus the private
//! [`mag_shl`](BigInt::mag_shl) / [`mag_shr`](BigInt::mag_shr)
//! limb-level helpers.

use core::ops::{Shl, Shr};

use super::{BigInt, ct_select_u64};

impl<const N: usize> BigInt<N> {
    /// Constant-time left shift of magnitude by `s` bits.
    ///
    /// Algorithm 3 (§3.3) from [Kouider et al.][ct-bigint]
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    pub(super) fn mag_shl(a: &[u64; N], s: u32) -> [u64; N] {
        let r = (s % 64) as u64;
        let j = (s / 64) as usize;
        let mut result = [0u64; N];
        // Indicator: 1 while i >= j (within the shifted range).
        let mut l: u64 = 1;
        let mut i = N;
        while i > 0 {
            i -= 1;
            // Zero out limbs below the shift boundary.
            result[i] = ct_select_u64(0, result[i], l);
            // l transitions to 0 when i < j.
            l &= (i >= j) as u64;
            if i >= j {
                // Shift the source limb and OR in the carry from the lower
                // limb.
                let src = if i >= j { a[i - j] } else { 0 };
                let carry = if i > j { a[i - j - 1] } else { 0 };
                // When r == 0 we must avoid shifting by 64 (undefined).
                let shifted = if r == 0 {
                    src
                } else {
                    (src << r) | (carry >> (64 - r))
                };
                result[i] = ct_select_u64(result[i], shifted, l);
            }
        }
        result
    }

    /// Constant-time right shift of magnitude by `s` bits.
    ///
    /// Algorithm 4 (§3.3) from [Kouider et al.][ct-bigint]
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    pub(super) fn mag_shr(a: &[u64; N], s: u32) -> [u64; N] {
        let r = (s % 64) as u64;
        let j = (s / 64) as usize;
        let mut result = [0u64; N];
        let mut i = 0;
        while i < N {
            let src_idx = i + j;
            let carry_idx = i + j + 1;
            let src = if src_idx < N { a[src_idx] } else { 0 };
            let carry = if carry_idx < N { a[carry_idx] } else { 0 };
            result[i] = if r == 0 {
                src
            } else {
                (src >> r) | (carry << (64 - r))
            };
            i += 1;
        }
        result
    }

    /// `floor(self / 2^s)`; backs the `Shr` impls.
    fn shr_floor(&self, s: u32) -> Self {
        let shifted = Self::mag_shr(&self.limbs, s);

        // Shifting back restores the magnitude iff no set bit was
        // dropped; a negative value with dropped bits rounds down.
        let (dropped, _) = Self::mag_sub(&self.limbs, &Self::mag_shl(&shifted, s));
        let round_down = self.sign & (1 - Self::mag_is_zero(&dropped));

        let (bumped, _) = Self::mag_add(&shifted, &Self::ONE.limbs);
        let limbs = Self::mag_select(&shifted, &bumped, round_down);
        Self {
            sign: self.sign & (1 - Self::mag_is_zero(&limbs)),
            limbs,
        }
    }
}

/// Constant-time left shift by `s` bits (multiply by `2^s`).
///
/// Algorithm 3 (§3.3) from [Kouider et al.][ct-bigint]
/// Runs in constant time w.r.t. both the value and the shift amount.
///
/// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
impl<const N: usize> Shl<u32> for BigInt<N> {
    type Output = Self;
    #[inline]
    fn shl(self, rhs: u32) -> Self {
        let limbs = Self::mag_shl(&self.limbs, rhs);
        Self {
            sign: self.sign,
            limbs,
        }
    }
}

impl<const N: usize> Shl<u32> for &BigInt<N> {
    type Output = BigInt<N>;
    #[inline]
    fn shl(self, rhs: u32) -> BigInt<N> {
        let limbs = BigInt::<N>::mag_shl(&self.limbs, rhs);
        BigInt::<N> {
            sign: self.sign,
            limbs,
        }
    }
}

/// Arithmetic right shift: `floor(self / 2^rhs)`.
///
/// Rounds toward negative infinity like `>>` on Rust's signed integers
/// and the reference implementation's `ibz_div_2exp`, so a negative
/// value with nonzero shifted-out bits rounds one further from zero.
/// Constant-time in the value and the shift amount.
impl<const N: usize> Shr<u32> for BigInt<N> {
    type Output = Self;
    #[inline]
    fn shr(self, rhs: u32) -> Self {
        self.shr_floor(rhs)
    }
}

impl<const N: usize> Shr<u32> for &BigInt<N> {
    type Output = BigInt<N>;
    #[inline]
    fn shr(self, rhs: u32) -> BigInt<N> {
        self.shr_floor(rhs)
    }
}
