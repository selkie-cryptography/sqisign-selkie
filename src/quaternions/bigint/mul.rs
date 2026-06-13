//! Multiplication on [`BigInt<N>`][super::BigInt]: the constant-time
//! [`ct_mul`][BigInt::ct_mul] entry point, the private
//! [`mag_mul`](BigInt::mag_mul) limb-level helper, and the
//! corresponding [`Mul`] trait impls.

use core::ops::Mul;

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Schoolbook multiplication of magnitudes, truncated to `N` limbs.
    ///
    /// Adapted from Table 1 (§3.1) of [Kouider et al.][ct-bigint]
    /// (schoolbook limb-by-limb).  For N=4 under `+adx,+bmi2`,
    /// dispatches to the dual-CF/OF-chain `asm!` variant in
    /// `super::arch::x86_64`; all other N (or other targets) use
    /// the portable schoolbook below.
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    pub(super) fn mag_mul(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "adx",
            target_feature = "bmi2",
        ))]
        if N == 4 {
            // SAFETY: const-N == 4, so the casts are between
            // `&[u64; 4]` and `&[u64; N]` with identical layouts.
            let r4 = unsafe {
                super::arch::x86_64::mag_mul_4_adx(
                    &*(a.as_ptr().cast::<[u64; 4]>()),
                    &*(b.as_ptr().cast::<[u64; 4]>()),
                )
            };
            let mut out = [0u64; N];
            // SAFETY: same const-N == 4.
            unsafe {
                core::ptr::copy_nonoverlapping(r4.as_ptr(), out.as_mut_ptr(), 4);
            }
            return out;
        }

        let mut result = [0u64; N];
        let mut i = 0;
        while i < N {
            let mut carry: u64 = 0;
            let mut j = 0;
            while j < N - i {
                // Single-`u128` multiply-accumulate: `result + a*b + carry`
                // fits in 128 bits (max is `2^128 - 1`), so the high half is
                // the carry. LLVM lowers this to `mul`/`umulh` + an `adds`/
                // `adcs` carry chain, avoiding the `cset`-per-limb the
                // `widening_mul` + double-`overflowing_add` form emits.
                let prod = result[i + j] as u128 + a[i] as u128 * b[j] as u128 + carry as u128;
                result[i + j] = prod as u64;
                carry = (prod >> 64) as u64;
                j += 1;
            }
            i += 1;
        }
        result
    }

    /// Constant-time signed multiplication.
    ///
    /// Sign is XOR of input signs (Table 1, §3.1 of [Kouider et
    /// al.][ct-bigint]).
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    /// Magnitude is schoolbook product, truncated to `N` limbs.
    pub fn ct_mul(&self, rhs: &Self) -> Self {
        let result_sign = self.sign ^ rhs.sign;
        let result_limbs = Self::mag_mul(&self.limbs, &rhs.limbs);

        // Canonicalize zero.
        let is_zero = Self::mag_is_zero(&result_limbs);
        let result_sign = result_sign & (1 - is_zero);

        Self {
            sign: result_sign,
            limbs: result_limbs,
        }
    }
}

impl<const N: usize> Mul for BigInt<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.ct_mul(&rhs)
    }
}

impl<const N: usize> Mul<&BigInt<N>> for BigInt<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Self) -> Self {
        self.ct_mul(rhs)
    }
}

impl<const N: usize> Mul<&BigInt<N>> for &BigInt<N> {
    type Output = BigInt<N>;
    #[inline]
    fn mul(self, rhs: &BigInt<N>) -> BigInt<N> {
        self.ct_mul(rhs)
    }
}
