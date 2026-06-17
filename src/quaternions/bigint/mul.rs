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
        // aarch64: column-scanning Comba multiply (register-held column
        // accumulator, no per-product result load/store). Wins on the
        // wide lattice multiplies; see `super::arch::aarch64`.
        #[cfg(target_arch = "aarch64")]
        {
            super::arch::aarch64::mag_mul_comba(a, b)
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
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

            // x86 with ADX: column-scanning Comba (3-register accumulator,
            // `mulx` + `add`/`adc`/`adc`). Wins on the wide lattice
            // multiplies, where the result is memory-resident and the
            // register-result dual-chain (`mag_mul_4_adx`) does not apply.
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "adx",
                target_feature = "bmi2",
            ))]
            {
                super::arch::x86_64::mag_mul_comba(a, b)
            }

            // Portable single-`u128` multiply-accumulate (no ADX, or
            // non-x86 non-aarch64 targets): `result + a*b + carry` fits in
            // 128 bits (max `2^128 - 1`), so the high half is the carry,
            // which lowers to a tight `mul`/`adc` chain without the
            // `widening_mul` + double-`overflowing_add` per-limb `cset`.
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "adx",
                target_feature = "bmi2",
            )))]
            {
                let mut result = [0u64; N];
                let mut i = 0;
                while i < N {
                    let mut carry: u64 = 0;
                    let mut j = 0;
                    while j < N - i {
                        let prod =
                            result[i + j] as u128 + a[i] as u128 * b[j] as u128 + carry as u128;
                        result[i + j] = prod as u64;
                        carry = (prod >> 64) as u64;
                        j += 1;
                    }
                    i += 1;
                }
                result
            }
        }
    }

    /// Variable-time schoolbook product over the operands' effective
    /// limb lengths `la`, `lb`, truncated to `N` limbs.
    ///
    /// Byte-identical to [`mag_mul`](Self::mag_mul): the skipped limbs
    /// (`a[la..]`, `b[lb..]`) are zero and contribute nothing. The loop
    /// bounds depend on operand magnitude, hence variable-time; reached
    /// via [`vt_mul`](Self::vt_mul) when `la * lb` is well below `N * N`,
    /// where it beats the full-width arch kernels.
    #[cfg(feature = "vartime")]
    fn mag_mul_short(a: &[u64; N], b: &[u64; N], la: usize, lb: usize) -> [u64; N] {
        let mut result = [0u64; N];

        let mut i = 0;
        while i < la {
            let mut carry: u64 = 0;
            let jmax = core::cmp::min(lb, N - i);

            let mut j = 0;
            while j < jmax {
                let prod = result[i + j] as u128 + a[i] as u128 * b[j] as u128 + carry as u128;
                result[i + j] = prod as u64;
                carry = (prod >> 64) as u64;
                j += 1;
            }

            // b[j >= lb] are zero, so the remainder of the row is pure
            // carry propagation, truncated at N (matching mag_mul's drop
            // of limbs past N).
            let mut k = i + jmax;
            while carry != 0 && k < N {
                let s = result[k] as u128 + carry as u128;
                result[k] = s as u64;
                carry = (s >> 64) as u64;
                k += 1;
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

    /// Variable-time-permitted signed multiplication: returns exactly
    /// what [`ct_mul`](Self::ct_mul) does, but under the `vartime`
    /// feature skips leading-zero limbs (looping over the operands'
    /// effective lengths) where that beats the full-width kernels.
    /// Without the feature it *is* `ct_mul`.
    ///
    /// Use only where constant-time is not required -- the `main`
    /// track's quaternion and lattice arithmetic. On the constant-time
    /// `next` build (feature off) every call site compiles to `ct_mul`.
    pub fn vt_mul(&self, rhs: &Self) -> Self {
        #[cfg(not(feature = "vartime"))]
        {
            self.ct_mul(rhs)
        }

        #[cfg(feature = "vartime")]
        {
            let la = Self::mag_effective_len(&self.limbs);
            let lb = Self::mag_effective_len(&rhs.limbs);
            let result_limbs = if la * lb * 4 < N * N {
                Self::mag_mul_short(&self.limbs, &rhs.limbs, la, lb)
            } else {
                Self::mag_mul(&self.limbs, &rhs.limbs)
            };

            let result_sign = self.sign ^ rhs.sign;
            let is_zero = Self::mag_is_zero(&result_limbs);
            let result_sign = result_sign & (1 - is_zero);

            Self {
                sign: result_sign,
                limbs: result_limbs,
            }
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
