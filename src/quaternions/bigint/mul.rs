//! Multiplication on [`BigInt<N>`][super::BigInt]: the constant-time
//! [`ct_mul`][BigInt::ct_mul] entry point, the private
//! [`mag_mul`](BigInt::mag_mul) limb-level helper, and the
//! corresponding [`Mul`] trait impls.

use core::ops::Mul;

use super::{BigInt, widening_mul};

impl<const N: usize> BigInt<N> {
    /// Schoolbook multiplication of magnitudes, truncated to `N` limbs.
    ///
    /// Dispatches by width: `N == 4` under x86-64 `+adx,+bmi2` uses the
    /// dual-CF/OF-chain `asm!` variant in [`super::arch::x86_64`];
    /// `N <= 16` uses the fixed-width schoolbook
    /// ([`Self::mag_mul_fixed`]); `N > 16` uses the leading-zero-skipping
    /// schoolbook ([`Self::mag_mul_effective`]). The two portable paths
    /// produce identical limbs (Table 1, §3.1 of [Kouider et
    /// al.][ct-bigint]); they differ only in timing.
    ///
    /// # Constant-time
    ///
    /// The `N <= 16` path is data-independent and covers every
    /// constant-time-required caller: the `BigInt<4>`
    /// `EndomorphismAction` action on secret torsion (field arithmetic
    /// uses no `ct_mul`; `MontReducer`'s modexp uses its own limb
    /// loops). The `N > 16` path is reached only by the variable-time
    /// quaternion/lattice/HNF arithmetic (working widths 30/48/110), so
    /// its operand-magnitude-dependent timing sits on that side's
    /// existing `TODO(ct)` debt rather than in a constant-time path.
    /// `TODO(ct)`: folded into the quaternion-side constant-time pass,
    /// which reworks these multiplies.
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

        if N <= 16 {
            Self::mag_mul_fixed(a, b)
        } else {
            Self::mag_mul_effective(a, b)
        }
    }

    /// Fixed-width truncated schoolbook product.
    ///
    /// Iterates the full `N x (N - i)` limb grid regardless of operand
    /// values, so its timing is independent of the operands.
    fn mag_mul_fixed(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let mut result = [0u64; N];
        let mut i = 0;
        while i < N {
            let mut carry: u64 = 0;
            let mut j = 0;
            while j < N - i {
                let (lo, hi) = widening_mul(a[i], b[j]);
                let (s1, c1) = result[i + j].overflowing_add(lo);
                let (s2, c2) = s1.overflowing_add(carry);
                result[i + j] = s2;
                carry = hi + (c1 as u64) + (c2 as u64);
                j += 1;
            }
            i += 1;
        }
        result
    }

    /// Truncated schoolbook product that skips leading-zero limbs.
    ///
    /// Byte-identical to [`Self::mag_mul_fixed`] (a zero limb
    /// contributes nothing), but bounds the loops by the operands'
    /// effective limb lengths and then propagates the final carry
    /// upward, matching the fixed loop's continued zero-limb carry
    /// propagation and its truncation past `N`. At the wide quaternion
    /// widths the operands are routinely far below `N` limbs (worst-case
    /// provisioned), so skipping the zero grid roughly halves the work.
    /// Timing depends on operand magnitude; see [`Self::mag_mul`]'s
    /// `# Constant-time` note.
    fn mag_mul_effective(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let la = {
            let mut k = N;
            while k > 0 && a[k - 1] == 0 {
                k -= 1;
            }
            k
        };
        let lb = {
            let mut k = N;
            while k > 0 && b[k - 1] == 0 {
                k -= 1;
            }
            k
        };

        let mut result = [0u64; N];
        let mut i = 0;
        while i < la {
            let ai = a[i];
            if ai != 0 {
                let mut carry: u64 = 0;
                let jmax = (N - i).min(lb);
                let mut j = 0;
                while j < jmax {
                    let (lo, hi) = widening_mul(ai, b[j]);
                    let (s1, c1) = result[i + j].overflowing_add(lo);
                    let (s2, c2) = s1.overflowing_add(carry);
                    result[i + j] = s2;
                    carry = hi + (c1 as u64) + (c2 as u64);
                    j += 1;
                }

                let mut k = i + jmax;
                while carry != 0 && k < N {
                    let (s, c) = result[k].overflowing_add(carry);
                    result[k] = s;
                    carry = c as u64;
                    k += 1;
                }
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
