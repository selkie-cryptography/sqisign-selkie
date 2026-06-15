//! Addition on [`BigInt<N>`][super::BigInt]: the constant-time
//! [`ct_add`][BigInt::ct_add] entry point, the private
//! [`mag_add`](BigInt::mag_add) limb-level helper, and the
//! corresponding [`Add`] trait impls.

use core::ops::Add;

use super::{BigInt, ct_select_u64};

impl<const N: usize> BigInt<N> {
    /// Constant-time unsigned addition of magnitudes. Returns `(limbs, carry)`.
    ///
    /// The `c1 | c2` carry computation is equivalent to `c1 + c2` here
    /// because in a chained add with carry-in ≤ 1, `c1` and `c2` can't
    /// both be 1 — but the bitwise-or form lets LLVM see the carry as
    /// strictly ≤ 1 and emit a tighter `adcs` chain instead of
    /// materializing an intermediate carry register per limb. LLVM
    /// lowers this to a fully-unrolled `adc`/`adcs` chain on both x86_64
    /// and aarch64, so there is no separate single-chain asm path; the
    /// only hand-written add is the fused dual-chain in [`Self::ct_add`].
    #[inline(always)]
    pub(super) const fn mag_add(a: &[u64; N], b: &[u64; N]) -> ([u64; N], u64) {
        let mut result = [0u64; N];
        let mut carry: u64 = 0;
        let mut i = 0;
        while i < N {
            let (s1, c1) = a[i].overflowing_add(b[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            result[i] = s2;
            carry = (c1 | c2) as u64;
            i += 1;
        }
        (result, carry)
    }

    /// Constant-time signed addition.
    ///
    /// If signs match: add magnitudes.
    /// If signs differ: subtract the smaller magnitude from the larger,
    /// with the sign of the larger.
    #[inline]
    pub fn ct_add(&self, rhs: &Self) -> Self {
        let same_sign = ((self.sign ^ rhs.sign) == 0) as u64;

        // The same-sign sum (Case 1) and the different-sign forward
        // difference plus ordering (Case 2) are obtained together. On
        // x86_64 + ADX one fused dual-chain pass yields both (sum on the
        // CF chain via `adcx`, difference on the OF chain via `adox`) --
        // a single read of each operand feeds both results, which LLVM
        // will not generate from two separate calls. Every other target
        // keeps the two portable carry chains, which LLVM already lowers
        // to a tight `adc`/`sbb` (x86) or `adcs`/`sbcs` (aarch64) pair.
        // `a_lt_b == 1` iff self < rhs (the subtraction borrowed), so
        // there is no separate `mag_cmp`, and the reverse difference is
        // the two's-complement negation of the forward one, so no second
        // subtraction.
        // Flat bindings (not a tuple-returning block) on both arms so
        // each lowers identically to the underlying call: a direct
        // destructure of the fused result on x86_64 ADX, and the two
        // separate carry chains on every other target.
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "adx",
            target_feature = "bmi2",
        ))]
        let (sum, diff_a, _carry, a_lt_b) =
            super::arch::x86_64::addsub_n_adx(&self.limbs, &rhs.limbs);

        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "adx",
            target_feature = "bmi2",
        )))]
        let (sum, _carry) = Self::mag_add(&self.limbs, &rhs.limbs);
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "adx",
            target_feature = "bmi2",
        )))]
        let (diff_a, a_lt_b) = Self::mag_sub(&self.limbs, &rhs.limbs);

        let self_ge = 1 - a_lt_b;
        let diff_b = Self::mag_negate(&diff_a);

        let diff_mag = Self::mag_select(&diff_b, &diff_a, self_ge);
        let diff_sign = ct_select_u64(rhs.sign, self.sign, self_ge);

        let result_limbs = Self::mag_select(&diff_mag, &sum, same_sign);
        let result_sign = ct_select_u64(diff_sign, self.sign, same_sign);

        // Canonicalize: if result is zero, sign must be 0.
        let is_zero = Self::mag_is_zero(&result_limbs);
        let result_sign = result_sign & (1 - is_zero);

        Self {
            sign: result_sign,
            limbs: result_limbs,
        }
    }
}

impl<const N: usize> Add for BigInt<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.ct_add(&rhs)
    }
}

impl<const N: usize> Add<&BigInt<N>> for BigInt<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &Self) -> Self {
        self.ct_add(rhs)
    }
}

impl<const N: usize> Add<&BigInt<N>> for &BigInt<N> {
    type Output = BigInt<N>;
    #[inline]
    fn add(self, rhs: &BigInt<N>) -> BigInt<N> {
        self.ct_add(rhs)
    }
}
