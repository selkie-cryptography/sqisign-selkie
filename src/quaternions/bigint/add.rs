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
    /// materializing an intermediate carry register per limb.
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

    /// Returns `(a + b, a - b, sum_carry, a_lt_b)` over the magnitudes,
    /// where `a_lt_b == 1` iff `a < b` (the subtraction borrowed).
    ///
    /// On x86_64 + ADX, for widths past a small-N threshold, both
    /// results come from one dual-chain `addsub_n_adx` pass (sum on the
    /// CF chain, difference on the OF chain); elsewhere from the portable
    /// two-pass [`Self::mag_add`] + [`Self::mag_sub`].  Both paths are
    /// constant-time.
    #[inline]
    fn mag_add_sub(a: &[u64; N], b: &[u64; N]) -> ([u64; N], [u64; N], u64, u64) {
        // The dual-chain pass wins once the carry-chain latency
        // dominates the per-limb load/store overhead; below that the
        // portable inlined adc/sbb chains are tighter.
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "adx",
            target_feature = "bmi2",
        ))]
        if N >= 8 {
            let mut sum = [0u64; N];
            let mut diff = [0u64; N];

            // SAFETY: cfg-gated on +adx,+bmi2; sum and diff have N
            // writable words, a and b N readable words, N >= 8 > 0.
            let (sum_carry, no_borrow) = unsafe {
                super::arch::x86_64::addsub_n_adx(
                    sum.as_mut_ptr(),
                    diff.as_mut_ptr(),
                    a.as_ptr(),
                    b.as_ptr(),
                    N as u64,
                )
            };

            return (sum, diff, sum_carry, 1 - no_borrow);
        }

        let (sum, sum_carry) = Self::mag_add(a, b);
        let (diff, borrow) = Self::mag_sub(a, b);

        (sum, diff, sum_carry, borrow)
    }

    /// Constant-time signed addition.
    ///
    /// If signs match: add magnitudes.
    /// If signs differ: subtract the smaller magnitude from the larger,
    /// with the sign of the larger.
    pub fn ct_add(&self, rhs: &Self) -> Self {
        let same_sign = ((self.sign ^ rhs.sign) == 0) as u64;

        // The same-sign sum and the different-sign forward difference,
        // fused into one dual-chain pass where it pays.  `mag_sub`'s
        // borrow is the ordering (a_lt_b == 1 iff self < rhs), so no
        // separate `mag_cmp`; and the reverse difference is the
        // two's-complement negation of the forward one, so no second
        // subtraction.
        let (sum, diff_a, _carry, a_lt_b) = Self::mag_add_sub(&self.limbs, &rhs.limbs);
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
