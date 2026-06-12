//! Multiplication on [`BigInt<N>`][super::BigInt]: the constant-time
//! [`ct_mul`][BigInt::ct_mul] entry point, the private
//! [`mag_mul`](BigInt::mag_mul) limb-level helper, and the
//! corresponding [`Mul`] trait impls.

use core::ops::Mul;

use super::{BigInt, widening_mul};

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

    /// Schoolbook product where the left operand `a` occupies at most
    /// `lhs_limbs` significant limbs, the rest provably zero.
    ///
    /// Identical result to [`mag_mul`](Self::mag_mul) under that
    /// precondition: the skipped outer rows `a[lhs_limbs..]` are zero and
    /// contribute nothing, and the inner loop over `b` is full, so carry
    /// propagation and truncation match `mag_mul` exactly. Does
    /// `lhs_limbs * N` partial products instead of `N * N`.
    ///
    /// `lhs_limbs` is a fixed, value-independent bound (a function of `N`
    /// and the calling algorithm, never of `a`), so the loop trip counts
    /// do not depend on operand values.
    fn mag_mul_lhs_bounded(a: &[u64; N], b: &[u64; N], lhs_limbs: usize) -> [u64; N] {
        let lhs_limbs = lhs_limbs.min(N);
        let mut result = [0u64; N];
        let mut i = 0;
        while i < lhs_limbs {
            let ai = a[i];
            let mut carry: u64 = 0;
            let mut j = 0;
            while j < N - i {
                let (lo, hi) = widening_mul(ai, b[j]);
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

    /// Schoolbook product where `a` occupies at most `ka` significant
    /// limbs and `b` at most `kb`, the rest provably zero.
    ///
    /// Identical result to [`mag_mul`](Self::mag_mul) under those
    /// preconditions, but does `ka * kb` partial products instead of
    /// `N * N`. Each row's trailing carry lands at `result[i + kb]`, a
    /// position no earlier row has written (the `mpn_mul` layout), so it
    /// is a plain accumulate with no cascade. When `i + j` reaches `N`
    /// the write is dropped, matching `mag_mul`'s truncation. `ka`/`kb`
    /// are fixed, value-independent bounds, so trip counts do not depend
    /// on operand values.
    fn mag_mul_both_bounded(a: &[u64; N], b: &[u64; N], ka: usize, kb: usize) -> [u64; N] {
        let ka = ka.min(N);
        let kb = kb.min(N);
        let mut result = [0u64; N];
        let mut i = 0;
        while i < ka {
            let ai = a[i];
            let mut carry: u64 = 0;
            let jmax = kb.min(N - i);
            let mut j = 0;
            while j < jmax {
                let (lo, hi) = widening_mul(ai, b[j]);
                let (s1, c1) = result[i + j].overflowing_add(lo);
                let (s2, c2) = s1.overflowing_add(carry);
                result[i + j] = s2;
                carry = hi + (c1 as u64) + (c2 as u64);
                j += 1;
            }
            if i + jmax < N {
                result[i + jmax] = result[i + jmax].wrapping_add(carry);
            }
            i += 1;
        }
        result
    }

    /// Signed multiply where `self` occupies at most `lhs_limbs` limbs and
    /// `rhs` at most `rhs_limbs`, both value-independent bounds.
    ///
    /// A faster equivalent of [`ct_mul`](Self::ct_mul) when both operands
    /// have proven limb bounds (e.g. an LLL size-reduction coefficient
    /// `floor(mu)` times a basis entry, each `<= N/2 + 1` limbs from the
    /// L2 norm and Lovasz bounds). The result equals `self.ct_mul(rhs)`
    /// whenever both preconditions hold.
    ///
    /// # Constant-time
    ///
    /// Trip counts depend only on `lhs_limbs`/`rhs_limbs`, never on the
    /// operand values; it does not branch on either operand's width.
    ///
    /// # Correctness
    ///
    /// If either operand exceeds its bound the product is silently
    /// truncated. Callers must pass proven bounds; debug builds assert them.
    #[must_use]
    pub fn ct_mul_both_bounded(&self, rhs: &Self, lhs_limbs: usize, rhs_limbs: usize) -> Self {
        debug_assert!(
            (self.bitsize() as usize).div_ceil(64) <= lhs_limbs
                && (rhs.bitsize() as usize).div_ceil(64) <= rhs_limbs,
            "ct_mul_both_bounded: an operand exceeds its declared limb bound"
        );
        let result_sign = self.sign ^ rhs.sign;
        let result_limbs =
            Self::mag_mul_both_bounded(&self.limbs, &rhs.limbs, lhs_limbs, rhs_limbs);

        let is_zero = Self::mag_is_zero(&result_limbs);
        let result_sign = result_sign & (1 - is_zero);

        Self {
            sign: result_sign,
            limbs: result_limbs,
        }
    }

    /// Signed multiply for a left operand with a known, value-independent
    /// limb bound `lhs_limbs`.
    ///
    /// A faster equivalent of [`ct_mul`](Self::ct_mul) when `self` is
    /// provably at most `lhs_limbs` significant limbs (e.g. an LLL
    /// size-reduction coefficient `floor(mu)`, bounded by `N/2 + 1` limbs
    /// from the L2 norm and Lovasz bounds). The result equals
    /// `self.ct_mul(rhs)` whenever that precondition holds.
    ///
    /// # Constant-time
    ///
    /// The loop counts depend only on `lhs_limbs`, a fixed bound, not on
    /// the operand values; this deliberately does not branch on `self`'s
    /// actual width. It adds no operand-dependent timing beyond the
    /// caller's existing variable-time behavior.
    ///
    /// # Correctness
    ///
    /// If `self` exceeds `lhs_limbs` limbs the product is silently
    /// truncated. Callers must pass a proven bound; debug builds assert it.
    #[must_use]
    pub fn ct_mul_lhs_bounded(&self, rhs: &Self, lhs_limbs: usize) -> Self {
        debug_assert!(
            (self.bitsize() as usize).div_ceil(64) <= lhs_limbs,
            "ct_mul_lhs_bounded: operand exceeds the declared limb bound"
        );
        let result_sign = self.sign ^ rhs.sign;
        let result_limbs = Self::mag_mul_lhs_bounded(&self.limbs, &rhs.limbs, lhs_limbs);

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
