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

    /// Variable-time unsigned addition of magnitudes over the operands'
    /// effective limb lengths. Returns `(limbs, carry)`.
    ///
    /// Byte-identical to [`mag_add`](Self::mag_add): the skipped limbs
    /// (`a[la..]`, `b[lb..]`) are zero, so the columns past
    /// `max(la, lb)` are pure carry propagation, and once the carry
    /// clears the remaining limbs stay zero. The loop bounds depend on
    /// operand magnitude, hence variable-time; reached via
    /// [`vt_add`](Self::vt_add) when both operands sit well below full
    /// width, where skipping the high zero limbs beats the full-width
    /// `adcs` chain.
    fn mag_add_short(a: &[u64; N], b: &[u64; N], la: usize, lb: usize) -> ([u64; N], u64) {
        let mut result = [0u64; N];
        let lmax = core::cmp::max(la, lb);

        let mut carry: u64 = 0;
        let mut i = 0;
        while i < lmax {
            let (s1, c1) = a[i].overflowing_add(b[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            result[i] = s2;
            carry = (c1 | c2) as u64;
            i += 1;
        }

        // Columns past lmax have both operands zero, so they are pure
        // carry propagation; once the carry clears, the rest stay zero
        // (matching mag_add's full-width result).
        while carry != 0 && i < N {
            let (s, c) = result[i].overflowing_add(carry);
            result[i] = s;
            carry = c as u64;
            i += 1;
        }

        (result, carry)
    }

    /// Variable-time unsigned subtraction of magnitudes over the
    /// operands' effective limb lengths. Returns `(limbs, borrow)`,
    /// borrow == 1 iff `a < b` (unsigned).
    ///
    /// Byte-identical to [`mag_sub`](Self::mag_sub): the skipped limbs
    /// are zero, so columns past `max(la, lb)` are pure borrow
    /// propagation. When the borrow is still set after the operand
    /// limbs (i.e. `a < b`), the result is the two's-complement
    /// negation, whose high limbs are all-ones; this loop fills them
    /// explicitly so the output matches the full-width result exactly.
    fn mag_sub_short(a: &[u64; N], b: &[u64; N], la: usize, lb: usize) -> ([u64; N], u64) {
        let mut result = [0u64; N];
        let lmax = core::cmp::max(la, lb);

        let mut borrow: u64 = 0;
        let mut i = 0;
        while i < lmax {
            let (d1, b1) = a[i].overflowing_sub(b[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            result[i] = d2;
            borrow = (b1 | b2) as u64;
            i += 1;
        }

        // Columns past lmax have both operands zero, so they reduce to
        // `0 - borrow`: while a borrow is outstanding each yields
        // `0xFFFF_FFFF_FFFF_FFFF` and re-borrows, exactly as the
        // full-width mag_sub produces for the `a < b` two's-complement
        // case. Once (if) the borrow clears, the rest stay zero.
        while borrow != 0 && i < N {
            result[i] = 0u64.wrapping_sub(borrow);
            i += 1;
        }

        (result, borrow)
    }

    /// Constant-time signed addition.
    ///
    /// If signs match: add magnitudes.
    /// If signs differ: subtract the smaller magnitude from the larger,
    /// with the sign of the larger.
    #[inline]
    pub fn ct_add(&self, rhs: &Self) -> Self {
        let same_sign = ((self.sign ^ rhs.sign) == 0) as u64;

        // Case 1: same sign -> add magnitudes, keep sign.
        let (sum, _carry) = Self::mag_add(&self.limbs, &rhs.limbs);

        // Case 2: different signs -> subtract the smaller magnitude from
        // the larger.  `mag_sub`'s borrow is the ordering (borrow == 1
        // iff self < rhs), so no separate `mag_cmp`; and the reverse
        // difference is the two's-complement negation of the forward
        // one, so no second `mag_sub`.
        let (diff_a, borrow) = Self::mag_sub(&self.limbs, &rhs.limbs);
        let self_ge = 1 - borrow;
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

    /// Variable-time-permitted signed addition: returns exactly what
    /// [`ct_add`](Self::ct_add) does, but skips leading-zero limbs
    /// (looping over the operands' effective lengths) when that beats
    /// the full-width sign-and-magnitude merge.
    ///
    /// Use only where constant-time is not required -- the quaternion
    /// and lattice arithmetic, where variable-time is the accepted
    /// posture. [`ct_add`](Self::ct_add) is preserved in source so a
    /// constant-time build can re-route these call sites back to it.
    #[inline]
    pub fn vt_add(&self, rhs: &Self) -> Self {
        let la = Self::mag_effective_len(&self.limbs);
        let lb = Self::mag_effective_len(&rhs.limbs);

        // Gate: only take the short path when both operands sit well
        // below full width, so the saved high-limb adds outweigh the
        // two O(N) effective-length scans plus the irreducible result
        // memset. K = 4 (same family as vt_mul's `la * lb * 4 < N*N`).
        if core::cmp::max(la, lb) * 4 >= N {
            return self.ct_add(rhs);
        }

        // Same sign-and-magnitude algorithm as ct_add, length-bounded.
        // Being variable-time, the differing-sign branch picks the
        // larger magnitude directly via the borrow rather than the
        // negate + two selects ct_add needs to stay branchless.
        let (result_sign, result_limbs) = if (self.sign ^ rhs.sign) == 0 {
            let (sum, _carry) = Self::mag_add_short(&self.limbs, &rhs.limbs, la, lb);
            (self.sign, sum)
        } else {
            let (diff, borrow) = Self::mag_sub_short(&self.limbs, &rhs.limbs, la, lb);
            if borrow == 0 {
                // self >= rhs: keep self's sign and the forward difference.
                (self.sign, diff)
            } else {
                // self < rhs: magnitude is rhs - self, sign is rhs's.
                let (rdiff, _b) = Self::mag_sub_short(&rhs.limbs, &self.limbs, lb, la);
                (rhs.sign, rdiff)
            }
        };

        // Canonicalize: if result is zero, sign must be 0.
        let is_zero = Self::mag_is_zero(&result_limbs);
        let result_sign = result_sign & (1 - is_zero);

        Self {
            sign: result_sign,
            limbs: result_limbs,
        }
    }

    /// In-place constant-time signed addition: `self += rhs`.
    ///
    /// Branch-free sign-and-magnitude merge identical to [`Self::ct_add`],
    /// writing the result into `self` rather than returning a fresh
    /// [`BigInt`]. The dataflow (and hence the side-channel profile) is
    /// the same; only the destination differs, so this stays CT-neutral
    /// while avoiding the return-value copy that matters at large `N`.
    #[inline]
    pub fn ct_add_assign(&mut self, rhs: &Self) {
        let same_sign = ((self.sign ^ rhs.sign) == 0) as u64;

        // Case 1: same sign -> add magnitudes, keep sign.
        let (sum, _carry) = Self::mag_add(&self.limbs, &rhs.limbs);

        // Case 2: different signs -> subtract the smaller magnitude from
        // the larger.  `mag_sub`'s borrow is the ordering (borrow == 1
        // iff self < rhs), so no separate `mag_cmp`; and the reverse
        // difference is the two's-complement negation of the forward
        // one, so no second `mag_sub`.
        let (diff_a, borrow) = Self::mag_sub(&self.limbs, &rhs.limbs);
        let self_ge = 1 - borrow;
        let diff_b = Self::mag_negate(&diff_a);

        let diff_mag = Self::mag_select(&diff_b, &diff_a, self_ge);
        let diff_sign = ct_select_u64(rhs.sign, self.sign, self_ge);

        let result_limbs = Self::mag_select(&diff_mag, &sum, same_sign);
        let result_sign = ct_select_u64(diff_sign, self.sign, same_sign);

        // Canonicalize: if result is zero, sign must be 0.
        let is_zero = Self::mag_is_zero(&result_limbs);
        self.sign = result_sign & (1 - is_zero);
        self.limbs = result_limbs;
    }
}

impl<const N: usize> core::ops::AddAssign<&BigInt<N>> for BigInt<N> {
    #[inline]
    fn add_assign(&mut self, rhs: &BigInt<N>) {
        self.ct_add_assign(rhs);
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
