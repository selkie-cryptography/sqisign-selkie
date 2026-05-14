//! Comparison on [`BigInt<N>`][super::BigInt]: the [`PartialEq`] /
//! [`Eq`] / [`Ord`] / [`PartialOrd`] trait impls, the constant-time
//! [`ConstantTimeEq`] impl, and the private limb-level
//! [`mag_cmp`](BigInt::mag_cmp), [`mag_eq`](BigInt::mag_eq), and
//! [`mag_is_zero`](BigInt::mag_is_zero) helpers.

use core::cmp::Ordering;

use subtle::{Choice, ConstantTimeEq};

use super::{BigInt, ct_gt_u64};

impl<const N: usize> BigInt<N> {
    /// Constant-time unsigned magnitude comparison.
    pub(super) fn mag_cmp(a: &[u64; N], b: &[u64; N]) -> Ordering {
        let mut gt: u64 = 0;
        let mut lt: u64 = 0;
        let mut i = N;
        while i > 0 {
            i -= 1;
            let undecided = 1 - (gt | lt);
            gt |= undecided & ct_gt_u64(a[i], b[i]);
            lt |= undecided & ct_gt_u64(b[i], a[i]);
        }
        if gt == 1 {
            Ordering::Greater
        } else if lt == 1 {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    }

    /// Constant-time unsigned magnitude equality.
    pub(super) fn mag_eq(a: &[u64; N], b: &[u64; N]) -> u64 {
        let mut acc = 0u64;
        let mut i = 0;
        while i < N {
            acc |= a[i] ^ b[i];
            i += 1;
        }
        (acc == 0) as u64
    }

    /// Constant-time unsigned magnitude is-zero test.
    pub(super) fn mag_is_zero(a: &[u64; N]) -> u64 {
        let mut acc = 0u64;
        let mut i = 0;
        while i < N {
            acc |= a[i];
            i += 1;
        }
        (acc == 0) as u64
    }
}

impl<const N: usize> ConstantTimeEq for BigInt<N> {
    fn ct_eq(&self, other: &Self) -> Choice {
        let both_zero = Self::mag_is_zero(&self.limbs) & Self::mag_is_zero(&other.limbs);
        let same_sign = ((self.sign ^ other.sign) == 0) as u64;
        let same_mag = Self::mag_eq(&self.limbs, &other.limbs);
        Choice::from(((both_zero | (same_sign & same_mag)) != 0) as u8)
    }
}

impl<const N: usize> Eq for BigInt<N> {}

impl<const N: usize> PartialEq for BigInt<N> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<const N: usize> Ord for BigInt<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        let a_neg = bool::from(self.is_negative()) as u64;
        let b_neg = bool::from(other.is_negative()) as u64;
        let a_zero = Self::mag_is_zero(&self.limbs);
        let b_zero = Self::mag_is_zero(&other.limbs);

        let mag_cmp = Self::mag_cmp(&self.limbs, &other.limbs);

        if a_neg == 0 && b_neg == 0 {
            if a_zero == 1 && b_zero == 1 {
                Ordering::Equal
            } else {
                mag_cmp
            }
        } else if a_neg == 1 && b_neg == 1 {
            mag_cmp.reverse()
        } else if a_neg == 1 {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

impl<const N: usize> PartialOrd for BigInt<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
