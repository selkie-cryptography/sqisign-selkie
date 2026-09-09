//! Bit-level operations on [`BigInt<N>`][super::BigInt]:
//! [`bitsize`][BigInt::bitsize], [`is_even`][BigInt::is_even],
//! [`is_odd`][BigInt::is_odd], [`trailing_zeros`][BigInt::trailing_zeros],
//! plus the private limb-level
//! [`mag_effective_len`](BigInt::mag_effective_len),
//! [`mag_trailing_zeros`](BigInt::mag_trailing_zeros), and
//! [`mag_bitsize`](BigInt::mag_bitsize) helpers.

use subtle::Choice;

use super::{BigInt, nbits64, trailing_zeros};

impl<const N: usize> BigInt<N> {
    /// Constant-time bit-size of the magnitude.
    ///
    /// Returns the position of the highest set bit (1-indexed), or 0
    /// if the value is zero. Iterates over all `N` limbs unconditionally.
    ///
    /// Algorithm 2 (§3.2) from [Kouider et al.][ct-bigint]
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    pub fn bitsize(&self) -> u32 {
        let mut k: u32 = 0;
        let mut i = N;
        while i > 0 {
            i -= 1;
            let s = nbits64(self.limbs[i]);
            let r = (k == 0) as u32;
            let t = r * s;
            let m = ((k == 0) as u32) & ((s > 0) as u32);
            k = k + t + (64 * i as u32) * m;
        }
        k
    }

    /// Returns `true` (as `Choice`) if the magnitude is even.
    #[inline]
    pub fn is_even(&self) -> Choice {
        Choice::from(((self.limbs[0] & 1) == 0) as u8)
    }

    /// Returns `true` (as `Choice`) if the magnitude is odd.
    #[inline]
    pub fn is_odd(&self) -> Choice {
        Choice::from((self.limbs[0] & 1) as u8)
    }

    /// Constant-time count of trailing zero bits (2-adic valuation).
    ///
    /// For zero, returns `N * 64` (the reference implementation's
    /// `ibz_two_adic` returns 0 there). Iterates over all `N` limbs and
    /// all 64 bits per limb unconditionally, with no data-dependent
    /// branches or memory accesses.
    pub fn trailing_zeros(&self) -> u32 {
        let mut k: u32 = 0;
        let mut found: u32 = 0;
        let mut i: usize = 0;
        while i < N {
            let limb = self.limbs[i];
            let tz = trailing_zeros(limb);
            // 1 if `limb != 0`, else 0.
            let limb_nz = ((limb | limb.wrapping_neg()) >> 63) as u32;
            // 1 only at the first nonzero limb encountered, low to high.
            let m = limb_nz & (1 - found);
            k += m * (64 * i as u32 + tz);
            found |= limb_nz;
            i += 1;
        }
        // All-zero contract: return `N * 64`.
        k + (1 - found) * (64 * N as u32)
    }

    /// Effective limb count of a magnitude (index of the highest nonzero
    /// limb plus one; 0 for an all-zero input).
    #[inline]
    pub(super) fn mag_effective_len(a: &[u64; N]) -> usize {
        let mut i = N;
        while i > 0 {
            if a[i - 1] != 0 {
                return i;
            }
            i -= 1;
        }
        0
    }

    /// Variable-time trailing-zero count of a magnitude array.
    ///
    /// Returns `N * 64` for an all-zero input. Used by the Stein binary
    /// GCD where variable-time is the design choice.
    pub(super) fn mag_trailing_zeros(a: &[u64; N]) -> u32 {
        let mut i = 0;
        while i < N {
            if a[i] != 0 {
                return (i as u32) * 64 + a[i].trailing_zeros();
            }
            i += 1;
        }
        (N as u32) * 64
    }

    /// Constant-time bitsize of a magnitude array.
    pub(super) fn mag_bitsize(a: &[u64; N]) -> u32 {
        let mut k: u32 = 0;
        let mut i = N;
        while i > 0 {
            i -= 1;
            let s = nbits64(a[i]);
            let r = (k == 0) as u32;
            let t = r * s;
            let m = ((k == 0) as u32) & ((s > 0) as u32);
            k = k + t + (64 * i as u32) * m;
        }
        k
    }
}
