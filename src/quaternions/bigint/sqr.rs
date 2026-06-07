//! Squaring on [`BigInt<N>`][super::BigInt]: the constant-time
//! [`ct_sqr`][BigInt::ct_sqr] entry point, the inherent
//! [`square`][BigInt::square] method, and the private
//! [`mag_sqr`](BigInt::mag_sqr) limb-level helper that exploits
//! cross-term symmetry.
//!
//! For N=4 under `+adx,+bmi2`, dispatches to `mag_sqr_4_adx` in
//! `super::arch::x86_64` (6 mulx vs the 10 of
//! [`mag_mul`][BigInt::mag_mul] applied to `(a, a)`).  All other
//! sizes use the portable schoolbook below, which mirrors the
//! cross-term recipe used by `super::modular`'s wide squaring,
//! specialized to the truncated `N`-limb output.

use super::{BigInt, widening_mul};

impl<const N: usize> BigInt<N> {
    /// Schoolbook squaring of a magnitude, truncated to `N` limbs.
    ///
    /// Exploits the symmetry `a[i]·a[j] == a[j]·a[i]` to issue roughly
    /// `N·(N+1)/2` limb-mults vs the `N^2` of [`Self::mag_mul`]`(a, a)`.
    /// For N=4 that's 6 mults instead of 10 (40% fewer); under
    /// `+adx,+bmi2` dispatches to the dual-CF/OF-chain `asm!` variant
    /// in `super::arch::x86_64`.
    pub(super) fn mag_sqr(a: &[u64; N]) -> [u64; N] {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "adx",
            target_feature = "bmi2",
        ))]
        if N == 4 {
            // SAFETY: const-N == 4 — casts are between `&[u64; 4]` and
            // `&[u64; N]` with identical layouts.
            let r4 =
                unsafe { super::arch::x86_64::mag_sqr_4_adx(&*(a.as_ptr().cast::<[u64; 4]>())) };

            let mut out = [0u64; N];
            // SAFETY: same const-N == 4.
            unsafe {
                core::ptr::copy_nonoverlapping(r4.as_ptr(), out.as_mut_ptr(), 4);
            }

            return out;
        }

        // Portable schoolbook with cross-term symmetry, truncated to N limbs.
        //
        // Phase 1: lower-triangle cross products a[i]·a[j] (i<j),
        //   accumulated at position i+j of an N-limb buffer.
        //   Contributions to positions >= N are dropped.
        // Phase 2: double the buffer in place (each cross appears twice
        //   in a²); the bit shifted out of position N-1 is dropped.
        // Phase 3: add diagonals a[i]² at position 2i, dropping
        //   contributions to positions >= N.

        let mut result = [0u64; N];

        let mut i = 0;
        while i < N {
            let mut carry: u64 = 0;
            let mut j = i + 1;

            while j < N {
                let pos = i + j;
                if pos >= N {
                    // Higher positions are discarded by truncation; the
                    // carry chain stops with them.
                    break;
                }

                let (lo, hi) = widening_mul(a[i], a[j]);
                let (s1, c1) = result[pos].overflowing_add(lo);
                let (s2, c2) = s1.overflowing_add(carry);
                result[pos] = s2;

                // Next iter writes to pos+1; accumulate the high half
                // plus this iter's two carries into `carry`.  `hi` is
                // at most 2^64 - 2 for a non-degenerate limb product,
                // leaving room for the two carry bits.
                carry = hi.wrapping_add(c1 as u64).wrapping_add(c2 as u64);
                j += 1;
            }

            i += 1;
        }

        // Phase 2: double the truncated buffer.  The bit shifted out
        // of the top is discarded by truncation.
        let mut carry: u64 = 0;
        let mut k = 0;
        while k < N {
            let new = (result[k] << 1) | carry;
            carry = result[k] >> 63;
            result[k] = new;
            k += 1;
        }

        // Phase 3: add diagonals a[i]² at position 2i, propagating
        // carries within the truncated range.
        let mut i = 0;
        while i < N {
            let pos_lo = 2 * i;
            if pos_lo >= N {
                break;
            }

            let (lo, hi) = widening_mul(a[i], a[i]);
            let (s, c) = result[pos_lo].overflowing_add(lo);
            result[pos_lo] = s;
            let mut carry = c as u64;

            let pos_hi = pos_lo + 1;
            if pos_hi < N {
                let (s2, c2) = result[pos_hi].overflowing_add(hi);
                let (s3, c3) = s2.overflowing_add(carry);
                result[pos_hi] = s3;
                carry = (c2 as u64) | (c3 as u64);

                let mut p = pos_hi + 1;
                while p < N && carry != 0 {
                    let (s4, c4) = result[p].overflowing_add(carry);
                    result[p] = s4;
                    carry = c4 as u64;
                    p += 1;
                }
            }

            i += 1;
        }

        result
    }

    /// Constant-time signed squaring.  The result is always
    /// non-negative.
    ///
    /// Magnitude is [`Self::mag_sqr`], truncated to `N` limbs.
    pub fn ct_sqr(&self) -> Self {
        let result_limbs = Self::mag_sqr(&self.limbs);

        Self {
            sign: 0,
            limbs: result_limbs,
        }
    }

    /// Inherent `self · self`.  Equivalent to `&self * &self` but goes
    /// through the cross-term-symmetric [`Self::ct_sqr`] rather than
    /// [`Self::ct_mul`], saving roughly 40% of limb-mults for N=4.
    #[must_use]
    pub fn square(&self) -> Self {
        self.ct_sqr()
    }
}
