//! Euclidean division on [`BigInt<N>`][super::BigInt]:
//! [`div_rem`][BigInt::div_rem], [`ct_mod`][BigInt::ct_mod],
//! [`divides`][BigInt::divides], and the private
//! [`mag_div_rem`](BigInt::mag_div_rem) limb-level helper
//! (Knuth Algorithm D).

use subtle::Choice;

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Constant-time Euclidean division: returns `(quotient, remainder)`
    /// such that `self = quotient * divisor + remainder` with
    /// `0 <= remainder < |divisor|`.
    ///
    /// The quotient sign follows standard Euclidean division convention:
    /// the remainder is always non-negative.
    ///
    /// Based on Algorithm 5 (§3.4) from [Kouider et al.][ct-bigint],
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    /// operating on magnitudes then adjusting signs.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is zero.
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        assert!(!bool::from(divisor.is_zero()), "division by zero");

        // Compute unsigned division on magnitudes.
        let (q_limbs, r_limbs) = Self::mag_div_rem(&self.limbs, &divisor.limbs);

        // Determine quotient sign: negative if signs differ and quotient != 0.
        let q_sign_raw = self.sign ^ divisor.sign;
        let r_is_zero = Self::mag_is_zero(&r_limbs);

        // Euclidean convention: if remainder is nonzero and the dividend
        // was negative, adjust: q = q + 1, r = |divisor| - r.
        let needs_adjust = self.sign & (1 - r_is_zero);

        // q_adjusted = q_mag + 1 (when adjusting)
        let one = {
            let mut l = [0u64; N];
            l[0] = 1;
            l
        };
        let (q_inc, _) = Self::mag_add(&q_limbs, &one);
        let q_final = Self::mag_select(&q_limbs, &q_inc, needs_adjust);

        // r_adjusted = |divisor| - r (when adjusting)
        let (r_adj, _) = Self::mag_sub(&divisor.limbs, &r_limbs);
        let r_final = Self::mag_select(&r_limbs, &r_adj, needs_adjust);

        // Quotient sign: q_sign_raw, but canonical if zero.
        let q_is_zero = Self::mag_is_zero(&q_final);
        let q_sign = q_sign_raw & (1 - q_is_zero);

        (
            Self {
                sign: q_sign,
                limbs: q_final,
            },
            Self {
                sign: 0,
                limbs: r_final,
            },
        )
    }

    /// Returns `true` if `self` divides `other` evenly.
    pub fn divides(&self, other: &Self) -> Choice {
        let (_, r) = other.div_rem(self);
        r.is_zero()
    }

    /// Unsigned Euclidean division of magnitudes via Knuth's Algorithm D
    /// (TAOCP §4.3.1).
    ///
    /// Returns `(quotient, remainder)` with `a == q·b + r` and `0 ≤ r < b`.
    ///
    /// Single-limb divisor takes a fast path. For multi-limb divisors the
    /// classical schoolbook algorithm is used: normalize divisor so its
    /// top bit is set, estimate each quotient digit from the top two
    /// dividend limbs over the top divisor limb (with a single fix-up
    /// step), multiply-subtract per digit, restore on borrow.
    ///
    /// **Variable-time.** Branches on effective lengths and on the
    /// at-most-1 fix-up step. Constant-time division will be reintroduced
    /// in a separate pass.
    pub(super) fn mag_div_rem(a: &[u64; N], b: &[u64; N]) -> ([u64; N], [u64; N]) {
        let n_b = Self::mag_effective_len(b);
        debug_assert!(n_b > 0, "division by zero");

        let m_a = Self::mag_effective_len(a);
        if m_a < n_b {
            return ([0u64; N], *a);
        }

        // Single-limb divisor: O(N) using u128 division.
        if n_b == 1 {
            let d = b[0] as u128;
            let mut q = [0u64; N];
            let mut r: u128 = 0;
            let mut i = m_a;
            while i > 0 {
                i -= 1;
                let num = (r << 64) | a[i] as u128;
                q[i] = (num / d) as u64;
                r = num % d;
            }
            let mut rem = [0u64; N];
            rem[0] = r as u64;
            return (q, rem);
        }

        // Knuth Algorithm D, multi-limb divisor.
        //
        // Normalization: shift left by `s` so the top bit of the
        // divisor's leading limb is set. The dividend grows by at
        // most one limb (`u_hi`). The divisor never exceeds n_b limbs
        // because we picked `s` to exactly fill the leading limb.
        let s = b[n_b - 1].leading_zeros();
        let mut u = [0u64; N];
        let mut u_hi: u64 = 0;
        let mut v = [0u64; N];
        if s == 0 {
            u.copy_from_slice(a);
            v[..n_b].copy_from_slice(&b[..n_b]);
        } else {
            u[0] = a[0] << s;
            for i in 1..N {
                u[i] = (a[i] << s) | (a[i - 1] >> (64 - s));
            }
            u_hi = a[N - 1] >> (64 - s);

            v[0] = b[0] << s;
            for i in 1..n_b {
                v[i] = (b[i] << s) | (b[i - 1] >> (64 - s));
            }
        }

        let v_hi = v[n_b - 1];
        let v_2nd = v[n_b - 2];

        let mut q = [0u64; N];

        // Quotient has at most (m_a - n_b + 1) limbs at indices [0, m_a - n_b].
        let m = m_a - n_b;
        let mut j = m + 1;
        while j > 0 {
            j -= 1;
            // u[j+n_b] is u_hi when j+n_b == N (only ever once, at j == m).
            let u_top = if j + n_b == N { u_hi } else { u[j + n_b] };
            let u_2nd = u[j + n_b - 1];
            let u_3rd = if j + n_b >= 2 { u[j + n_b - 2] } else { 0 };

            // q_hat estimate. If u_top == v_hi, q_hat would overflow u64,
            // so cap and let the fix-up loop refine.
            let top2 = ((u_top as u128) << 64) | u_2nd as u128;
            let (mut q_hat, mut r_hat): (u128, u128) = if u_top >= v_hi {
                (
                    u128::from(u64::MAX),
                    top2 - (u128::from(u64::MAX) * v_hi as u128),
                )
            } else {
                (top2 / v_hi as u128, top2 % v_hi as u128)
            };

            // Refine: at most 2 decrements (Knuth proves this).
            while r_hat >> 64 == 0 && q_hat * v_2nd as u128 > (r_hat << 64) | u_3rd as u128 {
                q_hat -= 1;
                r_hat += v_hi as u128;
            }
            let q_hat = q_hat as u64;

            // u[j..j+n_b+1] -= q_hat · v[0..n_b].
            // Two-stage borrow: low 64 of (q_hat · v[i] + borrow_in) is
            // subtracted from u[j+i], producing a possible new borrow.
            let mut borrow: u64 = 0;
            for i in 0..n_b {
                let prod = q_hat as u128 * v[i] as u128 + borrow as u128;
                let lo = prod as u64;
                let hi = (prod >> 64) as u64;
                let (diff, borrow_out) = u[j + i].overflowing_sub(lo);
                u[j + i] = diff;
                borrow = hi + borrow_out as u64;
            }
            // Subtract the final carry from the topmost dividend limb.
            let top_borrowed = if j + n_b == N {
                let (new_hi, b1) = u_hi.overflowing_sub(borrow);
                u_hi = new_hi;
                b1
            } else {
                let (new, b1) = u[j + n_b].overflowing_sub(borrow);
                u[j + n_b] = new;
                b1
            };

            if top_borrowed {
                // q_hat was 1 too large. Decrement and add v back.
                let q_hat_corrected = q_hat - 1;
                let mut carry: u64 = 0;
                for i in 0..n_b {
                    let sum = u[j + i] as u128 + v[i] as u128 + carry as u128;
                    u[j + i] = sum as u64;
                    carry = (sum >> 64) as u64;
                }
                // The add-back carry cancels the spurious borrow at the top.
                if j + n_b == N {
                    u_hi = u_hi.wrapping_add(carry);
                } else {
                    u[j + n_b] = u[j + n_b].wrapping_add(carry);
                }
                q[j] = q_hat_corrected;
            } else {
                q[j] = q_hat;
            }
        }

        // Denormalize remainder: shift u (low n_b limbs) right by s.
        let mut r = [0u64; N];
        if s == 0 {
            r[..n_b].copy_from_slice(&u[..n_b]);
        } else {
            for i in 0..n_b - 1 {
                r[i] = (u[i] >> s) | (u[i + 1] << (64 - s));
            }
            r[n_b - 1] = u[n_b - 1] >> s;
        }

        (q, r)
    }

    /// Bit-by-bit long division — kept around as the constant-time
    /// reference for div_rem until the CT pass replaces this.
    #[cfg(any())]
    fn mag_div_rem_bitwise(a: &[u64; N], b: &[u64; N]) -> ([u64; N], [u64; N]) {
        let mut r = *a;
        let mut q = [0u64; N];

        let bs_b = Self::mag_bitsize(b);
        let mut bit = Self::BITS;
        while bit > 0 {
            bit -= 1;

            let u = Self::mag_shl(b, bit);

            let (r_minus_u, borrow) = Self::mag_sub(&r, &u);
            let can_sub = 1 - borrow.min(1);

            let valid_shift = if bs_b == 0 {
                0u64
            } else if bit + bs_b <= Self::BITS {
                1u64
            } else {
                0u64
            };
            let do_sub = can_sub & valid_shift;

            r = Self::mag_select(&r, &r_minus_u, do_sub);

            let q_limb = (bit / 64) as usize;
            let q_bit = bit % 64;
            if q_limb < N {
                q[q_limb] |= do_sub << q_bit;
            }
        }

        (q, r)
    }
}
