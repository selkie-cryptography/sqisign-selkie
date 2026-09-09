//! Truncating division on [`BigInt<N>`][super::BigInt]:
//! [`vt_div_rem`][BigInt::vt_div_rem], [`vt_mod`][BigInt::vt_mod],
//! [`vt_divides`][BigInt::vt_divides], and the private
//! [`mag_div_rem`](BigInt::mag_div_rem) limb-level helper
//! (Knuth Algorithm D).

use subtle::Choice;

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Truncating division: returns `(quotient, remainder)` with
    /// `self = quotient * divisor + remainder`, the quotient rounded
    /// toward zero and `|remainder| < |divisor|`.
    ///
    /// The remainder takes the dividend's sign, as `/` and `%` do on
    /// Rust's signed integers and as the reference implementation's
    /// `ibz_div` does. [`vt_mod`](Self::vt_mod) gives the non-negative
    /// residue. Magnitude division is Knuth's Algorithm D; the sign
    /// fix-up is constant-time over its inputs.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is zero.
    ///
    /// # Constant-time
    ///
    /// Variable-time on both operands' effective lengths and on the
    /// at-most-one fix-up step inside the magnitude loop. `TODO(ct)`:
    /// a constant-time `ct_div_rem` (with `ct_mod` on top) replaces
    /// [`mag_div_rem`][Self::mag_div_rem] with the divider from
    /// [Kouider et al.][ct-bigint] before any caller on secret-derived
    /// inputs ships.
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    pub fn vt_div_rem(&self, divisor: &Self) -> (Self, Self) {
        assert!(!bool::from(divisor.is_zero()), "division by zero");

        let (q_limbs, r_limbs) = Self::mag_div_rem(&self.limbs, &divisor.limbs);

        // The quotient is negative iff the signs differ; the remainder
        // keeps the dividend's sign. Zero canonicalizes to sign 0.
        let q_sign = (self.sign ^ divisor.sign) & (1 - Self::mag_is_zero(&q_limbs));
        let r_sign = self.sign & (1 - Self::mag_is_zero(&r_limbs));

        (
            Self {
                sign: q_sign,
                limbs: q_limbs,
            },
            Self {
                sign: r_sign,
                limbs: r_limbs,
            },
        )
    }

    /// Returns a [`Choice`] set iff `self` divides `other` evenly.
    ///
    /// Variable-time: built on [`vt_div_rem`](Self::vt_div_rem).
    pub fn vt_divides(&self, other: &Self) -> Choice {
        let (_, r) = other.vt_div_rem(self);
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

        // Divisor normalization is already bounded to its n_b limbs.
        if s == 0 {
            v[..n_b].copy_from_slice(&b[..n_b]);
        } else {
            v[0] = b[0] << s;
            for i in 1..n_b {
                v[i] = (b[i] << s) | (b[i - 1] >> (64 - s));
            }
        }

        // Dividend normalization: shift left by s into u, with the top
        // shift-out in u_hi. Left full-width: a len-bounded shift
        // (loop 1..m_a) was tried and measured within noise -- the Knuth
        // core below is already length-bounded and the [0u64; N] memset is
        // irreducible at fixed width, so it is not worth a var-time opt on
        // this CT-debt path.
        if s == 0 {
            u.copy_from_slice(a);
        } else {
            u[0] = a[0] << s;
            for i in 1..N {
                u[i] = (a[i] << s) | (a[i - 1] >> (64 - s));
            }
            u_hi = a[N - 1] >> (64 - s);
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
}
