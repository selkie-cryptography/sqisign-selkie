//! Constant-time arbitrary-precision signed integer arithmetic.
//!
//! Fixed-width sign+magnitude representation for use in quaternion
//! algebra computations. Constant-time algorithms adapted from
//! [Kouider, Mukherjee, Jacquemin, and Kutas][ct-bigint].
//!
//! API patterns modeled after [RustCrypto `crypto-bigint`][cb].
//!
//! [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
//! [cb]: https://github.com/RustCrypto/crypto-bigint

use core::{
    cmp::Ordering,
    ops::{Mul, Neg, Sub},
};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

mod add;
mod cornacchia;
mod encoding;
mod gcd;
mod modular;
mod primes;
mod rand;
mod resize;
mod sqrt;
pub(crate) use modular::MontReducer;

#[cfg(test)]
mod tests;

/// Constant-time bit-size of a single 64-bit word.
///
/// Returns the position of the highest set bit (1-indexed), or 0 if `x == 0`.
/// Runs in constant time by iterating over all 64 bits unconditionally.
///
/// Algorithm 1 (§3.1) from [Kouider et al.][ct-bigint]
///
/// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
#[inline]
const fn nbits64(mut x: u64) -> u32 {
    let mut n: u32 = 0;
    let mut i: u32 = 0;
    while i < 64 {
        // If x != 0 then (x | x.wrapping_neg()) >> 63 == 1, else 0.
        n += ((x | x.wrapping_neg()) >> 63) as u32;
        x >>= 1;
        i += 1;
    }
    n
}

/// Constant-time trailing-zero count of a single 64-bit word.
///
/// Returns the position of the lowest set bit (0-indexed), or `64` if
/// `x == 0`. Iterates over all 64 bits unconditionally — does not call
/// `u64::trailing_zeros`, whose CT status depends on target codegen.
#[inline]
const fn trailing_zeros(x: u64) -> u32 {
    let mut tz: u32 = 0;
    let mut found: u32 = 0;
    let mut i: u32 = 0;
    while i < 64 {
        let bit = ((x >> i) & 1) as u32;
        // `m == 1` exactly at the first set bit, `0` thereafter and before.
        let m = bit & (1 - found);
        // Latch position into `tz` on that single iteration; no-op otherwise.
        tz += m * i;
        found |= bit;
        i += 1;
    }
    // If no bit was ever set, `found == 0` and we return 64.
    tz + (1 - found) * 64
}

/// Constant-time conditional move: returns `a` if `choice == 0`, `b` if
/// `choice == 1`. No branching.
#[inline]
const fn ct_select_u64(a: u64, b: u64, choice: u64) -> u64 {
    let mask = choice.wrapping_neg();
    a ^ (mask & (a ^ b))
}

/// Constant-time unsigned comparison: returns 1 if a > b, else 0.
#[inline]
const fn ct_gt_u64(a: u64, b: u64) -> u64 {
    let (_, borrow) = b.overflowing_sub(a);
    borrow as u64
}

/// 64x64 -> 128-bit widening multiplication.
#[inline]
const fn widening_mul(a: u64, b: u64) -> (u64, u64) {
    let full = (a as u128) * (b as u128);
    (full as u64, (full >> 64) as u64)
}

/// A fixed-width signed integer in sign+magnitude representation.
///
/// Constant-time with respect to the value: all operations iterate over
/// all `N` limbs unconditionally. The representation `D(a) = (sign,
/// [a_{N-1}, ..., a_0])` follows [Kouider et al.][ct-bigint] §2.1.
///
/// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
///
/// The sign is stored as a `u64`: 0 for non-negative, 1 for negative.
/// Zero is always represented with `sign == 0` (canonical form).
///
/// API patterns follow [RustCrypto `crypto-bigint`][cb].
///
/// [cb]: https://github.com/RustCrypto/crypto-bigint
#[derive(Clone)]
pub struct BigInt<const N: usize> {
    /// 0 for non-negative, 1 for negative.
    sign: u64,
    /// Magnitude in little-endian order: `limbs[0]` is the least significant.
    limbs: [u64; N],
}

impl<const N: usize> BigInt<N> {
    /// Zero.
    pub const ZERO: Self = Self {
        sign: 0,
        limbs: [0u64; N],
    };

    /// One.
    pub const ONE: Self = Self {
        sign: 0,
        limbs: {
            let mut l = [0u64; N];
            l[0] = 1;
            l
        },
    };

    /// Negative one.
    pub const MINUS_ONE: Self = Self {
        sign: 1,
        limbs: {
            let mut l = [0u64; N];
            l[0] = 1;
            l
        },
    };

    /// Two.
    pub const TWO: Self = Self {
        sign: 0,
        limbs: {
            let mut l = [0u64; N];
            l[0] = 2;
            l
        },
    };

    /// Three.
    pub const THREE: Self = Self {
        sign: 0,
        limbs: {
            let mut l = [0u64; N];
            l[0] = 3;
            l
        },
    };

    /// Total number of bits in the magnitude.
    pub const BITS: u32 = (N as u32) * 64;

    /// Total number of bytes in the magnitude.
    pub const BYTES: usize = N * 8;

    /// Creates a `BigInt` from a sign bit and little-endian limb array.
    #[inline]
    pub const fn from_sign_and_limbs(sign: u64, limbs: [u64; N]) -> Self {
        Self { sign, limbs }
    }

    /// Creates a non-negative `BigInt` from `N` little-endian `u64` limbs.
    ///
    /// Equivalent to `from_sign_and_limbs(0, limbs)`. Used heavily in
    /// precomputed constant tables where the sign is always positive.
    #[inline]
    pub const fn from_limbs(limbs: [u64; N]) -> Self {
        Self { sign: 0, limbs }
    }

    /// Creates a negative `BigInt` from `N` little-endian `u64` limbs
    /// representing the absolute value.
    ///
    /// Equivalent to `from_sign_and_limbs(1, limbs)`.
    #[inline]
    pub const fn from_limbs_neg(limbs: [u64; N]) -> Self {
        Self { sign: 1, limbs }
    }

    /// Creates a `BigInt` from an `i64`.
    #[inline]
    pub const fn from_i64(val: i64) -> Self {
        let sign = (val < 0) as u64;
        let abs = val.unsigned_abs();
        let mut limbs = [0u64; N];
        limbs[0] = abs;
        Self { sign, limbs }
    }

    /// Creates a `BigInt` from a `u64` (non-negative).
    #[inline]
    pub const fn from_u64(val: u64) -> Self {
        let mut limbs = [0u64; N];
        limbs[0] = val;
        Self { sign: 0, limbs }
    }

    /// Returns `true` (as `Choice`) if this value is zero.
    #[inline]
    pub fn is_zero(&self) -> Choice {
        let mut acc = 0u64;
        let mut i = 0;
        while i < N {
            acc |= self.limbs[i];
            i += 1;
        }
        Choice::from((acc == 0) as u8)
    }

    /// Returns `true` (as `Choice`) if this value is negative.
    #[inline]
    pub fn is_negative(&self) -> Choice {
        let neg = self.sign & 1;
        let nonzero = !bool::from(self.is_zero()) as u64;
        Choice::from((neg & nonzero) as u8)
    }

    /// Returns `true` (as `Choice`) if this value is positive (> 0).
    #[inline]
    pub fn is_positive(&self) -> Choice {
        let nonneg = 1 - (self.sign & 1);
        let nonzero = !bool::from(self.is_zero()) as u64;
        Choice::from((nonneg & nonzero) as u8)
    }

    /// Returns the absolute value.
    #[inline]
    pub fn abs(&self) -> Self {
        Self {
            sign: 0,
            limbs: self.limbs,
        }
    }

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
    /// For zero, returns `N * 64`. Iterates over all `N` limbs and all
    /// 64 bits per limb unconditionally, with no data-dependent branches
    /// or memory accesses. Called on secret-derived values in
    /// `SuitableIdeals` (dyadic valuation of `gcd(u, v)`).
    ///
    /// Counterpart to `ibz_two_adic` in the C reference, which delegates
    /// to GMP's variable-time `mpz_scan1`.
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

    /// Negation. Flips the sign bit.
    ///
    /// Table 1 (§3.1) from [Kouider et al.][ct-bigint]: `c_sign = 1 XOR
    /// a_sign`.
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    #[inline]
    pub fn wrapping_neg(&self) -> Self {
        Self {
            sign: self.sign ^ 1,
            limbs: self.limbs,
        }
    }

    /// Normalizes the representation: ensures zero has sign 0.
    #[inline]
    pub fn normalize(&mut self) {
        let is_zero = bool::from(self.is_zero()) as u64;
        self.sign &= 1 - is_zero;
    }

    /// Constant-time left shift by `s` bits (multiply by 2^s).
    ///
    /// Algorithm 3 (§3.3) from [Kouider et al.][ct-bigint]
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    /// Runs in constant time w.r.t. both the value and the shift amount.
    pub fn shl(&self, s: u32) -> Self {
        let limbs = Self::mag_shl(&self.limbs, s);
        Self {
            sign: self.sign,
            limbs,
        }
    }

    /// Constant-time right shift by `s` bits (divide by 2^s, rounding
    /// toward zero).
    ///
    /// Algorithm 4 (§3.3) from [Kouider et al.][ct-bigint]
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    pub fn shr(&self, s: u32) -> Self {
        let limbs = Self::mag_shr(&self.limbs, s);
        // Canonicalize zero.
        let is_zero = Self::mag_is_zero(&limbs);
        Self {
            sign: self.sign & (1 - is_zero),
            limbs,
        }
    }

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

    /// Constant-time modular reduction: `self mod modulus`.
    ///
    /// Returns a value in `[0, |modulus|)`. Uses Euclidean division.
    ///
    /// # Panics
    ///
    /// Panics if `modulus` is zero.
    #[inline]
    pub fn ct_mod(&self, modulus: &Self) -> Self {
        let (_, r) = self.div_rem(modulus);
        r
    }

    /// Returns the 2-adic valuation: the number of trailing zero bits
    /// in the magnitude. Returns 0 for zero.
    pub fn two_adic_val(&self) -> u32 {
        // Count trailing zeros in constant time by iterating all limbs.
        let mut count: u32 = 0;
        let mut still_zero = 1u64; // 1 while all limbs so far are zero
        let mut i = 0;
        while i < N {
            let limb = self.limbs[i];
            // CT trailing zeros for this limb: if limb == 0, contribute 64;
            // otherwise contribute trailing_zeros(limb).
            let limb_nonzero = ((limb | limb.wrapping_neg()) >> 63) as u32;
            let tz = if limb == 0 { 64 } else { limb.trailing_zeros() };
            // Only count if all previous limbs were zero.
            count += (still_zero as u32) * tz;
            // Once we hit a nonzero limb, stop counting.
            still_zero &= 1 - (limb_nonzero as u64);
            i += 1;
        }
        count
    }

    /// Integer exponentiation: `self^exp`.
    ///
    /// Uses a simple square-and-multiply. The exponent is public (not
    /// constant-time w.r.t. the exponent value).
    pub fn pow(&self, exp: u32) -> Self {
        if exp == 0 {
            return Self::ONE;
        }
        let mut result = Self::ONE;
        let mut base = *self;
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result = result.ct_mul(&base);
            }
            base = base.ct_mul(&base);
            e >>= 1;
        }
        result
    }

    /// Returns `true` if `self` divides `other` evenly.
    pub fn divides(&self, other: &Self) -> Choice {
        let (_, r) = other.div_rem(self);
        r.is_zero()
    }

    /// Constant-time unsigned subtraction of magnitudes. Returns `(limbs,
    /// borrow)`. Borrow is 1 if `a < b` (unsigned).
    ///
    /// Same `b1 | b2` simplification as in [`Self::mag_add`] for tighter
    /// `sbcs` chain codegen.
    #[inline(always)]
    const fn mag_sub(a: &[u64; N], b: &[u64; N]) -> ([u64; N], u64) {
        let mut result = [0u64; N];
        let mut borrow: u64 = 0;
        let mut i = 0;
        while i < N {
            let (d1, b1) = a[i].overflowing_sub(b[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            result[i] = d2;
            borrow = (b1 | b2) as u64;
            i += 1;
        }
        (result, borrow)
    }

    /// Constant-time unsigned magnitude comparison.
    fn mag_cmp(a: &[u64; N], b: &[u64; N]) -> Ordering {
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
    fn mag_eq(a: &[u64; N], b: &[u64; N]) -> u64 {
        let mut acc = 0u64;
        let mut i = 0;
        while i < N {
            acc |= a[i] ^ b[i];
            i += 1;
        }
        (acc == 0) as u64
    }

    /// Constant-time unsigned magnitude is-zero test.
    fn mag_is_zero(a: &[u64; N]) -> u64 {
        let mut acc = 0u64;
        let mut i = 0;
        while i < N {
            acc |= a[i];
            i += 1;
        }
        (acc == 0) as u64
    }

    /// Constant-time conditional select on limb arrays.
    fn mag_select(a: &[u64; N], b: &[u64; N], choice: u64) -> [u64; N] {
        let mut result = [0u64; N];
        let mut i = 0;
        while i < N {
            result[i] = ct_select_u64(a[i], b[i], choice);
            i += 1;
        }
        result
    }

    /// Constant-time left shift of magnitude by `s` bits.
    ///
    /// Algorithm 3 (§3.3) from [Kouider et al.][ct-bigint]
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    fn mag_shl(a: &[u64; N], s: u32) -> [u64; N] {
        let r = (s % 64) as u64;
        let j = (s / 64) as usize;
        let mut result = [0u64; N];
        // Indicator: 1 while i >= j (within the shifted range).
        let mut l: u64 = 1;
        let mut i = N;
        while i > 0 {
            i -= 1;
            // Zero out limbs below the shift boundary.
            result[i] = ct_select_u64(0, result[i], l);
            // l transitions to 0 when i < j.
            l &= (i >= j) as u64;
            if i >= j {
                // Shift the source limb and OR in the carry from the lower limb.
                let src = if i >= j { a[i - j] } else { 0 };
                let carry = if i > j { a[i - j - 1] } else { 0 };
                // When r == 0 we must avoid shifting by 64 (undefined).
                let shifted = if r == 0 {
                    src
                } else {
                    (src << r) | (carry >> (64 - r))
                };
                result[i] = ct_select_u64(result[i], shifted, l);
            }
        }
        result
    }

    /// Constant-time right shift of magnitude by `s` bits.
    ///
    /// Algorithm 4 (§3.3) from [Kouider et al.][ct-bigint]
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    fn mag_shr(a: &[u64; N], s: u32) -> [u64; N] {
        let r = (s % 64) as u64;
        let j = (s / 64) as usize;
        let mut result = [0u64; N];
        let mut i = 0;
        while i < N {
            let src_idx = i + j;
            let carry_idx = i + j + 1;
            let src = if src_idx < N { a[src_idx] } else { 0 };
            let carry = if carry_idx < N { a[carry_idx] } else { 0 };
            result[i] = if r == 0 {
                src
            } else {
                (src >> r) | (carry << (64 - r))
            };
            i += 1;
        }
        result
    }

    /// Effective limb count of a magnitude (index of the highest nonzero
    /// limb plus one; 0 for an all-zero input).
    #[inline]
    fn mag_effective_len(a: &[u64; N]) -> usize {
        let mut i = N;
        while i > 0 {
            if a[i - 1] != 0 {
                return i;
            }
            i -= 1;
        }
        0
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
    fn mag_div_rem(a: &[u64; N], b: &[u64; N]) -> ([u64; N], [u64; N]) {
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

    /// Variable-time trailing-zero count of a magnitude array.
    ///
    /// Returns `N * 64` for an all-zero input. Used by the Stein binary
    /// GCD where vartime is the design choice.
    fn mag_trailing_zeros(a: &[u64; N]) -> u32 {
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
    fn mag_bitsize(a: &[u64; N]) -> u32 {
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

    /// Schoolbook multiplication of magnitudes, truncated to `N` limbs.
    ///
    /// Adapted from Table 1 (§3.1) of [Kouider et al.][ct-bigint] (schoolbook
    /// limb-by-limb).
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    fn mag_mul(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
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

    /// Constant-time signed subtraction: `self - rhs`.
    #[inline]
    pub fn ct_sub(&self, rhs: &Self) -> Self {
        let neg_rhs = rhs.wrapping_neg();
        self.ct_add(&neg_rhs)
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

impl<const N: usize> Copy for BigInt<N> where [u64; N]: Copy {}

impl<const N: usize> Default for BigInt<N> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const N: usize> From<i64> for BigInt<N> {
    fn from(val: i64) -> Self {
        Self::from_i64(val)
    }
}

impl<const N: usize> From<u64> for BigInt<N> {
    fn from(val: u64) -> Self {
        Self::from_u64(val)
    }
}

impl<const N: usize> From<i32> for BigInt<N> {
    fn from(val: i32) -> Self {
        Self::from_i64(val as i64)
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

impl<const N: usize> ConditionallySelectable for BigInt<N> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let c = choice.unwrap_u8() as u64;
        Self {
            sign: ct_select_u64(a.sign, b.sign, c),
            limbs: Self::mag_select(&a.limbs, &b.limbs, c),
        }
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

impl<const N: usize> Sub for BigInt<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.ct_sub(&rhs)
    }
}

impl<const N: usize> Sub<&BigInt<N>> for BigInt<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &Self) -> Self {
        self.ct_sub(rhs)
    }
}

impl<const N: usize> Sub<&BigInt<N>> for &BigInt<N> {
    type Output = BigInt<N>;
    #[inline]
    fn sub(self, rhs: &BigInt<N>) -> BigInt<N> {
        self.ct_sub(rhs)
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

impl<const N: usize> Neg for BigInt<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}

impl<const N: usize> Neg for &BigInt<N> {
    type Output = BigInt<N>;
    #[inline]
    fn neg(self) -> BigInt<N> {
        self.wrapping_neg()
    }
}
