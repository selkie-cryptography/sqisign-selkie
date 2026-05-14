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

use core::cmp::Ordering;

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

mod add;
mod bits;
mod cornacchia;
mod div;
mod encoding;
mod gcd;
mod modular;
mod mul;
mod neg;
mod primes;
mod rand;
mod resize;
mod shift;
mod sqrt;
mod sub;
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

    /// Normalizes the representation: ensures zero has sign 0.
    #[inline]
    pub fn normalize(&mut self) {
        let is_zero = bool::from(self.is_zero()) as u64;
        self.sign &= 1 - is_zero;
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

    /// Constant-time conditional select on limb arrays.
    pub(super) fn mag_select(a: &[u64; N], b: &[u64; N], choice: u64) -> [u64; N] {
        let mut result = [0u64; N];
        let mut i = 0;
        while i < N {
            result[i] = ct_select_u64(a[i], b[i], choice);
            i += 1;
        }
        result
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
