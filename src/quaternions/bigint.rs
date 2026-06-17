//! Arbitrary-precision signed integer arithmetic over a fixed-width
//! sign+magnitude representation, for quaternion-algebra computation.
//!
//! # Constant-time vs variable-time (`ct_` / `vt_`)
//!
//! Arithmetic comes in two families, distinguished by prefix:
//!
//! - `ct_*` ([`ct_add`](BigInt::ct_add), [`ct_sub`](BigInt::ct_sub),
//!   [`ct_mul`](BigInt::ct_mul), [`ct_sqr`](BigInt::ct_sqr)) are
//!   **constant-time on every build**: data-independent control flow. `grep
//!   ct_` enumerates the constant-time surface.
//! - `vt_*` ([`vt_mul`](BigInt::vt_mul), [`vt_div_rem`](BigInt::vt_div_rem),
//!   [`vt_mod`](BigInt::vt_mod), [`vt_divides`](BigInt::vt_divides)) are
//!   **variable-time permitted**: a call site that does not require constant
//!   time. `grep vt_` is the variable-time (CT-debt) ledger.
//!
//! The split maps onto the project's two tracks via the `vartime` cargo
//! feature:
//!
//! - **`main`** (feature on): `vt_*` take a variable-time fast path that skips
//!   leading-zero limbs, matching the variable-time C reference.
//! - **`next`** (feature off): [`vt_mul`](BigInt::vt_mul) is exactly
//!   [`ct_mul`](BigInt::ct_mul), hence constant-time.
//!   [`vt_div_rem`](BigInt::vt_div_rem) / [`vt_mod`](BigInt::vt_mod) stay
//!   variable-time on both tracks for now -- a known CT gap (`TODO(ct)`) to be
//!   closed by a constant-time divider ([Kouider et al.][ct-bigint]).
//!
//! Operators route to the build default: `*` is
//! [`vt_mul`](BigInt::vt_mul). The `mag_*` helpers are the unsigned
//! limb-array kernels beneath both families; their timing is documented
//! per helper. API patterns follow [RustCrypto `crypto-bigint`][cb].
//!
//! [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
//! [cb]: https://github.com/RustCrypto/crypto-bigint

use subtle::Choice;

mod add;
mod arch;
mod bits;
mod cmp;
mod cornacchia;
mod ct;
mod div;
mod encoding;
mod from;
mod gcd;
mod mac;
mod modular;
mod mul;
mod neg;
mod pow;
mod primes;
mod rand;
mod resize;
mod shift;
mod sqr;
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

impl<const N: usize> Copy for BigInt<N> where [u64; N]: Copy {}

impl<const N: usize> Default for BigInt<N> {
    fn default() -> Self {
        Self::ZERO
    }
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
}
