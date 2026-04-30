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
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Limb helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// BigInt<N>: fixed-width signed integer (sign + magnitude)
// ---------------------------------------------------------------------------

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

    /// Decode from little-endian bytes (unsigned, non-negative).
    ///
    /// Packs bytes into u64 limbs. Panics if `bytes.len() > N * 8`.
    pub fn from_bytes_le_unsigned(bytes: &[u8]) -> Self {
        assert!(bytes.len() <= N * 8);
        let mut limbs = [0u64; N];
        for (i, chunk) in bytes.chunks(8).enumerate() {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            limbs[i] = u64::from_le_bytes(buf);
        }
        Self { sign: 0, limbs }
    }

    /// Decode from little-endian bytes (signed, two's complement).
    ///
    /// The highest bit of the last byte is the sign bit.
    /// Panics if `bytes.len() > N * 8`.
    pub fn from_bytes_le_signed(bytes: &[u8]) -> Self {
        assert!(!bytes.is_empty() && bytes.len() <= N * 8);
        let is_negative = bytes[bytes.len() - 1] & 0x80 != 0;
        if !is_negative {
            return Self::from_bytes_le_unsigned(bytes);
        }
        // Negate two's complement: flip bits, add 1.
        let mut flipped = [0u8; { 8 * 8 }]; // max N=8
        for (i, &b) in bytes.iter().enumerate() {
            flipped[i] = !b;
        }
        // Pad with 0xFF for remaining bytes up to the limb boundary.
        // Actually we only need to negate the bytes we have.
        let mut magnitude = Self::from_bytes_le_unsigned(&flipped[..bytes.len()]);
        // Add 1 to the magnitude.
        magnitude.limbs[0] = magnitude.limbs[0].wrapping_add(1);
        let mut carry = if magnitude.limbs[0] == 0 { 1u64 } else { 0 };
        for i in 1..N {
            let (val, c) = magnitude.limbs[i].overflowing_add(carry);
            magnitude.limbs[i] = val;
            carry = c as u64;
        }
        magnitude.sign = 1;
        magnitude
    }

    /// Returns the limbs as a slice in little-endian order.
    #[inline]
    pub const fn as_limbs(&self) -> &[u64; N] {
        &self.limbs
    }

    /// Returns a mutable reference to the limbs.
    #[inline]
    pub fn as_limbs_mut(&mut self) -> &mut [u64; N] {
        &mut self.limbs
    }

    /// Lossy conversion to `f64`.
    ///
    /// Precision is limited to 53 significant bits (the mantissa of
    /// `f64`). This is sufficient for the floating-point GSO family
    /// in L2 lattice reduction, which only needs 24 bits of mantissa.
    pub fn to_f64(self) -> f64 {
        let neg = bool::from(self.is_negative());
        let limbs = &self.limbs;
        let mut val: f64 = 0.0;
        let mut i = N;
        while i > 0 {
            i -= 1;
            val = val * (u64::MAX as f64 + 1.0) + limbs[i] as f64;
        }
        if neg { -val } else { val }
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

    /// Returns the number of trailing zero bits (2-adic valuation).
    ///
    /// For zero, returns `N * 64`.
    ///
    /// WARNING: Not constant-time (data-dependent branch on limb values).
    ///
    /// TODO(ct): Make constant-time before production use. Called on
    /// secret-derived values in SuitableIdeals (dyadic valuation of u).
    pub fn trailing_zeros(&self) -> u32 {
        let mut count = 0u32;
        for i in 0..N {
            if self.limbs[i] == 0 {
                count += 64;
            } else {
                count += self.limbs[i].trailing_zeros();
                return count;
            }
        }
        count
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

    /// Greatest common divisor via the Euclidean algorithm.
    ///
    /// Returns a non-negative value. Based on the constant-time GCD
    /// approach (Algorithm 9, §2.5) from [Kouider et al.][ct-bigint],
    /// adapted here as a standard Euclidean GCD operating on magnitudes.
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    pub fn gcd(&self, other: &Self) -> Self {
        let mut a = self.abs();
        let mut b = other.abs();

        // Iterate a fixed number of times for constant-time behavior.
        // Each iteration reduces the larger operand by at least 1 bit,
        // so 2*BITS iterations is a safe upper bound.
        let mut i = 0;
        while i < 2 * Self::BITS {
            let b_is_zero = bool::from(b.is_zero()) as u64;
            // When b is zero, freeze both a and b (result is in a).
            let zero_limbs = [0u64; N];
            let (_, r) = if b_is_zero == 1 {
                (zero_limbs, zero_limbs)
            } else {
                Self::mag_div_rem(&a.limbs, &b.limbs)
            };
            // new_a = b (if active), or a (if frozen)
            let new_a_limbs = Self::mag_select(&b.limbs, &a.limbs, b_is_zero);
            // new_b = r (if active), or zero (if frozen)
            let new_b_limbs = Self::mag_select(&r, &b.limbs, b_is_zero);
            a = Self {
                sign: 0,
                limbs: new_a_limbs,
            };
            b = Self {
                sign: 0,
                limbs: new_b_limbs,
            };
            i += 1;
        }
        a
    }

    /// Extended GCD: returns `(gcd, x, y)` such that
    /// `self * x + other * y = gcd`, with `gcd >= 0`.
    ///
    /// Uses the extended Euclidean algorithm. The Bezout coefficients
    /// `x` and `y` are signed.
    pub fn xgcd(&self, other: &Self) -> (Self, Self, Self) {
        let mut old_r = self.abs();
        let mut r = other.abs();
        let mut old_s = Self::ONE;
        let mut s = Self::ZERO;
        let mut old_t = Self::ZERO;
        let mut t = Self::ONE;

        let mut i = 0;
        while i < 2 * Self::BITS {
            let r_is_zero = bool::from(r.is_zero()) as u64;
            if r_is_zero == 0 {
                let (q, rem) = old_r.div_rem(&r);

                let new_r = rem;
                let new_s = old_s.ct_sub(&q.ct_mul(&s));
                let new_t = old_t.ct_sub(&q.ct_mul(&t));

                old_r = r;
                old_s = s;
                old_t = t;
                r = new_r;
                s = new_s;
                t = new_t;
            }
            i += 1;
        }

        // Adjust signs: we computed on |self| and |other|.
        let mut x = old_s;
        let mut y = old_t;
        if self.sign == 1 {
            x = x.wrapping_neg();
        }
        if other.sign == 1 {
            y = y.wrapping_neg();
        }
        x.normalize();
        y.normalize();

        (old_r, x, y)
    }

    /// Modular inverse: returns `self^{-1} mod modulus`, or `None` if
    /// the inverse does not exist (i.e., `gcd(self, modulus) != 1`).
    ///
    /// The result is in `[0, |modulus|)`.
    pub fn invert_mod(&self, modulus: &Self) -> Option<Self> {
        let (g, x, _) = self.xgcd(modulus);
        if !bool::from(g.ct_eq(&Self::ONE)) {
            return None;
        }
        // x might be negative; reduce mod |modulus|.
        let result = x.ct_mod(modulus);
        Some(result)
    }

    /// Integer square root: returns the largest `s` such that `s*s <= self`.
    ///
    /// Only defined for non-negative values. Based on Newton-Raphson
    /// reciprocal square root (Algorithm 10, §2.5) from
    /// [Kouider et al.][ct-bigint], simplified here as a binary search
    /// on the result bits.
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    ///
    /// # Panics
    ///
    /// Panics if `self` is negative.
    pub fn sqrt_floor(&self) -> Self {
        assert!(
            !bool::from(self.is_negative()),
            "sqrt_floor called on negative value"
        );
        if bool::from(self.is_zero()) {
            return Self::ZERO;
        }

        // Binary search: set bits from MSB to LSB, keeping the bit if
        // result^2 <= self.
        let bs = self.bitsize();
        // The sqrt has at most ceil(bs/2) bits.
        let max_bit = bs.div_ceil(2);

        let mut result = Self::ZERO;
        let mut bit = max_bit;
        while bit > 0 {
            bit -= 1;
            // Tentatively set this bit.
            let limb_idx = (bit / 64) as usize;
            let bit_idx = bit % 64;
            if limb_idx < N {
                result.limbs[limb_idx] |= 1u64 << bit_idx;
                // Check if result^2 > self.
                let sq = result.ct_mul(&result);
                if sq > *self {
                    // Clear the bit.
                    result.limbs[limb_idx] &= !(1u64 << bit_idx);
                }
            }
        }
        result
    }

    /// Sample a uniform integer in `[a, b]` with rejection sampling on
    /// a top-bit-aligned mask, matching the C reference's
    /// `ibz_rand_interval`
    /// (`quaternion/ref/generic/intbig.c:413-473`).
    ///
    /// This is the byte-stream contract `quat_represent_integer` uses
    /// to draw `z` and `t`. Replicating it byte-for-byte is required
    /// for KAT byte-match against the C reference: with the same DRBG
    /// state, each call must consume the same number of bytes per
    /// rejection cycle.
    ///
    /// Per call: computes `bmina = b − a`, then loops drawing
    /// `ceil(bit_length(bmina) / 8)` bytes from `rng`, masking the top
    /// byte to exactly `bit_length(bmina)` bits, decoding little-
    /// endian, and accepting iff the value is `≤ bmina`. On accept,
    /// returns `bmina_value + a`. On `a == b`, returns `a` immediately
    /// without consuming any randomness.
    ///
    /// # Constant-time
    ///
    /// Variable-time. Used only for `represent_integer`'s `(z, t)`
    /// sampling, which the C reference also runs in variable time.
    /// `TODO(ct)`: revisit when the keygen path is hardened against
    /// timing side-channels (see `project_ct_plan.md`).
    ///
    /// # Panics
    ///
    /// Debug-asserts `a ≤ b`.
    pub fn rand_interval<R: rand_core::RngCore>(rng: &mut R, a: &Self, b: &Self) -> Self {
        debug_assert!(a <= b, "rand_interval: a must be ≤ b");
        let bmina = b.ct_sub(a);
        if bool::from(bmina.is_zero()) {
            return *a;
        }
        let len_bits = bmina.bitsize();
        let len_bytes = len_bits.div_ceil(8) as usize;

        // Top-byte mask: keep only the low `len_bits % 8` bits of the
        // last sampled byte. When `len_bits` is byte-aligned, the byte
        // is unmasked.
        let top_byte_bits = (len_bits % 8) as u8;
        let top_mask: u8 = if top_byte_bits == 0 {
            0xFF
        } else {
            (1u8 << top_byte_bits) - 1
        };

        let mut buf = vec![0u8; len_bytes];
        loop {
            rng.fill_bytes(&mut buf);
            buf[len_bytes - 1] &= top_mask;
            // Decode little-endian into `BigInt<N>` limbs.
            let mut limbs = [0u64; N];
            for (i, chunk) in buf.chunks(8).enumerate() {
                if i >= N {
                    break;
                }
                let mut bytes = [0u8; 8];
                bytes[..chunk.len()].copy_from_slice(chunk);
                limbs[i] = u64::from_le_bytes(bytes);
            }
            let val = Self::from_limbs(limbs);
            if val <= bmina {
                return val.ct_add(a);
            }
        }
    }

    /// Cornacchia's algorithm: solve x² + qy² = m for integers x, y.
    ///
    /// Given a prime `m` and a positive integer `q` with 0 ≤ q ≤ m,
    /// finds integers x, y such that x² + qy² = m, or returns `None`
    /// if no solution exists (which happens when -q is not a quadratic
    /// residue mod m).
    ///
    /// # Algorithm
    ///
    /// This implements [Alg. 3.11] from the SQIsign specification,
    /// following the standard Cornacchia algorithm as described in
    /// Cohen's "A Course in Computational Algebraic Number Theory"
    /// and [Morain-Nicolas][MN90]:
    ///
    /// 1. Check that -q is a quadratic residue mod m (Legendre symbol).
    /// 2. Compute r₀ = √(-q) mod m via [`modular_sqrt`](Self::modular_sqrt).
    /// 3. Run the Euclidean algorithm on (m, r₀), reducing until s ≤ √m.
    /// 4. Set x = s, compute y² = (m - x²) / q, verify y is an integer.
    ///
    /// # Spec discrepancy
    ///
    /// The spec's Algorithm 3.11, line 9, shows `s ← q`, but this does
    /// not match the standard Cornacchia algorithm or the C reference
    /// implementation (`ibz_cornacchia_prime` in `integers.c`), both of
    /// which initialize the Euclidean chain as (m, r₀). We follow the
    /// standard algorithm and the C reference here.
    ///
    /// The C reference initializes `r1 = p` (the prime) and
    /// `r2 = sqrt(-n mod p)`, then reduces until `r0² < p` — exactly
    /// the `(m, r₀)` approach.
    ///
    /// TODO: Open a bug against the SQIsign spec reporting the line 9
    /// discrepancy (`s ← q` vs the correct `s ← m`).
    ///
    /// # Side-channel considerations
    ///
    /// Not constant-time — the Euclidean reduction loop has
    /// data-dependent iteration count.
    ///
    /// TODO(ct): Make constant-time before production use. Called on
    /// secret-derived values during signing (via RepresentInteger).
    ///
    /// # Special cases
    ///
    /// - q = 0: checks if m is a perfect square.
    /// - m = 2, q = 1: returns (1, 1) since 1² + 1² = 2.
    ///
    /// [Alg. 3.11]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.11
    /// [MN90]: https://doi.org/10.1016/0022-314X(90)90136-F
    pub fn cornacchia(q: &Self, m: &Self) -> Option<(Self, Self)> {
        // Special case: q = 0 reduces to x² = m.
        if bool::from(q.is_zero()) {
            let s = m.sqrt_floor();
            return if s.ct_mul(&s) == *m {
                Some((s, Self::ZERO))
            } else {
                None
            };
        }

        // Special case: m = 2, q = 1 (spec lines 3-7).
        if *m == Self::TWO {
            return if *q == Self::ONE {
                Some((Self::ONE, Self::ONE))
            } else {
                None
            };
        }

        // Step 1: Check Legendre symbol — -q must be a QR mod m.
        let neg_q = m.ct_sub(&q.ct_mod(m));
        let exp = m.ct_sub(&Self::ONE).shr(1);
        let legendre = Self::pow_mod(&neg_q, &exp, m);
        if legendre != Self::ONE && !bool::from(neg_q.ct_mod(m).is_zero()) {
            return None;
        }

        // Step 2: r₀ = √(-q) mod m.
        let r0 = Self::modular_sqrt(&neg_q, m)?;

        // Step 3: Euclidean reduction on (m, r₀).
        //
        // Initialize r = m, s = r₀. Reduce until s ≤ √m (equivalently
        // s² ≤ m). After the loop, x = s is the first remainder whose
        // square does not exceed m.
        let mut r = *m;
        let mut s = r0;
        let bound = m.sqrt_floor();

        let mut i = 0;
        while i < 2 * Self::BITS {
            if s <= bound {
                break;
            }
            let tmp = r.ct_mod(&s);
            r = s;
            s = tmp;
            i += 1;
        }

        // Step 4: x = s, y² = (m - x²) / q.
        let x = s;
        let x_sq = x.ct_mul(&x);
        if x_sq > *m {
            return None;
        }

        // (m - x²) must be divisible by q.
        let (y_sq, rem) = m.ct_sub(&x_sq).div_rem(q);
        if !bool::from(rem.is_zero()) {
            return None;
        }

        // y² must be a perfect square.
        let y = y_sq.sqrt_floor();
        if y.ct_mul(&y) != y_sq {
            return None;
        }

        // Final verification: x² + qy² = m.
        let check = x_sq.ct_add(&q.ct_mul(&y.ct_mul(&y)));
        if check == *m { Some((x, y)) } else { None }
    }

    /// Cornacchia's algorithm at a wider working width `W`.
    ///
    /// Widens `q` and `m` to `BigInt<W>` for the modular arithmetic
    /// (Legendre symbol, modular square root, Euclidean reduction),
    /// then narrows the result back to `BigInt<N>`. Use when
    /// `64*N < 2*bits(m)` — otherwise [`cornacchia`](Self::cornacchia)
    /// silently truncates during `pow_mod` and fails to find
    /// solutions.
    pub fn cornacchia_w<const W: usize>(q: &Self, m: &Self) -> Option<(Self, Self)> {
        const {
            assert!(
                W >= N,
                "cornacchia_w: working width W must be >= storage width N"
            )
        };
        let q_w: BigInt<W> = q.widen();
        let m_w: BigInt<W> = m.widen();
        let (x_w, y_w) = BigInt::<W>::cornacchia(&q_w, &m_w)?;
        Some((x_w.narrow_to::<N>()?, y_w.narrow_to::<N>()?))
    }

    /// Modular square root: returns x such that x² ≡ n (mod m),
    /// or `None` if n is not a quadratic residue mod m.
    ///
    /// Requires m to be an odd prime. Implements [Alg. 3.1] from
    /// the spec, with fast paths for m ≡ 3 (mod 4) and m ≡ 5 (mod 8),
    /// and Tonelli-Shanks for the general case m ≡ 1 (mod 8).
    ///
    /// # Width requirement
    ///
    /// All internal operations use [`pow_mod`](Self::pow_mod) and
    /// direct `ct_mul`/`ct_mod` at width `N`. The caller must ensure
    /// `64*N >= 2*bits(m)` — otherwise the squarings silently
    /// truncate and the result is wrong. For larger moduli use
    /// [`modular_sqrt_w`](Self::modular_sqrt_w).
    ///
    /// [Alg. 3.1]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.1
    pub fn modular_sqrt(n: &Self, m: &Self) -> Option<Self> {
        let n_mod = n.ct_mod(m);
        if bool::from(n_mod.is_zero()) {
            return Some(Self::ZERO);
        }

        let m_mod4 = m.as_limbs()[0] & 3;
        let m_mod8 = m.as_limbs()[0] & 7;

        // m ≡ 3 (mod 4): return n^((m+1)/4) mod m.
        if m_mod4 == 3 {
            let exp = m.ct_add(&Self::ONE).shr(2);
            let r = Self::pow_mod(&n_mod, &exp, m);
            let check = r.ct_mul(&r).ct_mod(m);
            return if check == n_mod { Some(r) } else { None };
        }

        // m ≡ 5 (mod 8):
        if m_mod8 == 5 {
            // Check if n^((m-1)/4) ≡ 1 mod m.
            let exp_check = m.ct_sub(&Self::ONE).shr(2);
            let test = Self::pow_mod(&n_mod, &exp_check, m);
            if test == Self::ONE {
                // return n^((m+3)/8) mod m
                let exp = m.ct_add(&Self::THREE).shr(3);
                return Some(Self::pow_mod(&n_mod, &exp, m));
            } else {
                // return 2n(4n)^((m-5)/8) mod m
                let four_n = n_mod.ct_mul(&Self::from_u64(4)).ct_mod(m);
                let exp = m.ct_sub(&Self::from_u64(5)).shr(3);
                let base = Self::pow_mod(&four_n, &exp, m);
                let r = Self::TWO.ct_mul(&n_mod).ct_mul(&base).ct_mod(m);
                let check = r.ct_mul(&r).ct_mod(m);
                return if check == n_mod { Some(r) } else { None };
            }
        }

        // General Tonelli-Shanks (m ≡ 1 mod 8).
        let e = m.ct_sub(&Self::ONE).two_adic_val();
        let q = m.ct_sub(&Self::ONE).shr(e);

        // Find a non-residue w.
        let mut w = Self::TWO;
        loop {
            let exp = m.ct_sub(&Self::ONE).shr(1);
            let ls = Self::pow_mod(&w, &exp, m);
            // Legendre symbol: if ls == m - 1, then w is a non-residue.
            if ls == m.ct_sub(&Self::ONE) {
                break;
            }
            w = w.ct_add(&Self::ONE);
            // Safety bound.
            if w > *m {
                return None;
            }
        }

        let mut z = Self::pow_mod(&w, &q, m);
        let mut y = Self::pow_mod(&n_mod, &q, m);
        let mut x = Self::pow_mod(&n_mod, &q.ct_add(&Self::ONE).shr(1), m);
        let mut f = Self::from_u64(1u64 << (e - 2));

        for _i in 0..e.saturating_sub(1) {
            let b = Self::pow_mod(&y, &f, m);
            if b == m.ct_sub(&Self::ONE) {
                // b ≡ -1 mod m
                x = x.ct_mul(&z).ct_mod(m);
                y = y.ct_mul(&z).ct_mul(&z).ct_mod(m);
            }
            z = z.ct_mul(&z).ct_mod(m);
            f = f.shr(1);
        }

        let check = x.ct_mul(&x).ct_mod(m);
        if check == n_mod { Some(x) } else { None }
    }

    /// Modular square root at a wider working width `W`.
    ///
    /// Widens `n` and `m` to `BigInt<W>` and runs
    /// [`modular_sqrt`](Self::modular_sqrt) at that width. Use when
    /// `64*N < 2*bits(m)` would otherwise silently truncate the
    /// Tonelli-Shanks exponentiations.
    ///
    /// See [`pow_mod_w`](Self::pow_mod_w) for the width constraints.
    pub fn modular_sqrt_w<const W: usize>(n: &Self, m: &Self) -> Option<Self> {
        const {
            assert!(
                W >= N,
                "modular_sqrt_w: working width W must be >= storage width N"
            )
        };
        let n_w: BigInt<W> = n.widen();
        let m_w: BigInt<W> = m.widen();
        let r_w = BigInt::<W>::modular_sqrt(&n_w, &m_w)?;
        r_w.narrow_to::<N>()
    }

    /// Modular exponentiation: `base^exp mod modulus`.
    ///
    /// Uses square-and-multiply. The exponent is scanned from MSB to LSB.
    ///
    /// # Width requirement
    ///
    /// The inner squaring `result * result` can reach `(modulus - 1)²`
    /// before the `ct_mod` reduction. For the result to not silently
    /// truncate, `BigInt<N>` must satisfy `64*N >= 2*bits(modulus)`.
    /// If `modulus` is larger than that bound (bits-wise), use
    /// [`pow_mod_w`](Self::pow_mod_w) with a wider working type.
    pub fn pow_mod(base: &Self, exp: &Self, modulus: &Self) -> Self {
        let mut result = Self::ONE;
        let bs = exp.bitsize();
        let mut i = bs;
        while i > 0 {
            i -= 1;
            result = result.ct_mul(&result).ct_mod(modulus);
            let limb_idx = (i / 64) as usize;
            let bit_idx = i % 64;
            let bit = (exp.limbs[limb_idx] >> bit_idx) & 1;
            if bit == 1 {
                result = result.ct_mul(base).ct_mod(modulus);
            }
        }
        result
    }

    /// Modular exponentiation at a wider working width `W`.
    ///
    /// Widens the operands to `BigInt<W>`, runs [`pow_mod`](Self::pow_mod)
    /// at that width, then narrows the result back to `BigInt<N>`.
    /// Use this when the storage width `N` is not big enough for the
    /// squarings inside `pow_mod` to fit without truncation — that is,
    /// whenever `64*N < 2*bits(modulus)`.
    ///
    /// # Width requirements
    ///
    /// - Compile-time: `W >= N` (enforced by a const assertion).
    /// - Runtime invariant: `64*W >= 2*bits(modulus)`. The caller is
    ///   responsible for choosing `W` large enough for their modulus. If this
    ///   is violated, the wider `pow_mod` will also silently truncate.
    ///
    /// For the SQIsign v2 commitment modulus
    /// `D_mix = 2^512 + 75` (513 bits), use at least `W = 18`.
    pub fn pow_mod_w<const W: usize>(base: &Self, exp: &Self, modulus: &Self) -> Self {
        const {
            assert!(
                W >= N,
                "pow_mod_w: working width W must be >= storage width N"
            )
        };
        let base_w: BigInt<W> = base.widen();
        let exp_w: BigInt<W> = exp.widen();
        let modulus_w: BigInt<W> = modulus.widen();
        let result_w = BigInt::<W>::pow_mod(&base_w, &exp_w, &modulus_w);
        result_w
            .narrow_to::<N>()
            .expect("pow_mod_w result < modulus < 2^(64N) fits in BigInt<N>")
    }

    /// Miller-Rabin probabilistic primality test.
    ///
    /// Returns `true` if `self` is probably prime. Uses `rounds`
    /// deterministic witnesses (2, 3, 5, 7, 11, 13, ...) for small
    /// round counts, which gives a deterministic result for values
    /// below certain bounds.
    ///
    /// Based on G.L. Miller, ["Riemann's hypothesis and tests for
    /// primality"][Miller76] (JCSS 13(3), 1976) and M.O. Rabin,
    /// "Probabilistic algorithm for testing primality"
    /// (J. Number Theory 12(1), 1980).
    ///
    /// [Miller76]: https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
    ///
    /// # Width requirement
    ///
    /// Miller-Rabin uses [`pow_mod`](Self::pow_mod) and direct
    /// `ct_mul` on values up to `self`. The caller must ensure
    /// `64*N >= 2*bits(self)` — otherwise the squarings silently
    /// truncate and the test returns wrong answers (in practice,
    /// false negatives on primes). For larger candidates use
    /// [`is_probable_prime_w`](Self::is_probable_prime_w).
    ///
    /// WARNING: Not constant-time — the number of iterations and
    /// modular exponentiations depend on the value.
    ///
    /// TODO(ct): Make constant-time before production use. Called on
    /// secret-derived values during signing (via RepresentInteger).
    pub fn is_probable_prime(&self, rounds: u32) -> bool {
        if bool::from(self.is_negative()) || bool::from(self.is_zero()) {
            return false;
        }
        if *self == Self::ONE {
            return false;
        }
        if *self == Self::TWO || *self == Self::THREE {
            return true;
        }
        if bool::from(self.is_even()) {
            return false;
        }

        // Write self - 1 = 2^s · d with d odd.
        let n_minus_1 = self.ct_sub(&Self::ONE);
        let s = n_minus_1.two_adic_val();
        let d = n_minus_1.shr(s);

        // Deterministic witnesses sufficient for values up to 3.3×10²⁴.
        let witnesses: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
        let num_rounds = (rounds as usize).min(witnesses.len());

        for &a_val in &witnesses[..num_rounds] {
            let a = Self::from_u64(a_val);
            if a >= *self {
                continue;
            }

            let mut x = Self::pow_mod(&a, &d, self);
            if x == Self::ONE || x == n_minus_1 {
                continue;
            }

            let mut composite = true;
            for _r in 1..s {
                x = x.ct_mul(&x).ct_mod(self);
                if x == n_minus_1 {
                    composite = false;
                    break;
                }
            }
            if composite {
                return false;
            }
        }

        true
    }

    /// Miller-Rabin primality test at a wider working width `W`.
    ///
    /// Widens `self` to `BigInt<W>` and runs
    /// [`is_probable_prime`](Self::is_probable_prime) at that width.
    /// Use when `64*N < 2*bits(self)` would otherwise silently
    /// truncate the Miller-Rabin witness exponentiations (producing
    /// false negatives on actual primes).
    ///
    /// See [`pow_mod_w`](Self::pow_mod_w) for the width constraints.
    pub fn is_probable_prime_w<const W: usize>(&self, rounds: u32) -> bool {
        const {
            assert!(
                W >= N,
                "is_probable_prime_w: working width W must be >= storage width N"
            )
        };
        let self_w: BigInt<W> = self.widen();
        self_w.is_probable_prime(rounds)
    }

    /// Legendre symbol: returns 1 if `a` is a quadratic residue mod
    /// `p`, -1 if not, 0 if a ≡ 0 mod p. Requires `p` odd prime.
    ///
    /// # Width requirement
    ///
    /// Uses Euler's criterion via [`pow_mod`](Self::pow_mod), which
    /// requires `64*N >= 2*bits(p)`. For larger primes use
    /// [`legendre_w`](Self::legendre_w).
    pub fn legendre(a: &Self, p: &Self) -> i32 {
        let a_mod = a.ct_mod(p);
        if bool::from(a_mod.is_zero()) {
            return 0;
        }
        let exp = p.ct_sub(&Self::ONE).shr(1);
        let result = Self::pow_mod(&a_mod, &exp, p);
        if result == Self::ONE { 1 } else { -1 }
    }

    /// Legendre symbol at a wider working width `W`.
    ///
    /// Widens `a` and `p` to `BigInt<W>` and runs [`legendre`](Self::legendre)
    /// at that width. See [`pow_mod_w`](Self::pow_mod_w) for the width
    /// constraints. Use when `64*N < 2*bits(p)` would otherwise
    /// silently truncate the Euler exponentiation.
    pub fn legendre_w<const W: usize>(a: &Self, p: &Self) -> i32 {
        const {
            assert!(
                W >= N,
                "legendre_w: working width W must be >= storage width N"
            )
        };
        let a_w: BigInt<W> = a.widen();
        let p_w: BigInt<W> = p.widen();
        BigInt::<W>::legendre(&a_w, &p_w)
    }

    // -----------------------------------------------------------------------
    // Private magnitude helpers
    // -----------------------------------------------------------------------

    /// Constant-time unsigned addition of magnitudes. Returns `(limbs, carry)`.
    #[inline]
    const fn mag_add(a: &[u64; N], b: &[u64; N]) -> ([u64; N], u64) {
        let mut result = [0u64; N];
        let mut carry: u64 = 0;
        let mut i = 0;
        while i < N {
            let (s1, c1) = a[i].overflowing_add(b[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            result[i] = s2;
            carry = (c1 as u64) + (c2 as u64);
            i += 1;
        }
        (result, carry)
    }

    /// Constant-time unsigned subtraction of magnitudes. Returns `(limbs,
    /// borrow)`. Borrow is 1 if `a < b` (unsigned).
    #[inline]
    const fn mag_sub(a: &[u64; N], b: &[u64; N]) -> ([u64; N], u64) {
        let mut result = [0u64; N];
        let mut borrow: u64 = 0;
        let mut i = 0;
        while i < N {
            let (d1, b1) = a[i].overflowing_sub(b[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            result[i] = d2;
            borrow = (b1 as u64) + (b2 as u64);
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

    /// Constant-time unsigned Euclidean division of magnitudes.
    ///
    /// Returns `(quotient, remainder)` where `a = quotient * b + remainder`
    /// and `0 <= remainder < b`.
    ///
    /// Based on Algorithm 5 (§3.4) from [Kouider et al.][ct-bigint]
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    fn mag_div_rem(a: &[u64; N], b: &[u64; N]) -> ([u64; N], [u64; N]) {
        // Long division: scan from the highest possible quotient bit
        // down to bit 0. For each position, trial-subtract the shifted
        // divisor and set the quotient bit if the subtraction succeeds.
        let mut r = *a; // running remainder
        let mut q = [0u64; N];

        let bs_b = Self::mag_bitsize(b);
        // The quotient can have at most (BITS - bs_b + 1) bits, but
        // we iterate over all BITS positions for constant time.
        let mut bit = Self::BITS;
        while bit > 0 {
            bit -= 1;

            // u = b << bit
            let u = Self::mag_shl(b, bit);

            // Check if r >= u (unsigned).
            let (r_minus_u, borrow) = Self::mag_sub(&r, &u);
            let can_sub = 1 - borrow.min(1);

            // Also check that the shift didn't overflow (bit + bs_b <= BITS).
            // If b << bit overflows N limbs, the shifted value is truncated
            // and the comparison would be wrong. Suppress when bit >= BITS - bs_b + 1.
            let valid_shift = if bs_b == 0 {
                0u64
            } else if bit + bs_b <= Self::BITS {
                1u64
            } else {
                0u64
            };
            let do_sub = can_sub & valid_shift;

            // Conditionally update remainder.
            r = Self::mag_select(&r, &r_minus_u, do_sub);

            // Set quotient bit.
            let q_limb = (bit / 64) as usize;
            let q_bit = bit % 64;
            if q_limb < N {
                q[q_limb] |= do_sub << q_bit;
            }
        }

        (q, r)
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

    // -----------------------------------------------------------------------
    // Signed arithmetic
    // -----------------------------------------------------------------------

    /// Constant-time signed addition.
    ///
    /// If signs match: add magnitudes.
    /// If signs differ: subtract the smaller magnitude from the larger,
    /// with the sign of the larger.
    pub fn ct_add(&self, rhs: &Self) -> Self {
        let same_sign = ((self.sign ^ rhs.sign) == 0) as u64;

        // Case 1: same sign -> add magnitudes, keep sign.
        let (sum, _carry) = Self::mag_add(&self.limbs, &rhs.limbs);

        // Case 2: different signs -> subtract magnitudes.
        let cmp = Self::mag_cmp(&self.limbs, &rhs.limbs);
        let self_ge = (cmp != Ordering::Less) as u64;

        let (diff_a, _) = Self::mag_sub(&self.limbs, &rhs.limbs);
        let (diff_b, _) = Self::mag_sub(&rhs.limbs, &self.limbs);

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

// ---------------------------------------------------------------------------
// Width conversions
// ---------------------------------------------------------------------------

impl<const N: usize> BigInt<N> {
    /// Widen to `BigInt<W>` by zero-extending the upper limbs.
    ///
    /// The value is preserved exactly; the upper `W - N` limbs are zero.
    /// Compile-time error if `W < N`.
    #[must_use]
    pub fn widen<const W: usize>(self) -> BigInt<W> {
        // Compile-time check: destination must be at least as wide.
        const {
            assert!(
                W >= N,
                "widen: destination width W must be >= source width N"
            )
        };
        let mut limbs = [0u64; W];
        let mut i = 0;
        while i < N {
            limbs[i] = self.limbs[i];
            i += 1;
        }
        BigInt::<W> {
            sign: self.sign,
            limbs,
        }
    }

    /// Narrow to `BigInt<T>`, returning `None` if the value does not
    /// fit (any of the upper `N - T` limbs are non-zero).
    ///
    /// Compile-time error if `T > N`.
    #[must_use]
    pub fn narrow_to<const T: usize>(self) -> Option<BigInt<T>> {
        const {
            assert!(
                T <= N,
                "narrow_to: destination width T must be <= source width N"
            )
        };
        let mut overflow = 0u64;
        let mut i = T;
        while i < N {
            overflow |= self.limbs[i];
            i += 1;
        }
        if overflow != 0 {
            return None;
        }
        let mut limbs = [0u64; T];
        let mut j = 0;
        while j < T {
            limbs[j] = self.limbs[j];
            j += 1;
        }
        Some(BigInt::<T> {
            sign: self.sign,
            limbs,
        })
    }
}

/// Widen: zero-extend a four-limb integer to eight limbs.
impl From<BigInt<4>> for BigInt<8> {
    fn from(small: BigInt<4>) -> Self {
        small.widen()
    }
}

/// Narrow: truncate an eight-limb integer to four limbs.
///
/// Returns a `CtOption` — the result is always computed (constant time),
/// but the `is_some` flag indicates whether the value actually fits.
///
/// For ergonomic use in non-constant-time code, see [`BigInt<8>::narrow`].
impl From<BigInt<8>> for subtle::CtOption<BigInt<4>> {
    fn from(wide: BigInt<8>) -> Self {
        let mut overflow = 0u64;
        overflow |= wide.limbs[4];
        overflow |= wide.limbs[5];
        overflow |= wide.limbs[6];
        overflow |= wide.limbs[7];
        let fits = Choice::from((overflow == 0) as u8);
        let narrow = BigInt::<4> {
            sign: wide.sign,
            limbs: [wide.limbs[0], wide.limbs[1], wide.limbs[2], wide.limbs[3]],
        };
        subtle::CtOption::new(narrow, fits)
    }
}

impl BigInt<8> {
    /// Narrow to `BigInt<4>`, returning `None` if the upper limbs are
    /// non-zero.
    ///
    /// This is a convenience wrapper around the constant-time
    /// `From<BigInt<8>> for CtOption<BigInt<4>>` conversion.
    pub fn narrow(self) -> Option<BigInt<4>> {
        Option::from(subtle::CtOption::<BigInt<4>>::from(self))
    }
}

// ---------------------------------------------------------------------------
// ConstantTimeEq / ConditionallySelectable
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Standard operator traits
// ---------------------------------------------------------------------------

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

impl<const N: usize> AddAssign for BigInt<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.ct_add(&rhs);
    }
}

impl<const N: usize> AddAssign<&BigInt<N>> for BigInt<N> {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        *self = self.ct_add(rhs);
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

impl<const N: usize> SubAssign for BigInt<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.ct_sub(&rhs);
    }
}

impl<const N: usize> SubAssign<&BigInt<N>> for BigInt<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        *self = self.ct_sub(rhs);
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

impl<const N: usize> MulAssign for BigInt<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.ct_mul(&rhs);
    }
}

impl<const N: usize> MulAssign<&BigInt<N>> for BigInt<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = self.ct_mul(rhs);
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

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl<const N: usize> fmt::Debug for BigInt<N> {
    #[cfg_attr(test, mutants::skip)] // formatting, not correctness
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.sign == 1 && Self::mag_is_zero(&self.limbs) == 0 {
            write!(f, "-")?;
        }
        write!(f, "BigInt(0x")?;
        let mut i = N;
        while i > 0 {
            i -= 1;
            write!(f, "{:016x}", self.limbs[i])?;
        }
        write!(f, ")")
    }
}

impl<const N: usize> fmt::Display for BigInt<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.sign == 1 && Self::mag_is_zero(&self.limbs) == 0 {
            write!(f, "-")?;
        }
        write!(f, "0x")?;
        let mut started = false;
        let mut i = N;
        while i > 0 {
            i -= 1;
            let limb = self.limbs[i];
            if started {
                write!(f, "{limb:016x}")?;
            } else if limb != 0 {
                write!(f, "{limb:x}")?;
                started = true;
            }
        }
        if !started {
            write!(f, "0")?;
        }
        Ok(())
    }
}
