//! Double-plus-exponent (DPE) floating-point type for LLL.
//!
//! A minimal software-float with `f64` precision (~53-bit mantissa)
//! and extended exponent range (`i64`). This matches the `dpe_t`
//! type used by the SQIsign C reference's L² implementation.
//!
//! Without extended exponents, converting large `BigInt` Gram entries
//! (> 2^1023) to `f64` yields infinity, and entries > ~2^53 lose
//! precision that causes the GSO in LLL to diverge or loop. The DPE
//! representation keeps the mantissa normalized in `[0.5, 1.0)` (or
//! zero) so all 53 bits of precision are preserved regardless of
//! magnitude.
//!
//! # Divergences
//!
//! The SQIsign spec (Algorithm 3.3) describes L² in terms of exact
//! rational arithmetic. The C reference and this implementation both
//! use approximate floating-point GSO with DPE for the Lovász
//! condition and size-reduction rounding. The integer basis and Gram
//! updates remain exact; only the GSO coefficients are approximate.

use core::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Sub, SubAssign},
};

use crate::quaternions::bigint::BigInt;

#[cfg(test)]
mod tests;

/// A double-precision float with an explicit base-2 exponent.
///
/// Represents the value `m * 2^e`. The mantissa is kept in
/// `[0.5, 1.0)` (or zero) after every operation.
#[derive(Clone, Copy, Debug)]
pub struct DoublePlusExponent {
    /// Mantissa in `[0.5, 1.0)` or zero.
    pub m: f64,
    /// Base-2 exponent.
    pub e: i64,
}

/// Decompose `x` into `(frac, exp)` such that `x = frac * 2^exp`
/// and `frac ∈ [0.5, 1.0)` (or zero). Pure bit manipulation, no
/// libm dependency.
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000_0000_0000;
    let biased_exp = ((bits >> 52) & 0x7FF) as i32;
    // Normal number: exponent = biased - 1022, mantissa forced to
    // biased exponent 0x3FE (= 1022) which represents [0.5, 1.0).
    let exp = biased_exp - 1022;
    let frac_bits = sign | 0x3FE0_0000_0000_0000 | (bits & 0x000F_FFFF_FFFF_FFFF);
    (f64::from_bits(frac_bits), exp)
}

/// Computes `x * 2^exp`. Pure bit manipulation.
fn ldexp(x: f64, exp: i32) -> f64 {
    // Multiply by 2^exp via a float whose exponent IS exp.
    // Split into two steps to avoid overflow for large exp.
    let half = exp / 2;
    let other = exp - half;
    let a = f64::from_bits(((1023i64 + half as i64) as u64) << 52);
    let b = f64::from_bits(((1023i64 + other as i64) as u64) << 52);
    x * a * b
}

impl DoublePlusExponent {
    /// The zero `DoublePlusExponent` value.
    pub const ZERO: Self = Self { m: 0.0, e: 0 };

    /// Converts from an `f64` (matches `dpe_set_d`).
    pub fn from_f64(v: f64) -> Self {
        if v == 0.0 {
            return Self::ZERO;
        }
        let (frac, exp) = frexp(v);
        Self {
            m: frac,
            e: exp as i64,
        }
    }

    /// Converts from a [`BigInt<N>`], byte-for-byte matching mini-GMP's
    /// `mini_mpz_get_d_2exp`
    /// (`the-sqisign/src/mini-gmp/mini-gmp-extra.c:41-65`), which is
    /// the function the C ref's `dpe_set_z` calls.
    ///
    /// Algorithm (from C ref):
    /// 1. If `v == 0`: return `(0.0, 0)`.
    /// 2. Set `e = bitsize(|v|)` (= `mpz_sizeinbase(op, 2)`).
    /// 3. If `e > DBL_MAX_EXP` (= 1024): shift `|v|` right by `(e − 1024)` so
    ///    the truncated value fits in `f64`'s normal range when converted.
    /// 4. Convert the (possibly shifted) magnitude to `f64` using mini-GMP's
    ///    `mpz_get_d` (truncation at bit 53, round-toward-zero — NOT
    ///    round-to-nearest).
    /// 5. Apply `frexp` to canonicalize the mantissa to `[1/2, 1)`; discard
    ///    `frexp`'s exponent (the bit length is what we return).
    /// 6. Negate the mantissa if `v < 0`.
    ///
    /// # Bit-exactness
    ///
    /// Byte-for-byte equal to `mini_mpz_get_d_2exp` on every input.
    /// Property-tested against the C reference by
    /// `tests/ffi_cref_lll.rs::to_dpe_*` under the `ffi-cref-lll`
    /// feature.
    pub fn from_bigint<const N: usize>(v: &BigInt<N>) -> Self {
        // The mantissa is the top 53 bits of |v|, truncated toward zero,
        // and the exponent is bitsize(|v|). `to_f64_trunc` computes the
        // same f64 from the whole magnitude, but everything below the top
        // 53 bits only scales it by a power of two, which `frexp`
        // discards. So read just the top one or two limbs around the MSB
        // and skip the full-width `abs`/shift/limb-loop. Bit-for-bit equal
        // to `frexp(to_f64_trunc(|v| >> max(0, bits - 1024))).0` (the
        // `mini_mpz_get_d_2exp` path); see the differential test in
        // `tests`.
        let limbs = v.as_limbs();
        let mut top_idx = N;
        while top_idx > 0 && limbs[top_idx - 1] == 0 {
            top_idx -= 1;
        }
        if top_idx == 0 {
            return Self::ZERO;
        }

        let top = limbs[top_idx - 1];
        let clz = top.leading_zeros();
        let bits = (top_idx as i64) * 64 - clz as i64;

        // m = clz + 53 - 64: bits the top limb has beyond/below 53. m <= 0
        // means the top limb already covers >= 53 bits (mask its low
        // -m bits); m > 0 means we need the top m bits of the next limb.
        let m: i32 = clz as i32 - 11;
        let raw: f64 = if m <= 0 {
            let masked = if m < 0 {
                top & (u64::MAX << ((-m) as u32))
            } else {
                top
            };
            masked as f64
        } else {
            // Top limb has 53 - m significant bits; take m more from the
            // next limb's high end. b = 2^64 (the limb's place value);
            // frexp normalizes away the place, so only the ratio matters.
            let b: f64 = (1u128 << 64) as f64;
            let mut x = (top as f64) * b;
            if top_idx >= 2 {
                let l2 = limbs[top_idx - 2] & (u64::MAX << ((64 - m) as u32));
                x += l2 as f64;
            }
            x
        };

        let mantissa = frexp(raw).0;
        let signed = if bool::from(v.is_negative()) {
            -mantissa
        } else {
            mantissa
        };
        Self { m: signed, e: bits }
    }

    /// Converts to a `BigInt<N>`, rounding to the nearest integer.
    ///
    /// This is the equivalent of the C reference's `ibz_set_dpe`:
    /// it converts the DPE mantissa to an integer and shifts left
    /// by the exponent. Needed because size-reduction μ values
    /// can exceed `i64` range (e.g., `μ[3][0] ≈ 2^260` before
    /// the first reduction pass); `round() as i64` would wrap.
    pub fn to_bigint<const N: usize>(self) -> BigInt<N> {
        if self.m == 0.0 {
            return BigInt::ZERO;
        }
        // Round mantissa to nearest integer in f64 (precision is
        // fine since mantissa is in [0.5, 1.0)).
        // Value = m * 2^e. Represent as integer:
        //   if e >= 0: round(m * 2^53) << (e - 53)
        //   if e < 53: round(m * 2^e)  (small, fits directly)
        let neg = self.m < 0.0;
        let abs_m = self.m.abs();

        if self.e <= 53 {
            // |value| fits in a single f64 → round directly.
            let value = ldexp(abs_m, self.e as i32);
            let rounded = value.round() as u64;
            let mut r = BigInt::<N>::from_u64(rounded);
            if neg {
                r = r.wrapping_neg();
            }
            return r;
        }

        // Shift mantissa into integer range: mantissa * 2^53
        // gives a 53-bit integer. Then shift left by (e - 53).
        let int_mantissa = (abs_m * ((1u64 << 53) as f64)).round() as u64;
        let mut r = BigInt::<N>::from_u64(int_mantissa);
        let shift = self.e - 53;
        if shift > 0 {
            r = r << shift as u32;
        } else if shift < 0 {
            r = r >> (-shift) as u32;
        }
        if neg {
            r = r.wrapping_neg();
        }
        r
    }

    /// Renormalizes so that `m ∈ [0.5, 1)`, adjusting `e` accordingly.
    fn normalize(mut self) -> Self {
        if self.m == 0.0 {
            self.e = 0;
            return self;
        }
        let (frac, exp) = frexp(self.m);
        self.m = frac;
        self.e += exp as i64;
        self
    }
}

impl<const N: usize> From<&BigInt<N>> for DoublePlusExponent {
    /// Converts via [`DoublePlusExponent::from_bigint`] — bit-exact
    /// with mini-GMP's `mini_mpz_get_d_2exp`. Total conversion (no
    /// `TryFrom` needed) since every [`BigInt`] has a finite
    /// `DoublePlusExponent` representation.
    #[inline]
    fn from(v: &BigInt<N>) -> Self {
        Self::from_bigint(v)
    }
}

impl<const N: usize> From<BigInt<N>> for DoublePlusExponent {
    /// Converts by reference (the value is small and `Copy`, but
    /// `from_bigint`'s only access pattern is read-only).
    #[inline]
    fn from(v: BigInt<N>) -> Self {
        Self::from_bigint(&v)
    }
}

impl Add for DoublePlusExponent {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        if self.m == 0.0 {
            return rhs;
        }
        if rhs.m == 0.0 {
            return self;
        }
        let (hi, lo) = if self.e >= rhs.e {
            (self, rhs)
        } else {
            (rhs, self)
        };
        let shift = hi.e - lo.e;
        if shift > 53 {
            return hi;
        }
        let aligned_lo = ldexp(lo.m, -(shift as i32));
        Self {
            m: hi.m + aligned_lo,
            e: hi.e,
        }
        .normalize()
    }
}

impl Sub for DoublePlusExponent {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + Self {
            m: -rhs.m,
            e: rhs.e,
        }
    }
}

impl SubAssign for DoublePlusExponent {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for DoublePlusExponent {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            m: self.m * rhs.m,
            e: self.e + rhs.e,
        }
        .normalize()
    }
}

impl Div for DoublePlusExponent {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self {
            m: self.m / rhs.m,
            e: self.e - rhs.e,
        }
        .normalize()
    }
}

impl PartialEq for DoublePlusExponent {
    fn eq(&self, other: &Self) -> bool {
        self.m == other.m && self.e == other.e
    }
}

impl PartialOrd for DoublePlusExponent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let diff = *self - *other;
        diff.m.partial_cmp(&0.0)
    }
}
