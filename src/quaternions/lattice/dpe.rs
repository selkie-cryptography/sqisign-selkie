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

use crate::quaternions::bigint::BigInt;

/// A double-precision float with an explicit base-2 exponent.
///
/// Represents the value `m * 2^e`. The mantissa is kept in
/// `[0.5, 1.0)` (or zero) after every operation.
#[derive(Clone, Copy)]
pub(crate) struct Dpe {
    /// Mantissa in `[0.5, 1.0)` or zero.
    pub(super) m: f64,
    /// Base-2 exponent.
    pub(super) e: i64,
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

/// Compute `x * 2^exp`. Pure bit manipulation.
fn ldexp(x: f64, exp: i32) -> f64 {
    // Multiply by 2^exp via a float whose exponent IS exp.
    // Split into two steps to avoid overflow for large exp.
    let half = exp / 2;
    let other = exp - half;
    let a = f64::from_bits(((1023i64 + half as i64) as u64) << 52);
    let b = f64::from_bits(((1023i64 + other as i64) as u64) << 52);
    x * a * b
}

impl Dpe {
    pub const ZERO: Self = Self { m: 0.0, e: 0 };

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

    /// Convert from a [`BigInt<N>`].
    ///
    /// Extracts the top ~53 bits of the absolute value as the
    /// mantissa, recording the remaining magnitude in the exponent.
    pub fn from_bigint<const N: usize>(v: &BigInt<N>) -> Self {
        let bits = v.bitsize();
        if bits == 0 {
            return Self::ZERO;
        }
        // Shift right to bring the top bits into f64 range,
        // then combine with the shift as exponent.
        let shift = if bits > 53 { bits - 53 } else { 0 };
        let truncated = v.shr(shift);
        let f = truncated.to_f64();
        let (frac, exp) = frexp(f);
        Self {
            m: frac,
            e: exp as i64 + shift as i64,
        }
    }

    /// Convert to a `BigInt<N>`, rounding to the nearest integer.
    ///
    /// This is the equivalent of the C reference's `ibz_set_dpe`:
    /// it converts the DPE mantissa to an integer and shifts left
    /// by the exponent. Needed because size-reduction μ values
    /// can exceed `i64` range (e.g., μ[3][0] ≈ 2^260 before
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
            r = r.shl(shift as u32);
        } else if shift < 0 {
            r = r.shr((-shift) as u32);
        }
        if neg {
            r = r.wrapping_neg();
        }
        r
    }

    /// Approximate `f64` value. May overflow to ±inf for very
    /// large exponents.
    pub fn to_f64(self) -> f64 {
        if self.m == 0.0 {
            return 0.0;
        }
        ldexp(self.m, self.e as i32)
    }

    pub fn abs(self) -> Self {
        Self {
            m: self.m.abs(),
            e: self.e,
        }
    }

    /// Round the represented value to the nearest integer (as
    /// `f64`). For values that fit in `i64`, this is exact.
    pub fn round(self) -> f64 {
        self.to_f64().round()
    }

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

impl core::ops::Add for Dpe {
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

impl core::ops::Sub for Dpe {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + Self {
            m: -rhs.m,
            e: rhs.e,
        }
    }
}

impl core::ops::SubAssign for Dpe {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::Mul for Dpe {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            m: self.m * rhs.m,
            e: self.e + rhs.e,
        }
        .normalize()
    }
}

impl core::ops::Div for Dpe {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self {
            m: self.m / rhs.m,
            e: self.e - rhs.e,
        }
        .normalize()
    }
}

impl PartialEq for Dpe {
    fn eq(&self, other: &Self) -> bool {
        self.m == other.m && self.e == other.e
    }
}

impl PartialOrd for Dpe {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        let diff = *self - *other;
        diff.m.partial_cmp(&0.0)
    }
}
