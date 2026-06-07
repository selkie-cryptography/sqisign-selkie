//! Encoding for [`BigInt<N>`][super::BigInt]: byte decode
//! ([`from_bytes_le_unsigned`][BigInt::from_bytes_le_unsigned],
//! [`from_bytes_le_signed`][BigInt::from_bytes_le_signed]), floating-point
//! conversion ([`to_f64`][BigInt::to_f64],
//! [`to_f64_trunc`][BigInt::to_f64_trunc]), and the `Debug`/`Display`
//! impls.

use core::fmt;

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Decodes from little-endian bytes (unsigned, non-negative).
    ///
    /// Packs bytes into u64 limbs. Caller must ensure
    /// `bytes.len() <= N * 8`; oversized input panics on the inevitable
    /// `limbs[i]` bounds-check inside the loop, but the
    /// `debug_assert!` here gives a clearer failure in test builds.
    /// All in-tree callers are audited and satisfy the precondition.
    pub fn from_bytes_le_unsigned(bytes: &[u8]) -> Self {
        debug_assert!(bytes.len() <= N * 8);
        let mut limbs = [0u64; N];
        for (i, chunk) in bytes.chunks(8).enumerate() {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            limbs[i] = u64::from_le_bytes(buf);
        }
        Self { sign: 0, limbs }
    }

    /// Decodes from little-endian bytes (signed, two's complement).
    ///
    /// The highest bit of the last byte is the sign bit. Same
    /// precondition as [`Self::from_bytes_le_unsigned`].
    pub fn from_bytes_le_signed(bytes: &[u8]) -> Self {
        debug_assert!(!bytes.is_empty() && bytes.len() <= N * 8);
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

    /// Converts the **magnitude** to `f64` byte-for-byte matching
    /// mini-GMP's `mpz_get_d`
    /// (`the-sqisign/src/mini-gmp/mini-gmp.c:1773-1808`).
    ///
    /// Mini-GMP processes limbs from most-significant down, masking
    /// off bits below the 53-bit mantissa boundary
    /// (round-toward-zero truncation, **not** round-to-nearest).
    /// This is the rounding semantics the C ref's `dpe_set_z` →
    /// `mini_mpz_get_d_2exp` chain relies on; matching it bit-exact
    /// is the load-bearing part of byte-equal LLL.
    ///
    /// Returns `0.0` for zero input. **Sign is ignored** — this
    /// returns the magnitude only, mirroring `mpz_get_d`'s "magnitude
    /// then negate" structure (callers apply the sign separately;
    /// see [`DoublePlusExponent::from_bigint`](crate::quaternions::lattice::dpe::DoublePlusExponent::from_bigint)).
    ///
    /// Distinct from [`to_f64`](Self::to_f64), which does
    /// signed conversion via repeated `val * 2^64 + limb`
    /// accumulation — bit-different at the rounding boundary.
    pub fn to_f64_trunc(self) -> f64 {
        let limbs = &self.limbs;
        let mut un = N;
        while un > 0 && limbs[un - 1] == 0 {
            un -= 1;
        }
        if un == 0 {
            return 0.0;
        }

        let mut l = limbs[un - 1];
        un -= 1;

        // m = clz(top_limb) + 53 - 64 = clz - 11. Range: m ∈ [-11, 52].
        // Negative ⇒ top limb already exceeds 53 mantissa bits; mask
        // off the low (-m) bits before converting.
        let mut m: i32 = (l.leading_zeros() as i32) + 53 - 64;
        if m < 0 {
            // (-m) ∈ [1, 11]; well-defined u64 shift.
            l &= u64::MAX.wrapping_shl((-m) as u32);
        }

        // B = 2^64 as f64 (exact: mantissa 1.0, exp 64).
        let b: f64 = (1u128 << 64) as f64;
        let mut x_d: f64 = l as f64;

        while un > 0 {
            un -= 1;
            x_d *= b;
            if m > 0 {
                let mut l2 = limbs[un];
                m -= 64;
                if m < 0 {
                    // (-m) ∈ [1, 63] in this branch (entry m ∈ [1, 52]).
                    l2 &= u64::MAX.wrapping_shl((-m) as u32);
                }
                x_d += l2 as f64;
            }
            // If m ≤ 0 from the outset (top limb already saturated),
            // we still apply `x *= B` but contribute nothing from
            // this limb — matches mini-GMP exactly.
        }

        x_d
    }
}

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
