//! Integer square root for [`BigInt<N>`][super::BigInt].

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Integer square root: returns the largest `s` such that
    /// `s·s ≤ self`, or `None` if `self` is negative.
    ///
    /// Based on Newton-Raphson reciprocal square root (Algorithm 10,
    /// §2.5) from [Kouider et al.][ct-bigint], simplified here as a
    /// binary search on the result bits.
    ///
    /// [ct-bigint]: https://eprint.iacr.org/2025/832.pdf
    #[must_use]
    pub fn sqrt_floor(&self) -> Option<Self> {
        if bool::from(self.is_negative()) {
            return None;
        }
        if bool::from(self.is_zero()) {
            return Some(Self::ZERO);
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
                let sq = result.vt_mul(&result);
                if sq > *self {
                    // Clear the bit.
                    result.limbs[limb_idx] &= !(1u64 << bit_idx);
                }
            }
        }
        Some(result)
    }
}
