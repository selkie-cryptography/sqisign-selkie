//! Scalars for elliptic curve point multiplication.
//!
//! A [`Scalar`] represents a non-negative integer used as a multiplier
//! in the elliptic curve group law: `[n]P = P + P + ... + P` (n times).
//! For SQIsign's torsion arithmetic, scalars are integers mod 2^f
//! where f = 248 is the torsion exponent.

use core::fmt;

use crate::quaternions::bigint::BigInt;

/// A scalar for elliptic curve point multiplication.
///
/// Stored as four little-endian u64 limbs (256-bit unsigned integer).
/// This is sufficient for all SQIsign scalar multiplications, where
/// the maximum scalar is bounded by 2^f = 2^248 < 2^256.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Scalar([u64; 4]);

impl Scalar {
    /// Total number of bits (fixed, used for constant-time iteration).
    pub const BITS: u32 = 256;

    /// The zero scalar.
    pub const ZERO: Self = Self([0; 4]);

    /// The scalar one.
    pub const ONE: Self = Self([1, 0, 0, 0]);

    /// Creates a scalar from a `u64`.
    #[inline]
    pub const fn from_u64(val: u64) -> Self {
        Self([val, 0, 0, 0])
    }

    /// Creates a scalar from four little-endian u64 limbs.
    #[inline]
    pub const fn from_limbs(limbs: [u64; 4]) -> Self {
        Self(limbs)
    }

    /// Returns the limbs in little-endian order.
    #[inline]
    pub const fn as_limbs(&self) -> &[u64; 4] {
        &self.0
    }

    /// Serialize to 32 little-endian bytes.
    #[must_use]
    pub fn to_le_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, limb) in self.0.iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        out
    }

    /// Returns a big-endian bit iterator over the scalar.
    ///
    /// Always yields exactly `bits` bits (zero-padded from the MSB).
    /// The iterator length is public and determines the number of
    /// ladder steps, so it must be a fixed public constant.
    pub fn bits_be(&self, bits: u32) -> impl Iterator<Item = bool> + '_ {
        (0..bits).rev().map(move |i| {
            let limb = (i / 64) as usize;
            let bit = (i % 64) as u64;
            if limb < 4 {
                (self.0[limb] >> bit) & 1 == 1
            } else {
                false
            }
        })
    }

    /// Returns the total number of significant bits.
    pub fn bit_length(&self) -> u32 {
        for i in (0..4).rev() {
            if self.0[i] != 0 {
                return (i as u32) * 64 + (64 - self.0[i].leading_zeros());
            }
        }
        0
    }

    // -----------------------------------------------------------------
    // Unsigned modular arithmetic mod 2^k.
    //
    // Every operation below is constant-time: fixed iteration counts,
    // no branches on values. The modulus 2^k is enforced by masking
    // (truncation), so no expensive modular reduction is needed.
    //
    // TODO: the runtime k parameter on every call is not great. We
    // considered `Scalar<const K: u32>` (const-generic, enables
    // `impl Add` etc.) but k varies per signature for M_chl
    // application (e_rsp_prime + 2 + r_rsp). Other options: newtype
    // carrying k at runtime, or a `ScalarRing` builder. Revisit when
    // the signing flow stabilizes and the set of k values is clearer.
    // -----------------------------------------------------------------

    /// Reduce mod 2^k by zeroing bits ≥ k.
    #[must_use]
    pub fn reduce_mod2k(&self, k: u32) -> Self {
        debug_assert!(k <= 256);
        let mut out = self.0;
        let full_limbs = (k / 64) as usize;
        let rem_bits = k % 64;
        if full_limbs < 4 {
            if rem_bits > 0 {
                out[full_limbs] &= (1u64 << rem_bits) - 1;
            } else {
                out[full_limbs] = 0;
            }
            for limb in out.iter_mut().skip(full_limbs + 1) {
                *limb = 0;
            }
        }
        Self(out)
    }

    /// Add mod 2^k.
    #[must_use]
    pub fn add_mod2k(&self, rhs: &Self, k: u32) -> Self {
        let mut out = [0u64; 4];
        let mut carry = 0u64;
        for (i, out_limb) in out.iter_mut().enumerate() {
            let (s, c1) = self.0[i].overflowing_add(rhs.0[i]);
            let (s, c2) = s.overflowing_add(carry);
            *out_limb = s;
            carry = (c1 as u64) + (c2 as u64);
        }
        Self(out).reduce_mod2k(k)
    }

    /// Subtract mod 2^k (wrapping).
    #[must_use]
    pub fn sub_mod2k(&self, rhs: &Self, k: u32) -> Self {
        let mut out = [0u64; 4];
        let mut borrow = 0u64;
        for (i, out_limb) in out.iter_mut().enumerate() {
            let (d, b1) = self.0[i].overflowing_sub(rhs.0[i]);
            let (d, b2) = d.overflowing_sub(borrow);
            *out_limb = d;
            borrow = (b1 as u64) + (b2 as u64);
        }
        Self(out).reduce_mod2k(k)
    }

    /// Multiply mod 2^k (schoolbook, keep low 256 bits).
    #[must_use]
    pub fn mul_mod2k(&self, rhs: &Self, k: u32) -> Self {
        let mut out = [0u64; 4];
        for i in 0..4 {
            let mut carry = 0u128;
            for j in 0..4 {
                if i + j < 4 {
                    let prod =
                        (self.0[i] as u128) * (rhs.0[j] as u128) + (out[i + j] as u128) + carry;
                    out[i + j] = prod as u64;
                    carry = prod >> 64;
                }
            }
        }
        Self(out).reduce_mod2k(k)
    }

    /// Invert mod 2^k via Hensel lifting.
    ///
    /// Returns `None` if `self` is even (no inverse mod 2^k).
    /// Constant-time: always performs ⌈log₂(k)⌉ iterations.
    #[must_use]
    pub fn inv_mod2k(&self, k: u32) -> Option<Self> {
        if self.0[0] & 1 == 0 {
            return None;
        }
        // x ≡ 1 mod 2 (any odd number is its own inverse mod 2).
        let mut x = Self::from_u64(1);
        // Lift: x ← x · (2 − self · x) mod 2^pow.
        // After each iteration, pow doubles (capped at k).
        let mut pow = 1u32;
        while pow < k {
            pow = (pow * 2).min(k);
            let ax = self.mul_mod2k(&x, pow);
            let two = Self::from_u64(2);
            let correction = two.sub_mod2k(&ax, pow);
            x = x.mul_mod2k(&correction, pow);
        }
        Some(x)
    }
}

impl From<u64> for Scalar {
    fn from(val: u64) -> Self {
        Self::from_u64(val)
    }
}

impl From<Scalar> for BigInt<4> {
    /// Convert a `Scalar` (unsigned) to a non-negative `BigInt<4>`.
    fn from(s: Scalar) -> Self {
        Self::from_limbs(s.0)
    }
}

impl From<&Scalar> for BigInt<4> {
    fn from(s: &Scalar) -> Self {
        Self::from_limbs(s.0)
    }
}

/// Convert a signed `BigInt<4>` to a `Scalar` (unsigned mod 2^256).
///
/// Negative values are reduced: `-x` becomes `2^256 - x`. This is
/// necessary because `BigInt` is sign-magnitude while `Scalar` is
/// unsigned modular. A previous version took the raw limbs without
/// checking the sign, silently mapping `-x` to `+x` — this
/// corrupted the action matrix whenever `decompose` returned
/// negative coefficients (which happens for most nontrivial
/// quaternion elements).
impl From<BigInt<4>> for Scalar {
    fn from(b: BigInt<4>) -> Self {
        if bool::from(b.is_negative()) {
            let abs_scalar = Self::from_limbs(*b.abs().as_limbs());
            let zero = Self::from_u64(0);
            zero.sub_mod2k(&abs_scalar, 256)
        } else {
            Self::from_limbs(*b.as_limbs())
        }
    }
}

impl subtle::ConditionallySelectable for Scalar {
    fn conditional_select(a: &Self, b: &Self, choice: subtle::Choice) -> Self {
        Self([
            u64::conditional_select(&a.0[0], &b.0[0], choice),
            u64::conditional_select(&a.0[1], &b.0[1], choice),
            u64::conditional_select(&a.0[2], &b.0[2], choice),
            u64::conditional_select(&a.0[3], &b.0[3], choice),
        ])
    }
}

impl fmt::Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scalar(0x")?;
        for i in (0..4).rev() {
            write!(f, "{:016x}", self.0[i])?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_be_small() {
        let s = Scalar::from_u64(0b1011);
        let bits: Vec<bool> = s.bits_be(4).collect();
        assert_eq!(bits, vec![true, false, true, true]);
    }

    #[test]
    fn bits_be_zero_padded() {
        let s = Scalar::from_u64(3); // 0b11
        let bits: Vec<bool> = s.bits_be(8).collect();
        assert_eq!(
            bits,
            vec![false, false, false, false, false, false, true, true]
        );
    }

    #[test]
    fn bit_length() {
        assert_eq!(Scalar::ZERO.bit_length(), 0);
        assert_eq!(Scalar::ONE.bit_length(), 1);
        assert_eq!(Scalar::from_u64(255).bit_length(), 8);
        assert_eq!(Scalar::from_u64(256).bit_length(), 9);
    }

    #[test]
    fn reduce_mod2k_masks_correctly() {
        let s = Scalar::from_u64(0xFF);
        assert_eq!(s.reduce_mod2k(4), Scalar::from_u64(0x0F));
        assert_eq!(s.reduce_mod2k(8), Scalar::from_u64(0xFF));
        assert_eq!(s.reduce_mod2k(1), Scalar::from_u64(1));
    }

    #[test]
    fn add_mod2k_wraps() {
        let a = Scalar::from_u64(250);
        let b = Scalar::from_u64(10);
        // 250 + 10 = 260 = 0x104, mod 2^8 = 4
        assert_eq!(a.add_mod2k(&b, 8), Scalar::from_u64(4));
    }

    #[test]
    fn sub_mod2k_wraps() {
        let a = Scalar::from_u64(3);
        let b = Scalar::from_u64(5);
        // 3 - 5 mod 2^8 = 254
        assert_eq!(a.sub_mod2k(&b, 8), Scalar::from_u64(254));
    }

    #[test]
    fn mul_mod2k_truncates() {
        let a = Scalar::from_u64(200);
        let b = Scalar::from_u64(200);
        // 200 * 200 = 40000 = 0x9C40, mod 2^8 = 0x40 = 64
        assert_eq!(a.mul_mod2k(&b, 8), Scalar::from_u64(64));
    }

    #[test]
    fn inv_mod2k_round_trip() {
        let a = Scalar::from_u64(7); // odd
        let inv = a.inv_mod2k(248).unwrap();
        let product = a.mul_mod2k(&inv, 248);
        assert_eq!(product, Scalar::ONE);
    }

    #[test]
    fn inv_mod2k_even_returns_none() {
        let a = Scalar::from_u64(6); // even
        assert!(a.inv_mod2k(248).is_none());
    }

    #[test]
    fn inv_mod2k_large_odd() {
        // 2^248 - 1 is odd
        let a = Scalar::from_limbs([
            u64::MAX,
            u64::MAX,
            u64::MAX,
            (1u64 << 56) - 1, // 248 bits
        ]);
        let inv = a.inv_mod2k(248).unwrap();
        let product = a.mul_mod2k(&inv, 248);
        assert_eq!(product, Scalar::ONE);
    }
}
