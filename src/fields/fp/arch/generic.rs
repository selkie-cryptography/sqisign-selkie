//! `Fp55` backend: radix-55 Montgomery on `[u64; 6]`.
//!
//! Cross-architecture baseline that compiles on any target. Each limb
//! holds 55 bits in unsaturated form. `p = 3 * 2^324 - 1` folds through
//! `P5 = 3 * 2^49` per column, and `R = 2^330 > 4p` means products need
//! no final conditional subtraction. Same layout as the C reference's
//! portable backend, whose fixed exponent chain is the oracle for
//! [`Fp55::pow_p3div4`].
//!
//! Tables in `params.rs` use this backend's limb shape as the source of
//! truth.

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::fields::fp::FP_ENCODED_BYTES;

/// Radix used for limb representation: 2^55.
const RADIX: u32 = 55;

/// Mask for a single radix-55 limb.
const MASK: u64 = (1u64 << RADIX) - 1;

/// The high-limb constant: c * 2^(324 - 5 * 55) = 3 * 2^49.
/// This is `p5` in the C reference: 0x6000000000000.
const P5: u64 = 3u64 << 49;

/// Number of limbs.
const LIMBS: usize = 6;

/// An element of the prime field F_p, where p = 3 * 2^324 - 1.
///
/// Internally stored in Montgomery form using unsaturated radix-2^55
/// representation with 6 limbs. Limbs are allowed to be slightly
/// unreduced between operations.
#[derive(Copy, Clone)]
pub struct Fp55(pub(crate) [u64; LIMBS]);

impl Fp55 {
    /// Constructs from raw radix-55 limbs (already in Montgomery form).
    ///
    /// This is a const constructor for embedding precomputed constants.
    /// The caller is responsible for ensuring the limbs represent a
    /// valid Montgomery-form field element.
    pub const fn from_limbs(limbs: [u64; LIMBS]) -> Fp55 {
        Fp55(limbs)
    }
}

impl core::fmt::Debug for Fp55 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Fp55({:?})", &self.0[..])
    }
}

/// R^2 mod p, used for converting to Montgomery form.
/// R = 2^(6 * 55) = 2^330.
///
/// Equals the C reference's `nres()` constant.
const R2: Fp55 = Fp55([
    0x5555555555571C,
    0x2AAAAAAAAAAAAA,
    0x55555555555555,
    0x2AAAAAAAAAAAAA,
    0x55555555555555,
    0x02AAAAAAAAAAAA,
]);

impl Fp55 {
    /// The additive identity (zero) in Montgomery form.
    pub const ZERO: Fp55 = Fp55([0, 0, 0, 0, 0, 0]);

    /// The multiplicative identity (one) in Montgomery form.
    pub const ONE: Fp55 = Fp55([0x15, 0, 0, 0, 0, 0x02000000000000]);

    /// The constant 2 in Montgomery form.
    pub const TWO: Fp55 = Fp55([0x2A, 0, 0, 0, 0, 0x04000000000000]);

    /// The constant 4 in Montgomery form.
    pub const FOUR: Fp55 = Fp55([0x55, 0, 0, 0, 0, 0x02000000000000]);

    /// The constant `2^-1 mod p` in Montgomery form.
    pub const TWO_INV: Fp55 = Fp55([0x0A, 0, 0, 0, 0, 0x04000000000000]);

    /// The constant `-1 mod p` in Montgomery form.
    pub const MINUS_ONE: Fp55 = Fp55([
        0x7FFFFFFFFFFFEA,
        0x7FFFFFFFFFFFFF,
        0x7FFFFFFFFFFFFF,
        0x7FFFFFFFFFFFFF,
        0x7FFFFFFFFFFFFF,
        0x03FFFFFFFFFFFF,
    ]);

    /// Constructs a field element from a small integer.
    pub fn from_small(x: u32) -> Fp55 {
        let mut a = Fp55::ZERO;
        a.0[0] = x as u64;
        a.to_montgomery()
    }

    /// Converts a field element in normal form to Montgomery form.
    fn to_montgomery(self) -> Fp55 {
        &self * &R2
    }

    /// Converts from Montgomery form back to normal (canonical) form.
    fn reduce_montgomery(self) -> Fp55 {
        let one = Fp55([1, 0, 0, 0, 0, 0]);
        let mut r = &self * &one;
        r.final_sub();
        r
    }

    /// Encodes this field element as 41 bytes, little-endian.
    pub fn to_bytes(self) -> [u8; FP_ENCODED_BYTES] {
        let c = self.reduce_montgomery();
        let mut out = [0u8; FP_ENCODED_BYTES];

        // Pack the 6 radix-55 limbs into 41 bytes, little-endian.
        let mut acc: u128 = 0;
        let mut bits = 0u32;
        let mut pos = 0;
        for &limb in c.0.iter() {
            acc |= (limb as u128) << bits;
            bits += RADIX;
            while bits >= 8 && pos < FP_ENCODED_BYTES {
                out[pos] = acc as u8;
                acc >>= 8;
                bits -= 8;
                pos += 1;
            }
        }
        if pos < FP_ENCODED_BYTES {
            out[pos] = acc as u8;
        }

        out
    }

    /// Decodes 41 bytes (little-endian) into a field element.
    ///
    /// Returns the element in Montgomery form. The input must be a
    /// canonical encoding (i.e., the value must be less than p).
    pub fn from_bytes(bytes: &[u8; FP_ENCODED_BYTES]) -> Fp55 {
        // Unpack 41 bytes into 6 radix-55 limbs.
        let mut limbs = [0u64; LIMBS];
        let mut acc: u128 = 0;
        let mut bits = 0u32;
        let mut limb_idx = 0;
        for &byte in bytes.iter() {
            acc |= (byte as u128) << bits;
            bits += 8;
            if bits >= RADIX && limb_idx < LIMBS - 1 {
                limbs[limb_idx] = (acc as u64) & MASK;
                acc >>= RADIX;
                bits -= RADIX;
                limb_idx += 1;
            }
        }
        limbs[limb_idx] = acc as u64;

        let mut r = Fp55(limbs);
        // Canonicalize: subtract p, add back if underflow.
        r.final_sub();
        r.to_montgomery()
    }

    /// Squares this field element.
    #[must_use]
    #[inline]
    pub fn square(&self) -> Fp55 {
        let a = &self.0;
        let (a0, a1, a2, a3, a4, a5) = (
            a[0] as u128,
            a[1] as u128,
            a[2] as u128,
            a[3] as u128,
            a[4] as u128,
            a[5] as u128,
        );
        let p5 = P5 as u128;

        // Column 0.
        let mut t: u128 = a0 * a0;
        let v0 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 1.
        t += (a0 * a1) * 2;
        let v1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 2.
        t += (a0 * a2) * 2 + a1 * a1;
        let v2 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 3.
        t += (a0 * a3 + a1 * a2) * 2;
        let v3 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 4.
        t += (a0 * a4 + a1 * a3) * 2 + a2 * a2;
        let v4 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 5 (start reduction: fold v0 * P5).
        t += (a0 * a5 + a1 * a4 + a2 * a3) * 2;
        t += (v0 as u128) * p5;
        let v5 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 6 (reduce v1).
        t += (a1 * a5 + a2 * a4) * 2 + a3 * a3;
        t += (v1 as u128) * p5;
        let c0 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 7 (reduce v2).
        t += (a2 * a5 + a3 * a4) * 2;
        t += (v2 as u128) * p5;
        let c1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 8 (reduce v3).
        t += (a3 * a5) * 2 + a4 * a4;
        t += (v3 as u128) * p5;
        let c2 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 9 (reduce v4).
        t += (a4 * a5) * 2;
        t += (v4 as u128) * p5;
        let c3 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 10 (reduce v5).
        t += a5 * a5;
        t += (v5 as u128) * p5;
        let c4 = (t as u64) & MASK;
        t >>= RADIX;

        Fp55([c0, c1, c2, c3, c4, t as u64])
    }

    /// Squares this element `n` times.
    #[must_use]
    pub fn pow2k(&self, n: u32) -> Fp55 {
        let mut r = *self;
        for _ in 0..n {
            r = r.square();
        }
        r
    }

    /// Computes `self^((p-3)/4)`, with (p-3)/4 = 2^323 + 2^322 - 1.
    ///
    /// This is used to derive inversions, square roots, and Legendre
    /// symbols. The addition chain is the C reference's `modpro`.
    #[must_use]
    pub(crate) fn pow_p3div4(&self) -> Fp55 {
        let x = *self;
        let z = x.square();
        let z = &x * &z;
        let t0 = z.pow2k(2);
        let t0 = &z * &t0;
        let t1 = t0.pow2k(4);
        let t0 = &t0 * &t1;
        let t1 = t0.pow2k(2);
        let z = &z * &t1;
        let t2 = z.pow2k(4);
        let t1 = t2.pow2k(4);
        let t3 = t1.pow2k(10);
        let t1 = &t1 * &t3;
        let t3 = t1.pow2k(6);
        let t2 = &t2 * &t3;
        let t2 = t2.pow2k(24);
        let t1 = &t1 * &t2;
        let t0 = &t0 * &t1;
        let t1 = t0.pow2k(10);
        let z = &z * &t1;
        let t1 = z.pow2k(58);
        let t1 = &t0 * &t1;
        let t0 = &x * &t1;
        let t2 = t0.square();
        let t1 = &t1 * &t2;
        let t2 = t1.pow2k(128);
        let t1 = &t1 * &t2;
        let t0 = &t0 * &t1;
        let t0 = t0.pow2k(68);
        &z * &t0
    }

    /// Computes the multiplicative inverse: self^(p-2).
    #[must_use]
    pub fn invert(&self) -> Fp55 {
        let t = self.pow_p3div4();
        let t = t.pow2k(2);
        self * &t
    }

    /// Tests whether this element is a quadratic residue (square) in F_p.
    pub fn is_square(&self) -> Choice {
        let r = self.pow_p3div4();
        let r = r.square();
        let r = &r * self;
        r.ct_eq(&Fp55::ONE) | self.ct_eq(&Fp55::ZERO)
    }

    /// Computes the square root (when self is a QR).
    ///
    /// The result is only meaningful when `self.is_square()` is true.
    #[must_use]
    pub fn sqrt(&self) -> Fp55 {
        let y = self.pow_p3div4();
        &y * self
    }

    /// Propagate carries through the limbs. Returns a sign/borrow indicator.
    fn prop(&mut self) -> u64 {
        let mut carry = self.0[0] as i64;
        carry >>= RADIX;
        self.0[0] &= MASK;
        for limb in &mut self.0[1..LIMBS - 1] {
            carry += *limb as i64;
            *limb = (carry as u64) & MASK;
            carry >>= RADIX;
        }
        self.0[LIMBS - 1] = self.0[LIMBS - 1].wrapping_add(carry as u64);
        // Return a mask: 0 if positive, all-ones if negative.
        let sign = (self.0[LIMBS - 1] >> 1) >> 62;
        sign.wrapping_neg()
    }

    /// Propagate carries and conditionally add p if negative.
    fn flatten(&mut self) {
        let carry = self.prop();
        self.0[0] = self.0[0].wrapping_sub(1u64 & carry);
        self.0[LIMBS - 1] = self.0[LIMBS - 1].wrapping_add(P5 & carry);
        self.prop();
    }

    /// Montgomery final subtraction: subtract p and flatten.
    fn final_sub(&mut self) {
        self.0[0] = self.0[0].wrapping_add(1);
        self.0[LIMBS - 1] = self.0[LIMBS - 1].wrapping_sub(P5);
        self.flatten();
    }
}

impl Fp55 {
    /// Returns `a1 * b1 + a2 * b2 mod p` with a single Montgomery reduction.
    ///
    /// Implements [Longa's interleaved sum-of-products][longa]
    /// (ePrint 2022/367, Algorithm 5). Both products are accumulated
    /// into one radix-2^55 column accumulator, and the `P5` Montgomery
    /// fold is applied once, which is cheaper than two separate `mul`s
    /// each carrying their own reduction.
    ///
    /// # Implementation
    ///
    /// The widest column (k = 5) sums at most `2 * 6 = 12` partial
    /// products, each `< (2^55)^2 = 2^110`, plus a reduction term
    /// `< 2^55 * P5 < 2^106` and a carry `< 2^60`, so the `u128`
    /// accumulator stays under `2^114`.
    ///
    /// [longa]: https://eprint.iacr.org/2022/367.pdf
    #[must_use]
    #[inline]
    pub fn sum_of_2_products(a1: &Fp55, b1: &Fp55, a2: &Fp55, b2: &Fp55) -> Fp55 {
        debug_assert!(a1.0.iter().all(|&x| x <= MASK), "a1 limb exceeds MASK");
        debug_assert!(b1.0.iter().all(|&x| x <= MASK), "b1 limb exceeds MASK");
        debug_assert!(a2.0.iter().all(|&x| x <= MASK), "a2 limb exceeds MASK");
        debug_assert!(b2.0.iter().all(|&x| x <= MASK), "b2 limb exceeds MASK");
        let (a, b) = (&a1.0, &b1.0);
        let (c, d) = (&a2.0, &b2.0);
        let p5 = P5 as u128;

        let mut t: u128 = 0;

        // Column 0.
        t += (a[0] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[0] as u128);
        let v0 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 1.
        t += (a[0] as u128) * (b[1] as u128);
        t += (a[1] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[1] as u128);
        t += (c[1] as u128) * (d[0] as u128);
        let v1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 2.
        t += (a[0] as u128) * (b[2] as u128);
        t += (a[1] as u128) * (b[1] as u128);
        t += (a[2] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[2] as u128);
        t += (c[1] as u128) * (d[1] as u128);
        t += (c[2] as u128) * (d[0] as u128);
        let v2 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 3.
        t += (a[0] as u128) * (b[3] as u128);
        t += (a[1] as u128) * (b[2] as u128);
        t += (a[2] as u128) * (b[1] as u128);
        t += (a[3] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[3] as u128);
        t += (c[1] as u128) * (d[2] as u128);
        t += (c[2] as u128) * (d[1] as u128);
        t += (c[3] as u128) * (d[0] as u128);
        let v3 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 4.
        t += (a[0] as u128) * (b[4] as u128);
        t += (a[1] as u128) * (b[3] as u128);
        t += (a[2] as u128) * (b[2] as u128);
        t += (a[3] as u128) * (b[1] as u128);
        t += (a[4] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[4] as u128);
        t += (c[1] as u128) * (d[3] as u128);
        t += (c[2] as u128) * (d[2] as u128);
        t += (c[3] as u128) * (d[1] as u128);
        t += (c[4] as u128) * (d[0] as u128);
        let v4 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 5 (start reduction: fold v0 * P5).
        t += (a[0] as u128) * (b[5] as u128);
        t += (a[1] as u128) * (b[4] as u128);
        t += (a[2] as u128) * (b[3] as u128);
        t += (a[3] as u128) * (b[2] as u128);
        t += (a[4] as u128) * (b[1] as u128);
        t += (a[5] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[5] as u128);
        t += (c[1] as u128) * (d[4] as u128);
        t += (c[2] as u128) * (d[3] as u128);
        t += (c[3] as u128) * (d[2] as u128);
        t += (c[4] as u128) * (d[1] as u128);
        t += (c[5] as u128) * (d[0] as u128);
        t += (v0 as u128) * p5;
        let v5 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 6 (reduce v1).
        t += (a[1] as u128) * (b[5] as u128);
        t += (a[2] as u128) * (b[4] as u128);
        t += (a[3] as u128) * (b[3] as u128);
        t += (a[4] as u128) * (b[2] as u128);
        t += (a[5] as u128) * (b[1] as u128);
        t += (c[1] as u128) * (d[5] as u128);
        t += (c[2] as u128) * (d[4] as u128);
        t += (c[3] as u128) * (d[3] as u128);
        t += (c[4] as u128) * (d[2] as u128);
        t += (c[5] as u128) * (d[1] as u128);
        t += (v1 as u128) * p5;
        let c0 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 7 (reduce v2).
        t += (a[2] as u128) * (b[5] as u128);
        t += (a[3] as u128) * (b[4] as u128);
        t += (a[4] as u128) * (b[3] as u128);
        t += (a[5] as u128) * (b[2] as u128);
        t += (c[2] as u128) * (d[5] as u128);
        t += (c[3] as u128) * (d[4] as u128);
        t += (c[4] as u128) * (d[3] as u128);
        t += (c[5] as u128) * (d[2] as u128);
        t += (v2 as u128) * p5;
        let c1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 8 (reduce v3).
        t += (a[3] as u128) * (b[5] as u128);
        t += (a[4] as u128) * (b[4] as u128);
        t += (a[5] as u128) * (b[3] as u128);
        t += (c[3] as u128) * (d[5] as u128);
        t += (c[4] as u128) * (d[4] as u128);
        t += (c[5] as u128) * (d[3] as u128);
        t += (v3 as u128) * p5;
        let c2 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 9 (reduce v4).
        t += (a[4] as u128) * (b[5] as u128);
        t += (a[5] as u128) * (b[4] as u128);
        t += (c[4] as u128) * (d[5] as u128);
        t += (c[5] as u128) * (d[4] as u128);
        t += (v4 as u128) * p5;
        let c3 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 10 (reduce v5).
        t += (a[5] as u128) * (b[5] as u128);
        t += (c[5] as u128) * (d[5] as u128);
        t += (v5 as u128) * p5;
        let c4 = (t as u64) & MASK;
        t >>= RADIX;

        Fp55([c0, c1, c2, c3, c4, t as u64])
    }

    /// Returns `a1 * b1 - a2 * b2 mod p` with a single Montgomery reduction.
    ///
    /// Negates `b2` (one limb-wise pass) and defers to
    /// [`Fp55::sum_of_2_products`]. Used by the `Fp2` product to
    /// compute `c0 = a0 * b0 - a1 * b1`.
    #[must_use]
    pub fn difference_of_2_products(a1: &Fp55, b1: &Fp55, a2: &Fp55, b2: &Fp55) -> Fp55 {
        let neg_b2 = -b2;
        Fp55::sum_of_2_products(a1, b1, a2, &neg_b2)
    }
}

impl<'b> Add<&'b Fp55> for &Fp55 {
    type Output = Fp55;

    /// Modular addition, reduced to less than 2p.
    #[inline]
    fn add(self, rhs: &'b Fp55) -> Fp55 {
        let mut n = Fp55([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
            self.0[5] + rhs.0[5],
        ]);
        // Subtract 2p.
        n.0[0] = n.0[0].wrapping_add(2);
        n.0[LIMBS - 1] = n.0[LIMBS - 1].wrapping_sub(2 * P5);
        let carry = n.prop();
        // Add 2p back if underflow.
        n.0[0] = n.0[0].wrapping_sub(2u64 & carry);
        n.0[LIMBS - 1] = n.0[LIMBS - 1].wrapping_add((2 * P5) & carry);
        n.prop();
        n
    }
}

impl<'b> Sub<&'b Fp55> for &Fp55 {
    type Output = Fp55;

    /// Modular subtraction, reduced to less than 2p.
    #[inline]
    fn sub(self, rhs: &'b Fp55) -> Fp55 {
        let mut n = Fp55([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
            self.0[2].wrapping_sub(rhs.0[2]),
            self.0[3].wrapping_sub(rhs.0[3]),
            self.0[4].wrapping_sub(rhs.0[4]),
            self.0[5].wrapping_sub(rhs.0[5]),
        ]);
        let carry = n.prop();
        n.0[0] = n.0[0].wrapping_sub(2u64 & carry);
        n.0[LIMBS - 1] = n.0[LIMBS - 1].wrapping_add((2 * P5) & carry);
        n.prop();
        n
    }
}

impl Neg for &Fp55 {
    type Output = Fp55;

    #[inline]
    fn neg(self) -> Fp55 {
        &Fp55::ZERO - self
    }
}

impl<'b> Mul<&'b Fp55> for &Fp55 {
    type Output = Fp55;

    /// Modular multiplication (Montgomery form), reduced to less than 2p.
    ///
    /// Schoolbook with interleaved reduction: column k drops v_k and
    /// adds v_k * P5 at column k + 5, since -p^-1 = 1 mod 2^55 for
    /// p = 3 * 2^324 - 1.
    #[inline]
    fn mul(self, rhs: &'b Fp55) -> Fp55 {
        let (a, b) = (&self.0, &rhs.0);
        let p5 = P5 as u128;

        let mut t: u128 = 0;

        // Column 0.
        t += (a[0] as u128) * (b[0] as u128);
        let v0 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 1.
        t += (a[0] as u128) * (b[1] as u128);
        t += (a[1] as u128) * (b[0] as u128);
        let v1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 2.
        t += (a[0] as u128) * (b[2] as u128);
        t += (a[1] as u128) * (b[1] as u128);
        t += (a[2] as u128) * (b[0] as u128);
        let v2 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 3.
        t += (a[0] as u128) * (b[3] as u128);
        t += (a[1] as u128) * (b[2] as u128);
        t += (a[2] as u128) * (b[1] as u128);
        t += (a[3] as u128) * (b[0] as u128);
        let v3 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 4.
        t += (a[0] as u128) * (b[4] as u128);
        t += (a[1] as u128) * (b[3] as u128);
        t += (a[2] as u128) * (b[2] as u128);
        t += (a[3] as u128) * (b[1] as u128);
        t += (a[4] as u128) * (b[0] as u128);
        let v4 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 5 (start reduction: fold v0 * P5).
        t += (a[0] as u128) * (b[5] as u128);
        t += (a[1] as u128) * (b[4] as u128);
        t += (a[2] as u128) * (b[3] as u128);
        t += (a[3] as u128) * (b[2] as u128);
        t += (a[4] as u128) * (b[1] as u128);
        t += (a[5] as u128) * (b[0] as u128);
        t += (v0 as u128) * p5;
        let v5 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 6 (reduce v1).
        t += (a[1] as u128) * (b[5] as u128);
        t += (a[2] as u128) * (b[4] as u128);
        t += (a[3] as u128) * (b[3] as u128);
        t += (a[4] as u128) * (b[2] as u128);
        t += (a[5] as u128) * (b[1] as u128);
        t += (v1 as u128) * p5;
        let c0 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 7 (reduce v2).
        t += (a[2] as u128) * (b[5] as u128);
        t += (a[3] as u128) * (b[4] as u128);
        t += (a[4] as u128) * (b[3] as u128);
        t += (a[5] as u128) * (b[2] as u128);
        t += (v2 as u128) * p5;
        let c1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 8 (reduce v3).
        t += (a[3] as u128) * (b[5] as u128);
        t += (a[4] as u128) * (b[4] as u128);
        t += (a[5] as u128) * (b[3] as u128);
        t += (v3 as u128) * p5;
        let c2 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 9 (reduce v4).
        t += (a[4] as u128) * (b[5] as u128);
        t += (a[5] as u128) * (b[4] as u128);
        t += (v4 as u128) * p5;
        let c3 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 10 (reduce v5).
        t += (a[5] as u128) * (b[5] as u128);
        t += (v5 as u128) * p5;
        let c4 = (t as u64) & MASK;
        t >>= RADIX;

        Fp55([c0, c1, c2, c3, c4, t as u64])
    }
}

// Convenience impls: owned variants delegate to reference impls.

impl Add<Fp55> for Fp55 {
    type Output = Fp55;

    fn add(self, rhs: Fp55) -> Fp55 {
        &self + &rhs
    }
}

impl Sub<Fp55> for Fp55 {
    type Output = Fp55;

    fn sub(self, rhs: Fp55) -> Fp55 {
        &self - &rhs
    }
}

impl Mul<Fp55> for Fp55 {
    type Output = Fp55;

    fn mul(self, rhs: Fp55) -> Fp55 {
        &self * &rhs
    }
}

impl Neg for Fp55 {
    type Output = Fp55;

    fn neg(self) -> Fp55 {
        -&self
    }
}

impl AddAssign<&Fp55> for Fp55 {
    fn add_assign(&mut self, rhs: &Fp55) {
        *self = &*self + rhs;
    }
}

impl SubAssign<&Fp55> for Fp55 {
    fn sub_assign(&mut self, rhs: &Fp55) {
        *self = &*self - rhs;
    }
}

impl MulAssign<&Fp55> for Fp55 {
    fn mul_assign(&mut self, rhs: &Fp55) {
        *self = &*self * rhs;
    }
}

impl AddAssign for Fp55 {
    fn add_assign(&mut self, rhs: Fp55) {
        *self += &rhs;
    }
}

impl SubAssign for Fp55 {
    fn sub_assign(&mut self, rhs: Fp55) {
        *self -= &rhs;
    }
}

impl MulAssign for Fp55 {
    fn mul_assign(&mut self, rhs: Fp55) {
        *self *= &rhs;
    }
}

impl ConstantTimeEq for Fp55 {
    fn ct_eq(&self, other: &Fp55) -> Choice {
        self.to_bytes().ct_eq(&other.to_bytes())
    }
}

impl ConditionallySelectable for Fp55 {
    fn conditional_select(a: &Fp55, b: &Fp55, choice: Choice) -> Fp55 {
        let mut out = [0u64; LIMBS];
        for (o, (x, y)) in out.iter_mut().zip(a.0.iter().zip(b.0.iter())) {
            *o = u64::conditional_select(x, y, choice);
        }
        Fp55(out)
    }
}

impl Eq for Fp55 {}

impl PartialEq for Fp55 {
    fn eq(&self, other: &Fp55) -> bool {
        self.ct_eq(other).into()
    }
}
