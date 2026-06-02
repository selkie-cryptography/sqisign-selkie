//! Field arithmetic modulo [p = 5 · 2²⁴⁸ − 1][§2.1].
//!
//! Elements are represented in unsaturated or reduced-radix radix-2⁵¹ form
//! as five `u64` limbs, matching the representation used by the SQIsign
//! reference implementation.  All arithmetic is performed in [Montgomery
//! form][§8.1]. This exploits the fact that SQIsign primes have the special
//! shape p = β · 2^α − 1, with a relatively small integer β and a large
//! integer α. Primes with this characteristic are often referred to as
//! Montgomery-friendly primes, which allow efficient Montgomery reduction.
//!
//! [§2.1]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.1
//! [§8.1]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.1

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

mod arch;

#[cfg(test)]
mod tests;

/// Number of bytes in a canonical encoding of an element of F_p.
pub const FP_ENCODED_BYTES: usize = 32;

/// Radix used for limb representation: 2^51.
const RADIX: u32 = 51;

/// Mask for a single radix-51 limb.
const MASK: u64 = (1u64 << RADIX) - 1;

/// The high-limb constant: c · 2^(248 - 4·51) = 5 · 2^(248-204) = 5 · 2^44.
/// This is `p4` in the C reference: 0x500000000000.
const P4: u64 = 5u64 << 44;

/// An element of the prime field F_p, where p = 5 · 2²⁴⁸ − 1.
///
/// Internally stored in Montgomery form using unsaturated radix-2⁵¹
/// representation with 5 limbs. Limbs are allowed to be slightly
/// unreduced between operations.
#[derive(Copy, Clone)]
pub struct Fp(pub(crate) [u64; 5]);

impl Fp {
    /// Constructs from raw radix-51 limbs (already in Montgomery form).
    ///
    /// This is a const constructor for embedding precomputed constants.
    /// The caller is responsible for ensuring the limbs represent a
    /// valid Montgomery-form field element.
    pub const fn from_limbs(limbs: [u64; 5]) -> Fp {
        Fp(limbs)
    }
}

impl core::fmt::Debug for Fp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Fp({:?})", &self.0[..])
    }
}

/// R² mod p, used for converting to Montgomery form.
/// R = 2^(5·51) = 2^255.
///
/// Taken from the C reference `nres()` function.
const R2: Fp = Fp([
    0x0004CCCCCCCCCF5C,
    0x0001999999999999,
    0x0003333333333333,
    0x0006666666666666,
    0x00000CCCCCCCCCCC,
]);

impl Fp {
    /// The additive identity (zero) in Montgomery form.
    pub const ZERO: Fp = Fp([0, 0, 0, 0, 0]);

    /// The multiplicative identity (one) in Montgomery form.
    pub const ONE: Fp = Fp([
        0x0000000000000019,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000300000000000,
    ]);

    /// The constant 2 in Montgomery form.
    pub const TWO: Fp = Fp([
        0x0000000000000032,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000600000000000,
    ]);

    /// The constant 4 in Montgomery form.
    pub const FOUR: Fp = Fp([
        0x0000000000000064,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000C00000000000,
    ]);

    /// The constant `-1 mod p` in Montgomery form.
    ///
    /// Used by the `NORMALIZATION_TRANSFORMS` precomputed matrices in
    /// the (2,2)-isogeny splitter (`src/surfaces/isogeny.rs`), where
    /// a `const` definition is required.
    pub const MINUS_ONE: Fp = Fp([
        0x0007FFFFFFFFFFE5,
        0x0007FFFFFFFFFFFF,
        0x0007FFFFFFFFFFFF,
        0x0007FFFFFFFFFFFF,
        0x00006FFFFFFFFFFF,
    ]);

    /// Constructs a field element from a small integer.
    pub fn from_small(x: u32) -> Fp {
        let mut a = Fp::ZERO;
        a.0[0] = x as u64;
        a.to_montgomery()
    }

    /// Converts a field element in normal form to Montgomery form.
    fn to_montgomery(self) -> Fp {
        &self * &R2
    }

    /// Converts from Montgomery form back to normal (canonical) form.
    fn reduce_montgomery(self) -> Fp {
        let one = Fp([1, 0, 0, 0, 0]);
        let mut r = &self * &one;
        r.final_sub();
        r
    }

    /// Encodes this field element as 32 bytes, little-endian.
    pub fn to_bytes(self) -> [u8; FP_ENCODED_BYTES] {
        let c = self.reduce_montgomery();
        let mut out = [0u8; 32];

        // Pack the 5 radix-51 limbs into 32 bytes, little-endian.
        // Limb 0: bits 0..50, Limb 1: bits 51..101, etc.
        let mut acc: u128 = 0;
        let mut bits = 0u32;
        let mut pos = 0;
        for &limb in c.0.iter() {
            acc |= (limb as u128) << bits;
            bits += RADIX;
            while bits >= 8 && pos < 32 {
                out[pos] = acc as u8;
                acc >>= 8;
                bits -= 8;
                pos += 1;
            }
        }
        if pos < 32 {
            out[pos] = acc as u8;
        }

        out
    }

    /// Decodes 32 bytes (little-endian) into a field element.
    ///
    /// Returns the element in Montgomery form. The input must be a
    /// canonical encoding (i.e., the value must be less than p).
    pub fn from_bytes(bytes: &[u8; FP_ENCODED_BYTES]) -> Fp {
        // Unpack 32 bytes into 5 radix-51 limbs.
        let mut limbs = [0u64; 5];
        let mut acc: u128 = 0;
        let mut bits = 0u32;
        let mut limb_idx = 0;
        for &byte in bytes.iter() {
            acc |= (byte as u128) << bits;
            bits += 8;
            if bits >= RADIX && limb_idx < 4 {
                limbs[limb_idx] = (acc as u64) & MASK;
                acc >>= RADIX;
                bits -= RADIX;
                limb_idx += 1;
            }
        }
        limbs[limb_idx] = acc as u64;

        let mut r = Fp(limbs);
        // Canonicalize: subtract p, add back if underflow.
        r.final_sub();
        // Convert to Montgomery form.
        r = r.to_montgomery();
        r
    }

    /// Squares this field element.
    #[must_use]
    pub fn square(&self) -> Fp {
        let a = &self.0;

        let tot = (a[0] as u128) * (a[0] as u128);
        let mut t: u128 = tot;
        let v0 = (t as u64) & MASK;
        t >>= RADIX;

        let tot = ((a[0] as u128) * (a[1] as u128)) * 2;
        t += tot;
        let v1 = (t as u64) & MASK;
        t >>= RADIX;

        let mut tot = ((a[0] as u128) * (a[2] as u128)) * 2;
        tot += (a[1] as u128) * (a[1] as u128);
        t += tot;
        let v2 = (t as u64) & MASK;
        t >>= RADIX;

        let tot = ((a[0] as u128) * (a[3] as u128) + (a[1] as u128) * (a[2] as u128)) * 2;
        t += tot;
        let v3 = (t as u64) & MASK;
        t >>= RADIX;

        let mut tot = ((a[0] as u128) * (a[4] as u128) + (a[1] as u128) * (a[3] as u128)) * 2;
        tot += (a[2] as u128) * (a[2] as u128);
        t += tot;
        t += (v0 as u128) * (P4 as u128);
        let v4 = (t as u64) & MASK;
        t >>= RADIX;

        let tot = ((a[1] as u128) * (a[4] as u128) + (a[2] as u128) * (a[3] as u128)) * 2;
        t += tot;
        t += (v1 as u128) * (P4 as u128);
        let c0 = (t as u64) & MASK;
        t >>= RADIX;

        let mut tot = ((a[2] as u128) * (a[4] as u128)) * 2;
        tot += (a[3] as u128) * (a[3] as u128);
        t += tot;
        t += (v2 as u128) * (P4 as u128);
        let c1 = (t as u64) & MASK;
        t >>= RADIX;

        let tot = ((a[3] as u128) * (a[4] as u128)) * 2;
        t += tot;
        t += (v3 as u128) * (P4 as u128);
        let c2 = (t as u64) & MASK;
        t >>= RADIX;

        let tot = (a[4] as u128) * (a[4] as u128);
        t += tot;
        t += (v4 as u128) * (P4 as u128);
        let c3 = (t as u64) & MASK;
        t >>= RADIX;

        Fp([c0, c1, c2, c3, t as u64])
    }

    /// Squares this element `n` times.
    #[must_use]
    pub fn pow2k(&self, n: u32) -> Fp {
        let mut r = *self;
        for _ in 0..n {
            r = r.square();
        }
        r
    }

    /// Computes the pow_p3div4: self^((p-3)/4).
    ///
    /// This is used to derive inversions, square roots, and Legendre symbols.
    /// The addition chain is taken from the C reference implementation.
    #[must_use]
    pub(crate) fn pow_p3div4(&self) -> Fp {
        let x = *self;
        let z = x.square(); // x^2
        let t0 = &x * &z; // x^3
        let z = t0.square(); // x^6
        let z = &x * &z; // x^7
        let t1 = z.square();
        let t3 = t1.square();
        let t2 = t3.square();
        let t4 = t2.pow2k(3);
        let t2 = &t2 * &t4;
        let t4 = t2.pow2k(6);
        let t2 = &t2 * &t4;
        let t4 = t2.pow2k(2);
        let t3 = &t3 * &t4;
        let t3 = t3.pow2k(13);
        let t2 = &t2 * &t3;
        let t3 = t2.pow2k(27);
        let t2 = &t2 * &t3;
        let z = &z * &t2;
        let t2 = z.pow2k(4);
        let t1 = &t1 * &t2;
        let t0 = &t0 * &t1;
        let t1 = &t1 * &t0;
        let t0 = &t1 * &t0;
        let t2 = &t0 * &t1;
        let t0 = &t0 * &t2;
        let t1 = &t1 * &t0;
        let t1 = t1.pow2k(63);
        let t1 = &t0 * &t1;
        let t1 = t1.pow2k(64);
        let t0 = &t0 * &t1;
        let t0 = t0.pow2k(57);
        &z * &t0
    }

    /// Computes the multiplicative inverse: self^(p-2).
    #[must_use]
    pub fn invert(&self) -> Fp {
        let t = self.pow_p3div4();
        let t = t.pow2k(2);
        self * &t
    }

    /// Tests whether this element is a quadratic residue (square) in F_p.
    pub fn is_square(&self) -> Choice {
        let r = self.pow_p3div4();
        let r = r.square();
        let r = &r * self;
        r.ct_eq(&Fp::ONE) | self.ct_eq(&Fp::ZERO)
    }

    /// Computes the square root (when self is a QR).
    ///
    /// The result is only meaningful when `self.is_square()` is true.
    #[must_use]
    pub fn sqrt(&self) -> Fp {
        let y = self.pow_p3div4();
        &y * self
    }

    /// Propagate carries through the limbs. Returns a sign/borrow indicator.
    fn prop(&mut self) -> u64 {
        let mut carry = self.0[0] as i64;
        carry >>= RADIX;
        self.0[0] &= MASK;
        for i in 1..4 {
            carry += self.0[i] as i64;
            self.0[i] = (carry as u64) & MASK;
            carry >>= RADIX;
        }
        self.0[4] = self.0[4].wrapping_add(carry as u64);
        // Return a mask: 0 if positive, all-ones if negative.
        let sign = (self.0[4] >> 1) >> 62;
        sign.wrapping_neg()
    }

    /// Propagate carries and conditionally add p if negative.
    fn flatten(&mut self) {
        let carry = self.prop();
        self.0[0] = self.0[0].wrapping_sub(1u64 & carry);
        self.0[4] = self.0[4].wrapping_add(P4 & carry);
        self.prop();
    }

    /// Montgomery final subtraction: subtract p and flatten.
    fn final_sub(&mut self) {
        self.0[0] = self.0[0].wrapping_add(1);
        self.0[4] = self.0[4].wrapping_sub(P4);
        self.flatten();
    }
}

impl Fp {
    /// Returns `a1·b1 + a2·b2 mod p` with a single Montgomery reduction.
    ///
    /// Implements [Longa's interleaved sum-of-products][longa]
    /// (ePrint 2022/367, Algorithm 5; SQIsign spec [Algorithm 8.1][spec]
    /// specialised to B=1).  Both products are accumulated into one
    /// radix-2^51 column accumulator, and the `P4 = 5·2^44` Montgomery
    /// fold is folded in once — cheaper than two separate `mul`s, which
    /// would each carry their own reduction.  Mirrors C ref's
    /// `fp2_mul_c0` / `fp2_mul_c1` (`src/gf/broadwell/lvl1/fp_asm.S`),
    /// which compute one Fp² coefficient each as a single fused asm op.
    ///
    /// # Implementation
    ///
    /// The widest column (k = 4) sums at most `2*5 = 10` partial products,
    /// each `< (2^51)^2 = 2^102`, plus a reduction term `< 2^51*P4 < 2^97`
    /// and a carry `< 2^55`, so the `u128` accumulator stays under
    /// `2^107.5`.
    ///
    /// [longa]: https://eprint.iacr.org/2022/367.pdf
    /// [spec]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.1
    #[must_use]
    #[rustfmt::skip]
    pub fn sum_of_2_products(a1: &Fp, b1: &Fp, a2: &Fp, b2: &Fp) -> Fp {
        debug_assert!(a1.0.iter().all(|&x| x <= MASK), "a1 limb exceeds MASK");
        debug_assert!(b1.0.iter().all(|&x| x <= MASK), "b1 limb exceeds MASK");
        debug_assert!(a2.0.iter().all(|&x| x <= MASK), "a2 limb exceeds MASK");
        debug_assert!(b2.0.iter().all(|&x| x <= MASK), "b2 limb exceeds MASK");
        let (a, b) = (&a1.0, &b1.0);
        let (c, d) = (&a2.0, &b2.0);
        let mut t: u128 = 0;

        // Column 0
        t += (a[0] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[0] as u128);
        let v0 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 1
        t += (a[0] as u128) * (b[1] as u128);
        t += (a[1] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[1] as u128);
        t += (c[1] as u128) * (d[0] as u128);
        let v1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 2
        t += (a[0] as u128) * (b[2] as u128);
        t += (a[1] as u128) * (b[1] as u128);
        t += (a[2] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[2] as u128);
        t += (c[1] as u128) * (d[1] as u128);
        t += (c[2] as u128) * (d[0] as u128);
        let v2 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 3
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

        // Column 4 (start reduction: fold v0 * P4)
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
        t += (v0 as u128) * (P4 as u128);
        let v4 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 5 (reduce v1)
        t += (a[1] as u128) * (b[4] as u128);
        t += (a[2] as u128) * (b[3] as u128);
        t += (a[3] as u128) * (b[2] as u128);
        t += (a[4] as u128) * (b[1] as u128);
        t += (c[1] as u128) * (d[4] as u128);
        t += (c[2] as u128) * (d[3] as u128);
        t += (c[3] as u128) * (d[2] as u128);
        t += (c[4] as u128) * (d[1] as u128);
        t += (v1 as u128) * (P4 as u128);
        let c0 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 6 (reduce v2)
        t += (a[2] as u128) * (b[4] as u128);
        t += (a[3] as u128) * (b[3] as u128);
        t += (a[4] as u128) * (b[2] as u128);
        t += (c[2] as u128) * (d[4] as u128);
        t += (c[3] as u128) * (d[3] as u128);
        t += (c[4] as u128) * (d[2] as u128);
        t += (v2 as u128) * (P4 as u128);
        let c1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 7 (reduce v3)
        t += (a[3] as u128) * (b[4] as u128);
        t += (a[4] as u128) * (b[3] as u128);
        t += (c[3] as u128) * (d[4] as u128);
        t += (c[4] as u128) * (d[3] as u128);
        t += (v3 as u128) * (P4 as u128);
        let c2 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 8 (reduce v4)
        t += (a[4] as u128) * (b[4] as u128);
        t += (c[4] as u128) * (d[4] as u128);
        t += (v4 as u128) * (P4 as u128);
        let c3 = (t as u64) & MASK;
        t >>= RADIX;

        Fp([c0, c1, c2, c3, t as u64])
    }

    /// Returns `a1·b1 − a2·b2 mod p` with a single Montgomery reduction.
    ///
    /// Negates `b2` (one limb-wise pass) and defers to
    /// [`Fp::sum_of_2_products`].  Used by `Fp²::mul` to compute
    /// `c0 = a0·b0 − a1·b1` per spec Algorithm 8.1.
    #[must_use]
    pub fn difference_of_2_products(a1: &Fp, b1: &Fp, a2: &Fp, b2: &Fp) -> Fp {
        let neg_b2 = -b2;
        Fp::sum_of_2_products(a1, b1, a2, &neg_b2)
    }
}

impl<'b> Add<&'b Fp> for &Fp {
    type Output = Fp;

    /// Modular addition, reduced to less than 2p.
    fn add(self, rhs: &'b Fp) -> Fp {
        let mut n = Fp([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
        ]);
        // Subtract 2p
        n.0[0] = n.0[0].wrapping_add(2);
        n.0[4] = n.0[4].wrapping_sub(2 * P4);
        let carry = n.prop();
        // Add 2p back if underflow
        n.0[0] = n.0[0].wrapping_sub(2u64 & carry);
        n.0[4] = n.0[4].wrapping_add((2 * P4) & carry);
        n.prop();
        n
    }
}

impl<'b> Sub<&'b Fp> for &Fp {
    type Output = Fp;

    /// Modular subtraction, reduced to less than 2p.
    fn sub(self, rhs: &'b Fp) -> Fp {
        let mut n = Fp([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
            self.0[2].wrapping_sub(rhs.0[2]),
            self.0[3].wrapping_sub(rhs.0[3]),
            self.0[4].wrapping_sub(rhs.0[4]),
        ]);
        let carry = n.prop();
        n.0[0] = n.0[0].wrapping_sub(2u64 & carry);
        n.0[4] = n.0[4].wrapping_add((2 * P4) & carry);
        n.prop();
        n
    }
}

impl Neg for &Fp {
    type Output = Fp;

    fn neg(self) -> Fp {
        &Fp::ZERO - self
    }
}

impl<'b> Mul<&'b Fp> for &Fp {
    type Output = Fp;

    /// Modular multiplication (Montgomery form), reduced to less than 2p.
    ///
    /// Uses the schoolbook method with interleaved reduction, exploiting
    /// the special shape p = 5 · 2²⁴⁸ − 1.
    #[rustfmt::skip]
    fn mul(self, rhs: &'b Fp) -> Fp {
        let (a, b) = (&self.0, &rhs.0);
        let mut t: u128 = 0;

        t += (a[0] as u128) * (b[0] as u128);
        let v0 = (t as u64) & MASK;
        t >>= RADIX;

        t += (a[0] as u128) * (b[1] as u128);
        t += (a[1] as u128) * (b[0] as u128);
        let v1 = (t as u64) & MASK;
        t >>= RADIX;

        t += (a[0] as u128) * (b[2] as u128);
        t += (a[1] as u128) * (b[1] as u128);
        t += (a[2] as u128) * (b[0] as u128);
        let v2 = (t as u64) & MASK;
        t >>= RADIX;

        t += (a[0] as u128) * (b[3] as u128);
        t += (a[1] as u128) * (b[2] as u128);
        t += (a[2] as u128) * (b[1] as u128);
        t += (a[3] as u128) * (b[0] as u128);
        let v3 = (t as u64) & MASK;
        t >>= RADIX;

        t += (a[0] as u128) * (b[4] as u128);
        t += (a[1] as u128) * (b[3] as u128);
        t += (a[2] as u128) * (b[2] as u128);
        t += (a[3] as u128) * (b[1] as u128);
        t += (a[4] as u128) * (b[0] as u128);
        t += (v0 as u128) * (P4 as u128);
        let v4 = (t as u64) & MASK;
        t >>= RADIX;

        t += (a[1] as u128) * (b[4] as u128);
        t += (a[2] as u128) * (b[3] as u128);
        t += (a[3] as u128) * (b[2] as u128);
        t += (a[4] as u128) * (b[1] as u128);
        t += (v1 as u128) * (P4 as u128);
        let c0 = (t as u64) & MASK;
        t >>= RADIX;

        t += (a[2] as u128) * (b[4] as u128);
        t += (a[3] as u128) * (b[3] as u128);
        t += (a[4] as u128) * (b[2] as u128);
        t += (v2 as u128) * (P4 as u128);
        let c1 = (t as u64) & MASK;
        t >>= RADIX;

        t += (a[3] as u128) * (b[4] as u128);
        t += (a[4] as u128) * (b[3] as u128);
        t += (v3 as u128) * (P4 as u128);
        let c2 = (t as u64) & MASK;
        t >>= RADIX;

        t += (a[4] as u128) * (b[4] as u128);
        t += (v4 as u128) * (P4 as u128);
        let c3 = (t as u64) & MASK;
        t >>= RADIX;

        Fp([c0, c1, c2, c3, t as u64])
    }
}

// Convenience impls: owned variants delegate to reference impls.

impl Add<Fp> for Fp {
    type Output = Fp;
    fn add(self, rhs: Fp) -> Fp {
        &self + &rhs
    }
}

impl Sub<Fp> for Fp {
    type Output = Fp;
    fn sub(self, rhs: Fp) -> Fp {
        &self - &rhs
    }
}

impl Mul<Fp> for Fp {
    type Output = Fp;
    fn mul(self, rhs: Fp) -> Fp {
        &self * &rhs
    }
}

impl Neg for Fp {
    type Output = Fp;
    fn neg(self) -> Fp {
        -&self
    }
}

impl AddAssign<&Fp> for Fp {
    fn add_assign(&mut self, rhs: &Fp) {
        *self = &*self + rhs;
    }
}

impl SubAssign<&Fp> for Fp {
    fn sub_assign(&mut self, rhs: &Fp) {
        *self = &*self - rhs;
    }
}

impl MulAssign<&Fp> for Fp {
    fn mul_assign(&mut self, rhs: &Fp) {
        *self = &*self * rhs;
    }
}

impl AddAssign for Fp {
    fn add_assign(&mut self, rhs: Fp) {
        *self += &rhs;
    }
}

impl SubAssign for Fp {
    fn sub_assign(&mut self, rhs: Fp) {
        *self -= &rhs;
    }
}

impl MulAssign for Fp {
    fn mul_assign(&mut self, rhs: Fp) {
        *self *= &rhs;
    }
}

impl ConstantTimeEq for Fp {
    fn ct_eq(&self, other: &Fp) -> Choice {
        self.to_bytes().ct_eq(&other.to_bytes())
    }
}

impl ConditionallySelectable for Fp {
    fn conditional_select(a: &Fp, b: &Fp, choice: Choice) -> Fp {
        Fp([
            u64::conditional_select(&a.0[0], &b.0[0], choice),
            u64::conditional_select(&a.0[1], &b.0[1], choice),
            u64::conditional_select(&a.0[2], &b.0[2], choice),
            u64::conditional_select(&a.0[3], &b.0[3], choice),
            u64::conditional_select(&a.0[4], &b.0[4], choice),
        ])
    }
}

impl Eq for Fp {}

impl PartialEq for Fp {
    fn eq(&self, other: &Fp) -> bool {
        self.ct_eq(other).into()
    }
}
