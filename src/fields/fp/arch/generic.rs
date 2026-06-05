//! `Fp51` backend — radix-51 Montgomery on `[u64; 5]`.
//!
//! Cross-architecture baseline that compiles on any target.  Each limb
//! holds 51 bits in unsaturated form, exploiting `p = 5 · 2^248 − 1`'s
//! Montgomery-friendly structure with `P4 = 5 · 2^44` as the per-column
//! fold multiplier.
//!
//! Tables in `params.rs` and `deuring/precomputed.rs` use this
//! backend's limb shape as the source of truth; every other backend's
//! `from_limbs` const-converts from `[u64; 5]` radix-51 limbs.

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::fields::fp::FP_ENCODED_BYTES;

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
pub struct Fp51(pub(crate) [u64; 5]);

impl Fp51 {
    /// Constructs from raw radix-51 limbs (already in Montgomery form).
    ///
    /// This is a const constructor for embedding precomputed constants.
    /// The caller is responsible for ensuring the limbs represent a
    /// valid Montgomery-form field element.
    pub const fn from_limbs(limbs: [u64; 5]) -> Fp51 {
        Fp51(limbs)
    }
}

impl core::fmt::Debug for Fp51 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Fp51({:?})", &self.0[..])
    }
}

/// R² mod p, used for converting to Montgomery form.
/// R = 2^(5·51) = 2^255.
///
/// Taken from the C reference `nres()` function.
const R2: Fp51 = Fp51([
    0x0004CCCCCCCCCF5C,
    0x0001999999999999,
    0x0003333333333333,
    0x0006666666666666,
    0x00000CCCCCCCCCCC,
]);

impl Fp51 {
    /// The additive identity (zero) in Montgomery form.
    pub const ZERO: Fp51 = Fp51([0, 0, 0, 0, 0]);

    /// The multiplicative identity (one) in Montgomery form.
    pub const ONE: Fp51 = Fp51([
        0x0000000000000019,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000300000000000,
    ]);

    /// The constant 2 in Montgomery form.
    pub const TWO: Fp51 = Fp51([
        0x0000000000000032,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000600000000000,
    ]);

    /// The constant 4 in Montgomery form.
    pub const FOUR: Fp51 = Fp51([
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
    pub const MINUS_ONE: Fp51 = Fp51([
        0x0007FFFFFFFFFFE5,
        0x0007FFFFFFFFFFFF,
        0x0007FFFFFFFFFFFF,
        0x0007FFFFFFFFFFFF,
        0x00006FFFFFFFFFFF,
    ]);

    /// Constructs a field element from a small integer.
    pub fn from_small(x: u32) -> Fp51 {
        let mut a = Fp51::ZERO;
        a.0[0] = x as u64;
        a.to_montgomery()
    }

    /// Converts a field element in normal form to Montgomery form.
    fn to_montgomery(self) -> Fp51 {
        &self * &R2
    }

    /// Converts from Montgomery form back to normal (canonical) form.
    fn reduce_montgomery(self) -> Fp51 {
        let one = Fp51([1, 0, 0, 0, 0]);
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
    pub fn from_bytes(bytes: &[u8; FP_ENCODED_BYTES]) -> Fp51 {
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

        let mut r = Fp51(limbs);
        // Canonicalize: subtract p, add back if underflow.
        r.final_sub();
        // Convert to Montgomery form.
        r = r.to_montgomery();
        r
    }

    /// Squares this field element.
    #[must_use]
    pub fn square(&self) -> Fp51 {
        let a = &self.0;

        // Column 0.
        let tot = (a[0] as u128) * (a[0] as u128);
        let mut t: u128 = tot;
        let v0 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 1.
        let tot = ((a[0] as u128) * (a[1] as u128)) * 2;
        t += tot;
        let v1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 2.
        let mut tot = ((a[0] as u128) * (a[2] as u128)) * 2;
        tot += (a[1] as u128) * (a[1] as u128);
        t += tot;
        let v2 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 3.
        let tot = ((a[0] as u128) * (a[3] as u128) + (a[1] as u128) * (a[2] as u128)) * 2;
        t += tot;
        let v3 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 4 (start reduction: fold v0 * P4).
        let mut tot = ((a[0] as u128) * (a[4] as u128) + (a[1] as u128) * (a[3] as u128)) * 2;
        tot += (a[2] as u128) * (a[2] as u128);
        t += tot;
        t += (v0 as u128) * (P4 as u128);
        let v4 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 5 (reduce v1).
        let tot = ((a[1] as u128) * (a[4] as u128) + (a[2] as u128) * (a[3] as u128)) * 2;
        t += tot;
        t += (v1 as u128) * (P4 as u128);
        let c0 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 6 (reduce v2).
        let mut tot = ((a[2] as u128) * (a[4] as u128)) * 2;
        tot += (a[3] as u128) * (a[3] as u128);
        t += tot;
        t += (v2 as u128) * (P4 as u128);
        let c1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 7 (reduce v3).
        let tot = ((a[3] as u128) * (a[4] as u128)) * 2;
        t += tot;
        t += (v3 as u128) * (P4 as u128);
        let c2 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 8 (reduce v4).
        let tot = (a[4] as u128) * (a[4] as u128);
        t += tot;
        t += (v4 as u128) * (P4 as u128);
        let c3 = (t as u64) & MASK;
        t >>= RADIX;

        Fp51([c0, c1, c2, c3, t as u64])
    }

    /// Squares this element `n` times.
    #[must_use]
    pub fn pow2k(&self, n: u32) -> Fp51 {
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
    pub(crate) fn pow_p3div4(&self) -> Fp51 {
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
    pub fn invert(&self) -> Fp51 {
        let t = self.pow_p3div4();
        let t = t.pow2k(2);
        self * &t
    }

    /// Tests whether this element is a quadratic residue (square) in F_p.
    pub fn is_square(&self) -> Choice {
        let r = self.pow_p3div4();
        let r = r.square();
        let r = &r * self;
        r.ct_eq(&Fp51::ONE) | self.ct_eq(&Fp51::ZERO)
    }

    /// Computes the square root (when self is a QR).
    ///
    /// The result is only meaningful when `self.is_square()` is true.
    #[must_use]
    pub fn sqrt(&self) -> Fp51 {
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

impl Fp51 {
    /// Returns `a1·b1 + a2·b2 mod p` with a single Montgomery reduction.
    ///
    /// Implements [Longa's interleaved sum-of-products][longa]
    /// (ePrint 2022/367, Algorithm 5; SQIsign spec [Algorithm 8.1][spec]
    /// specialised to B=1).  Both products are accumulated into one
    /// radix-2^51 column accumulator, and the `P4 = 5·2^44` Montgomery
    /// fold is folded in once — cheaper than two separate `mul`s, which
    /// would each carry their own reduction.  Mirrors C ref's
    /// `fp2_mul_c0` / `fp2_mul_c1` (`src/gf/broadwell/lvl1/fp_asm.S`),
    /// which compute one Fp51² coefficient each as a single fused asm op.
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
    pub fn sum_of_2_products(a1: &Fp51, b1: &Fp51, a2: &Fp51, b2: &Fp51) -> Fp51 {
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

        Fp51([c0, c1, c2, c3, t as u64])
    }

    /// Returns `a1·b1 − a2·b2 mod p` with a single Montgomery reduction.
    ///
    /// Negates `b2` (one limb-wise pass) and defers to
    /// [`Fp51::sum_of_2_products`].  Used by `Fp51²::mul` to compute
    /// `c0 = a0·b0 − a1·b1` per spec Algorithm 8.1.
    #[must_use]
    pub fn difference_of_2_products(a1: &Fp51, b1: &Fp51, a2: &Fp51, b2: &Fp51) -> Fp51 {
        let neg_b2 = -b2;
        Fp51::sum_of_2_products(a1, b1, a2, &neg_b2)
    }
}

// The `sum_of_N_products` / `difference_of_N_products` family below
// is one explicit method body per supported t-value (Longa Alg 5).
// An earlier experiment folded all the bodies into a single
// `pub fn sum_of_products_generic<const N: usize>(pairs: [(&Fp51, &Fp51);
// N]) -> Fp51` over `for k in 0..9 { for &(a, b) in pairs.iter() {...}}`.
// On M4 with rustc 1.95.0 / opt-level 3 the generic measured 2.88x
// slower than the longhand at N=4 (61.61 ns vs 21.40 ns) and 1.33x
// slower than the naive 6-mul-and-add baseline at N=6 (89.59 ns vs
// 67.45 ns) -- LLVM didn't unroll the nested loops despite N being a
// compile-time constant, leaving runtime branches in the hot path
// and destroying the fused-reduction win.  The explicit bodies map
// one column per source paragraph and are auditable line-by-line;
// they pay in LOC (~200 / method) for guaranteed codegen.

impl Fp51 {
    /// Returns `a1·b1 + a2·b2 + a3·b3 + a4·b4 mod p` with a single
    /// Montgomery reduction.
    ///
    /// The t=4 extension of [`Fp51::sum_of_2_products`] -- four products
    /// share one fused-column accumulator and one `P4` reduction pass.
    /// Used by `Fp2::sum_of_2_products` to compute each coefficient of
    /// an `a·b + c·d` Fp² expression with one reduction.
    ///
    /// # Implementation
    ///
    /// Column 4 (widest) sums at most `4 * 5 = 20` partial products,
    /// each `< (2^51)^2 = 2^102`, plus a reduction term `< 2^51·P4 < 2^98`
    /// and an incoming carry `< 2^55.5`, so the `u128` accumulator stays
    /// under `2^106.5` -- comfortably within `2^128`.
    #[must_use]
    #[rustfmt::skip]
    pub fn sum_of_4_products(pairs: [(&Fp51, &Fp51); 4]) -> Fp51 {
        let [(a1, b1), (a2, b2), (a3, b3), (a4, b4)] = pairs;
        debug_assert!(a1.0.iter().all(|&x| x <= MASK), "a1 limb exceeds MASK");
        debug_assert!(b1.0.iter().all(|&x| x <= MASK), "b1 limb exceeds MASK");
        debug_assert!(a2.0.iter().all(|&x| x <= MASK), "a2 limb exceeds MASK");
        debug_assert!(b2.0.iter().all(|&x| x <= MASK), "b2 limb exceeds MASK");
        debug_assert!(a3.0.iter().all(|&x| x <= MASK), "a3 limb exceeds MASK");
        debug_assert!(b3.0.iter().all(|&x| x <= MASK), "b3 limb exceeds MASK");
        debug_assert!(a4.0.iter().all(|&x| x <= MASK), "a4 limb exceeds MASK");
        debug_assert!(b4.0.iter().all(|&x| x <= MASK), "b4 limb exceeds MASK");
        let (a, b) = (&a1.0, &b1.0);
        let (c, d) = (&a2.0, &b2.0);
        let (e, f) = (&a3.0, &b3.0);
        let (g, h) = (&a4.0, &b4.0);
        let mut t: u128 = 0;

        // Column 0
        t += (a[0] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[0] as u128);
        t += (e[0] as u128) * (f[0] as u128);
        t += (g[0] as u128) * (h[0] as u128);
        let v0 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 1
        t += (a[0] as u128) * (b[1] as u128);
        t += (a[1] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[1] as u128);
        t += (c[1] as u128) * (d[0] as u128);
        t += (e[0] as u128) * (f[1] as u128);
        t += (e[1] as u128) * (f[0] as u128);
        t += (g[0] as u128) * (h[1] as u128);
        t += (g[1] as u128) * (h[0] as u128);
        let v1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 2
        t += (a[0] as u128) * (b[2] as u128);
        t += (a[1] as u128) * (b[1] as u128);
        t += (a[2] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[2] as u128);
        t += (c[1] as u128) * (d[1] as u128);
        t += (c[2] as u128) * (d[0] as u128);
        t += (e[0] as u128) * (f[2] as u128);
        t += (e[1] as u128) * (f[1] as u128);
        t += (e[2] as u128) * (f[0] as u128);
        t += (g[0] as u128) * (h[2] as u128);
        t += (g[1] as u128) * (h[1] as u128);
        t += (g[2] as u128) * (h[0] as u128);
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
        t += (e[0] as u128) * (f[3] as u128);
        t += (e[1] as u128) * (f[2] as u128);
        t += (e[2] as u128) * (f[1] as u128);
        t += (e[3] as u128) * (f[0] as u128);
        t += (g[0] as u128) * (h[3] as u128);
        t += (g[1] as u128) * (h[2] as u128);
        t += (g[2] as u128) * (h[1] as u128);
        t += (g[3] as u128) * (h[0] as u128);
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
        t += (e[0] as u128) * (f[4] as u128);
        t += (e[1] as u128) * (f[3] as u128);
        t += (e[2] as u128) * (f[2] as u128);
        t += (e[3] as u128) * (f[1] as u128);
        t += (e[4] as u128) * (f[0] as u128);
        t += (g[0] as u128) * (h[4] as u128);
        t += (g[1] as u128) * (h[3] as u128);
        t += (g[2] as u128) * (h[2] as u128);
        t += (g[3] as u128) * (h[1] as u128);
        t += (g[4] as u128) * (h[0] as u128);
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
        t += (e[1] as u128) * (f[4] as u128);
        t += (e[2] as u128) * (f[3] as u128);
        t += (e[3] as u128) * (f[2] as u128);
        t += (e[4] as u128) * (f[1] as u128);
        t += (g[1] as u128) * (h[4] as u128);
        t += (g[2] as u128) * (h[3] as u128);
        t += (g[3] as u128) * (h[2] as u128);
        t += (g[4] as u128) * (h[1] as u128);
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
        t += (e[2] as u128) * (f[4] as u128);
        t += (e[3] as u128) * (f[3] as u128);
        t += (e[4] as u128) * (f[2] as u128);
        t += (g[2] as u128) * (h[4] as u128);
        t += (g[3] as u128) * (h[3] as u128);
        t += (g[4] as u128) * (h[2] as u128);
        t += (v2 as u128) * (P4 as u128);
        let c1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 7 (reduce v3)
        t += (a[3] as u128) * (b[4] as u128);
        t += (a[4] as u128) * (b[3] as u128);
        t += (c[3] as u128) * (d[4] as u128);
        t += (c[4] as u128) * (d[3] as u128);
        t += (e[3] as u128) * (f[4] as u128);
        t += (e[4] as u128) * (f[3] as u128);
        t += (g[3] as u128) * (h[4] as u128);
        t += (g[4] as u128) * (h[3] as u128);
        t += (v3 as u128) * (P4 as u128);
        let c2 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 8 (reduce v4)
        t += (a[4] as u128) * (b[4] as u128);
        t += (c[4] as u128) * (d[4] as u128);
        t += (e[4] as u128) * (f[4] as u128);
        t += (g[4] as u128) * (h[4] as u128);
        t += (v4 as u128) * (P4 as u128);
        let c3 = (t as u64) & MASK;
        t >>= RADIX;

        Fp51([c0, c1, c2, c3, t as u64])
    }

    /// Returns the t=4 sum-of-products with the last pair subtracted,
    /// `pairs[0]·pairs[1] + pairs[2] - pairs[3]` (each ·), with a single
    /// Montgomery reduction.
    ///
    /// Negates the last pair's `b` and defers to
    /// [`Fp51::sum_of_4_products`].  More complex sign patterns (e.g. the
    /// Fp²-Karatsuba real coefficient `a0·b0 - a1·b1 + c0·d0 - c1·d1`,
    /// which has two negations) are expressed by the caller pre-negating
    /// individual pair entries -- each costs one limb-wise pass.
    #[must_use]
    pub fn difference_of_4_products(pairs: [(&Fp51, &Fp51); 4]) -> Fp51 {
        let [(a1, b1), (a2, b2), (a3, b3), (a4, b4)] = pairs;
        let neg_b4 = -b4;
        Fp51::sum_of_4_products([(a1, b1), (a2, b2), (a3, b3), (a4, &neg_b4)])
    }

    /// Returns the sum of six base-field products with a single
    /// Montgomery reduction.
    ///
    /// The t=6 extension of [`Fp51::sum_of_2_products`].  Used by
    /// `Fp2::sum_of_3_products` to compute each coefficient of an
    /// `a*b + c*d + e*f` Fp² expression with one reduction.
    ///
    /// # Implementation
    ///
    /// Column 4 (widest) sums at most `6 * 5 = 30` partial products,
    /// each `< (2^51)^2 = 2^102`, plus a reduction term `< 2^98` and
    /// an incoming carry `< 2^56`, so the `u128` accumulator stays
    /// under `2^106.9` -- comfortably within `2^128`.
    #[must_use]
    #[rustfmt::skip]
    pub fn sum_of_6_products(pairs: [(&Fp51, &Fp51); 6]) -> Fp51 {
        let [(a1, b1), (a2, b2), (a3, b3), (a4, b4), (a5, b5), (a6, b6)] = pairs;
        debug_assert!(a1.0.iter().all(|&x| x <= MASK), "a1 limb exceeds MASK");
        debug_assert!(b1.0.iter().all(|&x| x <= MASK), "b1 limb exceeds MASK");
        debug_assert!(a2.0.iter().all(|&x| x <= MASK), "a2 limb exceeds MASK");
        debug_assert!(b2.0.iter().all(|&x| x <= MASK), "b2 limb exceeds MASK");
        debug_assert!(a3.0.iter().all(|&x| x <= MASK), "a3 limb exceeds MASK");
        debug_assert!(b3.0.iter().all(|&x| x <= MASK), "b3 limb exceeds MASK");
        debug_assert!(a4.0.iter().all(|&x| x <= MASK), "a4 limb exceeds MASK");
        debug_assert!(b4.0.iter().all(|&x| x <= MASK), "b4 limb exceeds MASK");
        debug_assert!(a5.0.iter().all(|&x| x <= MASK), "a5 limb exceeds MASK");
        debug_assert!(b5.0.iter().all(|&x| x <= MASK), "b5 limb exceeds MASK");
        debug_assert!(a6.0.iter().all(|&x| x <= MASK), "a6 limb exceeds MASK");
        debug_assert!(b6.0.iter().all(|&x| x <= MASK), "b6 limb exceeds MASK");
        let (a, b) = (&a1.0, &b1.0);
        let (c, d) = (&a2.0, &b2.0);
        let (e, f) = (&a3.0, &b3.0);
        let (g, h) = (&a4.0, &b4.0);
        let (i, j) = (&a5.0, &b5.0);
        let (k, l) = (&a6.0, &b6.0);
        let mut t: u128 = 0;

        // Column 0
        t += (a[0] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[0] as u128);
        t += (e[0] as u128) * (f[0] as u128);
        t += (g[0] as u128) * (h[0] as u128);
        t += (i[0] as u128) * (j[0] as u128);
        t += (k[0] as u128) * (l[0] as u128);
        let v0 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 1
        t += (a[0] as u128) * (b[1] as u128);
        t += (a[1] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[1] as u128);
        t += (c[1] as u128) * (d[0] as u128);
        t += (e[0] as u128) * (f[1] as u128);
        t += (e[1] as u128) * (f[0] as u128);
        t += (g[0] as u128) * (h[1] as u128);
        t += (g[1] as u128) * (h[0] as u128);
        t += (i[0] as u128) * (j[1] as u128);
        t += (i[1] as u128) * (j[0] as u128);
        t += (k[0] as u128) * (l[1] as u128);
        t += (k[1] as u128) * (l[0] as u128);
        let v1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 2
        t += (a[0] as u128) * (b[2] as u128);
        t += (a[1] as u128) * (b[1] as u128);
        t += (a[2] as u128) * (b[0] as u128);
        t += (c[0] as u128) * (d[2] as u128);
        t += (c[1] as u128) * (d[1] as u128);
        t += (c[2] as u128) * (d[0] as u128);
        t += (e[0] as u128) * (f[2] as u128);
        t += (e[1] as u128) * (f[1] as u128);
        t += (e[2] as u128) * (f[0] as u128);
        t += (g[0] as u128) * (h[2] as u128);
        t += (g[1] as u128) * (h[1] as u128);
        t += (g[2] as u128) * (h[0] as u128);
        t += (i[0] as u128) * (j[2] as u128);
        t += (i[1] as u128) * (j[1] as u128);
        t += (i[2] as u128) * (j[0] as u128);
        t += (k[0] as u128) * (l[2] as u128);
        t += (k[1] as u128) * (l[1] as u128);
        t += (k[2] as u128) * (l[0] as u128);
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
        t += (e[0] as u128) * (f[3] as u128);
        t += (e[1] as u128) * (f[2] as u128);
        t += (e[2] as u128) * (f[1] as u128);
        t += (e[3] as u128) * (f[0] as u128);
        t += (g[0] as u128) * (h[3] as u128);
        t += (g[1] as u128) * (h[2] as u128);
        t += (g[2] as u128) * (h[1] as u128);
        t += (g[3] as u128) * (h[0] as u128);
        t += (i[0] as u128) * (j[3] as u128);
        t += (i[1] as u128) * (j[2] as u128);
        t += (i[2] as u128) * (j[1] as u128);
        t += (i[3] as u128) * (j[0] as u128);
        t += (k[0] as u128) * (l[3] as u128);
        t += (k[1] as u128) * (l[2] as u128);
        t += (k[2] as u128) * (l[1] as u128);
        t += (k[3] as u128) * (l[0] as u128);
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
        t += (e[0] as u128) * (f[4] as u128);
        t += (e[1] as u128) * (f[3] as u128);
        t += (e[2] as u128) * (f[2] as u128);
        t += (e[3] as u128) * (f[1] as u128);
        t += (e[4] as u128) * (f[0] as u128);
        t += (g[0] as u128) * (h[4] as u128);
        t += (g[1] as u128) * (h[3] as u128);
        t += (g[2] as u128) * (h[2] as u128);
        t += (g[3] as u128) * (h[1] as u128);
        t += (g[4] as u128) * (h[0] as u128);
        t += (i[0] as u128) * (j[4] as u128);
        t += (i[1] as u128) * (j[3] as u128);
        t += (i[2] as u128) * (j[2] as u128);
        t += (i[3] as u128) * (j[1] as u128);
        t += (i[4] as u128) * (j[0] as u128);
        t += (k[0] as u128) * (l[4] as u128);
        t += (k[1] as u128) * (l[3] as u128);
        t += (k[2] as u128) * (l[2] as u128);
        t += (k[3] as u128) * (l[1] as u128);
        t += (k[4] as u128) * (l[0] as u128);
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
        t += (e[1] as u128) * (f[4] as u128);
        t += (e[2] as u128) * (f[3] as u128);
        t += (e[3] as u128) * (f[2] as u128);
        t += (e[4] as u128) * (f[1] as u128);
        t += (g[1] as u128) * (h[4] as u128);
        t += (g[2] as u128) * (h[3] as u128);
        t += (g[3] as u128) * (h[2] as u128);
        t += (g[4] as u128) * (h[1] as u128);
        t += (i[1] as u128) * (j[4] as u128);
        t += (i[2] as u128) * (j[3] as u128);
        t += (i[3] as u128) * (j[2] as u128);
        t += (i[4] as u128) * (j[1] as u128);
        t += (k[1] as u128) * (l[4] as u128);
        t += (k[2] as u128) * (l[3] as u128);
        t += (k[3] as u128) * (l[2] as u128);
        t += (k[4] as u128) * (l[1] as u128);
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
        t += (e[2] as u128) * (f[4] as u128);
        t += (e[3] as u128) * (f[3] as u128);
        t += (e[4] as u128) * (f[2] as u128);
        t += (g[2] as u128) * (h[4] as u128);
        t += (g[3] as u128) * (h[3] as u128);
        t += (g[4] as u128) * (h[2] as u128);
        t += (i[2] as u128) * (j[4] as u128);
        t += (i[3] as u128) * (j[3] as u128);
        t += (i[4] as u128) * (j[2] as u128);
        t += (k[2] as u128) * (l[4] as u128);
        t += (k[3] as u128) * (l[3] as u128);
        t += (k[4] as u128) * (l[2] as u128);
        t += (v2 as u128) * (P4 as u128);
        let c1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 7 (reduce v3)
        t += (a[3] as u128) * (b[4] as u128);
        t += (a[4] as u128) * (b[3] as u128);
        t += (c[3] as u128) * (d[4] as u128);
        t += (c[4] as u128) * (d[3] as u128);
        t += (e[3] as u128) * (f[4] as u128);
        t += (e[4] as u128) * (f[3] as u128);
        t += (g[3] as u128) * (h[4] as u128);
        t += (g[4] as u128) * (h[3] as u128);
        t += (i[3] as u128) * (j[4] as u128);
        t += (i[4] as u128) * (j[3] as u128);
        t += (k[3] as u128) * (l[4] as u128);
        t += (k[4] as u128) * (l[3] as u128);
        t += (v3 as u128) * (P4 as u128);
        let c2 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 8 (reduce v4)
        t += (a[4] as u128) * (b[4] as u128);
        t += (c[4] as u128) * (d[4] as u128);
        t += (e[4] as u128) * (f[4] as u128);
        t += (g[4] as u128) * (h[4] as u128);
        t += (i[4] as u128) * (j[4] as u128);
        t += (k[4] as u128) * (l[4] as u128);
        t += (v4 as u128) * (P4 as u128);
        let c3 = (t as u64) & MASK;
        t >>= RADIX;

        Fp51([c0, c1, c2, c3, t as u64])
    }

    /// Returns the t=6 sum-of-products with the last pair subtracted.
    ///
    /// Negates the last pair's `b` and defers to
    /// [`Fp51::sum_of_6_products`].  Multi-negation patterns (e.g. the
    /// `Fp²::sum_of_3_products` real coefficient with three subtracted
    /// terms) are expressed by the caller pre-negating individual pair
    /// entries -- each costs one limb-wise pass.
    #[must_use]
    pub fn difference_of_6_products(pairs: [(&Fp51, &Fp51); 6]) -> Fp51 {
        let [p1, p2, p3, p4, p5, (a6, b6)] = pairs;
        let neg_b6 = -b6;
        Fp51::sum_of_6_products([p1, p2, p3, p4, p5, (a6, &neg_b6)])
    }
}

impl<'b> Add<&'b Fp51> for &Fp51 {
    type Output = Fp51;

    /// Modular addition, reduced to less than 2p.
    fn add(self, rhs: &'b Fp51) -> Fp51 {
        let mut n = Fp51([
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

impl<'b> Sub<&'b Fp51> for &Fp51 {
    type Output = Fp51;

    /// Modular subtraction, reduced to less than 2p.
    fn sub(self, rhs: &'b Fp51) -> Fp51 {
        let mut n = Fp51([
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

impl Neg for &Fp51 {
    type Output = Fp51;

    fn neg(self) -> Fp51 {
        &Fp51::ZERO - self
    }
}

impl<'b> Mul<&'b Fp51> for &Fp51 {
    type Output = Fp51;

    /// Modular multiplication (Montgomery form), reduced to less than 2p.
    ///
    /// Uses the schoolbook method with interleaved reduction, exploiting
    /// the special shape p = 5 · 2²⁴⁸ − 1.
    #[rustfmt::skip]
    fn mul(self, rhs: &'b Fp51) -> Fp51 {
        let (a, b) = (&self.0, &rhs.0);
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

        // Column 4 (start reduction: fold v0 * P4).
        t += (a[0] as u128) * (b[4] as u128);
        t += (a[1] as u128) * (b[3] as u128);
        t += (a[2] as u128) * (b[2] as u128);
        t += (a[3] as u128) * (b[1] as u128);
        t += (a[4] as u128) * (b[0] as u128);
        t += (v0 as u128) * (P4 as u128);
        let v4 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 5 (reduce v1).
        t += (a[1] as u128) * (b[4] as u128);
        t += (a[2] as u128) * (b[3] as u128);
        t += (a[3] as u128) * (b[2] as u128);
        t += (a[4] as u128) * (b[1] as u128);
        t += (v1 as u128) * (P4 as u128);
        let c0 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 6 (reduce v2).
        t += (a[2] as u128) * (b[4] as u128);
        t += (a[3] as u128) * (b[3] as u128);
        t += (a[4] as u128) * (b[2] as u128);
        t += (v2 as u128) * (P4 as u128);
        let c1 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 7 (reduce v3).
        t += (a[3] as u128) * (b[4] as u128);
        t += (a[4] as u128) * (b[3] as u128);
        t += (v3 as u128) * (P4 as u128);
        let c2 = (t as u64) & MASK;
        t >>= RADIX;

        // Column 8 (reduce v4).
        t += (a[4] as u128) * (b[4] as u128);
        t += (v4 as u128) * (P4 as u128);
        let c3 = (t as u64) & MASK;
        t >>= RADIX;

        Fp51([c0, c1, c2, c3, t as u64])
    }
}

// Convenience impls: owned variants delegate to reference impls.

impl Add<Fp51> for Fp51 {
    type Output = Fp51;
    fn add(self, rhs: Fp51) -> Fp51 {
        &self + &rhs
    }
}

impl Sub<Fp51> for Fp51 {
    type Output = Fp51;
    fn sub(self, rhs: Fp51) -> Fp51 {
        &self - &rhs
    }
}

impl Mul<Fp51> for Fp51 {
    type Output = Fp51;
    fn mul(self, rhs: Fp51) -> Fp51 {
        &self * &rhs
    }
}

impl Neg for Fp51 {
    type Output = Fp51;
    fn neg(self) -> Fp51 {
        -&self
    }
}

impl AddAssign<&Fp51> for Fp51 {
    fn add_assign(&mut self, rhs: &Fp51) {
        *self = &*self + rhs;
    }
}

impl SubAssign<&Fp51> for Fp51 {
    fn sub_assign(&mut self, rhs: &Fp51) {
        *self = &*self - rhs;
    }
}

impl MulAssign<&Fp51> for Fp51 {
    fn mul_assign(&mut self, rhs: &Fp51) {
        *self = &*self * rhs;
    }
}

impl AddAssign for Fp51 {
    fn add_assign(&mut self, rhs: Fp51) {
        *self += &rhs;
    }
}

impl SubAssign for Fp51 {
    fn sub_assign(&mut self, rhs: Fp51) {
        *self -= &rhs;
    }
}

impl MulAssign for Fp51 {
    fn mul_assign(&mut self, rhs: Fp51) {
        *self *= &rhs;
    }
}

impl ConstantTimeEq for Fp51 {
    fn ct_eq(&self, other: &Fp51) -> Choice {
        self.to_bytes().ct_eq(&other.to_bytes())
    }
}

impl ConditionallySelectable for Fp51 {
    fn conditional_select(a: &Fp51, b: &Fp51, choice: Choice) -> Fp51 {
        Fp51([
            u64::conditional_select(&a.0[0], &b.0[0], choice),
            u64::conditional_select(&a.0[1], &b.0[1], choice),
            u64::conditional_select(&a.0[2], &b.0[2], choice),
            u64::conditional_select(&a.0[3], &b.0[3], choice),
            u64::conditional_select(&a.0[4], &b.0[4], choice),
        ])
    }
}

impl Eq for Fp51 {}

impl PartialEq for Fp51 {
    fn eq(&self, other: &Fp51) -> bool {
        self.ct_eq(other).into()
    }
}
