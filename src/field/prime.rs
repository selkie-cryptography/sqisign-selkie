//! Prime order finite field arithmetic of the form p = 2^a·3^b +/- 1
//!
//! Uses optimized Comba (aka product scanning form-based Montgomery reduction
//! for primes of the shape p = 2^a·3^b−1, from

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use subtle::{Choice, ConstantTimeEq};

use zeroize::DefaultIsZeroes;

use crate::field::*;

// /// The number of 64-bit limbs needed to represent a 434-bit
// /// [`PrimeFieldElement`].
// const LIMBS: usize = 7;

// /// p434 = 2^216·3^137-1
// ///
// /// Encoded with the least significant octet (and digit) located at the
// leftmost /// position (i.e., little endian format).
// ///
// /// Extracted from [PQ-Crypto].
// ///
// /// [PQCrypto-SIDH]: https://github.com/microsoft/PQCrypto-SIDH/blob/effa607f244768cdd38f930887076373604eaa78/src/P434/P434.c#L24
// const MODULUS: [u64; LIMBS] = [
//     0xffffffffffffffff,
//     0xffffffffffffffff,
//     0xffffffffffffffff,
//     0xfdc1767ae2ffffff,
//     0x7bc65c783158aea3,
//     0x6cfc5fd681c52056,
//     0x0002341f27177344,
// ];

// /// p434 * 2
// ///
// /// Encoded with the least significant octet (and digit) located at the
// leftmost /// position (i.e., little endian format).
// ///
// /// Extracted from [PQ-Crypto].
// ///
// /// [PQCrypto-SIDH]: https://github.com/microsoft/PQCrypto-SIDH/blob/effa607f244768cdd38f930887076373604eaa78/src/P434/P434.c#L26
// const MODULUS_X_2: [u64; LIMBS] = [
//     0xfffffffffffffffe,
//     0xffffffffffffffff,
//     0xffffffffffffffff,
//     0xfb82ecf5c5ffffff,
//     0xf78cb8f062b15d47,
//     0xd9f8bfad038a40ac,
//     0x0004683e4e2ee688,
// ];

// /// p434 + 1 := 2^216·3^137
// ///
// /// Encoded with the least significant octet (and digit) located at the
// leftmost /// position (i.e., little endian format).
// ///
// /// Extracted from [PQ-Crypto].
// ///
// /// [PQCrypto-SIDH]: https://github.com/microsoft/PQCrypto-SIDH/blob/effa607f244768cdd38f930887076373604eaa78/src/P434/P434.c#L30
// const MODULUS_PLUS_1: [u64; LIMBS] = [
//     0x0000000000000000,
//     0x0000000000000000,
//     0x0000000000000000,
//     0xfdc1767ae3000000,
//     0x7bc65c783158aea3,
//     0x6cfc5fd681c52056,
//     0x0002341f27177344,
// ];

/// Large finite field elements are represented as an array of smaller CPU
/// word-size integers called "limbs", generic over the word type `W`.
///
/// Instances of `W` are expected to be primitives like `u32`, `u64`, etc.
#[derive(Copy, Clone, Debug, Default, Hash)]
pub struct Limb<W>(pub W);

impl<W> Limb<W> {
    /// The value `0`.
    pub const ZERO: Self = Limb(0);

    /// The value `1`.
    pub const ONE: Self = Limb(1);

    /// Maximum value this [`Limb`] can express.
    pub const MAX: Self = Limb(W::MAX);

    /// Size of the inner integer in bits.
    pub const BITS: usize = W::BITS;

    /// Size of the inner integ er in bytes.
    pub const BYTES: usize = Self::BITS / 8;

    /// Wrapper around the underlying `W::wrapping_sub()`
    pub fn wrapping_sub(self, other: Self) -> Self {
        Self(self.0.wrapping_sub(other.0))
    }
}

/// A `Limb` defined over 32-bit words (`u32`).
pub type Limb32 = Limb<u32>;

/// A `Limb` defined over 64-bit words (`u64`).
pub type Limb64 = Limb<u64>;

/// The modulus used to define a finite field.
///
/// Large finite field elements are represented as an array of smaller CPU
/// word-size integers called "limbs". This trait defines the representation of
/// the modulus of the field as its words, limb schedule, and related consts
/// that are useful for doing field arithmetic.
///
/// Note bene: this trait does not enforce that the value of `Modulus::MODULUS`
/// is prime, yet.
// TODO(dconnolly): compute LIMBS (and LIMBS_X_2, etc) from MODULUS /
// Limb::BYTES
pub trait Modulus {
    type Limb;

    const LIMBS: usize;

    const LIMBS_X_2: usize = Self::LIMBS * 2;

    const MODULUS: [Self::Limb; Self::LIMBS];

    const MODULUS_X_2: [Self::Limb; Self::LIMBS];

    const MODULUS_PLUS_1: [Self::Limb; Self::LIMBS];

    fn num_bytes() -> usize {
        let bits = Self::Limb::BITS * Self::LIMBS;
        bits / 8
    }
}

/// Element of 𝔽ₚ, where p is a prime modulus
#[derive(Copy, Clone, Default)]
pub struct PrimeFieldElement<M: Modulus>(pub(crate) [M::Limb; M::LIMBS]);

impl<M> PrimeFieldElement<M>
where
    M: Modulus,
{
    /// The value `0` in this field.
    pub const ZERO: Self = Default::default();

    /// The value `1` in this field.
    pub const ONE: Self = {
        let mut limbs = Self::ZERO;
        limbs.0[0] = M::Limb::ONE;
        limbs
    };

    // TODO
    pub fn to_bytes(&self) -> [u8; M::num_bytes()] {
        unimplemented!()
    }

    // TODO
    pub fn from_bytes(_bytes: &[u8; M::num_bytes()]) -> Self {
        unimplemented!()
    }

    /// Optimized Comba-based Montgomery reduction for p = 2^216·3^137-1
    ///
    /// Reduces a double-wide set of limbs (like you may get after a
    /// multiplication) by p, using Montgomery reduction in product scanning
    /// form (also known as Comba), tailored for efficiency modulo primes of
    /// the shape 2^a·3^b - 1.
    ///
    /// The `- 1` component of p simplifies the computation to rounding down
    /// without it. Also, for Montgomery reduction, where `p' = -p^ - 1 mod
    /// 2^448`, `p' - 1` contains multiple limbs of value 0. In total, the
    /// cost of computing the Montgomery residue `c = a·R^-1 mod p` where `R
    /// = 2^448` is `s(s-⌊216/w⌋)` multiplications for a word-size `w`. With `w
    /// = 64` and `s = 7` we get 28 mul's, vs the `s^2 + s` or 56 mul's for
    /// regular Montgomery reduction of a `2s`-limb input.
    ///
    /// See Algorithm 1 from ["Efficient algorithms for supersingular isogeny
    /// Diffie-Hellman"](https://eprint.iacr.org/2016/413.pdf)
    // TODO(dconnolly): operate over Limbs not Words
    #[inline]
    fn reduce(a: [M::Limb; M::LIMBS * 2]) -> Self {
        let mut c: [M::Limb; M::LIMBS] = Default::default();

        // z = ⌊e_A / w⌋, so z = ⌊216 / 64⌋ = 3
        let mut z = 3;

        let mut t: M::Word = Default::default();
        let mut u: M::Word = Default::default();
        let mut v: M::Word = Default::default();

        for i in 0..M::LIMBS {
            let mut carry: M::Limb;

            for j in 0..i {
                // XXX if this being weird, try (i.wrapping_sub(z) + 1)
                if j < ((i + 1).wrapping_sub(z)) {
                    // (t, u, v) = cj × pˆi−j + (t, u, v)
                    let cp = mul(c[j], M::MODULUS_PLUS_1[i - j]);
                    (carry, v) = addc(cp.0, v, 0);
                    (carry, u) = addc(cp.1, u, carry);
                    t += carry;
                }
            }

            (carry, v) = addc(v, a[i], 0);
            (carry, u) = addc(u, 0, carry);
            t += carry;

            c[i] = v;

            (v, u, t) = (u, t, 0);
        }

        for i in M::LIMBS..(2 * M::LIMBS - 1) {
            if z > 0 {
                z -= 1;
            }

            let mut carry: M::Limb;

            for j in (i - M::LIMBS + 1)..M::LIMBS {
                if j < (M::LIMBS - z) {
                    // (t, u, v) = cj × pˆi−j + (t, u, v)
                    let cp = mul(c[j], M::MODULUS_PLUS_1[i - j]);
                    (carry, v) = addc(cp.0, v, 0);
                    (carry, u) = addc(cp.1, u, carry);
                    t += carry;
                }
            }

            (carry, v) = addc(v, a[i], 0);
            (carry, u) = addc(u, 0, carry);
            t += carry;

            c[i - M::LIMBS] = v;

            (v, u, t) = (u, t, 0);
        }

        (_, c[M::LIMBS - 1]) = addc(v, a[2 * M::LIMBS - 1], 0);

        Self(c)
    }
}

impl Add for PrimeFieldElement {
    type Output = PrimeFieldElement;

    fn add(self, rhs: PrimeFieldElement) -> PrimeFieldElement {
        &self + &rhs
    }
}

impl Add<&PrimeFieldElement> for PrimeFieldElement {
    type Output = PrimeFieldElement;

    fn add(self, rhs: &PrimeFieldElement) -> PrimeFieldElement {
        &self + rhs
    }
}

impl<'a, 'b, M> Add<&'b PrimeFieldElement<M>> for &'a PrimeFieldElement<M>
where
    M: Modulus,
{
    type Output = PrimeFieldElement<M>;

    fn add(self, rhs: &'b PrimeFieldElement<M>) -> PrimeFieldElement<M> {
        //let mut out: [M::Word; M::LIMBS] = Default::default();
        let mut out = Self::ZERO; // Self::zero();

        let mut carry = M::Lib::ZERO;
        for i in 0..M::LIMBS {
            (carry, out[i]) = addc(self.0[i], rhs.0[i], carry);
        }

        carry = 0;
        for i in 0..M::LIMBS {
            (carry, out[i]) = subc(out[i], M::MODULUS_X_2[i], carry);
        }

        // let mask: M::Word = Default::default();
        let mask = M::Limb::ZERO;
        mask.wrapping_sub(carry);
        for i in 0..M::LIMBS {
            (carry, out[i]) = addc(out[i], M::MODULUS_X_2[i] & mask, carry);
        }

        PrimeFieldElement(out)
    }
}

impl AddAssign for PrimeFieldElement {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs
    }
}

impl<'a> AddAssign<&'a PrimeFieldElement> for PrimeFieldElement {
    fn add_assign(&mut self, rhs: &'a Self) {
        *self = *self + *rhs
    }
}

impl ConstantTimeEq for PrimeFieldElement {
    /// Check whether two `PrimeFieldElement`s are equal, runtime independent of
    /// the value of the value.
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

// impl Debug for PrimeFieldElement {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         f.debug_tuple("PrimeFieldElement")
//             .field(&hex::encode(&self.to_bytes()))
//             .finish()
//     }
// }

impl DefaultIsZeroes for PrimeFieldElement {}

impl Eq for PrimeFieldElement {}

impl FiniteField for PrimeFieldElement {
    fn zero() -> Self {
        Self::default()
    }

    fn one() -> Self {
        // TODO
        unimplemented!()
    }

    fn double(&self) -> Self {
        self + self
    }

    fn square(&self) -> Self {
        self * self
    }

    fn invert(&self) -> Self {
        // TODO
        unimplemented!()
    }
}

impl Mul for PrimeFieldElement {
    type Output = PrimeFieldElement;

    fn mul(self, rhs: Self) -> PrimeFieldElement {
        &self * &rhs
    }
}

impl<'a> Mul<&'a PrimeFieldElement> for PrimeFieldElement {
    type Output = PrimeFieldElement;

    fn mul(self, rhs: &'a Self) -> PrimeFieldElement {
        &self * rhs
    }
}

impl<'a, 'b, M> Mul<&'b PrimeFieldElement<M>> for &'a PrimeFieldElement<M>
where
    M: Modulus,
{
    type Output = PrimeFieldElement<M>;

    // Schoolbook multiply
    //
    // TODO: Karatsuba?
    fn mul(self, rhs: &'b PrimeFieldElement<M>) -> PrimeFieldElement<M> {
        // const LIMBS_1: usize = M::LIMBS - 1;

        let LIMBS_1: usize = M::LIMBS - 1;

        let mut w = [0u64; M::LIMBS * 2];

        let mut carry: u64;
        for i in 0..M::LIMBS {
            (w[i], carry) = mac(w[i], self.0[i], rhs.0[0], 0);

            for j in 1..LIMBS_1 {
                (w[i + j], carry) = mac(w[i], self.0[i], rhs.0[j], carry);
            }
            (w[i + LIMBS_1], w[i + M::LIMBS]) =
                mac(w[i + LIMBS_1], self.0[i], rhs.0[LIMBS_1], carry);
        }

        // Reduce!
        PrimeFieldElement::reduce(w)
    }
}

impl MulAssign for PrimeFieldElement {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for PrimeFieldElement {
    type Output = PrimeFieldElement;

    fn neg(self) -> PrimeFieldElement {
        &PrimeFieldElement::zero() - &self
    }
}

impl PartialEq for PrimeFieldElement {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl Sub for PrimeFieldElement {
    type Output = PrimeFieldElement;

    fn sub(self, rhs: PrimeFieldElement) -> PrimeFieldElement {
        &self - &rhs
    }
}

impl Sub<&PrimeFieldElement> for PrimeFieldElement {
    type Output = PrimeFieldElement;

    fn sub(self, rhs: &PrimeFieldElement) -> PrimeFieldElement {
        &self - rhs
    }
}

impl<M> Sub<&PrimeFieldElement<M>> for &PrimeFieldElement<M>
where
    M: Modulus,
{
    type Output = PrimeFieldElement;

    fn sub(self, rhs: &PrimeFieldElement) -> PrimeFieldElement {
        let mut out: [M::Word; M::LIMBS] = Default::default();

        let mut borrow = 0;
        for i in 0..M::LIMBS {
            (borrow, out[i]) = subc(self.0[i], rhs.0[i], borrow);
        }

        let mask = 0u64.wrapping_sub(borrow);

        borrow = 0;
        for i in 0..M::LIMBS {
            (borrow, out[i]) = addc(out[i], M::MODULUS_X_2[i] & mask, borrow);
        }

        PrimeFieldElement(out)
    }
}

impl SubAssign for PrimeFieldElement {
    fn sub_assign(&mut self, rhs: PrimeFieldElement) {
        *self -= &rhs
    }
}

impl SubAssign<&PrimeFieldElement> for PrimeFieldElement {
    fn sub_assign(&mut self, rhs: &PrimeFieldElement) {
        *self = *self - rhs
    }
}
