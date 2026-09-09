//! Quadratic extension field F_{p²} = F_p(i), where i² = −1.
//!
//! Elements are represented as a + b·i with a, b ∈ F_p.
//! Since p ≡ 3 (mod 4), −1 is a quadratic non-residue in F_p,
//! so F_p(i) is a valid degree-2 extension.
//!
//! See [§4.2] of the SQIsign spec.
//!
//! [§4.2]: https://sqisign.org/spec/sqisign-20260901.pdf#sec:finite-fields

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use subtle::{
    Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
};

use crate::fields::fp::{FP_ENCODED_BYTES, Fp};

#[cfg(test)]
mod tests;

/// Number of bytes in a canonical encoding of an element of F_{p²}.
pub const FP2_ENCODED_BYTES: usize = 2 * FP_ENCODED_BYTES;

/// An element of the quadratic extension field F_{p²} = F_p(i).
///
/// Represented as `a + b·i` where `a` is the "real" part and `b` the
/// "imaginary" part, with i² = −1.
#[derive(Copy, Clone, Debug)]
pub struct Fp2 {
    /// Real part.
    pub(crate) a: Fp,
    /// Imaginary part.
    pub(crate) b: Fp,
}

impl Fp2 {
    /// The additive identity in F_{p²}.
    pub const ZERO: Fp2 = Fp2 {
        a: Fp::ZERO,
        b: Fp::ZERO,
    };

    /// The multiplicative identity in F_{p²}.
    pub const ONE: Fp2 = Fp2 {
        a: Fp::ONE,
        b: Fp::ZERO,
    };

    /// The element i (the generator of the extension).
    pub const I: Fp2 = Fp2 {
        a: Fp::ZERO,
        b: Fp::ONE,
    };

    /// The constant `-1 mod p` in `F_{p²}`, with imaginary part zero.
    pub const MINUS_ONE: Fp2 = Fp2 {
        a: Fp::MINUS_ONE,
        b: Fp::ZERO,
    };

    /// Constructs an element from its real and imaginary parts.
    pub const fn new(a: Fp, b: Fp) -> Fp2 {
        Fp2 { a, b }
    }

    /// Constructs from a base field element (imaginary part is zero).
    pub const fn from_fp(a: Fp) -> Fp2 {
        Fp2 { a, b: Fp::ZERO }
    }

    /// Squares this element.
    ///
    /// Uses the identity (a + bi)² = (a² − b²) + 2ab·i, optimized as:
    ///   real = (a + b)(a − b)
    ///   imag = 2ab
    #[must_use]
    #[inline]
    pub fn square(&self) -> Fp2 {
        let a_plus_b = &self.a + &self.b;
        let a_minus_b = &self.a - &self.b;
        let two_ab = &self.a * &self.b;
        Fp2 {
            a: &a_plus_b * &a_minus_b,
            b: &two_ab + &two_ab,
        }
    }

    /// Computes the norm: N(a + bi) = a² + b².
    #[must_use]
    pub fn norm(&self) -> Fp {
        &self.a.square() + &self.b.square()
    }

    /// Computes the conjugate: conj(a + bi) = a − bi.
    #[must_use]
    pub fn conjugate(&self) -> Fp2 {
        Fp2 {
            a: self.a,
            b: -&self.b,
        }
    }

    /// Computes the multiplicative inverse.
    ///
    /// (a + bi)⁻¹ = (a − bi) / (a² + b²)
    #[must_use]
    pub fn invert(&self) -> Fp2 {
        let n_inv = self.norm().invert();
        Fp2 {
            a: &self.a * &n_inv,
            b: &(-&self.b) * &n_inv,
        }
    }

    /// Tests whether this element is a square in F_{p²}.
    ///
    /// An element a ∈ F_{p²} is a square iff a^{p+1} is a square in F_p,
    /// since a^{p+1} = N(a) = a² + b².
    pub fn is_square(&self) -> Choice {
        self.norm().is_square() | self.ct_eq(&Fp2::ZERO)
    }

    /// Computes a square root in F_{p²} ([Alg. 4.35]).
    ///
    /// Given `x = x₀ + x₁·i`, returns one of the two roots `±r` with
    /// `r² = x`. Which root is returned is fixed by the algorithm (the
    /// F_p roots come from fixed exponent chains), not by a sign rule,
    /// and matches the C reference's `fp2_sqrt`. The result is only
    /// meaningful when `self.is_square()` is true.
    ///
    /// # Algorithm
    ///
    /// 1. δ ← √(x₀² + x₁²) ∈ F_p; if x₁ = 0, δ ← x₀
    /// 2. r₀ ← x₀ + δ,  t ← 2·r₀
    /// 3. r₁ ← t^{(p−3)/4}
    /// 4. r₀ ← r₀·r₁,  r₁ ← x₁·r₁
    /// 5. If (2·r₀)² = t: return r₀ + r₁·i, else return r₁ − r₀·i
    ///
    /// The step-1 select covers x₁ = 0 with x₀ a non-residue, where
    /// δ = −x₀ would zero r₀ and collapse the result.
    ///
    /// # Divergences
    ///
    /// The v2 C reference appended a parity canonicalization (negate
    /// if re(r) is odd, or re(r) = 0 and im(r) is odd). The v3 C
    /// reference and spec do not; neither does this function.
    ///
    /// # Constant-time
    ///
    /// Constant-time on `self`.
    ///
    /// [Alg. 4.35]: https://sqisign.org/spec/sqisign-20260901.pdf#algorithm.4.35
    #[must_use]
    pub fn sqrt(&self) -> Fp2 {
        let delta = self.norm().sqrt();
        let delta = Fp::conditional_select(&delta, &self.a, self.b.ct_eq(&Fp::ZERO));
        let x0 = &self.a + &delta;
        let t0 = &x0 + &x0;
        let x1 = t0.pow_p3div4();
        let x0 = &x0 * &x1;
        let x1 = &self.b * &x1;
        let t1 = (&x0 + &x0).square();

        let is_eq = t1.ct_eq(&t0);
        Fp2 {
            a: Fp::conditional_select(&x1, &x0, is_eq),
            b: Fp::conditional_select(&(-&x0), &x1, is_eq),
        }
    }

    /// Halves this element: multiplies both parts by `2^-1` ([Alg. 4.32]).
    ///
    /// [Alg. 4.32]: https://sqisign.org/spec/sqisign-20260901.pdf#algorithm.4.32
    #[must_use]
    pub fn half(&self) -> Fp2 {
        Fp2 {
            a: &self.a * &Fp::TWO_INV,
            b: &self.b * &Fp::TWO_INV,
        }
    }

    /// Inverts every element in place with one field inversion
    /// ([Alg. 4.36]).
    ///
    /// If any input is zero, every output is zero, as in the C
    /// reference; callers that need the distinction test for zero
    /// first.
    ///
    /// # Constant-time
    ///
    /// Constant-time on the element values; the slice length is public.
    ///
    /// [Alg. 4.36]: https://sqisign.org/spec/sqisign-20260901.pdf#algorithm.4.36
    pub fn batch_invert(elems: &mut [Fp2]) {
        let n = elems.len();
        if n == 0 {
            return;
        }

        // prefix[k] = elems[0] * ... * elems[k].
        let mut prefix = Vec::with_capacity(n);
        let mut acc = Fp2::ONE;
        for e in elems.iter() {
            acc = &acc * e;
            prefix.push(acc);
        }

        // Walk back: inv(elems[k]) = prefix[k - 1] * inv(prefix[k]).
        let mut inv = acc.invert();
        for k in (1..n).rev() {
            let e_inv = &prefix[k - 1] * &inv;
            inv = &inv * &elems[k];
            elems[k] = e_inv;
        }
        elems[0] = inv;
    }

    /// Encodes this element as 82 bytes (real part ‖ imaginary part).
    pub fn to_bytes(self) -> [u8; FP2_ENCODED_BYTES] {
        let mut out = [0u8; FP2_ENCODED_BYTES];
        out[..FP_ENCODED_BYTES].copy_from_slice(&self.a.to_bytes());
        out[FP_ENCODED_BYTES..].copy_from_slice(&self.b.to_bytes());
        out
    }

    /// Decodes 82 bytes into an F_{p²} element.
    pub fn from_bytes(bytes: &[u8; FP2_ENCODED_BYTES]) -> Fp2 {
        let a = Fp::from_bytes(bytes[..FP_ENCODED_BYTES].try_into().expect("slice length"));
        let b = Fp::from_bytes(bytes[FP_ENCODED_BYTES..].try_into().expect("slice length"));
        Fp2 { a, b }
    }
}

impl<'b> Add<&'b Fp2> for &Fp2 {
    type Output = Fp2;
    #[inline]
    fn add(self, rhs: &'b Fp2) -> Fp2 {
        Fp2 {
            a: &self.a + &rhs.a,
            b: &self.b + &rhs.b,
        }
    }
}

impl<'b> Sub<&'b Fp2> for &Fp2 {
    type Output = Fp2;
    #[inline]
    fn sub(self, rhs: &'b Fp2) -> Fp2 {
        Fp2 {
            a: &self.a - &rhs.a,
            b: &self.b - &rhs.b,
        }
    }
}

impl Neg for &Fp2 {
    type Output = Fp2;
    #[inline]
    fn neg(self) -> Fp2 {
        Fp2 {
            a: -&self.a,
            b: -&self.b,
        }
    }
}

impl<'b> Mul<&'b Fp2> for &Fp2 {
    type Output = Fp2;

    /// Fp² Montgomery multiplication via Longa's fused sum-of-products.
    ///
    /// For `c = (a0 + a1·i)(b0 + b1·i) = (a0·b0 − a1·b1) + (a0·b1 + a1·b0)·i`,
    /// computes each coefficient with one fused
    /// `Fp::sum_of_2_products` (resp. `Fp::difference_of_2_products`)
    /// — one Montgomery reduction per coefficient, two total.
    ///
    /// This is the SQIsign spec's
    /// [`OptimizedPartialFp2Mul`][spec] (Algorithm 8.1) lifted from
    /// the 64-bit Intel optimized path's `fp2_mul_c0` / `fp2_mul_c1`
    /// (C ref `src/gf/broadwell/lvl1/fp_asm.S`) into portable Rust.
    /// Trades the asymmetric Karatsuba 3-mul shape (3M + 5A + 3
    /// reductions) for the symmetric Algorithm 8.1 shape (2 fused
    /// mul-pairs + 2 reductions).
    ///
    /// [spec]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.1
    #[inline]
    fn mul(self, rhs: &'b Fp2) -> Fp2 {
        Fp2 {
            a: Fp::difference_of_2_products(&self.a, &rhs.a, &self.b, &rhs.b),
            b: Fp::sum_of_2_products(&self.a, &rhs.b, &self.b, &rhs.a),
        }
    }
}

/// Multiplies an F_{p²} element by a base field element.
impl<'b> Mul<&'b Fp> for &Fp2 {
    type Output = Fp2;
    #[inline]
    fn mul(self, rhs: &'b Fp) -> Fp2 {
        Fp2 {
            a: &self.a * rhs,
            b: &self.b * rhs,
        }
    }
}

// Owned convenience impls

impl Add for Fp2 {
    type Output = Fp2;
    #[inline]
    fn add(self, rhs: Fp2) -> Fp2 {
        &self + &rhs
    }
}

impl Sub for Fp2 {
    type Output = Fp2;
    #[inline]
    fn sub(self, rhs: Fp2) -> Fp2 {
        &self - &rhs
    }
}

impl Mul for Fp2 {
    type Output = Fp2;
    #[inline]
    fn mul(self, rhs: Fp2) -> Fp2 {
        &self * &rhs
    }
}

impl Neg for Fp2 {
    type Output = Fp2;
    #[inline]
    fn neg(self) -> Fp2 {
        -&self
    }
}

impl AddAssign<&Fp2> for Fp2 {
    fn add_assign(&mut self, rhs: &Fp2) {
        *self = &*self + rhs;
    }
}

impl SubAssign<&Fp2> for Fp2 {
    fn sub_assign(&mut self, rhs: &Fp2) {
        *self = &*self - rhs;
    }
}

impl MulAssign<&Fp2> for Fp2 {
    fn mul_assign(&mut self, rhs: &Fp2) {
        *self = &*self * rhs;
    }
}

impl AddAssign for Fp2 {
    fn add_assign(&mut self, rhs: Fp2) {
        *self += &rhs;
    }
}

impl SubAssign for Fp2 {
    fn sub_assign(&mut self, rhs: Fp2) {
        *self -= &rhs;
    }
}

impl MulAssign for Fp2 {
    fn mul_assign(&mut self, rhs: Fp2) {
        *self *= &rhs;
    }
}

impl ConstantTimeEq for Fp2 {
    fn ct_eq(&self, other: &Fp2) -> Choice {
        self.a.ct_eq(&other.a) & self.b.ct_eq(&other.b)
    }
}

impl ConstantTimeGreater for Fp2 {
    /// `Fp2Compare`, the total order of Definition 4.2.1 ([§4.2.1.5]):
    /// imaginary parts first, then real parts, each as canonical
    /// integers in `[0, p)`. `NormalizeMontgomery` selects the largest
    /// of the six candidate coefficients by it.
    ///
    /// Scans the canonical encoding from its top byte, which orders the
    /// imaginary half before the real half; the same scan as the C
    /// reference's `fp2_less_than`, and independent of any backend's
    /// limb layout.
    ///
    /// [§4.2.1.5]: https://sqisign.org/spec/sqisign-20260901.pdf#subsubsection.4.2.1.5
    fn ct_gt(&self, other: &Fp2) -> Choice {
        let a = self.to_bytes();
        let b = other.to_bytes();
        let mut lt = Choice::from(0);
        let mut gt = Choice::from(0);
        for (x, y) in a.iter().rev().zip(b.iter().rev()) {
            let undecided = !(lt | gt);
            lt |= undecided & x.ct_lt(y);
            gt |= undecided & x.ct_gt(y);
        }
        gt
    }
}

impl ConstantTimeLess for Fp2 {}

impl ConditionallySelectable for Fp2 {
    #[inline]
    fn conditional_select(a: &Fp2, b: &Fp2, choice: Choice) -> Fp2 {
        Fp2 {
            a: Fp::conditional_select(&a.a, &b.a, choice),
            b: Fp::conditional_select(&a.b, &b.b, choice),
        }
    }
}

impl Eq for Fp2 {}

impl PartialEq for Fp2 {
    fn eq(&self, other: &Fp2) -> bool {
        self.ct_eq(other).into()
    }
}
