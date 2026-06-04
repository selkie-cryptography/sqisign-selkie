//! Quadratic extension field F_{p²} = F_p(i), where i² = −1.
//!
//! Elements are represented as a + b·i with a, b ∈ F_p.
//! Since p ≡ 3 (mod 4), −1 is a quadratic non-residue in F_p,
//! so F_p(i) is a valid degree-2 extension.
//!
//! See [§2.1.2] of the SQIsign spec.
//!
//! [§2.1.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.1.2

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

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
    pub fn square(&self) -> Fp2 {
        let a_plus_b = &self.a + &self.b;
        let a_minus_b = &self.a - &self.b;
        let two_ab = &self.a * &self.b;
        Fp2 {
            a: &a_plus_b * &a_minus_b,
            b: &two_ab + &two_ab,
        }
    }

    /// Returns `a*b + c*d` in F_{p²} with two fused Montgomery reductions.
    ///
    /// Each Fp² coefficient is one t=4 fused
    /// [`Fp::sum_of_products_4`] over the four base-field products that
    /// make it up — total two Mont reductions, vs four reductions for
    /// two independent `Fp²::mul` operations followed by an add (each
    /// `mul` is itself an Algorithm 8.1 t=2 fused pair).
    ///
    /// Expands the imaginary unit identity `i² = -1`:
    /// `a*b + c*d = (a0·b0 − a1·b1 + c0·d0 − c1·d1)
    ///             + (a0·b1 + a1·b0 + c0·d1 + c1·d0)·i`
    /// where `a = a0 + a1·i`, etc.  The real coefficient's two
    /// subtractions are handled by pre-negating `b.b` and `d.b` (one
    /// limb-wise pass each) before the fused call.
    #[must_use]
    pub fn sum_of_products(a: &Fp2, b: &Fp2, c: &Fp2, d: &Fp2) -> Fp2 {
        let neg_b_im = -&b.b;
        let neg_d_im = -&d.b;
        Fp2 {
            a: Fp::sum_of_products_4([
                (&a.a, &b.a),
                (&a.b, &neg_b_im),
                (&c.a, &d.a),
                (&c.b, &neg_d_im),
            ]),
            b: Fp::sum_of_products_4([(&a.a, &b.b), (&a.b, &b.a), (&c.a, &d.b), (&c.b, &d.a)]),
        }
    }

    /// Returns `a*b - c*d` in F_{p²} with two fused Montgomery reductions.
    ///
    /// Negates `d` (two limb-wise passes) and defers to
    /// [`Fp2::sum_of_products`], since `a*b - c*d = a*b + c*(-d)`.
    #[must_use]
    pub fn difference_of_products(a: &Fp2, b: &Fp2, c: &Fp2, d: &Fp2) -> Fp2 {
        let neg_d = -d;
        Fp2::sum_of_products(a, b, c, &neg_d)
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

    /// Computes the square root in F_{p²}.
    ///
    /// Uses the algorithm from [SQIsign spec, §2.1.2, Algorithm 8.2].
    /// The result is only meaningful when `self.is_square()` is true.
    #[must_use]
    /// Computes a canonical square root in F_{p²}.
    ///
    /// Given `a = a₀ + a₁·i`, returns the unique `r` such that `r² = a`
    /// and `r` is _even_: its real part is even (as an integer in
    /// `[0, p)`), or if the real part is zero, its imaginary part is
    /// even. The other root is `−r`, which is odd.
    ///
    /// # Why canonicalization matters
    ///
    /// Every nonzero square in F_{p²} has two roots, `r` and `−r`.
    /// Without a canonical choice, any function that branches on a
    /// square root — such as `projective_difference` (Proposition 3
    /// of [Costello–Hisil–Renes 2017][CHR17]) — becomes
    /// nondeterministic: the two roots yield two distinct projective
    /// points that are NOT projectively equivalent.
    ///
    /// For SQIsign verification the consequences cascade: a wrong
    /// root in `projective_difference` produces the wrong third point
    /// of the torsion basis `(P, P−Q, Q)`, which makes `scalar_mul_add`
    /// compute the wrong challenge kernel, which makes the challenge
    /// isogeny land on the wrong curve, making every subsequent step
    /// diverge.
    ///
    /// The canonical-even convention matches the C reference's
    /// `fp2_sqrt` ([Aardal et al. 2024][ABCDK24], §3.1).
    ///
    /// # Algorithm
    ///
    /// Uses the Tonelli–Shanks-style formula from [ABCDK24]:
    ///
    /// 1. δ ← √(a₀² + a₁²) ∈ F_p  (the Fp norm's square root)
    /// 2. x₀ ← a₀ + δ,  t₀ ← 2·x₀
    /// 3. x₁ ← t₀^{(p−3)/4}
    /// 4. x₀ ← x₀·x₁,  x₁ ← a₁·x₁
    /// 5. If (2·x₀)² = t₀: r ← x₀ + x₁·i else:              r ← x₁ − x₀·i
    /// 6. If re(r) is odd, or re(r) = 0 and im(r) is odd: r ← −r
    ///
    /// Step 6 is the canonicalization.
    ///
    /// [CHR17]: https://eprint.iacr.org/2017/518
    /// [ABCDK24]: https://eprint.iacr.org/2024/1563
    pub fn sqrt(&self) -> Fp2 {
        // Steps 1–5: compute a square root (either r or −r).
        let delta = self.norm().sqrt();
        let x0 = &self.a + &delta;
        let t0 = &x0 + &x0;
        let x1 = t0.pow_p3div4();
        let x0 = &x0 * &x1;
        let x1 = &self.b * &x1;
        let t1 = (&x0 + &x0).square();

        let is_eq = t1.ct_eq(&t0);
        let re = Fp::conditional_select(&x1, &x0, is_eq);
        let im = Fp::conditional_select(&(-&x0), &x1, is_eq);

        // Step 6: canonicalize to the even root.
        //
        // "Even" means the least-significant bit of the canonical
        // encoding of re(r) is 0. If re(r) = 0, we check im(r)
        // instead. This is the lexicographic tie-breaking used by the
        // C reference (`fp2_sqrt` in `fp2.c`).
        let re_bytes = re.to_bytes();
        let im_bytes = im.to_bytes();
        let re_is_odd = Choice::from(re_bytes[0] & 1);
        let re_is_zero = re.ct_eq(&Fp::ZERO);
        let im_is_odd = Choice::from(im_bytes[0] & 1);
        let negate = re_is_odd | (re_is_zero & im_is_odd);
        let neg_re = -&re;
        let neg_im = -&im;
        Fp2 {
            a: Fp::conditional_select(&re, &neg_re, negate),
            b: Fp::conditional_select(&im, &neg_im, negate),
        }
    }

    /// Encodes this element as 64 bytes (real part ‖ imaginary part).
    pub fn to_bytes(self) -> [u8; FP2_ENCODED_BYTES] {
        let mut out = [0u8; FP2_ENCODED_BYTES];
        out[..FP_ENCODED_BYTES].copy_from_slice(&self.a.to_bytes());
        out[FP_ENCODED_BYTES..].copy_from_slice(&self.b.to_bytes());
        out
    }

    /// Decodes 64 bytes into an F_{p²} element.
    pub fn from_bytes(bytes: &[u8; FP2_ENCODED_BYTES]) -> Fp2 {
        let a = Fp::from_bytes(bytes[..FP_ENCODED_BYTES].try_into().expect("slice length"));
        let b = Fp::from_bytes(bytes[FP_ENCODED_BYTES..].try_into().expect("slice length"));
        Fp2 { a, b }
    }
}

impl<'b> Add<&'b Fp2> for &Fp2 {
    type Output = Fp2;
    fn add(self, rhs: &'b Fp2) -> Fp2 {
        Fp2 {
            a: &self.a + &rhs.a,
            b: &self.b + &rhs.b,
        }
    }
}

impl<'b> Sub<&'b Fp2> for &Fp2 {
    type Output = Fp2;
    fn sub(self, rhs: &'b Fp2) -> Fp2 {
        Fp2 {
            a: &self.a - &rhs.a,
            b: &self.b - &rhs.b,
        }
    }
}

impl Neg for &Fp2 {
    type Output = Fp2;
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
    /// `Fp::sum_of_products` (resp. `Fp::difference_of_products`)
    /// — one Montgomery reduction per coefficient, two total.
    ///
    /// This is the SQIsign spec's
    /// [`OptimizedPartialFp2Mul`][spec] (Algorithm 8.1) lifted from
    /// the 64-bit Intel optimised path into the dispatched `Fp` API, so
    /// every backend benefits.  Trades the asymmetric Karatsuba 3-mul
    /// shape (3M + 5A + 3 reductions) for the symmetric Algorithm 8.1
    /// shape (2 fused mul-pairs + 2 reductions).  Longa shows the fused
    /// form wins on BLS12-381 (ePrint 2022/367); whether it wins at
    /// SQIsign's n=5 radix-51 layout is empirical and measured by the
    /// `fp2_mul` bench.
    ///
    /// [spec]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.1
    fn mul(self, rhs: &'b Fp2) -> Fp2 {
        Fp2 {
            a: Fp::difference_of_products(&self.a, &rhs.a, &self.b, &rhs.b),
            b: Fp::sum_of_products(&self.a, &rhs.b, &self.b, &rhs.a),
        }
    }
}

/// Multiplies an F_{p²} element by a base field element.
impl<'b> Mul<&'b Fp> for &Fp2 {
    type Output = Fp2;
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
    fn add(self, rhs: Fp2) -> Fp2 {
        &self + &rhs
    }
}

impl Sub for Fp2 {
    type Output = Fp2;
    fn sub(self, rhs: Fp2) -> Fp2 {
        &self - &rhs
    }
}

impl Mul for Fp2 {
    type Output = Fp2;
    fn mul(self, rhs: Fp2) -> Fp2 {
        &self * &rhs
    }
}

impl Neg for Fp2 {
    type Output = Fp2;
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

impl ConditionallySelectable for Fp2 {
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
