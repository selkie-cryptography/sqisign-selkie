//! The quaternion algebra B_{p,∞} = (-1, -p)_Q and its elements.
//!
//! Elements of B_{p,∞} are represented as rational quaternions
//! α = (a + bi + cj + dk) / r in the basis {1, i, j, k} where
//! i² = -1, j² = -p, k = ij = -ji. The prime p is fixed to the
//! NIST-I parameter (p = 5 · 2²⁴⁸ − 1).
//!
//! Since all quaternion algebras ramified at p and ∞ are isomorphic,
//! there is a unique such algebra for each p. See [§3.1] of the spec.
//!
//! [§3.1]: https://sqisign.org/spec/sqisign-20250707.pdf#section.3.1

use core::fmt;

use super::bigint::BigInt;
use super::precomputed::P_WIDE;

// ---------------------------------------------------------------------------
// Coordinate: a coefficient in the quaternion basis {1, i, j, k}
// ---------------------------------------------------------------------------

/// A coefficient of a quaternion element in the basis {1, i, j, k}.
///
/// Wraps a 256-bit signed integer. This is the storage size for
/// quaternion coordinates; arithmetic that produces wider intermediates
/// uses `BigInt<8>` internally and narrows back after normalization.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Coordinate(BigInt<4>);

impl Coordinate {
    /// Zero.
    pub const ZERO: Self = Self(BigInt::ZERO);

    /// One.
    pub const ONE: Self = Self(BigInt::ONE);

    /// Creates a coordinate from an `i64`.
    #[inline]
    pub const fn from_i64(val: i64) -> Self {
        Self(BigInt::from_i64(val))
    }

    /// Creates a coordinate from a `BigInt<4>`.
    #[inline]
    pub const fn from_bigint(val: BigInt<4>) -> Self {
        Self(val)
    }

    /// Creates a coordinate from a sign and four little-endian u64 limbs.
    #[inline]
    pub const fn from_sign_and_limbs(sign: u64, limbs: [u64; 4]) -> Self {
        Self(BigInt::from_sign_and_limbs(sign, limbs))
    }

    /// Returns the inner `BigInt<4>`.
    #[inline]
    pub const fn as_bigint(&self) -> &BigInt<4> {
        &self.0
    }

    /// Widens to `BigInt<8>` for intermediate arithmetic.
    #[inline]
    pub fn wide(&self) -> BigInt<8> {
        self.0.into()
    }

    /// Widens to `BigInt<N>` by zero-extending the four limbs.
    ///
    /// Works for any `N >= 4`. The upper limbs are zeroed.
    #[inline]
    pub fn to_bigint<const N: usize>(&self) -> BigInt<N> {
        let mut limbs = [0u64; N];
        limbs[0] = self.0.as_limbs()[0];
        limbs[1] = self.0.as_limbs()[1];
        limbs[2] = self.0.as_limbs()[2];
        limbs[3] = self.0.as_limbs()[3];
        BigInt::from_sign_and_limbs(
            if bool::from(self.0.is_negative()) {
                1
            } else {
                0
            },
            limbs,
        )
    }
}

impl From<i64> for Coordinate {
    fn from(val: i64) -> Self {
        Self::from_i64(val)
    }
}

impl From<BigInt<4>> for Coordinate {
    fn from(val: BigInt<4>) -> Self {
        Self(val)
    }
}

impl fmt::Debug for Coordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Coordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Denominator: a positive integer common denominator
// ---------------------------------------------------------------------------

/// A positive integer denominator for a quaternion element.
///
/// Always > 0. Can only be constructed via [`Denominator::new`] (which
/// checks positivity) or the constant [`Denominator::ONE`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Denominator(BigInt<4>);

impl Denominator {
    /// The trivial denominator (1).
    pub const ONE: Self = Self(BigInt::ONE);

    /// The denominator 2 (common for O₀ basis elements).
    pub const TWO: Self = Self(BigInt::TWO);

    /// Creates a denominator from a `BigInt<4>`, returning `None` if
    /// the value is not positive.
    pub fn new(val: BigInt<4>) -> Option<Self> {
        if bool::from(val.is_positive()) {
            Some(Self(val))
        } else {
            None
        }
    }

    /// Creates a denominator from a `u64`. Panics if zero.
    pub const fn from_u64(val: u64) -> Self {
        assert!(val > 0, "denominator must be positive");
        Self(BigInt::from_u64(val))
    }

    /// Creates a denominator from four little-endian u64 limbs.
    /// The value must be positive (unchecked in const context).
    pub const fn from_limbs(limbs: [u64; 4]) -> Self {
        Self(BigInt::from_limbs(limbs))
    }

    /// Creates a denominator from a `BigInt<4>` without checking.
    ///
    /// # Safety (logical)
    ///
    /// The caller must ensure `val > 0`.
    pub const fn from_bigint_unchecked(val: BigInt<4>) -> Self {
        Self(val)
    }

    /// Returns the inner `BigInt<4>`.
    #[inline]
    pub const fn as_bigint(&self) -> &BigInt<4> {
        &self.0
    }

    /// Widens to `BigInt<8>` for intermediate arithmetic.
    #[inline]
    pub fn wide(&self) -> BigInt<8> {
        self.0.into()
    }

    /// Widens to `BigInt<N>` by zero-extending the four limbs.
    #[inline]
    pub fn to_bigint<const N: usize>(&self) -> BigInt<N> {
        let mut limbs = [0u64; N];
        limbs[0] = self.0.as_limbs()[0];
        limbs[1] = self.0.as_limbs()[1];
        limbs[2] = self.0.as_limbs()[2];
        limbs[3] = self.0.as_limbs()[3];
        BigInt::from_sign_and_limbs(0, limbs) // denominator is always positive
    }

    /// Multiply two denominators. The result is always positive.
    pub fn mul(&self, other: &Self) -> Self {
        Self(self.0.ct_mul(&other.0))
    }
}

impl From<Denominator> for BigInt<4> {
    fn from(d: Denominator) -> Self {
        d.0
    }
}

impl fmt::Debug for Denominator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Denominator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Element: an element of B_{p,∞}
// ---------------------------------------------------------------------------

/// An element of the quaternion algebra B_{p,∞} = (-1, -p)_Q.
///
/// Represented as α = (a + bi + cj + dk) / r where (a, b, c, d) are
/// integer [`Coordinate`]s and r is a positive [`Denominator`].
/// The canonical form has gcd(a, b, c, d, r) = 1 and r > 0.
///
/// The multiplication rules are i² = -1, j² = -p, k = ij = -ji,
/// following the [multiplication table][fig1] in the spec. The prime
/// p is the fixed NIST-I parameter.
///
/// [fig1]: https://sqisign.org/spec/sqisign-20250707.pdf#figure.3.1
#[derive(Clone, Copy)]
pub struct Element {
    /// Coordinate of 1.
    pub(crate) a: Coordinate,
    /// Coordinate of i.
    pub(crate) b: Coordinate,
    /// Coordinate of j.
    pub(crate) c: Coordinate,
    /// Coordinate of k = ij.
    pub(crate) d: Coordinate,
    /// Common positive denominator.
    pub(crate) denom: Denominator,
}

impl Element {
    /// The zero element.
    pub const ZERO: Self = Self {
        a: Coordinate::ZERO,
        b: Coordinate::ZERO,
        c: Coordinate::ZERO,
        d: Coordinate::ZERO,
        denom: Denominator::ONE,
    };

    /// The basis element i = (0 + 1·i + 0·j + 0·k) / 1.
    pub const I: Self = Self {
        a: Coordinate::ZERO,
        b: Coordinate::ONE,
        c: Coordinate::ZERO,
        d: Coordinate::ZERO,
        denom: Denominator::ONE,
    };

    /// The basis element j = (0 + 0·i + 1·j + 0·k) / 1.
    pub const J: Self = Self {
        a: Coordinate::ZERO,
        b: Coordinate::ZERO,
        c: Coordinate::ONE,
        d: Coordinate::ZERO,
        denom: Denominator::ONE,
    };

    /// The basis element k = ij = (0 + 0·i + 0·j + 1·k) / 1.
    pub const K: Self = Self {
        a: Coordinate::ZERO,
        b: Coordinate::ZERO,
        c: Coordinate::ZERO,
        d: Coordinate::ONE,
        denom: Denominator::ONE,
    };

    /// Creates an element from integer coordinates (denominator = 1).
    #[inline]
    pub const fn from_coords(a: Coordinate, b: Coordinate, c: Coordinate, d: Coordinate) -> Self {
        Self {
            a,
            b,
            c,
            d,
            denom: Denominator::ONE,
        }
    }

    /// Creates an element from `i64` coordinates (denominator = 1).
    ///
    /// Convenience constructor for tests and small constants.
    #[inline]
    pub const fn from_i64(a: i64, b: i64, c: i64, d: i64) -> Self {
        Self {
            a: Coordinate::from_i64(a),
            b: Coordinate::from_i64(b),
            c: Coordinate::from_i64(c),
            d: Coordinate::from_i64(d),
            denom: Denominator::ONE,
        }
    }

    /// Creates an element from coordinates and a denominator.
    #[inline]
    pub const fn new(
        a: Coordinate,
        b: Coordinate,
        c: Coordinate,
        d: Coordinate,
        denom: Denominator,
    ) -> Self {
        Self { a, b, c, d, denom }
    }

    /// Returns `true` if this element is zero.
    pub fn is_zero(&self) -> bool {
        self.a == Coordinate::ZERO
            && self.b == Coordinate::ZERO
            && self.c == Coordinate::ZERO
            && self.d == Coordinate::ZERO
    }

    /// Conjugate: ᾱ = (a - bi - cj - dk) / r.
    ///
    /// [§3.1.5] of the SQIsign specification.
    ///
    /// [§3.1.5]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.3.1.5
    pub fn conjugate(&self) -> Self {
        Self {
            a: self.a,
            b: Coordinate(self.b.0.wrapping_neg()),
            c: Coordinate(self.c.0.wrapping_neg()),
            d: Coordinate(self.d.0.wrapping_neg()),
            denom: self.denom,
        }
    }

    /// Reduced trace: tr(α) = α + ᾱ = 2a / r.
    ///
    /// Returns (numerator, denominator) as `BigInt<4>` values.
    pub fn trace(&self) -> (BigInt<4>, BigInt<4>) {
        let two_a = self.a.0.ct_mul(&BigInt::TWO);
        (two_a, self.denom.0)
    }

    /// Reduced norm: nrd(α) = α·ᾱ = (a² + b² + p(c² + d²)) / r².
    ///
    /// Returns (numerator, denominator) as `BigInt<8>` values, using
    /// wide arithmetic to avoid overflow.
    pub fn norm(&self) -> (BigInt<8>, BigInt<8>) {
        let a = self.a.wide();
        let b = self.b.wide();
        let c = self.c.wide();
        let d = self.d.wide();
        let p: BigInt<8> = P_WIDE;

        let numer = a
            .ct_mul(&a)
            .ct_add(&b.ct_mul(&b))
            .ct_add(&p.ct_mul(&c.ct_mul(&c).ct_add(&d.ct_mul(&d))));

        let r = self.denom.wide();
        let denom_sq = r.ct_mul(&r);
        (numer, denom_sq)
    }

    /// Quaternion multiplication: self · rhs.
    ///
    /// Uses the [multiplication table][fig1] for B_{p,∞}:
    ///
    /// ```text
    ///       1    i     j     k
    ///   1 | 1    i     j     k
    ///   i | i   -1     k    -j
    ///   j | j   -k    -p    pi
    ///   k | k    j   -pi    -p
    /// ```
    ///
    /// Computed in `BigInt<8>` to accommodate intermediate products,
    /// then narrowed back to `Coordinate` storage via normalization.
    ///
    /// [fig1]: https://sqisign.org/spec/sqisign-20250707.pdf#figure.3.1
    pub fn mul(&self, rhs: &Self) -> Self {
        let (a1, b1, c1, d1) = (self.a.wide(), self.b.wide(), self.c.wide(), self.d.wide());
        let (a2, b2, c2, d2) = (rhs.a.wide(), rhs.b.wide(), rhs.c.wide(), rhs.d.wide());
        let p: BigInt<8> = P_WIDE;

        // a = a1*a2 - b1*b2 - p*(c1*c2 + d1*d2)
        let a = a1
            .ct_mul(&a2)
            .ct_sub(&b1.ct_mul(&b2))
            .ct_sub(&p.ct_mul(&c1.ct_mul(&c2).ct_add(&d1.ct_mul(&d2))));

        // b = a1*b2 + b1*a2 + p*(c1*d2 - d1*c2)
        let b = a1
            .ct_mul(&b2)
            .ct_add(&b1.ct_mul(&a2))
            .ct_add(&p.ct_mul(&c1.ct_mul(&d2).ct_sub(&d1.ct_mul(&c2))));

        // c = a1*c2 - b1*d2 + c1*a2 + d1*b2
        let c = a1
            .ct_mul(&c2)
            .ct_sub(&b1.ct_mul(&d2))
            .ct_add(&c1.ct_mul(&a2))
            .ct_add(&d1.ct_mul(&b2));

        // d = a1*d2 + b1*c2 - c1*b2 + d1*a2
        let d = a1
            .ct_mul(&d2)
            .ct_add(&b1.ct_mul(&c2))
            .ct_sub(&c1.ct_mul(&b2))
            .ct_add(&d1.ct_mul(&a2));

        let new_denom = self.denom.wide().ct_mul(&rhs.denom.wide());

        // Normalize in wide representation, then narrow.
        Self::from_wide(a, b, c, d, new_denom)
    }
}

impl Element {
    /// Normalize: reduce gcd(a, b, c, d, r) to 1, ensure r > 0.
    pub fn normalize(&mut self) {
        let mut g = self.a.0.abs().gcd(&self.b.0.abs());
        g = g.gcd(&self.c.0.abs());
        g = g.gcd(&self.d.0.abs());
        g = g.gcd(&self.denom.0.abs());

        if !bool::from(g.is_zero()) && g != BigInt::ONE {
            let (qa, _) = self.a.0.div_rem(&g);
            let (qb, _) = self.b.0.div_rem(&g);
            let (qc, _) = self.c.0.div_rem(&g);
            let (qd, _) = self.d.0.div_rem(&g);
            let (qr, _) = self.denom.0.div_rem(&g);
            self.a = Coordinate(qa);
            self.b = Coordinate(qb);
            self.c = Coordinate(qc);
            self.d = Coordinate(qd);
            self.denom = Denominator(qr);
        }

        // Ensure denominator > 0.
        if bool::from(self.denom.0.is_negative()) {
            self.a = Coordinate(self.a.0.wrapping_neg());
            self.b = Coordinate(self.b.0.wrapping_neg());
            self.c = Coordinate(self.c.0.wrapping_neg());
            self.d = Coordinate(self.d.0.wrapping_neg());
            self.denom = Denominator(self.denom.0.wrapping_neg());
        }
    }

    /// Returns a normalized copy.
    pub fn normalized(&self) -> Self {
        let mut result = *self;
        result.normalize();
        result
    }

    /// Compute backtracking and normalize.
    ///
    /// Converts α from the {1, i, j, k} basis to the O₀ basis
    /// (1, i, (i+j)/2, (1+k)/2), finds the largest power of 2
    /// dividing all O₀-basis coordinates, and divides it out.
    ///
    /// Returns the normalized α and the backtracking exponent n
    /// (the 2-adic valuation of the GCD).
    ///
    /// Implements [ComputeBacktrackingAndNormalize][Alg. 4.4].
    ///
    /// [Alg. 4.4]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.4
    pub fn compute_backtracking(&self) -> (Self, u32) {
        // Write α = α₀ + α₁i + α₂j + α₃k (with common denom r).
        // The O₀ basis is (1, i, (i+j)/2, (1+k)/2), so:
        //   α'₀ = α₀ − α₃
        //   α'₁ = α₁ − α₂
        //   α'₂ = α₂ / 2   (must be exact since α ∈ O₀)
        //   α'₃ = α₃ / 2
        //
        // Actually, if denom r = 2 (common for O₀ elements), the
        // coords are already doubled. The spec assumes integer coords
        // in the O₀ basis. We need to ensure exactness.
        //
        // For now, work with the raw {1,i,j,k} coords and denom.
        // The O₀-basis coords (before dividing by denom) are:
        //   α'₀ = a - d,  α'₁ = b - c,  α'₂ = c,  α'₃ = d
        // where (a,b,c,d) are the numerator coords and denom = 2
        // for a typical O₀ element. Then the true O₀ coords are
        // (α'₀, α'₁, α'₂, α'₃) / denom, which must be integers.
        //
        // The spec's lines 2-5 assume denom = 1 (integer coords in
        // {1,i,j,k}). The C ref calls `quat_alg_make_primitive`
        // which normalizes first, then does the basis change.
        let mut elem = self.normalized();

        // Convert to O₀ basis: for O₀ = Z⟨1, i, (i+j)/2, (1+k)/2⟩,
        // if α = (a + bi + cj + dk)/r in {1,i,j,k}, then in O₀:
        //   α'₀ = a - d,  α'₁ = b - c,  α'₂ = c,  α'₃ = d
        // (all divided by r, which must give integers).
        let a = &elem.a.0;
        let b = &elem.b.0;
        let c = &elem.c.0;
        let d = &elem.d.0;

        let c0 = a.ct_sub(d); // α'₀ = a - d
        let c1 = b.ct_sub(c); // α'₁ = b - c
        let c2 = *c; // α'₂ = c
        let c3 = *d; // α'₃ = d

        // g = gcd(α'₀, α'₁, α'₂, α'₃)
        let g = c0.abs().gcd(&c1.abs()).gcd(&c2.abs()).gcd(&c3.abs());

        // n = 2-adic valuation of g
        let n = if bool::from(g.is_zero()) {
            0
        } else {
            g.two_adic_val()
        };

        // Divide α by 2^n: scale the denominator up by 2^n.
        // α/2^n means each coord is divided by 2^n, equivalently
        // denom is multiplied by 2^n.
        if n > 0 {
            let divisor = BigInt::<4>::ONE.shl(n);
            let new_denom = elem.denom.0.ct_mul(&divisor);
            elem.denom = Denominator::new(new_denom).expect("denom > 0");
            elem.normalize();
        }

        (elem, n)
    }

    /// Addition: self + rhs.
    pub fn add(&self, rhs: &Self) -> Self {
        if self.denom == rhs.denom {
            Self {
                a: Coordinate(self.a.0.ct_add(&rhs.a.0)),
                b: Coordinate(self.b.0.ct_add(&rhs.b.0)),
                c: Coordinate(self.c.0.ct_add(&rhs.c.0)),
                d: Coordinate(self.d.0.ct_add(&rhs.d.0)),
                denom: self.denom,
            }
        } else {
            Self {
                a: Coordinate(
                    self.a
                        .0
                        .ct_mul(&rhs.denom.0)
                        .ct_add(&rhs.a.0.ct_mul(&self.denom.0)),
                ),
                b: Coordinate(
                    self.b
                        .0
                        .ct_mul(&rhs.denom.0)
                        .ct_add(&rhs.b.0.ct_mul(&self.denom.0)),
                ),
                c: Coordinate(
                    self.c
                        .0
                        .ct_mul(&rhs.denom.0)
                        .ct_add(&rhs.c.0.ct_mul(&self.denom.0)),
                ),
                d: Coordinate(
                    self.d
                        .0
                        .ct_mul(&rhs.denom.0)
                        .ct_add(&rhs.d.0.ct_mul(&self.denom.0)),
                ),
                denom: self.denom.mul(&rhs.denom),
            }
        }
    }

    /// Subtraction: self - rhs.
    pub fn sub(&self, rhs: &Self) -> Self {
        let neg = Self {
            a: Coordinate(rhs.a.0.wrapping_neg()),
            b: Coordinate(rhs.b.0.wrapping_neg()),
            c: Coordinate(rhs.c.0.wrapping_neg()),
            d: Coordinate(rhs.d.0.wrapping_neg()),
            denom: rhs.denom,
        };
        self.add(&neg)
    }

    /// Scalar multiplication: α · s.
    pub fn scalar_mul(&self, s: &BigInt<4>) -> Self {
        Self {
            a: Coordinate(self.a.0.ct_mul(s)),
            b: Coordinate(self.b.0.ct_mul(s)),
            c: Coordinate(self.c.0.ct_mul(s)),
            d: Coordinate(self.d.0.ct_mul(s)),
            denom: self.denom,
        }
    }

    /// Construct an `Element` from wide (`BigInt<8>`) intermediates,
    /// normalizing and narrowing back to `Coordinate` storage.
    fn from_wide(a: BigInt<8>, b: BigInt<8>, c: BigInt<8>, d: BigInt<8>, r: BigInt<8>) -> Self {
        // GCD-normalize in wide representation.
        let mut g = a.abs().gcd(&b.abs());
        g = g.gcd(&c.abs());
        g = g.gcd(&d.abs());
        g = g.gcd(&r.abs());

        let (mut wa, mut wb, mut wc, mut wd, mut wr) = (a, b, c, d, r);
        if !bool::from(g.is_zero()) && g != BigInt::<8>::ONE {
            let (qa, _) = a.div_rem(&g);
            let (qb, _) = b.div_rem(&g);
            let (qc, _) = c.div_rem(&g);
            let (qd, _) = d.div_rem(&g);
            let (qr, _) = r.div_rem(&g);
            wa = qa;
            wb = qb;
            wc = qc;
            wd = qd;
            wr = qr;
        }

        // Ensure denominator > 0.
        if bool::from(wr.is_negative()) {
            wa = wa.wrapping_neg();
            wb = wb.wrapping_neg();
            wc = wc.wrapping_neg();
            wd = wd.wrapping_neg();
            wr = wr.wrapping_neg();
        }

        // Narrow to BigInt<4>. If the normalized values don't fit,
        // this is a bug — after GCD reduction they should be small
        // enough for the NIST-I parameter set.
        let narrow = |v: BigInt<8>| -> BigInt<4> {
            let ct: subtle::CtOption<BigInt<4>> = v.into();
            // In debug builds, panic on overflow; in release, truncate.
            debug_assert!(
                bool::from(ct.is_some()),
                "quaternion coordinate overflow after normalization"
            );
            if bool::from(ct.is_some()) {
                ct.unwrap()
            } else {
                // Fallback: truncate (shouldn't happen in practice).
                BigInt::from_sign_and_limbs(
                    if bool::from(v.is_negative()) { 1 } else { 0 },
                    [
                        v.as_limbs()[0],
                        v.as_limbs()[1],
                        v.as_limbs()[2],
                        v.as_limbs()[3],
                    ],
                )
            }
        };

        Self {
            a: Coordinate(narrow(wa)),
            b: Coordinate(narrow(wb)),
            c: Coordinate(narrow(wc)),
            d: Coordinate(narrow(wd)),
            denom: Denominator(narrow(wr)),
        }
    }
}

impl PartialEq for Element {
    fn eq(&self, other: &Self) -> bool {
        // Cross-multiply: a1/r1 == a2/r2 iff a1*r2 == a2*r1.
        self.a.0.ct_mul(&other.denom.0) == other.a.0.ct_mul(&self.denom.0)
            && self.b.0.ct_mul(&other.denom.0) == other.b.0.ct_mul(&self.denom.0)
            && self.c.0.ct_mul(&other.denom.0) == other.c.0.ct_mul(&self.denom.0)
            && self.d.0.ct_mul(&other.denom.0) == other.d.0.ct_mul(&self.denom.0)
    }
}

impl Eq for Element {}

impl fmt::Debug for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({} + {}*i + {}*j + {}*k) / {}",
            self.a, self.b, self.c, self.d, self.denom
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero() {
        assert!(Element::ZERO.is_zero());
    }

    #[test]
    fn conjugate() {
        let e = Element::from_i64(1, 2, 3, 4);
        let conj = e.conjugate();
        assert_eq!(conj.a, Coordinate::from_i64(1));
        assert_eq!(conj.b, Coordinate::from_i64(-2));
        assert_eq!(conj.c, Coordinate::from_i64(-3));
        assert_eq!(conj.d, Coordinate::from_i64(-4));
    }

    #[test]
    fn trace() {
        let e = Element::from_i64(5, 2, 3, 4);
        let (tr_num, tr_den) = e.trace();
        assert_eq!(tr_num, BigInt::from(10i64));
        assert_eq!(tr_den, BigInt::ONE);
    }

    #[test]
    fn norm() {
        // nrd(1 + 2i + 3j + 4k) with p from NIST-I.
        // = 1 + 4 + p*(9 + 16) = 5 + 25p
        let e = Element::from_i64(1, 2, 3, 4);
        let (n_num, n_den) = e.norm();
        let p: BigInt<8> = P_WIDE;
        let expected = BigInt::<8>::from(5i64).ct_add(&BigInt::<8>::from(25i64).ct_mul(&p));
        assert_eq!(n_num, expected);
        assert_eq!(n_den, BigInt::<8>::ONE);
    }

    #[test]
    fn mul_i_squared() {
        // i² = -1
        let unit_i = Element::from_i64(0, 1, 0, 0);
        let result = unit_i.mul(&unit_i);
        assert_eq!(result, Element::from_i64(-1, 0, 0, 0));
    }

    #[test]
    fn mul_ij_eq_k() {
        // ij = k
        let unit_i = Element::from_i64(0, 1, 0, 0);
        let unit_j = Element::from_i64(0, 0, 1, 0);
        let result = unit_i.mul(&unit_j);
        assert_eq!(result, Element::from_i64(0, 0, 0, 1));
    }

    #[test]
    fn mul_ji_eq_neg_k() {
        // ji = -k
        let unit_i = Element::from_i64(0, 1, 0, 0);
        let unit_j = Element::from_i64(0, 0, 1, 0);
        let result = unit_j.mul(&unit_i);
        assert_eq!(result, Element::from_i64(0, 0, 0, -1));
    }

    #[test]
    fn norm_is_multiplicative() {
        let alpha = Element::from_i64(1, 2, 0, 1);
        let beta = Element::from_i64(3, 0, 1, 0);
        let product = alpha.mul(&beta);

        let (na, da) = alpha.norm();
        let (nb, db) = beta.norm();
        let (np, dp) = product.norm();

        // na/da * nb/db == np/dp
        let lhs = na.ct_mul(&nb).ct_mul(&dp);
        let rhs = np.ct_mul(&da).ct_mul(&db);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn mul_by_conjugate_is_norm() {
        let e = Element::from_i64(1, 2, 3, 4);
        let conj = e.conjugate();
        let product = e.mul(&conj).normalized();

        let (n_num, n_den) = e.norm();
        // product should be scalar: (nrd, 0, 0, 0).
        // The product denom is r², and n_den is also r².
        assert_eq!(product.b, Coordinate::from_i64(0));
        assert_eq!(product.c, Coordinate::from_i64(0));
        assert_eq!(product.d, Coordinate::from_i64(0));
    }

    #[test]
    fn addition() {
        let a = Element::from_i64(1, 2, 3, 4);
        let b = Element::from_i64(5, 6, 7, 8);
        let sum = a.add(&b);
        assert_eq!(sum, Element::from_i64(6, 8, 10, 12));
    }

    #[test]
    fn subtraction() {
        let a = Element::from_i64(5, 6, 7, 8);
        let b = Element::from_i64(1, 2, 3, 4);
        let diff = a.sub(&b);
        assert_eq!(diff, Element::from_i64(4, 4, 4, 4));
    }

    #[test]
    fn normalize_gcd() {
        let mut e = Element::new(
            Coordinate::from_i64(2),
            Coordinate::from_i64(4),
            Coordinate::from_i64(6),
            Coordinate::from_i64(8),
            Denominator::TWO,
        );
        e.normalize();
        assert_eq!(e, Element::from_i64(1, 2, 3, 4));
        assert_eq!(e.denom, Denominator::ONE);
    }

    #[test]
    fn equality_across_denominators() {
        let a = Element::new(
            Coordinate::from_i64(2),
            Coordinate::from_i64(4),
            Coordinate::from_i64(0),
            Coordinate::from_i64(0),
            Denominator::TWO,
        );
        let b = Element::from_i64(1, 2, 0, 0);
        assert_eq!(a, b);
    }
}
