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
//!
//! The types [`Coordinate`], [`Denominator`], and [`Element`] are generic
//! over a const parameter `N` (the number of `u64` limbs in `BigInt<N>`).
//! Use `Element<4>` for verification/torsion (256-bit) and `Element<110>`
//! for signing (7040-bit, per Kim et al. ePrint 2025/1649).
//!
//! # Divergence from spec and C reference
//!
//! The spec and C reference use GMP (arbitrary-precision integers) for
//! quaternion arithmetic. We use fixed-width `BigInt<N>` following the
//! bounds proven by Kim et al. (ePrint 2025/1649): all intermediates
//! during NIST-I signing fit in 7,026 bits (110 u64 limbs). This
//! enables constant-time arithmetic without dynamic allocation.
//!
//! `Element::mul` and `Element::norm` are currently only implemented
//! for `Element<4>` (they widen to `BigInt<8>` internally). For
//! `Element<110>`, these operations will use modular arithmetic
//! per Kim et al.'s modified IdealMultiplication (Appendix B).
// TODO: `N: usize` allows arbitrarily large values, but Kim et al.
// prove N=110 is the worst-case for NIST-I signing. We'd like to
// restrict N at the type level (e.g., a sealed trait or a bounded
// const generic when Rust supports `where N <= 110`) to prevent
// accidentally using oversized types. For now, the valid values
// are: N=4 (verification), N=9 (D_MIX commitment), N=110 (signing).

use core::fmt;

use super::{bigint::BigInt, precomputed::P_WIDE};

// ---------------------------------------------------------------------------
// Coordinate: a coefficient in the quaternion basis {1, i, j, k}
// ---------------------------------------------------------------------------

/// A coefficient of a quaternion element in the basis {1, i, j, k}.
///
/// Wraps a `BigInt<N>` signed integer. The limb count `N` determines
/// the maximum magnitude: `N = 4` gives 256-bit coordinates (sufficient
/// for verification and torsion basis), `N = 110` gives 7040-bit
/// coordinates (sufficient for signing intermediates).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Coordinate<const N: usize>(BigInt<N>);

impl<const N: usize> Coordinate<N> {
    /// Zero.
    pub const ZERO: Self = Self(BigInt::ZERO);

    /// One.
    pub const ONE: Self = Self(BigInt::ONE);

    /// Creates a coordinate from an `i64`.
    #[inline]
    pub const fn from_i64(val: i64) -> Self {
        Self(BigInt::from_i64(val))
    }

    /// Creates a coordinate from a `BigInt<N>`.
    #[inline]
    pub const fn from_bigint(val: BigInt<N>) -> Self {
        Self(val)
    }

    /// Creates a coordinate from a sign and `N` little-endian u64 limbs.
    #[inline]
    pub const fn from_sign_and_limbs(sign: u64, limbs: [u64; N]) -> Self {
        Self(BigInt::from_sign_and_limbs(sign, limbs))
    }

    /// Creates a non-negative coordinate from `N` little-endian `u64` limbs.
    ///
    /// Delegates to [`BigInt::from_limbs`].
    #[inline]
    pub const fn from_limbs(limbs: [u64; N]) -> Self {
        Self(BigInt::from_limbs(limbs))
    }

    /// Creates a negative coordinate from `N` little-endian `u64` limbs
    /// representing the absolute value.
    ///
    /// Delegates to [`BigInt::from_limbs_neg`].
    #[inline]
    pub const fn from_limbs_neg(limbs: [u64; N]) -> Self {
        Self(BigInt::from_limbs_neg(limbs))
    }

    /// Returns the inner `BigInt<N>`.
    #[inline]
    pub const fn as_bigint(&self) -> &BigInt<N> {
        &self.0
    }

    /// Consumes `self` and returns the inner `BigInt<N>`.
    #[inline]
    pub fn to_bigint(self) -> BigInt<N> {
        self.0
    }
}

/// Widening methods available only for `Coordinate<4>`.
impl Coordinate<4> {
    /// Widens to `BigInt<8>` for intermediate arithmetic.
    #[inline]
    pub fn wide(&self) -> BigInt<8> {
        self.0.into()
    }

    /// Widens to `BigInt<M>` by zero-extending the four limbs.
    ///
    /// Works for any `M >= 4`. The upper limbs are zeroed.
    #[inline]
    pub fn widen<const M: usize>(self) -> BigInt<M> {
        let mut limbs = [0u64; M];
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

impl<const N: usize> From<i64> for Coordinate<N> {
    fn from(val: i64) -> Self {
        Self::from_i64(val)
    }
}

impl<const N: usize> From<BigInt<N>> for Coordinate<N> {
    fn from(val: BigInt<N>) -> Self {
        Self(val)
    }
}

impl<const N: usize> fmt::Debug for Coordinate<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<const N: usize> fmt::Display for Coordinate<N> {
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
pub struct Denominator<const N: usize>(BigInt<N>);

impl<const N: usize> Denominator<N> {
    /// The trivial denominator (1).
    pub const ONE: Self = Self(BigInt::ONE);

    /// The denominator 2 (common for O₀ basis elements).
    pub const TWO: Self = Self(BigInt::TWO);

    /// Creates a denominator from a `BigInt<N>`, returning `None` if
    /// the value is not positive.
    pub fn new(val: BigInt<N>) -> Option<Self> {
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

    /// Creates a denominator from `N` little-endian u64 limbs.
    /// The value must be positive (unchecked in const context).
    pub const fn from_limbs(limbs: [u64; N]) -> Self {
        Self(BigInt::from_limbs(limbs))
    }

    /// Creates a denominator from a `BigInt<N>` without checking.
    ///
    /// # Safety (logical)
    ///
    /// The caller must ensure `val > 0`.
    pub const fn from_bigint_unchecked(val: BigInt<N>) -> Self {
        Self(val)
    }

    /// Returns the inner `BigInt<N>`.
    #[inline]
    pub const fn as_bigint(&self) -> &BigInt<N> {
        &self.0
    }

    /// Multiply two denominators. The result is always positive.
    pub fn mul(&self, other: &Self) -> Self {
        Self(self.0.ct_mul(&other.0))
    }
}

/// Widening methods available only for `Denominator<4>`.
impl Denominator<4> {
    /// Widens to `BigInt<8>` for intermediate arithmetic.
    #[inline]
    pub fn wide(&self) -> BigInt<8> {
        self.0.into()
    }

    /// Widens to `BigInt<M>` by zero-extending the four limbs.
    #[inline]
    pub fn widen<const M: usize>(self) -> BigInt<M> {
        let mut limbs = [0u64; M];
        limbs[0] = self.0.as_limbs()[0];
        limbs[1] = self.0.as_limbs()[1];
        limbs[2] = self.0.as_limbs()[2];
        limbs[3] = self.0.as_limbs()[3];
        BigInt::from_sign_and_limbs(0, limbs) // denominator is always positive
    }
}

impl<const N: usize> From<Denominator<N>> for BigInt<N> {
    fn from(d: Denominator<N>) -> Self {
        d.0
    }
}

impl<const N: usize> fmt::Debug for Denominator<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<const N: usize> fmt::Display for Denominator<N> {
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
pub struct Element<const N: usize> {
    /// Coordinate of 1.
    pub(crate) a: Coordinate<N>,
    /// Coordinate of i.
    pub(crate) b: Coordinate<N>,
    /// Coordinate of j.
    pub(crate) c: Coordinate<N>,
    /// Coordinate of k = ij.
    pub(crate) d: Coordinate<N>,
    /// Common positive denominator.
    pub(crate) denom: Denominator<N>,
}

impl<const N: usize> Element<N> {
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
    pub const fn from_coords(
        a: Coordinate<N>,
        b: Coordinate<N>,
        c: Coordinate<N>,
        d: Coordinate<N>,
    ) -> Self {
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
        a: Coordinate<N>,
        b: Coordinate<N>,
        c: Coordinate<N>,
        d: Coordinate<N>,
        denom: Denominator<N>,
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
    /// Returns (numerator, denominator) as `BigInt<N>` values.
    pub fn trace(&self) -> (BigInt<N>, BigInt<N>) {
        let two_a = self.a.0.ct_mul(&BigInt::TWO);
        (two_a, self.denom.0)
    }

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
    pub fn scalar_mul(&self, s: &BigInt<N>) -> Self {
        Self {
            a: Coordinate(self.a.0.ct_mul(s)),
            b: Coordinate(self.b.0.ct_mul(s)),
            c: Coordinate(self.c.0.ct_mul(s)),
            d: Coordinate(self.d.0.ct_mul(s)),
            denom: self.denom,
        }
    }
}

impl<const N: usize> Element<N> {
    /// Embed p into `BigInt<N>`.
    fn p_at_width() -> BigInt<N> {
        let p8: BigInt<8> = P_WIDE;
        let mut limbs = [0u64; N];
        let len = p8.as_limbs().len().min(N);
        limbs[..len].copy_from_slice(&p8.as_limbs()[..len]);
        BigInt::from_sign_and_limbs(0, limbs)
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
    pub fn compute_backtracking(&self) -> Option<(Self, u32)> {
        // Convert to O₀ basis: for O₀ = Z⟨1, i, (i+j)/2, (1+k)/2⟩,
        // if α = (a + bi + cj + dk)/r in {1,i,j,k}, then in O₀:
        //   α'₀ = a - d,  α'₁ = b - c,  α'₂ = c,  α'₃ = d
        // (all divided by r, which must give integers).
        let mut elem = self.normalized();
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
        if n > 0 {
            let divisor = BigInt::<N>::ONE.shl(n);
            let new_denom = elem.denom.0.ct_mul(&divisor);
            elem.denom = Denominator::new(new_denom)?;
            elem.normalize();
        }

        Some((elem, n))
    }

    /// Divides α by the GCD of its `O₀`-basis coordinates, returning
    /// the primitive representative and the extracted integer factor.
    ///
    /// A quaternion element of a maximal order `O` is *primitive* when
    /// no integer `k > 1` satisfies `α ∈ k · O`. Equivalently, the
    /// GCD of its coordinates in the `O`-basis is 1.
    ///
    /// The C reference primitivizes the response-phase quaternion
    /// fully before constructing `I_com,rsp`; with only 2-adic
    /// backtracking normalization (as [`compute_backtracking`]
    /// provides), an odd integer factor `g` remains in `α` and the
    /// declared ideal norm `n(I) = N(I_com) · q_rsp` overstates the
    /// actual covolume-derived norm by `g²`. Calling
    /// `make_primitive_odd` after `compute_backtracking` removes
    /// the odd content and restores the consistency.
    ///
    /// The returned scalar `g` is odd (any power of 2 was already
    /// removed by `compute_backtracking`).
    ///
    /// [`compute_backtracking`]: Self::compute_backtracking
    pub fn make_primitive_odd(&self) -> Option<(Self, BigInt<N>)> {
        let mut elem = self.normalized();
        let a = &elem.a.0;
        let b = &elem.b.0;
        let c = &elem.c.0;
        let d = &elem.d.0;

        // `O₀`-basis coordinates of α (see `compute_backtracking`).
        let c0 = a.ct_sub(d);
        let c1 = b.ct_sub(c);
        let c2 = *c;
        let c3 = *d;

        let g = c0.abs().gcd(&c1.abs()).gcd(&c2.abs()).gcd(&c3.abs());
        let one = BigInt::<N>::ONE;
        if bool::from(g.is_zero()) || g == one {
            return Some((elem, one));
        }

        let new_denom = elem.denom.0.ct_mul(&g);
        elem.denom = Denominator::new(new_denom)?;
        elem.normalize();

        Some((elem, g))
    }

    /// Narrow all coordinates and denominator to `BigInt<M>`, returning
    /// `None` if any of them overflow.
    ///
    /// Useful after sampling at a wide width (for radius headroom)
    /// when subsequent operations need a narrower representative.
    /// Succeeds only when every individual limb above position `M` is
    /// zero in each of the five `BigInt<N>` fields.
    pub fn narrow_to<const M: usize>(&self) -> Option<Element<M>> {
        const { assert!(M <= N, "Element::narrow_to: M must be <= N") };
        Some(Element {
            a: Coordinate(self.a.0.narrow_to::<M>()?),
            b: Coordinate(self.b.0.narrow_to::<M>()?),
            c: Coordinate(self.c.0.narrow_to::<M>()?),
            d: Coordinate(self.d.0.narrow_to::<M>()?),
            denom: Denominator(self.denom.0.narrow_to::<M>()?),
        })
    }

    /// Reduced norm at a wider working width `W`.
    ///
    /// Returns `(numerator, denominator²)` at `BigInt<W>`. Computes
    /// nrd(α) = (a² + b² + p(c² + d²)) / r² after widening the
    /// coordinates to `BigInt<W>`, so there is no overflow provided
    /// `W` is large enough for the squared sum.
    ///
    /// # Width requirement
    ///
    /// `W >= N` (compile-time enforced). For correctness of the inner
    /// arithmetic, `W` must also satisfy
    /// `64*W >= 2*bits(max_coord) + bits(p)` where `max_coord` is the
    /// largest magnitude among `a, b, c, d, r`. For NIST-I with
    /// coordinates up to ~2^574 and `p ~ 2^251`, pick `W >= 22`.
    pub fn norm_w<const W: usize>(&self) -> (BigInt<W>, BigInt<W>) {
        const { assert!(W >= N, "norm_w: W must be >= N") };
        let a: BigInt<W> = self.a.0.widen();
        let b: BigInt<W> = self.b.0.widen();
        let c: BigInt<W> = self.c.0.widen();
        let d: BigInt<W> = self.d.0.widen();
        let r: BigInt<W> = self.denom.0.widen();
        let p = {
            let p8: BigInt<8> = P_WIDE;
            let mut limbs = [0u64; W];
            let len = p8.as_limbs().len().min(W);
            limbs[..len].copy_from_slice(&p8.as_limbs()[..len]);
            BigInt::<W>::from_sign_and_limbs(0, limbs)
        };

        let numer = a
            .ct_mul(&a)
            .ct_add(&b.ct_mul(&b))
            .ct_add(&p.ct_mul(&c.ct_mul(&c).ct_add(&d.ct_mul(&d))));
        let denom_sq = r.ct_mul(&r);
        (numer, denom_sq)
    }

    /// Reduced norm: nrd(α) = (a² + b² + p(c² + d²)) / r².
    ///
    /// Returns `(numerator, denominator²)` as `BigInt<N>`.
    ///
    /// **Precondition:** coordinates must use at most N/2 limbs so
    /// that products don't overflow `BigInt<N>`. This is guaranteed
    /// for L2-reduced lattice elements at N ≥ 8.
    pub fn norm_direct(&self) -> (BigInt<N>, BigInt<N>) {
        let (a, b, c, d) = (&self.a.0, &self.b.0, &self.c.0, &self.d.0);
        let p = Self::p_at_width();

        let numer = a
            .ct_mul(a)
            .ct_add(&b.ct_mul(b))
            .ct_add(&p.ct_mul(&c.ct_mul(c).ct_add(&d.ct_mul(d))));

        let r = &self.denom.0;
        let denom_sq = r.ct_mul(r);
        (numer, denom_sq)
    }

    /// Quaternion multiplication: self · rhs.
    ///
    /// Uses the [multiplication table][fig1] for B_{p,∞} = (-1, -p):
    /// ```text
    ///       1    i     j     k
    ///   1 | 1    i     j     k
    ///   i | i   -1     k    -j
    ///   j | j   -k    -p    pi
    ///   k | k    j   -pi    -p
    /// ```
    ///
    /// Multiplies directly at `BigInt<N>` width — no widening.
    ///
    /// **Precondition:** coordinates must use at most N/2 limbs so
    /// that products don't overflow `BigInt<N>`.
    ///
    /// [fig1]: https://sqisign.org/spec/sqisign-20250707.pdf#figure.3.1
    pub fn mul_direct(&self, rhs: &Self) -> Self {
        let (a1, b1, c1, d1) = (&self.a.0, &self.b.0, &self.c.0, &self.d.0);
        let (a2, b2, c2, d2) = (&rhs.a.0, &rhs.b.0, &rhs.c.0, &rhs.d.0);
        let p = Self::p_at_width();

        let a = a1
            .ct_mul(a2)
            .ct_sub(&b1.ct_mul(b2))
            .ct_sub(&p.ct_mul(&c1.ct_mul(c2).ct_add(&d1.ct_mul(d2))));
        let b = a1
            .ct_mul(b2)
            .ct_add(&b1.ct_mul(a2))
            .ct_add(&p.ct_mul(&c1.ct_mul(d2).ct_sub(&d1.ct_mul(c2))));
        let c = a1
            .ct_mul(c2)
            .ct_sub(&b1.ct_mul(d2))
            .ct_add(&c1.ct_mul(a2))
            .ct_add(&d1.ct_mul(b2));
        let d = a1
            .ct_mul(d2)
            .ct_add(&b1.ct_mul(c2))
            .ct_sub(&c1.ct_mul(b2))
            .ct_add(&d1.ct_mul(a2));

        let new_denom = self.denom.0.ct_mul(&rhs.denom.0);

        Self {
            a: Coordinate(a),
            b: Coordinate(b),
            c: Coordinate(c),
            d: Coordinate(d),
            denom: Denominator::from_bigint_unchecked(new_denom),
        }
    }
}

/// Methods that require widening to `BigInt<8>` intermediates.
///
/// `norm` and `mul` widen to `BigInt<8>` then narrow back. For
/// `Element<4>` this avoids overflow. For larger N, use the generic
/// versions on `impl<N> Element<N>` which multiply directly at
/// width N (safe when coordinates use at most N/2 limbs).
impl Element<4> {
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

    /// Quaternion multiplication: self · rhs (widening to `BigInt<8>`).
    ///
    /// [fig1]: https://sqisign.org/spec/sqisign-20250707.pdf#figure.3.1
    pub fn mul(&self, rhs: &Self) -> Self {
        let (a1, b1, c1, d1) = (self.a.wide(), self.b.wide(), self.c.wide(), self.d.wide());
        let (a2, b2, c2, d2) = (rhs.a.wide(), rhs.b.wide(), rhs.c.wide(), rhs.d.wide());
        let p: BigInt<8> = P_WIDE;

        let a = a1
            .ct_mul(&a2)
            .ct_sub(&b1.ct_mul(&b2))
            .ct_sub(&p.ct_mul(&c1.ct_mul(&c2).ct_add(&d1.ct_mul(&d2))));
        let b = a1
            .ct_mul(&b2)
            .ct_add(&b1.ct_mul(&a2))
            .ct_add(&p.ct_mul(&c1.ct_mul(&d2).ct_sub(&d1.ct_mul(&c2))));
        let c = a1
            .ct_mul(&c2)
            .ct_sub(&b1.ct_mul(&d2))
            .ct_add(&c1.ct_mul(&a2))
            .ct_add(&d1.ct_mul(&b2));
        let d = a1
            .ct_mul(&d2)
            .ct_add(&b1.ct_mul(&c2))
            .ct_sub(&c1.ct_mul(&b2))
            .ct_add(&d1.ct_mul(&a2));

        let new_denom = self.denom.wide().ct_mul(&rhs.denom.wide());
        Self::from_wide(a, b, c, d, new_denom)
    }

    /// Construct an `Element<4>` from wide (`BigInt<8>`) intermediates,
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

        // Narrow to `BigInt<4>`. If the normalized values don't fit,
        // the product genuinely exceeds `Element<4>`'s budget and
        // the caller picked the wrong width. Panicking here is
        // strictly better than silently truncating: narrow-path
        // callers (e.g. `LeftIdeal<4>::random_norm`) that expect
        // the product to fit must ensure their inputs are bounded
        // (`|coord| < 2^127`-ish for a `p ≈ 2^250` quaternion
        // algebra); wide-input callers should use
        // `Element<N>::mul_direct` at a width with headroom.
        //
        // Previously this was a `debug_assert!` + release-truncate,
        // which let silent numerical corruption escape into
        // downstream ideals and action matrices (see Task #28 for
        // one concrete instance).
        let narrow = |v: BigInt<8>| -> BigInt<4> {
            let ct: subtle::CtOption<BigInt<4>> = v.into();
            assert!(
                bool::from(ct.is_some()),
                "Element<4>::mul: quaternion coordinate overflow after GCD \
                 normalization (product exceeds BigInt<4>'s 256-bit budget; \
                 caller should use Element<N>::mul_direct at a wider N)"
            );
            ct.unwrap()
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

impl<const N: usize> PartialEq for Element<N> {
    fn eq(&self, other: &Self) -> bool {
        // Cross-multiply: a1/r1 == a2/r2 iff a1*r2 == a2*r1.
        self.a.0.ct_mul(&other.denom.0) == other.a.0.ct_mul(&self.denom.0)
            && self.b.0.ct_mul(&other.denom.0) == other.b.0.ct_mul(&self.denom.0)
            && self.c.0.ct_mul(&other.denom.0) == other.c.0.ct_mul(&self.denom.0)
            && self.d.0.ct_mul(&other.denom.0) == other.d.0.ct_mul(&self.denom.0)
    }
}

impl<const N: usize> Eq for Element<N> {}

impl<const N: usize> fmt::Debug for Element<N> {
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
        assert!(Element::<4>::ZERO.is_zero());
    }

    #[test]
    fn conjugate() {
        let e = Element::<4>::from_i64(1, 2, 3, 4);
        let conj = e.conjugate();
        assert_eq!(conj.a, Coordinate::from_i64(1));
        assert_eq!(conj.b, Coordinate::from_i64(-2));
        assert_eq!(conj.c, Coordinate::from_i64(-3));
        assert_eq!(conj.d, Coordinate::from_i64(-4));
    }

    #[test]
    fn trace() {
        let e = Element::<4>::from_i64(5, 2, 3, 4);
        let (tr_num, tr_den) = e.trace();
        assert_eq!(tr_num, BigInt::from(10i64));
        assert_eq!(tr_den, BigInt::ONE);
    }

    #[test]
    fn norm() {
        // nrd(1 + 2i + 3j + 4k) with p from NIST-I.
        // = 1 + 4 + p*(9 + 16) = 5 + 25p
        let e = Element::<4>::from_i64(1, 2, 3, 4);
        let (n_num, n_den) = e.norm();
        let p: BigInt<8> = P_WIDE;
        let expected = BigInt::<8>::from(5i64).ct_add(&BigInt::<8>::from(25i64).ct_mul(&p));
        assert_eq!(n_num, expected);
        assert_eq!(n_den, BigInt::<8>::ONE);
    }

    #[test]
    fn norm_w_matches_norm() {
        // `norm_w::<W>` at W = N' > 8 should match `norm` (widened).
        let e = Element::<4>::from_i64(1, 2, 3, 4);
        let (num8, den8) = e.norm();
        let (num22, den22) = e.norm_w::<22>();
        assert_eq!(num22, num8.widen::<22>());
        assert_eq!(den22, den8.widen::<22>());
    }

    #[test]
    fn mul_i_squared() {
        // i² = -1
        let unit_i = Element::<4>::from_i64(0, 1, 0, 0);
        let result = unit_i.mul(&unit_i);
        assert_eq!(result, Element::<4>::from_i64(-1, 0, 0, 0));
    }

    #[test]
    fn mul_ij_eq_k() {
        // ij = k
        let unit_i = Element::<4>::from_i64(0, 1, 0, 0);
        let unit_j = Element::<4>::from_i64(0, 0, 1, 0);
        let result = unit_i.mul(&unit_j);
        assert_eq!(result, Element::<4>::from_i64(0, 0, 0, 1));
    }

    #[test]
    fn mul_ji_eq_neg_k() {
        // ji = -k
        let unit_i = Element::<4>::from_i64(0, 1, 0, 0);
        let unit_j = Element::<4>::from_i64(0, 0, 1, 0);
        let result = unit_j.mul(&unit_i);
        assert_eq!(result, Element::<4>::from_i64(0, 0, 0, -1));
    }

    #[test]
    fn norm_is_multiplicative() {
        let alpha = Element::<4>::from_i64(1, 2, 0, 1);
        let beta = Element::<4>::from_i64(3, 0, 1, 0);
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
        let e = Element::<4>::from_i64(1, 2, 3, 4);
        let conj = e.conjugate();
        let product = e.mul(&conj).normalized();

        let (_n_num, _n_den) = e.norm();
        // product should be scalar: (nrd, 0, 0, 0).
        // The product denom is r², and n_den is also r².
        assert_eq!(product.b, Coordinate::from_i64(0));
        assert_eq!(product.c, Coordinate::from_i64(0));
        assert_eq!(product.d, Coordinate::from_i64(0));
    }

    #[test]
    fn addition() {
        let a = Element::<4>::from_i64(1, 2, 3, 4);
        let b = Element::<4>::from_i64(5, 6, 7, 8);
        let sum = a.add(&b);
        assert_eq!(sum, Element::<4>::from_i64(6, 8, 10, 12));
    }

    #[test]
    fn subtraction() {
        let a = Element::<4>::from_i64(5, 6, 7, 8);
        let b = Element::<4>::from_i64(1, 2, 3, 4);
        let diff = a.sub(&b);
        assert_eq!(diff, Element::<4>::from_i64(4, 4, 4, 4));
    }

    #[test]
    fn normalize_gcd() {
        let mut e = Element::<4>::new(
            Coordinate::from_i64(2),
            Coordinate::from_i64(4),
            Coordinate::from_i64(6),
            Coordinate::from_i64(8),
            Denominator::TWO,
        );
        e.normalize();
        assert_eq!(e, Element::<4>::from_i64(1, 2, 3, 4));
        assert_eq!(e.denom, Denominator::ONE);
    }

    #[test]
    fn equality_across_denominators() {
        let a = Element::<4>::new(
            Coordinate::from_i64(2),
            Coordinate::from_i64(4),
            Coordinate::from_i64(0),
            Coordinate::from_i64(0),
            Denominator::TWO,
        );
        let b = Element::<4>::from_i64(1, 2, 0, 0);
        assert_eq!(a, b);
    }
}
