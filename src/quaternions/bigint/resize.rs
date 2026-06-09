//! Width conversion between [`BigInt<N>`][super::BigInt] sizes:
//! [`widen`][BigInt::widen] and [`narrow_to`][BigInt::narrow_to],
//! plus the `From`/`TryFrom`-style conversions between `BigInt<4>`
//! and `BigInt<8>`.

use subtle::{Choice, CtOption};

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Widen to `BigInt<W>` by zero-extending the upper limbs.
    ///
    /// The value is preserved exactly; the upper `W - N` limbs are zero.
    /// Compile-time error if `W < N`.
    #[must_use]
    pub fn widen<const W: usize>(self) -> BigInt<W> {
        // Compile-time check: destination must be at least as wide.
        const {
            assert!(
                W >= N,
                "widen: destination width W must be >= source width N"
            )
        };
        let mut limbs = [0u64; W];
        let mut i = 0;
        while i < N {
            limbs[i] = self.limbs[i];
            i += 1;
        }
        BigInt::<W> {
            sign: self.sign,
            limbs,
        }
    }

    /// Narrow to `BigInt<T>`, returning `None` if the value does not
    /// fit (any of the upper `N - T` limbs are non-zero).
    ///
    /// Compile-time error if `T > N`.
    #[must_use]
    pub fn narrow_to<const T: usize>(self) -> Option<BigInt<T>> {
        const {
            assert!(
                T <= N,
                "narrow_to: destination width T must be <= source width N"
            )
        };
        let mut overflow = 0u64;
        let mut i = T;
        while i < N {
            overflow |= self.limbs[i];
            i += 1;
        }
        if overflow != 0 {
            return None;
        }
        let mut limbs = [0u64; T];
        let mut j = 0;
        while j < T {
            limbs[j] = self.limbs[j];
            j += 1;
        }
        Some(BigInt::<T> {
            sign: self.sign,
            limbs,
        })
    }

    /// Copies `self` into width `W`, taking the low `min(N, W)` limbs and
    /// preserving the sign, with no fit assertion.
    ///
    /// Unlike [`Self::widen`] / [`Self::narrow_to`] this places no
    /// compile-time relation on `N` and `W`, so it type-checks inside the
    /// runtime-dispatched width ladder of [`BigInt::is_probable_prime_auto`],
    /// where the dispatch provably picks a width that holds the candidate
    /// value (any dropped high limbs are zero).  Not for general use.
    #[must_use]
    pub(super) fn resize_unchecked<const W: usize>(self) -> BigInt<W> {
        let mut limbs = [0u64; W];
        let n = if N < W { N } else { W };

        let mut i = 0;
        while i < n {
            limbs[i] = self.limbs[i];
            i += 1;
        }

        BigInt::<W> {
            sign: self.sign,
            limbs,
        }
    }
}

/// Widen: zero-extend a four-limb integer to eight limbs.
impl From<BigInt<4>> for BigInt<8> {
    fn from(small: BigInt<4>) -> Self {
        small.widen()
    }
}

/// Narrow: truncate an eight-limb integer to four limbs.
///
/// Returns a `CtOption` — the result is always computed (constant time),
/// but the `is_some` flag indicates whether the value actually fits.
///
/// For ergonomic use in non-constant-time code, see `BigInt<8>::narrow`.
impl From<BigInt<8>> for CtOption<BigInt<4>> {
    fn from(wide: BigInt<8>) -> Self {
        let mut overflow = 0u64;
        overflow |= wide.limbs[4];
        overflow |= wide.limbs[5];
        overflow |= wide.limbs[6];
        overflow |= wide.limbs[7];
        let fits = Choice::from((overflow == 0) as u8);
        let narrow = BigInt::<4> {
            sign: wide.sign,
            limbs: [wide.limbs[0], wide.limbs[1], wide.limbs[2], wide.limbs[3]],
        };
        CtOption::new(narrow, fits)
    }
}

impl BigInt<8> {
    /// Narrow to `BigInt<4>`, returning `None` if the upper limbs are
    /// non-zero.
    ///
    /// This is a convenience wrapper around the constant-time
    /// `From<BigInt<8>> for CtOption<BigInt<4>>` conversion.
    pub fn narrow(self) -> Option<BigInt<4>> {
        Option::from(CtOption::<BigInt<4>>::from(self))
    }
}
