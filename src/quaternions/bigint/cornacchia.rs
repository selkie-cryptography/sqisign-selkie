//! Cornacchia's algorithm on [`BigInt<N>`][super::BigInt]: solves
//! `x² + qy² = m` over the integers, with `m` prime and `0 ≤ q ≤ m`.

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Cornacchia's algorithm: solve x² + qy² = m for integers x, y.
    ///
    /// Given a prime `m` and a positive integer `q` with 0 ≤ q ≤ m,
    /// finds integers x, y such that x² + qy² = m, or returns `None`
    /// if no solution exists (which happens when -q is not a quadratic
    /// residue mod m).
    ///
    /// # Algorithm
    ///
    /// This implements [Alg. 3.11] from the SQIsign specification,
    /// following the standard Cornacchia algorithm as described in
    /// Cohen's "A Course in Computational Algebraic Number Theory"
    /// and [Morain-Nicolas][MN90]:
    ///
    /// 1. Check that -q is a quadratic residue mod m (Legendre symbol).
    /// 2. Compute r₀ = √(-q) mod m via [`modular_sqrt`](Self::modular_sqrt).
    /// 3. Run the Euclidean algorithm on (m, r₀), reducing until s ≤ √m.
    /// 4. Set x = s, compute y² = (m - x²) / q, verify y is an integer.
    ///
    /// # Spec discrepancy
    ///
    /// The spec's Algorithm 3.11, line 9, shows `s ← q`, but this does
    /// not match the standard Cornacchia algorithm or the C reference
    /// implementation (`ibz_cornacchia_prime` in `integers.c`), both of
    /// which initialize the Euclidean chain as (m, r₀). We follow the
    /// standard algorithm and the C reference here.
    ///
    /// The C reference initializes `r1 = p` (the prime) and
    /// `r2 = sqrt(-n mod p)`, then reduces until `r0² < p` — exactly
    /// the `(m, r₀)` approach.
    ///
    /// TODO: Open a bug against the SQIsign spec reporting the line 9
    /// discrepancy (`s ← q` vs the correct `s ← m`).
    ///
    /// # Side-channel considerations
    ///
    /// Not constant-time — the Euclidean reduction loop has
    /// data-dependent iteration count.
    ///
    /// TODO(ct): Make constant-time before production use. Called on
    /// secret-derived values during signing (via RepresentInteger).
    ///
    /// # Special cases
    ///
    /// - q = 0: checks if m is a perfect square.
    /// - m = 2, q = 1: returns (1, 1) since 1² + 1² = 2.
    ///
    /// [Alg. 3.11]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.11
    /// [MN90]: https://doi.org/10.1016/0022-314X(90)90136-F
    pub fn cornacchia(q: &Self, m: &Self) -> Option<(Self, Self)> {
        // Special case: q = 0 reduces to x² = m.
        if bool::from(q.is_zero()) {
            let s = m.sqrt_floor()?;
            return if s.ct_mul(&s) == *m {
                Some((s, Self::ZERO))
            } else {
                None
            };
        }

        // Special case: m = 2, q = 1 (spec lines 3-7).
        if *m == Self::TWO {
            return if *q == Self::ONE {
                Some((Self::ONE, Self::ONE))
            } else {
                None
            };
        }

        // Step 1: Check Legendre symbol — -q must be a QR mod m.
        let neg_q = m.ct_sub(&q.vt_mod(m));
        let exp = m.ct_sub(&Self::ONE) >> 1;
        let legendre = Self::pow_mod(&neg_q, &exp, m);
        if legendre != Self::ONE && !bool::from(neg_q.vt_mod(m).is_zero()) {
            return None;
        }

        // Step 2: r₀ = √(-q) mod m.
        let r0 = Self::modular_sqrt(&neg_q, m)?;

        // Step 3: Euclidean reduction on (m, r₀).
        //
        // Initialize r = m, s = r₀. Reduce until s ≤ √m (equivalently
        // s² ≤ m). After the loop, x = s is the first remainder whose
        // square does not exceed m.
        let mut r = *m;
        let mut s = r0;
        let bound = m.sqrt_floor()?;

        let mut i = 0;
        while i < 2 * Self::BITS {
            if s <= bound {
                break;
            }
            let tmp = r.vt_mod(&s);
            r = s;
            s = tmp;
            i += 1;
        }

        // Step 4: x = s, y² = (m - x²) / q.
        let x = s;
        let x_sq = x.ct_mul(&x);
        if x_sq > *m {
            return None;
        }

        // (m - x²) must be divisible by q.
        let (y_sq, rem) = m.ct_sub(&x_sq).vt_div_rem(q);
        if !bool::from(rem.is_zero()) {
            return None;
        }

        // y² must be a perfect square.
        let y = y_sq.sqrt_floor()?;
        if y.ct_mul(&y) != y_sq {
            return None;
        }

        // Final verification: x² + qy² = m.
        let check = x_sq.ct_add(&q.ct_mul(&y.ct_mul(&y)));
        if check == *m { Some((x, y)) } else { None }
    }

    /// Cornacchia's algorithm at a wider working width `W`.
    ///
    /// Widens `q` and `m` to `BigInt<W>` for the modular arithmetic
    /// (Legendre symbol, modular square root, Euclidean reduction),
    /// then narrows the result back to `BigInt<N>`. Use when `m` does
    /// not fit in `N` limbs.
    pub fn cornacchia_w<const W: usize>(q: &Self, m: &Self) -> Option<(Self, Self)> {
        const {
            assert!(
                W >= N,
                "cornacchia_w: working width W must be >= storage width N"
            )
        };
        let q_w: BigInt<W> = q.widen();
        let m_w: BigInt<W> = m.widen();
        let (x_w, y_w) = BigInt::<W>::cornacchia(&q_w, &m_w)?;
        Some((x_w.narrow_to::<N>()?, y_w.narrow_to::<N>()?))
    }

    /// Cornacchia at the tightest working width that still holds `m`,
    /// narrowing from the caller's ceiling `WMAX` when `m` is small.
    ///
    /// Every modular multiply inside [`cornacchia`](Self::cornacchia)
    /// (Legendre via [`pow_mod`](Self::pow_mod), the square root via
    /// [`modular_sqrt`](Self::modular_sqrt)) now runs in Montgomery form,
    /// so nothing truncates and the only width requirement is that `W`
    /// holds `m`: `64*W >= bits(m)`. The chosen width is
    /// `ceil(bits(m)/64) + 1` (one limb of margin); `q <= m` fits in the
    /// same width. `WMAX` is the caller's already-proven-safe ceiling,
    /// and the dispatch only narrows below it and falls back to it, so
    /// the result is identical to
    /// [`cornacchia_w`](Self::cornacchia_w)`::<WMAX>` for every input.
    /// Cost scales `O(W^2)` in the exponentiation and the Euclidean
    /// reduction, so the narrowing is large: an `m` of 273 bits runs at
    /// `W = 6` rather than `WMAX = 17`.
    ///
    /// # Constant-time
    ///
    /// `bits(m)` selects the width, so this is variable-time in the
    /// magnitude of `m`. Cornacchia is part of the norm-equation prime
    /// search of `represent_integer`, which is already the variable-time
    /// Cornacchia/Basso side-channel surface scheduled for the
    /// constant-time pass; this adds no new class of leak and is closed
    /// there wholesale.
    #[must_use]
    pub fn cornacchia_auto<const WMAX: usize>(q: &Self, m: &Self) -> Option<(Self, Self)> {
        let bits = Self::mag_bitsize(&m.limbs) as usize;
        let need = bits.div_ceil(64) + 1;

        if WMAX > 6 && need <= 6 {
            return Self::corn_at::<6>(q, m);
        }
        if WMAX > 7 && need <= 7 {
            return Self::corn_at::<7>(q, m);
        }
        if WMAX > 8 && need <= 8 {
            return Self::corn_at::<8>(q, m);
        }
        if WMAX > 9 && need <= 9 {
            return Self::corn_at::<9>(q, m);
        }
        if WMAX > 10 && need <= 10 {
            return Self::corn_at::<10>(q, m);
        }
        if WMAX > 12 && need <= 12 {
            return Self::corn_at::<12>(q, m);
        }
        if WMAX > 14 && need <= 14 {
            return Self::corn_at::<14>(q, m);
        }
        if WMAX > 16 && need <= 16 {
            return Self::corn_at::<16>(q, m);
        }

        Self::corn_at::<WMAX>(q, m)
    }

    /// Resizes `q` and `m` to width `W` (the dispatch guarantees `W`
    /// holds them), runs [`cornacchia`](Self::cornacchia), and resizes
    /// the result back to `N`.
    #[must_use]
    fn corn_at<const W: usize>(q: &Self, m: &Self) -> Option<(Self, Self)> {
        let q_w = q.resize_unchecked::<W>();
        let m_w = m.resize_unchecked::<W>();
        let (x_w, y_w) = BigInt::<W>::cornacchia(&q_w, &m_w)?;
        Some((x_w.resize_unchecked::<N>(), y_w.resize_unchecked::<N>()))
    }
}
