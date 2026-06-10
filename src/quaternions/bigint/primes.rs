//! Miller-Rabin probabilistic primality testing for
//! [`BigInt<N>`][super::BigInt].

use subtle::{Choice, ConstantTimeEq};

use super::{BigInt, MontReducer};

/// Miller-Rabin witness bases, reused as the small-prime trial set in
/// [`BigInt::is_probable_prime`]'s pre-screen. Deterministic for values
/// below 3.3*10^24; probabilistic above.
const MR_WITNESSES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

impl<const N: usize> BigInt<N> {
    /// Miller-Rabin probabilistic primality test.
    ///
    /// Returns `true` if `self` is probably prime. Uses `rounds`
    /// deterministic witnesses (2, 3, 5, 7, 11, 13, ...) for small
    /// round counts, which gives a deterministic result for values
    /// below certain bounds.
    ///
    /// Based on G.L. Miller, ["Riemann's hypothesis and tests for
    /// primality"][Miller76] (JCSS 13(3), 1976) and M.O. Rabin,
    /// "Probabilistic algorithm for testing primality"
    /// (J. Number Theory 12(1), 1980).
    ///
    /// [Miller76]: https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
    ///
    /// # Width requirement
    ///
    /// Miller-Rabin uses [`pow_mod`](Self::pow_mod) and direct
    /// `ct_mul` on values up to `self`. The caller must ensure
    /// `64*N >= 2*bits(self)` — otherwise the squarings silently
    /// truncate and the test returns wrong answers (in practice,
    /// false negatives on primes). For larger candidates use
    /// [`is_probable_prime_w`](Self::is_probable_prime_w).
    ///
    /// WARNING: Not constant-time — the number of iterations and
    /// modular exponentiations depend on the value.
    ///
    /// TODO(ct): Make constant-time before production use. Called on
    /// secret-derived values during signing (via RepresentInteger).
    pub fn is_probable_prime(&self, rounds: u32) -> bool {
        if bool::from(self.is_negative()) || bool::from(self.is_zero()) {
            return false;
        }
        if *self == Self::ONE {
            return false;
        }
        if *self == Self::TWO || *self == Self::THREE {
            return true;
        }
        if bool::from(self.is_even()) {
            return false;
        }

        // Trial-divide by the small odd primes Miller-Rabin uses as
        // witnesses, before the Montgomery setup and exponentiations. A
        // multiple of any used base `p` (with `self != p`) fails that
        // base's round regardless, so this rejects exactly the same
        // composites -- without the modular exponentiations (and the
        // `MontReducer::new` cost) that most rejected candidates pay.
        if bool::from(self.has_small_witness_factor(rounds)) {
            return false;
        }

        // Build the Montgomery context once and delegate. (Per the CT
        // note on `MontReducer`: this within-function context is safe even
        // when `self` is a secret-derived prime candidate.)
        let ctx = MontReducer::<N>::new(self).expect("self is odd > 1 by the early returns above");
        self.is_probable_prime_with_ctx(rounds, &ctx)
    }

    /// Returns whether one of the first `rounds` Miller-Rabin witness
    /// bases ([`MR_WITNESSES`]) divides `self`, excluding the case
    /// `self` equals that base.
    ///
    /// Pre-filter for [`is_probable_prime`](Self::is_probable_prime):
    /// the bases are small primes, and any composite divisible by a base
    /// `p` is rejected by `p`'s Miller-Rabin round anyway (for `self >
    /// p`, `p^d` is divisible by `p` so it can be neither `1` nor `-1`
    /// mod `self`). Screening here is therefore output-identical to
    /// Miller-Rabin while skipping the exponentiations for the common
    /// case of a composite with a tiny factor.
    ///
    /// # Constant-time
    ///
    /// Variable-time. `TODO(ct)`: gates the variable-time Miller-Rabin
    /// test on a secret-derived candidate (Algorithm 4.2 line 16, via
    /// RepresentInteger). The scan itself takes a data-independent path
    /// -- a fixed base set, no early exit, `ct_mod` -- so it reveals no
    /// more than the surrounding test it feeds.
    fn has_small_witness_factor(&self, rounds: u32) -> Choice {
        let num_rounds = (rounds as usize).min(MR_WITNESSES.len());

        let mut composite = Choice::from(0u8);
        for &p in &MR_WITNESSES[..num_rounds] {
            let p_big = Self::from_u64(p);
            let divides = self.ct_mod(&p_big).is_zero();
            let is_self = self.ct_eq(&p_big);
            composite |= divides & !is_self;
        }

        composite
    }

    /// Same as [`is_probable_prime`](Self::is_probable_prime) but with
    /// a caller-provided `MontReducer`. The caller must ensure
    /// `reducer` is built for `self` — passing a mismatched reducer
    /// returns garbage (debug-asserted).
    ///
    /// Use case: known-fixed moduli (e.g. `params::D_MIX_W18_MOD`)
    /// where the `MontReducer` is built at compile time as a `const` —
    /// saves the ~tens of µs `MontReducer::new` cost per primality
    /// test (dominated by the `128·N` doublings to compute `R²`).
    ///
    /// Internal API. Crate-private because it exposes a `MontReducer`,
    /// which is implementation detail; external callers use
    /// [`Self::is_probable_prime`].
    pub(crate) fn is_probable_prime_with_ctx(&self, rounds: u32, ctx: &MontReducer<N>) -> bool {
        debug_assert_eq!(
            ctx.modulus_limbs(),
            &self.limbs,
            "is_probable_prime_with_ctx: ctx must be for `self`",
        );
        if bool::from(self.is_negative()) || bool::from(self.is_zero()) {
            return false;
        }
        if *self == Self::ONE {
            return false;
        }
        if *self == Self::TWO || *self == Self::THREE {
            return true;
        }
        if bool::from(self.is_even()) {
            return false;
        }

        // Write self - 1 = 2^s · d with d odd.
        let n_minus_1 = self.ct_sub(&Self::ONE);
        let s = n_minus_1.two_adic_val();
        let d = n_minus_1 >> s;

        let num_rounds = (rounds as usize).min(MR_WITNESSES.len());

        for &a_val in &MR_WITNESSES[..num_rounds] {
            let a = Self::from_u64(a_val);
            if a >= *self {
                continue;
            }

            let mut x = ctx.pow(&a, &d);
            if x == Self::ONE || x == n_minus_1 {
                continue;
            }

            let mut composite = true;
            for _r in 1..s {
                x = x.ct_mul(&x).ct_mod(self);
                if x == n_minus_1 {
                    composite = false;
                    break;
                }
            }
            if composite {
                return false;
            }
        }

        true
    }

    /// Miller-Rabin primality test at a wider working width `W`.
    ///
    /// Widens `self` to `BigInt<W>` and runs
    /// [`is_probable_prime`](Self::is_probable_prime) at that width.
    /// Use when `64*N < 2*bits(self)` would otherwise silently
    /// truncate the Miller-Rabin witness exponentiations (producing
    /// false negatives on actual primes).
    ///
    /// See [`pow_mod_w`](Self::pow_mod_w) for the width constraints.
    pub fn is_probable_prime_w<const W: usize>(&self, rounds: u32) -> bool {
        const {
            assert!(
                W >= N,
                "is_probable_prime_w: working width W must be >= storage width N"
            )
        };
        let self_w: BigInt<W> = self.widen();
        self_w.is_probable_prime(rounds)
    }

    /// Miller-Rabin at the tightest working width that still holds the
    /// Montgomery exponentiation, narrowing from the caller's ceiling
    /// `WMAX` when the candidate is small.
    ///
    /// The Montgomery REDC truncates unless `64*W >= 2*bits(self)` (the
    /// confirmed floor; below it the test silently mis-decides), so the
    /// chosen width is `ceil(2*bits/64) + 1` limb of margin.  `WMAX` is
    /// the caller's already-proven-safe ceiling (the width it would
    /// otherwise fix unconditionally); the dispatch only narrows *below*
    /// `WMAX` and falls back to it, so the primality decision is
    /// identical to
    /// [`is_probable_prime_w`](Self::is_probable_prime_w)`::<WMAX>`
    /// for every candidate.  The win is that the common small candidates
    /// (the norm-equation primes are usually far below the worst-case
    /// `WMAX`) run their `O(W^2)` Montgomery arithmetic at a much smaller
    /// width.
    ///
    /// # Constant-time
    ///
    /// The candidate's bit length selects the width, so this is
    /// variable-time in the candidate magnitude.  Primality on
    /// secret-derived candidates is already variable-time (the
    /// norm-equation prime search is the Cornacchia/Basso side-channel
    /// surface scheduled for the constant-time pass); this adds no new
    /// class of leak and is closed there wholesale.
    #[must_use]
    pub fn is_probable_prime_auto<const WMAX: usize>(&self, rounds: u32) -> bool {
        let bits = Self::mag_bitsize(&self.limbs) as usize;
        let need = (2 * bits).div_ceil(64) + 1;

        if WMAX > 8 && need <= 8 {
            return self.prime_at::<8>(rounds);
        }
        if WMAX > 10 && need <= 10 {
            return self.prime_at::<10>(rounds);
        }
        if WMAX > 12 && need <= 12 {
            return self.prime_at::<12>(rounds);
        }
        if WMAX > 14 && need <= 14 {
            return self.prime_at::<14>(rounds);
        }
        if WMAX > 16 && need <= 16 {
            return self.prime_at::<16>(rounds);
        }
        if WMAX > 18 && need <= 18 {
            return self.prime_at::<18>(rounds);
        }
        if WMAX > 20 && need <= 20 {
            return self.prime_at::<20>(rounds);
        }
        if WMAX > 24 && need <= 24 {
            return self.prime_at::<24>(rounds);
        }

        self.prime_at::<WMAX>(rounds)
    }

    /// Resizes the candidate to width `W` (value-preserving: the caller
    /// guarantees `W` holds it) and runs
    /// [`is_probable_prime`](Self::is_probable_prime).
    #[must_use]
    fn prime_at<const W: usize>(&self, rounds: u32) -> bool {
        self.resize_unchecked::<W>().is_probable_prime(rounds)
    }
}
