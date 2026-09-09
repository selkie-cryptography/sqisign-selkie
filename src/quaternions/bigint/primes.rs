//! Miller-Rabin probabilistic primality testing for
//! [`BigInt<N>`][super::BigInt].

use super::{BigInt, MontReducer};

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
    /// # Width
    ///
    /// The whole test (witness exponentiation and the strong-prime
    /// square-chain) runs in Montgomery form via
    /// [`MontReducer::is_strong_probable_prime`], whose internal
    /// `2N`-limb products are folded back by REDC without truncation.
    /// It is therefore correct for any `self` that fits in `N` limbs;
    /// there is no `64*N >= 2*bits` floor. (The earlier canonical-form
    /// square-chain squared with a fixed-width `N`-limb multiply that
    /// truncated `x²`, which did impose that floor.)
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

        // Small-prime trial-division pre-screen. A candidate divisible by a
        // small prime (and larger than it) is composite, so the witness
        // exponentiation below would reject it too: the accept/reject
        // decision, and every downstream signature, is unchanged. Trial
        // division is verdict-preserving for any prime set, so we extend
        // past the reference's mini-gmp `mpz_probab_prime_p` (which screens
        // only `3*5*...*29 = 0xc0cfd797` to fit `u32`); real GMP screens a
        // far larger table for the same reason. The win is rejecting the
        // bulk of random composites with a single-limb `vt_mod` per word
        // plus a few `u64` remainders, instead of building a `MontReducer`
        // (the `R^2` setup) and running a modular exponentiation.
        //
        // Each word is the product of its primes and fits in `u64`, so
        // `r = self mod word` carries `self mod p` for every prime `p` in
        // that word (each divides it), recovered as `r % p`. Primes 3..29
        // catch ~68% of odd composites; adding 31..73 reaches ~75%. The
        // second word runs only on candidates the first did not reject.
        // The `self != p` guard keeps the predicate correct when `self` is
        // itself one of these primes.
        const TRIAL_WORDS: [(u64, &[u64]); 2] = [
            (
                3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29,
                &[3, 5, 7, 11, 13, 17, 19, 23, 29],
            ),
            (
                31u64 * 37 * 41 * 43 * 47 * 53 * 59 * 61 * 67 * 71 * 73,
                &[31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73],
            ),
        ];
        for (word, primes) in TRIAL_WORDS {
            let r = self.vt_mod(&Self::from_u64(word)).as_limbs()[0];
            for &p in primes {
                if r.is_multiple_of(p) && *self != Self::from_u64(p) {
                    return false;
                }
            }
        }

        // Build the Montgomery context once and delegate. (Per the CT
        // note on `MontReducer`: this within-function context is safe even
        // when `self` is a secret-derived prime candidate.)
        let ctx = MontReducer::<N>::new(self).expect("self is odd > 1 by the early returns above");
        self.is_probable_prime_with_ctx(rounds, &ctx)
    }

    /// Same as [`is_probable_prime`](Self::is_probable_prime) but with
    /// a caller-provided `MontReducer`. The caller must ensure
    /// `reducer` is built for `self` — passing a mismatched reducer
    /// returns garbage (debug-asserted).
    ///
    /// Use case: a fixed modulus whose `MontReducer` is a compile-time
    /// `const`, saving the `MontReducer::new` cost per primality test
    /// (dominated by the `128·N` doublings that compute `R²`).
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
        let n_minus_1 = self.vt_sub(&Self::ONE);
        let s = n_minus_1.trailing_zeros();
        let d = n_minus_1 >> s;

        // Deterministic witnesses sufficient for values up to 3.3×10²⁴.
        let witnesses: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
        let num_rounds = (rounds as usize).min(witnesses.len());

        for &a_val in &witnesses[..num_rounds] {
            let a = Self::from_u64(a_val);
            if a >= *self {
                continue;
            }

            if !ctx.is_strong_probable_prime(&a, &d, s, &n_minus_1) {
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
    /// candidate, narrowing from the caller's ceiling `WMAX` when the
    /// candidate is small.
    ///
    /// The whole test runs in Montgomery form (see
    /// [`is_probable_prime`](Self::is_probable_prime)), which never
    /// truncates, so the only width requirement is that `W` holds the
    /// candidate: `64*W >= bits(self)`. The chosen width is therefore
    /// `ceil(bits/64) + 1` (one limb of margin). `WMAX` is the caller's
    /// already-proven-safe ceiling (the width it would otherwise fix
    /// unconditionally); the dispatch only narrows *below* `WMAX` and
    /// falls back to it, so the primality decision is identical to
    /// [`is_probable_prime_w`](Self::is_probable_prime_w)`::<WMAX>` for
    /// every candidate. The win is that the common candidates (the
    /// norm-equation primes are 273 to ~515 bits, far below the
    /// worst-case `WMAX`) run their `O(W^2)` Montgomery arithmetic at a
    /// much smaller width: a 273-bit candidate narrows from `WMAX = 17`
    /// to `W = 6`, an ~8x reduction in limb-mults per squaring.
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
        let need = bits.div_ceil(64) + 1;

        if WMAX > 6 && need <= 6 {
            return self.prime_at::<6>(rounds);
        }
        if WMAX > 7 && need <= 7 {
            return self.prime_at::<7>(rounds);
        }
        if WMAX > 8 && need <= 8 {
            return self.prime_at::<8>(rounds);
        }
        if WMAX > 9 && need <= 9 {
            return self.prime_at::<9>(rounds);
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
