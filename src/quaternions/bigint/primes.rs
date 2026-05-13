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
        let d = n_minus_1.shr(s);

        // Deterministic witnesses sufficient for values up to 3.3×10²⁴.
        let witnesses: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
        let num_rounds = (rounds as usize).min(witnesses.len());

        for &a_val in &witnesses[..num_rounds] {
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
}
