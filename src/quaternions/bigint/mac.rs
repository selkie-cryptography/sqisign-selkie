//! Carry-save multiply-accumulate for sum-of-products on
//! [`BigInt<N>`][super::BigInt].
//!
//! A sum-of-products `acc = sum_i t_i` written as a chain of
//! [`ct_add`](BigInt::ct_add) calls pays the full sign-and-magnitude
//! merge (one `mag_add`, one `mag_sub`, one `mag_negate`, and three
//! `mag_select`s, plus zero canonicalization) on *every* term. The
//! carry-save accumulator below sorts each term into one of two
//! unsigned running totals by sign and adds it with a single
//! carry-propagating `mag_add`, deferring the sign-and-magnitude merge
//! to one final subtraction after the loop. One sign merge instead of
//! `k`.
//!
//! # Correctness (byte-identical to the `ct_add` chain)
//!
//! Each running total is an `N`-limb magnitude plus one `u64` headroom
//! word. Each term magnitude is `< 2^(64N)`, so summing up to `2^64`
//! terms keeps the carry out of limb `N - 1` within that headroom word
//! and never overflows. The final signed difference `pos - neg`, taken
//! modulo `2^(64N)` with the sign of the larger magnitude, equals the
//! true signed integer sum of the terms whenever that sum fits in `N`
//! limbs.
//!
//! The `ct_add` chain it replaces likewise equals the true signed sum
//! whenever no same-sign partial overflows `N` limbs (it discards the
//! top carry otherwise). At every call site the width `N` is chosen so
//! the exact sum fits — that is the whole reason those sites widen to
//! `W` before accumulating — so the two agree byte-for-byte. The KAT
//! and proptest suites pin this.
//!
//! Constant-time: every term is processed by the same straight-line
//! add regardless of value; the sign only selects which of the two
//! accumulators receives it, via a branchless mask.

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Sum of signed terms via a carry-save accumulator.
    ///
    /// Computes `sum_i terms[i]` with a single sign-and-magnitude merge
    /// at the end, in place of one [`ct_add`](BigInt::ct_add) per term.
    /// Byte-identical to the equivalent `ct_add` chain whenever the
    /// exact sum fits in `N` limbs (see the module docs). Constant-time.
    #[must_use]
    pub fn mac_sum(terms: &[Self]) -> Self {
        let mut pos = [0u64; N];
        let mut pos_hi: u64 = 0;
        let mut neg = [0u64; N];
        let mut neg_hi: u64 = 0;
        for t in terms {
            // Branchless: mask the magnitude into the matching
            // accumulator. The non-matching accumulator gets a zero
            // add, so the dataflow stays value-independent.
            let pos_mask = t.sign.wrapping_sub(1); // all-ones iff sign==0, else 0
            let neg_mask = !pos_mask;
            let mut cp: u64 = 0;
            let mut cn: u64 = 0;
            let mut i = 0;
            while i < N {
                let limb = t.limbs[i];
                let (s1, c1) = pos[i].overflowing_add(limb & pos_mask);
                let (s2, c2) = s1.overflowing_add(cp);
                pos[i] = s2;
                cp = (c1 | c2) as u64;

                let (d1, e1) = neg[i].overflowing_add(limb & neg_mask);
                let (d2, e2) = d1.overflowing_add(cn);
                neg[i] = d2;
                cn = (e1 | e2) as u64;
                i += 1;
            }
            pos_hi = pos_hi.wrapping_add(cp);
            neg_hi = neg_hi.wrapping_add(cn);
        }
        Self::mac_finalize(&pos, pos_hi, &neg, neg_hi)
    }

    /// Final signed difference `pos - neg`, reduced to `N` limbs.
    ///
    /// `pos`/`neg` are `N`-limb magnitudes each carrying one `u64`
    /// headroom word (`pos_hi`/`neg_hi`). Returns the signed `N`-limb
    /// result with the sign of the larger, matching the
    /// sign-and-magnitude convention of [`ct_add`](BigInt::ct_add).
    /// Constant-time.
    #[inline]
    fn mac_finalize(pos: &[u64; N], pos_hi: u64, neg: &[u64; N], neg_hi: u64) -> Self {
        // Wide subtraction over N+1 words: borrow == 1 iff pos < neg.
        let (diff_a, borrow) = Self::mag_sub(pos, neg);
        let (hi_a, hb) = pos_hi.overflowing_sub(neg_hi);
        let (_hi_a2, hb2) = hi_a.overflowing_sub(borrow);
        let pos_ge = 1 - (hb | hb2) as u64;

        // |neg - pos| = two's-complement negation of (pos - neg) over
        // the low N limbs; the headroom word is not part of the result.
        let diff_b = Self::mag_negate(&diff_a);

        let limbs = Self::mag_select(&diff_b, &diff_a, pos_ge);

        // Sign: negative iff neg > pos; canonical zero is non-negative.
        let result_sign = (1 - pos_ge) & (1 - Self::mag_is_zero(&limbs));
        Self {
            sign: result_sign,
            limbs,
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::BigInt;

    /// Reference: the per-term `ct_add` chain that `mac_sum` replaces.
    fn chain_sum<const N: usize>(terms: &[BigInt<N>]) -> BigInt<N> {
        let mut acc = BigInt::<N>::ZERO;
        for t in terms {
            acc = acc.ct_add(t);
        }
        acc
    }

    /// Random `BigInt<8>` with the top four limbs zero, so it fits in
    /// 256 bits. Summing up to 16 of these stays well inside 512 bits,
    /// keeping us in the "exact sum fits in `N`" regime where `mac_sum`
    /// is byte-identical to the `ct_add` chain.
    fn arb_term() -> impl Strategy<Value = BigInt<8>> {
        (any::<bool>(), any::<[u64; 4]>()).prop_map(|(neg, lo)| {
            let mut limbs = [0u64; 8];
            limbs[..4].copy_from_slice(&lo);
            BigInt::from_sign_and_limbs(u64::from(neg), limbs)
        })
    }

    proptest! {
        #[test]
        fn mac_sum_matches_chain(terms in prop::collection::vec(arb_term(), 0..=16)) {
            prop_assert_eq!(BigInt::mac_sum(&terms), chain_sum(&terms));
        }
    }

    #[test]
    fn mac_sum_empty_is_zero() {
        let terms: [BigInt<4>; 0] = [];
        assert_eq!(BigInt::mac_sum(&terms), BigInt::ZERO);
    }

    #[test]
    fn mac_sum_cancels_to_zero() {
        let a = BigInt::<4>::from_i64(123_456_789);
        let terms = [a, -a, a, -a];
        let s = BigInt::mac_sum(&terms);
        assert!(bool::from(s.is_zero()));
        assert_eq!(s.as_limbs(), &[0u64; 4]);
    }

    #[test]
    fn mac_sum_mixed_sign() {
        let terms = [
            BigInt::<4>::from_i64(10),
            BigInt::<4>::from_i64(-3),
            BigInt::<4>::from_i64(-2),
            BigInt::<4>::from_i64(7),
        ];
        // 10 - 3 - 2 + 7 = 12
        assert_eq!(BigInt::mac_sum(&terms), BigInt::from_i64(12));
    }
}
