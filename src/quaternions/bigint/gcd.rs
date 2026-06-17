//! GCD and extended GCD on [`BigInt<N>`][super::BigInt], plus the
//! [`invert_mod`][BigInt::invert_mod] modular-inverse routine built
//! on top of `xgcd`.

use core::cmp::Ordering;

use subtle::ConstantTimeEq;

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Greatest common divisor. Returns a non-negative value.
    ///
    /// Dispatches by storage width: `gcd_stein` (binary GCD) for
    /// `N < 8`, [`Self::gcd_lehmer`] for `N >= 8`. Both produce identical
    /// values; the crossover (microbenched on M4) lands between `N = 4`
    /// (Stein wins narrowly) and `N = 8` (Lehmer ~2.7x), widening with
    /// `N` because Lehmer's per-iteration cost tracks operand magnitude
    /// rather than Stein's `O(N^2 * 64)` bit-step count.
    ///
    /// **Variable-time.** Both backends leak operand structure (iteration
    /// count, shift amounts / quotient values, the swap branch). Mirrors
    /// the `main` track's GMP-equivalent posture; CT GCD lives elsewhere.
    pub fn gcd(&self, other: &Self) -> Self {
        if N >= 8 {
            return self.gcd_lehmer(other);
        }
        self.gcd_stein(other)
    }

    /// Stein's binary GCD, retained as the differential oracle the Lehmer
    /// path is validated against (see the `lehmer_gcd_*` proptests) and
    /// the `N < 8` fast path where its bit-stepping still beats Lehmer's
    /// per-word batching.
    pub(super) fn gcd_stein(&self, other: &Self) -> Self {
        let mut a = self.abs().limbs;
        let mut b = other.abs().limbs;

        // Special-case zero inputs: gcd(0, x) = x, gcd(0, 0) = 0.
        if Self::mag_is_zero(&a) == 1 {
            return Self { sign: 0, limbs: b };
        }
        if Self::mag_is_zero(&b) == 1 {
            return Self { sign: 0, limbs: a };
        }

        // Strip the largest power of 2 dividing both, applied at the end.
        let shift = Self::mag_trailing_zeros(&a).min(Self::mag_trailing_zeros(&b));
        a = Self::mag_shr(&a, shift);
        b = Self::mag_shr(&b, shift);

        // Make a odd. b may still be even on entry to the loop.
        a = Self::mag_shr(&a, Self::mag_trailing_zeros(&a));

        loop {
            // Make b odd; both operands odd from here.
            b = Self::mag_shr(&b, Self::mag_trailing_zeros(&b));

            // Ensure a <= b so the subtraction below has no borrow.
            if Self::mag_cmp(&a, &b) == Ordering::Greater {
                core::mem::swap(&mut a, &mut b);
            }

            // b := b - a. Both odd, so result is even and the next
            // iteration's shift makes progress.
            let (new_b, _) = Self::mag_sub(&b, &a);
            b = new_b;

            if Self::mag_is_zero(&b) == 1 {
                break;
            }
        }

        Self {
            sign: 0,
            limbs: Self::mag_shl(&a, shift),
        }
    }

    /// Greatest common divisor via Lehmer's algorithm (HAC 14.57).
    ///
    /// Returns a non-negative value, byte-identical to `gcd_stein`
    /// on every input (proptested at N=4/8/16/30/60/150 plus the N=500
    /// signing-lattice regression). Where Stein replaces division with
    /// shifts, Lehmer batches many Euclidean quotient steps into a single
    /// multi-precision update:
    /// it extracts the leading 64-bit words of the two operands, runs
    /// single-word Euclid on them while the quotient is unambiguous,
    /// accumulating a 2x2 integer transform `[[a00, a01], [a10, a11]]`,
    /// then applies that transform to the full-width operands in one
    /// pass. When the leading words can't disambiguate the next quotient
    /// (the cofactor matrix would still be the identity), it falls back
    /// to one full-width Euclidean division step. Each outer iteration
    /// retires roughly a full word of both operands, so total cost is
    /// O(n) word-divisions plus O(n) full-width updates of O(n) work,
    /// i.e. O(n^2) word operations versus Stein's O(n^2 * 64) bit-steps.
    ///
    /// **Variable-time.** Quotient values, iteration count, and the
    /// leading-word disambiguation branch all leak operand structure.
    /// This is the `main`-track Lehmer that mirrors GMP's var-time
    /// posture; the CT GCD lives elsewhere.
    pub fn gcd_lehmer(&self, other: &Self) -> Self {
        let mut a = self.abs().limbs;
        let mut b = other.abs().limbs;

        if Self::mag_is_zero(&a) == 1 {
            return Self { sign: 0, limbs: b };
        }
        if Self::mag_is_zero(&b) == 1 {
            return Self { sign: 0, limbs: a };
        }

        // Keep a >= b throughout.
        if Self::mag_cmp(&a, &b) == Ordering::Less {
            core::mem::swap(&mut a, &mut b);
        }

        loop {
            let len_b = Self::mag_effective_len(&b);
            if len_b == 0 {
                break;
            }
            // Once b fits in a single word, finish with plain u64 Euclid:
            // reduce a mod b[0], then run single-precision gcd.
            if len_b == 1 {
                let d = b[0];
                let mut r: u128 = 0;
                let len_a = Self::mag_effective_len(&a);
                let mut i = len_a;
                while i > 0 {
                    i -= 1;
                    r = ((r << 64) | a[i] as u128) % d as u128;
                }
                let mut g = d;
                let mut rr = r as u64;
                while rr != 0 {
                    let t = g % rr;
                    g = rr;
                    rr = t;
                }
                let mut limbs = [0u64; N];
                limbs[0] = g;
                a = limbs;
                break;
            }

            // Extract the top words of a and b aligned to a's top limb.
            // `s` left-normalizes a's MSB to bit 63; b is taken from the
            // same limb position so the windows are comparable.
            let len_a = Self::mag_effective_len(&a);
            let s = a[len_a - 1].leading_zeros();
            let a_hat = Self::top_word(&a, len_a, s);
            let b_hat = Self::top_word(&b, len_a, s);

            // Single-word Lehmer inner loop on (u, v) = (a_hat, b_hat),
            // following Knuth TAOCP 4.5.2 Algorithm L. The 2x2 cofactor
            // matrix [[a00, a01], [a10, a11]] tracks the transform mapping
            // the original (a, b) windows to (u, v): the entries carry
            // their own signs (a00, a11 stay non-negative; a01, a10 stay
            // non-positive). A step is accepted only while the single-word
            // quotient estimate is unambiguous (low and high bounds agree).
            let (mut u, mut v) = (a_hat, b_hat);
            let (mut a00, mut a01, mut a10, mut a11) = (1i128, 0i128, 0i128, 1i128);
            let mut steps = 0u32;
            loop {
                let denom_c = (v as i128) + a10;
                let denom_d = (v as i128) + a11;
                if denom_c == 0 || denom_d == 0 {
                    break;
                }
                let q = ((u as i128) + a00) / denom_c;
                let q2 = ((u as i128) + a01) / denom_d;
                if q != q2 {
                    break;
                }
                // Apply quotient q: (u, v) <- (v, u - q*v) with the matrix
                // rows updated likewise.
                let new_v = (u as i128) - q * (v as i128);
                u = v;
                v = new_v as u64;
                let n0 = a00 - q * a10;
                let n1 = a01 - q * a11;
                a00 = a10;
                a01 = a11;
                a10 = n0;
                a11 = n1;
                steps += 1;
            }

            if steps == 0 {
                // Leading words couldn't disambiguate: one full Euclid step.
                let (_, r) = Self::mag_div_rem(&a, &b);
                a = b;
                b = r;
            } else {
                // Apply the cofactor matrix to the full-width operands:
                // (a, b) <- (a00*a + a01*b, a10*a + a11*b), all non-negative.
                let (new_a, new_b) = Self::apply_cofactor_matrix(&a, &b, a00, a01, a10, a11);
                a = new_a;
                b = new_b;
                if Self::mag_cmp(&a, &b) == Ordering::Less {
                    core::mem::swap(&mut a, &mut b);
                }
            }
        }

        Self { sign: 0, limbs: a }
    }

    /// Extracts the top 64-bit window of magnitude `a` left-normalized by
    /// `s` bits, where `len` is `a`'s effective limb length and `s` is the
    /// leading-zero count of `a`'s top limb (so the window's MSB lands at
    /// bit 63 when read from `a`'s leading limb).
    ///
    /// `a` is read at limb index `len - 1`; the same `s` and limb index
    /// are used for `b` so [`Self::gcd_lehmer`]'s two single-word windows
    /// share a bit alignment and their quotient estimates are comparable.
    #[inline]
    fn top_word(a: &[u64; N], len: usize, s: u32) -> u64 {
        let hi = a[len - 1];
        if s == 0 {
            return hi;
        }
        let lo = if len >= 2 { a[len - 2] } else { 0 };
        (hi << s) | (lo >> (64 - s))
    }

    /// Computes `(a00*a + a01*b, a10*a + a11*b)` for the signed
    /// single-word cofactors produced by [`Self::gcd_lehmer`]'s inner
    /// loop. Each row's value is a non-negative Euclid remainder smaller
    /// than `max(a, b)`, so it fits in `N` limbs even though the
    /// individual products `|c|*a` may not. The two coefficients are
    /// never both negative (that would give a negative result); the three
    /// remaining sign cases reduce to one fused multiply-add (both
    /// non-negative) or a fused multiply-subtract (mixed signs), neither
    /// of which materializes the oversized product.
    fn apply_cofactor_matrix(
        a: &[u64; N],
        b: &[u64; N],
        a00: i128,
        a01: i128,
        a10: i128,
        a11: i128,
    ) -> ([u64; N], [u64; N]) {
        let row = |c0: i128, c1: i128| -> [u64; N] {
            match (c0 >= 0, c1 >= 0) {
                (true, true) => Self::fused_mul_add(a, c0 as u64, b, c1 as u64),
                (true, false) => Self::fused_mul_sub(a, c0 as u64, b, c1.unsigned_abs() as u64),
                (false, true) => Self::fused_mul_sub(b, c1 as u64, a, c0.unsigned_abs() as u64),
                // Both negative would yield a negative combination, which
                // the Euclid remainder invariant rules out; only (0, 0)
                // reaches here and produces zero.
                (false, false) => [0u64; N],
            }
        };
        (row(a00, a01), row(a10, a11))
    }

    /// Computes `p*pw + m*mw` over magnitudes, truncated to `N` limbs.
    /// The caller guarantees the true sum fits, so the top carry is zero.
    ///
    /// The two products are accumulated in separate carry chains so no
    /// single `u128` ever has to hold their full sum (which would need 129
    /// bits): `p*pw` runs in `carry_p`, `m*mw` in `carry_m`, and the two
    /// low limbs plus a small running `add_carry` are combined per limb.
    #[inline]
    fn fused_mul_add(p: &[u64; N], pw: u64, m: &[u64; N], mw: u64) -> [u64; N] {
        let mut out = [0u64; N];
        let mut carry_p: u64 = 0;
        let mut carry_m: u64 = 0;
        let mut add_carry: u64 = 0;
        let mut i = 0;
        while i < N {
            let prod_p = p[i] as u128 * pw as u128 + carry_p as u128;
            carry_p = (prod_p >> 64) as u64;
            let prod_m = m[i] as u128 * mw as u128 + carry_m as u128;
            carry_m = (prod_m >> 64) as u64;

            let sum = prod_p as u64 as u128 + prod_m as u64 as u128 + add_carry as u128;
            out[i] = sum as u64;
            add_carry = (sum >> 64) as u64;
            i += 1;
        }
        out
    }

    /// Computes `p*pw - m*mw` over magnitudes, where the true result is
    /// known to be non-negative and to fit in `N` limbs. The product and
    /// difference are interleaved per limb so the (possibly oversized)
    /// intermediate `p*pw` is never stored, only its running low limbs.
    /// Returns the `N`-limb magnitude.
    #[inline]
    fn fused_mul_sub(p: &[u64; N], pw: u64, m: &[u64; N], mw: u64) -> [u64; N] {
        let mut out = [0u64; N];
        let mut add_carry: u64 = 0;
        let mut sub_borrow: u64 = 0;
        let mut i = 0;
        while i < N {
            let prod = p[i] as u128 * pw as u128 + add_carry as u128;
            add_carry = (prod >> 64) as u64;
            let plo = prod as u64;

            let sub = m[i] as u128 * mw as u128 + sub_borrow as u128;
            sub_borrow = (sub >> 64) as u64;
            let slo = sub as u64;

            let (diff, borrow) = plo.overflowing_sub(slo);
            out[i] = diff;
            sub_borrow += borrow as u64;
            i += 1;
        }
        out
    }

    /// Extended GCD: returns `(gcd, x, y)` such that
    /// `self * x + other * y = gcd`, with `gcd >= 0`.
    ///
    /// Dispatches to [`Self::xgcd_binary`] at the smallest fixed working
    /// width that safely holds the operands and their Bezout cofactors.
    /// The quaternion lattice code calls this on values stored in wide
    /// `BigInt<N>` (N up to ~500) whose actual magnitudes are far
    /// smaller, so each halve/subtract step would otherwise run across
    /// hundreds of always-zero high limbs. Narrowing first makes the
    /// per-iteration cost track the operand size, not the storage width;
    /// the result is identical to running at width `N`.
    ///
    /// **Variable-time**, same sources of leakage as [`Self::gcd`] plus
    /// cofactor sign branches.
    #[must_use]
    pub fn xgcd(&self, other: &Self) -> (Self, Self, Self) {
        // Operands and the binary-GCD cofactors are bounded in magnitude
        // by `max(|self|, |other|)`, so `bits/64 + 2` limbs (one for the
        // bit-length boundary, one of slack) hold every intermediate.
        let bits = Self::mag_bitsize(&self.limbs).max(Self::mag_bitsize(&other.limbs)) as usize;
        let needed = bits / 64 + 2;

        if N > 8 && needed <= 8 {
            return self.xgcd_narrowed::<8>(other);
        }
        if N > 16 && needed <= 16 {
            return self.xgcd_narrowed::<16>(other);
        }
        if N > 32 && needed <= 32 {
            return self.xgcd_narrowed::<32>(other);
        }
        if N > 64 && needed <= 64 {
            return self.xgcd_narrowed::<64>(other);
        }
        if N > 128 && needed <= 128 {
            return self.xgcd_narrowed::<128>(other);
        }
        if N > 256 && needed <= 256 {
            return self.xgcd_narrowed::<256>(other);
        }

        self.xgcd_binary(other)
    }

    /// Runs [`Self::xgcd_binary`] at narrower working width `M`, resizing
    /// the `(gcd, x, y)` result back to `N`.
    ///
    /// The caller ([`Self::xgcd`]) chooses `M >= needed`, so the operand
    /// magnitudes (and hence all cofactors) fit in `M` limbs and the
    /// narrowed computation produces the same values as width `N`; only
    /// leading zero limbs are dropped.
    #[must_use]
    fn xgcd_narrowed<const M: usize>(&self, other: &Self) -> (Self, Self, Self) {
        let a = self.resize_for_xgcd::<M>();
        let b = other.resize_for_xgcd::<M>();

        let (g, x, y) = a.xgcd_binary(&b);

        (
            g.resize_for_xgcd::<N>(),
            x.resize_for_xgcd::<N>(),
            y.resize_for_xgcd::<N>(),
        )
    }

    /// Copies `self` into width `W`, taking the low `min(N, W)` limbs and
    /// preserving the sign — without the fit assertions of
    /// [`Self::widen`] / [`Self::narrow_to`].
    ///
    /// Used only by [`Self::xgcd_narrowed`], where the dispatch in
    /// [`Self::xgcd`] provably picks a width holding every value, so any
    /// dropped high limbs are zero. The runtime `if N > M` guards mean
    /// the shrinking direction never executes for `W < N` (and vice
    /// versa); the assertion-free copy only exists so those statically
    /// unreachable instantiations still type-check. Not for general use.
    #[must_use]
    fn resize_for_xgcd<const W: usize>(self) -> BigInt<W> {
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

    /// Stein's binary extended GCD (HAC algorithm 14.61). Same shape as
    /// [`Self::gcd`] but tracks Bezout cofactors through the halving and
    /// subtract steps; when the cofactor pair isn't both even, the
    /// originals (post common-factor strip) are added/subtracted to make
    /// them so before halving.
    ///
    /// Runs entirely at the storage width `N`; [`Self::xgcd`] narrows to
    /// a tight width before calling this.
    #[must_use]
    fn xgcd_binary(&self, other: &Self) -> (Self, Self, Self) {
        let a_abs = self.abs();
        let b_abs = other.abs();

        // Edge cases: gcd(0, b) = |b| with cofactors (0, sign(b)); symmetric.
        // The cofactors must satisfy `self · u + other · v = gcd ≥ 0`. For
        // negative inputs the early-returned cofactor needs the matching
        // sign so the identity holds (e.g. `xgcd(-5, 0)` must return
        // `(5, -1, 0)` so that `(-5) · (-1) + 0 · 0 = 5`, not `(5, 1, 0)`
        // which gives `-5 ≠ 5`).
        if bool::from(a_abs.is_zero()) {
            let v_sign = if other.sign == 1 {
                Self::ONE.wrapping_neg()
            } else {
                Self::ONE
            };
            return (b_abs, Self::ZERO, v_sign);
        }
        if bool::from(b_abs.is_zero()) {
            let u_sign = if self.sign == 1 {
                Self::ONE.wrapping_neg()
            } else {
                Self::ONE
            };
            return (a_abs, u_sign, Self::ZERO);
        }

        // Strip common factor of 2; reapplied to gcd at the end.
        // Cofactors are computed against the stripped operands `(x, y)`,
        // and that's also what satisfies `x_co · self + y_co · other == g`,
        // since stripping a common factor doesn't change the cofactor
        // identity.
        let g_shift =
            Self::mag_trailing_zeros(&a_abs.limbs).min(Self::mag_trailing_zeros(&b_abs.limbs));
        let x_lim = Self::mag_shr(&a_abs.limbs, g_shift);
        let y_lim = Self::mag_shr(&b_abs.limbs, g_shift);
        let x = Self {
            sign: 0,
            limbs: x_lim,
        };
        let y = Self {
            sign: 0,
            limbs: y_lim,
        };

        // Invariant: `u = aa·x + bb·y` and `v = cc·x + dd·y`.
        let mut u = x_lim;
        let mut v = y_lim;
        let mut aa = Self::ONE;
        let mut bb = Self::ZERO;
        let mut cc = Self::ZERO;
        let mut dd = Self::ONE;

        loop {
            // Halve u while the invariant holds; adjust cofactors.
            while u[0] & 1 == 0 {
                u = Self::mag_shr(&u, 1);
                if (aa.limbs[0] | bb.limbs[0]) & 1 == 0 {
                    aa = Self::halve_even(&aa);
                    bb = Self::halve_even(&bb);
                } else {
                    aa = Self::halve_even(&(aa + &y));
                    bb = Self::halve_even(&(bb - &x));
                }
            }
            // Halve v likewise.
            while v[0] & 1 == 0 {
                v = Self::mag_shr(&v, 1);
                if (cc.limbs[0] | dd.limbs[0]) & 1 == 0 {
                    cc = Self::halve_even(&cc);
                    dd = Self::halve_even(&dd);
                } else {
                    cc = Self::halve_even(&(cc + &y));
                    dd = Self::halve_even(&(dd - &x));
                }
            }

            // Both u, v odd. Subtract smaller from larger; result is
            // even, picked up by the next iteration's halve loop.
            if Self::mag_cmp(&u, &v) != Ordering::Less {
                let (new_u, _) = Self::mag_sub(&u, &v);
                u = new_u;
                aa = aa - &cc;
                bb = bb - &dd;
            } else {
                let (new_v, _) = Self::mag_sub(&v, &u);
                v = new_v;
                cc = cc - &aa;
                dd = dd - &bb;
            }

            if Self::mag_is_zero(&u) == 1 {
                break;
            }
        }

        let g = Self {
            sign: 0,
            limbs: Self::mag_shl(&v, g_shift),
        };

        // Adjust signs to undo our `abs()` of the inputs.
        let mut x_co = if self.sign == 1 {
            cc.wrapping_neg()
        } else {
            cc
        };
        let mut y_co = if other.sign == 1 {
            dd.wrapping_neg()
        } else {
            dd
        };
        x_co.normalize();
        y_co.normalize();

        (g, x_co, y_co)
    }

    /// Halves a value known to be even. Sign preserved (no floor-vs-trunc
    /// issue since we only halve even values).
    #[inline]
    fn halve_even(a: &Self) -> Self {
        debug_assert!(a.limbs[0] & 1 == 0, "halve_even on odd value");
        let limbs = Self::mag_shr(&a.limbs, 1);
        let zero = Self::mag_is_zero(&limbs) == 1;
        Self {
            sign: if zero { 0 } else { a.sign },
            limbs,
        }
    }

    /// Modular inverse: returns `self^{-1} mod modulus`, or `None` if
    /// the inverse does not exist (i.e., `gcd(self, modulus) != 1`).
    ///
    /// The result is in `[0, |modulus|)`.
    pub fn invert_mod(&self, modulus: &Self) -> Option<Self> {
        let (g, x, _) = self.xgcd(modulus);
        if !bool::from(g.ct_eq(&Self::ONE)) {
            return None;
        }
        // x might be negative; reduce mod |modulus|.
        let result = x.vt_mod(modulus);
        Some(result)
    }
}
