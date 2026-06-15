//! GCD and extended GCD on [`BigInt<N>`][super::BigInt], plus the
//! [`invert_mod`][BigInt::invert_mod] modular-inverse routine built
//! on top of `xgcd`.

use core::cmp::Ordering;

use subtle::ConstantTimeEq;

use super::{BigInt, ct_select_u64};

impl<const N: usize> BigInt<N> {
    /// Greatest common divisor via Stein's binary algorithm.
    ///
    /// Returns a non-negative value. Replaces division with shifts and
    /// subtractions: each iteration strips trailing zeros from the
    /// smaller operand and subtracts. Total iterations are bounded by
    /// `2·BITS` and each does O(N) limb work, so total cost is O(N²·64)
    /// — versus O(BITS²·N) for Euclidean.
    ///
    /// **Variable-time.** Iteration count, shift amounts, and the
    /// swap-on-greater branch all leak information about the inputs.
    /// Constant-time GCD will be reintroduced in a separate pass.
    pub fn gcd(&self, other: &Self) -> Self {
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
            //
            // The cofactor updates mutate `aa`/`bb` in place so the
            // per-iteration return-by-value copies disappear. The
            // `add_then_halve_even` / `sub_then_halve_even` paths fold the
            // signed add/subtract and the subsequent halving into one pass,
            // eliding the intermediate `BigInt` that `(aa + &y)` would
            // otherwise materialize before `halve_even` copied it again. The
            // dataflow (and hence the branch structure) is identical to the
            // by-value form.
            while u[0] & 1 == 0 {
                u = Self::mag_shr(&u, 1);
                if (aa.limbs[0] | bb.limbs[0]) & 1 == 0 {
                    aa.halve_even_assign();
                    bb.halve_even_assign();
                } else {
                    aa.add_then_halve_even(&y);
                    bb.sub_then_halve_even(&x);
                }
            }
            // Halve v likewise.
            while v[0] & 1 == 0 {
                v = Self::mag_shr(&v, 1);
                if (cc.limbs[0] | dd.limbs[0]) & 1 == 0 {
                    cc.halve_even_assign();
                    dd.halve_even_assign();
                } else {
                    cc.add_then_halve_even(&y);
                    dd.sub_then_halve_even(&x);
                }
            }

            // Both u, v odd. Subtract smaller from larger; result is
            // even, picked up by the next iteration's halve loop.
            if Self::mag_cmp(&u, &v) != Ordering::Less {
                let (new_u, _) = Self::mag_sub(&u, &v);
                u = new_u;
                aa.ct_sub_assign(&cc);
                bb.ct_sub_assign(&dd);
            } else {
                let (new_v, _) = Self::mag_sub(&v, &u);
                v = new_v;
                cc.ct_sub_assign(&aa);
                dd.ct_sub_assign(&bb);
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

    /// Halves a value known to be even, in place. Sign preserved (no
    /// floor-vs-trunc issue since we only halve even values), and
    /// re-canonicalized to sign 0 when the result is zero.
    ///
    /// In-place counterpart to the by-value halving the binary xgcd used
    /// to do; the cofactor loop calls this once per halving step, so
    /// avoiding the return-value copy matters.
    #[inline]
    fn halve_even_assign(&mut self) {
        debug_assert!(self.limbs[0] & 1 == 0, "halve_even on odd value");
        self.limbs = Self::mag_shr(&self.limbs, 1);
        let zero = Self::mag_is_zero(&self.limbs);
        self.sign &= 1 - zero;
    }

    /// In-place signed add: `self += rhs`. Branch-free sign-and-magnitude
    /// merge identical to [`Self::ct_add`], writing the result into `self`
    /// rather than returning a fresh [`BigInt`].
    ///
    /// Used by the binary xgcd cofactor loop, where folding the add into
    /// the destination removes one per-iteration copy.
    #[inline]
    fn ct_add_assign(&mut self, rhs: &Self) {
        self.ct_add_assign_signed(&rhs.limbs, rhs.sign);
    }

    /// In-place signed subtract: `self -= rhs`. Equivalent to adding the
    /// negation of `rhs`, but passes the flipped sign straight through to
    /// [`Self::ct_add_assign_signed`] so no negated temporary is built.
    #[inline]
    fn ct_sub_assign(&mut self, rhs: &Self) {
        // Negate rhs's sign, clamping the canonical-zero invariant (a zero
        // magnitude stays sign 0) exactly as [`Self::wrapping_neg`] does.
        let rhs_is_zero = Self::mag_is_zero(&rhs.limbs);
        let neg_sign = (rhs.sign ^ 1) & (1 - rhs_is_zero);
        self.ct_add_assign_signed(&rhs.limbs, neg_sign);
    }

    /// In-place core of the signed add/subtract: `self += sign(rhs_sign) ·
    /// |rhs_limbs|`. Mirrors [`Self::ct_add`] limb-for-limb (same
    /// `mag_add` / `mag_sub` / `mag_select` dataflow and branch structure),
    /// so it is constant-time-neutral and bit-for-bit identical to the
    /// by-value path; it only writes into `self` instead of allocating a
    /// result.
    #[inline]
    fn ct_add_assign_signed(&mut self, rhs_limbs: &[u64; N], rhs_sign: u64) {
        let same_sign = ((self.sign ^ rhs_sign) == 0) as u64;

        // Case 1: same sign -> add magnitudes, keep sign.
        let (sum, _carry) = Self::mag_add(&self.limbs, rhs_limbs);

        // Case 2: different signs -> subtract the smaller magnitude from the
        // larger. `mag_sub`'s borrow is the ordering, and the reverse
        // difference is the two's-complement negation of the forward one.
        let (diff_a, borrow) = Self::mag_sub(&self.limbs, rhs_limbs);
        let self_ge = 1 - borrow;
        let diff_b = Self::mag_negate(&diff_a);

        let diff_mag = Self::mag_select(&diff_b, &diff_a, self_ge);
        let diff_sign = ct_select_u64(rhs_sign, self.sign, self_ge);

        let result_limbs = Self::mag_select(&diff_mag, &sum, same_sign);
        let result_sign = ct_select_u64(diff_sign, self.sign, same_sign);

        // Canonicalize: if result is zero, sign must be 0.
        let is_zero = Self::mag_is_zero(&result_limbs);

        self.limbs = result_limbs;
        self.sign = result_sign & (1 - is_zero);
    }

    /// In-place fused `self = (self + rhs) / 2`, where the sum is known
    /// even. One signed add followed by an even-halving, both writing into
    /// `self`. Replaces `halve_even(&(self + rhs))`, whose intermediate
    /// `BigInt` from `self + rhs` and second copy in `halve_even` the
    /// compiler cannot elide.
    #[inline]
    fn add_then_halve_even(&mut self, rhs: &Self) {
        self.ct_add_assign(rhs);
        self.halve_even_assign();
    }

    /// In-place fused `self = (self - rhs) / 2`, where the difference is
    /// known even. Subtraction counterpart to [`Self::add_then_halve_even`].
    #[inline]
    fn sub_then_halve_even(&mut self, rhs: &Self) {
        self.ct_sub_assign(rhs);
        self.halve_even_assign();
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
        let result = x.ct_mod(modulus);
        Some(result)
    }
}
