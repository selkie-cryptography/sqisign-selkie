//! GCD and extended GCD on [`BigInt<N>`][super::BigInt], plus the
//! [`invert_mod`][BigInt::invert_mod] modular-inverse routine built
//! on top of `xgcd`.

use core::cmp::Ordering;

use subtle::ConstantTimeEq;

use super::BigInt;

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
    /// Stein's binary extended GCD (HAC algorithm 14.61). Same shape as
    /// [`Self::gcd`] but tracks Bezout cofactors through the halving and
    /// subtract steps; when the cofactor pair isn't both even, the
    /// originals (post common-factor strip) are added/subtracted to make
    /// them so before halving.
    ///
    /// **Variable-time**, same sources of leakage as [`Self::gcd`] plus
    /// cofactor sign branches.
    pub fn xgcd(&self, other: &Self) -> (Self, Self, Self) {
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
        let result = x.ct_mod(modulus);
        Some(result)
    }
}
