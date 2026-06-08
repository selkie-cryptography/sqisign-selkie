//! Modular arithmetic over [`BigInt<N>`][super::BigInt].
//!
//! Holds the Montgomery-reduction context type [`MontReducer`] and
//! the high-level modular-exponentiation entry points
//! [`pow_mod`][BigInt::pow_mod] and [`pow_mod_w`][BigInt::pow_mod_w].

use core::cmp::Ordering;

use super::BigInt;

/// Precomputed Montgomery reducer for a runtime-supplied odd modulus.
///
/// Distinct from [`crate::fields::fp::Fp`] (which hardcodes the SQIsign
/// curve prime in its type identity): a `MontReducer<N>` is built at
/// runtime via [`MontReducer::new`] for any odd `n < 2^{64N}`. The
/// same struct shape backs every modulus used in
/// signing's quaternion-side arithmetic (`D_mix`, ramification primes,
/// Miller-Rabin candidates during `random_prime_norm`). Operates on
/// canonical-form [`BigInt<N>`] values; the Montgomery representation
/// is internal.
///
/// Montgomery form represents a value `x` as `x · R mod n` where
/// `R = 2^{64N}`. Multiplication in this form costs one CIOS
/// multiply-and-reduce (no division) per op, making chains like
/// `pow_mod` substantially cheaper than schoolbook reduction.
///
/// # Constant-time
///
/// **Variable-time.** The final conditional subtract in `mul`/`square`
/// branches on a data-dependent comparison, and the modular-inverse
/// precomputation uses early-exit Newton iteration. Constant-time
/// Montgomery will be reintroduced in a separate pass.
///
/// **Caching note.** The cached fields (`n_inv_neg`, `r_squared`) are
/// derived purely from `n` and leak nothing the modulus didn't. A
/// within-operation `MontReducer` is safe even when the modulus is
/// secret-derived (e.g. a Miller-Rabin candidate). Cross-operation
/// reuse keyed on a secret modulus, however, opens a cache-occupancy
/// timing channel on the modulus itself; scope caching to within an
/// operation in those cases when CT lands.
pub(crate) struct MontReducer<const N: usize> {
    /// The modulus (odd, nonzero).
    n: [u64; N],
    /// `-n^{-1} mod 2^64`. Used per-iteration in CIOS to choose the
    /// reduction constant `m_i` so that `t[0] + m_i · n[0] ≡ 0
    /// (mod 2^64)`.
    n_inv_neg: u64,
    /// `R^2 mod n` where `R = 2^{64N}`. Used by [`Self::to_montgomery`].
    r_squared: [u64; N],
}

impl<const N: usize> MontReducer<N> {
    /// Returns a reducer for the given odd modulus, or `None` if the
    /// modulus is even or zero.
    pub(crate) fn new(modulus: &BigInt<N>) -> Option<Self> {
        // Modulus must be odd (n_inv_neg only exists then) and nonzero.
        if modulus.limbs[0] & 1 == 0 || bool::from(modulus.is_zero()) {
            return None;
        }
        let n = modulus.limbs;
        let n_inv_neg = Self::neg_inv_mod_2_64(n[0]);
        let r_squared = Self::compute_r_squared(&n);
        Some(Self {
            n,
            n_inv_neg,
            r_squared,
        })
    }

    /// Returns the modulus this reducer was built for, as a limb slice.
    ///
    /// Used by `*_with_ctx` callers to debug-assert that the supplied
    /// reducer matches their modulus.
    pub(crate) fn modulus_limbs(&self) -> &[u64; N] {
        &self.n
    }

    /// Returns `x · R mod n` — i.e. converts a canonical-form magnitude
    /// in `[0, n)` into Montgomery form.
    fn to_montgomery(&self, x: &[u64; N]) -> [u64; N] {
        self.mul(x, &self.r_squared)
    }

    /// Returns `x · R^{-1} mod n` — i.e. converts a Montgomery-form
    /// magnitude back to canonical form.
    fn reduce_montgomery(&self, x: &[u64; N]) -> [u64; N] {
        let mut one = [0u64; N];
        one[0] = 1;
        self.mul(x, &one)
    }

    /// Returns `a · b · R^{-1} mod n` for two Montgomery-form
    /// magnitudes. Inputs must be in `[0, n)`; output is in `[0, n)`.
    ///
    /// Uses the **CIOS** (Coarsely Integrated Operand Scanning)
    /// schedule per [Acar 1996][acar96], §5: each outer iteration
    /// over `b`'s limbs alternates one multiply step (`t += a · b[i]`)
    /// with one reduction step (add `m·n` so the bottom limb cancels,
    /// shift down). This interleaving keeps the working buffer at
    /// `N+2` limbs throughout, vs `2N+1` for the separated form
    /// (SOS), and is the fastest of Acar's variants in software.
    ///
    /// [acar96]: https://www.microsoft.com/en-us/research/wp-content/uploads/1996/01/j37acmon.pdf
    fn mul(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let n = &self.n;
        let n_inv = self.n_inv_neg;

        // `t` holds N+2 limbs as (t[0..N], t_n, t_np1). t_np1 is local
        // to each outer iteration since it's written in the multiply
        // phase and consumed in the reduce phase, never carried across.
        let mut t = [0u64; N];
        let mut t_n: u64 = 0;

        for &b_i in b {
            // Multiply phase: t += a · b_i
            let mut c: u64 = 0;
            for j in 0..N {
                let prod = t[j] as u128 + a[j] as u128 * b_i as u128 + c as u128;
                t[j] = prod as u64;
                c = (prod >> 64) as u64;
            }
            let sum = t_n as u128 + c as u128;
            t_n = sum as u64;
            let t_np1 = (sum >> 64) as u64;

            // Reduce phase: m chosen so t[0] + m·n[0] ≡ 0 (mod 2^64);
            // then add m·n and shift down by one limb.
            let m = t[0].wrapping_mul(n_inv);
            let prod = t[0] as u128 + m as u128 * n[0] as u128;
            // The low 64 bits of this prod are zero by construction.
            let mut c = (prod >> 64) as u64;
            for j in 1..N {
                let prod = t[j] as u128 + m as u128 * n[j] as u128 + c as u128;
                t[j - 1] = prod as u64;
                c = (prod >> 64) as u64;
            }
            let sum = t_n as u128 + c as u128;
            t[N - 1] = sum as u64;
            t_n = t_np1.wrapping_add((sum >> 64) as u64);
        }

        // After N iterations the result is in t[0..N] plus an at-most-1
        // overflow bit in t_n. By Montgomery's bound the value is in
        // `[0, 2n)`, so a single conditional subtract reduces.
        if t_n != 0 || BigInt::<N>::mag_cmp(&t, n) != Ordering::Less {
            let (sub, _) = BigInt::<N>::mag_sub(&t, n);
            t = sub;
        }
        t
    }

    /// Returns `a · a · R^{-1} mod n` for a Montgomery-form magnitude.
    ///
    /// Decoupled implementation: computes the full 2N-limb `a²` using
    /// schoolbook squaring with cross-term reuse (`~N(N+1)/2`
    /// limb-mults, vs `N²` for [`Self::mul`]`(a, a)`), then applies
    /// Montgomery REDC. Saves roughly 25% of the multiply phase
    /// relative to `mul(a, a)`.
    ///
    /// Input `a` must be in Montgomery form, in `[0, n)`. Output is in
    /// `[0, n)`.
    fn square(&self, a: &[u64; N]) -> [u64; N] {
        let (lo, hi) = self.square_wide(a);
        self.reduce_wide(lo, hi)
    }

    /// Returns the full `2N`-limb product `a²` as `(low N, high N)`
    /// halves, using cross-term symmetry.
    ///
    /// Phases:
    /// 1. Compute lower-triangle cross products `a[i] · a[j]` for `i<j`,
    ///    accumulated into position `i+j` of the 2N-limb buffer.
    /// 2. Double the entire buffer (cross products contribute twice in `a²`).
    /// 3. Add diagonal squares `a[i]²` at position `2i`.
    fn square_wide(&self, a: &[u64; N]) -> ([u64; N], [u64; N]) {
        let mut lo = [0u64; N];
        let mut hi = [0u64; N];

        // Phase 1: lower-triangle cross products. Each pair (i, j) with
        // i < j contributes a[i]·a[j] to position i+j, accumulating
        // across outer iterations.
        let mut i = 0;
        while i < N {
            let mut carry: u64 = 0;
            let mut j = i + 1;
            while j < N {
                let pos = i + j;
                let cur = if pos < N { lo[pos] } else { hi[pos - N] };
                let prod = a[i] as u128 * a[j] as u128 + cur as u128 + carry as u128;
                if pos < N {
                    lo[pos] = prod as u64;
                } else {
                    hi[pos - N] = prod as u64;
                }
                carry = (prod >> 64) as u64;
                j += 1;
            }
            // Final carry of this iteration goes to position i+N (in `hi`).
            // hi[i] hasn't been written by any prior iter (each iter k
            // writes its final carry to hi[k]; inner-loop accumulating
            // writes to hi[k] for k > 0 happen via larger outer iters
            // and read-modify-write the existing value).
            hi[i] = carry;
            i += 1;
        }

        // Phase 2: double the entire 2N-limb buffer (cross products
        // appear twice in a²; doubling here lets Phase 3 add the
        // diagonal squares un-doubled).
        let mut carry: u64 = 0;
        let mut k = 0;
        while k < N {
            let new = (lo[k] << 1) | carry;
            carry = lo[k] >> 63;
            lo[k] = new;
            k += 1;
        }
        let mut k = 0;
        while k < N {
            let new = (hi[k] << 1) | carry;
            carry = hi[k] >> 63;
            hi[k] = new;
            k += 1;
        }
        // Final `carry` overflows position 2N. For SQIsign sizes with
        // `a < n < 2^(64N)`, this can't happen — a² < 2^(128N) so the
        // 2N-limb buffer never fills its top bit before doubling.
        debug_assert_eq!(carry, 0, "square_wide: phase-2 doubling overflow");

        // Phase 3: add diagonal squares a[i]² at position 2i.
        let mut carry: u64 = 0;
        let mut i = 0;
        while i < N {
            let prod = a[i] as u128 * a[i] as u128;
            let plo = prod as u64;
            let phi = (prod >> 64) as u64;

            let pos_lo = 2 * i;
            let cur_lo = if pos_lo < N {
                lo[pos_lo]
            } else {
                hi[pos_lo - N]
            };
            let (s, c1) = cur_lo.overflowing_add(plo);
            let (s, c2) = s.overflowing_add(carry);
            if pos_lo < N {
                lo[pos_lo] = s;
            } else {
                hi[pos_lo - N] = s;
            }
            let mid_carry = (c1 | c2) as u64;

            let pos_hi = 2 * i + 1;
            // pos_hi = 2i+1, max at i=N-1 is 2N-1, always valid.
            let cur_hi = if pos_hi < N {
                lo[pos_hi]
            } else {
                hi[pos_hi - N]
            };
            let (s, c1) = cur_hi.overflowing_add(phi);
            let (s, c2) = s.overflowing_add(mid_carry);
            if pos_hi < N {
                lo[pos_hi] = s;
            } else {
                hi[pos_hi - N] = s;
            }
            carry = (c1 | c2) as u64;
            i += 1;
        }
        debug_assert_eq!(carry, 0, "square_wide: phase-3 final carry overflow");

        (lo, hi)
    }

    /// Montgomery REDC on a 2N-limb input. Computes
    /// `(lo + hi · R) · R^{-1} mod n` where `R = 2^{64N}`.
    ///
    /// Same shape as [`Self::mul`]'s reduce phase, applied `N` times
    /// to a pre-multiplied 2N-limb input. The `(t_n, t_np1)` pair
    /// tracks the working window's overflow above the N-limb `t`.
    fn reduce_wide(&self, lo: [u64; N], hi: [u64; N]) -> [u64; N] {
        let n = &self.n;
        let n_inv = self.n_inv_neg;

        // Working window: t[0..N] + t_n + t_np1 (= conceptual N+2 limbs).
        // Initialize from (lo, hi[0], hi[1]) — the bottom of the input.
        let mut t = lo;
        let mut t_n: u64 = if N >= 1 { hi[0] } else { 0 };
        let mut t_np1: u64 = if N >= 2 { hi[1] } else { 0 };

        let mut round = 0;
        while round < N {
            let m = t[0].wrapping_mul(n_inv);
            // Low 64 bits of (t[0] + m·n[0]) are zero by choice of m.
            let prod = t[0] as u128 + m as u128 * n[0] as u128;
            let mut c = (prod >> 64) as u64;
            let mut j = 1;
            while j < N {
                let prod = t[j] as u128 + m as u128 * n[j] as u128 + c as u128;
                t[j - 1] = prod as u64;
                c = (prod >> 64) as u64;
                j += 1;
            }
            let sum = t_n as u128 + c as u128;
            t[N - 1] = sum as u64;
            let new_high_carry = (sum >> 64) as u64;

            // Shift t_np1 down into t_n; shift in the next hi limb at t_np1.
            t_n = t_np1.wrapping_add(new_high_carry);
            let next_hi_idx = round + 2;
            t_np1 = if next_hi_idx < N { hi[next_hi_idx] } else { 0 };

            round += 1;
        }

        // After N rounds: result is in `t`; t_n is at-most-1 overflow;
        // t_np1 should be 0 (all hi limbs consumed).
        debug_assert_eq!(t_np1, 0, "reduce_wide: t_np1 nonzero at end");

        if t_n != 0 || BigInt::<N>::mag_cmp(&t, n) != Ordering::Less {
            let (sub, _) = BigInt::<N>::mag_sub(&t, n);
            t = sub;
        }
        t
    }

    /// Modular exponentiation: `base^exp mod n`. Computed in Montgomery
    /// form using a 4-bit fixed-window scheme: precompute table of
    /// `base^i` for `i in 0..16`, then process the exponent four bits at
    /// a time with 4 squarings + 1 multiply per window.
    ///
    /// **Variable-time on the exponent.** The window-indexed table
    /// access is a cache-line side channel on the exponent bits — fine
    /// for SQIsign's Miller-Rabin (witness exponents are public and
    /// pre-determined small primes minus one) and for any other
    /// public-exponent path. For a future secret-exponent CT path the
    /// table read needs to be made oblivious (scan all 16 entries with
    /// `subtle::ConditionallySelectable`).
    pub(crate) fn pow(&self, base: &BigInt<N>, exp: &BigInt<N>) -> BigInt<N> {
        // Reduce base mod n then convert to Mont form.
        let base_red = BigInt::<N>::mag_div_rem(&base.limbs, &self.n).1;
        let base_m = self.to_montgomery(&base_red);

        // table[i] = base^i in Mont form, for i in 0..16. table[0] = 1·R = R mod n.
        let mut one = [0u64; N];
        one[0] = 1;
        let mut table: [[u64; N]; 16] = [[0u64; N]; 16];
        table[0] = self.mul(&self.r_squared, &one);
        table[1] = base_m;
        let mut i = 2;
        while i < 16 {
            table[i] = self.mul(&table[i - 1], &base_m);
            i += 1;
        }

        let bs = exp.bitsize();
        if bs == 0 {
            // base^0 = 1.
            return BigInt {
                sign: 0,
                limbs: self.reduce_montgomery(&table[0]),
            };
        }

        // Scan exponent in 4-bit windows from the most-significant
        // nibble down. Initial state is `1` in Mont form (= table[0]),
        // so the very first window doesn't need its 4 squarings — we
        // can directly install table[w_top] (or leave result at 1 if
        // w_top == 0).
        let nibbles = bs.div_ceil(4);
        let mut result = table[0];
        let mut nib_rev = nibbles;
        while nib_rev > 0 {
            nib_rev -= 1;
            let lo_bit = nib_rev * 4;

            // Extract bits [lo_bit, lo_bit+4) into a 4-bit window.
            let mut w: u64 = 0;
            let mut b: u32 = 0;
            while b < 4 {
                let bit_idx = lo_bit + b;
                if bit_idx < bs {
                    let li = (bit_idx / 64) as usize;
                    let bo = bit_idx % 64;
                    w |= ((exp.limbs[li] >> bo) & 1) << b;
                }
                b += 1;
            }
            let w = w as usize;

            // First-window optimization: result is still 1, so 4
            // squarings are no-ops (1^2 = 1 each). Skip them.
            if nib_rev != nibbles - 1 {
                result = self.square(&result);
                result = self.square(&result);
                result = self.square(&result);
                result = self.square(&result);
            }
            // Multiply by table[w] (skip if w == 0, since table[0] = 1).
            if w != 0 {
                result = self.mul(&result, &table[w]);
            }
        }

        BigInt {
            sign: 0,
            limbs: self.reduce_montgomery(&result),
        }
    }

    /// Returns `-u^{-1} mod 2^64` for odd `u`.
    ///
    /// Newton iteration on `f(x) = u·x − 1` converges quadratically:
    /// from a 5-bit accurate seed `(3·u XOR 2)` (Hacker's Delight),
    /// four iterations reach 80-bit precision, well past the u64
    /// window. Used to derive [`Self::n_inv_neg`].
    const fn neg_inv_mod_2_64(u: u64) -> u64 {
        debug_assert!(u & 1 == 1);
        // 5-bit accurate seed: 3·u XOR 2 ≡ u^{-1} (mod 32) for odd u.
        let mut x: u64 = u.wrapping_mul(3) ^ 2;
        // Newton: x_{k+1} = x_k · (2 - u · x_k); precision doubles each iter.
        x = x.wrapping_mul(2u64.wrapping_sub(u.wrapping_mul(x))); // 10 bits
        x = x.wrapping_mul(2u64.wrapping_sub(u.wrapping_mul(x))); // 20 bits
        x = x.wrapping_mul(2u64.wrapping_sub(u.wrapping_mul(x))); // 40 bits
        x = x.wrapping_mul(2u64.wrapping_sub(u.wrapping_mul(x))); // 80 bits
        x.wrapping_neg()
    }

    /// Returns `R^2 mod n` where `R = 2^{64N}`. Used by
    /// [`Self::to_montgomery`] to convert canonical-form values into
    /// Montgomery form via a single multiply.
    ///
    /// Computed by `128·N` rounds of doubling-and-conditional-subtract
    /// (`O(N²)` time, one-shot per modulus). `const fn` so callers
    /// embedding a fixed modulus can fold the entire setup into
    /// compile-time constant evaluation. The conditional subtract is
    /// driven by
    /// `mag_sub`'s borrow flag rather than `mag_cmp` because
    /// `mag_cmp` returns `Ordering`, which isn't `const`-callable in
    /// this crate's `MSRV` window — the borrow flag tells us
    /// `x >= n` for free.
    const fn compute_r_squared(n: &[u64; N]) -> [u64; N] {
        let mut x = [0u64; N];
        x[0] = 1;
        let mut iter = 0;
        let target = 128u32 * N as u32;
        while iter < target {
            // x := 2x; if x >= n or carried out, x -= n.
            let mut carry: u64 = 0;
            let mut i = 0;
            while i < N {
                let new = (x[i] << 1) | carry;
                carry = x[i] >> 63;
                x[i] = new;
                i += 1;
            }
            // Try the subtraction unconditionally; `borrow == 0` means
            // `x >= n`. Combined with `carry == 1` (overflowed past
            // `2^{64N}`), we always want to subtract in those cases.
            let (sub, borrow) = BigInt::<N>::mag_sub(&x, n);
            if carry == 1 || borrow == 0 {
                x = sub;
            }
            iter += 1;
        }
        x
    }
}

impl<const N: usize> BigInt<N> {
    /// Modular reduction: `self mod modulus`. Returns a value in
    /// `[0, |modulus|)`.
    ///
    /// Uses Euclidean division (Knuth Algorithm D) — see
    /// [`div_rem`](Self::div_rem) for the underlying routine.
    ///
    /// # Panics
    ///
    /// Panics if `modulus` is zero.
    #[inline]
    pub fn ct_mod(&self, modulus: &Self) -> Self {
        let (_, r) = self.div_rem(modulus);
        r
    }

    /// Modular exponentiation: `base^exp mod modulus`.
    ///
    /// Uses Montgomery arithmetic when the modulus is odd (the common
    /// case): conversion in/out plus square-and-multiply with CIOS
    /// Montgomery multiplication, no per-step division. Falls back to
    /// schoolbook square-and-multiply (`ct_mul` + `ct_mod`) for even
    /// moduli, where Montgomery doesn't apply.
    ///
    /// # Width requirement
    ///
    /// The inner squaring `result * result` can reach `(modulus - 1)²`
    /// before the reduction. For the result to not silently truncate,
    /// `BigInt<N>` must satisfy `64*N >= 2*bits(modulus)`. If `modulus`
    /// is larger than that bound, use [`pow_mod_w`](Self::pow_mod_w)
    /// with a wider working type.
    pub fn pow_mod(base: &Self, exp: &Self, modulus: &Self) -> Self {
        // Montgomery requires an odd modulus.
        if let Some(ctx) = MontReducer::<N>::new(modulus) {
            return ctx.pow(base, exp);
        }
        Self::pow_mod_schoolbook(base, exp, modulus)
    }

    /// Schoolbook square-and-multiply fallback for even moduli. Kept
    /// public(crate) so MontReducer::pow can delegate when the exponent
    /// loop trivially terminates.
    fn pow_mod_schoolbook(base: &Self, exp: &Self, modulus: &Self) -> Self {
        let mut result = Self::ONE;
        let bs = exp.bitsize();
        let mut i = bs;
        while i > 0 {
            i -= 1;
            result = result.ct_mul(&result).ct_mod(modulus);
            let limb_idx = (i / 64) as usize;
            let bit_idx = i % 64;
            let bit = (exp.limbs[limb_idx] >> bit_idx) & 1;
            if bit == 1 {
                result = result.ct_mul(base).ct_mod(modulus);
            }
        }
        result
    }

    /// Modular exponentiation at a wider working width `W`.
    ///
    /// Widens the operands to `BigInt<W>`, runs [`pow_mod`](Self::pow_mod)
    /// at that width, then narrows the result back to `BigInt<N>`.
    /// Use this when the storage width `N` is not big enough for the
    /// squarings inside `pow_mod` to fit without truncation — that is,
    /// whenever `64*N < 2*bits(modulus)`.
    ///
    /// # Width requirements
    ///
    /// - Compile-time: `W >= N` (enforced by a const assertion).
    /// - Runtime invariant: `64*W >= 2*bits(modulus)`. The caller is
    ///   responsible for choosing `W` large enough for their modulus. If this
    ///   is violated, the wider `pow_mod` will also silently truncate.
    ///
    /// For the SQIsign v2 commitment modulus
    /// `D_mix = 2^512 + 75` (513 bits), use at least `W = 18`.
    pub fn pow_mod_w<const W: usize>(base: &Self, exp: &Self, modulus: &Self) -> Self {
        const {
            assert!(
                W >= N,
                "pow_mod_w: working width W must be >= storage width N"
            )
        };
        let base_w: BigInt<W> = base.widen();
        let exp_w: BigInt<W> = exp.widen();
        let modulus_w: BigInt<W> = modulus.widen();
        let result_w = BigInt::<W>::pow_mod(&base_w, &exp_w, &modulus_w);
        result_w
            .narrow_to::<N>()
            .expect("pow_mod_w result < modulus < 2^(64N) fits in BigInt<N>")
    }
}

impl<const N: usize> BigInt<N> {
    /// Modular square root: returns x such that x² ≡ n (mod m),
    /// or `None` if n is not a quadratic residue mod m.
    ///
    /// Requires m to be an odd prime. Implements [Alg. 3.1] from
    /// the spec, with fast paths for m ≡ 3 (mod 4) and m ≡ 5 (mod 8),
    /// and Tonelli-Shanks for the general case m ≡ 1 (mod 8).
    ///
    /// # Width requirement
    ///
    /// All internal operations use [`pow_mod`](Self::pow_mod) and
    /// direct `ct_mul`/`ct_mod` at width `N`. The caller must ensure
    /// `64*N >= 2*bits(m)` — otherwise the squarings silently
    /// truncate and the result is wrong. For larger moduli use
    /// [`modular_sqrt_w`](Self::modular_sqrt_w).
    ///
    /// [Alg. 3.1]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.1
    pub fn modular_sqrt(n: &Self, m: &Self) -> Option<Self> {
        let n_mod = n.ct_mod(m);
        if bool::from(n_mod.is_zero()) {
            return Some(Self::ZERO);
        }

        // Build Montgomery context once for this modulus and share
        // across all pow_mod calls. The general Tonelli-Shanks branch
        // below issues 4+ pow_mods per call plus more inside the
        // Newton-style adjustment loop; without caching, each would
        // rebuild Newton iteration for n_inv and the 128·N doublings
        // for R². Fall back to the schoolbook path if `m` is even
        // (Montgomery requires an odd modulus).
        //
        // CT note: see `MontReducer`'s doc. Within-function caching is
        // safe even when `m` is secret-derived (e.g. via
        // `cornacchia` from `random_prime_norm`) because the context
        // is dropped before this function returns — no cross-call
        // cache-occupancy channel.
        let ctx = MontReducer::<N>::new(m);

        // Helper: mod-pow either via cached ctx or schoolbook fallback.
        let pow = |b: &Self, e: &Self| -> Self {
            match &ctx {
                Some(c) => c.pow(b, e),
                None => Self::pow_mod(b, e, m),
            }
        };

        let m_mod4 = m.as_limbs()[0] & 3;
        let m_mod8 = m.as_limbs()[0] & 7;

        // m ≡ 3 (mod 4): return n^((m+1)/4) mod m.
        if m_mod4 == 3 {
            let exp = m.ct_add(&Self::ONE) >> 2;
            let r = pow(&n_mod, &exp);
            let check = r.ct_mul(&r).ct_mod(m);
            return if check == n_mod { Some(r) } else { None };
        }

        // m ≡ 5 (mod 8):
        if m_mod8 == 5 {
            // Check if n^((m-1)/4) ≡ 1 mod m.
            let exp_check = m.ct_sub(&Self::ONE) >> 2;
            let test = pow(&n_mod, &exp_check);
            if test == Self::ONE {
                // return n^((m+3)/8) mod m
                let exp = m.ct_add(&Self::THREE) >> 3;
                return Some(pow(&n_mod, &exp));
            } else {
                // return 2n(4n)^((m-5)/8) mod m
                let four_n = n_mod.ct_mul(&Self::from_u64(4)).ct_mod(m);
                let exp = m.ct_sub(&Self::from_u64(5)) >> 3;
                let base = pow(&four_n, &exp);
                let r = Self::TWO.ct_mul(&n_mod).ct_mul(&base).ct_mod(m);
                let check = r.ct_mul(&r).ct_mod(m);
                return if check == n_mod { Some(r) } else { None };
            }
        }

        // General Tonelli-Shanks (m ≡ 1 mod 8).
        let e = m.ct_sub(&Self::ONE).two_adic_val();
        let q = m.ct_sub(&Self::ONE) >> e;

        // Find a non-residue w.
        let mut w = Self::TWO;
        loop {
            let exp = m.ct_sub(&Self::ONE) >> 1;
            let ls = pow(&w, &exp);
            // Legendre symbol: if ls == m - 1, then w is a non-residue.
            if ls == m.ct_sub(&Self::ONE) {
                break;
            }
            w = w.ct_add(&Self::ONE);
            // Safety bound.
            if w > *m {
                return None;
            }
        }

        let mut z = pow(&w, &q);
        let mut y = pow(&n_mod, &q);
        let mut x = pow(&n_mod, &(q.ct_add(&Self::ONE) >> 1));
        let mut f = Self::from_u64(1u64 << (e - 2));

        for _i in 0..e.saturating_sub(1) {
            let b = pow(&y, &f);
            if b == m.ct_sub(&Self::ONE) {
                // b ≡ -1 mod m
                x = x.ct_mul(&z).ct_mod(m);
                y = y.ct_mul(&z).ct_mul(&z).ct_mod(m);
            }
            z = z.ct_mul(&z).ct_mod(m);
            f = f >> 1;
        }

        let check = x.ct_mul(&x).ct_mod(m);
        if check == n_mod { Some(x) } else { None }
    }

    /// Modular square root at a wider working width `W`.
    ///
    /// Widens `n` and `m` to `BigInt<W>` and runs
    /// [`modular_sqrt`](Self::modular_sqrt) at that width. Use when
    /// `64*N < 2*bits(m)` would otherwise silently truncate the
    /// Tonelli-Shanks exponentiations.
    ///
    /// See [`pow_mod_w`](Self::pow_mod_w) for the width constraints.
    pub fn modular_sqrt_w<const W: usize>(n: &Self, m: &Self) -> Option<Self> {
        const {
            assert!(
                W >= N,
                "modular_sqrt_w: working width W must be >= storage width N"
            )
        };
        let n_w: BigInt<W> = n.widen();
        let m_w: BigInt<W> = m.widen();
        let r_w = BigInt::<W>::modular_sqrt(&n_w, &m_w)?;
        r_w.narrow_to::<N>()
    }
    /// Legendre symbol: returns 1 if `a` is a quadratic residue mod
    /// `p`, -1 if not, 0 if a ≡ 0 mod p. Requires `p` odd prime.
    ///
    /// # Width requirement
    ///
    /// Uses Euler's criterion via [`pow_mod`](Self::pow_mod), which
    /// requires `64*N >= 2*bits(p)`. For larger primes use
    /// [`legendre_w`](Self::legendre_w).
    pub fn legendre(a: &Self, p: &Self) -> i32 {
        let a_mod = a.ct_mod(p);
        if bool::from(a_mod.is_zero()) {
            return 0;
        }
        let exp = p.ct_sub(&Self::ONE) >> 1;
        let result = Self::pow_mod(&a_mod, &exp, p);
        if result == Self::ONE { 1 } else { -1 }
    }

    /// Legendre symbol at a wider working width `W`.
    ///
    /// Widens `a` and `p` to `BigInt<W>` and runs [`legendre`](Self::legendre)
    /// at that width. See [`pow_mod_w`](Self::pow_mod_w) for the width
    /// constraints. Use when `64*N < 2*bits(p)` would otherwise
    /// silently truncate the Euler exponentiation.
    pub fn legendre_w<const W: usize>(a: &Self, p: &Self) -> i32 {
        const {
            assert!(
                W >= N,
                "legendre_w: working width W must be >= storage width N"
            )
        };
        let a_w: BigInt<W> = a.widen();
        let p_w: BigInt<W> = p.widen();
        BigInt::<W>::legendre(&a_w, &p_w)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{BigInt, MontReducer};

    /// Montgomery round-trip oracle: `to_mont -> mul -> reduce_mont`
    /// must equal the independent schoolbook `(x * y) mod n`.
    ///
    /// The oracle multiplies at a doubled width `2N` (so the `x*y`
    /// product never truncates) and reduces — a fully independent path
    /// that cross-checks `MontReducer::mul`'s CIOS loop. Inputs are
    /// reduced mod `n` first so they satisfy `mul`'s `[0, n)` contract.
    fn check_mont_mul<const N: usize, const N2: usize>(
        x: &BigInt<N>,
        y: &BigInt<N>,
        n: &BigInt<N>,
    ) {
        let Some(ctx) = MontReducer::<N>::new(n) else {
            return; // even modulus: Montgomery doesn't apply
        };
        let x_red = x.ct_mod(n);
        let y_red = y.ct_mod(n);

        let xm = ctx.to_montgomery(&x_red.limbs);
        let ym = ctx.to_montgomery(&y_red.limbs);
        let prod = ctx.reduce_montgomery(&ctx.mul(&xm, &ym));
        let via_mont = BigInt::<N>::from_limbs(prod);

        // Oracle: widen to 2N, full-width multiply, reduce, narrow back.
        let xw: BigInt<N2> = x_red.widen();
        let yw: BigInt<N2> = y_red.widen();
        let nw: BigInt<N2> = n.widen();
        let via_oracle = (xw.ct_mul(&yw).ct_mod(&nw))
            .narrow_to::<N>()
            .expect("product mod n < n < 2^(64N) fits in N limbs");
        assert_eq!(via_mont, via_oracle);
    }

    /// Builds an odd `BigInt<N>` >= 3 from random limbs (forces bit 0 set
    /// and a nonzero high half so the modulus is a genuine `N`-limb odd).
    fn arb_odd_modulus<const N: usize>() -> impl Strategy<Value = BigInt<N>> {
        prop::array::uniform(any::<u64>()).prop_map(|mut limbs| {
            limbs[0] |= 1;
            limbs[N - 1] |= 1 << 63;
            BigInt::from_limbs(limbs)
        })
    }

    fn arb_bigint<const N: usize>() -> impl Strategy<Value = BigInt<N>> {
        prop::array::uniform(any::<u64>()).prop_map(BigInt::from_limbs)
    }

    proptest! {
        #[test]
        fn prop_mont_mul_matches_schoolbook_n4(
            x in arb_bigint::<4>(), y in arb_bigint::<4>(), n in arb_odd_modulus::<4>(),
        ) {
            check_mont_mul::<4, 8>(&x, &y, &n);
        }

        #[test]
        fn prop_mont_mul_matches_schoolbook_n9(
            x in arb_bigint::<9>(), y in arb_bigint::<9>(), n in arb_odd_modulus::<9>(),
        ) {
            check_mont_mul::<9, 18>(&x, &y, &n);
        }

        #[test]
        fn prop_mont_mul_matches_schoolbook_n17(
            x in arb_bigint::<17>(), y in arb_bigint::<17>(), n in arb_odd_modulus::<17>(),
        ) {
            check_mont_mul::<17, 34>(&x, &y, &n);
        }

        #[test]
        fn prop_mont_mul_matches_schoolbook_n18(
            x in arb_bigint::<18>(), y in arb_bigint::<18>(), n in arb_odd_modulus::<18>(),
        ) {
            check_mont_mul::<18, 36>(&x, &y, &n);
        }
    }

    #[test]
    fn mont_mul_edge_cases_n17() {
        // n = 2^(64*17) - 1 (odd), x = y = n - 1.
        let n_limbs = [u64::MAX; 17];
        let n = BigInt::<17>::from_limbs(n_limbs);
        let mut xm1 = n_limbs;
        xm1[0] -= 1;
        let x = BigInt::<17>::from_limbs(xm1);
        check_mont_mul::<17, 34>(&x, &x, &n);
        check_mont_mul::<17, 34>(&BigInt::ZERO, &x, &n);
        check_mont_mul::<17, 34>(&BigInt::ONE, &x, &n);
    }
}
