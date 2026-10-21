//! `Fp64` backend: radix-2^64 Montgomery on `[u64; 6]`.
//!
//! Storage matches the SQIsign C reference's saturated 64-bit backend
//! (`src/gf/sat64/p324_3`): six full-width u64 limbs, 58 spare bits
//! above the 326-bit modulus.  Designed for MULX + dual-chain
//! ADCX/ADOX asm schoolbook; full-width limbs let each partial
//! product's `(hi, lo)` land at adjacent accumulator positions for
//! row-major dual-chain accumulation.
//!
//! Montgomery factor `R = 2^384`.  The `from_limbs([u64; 6])`
//! const-bridge below repacks the portable backend's radix-55 limbs
//! (`R = 2^330`) and rescales by `2^54` through a compile-time
//! Montgomery multiplication, so the same `pub const FOO: Fp =
//! Fp::from_limbs([...])` precomputed tables compile identically
//! across backends.
//!
//! Reduction uses `p + 1 = 0x30 * 2^320` and `-p^-1 = 1 mod 2^64`:
//! each CIOS step drops the low accumulator limb and adds that limb
//! times `0x30` at limb positions 5 and 6.

use core::{
    arch::asm,
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::fields::fp::FP_ENCODED_BYTES;

/// Number of limbs.
const LIMBS: usize = 6;

/// The top limb of `p + 1 = 3 * 2^324` in `[u64; 6]` LE form (the only
/// non-zero limb, at position 5).  The CIOS-style Montgomery reduction
/// in the asm multiplies the low accumulator limb by this constant to
/// fold the reduction into the upper limbs without a modular inverse.
/// Read through a RIP-relative memory operand so the asm stays within
/// the 14 allocatable general registers.
static P_PLUS_1_HI: u64 = 0x30;

#[cfg(test)]
mod tests;

/// An element of `F_p` (where `p = 3 * 2^324 - 1`) in radix-2^64
/// Montgomery form, packed into six full-width u64 limbs.
///
/// **Lazy reduction**: values are kept in `[0, 2p)`, not canonical
/// `[0, p)` (matching the C reference, whose `fp_mul` skips the final
/// conditional subtract).  `mul`/`square` accept and return `[0, 2p)`;
/// `add`/`sub`/`neg` reduce to `[0, 2p)`.  A field element therefore
/// has two representatives (`x` and `x + p`), so `PartialEq`, `ct_eq`,
/// and `to_bytes` normalize to `[0, p)` first.  `PartialEq`/`Eq` are
/// hand-written, not derived.
#[derive(Copy, Clone)]
pub struct Fp64(pub(crate) [u64; LIMBS]);

impl Fp64 {
    /// Constructs from raw radix-2^64 Montgomery limbs.
    ///
    /// Caller is responsible for the limbs already encoding a valid
    /// Montgomery-form field element.  For the cross-backend
    /// const-bridge from the portable backend's radix-55 limbs, use
    /// [`Fp64::from_limbs`] instead.
    pub const fn from_raw(limbs: [u64; LIMBS]) -> Self {
        Self(limbs)
    }

    /// The additive identity in Montgomery form.
    pub const ZERO: Self = Self([0; LIMBS]);

    /// The multiplicative identity in Montgomery form: `2^384 mod p`.
    pub const ONE: Self = Self([0x0555555555555555, 0, 0, 0, 0, 0x0000000000000010]);

    /// `2 * ONE` mod p, in Montgomery form.
    pub const TWO: Self = Self([0x0AAAAAAAAAAAAAAA, 0, 0, 0, 0, 0x0000000000000020]);

    /// `4 * ONE` mod p, in Montgomery form.
    pub const FOUR: Self = Self([0x1555555555555555, 0, 0, 0, 0, 0x0000000000000010]);

    /// `2^-1 mod p` in Montgomery form.
    pub const TWO_INV: Self = Self([0x02AAAAAAAAAAAAAA, 0, 0, 0, 0, 0x0000000000000020]);

    /// `-1 mod p` in Montgomery form.
    pub const MINUS_ONE: Self = Self([
        0xFAAAAAAAAAAAAAAA,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x000000000000001F,
    ]);

    /// The modulus `p = 3 * 2^324 - 1`.  Not a canonical element, but
    /// load-bearing for the conditional subtracts.
    pub(crate) const P: Self = Self([
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x000000000000002F,
    ]);

    /// `2p`, the reduction constant for the lazy-`[0, 2p)` invariant.
    /// `add`/`sub`/`neg` produce values in `[0, 4p)` and subtract this
    /// once to land back in `[0, 2p)`.
    const TWO_P: Self = Self([
        0xFFFFFFFFFFFFFFFE,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x000000000000005F,
    ]);

    /// `R^2 mod p` for converting in/out of Montgomery form (`R = 2^384`).
    pub(crate) const R2: Self = Self([
        0xC71C71C71C71C71C,
        0x5571C71C71C71C71,
        0x5555555555555555,
        0x5555555555555555,
        0x5555555555555555,
        0x0000000000000015,
    ]);

    /// Const-bridge from the portable backend's radix-2^55 Montgomery limbs.
    ///
    /// Repacks `[u64; 6]` radix-2^55 into `[u64; 6]` radix-2^64 (the
    /// integer value `v * 2^330 mod p`, possibly unreduced) and then
    /// Montgomery-multiplies by `K = 2^438 mod p`, which lands the value
    /// at `v * 2^330 * 2^438 * 2^-384 = v * 2^384`.  The compile-time
    /// multiply is [`Fp64::mont_mul_const`], the plain-Rust twin of the
    /// asm schedule.
    ///
    /// Signature-compatible with the portable backend's `from_limbs`
    /// and with `Fp29::from_limbs` / `Fp26::from_limbs`, so the same
    /// `pub const FOO: Fp = Fp::from_limbs([...])` tables compile across
    /// every backend.
    pub const fn from_limbs(portable_mont: [u64; LIMBS]) -> Self {
        // K = 2^438 mod p in radix-2^64.
        const K: [u64; LIMBS] = [
            0x5555555555555555,
            0x0001555555555555,
            0,
            0,
            0,
            0x0000000000000010,
        ];

        let mut acc: u128 = 0;
        let mut bits: u32 = 0;
        let mut src = 0;
        let mut out = [0u64; LIMBS];
        let mut i = 0;
        while i < LIMBS {
            while bits < 64 && src < LIMBS {
                acc |= (portable_mont[src] as u128) << bits;
                bits += 55;
                src += 1;
            }
            out[i] = acc as u64;
            acc >>= 64;
            bits = bits.saturating_sub(64);
            i += 1;
        }

        Self(Self::mont_mul_const(out, K))
    }

    /// Plain-Rust Montgomery multiplication with the asm's schedule:
    /// six CIOS steps, each dropping the low limb and folding it times
    /// `0x30` into limbs 5 and 6.  Used at compile time by
    /// [`Fp64::from_limbs`] and by the tests as the asm's oracle.
    pub(crate) const fn mont_mul_const(a: [u64; LIMBS], b: [u64; LIMBS]) -> [u64; LIMBS] {
        // Seven-limb accumulator; `t[0]` is always the limb about to be
        // folded away.
        let mut t = [0u64; LIMBS + 1];
        let mut k = 0;
        while k < LIMBS {
            // t += a[k] * b.
            let mut carry: u128 = 0;
            let mut j = 0;
            while j < LIMBS {
                let v = (t[j] as u128) + (a[k] as u128) * (b[j] as u128) + carry;
                t[j] = v as u64;
                carry = v >> 64;
                j += 1;
            }
            t[LIMBS] = t[LIMBS].wrapping_add(carry as u64);

            // Fold: t += t[0] * p.  With p + 1 = 0x30 * 2^320 this zeroes
            // t[0] and adds t[0] * 0x30 at limbs 5 and 6.
            let q = t[0] as u128;
            let f = q * (P_PLUS_1_HI as u128);
            let v5 = (t[5] as u128) + (f as u64 as u128);
            t[5] = v5 as u64;
            t[6] = t[6]
                .wrapping_add((f >> 64) as u64)
                .wrapping_add((v5 >> 64) as u64);

            // Shift down one limb.
            let mut j = 0;
            while j < LIMBS {
                t[j] = t[j + 1];
                j += 1;
            }
            t[LIMBS] = 0;
            k += 1;
        }
        [t[0], t[1], t[2], t[3], t[4], t[5]]
    }

    /// Conditional subtract of `p` in a `const` context.
    ///
    /// Returns `self - p` if `self >= p`; else `self` unchanged.
    const fn cond_sub_p_const(self) -> Self {
        let p = Self::P.0;
        let mut d = [0u64; LIMBS];
        let mut borrow = false;
        let mut i = 0;
        while i < LIMBS {
            let (x, b1) = self.0[i].overflowing_sub(p[i]);
            let (y, b2) = x.overflowing_sub(borrow as u64);
            d[i] = y;
            borrow = b1 | b2;
            i += 1;
        }
        if borrow { self } else { Self(d) }
    }

    /// Reduces `self` from `[0, 2p)` to `[0, p)` by conditionally
    /// subtracting `p`, branch-free via [`ConditionallySelectable`].
    fn final_sub_p(self) -> Self {
        let p = Self::P.0;
        let mut d = [0u64; LIMBS];
        let mut borrow = false;
        for i in 0..LIMBS {
            let (x, b) = self.0[i].borrowing_sub(p[i], borrow);
            d[i] = x;
            borrow = b;
        }
        // `borrow` iff `self < p`; in that case keep `self`.
        Self::conditional_select(&Self(d), &self, Choice::from(borrow as u8))
    }

    /// Reduces a value in `[0, 4p)` to `[0, 2p)` by subtracting `2p`
    /// when `self >= 2p` (constant-time via a borrow-masked select).
    fn cond_sub_2p(self) -> Self {
        let tp = Self::TWO_P.0;
        let mut d = [0u64; LIMBS];
        let mut borrow = false;
        for i in 0..LIMBS {
            let (x, b) = self.0[i].borrowing_sub(tp[i], borrow);
            d[i] = x;
            borrow = b;
        }
        Self::conditional_select(&Self(d), &self, Choice::from(borrow as u8))
    }

    /// Conditionally adds `2p` to `self` (mod 2^384) when `cond`.  Used
    /// by `Sub`: an underflowing `a - b` (with `a, b` in `[0, 2p)`)
    /// wraps; adding `2p` back lands the result in `[0, 2p)`.
    fn cond_add_2p(self, cond: Choice) -> Self {
        let tp = Self::TWO_P.0;
        let mut s = [0u64; LIMBS];
        let mut carry = false;
        for i in 0..LIMBS {
            let (x, c) = self.0[i].carrying_add(tp[i], carry);
            s[i] = x;
            carry = c;
        }
        Self::conditional_select(&self, &Self(s), cond)
    }

    /// Montgomery multiplication: `self * rhs * R^{-1} mod p` with
    /// `R = 2^384`.
    ///
    /// MULX schoolbook + dual-chain ADCX/ADOX accumulation, interleaved
    /// with CIOS-style Montgomery reduction folded through the
    /// special-form constant `p + 1 = 0x30 * 2^320`.  The 36 partial
    /// products span a 7-limb rotating accumulator; after each row the
    /// low limb is folded into limbs 5 and 6, leaving the rotated
    /// accumulator one limb shorter for the next row.
    ///
    /// Inputs and output are in `[0, 2p)`: for inputs below `2p` the
    /// product is below `4p^2 < 2p * R`.
    ///
    /// # Constant-time
    ///
    /// No data-dependent branches or memory addressing.
    #[inline]
    fn mul_montgomery(a: &Self, b: &Self) -> Self {
        let mut out = [0u64; LIMBS];
        // SAFETY: cfg-gated at the parent module on bmi2 + adx; the asm
        // uses MULX/ADCX/ADOX unconditionally.  Reads 48 bytes from each
        // of `a` and `b`, writes 48 bytes to `out`, reads the static
        // fold constant through a RIP-relative operand.  `nostack`:
        // no stack use.
        unsafe {
            asm!(
                // Prologue: a[0] * b into the 7-limb accumulator.
                "mov rdx, qword ptr [{a} + 0]",
                "mulx {z1}, {z0}, qword ptr [{b} + 0]",
                "xor eax, eax",
                "mulx {z2}, {t1}, qword ptr [{b} + 8]",
                "adox {z1}, {t1}",
                "mulx {z3}, {t1}, qword ptr [{b} + 16]",
                "adox {z2}, {t1}",
                "mulx {z4}, {t1}, qword ptr [{b} + 24]",
                "adox {z3}, {t1}",
                "mulx {z5}, {t1}, qword ptr [{b} + 32]",
                "adox {z4}, {t1}",
                "mulx {z6}, {t1}, qword ptr [{b} + 40]",
                "adox {z5}, {t1}",
                "adox {z6}, rax",
                // Step 0: fold limb 0 through p + 1 = 0x30 * 2^320, then rotate.
                "mov rdx, {z0}",
                "mulx {t0}, {t1}, qword ptr [rip + {p1}]",
                "xor eax, eax",
                "adox {z5}, {t1}",
                "adox {z6}, {t0}",
                // Row 1: accumulate a[1] * b; the freed slot becomes the top limb.
                "mov rdx, qword ptr [{a} + 8]",
                "mulx {t0}, {t1}, qword ptr [{b} + 0]",
                "xor {z0:e}, {z0:e}",
                "adox {z1}, {t1}",
                "adox {z2}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 8]",
                "adcx {z2}, {t1}",
                "adox {z3}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 16]",
                "adcx {z3}, {t1}",
                "adox {z4}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 24]",
                "adcx {z4}, {t1}",
                "adox {z5}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 32]",
                "adcx {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 40]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "adc {z0}, 0",
                // Step 1: fold limb 0 through p + 1 = 0x30 * 2^320, then rotate.
                "mov rdx, {z1}",
                "mulx {t0}, {t1}, qword ptr [rip + {p1}]",
                "xor eax, eax",
                "adox {z6}, {t1}",
                "adox {z0}, {t0}",
                // Row 2: accumulate a[2] * b; the freed slot becomes the top limb.
                "mov rdx, qword ptr [{a} + 16]",
                "mulx {t0}, {t1}, qword ptr [{b} + 0]",
                "xor {z1:e}, {z1:e}",
                "adox {z2}, {t1}",
                "adox {z3}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 8]",
                "adcx {z3}, {t1}",
                "adox {z4}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 16]",
                "adcx {z4}, {t1}",
                "adox {z5}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 24]",
                "adcx {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 32]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 40]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "adc {z1}, 0",
                // Step 2: fold limb 0 through p + 1 = 0x30 * 2^320, then rotate.
                "mov rdx, {z2}",
                "mulx {t0}, {t1}, qword ptr [rip + {p1}]",
                "xor eax, eax",
                "adox {z0}, {t1}",
                "adox {z1}, {t0}",
                // Row 3: accumulate a[3] * b; the freed slot becomes the top limb.
                "mov rdx, qword ptr [{a} + 24]",
                "mulx {t0}, {t1}, qword ptr [{b} + 0]",
                "xor {z2:e}, {z2:e}",
                "adox {z3}, {t1}",
                "adox {z4}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 8]",
                "adcx {z4}, {t1}",
                "adox {z5}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 16]",
                "adcx {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 24]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 32]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 40]",
                "adcx {z1}, {t1}",
                "adox {z2}, {t0}",
                "adc {z2}, 0",
                // Step 3: fold limb 0 through p + 1 = 0x30 * 2^320, then rotate.
                "mov rdx, {z3}",
                "mulx {t0}, {t1}, qword ptr [rip + {p1}]",
                "xor eax, eax",
                "adox {z1}, {t1}",
                "adox {z2}, {t0}",
                // Row 4: accumulate a[4] * b; the freed slot becomes the top limb.
                "mov rdx, qword ptr [{a} + 32]",
                "mulx {t0}, {t1}, qword ptr [{b} + 0]",
                "xor {z3:e}, {z3:e}",
                "adox {z4}, {t1}",
                "adox {z5}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 8]",
                "adcx {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 16]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 24]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 32]",
                "adcx {z1}, {t1}",
                "adox {z2}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 40]",
                "adcx {z2}, {t1}",
                "adox {z3}, {t0}",
                "adc {z3}, 0",
                // Step 4: fold limb 0 through p + 1 = 0x30 * 2^320, then rotate.
                "mov rdx, {z4}",
                "mulx {t0}, {t1}, qword ptr [rip + {p1}]",
                "xor eax, eax",
                "adox {z2}, {t1}",
                "adox {z3}, {t0}",
                // Row 5: accumulate a[5] * b; the freed slot becomes the top limb.
                "mov rdx, qword ptr [{a} + 40]",
                "mulx {t0}, {t1}, qword ptr [{b} + 0]",
                "xor {z4:e}, {z4:e}",
                "adox {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 8]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 16]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 24]",
                "adcx {z1}, {t1}",
                "adox {z2}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 32]",
                "adcx {z2}, {t1}",
                "adox {z3}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 40]",
                "adcx {z3}, {t1}",
                "adox {z4}, {t0}",
                "adc {z4}, 0",
                // Step 5: fold limb 0 through p + 1 = 0x30 * 2^320, then rotate.
                "mov rdx, {z5}",
                "mulx {t0}, {t1}, qword ptr [rip + {p1}]",
                "xor eax, eax",
                "adox {z3}, {t1}",
                "adox {z4}, {t0}",
                // Result: positions 0..5 after the last rotation.
                "mov qword ptr [{out} + 0], {z6}",
                "mov qword ptr [{out} + 8], {z0}",
                "mov qword ptr [{out} + 16], {z1}",
                "mov qword ptr [{out} + 24], {z2}",
                "mov qword ptr [{out} + 32], {z3}",
                "mov qword ptr [{out} + 40], {z4}",
                a = in(reg) a.0.as_ptr(),
                b = in(reg) b.0.as_ptr(),
                out = in(reg) out.as_mut_ptr(),
                p1 = sym P_PLUS_1_HI,
                z0 = out(reg) _,
                z1 = out(reg) _,
                z2 = out(reg) _,
                z3 = out(reg) _,
                z4 = out(reg) _,
                z5 = out(reg) _,
                z6 = out(reg) _,
                t0 = out(reg) _,
                t1 = out(reg) _,
                out("rax") _,
                out("rdx") _,
                options(nostack),
            );
        }
        Self(out)
    }

    /// Modular squaring: `self * self mod p`, via [`Fp64::mul_montgomery`].
    #[inline]
    #[must_use]
    pub fn square(&self) -> Self {
        Self::mul_montgomery(self, self)
    }

    /// Computes `m[0..6] * s[6..12] + m[6..12] * s[0..6] (mod p)`, in
    /// `[0, 2p)`.
    ///
    /// One fused CIOS pass: per column it accumulates a limb of both
    /// 6x6 products before a single shared Montgomery reduction.
    /// Sharing the reduction across the two products halves the
    /// reduction work versus two separate multiplications.  For inputs
    /// in `[0, 2p)` the value `A*B + C*D < 8p^2 < 2p * R`, so the
    /// output is `< 2p`.
    ///
    /// # Constant-time
    ///
    /// No data-dependent branches or memory addressing.
    // `inline(never)`: the asm holds 14 register operands, which fits a
    // standalone frame but not when inlined into the two-coordinate
    // `Fp2::mul`.
    #[inline(never)]
    fn sum_of_products_packed(m: &[u64; 2 * LIMBS], s: &[u64; 2 * LIMBS]) -> Self {
        let mut o = [0u64; LIMBS];
        // SAFETY: cfg-gated on bmi2 + adx; no operand-dependent memory
        // or control flow.  Reads m[0..12], s[0..12]; writes o[0..6].
        unsafe {
            asm!(
                // Column 0: b1[0] * a1 (prologue chain) plus b2[0] * a2, then fold.
                "mov rdx, qword ptr [{s} + 48]",
                "mulx {z1}, {z0}, qword ptr [{m} + 0]",
                "xor eax, eax",
                "mulx {z2}, {t1}, qword ptr [{m} + 8]",
                "adox {z1}, {t1}",
                "mulx {z3}, {t1}, qword ptr [{m} + 16]",
                "adox {z2}, {t1}",
                "mulx {z4}, {t1}, qword ptr [{m} + 24]",
                "adox {z3}, {t1}",
                "mulx {z5}, {t1}, qword ptr [{m} + 32]",
                "adox {z4}, {t1}",
                "mulx {z6}, {t1}, qword ptr [{m} + 40]",
                "adox {z5}, {t1}",
                "adox {z6}, rax",
                "mov rdx, qword ptr [{s} + 0]",
                "mulx {t0}, {t1}, qword ptr [{m} + 48]",
                "xor eax, eax",
                "adox {z0}, {t1}",
                "adox {z1}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 56]",
                "adcx {z1}, {t1}",
                "adox {z2}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 64]",
                "adcx {z2}, {t1}",
                "adox {z3}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 72]",
                "adcx {z3}, {t1}",
                "adox {z4}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 80]",
                "adcx {z4}, {t1}",
                "adox {z5}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 88]",
                "adcx {z5}, {t1}",
                "adox {z6}, {t0}",
                "adc {z6}, 0",
                "mov rdx, {z0}",
                "mulx {t0}, {t1}, qword ptr [rip + {p1}]",
                "xor eax, eax",
                "adox {z5}, {t1}",
                "adox {z6}, {t0}",
                // Column 1: b1[1] * a1 and b2[1] * a2 into the rotated accumulator, then fold.
                "mov rdx, qword ptr [{s} + 56]",
                "mulx {t0}, {t1}, qword ptr [{m} + 0]",
                "xor {z0:e}, {z0:e}",
                "adox {z1}, {t1}",
                "adox {z2}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 8]",
                "adcx {z2}, {t1}",
                "adox {z3}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 16]",
                "adcx {z3}, {t1}",
                "adox {z4}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 24]",
                "adcx {z4}, {t1}",
                "adox {z5}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 32]",
                "adcx {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 40]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "adc {z0}, 0",
                "mov rdx, qword ptr [{s} + 8]",
                "mulx {t0}, {t1}, qword ptr [{m} + 48]",
                "xor eax, eax",
                "adox {z1}, {t1}",
                "adox {z2}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 56]",
                "adcx {z2}, {t1}",
                "adox {z3}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 64]",
                "adcx {z3}, {t1}",
                "adox {z4}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 72]",
                "adcx {z4}, {t1}",
                "adox {z5}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 80]",
                "adcx {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 88]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "adc {z0}, 0",
                "mov rdx, {z1}",
                "mulx {t0}, {t1}, qword ptr [rip + {p1}]",
                "xor eax, eax",
                "adox {z6}, {t1}",
                "adox {z0}, {t0}",
                // Column 2: b1[2] * a1 and b2[2] * a2 into the rotated accumulator, then fold.
                "mov rdx, qword ptr [{s} + 64]",
                "mulx {t0}, {t1}, qword ptr [{m} + 0]",
                "xor {z1:e}, {z1:e}",
                "adox {z2}, {t1}",
                "adox {z3}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 8]",
                "adcx {z3}, {t1}",
                "adox {z4}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 16]",
                "adcx {z4}, {t1}",
                "adox {z5}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 24]",
                "adcx {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 32]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 40]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "adc {z1}, 0",
                "mov rdx, qword ptr [{s} + 16]",
                "mulx {t0}, {t1}, qword ptr [{m} + 48]",
                "xor eax, eax",
                "adox {z2}, {t1}",
                "adox {z3}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 56]",
                "adcx {z3}, {t1}",
                "adox {z4}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 64]",
                "adcx {z4}, {t1}",
                "adox {z5}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 72]",
                "adcx {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 80]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 88]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "adc {z1}, 0",
                "mov rdx, {z2}",
                "mulx {t0}, {t1}, qword ptr [rip + {p1}]",
                "xor eax, eax",
                "adox {z0}, {t1}",
                "adox {z1}, {t0}",
                // Column 3: b1[3] * a1 and b2[3] * a2 into the rotated accumulator, then fold.
                "mov rdx, qword ptr [{s} + 72]",
                "mulx {t0}, {t1}, qword ptr [{m} + 0]",
                "xor {z2:e}, {z2:e}",
                "adox {z3}, {t1}",
                "adox {z4}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 8]",
                "adcx {z4}, {t1}",
                "adox {z5}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 16]",
                "adcx {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 24]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 32]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 40]",
                "adcx {z1}, {t1}",
                "adox {z2}, {t0}",
                "adc {z2}, 0",
                "mov rdx, qword ptr [{s} + 24]",
                "mulx {t0}, {t1}, qword ptr [{m} + 48]",
                "xor eax, eax",
                "adox {z3}, {t1}",
                "adox {z4}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 56]",
                "adcx {z4}, {t1}",
                "adox {z5}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 64]",
                "adcx {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 72]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 80]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 88]",
                "adcx {z1}, {t1}",
                "adox {z2}, {t0}",
                "adc {z2}, 0",
                "mov rdx, {z3}",
                "mulx {t0}, {t1}, qword ptr [rip + {p1}]",
                "xor eax, eax",
                "adox {z1}, {t1}",
                "adox {z2}, {t0}",
                // Column 4: b1[4] * a1 and b2[4] * a2 into the rotated accumulator, then fold.
                "mov rdx, qword ptr [{s} + 80]",
                "mulx {t0}, {t1}, qword ptr [{m} + 0]",
                "xor {z3:e}, {z3:e}",
                "adox {z4}, {t1}",
                "adox {z5}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 8]",
                "adcx {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 16]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 24]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 32]",
                "adcx {z1}, {t1}",
                "adox {z2}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 40]",
                "adcx {z2}, {t1}",
                "adox {z3}, {t0}",
                "adc {z3}, 0",
                "mov rdx, qword ptr [{s} + 32]",
                "mulx {t0}, {t1}, qword ptr [{m} + 48]",
                "xor eax, eax",
                "adox {z4}, {t1}",
                "adox {z5}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 56]",
                "adcx {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 64]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 72]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 80]",
                "adcx {z1}, {t1}",
                "adox {z2}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 88]",
                "adcx {z2}, {t1}",
                "adox {z3}, {t0}",
                "adc {z3}, 0",
                "mov rdx, {z4}",
                "mulx {t0}, {t1}, qword ptr [rip + {p1}]",
                "xor eax, eax",
                "adox {z2}, {t1}",
                "adox {z3}, {t0}",
                // Column 5: b1[5] * a1 and b2[5] * a2 into the rotated accumulator, then fold.
                "mov rdx, qword ptr [{s} + 88]",
                "mulx {t0}, {t1}, qword ptr [{m} + 0]",
                "xor {z4:e}, {z4:e}",
                "adox {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 8]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 16]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 24]",
                "adcx {z1}, {t1}",
                "adox {z2}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 32]",
                "adcx {z2}, {t1}",
                "adox {z3}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 40]",
                "adcx {z3}, {t1}",
                "adox {z4}, {t0}",
                "adc {z4}, 0",
                "mov rdx, qword ptr [{s} + 40]",
                "mulx {t0}, {t1}, qword ptr [{m} + 48]",
                "xor eax, eax",
                "adox {z5}, {t1}",
                "adox {z6}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 56]",
                "adcx {z6}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 64]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 72]",
                "adcx {z1}, {t1}",
                "adox {z2}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 80]",
                "adcx {z2}, {t1}",
                "adox {z3}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{m} + 88]",
                "adcx {z3}, {t1}",
                "adox {z4}, {t0}",
                "adc {z4}, 0",
                "mov rdx, {z5}",
                "mulx {t0}, {t1}, qword ptr [rip + {p1}]",
                "xor eax, eax",
                "adox {z3}, {t1}",
                "adox {z4}, {t0}",
                // Result: positions 0..5 after the last rotation.
                "mov qword ptr [{o} + 0], {z6}",
                "mov qword ptr [{o} + 8], {z0}",
                "mov qword ptr [{o} + 16], {z1}",
                "mov qword ptr [{o} + 24], {z2}",
                "mov qword ptr [{o} + 32], {z3}",
                "mov qword ptr [{o} + 40], {z4}",
                m = in(reg) m.as_ptr(),
                s = in(reg) s.as_ptr(),
                o = in(reg) o.as_mut_ptr(),
                p1 = sym P_PLUS_1_HI,
                z0 = out(reg) _,
                z1 = out(reg) _,
                z2 = out(reg) _,
                z3 = out(reg) _,
                z4 = out(reg) _,
                z5 = out(reg) _,
                z6 = out(reg) _,
                t0 = out(reg) _,
                t1 = out(reg) _,
                out("rax") _,
                out("rdx") _,
                options(nostack),
            );
        }
        Self(o)
    }

    /// Returns `a1 * b1 + a2 * b2 (mod p)`, in `[0, 2p)`, via the
    /// shared-reduction kernel [`Fp64::sum_of_products_packed`].
    ///
    /// # Constant-time
    ///
    /// Constant-time on all four operands.
    #[inline]
    #[must_use]
    pub fn sum_of_2_products(a1: &Self, b1: &Self, a2: &Self, b2: &Self) -> Self {
        let mut m = [0u64; 2 * LIMBS];
        m[..LIMBS].copy_from_slice(&a1.0);
        m[LIMBS..].copy_from_slice(&a2.0);
        let mut s = [0u64; 2 * LIMBS];
        s[..LIMBS].copy_from_slice(&b2.0);
        s[LIMBS..].copy_from_slice(&b1.0);

        Self::sum_of_products_packed(&m, &s)
    }

    /// Returns `a1 * b1 - a2 * b2 (mod p)`, in `[0, 2p)`.
    ///
    /// The subtrahend's `b2` is negated as `2p - b2` (in `(0, 2p]` for
    /// `b2` in `[0, 2p)`, so no underflow), turning the difference into
    /// a non-negative sum handled by the same kernel.
    ///
    /// # Constant-time
    ///
    /// Constant-time on all four operands.
    #[inline]
    #[must_use]
    pub fn difference_of_2_products(a1: &Self, b1: &Self, a2: &Self, b2: &Self) -> Self {
        let tp = Self::TWO_P.0;
        let mut n = [0u64; LIMBS];
        let mut borrow = false;
        for i in 0..LIMBS {
            let (x, b) = tp[i].borrowing_sub(b2.0[i], borrow);
            n[i] = x;
            borrow = b;
        }

        let mut m = [0u64; 2 * LIMBS];
        m[..LIMBS].copy_from_slice(&a1.0);
        m[LIMBS..].copy_from_slice(&a2.0);
        let mut s = [0u64; 2 * LIMBS];
        s[..LIMBS].copy_from_slice(&n);
        s[LIMBS..].copy_from_slice(&b1.0);

        Self::sum_of_products_packed(&m, &s)
    }

    /// Constructs a field element from a small integer.
    #[must_use]
    pub fn from_small(x: u32) -> Self {
        let canonical = Self([x as u64, 0, 0, 0, 0, 0]);
        &canonical * &Self::R2
    }

    /// Decodes 41 bytes (little-endian) into a Montgomery-form `Fp64`.
    ///
    /// The input must be a canonical encoding (value `< p`).  Limbs 0..4
    /// take eight bytes each; limb 5 takes the last byte (`p < 2^326`).
    #[must_use]
    pub fn from_bytes(bytes: &[u8; FP_ENCODED_BYTES]) -> Self {
        let mut limbs = [0u64; LIMBS];
        let (chunks, rest) = bytes.as_slice().as_chunks::<8>();
        for (limb, chunk) in limbs.iter_mut().zip(chunks) {
            *limb = u64::from_le_bytes(*chunk);
        }
        limbs[LIMBS - 1] = rest[0] as u64;
        &Self(limbs) * &Self::R2
    }

    /// Encodes a Montgomery-form `Fp64` as 41 bytes, little-endian.
    ///
    /// Exits Montgomery form via `mul` by raw `1`, then `final_sub_p`
    /// normalizes the lazy `[0, 2p)` result to `[0, p)` before encoding.
    #[must_use]
    pub fn to_bytes(self) -> [u8; FP_ENCODED_BYTES] {
        let one_raw = Self([1, 0, 0, 0, 0, 0]);
        let canonical = (&self * &one_raw).final_sub_p();
        let mut out = [0u8; FP_ENCODED_BYTES];
        for (i, limb) in canonical.0[..LIMBS - 1].iter().enumerate() {
            out[8 * i..8 * i + 8].copy_from_slice(&limb.to_le_bytes());
        }
        out[FP_ENCODED_BYTES - 1] = canonical.0[LIMBS - 1] as u8;
        out
    }

    /// Squares `self` `n` times in succession.
    #[must_use]
    pub fn pow2k(&self, n: u32) -> Self {
        let mut r = *self;
        for _ in 0..n {
            r = r.square();
        }
        r
    }

    /// Computes `self^((p - 3)/4)`.
    ///
    /// Same addition chain as the portable backend's `pow_p3div4`.
    #[must_use]
    pub(crate) fn pow_p3div4(&self) -> Self {
        let x = *self;
        let z = x.square();
        let z = &x * &z;
        let t0 = z.pow2k(2);
        let t0 = &z * &t0;
        let t1 = t0.pow2k(4);
        let t0 = &t0 * &t1;
        let t1 = t0.pow2k(2);
        let z = &z * &t1;
        let t2 = z.pow2k(4);
        let t1 = t2.pow2k(4);
        let t3 = t1.pow2k(10);
        let t1 = &t1 * &t3;
        let t3 = t1.pow2k(6);
        let t2 = &t2 * &t3;
        let t2 = t2.pow2k(24);
        let t1 = &t1 * &t2;
        let t0 = &t0 * &t1;
        let t1 = t0.pow2k(10);
        let z = &z * &t1;
        let t1 = z.pow2k(58);
        let t1 = &t0 * &t1;
        let t0 = &x * &t1;
        let t2 = t0.square();
        let t1 = &t1 * &t2;
        let t2 = t1.pow2k(128);
        let t1 = &t1 * &t2;
        let t0 = &t0 * &t1;
        let t0 = t0.pow2k(68);
        &z * &t0
    }

    /// Computes the multiplicative inverse: `self^(p - 2) mod p`.
    ///
    /// Returns garbage if `self == 0` (no inverse exists).
    #[must_use]
    pub fn invert(&self) -> Self {
        let t = self.pow_p3div4();
        let t = t.pow2k(2);
        self * &t
    }

    /// Tests whether `self` is a quadratic residue in `F_p`.
    #[must_use]
    pub fn is_square(&self) -> Choice {
        let r = self.pow_p3div4();
        let r = r.square();
        let r = &r * self;
        r.ct_eq(&Self::ONE) | self.ct_eq(&Self::ZERO)
    }

    /// Computes the square root, when `self.is_square()` is true.
    ///
    /// The result is only meaningful when `self.is_square()` is set.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        let y = self.pow_p3div4();
        &y * self
    }
}

impl ConstantTimeEq for Fp64 {
    /// Equality on the canonical representative.
    fn ct_eq(&self, other: &Self) -> Choice {
        let a = self.final_sub_p();
        let b = other.final_sub_p();
        a.0.ct_eq(&b.0)
    }
}

impl PartialEq for Fp64 {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl Eq for Fp64 {}

impl fmt::Debug for Fp64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp64({:?})", &self.0[..])
    }
}

impl ConditionallySelectable for Fp64 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let mut out = [0u64; LIMBS];
        for (o, (x, y)) in out.iter_mut().zip(a.0.iter().zip(b.0.iter())) {
            *o = u64::conditional_select(x, y, choice);
        }
        Self(out)
    }
}

impl<'b> Add<&'b Fp64> for &Fp64 {
    type Output = Fp64;

    /// `a + b` reduced to `[0, 2p)`: the raw sum is below `4p < 2^384`, so
    /// one conditional subtract of `2p` suffices.
    #[inline]
    fn add(self, rhs: &'b Fp64) -> Fp64 {
        let mut s = [0u64; LIMBS];
        let mut carry = false;
        for (o, (x, y)) in s.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            let (sum, c) = x.carrying_add(*y, carry);
            *o = sum;
            carry = c;
        }
        Fp64(s).cond_sub_2p()
    }
}

impl<'b> Sub<&'b Fp64> for &Fp64 {
    type Output = Fp64;

    /// `a - b` reduced to `[0, 2p)`: on borrow, add `2p` back.
    #[inline]
    fn sub(self, rhs: &'b Fp64) -> Fp64 {
        let mut d = [0u64; LIMBS];
        let mut borrow = false;
        for (o, (x, y)) in d.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            let (diff, b) = x.borrowing_sub(*y, borrow);
            *o = diff;
            borrow = b;
        }
        Fp64(d).cond_add_2p(Choice::from(borrow as u8))
    }
}

impl Neg for &Fp64 {
    type Output = Fp64;

    #[inline]
    fn neg(self) -> Fp64 {
        &Fp64::ZERO - self
    }
}

impl Add<Fp64> for Fp64 {
    type Output = Fp64;

    fn add(self, rhs: Fp64) -> Fp64 {
        &self + &rhs
    }
}

impl Sub<Fp64> for Fp64 {
    type Output = Fp64;

    fn sub(self, rhs: Fp64) -> Fp64 {
        &self - &rhs
    }
}

impl Neg for Fp64 {
    type Output = Fp64;

    fn neg(self) -> Fp64 {
        -&self
    }
}

impl AddAssign<&Fp64> for Fp64 {
    fn add_assign(&mut self, rhs: &Fp64) {
        *self = &*self + rhs;
    }
}

impl SubAssign<&Fp64> for Fp64 {
    fn sub_assign(&mut self, rhs: &Fp64) {
        *self = &*self - rhs;
    }
}

impl AddAssign for Fp64 {
    fn add_assign(&mut self, rhs: Fp64) {
        *self += &rhs;
    }
}

impl SubAssign for Fp64 {
    fn sub_assign(&mut self, rhs: Fp64) {
        *self -= &rhs;
    }
}

impl<'b> Mul<&'b Fp64> for &Fp64 {
    type Output = Fp64;

    #[inline]
    fn mul(self, rhs: &'b Fp64) -> Fp64 {
        Fp64::mul_montgomery(self, rhs)
    }
}

impl Mul<Fp64> for Fp64 {
    type Output = Fp64;

    fn mul(self, rhs: Fp64) -> Fp64 {
        &self * &rhs
    }
}

impl MulAssign<&Fp64> for Fp64 {
    fn mul_assign(&mut self, rhs: &Fp64) {
        *self = &*self * rhs;
    }
}

impl MulAssign for Fp64 {
    fn mul_assign(&mut self, rhs: Fp64) {
        *self *= &rhs;
    }
}

// Compile-time checks: the const bridge must land the portable
// backend's named constants on this backend's.
const _: () = {
    let one = Fp64::from_limbs(crate::fields::fp::arch::generic::Fp55::ONE.0).cond_sub_p_const();
    let mut i = 0;
    while i < LIMBS {
        assert!(one.0[i] == Fp64::ONE.0[i]);
        i += 1;
    }
    let minus_one =
        Fp64::from_limbs(crate::fields::fp::arch::generic::Fp55::MINUS_ONE.0).cond_sub_p_const();
    let mut i = 0;
    while i < LIMBS {
        assert!(minus_one.0[i] == Fp64::MINUS_ONE.0[i]);
        i += 1;
    }
};
