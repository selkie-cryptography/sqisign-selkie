//! `Fp64` backend -- radix-2^64 Montgomery on `[u64; 4]`.
//!
//! Storage matches the SQIsign C reference's broadwell-lvl1 backend
//! (`src/gf/broadwell/lvl1/gf5248.c`): four full-width u64 limbs, no
//! spare-bit headroom.  Designed for MULX + dual-chain ADCX/ADOX asm
//! schoolbook; full-width limbs let each partial product's `(hi, lo)`
//! land at adjacent accumulator positions for row-major dual-chain
//! accumulation.
//!
//! Montgomery factor `R = 2^256`.  This differs from `Fp51`'s
//! `R = 2^255` -- the `from_limbs([u64; 5])` const-bridge below
//! handles both the radix repack and the factor-of-2 scaling in one
//! pass so the same `pub const FOO: Fp = Fp::from_limbs([...])`
//! precomputed-constant tables compile identically across backends.
//!
//! Constants verified against C ref's `gf5248_ONE`, `gf5248_MINUS_ONE`,
//! `R2`, `MODULUS` in `src/gf/broadwell/lvl1/gf5248.c:10-19`.
//!
//! # Contents
//!
//! Today: storage, named constants, `from_limbs([u64; 5])`
//! cross-backend const-bridge, `ConditionallySelectable`, modular
//! `Add` / `Sub` / `Neg` in pure Rust, and Montgomery `Mul` via the
//! `fp_mul` asm port of C ref's `FPMUL256x256` (4x4 MULX + dual ADX
//! schoolbook with interleaved CIOS reduction).  Subsequent commits
//! add the remaining asm leaves (`fp_sqr`, `fp2_mul_c0`,
//! `fp2_mul_c1`) and the higher-level ops layered over them
//! (`square`, `invert`, `sqrt`, `pow`, `sum_of_2_products`,
//! `difference_of_2_products`).  Montgomery-form byte conversions
//! (`From<[u8; 32]>` / `to_bytes`) land alongside the higher-level
//! ops since they internally use `Mul`-by-`R2` to enter the form.

#![allow(dead_code)] // dispatcher activation lands in a later commit.

use core::{
    arch::asm,
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use super::super::super::FP_ENCODED_BYTES;

/// The top limb of `p + 1 = 5 * 2^248` in `[u64; 4]` LE form (= the
/// only non-zero limb).  The CIOS-style Montgomery reduction in
/// [`Fp64::mul`]'s asm multiplies the low accumulator limb by this
/// constant to fold the reduction into the upper limbs without a
/// modular inverse.  Matches C ref's `p_plus_1` at
/// `src/gf/broadwell/lvl1/fp_asm.S:12`.
const P_PLUS_1_HI: u64 = 0x0500_0000_0000_0000;

#[cfg(test)]
mod tests;

/// An element of `F_p` (where `p = 5 * 2^248 - 1`) in radix-2^64
/// Montgomery form, packed into four full-width u64 limbs.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Fp64(pub(crate) [u64; 4]);

impl Fp64 {
    /// Constructs from raw radix-2^64 Montgomery limbs.
    ///
    /// Caller is responsible for the limbs already encoding a valid
    /// Montgomery-form field element.  For the cross-backend
    /// const-bridge from `Fp51`'s radix-51 limbs, use
    /// [`Fp64::from_limbs`] instead.
    pub const fn from_raw(limbs: [u64; 4]) -> Self {
        Self(limbs)
    }

    /// The additive identity in Montgomery form.
    pub const ZERO: Self = Self([0, 0, 0, 0]);

    /// The multiplicative identity in Montgomery form.
    ///
    /// `R mod p` where `R = 2^256`.  Equals `51 + 2^248` because
    /// `R = 256 * 2^248 = 51 + 2^248 (mod p)`, using `5 * 2^248 = 1
    /// (mod p)` so `1/5 = 2^248 (mod p)`.
    ///
    /// Matches C ref's `gf5248_ONE`.
    pub const ONE: Self = Self([
        0x0000000000000033,
        0x0000000000000000,
        0x0000000000000000,
        0x0100000000000000,
    ]);

    /// `2 * ONE` mod p, in Montgomery form.
    pub const TWO: Self = Self([
        0x0000000000000066,
        0x0000000000000000,
        0x0000000000000000,
        0x0200000000000000,
    ]);

    /// `4 * ONE` mod p, in Montgomery form.
    pub const FOUR: Self = Self([
        0x00000000000000CC,
        0x0000000000000000,
        0x0000000000000000,
        0x0400000000000000,
    ]);

    /// `-1 mod p` in Montgomery form.
    ///
    /// Equals `p - ONE = 2^250 - 52`.  Matches C ref's `gf5248_MINUS_ONE`.
    pub const MINUS_ONE: Self = Self([
        0xFFFFFFFFFFFFFFCC,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x03FFFFFFFFFFFFFF,
    ]);

    /// The modulus `p = 5 * 2^248 - 1`.  Not a canonical element, but
    /// load-bearing for `from_limbs`'s conditional subtract and (in
    /// later commits) the canonicalization tail of `Fp64::add`.
    pub(crate) const P: Self = Self([
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x04FFFFFFFFFFFFFF,
    ]);

    /// `R^2 mod p` for converting in/out of Montgomery form (`R = 2^256`).
    /// Matches C ref's `R2`.
    pub(crate) const R2: Self = Self([
        0x3333333333333D70,
        0x3333333333333333,
        0x3333333333333333,
        0x0333333333333333,
    ]);

    /// Const-bridge from `Fp51`'s radix-2^51 Montgomery limbs.
    ///
    /// Three-step:
    ///
    /// 1. Bit-repack `[u64; 5]` radix-2^51 -> `[u64; 4]` radix-2^64.
    /// 2. Canonicalize the repacked integer.  `Fp51`'s named constants (`ONE`,
    ///    `TWO`, `FOUR`, `MINUS_ONE`) store integer values in `[0, ~7p)` --
    ///    they're encoded for ease of Fp51's internal "less than 2p" Add
    ///    contract, not at minimal Mont form.  E.g. `Fp51::FOUR` integer is
    ///    `100 + 12 * 2^248 ~= 2.4p`. Fully reducing requires up to ~7
    ///    conditional subtracts of `p`; we do 16 as a safety margin (each call
    ///    past canonical is a no-op).
    /// 3. Multiply by 2 mod p (`Fp51`'s `R = 2^255`, `Fp64`'s `R = 2^256`,
    ///    ratio 2).  Implemented as a 1-bit left-shift plus a single
    ///    conditional subtract of `p` to canonicalize the doubled value (in
    ///    `[0, 2p)` since input is `[0, p)`).
    ///
    /// Signature-compatible with [`Fp51::from_limbs`][f51] and
    /// `arch::aarch64::neon::Fp29::from_limbs`, so the same
    /// `pub const FOO: Fp = Fp::from_limbs([...])` precomputed
    /// constants in `params.rs` and `deuring/precomputed.rs` compile
    /// across every backend.
    ///
    /// [f51]: super::super::generic::Fp51::from_limbs
    pub const fn from_limbs(portable_mont: [u64; 5]) -> Self {
        let l0 = portable_mont[0];
        let l1 = portable_mont[1];
        let l2 = portable_mont[2];
        let l3 = portable_mont[3];
        let l4 = portable_mont[4];

        // Step 1: repack radix-2^51 -> radix-2^64.  Each output limb's
        // bits come from at most two input limbs.  (Input bit
        // positions: l0=0..50, l1=51..101, l2=102..152, l3=153..203,
        // l4=204..254.)
        let v0 = l0 | (l1 << 51);
        let v1 = (l1 >> 13) | (l2 << 38);
        let v2 = (l2 >> 26) | (l3 << 25);
        let v3 = (l3 >> 39) | (l4 << 12);

        // Step 2: fully canonicalize the repacked integer.  Worst-case
        // Fp51 named constants encode integers around `~6.4p`; loop
        // 16 times to safely reduce to `[0, p)`.
        let mut acc = Self([v0, v1, v2, v3]);
        let mut i = 0;
        while i < 16 {
            acc = acc.cond_sub_p_const(false);
            i += 1;
        }

        // Step 3: multiply by 2 (1-bit left shift across the 4
        // limbs).  Input is now in `[0, p)`, `p < 2^252`, so the
        // doubled value is in `[0, 2p) < 2^253` and never produces a
        // bit-256 overflow.  Single conditional subtract suffices.
        let v0 = acc.0[0];
        let v1 = acc.0[1];
        let v2 = acc.0[2];
        let v3 = acc.0[3];
        let s0 = v0 << 1;
        let s1 = (v1 << 1) | (v0 >> 63);
        let s2 = (v2 << 1) | (v1 >> 63);
        let s3 = (v3 << 1) | (v2 >> 63);

        Self([s0, s1, s2, s3]).cond_sub_p_const(false)
    }

    /// Conditional subtract of `p` in a `const` context.
    ///
    /// Returns `self - p` if `self >= p` or `force` is true; else
    /// returns `self` unchanged.  Used by [`Fp64::from_limbs`] to
    /// canonicalize at compile time.  The runtime equivalent is
    /// [`Fp64::final_sub_p`], which uses
    /// `ConditionallySelectable` for branch-freedom (not available
    /// in `const fn` yet).
    const fn cond_sub_p_const(self, force: bool) -> Self {
        let p = Self::P.0;

        let (d0, b0) = self.0[0].overflowing_sub(p[0]);

        let (d1_a, b1_a) = self.0[1].overflowing_sub(p[1]);
        let (d1, b1_b) = d1_a.overflowing_sub(b0 as u64);
        let b1 = b1_a | b1_b;

        let (d2_a, b2_a) = self.0[2].overflowing_sub(p[2]);
        let (d2, b2_b) = d2_a.overflowing_sub(b1 as u64);
        let b2 = b2_a | b2_b;

        let (d3_a, b3_a) = self.0[3].overflowing_sub(p[3]);
        let (d3, b3_b) = d3_a.overflowing_sub(b2 as u64);
        let b3 = b3_a | b3_b;

        let take = force || !b3;

        Self([
            if take { d0 } else { self.0[0] },
            if take { d1 } else { self.0[1] },
            if take { d2 } else { self.0[2] },
            if take { d3 } else { self.0[3] },
        ])
    }

    /// Reduce `self` from `[0, 2p)` to `[0, p)` by conditionally
    /// subtracting `p`.
    ///
    /// Speculatively subtracts `p` and uses the final borrow to
    /// select: borrow set means `self < p` (keep `self` unchanged),
    /// borrow clear means `self >= p` (use the subtracted form).
    /// Branch-free via [`ConditionallySelectable`].
    fn final_sub_p(self) -> Self {
        let p = Self::P.0;
        let (d0, b0) = self.0[0].borrowing_sub(p[0], false);
        let (d1, b1) = self.0[1].borrowing_sub(p[1], b0);
        let (d2, b2) = self.0[2].borrowing_sub(p[2], b1);
        let (d3, b3) = self.0[3].borrowing_sub(p[3], b2);
        let subbed = Self([d0, d1, d2, d3]);
        // `b3 = true` iff `self < p`; in that case keep `self`.
        Self::conditional_select(&subbed, &self, Choice::from(b3 as u8))
    }

    /// Conditionally add `p` to `self`.
    ///
    /// Returns `self + p (mod 2^256)` if `cond` is true, else `self`
    /// unchanged.  Used by `Sub` to add `p` back when the raw
    /// subtraction underflowed.
    fn cond_add_p(self, cond: Choice) -> Self {
        let p = Self::P.0;
        let (a0, c0) = self.0[0].carrying_add(p[0], false);
        let (a1, c1) = self.0[1].carrying_add(p[1], c0);
        let (a2, c2) = self.0[2].carrying_add(p[2], c1);
        let (a3, _c3) = self.0[3].carrying_add(p[3], c2);
        let added = Self([a0, a1, a2, a3]);
        Self::conditional_select(&self, &added, cond)
    }

    /// Montgomery multiplication: `self * rhs * R^{-1} mod p` with
    /// `R = 2^256`.
    ///
    /// MULX schoolbook + dual-chain ADCX/ADOX accumulation,
    /// interleaved with CIOS-style Montgomery reduction folded
    /// through the special-form constant `p + 1 = 5 * 2^248`.  Ports
    /// C ref's `fp_mul` + `FPMUL256x256` macro at
    /// `src/gf/broadwell/lvl1/fp_asm.S:434` / `:288`.
    ///
    /// Inputs and output are in `[0, p)`.  The 16 partial products
    /// span 5 limbs of accumulator; after each row of MULADD64x256
    /// the low limb is folded into the upper limbs via MULADD64x64
    /// using `P_PLUS_1_HI`, leaving the rotated accumulator one limb
    /// shorter for the next row.
    #[inline]
    fn mul_montgomery(a: &Self, b: &Self) -> Self {
        let mut out = [0u64; 4];
        // SAFETY: cfg-gated at the parent module on bmi2 + adx; asm
        // uses MULX/ADCX/ADOX unconditionally.  Reads 32 bytes from
        // each of `a` and `b`, writes 32 bytes to `out`.  `nostack`
        // since the prologue saves no GPRs (rustc handles
        // callee-saved register preservation via the `out(reg)`
        // declarations).
        unsafe {
            asm!(
                // Prologue: (r8..r12) = a[0] * b (5 limbs, ADOX chain).
                "mov rdx, qword ptr [{a} + 0]",
                "mulx {z1}, {z0}, qword ptr [{b} + 0]",
                "xor eax, eax",
                "mulx {z2}, {t1}, qword ptr [{b} + 8]",
                "adox {z1}, {t1}",
                "mulx {z3}, {t1}, qword ptr [{b} + 16]",
                "adox {z2}, {t1}",
                "mulx {z4}, {t1}, qword ptr [{b} + 24]",
                "adox {z3}, {t1}",
                "adox {z4}, rax",

                // Iter 0: reduce z0 (mul by p+1 top); then accumulate
                // a[1] * b. After: accumulator slots rotate -- the
                // "new z4" is what was z0.

                // MULADD64x64(reduce z0): mulx with p+1's top limb.
                // The product (z0 * P_PLUS_1_HI) is a u128 representing
                // an integer value at limb position 3 (since P_PLUS_1_HI
                // is at limb 3 of p+1).  So T1 (= LO) lands at limb 3 =
                // z3, T0 (= HI) at limb 4 = z4.
                "mov rdx, {z0}",
                "mulx {t0}, {t1}, {p1hi}",
                "xor eax, eax",
                "adox {z3}, {t1}",
                "adox {z4}, {t0}",

                // MULADD64x256(a[1] * b, accumulate into z1:z4:z0):
                // first mulx primes z1, z2 via ADOX; subsequent
                // mulx-pairs interleave ADCX (lo chain into z2..z4)
                // and ADOX (hi chain into z2..z0).  C is z0 (= the
                // freed slot).
                "mov rdx, qword ptr [{a} + 8]",
                "mulx {t0}, {t1}, qword ptr [{b} + 0]",
                "xor {z0}, {z0}",
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
                "adox {z0}, {t0}",
                "adc {z0}, 0",

                // Iter 1: reduce z1; accumulate a[2] * b into z2..z0:z1.
                // After iter 0's rotation, conceptual position-3 = z4
                // and position-4 = z0 (the freed slot).
                "mov rdx, {z1}",
                "mulx {t0}, {t1}, {p1hi}",
                "xor eax, eax",
                "adox {z4}, {t1}",
                "adox {z0}, {t0}",

                "mov rdx, qword ptr [{a} + 16]",
                "mulx {t0}, {t1}, qword ptr [{b} + 0]",
                "xor {z1}, {z1}",
                "adox {z2}, {t1}",
                "adox {z3}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 8]",
                "adcx {z3}, {t1}",
                "adox {z4}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 16]",
                "adcx {z4}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 24]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "adc {z1}, 0",

                // Iter 2: reduce z2; accumulate a[3] * b into z3..z1:z2.
                // After iter 1's rotation, conceptual position-3 = z0
                // and position-4 = z1.
                "mov rdx, {z2}",
                "mulx {t0}, {t1}, {p1hi}",
                "xor eax, eax",
                "adox {z0}, {t1}",
                "adox {z1}, {t0}",

                "mov rdx, qword ptr [{a} + 24]",
                "mulx {t0}, {t1}, qword ptr [{b} + 0]",
                "xor {z2}, {z2}",
                "adox {z3}, {t1}",
                "adox {z4}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 8]",
                "adcx {z4}, {t1}",
                "adox {z0}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 16]",
                "adcx {z0}, {t1}",
                "adox {z1}, {t0}",
                "mulx {t0}, {t1}, qword ptr [{b} + 24]",
                "adcx {z1}, {t1}",
                "adox {z2}, {t0}",
                "adc {z2}, 0",

                // Iter 3: final reduction; no further row addition.
                // After iter 2's rotation, conceptual position-3 = z1
                // and position-4 = z2.
                "mov rdx, {z3}",
                "mulx {t0}, {t1}, {p1hi}",
                "xor eax, eax",
                "adox {z1}, {t1}",
                "adox {z2}, {t0}",

                // Result lands in (z4, z0, z1, z2) -- four limbs after
                // four rotations.  Store to out[0..3].
                "mov qword ptr [{out} + 0], {z4}",
                "mov qword ptr [{out} + 8], {z0}",
                "mov qword ptr [{out} + 16], {z1}",
                "mov qword ptr [{out} + 24], {z2}",

                a = in(reg) a.0.as_ptr(),
                b = in(reg) b.0.as_ptr(),
                out = in(reg) out.as_mut_ptr(),
                p1hi = in(reg) P_PLUS_1_HI,
                z0 = out(reg) _,
                z1 = out(reg) _,
                z2 = out(reg) _,
                z3 = out(reg) _,
                z4 = out(reg) _,
                t0 = out(reg) _,
                t1 = out(reg) _,
                out("rax") _,
                out("rdx") _,
                options(nostack),
            );
        }
        // Output may be up to ~2p; canonicalize via single conditional
        // subtract.  (C ref omits this; the C-side `fp_normalize`
        // handles it at the trait boundary.  We canonicalize at every
        // op so Fp64 values are always in [0, p).)
        Self(out).final_sub_p()
    }

    /// Full 8-limb product `a^2` via dual-carry ADCX/ADOX, exploiting
    /// squaring symmetry: the 6 cross products `a[i]*a[j]` (i<j) are
    /// accumulated once with the CF (ADCX) and OF (ADOX) chains running
    /// in parallel, the buffer is doubled, then the 4 diagonal squares
    /// `a[i]^2` are folded in -- 10 MULX total.  Output is the raw
    /// 512-bit square (unreduced); the caller applies Montgomery REDC.
    ///
    /// Each cross-product row's two carry-outs are folded into the next
    /// limb (`adcx top, 0` / `adox next, 0` / `adc next, 0`) before the
    /// following row's `xor` resets the flags, so neither chain drops a
    /// carry.  The doubling and diagonal phases use a single CF chain
    /// (MULX and MOV preserve CF).
    ///
    /// # Safety
    ///
    /// cfg-gated on `+adx` and `+bmi2`; MULX/ADCX/ADOX are used
    /// unconditionally.  Reads 32 bytes from `a`, writes 64 bytes to the
    /// returned buffer; `nostack`.
    #[inline]
    fn sqr_wide_adx(a: &[u64; 4]) -> [u64; 8] {
        let mut t = [0u64; 8];
        // SAFETY: cfg-gated +adx/+bmi2; the asm has no operand-dependent
        // memory access or control flow.  Reads a[0..4] via its pointer,
        // writes t[0..8] via its pointer.
        unsafe {
            asm!(
                // Zero the eight accumulators.
                "xor {z0:e}, {z0:e}",
                "xor {z1:e}, {z1:e}",
                "xor {z2:e}, {z2:e}",
                "xor {z3:e}, {z3:e}",
                "xor {z4:e}, {z4:e}",
                "xor {z5:e}, {z5:e}",
                "xor {z6:e}, {z6:e}",
                "xor {z7:e}, {z7:e}",

                // Row 0: a[0] * (a[1], a[2], a[3]) -> z1..z4, fold to z5.
                "mov rdx, qword ptr [{a} + 0]",
                "xor eax, eax",
                "mulx {hi}, {lo}, qword ptr [{a} + 8]",
                "adox {z1}, {lo}",
                "adox {z2}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{a} + 16]",
                "adcx {z2}, {lo}",
                "adox {z3}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{a} + 24]",
                "adcx {z3}, {lo}",
                "adox {z4}, {hi}",
                "adcx {z4}, rax",
                "adox {z5}, rax",
                "adc {z5}, 0",

                // Row 1: a[1] * (a[2], a[3]) -> z3..z5, fold to z6.
                "mov rdx, qword ptr [{a} + 8]",
                "xor eax, eax",
                "mulx {hi}, {lo}, qword ptr [{a} + 16]",
                "adox {z3}, {lo}",
                "adox {z4}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{a} + 24]",
                "adcx {z4}, {lo}",
                "adox {z5}, {hi}",
                "adcx {z5}, rax",
                "adox {z6}, rax",
                "adc {z6}, 0",

                // Row 2: a[2] * a[3] -> z5, z6, fold to z7.
                "mov rdx, qword ptr [{a} + 16]",
                "xor eax, eax",
                "mulx {hi}, {lo}, qword ptr [{a} + 24]",
                "adox {z5}, {lo}",
                "adox {z6}, {hi}",
                "adcx {z6}, rax",
                "adox {z7}, rax",
                "adc {z7}, 0",

                // Double z1..z7 (cross terms count twice); z0 stays 0.
                "add {z1}, {z1}",
                "adc {z2}, {z2}",
                "adc {z3}, {z3}",
                "adc {z4}, {z4}",
                "adc {z5}, {z5}",
                "adc {z6}, {z6}",
                "adc {z7}, {z7}",

                // Diagonals a[i]^2 at (2i, 2i+1); single CF chain.
                "mov rdx, qword ptr [{a} + 0]",
                "mulx {hi}, {lo}, rdx",
                "add {z0}, {lo}",
                "adc {z1}, {hi}",
                "mov rdx, qword ptr [{a} + 8]",
                "mulx {hi}, {lo}, rdx",
                "adc {z2}, {lo}",
                "adc {z3}, {hi}",
                "mov rdx, qword ptr [{a} + 16]",
                "mulx {hi}, {lo}, rdx",
                "adc {z4}, {lo}",
                "adc {z5}, {hi}",
                "mov rdx, qword ptr [{a} + 24]",
                "mulx {hi}, {lo}, rdx",
                "adc {z6}, {lo}",
                "adc {z7}, {hi}",

                // Store the 8-limb product.
                "mov qword ptr [{t} + 0], {z0}",
                "mov qword ptr [{t} + 8], {z1}",
                "mov qword ptr [{t} + 16], {z2}",
                "mov qword ptr [{t} + 24], {z3}",
                "mov qword ptr [{t} + 32], {z4}",
                "mov qword ptr [{t} + 40], {z5}",
                "mov qword ptr [{t} + 48], {z6}",
                "mov qword ptr [{t} + 56], {z7}",

                a = in(reg) a.as_ptr(),
                t = in(reg) t.as_mut_ptr(),
                z0 = out(reg) _,
                z1 = out(reg) _,
                z2 = out(reg) _,
                z3 = out(reg) _,
                z4 = out(reg) _,
                z5 = out(reg) _,
                z6 = out(reg) _,
                z7 = out(reg) _,
                lo = out(reg) _,
                hi = out(reg) _,
                out("rax") _,
                out("rdx") _,
                options(nostack),
            );
        }
        t
    }

    /// Modular squaring: `self * self mod p`.
    ///
    /// Symmetric squaring: forms the full 2N-limb product `a^2` using
    /// cross-term symmetry (each `a[i]*a[j]`, `i<j`, computed once then
    /// doubled, plus the diagonal `a[i]^2`), then a `p+1`-trick
    /// Montgomery REDC.  Totals ~14 `u128` limb-mults (10 product + 4
    /// reduction) vs the 20 of `mul_montgomery` (16 + 4), which is why
    /// `fp_square` beats `fp_mul`.  C ref's `fp_sqr` is `jmp fp_mul`
    /// (no symmetric shortcut); this is a Selkie divergence motivated
    /// by the bench gap.
    ///
    /// The product is the dual-carry [`Fp64::sqr_wide_adx`] asm; the
    /// `p+1` REDC is `u128` Rust (only 4 limb-mults).  Validated against
    /// the Montgomery definition (cycle-accurate flag sim, 500k random +
    /// edges) and gated on CI by `square_matches_mul` /
    /// `square_matches_fp51`.
    #[inline]
    #[must_use]
    pub fn square(&self) -> Self {
        // Phases 1-3: the full 8-limb product a^2 via dual-carry
        // ADCX/ADOX asm (symmetric cross-terms, doubling, diagonals).
        let mut t = Self::sqr_wide_adx(&self.0);

        // Phase 4: Montgomery REDC via the p+1 identity.  Since
        // `p[0] == -1`, the multiplier is `n_inv == 1`, so `m = t[i]`
        // and `t += m*p` cancels limb i exactly (`t[i] - m == 0`, no
        // borrow).  And `p + 1 == P_PLUS_1_HI * 2^192`, so the only
        // nonzero add is `m * P_PLUS_1_HI` at limbs i+3, i+4 -- 4
        // limb-mults total vs 16 for a full-modulus REDC.  Limbs 0..3
        // are consumed (zeroed) and dropped; the result is t[4..8] in
        // [0, 2p).
        let mut i = 0;
        while i < 4 {
            let m = t[i];
            let prod = m as u128 * P_PLUS_1_HI as u128;
            let plo = prod as u64;
            let phi = (prod >> 64) as u64;

            let (s, c0) = t[i + 3].overflowing_add(plo);
            t[i + 3] = s;
            let (s, c1) = t[i + 4].overflowing_add(phi);
            let (s, c2) = s.overflowing_add(c0 as u64);
            t[i + 4] = s;

            let mut carry = (c1 as u64) | (c2 as u64);
            let mut idx = i + 5;
            while idx < 8 && carry != 0 {
                let (s, c) = t[idx].overflowing_add(carry);
                t[idx] = s;
                carry = c as u64;
                idx += 1;
            }
            i += 1;
        }

        Self([t[4], t[5], t[6], t[7]]).final_sub_p()
    }

    /// Returns `a1 * b1 + a2 * b2 (mod p)`.
    ///
    /// Fused-shape op used by `Fp2::mul`'s Algorithm 8.1 path: each
    /// `Fp2` coefficient is one sum-of-2-products.  Composed for now
    /// from two `Mul`s and an `Add`; C ref ships a fused asm
    /// (`fp2_mul_c1` at `src/gf/broadwell/lvl1/fp_asm.S:220`) that
    /// shares the Montgomery reduction across both partial products.
    /// Drop-in asm replacement lands in a later commit once the
    /// scalar surface stabilizes.
    #[inline]
    #[must_use]
    pub fn sum_of_2_products(a1: &Self, b1: &Self, a2: &Self, b2: &Self) -> Self {
        &(a1 * b1) + &(a2 * b2)
    }

    /// Returns `a1 * b1 - a2 * b2 (mod p)`.
    ///
    /// Companion to [`Fp64::sum_of_2_products`]; computes the other
    /// `Fp2::mul` coefficient (Algorithm 8.1).  C ref's
    /// `fp2_mul_c0` at `src/gf/broadwell/lvl1/fp_asm.S:134` provides
    /// the fused-asm equivalent.
    #[inline]
    #[must_use]
    pub fn difference_of_2_products(a1: &Self, b1: &Self, a2: &Self, b2: &Self) -> Self {
        &(a1 * b1) - &(a2 * b2)
    }

    /// Constructs a field element from a small integer.
    ///
    /// Inserts `x` at limb 0 of the canonical form, then enters
    /// Montgomery form by multiplying by `R^2`.
    #[must_use]
    pub fn from_small(x: u32) -> Self {
        let canonical = Self([x as u64, 0, 0, 0]);
        &canonical * &Self::R2
    }

    /// Decodes 32 bytes (little-endian) into a Montgomery-form `Fp64`.
    ///
    /// The input must be a canonical encoding (value `< p`).  Parses
    /// the bytes into `[u64; 4]` canonical form, then multiplies by
    /// `R^2` to enter Montgomery scaling.
    #[must_use]
    pub fn from_bytes(bytes: &[u8; FP_ENCODED_BYTES]) -> Self {
        let canonical = Self([
            u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]),
            u64::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15],
            ]),
            u64::from_le_bytes([
                bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22],
                bytes[23],
            ]),
            u64::from_le_bytes([
                bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30],
                bytes[31],
            ]),
        ]);
        &canonical * &Self::R2
    }

    /// Encodes a Montgomery-form `Fp64` as 32 bytes, little-endian.
    ///
    /// Exits Montgomery form via `mul` by `1` (= raw `[1, 0, 0, 0]`),
    /// which performs the Montgomery reduction without re-entering
    /// the scaled form, then encodes the canonical `[u64; 4]`.
    #[must_use]
    pub fn to_bytes(self) -> [u8; FP_ENCODED_BYTES] {
        let one_raw = Self([1, 0, 0, 0]);
        let canonical = &self * &one_raw;
        let mut out = [0u8; FP_ENCODED_BYTES];
        out[0..8].copy_from_slice(&canonical.0[0].to_le_bytes());
        out[8..16].copy_from_slice(&canonical.0[1].to_le_bytes());
        out[16..24].copy_from_slice(&canonical.0[2].to_le_bytes());
        out[24..32].copy_from_slice(&canonical.0[3].to_le_bytes());
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
    /// Used to derive [`Fp64::invert`], [`Fp64::sqrt`], and
    /// [`Fp64::is_square`].  Addition chain ported verbatim from
    /// [`Fp51::pow_p3div4`] (same prime, same chain).
    ///
    /// [`Fp51::pow_p3div4`]: super::super::generic::Fp51::pow_p3div4
    #[must_use]
    pub(crate) fn pow_p3div4(&self) -> Self {
        let x = *self;
        let z = x.square();
        let t0 = &x * &z;
        let z = t0.square();
        let z = &x * &z;
        let t1 = z.square();
        let t3 = t1.square();
        let t2 = t3.square();
        let t4 = t2.pow2k(3);
        let t2 = &t2 * &t4;
        let t4 = t2.pow2k(6);
        let t2 = &t2 * &t4;
        let t4 = t2.pow2k(2);
        let t3 = &t3 * &t4;
        let t3 = t3.pow2k(13);
        let t2 = &t2 * &t3;
        let t3 = t2.pow2k(27);
        let t2 = &t2 * &t3;
        let z = &z * &t2;
        let t2 = z.pow2k(4);
        let t1 = &t1 * &t2;
        let t0 = &t0 * &t1;
        let t1 = &t1 * &t0;
        let t0 = &t1 * &t0;
        let t2 = &t0 * &t1;
        let t0 = &t0 * &t2;
        let t1 = &t1 * &t0;
        let t1 = t1.pow2k(63);
        let t1 = &t0 * &t1;
        let t1 = t1.pow2k(64);
        let t0 = &t0 * &t1;
        let t0 = t0.pow2k(57);
        &z * &t0
    }

    /// Computes the multiplicative inverse: `self^(p - 2) mod p`.
    ///
    /// Returns garbage if `self == 0` (no inverse exists).  Uses
    /// Fermat's little theorem via [`Fp64::pow_p3div4`].
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
    /// The result is only meaningful when `self.is_square()` is set;
    /// callers that don't know upfront should check before using the
    /// output.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        let y = self.pow_p3div4();
        &y * self
    }
}

impl ConstantTimeEq for Fp64 {
    /// Constant-time equality on canonical `Fp64` values.
    ///
    /// Limb-wise via `subtle`'s `ConstantTimeEq` on `u64`.  Requires
    /// both operands to be in canonical form (which they always are
    /// after any `Fp64` op).
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0[0].ct_eq(&other.0[0])
            & self.0[1].ct_eq(&other.0[1])
            & self.0[2].ct_eq(&other.0[2])
            & self.0[3].ct_eq(&other.0[3])
    }
}

impl fmt::Debug for Fp64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp64({:?})", &self.0[..])
    }
}

impl ConditionallySelectable for Fp64 {
    /// Constant-time select between `a` and `b` per `choice`.
    /// Limb-wise via `subtle::u64::conditional_select`.
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self([
            u64::conditional_select(&a.0[0], &b.0[0], choice),
            u64::conditional_select(&a.0[1], &b.0[1], choice),
            u64::conditional_select(&a.0[2], &b.0[2], choice),
            u64::conditional_select(&a.0[3], &b.0[3], choice),
        ])
    }
}

impl<'b> Add<&'b Fp64> for &Fp64 {
    type Output = Fp64;

    /// Modular addition.  Inputs in `[0, p)`; output in `[0, p)`.
    ///
    /// Sum of two canonical values fits in 4 limbs without bit-256
    /// overflow (`2p < 2^253`); a single conditional subtract of `p`
    /// canonicalizes.
    fn add(self, rhs: &'b Fp64) -> Fp64 {
        let (r0, c0) = self.0[0].carrying_add(rhs.0[0], false);
        let (r1, c1) = self.0[1].carrying_add(rhs.0[1], c0);
        let (r2, c2) = self.0[2].carrying_add(rhs.0[2], c1);
        let (r3, _c3) = self.0[3].carrying_add(rhs.0[3], c2);
        Fp64([r0, r1, r2, r3]).final_sub_p()
    }
}

impl<'b> Sub<&'b Fp64> for &Fp64 {
    type Output = Fp64;

    /// Modular subtraction.  Inputs in `[0, p)`; output in `[0, p)`.
    ///
    /// Subtract limb-wise; a final borrow means the raw difference
    /// underflowed (`self < rhs`), in which case [`Fp64::cond_add_p`]
    /// adds `p` back to canonicalize.
    fn sub(self, rhs: &'b Fp64) -> Fp64 {
        let (r0, b0) = self.0[0].borrowing_sub(rhs.0[0], false);
        let (r1, b1) = self.0[1].borrowing_sub(rhs.0[1], b0);
        let (r2, b2) = self.0[2].borrowing_sub(rhs.0[2], b1);
        let (r3, b3) = self.0[3].borrowing_sub(rhs.0[3], b2);
        Fp64([r0, r1, r2, r3]).cond_add_p(Choice::from(b3 as u8))
    }
}

impl Neg for &Fp64 {
    type Output = Fp64;

    /// Modular negation: `p - self mod p`.
    ///
    /// Computes `p - self`; for `self == 0` the raw result is `p` and
    /// [`Fp64::final_sub_p`] reduces it to `0`.
    fn neg(self) -> Fp64 {
        let p = Fp64::P.0;
        let (d0, b0) = p[0].borrowing_sub(self.0[0], false);
        let (d1, b1) = p[1].borrowing_sub(self.0[1], b0);
        let (d2, b2) = p[2].borrowing_sub(self.0[2], b1);
        let (d3, _b3) = p[3].borrowing_sub(self.0[3], b2);
        Fp64([d0, d1, d2, d3]).final_sub_p()
    }
}

// Convenience: owned variants delegate to reference impls.

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

    /// Modular Montgomery multiplication.  Dispatches to
    /// [`Fp64::mul_montgomery`].
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
