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
///
/// **Lazy reduction**: values are kept in `[0, 2p)`, not canonical
/// `[0, p)` (matching the C reference, whose `fp_mul` skips the final
/// conditional subtract -- ~27% per mul).  `mul`/`square` accept and
/// return `[0, 2p)`; `add`/`sub`/`neg` reduce to `[0, 2p)`.  A field
/// element therefore has two representatives (`x` and `x + p`), so
/// `PartialEq`, `ct_eq`, and `to_bytes` normalize to `[0, p)` first --
/// `PartialEq`/`Eq` are hand-written, not derived (a structural compare
/// would call `x` and `x + p` unequal).
#[derive(Copy, Clone)]
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

    /// `2p`, the reduction constant for the lazy-`[0, 2p)` invariant.
    /// `add`/`sub`/`neg` produce values in `[0, 4p)` and subtract this
    /// once to land back in `[0, 2p)`.
    const TWO_P: Self = Self([
        0xFFFFFFFFFFFFFFFE,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x09FFFFFFFFFFFFFF,
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

    /// Reduces a value in `[0, 4p)` to `[0, 2p)` by subtracting `2p`
    /// when `self >= 2p` (constant-time via a borrow-masked select).
    /// The lazy-invariant analog of [`Fp64::final_sub_p`].
    fn cond_sub_2p(self) -> Self {
        let tp = Self::TWO_P.0;
        let (d0, b0) = self.0[0].borrowing_sub(tp[0], false);
        let (d1, b1) = self.0[1].borrowing_sub(tp[1], b0);
        let (d2, b2) = self.0[2].borrowing_sub(tp[2], b1);
        let (d3, b3) = self.0[3].borrowing_sub(tp[3], b2);
        let subbed = Self([d0, d1, d2, d3]);
        // `b3 = true` iff `self < 2p`; in that case keep `self`.
        Self::conditional_select(&subbed, &self, Choice::from(b3 as u8))
    }

    /// Conditionally adds `2p` to `self` (mod 2^256) when `cond`.  Used
    /// by `Sub`: an underflowing `a - b` (with `a, b` in `[0, 2p)`)
    /// wraps; adding `2p` back lands the result in `[0, 2p)`.
    fn cond_add_2p(self, cond: Choice) -> Self {
        let tp = Self::TWO_P.0;
        let (a0, c0) = self.0[0].carrying_add(tp[0], false);
        let (a1, c1) = self.0[1].carrying_add(tp[1], c0);
        let (a2, c2) = self.0[2].carrying_add(tp[2], c1);
        let (a3, _c3) = self.0[3].carrying_add(tp[3], c2);
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
        // Lazy reduction: leave the result in [0, 2p) (Montgomery's
        // bound for inputs < 2p, since 4p < R).  No final conditional
        // subtract -- matching C ref's `fp_mul`.  Normalization to
        // [0, p) happens only at boundaries (`to_bytes`, equality).
        Self(out)
    }

    /// Modular squaring: `self * self mod p`.
    ///
    /// Squares via [`Fp64::mul_montgomery`], matching C ref's `fp_sqr`
    /// (`jmp fp_mul`).  At 4 limbs a dedicated symmetric square is a net
    /// loss: it saves a few MULX over the schoolbook product, but the
    /// doubling pass plus the extra ADCX/ADOX/ADC carry-folding add more
    /// than that back, so it runs more instructions than the dual-carry
    /// mul.  A symmetric / single-block-asm square was implemented and
    /// reverted on that evidence (gungraun per-op instruction counts).
    #[inline]
    #[must_use]
    pub fn square(&self) -> Self {
        Self::mul_montgomery(self, self)
    }

    /// Full 4x4 -> 8-limb product `a * b` (no Montgomery reduction), via
    /// dual-carry ADCX/ADOX.  Each row `a[i]*b` accumulates on the CF
    /// (ADCX) and OF (ADOX) chains; the two carry-outs fold into the next
    /// limb before the next row resets the flags.
    ///
    /// Building block for measuring the fused-`fp2` reduction-sharing win
    /// (`fp64_mul` cycles minus this = one Montgomery reduction).  Not yet
    /// on a production path.
    ///
    /// # Safety
    ///
    /// cfg-gated on `+adx` + `+bmi2`; reads 32 bytes from each of `a`, `b`.
    #[inline]
    pub(crate) fn mul_wide_adx(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
        let mut t = [0u64; 8];
        // SAFETY: cfg-gated +adx/+bmi2; no operand-dependent memory or
        // control flow.  Reads a[0..4], b[0..4]; writes t[0..8].
        unsafe {
            asm!(
                "xor {z0:e}, {z0:e}",
                "xor {z1:e}, {z1:e}",
                "xor {z2:e}, {z2:e}",
                "xor {z3:e}, {z3:e}",
                "xor {z4:e}, {z4:e}",
                "xor {z5:e}, {z5:e}",
                "xor {z6:e}, {z6:e}",
                "xor {z7:e}, {z7:e}",

                // Row 0: a[0] * b[0..3] -> z0..z4, fold to z5.
                "mov rdx, qword ptr [{a} + 0]",
                "xor eax, eax",
                "mulx {hi}, {lo}, qword ptr [{b} + 0]",
                "adox {z0}, {lo}",
                "adox {z1}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{b} + 8]",
                "adcx {z1}, {lo}",
                "adox {z2}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{b} + 16]",
                "adcx {z2}, {lo}",
                "adox {z3}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{b} + 24]",
                "adcx {z3}, {lo}",
                "adox {z4}, {hi}",
                "adcx {z4}, rax",
                "adox {z5}, rax",
                "adc {z5}, 0",

                // Row 1: a[1] * b[0..3] -> z1..z5, fold to z6.
                "mov rdx, qword ptr [{a} + 8]",
                "xor eax, eax",
                "mulx {hi}, {lo}, qword ptr [{b} + 0]",
                "adox {z1}, {lo}",
                "adox {z2}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{b} + 8]",
                "adcx {z2}, {lo}",
                "adox {z3}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{b} + 16]",
                "adcx {z3}, {lo}",
                "adox {z4}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{b} + 24]",
                "adcx {z4}, {lo}",
                "adox {z5}, {hi}",
                "adcx {z5}, rax",
                "adox {z6}, rax",
                "adc {z6}, 0",

                // Row 2: a[2] * b[0..3] -> z2..z6, fold to z7.
                "mov rdx, qword ptr [{a} + 16]",
                "xor eax, eax",
                "mulx {hi}, {lo}, qword ptr [{b} + 0]",
                "adox {z2}, {lo}",
                "adox {z3}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{b} + 8]",
                "adcx {z3}, {lo}",
                "adox {z4}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{b} + 16]",
                "adcx {z4}, {lo}",
                "adox {z5}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{b} + 24]",
                "adcx {z5}, {lo}",
                "adox {z6}, {hi}",
                "adcx {z6}, rax",
                "adox {z7}, rax",
                "adc {z7}, 0",

                // Row 3: a[3] * b[0..3] -> z3..z7 (product fits 8 limbs).
                "mov rdx, qword ptr [{a} + 24]",
                "xor eax, eax",
                "mulx {hi}, {lo}, qword ptr [{b} + 0]",
                "adox {z3}, {lo}",
                "adox {z4}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{b} + 8]",
                "adcx {z4}, {lo}",
                "adox {z5}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{b} + 16]",
                "adcx {z5}, {lo}",
                "adox {z6}, {hi}",
                "mulx {hi}, {lo}, qword ptr [{b} + 24]",
                "adcx {z6}, {lo}",
                "adox {z7}, {hi}",
                "adcx {z7}, rax",

                "mov qword ptr [{t} + 0], {z0}",
                "mov qword ptr [{t} + 8], {z1}",
                "mov qword ptr [{t} + 16], {z2}",
                "mov qword ptr [{t} + 24], {z3}",
                "mov qword ptr [{t} + 32], {z4}",
                "mov qword ptr [{t} + 40], {z5}",
                "mov qword ptr [{t} + 48], {z6}",
                "mov qword ptr [{t} + 56], {z7}",

                a = in(reg) a.as_ptr(),
                b = in(reg) b.as_ptr(),
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
    /// then `final_sub_p` normalizes the lazy `[0, 2p)` result to the
    /// canonical `[0, p)` before encoding.
    #[must_use]
    pub fn to_bytes(self) -> [u8; FP_ENCODED_BYTES] {
        let one_raw = Self([1, 0, 0, 0]);
        let canonical = (&self * &one_raw).final_sub_p();
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
    /// Constant-time equality, lazy-reduction aware.
    ///
    /// Inputs are in `[0, 2p)`, where a field element has two
    /// representatives (`x` and `x + p`), so both operands are
    /// normalized to canonical `[0, p)` via `final_sub_p` before the
    /// limb-wise compare.
    fn ct_eq(&self, other: &Self) -> Choice {
        let a = self.final_sub_p();
        let b = other.final_sub_p();
        a.0[0].ct_eq(&b.0[0])
            & a.0[1].ct_eq(&b.0[1])
            & a.0[2].ct_eq(&b.0[2])
            & a.0[3].ct_eq(&b.0[3])
    }
}

impl PartialEq for Fp64 {
    /// Field equality, lazy-reduction aware: normalizes both sides to
    /// `[0, p)` so `x` and `x + p` (the same element) compare equal.
    /// Routes through the constant-time [`ConstantTimeEq`] path.
    fn eq(&self, other: &Self) -> bool {
        bool::from(self.ct_eq(other))
    }
}

impl Eq for Fp64 {}

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

    /// Modular addition.  Inputs in `[0, 2p)`; output in `[0, 2p)`.
    ///
    /// The sum is in `[0, 4p)`, which fits 4 limbs (`4p < 2^253`, so the
    /// top carry is always 0); `cond_sub_2p` brings it back to `[0, 2p)`.
    fn add(self, rhs: &'b Fp64) -> Fp64 {
        let (r0, c0) = self.0[0].carrying_add(rhs.0[0], false);
        let (r1, c1) = self.0[1].carrying_add(rhs.0[1], c0);
        let (r2, c2) = self.0[2].carrying_add(rhs.0[2], c1);
        let (r3, _c3) = self.0[3].carrying_add(rhs.0[3], c2);
        Fp64([r0, r1, r2, r3]).cond_sub_2p()
    }
}

impl<'b> Sub<&'b Fp64> for &Fp64 {
    type Output = Fp64;

    /// Modular subtraction.  Inputs in `[0, 2p)`; output in `[0, 2p)`.
    ///
    /// `a - b` with `a, b` in `[0, 2p)` is in `(-2p, 2p)`; a final
    /// borrow means it underflowed (wrapped mod 2^256), and adding `2p`
    /// back lands the result in `[0, 2p)`.
    fn sub(self, rhs: &'b Fp64) -> Fp64 {
        let (r0, b0) = self.0[0].borrowing_sub(rhs.0[0], false);
        let (r1, b1) = self.0[1].borrowing_sub(rhs.0[1], b0);
        let (r2, b2) = self.0[2].borrowing_sub(rhs.0[2], b1);
        let (r3, b3) = self.0[3].borrowing_sub(rhs.0[3], b2);
        Fp64([r0, r1, r2, r3]).cond_add_2p(Choice::from(b3 as u8))
    }
}

impl Neg for &Fp64 {
    type Output = Fp64;

    /// Modular negation: `-self mod p`.  Input in `[0, 2p)`; output in
    /// `[0, 2p)`.
    ///
    /// Computes `2p - self` (in `(0, 2p]` for `self` in `[0, 2p)`);
    /// `cond_sub_2p` maps the `self == 0` case (`2p`) back to `0`.  Uses
    /// `2p` rather than `p` so the subtraction never underflows for a
    /// non-canonical `self` in `[p, 2p)`.
    fn neg(self) -> Fp64 {
        let tp = Fp64::TWO_P.0;
        let (d0, b0) = tp[0].borrowing_sub(self.0[0], false);
        let (d1, b1) = tp[1].borrowing_sub(self.0[1], b0);
        let (d2, b2) = tp[2].borrowing_sub(self.0[2], b1);
        let (d3, _b3) = tp[3].borrowing_sub(self.0[3], b2);
        Fp64([d0, d1, d2, d3]).cond_sub_2p()
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
