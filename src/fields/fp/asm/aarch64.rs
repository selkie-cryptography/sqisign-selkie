//! Hand-written aarch64 implementations of [`Fp`] primitives.
//!
//! Active under `cfg(target_arch = "aarch64")`. Each function mirrors
//! the algorithmic shape of the inline Rust fallback in
//! [`super::super`] but pins the limb accumulators in registers and
//! uses native carry-propagation instructions. Property tests in
//! [`super::super::tests`] exercise the asm path on every aarch64 CI
//! runner.
//!
//! Register conventions inside each `asm!` block:
//! - `x4..x8`  hold the 5-limb running result (limb 0 in `x4`,
//!   limb 4 in `x8`)
//! - `x9, x10` are scratch (carry word, sign mask, masked constants)
//! - `{lhs}, {rhs}, {out}, {two_p4}` are register operands picked by
//!   the compiler from the un-clobbered pool

use super::super::Fp;

/// `2 * P4 = 2 * (5 << 44)`. Used as the high-limb encoding of `2p`
/// in the speculative-subtract / conditional-add-back trick shared by
/// [`add`] and [`sub`].
const TWO_P4: u64 = 2 * super::super::P4;

/// Modular addition, reduced to less than 2p.
///
/// Algorithm matches the inline Rust fallback in `super::super`:
/// 1. Five per-limb adds (no carry chain — limbs are unsaturated 51-bit
///    so 51+51 = 52 bits fits a u64).
/// 2. Speculative subtract of 2p, encoded as `+2` to limb 0 and
///    `-2*P4` to limb 4.
/// 3. Signed carry propagation through limbs 0..3 with `asr #51`.
/// 4. Sign mask = `asr #63` of limb 4 (0 if non-negative, all-1s if
///    negative).
/// 5. Conditional add-back of 2p (mask-and `2` from limb 0, mask-and
///    `2*P4` to limb 4).
/// 6. Final signed carry propagation.
///
/// Constant-time: all arithmetic and shifts are data-independent;
/// the conditional add-back is implemented with mask AND, no branch.
///
/// `#[inline(always)]` is load-bearing: without it, the wrapper
/// function call across the module boundary costs more than the asm
/// saves, making `Fp::add` slower in benches than the inlined Rust
/// fallback (see commit notes).
#[inline(always)]
#[must_use]
pub(in crate::fields::fp) fn add(lhs: &Fp, rhs: &Fp) -> Fp {
    let mut out = [0u64; 5];
    // SAFETY: All memory accesses are through pointers derived from
    // valid `&Fp` references and a stack-allocated `[u64; 5]` of the
    // expected size. The asm block reads 5 u64s from each of `lhs`
    // and `rhs` and writes 5 u64s to `out`, matching the layout of
    // `Fp([u64; 5])`. No unaligned accesses (Rust guarantees u64
    // alignment for the underlying storage).
    unsafe {
        core::arch::asm!(
            // Load lhs limbs into x4..x8.
            "ldp  x4, x5, [{lhs}]",
            "ldp  x6, x7, [{lhs}, #16]",
            "ldr  x8, [{lhs}, #32]",
            // Load rhs limbs in pairs into x9/x10, add into running
            // result, freeing the load registers immediately for
            // reuse on the next pair.
            "ldp  x9, x10, [{rhs}]",
            "add  x4, x4, x9",
            "add  x5, x5, x10",
            "ldp  x9, x10, [{rhs}, #16]",
            "add  x6, x6, x9",
            "add  x7, x7, x10",
            "ldr  x9, [{rhs}, #32]",
            "add  x8, x8, x9",
            // Subtract 2p: `+2` in limb 0, `-2*P4` in limb 4.
            "add  x4, x4, #2",
            "sub  x8, x8, {two_p4}",
            // First prop(): signed carry propagation through limbs
            // 0..3, then absorb the final carry into limb 4 without
            // masking.
            "asr  x9, x4, #51",
            "and  x4, x4, {mask}",
            "add  x9, x9, x5",
            "and  x5, x9, {mask}",
            "asr  x9, x9, #51",
            "add  x9, x9, x6",
            "and  x6, x9, {mask}",
            "asr  x9, x9, #51",
            "add  x9, x9, x7",
            "and  x7, x9, {mask}",
            "asr  x9, x9, #51",
            "add  x8, x8, x9",
            // Sign mask: `asr #63` spreads the top bit of limb 4 to
            // all 64 bits (0 if non-negative, all-1s if negative).
            "asr  x9, x8, #63",
            // Conditional add-back of 2p (no branch).
            "and  x10, x9, #2",
            "sub  x4, x4, x10",
            "and  x10, x9, {two_p4}",
            "add  x8, x8, x10",
            // Second prop(): clean up the limbs after the conditional
            // fix-up.
            "asr  x9, x4, #51",
            "and  x4, x4, {mask}",
            "add  x9, x9, x5",
            "and  x5, x9, {mask}",
            "asr  x9, x9, #51",
            "add  x9, x9, x6",
            "and  x6, x9, {mask}",
            "asr  x9, x9, #51",
            "add  x9, x9, x7",
            "and  x7, x9, {mask}",
            "asr  x9, x9, #51",
            "add  x8, x8, x9",
            // Store result into out[0..5].
            "stp  x4, x5, [{out}]",
            "stp  x6, x7, [{out}, #16]",
            "str  x8, [{out}, #32]",
            lhs = in(reg) lhs.0.as_ptr(),
            rhs = in(reg) rhs.0.as_ptr(),
            out = in(reg) out.as_mut_ptr(),
            two_p4 = in(reg) TWO_P4,
            mask = const super::super::MASK,
            out("x4") _, out("x5") _, out("x6") _, out("x7") _,
            out("x8") _, out("x9") _, out("x10") _,
            options(nostack, preserves_flags),
        );
    }
    Fp(out)
}

/// Modular multiplication (Montgomery form), reduced to less than 2p.
///
/// 9-column schoolbook radix-2⁵¹ multiplication with interleaved
/// Montgomery reduction. For each column k, the partial products
/// `a[i] * b[j]` (i + j = k) are accumulated into a 128-bit running
/// total `(t_lo, t_hi)`; for k ≥ 4, the reduction term `v_(k-4) * P4`
/// is also folded in. After each column the low 51 bits are extracted
/// (as `v_k` for k < 5, then directly as result limb `c_(k-5)` for
/// k ≥ 5) and the accumulator is shifted right by 51 bits via `extr`.
///
/// Each partial product is `mul + umulh + adds + adc` (4 instructions);
/// each column finalize is `and + extr + lsr` (3 instructions). The
/// 5 inputs of each side, the running 2-word accumulator, the 5
/// `v_k`s, and 2 multiply temporaries fit comfortably in the 24-ish
/// callee-clobbered GPRs we use here.
///
/// `#[inline(always)]` for the same reason as [`add`].
#[inline(always)]
#[must_use]
pub(in crate::fields::fp) fn mul(lhs: &Fp, rhs: &Fp) -> Fp {
    let mut out = [0u64; 5];
    // SAFETY: see `add`.
    unsafe {
        core::arch::asm!(
            // ===== LOAD =====
            "ldp  {a0}, {a1}, [{lhs}]",
            "ldp  {a2}, {a3}, [{lhs}, #16]",
            "ldr  {a4}, [{lhs}, #32]",
            "ldp  {b0}, {b1}, [{rhs}]",
            "ldp  {b2}, {b3}, [{rhs}, #16]",
            "ldr  {b4}, [{rhs}, #32]",

            // ===== Column 0: a[0]*b[0] =====
            "mul   {tlo}, {a0}, {b0}",
            "umulh {thi}, {a0}, {b0}",
            "and   {v0}, {tlo}, {mask}",
            "extr  {tlo}, {thi}, {tlo}, #51",
            "lsr   {thi}, {thi}, #51",

            // ===== Column 1: a[0]*b[1] + a[1]*b[0] =====
            "mul   {t1}, {a0}, {b1}",
            "umulh {t2}, {a0}, {b1}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a1}, {b0}",
            "umulh {t2}, {a1}, {b0}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "and   {v1}, {tlo}, {mask}",
            "extr  {tlo}, {thi}, {tlo}, #51",
            "lsr   {thi}, {thi}, #51",

            // ===== Column 2: a[0]*b[2] + a[1]*b[1] + a[2]*b[0] =====
            "mul   {t1}, {a0}, {b2}",
            "umulh {t2}, {a0}, {b2}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a1}, {b1}",
            "umulh {t2}, {a1}, {b1}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a2}, {b0}",
            "umulh {t2}, {a2}, {b0}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "and   {v2}, {tlo}, {mask}",
            "extr  {tlo}, {thi}, {tlo}, #51",
            "lsr   {thi}, {thi}, #51",

            // ===== Column 3: a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0] =====
            "mul   {t1}, {a0}, {b3}",
            "umulh {t2}, {a0}, {b3}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a1}, {b2}",
            "umulh {t2}, {a1}, {b2}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a2}, {b1}",
            "umulh {t2}, {a2}, {b1}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a3}, {b0}",
            "umulh {t2}, {a3}, {b0}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "and   {v3}, {tlo}, {mask}",
            "extr  {tlo}, {thi}, {tlo}, #51",
            "lsr   {thi}, {thi}, #51",

            // ===== Column 4: a[0]*b[4] + ... + a[4]*b[0] + v0*P4 =====
            "mul   {t1}, {a0}, {b4}",
            "umulh {t2}, {a0}, {b4}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a1}, {b3}",
            "umulh {t2}, {a1}, {b3}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a2}, {b2}",
            "umulh {t2}, {a2}, {b2}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a3}, {b1}",
            "umulh {t2}, {a3}, {b1}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a4}, {b0}",
            "umulh {t2}, {a4}, {b0}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            // v0 * P4 reduction term
            "mul   {t1}, {v0}, {p4}",
            "umulh {t2}, {v0}, {p4}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "and   {v4}, {tlo}, {mask}",
            "extr  {tlo}, {thi}, {tlo}, #51",
            "lsr   {thi}, {thi}, #51",

            // ===== Column 5: a[1]*b[4] + ... + a[4]*b[1] + v1*P4 =====
            // v0 is consumed; reuse its register for c0 by overwriting.
            "mul   {t1}, {a1}, {b4}",
            "umulh {t2}, {a1}, {b4}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a2}, {b3}",
            "umulh {t2}, {a2}, {b3}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a3}, {b2}",
            "umulh {t2}, {a3}, {b2}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a4}, {b1}",
            "umulh {t2}, {a4}, {b1}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {v1}, {p4}",
            "umulh {t2}, {v1}, {p4}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "and   {v0}, {tlo}, {mask}",     // c0 (overwriting consumed v0)
            "extr  {tlo}, {thi}, {tlo}, #51",
            "lsr   {thi}, {thi}, #51",

            // ===== Column 6: a[2]*b[4] + a[3]*b[3] + a[4]*b[2] + v2*P4 =====
            "mul   {t1}, {a2}, {b4}",
            "umulh {t2}, {a2}, {b4}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a3}, {b3}",
            "umulh {t2}, {a3}, {b3}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a4}, {b2}",
            "umulh {t2}, {a4}, {b2}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {v2}, {p4}",
            "umulh {t2}, {v2}, {p4}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "and   {v1}, {tlo}, {mask}",     // c1 (overwriting consumed v1)
            "extr  {tlo}, {thi}, {tlo}, #51",
            "lsr   {thi}, {thi}, #51",

            // ===== Column 7: a[3]*b[4] + a[4]*b[3] + v3*P4 =====
            "mul   {t1}, {a3}, {b4}",
            "umulh {t2}, {a3}, {b4}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {a4}, {b3}",
            "umulh {t2}, {a4}, {b3}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {v3}, {p4}",
            "umulh {t2}, {v3}, {p4}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "and   {v2}, {tlo}, {mask}",     // c2 (overwriting consumed v2)
            "extr  {tlo}, {thi}, {tlo}, #51",
            "lsr   {thi}, {thi}, #51",

            // ===== Column 8: a[4]*b[4] + v4*P4 =====
            "mul   {t1}, {a4}, {b4}",
            "umulh {t2}, {a4}, {b4}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "mul   {t1}, {v4}, {p4}",
            "umulh {t2}, {v4}, {p4}",
            "adds  {tlo}, {tlo}, {t1}",
            "adc   {thi}, {thi}, {t2}",
            "and   {v3}, {tlo}, {mask}",     // c3 (overwriting consumed v3)
            "extr  {tlo}, {thi}, {tlo}, #51", // c4 in tlo

            // ===== STORE c0..c4 =====
            "stp   {v0}, {v1}, [{out}]",       // c0, c1
            "stp   {v2}, {v3}, [{out}, #16]",  // c2, c3
            "str   {tlo},      [{out}, #32]",  // c4

            lhs = in(reg) lhs.0.as_ptr(),
            rhs = in(reg) rhs.0.as_ptr(),
            out = in(reg) out.as_mut_ptr(),
            p4 = in(reg) super::super::P4,
            mask = const super::super::MASK,
            a0 = out(reg) _, a1 = out(reg) _, a2 = out(reg) _, a3 = out(reg) _, a4 = out(reg) _,
            b0 = out(reg) _, b1 = out(reg) _, b2 = out(reg) _, b3 = out(reg) _, b4 = out(reg) _,
            tlo = out(reg) _, thi = out(reg) _,
            t1 = out(reg) _, t2 = out(reg) _,
            v0 = out(reg) _, v1 = out(reg) _, v2 = out(reg) _, v3 = out(reg) _, v4 = out(reg) _,
            options(nostack),
        );
    }
    Fp(out)
}
