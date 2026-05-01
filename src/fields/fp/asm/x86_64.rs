//! Hand-written x86_64 (baseline, no BMI2/ADX) implementations of
//! [`Fp`] primitives.
//!
//! Active under `cfg(target_arch = "x86_64")` AND the
//! `x86_64-radix-51-asm` Cargo feature. Each function mirrors the
//! algorithmic shape of the inline Rust fallback in [`super::super`]
//! but pins limb accumulators in registers and uses native carry-
//! propagation.
//!
//! Baseline-only — uses `mul reg/mem` (which destroys `rax`/`rdx` and
//! sets flags) plus a single `adc` carry chain. No `mulx`, `adcx`,
//! `adox`. This compiles and runs on every Tier-1 x86_64 target; if
//! we ever want to exploit BMI2/ADX (Broadwell+, ~2014), that goes in
//! a separate cfg path opt-in via `RUSTFLAGS` rather than runtime
//! detection.
//!
//! `mul reg/mem` reads the multiplicand from `rax` and writes the
//! 128-bit result to `rdx:rax`, so the accumulator must live in
//! different registers (we use compiler-picked operands `tlo` /
//! `thi`). Each partial product is `mov rax, a_i; mul b_j; add tlo,
//! rax; adc thi, rdx` — 4 instructions, same shape as the aarch64
//! version but with `rax`/`rdx` constrained.
//!
//! Property tests in [`super::super::tests`] cover both this asm
//! path (when the feature is on) and the Rust fallback (always).

use super::super::Fp;

/// `2 * P4 = 2 * (5 << 44)`. Used as the high-limb encoding of `2p`
/// in the speculative-subtract trick.
const TWO_P4: u64 = 2 * super::super::P4;

/// Modular addition, reduced to less than 2p.
///
/// Algorithm matches the inline Rust fallback in `super::super`:
/// 1. Per-limb adds (no carry chain — radix-51 unsaturated, 51+51
///    fits a u64).
/// 2. Speculative subtract of 2p (`+2` to limb 0, `-2*P4` to limb 4).
/// 3. Signed carry propagation through limbs 0..3 with `sar 51`.
/// 4. Sign mask = `sar 63` of limb 4.
/// 5. Conditional add-back of 2p via mask AND.
/// 6. Final signed carry propagation.
///
/// Constant-time: data-independent shifts and arithmetic; conditional
/// add-back via mask AND, no branch.
///
/// `#[inline(always)]` for the same reason as the aarch64 version —
/// without it, the wrapper-function call overhead dominates the asm.
#[inline(always)]
#[must_use]
pub(in crate::fields::fp) fn add(lhs: &Fp, rhs: &Fp) -> Fp {
    let mut out = [0u64; 5];
    // SAFETY: pointers derived from valid `&Fp` references and a
    // stack `[u64; 5]`. Asm reads 5 u64s from each side, writes 5
    // u64s to `out`. Layout matches `Fp([u64; 5])`.
    unsafe {
        core::arch::asm!(
            // Load lhs limbs.
            "mov  {l0}, qword ptr [{lhs}]",
            "mov  {l1}, qword ptr [{lhs} + 8]",
            "mov  {l2}, qword ptr [{lhs} + 16]",
            "mov  {l3}, qword ptr [{lhs} + 24]",
            "mov  {l4}, qword ptr [{lhs} + 32]",
            // Add rhs limbs (memory operand, no carry needed —
            // unsaturated 51-bit limbs sum to ≤ 52 bits in u64).
            "add  {l0}, qword ptr [{rhs}]",
            "add  {l1}, qword ptr [{rhs} + 8]",
            "add  {l2}, qword ptr [{rhs} + 16]",
            "add  {l3}, qword ptr [{rhs} + 24]",
            "add  {l4}, qword ptr [{rhs} + 32]",
            // Subtract 2p: `+2` in limb 0, `-2*P4` in limb 4.
            "add  {l0}, 2",
            "sub  {l4}, {two_p4}",
            // First prop(): signed carry propagation.
            "mov  {c}, {l0}",
            "sar  {c}, 51",
            "and  {l0}, {mask}",
            "add  {c}, {l1}",
            "mov  {l1}, {c}",
            "and  {l1}, {mask}",
            "sar  {c}, 51",
            "add  {c}, {l2}",
            "mov  {l2}, {c}",
            "and  {l2}, {mask}",
            "sar  {c}, 51",
            "add  {c}, {l3}",
            "mov  {l3}, {c}",
            "and  {l3}, {mask}",
            "sar  {c}, 51",
            "add  {l4}, {c}",
            // Sign mask: sar 63 of limb 4 (0 if non-negative,
            // all-1s if negative).
            "mov  {c}, {l4}",
            "sar  {c}, 63",
            // Conditional add-back of 2p (no branch).
            "mov  {tmp}, {c}",
            "and  {tmp}, 2",
            "sub  {l0}, {tmp}",
            "mov  {tmp}, {two_p4}",
            "and  {tmp}, {c}",
            "add  {l4}, {tmp}",
            // Second prop().
            "mov  {c}, {l0}",
            "sar  {c}, 51",
            "and  {l0}, {mask}",
            "add  {c}, {l1}",
            "mov  {l1}, {c}",
            "and  {l1}, {mask}",
            "sar  {c}, 51",
            "add  {c}, {l2}",
            "mov  {l2}, {c}",
            "and  {l2}, {mask}",
            "sar  {c}, 51",
            "add  {c}, {l3}",
            "mov  {l3}, {c}",
            "and  {l3}, {mask}",
            "sar  {c}, 51",
            "add  {l4}, {c}",
            // Store result.
            "mov  qword ptr [{out}], {l0}",
            "mov  qword ptr [{out} + 8], {l1}",
            "mov  qword ptr [{out} + 16], {l2}",
            "mov  qword ptr [{out} + 24], {l3}",
            "mov  qword ptr [{out} + 32], {l4}",
            lhs = in(reg) lhs.0.as_ptr(),
            rhs = in(reg) rhs.0.as_ptr(),
            out = in(reg) out.as_mut_ptr(),
            two_p4 = in(reg) TWO_P4,
            mask = in(reg) super::super::MASK,
            l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _,
            l3 = out(reg) _, l4 = out(reg) _,
            c = out(reg) _, tmp = out(reg) _,
            options(nostack),
        );
    }
    Fp(out)
}

/// Modular multiplication (Montgomery form), reduced to less than 2p.
///
/// 9-column schoolbook radix-2⁵¹ multiplication with interleaved
/// Montgomery reduction. Each partial product is `mov rax, a_i; mul
/// b_j; add tlo, rax; adc thi, rdx`. Each column finalize is `mov
/// rax, tlo; and rax, mask; store; shrd tlo, thi, 51; shr thi, 51`.
///
/// **Register pressure trick**: x86_64 baseline only has 14 usable
/// GPRs; we can't keep both sides' 5 limbs in registers (10) plus
/// the running accumulator (2) plus rax/rdx (mul scratch) plus
/// `v_0..v_4` (5) plus pointers/constants. Instead:
/// - `a` limbs stay in registers (5 regs).
/// - `b` limbs are loaded from memory each multiply (5 mem operands
///   per column on average).
/// - `v_k` values are written to `out[k]` (the result buffer doubles
///   as scratch) when computed in cols 0..4, then read back from
///   `out[k]` for the reduction term `v_k * P4` in cols 4..8. Once
///   `v_k` is consumed, that slot is overwritten with `c_(k-5)` from
///   col 5 onward.
///
/// `#[inline(always)]` for the same reason as `add`.
#[inline(always)]
#[must_use]
pub(in crate::fields::fp) fn mul(lhs: &Fp, rhs: &Fp) -> Fp {
    let mut out = [0u64; 5];
    // SAFETY: see `add`. The asm additionally reads/writes `out[..]`
    // in interleaved fashion (writing `v_k` then reading + overwriting)
    // — all within the bounds of the 5-element array.
    unsafe {
        core::arch::asm!(
            // ===== LOAD a[0..4] into 5 registers =====
            "mov  {a0}, qword ptr [{lhs}]",
            "mov  {a1}, qword ptr [{lhs} + 8]",
            "mov  {a2}, qword ptr [{lhs} + 16]",
            "mov  {a3}, qword ptr [{lhs} + 24]",
            "mov  {a4}, qword ptr [{lhs} + 32]",

            // ===== Column 0: a[0] * b[0] =====
            "mov  rax, {a0}",
            "mul  qword ptr [{rhs}]",
            "mov  {tlo}, rax",
            "mov  {thi}, rdx",
            // v0 = tlo & mask; out[0] = v0
            "mov  rax, {tlo}",
            "and  rax, {mask}",
            "mov  qword ptr [{out}], rax",
            // shift t right by 51
            "shrd {tlo}, {thi}, 51",
            "shr  {thi}, 51",

            // ===== Column 1: a[0]*b[1] + a[1]*b[0] =====
            "mov  rax, {a0}",
            "mul  qword ptr [{rhs} + 8]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a1}",
            "mul  qword ptr [{rhs}]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {tlo}",
            "and  rax, {mask}",
            "mov  qword ptr [{out} + 8], rax",
            "shrd {tlo}, {thi}, 51",
            "shr  {thi}, 51",

            // ===== Column 2: a[0]*b[2] + a[1]*b[1] + a[2]*b[0] =====
            "mov  rax, {a0}",
            "mul  qword ptr [{rhs} + 16]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a1}",
            "mul  qword ptr [{rhs} + 8]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a2}",
            "mul  qword ptr [{rhs}]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {tlo}",
            "and  rax, {mask}",
            "mov  qword ptr [{out} + 16], rax",
            "shrd {tlo}, {thi}, 51",
            "shr  {thi}, 51",

            // ===== Column 3: a[0]*b[3]+a[1]*b[2]+a[2]*b[1]+a[3]*b[0] =====
            "mov  rax, {a0}",
            "mul  qword ptr [{rhs} + 24]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a1}",
            "mul  qword ptr [{rhs} + 16]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a2}",
            "mul  qword ptr [{rhs} + 8]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a3}",
            "mul  qword ptr [{rhs}]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {tlo}",
            "and  rax, {mask}",
            "mov  qword ptr [{out} + 24], rax",
            "shrd {tlo}, {thi}, 51",
            "shr  {thi}, 51",

            // ===== Column 4: a[0]*b[4]+...+a[4]*b[0] + v0*P4 =====
            "mov  rax, {a0}",
            "mul  qword ptr [{rhs} + 32]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a1}",
            "mul  qword ptr [{rhs} + 24]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a2}",
            "mul  qword ptr [{rhs} + 16]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a3}",
            "mul  qword ptr [{rhs} + 8]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a4}",
            "mul  qword ptr [{rhs}]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            // v0 * P4 reduction term: load v0 from out[0]
            "mov  rax, qword ptr [{out}]",
            "mul  {p4}",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            // v4 = tlo & mask; out[4] = v4
            "mov  rax, {tlo}",
            "and  rax, {mask}",
            "mov  qword ptr [{out} + 32], rax",
            "shrd {tlo}, {thi}, 51",
            "shr  {thi}, 51",

            // ===== Column 5: a[1]*b[4]+...+a[4]*b[1] + v1*P4 =====
            "mov  rax, {a1}",
            "mul  qword ptr [{rhs} + 32]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a2}",
            "mul  qword ptr [{rhs} + 24]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a3}",
            "mul  qword ptr [{rhs} + 16]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a4}",
            "mul  qword ptr [{rhs} + 8]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            // v1 * P4 reduction term
            "mov  rax, qword ptr [{out} + 8]",
            "mul  {p4}",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            // c0 = tlo & mask; out[0] = c0 (overwrites consumed v0)
            "mov  rax, {tlo}",
            "and  rax, {mask}",
            "mov  qword ptr [{out}], rax",
            "shrd {tlo}, {thi}, 51",
            "shr  {thi}, 51",

            // ===== Column 6: a[2]*b[4]+a[3]*b[3]+a[4]*b[2] + v2*P4 =====
            "mov  rax, {a2}",
            "mul  qword ptr [{rhs} + 32]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a3}",
            "mul  qword ptr [{rhs} + 24]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a4}",
            "mul  qword ptr [{rhs} + 16]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            // v2 * P4
            "mov  rax, qword ptr [{out} + 16]",
            "mul  {p4}",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            // c1 = tlo & mask; out[1] = c1 (overwrites consumed v1)
            "mov  rax, {tlo}",
            "and  rax, {mask}",
            "mov  qword ptr [{out} + 8], rax",
            "shrd {tlo}, {thi}, 51",
            "shr  {thi}, 51",

            // ===== Column 7: a[3]*b[4]+a[4]*b[3] + v3*P4 =====
            "mov  rax, {a3}",
            "mul  qword ptr [{rhs} + 32]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            "mov  rax, {a4}",
            "mul  qword ptr [{rhs} + 24]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            // v3 * P4
            "mov  rax, qword ptr [{out} + 24]",
            "mul  {p4}",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            // c2 = tlo & mask; out[2] = c2 (overwrites consumed v2)
            "mov  rax, {tlo}",
            "and  rax, {mask}",
            "mov  qword ptr [{out} + 16], rax",
            "shrd {tlo}, {thi}, 51",
            "shr  {thi}, 51",

            // ===== Column 8: a[4]*b[4] + v4*P4 =====
            "mov  rax, {a4}",
            "mul  qword ptr [{rhs} + 32]",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            // v4 * P4
            "mov  rax, qword ptr [{out} + 32]",
            "mul  {p4}",
            "add  {tlo}, rax",
            "adc  {thi}, rdx",
            // c3 = tlo & mask; out[3] = c3 (overwrites consumed v3)
            "mov  rax, {tlo}",
            "and  rax, {mask}",
            "mov  qword ptr [{out} + 24], rax",
            // c4 = (thi:tlo) >> 51 ; store to out[4]
            "shrd {tlo}, {thi}, 51",
            "mov  qword ptr [{out} + 32], {tlo}",

            lhs = in(reg) lhs.0.as_ptr(),
            rhs = in(reg) rhs.0.as_ptr(),
            out = in(reg) out.as_mut_ptr(),
            p4 = in(reg) super::super::P4,
            mask = in(reg) super::super::MASK,
            a0 = out(reg) _, a1 = out(reg) _, a2 = out(reg) _,
            a3 = out(reg) _, a4 = out(reg) _,
            tlo = out(reg) _, thi = out(reg) _,
            // mul writes rdx:rax; declare both clobbered.
            out("rax") _, out("rdx") _,
            options(nostack),
        );
    }
    Fp(out)
}
