//! x86_64 ADX backend for [`BigInt`][super::super::BigInt] primitives.
//!
//! `core::arch::asm!` schoolbook multiplication for the N=4 case
//! (truncated 256-bit × 256-bit → 256-bit), interleaving MULX (BMI2)
//! with dual-carry ADCX (CF chain) and ADOX (OF chain).  The two
//! chains run on independent flags, doubling effective carry
//! throughput vs the single-CF-chain rustc emits from `widening_mul`
//! + `overflowing_add`.
//!
//! As of rustc 1.95 / LLVM 19, both `_addcarry_u64` and
//! `_addcarryx_u64` lower to ADCX; ADOX has no stable intrinsic, so
//! explicit asm is the only path to the dual-chain win.  Codegen
//! inspection of the portable schoolbook confirms it already emits
//! `mulx + adc` — everything beyond that needs ADCX+ADOX interleave.
//!
//! Constant-time: MULX/ADCX/ADOX operate on register-width words with
//! no data-dependent control flow or memory access, and have
//! data-independent latency on Broadwell+ / Excavator+ / Zen+.
//! Microarchitectural CT must be reverified per CPU family via
//! ctgrind / dudect when this lands.

#[cfg(all(target_feature = "adx", target_feature = "bmi2"))]
use core::arch::asm;

/// 4×4 limb truncated schoolbook multiplication via dual-chain ADX.
///
/// Returns the low 4 limbs of `a · b`, discarding the upper 256 bits.
///
/// # Safety
///
/// `target_feature = "adx"` and `"bmi2"` are cfg-required; the asm
/// uses MULX/ADCX/ADOX unconditionally.  Available on Broadwell+
/// Intel (2014) and Excavator+ / Zen+ AMD (2015+).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "adx",
    target_feature = "bmi2",
))]
#[inline]
pub(in super::super) fn mag_mul_4_adx(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut r0: u64;
    let mut r1: u64;
    let mut r2: u64;
    let mut r3: u64;

    // SAFETY: cfg-gated on +adx and +bmi2; no memory writes
    // (readonly); no stack usage (nostack).  Reads exactly 32 bytes
    // from each of a and b through their raw pointers.
    unsafe {
        asm!(
            // Zero the four accumulator registers.
            "xor {r0:e}, {r0:e}",
            "xor {r1:e}, {r1:e}",
            "xor {r2:e}, {r2:e}",
            "xor {r3:e}, {r3:e}",

            // Row 0: a[0] * b[0..3]
            "mov rdx, qword ptr [{a} + 0]",
            "xor eax, eax",                              // clear CF, OF
            "mulx {hi}, {lo}, qword ptr [{b} + 0]",
            "adcx {r0}, {lo}",
            "adox {r1}, {hi}",
            "mulx {hi}, {lo}, qword ptr [{b} + 8]",
            "adcx {r1}, {lo}",
            "adox {r2}, {hi}",
            "mulx {hi}, {lo}, qword ptr [{b} + 16]",
            "adcx {r2}, {lo}",
            "adox {r3}, {hi}",
            "mulx {hi}, {lo}, qword ptr [{b} + 24]",
            "adcx {r3}, {lo}",
            // hi at position 4 discarded (truncation); final CF/OF
            // also discarded.

            // Row 1: a[1] * b[0..2]
            "mov rdx, qword ptr [{a} + 8]",
            "xor eax, eax",
            "mulx {hi}, {lo}, qword ptr [{b} + 0]",
            "adcx {r1}, {lo}",
            "adox {r2}, {hi}",
            "mulx {hi}, {lo}, qword ptr [{b} + 8]",
            "adcx {r2}, {lo}",
            "adox {r3}, {hi}",
            "mulx {hi}, {lo}, qword ptr [{b} + 16]",
            "adcx {r3}, {lo}",

            // Row 2: a[2] * b[0..1]
            "mov rdx, qword ptr [{a} + 16]",
            "xor eax, eax",
            "mulx {hi}, {lo}, qword ptr [{b} + 0]",
            "adcx {r2}, {lo}",
            "adox {r3}, {hi}",
            "mulx {hi}, {lo}, qword ptr [{b} + 8]",
            "adcx {r3}, {lo}",

            // Row 3: a[3] * b[0]
            "mov rdx, qword ptr [{a} + 24]",
            "xor eax, eax",
            "mulx {hi}, {lo}, qword ptr [{b} + 0]",
            "adcx {r3}, {lo}",

            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            r0 = out(reg) r0,
            r1 = out(reg) r1,
            r2 = out(reg) r2,
            r3 = out(reg) r3,
            lo = out(reg) _,
            hi = out(reg) _,
            out("rax") _,
            out("rdx") _,
            options(nostack, readonly),
        );
    }

    [r0, r1, r2, r3]
}
