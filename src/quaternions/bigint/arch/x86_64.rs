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

/// Runs the `N` Montgomery-reduction rounds in place on `t` (initialized
/// by the caller to the low half of the `2N`-limb input), sliding in the
/// high half `hi`, and returns the overflow limb `t_n`.
///
/// Each round picks `m = t[0] * n_inv (mod 2^64)`, adds `m*n` with a
/// `mulx` + `add`/`adc` chain, and shifts the window down one limb; the
/// caller does the final conditional subtract (using the returned
/// `t_n`), matching the portable `reduce_wide`. The accumulator `t` is
/// memory-resident, so this uses a single CF chain (`adcx`/`adox` cannot
/// target memory); the win over the portable form is dropping the
/// per-limb `setb` carry-materialization and the redundant scalar
/// spills. Requires `N >= 2`.
///
/// # Safety
///
/// `target_feature = "adx"` and `"bmi2"` are cfg-required (`mulx`).
/// Reads/writes `N` `u64` through `t`, reads `N` `u64` from each of `hi`
/// and `n`. No stack use.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "adx",
    target_feature = "bmi2",
))]
#[inline]
pub(in super::super) fn mont_redc<const N: usize>(
    t: &mut [u64; N],
    hi: &[u64; N],
    n: &[u64; N],
    n_inv: u64,
) -> u64 {
    const { assert!(N >= 2, "mont_redc: N >= 2") };

    // Precompute the slide source `&hi[2]` and the initial window high
    // limbs in Rust, so the asm needs neither an `hi` base pointer nor an
    // end pointer (x86 register budget): `t_np1 = hi[round+2]` is gated
    // on `round < N-2` against a const, and `hs` advances each round.
    let hs0 = hi.as_ptr().wrapping_add(2);
    let t_n: u64;

    // SAFETY: cfg-gated +adx,+bmi2. `tp` read+written for `N` limbs;
    // `np` read for `N` limbs; `hs` reads `hi[2..N]` while `round < N-2`.
    unsafe {
        asm!(
            "xor   {rc:e}, {rc:e}",                  // round = 0

            // Round rc: rdx = m = t[0]*n_inv; c = (t[0] + m*n[0]) >> 64.
            "2:",
            "mov   rdx, [{tp}]",
            "imul  rdx, {ninv}",
            "mulx  {hi_p}, {lo}, qword ptr [{np}]",
            "mov   {c}, [{tp}]",
            "add   {c}, {lo}",                       // low limb cancels; take CF
            "mov   {c}, {hi_p}",
            "adc   {c}, 0",

            // Inner j = 1..N: t[j-1] = t[j] + m*n[j] + c; c = high.
            "mov   {jb}, 8",
            "3:",
            "cmp   {jb}, {nb}",
            "jae   4f",
            "mulx  {hi_p}, {lo}, qword ptr [{np} + {jb}]",
            "mov   {tmp}, [{tp} + {jb}]",
            "add   {tmp}, {c}",
            "adc   {hi_p}, 0",
            "add   {tmp}, {lo}",
            "adc   {hi_p}, 0",
            "mov   {c}, {hi_p}",
            "mov   [{tp} + {jb} - 8], {tmp}",
            "add   {jb}, 8",
            "jmp   3b",

            // t[N-1] = (t_n + c) low; t_n = t_np1 + carry; slide t_np1.
            "4:",
            "add   {tn}, {c}",
            "mov   [{tp} + {nbm8}], {tn}",
            "adc   {tnp1}, 0",
            "mov   {tn}, {tnp1}",
            "cmp   {rc:e}, {nm2}",
            "jae   5f",
            "mov   {tnp1}, [{hs}]",
            "jmp   6f",
            "5:",
            "xor   {tnp1:e}, {tnp1:e}",
            "6:",
            "add   {hs}, 8",

            // rc += 1; loop while rc < N.
            "add   {rc:e}, 1",
            "cmp   {rc:e}, {ncount}",
            "jb    2b",

            tp = in(reg) t.as_mut_ptr(),
            np = in(reg) n.as_ptr(),
            ninv = in(reg) n_inv,
            nb = const N * 8,
            nbm8 = const (N - 1) * 8,
            ncount = const N,
            nm2 = const N - 2,
            hs = inout(reg) hs0 => _,
            tn = inout(reg) hi[0] => t_n,
            tnp1 = inout(reg) hi[1] => _,
            rc = out(reg) _,
            c = out(reg) _,
            lo = out(reg) _,
            hi_p = out(reg) _,
            tmp = out(reg) _,
            jb = out(reg) _,
            out("rdx") _,
            options(nostack),
        );
    }

    t_n
}

/// 4-limb truncated squaring via dual-chain ADX, exploiting cross-term
/// symmetry `a[i]·a[j] == a[j]·a[i]`.
///
/// Returns the low 4 limbs of `a · a`, discarding the upper 256 bits.
/// Issues 6 `mulx` instructions vs the 10 of [`mag_mul_4_adx`] applied to
/// `(a, a)`: 4 cross products (`a[0]·a[1..3]`, `a[1]·a[2]`, each
/// contributing to the truncated output once doubled) plus 2 diagonals
/// (`a[0]²`, `a[1]²`).  The remaining diagonals (`a[2]²`, `a[3]²`) and
/// crosses (`a[1]·a[3]`, `a[2]·a[3]`) land at positions 4-6, outside the
/// truncated output.
///
/// Mirrors the cross-term recipe in `super::super::modular`'s wide
/// squaring, specialized to N=4 and truncated.
///
/// # Safety
///
/// `target_feature = "adx"` and `"bmi2"` are cfg-required; the asm uses
/// MULX/ADCX/ADOX unconditionally.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "adx",
    target_feature = "bmi2",
))]
#[inline]
pub(in super::super) fn mag_sqr_4_adx(a: &[u64; 4]) -> [u64; 4] {
    let mut r0: u64;
    let mut r1: u64;
    let mut r2: u64;
    let mut r3: u64;

    // SAFETY: cfg-gated on +adx and +bmi2; readonly (no memory writes);
    // nostack.  Reads exactly 32 bytes from `a` through its raw pointer.
    unsafe {
        asm!(
            // Pre-doubling target positions for the cross-product sum:
            //   pos 1 = lo(a[0]·a[1])
            //   pos 2 = hi(a[0]·a[1]) + lo(a[0]·a[2])
            //   pos 3 = hi(a[0]·a[2]) + lo(a[0]·a[3]) + lo(a[1]·a[2])
            // Position-4 contributions (hi of a[0]·a[3], hi of a[1]·a[2],
            // carries) are discarded.
            "xor {r3:e}, {r3:e}",

            // Cross-products fanning out from a[0]: a[0]·a[1..3].
            "mov rdx, qword ptr [{a} + 0]",
            "mulx {r2}, {r1}, qword ptr [{a} + 8]",

            "mulx {hi}, {lo}, qword ptr [{a} + 16]",
            "add {r2}, {lo}",
            "adc {r3}, {hi}",

            "mulx {hi}, {lo}, qword ptr [{a} + 24]",
            "add {r3}, {lo}",

            // Last cross-product a[1]·a[2]; only the low half stays in range.
            "mov rdx, qword ptr [{a} + 8]",
            "mulx {hi}, {lo}, qword ptr [{a} + 16]",
            "add {r3}, {lo}",

            // Double (r1, r2, r3) <<= 1; bit shifted out of r3 discarded.
            "shl {r1}, 1",
            "adc {r2}, {r2}",
            "adc {r3}, {r3}",

            // Diagonal a[0]^2 lands at (pos 0, pos 1).
            "mov rdx, qword ptr [{a} + 0]",
            "mulx {hi}, {r0}, rdx",
            "add {r1}, {hi}",

            // Diagonal a[1]^2 lands at (pos 2, pos 3).
            "mov rdx, qword ptr [{a} + 8]",
            "mulx {hi}, {lo}, rdx",
            "adc {r2}, {lo}",
            "adc {r3}, {hi}",

            a = in(reg) a.as_ptr(),
            r0 = out(reg) r0,
            r1 = out(reg) r1,
            r2 = out(reg) r2,
            r3 = out(reg) r3,
            lo = out(reg) _,
            hi = out(reg) _,
            out("rdx") _,
            options(nostack, readonly),
        );
    }

    [r0, r1, r2, r3]
}
