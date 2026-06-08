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

/// One CIOS Montgomery iteration via dual-chain ADX: folds `a · b_i`
/// into the accumulator, then runs one reduction round.
///
/// `t` is the `N+1`-word accumulator (`t[0..N]` plus a top overflow
/// word at `t[N]`). On entry it holds the partial product after the
/// previous iterations; on exit it holds the value after folding the
/// `i`-th limb of `b` (`b_i`) and reducing once (the result is shifted
/// down one word, so a full `mul` runs this `N` times).
///
/// Carry management: the multiply row and reduce row each open with a
/// fresh `xor` (clearing CF and OF), run mulx + adcx (CF chain) + adox
/// (OF chain) over the `N` limbs via a `loop`-counted inner loop —
/// `loop` and `lea` preserve both flags, so the dual chains survive the
/// iteration counter. The two residual carries plus the cross-iteration
/// top word are folded with plain `add`/`adc` after each row. The final
/// word shift (`>> 64`, dropping the canceled `t[0]`) is `N` plain
/// `mov`s after both flag chains are done.
///
/// # Safety
///
/// `target_feature = "adx"` and `"bmi2"` are cfg-required; the asm uses
/// MULX/ADCX/ADOX unconditionally. `t` must point to `N+1` writable
/// words; `a` and `n` to `N` readable words each.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "adx",
    target_feature = "bmi2",
))]
#[inline]
unsafe fn mont_cios_iter(
    t: *mut u64,
    a: *const u64,
    n: *const u64,
    b_i: u64,
    n_inv: u64,
    len: u64,
) {
    // SAFETY: cfg-gated on +adx,+bmi2; caller guarantees t has len+1
    // writable words and a,n have len readable words. nostack.
    unsafe {
        // adcx/adox require a register destination (only the source may
        // be memory), so each accumulator word is load-modify-stored
        // through a temp register (`tmp`). The CF/OF chains live in the
        // flags across the load/store movs, which don't touch CF/OF.
        // Register-pressure note: `a` and `n` are rolled in place (and
        // thus clobbered) to avoid holding separate base + cursor regs;
        // neither base is needed after its row. A zeroed `hi` doubles as
        // the carry-fold source after each multiply chain (mulx leaves
        // `hi` dead at the residual-fold point).
        asm!(
            // ---- Multiply row: t += a * b_i ----
            "mov rdx, {b_i}",
            "xor {hi:e}, {hi:e}",                 // clear CF, OF; hi = 0
            "mov rcx, {len}",
            "mov {tp}, {t}",
            "2:",                                 // multiply-row loop
            "mulx {hi}, {lo}, qword ptr [{a}]",
            "mov {tmp}, qword ptr [{tp}]",
            "adcx {tmp}, {lo}",                   // CF chain: t[j] += lo
            "mov qword ptr [{tp}], {tmp}",
            "mov {tmp}, qword ptr [{tp} + 8]",
            "adox {tmp}, {hi}",                   // OF chain: t[j+1] += hi
            "mov qword ptr [{tp} + 8], {tmp}",
            "lea {a}, [{a} + 8]",                 // flag-preserving advance
            "lea {tp}, [{tp} + 8]",
            "loop 2b",                            // dec rcx, preserve flags
            // tp now points at t[len]. Fold residual CF into t[len];
            // the residual OF plus that carry-out becomes the
            // cross-row top word `th`. `mov`-zeroing preserves CF/OF.
            "mov {lo}, 0",                        // zero source (no flag clobber)
            "mov {tmp}, qword ptr [{tp}]",
            "adcx {tmp}, {lo}",                   // t[len] += CF; CF = carry-out
            "mov qword ptr [{tp}], {tmp}",
            "mov {th}, 0",
            "adox {th}, {lo}",                    // th = OF
            "adc {th}, {lo}",                     // th += carry-out (CF)

            // ---- Reduce row: m = t[0]*n_inv; t += m*n; shift ----
            "mov rdx, qword ptr [{t}]",
            "imul rdx, {n_inv}",                  // m = t[0] * n_inv mod 2^64
            "xor {hi:e}, {hi:e}",                 // clear CF, OF; hi = 0
            "mov {tp}, {t}",
            // j=0: t[0] += lo cancels to 0 (CF out); t[1] += hi.
            "mulx {hi}, {lo}, qword ptr [{n}]",
            "mov {tmp}, qword ptr [{tp}]",
            "adcx {tmp}, {lo}",                   // t[0] -> 0, CF out
            "mov qword ptr [{tp}], {tmp}",
            "mov {tmp}, qword ptr [{tp} + 8]",
            "adox {tmp}, {hi}",
            "mov qword ptr [{tp} + 8], {tmp}",
            "lea {n}, [{n} + 8]",
            "lea {tp}, [{tp} + 8]",
            // j=1..len-1. `lea` sets the counter to len-1 without
            // touching CF/OF (the j=0 chains must reach the loop intact).
            "lea rcx, [{len} - 1]",
            "3:",
            "mulx {hi}, {lo}, qword ptr [{n}]",
            "mov {tmp}, qword ptr [{tp}]",
            "adcx {tmp}, {lo}",
            "mov qword ptr [{tp}], {tmp}",
            "mov {tmp}, qword ptr [{tp} + 8]",
            "adox {tmp}, {hi}",
            "mov qword ptr [{tp} + 8], {tmp}",
            "lea {n}, [{n} + 8]",
            "lea {tp}, [{tp} + 8]",
            "loop 3b",
            // tp points at t[len]. Fold residual CF into t[len]; the
            // post-shift top word is `th + OF + carry-out` (the
            // conceptual t[len+1]). `mov`-zeroing preserves CF/OF.
            "mov {hi}, 0",                        // zero source (no flag clobber)
            "mov {tmp}, qword ptr [{tp}]",
            "adcx {tmp}, {hi}",                   // t[len] += CF; CF = carry-out
            "mov qword ptr [{tp}], {tmp}",
            "adox {th}, {hi}",                    // th += OF
            "adc {th}, {hi}",                     // th += carry-out (CF)
            // `th` now holds the conceptual t[len+1] word.

            // ---- Shift down one word: t[k] = t[k+1] for k<len ----
            "mov rcx, {len}",
            "mov {tp}, {t}",
            "4:",
            "mov {lo}, qword ptr [{tp} + 8]",
            "mov qword ptr [{tp}], {lo}",
            "lea {tp}, [{tp} + 8]",
            "loop 4b",
            // After shift, the old t[len] sits at t[len-1]; the new top
            // word t[len] is the saved conceptual t[len+1] (`th`).
            "mov qword ptr [{tp}], {th}",

            t = in(reg) t,
            a = inout(reg) a => _,
            n = inout(reg) n => _,
            b_i = in(reg) b_i,
            n_inv = in(reg) n_inv,
            len = in(reg) len,
            tp = out(reg) _,
            th = out(reg) _,
            lo = out(reg) _,
            hi = out(reg) _,
            tmp = out(reg) _,
            out("rcx") _,
            out("rdx") _,
            options(nostack),
        );
    }
}

/// CIOS Montgomery multiply via dual-chain ADX for arbitrary width `N`.
///
/// Computes `a · b · R^{-1} mod n` (where `R = 2^{64N}`) for
/// Montgomery-form operands in `[0, n)`, returning a result in
/// `[0, n)`. Runs [`mont_cios_iter`] once per limb of `b`, then a
/// single conditional subtract. This is the asm counterpart of the
/// portable CIOS loop in `super::super::modular::MontReducer::mul`,
/// which stays the correctness oracle.
///
/// # Safety
///
/// `target_feature = "adx"` and `"bmi2"` are cfg-required.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "adx",
    target_feature = "bmi2",
))]
#[inline]
pub(in super::super) fn mont_mul_adx<const N: usize>(
    a: &[u64; N],
    b: &[u64; N],
    n: &[u64; N],
    n_inv: u64,
) -> [u64; N] {
    // Contiguous N+1-word accumulator: t[0..N] plus a top overflow word
    // at index N. `MONT_ADX_MAX` caps the supported width; the caller
    // (`MontReducer::mul`) only dispatches here when `N + 1 <= MONT_ADX_MAX`.
    let mut buf = [0u64; MONT_ADX_MAX];
    debug_assert!(N < MONT_ADX_MAX, "mont_mul_adx: N+1 exceeds MONT_ADX_MAX");

    for &b_i in b {
        // SAFETY: buf has >= N+1 words; a and n have N words each.
        unsafe {
            mont_cios_iter(
                buf.as_mut_ptr(),
                a.as_ptr(),
                n.as_ptr(),
                b_i,
                n_inv,
                N as u64,
            );
        }
    }

    let mut t = [0u64; N];
    t.copy_from_slice(&buf[..N]);
    let t_hi = buf[N];

    // Result in [0, 2n): one conditional subtract reduces to [0, n).
    if t_hi != 0 || super::super::BigInt::<N>::mag_cmp(&t, n) != core::cmp::Ordering::Less {
        let (sub, _) = super::super::BigInt::<N>::mag_sub(&t, n);
        t = sub;
    }
    t
}

/// Upper bound on `N + 1` for [`mont_mul_adx`]'s stack accumulator.
/// Covers every Montgomery width SQIsign uses (storage `N <= 18`,
/// `pow_mod_w` working width `W = 18`) with headroom.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "adx",
    target_feature = "bmi2",
))]
pub(in super::super) const MONT_ADX_MAX: usize = 33;

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
