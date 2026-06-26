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

/// Column-scanning (Comba) truncated schoolbook multiply: returns the
/// low `N` limbs of `a * b`.
///
/// For the wide widths (`N = 30`, `N = 60`) the result cannot live in
/// registers, so the dual-chain [`mag_mul_4_adx`] form (which needs a
/// register-resident accumulator, and whose `adcx`/`adox` cannot target
/// memory) does not apply. Comba instead keeps a 192-bit accumulator
/// (`acc0:acc1:acc2`) in three registers across the whole product,
/// independent of `N`: each column sums its partial products into the
/// accumulator with one `mulx` plus an `add`/`adc`/`adc` chain, stores
/// one output limb, then shifts the accumulator down. The chain
/// completes within each product (one CF chain), so the inner loop uses
/// ordinary `cmp`/branch control. Versus the portable `u128` row form
/// this drops the `setb` carry-materialization and the per-product
/// result load/store, cutting the per-product instruction count.
///
/// # Safety
///
/// `target_feature = "adx"` and `"bmi2"` are cfg-required (`mulx`).
/// Reads `N` `u64` from each of `a` and `b`, writes `N` `u64` to the
/// output. No stack use. `N >= 1`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "adx",
    target_feature = "bmi2",
))]
#[inline]
pub(in super::super) fn mag_mul_comba<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    const { assert!(N >= 1, "mag_mul_comba: N >= 1") };
    let mut out = [0u64; N];

    // SAFETY: cfg-gated on +adx,+bmi2. `ap`/`bp` are read for `N` limbs;
    // `ocur` (initialized to the output pointer) is written for `N`
    // limbs. Loop bounds are pointer comparisons over those `N`-limb
    // ranges (a width, not data). nostack: no spills.
    unsafe {
        asm!(
            // bcol = &b[0] (column-0 start); oend = &out[N]; acc = 0.
            "mov   {bcol}, {bp}",
            "lea   {oend}, [{ocur} + {nb}]",
            "xor   {a0:e}, {a0:e}",
            "xor   {a1:e}, {a1:e}",
            "xor   {a2:e}, {a2:e}",

            // Column k: ai = &a[0], bj = &b[k] (i ascends, j descends).
            "2:",
            "mov   {ai}, {ap}",
            "mov   {bj}, {bcol}",

            // Inner: acc += a[i] * b[j] for j = k..0.
            "3:",
            "mov   rdx, [{ai}]",
            "add   {ai}, 8",
            "mulx  {hi}, {lo}, qword ptr [{bj}]",
            "add   {a0}, {lo}",
            "adc   {a1}, {hi}",
            "adc   {a2}, 0",
            "sub   {bj}, 8",
            "cmp   {bj}, {bp}",
            "jae   3b",

            // Emit out[k] = acc0, shift the accumulator down one limb.
            "mov   [{ocur}], {a0}",
            "mov   {a0}, {a1}",
            "mov   {a1}, {a2}",
            "xor   {a2:e}, {a2:e}",
            "add   {ocur}, 8",
            "add   {bcol}, 8",
            "cmp   {ocur}, {oend}",
            "jne   2b",

            ap = in(reg) a.as_ptr(),
            bp = in(reg) b.as_ptr(),
            ocur = inout(reg) out.as_mut_ptr() => _,
            nb = const N * 8,
            bcol = out(reg) _,
            oend = out(reg) _,
            ai = out(reg) _,
            bj = out(reg) _,
            a0 = out(reg) _,
            a1 = out(reg) _,
            a2 = out(reg) _,
            lo = out(reg) _,
            hi = out(reg) _,
            out("rdx") _,
            options(nostack),
        );
    }

    out
}

/// Montgomery REDC on a `2N`-limb input `(lo, hi)`, returning the reduced
/// `N`-limb value and the at-most-1 overflow above the `N`-limb window.
///
/// Mirrors the portable `MontReducer::reduce_wide` recipe — N rounds of
///
/// > `m = t[0] · n_inv (mod 2^64)`,
/// > `t[j-1] = (t[j] + m·n[j] + carry) (mod 2^64)` for `j = 1..N`,
/// > `t[N-1] = t_n + carry`,
/// > shift `t_n = t_np1 + new_high_carry`, `t_np1 = hi[round+2]` or 0.
///
/// The inner mul-add-shift loop uses `mulx` (BMI2 — destination registers
/// independent of `rdx`, so no spills around the multiplier) plus a single
/// `add`/`adc`/`adc` carry chain. Dual ADCX/ADOX interleave is left for a
/// follow-up — this is the smaller-blast-radius first attempt, matched to
/// the runtime-`N` inner loop already used by `mag_mul_comba`.
///
/// Caller compares `(t_n, t)` against `n` and subtracts when out of range.
///
/// # Safety
///
/// `target_feature = "adx"` and `"bmi2"` are cfg-required (`mulx`).
/// Reads `N` `u64` from each of `hi` and `n`; writes `N` `u64` to the
/// returned `t`. No stack use beyond the result array. `N >= 1`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "adx",
    target_feature = "bmi2",
))]
#[inline]
#[allow(dead_code)] // wired via cfg in modular.rs::reduce_wide; CI gates the perf trial
pub(in super::super) fn redc_adx<const N: usize>(
    lo: [u64; N],
    hi: &[u64; N],
    n: &[u64; N],
    n_inv_neg: u64,
) -> ([u64; N], u64) {
    const { assert!(N >= 1, "redc_adx: N >= 1") };

    let mut t = lo;
    let mut t_n: u64 = hi[0];
    let mut t_np1: u64 = if N >= 2 { hi[1] } else { 0 };

    // SAFETY: cfg-gated on +adx,+bmi2. `t_ptr` written N limbs;
    // `n_ptr`/`hi_ptr` read N limbs. Loop bounds depend only on `N` (a
    // width, not data). `t_n`/`t_np1` round-trip through `inout(reg)`.
    // nostack: no spills.
    unsafe {
        asm!(
            "xor   {round_b:e}, {round_b:e}",
            // ---- outer: per-round REDC ----
            "2:",
            // m = t[0] * n_inv_neg; mulx will keep rdx = m through the
            // round (BMI2 mulx doesn't clobber rdx).
            "mov   rdx, qword ptr [{t_ptr}]",
            "imul  rdx, {n_inv}",
            // j = 0: t[0] + m·n[0] has low bits = 0 by choice of m;
            // capture the high half + carry-out into the running c.
            "mulx  {ph}, {pl}, qword ptr [{n_ptr}]",
            "add   {pl}, qword ptr [{t_ptr}]",
            "mov   {c}, {ph}",
            "adc   {c}, 0",
            // inner: j = 1..N. j_b counts bytes (= 8 * j).
            "mov   {j_b}, 8",
            "3:",
            "mulx  {ph}, {pl}, qword ptr [{n_ptr} + {j_b}]",
            "add   {pl}, {c}",
            "adc   {ph}, 0",
            "add   {pl}, qword ptr [{t_ptr} + {j_b}]",
            "adc   {ph}, 0",
            "mov   {c}, {ph}",
            "lea   {jm1_b}, [{j_b} - 8]",
            "mov   qword ptr [{t_ptr} + {jm1_b}], {pl}",
            "add   {j_b}, 8",
            "cmp   {j_b}, {nb}",
            "jb    3b",
            // t[N-1] = t_n + c   (with carry-out into hc).
            "mov   {pl}, {t_n}",
            "add   {pl}, {c}",
            "mov   {hc:e}, 0",
            "adc   {hc}, 0",
            "lea   {jm1_b}, [{nb} - 8]",
            "mov   qword ptr [{t_ptr} + {jm1_b}], {pl}",
            // t_n = t_np1 + high_carry  (wrapping; final guard in Rust).
            "mov   {t_n}, {t_np1}",
            "add   {t_n}, {hc}",
            // t_np1 = hi[round + 2] if (round + 2) < N else 0
            "lea   {idx_b}, [{round_b} + 16]",
            "cmp   {idx_b}, {nb}",
            "jae   4f",
            "mov   {t_np1}, qword ptr [{hi_ptr} + {idx_b}]",
            "jmp   5f",
            "4:",
            "xor   {t_np1:e}, {t_np1:e}",
            "5:",
            // round_b += 8; loop while round_b < N * 8.
            "add   {round_b}, 8",
            "cmp   {round_b}, {nb}",
            "jb    2b",

            t_ptr   = in(reg) t.as_mut_ptr(),
            hi_ptr  = in(reg) hi.as_ptr(),
            n_ptr   = in(reg) n.as_ptr(),
            nb      = const N * 8,
            n_inv   = in(reg) n_inv_neg,
            t_n     = inout(reg) t_n,
            t_np1   = inout(reg) t_np1,
            round_b = out(reg) _,
            j_b     = out(reg) _,
            jm1_b   = out(reg) _,
            c       = out(reg) _,
            pl      = out(reg) _,
            ph      = out(reg) _,
            hc      = out(reg) _,
            idx_b   = out(reg) _,
            out("rdx") _,
            options(nostack),
        );
    }

    debug_assert_eq!(t_np1, 0, "redc_adx: t_np1 nonzero at end");

    (t, t_n)
}
