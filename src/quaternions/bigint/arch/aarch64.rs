//! aarch64 backend for [`BigInt`][super::super::BigInt] primitives.
//!
//! [`mag_mul_comba`] is a column-scanning (Comba / product-scanning)
//! truncated schoolbook multiply. Unlike the row-scanning form LLVM
//! emits from the portable `widening_mul` loop, it keeps a 192-bit
//! column accumulator (`acc0:acc1:acc2`) in registers and folds each
//! `a[i]*b[j]` in with one `mul`/`umulh` plus an `adds`/`adcs`/`adc`
//! triple, storing one output limb per column. That removes the
//! per-product result load/store and the per-product carry
//! materialization the row form pays, which is the win on Apple
//! Silicon where those stack round-trips dominate the wide
//! (`N = 30`, `N = 60`) lattice multiplies.
//!
//! aarch64 has a single carry flag (no `adcx`/`adox` dual chain), so
//! the column accumulation is serial; the gain is from cutting memory
//! traffic, not from parallel carry chains.
//!
//! Constant-time: `mul`/`umulh`/`adcs` are register-only, data-flow
//! with no data-dependent control flow or memory addressing (the loop
//! bounds depend only on `N`, a compile-time width). Microarchitectural
//! CT must be reverified per Apple-Silicon revision (M1-M4 differ in
//! multiplier latency).
//!
//! # Note on field arithmetic
//!
//! The much larger aarch64 lever for SQIsign is in
//! [`crate::fields::fp`], not here. Per De Feo et al. 2026/394,
//! a vectorised radix-29 `Fp` is the ~1.22x speedup on Apple M1.
//! That backend lives at `crate::fields::fp::arch::aarch64`; this
//! module covers only the quaternion-side `BigInt` primitives.

#[cfg(target_arch = "aarch64")]
use core::arch::asm;

/// Column-scanning (Comba) truncated schoolbook multiply: returns the
/// low `N` limbs of `a * b`.
///
/// For each output column `k` in `0..N`, sums `a[i] * b[k-i]` for
/// `i in 0..=k` into a 192-bit accumulator carried across columns,
/// emits `out[k]`, then shifts the accumulator down one limb. The
/// products at positions `>= N` are never formed (truncation).
///
/// # Safety
///
/// Reads `N` `u64` from each of `a` and `b` and writes `N` `u64` to the
/// output through register-held pointers. No stack use. `N >= 1`
/// (`N == 0` would run zero columns and return the zero array, but the
/// caller never instantiates `BigInt<0>`).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(in super::super) fn mag_mul_comba<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    let mut out = [0u64; N];

    // SAFETY: `aptr`/`bptr` are read for `N` limbs, `optr` written for
    // `N` limbs; the loops are bounded by `N` (a width, not data), and
    // every other operand is a scratch register. `nostack`: no spills.
    unsafe {
        asm!(
            // acc0:acc1:acc2 = 0; k = 0.
            "mov   {k}, xzr",
            "mov   {a0}, xzr",
            "mov   {a1}, xzr",
            "mov   {a2}, xzr",

            // Column k: ai = &a[0], bj = &b[k], cnt = k + 1.
            "2:",
            "mov   {ai}, {aptr}",
            "add   {bj}, {bptr}, {k}, lsl #3",
            "add   {cnt}, {k}, #1",

            // Inner: i = 0..=k, j = k - i. acc += a[i] * b[j].
            "3:",
            "ldr   {x}, [{ai}], #8",               // x = a[i]; ai += 8
            "ldr   {y}, [{bj}]",                   // y = b[j]
            "sub   {bj}, {bj}, #8",                // bj -= 8 (j decreases)
            "mul   {pl}, {x}, {y}",
            "umulh {ph}, {x}, {y}",
            "adds  {a0}, {a0}, {pl}",
            "adcs  {a1}, {a1}, {ph}",
            "adc   {a2}, {a2}, xzr",
            "subs  {cnt}, {cnt}, #1",
            "b.ne  3b",

            // Emit out[k] = acc0, shift the accumulator down one limb.
            "str   {a0}, [{optr}, {k}, lsl #3]",
            "mov   {a0}, {a1}",
            "mov   {a1}, {a2}",
            "mov   {a2}, xzr",

            // k += 1; repeat while k < N.
            "add   {k}, {k}, #1",
            "cmp   {k}, {n}",
            "b.lo  2b",

            aptr = in(reg) a.as_ptr(),
            bptr = in(reg) b.as_ptr(),
            optr = in(reg) out.as_mut_ptr(),
            n = in(reg) N,
            k = out(reg) _,
            a0 = out(reg) _,
            a1 = out(reg) _,
            a2 = out(reg) _,
            ai = out(reg) _,
            bj = out(reg) _,
            cnt = out(reg) _,
            x = out(reg) _,
            y = out(reg) _,
            pl = out(reg) _,
            ph = out(reg) _,
            options(nostack),
        );
    }

    out
}
