//! Architecture-specific backends for [`BigInt`][super::BigInt] primitives.
//!
//! Per-target backend selected at compile time. `x86_64` provides ADX
//! (`asm!` MULX + ADCX/ADOX dual-chain) multiply/square plus the fused
//! dual-chain add-and-subtract (`addsub_n_adx`); `aarch64` provides a
//! Comba multiply; everything else routes through `fallback` (pure
//! Rust). Standalone add and subtract stay portable on every target:
//! LLVM lowers the carry chains to `adc`/`sbb` (x86) or `adcs`/`sbcs`
//! (aarch64) as tightly as hand-written asm, so the only hand-written
//! add/sub is the fused two-output pass LLVM will not generate.
//!
//! Callers cfg-dispatch on a per-function basis to the available arch
//! entry — see e.g. `BigInt::mag_mul` (`arch::x86_64::mag_mul_4_adx`
//! for N=4 under `+adx` + `+bmi2`, Comba otherwise) and `BigInt::ct_add`
//! (`arch::x86_64::addsub_n_adx` on x86_64 ADX, the portable `mag_add`
//! + `mag_sub` carry chains otherwise).

#[cfg(target_arch = "x86_64")]
pub(super) mod x86_64;

#[cfg(target_arch = "aarch64")]
pub(super) mod aarch64;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[allow(dead_code)] // empty until called from non-x86_64 / non-aarch64 builds
pub(super) mod fallback;
