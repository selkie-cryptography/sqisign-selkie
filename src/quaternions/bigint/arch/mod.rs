//! Architecture-specific backends for [`BigInt`][super::BigInt] primitives.
//!
//! Per-target backend selected at compile time.  `x86_64` provides
//! ADX (`asm!` MULX + ADCX/ADOX dual-chain) versions of `mag_mul`;
//! `aarch64` is reserved for future NEON-assisted `umulh`/`umull`
//! chains; everything else routes through `fallback` (pure Rust,
//! mirrors the in-place `bigint::{add,sub,mul,modular}` code).
//!
//! Callers (`mul.rs` etc.) cfg-dispatch on a per-function basis to
//! the available arch entry — see e.g. `BigInt::mag_mul`, which uses
//! `arch::x86_64::mag_mul_4_adx` for N=4 under `+adx` + `+bmi2` and
//! the portable schoolbook otherwise.

#[cfg(target_arch = "x86_64")]
pub(super) mod x86_64;

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)] // empty until NEON implementations land
pub(super) mod aarch64;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[allow(dead_code)] // empty until called from non-x86_64 / non-aarch64 builds
pub(super) mod fallback;
