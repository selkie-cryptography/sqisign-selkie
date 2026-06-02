//! aarch64 NEON backend for [`BigInt`][super::super::BigInt] primitives.
//!
//! Future home for:
//!
//! - **NEON-assisted `mag_mul`** using `umulh` + `mul` to get the high and low
//!   halves of a `u64 × u64` product separately, then accumulating via paired
//!   `adcs`/`adds` carry chains.  The absence of x86's dual-accumulator
//!   (`adcx`/`adox`) instruction pair caps the win below the ADX number, but
//!   Apple Silicon's superscalar dispatch absorbs enough of the carry-chain
//!   serialization that ~10–20% on `mag_mul` is realistic.
//! - **NEON-assisted Montgomery REDC** following the same pattern.
//! - **NEON-assisted `mag_add` / `mag_sub`** using `adcs`/`sbcs` loops.  LLVM
//!   already lowers the existing `overflowing_add` chain to these on `aarch64`;
//!   the win here is mostly from pinning the loop structure rather than from
//!   new intrinsics.
//!
//! Each implementation is gated on
//! `cfg(target_feature = "neon")`, which is always enabled on
//! `aarch64` ABI targets (it is part of the base ISA, not an
//! extension on the Apple Silicon / Cortex-A series we care about).
//!
//! Constant-time: same posture as the x86_64 backend — every
//! intrinsic is data-flow-only, but microarchitectural CT must be
//! verified per Apple-Silicon revision (M1–M4 each have slightly
//! different multiplier latencies).
//!
//! # Note on field arithmetic
//!
//! The much larger NEON-on-aarch64 lever for SQIsign is in
//! [`crate::fields::fp`], not here.  Per De Feo et al. 2026/394,
//! a vectorised radix-29 `Fp` is the ~1.22× speedup on Apple M1.
//! That backend lives at `crate::fields::fp::arch::aarch64`; this
//! module covers only the quaternion-side `BigInt` primitives.
