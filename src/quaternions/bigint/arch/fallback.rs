//! Portable fallback backend for [`BigInt`][super::super::BigInt] primitives.
//!
//! Active on targets that are neither `x86_64` nor `aarch64` — at
//! the time of writing, that excludes only embedded and esoteric
//! architectures (Cortex-M, RISC-V, ppc, wasm32, …) which are not
//! deployment targets for sqisign-selkie.  The module exists so the
//! `cfg`-dispatch in [`super::mod`] always resolves to a concrete
//! backend.
//!
//! Implementations here mirror the current scalar Rust code that
//! already lives in `bigint::{add, sub, mul, modular}` modules.
//! When the x86_64 / aarch64 backends fill in their intrinsics, the
//! relevant primitive moves to those files; this fallback retains
//! the `u128`-arithmetic version for any target whose codegen
//! does not pattern-match to ADX-equivalent or `umulh`-equivalent
//! instruction sequences.
