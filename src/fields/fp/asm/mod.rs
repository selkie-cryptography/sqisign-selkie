//! Per-architecture asm backends for [`super::Fp`] arithmetic.
//!
//! Each backend implements the same primitives that the operator
//! impls in [`super`] otherwise compute inline. The dispatch is via
//! `cfg(target_arch = ...)` early-return at the top of each operator
//! impl. Property tests in [`super::tests`] exercise the asm path on
//! every CI runner whose `target_arch` matches a backend here, and
//! exercise the inline Rust fallback on every other runner.

#[cfg(target_arch = "aarch64")]
pub(in crate::fields::fp) mod aarch64;
