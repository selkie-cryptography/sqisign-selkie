//! x86_64 intrinsics backend for [`BigInt`][super::super::BigInt] primitives.
//!
//! Future home for:
//!
//! - **ADX-based `mag_mul`** using [`core::arch::x86_64::_mulx_u64`] for `u64 *
//!   u64 -> u128` without consuming the carry flag, plus
//!   [`core::arch::x86_64::_addcarryx_u64`] / `_addcarry_u64` double-chain
//!   accumulation for the schoolbook product.
//! - **ADX-based Montgomery REDC** (`square_wide`, `reduce_wide`,
//!   `MontReducer::mul`) using the same dual-carry pattern to interleave the
//!   multiply and reduce phases of CIOS.
//! - **ADX-based `mag_add` / `mag_sub`** using `_addcarry_u64` /
//!   `_subborrow_u64`.  The win here is modest (single-carry chain, no
//!   double-accumulator leverage) but mechanical given the same infrastructure.
//!
//! Each implementation should be gated on
//! `cfg(target_feature = "bmi2")` (for `_mulx_u64`) and
//! `cfg(target_feature = "adx")` (for the dual-carry intrinsics).
//! Compile-time enabled targets dispatch directly; runtime detection
//! via `is_x86_feature_detected!("bmi2")` and `..."adx"` is the
//! fallback path for portable binaries.
//!
//! Constant-time: every intrinsic operates on register-width words
//! with no data-dependent control flow, so the ADX path is CT by
//! construction.  Per-platform audit (e.g. via `ctgrind`) is
//! required because microarchitectural side channels in `mulx` /
//! `adcx` / `adox` are CPU-specific.
