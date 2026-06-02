//! Architecture-specific backends for [`BigInt`][super::BigInt] primitives.
//!
//! Skeleton module that re-exports the active backend.  The
//! backend is selected at compile time by the target architecture:
//!
//! - `x86_64`: see file `arch/x86_64.rs`.  Future home for ADX (`_mulx_u64`,
//!   `_addcarry_u64`, `_addcarryx_u64`) implementations of the `mag_mul` /
//!   `mag_add` / `mag_sub` / Montgomery REDC primitives.
//! - `aarch64`: see file `arch/aarch64.rs`.  Future home for NEON (`umulh` /
//!   `umull` + carry chains) implementations of the same primitives.
//! - any other target: see file `arch/fallback.rs`.  Pure-Rust `u128`-based
//!   implementations equivalent to the current scalar code.
//!
//! # Current state
//!
//! All three backends are empty — they exist only to anchor the
//! `cfg`-dispatch pattern.  Future PRs filling in arch-specific
//! intrinsics add their bodies to the matching file behind feature
//! detection (`cfg(target_feature = "bmi2")` for ADX,
//! `cfg(target_feature = "neon")` for NEON).
//!
//! Callers do not import these submodules directly.  When the
//! arch-specific primitives are filled in, this `mod.rs` will
//! re-export the active backend's API so the rest of `bigint`
//! sees a single namespace.
//!
//! # Why the skeleton lands first
//!
//! Both ADX (x86_64) and NEON (aarch64) work proceeds in parallel
//! on separate branches; landing the dispatch skeleton first means
//! both branches share one merge target and one cfg convention,
//! and the rest of the `bigint` module never sees more than one
//! arch-specific code path at a time.

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // empty until ADX implementations land
pub(super) mod x86_64;

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)] // empty until NEON implementations land
pub(super) mod aarch64;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[allow(dead_code)] // empty until called from non-x86_64 / non-aarch64 builds
pub(super) mod fallback;
