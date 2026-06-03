//! aarch64 implementations of [`Fp`][super::super::Fp] arithmetic.
//!
//! Each submodule is a complete `Fp` backend specialised for an aarch64
//! instruction subset.  The arch dispatcher in [`super`] picks one of
//! these via `cfg(sqisign_selkie_arch = "...")` set by `build.rs`.
//!
//! Currently only [`neon`] exists.  Future variants (SVE2 once Apple ships
//! it on a wider product line, or any other AArch64 extension that
//! materially changes the tightest-codegen layout) slot in alongside
//! `neon.rs` here without touching the dispatcher's surface.

pub mod neon;
