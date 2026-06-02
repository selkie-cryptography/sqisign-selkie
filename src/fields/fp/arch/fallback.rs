//! Portable fallback backend for [`Fp`][super::super::Fp] arithmetic.
//!
//! Active on targets other than `aarch64`.  On `x86_64` this is
//! the production path today, since AVX-512-IFMA52 (the natural
//! vectorisation for our radix-51 layout) is outside sqisign-selkie's
//! target set; see [`super`][parent module] for the rationale.
//!
//! [parent module]: super
//!
//! Implementations mirror the current scalar Rust code that lives
//! in `fields::fp::mod`: radix-51 packed limbs, schoolbook
//! multiplication with interleaved Montgomery reduction via the
//! `P4 = 5·2^44` constant.  This is already at the limit of scalar
//! Rust for `p = 5·2^248 − 1`; further wins on `x86_64` come from
//! ADX on the *quaternion-side* `BigInt` primitives
//! ([`crate::quaternions::bigint::arch::x86_64`]), not from `Fp`.
