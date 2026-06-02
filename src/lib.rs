#![doc = include_str!("../README.md")]
// Projective coordinates traditionally use uppercase letters,
// lowercase for affine coordinates.
#![allow(non_snake_case)]
#![allow(clippy::op_ref)]
#![deny(
    missing_docs,
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used
)]
// Tests are allowed to use unwrap() / Option::unwrap() / Result::unwrap();
// production code is not (enforced by the deny above). This must come
// after the deny to override it.
#![cfg_attr(test, allow(clippy::unwrap_used))]
#![warn(rust_2018_idioms, unused_lifetimes, unused_qualifications)]

// Diagnostic eprintln gated on the `SELKIE_TRACE` env var, used in
// place of `#[cfg(test)] eprintln!(...)` to keep CI nextest output
// quiet by default. Set `SELKIE_TRACE=1` to re-enable verbose
// per-iteration progress prints during local debugging. Compiles to
// nothing in non-test builds (the body's `#[cfg(test)]` strips the
// `if` statement).
#[cfg(test)]
macro_rules! selkie_trace {
    ($($arg:tt)*) => {{
        if ::std::env::var_os("SELKIE_TRACE").is_some() {
            ::std::eprintln!($($arg)*);
        }
    }};
}
#[cfg(test)]
pub(crate) use selkie_trace;

// Internal modules: pub(crate) by default, pub with expose-internals.

// NIST-I parameter set constants
#[cfg(not(feature = "expose-internals"))]
pub(crate) mod params;
#[cfg(feature = "expose-internals")]
pub mod params;

/// Thread-local flag for gating L² LLL traces to specific call sites.
/// Set to `true` by `sample_from_ball` around its LLL call so we can
/// byte-diff just that one call against C-ref.
#[cfg(test)]
pub mod l2_trace_active {
    use std::cell::Cell;
    thread_local! {
        static ACTIVE: Cell<bool> = const { Cell::new(false) };
    }
    /// Sets the active flag.
    pub fn set(v: bool) {
        ACTIVE.with(|c| c.set(v));
    }
    /// Reads the active flag.
    pub fn get() -> bool {
        ACTIVE.with(|c| c.get())
    }
}

// Finite field arithmetic (F_p and F_{p^2})
#[cfg(not(feature = "expose-internals"))]
pub(crate) mod fields;
#[cfg(feature = "expose-internals")]
pub mod fields;

// Elliptic curves, points, and isogenies between them
#[cfg(not(feature = "expose-internals"))]
pub(crate) mod curves;
#[cfg(feature = "expose-internals")]
pub mod curves;

// Abelian surfaces and (2,2)-isogenies using theta coordinates
#[cfg(not(feature = "expose-internals"))]
pub(crate) mod surfaces;
#[cfg(feature = "expose-internals")]
pub mod surfaces;

// Quaternion algebra and big integer arithmetic
#[cfg(not(feature = "expose-internals"))]
pub(crate) mod quaternions;
#[cfg(feature = "expose-internals")]
pub mod quaternions;

// Deuring correspondence: ideal ↔ isogeny bridge
#[cfg(not(feature = "expose-internals"))]
pub(crate) mod deuring;
#[cfg(feature = "expose-internals")]
pub mod deuring;

// Always-private modules.

// Challenge hash function
pub(crate) mod hash;

// AES256-CTR-DRBG (SP 800-90A) used by _derand entry points
pub(crate) mod drbg;

// Key types and signatures
#[cfg(not(feature = "expose-internals"))]
pub(crate) mod keys;
#[cfg(feature = "expose-internals")]
pub mod keys;

// Public API.

pub use keys::{Signature, SignatureError, SigningKey, VerifyingKey};
pub use params::{SIGNATURE_BYTES, SIGNING_KEY_BYTES, VERIFYING_KEY_BYTES};
