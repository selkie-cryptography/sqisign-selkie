#![doc = include_str!("../README.md")]
// Projective coordinates traditionally use uppercase letters,
// lowercase for affine coordinates.
#![allow(non_snake_case)]
#![allow(clippy::op_ref)]
// Many items are work-in-progress (sign() and keygen are not yet wired
// up end-to-end). Re-enable dead_code lints once the signing flow is
// complete and unused infrastructure has been pruned.
#![allow(dead_code)]
#![deny(missing_docs, clippy::unwrap_used)]
// Tests are allowed to use unwrap() / Option::unwrap() / Result::unwrap();
// production code is not (enforced by the deny above). This must come
// after the deny to override it.
#![cfg_attr(test, allow(clippy::unwrap_used))]
#![warn(rust_2018_idioms, unused_lifetimes, unused_qualifications)]

// --- Internal modules (pub(crate) by default, pub with expose-internals) ---

// NIST-I parameter set constants
#[cfg(not(feature = "expose-internals"))]
pub(crate) mod params;
#[cfg(feature = "expose-internals")]
pub mod params;

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

// --- Always-private modules ---

// Challenge hash function
pub(crate) mod hash;

// AES256-CTR-DRBG (SP 800-90A) used by _derand entry points
pub(crate) mod drbg;

// Key types and signatures
#[cfg(not(feature = "expose-internals"))]
pub(crate) mod keys;
#[cfg(feature = "expose-internals")]
pub mod keys;

// --- Public API ---

pub use keys::{Signature, SignatureError, SigningKey, VerifyingKey};
pub use params::{SIGNATURE_BYTES, SIGNING_KEY_BYTES, VERIFYING_KEY_BYTES};
