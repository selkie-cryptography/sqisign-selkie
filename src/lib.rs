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

// NIST-I parameter set constants
pub mod params;

// Finite field arithmetic (F_p and F_{p^2})
pub mod fields;

// Elliptic curves, points, and isogenies between them
pub mod curves;

// Abelian surfaces and (2,2)-isogenies using theta coordinates
pub mod surfaces;

// Challenge hash function
pub(crate) mod hash;

// Quaternion algebra and big integer arithmetic
pub(crate) mod quaternions;

// Deuring correspondence: ideal ↔ isogeny bridge
pub(crate) mod deuring;

// AES256-CTR-DRBG (SP 800-90A) used by _derand entry points
pub(crate) mod drbg;

// Key types and signatures
pub mod keys;

pub use keys::{Signature, SignatureError, SigningKey, VerifyingKey};
pub use params::{SIGNATURE_BYTES, SIGNING_KEY_BYTES, VERIFYING_KEY_BYTES};
