//! Selkie SQIsign: Rust implementation of the SQIsign isogeny-based
//! signature scheme.
//!
//! Implements [SQIsign-353] targeting the NIST-I parameter set where
//! p = 5·2²⁴⁸ − 1) as specified in the [v2.0.1 revision][spec].
//!
//! [spec]: https://sqisign.org/spec/sqisign-20250707.pdf

// Projective coordinates traditionally use uppercase letters,
// lowercase for affine coordinates.
#![allow(non_snake_case)]
#![deny(missing_docs, clippy::unwrap_used)]
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

// Key types and signatures
pub mod keys;

pub use keys::{Signature, SigningKey, VerifyingKey};
pub use keys::SignatureError;
pub use params::{SIGNATURE_BYTES, SIGNING_KEY_BYTES, VERIFYING_KEY_BYTES};
