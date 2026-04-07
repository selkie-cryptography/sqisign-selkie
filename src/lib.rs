//! Selkie SQIsign: Rust implementation of the SQIsign isogeny-based
//! signature scheme.
//!
//! Implements [SQIsign-353] targeting the NIST-I parameter set where
//! p = 5·2²⁴⁸ − 1) as specified in the [v2.0.1 revision][spec].
//!
//! # Implementation notes
//!
//! This implementation follows the spec directly wherever possible.
//! Where the spec is ambiguous or silent, the [C reference][cref]
//! is used as ground truth. Key divergences:
//!
//! - **Fp2 sqrt canonicalization** ([`fields::fp2`]): the spec does not specify
//!   which root to return; the C ref canonicalizes to even real part. Critical
//!   for deterministic `projective_difference`.
//! - **Torsion basis slot convention** ([`curves`]): the C ref swaps Q and P−Q
//!   in basis slots so the ladder computes `P + [m](P−Q)`.
//! - **Jacobian coordinates for gluing** ([`surfaces`]): the spec describes
//!   only x-only arithmetic; the C ref uses Jacobian doubling in the
//!   (2,2)-isogeny gluing to get a specific projective representative. Our impl
//!   matches the C ref.
//! - **Fixed-precision quaternion arithmetic** ([`quaternions`]): the C ref
//!   uses GMP (arbitrary precision); we use fixed-width `BigInt<N>` with N=110
//!   limbs (7,040 bits) as the worst-case bound proven by [Kim et al.][kim]
//!   (ePrint 2025/1649).
//! - **Type-safe scalars**: `Challenge`, `Scalar`, `TorsionExponent` are
//!   newtypes enforcing domain constraints at compile time.
//!
//! See `latex/spec-review.tex` for full documentation of spec gaps
//! and C reference conventions.
//!
//! [spec]: https://sqisign.org/spec/sqisign-20250707.pdf
//! [cref]: https://github.com/SQISign/the-sqisign
//! [kim]: https://eprint.iacr.org/2025/1649

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

// Key types and signatures
pub mod keys;

pub use keys::{Signature, SignatureError, SigningKey, VerifyingKey};
pub use params::{SIGNATURE_BYTES, SIGNING_KEY_BYTES, VERIFYING_KEY_BYTES};
