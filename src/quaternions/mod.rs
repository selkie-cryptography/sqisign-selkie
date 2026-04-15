//! Quaternion algebras and ideals.
//!
//! This module implements the quaternion algebra B_{p,∞} = (-1, -p)_Q
//! and its associated lattice, ideal, and order structures needed for
//! SQIsign key generation and signing.

#[cfg(not(feature = "bench-internals"))]
pub(crate) mod algebra;
#[cfg(feature = "bench-internals")]
pub mod algebra;

#[cfg(not(feature = "bench-internals"))]
pub(crate) mod bigint;
#[cfg(feature = "bench-internals")]
pub mod bigint;

#[cfg(not(feature = "bench-internals"))]
pub(crate) mod ideal;
#[cfg(feature = "bench-internals")]
pub mod ideal;

#[cfg(not(feature = "bench-internals"))]
pub(crate) mod lattice;
#[cfg(feature = "bench-internals")]
pub mod lattice;

#[cfg(not(feature = "bench-internals"))]
pub(crate) mod linear;
#[cfg(feature = "bench-internals")]
pub mod linear;

#[cfg(not(feature = "bench-internals"))]
pub(crate) mod precomputed;
#[cfg(feature = "bench-internals")]
pub mod precomputed;
