//! Quaternion algebras and ideals.
//!
//! This module implements the quaternion algebra B_{p,∞} = (-1, -p)_Q
//! and its associated lattice, ideal, and order structures needed for
//! SQIsign key generation and signing.

#[cfg(not(feature = "expose-internals"))]
pub(crate) mod algebra;
#[cfg(feature = "expose-internals")]
pub mod algebra;

#[cfg(not(feature = "expose-internals"))]
pub(crate) mod bigint;
#[cfg(feature = "expose-internals")]
pub mod bigint;

#[cfg(not(feature = "expose-internals"))]
pub(crate) mod ideal;
#[cfg(feature = "expose-internals")]
pub mod ideal;

#[cfg(not(feature = "expose-internals"))]
pub(crate) mod lattice;
#[cfg(feature = "expose-internals")]
pub mod lattice;

#[cfg(not(feature = "expose-internals"))]
pub(crate) mod linear;
#[cfg(feature = "expose-internals")]
pub mod linear;

#[cfg(not(feature = "expose-internals"))]
pub(crate) mod precomputed;
#[cfg(feature = "expose-internals")]
pub mod precomputed;
