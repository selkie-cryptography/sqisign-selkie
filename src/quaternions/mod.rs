//! Quaternion algebras and ideals.
//!
//! This module implements the quaternion algebra B_{p,∞} = (-1, -p)_Q
//! and its associated lattice, ideal, and order structures needed for
//! SQIsign key generation and signing.

pub(crate) mod algebra;
pub(crate) mod bigint;
pub(crate) mod ideal;
pub(crate) mod lattice;
pub(crate) mod linear;
pub(crate) mod precomputed;
