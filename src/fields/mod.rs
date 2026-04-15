//! Finite field arithmetic for SQIsign-353.
//!
//! - [`fp`]: The prime field F_p, where p = 5 · 2²⁴⁸ − 1.
//! - [`fp2`]: The quadratic extension F_{p²} = F_p(i), i² = −1.

pub mod fp;
pub mod fp2;
