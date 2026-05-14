//! 4-element integer vectors and 4×4 integer matrices over [`BigInt`].
//!
//! These types represent the fixed-dimension vectors and matrices used
//! in quaternion lattice arithmetic: basis matrices, coordinate vectors,
//! and Gram matrices. All dimensions are 4, matching the rank of the
//! quaternion algebra B_{p,∞}.
//!
//! [`BigInt`]: super::bigint::BigInt

mod matrix;
mod vector;

pub use matrix::Matrix;
pub use vector::Vector;

#[cfg(test)]
mod tests;
