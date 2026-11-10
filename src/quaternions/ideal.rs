//! Ideal algorithms for SQIsign key generation and signing.
//!
//! Algorithms that operate across multiple quaternion types (elements,
//! orders, ideals) and don't naturally belong to a single type.
//!
//! - [`ExtremalOrder::represent_integer`]: find γ ∈ O with nrd(γ) = M ([Alg.
//!   3.12])
//! - [`ExtremalOrder::represent_integer_any`]: same, trying all precomputed
//!   orders
//! - [`LeftIdeal::smallest_equiv_with_delta`]: find J ∼ I with smaller norm
//!   (used during [Alg. 3.9])
//!
//! `RandomEquivalentPrimeIdeal` ([Alg. 3.9]) is defined as
//! [`LeftIdeal<8>::reduce_to_prime_norm`] in the lattice module.
//!
//! [`ExtremalOrder::represent_integer`]: super::lattice::ExtremalOrder::represent_integer
//! [`ExtremalOrder::represent_integer_any`]: super::lattice::ExtremalOrder::represent_integer_any
//! [`LeftIdeal::smallest_equiv_with_delta`]: super::lattice::LeftIdeal::smallest_equiv_with_delta
//! [`LeftIdeal<8>::reduce_to_prime_norm`]: super::lattice::LeftIdeal::reduce_to_prime_norm
//! [Alg. 3.9]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.9
//! [Alg. 3.12]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.12

mod represent_integer;
mod smallest_equiv;

#[cfg(test)]
mod tests;
