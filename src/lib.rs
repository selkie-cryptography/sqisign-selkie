//! Selkie: pure-Rust supersingular isogeny cryptography.

// Projective coordinates traditionally use uppercase letters,
// lowercase for affine coordinates, and uppercase for curve equation
// coefficients.
#![allow(non_snake_case)]
// #![doc = include_str!("../README.md")]

// Used for isogenies, still requires nightly (as of 2023-04-02)
#![feature(associated_type_defaults)]
#![feature(fn_traits)]
#![feature(generic_const_exprs)]
#![feature(unboxed_closures)]

// TODO: proptest correctness against crypto-bigint(P434), arkworks::ff

pub mod curve;
pub mod field;
pub mod isogeny;
