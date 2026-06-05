//! Const-bridge tests for [`Fp64::from_limbs`].
//!
//! Verifies that converting `Fp51`'s named constants via the
//! `Fp51` → `Fp64` const-bridge produces the same numerical values
//! as `Fp64`'s own named constants — i.e. the bridge and the C-ref-
//! derived constants agree.

use super::Fp64;
use crate::fields::fp::arch::generic::Fp51;

#[test]
fn from_limbs_matches_zero() {
    assert_eq!(Fp64::from_limbs(Fp51::ZERO.0), Fp64::ZERO);
}

#[test]
fn from_limbs_matches_one() {
    assert_eq!(Fp64::from_limbs(Fp51::ONE.0), Fp64::ONE);
}

#[test]
fn from_limbs_matches_two() {
    assert_eq!(Fp64::from_limbs(Fp51::TWO.0), Fp64::TWO);
}

#[test]
fn from_limbs_matches_four() {
    assert_eq!(Fp64::from_limbs(Fp51::FOUR.0), Fp64::FOUR);
}

#[test]
fn from_limbs_matches_minus_one() {
    assert_eq!(Fp64::from_limbs(Fp51::MINUS_ONE.0), Fp64::MINUS_ONE);
}
