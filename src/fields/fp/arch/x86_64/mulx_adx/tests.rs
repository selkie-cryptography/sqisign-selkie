//! `Fp64` scaffold tests: const-bridge from `Fp51`'s limbs, and the
//! constant-time-select trait impl.

use proptest::prelude::*;
use subtle::{Choice, ConditionallySelectable};

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

#[test]
fn conditional_select_picks_a_when_false() {
    let a = Fp64::from_raw([1, 2, 3, 4]);
    let b = Fp64::from_raw([10, 20, 30, 40]);
    let r = Fp64::conditional_select(&a, &b, Choice::from(0));
    assert_eq!(r, a);
}

#[test]
fn conditional_select_picks_b_when_true() {
    let a = Fp64::from_raw([1, 2, 3, 4]);
    let b = Fp64::from_raw([10, 20, 30, 40]);
    let r = Fp64::conditional_select(&a, &b, Choice::from(1));
    assert_eq!(r, b);
}

proptest! {
    /// `conditional_select(a, b, choice)` agrees with the trivial
    /// branchy choice on arbitrary inputs.
    #[test]
    fn conditional_select_matches_branchy(
        a_limbs in any::<[u64; 4]>(),
        b_limbs in any::<[u64; 4]>(),
        c in any::<bool>(),
    ) {
        let a = Fp64::from_raw(a_limbs);
        let b = Fp64::from_raw(b_limbs);
        let r = Fp64::conditional_select(&a, &b, Choice::from(c as u8));
        let expected = if c { b } else { a };
        prop_assert_eq!(r, expected);
    }
}
