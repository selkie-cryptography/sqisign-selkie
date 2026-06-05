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

/// Maps `Fp51`'s known constants through the `from_limbs` bridge.
/// Sources canonical `Fp64` values without needing Fp64-native byte
/// I/O (which lands once `fp_mul` is in).
fn canon(fp51: Fp51) -> Fp64 {
    Fp64::from_limbs(fp51.0)
}

#[test]
fn add_identity_zero() {
    let one = canon(Fp51::ONE);
    assert_eq!(one + Fp64::ZERO, one);
    assert_eq!(Fp64::ZERO + one, one);
}

#[test]
fn add_one_one_is_two() {
    let one = canon(Fp51::ONE);
    let two = canon(Fp51::TWO);
    assert_eq!(one + one, two);
}

#[test]
fn add_two_two_is_four() {
    let two = canon(Fp51::TWO);
    let four = canon(Fp51::FOUR);
    assert_eq!(two + two, four);
}

#[test]
fn sub_self_is_zero() {
    let one = canon(Fp51::ONE);
    assert_eq!(one - one, Fp64::ZERO);
}

#[test]
fn sub_zero_one_is_minus_one() {
    let one = canon(Fp51::ONE);
    let minus_one = canon(Fp51::MINUS_ONE);
    assert_eq!(Fp64::ZERO - one, minus_one);
}

#[test]
fn neg_zero_is_zero() {
    assert_eq!(-Fp64::ZERO, Fp64::ZERO);
}

#[test]
fn neg_one_is_minus_one() {
    let one = canon(Fp51::ONE);
    let minus_one = canon(Fp51::MINUS_ONE);
    assert_eq!(-one, minus_one);
}

#[test]
fn neg_neg_is_identity() {
    let one = canon(Fp51::ONE);
    assert_eq!(-(-one), one);
}

#[test]
fn add_minus_one_one_is_zero() {
    let one = canon(Fp51::ONE);
    let minus_one = canon(Fp51::MINUS_ONE);
    assert_eq!(one + minus_one, Fp64::ZERO);
}

/// Generate a canonical `Fp64` value through the `Fp51`-bridge chain:
/// random bytes -> Fp51 (canonical, in Mont form) -> Fp64 via bridge.
fn arb_fp64() -> impl Strategy<Value = Fp64> {
    any::<[u8; 32]>().prop_map(|bytes| {
        let fp51 = Fp51::from_bytes(&bytes);
        Fp64::from_limbs(fp51.0)
    })
}

proptest! {
    /// `a + b == b + a`.
    #[test]
    fn add_commutes(a in arb_fp64(), b in arb_fp64()) {
        prop_assert_eq!(&a + &b, &b + &a);
    }

    /// `a + 0 == a`.
    #[test]
    fn add_identity(a in arb_fp64()) {
        prop_assert_eq!(&a + &Fp64::ZERO, a);
    }

    /// `a + (-a) == 0`.
    #[test]
    fn add_inverse(a in arb_fp64()) {
        prop_assert_eq!(&a + &(-&a), Fp64::ZERO);
    }

    /// `(a + b) + c == a + (b + c)`.
    #[test]
    fn add_associates(a in arb_fp64(), b in arb_fp64(), c in arb_fp64()) {
        let lhs = &(&a + &b) + &c;
        let rhs = &a + &(&b + &c);
        prop_assert_eq!(lhs, rhs);
    }

    /// `a - a == 0`.
    #[test]
    fn sub_self(a in arb_fp64()) {
        prop_assert_eq!(&a - &a, Fp64::ZERO);
    }

    /// `a - b == a + (-b)`.
    #[test]
    fn sub_via_neg(a in arb_fp64(), b in arb_fp64()) {
        prop_assert_eq!(&a - &b, &a + &(-&b));
    }

    /// `-(-a) == a`.
    #[test]
    fn neg_involutes(a in arb_fp64()) {
        prop_assert_eq!(-(-&a), a);
    }
}
