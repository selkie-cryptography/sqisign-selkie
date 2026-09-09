use proptest::prelude::*;
use subtle::{ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess};

use super::*;
use crate::fields::fp::{FP_ENCODED_BYTES, Fp};

fn arb_fp() -> impl Strategy<Value = Fp> {
    any::<[u8; FP_ENCODED_BYTES]>().prop_map(|mut b| {
        // Keep the value below p (top byte of p is 0x2f).
        b[FP_ENCODED_BYTES - 1] &= 0x0F;
        Fp::from_bytes(&b)
    })
}

fn arb_fp2() -> impl Strategy<Value = Fp2> {
    (arb_fp(), arb_fp()).prop_map(|(a, b)| Fp2::new(a, b))
}

#[test]
fn i_squared_is_minus_one() {
    let i2 = Fp2::I.square();
    assert_eq!(i2, -Fp2::ONE);
}

#[test]
fn conjugate_mul_is_norm() {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let n = &a * &a.conjugate();
    // Should be a real number equal to the norm
    assert_eq!(n.b, Fp::ZERO);
    assert_eq!(n.a, a.norm());
}

#[test]
fn inversion() {
    let a = Fp2::new(Fp::from_small(5), Fp::from_small(13));
    let a_inv = a.invert();
    assert_eq!(&a * &a_inv, Fp2::ONE);
}

#[test]
fn karatsuba_matches_schoolbook() {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let b = Fp2::new(Fp::from_small(11), Fp::from_small(5));
    // (3 + 7i)(11 + 5i) = 33 + 15i + 77i + 35i^2 = 33 - 35 + (15+77)i = -2 +
    // 92i
    let c = &a * &b;
    // Check via from_small arithmetic
    let expected_real = &Fp::from_small(33) - &Fp::from_small(35);
    let expected_imag = &Fp::from_small(15) + &Fp::from_small(77);
    assert_eq!(c.a, expected_real);
    assert_eq!(c.b, expected_imag);
}

#[test]
fn roundtrip_bytes() {
    let a = Fp2::new(Fp::from_small(42), Fp::from_small(99));
    let bytes = a.to_bytes();
    let b = Fp2::from_bytes(&bytes);
    assert_eq!(a, b);
}

#[test]
fn half_doubles_back() {
    let a = Fp2::new(Fp::from_small(7), Fp::from_small(9));
    assert_eq!(&a.half() + &a.half(), a);
}

/// Spec convention: `-1 = i^2` and the root selected is `-i`.
#[test]
fn sqrt_of_minus_one_is_minus_i() {
    assert!(bool::from(Fp2::MINUS_ONE.is_square()));
    assert_eq!(Fp2::MINUS_ONE.sqrt(), -Fp2::I);
}

/// Spec convention: of the two roots `+-(2 + 3i)` of `(2 + 3i)^2`,
/// the one selected is `-2 - 3i`.
#[test]
fn sqrt_of_two_plus_three_i_squared() {
    let a = Fp2::new(Fp::from_small(2), Fp::from_small(3));
    let r = a.square().sqrt();
    assert_eq!(r, -a);
}

/// `5` is a non-residue in `F_p` (`p = 3 mod 4` with `p = 3 mod 5`),
/// so its square root in `F_p2` is purely imaginary.
#[test]
fn sqrt_of_five_is_purely_imaginary() {
    let five = Fp2::from_fp(Fp::from_small(5));
    let r = five.sqrt();
    assert_eq!(r.a, Fp::ZERO);
    assert_eq!(r.square(), five);
}

#[test]
fn batch_invert_matches_invert() {
    let mut elems = [
        Fp2::new(Fp::from_small(1), Fp::from_small(2)),
        Fp2::new(Fp::from_small(3), Fp::from_small(0)),
        Fp2::new(Fp::from_small(0), Fp::from_small(4)),
        Fp2::new(Fp::from_small(5), Fp::from_small(6)),
    ];
    let expected: Vec<Fp2> = elems.iter().map(Fp2::invert).collect();
    Fp2::batch_invert(&mut elems);
    assert_eq!(elems.to_vec(), expected);
}

/// Definition 4.2.1 orders by the imaginary part first: `i` exceeds
/// every element of `F_p`.
#[test]
fn compare_is_imaginary_first() {
    let p_minus_one = Fp2::from_fp(Fp::MINUS_ONE);
    assert!(bool::from(Fp2::I.ct_gt(&p_minus_one)));
    assert!(bool::from(p_minus_one.ct_lt(&Fp2::I)));
    assert!(!bool::from(p_minus_one.ct_gt(&Fp2::I)));
}

/// Equal imaginary parts fall through to the real part.
#[test]
fn compare_breaks_ties_on_real_part() {
    let one_plus_i = &Fp2::ONE + &Fp2::I;
    assert!(bool::from(one_plus_i.ct_gt(&Fp2::I)));
    assert!(!bool::from(Fp2::I.ct_gt(&one_plus_i)));
    assert!(!bool::from(Fp2::I.ct_gt(&Fp2::I)));
    assert!(!bool::from(Fp2::I.ct_lt(&Fp2::I)));
}

/// The order is on plain integers, not Montgomery residues: on the
/// six-limb backends `4R mod p` wraps below `2R mod p`.
#[test]
fn compare_orders_integers_not_montgomery_residues() {
    let (two, four) = (
        Fp2::from_fp(Fp::from_small(2)),
        Fp2::from_fp(Fp::from_small(4)),
    );
    assert!(bool::from(four.ct_gt(&two)));
    assert!(bool::from(two.ct_lt(&four)));
    assert!(bool::from(Fp2::ONE.ct_lt(&Fp2::from_fp(Fp::MINUS_ONE))));
}

proptest! {
    #[test]
    fn prop_fp2_add_commutative(a in arb_fp2(), b in arb_fp2()) {
        prop_assert_eq!(a + b, b + a);
    }

    #[test]
    fn prop_fp2_add_associative(a in arb_fp2(), b in arb_fp2(), c in arb_fp2()) {
        prop_assert_eq!((a + b) + c, a + (b + c));
    }

    #[test]
    fn prop_fp2_add_identity(a in arb_fp2()) {
        prop_assert_eq!(a + Fp2::ZERO, a);
    }

    #[test]
    fn prop_fp2_mul_commutative(a in arb_fp2(), b in arb_fp2()) {
        prop_assert_eq!(a * b, b * a);
    }

    #[test]
    fn prop_fp2_mul_associative(a in arb_fp2(), b in arb_fp2(), c in arb_fp2()) {
        prop_assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn prop_fp2_mul_identity(a in arb_fp2()) {
        prop_assert_eq!(a * Fp2::ONE, a);
    }

    #[test]
    fn prop_fp2_distributive(a in arb_fp2(), b in arb_fp2(), c in arb_fp2()) {
        prop_assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn prop_fp2_sub_is_add_neg(a in arb_fp2(), b in arb_fp2()) {
        prop_assert_eq!(a - b, a + (-b));
    }

    #[test]
    fn prop_fp2_double_neg(a in arb_fp2()) {
        prop_assert_eq!(-(-a), a);
    }

    #[test]
    fn prop_fp2_square_equals_mul(a in arb_fp2()) {
        prop_assert_eq!(a.square(), a * a);
    }

    #[test]
    fn prop_fp2_inversion(a in arb_fp2()) {
        prop_assume!(!bool::from(a.ct_eq(&Fp2::ZERO)));
        prop_assert_eq!(a * a.invert(), Fp2::ONE);
    }

    #[test]
    fn prop_fp2_half(a in arb_fp2()) {
        prop_assert_eq!(a.half() + a.half(), a);
    }

    #[test]
    fn prop_fp2_sqrt_of_square(a in arb_fp2()) {
        let a2 = a.square();
        prop_assert!(bool::from(a2.is_square()));
        let r = a2.sqrt();
        prop_assert!(r == a || r == -a);
    }

    #[test]
    fn prop_fp2_batch_invert(a in arb_fp2(), b in arb_fp2(), c in arb_fp2()) {
        prop_assume!(!bool::from(a.ct_eq(&Fp2::ZERO)));
        prop_assume!(!bool::from(b.ct_eq(&Fp2::ZERO)));
        prop_assume!(!bool::from(c.ct_eq(&Fp2::ZERO)));
        let mut elems = [a, b, c];
        Fp2::batch_invert(&mut elems);
        prop_assert_eq!(elems, [a.invert(), b.invert(), c.invert()]);
    }

    #[test]
    fn prop_fp2_conjugate_mul_is_norm(a in arb_fp2()) {
        // a * conj(a) should be a real element (imaginary part zero).
        let n = a * a.conjugate();
        prop_assert_eq!(n, n.conjugate());
    }

    #[test]
    fn prop_fp2_norm_multiplicative(a in arb_fp2(), b in arb_fp2()) {
        // norm(a*b) == norm(a) * norm(b), where norm = a * conj(a).
        let norm_a = a * a.conjugate();
        let norm_b = b * b.conjugate();
        let norm_ab = (a * b) * (a * b).conjugate();
        prop_assert_eq!(norm_ab, norm_a * norm_b);
    }

    #[test]
    fn prop_fp2_conjugate_anti_automorphism(a in arb_fp2(), b in arb_fp2()) {
        // conj(a * b) == conj(b) * conj(a)
        // (for commutative Fp2 this equals conj(a) * conj(b))
        prop_assert_eq!((a * b).conjugate(), a.conjugate() * b.conjugate());
    }

    /// Strict total order: exactly one of `<`, `=`, `>` holds, and
    /// `<` is the mirror of `>`.
    #[test]
    fn prop_fp2_compare_is_a_strict_total_order(a in arb_fp2(), b in arb_fp2(), c in arb_fp2()) {
        let (lt, gt, eq) = (bool::from(a.ct_lt(&b)), bool::from(a.ct_gt(&b)), bool::from(a.ct_eq(&b)));
        prop_assert_eq!(usize::from(lt) + usize::from(gt) + usize::from(eq), 1);
        prop_assert_eq!(lt, bool::from(b.ct_gt(&a)));
        if bool::from(a.ct_lt(&b)) && bool::from(b.ct_lt(&c)) {
            prop_assert!(bool::from(a.ct_lt(&c)));
        }
    }

    /// Agrees with integer comparison of the canonical coordinates,
    /// imaginary part first.
    #[test]
    fn prop_fp2_compare_is_imaginary_then_real(a in arb_fp2(), b in arb_fp2()) {
        let key = |x: &Fp2| (x.b.to_bytes().iter().rev().copied().collect::<Vec<u8>>(), x.a.to_bytes().iter().rev().copied().collect::<Vec<u8>>());
        prop_assert_eq!(bool::from(a.ct_lt(&b)), key(&a) < key(&b));
    }

    #[test]
    fn prop_fp2_serialization_roundtrip(a in arb_fp2()) {
        let bytes = a.to_bytes();
        let b = Fp2::from_bytes(&bytes);
        prop_assert_eq!(a, b);
    }
}
