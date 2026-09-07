use proptest::prelude::*;
use subtle::ConstantTimeEq;

use super::*;
use crate::fields::fp::Fp;

fn arb_fp() -> impl Strategy<Value = Fp> {
    any::<[u8; 32]>().prop_map(|b| Fp::from_bytes(&b))
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

    #[test]
    fn prop_fp2_serialization_roundtrip(a in arb_fp2()) {
        let bytes = a.to_bytes();
        let b = Fp2::from_bytes(&bytes);
        prop_assert_eq!(a, b);
    }
}
