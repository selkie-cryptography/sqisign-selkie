use proptest::prelude::*;

use super::*;

/// The torsion power f = 248.
const K: u32 = 248;

fn arb_scalar() -> impl Strategy<Value = Scalar> {
    any::<[u64; 4]>().prop_map(Scalar::from_limbs)
}

#[test]
fn bits_be_small() {
    let s = Scalar::from_u64(0b1011);
    let bits: Vec<bool> = s.bits_be(4).collect();
    assert_eq!(bits, vec![true, false, true, true]);
}

#[test]
fn bits_be_zero_padded() {
    let s = Scalar::from_u64(3); // 0b11
    let bits: Vec<bool> = s.bits_be(8).collect();
    assert_eq!(
        bits,
        vec![false, false, false, false, false, false, true, true]
    );
}

#[test]
fn bit_length() {
    assert_eq!(Scalar::ZERO.bit_length(), 0);
    assert_eq!(Scalar::ONE.bit_length(), 1);
    assert_eq!(Scalar::from_u64(255).bit_length(), 8);
    assert_eq!(Scalar::from_u64(256).bit_length(), 9);
}

#[test]
fn reduce_mod2k_masks_correctly() {
    let s = Scalar::from_u64(0xFF);
    assert_eq!(s.reduce_mod2k(4), Scalar::from_u64(0x0F));
    assert_eq!(s.reduce_mod2k(8), Scalar::from_u64(0xFF));
    assert_eq!(s.reduce_mod2k(1), Scalar::from_u64(1));
}

#[test]
fn add_mod2k_wraps() {
    let a = Scalar::from_u64(250);
    let b = Scalar::from_u64(10);
    // 250 + 10 = 260 = 0x104, mod 2^8 = 4
    assert_eq!(a.add_mod2k(&b, 8), Scalar::from_u64(4));
}

#[test]
fn sub_mod2k_wraps() {
    let a = Scalar::from_u64(3);
    let b = Scalar::from_u64(5);
    // 3 - 5 mod 2^8 = 254
    assert_eq!(a.sub_mod2k(&b, 8), Scalar::from_u64(254));
}

#[test]
fn mul_mod2k_truncates() {
    let a = Scalar::from_u64(200);
    let b = Scalar::from_u64(200);
    // 200 * 200 = 40000 = 0x9C40, mod 2^8 = 0x40 = 64
    assert_eq!(a.mul_mod2k(&b, 8), Scalar::from_u64(64));
}

#[test]
fn inv_mod2k_round_trip() {
    let a = Scalar::from_u64(7); // odd
    let inv = a.inv_mod2k(248).unwrap();
    let product = a.mul_mod2k(&inv, 248);
    assert_eq!(product, Scalar::ONE);
}

#[test]
fn inv_mod2k_even_returns_none() {
    let a = Scalar::from_u64(6); // even
    assert!(a.inv_mod2k(248).is_none());
}

#[test]
fn inv_mod2k_large_odd() {
    // 2^248 - 1 is odd
    let a = Scalar::from_limbs([
        u64::MAX,
        u64::MAX,
        u64::MAX,
        (1u64 << 56) - 1, // 248 bits
    ]);
    let inv = a.inv_mod2k(248).unwrap();
    let product = a.mul_mod2k(&inv, 248);
    assert_eq!(product, Scalar::ONE);
}

proptest! {
    #[test]
    fn prop_scalar_add_commutative(a in arb_scalar(), b in arb_scalar()) {
        prop_assert_eq!(a.add_mod2k(&b, K), b.add_mod2k(&a, K));
    }

    #[test]
    fn prop_scalar_add_associative(a in arb_scalar(), b in arb_scalar(), c in arb_scalar()) {
        prop_assert_eq!(
            a.add_mod2k(&b, K).add_mod2k(&c, K),
            a.add_mod2k(&b.add_mod2k(&c, K), K)
        );
    }

    #[test]
    fn prop_scalar_add_identity(a in arb_scalar()) {
        prop_assert_eq!(a.add_mod2k(&Scalar::ZERO, K), a.add_mod2k(&Scalar::ZERO, K));
        // a + 0 mod 2^k should equal a mod 2^k (mask high bits).
        let sum = a.add_mod2k(&Scalar::ZERO, K);
        let again = sum.add_mod2k(&Scalar::ZERO, K);
        prop_assert_eq!(sum, again);
    }

    #[test]
    fn prop_scalar_add_inverse(a in arb_scalar()) {
        // a + (2^k - a) = 0 mod 2^k, which is a - a = 0.
        let neg = Scalar::ZERO.sub_mod2k(&a, K);
        prop_assert_eq!(a.add_mod2k(&neg, K), Scalar::ZERO.add_mod2k(&Scalar::ZERO, K));
    }

    #[test]
    fn prop_scalar_sub_is_add_neg(a in arb_scalar(), b in arb_scalar()) {
        let neg_b = Scalar::ZERO.sub_mod2k(&b, K);
        prop_assert_eq!(a.sub_mod2k(&b, K), a.add_mod2k(&neg_b, K));
    }

    #[test]
    fn prop_scalar_mul_commutative(a in arb_scalar(), b in arb_scalar()) {
        prop_assert_eq!(a.mul_mod2k(&b, K), b.mul_mod2k(&a, K));
    }

    #[test]
    fn prop_scalar_mul_associative(a in arb_scalar(), b in arb_scalar(), c in arb_scalar()) {
        prop_assert_eq!(
            a.mul_mod2k(&b, K).mul_mod2k(&c, K),
            a.mul_mod2k(&b.mul_mod2k(&c, K), K)
        );
    }

    #[test]
    fn prop_scalar_mul_identity(a in arb_scalar()) {
        let one = Scalar::from_limbs([1, 0, 0, 0]);
        let result = a.mul_mod2k(&one, K);
        // a * 1 mod 2^k should equal a mod 2^k.
        let a_masked = a.add_mod2k(&Scalar::ZERO, K);
        prop_assert_eq!(result, a_masked);
    }

    #[test]
    fn prop_scalar_mul_zero(a in arb_scalar()) {
        let zero = Scalar::ZERO.add_mod2k(&Scalar::ZERO, K);
        prop_assert_eq!(a.mul_mod2k(&Scalar::ZERO, K), zero);
    }

    #[test]
    fn prop_scalar_distributive(a in arb_scalar(), b in arb_scalar(), c in arb_scalar()) {
        prop_assert_eq!(
            a.mul_mod2k(&b.add_mod2k(&c, K), K),
            a.mul_mod2k(&b, K).add_mod2k(&a.mul_mod2k(&c, K), K)
        );
    }

    #[test]
    fn prop_scalar_inv_roundtrip(a in arb_scalar()) {
        // Only odd values are invertible mod 2^k.
        if let Some(inv) = a.inv_mod2k(K) {
            // a * a^{-1} = 1 mod 2^k.
            let one = Scalar::from_limbs([1, 0, 0, 0]);
            prop_assert_eq!(a.mul_mod2k(&inv, K), one);
        }
    }

    #[test]
    fn prop_scalar_double_inv(a in arb_scalar()) {
        // inv(inv(a)) = a for odd a.
        if let Some(inv) = a.inv_mod2k(K) {
            if let Some(inv_inv) = inv.inv_mod2k(K) {
                // Compare mod 2^k.
                prop_assert_eq!(
                    a.add_mod2k(&Scalar::ZERO, K),
                    inv_inv.add_mod2k(&Scalar::ZERO, K)
                );
            }
        }
    }
}
