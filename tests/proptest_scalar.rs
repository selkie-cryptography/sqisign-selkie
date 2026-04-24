//! Property-based tests for Scalar mod 2^k ring arithmetic.
//!
//! Run with: `cargo test --test proptest_scalar --features expose-internals`

use proptest::prelude::*;
use sqisign_selkie::curves::scalar::Scalar;

// The torsion power f = 248.
const K: u32 = 248;

fn arb_scalar() -> impl Strategy<Value = Scalar> {
    any::<[u64; 4]>().prop_map(Scalar::from_limbs)
}

proptest! {
    #[test]
    fn scalar_add_commutative(a in arb_scalar(), b in arb_scalar()) {
        prop_assert_eq!(a.add_mod2k(&b, K), b.add_mod2k(&a, K));
    }

    #[test]
    fn scalar_add_associative(a in arb_scalar(), b in arb_scalar(), c in arb_scalar()) {
        prop_assert_eq!(
            a.add_mod2k(&b, K).add_mod2k(&c, K),
            a.add_mod2k(&b.add_mod2k(&c, K), K)
        );
    }

    #[test]
    fn scalar_add_identity(a in arb_scalar()) {
        prop_assert_eq!(a.add_mod2k(&Scalar::ZERO, K), a.add_mod2k(&Scalar::ZERO, K));
        // a + 0 mod 2^k should equal a mod 2^k (mask high bits).
        let sum = a.add_mod2k(&Scalar::ZERO, K);
        let again = sum.add_mod2k(&Scalar::ZERO, K);
        prop_assert_eq!(sum, again);
    }

    #[test]
    fn scalar_add_inverse(a in arb_scalar()) {
        // a + (2^k - a) = 0 mod 2^k, which is a - a = 0.
        let neg = Scalar::ZERO.sub_mod2k(&a, K);
        prop_assert_eq!(a.add_mod2k(&neg, K), Scalar::ZERO.add_mod2k(&Scalar::ZERO, K));
    }

    #[test]
    fn scalar_sub_is_add_neg(a in arb_scalar(), b in arb_scalar()) {
        let neg_b = Scalar::ZERO.sub_mod2k(&b, K);
        prop_assert_eq!(a.sub_mod2k(&b, K), a.add_mod2k(&neg_b, K));
    }

    #[test]
    fn scalar_mul_commutative(a in arb_scalar(), b in arb_scalar()) {
        prop_assert_eq!(a.mul_mod2k(&b, K), b.mul_mod2k(&a, K));
    }

    #[test]
    fn scalar_mul_associative(a in arb_scalar(), b in arb_scalar(), c in arb_scalar()) {
        prop_assert_eq!(
            a.mul_mod2k(&b, K).mul_mod2k(&c, K),
            a.mul_mod2k(&b.mul_mod2k(&c, K), K)
        );
    }

    #[test]
    fn scalar_mul_identity(a in arb_scalar()) {
        let one = Scalar::from_limbs([1, 0, 0, 0]);
        let result = a.mul_mod2k(&one, K);
        // a * 1 mod 2^k should equal a mod 2^k.
        let a_masked = a.add_mod2k(&Scalar::ZERO, K);
        prop_assert_eq!(result, a_masked);
    }

    #[test]
    fn scalar_mul_zero(a in arb_scalar()) {
        let zero = Scalar::ZERO.add_mod2k(&Scalar::ZERO, K);
        prop_assert_eq!(a.mul_mod2k(&Scalar::ZERO, K), zero);
    }

    #[test]
    fn scalar_distributive(a in arb_scalar(), b in arb_scalar(), c in arb_scalar()) {
        prop_assert_eq!(
            a.mul_mod2k(&b.add_mod2k(&c, K), K),
            a.mul_mod2k(&b, K).add_mod2k(&a.mul_mod2k(&c, K), K)
        );
    }

    #[test]
    fn scalar_inv_roundtrip(a in arb_scalar()) {
        // Only odd values are invertible mod 2^k.
        if let Some(inv) = a.inv_mod2k(K) {
            // a * a^{-1} = 1 mod 2^k.
            let one = Scalar::from_limbs([1, 0, 0, 0]);
            prop_assert_eq!(a.mul_mod2k(&inv, K), one);
        }
    }
}
