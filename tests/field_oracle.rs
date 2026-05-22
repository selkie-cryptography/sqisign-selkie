// reason: `Fp` / `Fp2` only impl operators on `&` references (see
// `fields::fp2::mod::impl Add for &Fp2`); the crate root carries the
// same allow. Tests don't inherit it, so re-apply at file scope.
#![allow(clippy::op_ref)]
//! Differential proptests for `Fp` and `Fp2` against `num-bigint`.
//!
//! The reference computes the result as an unbounded `num_bigint`
//! value, reduces it mod `p = 5·2²⁴⁸ − 1` (or, for `Fp2`, performs
//! the `(a, b)`-pair arithmetic with `i² = −1` and reduces each
//! coordinate mod `p`), then encodes the reduced value as 32 (or
//! 64) little-endian bytes. The same input bytes go through `Fp` /
//! `Fp2`'s own `from_bytes` → op → `to_bytes` path. Equality on the
//! encoded bytes is the property under test.
//!
//! Both sides see *canonical* (`< p`) inputs: random 32-byte vectors
//! are reduced mod `p` before being handed to `Fp::from_bytes`, since
//! the decoder's contract demands canonical inputs.

use num_bigint::{BigInt as NumBigInt, Sign};
use num_traits::{One, Signed, Zero};
use proptest::prelude::*;
use sqisign_selkie::{
    fields::{fp::Fp, fp2::Fp2},
    params::{FP_ENCODED_BYTES, FP2_ENCODED_BYTES},
};

/// `p = 5 · 2²⁴⁸ − 1`, the base prime, as an unbounded integer.
fn p() -> NumBigInt {
    (NumBigInt::from(5u32) << 248) - NumBigInt::one()
}

/// Lifts 32 LE bytes into a non-negative `num_bigint::BigInt`.
fn bytes_to_num(bytes: &[u8]) -> NumBigInt {
    NumBigInt::from_bytes_le(Sign::Plus, bytes)
}

/// Reduces an unbounded integer to `[0, p)` and encodes it as 32 LE
/// bytes — the canonical wire form `Fp::from_bytes` expects.
fn num_to_fp_bytes(n: &NumBigInt) -> [u8; FP_ENCODED_BYTES] {
    let p = p();
    let mut r = n % &p;
    if r.is_negative() {
        r += &p;
    }
    let (_, mag) = r.to_bytes_le();
    let mut out = [0u8; FP_ENCODED_BYTES];
    out[..mag.len()].copy_from_slice(&mag);
    out
}

/// Reduces 32 input bytes to a canonical Fp encoding by interpreting
/// them as an unsigned integer and taking mod p.
fn canon_fp_bytes(bytes: &[u8; FP_ENCODED_BYTES]) -> [u8; FP_ENCODED_BYTES] {
    num_to_fp_bytes(&bytes_to_num(bytes))
}

/// Splits 64 input bytes into two canonical Fp encodings (real, imag).
fn canon_fp2_bytes(
    bytes: &[u8; FP2_ENCODED_BYTES],
) -> ([u8; FP_ENCODED_BYTES], [u8; FP_ENCODED_BYTES]) {
    let mut a = [0u8; FP_ENCODED_BYTES];
    let mut b = [0u8; FP_ENCODED_BYTES];
    a.copy_from_slice(&bytes[..FP_ENCODED_BYTES]);
    b.copy_from_slice(&bytes[FP_ENCODED_BYTES..]);
    (canon_fp_bytes(&a), canon_fp_bytes(&b))
}

/// Reassembles two canonical Fp encodings into a 64-byte Fp2 encoding.
fn join_fp2_bytes(
    a: &[u8; FP_ENCODED_BYTES],
    b: &[u8; FP_ENCODED_BYTES],
) -> [u8; FP2_ENCODED_BYTES] {
    let mut out = [0u8; FP2_ENCODED_BYTES];
    out[..FP_ENCODED_BYTES].copy_from_slice(a);
    out[FP_ENCODED_BYTES..].copy_from_slice(b);
    out
}

proptest! {
    #[test]
    fn oracle_fp_roundtrip(bytes in any::<[u8; FP_ENCODED_BYTES]>()) {
        let canon = canon_fp_bytes(&bytes);
        let element = Fp::from_bytes(&canon);
        prop_assert_eq!(element.to_bytes(), canon);
    }

    #[test]
    fn oracle_fp_add(ab in any::<[u8; FP2_ENCODED_BYTES]>()) {
        let (a_bytes, b_bytes) = canon_fp2_bytes(&ab);
        let a_ref = bytes_to_num(&a_bytes);
        let b_ref = bytes_to_num(&b_bytes);

        let ours = (Fp::from_bytes(&a_bytes) + Fp::from_bytes(&b_bytes)).to_bytes();
        let theirs = num_to_fp_bytes(&(a_ref + b_ref));
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_fp_sub(ab in any::<[u8; FP2_ENCODED_BYTES]>()) {
        let (a_bytes, b_bytes) = canon_fp2_bytes(&ab);
        let a_ref = bytes_to_num(&a_bytes);
        let b_ref = bytes_to_num(&b_bytes);

        let ours = (Fp::from_bytes(&a_bytes) - Fp::from_bytes(&b_bytes)).to_bytes();
        let theirs = num_to_fp_bytes(&(a_ref - b_ref));
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_fp_mul(ab in any::<[u8; FP2_ENCODED_BYTES]>()) {
        let (a_bytes, b_bytes) = canon_fp2_bytes(&ab);
        let a_ref = bytes_to_num(&a_bytes);
        let b_ref = bytes_to_num(&b_bytes);

        let ours = (Fp::from_bytes(&a_bytes) * Fp::from_bytes(&b_bytes)).to_bytes();
        let theirs = num_to_fp_bytes(&(a_ref * b_ref));
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_fp_square(a_bytes in any::<[u8; FP_ENCODED_BYTES]>()) {
        let canon = canon_fp_bytes(&a_bytes);
        let a_ref = bytes_to_num(&canon);

        let ours = Fp::from_bytes(&canon).square().to_bytes();
        let theirs = num_to_fp_bytes(&(&a_ref * &a_ref));
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_fp_neg(a_bytes in any::<[u8; FP_ENCODED_BYTES]>()) {
        let canon = canon_fp_bytes(&a_bytes);
        let a_ref = bytes_to_num(&canon);

        let ours = (-Fp::from_bytes(&canon)).to_bytes();
        let theirs = num_to_fp_bytes(&(-a_ref));
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_fp_invert(a_bytes in any::<[u8; FP_ENCODED_BYTES]>()) {
        let canon = canon_fp_bytes(&a_bytes);
        let a_ref = bytes_to_num(&canon);

        // `Fp::invert` on zero is undefined; skip.
        prop_assume!(!a_ref.is_zero());

        let ours = Fp::from_bytes(&canon).invert().to_bytes();
        // Reference: `a^(p-2) mod p`.
        let p = p();
        let exp = &p - 2u32;
        let inv_ref = a_ref.modpow(&exp, &p);
        let theirs = num_to_fp_bytes(&inv_ref);
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_fp2_roundtrip(bytes in any::<[u8; FP2_ENCODED_BYTES]>()) {
        let (a, b) = canon_fp2_bytes(&bytes);
        let joined = join_fp2_bytes(&a, &b);
        prop_assert_eq!(Fp2::from_bytes(&joined).to_bytes(), joined);
    }

    #[test]
    fn oracle_fp2_add(lhs in any::<[u8; FP2_ENCODED_BYTES]>(), rhs in any::<[u8; FP2_ENCODED_BYTES]>()) {
        let (la, lb) = canon_fp2_bytes(&lhs);
        let (ra, rb) = canon_fp2_bytes(&rhs);

        let x = Fp2::from_bytes(&join_fp2_bytes(&la, &lb));
        let y = Fp2::from_bytes(&join_fp2_bytes(&ra, &rb));
        let ours = (&x + &y).to_bytes();

        let theirs_a = num_to_fp_bytes(&(bytes_to_num(&la) + bytes_to_num(&ra)));
        let theirs_b = num_to_fp_bytes(&(bytes_to_num(&lb) + bytes_to_num(&rb)));
        prop_assert_eq!(ours, join_fp2_bytes(&theirs_a, &theirs_b));
    }

    #[test]
    fn oracle_fp2_sub(lhs in any::<[u8; FP2_ENCODED_BYTES]>(), rhs in any::<[u8; FP2_ENCODED_BYTES]>()) {
        let (la, lb) = canon_fp2_bytes(&lhs);
        let (ra, rb) = canon_fp2_bytes(&rhs);

        let x = Fp2::from_bytes(&join_fp2_bytes(&la, &lb));
        let y = Fp2::from_bytes(&join_fp2_bytes(&ra, &rb));
        let ours = (&x - &y).to_bytes();

        let theirs_a = num_to_fp_bytes(&(bytes_to_num(&la) - bytes_to_num(&ra)));
        let theirs_b = num_to_fp_bytes(&(bytes_to_num(&lb) - bytes_to_num(&rb)));
        prop_assert_eq!(ours, join_fp2_bytes(&theirs_a, &theirs_b));
    }

    #[test]
    fn oracle_fp2_mul(lhs in any::<[u8; FP2_ENCODED_BYTES]>(), rhs in any::<[u8; FP2_ENCODED_BYTES]>()) {
        let (la, lb) = canon_fp2_bytes(&lhs);
        let (ra, rb) = canon_fp2_bytes(&rhs);

        let x = Fp2::from_bytes(&join_fp2_bytes(&la, &lb));
        let y = Fp2::from_bytes(&join_fp2_bytes(&ra, &rb));
        let ours = (&x * &y).to_bytes();

        // `(a + bi)(c + di) = (ac − bd) + (ad + bc)·i`.
        let a = bytes_to_num(&la);
        let b = bytes_to_num(&lb);
        let c = bytes_to_num(&ra);
        let d = bytes_to_num(&rb);
        let theirs_a = num_to_fp_bytes(&(&a * &c - &b * &d));
        let theirs_b = num_to_fp_bytes(&(&a * &d + &b * &c));
        prop_assert_eq!(ours, join_fp2_bytes(&theirs_a, &theirs_b));
    }

    #[test]
    fn oracle_fp2_square(bytes in any::<[u8; FP2_ENCODED_BYTES]>()) {
        let (a_bytes, b_bytes) = canon_fp2_bytes(&bytes);
        let x = Fp2::from_bytes(&join_fp2_bytes(&a_bytes, &b_bytes));
        let ours = x.square().to_bytes();

        let a = bytes_to_num(&a_bytes);
        let b = bytes_to_num(&b_bytes);
        let theirs_a = num_to_fp_bytes(&(&a * &a - &b * &b));
        let theirs_b = num_to_fp_bytes(&(2u32 * &a * &b));
        prop_assert_eq!(ours, join_fp2_bytes(&theirs_a, &theirs_b));
    }

    #[test]
    fn oracle_fp2_norm(bytes in any::<[u8; FP2_ENCODED_BYTES]>()) {
        let (a_bytes, b_bytes) = canon_fp2_bytes(&bytes);
        let x = Fp2::from_bytes(&join_fp2_bytes(&a_bytes, &b_bytes));
        let ours = x.norm().to_bytes();

        let a = bytes_to_num(&a_bytes);
        let b = bytes_to_num(&b_bytes);
        let theirs = num_to_fp_bytes(&(&a * &a + &b * &b));
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_fp2_conjugate(bytes in any::<[u8; FP2_ENCODED_BYTES]>()) {
        let (a_bytes, b_bytes) = canon_fp2_bytes(&bytes);
        let x = Fp2::from_bytes(&join_fp2_bytes(&a_bytes, &b_bytes));
        let ours = x.conjugate().to_bytes();

        let theirs_b = num_to_fp_bytes(&(-bytes_to_num(&b_bytes)));
        prop_assert_eq!(ours, join_fp2_bytes(&a_bytes, &theirs_b));
    }

    #[test]
    fn oracle_fp2_invert_roundtrip(bytes in any::<[u8; FP2_ENCODED_BYTES]>()) {
        let (a_bytes, b_bytes) = canon_fp2_bytes(&bytes);
        let a = bytes_to_num(&a_bytes);
        let b = bytes_to_num(&b_bytes);
        let p = p();
        let norm = (&a * &a + &b * &b) % &p;
        prop_assume!(!norm.is_zero());

        let x = Fp2::from_bytes(&join_fp2_bytes(&a_bytes, &b_bytes));
        let product = (&x * &x.invert()).to_bytes();
        prop_assert_eq!(product, Fp2::ONE.to_bytes());
    }
}
