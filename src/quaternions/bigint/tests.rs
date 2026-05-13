use super::*;

type I256 = BigInt<4>;

#[test]
fn zero_is_zero() {
    let z = I256::ZERO;
    assert!(bool::from(z.is_zero()));
    assert!(!bool::from(z.is_negative()));
    assert!(!bool::from(z.is_positive()));
}

#[test]
fn from_i64_positive() {
    let x = I256::from(42i64);
    assert!(!bool::from(x.is_zero()));
    assert!(bool::from(x.is_positive()));
    assert!(!bool::from(x.is_negative()));
    assert_eq!(x.as_limbs()[0], 42);
}

#[test]
fn from_i64_negative() {
    let x = I256::from(-7i64);
    assert!(bool::from(x.is_negative()));
    assert!(!bool::from(x.is_positive()));
    assert_eq!(x.as_limbs()[0], 7);
}

#[test]
fn negation() {
    let x = I256::from(5i64);
    let neg_x = -x;
    assert!(bool::from(neg_x.is_negative()));
    assert_eq!(neg_x.as_limbs()[0], 5);
    assert_eq!(x, -neg_x);
}

#[test]
fn negation_of_zero() {
    let z = I256::ZERO;
    assert_eq!(z, -z);
}

#[test]
fn add_positive() {
    assert_eq!(I256::from(10i64) + I256::from(20i64), I256::from(30i64));
}

#[test]
fn add_negative() {
    assert_eq!(I256::from(-10i64) + I256::from(-20i64), I256::from(-30i64));
}

#[test]
fn add_mixed_signs() {
    assert_eq!(I256::from(30i64) + I256::from(-10i64), I256::from(20i64));
    assert_eq!(I256::from(-30i64) + I256::from(10i64), I256::from(-20i64));
}

#[test]
fn add_to_zero() {
    let c = I256::from(42i64) + I256::from(-42i64);
    assert!(bool::from(c.is_zero()));
    assert!(!bool::from(c.is_negative()));
}

#[test]
fn subtraction() {
    assert_eq!(I256::from(10i64) - I256::from(3i64), I256::from(7i64));
    assert_eq!(I256::from(3i64) - I256::from(10i64), I256::from(-7i64));
}

#[test]
fn multiplication() {
    assert_eq!(I256::from(6i64) * I256::from(7i64), I256::from(42i64));
}

#[test]
fn multiplication_mixed_signs() {
    assert_eq!(I256::from(-6i64) * I256::from(7i64), I256::from(-42i64));
    assert_eq!(I256::from(-6i64) * I256::from(-7i64), I256::from(42i64));
}

#[test]
fn multiplication_by_zero() {
    let r = I256::from(12345i64) * I256::ZERO;
    assert!(bool::from(r.is_zero()));
    assert!(!bool::from(r.is_negative()));
}

#[test]
fn ordering() {
    let a = I256::from(10i64);
    let b = I256::from(20i64);
    let c = I256::from(-5i64);
    assert!(a < b);
    assert!(c < a);
    assert!(c < b);
    assert!(b > a);
    assert!(a > c);
}

#[test]
fn ordering_negatives() {
    assert!(I256::from(-20i64) < I256::from(-10i64));
}

#[test]
fn bitsize_single_limb() {
    assert_eq!(nbits64(0), 0);
    assert_eq!(nbits64(1), 1);
    assert_eq!(nbits64(2), 2);
    assert_eq!(nbits64(255), 8);
    assert_eq!(nbits64(256), 9);
    assert_eq!(nbits64(u64::MAX), 64);
}

#[test]
fn bitsize_bigint() {
    assert_eq!(I256::ZERO.bitsize(), 0);
    assert_eq!(I256::ONE.bitsize(), 1);
    assert_eq!(I256::from(255i64).bitsize(), 8);
    assert_eq!(I256::from(-255i64).bitsize(), 8);
}

#[test]
fn trailing_zeros_single_limb() {
    assert_eq!(trailing_zeros(0), 64);
    assert_eq!(trailing_zeros(1), 0);
    assert_eq!(trailing_zeros(2), 1);
    assert_eq!(trailing_zeros(4), 2);
    assert_eq!(trailing_zeros(8), 3);
    assert_eq!(trailing_zeros(12), 2); // 0b1100
    assert_eq!(trailing_zeros(1 << 63), 63);
    assert_eq!(trailing_zeros(u64::MAX), 0);
    assert_eq!(trailing_zeros(0xFFFF_FFFF_0000_0000), 32);
}

#[test]
fn trailing_zeros_basic() {
    assert_eq!(I256::from_u64(1).trailing_zeros(), 0);
    assert_eq!(I256::from_u64(2).trailing_zeros(), 1);
    assert_eq!(I256::from_u64(4).trailing_zeros(), 2);
    assert_eq!(I256::from_u64(8).trailing_zeros(), 3);
    assert_eq!(I256::from_u64(12).trailing_zeros(), 2); // 0b1100
    assert_eq!(I256::ZERO.trailing_zeros(), 256); // 4 * 64
}

#[test]
fn trailing_zeros_limb_boundaries() {
    // 1 << 64: low limb zero, second limb has bit 0 set.
    assert_eq!(I256::from_limbs([0, 1, 0, 0]).trailing_zeros(), 64);
    // 1 << 127: highest bit of second limb.
    assert_eq!(
        I256::from_limbs([0, 1u64 << 63, 0, 0]).trailing_zeros(),
        127
    );
    // 1 << 128: bit 0 of third limb.
    assert_eq!(I256::from_limbs([0, 0, 1, 0]).trailing_zeros(), 128);
    // 1 << 192: bit 0 of high limb.
    assert_eq!(I256::from_limbs([0, 0, 0, 1]).trailing_zeros(), 192);
    // 1 << 255: highest representable bit.
    assert_eq!(
        I256::from_limbs([0, 0, 0, 1u64 << 63]).trailing_zeros(),
        255
    );
}

#[test]
fn trailing_zeros_only_lowest_set_bit_matters() {
    // High limbs set must NOT mask the low-limb bit. Catches "for i in 0..1"
    // truncation and "delete return" mutants on the old early-return impl.
    assert_eq!(
        I256::from_limbs([1, u64::MAX, u64::MAX, u64::MAX]).trailing_zeros(),
        0
    );
    // Low limbs zero, mid limb has its lowest bit set, high limbs garbage.
    assert_eq!(I256::from_limbs([0, 0, 1, u64::MAX]).trailing_zeros(), 128);
    // Low limb zero, second limb's bit 5 set, third nonzero — answer is 64+5.
    assert_eq!(
        I256::from_limbs([0, 1u64 << 5, 0xDEAD_BEEF, 0]).trailing_zeros(),
        64 + 5
    );
}

#[test]
fn trailing_zeros_negative_uses_magnitude() {
    // Sign is irrelevant; v_2 reads from |x|.
    assert_eq!(I256::from(-12i64).trailing_zeros(), 2);
    assert_eq!(I256::from(-1i64).trailing_zeros(), 0);
}

#[test]
fn even_odd() {
    assert!(bool::from(I256::ZERO.is_even()));
    assert!(bool::from(I256::from(2i64).is_even()));
    assert!(bool::from(I256::from(-4i64).is_even()));
    assert!(bool::from(I256::ONE.is_odd()));
    assert!(bool::from(I256::from(3i64).is_odd()));
    assert!(bool::from(I256::from(-7i64).is_odd()));
}

#[test]
fn carry_propagation() {
    let a = BigInt::<2>::from_sign_and_limbs(0, [u64::MAX, 0]);
    let b = BigInt::<2>::from(1i64);
    let c = a + b;
    assert_eq!(c.as_limbs()[0], 0);
    assert_eq!(c.as_limbs()[1], 1);
}

#[test]
fn mul_carry_propagation() {
    let a = BigInt::<2>::from_sign_and_limbs(0, [u64::MAX, 0]);
    let b = BigInt::<2>::from(2i64);
    let c = a * b;
    assert_eq!(c.as_limbs()[0], u64::MAX - 1);
    assert_eq!(c.as_limbs()[1], 1);
}

#[test]
fn display() {
    assert_eq!(format!("{}", I256::ZERO), "0x0");
    assert_eq!(format!("{}", I256::from(255i64)), "0xff");
    assert_eq!(format!("{}", I256::from(-255i64)), "-0xff");
}

#[test]
fn constants() {
    assert_eq!(I256::ONE, I256::from(1i64));
    assert_eq!(I256::MINUS_ONE, I256::from(-1i64));
    assert_eq!(I256::TWO, I256::from(2i64));
    assert_eq!(I256::THREE, I256::from(3i64));
}

#[test]
fn widen_4_to_8() {
    let small = BigInt::<4>::from(42i64);
    let wide: BigInt<8> = small.into();
    assert_eq!(wide.as_limbs()[0], 42);
    assert_eq!(wide, BigInt::<8>::from(42i64));
}

#[test]
fn widen_preserves_sign() {
    let small = BigInt::<4>::from(-7i64);
    let wide: BigInt<8> = small.into();
    assert!(bool::from(wide.is_negative()));
    assert_eq!(wide, BigInt::<8>::from(-7i64));
}

#[test]
fn narrow_8_to_4() {
    let wide = BigInt::<8>::from(99i64);
    let ct: subtle::CtOption<BigInt<4>> = wide.into();
    assert!(bool::from(ct.is_some()));
    assert_eq!(ct.unwrap(), BigInt::<4>::from(99i64));
}

#[test]
fn narrow_overflow_fails() {
    let mut wide = BigInt::<8>::from(1i64);
    wide.as_limbs_mut()[4] = 1;
    let ct: subtle::CtOption<BigInt<4>> = wide.into();
    assert!(!bool::from(ct.is_some()));
}

#[test]
fn narrow_method_basic() {
    // Production callers (signing.rs, ideal.rs) use the inherent
    // `BigInt<8>::narrow()` method, not the `From<...>` trait. Exercise
    // it directly so a `narrow -> None` mutation cannot survive.
    let wide = BigInt::<8>::from(99i64);
    let narrow = wide.narrow().expect("99 fits in BigInt<4>");
    assert_eq!(narrow, BigInt::<4>::from(99i64));

    let neg = BigInt::<8>::from(-12345i64);
    let neg_narrow = neg.narrow().expect("-12345 fits in BigInt<4>");
    assert_eq!(neg_narrow, BigInt::<4>::from(-12345i64));
}

#[test]
fn narrow_method_overflow() {
    let mut wide = BigInt::<8>::from(1i64);
    wide.as_limbs_mut()[4] = 1;
    assert!(wide.narrow().is_none());
}

#[test]
fn generic_widen_4_to_9() {
    let small = BigInt::<4>::from(-99i64);
    let wide: BigInt<9> = small.widen();
    assert_eq!(wide.as_limbs()[0], 99);
    assert_eq!(wide.as_limbs()[4], 0);
    assert!(bool::from(wide.is_negative()));
}

#[test]
fn generic_narrow_to() {
    let wide = BigInt::<9>::from(123i64);
    let narrow: BigInt<4> = wide.narrow_to().unwrap();
    assert_eq!(narrow, BigInt::<4>::from(123i64));
}

#[test]
fn generic_narrow_to_overflow() {
    let mut wide = BigInt::<9>::from(1i64);
    wide.as_limbs_mut()[5] = 1;
    assert!(wide.narrow_to::<4>().is_none());
}

#[test]
fn widen_narrow_roundtrip() {
    let orig = BigInt::<4>::from(-12345i64);
    let wide: BigInt<8> = orig.widen();
    let back: BigInt<4> = wide.narrow_to().unwrap();
    assert_eq!(orig, back);
}

#[test]
fn shl_small() {
    let a = I256::from(1i64);
    assert_eq!(a.shl(0), I256::from(1i64));
    assert_eq!(a.shl(1), I256::from(2i64));
    assert_eq!(a.shl(8), I256::from(256i64));
}

#[test]
fn shl_across_limbs() {
    let a = I256::from(1i64);
    let shifted = a.shl(64);
    assert_eq!(shifted.as_limbs()[0], 0);
    assert_eq!(shifted.as_limbs()[1], 1);
}

#[test]
fn shl_preserves_sign() {
    let a = I256::from(-3i64);
    let shifted = a.shl(4);
    assert!(bool::from(shifted.is_negative()));
    assert_eq!(shifted.as_limbs()[0], 48); // 3 << 4 = 48
}

#[test]
fn shr_small() {
    let a = I256::from(256i64);
    assert_eq!(a.shr(0), I256::from(256i64));
    assert_eq!(a.shr(1), I256::from(128i64));
    assert_eq!(a.shr(8), I256::from(1i64));
    assert_eq!(a.shr(9), I256::ZERO);
}

#[test]
fn shr_across_limbs() {
    let a = I256::from_sign_and_limbs(0, [0, 1, 0, 0]); // 2^64
    let shifted = a.shr(64);
    assert_eq!(shifted, I256::from(1i64));
}

#[test]
fn div_rem_basic() {
    let a = I256::from(17i64);
    let b = I256::from(5i64);
    let (q, r) = a.div_rem(&b);
    assert_eq!(q, I256::from(3i64));
    assert_eq!(r, I256::from(2i64));
}

#[test]
fn div_rem_exact() {
    let a = I256::from(42i64);
    let b = I256::from(6i64);
    let (q, r) = a.div_rem(&b);
    assert_eq!(q, I256::from(7i64));
    assert_eq!(r, I256::ZERO);
}

#[test]
fn div_rem_negative_dividend() {
    // Euclidean: -17 = (-4) * 5 + 3
    let a = I256::from(-17i64);
    let b = I256::from(5i64);
    let (q, r) = a.div_rem(&b);
    assert_eq!(q, I256::from(-4i64));
    assert_eq!(r, I256::from(3i64));
    assert!(bool::from(r.is_positive()) || bool::from(r.is_zero()));
}

#[test]
fn div_rem_negative_divisor() {
    // 17 = (-3) * (-5) + 2
    let a = I256::from(17i64);
    let b = I256::from(-5i64);
    let (q, r) = a.div_rem(&b);
    assert_eq!(q, I256::from(-3i64));
    assert_eq!(r, I256::from(2i64));
}

#[test]
fn div_rem_both_negative() {
    // -17 = 4 * (-5) + 3
    let a = I256::from(-17i64);
    let b = I256::from(-5i64);
    let (q, r) = a.div_rem(&b);
    assert_eq!(q, I256::from(4i64));
    assert_eq!(r, I256::from(3i64));
}

#[test]
fn ct_mod_basic() {
    assert_eq!(
        I256::from(17i64).ct_mod(&I256::from(5i64)),
        I256::from(2i64)
    );
    assert_eq!(
        I256::from(-17i64).ct_mod(&I256::from(5i64)),
        I256::from(3i64)
    );
}

#[test]
fn two_adic_val_tests() {
    assert_eq!(I256::from(1i64).two_adic_val(), 0);
    assert_eq!(I256::from(2i64).two_adic_val(), 1);
    assert_eq!(I256::from(8i64).two_adic_val(), 3);
    assert_eq!(I256::from(12i64).two_adic_val(), 2); // 12 = 4 * 3
    assert_eq!(I256::from(-24i64).two_adic_val(), 3); // 24 = 8 * 3
}

#[test]
fn pow_tests() {
    assert_eq!(I256::from(2i64).pow(10), I256::from(1024i64));
    assert_eq!(I256::from(3i64).pow(0), I256::ONE);
    assert_eq!(I256::from(-2i64).pow(3), I256::from(-8i64));
    assert_eq!(I256::from(-2i64).pow(4), I256::from(16i64));
}

#[test]
fn divides_tests() {
    assert!(bool::from(I256::from(3i64).divides(&I256::from(12i64))));
    assert!(!bool::from(I256::from(5i64).divides(&I256::from(12i64))));
    assert!(bool::from(I256::from(1i64).divides(&I256::from(7i64))));
}

#[test]
fn gcd_basic() {
    assert_eq!(I256::from(12i64).gcd(&I256::from(8i64)), I256::from(4i64));
    assert_eq!(I256::from(17i64).gcd(&I256::from(13i64)), I256::from(1i64));
    assert_eq!(I256::from(0i64).gcd(&I256::from(5i64)), I256::from(5i64));
    assert_eq!(I256::from(5i64).gcd(&I256::from(0i64)), I256::from(5i64));
}

#[test]
fn gcd_negative() {
    // GCD should always return non-negative.
    assert_eq!(I256::from(-12i64).gcd(&I256::from(8i64)), I256::from(4i64));
    assert_eq!(I256::from(-12i64).gcd(&I256::from(-8i64)), I256::from(4i64));
}

#[test]
fn xgcd_basic() {
    let a = I256::from(35i64);
    let b = I256::from(15i64);
    let (g, x, y) = a.xgcd(&b);
    assert_eq!(g, I256::from(5i64));
    // Verify Bezout identity: a*x + b*y == g.
    let lhs = a.ct_mul(&x).ct_add(&b.ct_mul(&y));
    assert_eq!(lhs, g);
}

#[test]
fn xgcd_coprime() {
    let a = I256::from(17i64);
    let b = I256::from(13i64);
    let (g, x, y) = a.xgcd(&b);
    assert_eq!(g, I256::ONE);
    let lhs = a.ct_mul(&x).ct_add(&b.ct_mul(&y));
    assert_eq!(lhs, g);
}

#[test]
fn xgcd_negative() {
    let a = I256::from(-35i64);
    let b = I256::from(15i64);
    let (g, x, y) = a.xgcd(&b);
    assert_eq!(g, I256::from(5i64));
    let lhs = a.ct_mul(&x).ct_add(&b.ct_mul(&y));
    assert_eq!(lhs, g);
}

#[test]
fn invert_mod_basic() {
    // 3^{-1} mod 7 = 5, since 3*5 = 15 = 1 mod 7.
    let inv = I256::from(3i64).invert_mod(&I256::from(7i64));
    assert_eq!(inv, Some(I256::from(5i64)));
}

#[test]
fn invert_mod_no_inverse() {
    // 6 and 9 share factor 3, no inverse.
    let inv = I256::from(6i64).invert_mod(&I256::from(9i64));
    assert_eq!(inv, None);
}

#[test]
fn invert_mod_verify() {
    let a = I256::from(11i64);
    let m = I256::from(23i64);
    let inv = a.invert_mod(&m).expect("inverse should exist");
    let product = a.ct_mul(&inv).ct_mod(&m);
    assert_eq!(product, I256::ONE);
}

#[test]
fn sqrt_floor_basic() {
    assert_eq!(I256::ZERO.sqrt_floor(), I256::ZERO);
    assert_eq!(I256::ONE.sqrt_floor(), I256::ONE);
    assert_eq!(I256::from(4i64).sqrt_floor(), I256::from(2i64));
    assert_eq!(I256::from(9i64).sqrt_floor(), I256::from(3i64));
    assert_eq!(I256::from(10i64).sqrt_floor(), I256::from(3i64));
    assert_eq!(I256::from(99i64).sqrt_floor(), I256::from(9i64));
    assert_eq!(I256::from(100i64).sqrt_floor(), I256::from(10i64));
}

#[test]
fn pow_mod_basic() {
    // 2^10 mod 1000 = 1024 mod 1000 = 24
    let r = I256::pow_mod(&I256::from(2i64), &I256::from(10i64), &I256::from(1000i64));
    assert_eq!(r, I256::from(24i64));
}

#[test]
fn pow_mod_fermat() {
    // Fermat's little theorem: a^(p-1) = 1 mod p for prime p.
    let a = I256::from(3i64);
    let p = I256::from(17i64);
    let exp = I256::from(16i64); // p - 1
    let r = I256::pow_mod(&a, &exp, &p);
    assert_eq!(r, I256::ONE);
}

#[test]
fn modular_sqrt_3mod4() {
    // 4 is a QR mod 7 (7 ≡ 3 mod 4). sqrt(4) mod 7 = 2.
    let r = I256::modular_sqrt(&I256::from(4i64), &I256::from(7i64));
    let r = r.expect("4 is a QR mod 7");
    assert_eq!(r.ct_mul(&r).ct_mod(&I256::from(7i64)), I256::from(4i64));
}

#[test]
fn modular_sqrt_5mod8() {
    // 3 is a QR mod 13 (13 ≡ 5 mod 8). Check sqrt(3)² ≡ 3 mod 13.
    let r = I256::modular_sqrt(&I256::from(3i64), &I256::from(13i64));
    let r = r.expect("3 is a QR mod 13");
    assert_eq!(r.ct_mul(&r).ct_mod(&I256::from(13i64)), I256::from(3i64));
}

#[test]
fn modular_sqrt_non_residue() {
    // 3 is not a QR mod 7.
    let r = I256::modular_sqrt(&I256::from(3i64), &I256::from(7i64));
    assert!(r.is_none());
}

// Width-aware regression tests (see paper §5 "Bugs from Fixed-Width
// Arithmetic"). Confirm the `_w` variants give correct answers on
// moduli where the plain versions would silently truncate.

#[test]
fn pow_mod_w_dmix_fermat() {
    // Fermat's little theorem: 2^(D_mix - 1) ≡ 1 (mod D_mix) for the
    // 513-bit prime D_mix = 2^512 + 75. Requires working width W ≥ 17.
    use crate::params::D_MIX;
    let two = BigInt::<9>::from_u64(2);
    let exp = D_MIX.ct_sub(&BigInt::<9>::ONE);
    let r = BigInt::<9>::pow_mod_w::<18>(&two, &exp, &D_MIX);
    assert_eq!(r, BigInt::<9>::ONE, "2^(D_mix - 1) must be 1 mod D_mix");
}

#[test]
fn legendre_w_dmix_square() {
    // `4 = 2²` is a perfect square, so `Legendre(4, N) = 1` for any
    // odd prime N. Requires the wide variant for 513-bit D_mix.
    use crate::params::D_MIX;
    let four = BigInt::<9>::from_u64(4);
    assert_eq!(BigInt::<9>::legendre_w::<18>(&four, &D_MIX), 1);
}

#[test]
fn is_probable_prime_w_dmix_is_prime() {
    // D_mix = 2^512 + 75 is the smallest prime greater than 2^512.
    // Requires the wide variant; the plain version returns false.
    use crate::params::D_MIX;
    assert!(D_MIX.is_probable_prime_w::<18>(12));
}

#[test]
fn modular_sqrt_w_dmix_roundtrip() {
    // For a random perfect square n² mod D_mix, sqrt(n²) must equal
    // n or D_mix − n.
    use crate::params::D_MIX;
    let n = BigInt::<9>::from_u64(12345);
    let n_sq = n.ct_mul(&n).ct_mod(&D_MIX);
    let r = BigInt::<9>::modular_sqrt_w::<18>(&n_sq, &D_MIX).expect("n² is a QR mod D_mix");
    assert!(r == n || r == D_MIX.ct_sub(&n));
}

#[test]
fn cornacchia_basic() {
    // Solve x² + y² = 5 (q=1, m=5). Solution: (1, 2) or (2, 1).
    let result = I256::cornacchia(&I256::ONE, &I256::from(5i64));
    let (x, y) = result.expect("5 = x² + y² should have a solution");
    assert_eq!(x.ct_mul(&x).ct_add(&y.ct_mul(&y)), I256::from(5i64));
}

#[test]
fn cornacchia_with_q() {
    // Solve x² + 3y² = 7 (q=3, m=7). Solution: (2, 1).
    let result = I256::cornacchia(&I256::from(3i64), &I256::from(7i64));
    let (x, y) = result.expect("7 = x² + 3y² should have a solution");
    let check = x.ct_mul(&x).ct_add(&I256::from(3i64).ct_mul(&y.ct_mul(&y)));
    assert_eq!(check, I256::from(7i64));
}

#[test]
fn cornacchia_no_solution() {
    // x² + y² = 3 has no solution (3 ≡ 3 mod 4, not sum of two squares).
    let result = I256::cornacchia(&I256::ONE, &I256::from(3i64));
    assert!(result.is_none());
}

#[test]
fn rand_interval_zero_range_no_rng_consumed() {
    use crate::drbg::Aes256CtrDrbg;

    // a == b: rand_interval must return a without consuming any randomness.
    let mut drbg = Aes256CtrDrbg::new(&[0u8; 48]);
    let initial_consumed = drbg.bytes_consumed();
    let a = I256::from(42i64);
    let v = I256::rand_interval(&mut drbg, &a, &a);
    assert_eq!(v, a);
    assert_eq!(drbg.bytes_consumed(), initial_consumed);
}

#[test]
fn rand_interval_in_range_for_many_samples() {
    use crate::drbg::Aes256CtrDrbg;

    // Every sample must lie in `[a, b]` inclusive across a wide range.
    let mut drbg = Aes256CtrDrbg::new(&[1u8; 48]);
    let a = I256::from(100i64);
    let b = I256::from(1_000_000i64);
    for _ in 0..200 {
        let v = I256::rand_interval(&mut drbg, &a, &b);
        assert!(v >= a, "value {v} below lower bound {a}");
        assert!(v <= b, "value {v} above upper bound {b}");
    }
}

#[test]
fn rand_interval_deterministic_under_same_seed() {
    use crate::drbg::Aes256CtrDrbg;

    // Same DRBG seed must produce the same sequence of samples.
    let seed = [7u8; 48];
    let a = I256::from(1i64);
    let b = I256::from(1i64 << 40);

    let mut d1 = Aes256CtrDrbg::new(&seed);
    let mut d2 = Aes256CtrDrbg::new(&seed);
    for _ in 0..50 {
        let v1 = I256::rand_interval(&mut d1, &a, &b);
        let v2 = I256::rand_interval(&mut d2, &a, &b);
        assert_eq!(v1, v2, "same-seed DRBGs must produce identical samples");
    }
}

#[test]
fn rand_interval_byte_aligned_bound() {
    use crate::drbg::Aes256CtrDrbg;

    // Range `[0, 2^64 - 1]`: bit-length is exactly 64 (byte-aligned),
    // so the top-byte mask is `0xFF` and every drawn 8-byte value is
    // accepted on the first try.
    let mut drbg = Aes256CtrDrbg::new(&[2u8; 48]);
    let a = I256::ZERO;
    let b = I256::from_limbs([u64::MAX, 0, 0, 0]);
    let initial = drbg.bytes_consumed();
    let _ = I256::rand_interval(&mut drbg, &a, &b);
    // Single accepted draw consumes exactly 8 bytes; with `top_mask =
    // 0xFF` and `bmina = 2^64 − 1`, every drawn value satisfies
    // `val ≤ bmina`, so no rejection cycle.
    assert_eq!(drbg.bytes_consumed() - initial, 8);
}

#[test]
fn rand_interval_partial_byte_bound() {
    use crate::drbg::Aes256CtrDrbg;

    // Range `[0, 2^70 − 1]`: bit-length is exactly 70, so
    // `len_bytes = 9` and the top byte must be masked to its low 6
    // bits. Verify all samples respect the bound (no over-large
    // values leaking through a wrong mask).
    let mut drbg = Aes256CtrDrbg::new(&[3u8; 48]);
    let a = I256::ZERO;
    // b = 2^70 − 1: limb 0 = 0xFFFFFFFF_FFFFFFFF, limb 1 = 0x3F.
    let b = I256::from_limbs([u64::MAX, 0x3F, 0, 0]);
    for _ in 0..100 {
        let v = I256::rand_interval(&mut drbg, &a, &b);
        assert!(v >= a);
        assert!(v <= b, "value {v} > 2^70 − 1");
    }
}

use proptest::prelude::*;

/// Generates a random `BigInt<4>` from a sign bit and four u64 limbs.
fn arb_bigint4() -> impl Strategy<Value = BigInt<4>> {
    (any::<bool>(), any::<[u64; 4]>())
        .prop_map(|(neg, limbs)| BigInt::from_sign_and_limbs(if neg { 1 } else { 0 }, limbs))
}

/// Generates a small `BigInt<4>` (fits in i64) for tests where overflow
/// in multiplication would wrap and obscure the algebraic property.
fn arb_small_bigint4() -> impl Strategy<Value = BigInt<4>> {
    any::<i32>().prop_map(|v| BigInt::from_i64(v as i64))
}

proptest! {
    #[test]
    fn bigint_add_commutative(a in arb_bigint4(), b in arb_bigint4()) {
        prop_assert_eq!(a + b, b + a);
    }

    // Uses small values to avoid overflow — BigInt<4> addition wraps on
    // 256-bit overflow, breaking associativity for full-range inputs.
    #[test]
    fn bigint_add_associative(a in arb_small_bigint4(), b in arb_small_bigint4(), c in arb_small_bigint4()) {
        prop_assert_eq!((a + b) + c, a + (b + c));
    }

    #[test]
    fn bigint_add_identity(a in arb_bigint4()) {
        prop_assert_eq!(a + BigInt::ZERO, a);
        prop_assert_eq!(BigInt::ZERO + a, a);
    }

    #[test]
    fn bigint_add_inverse(a in arb_bigint4()) {
        prop_assert_eq!(a + (-a), BigInt::ZERO);
        prop_assert_eq!((-a) + a, BigInt::ZERO);
    }

    #[test]
    fn bigint_sub_is_add_neg(a in arb_bigint4(), b in arb_bigint4()) {
        prop_assert_eq!(a - b, a + (-b));
    }

    #[test]
    fn bigint_double_neg(a in arb_bigint4()) {
        prop_assert_eq!(-(-a), a);
    }

    #[test]
    fn bigint_mul_commutative(a in arb_small_bigint4(), b in arb_small_bigint4()) {
        prop_assert_eq!(a * b, b * a);
    }

    #[test]
    fn bigint_mul_identity(a in arb_bigint4()) {
        prop_assert_eq!(a * BigInt::ONE, a);
        prop_assert_eq!(BigInt::ONE * a, a);
    }

    #[test]
    fn bigint_mul_zero(a in arb_bigint4()) {
        prop_assert_eq!(a * BigInt::ZERO, BigInt::ZERO);
    }

    #[test]
    fn bigint_mul_minus_one(a in arb_bigint4()) {
        prop_assert_eq!(a * BigInt::MINUS_ONE, -a);
    }

    #[test]
    fn bigint_distributive(a in arb_small_bigint4(), b in arb_small_bigint4(), c in arb_small_bigint4()) {
        prop_assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn bigint_mul_associative(a in arb_small_bigint4(), b in arb_small_bigint4(), c in arb_small_bigint4()) {
        prop_assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn bigint_abs_nonnegative(a in arb_bigint4()) {
        let abs_a = a.abs();
        // abs(a) is non-negative (sign == 0) unless a is zero.
        prop_assert!(!bool::from(abs_a.is_negative()) || bool::from(abs_a.is_zero()));
    }

    #[test]
    fn bigint_abs_idempotent(a in arb_bigint4()) {
        prop_assert_eq!(a.abs().abs(), a.abs());
    }
}

// Division and GCD properties.
proptest! {
    #[test]
    fn bigint_div_rem_identity(a in arb_small_bigint4(), d in arb_small_bigint4()) {
        // a = q * d + r, with 0 <= r < |d|.
        prop_assume!(!bool::from(d.is_zero()));
        let (q, r) = a.div_rem(&d);
        prop_assert_eq!(q * d + r, a);
    }

    #[test]
    fn bigint_div_rem_remainder_nonnegative(a in arb_small_bigint4(), d in arb_small_bigint4()) {
        prop_assume!(!bool::from(d.is_zero()));
        let (_, r) = a.div_rem(&d);
        prop_assert!(!bool::from(r.is_negative()));
    }

    #[test]
    fn bigint_gcd_commutative(a in arb_small_bigint4(), b in arb_small_bigint4()) {
        prop_assert_eq!(a.gcd(&b), b.gcd(&a));
    }

    #[test]
    fn bigint_gcd_divides_both(a in arb_small_bigint4(), b in arb_small_bigint4()) {
        let g = a.gcd(&b);
        if !bool::from(g.is_zero()) {
            let (_, ra) = a.div_rem(&g);
            let (_, rb) = b.div_rem(&g);
            prop_assert!(bool::from(ra.is_zero()), "gcd does not divide a");
            prop_assert!(bool::from(rb.is_zero()), "gcd does not divide b");
        }
    }

    #[test]
    fn bigint_gcd_with_zero(a in arb_small_bigint4()) {
        // gcd(a, 0) = |a|.
        prop_assert_eq!(a.gcd(&BigInt::ZERO), a.abs());
    }

    #[test]
    fn bigint_gcd_idempotent(a in arb_small_bigint4()) {
        // gcd(a, a) = |a|.
        prop_assert_eq!(a.gcd(&a), a.abs());
    }
}

// Shift and valuation properties.
proptest! {
    #[test]
    fn bigint_shl_shr_roundtrip(a in arb_small_bigint4(), k in 0u32..64) {
        // (a << k) >> k == a for small a where no bits are lost.
        let shifted = a.shl(k).shr(k);
        prop_assert_eq!(shifted, a);
    }

    #[test]
    fn bigint_shl_zero(a in arb_bigint4()) {
        prop_assert_eq!(a.shl(0), a);
    }

    #[test]
    fn bigint_shr_zero(a in arb_bigint4()) {
        prop_assert_eq!(a.shr(0), a);
    }

    #[test]
    fn bigint_two_adic_val_of_power_of_two(k in 1u32..200) {
        // v_2(2^k) = k.
        let val = BigInt::<4>::ONE.shl(k);
        prop_assert_eq!(val.two_adic_val(), k);
    }

    #[test]
    fn bigint_two_adic_val_of_odd(a in arb_small_bigint4()) {
        // An odd number has v_2 = 0.
        prop_assume!(!bool::from(a.is_zero()));
        let odd = a.abs().shl(1) + BigInt::ONE; // 2|a| + 1 is always odd
        prop_assert_eq!(odd.two_adic_val(), 0);
    }
}
