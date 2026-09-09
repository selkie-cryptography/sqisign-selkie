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
    let wide = BigInt::<8>::from_limbs([1, 0, 0, 0, 1, 0, 0, 0]);
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
    let wide = BigInt::<8>::from_limbs([1, 0, 0, 0, 1, 0, 0, 0]);
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
    let wide = BigInt::<9>::from_limbs([1, 0, 0, 0, 0, 1, 0, 0, 0]);
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
    assert_eq!(a << 0, I256::from(1i64));
    assert_eq!(a << 1, I256::from(2i64));
    assert_eq!(a << 8, I256::from(256i64));
}

#[test]
fn shl_across_limbs() {
    let a = I256::from(1i64);
    let shifted = a << 64;
    assert_eq!(shifted.as_limbs()[0], 0);
    assert_eq!(shifted.as_limbs()[1], 1);
}

#[test]
fn shl_preserves_sign() {
    let a = I256::from(-3i64);
    let shifted = a << 4;
    assert!(bool::from(shifted.is_negative()));
    assert_eq!(shifted.as_limbs()[0], 48); // 3 << 4 = 48
}

#[test]
fn shr_small() {
    let a = I256::from(256i64);
    assert_eq!(a >> 0, I256::from(256i64));
    assert_eq!(a >> 1, I256::from(128i64));
    assert_eq!(a >> 8, I256::from(1i64));
    assert_eq!(a >> 9, I256::ZERO);
}

#[test]
fn shr_across_limbs() {
    let a = I256::from_sign_and_limbs(0, [0, 1, 0, 0]); // 2^64
    let shifted = a >> 64;
    assert_eq!(shifted, I256::from(1i64));
}

#[test]
fn shr_negative_rounds_toward_negative_infinity() {
    assert_eq!(I256::from(-1i64) >> 1, I256::from(-1i64));
    assert_eq!(I256::from(-2i64) >> 1, I256::from(-1i64));
    assert_eq!(I256::from(-3i64) >> 1, I256::from(-2i64));
    assert_eq!(I256::from(-4i64) >> 2, I256::from(-1i64));
    assert_eq!(I256::from(-5i64) >> 2, I256::from(-2i64));
    assert_eq!(I256::from(-1i64) >> 300, I256::from(-1i64));

    // Dropped bits below a limb boundary still round down.
    let minus_2_64 = I256::from_sign_and_limbs(1, [0, 1, 0, 0]);
    assert_eq!(minus_2_64 >> 64, I256::from(-1i64));
    let minus_2_64_plus_1 = I256::from_sign_and_limbs(1, [1, 1, 0, 0]);
    assert_eq!(minus_2_64_plus_1 >> 64, I256::from(-2i64));
}

#[test]
fn div_rem_basic() {
    let a = I256::from(17i64);
    let b = I256::from(5i64);
    let (q, r) = a.vt_div_rem(&b);
    assert_eq!(q, I256::from(3i64));
    assert_eq!(r, I256::from(2i64));
}

#[test]
fn div_rem_exact() {
    let a = I256::from(42i64);
    let b = I256::from(6i64);
    let (q, r) = a.vt_div_rem(&b);
    assert_eq!(q, I256::from(7i64));
    assert_eq!(r, I256::ZERO);
}

#[test]
fn div_rem_negative_dividend() {
    // Truncating: -17 = (-3) * 5 + (-2); the remainder keeps the
    // dividend's sign.
    let a = I256::from(-17i64);
    let b = I256::from(5i64);
    let (q, r) = a.vt_div_rem(&b);
    assert_eq!(q, I256::from(-3i64));
    assert_eq!(r, I256::from(-2i64));
}

#[test]
fn div_rem_negative_divisor() {
    // 17 = (-3) * (-5) + 2
    let a = I256::from(17i64);
    let b = I256::from(-5i64);
    let (q, r) = a.vt_div_rem(&b);
    assert_eq!(q, I256::from(-3i64));
    assert_eq!(r, I256::from(2i64));
}

#[test]
fn div_rem_both_negative() {
    // -17 = 3 * (-5) + (-2)
    let a = I256::from(-17i64);
    let b = I256::from(-5i64);
    let (q, r) = a.vt_div_rem(&b);
    assert_eq!(q, I256::from(3i64));
    assert_eq!(r, I256::from(-2i64));
}

#[test]
fn div_rem_exact_negative_has_zero_remainder_with_sign_zero() {
    let (q, r) = I256::from(-15i64).vt_div_rem(&I256::from(5i64));
    assert_eq!(q, I256::from(-3i64));
    assert_eq!(r, I256::ZERO);
    assert!(!bool::from(r.is_negative()));
}

#[test]
fn vt_mod_is_non_negative_and_ignores_divisor_sign() {
    let five = I256::from(5i64);
    let minus_five = I256::from(-5i64);
    assert_eq!(I256::from(17i64).vt_mod(&five), I256::from(2i64));
    assert_eq!(I256::from(-17i64).vt_mod(&five), I256::from(3i64));
    assert_eq!(I256::from(17i64).vt_mod(&minus_five), I256::from(2i64));
    assert_eq!(I256::from(-17i64).vt_mod(&minus_five), I256::from(3i64));
    assert_eq!(I256::from(-15i64).vt_mod(&five), I256::ZERO);
}

#[test]
fn gcd_of_zeros_is_zero_at_every_dispatch_width() {
    assert_eq!(BigInt::<4>::ZERO.gcd(&BigInt::<4>::ZERO), BigInt::<4>::ZERO);
    assert_eq!(BigInt::<8>::ZERO.gcd(&BigInt::<8>::ZERO), BigInt::<8>::ZERO);
    assert_eq!(
        BigInt::<30>::ZERO.gcd(&BigInt::<30>::ZERO),
        BigInt::<30>::ZERO
    );

    let minus_six = BigInt::<8>::from(-6i64);
    assert_eq!(BigInt::<8>::ZERO.gcd(&minus_six), BigInt::<8>::from(6i64));
    assert_eq!(minus_six.gcd(&BigInt::<8>::ZERO), BigInt::<8>::from(6i64));
}

#[test]
fn vt_divides_tests() {
    assert!(bool::from(I256::from(3i64).vt_divides(&I256::from(12i64))));
    assert!(!bool::from(I256::from(5i64).vt_divides(&I256::from(12i64))));
    assert!(bool::from(I256::from(1i64).vt_divides(&I256::from(7i64))));
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

// Regression: the exact N=500 input from sign KAT 082 that exposed the
// cofactor-matrix both-non-negative row bug (Lehmer collapsed the odd
// part of the gcd, returning a 1507-bit value for a 3143-bit gcd). Both
// operands share a 1506-bit power-of-two factor and differ in length.
#[test]
fn lehmer_gcd_n500_kat082_regression() {
    let mut al = [0u64; 500];
    let a_tail = [
        16020870146050490368u64,
        6441575464503925197,
        10438403543318052669,
        6273953587472686692,
        12669500009644519806,
        13575231851056910516,
        16531505793299612324,
        12147896820382623561,
        12201412089280309712,
        13363937555882238624,
        7483082758042540783,
        4296080171076512387,
        2105985607161184141,
        11921311919214306120,
        6281329449158348896,
        7352118625919954864,
        3682704103088376909,
        3630503677488505233,
        16620472580358968276,
        803929341612212748,
        13936204271398113154,
        12340270009885320974,
        7151593045736592073,
        8365807683319818920,
        1181258786873885634,
        16748251176927411295,
        11801211617407249874,
        17678181516158542043,
        8618148130326644574,
        15683597732090426596,
        3805116,
    ];
    al[23..23 + a_tail.len()].copy_from_slice(&a_tail);
    let mut bl = [0u64; 500];
    let b_tail = [
        11942447858049089536u64,
        13902625029327768126,
        11663097056482124493,
        13807830429925569945,
        14883174261547469373,
        4615077038402999884,
        10885811495559111387,
        10259729562784726163,
        18015543830526546403,
        14011716040121428515,
        2474928486255298158,
        8955743019246365953,
        4203068231206608123,
        3454630796943493064,
        17976048411414720379,
        17419681015382949383,
        2672584484290635119,
        10060937796968091305,
        12343821835388960437,
        14865773467022516500,
        869165700745454735,
        17782786244527104436,
        15078561100211413377,
        5542193020527508586,
        1629344405830098200,
        15379494206881480743,
        5884110446339506177,
        17121702374425667366,
        15956447681082510272,
        9304185429433213149,
        452,
    ];
    bl[31..31 + b_tail.len()].copy_from_slice(&b_tail);
    let a = BigInt::<500>::from_sign_and_limbs(0, al);
    let b = BigInt::<500>::from_sign_and_limbs(0, bl);
    assert_eq!(a.gcd_lehmer(&b), a.gcd_stein(&b));
    assert_eq!(a.gcd_lehmer(&b).bitsize(), 3143);
}

// Full-width byte-identity probe: gcd_lehmer vs Stein on operands that
// fill ALL N limbs (no headroom), the case real W-wide gcd callers hit.
#[test]
fn lehmer_gcd_full_width_probe() {
    use proptest::test_runner::{Config, TestRunner};
    fn probe<const N: usize>(runner: &mut TestRunner) {
        let strat = prop::collection::vec(any::<u64>(), 2 * N);
        runner
            .run(&strat, |raw| {
                let mut al = [0u64; N];
                let mut bl = [0u64; N];
                al.copy_from_slice(&raw[..N]);
                bl.copy_from_slice(&raw[N..]);
                let a = BigInt::<N>::from_sign_and_limbs(0, al);
                let b = BigInt::<N>::from_sign_and_limbs(0, bl);
                prop_assert_eq!(a.gcd_lehmer(&b), a.gcd_stein(&b));
                Ok(())
            })
            .unwrap();
    }
    // Mixed-length probe: independently random significant lengths for a
    // and b (including 0 and 1), stressing the single-word finish and the
    // division-step fallback that uniform full-width inputs never reach.
    fn probe_mixed<const N: usize>(runner: &mut TestRunner) {
        let strat = (
            prop::collection::vec(any::<u64>(), N),
            prop::collection::vec(any::<u64>(), N),
            0usize..=N,
            0usize..=N,
        );
        runner
            .run(&strat, |(ra, rb, la, lb)| {
                let mut al = [0u64; N];
                let mut bl = [0u64; N];
                al[..la].copy_from_slice(&ra[..la]);
                bl[..lb].copy_from_slice(&rb[..lb]);
                let a = BigInt::<N>::from_sign_and_limbs(0, al);
                let b = BigInt::<N>::from_sign_and_limbs(0, bl);
                prop_assert_eq!(a.gcd_lehmer(&b), a.gcd_stein(&b));
                Ok(())
            })
            .unwrap();
    }

    // Signed probe: random signs on both operands, exercising the
    // abs()/canonical-zero edges the gcd callers (e.g. det1.gcd(det2))
    // reach with negative inputs.
    fn probe_signed<const N: usize>(runner: &mut TestRunner) {
        let strat = (
            prop::collection::vec(any::<u64>(), N),
            prop::collection::vec(any::<u64>(), N),
            any::<bool>(),
            any::<bool>(),
            0usize..=N,
            0usize..=N,
        );
        runner
            .run(&strat, |(ra, rb, sa, sb, la, lb)| {
                let mut al = [0u64; N];
                let mut bl = [0u64; N];
                al[..la].copy_from_slice(&ra[..la]);
                bl[..lb].copy_from_slice(&rb[..lb]);
                let a = BigInt::<N>::from_sign_and_limbs(u64::from(sa), al);
                let b = BigInt::<N>::from_sign_and_limbs(u64::from(sb), bl);
                prop_assert_eq!(a.gcd_lehmer(&b), a.gcd_stein(&b));
                Ok(())
            })
            .unwrap();
    }

    // Shifted probe: both operands share a large power-of-two factor
    // (their nonzero limbs sit in a high window, low limbs all zero) and
    // have independent, often-unequal effective lengths. This is the
    // signing-lattice input class (gcd of scaled determinants) that
    // tripped the cofactor-matrix both-non-negative row bug.
    fn probe_shifted<const N: usize>(runner: &mut TestRunner) {
        let strat = (
            prop::collection::vec(any::<u64>(), N),
            prop::collection::vec(any::<u64>(), N),
            0usize..N,
            0usize..N,
            0usize..N,
        );
        runner
            .run(&strat, |(ra, rb, base, la, lb)| {
                let mut al = [0u64; N];
                let mut bl = [0u64; N];
                let enda = (base + la).min(N);
                let endb = (base + lb).min(N);
                al[base..enda].copy_from_slice(&ra[..enda - base]);
                bl[base..endb].copy_from_slice(&rb[..endb - base]);
                let a = BigInt::<N>::from_sign_and_limbs(0, al);
                let b = BigInt::<N>::from_sign_and_limbs(0, bl);
                prop_assert_eq!(a.gcd_lehmer(&b), a.gcd_stein(&b));
                Ok(())
            })
            .unwrap();
    }

    let mut runner = TestRunner::new(Config::with_cases(3000));
    probe::<8>(&mut runner);
    probe::<16>(&mut runner);
    probe::<30>(&mut runner);
    probe_mixed::<8>(&mut runner);
    probe_mixed::<16>(&mut runner);
    probe_mixed::<30>(&mut runner);
    probe_mixed::<60>(&mut runner);
    probe_mixed::<150>(&mut runner);
    probe_signed::<8>(&mut runner);
    probe_signed::<30>(&mut runner);
    probe_shifted::<16>(&mut runner);
    probe_shifted::<60>(&mut runner);
    probe_shifted::<150>(&mut runner);
}

// Exhaustive small-value byte-identity of gcd_lehmer vs Stein gcd, plus
// a few wide structured cases that stress the cofactor-matrix update.
#[test]
fn lehmer_gcd_small_exhaustive() {
    for a in 0u64..=200 {
        for b in 0u64..=200 {
            let av = I256::from_limbs([a, 0, 0, 0]);
            let bv = I256::from_limbs([b, 0, 0, 0]);
            let stein = av.gcd(&bv);
            let lehmer = av.gcd_lehmer(&bv);
            assert_eq!(stein, lehmer, "gcd({a},{b}) stein != lehmer");
        }
    }
    // A few wide structured cases.
    let cases = [
        (
            [0xFFFF_FFFF_FFFF_FFFFu64, 0xFFFF_FFFF, 0, 0],
            [0x1_0000_0000u64, 1, 0, 0],
        ),
        (
            [0xDEAD_BEEF_CAFE_BABEu64, 0x1234_5678, 0xABCD, 0],
            [0xFEDC_BA98u64, 0x9999, 0, 0],
        ),
        (
            [u64::MAX, u64::MAX, u64::MAX, 0],
            [u64::MAX, u64::MAX, 0, 0],
        ),
    ];
    for (al, bl) in cases {
        let av = I256::from_limbs(al);
        let bv = I256::from_limbs(bl);
        assert_eq!(
            av.gcd(&bv),
            av.gcd_lehmer(&bv),
            "wide gcd mismatch {al:?} {bl:?}"
        );
    }
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
    let product = a.ct_mul(&inv).vt_mod(&m);
    assert_eq!(product, I256::ONE);
}

#[test]
fn sqrt_floor_basic() {
    assert_eq!(I256::ZERO.sqrt_floor(), Some(I256::ZERO));
    assert_eq!(I256::ONE.sqrt_floor(), Some(I256::ONE));
    assert_eq!(I256::from(4i64).sqrt_floor(), Some(I256::from(2i64)));
    assert_eq!(I256::from(9i64).sqrt_floor(), Some(I256::from(3i64)));
    assert_eq!(I256::from(10i64).sqrt_floor(), Some(I256::from(3i64)));
    assert_eq!(I256::from(99i64).sqrt_floor(), Some(I256::from(9i64)));
    assert_eq!(I256::from(100i64).sqrt_floor(), Some(I256::from(10i64)));
    assert_eq!(I256::from(-1i64).sqrt_floor(), None);
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
    assert_eq!(r.ct_mul(&r).vt_mod(&I256::from(7i64)), I256::from(4i64));
}

#[test]
fn modular_sqrt_5mod8() {
    // 3 is a QR mod 13 (13 ≡ 5 mod 8). Check sqrt(3)² ≡ 3 mod 13.
    let r = I256::modular_sqrt(&I256::from(3i64), &I256::from(13i64));
    let r = r.expect("3 is a QR mod 13");
    assert_eq!(r.ct_mul(&r).vt_mod(&I256::from(13i64)), I256::from(3i64));
}

#[test]
fn modular_sqrt_non_residue() {
    // 3 is not a QR mod 7.
    let r = I256::modular_sqrt(&I256::from(3i64), &I256::from(7i64));
    assert!(r.is_none());
}

// Width-aware regression tests at extreme widths where the plain
// `BigInt<N>` ops would silently truncate. Confirm the `_w` variants
// give correct answers on moduli that exceed the narrow path.

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
    let n_sq = n.ct_mul(&n).vt_mod(&D_MIX);
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
    fn prop_bigint_add_commutative(a in arb_bigint4(), b in arb_bigint4()) {
        prop_assert_eq!(a + b, b + a);
    }

    // Uses small values to avoid overflow — BigInt<4> addition wraps on
    // 256-bit overflow, breaking associativity for full-range inputs.
    #[test]
    fn prop_bigint_add_associative(a in arb_small_bigint4(), b in arb_small_bigint4(), c in arb_small_bigint4()) {
        prop_assert_eq!((a + b) + c, a + (b + c));
    }

    #[test]
    fn prop_bigint_add_identity(a in arb_bigint4()) {
        prop_assert_eq!(a + BigInt::ZERO, a);
        prop_assert_eq!(BigInt::ZERO + a, a);
    }

    #[test]
    fn prop_bigint_add_inverse(a in arb_bigint4()) {
        prop_assert_eq!(a + (-a), BigInt::ZERO);
        prop_assert_eq!((-a) + a, BigInt::ZERO);
    }

    #[test]
    fn prop_bigint_sub_is_add_neg(a in arb_bigint4(), b in arb_bigint4()) {
        prop_assert_eq!(a - b, a + (-b));
    }

    #[test]
    fn prop_bigint_double_neg(a in arb_bigint4()) {
        prop_assert_eq!(-(-a), a);
    }

    #[test]
    fn prop_bigint_mul_commutative(a in arb_small_bigint4(), b in arb_small_bigint4()) {
        prop_assert_eq!(a * b, b * a);
    }

    #[test]
    fn prop_bigint_mul_identity(a in arb_bigint4()) {
        prop_assert_eq!(a * BigInt::ONE, a);
        prop_assert_eq!(BigInt::ONE * a, a);
    }

    #[test]
    fn prop_bigint_mul_zero(a in arb_bigint4()) {
        prop_assert_eq!(a * BigInt::ZERO, BigInt::ZERO);
    }

    #[test]
    fn prop_bigint_mul_minus_one(a in arb_bigint4()) {
        prop_assert_eq!(a * BigInt::MINUS_ONE, -a);
    }

    #[test]
    fn prop_bigint_distributive(a in arb_small_bigint4(), b in arb_small_bigint4(), c in arb_small_bigint4()) {
        prop_assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn prop_bigint_mul_associative(a in arb_small_bigint4(), b in arb_small_bigint4(), c in arb_small_bigint4()) {
        prop_assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn prop_bigint_abs_nonnegative(a in arb_bigint4()) {
        let abs_a = a.abs();
        // abs(a) is non-negative (sign == 0) unless a is zero.
        prop_assert!(!bool::from(abs_a.is_negative()) || bool::from(abs_a.is_zero()));
    }

    #[test]
    fn prop_bigint_abs_idempotent(a in arb_bigint4()) {
        prop_assert_eq!(a.abs().abs(), a.abs());
    }
}

// Division and GCD properties.
proptest! {
    #[test]
    fn prop_bigint_div_rem_identity(a in arb_small_bigint4(), d in arb_small_bigint4()) {
        // a = q * d + r, with |r| < |d|.
        prop_assume!(!bool::from(d.is_zero()));
        let (q, r) = a.vt_div_rem(&d);
        prop_assert_eq!(q * d + r, a);
    }

    #[test]
    fn prop_bigint_div_rem_remainder_takes_dividend_sign(a in arb_small_bigint4(), d in arb_small_bigint4()) {
        prop_assume!(!bool::from(d.is_zero()));
        let (_, r) = a.vt_div_rem(&d);
        prop_assert!(bool::from(r.is_zero()) || r.is_negative().unwrap_u8() == a.is_negative().unwrap_u8());
        prop_assert!(r.abs() < d.abs());
    }

    #[test]
    fn prop_bigint_vt_mod_non_negative(a in arb_small_bigint4(), d in arb_small_bigint4()) {
        prop_assume!(!bool::from(d.is_zero()));
        let r = a.vt_mod(&d);
        prop_assert!(!bool::from(r.is_negative()));
        prop_assert!(r < d.abs());
        prop_assert_eq!(a.vt_sub(&r).vt_mod(&d), BigInt::ZERO);
    }

    #[test]
    fn prop_bigint_gcd_commutative(a in arb_small_bigint4(), b in arb_small_bigint4()) {
        prop_assert_eq!(a.gcd(&b), b.gcd(&a));
    }

    #[test]
    fn prop_bigint_gcd_divides_both(a in arb_small_bigint4(), b in arb_small_bigint4()) {
        let g = a.gcd(&b);
        if !bool::from(g.is_zero()) {
            let (_, ra) = a.vt_div_rem(&g);
            let (_, rb) = b.vt_div_rem(&g);
            prop_assert!(bool::from(ra.is_zero()), "gcd does not divide a");
            prop_assert!(bool::from(rb.is_zero()), "gcd does not divide b");
        }
    }

    #[test]
    fn prop_bigint_gcd_with_zero(a in arb_small_bigint4()) {
        // gcd(a, 0) = |a|.
        prop_assert_eq!(a.gcd(&BigInt::ZERO), a.abs());
    }

    #[test]
    fn prop_bigint_gcd_idempotent(a in arb_small_bigint4()) {
        // gcd(a, a) = |a|.
        prop_assert_eq!(a.gcd(&a), a.abs());
    }
}

// Shift and valuation properties.
proptest! {
    #[test]
    fn prop_bigint_shl_shr_roundtrip(a in arb_small_bigint4(), k in 0u32..64) {
        // (a << k) >> k == a for small a where no bits are lost.
        let shifted = (a << k) >> k;
        prop_assert_eq!(shifted, a);
    }

    #[test]
    fn prop_bigint_shl_zero(a in arb_bigint4()) {
        prop_assert_eq!(a << 0, a);
    }

    #[test]
    fn prop_bigint_shr_zero(a in arb_bigint4()) {
        prop_assert_eq!(a >> 0, a);
    }

    #[test]
    fn prop_bigint_trailing_zeros_of_power_of_two(k in 1u32..200) {
        // v_2(2^k) = k.
        let val = BigInt::<4>::ONE << k;
        prop_assert_eq!(val.trailing_zeros(), k);
    }

    #[test]
    fn prop_bigint_trailing_zeros_of_odd(a in arb_small_bigint4()) {
        // An odd number has v_2 = 0.
        prop_assume!(!bool::from(a.is_zero()));
        let odd = (a.abs() << 1) + BigInt::ONE; // 2|a| + 1 is always odd
        prop_assert_eq!(odd.trailing_zeros(), 0);
    }

    #[test]
    fn prop_bigint_shr_is_floor_division(a in arb_bigint4(), k in 0u32..=255) {
        // a >> k == floor(a / 2^k): the residue a - (a >> k) * 2^k lies
        // in [0, 2^k). Reconstructed one limb wider so `q << k` cannot
        // wrap for a negative `a` near the width limit.
        let q = (a >> k).widen::<5>();
        let residue = a.widen::<5>().vt_sub(&(q << k));
        prop_assert!(!bool::from(residue.is_negative()));
        prop_assert!(residue < (BigInt::<5>::ONE << k));
    }
}

// Squaring vs multiplication agreement.  ct_sqr exploits cross-term
// symmetry (6 mulx on x86_64 ADX vs 10 for ct_mul) — proves the
// arithmetic agrees with the schoolbook ct_mul on the truncated 4-limb
// output for random and edge-case inputs.
proptest! {
    #[test]
    fn prop_bigint_sqr_matches_mul(a in arb_bigint4()) {
        let via_sqr = a.ct_sqr();
        let via_mul = &a * &a;
        prop_assert_eq!(via_sqr, via_mul);
    }

    #[test]
    fn prop_bigint_square_method_matches_mul(a in arb_bigint4()) {
        prop_assert_eq!(a.square(), &a * &a);
    }

    #[test]
    fn prop_bigint_sqr_negative_input_nonneg(a in arb_bigint4()) {
        // (-a)^2 == a^2, and the result sign is always 0.
        let neg_a = -a;
        let sq = neg_a.square();
        prop_assert_eq!(sq, a.square());
        prop_assert_eq!(sq.sign, 0);
    }
}

// Direct reference oracle for the truncated `mag_mul` across the wide
// widths the lattice multiplies hit (N = 30, 60), where the aarch64
// build routes `ct_mul` through the column-scanning Comba `asm!`. The
// reference is an independent row-scanning u128 schoolbook, so it
// exercises a different code path (scan order, Rust vs asm) than the
// implementation under test. Deterministic splitmix64 inputs plus an
// all-ones edge case stress the carry chain.
#[test]
fn mag_mul_matches_reference_multi_width() {
    fn reference<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let mut r = [0u64; N];
        let mut i = 0;
        while i < N {
            let mut carry: u128 = 0;
            let mut j = 0;
            while j < N - i {
                let t = r[i + j] as u128 + a[i] as u128 * b[j] as u128 + carry;
                r[i + j] = t as u64;
                carry = t >> 64;
                j += 1;
            }
            i += 1;
        }
        r
    }

    fn check<const N: usize>(seed: &mut u64) {
        let mut next = || {
            *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        for _ in 0..200 {
            let a: [u64; N] = core::array::from_fn(|_| next());
            let b: [u64; N] = core::array::from_fn(|_| next());

            let got = BigInt::<N>::from_limbs(a) * BigInt::<N>::from_limbs(b);
            assert_eq!(*got.as_limbs(), reference(&a, &b), "N={N} random mismatch");
        }

        // All-ones * all-ones: maximal carry propagation.
        let ones = [u64::MAX; N];
        let got = BigInt::<N>::from_limbs(ones) * BigInt::<N>::from_limbs(ones);
        assert_eq!(
            *got.as_limbs(),
            reference(&ones, &ones),
            "N={N} all-ones mismatch"
        );
    }

    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    check::<4>(&mut seed);
    check::<5>(&mut seed);
    check::<8>(&mut seed);
    check::<9>(&mut seed);
    check::<30>(&mut seed);
    check::<60>(&mut seed);
}

#[test]
fn bigint_sqr_zero() {
    let z = BigInt::<4>::ZERO;
    assert_eq!(z.square(), z);
}

#[test]
fn bigint_sqr_one() {
    let one = BigInt::<4>::ONE;
    assert_eq!(one.square(), one);
}

#[test]
fn bigint_sqr_small_values() {
    // (2^32)^2 == 2^64, which straddles limb 0 and limb 1.
    let x = BigInt::<4>::from(1u64 << 32);
    let sq = x.square();
    assert_eq!(sq.limbs[0], 0);
    assert_eq!(sq.limbs[1], 1);
    assert_eq!(sq.limbs[2], 0);
    assert_eq!(sq.limbs[3], 0);
}

#[test]
fn bigint_sqr_max_limb_zero() {
    // (2^64 - 1)^2 == 2^128 - 2^65 + 1.
    let x = BigInt::<4> {
        sign: 0,
        limbs: [u64::MAX, 0, 0, 0],
    };
    let sq = x.square();
    let via_mul = &x * &x;
    assert_eq!(sq, via_mul);
    assert_eq!(sq.limbs[0], 1);
    assert_eq!(sq.limbs[1], u64::MAX - 1); // 2^64 - 2
}

/// Generates a `BigInt<N>` with a random significant-limb count (from
/// zero up to `N - 2`) and a random sign, so the [`BigInt::xgcd`] width
/// dispatch is exercised across every rung and the full-width
/// fallthrough.
///
/// The `N - 2` cap leaves the two limbs of headroom the binary xgcd
/// needs for its Bezout cofactors (the intermediate `aa + y` reaches one
/// limb beyond the operands). Real callers always supply that headroom
/// (HNF runs at `W` far wider than its reduced-mod-`D` operands);
/// operands that fill the entire storage width overflow the cofactor
/// regardless of the narrowing, in the original code too.
fn arb_wide_bigint<const N: usize>() -> impl Strategy<Value = BigInt<N>> {
    (
        any::<bool>(),
        0usize..=N.saturating_sub(2),
        prop::collection::vec(any::<u64>(), N),
    )
        .prop_map(|(neg, sig, raw)| {
            let mut limbs = [0u64; N];
            for (i, limb) in raw.iter().enumerate().take(sig) {
                limbs[i] = *limb;
            }

            BigInt::from_sign_and_limbs(u64::from(neg), limbs)
        })
}

/// Like `arb_wide_bigint` but guarantees a nonzero magnitude in exactly
/// `sig` low limbs (`1..=cap`), so the divisor's effective length is
/// controlled. Used to exercise `vt_div_rem` over both the single-limb
/// fast path (`sig == 1`) and the multi-limb Knuth core (`sig > 1`).
fn arb_wide_divisor<const N: usize>(cap: usize) -> impl Strategy<Value = BigInt<N>> {
    (
        any::<bool>(),
        1usize..=cap,
        prop::collection::vec(1u64.., N),
    )
        .prop_map(|(neg, sig, raw)| {
            let mut limbs = [0u64; N];
            for (i, limb) in raw.iter().enumerate().take(sig) {
                limbs[i] = *limb;
            }

            BigInt::from_sign_and_limbs(u64::from(neg), limbs)
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // vt_mul returns exactly what ct_mul does, at every width and
    // occupancy. arb_wide_bigint sweeps the significant-limb count from 0
    // to N-2, covering both the len-tracking short path (low occupancy)
    // and the full-width fallback (high occupancy).
    #[test]
    fn prop_vt_mul_matches_ct_mul_8(a in arb_wide_bigint::<8>(), b in arb_wide_bigint::<8>()) {
        prop_assert_eq!(a.vt_mul(&b), a.ct_mul(&b));
    }

    #[test]
    fn prop_vt_mul_matches_ct_mul_16(a in arb_wide_bigint::<16>(), b in arb_wide_bigint::<16>()) {
        prop_assert_eq!(a.vt_mul(&b), a.ct_mul(&b));
    }

    #[test]
    fn prop_vt_mul_matches_ct_mul_60(a in arb_wide_bigint::<60>(), b in arb_wide_bigint::<60>()) {
        prop_assert_eq!(a.vt_mul(&b), a.ct_mul(&b));
    }

    // vt_add / vt_sub return exactly what ct_add / ct_sub do, at every
    // width and occupancy. arb_wide_bigint sweeps the significant-limb
    // count from 0 to N-2 over both signs, covering the length-bounded
    // short path (low occupancy, both same-sign and differing-sign with
    // either operand the larger), the borrow-propagating two's-complement
    // case, and the full-width fallback (high occupancy).
    #[test]
    fn prop_vt_add_matches_ct_add_8(a in arb_wide_bigint::<8>(), b in arb_wide_bigint::<8>()) {
        prop_assert_eq!(a.vt_add(&b), a.ct_add(&b));
    }

    #[test]
    fn prop_vt_add_matches_ct_add_60(a in arb_wide_bigint::<60>(), b in arb_wide_bigint::<60>()) {
        prop_assert_eq!(a.vt_add(&b), a.ct_add(&b));
    }

    #[test]
    fn prop_vt_sub_matches_ct_sub_8(a in arb_wide_bigint::<8>(), b in arb_wide_bigint::<8>()) {
        prop_assert_eq!(a.vt_sub(&b), a.ct_sub(&b));
    }

    #[test]
    fn prop_vt_sub_matches_ct_sub_60(a in arb_wide_bigint::<60>(), b in arb_wide_bigint::<60>()) {
        prop_assert_eq!(a.vt_sub(&b), a.ct_sub(&b));
    }

    // xgcd narrows wide-N operands to a tight working width before
    // running the binary algorithm; the result must equal running at the
    // full storage width. Cross-check the gcd against the independent
    // full-width `gcd`, and the cofactors against the Bezout identity
    // `a*x + b*y == g` evaluated at double width so the products do not
    // truncate.
    #[test]
    fn prop_xgcd_dispatch_wide_60(a in arb_wide_bigint::<60>(), b in arb_wide_bigint::<60>()) {
        let (g, x, y) = a.xgcd(&b);

        prop_assert_eq!(g, a.gcd(&b));

        let lhs = a
            .widen::<130>()
            .ct_mul(&x.widen::<130>())
            .ct_add(&b.widen::<130>().ct_mul(&y.widen::<130>()));
        prop_assert_eq!(lhs, g.widen::<130>());
    }

    // Wider storage to exercise the 64/128/256-limb rungs and the
    // full-width fallthrough at needed > 256.
    #[test]
    fn prop_xgcd_dispatch_wide_300(a in arb_wide_bigint::<300>(), b in arb_wide_bigint::<300>()) {
        let (g, x, y) = a.xgcd(&b);

        prop_assert_eq!(g, a.gcd(&b));

        let lhs = a
            .widen::<610>()
            .ct_mul(&x.widen::<610>())
            .ct_add(&b.widen::<610>().ct_mul(&y.widen::<610>()));
        prop_assert_eq!(lhs, g.widen::<610>());
    }

    // vt_div_rem's truncating identity at wide N with low-occupancy
    // dividends. arb_wide_bigint sweeps the dividend's significant-limb
    // count from 0 up, exercising the len-tracked normalization short
    // path (m_a < N) alongside the single-limb (sig == 1) and multi-limb
    // (sig > 1) divisor paths. The identity `a == q*b + r` with
    // `|r| < |b|` is checked at double width so `q*b` cannot truncate.
    #[test]
    fn prop_vt_div_rem_truncating_16(
        a in arb_wide_bigint::<16>(),
        b in arb_wide_divisor::<16>(8),
    ) {
        let (q, r) = a.vt_div_rem(&b);

        // |r| < |b|, and r is zero or carries the dividend's sign.
        prop_assert!(r.abs() < b.abs());
        prop_assert!(bool::from(r.is_zero()) || r.is_negative().unwrap_u8() == a.is_negative().unwrap_u8());

        // a == q*b + r, widened so the product does not truncate.
        let lhs = q.widen::<33>().ct_mul(&b.widen::<33>()).ct_add(&r.widen::<33>());
        prop_assert_eq!(lhs, a.widen::<33>());
    }

    #[test]
    fn prop_vt_div_rem_truncating_60(
        a in arb_wide_bigint::<60>(),
        b in arb_wide_divisor::<60>(20),
    ) {
        let (q, r) = a.vt_div_rem(&b);

        prop_assert!(r.abs() < b.abs());
        prop_assert!(bool::from(r.is_zero()) || r.is_negative().unwrap_u8() == a.is_negative().unwrap_u8());

        let lhs = q.widen::<121>().ct_mul(&b.widen::<121>()).ct_add(&r.widen::<121>());
        prop_assert_eq!(lhs, a.widen::<121>());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    // Lehmer gcd must be byte-identical to Stein gcd at every width.
    #[test]
    fn prop_lehmer_gcd_matches_stein_4(a in arb_bigint4(), b in arb_bigint4()) {
        prop_assert_eq!(a.gcd_lehmer(&b), a.gcd_stein(&b));
    }

    #[test]
    fn prop_lehmer_gcd_matches_stein_8(a in arb_wide_bigint::<8>(), b in arb_wide_bigint::<8>()) {
        prop_assert_eq!(a.gcd_lehmer(&b), a.gcd_stein(&b));
    }

    #[test]
    fn prop_lehmer_gcd_matches_stein_30(a in arb_wide_bigint::<30>(), b in arb_wide_bigint::<30>()) {
        prop_assert_eq!(a.gcd_lehmer(&b), a.gcd_stein(&b));
    }

    #[test]
    fn prop_lehmer_gcd_matches_stein_60(a in arb_wide_bigint::<60>(), b in arb_wide_bigint::<60>()) {
        prop_assert_eq!(a.gcd_lehmer(&b), a.gcd_stein(&b));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(600))]

    // is_probable_prime_auto must be transparent: the same decision as the
    // fixed-width is_probable_prime_w it narrows from, for every candidate
    // size (tiny narrows to a small rung; large falls back to WMAX).
    #[test]
    fn prop_prime_auto_matches_fixed_w17(
        sig in 1usize..=16,
        raw in prop::collection::vec(any::<u64>(), 16),
    ) {
        let mut l = [0u64; 16];
        for (i, v) in raw.iter().enumerate().take(sig) {
            l[i] = *v;
        }
        l[0] |= 1;

        let n = BigInt::<16>::from_sign_and_limbs(0, l);
        prop_assert_eq!(n.is_probable_prime_auto::<17>(12), n.is_probable_prime_w::<17>(12));
    }

    #[test]
    fn prop_prime_auto_matches_fixed_w30(
        sig in 1usize..=16,
        raw in prop::collection::vec(any::<u64>(), 16),
    ) {
        let mut l = [0u64; 16];
        for (i, v) in raw.iter().enumerate().take(sig) {
            l[i] = *v;
        }
        l[0] |= 1;

        let n = BigInt::<16>::from_sign_and_limbs(0, l);
        prop_assert_eq!(n.is_probable_prime_auto::<30>(12), n.is_probable_prime_w::<30>(12));
    }
}
