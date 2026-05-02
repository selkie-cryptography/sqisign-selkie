use sqisign_selkie::quaternions::bigint::BigInt;

fn main() {
    divan::main();
}

fn sample_a() -> BigInt<4> {
    BigInt::from_limbs([
        0xDEAD_BEEF_CAFE_BABE,
        0x1234_5678_9ABC_DEF0,
        0xFEDC_BA98_7654_3210,
        0x0123_4567_89AB_CDEF,
    ])
}

fn sample_b() -> BigInt<4> {
    BigInt::from_limbs([
        0xAAAA_BBBB_CCCC_DDDD,
        0x1111_2222_3333_4444,
        0x5555_6666_7777_8888,
        0x0000_0000_0000_0003,
    ])
}

fn modulus_192() -> BigInt<4> {
    BigInt::from_limbs([
        0xFFFF_FFFF_FFFF_FFC5,
        0xFFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF,
        0x0000_0000_0000_0000,
    ])
}

#[divan::bench]
fn add(bencher: divan::Bencher) {
    let a = sample_a();
    let b = sample_b();
    bencher.bench(|| divan::black_box(a) + divan::black_box(b));
}

#[divan::bench]
fn sub(bencher: divan::Bencher) {
    let a = sample_a();
    let b = sample_b();
    bencher.bench(|| divan::black_box(a) - divan::black_box(b));
}

#[divan::bench]
fn mul(bencher: divan::Bencher) {
    let a = sample_a();
    let b = sample_b();
    bencher.bench(|| divan::black_box(a) * divan::black_box(b));
}

#[divan::bench]
fn div_rem(bencher: divan::Bencher) {
    let a = sample_a();
    let b = BigInt::<4>::from_limbs([0x1234_5678, 0, 0, 0]);
    bencher.bench(|| divan::black_box(&a).div_rem(divan::black_box(&b)));
}

#[divan::bench]
fn gcd(bencher: divan::Bencher) {
    let a = sample_a();
    let b = sample_b();
    bencher.bench(|| divan::black_box(&a).gcd(divan::black_box(&b)));
}

#[divan::bench]
fn xgcd(bencher: divan::Bencher) {
    let a = sample_a();
    let b = sample_b();
    bencher.bench(|| divan::black_box(&a).xgcd(divan::black_box(&b)));
}

#[divan::bench]
fn pow_mod(bencher: divan::Bencher) {
    let base = sample_a();
    let modulus = modulus_192();
    let exp = sample_b();
    bencher.bench(|| {
        BigInt::pow_mod(
            divan::black_box(&base),
            divan::black_box(&exp),
            divan::black_box(&modulus),
        )
    });
}

#[divan::bench]
fn is_probable_prime(bencher: divan::Bencher) {
    let n = modulus_192();
    bencher.bench(|| divan::black_box(&n).is_probable_prime(8));
}

#[divan::bench]
fn cornacchia(bencher: divan::Bencher) {
    let q = BigInt::<4>::from_limbs([3, 0, 0, 0]);
    let m = modulus_192();
    bencher.bench(|| BigInt::cornacchia(divan::black_box(&q), divan::black_box(&m)));
}

#[divan::bench]
fn sqrt_floor(bencher: divan::Bencher) {
    let a = sample_a();
    bencher.bench(|| divan::black_box(&a).sqrt_floor());
}

#[divan::bench]
fn invert_mod(bencher: divan::Bencher) {
    let a = sample_a();
    let modulus = modulus_192();
    bencher.bench(|| divan::black_box(&a).invert_mod(divan::black_box(&modulus)));
}

#[divan::bench]
fn mul_wide(bencher: divan::Bencher) {
    let a = BigInt::<8>::from_limbs([
        0xDEAD_BEEF,
        0x1234_5678,
        0xFEDC_BA98,
        0x0123_4567,
        0xAAAA_BBBB,
        0xCCCC_DDDD,
        0x1111_2222,
        0x3333_4444,
    ]);
    let b = BigInt::<8>::from_limbs([
        0xCAFE_BABE,
        0x9ABC_DEF0,
        0x7654_3210,
        0x89AB_CDEF,
        0x5555_6666,
        0x7777_8888,
        0x9999_AAAA,
        0xBBBB_CCCC,
    ]);
    bencher.bench(|| divan::black_box(a) * divan::black_box(b));
}

#[divan::bench]
fn div_rem_wide(bencher: divan::Bencher) {
    let a = BigInt::<8>::from_limbs([
        0xDEAD_BEEF,
        0x1234_5678,
        0xFEDC_BA98,
        0x0123_4567,
        0xAAAA_BBBB,
        0xCCCC_DDDD,
        0x1111_2222,
        0x3333_4444,
    ]);
    let b = BigInt::<8>::from_limbs([
        0xCAFE_BABE,
        0x9ABC_DEF0,
        0x7654_3210,
        0x89AB_CDEF,
        0,
        0,
        0,
        0,
    ]);
    bencher.bench(|| divan::black_box(&a).div_rem(divan::black_box(&b)));
}

// `BigInt<8>` is the dominant width on the signing hot path (quaternion
// `wide()` intermediates, ideal `_w` arithmetic, Miller-Rabin candidates
// during prime-norm sampling). `BigInt<30>` shows up in the response
// phase intersection. gcd / xgcd / invert_mod scale O(BITS²·N) under
// the current Euclidean impl, so the wider widths are where the binary
// GCD payoff lands.

fn sample_a_8() -> BigInt<8> {
    BigInt::from_limbs([
        0xDEAD_BEEF_CAFE_BABE,
        0x1234_5678_9ABC_DEF0,
        0xFEDC_BA98_7654_3210,
        0x0123_4567_89AB_CDEF,
        0xAAAA_BBBB_CCCC_DDDD,
        0x1111_2222_3333_4444,
        0x5555_6666_7777_8888,
        0x0123_4567_89AB_CDEF,
    ])
}

fn sample_b_8() -> BigInt<8> {
    BigInt::from_limbs([
        0xAAAA_BBBB_CCCC_DDDD,
        0x1111_2222_3333_4444,
        0x5555_6666_7777_8888,
        0x9999_AAAA_BBBB_CCCC,
        0xDEAD_BEEF_CAFE_BABE,
        0x1234_5678_9ABC_DEF0,
        0xFEDC_BA98_7654_3210,
        0x0000_0000_0000_0007,
    ])
}

fn modulus_512() -> BigInt<8> {
    // 2^511 - 187 (prime). Top limb 0x7FFF... so the widening intermediates
    // in modular ops don't overflow.
    BigInt::from_limbs([
        0xFFFF_FFFF_FFFF_FF45,
        0xFFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF,
        0x7FFF_FFFF_FFFF_FFFF,
    ])
}

#[divan::bench]
fn gcd_wide(bencher: divan::Bencher) {
    let a = sample_a_8();
    let b = sample_b_8();
    bencher.bench(|| divan::black_box(&a).gcd(divan::black_box(&b)));
}

#[divan::bench]
fn xgcd_wide(bencher: divan::Bencher) {
    let a = sample_a_8();
    let b = sample_b_8();
    bencher.bench(|| divan::black_box(&a).xgcd(divan::black_box(&b)));
}

#[divan::bench]
fn invert_mod_wide(bencher: divan::Bencher) {
    let a = sample_a_8();
    let m = modulus_512();
    bencher.bench(|| divan::black_box(&a).invert_mod(divan::black_box(&m)));
}

#[divan::bench(sample_count = 20)]
fn pow_mod_wide(bencher: divan::Bencher) {
    let base = sample_a_8();
    let exp = sample_b_8();
    let m = modulus_512();
    bencher.bench(|| {
        BigInt::pow_mod(
            divan::black_box(&base),
            divan::black_box(&exp),
            divan::black_box(&m),
        )
    });
}

#[divan::bench(sample_count = 10)]
fn is_probable_prime_wide(bencher: divan::Bencher) {
    let n = modulus_512();
    bencher.bench(|| divan::black_box(&n).is_probable_prime(8));
}

// `BigInt<30>` width appears in the response-phase intersection
// (`lattice.rs` ~line 2335). Single bench to track scaling at the
// extreme end; the response phase is where slow gcd hurts most.
fn sample_a_30() -> BigInt<30> {
    let mut limbs = [0u64; 30];
    for (i, l) in limbs.iter_mut().enumerate() {
        *l = 0xDEAD_BEEF_CAFE_BABE ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    BigInt::from_limbs(limbs)
}

fn sample_b_30() -> BigInt<30> {
    let mut limbs = [0u64; 30];
    for (i, l) in limbs.iter_mut().enumerate() {
        *l = 0xAAAA_BBBB_CCCC_DDDD ^ (i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    }
    // Force odd so gcd has something to chew on.
    limbs[0] |= 1;
    BigInt::from_limbs(limbs)
}

#[divan::bench(sample_count = 5)]
fn gcd_30(bencher: divan::Bencher) {
    let a = sample_a_30();
    let b = sample_b_30();
    bencher.bench(|| divan::black_box(&a).gcd(divan::black_box(&b)));
}
