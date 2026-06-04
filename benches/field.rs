#[cfg(target_arch = "aarch64")]
use sqisign_selkie::fields::fp::arch::aarch64::neon::{Fp29, Fp29x4};
#[cfg(target_arch = "x86_64")]
use sqisign_selkie::fields::fp::arch::x86_64::avx2::Fp26;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use sqisign_selkie::fields::fp::arch::x86_64::avx2::Fp26x4;
use sqisign_selkie::fields::{fp::Fp, fp2::Fp2};

fn main() {
    divan::main();
}

#[divan::bench]
fn fp_mul(bencher: divan::Bencher) {
    let a = Fp::from_small(17);
    let b = Fp::from_small(42);
    bencher.bench(|| divan::black_box(a) * divan::black_box(b));
}

#[divan::bench]
fn fp_square(bencher: divan::Bencher) {
    let a = Fp::from_small(17);
    bencher.bench(|| divan::black_box(&a).square());
}

#[divan::bench]
fn fp_invert(bencher: divan::Bencher) {
    let a = Fp::from_small(17);
    bencher.bench(|| divan::black_box(&a).invert());
}

#[divan::bench]
fn fp_sqrt(bencher: divan::Bencher) {
    let a = Fp::from_small(17).square();
    bencher.bench(|| divan::black_box(&a).sqrt());
}

#[divan::bench]
fn fp2_mul(bencher: divan::Bencher) {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let b = Fp2::new(Fp::from_small(11), Fp::from_small(13));
    bencher.bench(|| divan::black_box(a) * divan::black_box(b));
}

/// Baseline pair: two independent `Fp²::mul` calls feeding an add.
/// Pairs with [`fp2_sum_of_2_products`] to measure the fused-reduction
/// win at the Fp² level (`a*b + c*d` = 2 Fp² muls + 1 add = 4 reductions,
/// vs the t=4 fused path's 2 reductions).
#[divan::bench]
fn fp2_mul_pair_then_add(bencher: divan::Bencher) {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let b = Fp2::new(Fp::from_small(11), Fp::from_small(13));
    let c = Fp2::new(Fp::from_small(17), Fp::from_small(19));
    let d = Fp2::new(Fp::from_small(23), Fp::from_small(29));
    bencher.bench(|| {
        (divan::black_box(&a) * divan::black_box(&b))
            + (divan::black_box(&c) * divan::black_box(&d))
    });
}

/// `Fp²::sum_of_2_products(a, b, c, d) = a*b + c*d` via the t=4 fused
/// Fp call.  Compare against [`fp2_mul_pair_then_add`].
#[divan::bench]
fn fp2_sum_of_2_products(bencher: divan::Bencher) {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let b = Fp2::new(Fp::from_small(11), Fp::from_small(13));
    let c = Fp2::new(Fp::from_small(17), Fp::from_small(19));
    let d = Fp2::new(Fp::from_small(23), Fp::from_small(29));
    bencher.bench(|| {
        Fp2::sum_of_2_products(
            divan::black_box(&a),
            divan::black_box(&b),
            divan::black_box(&c),
            divan::black_box(&d),
        )
    });
}

/// `Fp::sum_of_6_products` direct measurement — six fused Fp products,
/// one Mont reduction.
#[divan::bench]
fn fp_sum_of_6_products(bencher: divan::Bencher) {
    let a = Fp::from_small(3);
    let b = Fp::from_small(7);
    let c = Fp::from_small(11);
    let d = Fp::from_small(13);
    let e = Fp::from_small(17);
    let f = Fp::from_small(19);
    let g = Fp::from_small(23);
    let h = Fp::from_small(29);
    let i = Fp::from_small(31);
    let j = Fp::from_small(37);
    let k = Fp::from_small(41);
    let l = Fp::from_small(43);
    bencher.bench(|| {
        Fp::sum_of_6_products([
            (divan::black_box(&a), divan::black_box(&b)),
            (divan::black_box(&c), divan::black_box(&d)),
            (divan::black_box(&e), divan::black_box(&f)),
            (divan::black_box(&g), divan::black_box(&h)),
            (divan::black_box(&i), divan::black_box(&j)),
            (divan::black_box(&k), divan::black_box(&l)),
        ])
    });
}

/// Naive baseline for t=6: six independent `Fp::mul`s and five adds.
#[divan::bench]
fn fp_six_muls_naive(bencher: divan::Bencher) {
    let a = Fp::from_small(3);
    let b = Fp::from_small(7);
    let c = Fp::from_small(11);
    let d = Fp::from_small(13);
    let e = Fp::from_small(17);
    let f = Fp::from_small(19);
    let g = Fp::from_small(23);
    let h = Fp::from_small(29);
    let i = Fp::from_small(31);
    let j = Fp::from_small(37);
    let k = Fp::from_small(41);
    let l = Fp::from_small(43);
    bencher.bench(|| {
        let p0 = divan::black_box(&a) * divan::black_box(&b);
        let p1 = divan::black_box(&c) * divan::black_box(&d);
        let p2 = divan::black_box(&e) * divan::black_box(&f);
        let p3 = divan::black_box(&g) * divan::black_box(&h);
        let p4 = divan::black_box(&i) * divan::black_box(&j);
        let p5 = divan::black_box(&k) * divan::black_box(&l);
        p0 + p1 + p2 + p3 + p4 + p5
    });
}

/// `Fp²::sum_of_3_products(a, b, c, d, e, f) = a*b + c*d + e*f` via the
/// t=6 fused Fp call.  Pair with [`fp2_three_muls_naive`] to measure
/// the 4 saved reductions per call.
#[divan::bench]
fn fp2_sum_of_3_products(bencher: divan::Bencher) {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let b = Fp2::new(Fp::from_small(11), Fp::from_small(13));
    let c = Fp2::new(Fp::from_small(17), Fp::from_small(19));
    let d = Fp2::new(Fp::from_small(23), Fp::from_small(29));
    let e = Fp2::new(Fp::from_small(31), Fp::from_small(37));
    let f = Fp2::new(Fp::from_small(41), Fp::from_small(43));
    bencher.bench(|| {
        Fp2::sum_of_3_products(
            divan::black_box(&a),
            divan::black_box(&b),
            divan::black_box(&c),
            divan::black_box(&d),
            divan::black_box(&e),
            divan::black_box(&f),
        )
    });
}

/// Naive baseline: three independent `Fp²::mul`s feeding two adds.
#[divan::bench]
fn fp2_three_muls_naive(bencher: divan::Bencher) {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let b = Fp2::new(Fp::from_small(11), Fp::from_small(13));
    let c = Fp2::new(Fp::from_small(17), Fp::from_small(19));
    let d = Fp2::new(Fp::from_small(23), Fp::from_small(29));
    let e = Fp2::new(Fp::from_small(31), Fp::from_small(37));
    let f = Fp2::new(Fp::from_small(41), Fp::from_small(43));
    bencher.bench(|| {
        let p0 = divan::black_box(&a) * divan::black_box(&b);
        let p1 = divan::black_box(&c) * divan::black_box(&d);
        let p2 = divan::black_box(&e) * divan::black_box(&f);
        p0 + p1 + p2
    });
}

/// `Fp::sum_of_4_products` direct measurement — four fused Fp products,
/// one Mont reduction.
#[divan::bench]
fn fp_sum_of_4_products(bencher: divan::Bencher) {
    let a = Fp::from_small(3);
    let b = Fp::from_small(7);
    let c = Fp::from_small(11);
    let d = Fp::from_small(13);
    let e = Fp::from_small(17);
    let f = Fp::from_small(19);
    let g = Fp::from_small(23);
    let h = Fp::from_small(29);
    bencher.bench(|| {
        Fp::sum_of_4_products([
            (divan::black_box(&a), divan::black_box(&b)),
            (divan::black_box(&c), divan::black_box(&d)),
            (divan::black_box(&e), divan::black_box(&f)),
            (divan::black_box(&g), divan::black_box(&h)),
        ])
    });
}

#[divan::bench]
fn fp2_square(bencher: divan::Bencher) {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    bencher.bench(|| divan::black_box(&a).square());
}

#[divan::bench]
fn fp2_invert(bencher: divan::Bencher) {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    bencher.bench(|| divan::black_box(&a).invert());
}

#[divan::bench]
fn fp2_sqrt(bencher: divan::Bencher) {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7)).square();
    bencher.bench(|| divan::black_box(&a).sqrt());
}

/// Four independent scalar `Fp::mul`s — the baseline the NEON path must beat.
#[cfg(target_arch = "aarch64")]
#[divan::bench]
fn fp_mul_4_independent(bencher: divan::Bencher) {
    let a = [
        Fp::from_small(3),
        Fp::from_small(7),
        Fp::from_small(11),
        Fp::from_small(13),
    ];
    let b = [
        Fp::from_small(17),
        Fp::from_small(19),
        Fp::from_small(23),
        Fp::from_small(29),
    ];
    bencher.bench(|| {
        let aa = divan::black_box(&a);
        let bb = divan::black_box(&b);
        [aa[0] * bb[0], aa[1] * bb[1], aa[2] * bb[2], aa[3] * bb[3]]
    });
}

/// One vectorised `Fp29x4::mul` — computes four independent products in one
/// NEON Karatsuba-decomposed Montgomery multiplication.  Compare against
/// `fp_mul_4_independent` for the scalar-vs-NEON crossover.
#[cfg(target_arch = "aarch64")]
#[divan::bench]
fn fp29x4_mul_neon(bencher: divan::Bencher) {
    let a29 = [
        Fp29::from_small(3),
        Fp29::from_small(7),
        Fp29::from_small(11),
        Fp29::from_small(13),
    ];
    let b29 = [
        Fp29::from_small(17),
        Fp29::from_small(19),
        Fp29::from_small(23),
        Fp29::from_small(29),
    ];
    let a4 = Fp29x4::from_scalars(&a29);
    let b4 = Fp29x4::from_scalars(&b29);
    bencher.bench(|| divan::black_box(&a4).mul(divan::black_box(&b4)));
}

/// Four independent scalar `Fp::square` calls — the baseline for the
/// vectorised square path.
#[cfg(target_arch = "aarch64")]
#[divan::bench]
fn fp_square_4_independent(bencher: divan::Bencher) {
    let a = [
        Fp::from_small(3),
        Fp::from_small(7),
        Fp::from_small(11),
        Fp::from_small(13),
    ];
    bencher.bench(|| {
        let aa = divan::black_box(&a);
        [
            aa[0].square(),
            aa[1].square(),
            aa[2].square(),
            aa[3].square(),
        ]
    });
}

/// Vectorised `Fp29x4::square` — Karatsuba structure with symmetric
/// sub-squares; cross-terms doubled via `vshlq_n_u64::<1>` rather than
/// the two `vmlal_u32` calls a straight mul would do.
#[cfg(target_arch = "aarch64")]
#[divan::bench]
fn fp29x4_square_neon(bencher: divan::Bencher) {
    let a29 = [
        Fp29::from_small(3),
        Fp29::from_small(7),
        Fp29::from_small(11),
        Fp29::from_small(13),
    ];
    let a4 = Fp29x4::from_scalars(&a29);
    bencher.bench(|| divan::black_box(&a4).square());
}

/// Standalone scalar `Fp29::mul` — for comparing radix-29 vs radix-51 cost
/// at the single-product level, isolating the radix change from the NEON
/// vectorisation factor.
#[cfg(target_arch = "aarch64")]
#[divan::bench]
fn fp29_mul_scalar(bencher: divan::Bencher) {
    let a = Fp29::from_small(17);
    let b = Fp29::from_small(42);
    bencher.bench(|| divan::black_box(&a) * divan::black_box(&b));
}

/// Four independent scalar `Fp::mul`s — the x86_64 baseline the AVX2
/// Fp26x4 path must beat.
#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn fp_mul_4_independent_x86_64(bencher: divan::Bencher) {
    let a = [
        Fp::from_small(3),
        Fp::from_small(7),
        Fp::from_small(11),
        Fp::from_small(13),
    ];
    let b = [
        Fp::from_small(17),
        Fp::from_small(19),
        Fp::from_small(23),
        Fp::from_small(29),
    ];
    bencher.bench(|| {
        let aa = divan::black_box(&a);
        let bb = divan::black_box(&b);
        [aa[0] * bb[0], aa[1] * bb[1], aa[2] * bb[2], aa[3] * bb[3]]
    });
}

/// Four independent scalar `Fp::square` calls — x86_64 baseline.
#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn fp_square_4_independent_x86_64(bencher: divan::Bencher) {
    let a = [
        Fp::from_small(3),
        Fp::from_small(7),
        Fp::from_small(11),
        Fp::from_small(13),
    ];
    bencher.bench(|| {
        let aa = divan::black_box(&a);
        [
            aa[0].square(),
            aa[1].square(),
            aa[2].square(),
            aa[3].square(),
        ]
    });
}

/// Standalone scalar `Fp26::mul` — for comparing radix-26 vs radix-51
/// cost at the single-product level, isolating the radix change from
/// the AVX2 vectorisation factor.
#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn fp26_mul_scalar(bencher: divan::Bencher) {
    let a = Fp26::from_small(17);
    let b = Fp26::from_small(42);
    bencher.bench(|| divan::black_box(&a) * divan::black_box(&b));
}

/// One vectorised `Fp26x4::mul` — four independent Montgomery products
/// in one AVX2 schoolbook CIOS.  Compare against
/// `fp_mul_4_independent_x86_64` for the scalar-vs-AVX2 crossover.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[divan::bench]
fn fp26x4_mul_avx2(bencher: divan::Bencher) {
    let a26 = [
        Fp26::from_small(3),
        Fp26::from_small(7),
        Fp26::from_small(11),
        Fp26::from_small(13),
    ];
    let b26 = [
        Fp26::from_small(17),
        Fp26::from_small(19),
        Fp26::from_small(23),
        Fp26::from_small(29),
    ];
    let a4 = Fp26x4::from_scalars(&a26);
    let b4 = Fp26x4::from_scalars(&b26);
    bencher.bench(|| divan::black_box(&a4).mul(divan::black_box(&b4)));
}

/// Vectorised `Fp26x4::square` — currently delegates to `mul`, baseline
/// for the future symmetric-cross-term optimisation.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[divan::bench]
fn fp26x4_square_avx2(bencher: divan::Bencher) {
    let a26 = [
        Fp26::from_small(3),
        Fp26::from_small(7),
        Fp26::from_small(11),
        Fp26::from_small(13),
    ];
    let a4 = Fp26x4::from_scalars(&a26);
    bencher.bench(|| divan::black_box(&a4).square());
}
