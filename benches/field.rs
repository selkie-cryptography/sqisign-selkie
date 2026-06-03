#[cfg(target_arch = "aarch64")]
use sqisign_selkie::fields::fp::arch::aarch64::{Fp29, Fp29x4};
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
    let a_fp = [
        Fp::from_small(3),
        Fp::from_small(7),
        Fp::from_small(11),
        Fp::from_small(13),
    ];
    let b_fp = [
        Fp::from_small(17),
        Fp::from_small(19),
        Fp::from_small(23),
        Fp::from_small(29),
    ];
    let a29 = [
        Fp29::from(a_fp[0]),
        Fp29::from(a_fp[1]),
        Fp29::from(a_fp[2]),
        Fp29::from(a_fp[3]),
    ];
    let b29 = [
        Fp29::from(b_fp[0]),
        Fp29::from(b_fp[1]),
        Fp29::from(b_fp[2]),
        Fp29::from(b_fp[3]),
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
    let a_fp = [
        Fp::from_small(3),
        Fp::from_small(7),
        Fp::from_small(11),
        Fp::from_small(13),
    ];
    let a29 = [
        Fp29::from(a_fp[0]),
        Fp29::from(a_fp[1]),
        Fp29::from(a_fp[2]),
        Fp29::from(a_fp[3]),
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
    let a = Fp29::from(Fp::from_small(17));
    let b = Fp29::from(Fp::from_small(42));
    bencher.bench(|| divan::black_box(&a).mul(divan::black_box(&b)));
}
