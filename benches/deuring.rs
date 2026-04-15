use sqisign_selkie::quaternions::{
    bigint::BigInt, lattice::LeftIdeal, precomputed::EXTREMAL_ORDERS,
};

fn main() {
    divan::main();
}

#[divan::bench(sample_count = 10)]
fn ideal_to_isogeny(bencher: divan::Bencher) {
    let order = &EXTREMAL_ORDERS[0];
    let norm = BigInt::<4>::from_limbs([0xDEAD_BEEF_CAFE_BABE, 0, 0, 0]);
    if let Some(ideal) = LeftIdeal::<4>::random_prime_norm(&norm, order) {
        bencher
            .with_inputs(|| ideal.clone())
            .bench_values(|i| i.to_isogeny());
    }
}

#[divan::bench(sample_count = 10)]
fn reduce_to_prime_norm(bencher: divan::Bencher) {
    let order = &EXTREMAL_ORDERS[0];
    let norm = BigInt::<4>::from_limbs([0xCAFE_BABE_0000_0001, 0, 0, 0]);
    if let Some(ideal) = LeftIdeal::<4>::random_norm(&norm, order) {
        bencher.with_inputs(|| ideal.clone()).bench_values(|mut i| {
            let mut rng = rand_core::OsRng;
            i.reduce_to_prime_norm::<4, _>(&mut rng);
            i
        });
    }
}

#[divan::bench(sample_count = 10)]
fn represent_integer(bencher: divan::Bencher) {
    let order = EXTREMAL_ORDERS[0].widen::<8>();
    let m = BigInt::<8>::from_limbs([
        0xDEAD_BEEF_CAFE_BAB1,
        0x1234_5678_9ABC_DEF0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]);
    bencher.bench(|| order.represent_integer(divan::black_box(&m), false));
}
