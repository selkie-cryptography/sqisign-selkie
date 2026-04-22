use sqisign_selkie::quaternions::{
    algebra::{Coordinate, Denominator, Element},
    bigint::BigInt,
    lattice::{Lattice, LeftIdeal},
    precomputed::EXTREMAL_ORDERS,
};

fn main() {
    divan::main();
}

fn sample_element() -> Element<4> {
    Element::new(
        Coordinate::from_limbs([0xDEAD_BEEF, 0x1234_5678, 0, 0]),
        Coordinate::from_limbs([0xCAFE_BABE, 0x9ABC_DEF0, 0, 0]),
        Coordinate::from_limbs([0x1111_2222, 0x3333_4444, 0, 0]),
        Coordinate::from_limbs([0x5555_6666, 0x7777_8888, 0, 0]),
        Denominator::ONE,
    )
}

fn sample_element_b() -> Element<4> {
    Element::new(
        Coordinate::from_limbs([0xAAAA_BBBB, 0xCCCC_DDDD, 0, 0]),
        Coordinate::from_limbs([0xEEEE_FFFF, 0x0000_1111, 0, 0]),
        Coordinate::from_limbs([0x2222_3333, 0x4444_5555, 0, 0]),
        Coordinate::from_limbs([0x6666_7777, 0x8888_9999, 0, 0]),
        Denominator::ONE,
    )
}

// --- Element operations ---

#[divan::bench]
fn element_mul(bencher: divan::Bencher) {
    let a = sample_element();
    let b = sample_element_b();
    bencher.bench(|| divan::black_box(&a).mul(divan::black_box(&b)));
}

#[divan::bench]
fn element_norm(bencher: divan::Bencher) {
    let a = sample_element();
    bencher.bench(|| divan::black_box(&a).norm());
}

#[divan::bench]
fn element_conjugate(bencher: divan::Bencher) {
    let a = sample_element();
    bencher.bench(|| divan::black_box(&a).conjugate());
}

#[divan::bench]
fn element_trace(bencher: divan::Bencher) {
    let a = sample_element();
    bencher.bench(|| divan::black_box(&a).trace());
}

#[divan::bench]
fn element_scalar_mul(bencher: divan::Bencher) {
    let a = sample_element();
    let s = BigInt::<4>::from_limbs([0xDEAD_BEEF_CAFE_BABE, 0x1234, 0, 0]);
    bencher.bench(|| divan::black_box(&a).scalar_mul(divan::black_box(&s)));
}

#[divan::bench]
fn element_normalize(bencher: divan::Bencher) {
    let a = Element::new(
        Coordinate::from_limbs([0xDEAD_BEEF_0000_0002, 0x1234_5678, 0, 0]),
        Coordinate::from_limbs([0xCAFE_BABE_0000_0004, 0x9ABC_DEF0, 0, 0]),
        Coordinate::from_limbs([0x1111_2222_0000_0006, 0x3333_4444, 0, 0]),
        Coordinate::from_limbs([0x5555_6666_0000_0008, 0x7777_8888, 0, 0]),
        Denominator::TWO,
    );
    bencher.with_inputs(|| a).bench_values(|mut e| {
        e.normalize();
        e
    });
}

// --- Lattice operations ---

#[divan::bench]
fn lattice_intersection(bencher: divan::Bencher) {
    let lat1: Lattice<4> = *EXTREMAL_ORDERS[0].order().lattice();
    let lat2: Lattice<4> = *EXTREMAL_ORDERS[1].order().lattice();
    bencher.bench(|| divan::black_box(&lat1).intersection(divan::black_box(&lat2)));
}

#[divan::bench]
fn lattice_product(bencher: divan::Bencher) {
    let lat1: Lattice<4> = *EXTREMAL_ORDERS[0].order().lattice();
    let lat2: Lattice<4> = *EXTREMAL_ORDERS[1].order().lattice();
    bencher.bench(|| divan::black_box(&lat1).product(divan::black_box(&lat2)));
}

#[divan::bench]
fn lattice_decompose(bencher: divan::Bencher) {
    let lat: Lattice<4> = *EXTREMAL_ORDERS[0].order().lattice();
    let elem = sample_element();
    bencher.bench(|| divan::black_box(&lat).decompose(divan::black_box(&elem)));
}

#[divan::bench]
fn lattice_conjugate(bencher: divan::Bencher) {
    let lat: Lattice<4> = *EXTREMAL_ORDERS[0].order().lattice();
    bencher.bench(|| divan::black_box(&lat).conjugate());
}

// --- LeftIdeal operations ---

#[divan::bench(sample_count = 10)]
fn ideal_random_prime_norm(bencher: divan::Bencher) {
    let order = &EXTREMAL_ORDERS[0];
    let norm = BigInt::<4>::from_limbs([0xDEAD_BEEF_CAFE_BAB1, 0, 0, 0]);
    bencher.bench(|| LeftIdeal::<4>::random_prime_norm(&norm, order));
}

#[divan::bench]
fn ideal_inverse(bencher: divan::Bencher) {
    let order = &EXTREMAL_ORDERS[0];
    let norm = BigInt::<4>::from_limbs([0xDEAD_BEEF_CAFE_BAB1, 0, 0, 0]);
    if let Some(ideal) = LeftIdeal::<4>::random_prime_norm(&norm, order) {
        bencher.bench(|| divan::black_box(&ideal).inverse());
    }
}

#[divan::bench]
fn ideal_right_order(bencher: divan::Bencher) {
    let order = &EXTREMAL_ORDERS[0];
    let norm = BigInt::<4>::from_limbs([0xDEAD_BEEF_CAFE_BAB1, 0, 0, 0]);
    if let Some(ideal) = LeftIdeal::<4>::random_prime_norm(&norm, order) {
        bencher.bench(|| divan::black_box(&ideal).right_order());
    }
}

#[divan::bench]
fn ideal_generator(bencher: divan::Bencher) {
    let order = &EXTREMAL_ORDERS[0];
    let norm = BigInt::<4>::from_limbs([0xDEAD_BEEF_CAFE_BAB1, 0, 0, 0]);
    if let Some(ideal) = LeftIdeal::<4>::random_prime_norm(&norm, order) {
        bencher.bench(|| divan::black_box(&ideal).generator());
    }
}
