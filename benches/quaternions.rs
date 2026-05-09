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

// --- Element operations ---

// Wider samples for `mul_direct` benching. The narrow `Element<4>::mul`
// requires `|coord| · p < 2^256` (i.e. coords < ~2^4), which is
// uncharacteristic of real signing-side callers; they widen to
// `Element<N>` with N large enough that `p · coord^2` fits in
// `BigInt<N>` (W ≥ 7 for ~2^96 coords; production uses Element<18> on
// the response phase). Use the same magnitude inputs as the narrow
// samples, just zero-extended to 8 limbs.
fn sample_element_wide() -> Element<8> {
    Element::new(
        Coordinate::from_limbs([0xDEAD_BEEF, 0x1234_5678, 0, 0, 0, 0, 0, 0]),
        Coordinate::from_limbs([0xCAFE_BABE, 0x9ABC_DEF0, 0, 0, 0, 0, 0, 0]),
        Coordinate::from_limbs([0x1111_2222, 0x3333_4444, 0, 0, 0, 0, 0, 0]),
        Coordinate::from_limbs([0x5555_6666, 0x7777_8888, 0, 0, 0, 0, 0, 0]),
        Denominator::ONE,
    )
}

fn sample_element_b_wide() -> Element<8> {
    Element::new(
        Coordinate::from_limbs([0xAAAA_BBBB, 0xCCCC_DDDD, 0, 0, 0, 0, 0, 0]),
        Coordinate::from_limbs([0xEEEE_FFFF, 0x0000_1111, 0, 0, 0, 0, 0, 0]),
        Coordinate::from_limbs([0x2222_3333, 0x4444_5555, 0, 0, 0, 0, 0, 0]),
        Coordinate::from_limbs([0x6666_7777, 0x8888_9999, 0, 0, 0, 0, 0, 0]),
        Denominator::ONE,
    )
}

#[divan::bench]
fn element_mul(bencher: divan::Bencher) {
    let a = sample_element_wide();
    let b = sample_element_b_wide();
    bencher.bench(|| divan::black_box(&a).mul_direct(divan::black_box(&b)));
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

/// Construct `O·⟨α, N⟩` from a quaternion generator + norm. Hot in
/// keygen (response-ideal construction) and sign.
#[divan::bench]
fn ideal_from_generator(bencher: divan::Bencher) {
    let order = EXTREMAL_ORDERS[0].order();
    let alpha = sample_element();
    let norm = BigInt::<4>::from_limbs([0xDEAD_BEEF_CAFE_BAB1, 0, 0, 0]);
    bencher.bench(|| {
        LeftIdeal::<4>::from_generator(
            divan::black_box(&alpha),
            divan::black_box(&norm),
            divan::black_box(order),
        )
    });
}

// --- Hot-path primitives ---
//
// `intersection_via_kernel` is a major sign-side cost; the working
// width `W` is the dominant lever. The three explicit-W variants
// below mirror the three production call sites:
//
//   - `W = 60`  — sub-step of `sample_from_ball` (gram inputs).
//   - `W = 120` — refresh_norm on intersected ideals.
//   - `W = 150` — the response-phase intersection in `sign_with_rng`.
//
// Const generics can't take a runtime argument, so each width gets
// its own divan bench.

#[divan::bench(sample_count = 30)]
fn intersection_via_kernel_w60(bencher: divan::Bencher) {
    let lat1: Lattice<4> = *EXTREMAL_ORDERS[0].order().lattice();
    let lat2: Lattice<4> = *EXTREMAL_ORDERS[1].order().lattice();
    bencher
        .bench(|| divan::black_box(&lat1).intersection_via_kernel::<60>(divan::black_box(&lat2)));
}

#[divan::bench(sample_count = 20)]
fn intersection_via_kernel_w120(bencher: divan::Bencher) {
    let lat1: Lattice<4> = *EXTREMAL_ORDERS[0].order().lattice();
    let lat2: Lattice<4> = *EXTREMAL_ORDERS[1].order().lattice();
    bencher
        .bench(|| divan::black_box(&lat1).intersection_via_kernel::<120>(divan::black_box(&lat2)));
}

#[divan::bench(sample_count = 10)]
fn intersection_via_kernel_w150(bencher: divan::Bencher) {
    let lat1: Lattice<4> = *EXTREMAL_ORDERS[0].order().lattice();
    let lat2: Lattice<4> = *EXTREMAL_ORDERS[1].order().lattice();
    bencher
        .bench(|| divan::black_box(&lat1).intersection_via_kernel::<150>(divan::black_box(&lat2)));
}

/// `sample_from_ball` — the dual-LLL sampling at the heart of the
/// response phase. The radius drives most of the cost; benchmark at
/// a sub-production size (~256 bits) for tolerable bench time. The
/// production response radius is closer to ~600 bits.
#[divan::bench(sample_count = 10)]
fn sample_from_ball(bencher: divan::Bencher) {
    let lat: Lattice<4> = *EXTREMAL_ORDERS[0].order().lattice();
    // 256-bit radius: 2^255 + 1 (any nontrivial value of about that
    // magnitude exercises the dual-LLL setup + sampling loop).
    let mut radius_limbs = [0u64; 4];
    radius_limbs[3] = 1u64 << 63;
    let radius = BigInt::<4>::from_limbs(radius_limbs);
    bencher.bench(|| {
        let mut rng = rand_core::OsRng;
        divan::black_box(&lat).sample_from_ball::<8, _>(divan::black_box(&radius), &mut rng)
    });
}

/// `refresh_norm` — recovers the ideal's stored norm from its
/// lattice covolume. Hot in sign (called twice per response-phase
/// iteration) and cheap relative to the intersections that produce
/// the ideal it's called on.
#[divan::bench]
fn ideal_refresh_norm(bencher: divan::Bencher) {
    let order = &EXTREMAL_ORDERS[0];
    let norm = BigInt::<4>::from_limbs([0xDEAD_BEEF_CAFE_BAB1, 0, 0, 0]);
    if let Some(ideal) = LeftIdeal::<4>::random_prime_norm(&norm, order) {
        bencher.with_inputs(|| ideal).bench_values(|mut i| {
            i.refresh_norm::<8>();
            i
        });
    }
}
