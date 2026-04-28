//! Deterministic instruction-count benchmarks via iai-callgrind.
//!
//! Measures instructions, L1/L2 cache misses, and branch mispredictions
//! per function. Deterministic across CI runners — no timing noise.
//!
//! Requires Valgrind: `apt install valgrind` or `brew install valgrind`.
//! Run with: `cargo bench --bench iai --features expose-internals`

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use sqisign_selkie::{
    curves::{
        Scalar,
        montgomery::{Curve, ProjectiveXOnlyPoint},
    },
    fields::{fp::Fp, fp2::Fp2},
    params::BASIS_E0_P_X,
};

// --- Fp arithmetic ---

#[library_benchmark]
fn fp_mul() -> Fp {
    let a = black_box(Fp::from_bytes(&[0x42; 32]));
    let b = black_box(Fp::from_bytes(&[0x99; 32]));
    a * b
}

#[library_benchmark]
fn fp_add() -> Fp {
    let a = black_box(Fp::from_bytes(&[0x42; 32]));
    let b = black_box(Fp::from_bytes(&[0x99; 32]));
    a + b
}

#[library_benchmark]
fn fp_sub() -> Fp {
    let a = black_box(Fp::from_bytes(&[0x42; 32]));
    let b = black_box(Fp::from_bytes(&[0x99; 32]));
    a - b
}

#[library_benchmark]
fn fp_square() -> Fp {
    let a = black_box(Fp::from_bytes(&[0x42; 32]));
    a * a
}

// --- Fp2 arithmetic ---

#[library_benchmark]
fn fp2_mul() -> Fp2 {
    let a = black_box(Fp2::new(
        Fp::from_bytes(&[0x42; 32]),
        Fp::from_bytes(&[0x11; 32]),
    ));
    let b = black_box(Fp2::new(
        Fp::from_bytes(&[0x99; 32]),
        Fp::from_bytes(&[0x55; 32]),
    ));
    a * b
}

// --- Curve operations ---

#[library_benchmark]
fn scalar_mul() -> ProjectiveXOnlyPoint {
    let p = black_box(ProjectiveXOnlyPoint::from_affine_x(
        BASIS_E0_P_X,
        &Curve::E0,
    ));
    let s = black_box(Scalar::from_limbs([
        0xDEAD_BEEF,
        0xCAFE_BABE,
        0x1234_5678,
        0x9ABC_DEF0,
    ]));
    p.scalar_mul(&s)
}

#[library_benchmark]
fn point_double() -> ProjectiveXOnlyPoint {
    let p = black_box(ProjectiveXOnlyPoint::from_affine_x(
        BASIS_E0_P_X,
        &Curve::E0,
    ));
    p.double()
}

// --- Parsing ---

#[library_benchmark]
fn vk_parse() {
    let bytes = black_box([0u8; 65]);
    let _ = black_box(sqisign_selkie::VerifyingKey::from_bytes(&bytes));
}

#[library_benchmark]
fn sig_parse() {
    let bytes = black_box([0u8; 148]);
    let _ = black_box(sqisign_selkie::Signature::from_bytes(&bytes));
}

#[library_benchmark]
fn sk_parse() {
    let sk_hex = sqisign_selkie::keys::kat_data::KAT_VECTORS[0].2;
    let sk_bytes = hex::decode(sk_hex).unwrap();
    let sk_arr: &[u8; sqisign_selkie::SIGNING_KEY_BYTES] = sk_bytes.as_slice().try_into().unwrap();
    let _ = black_box(sqisign_selkie::SigningKey::from_bytes(sk_arr));
}

// --- Top-level operations ---

// KAT vector 0 verification: parse pk + sig, verify.
#[library_benchmark]
fn kat_verify() {
    let pk_hex = sqisign_selkie::keys::kat_data::KAT_VECTORS[0].1;
    let sm_hex = sqisign_selkie::keys::kat_data::KAT_VECTORS[0].4;
    let pk_bytes = hex::decode(pk_hex).unwrap();
    let sm_bytes = hex::decode(sm_hex).unwrap();
    let sig_bytes: &[u8; sqisign_selkie::SIGNATURE_BYTES] = sm_bytes
        [..sqisign_selkie::SIGNATURE_BYTES]
        .try_into()
        .unwrap();
    let msg = &sm_bytes[sqisign_selkie::SIGNATURE_BYTES..];

    let vk =
        sqisign_selkie::VerifyingKey::from_bytes(pk_bytes.as_slice().try_into().unwrap()).unwrap();
    let sig = sqisign_selkie::Signature::from_bytes(sig_bytes).unwrap();
    let _ = black_box(vk.verify(msg, &sig));
}

// Deterministic keygen from KAT seed 0.
// Too slow under Valgrind (~30 min). Excluded from the benchmark
// group below; uncomment in `operations` to run manually.
#[library_benchmark]
fn kat_keygen() {
    let seed_hex = sqisign_selkie::keys::kat_data::KAT_VECTORS[0].0;
    let seed_bytes = hex::decode(seed_hex).unwrap();
    let seed: [u8; 48] = seed_bytes.as_slice().try_into().unwrap();
    let _ = black_box(sqisign_selkie::SigningKey::generate_derand(&seed));
}

// Deterministic sign with KAT key 0.
// Too slow under Valgrind (~30 min). Excluded from the benchmark
// group below; uncomment in `operations` to run manually.
#[library_benchmark]
fn kat_sign() {
    let sk_hex = sqisign_selkie::keys::kat_data::KAT_VECTORS[0].2;
    let sk_bytes = hex::decode(sk_hex).unwrap();
    let sk =
        sqisign_selkie::SigningKey::from_bytes(sk_bytes.as_slice().try_into().unwrap()).unwrap();
    let msg = b"iai benchmark message";
    let seed = [0x42u8; 48];
    let _ = black_box(sk.sign_derand(msg, &seed));
}

library_benchmark_group!(
    name = field;
    benchmarks = fp_mul, fp_add, fp_sub, fp_square, fp2_mul
);

library_benchmark_group!(
    name = curves;
    benchmarks = scalar_mul, point_double
);

library_benchmark_group!(
    name = parsing;
    benchmarks = vk_parse, sig_parse, sk_parse
);

library_benchmark_group!(
    name = operations;
    // TODO: re-enable kat_keygen and kat_sign once they complete under
    // Valgrind within the CI timeout (currently ~30 min each).
    benchmarks = kat_verify
);

main!(
    library_benchmark_groups = field,
    curves,
    parsing,
    operations
);
