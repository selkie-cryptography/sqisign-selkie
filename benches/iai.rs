//! Deterministic instruction-count benchmarks via iai-callgrind.
//!
//! Measures instructions, L1/L2 cache misses, and branch mispredictions
//! per function. Deterministic across CI runners — no timing noise.
//!
//! Requires Valgrind: `apt install valgrind` or `brew install valgrind`.
//! Run with: `cargo bench --bench iai --features expose-internals`

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use sqisign_selkie::{
    curves::{
        Scalar,
        montgomery::{Curve, ProjectiveXOnlyPoint},
    },
    fields::{fp::Fp, fp2::Fp2},
    params::{BASIS_E0_P_X, BASIS_E0_Q_X},
};
use std::hint::black_box;

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
    let p = black_box(ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &Curve::E0));
    let s = black_box(Scalar::from_limbs([0xDEAD_BEEF, 0xCAFE_BABE, 0x1234_5678, 0x9ABC_DEF0]));
    p.scalar_mul(&s)
}

#[library_benchmark]
fn point_double() -> ProjectiveXOnlyPoint {
    let p = black_box(ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &Curve::E0));
    p.double()
}

// --- Signature parsing ---

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
    benchmarks = vk_parse, sig_parse
);

main!(library_benchmark_groups = field, curves, parsing);
