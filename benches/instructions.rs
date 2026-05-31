//! Deterministic profile benchmarks via gungraun (the renamed
//! iai-callgrind).
//!
//! Runs each bench under Valgrind/callgrind with cache + branch
//! simulation, so every run emits `Ir` (instructions), `EstimatedCycles`,
//! L1/LL cache misses, and branch mispredicts --- all deterministic
//! across CI runners (no timing noise).  A per-benchmark `Ir` flamegraph
//! is written alongside each summary as
//! `callgrind.<bench>.total.Ir.flamegraph.svg`.
//!
//! Sharded under the `Profile` workflow (`.github/workflows/profile.yml`),
//! one matrix job per bench group / slow sqisign bench, so the wall-clock
//! cost of cache simulation is bounded by the longest single bench rather
//! than their sum.
//!
//! Requires Valgrind: `apt install valgrind` or `brew install valgrind`.
//! Run with: `cargo bench --bench instructions --features expose-internals`

mod common;

use std::hint::black_box;

use gungraun::{
    Callgrind, EventKind, FlamegraphConfig, LibraryBenchmarkConfig, library_benchmark,
    library_benchmark_group, main,
};
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
    let sk_arr = common::kat0_sk_bytes();
    let _ = black_box(sqisign_selkie::SigningKey::from_bytes(&sk_arr));
}

// --- Top-level operations ---

// KAT vector 0 verification: parse vk + sig, verify.
#[library_benchmark]
fn kat_verify() {
    let (vk, sig, msg) = common::kat0_vk_sig_msg();
    let _ = black_box(vk.verify(&msg, &sig));
}

// Deterministic keygen from KAT seed 0.
#[library_benchmark]
fn kat_keygen() {
    let seed = common::kat0_seed();
    let _ = black_box(sqisign_selkie::SigningKey::generate_derand(&seed));
}

// Deterministic sign with KAT key 0 — the response-phase flat profile and
// flamegraph that drive optimization targeting.
#[library_benchmark]
fn kat_sign() {
    let sk = common::kat0_signing_key();
    let msg = b"instructions benchmark message";
    let randomness = [0x42u8; 48];
    let _ = black_box(sk.sign_derand(msg, &randomness));
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
    name = sqisign;
    // Top-level sqisign keygen/sign/verify under Valgrind. keygen and sign are
    // the slow, high-value profiles — their flamegraphs locate the targets.
    benchmarks = kat_verify, kat_sign, kat_keygen
);

main!(
    config = LibraryBenchmarkConfig::default().tool(
        // gungraun defaults `--cache-sim=yes --branch-sim=yes`, which is
        // what we want here: every bench produces estimated_cycles, L1/LL
        // cache misses, and branch mispredicts alongside Ir.  Cache
        // simulation roughly doubles per-bench wall-clock; the workflow
        // shards across benches to keep total wall-clock bounded.
        Callgrind::default()
            .flamegraph(FlamegraphConfig::default().event_kinds([EventKind::Ir])),
    );
    library_benchmark_groups = field,
    curves,
    parsing,
    sqisign
);
