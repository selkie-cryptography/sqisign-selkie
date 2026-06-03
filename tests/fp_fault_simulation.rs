//! Consumer for `tests/vectors/fp_fault_simulation.json`.
//!
//! The JSON file is the source of truth: each entry pairs an
//! `(a, b, op)` input set with the expected canonical byte output of
//! the corresponding `Fp` operation. This test loads the file, runs
//! every entry through production `Fp`, and asserts byte-equal
//! output.
//!
//! Each vector targets a specific carry-propagation, modular-
//! reduction, or accumulator-overflow fault class — limb fenceposts
//! at radix-2⁵¹ boundaries, modulus crossings, multiplication
//! accumulator stress at the top of the canonical range. cargo-
//! mutants only swaps local operators; it cannot reshape limb arrays
//! or change reduction strategy, so these inputs sit on bugs the
//! mutator structurally can't generate.
//!
//! Regenerate the JSON when the boundary set evolves:
//!
//! ```sh
//! cargo run --release --features expose-internals --example fp-fault-vectors-gen
//! ```
//!
//! Methodology: vector-forge skill, `references/fault-simulation.md`.

use serde::Deserialize;
use sqisign_selkie::{fields::fp::Fp, params::FP_ENCODED_BYTES};

/// One arithmetic operation under test. `Square` and `Neg` are unary
/// (`b` is ignored); the rest are binary.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Op {
    Add,
    Sub,
    Mul,
    Neg,
    Square,
}

/// JSON shape for the boundary-vector file.
#[derive(Deserialize)]
struct TestFile {
    #[serde(rename = "testGroups")]
    test_groups: Vec<TestGroup>,
}

#[derive(Deserialize)]
struct TestGroup {
    tests: Vec<TestVector>,
}

/// One boundary vector: `(a, b, op) -> expected`, all canonical Fp
/// encoded as 32 LE bytes.
#[derive(Deserialize)]
struct TestVector {
    #[serde(rename = "tcId")]
    tc_id: u32,
    comment: String,
    a: String,
    #[serde(default)]
    b: String,
    op: Op,
    expected: String,
    #[allow(dead_code)]
    flags: Vec<String>,
}

/// Runs `op` through production `Fp` and returns the canonical byte
/// encoding.
fn fp_op(
    op: Op,
    a_bytes: &[u8; FP_ENCODED_BYTES],
    b_bytes: &[u8; FP_ENCODED_BYTES],
) -> [u8; FP_ENCODED_BYTES] {
    let a = Fp::from_bytes(a_bytes);
    let b = Fp::from_bytes(b_bytes);
    match op {
        Op::Add => (a + b).to_bytes(),
        Op::Sub => (a - b).to_bytes(),
        Op::Mul => (a * b).to_bytes(),
        Op::Neg => (-a).to_bytes(),
        Op::Square => a.square().to_bytes(),
    }
}

#[test]
fn fp_fault_simulation_vectors() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/vectors/fp_fault_simulation.json");
    let json = std::fs::read_to_string(&path).expect("read fp_fault_simulation.json");
    let file: TestFile = serde_json::from_str(&json).expect("parse fp_fault_simulation.json");

    let mut total = 0usize;
    for group in &file.test_groups {
        for v in &group.tests {
            let a_vec = hex::decode(&v.a).expect("decode a");
            let b_vec = hex::decode(&v.b).expect("decode b");
            let exp_vec = hex::decode(&v.expected).expect("decode expected");
            let a_bytes: [u8; FP_ENCODED_BYTES] = a_vec.as_slice().try_into().expect("a len");
            let b_bytes: [u8; FP_ENCODED_BYTES] = b_vec.as_slice().try_into().expect("b len");
            let exp_bytes: [u8; FP_ENCODED_BYTES] =
                exp_vec.as_slice().try_into().expect("expected len");

            let got = fp_op(v.op, &a_bytes, &b_bytes);
            assert_eq!(
                got, exp_bytes,
                "tcId {} ({}): production Fp diverges from boundary expectation",
                v.tc_id, v.comment,
            );
            total += 1;
        }
    }
    eprintln!("fp_fault_simulation_vectors: {total} vectors passed");
}
