//! Consumer for `tests/vectors/fp2_fault_simulation.json`.
//!
//! The JSON file is the source of truth: each entry pairs an
//! `(a, b, op)` input set with the expected canonical byte output of
//! the corresponding `Fp2` operation. This test loads the file, runs
//! every entry through production `Fp2`, and asserts byte-equal
//! output.
//!
//! Companion to `tests/fp_fault_simulation.rs`. Where the Fp set pins
//! limb-level carry / reduction faults at the multi-precision level,
//! this Fp² set pins formula-level faults that live above Fp:
//! cross-term sign errors in mul (`ac − bd` vs `ac + bd`), the
//! `i² = -1` axiom, the `(a² − b², 2ab)` square specialization, and
//! component swap (real vs imag) in any of those formulas.
//!
//! Regenerate the JSON when the boundary set evolves:
//!
//! ```sh
//! cargo run --release --features expose-internals --example fp2-fault-vectors-gen
//! ```
//!
//! Methodology: vector-forge skill, `references/fault-simulation.md`.

use serde::Deserialize;
use sqisign_selkie::{fields::fp2::Fp2, params::FP2_ENCODED_BYTES};

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

/// One boundary vector: `(a, b, op) -> expected`. All values are
/// canonical Fp² (real and imag components < p), each 41 LE bytes.
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

/// Runs `op` through production `Fp2` and returns the canonical byte
/// encoding.
fn fp2_op(
    op: Op,
    a_bytes: &[u8; FP2_ENCODED_BYTES],
    b_bytes: &[u8; FP2_ENCODED_BYTES],
) -> [u8; FP2_ENCODED_BYTES] {
    let a = Fp2::from_bytes(a_bytes);
    let b = Fp2::from_bytes(b_bytes);
    match op {
        Op::Add => (a + b).to_bytes(),
        Op::Sub => (a - b).to_bytes(),
        Op::Mul => (a * b).to_bytes(),
        Op::Neg => (-a).to_bytes(),
        Op::Square => a.square().to_bytes(),
    }
}

#[test]
fn fp2_fault_simulation_vectors() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/vectors/fp2_fault_simulation.json");
    let json = std::fs::read_to_string(&path).expect("read fp2_fault_simulation.json");
    let file: TestFile = serde_json::from_str(&json).expect("parse fp2_fault_simulation.json");

    let mut total = 0usize;
    for group in &file.test_groups {
        for v in &group.tests {
            let a_vec = hex::decode(&v.a).expect("decode a");
            let b_vec = hex::decode(&v.b).expect("decode b");
            let exp_vec = hex::decode(&v.expected).expect("decode expected");
            let a_bytes: [u8; FP2_ENCODED_BYTES] = a_vec.as_slice().try_into().expect("a len");
            let b_bytes: [u8; FP2_ENCODED_BYTES] = b_vec.as_slice().try_into().expect("b len");
            let exp_bytes: [u8; FP2_ENCODED_BYTES] =
                exp_vec.as_slice().try_into().expect("expected len");

            let got = fp2_op(v.op, &a_bytes, &b_bytes);
            assert_eq!(
                got, exp_bytes,
                "tcId {} ({}): production Fp² diverges from boundary expectation",
                v.tc_id, v.comment,
            );
            total += 1;
        }
    }
    eprintln!("fp2_fault_simulation_vectors: {total} vectors passed");
}
