//! Consumer for `tests/vectors/bigint_fault_simulation.json`.
//!
//! The JSON file is the source of truth: each entry pairs an
//! `(a, b, op)` input set with the expected canonical byte output of
//! the corresponding `BigInt<4>` operation. This test loads the file,
//! runs every entry through production `BigInt<4>`, and asserts
//! byte-equal output.
//!
//! Each vector targets a sign+magnitude representation fault:
//!
//! * carry propagation across the 4-limb fenceposts in add,
//! * sign-cancellation canonicalization (no negative zero),
//! * sub sign flip + borrow,
//! * mul wraparound mod 2²⁵⁶,
//! * sign mix on mul.
//!
//! `bigint_oracle.rs` proptest catches these statistically against a
//! `num-bigint` reference; this set locks specific boundary inputs
//! that proptest only hits by chance, and pins canonical-zero
//! handling deterministically.
//!
//! Regenerate the JSON when the boundary set evolves:
//!
//! ```sh
//! cargo run --release --features expose-internals --example bigint-fault-vectors-gen
//! ```
//!
//! Methodology: vector-forge skill, `references/fault-simulation.md`.

use serde::Deserialize;
use sqisign_selkie::quaternions::bigint::BigInt;

/// Each `BigInt<4>` wire-encodes as 33 bytes: sign at byte 0, magnitude
/// in bytes 1..33 as little-endian per-limb.
const BIGINT4_ENCODED_BYTES: usize = 1 + 32;

/// One arithmetic operation under test. `Neg` is unary; the rest are
/// binary.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Op {
    Add,
    Sub,
    Mul,
    Neg,
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

/// One boundary vector: `(a, b, op) -> expected`. Each `BigInt<4>` is
/// 33 LE hex bytes (1 sign byte + 32 magnitude bytes).
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

/// Constructs a `BigInt<4>` from 33 sign+magnitude LE bytes.
fn bigint_from_encoded(bytes: &[u8; BIGINT4_ENCODED_BYTES]) -> BigInt<4> {
    let sign = u64::from(bytes[0]);
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let mut le = [0u8; 8];
        le.copy_from_slice(&bytes[1 + i * 8..1 + (i + 1) * 8]);
        *limb = u64::from_le_bytes(le);
    }
    BigInt::from_sign_and_limbs(sign, limbs)
}

/// Encodes a `BigInt<4>` as 33 LE bytes, canonicalizing zero to
/// `sign = 0`.
fn bigint_encode(b: &BigInt<4>) -> [u8; BIGINT4_ENCODED_BYTES] {
    let mut out = [0u8; BIGINT4_ENCODED_BYTES];
    let mut magnitude = [0u8; 32];
    for (i, limb) in b.as_limbs().iter().enumerate() {
        magnitude[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    out[0] = if magnitude == [0u8; 32] {
        0
    } else if bool::from(b.is_negative()) {
        1
    } else {
        0
    };
    out[1..].copy_from_slice(&magnitude);
    out
}

/// Runs `op` through production `BigInt<4>` and returns the canonical
/// byte encoding.
fn bigint_op(
    op: Op,
    a_bytes: &[u8; BIGINT4_ENCODED_BYTES],
    b_bytes: &[u8; BIGINT4_ENCODED_BYTES],
) -> [u8; BIGINT4_ENCODED_BYTES] {
    let a = bigint_from_encoded(a_bytes);
    let b = bigint_from_encoded(b_bytes);
    let result = match op {
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
        Op::Neg => -a,
    };
    bigint_encode(&result)
}

#[test]
fn bigint_fault_simulation_vectors() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/vectors/bigint_fault_simulation.json");
    let json = std::fs::read_to_string(&path).expect("read bigint_fault_simulation.json");
    let file: TestFile = serde_json::from_str(&json).expect("parse bigint_fault_simulation.json");

    let mut total = 0usize;
    for group in &file.test_groups {
        for v in &group.tests {
            let a_vec = hex::decode(&v.a).expect("decode a");
            let b_vec = hex::decode(&v.b).expect("decode b");
            let exp_vec = hex::decode(&v.expected).expect("decode expected");
            let a_bytes: [u8; BIGINT4_ENCODED_BYTES] = a_vec.as_slice().try_into().expect("a len");
            let b_bytes: [u8; BIGINT4_ENCODED_BYTES] = b_vec.as_slice().try_into().expect("b len");
            let exp_bytes: [u8; BIGINT4_ENCODED_BYTES] =
                exp_vec.as_slice().try_into().expect("expected len");

            let got = bigint_op(v.op, &a_bytes, &b_bytes);
            assert_eq!(
                got, exp_bytes,
                "tcId {} ({}): production BigInt<4> diverges from boundary expectation",
                v.tc_id, v.comment,
            );
            total += 1;
        }
    }
    eprintln!("bigint_fault_simulation_vectors: {total} vectors passed");
}
