//! Regenerates `tests/vectors/bigint_fault_simulation.json` from the
//! boundary-input definitions below.
//!
//! The JSON file is the source of truth for the
//! [`bigint_fault_simulation_vectors`](../tests/bigint_fault_simulation.rs)
//! test; this generator exists so reviewers can audit the
//! `(input, op) -> expected` recipe instead of staring at hex bytes,
//! and so the file can be regenerated when the boundary set evolves.
//!
//! Run:
//!
//! ```sh
//! cargo run --release --features expose-internals --example bigint-fault-vectors-gen
//! ```
//!
//! Reference results are computed via `num-bigint` at unbounded
//! precision, then truncated mod 2²⁵⁶ to match `BigInt<4>`'s fixed
//! width. The generator emits a self-check pass against production
//! `BigInt<4>` before writing.
//!
//! Methodology: vector-forge skill, `references/fault-simulation.md`.

use num_bigint::{BigInt as NumBigInt, Sign};
use num_traits::{One, Signed, Zero};
use sqisign_selkie::quaternions::bigint::BigInt;

/// Each `BigInt<4>` wire-encodes as 33 bytes: sign at byte 0
/// (0 = non-negative, 1 = negative), magnitude in bytes 1..33 as
/// little-endian per limb.
const BIGINT4_ENCODED_BYTES: usize = 1 + 32;

/// Lifts a `BigInt<4>` to a signed `num_bigint::BigInt`.
fn to_num(a: &BigInt<4>) -> NumBigInt {
    let mut bytes = [0u8; 32];
    for (i, limb) in a.as_limbs().iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    let magnitude = NumBigInt::from_bytes_le(Sign::Plus, &bytes);
    if bool::from(a.is_negative()) {
        -magnitude
    } else {
        magnitude
    }
}

/// Encodes `(sign, magnitude_bytes)` as 33 LE bytes.
fn encode(sign: u8, magnitude: &[u8; 32]) -> [u8; BIGINT4_ENCODED_BYTES] {
    let mut out = [0u8; BIGINT4_ENCODED_BYTES];
    out[0] = sign;
    out[1..].copy_from_slice(magnitude);
    out
}

/// Truncates a `num_bigint::BigInt` to `BigInt<4>` semantics: take the
/// low 256 bits, with the sign preserved when the magnitude is
/// nonzero. Mirrors the `truncate::<4>` helper in `bigint_oracle.rs`.
fn encode_num(n: &NumBigInt) -> [u8; BIGINT4_ENCODED_BYTES] {
    let modulus: NumBigInt = NumBigInt::one() << 256;
    let mut magnitude_signed = n % &modulus;
    if magnitude_signed.is_negative() {
        // Truncated remainder; preserve the negative sign on the
        // magnitude. The magnitude bytes are the absolute value's LE
        // encoding; the sign byte carries the sign.
        magnitude_signed = -magnitude_signed;
    }
    let (sign_kind, mag_bytes) = magnitude_signed.to_bytes_le();
    debug_assert!(sign_kind != Sign::Minus);
    let mut magnitude = [0u8; 32];
    magnitude[..mag_bytes.len()].copy_from_slice(&mag_bytes);
    // Canonical zero: sign = 0 regardless of input sign.
    let sign = if magnitude == [0u8; 32] {
        0
    } else if n.is_negative() {
        1
    } else {
        0
    };
    encode(sign, &magnitude)
}

/// One arithmetic operation under test.
#[derive(Clone, Copy, Debug)]
enum Op {
    Add,
    Sub,
    Mul,
    Neg,
}

impl Op {
    fn as_str(self) -> &'static str {
        match self {
            Op::Add => "add",
            Op::Sub => "sub",
            Op::Mul => "mul",
            Op::Neg => "neg",
        }
    }
}

/// Boundary input as a symbolic name plus the signed
/// `num_bigint::BigInt` value. The name appears in the generated
/// comment; the integer drives the encoding.
struct BoundaryInput {
    name: &'static str,
    value: NumBigInt,
}

/// Returns the boundary inputs.
fn boundary_inputs() -> Vec<BoundaryInput> {
    let one: NumBigInt = NumBigInt::one();
    let zero: NumBigInt = NumBigInt::zero();
    let five: NumBigInt = NumBigInt::from(5u32);
    let seven: NumBigInt = NumBigInt::from(7u32);
    let three: NumBigInt = NumBigInt::from(3u32);
    let twentyfive: NumBigInt = NumBigInt::from(25u32);
    let twofivesix_m1: NumBigInt = (&one << 256) - &one;
    let twofivesix_m2: NumBigInt = (&one << 256) - NumBigInt::from(2u32);
    let twosixfour_m1: NumBigInt = (&one << 64) - &one;
    let twosixfour: NumBigInt = &one << 64;
    let twooneeight_m1: NumBigInt = (&one << 128) - &one;
    let twooneeight: NumBigInt = &one << 128;
    let two192_m1: NumBigInt = (&one << 192) - &one;
    let two192: NumBigInt = &one << 192;
    let two255: NumBigInt = &one << 255;
    let two32: NumBigInt = &one << 32;
    vec![
        BoundaryInput {
            name: "0",
            value: zero.clone(),
        },
        BoundaryInput {
            name: "1",
            value: one.clone(),
        },
        BoundaryInput {
            name: "3",
            value: three.clone(),
        },
        BoundaryInput {
            name: "5",
            value: five.clone(),
        },
        BoundaryInput {
            name: "7",
            value: seven.clone(),
        },
        BoundaryInput {
            name: "25",
            value: twentyfive.clone(),
        },
        BoundaryInput {
            name: "-3",
            value: -three.clone(),
        },
        BoundaryInput {
            name: "-5",
            value: -five.clone(),
        },
        BoundaryInput {
            name: "-7",
            value: -seven.clone(),
        },
        BoundaryInput {
            name: "2^32",
            value: two32.clone(),
        },
        BoundaryInput {
            name: "2^64 - 1",
            value: twosixfour_m1.clone(),
        },
        BoundaryInput {
            name: "2^64",
            value: twosixfour.clone(),
        },
        BoundaryInput {
            name: "2^128 - 1",
            value: twooneeight_m1.clone(),
        },
        BoundaryInput {
            name: "2^128",
            value: twooneeight.clone(),
        },
        BoundaryInput {
            name: "2^192 - 1",
            value: two192_m1.clone(),
        },
        BoundaryInput {
            name: "2^192",
            value: two192.clone(),
        },
        BoundaryInput {
            name: "2^255",
            value: two255.clone(),
        },
        BoundaryInput {
            name: "2^256 - 1",
            value: twofivesix_m1.clone(),
        },
        BoundaryInput {
            name: "2^256 - 2",
            value: twofivesix_m2.clone(),
        },
        BoundaryInput {
            name: "-(2^256 - 1)",
            value: -twofivesix_m1.clone(),
        },
    ]
}

/// Returns `(a, b, op, comment)` tuples whose expected outputs become
/// the vector set.
fn vector_recipes(inputs: &[BoundaryInput]) -> Vec<(usize, usize, Op, String)> {
    let by_name = |n: &str| {
        inputs
            .iter()
            .position(|x| x.name == n)
            .unwrap_or_else(|| panic!("missing boundary input {n}"))
    };
    vec![
        // Add: carry propagation across the 4-limb fenceposts.
        (
            by_name("2^64 - 1"),
            by_name("1"),
            Op::Add,
            "(2^64 - 1) + 1 = 2^64 — carry limb 0 -> limb 1".into(),
        ),
        (
            by_name("2^128 - 1"),
            by_name("1"),
            Op::Add,
            "(2^128 - 1) + 1 = 2^128 — carry limbs 0..1 -> limb 2".into(),
        ),
        (
            by_name("2^192 - 1"),
            by_name("1"),
            Op::Add,
            "(2^192 - 1) + 1 = 2^192 — carry limbs 0..2 -> limb 3".into(),
        ),
        (
            by_name("2^256 - 1"),
            by_name("1"),
            Op::Add,
            "(2^256 - 1) + 1 = 0 — full wrap mod 2^256, canonical-zero sign".into(),
        ),
        (
            by_name("2^255"),
            by_name("2^255"),
            Op::Add,
            "2^255 + 2^255 = 0 mod 2^256 — top-bit overflow wraps to canonical zero".into(),
        ),
        // Sign cancellation.
        (
            by_name("5"),
            by_name("-5"),
            Op::Add,
            "5 + (-5) = 0 — sign cancellation must canonicalize zero (sign=0)".into(),
        ),
        (
            by_name("5"),
            by_name("-3"),
            Op::Add,
            "5 + (-3) = 2 — mixed sign, positive result".into(),
        ),
        (
            by_name("-5"),
            by_name("-7"),
            Op::Add,
            "-5 + -7 = -12 — both negative".into(),
        ),
        // Sub: sign flip + borrow.
        (
            by_name("0"),
            by_name("1"),
            Op::Sub,
            "0 - 1 = -1 — sub from zero flips sign".into(),
        ),
        (
            by_name("5"),
            by_name("7"),
            Op::Sub,
            "5 - 7 = -2 — magnitude difference, sign flips negative".into(),
        ),
        (
            by_name("2^256 - 1"),
            by_name("1"),
            Op::Sub,
            "(2^256 - 1) - 1 = 2^256 - 2 — borrow propagates through limbs".into(),
        ),
        (
            by_name("0"),
            by_name("2^256 - 1"),
            Op::Sub,
            "0 - (2^256 - 1) = -(2^256 - 1) — full underflow into negative-max".into(),
        ),
        // Neg: zero canonicalization, sign flip, max-magnitude.
        (
            by_name("0"),
            by_name("0"),
            Op::Neg,
            "-(0) = 0 — neg of zero must remain sign=0 (no negative zero)".into(),
        ),
        (
            by_name("-5"),
            by_name("0"),
            Op::Neg,
            "-(-5) = 5 — double negation".into(),
        ),
        (
            by_name("2^256 - 1"),
            by_name("0"),
            Op::Neg,
            "-(2^256 - 1) = -(2^256 - 1) — neg at canonical-max magnitude".into(),
        ),
        // Mul: limb boundaries, wraparound, sign mix.
        (
            by_name("2^32"),
            by_name("2^32"),
            Op::Mul,
            "2^32 * 2^32 = 2^64 — output crosses limb 0 -> limb 1 boundary".into(),
        ),
        (
            by_name("2^128"),
            by_name("2^128"),
            Op::Mul,
            "2^128 * 2^128 = 0 mod 2^256 — wraparound on conceptual 2^256".into(),
        ),
        (
            by_name("2^256 - 1"),
            by_name("2^256 - 1"),
            Op::Mul,
            "(2^256 - 1)^2 = 1 mod 2^256 — full schoolbook stress".into(),
        ),
        (
            by_name("-5"),
            by_name("3"),
            Op::Mul,
            "(-5) * 3 = -15 — sign-mix produces negative result".into(),
        ),
        (
            by_name("-5"),
            by_name("-5"),
            Op::Mul,
            "(-5) * (-5) = 25 — both negative cancel to positive".into(),
        ),
    ]
}

/// Applies `op` to the unbounded values; truncation happens at encode
/// time via [`encode_num`].
fn apply_op(op: Op, a: &NumBigInt, b: &NumBigInt) -> NumBigInt {
    match op {
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
        Op::Neg => -a,
    }
}

/// Constructs a `BigInt<4>` from sign+magnitude bytes.
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

/// Encodes a `BigInt<4>` as 33 LE bytes (sign byte + 32 magnitude
/// bytes), canonicalizing zero to `sign = 0`.
fn bigint_encode(b: &BigInt<4>) -> [u8; BIGINT4_ENCODED_BYTES] {
    let mut magnitude = [0u8; 32];
    for (i, limb) in b.as_limbs().iter().enumerate() {
        magnitude[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    let sign = if magnitude == [0u8; 32] {
        0
    } else if bool::from(b.is_negative()) {
        1
    } else {
        0
    };
    encode(sign, &magnitude)
}

/// Runs `op` through production `BigInt<4>` and returns the canonical
/// byte encoding. Used as the in-generator self-check.
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

fn main() {
    let inputs = boundary_inputs();
    let recipes = vector_recipes(&inputs);

    let mut entries: Vec<String> = Vec::new();
    for (tc, (a_idx, b_idx, op, comment)) in recipes.iter().enumerate() {
        let a_num = &inputs[*a_idx].value;
        let b_num = &inputs[*b_idx].value;
        let a_bytes = encode_num(a_num);
        let b_bytes = encode_num(b_num);
        let expected_num = apply_op(*op, a_num, b_num);
        let expected_bytes = encode_num(&expected_num);

        let production = bigint_op(*op, &a_bytes, &b_bytes);
        if production != expected_bytes {
            // Surface both reference and production for debugging.
            let prod_bi = bigint_from_encoded(&production);
            let exp_bi = bigint_from_encoded(&expected_bytes);
            panic!(
                "vector {tc} ({comment}):\n  reference = {expected_num} (= {})\n  production = {} (encoded {})",
                to_num(&exp_bi),
                to_num(&prod_bi),
                hex::encode(production),
            );
        }

        entries.push(format!(
            r#"        {{
          "tcId": {tc_id},
          "comment": "{comment}",
          "a": "{a_hex}",
          "b": "{b_hex}",
          "op": "{op_str}",
          "expected": "{exp_hex}",
          "flags": ["FaultSimulation"]
        }}"#,
            tc_id = tc + 1,
            comment = comment,
            a_hex = hex::encode(a_bytes),
            b_hex = hex::encode(b_bytes),
            op_str = op.as_str(),
            exp_hex = hex::encode(expected_bytes),
        ));
    }

    let n = entries.len();
    let json = format!(
        r#"{{
  "algorithm": "SQIsign_248_BigInt4",
  "schema": "bigint_fault_simulation_schema.json",
  "generatorVersion": "sqisign-selkie-0.0.1",
  "numberOfTests": {n},
  "header": [
    "Boundary-input vectors for BigInt<4> arithmetic (sign+magnitude over [u64; 4]).",
    "Each vector targets a sign+magnitude representation fault: carry across the",
    "4-limb fenceposts, sign-cancellation canonicalization (no negative zero),",
    "sub sign flip, mul wraparound mod 2^256, sign mix on mul.",
    "All operations are mod 2^(N*64); the reference computes in unbounded",
    "precision and truncates at encode time.",
    "Methodology: vector-forge skill, references/fault-simulation.md.",
    "Encoding: 33 LE bytes per BigInt = 1 sign byte (0 = non-negative, 1 = negative)",
    "followed by 32 magnitude bytes LE. Zero canonicalizes to sign = 0."
  ],
  "notes": {{
    "FaultSimulation": {{
      "bugType": "FUNCTIONALITY",
      "description": "Boundary input crafted to detect a specific carry, sign-canonicalization, or wraparound fault in BigInt<4>'s sign+magnitude arithmetic. Any conformant impl must produce the byte-equal expected output."
    }}
  }},
  "testGroups": [
    {{
      "type": "BigInt4Arithmetic",
      "tests": [
{}
      ]
    }}
  ]
}}
"#,
        entries.join(",\n"),
    );

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/vectors/bigint_fault_simulation.json");
    std::fs::write(&path, json).expect("write bigint_fault_simulation.json");
    eprintln!("wrote {n} vectors to {}", path.display());
}
