//! Regenerates `tests/vectors/fp_fault_simulation.json` from the
//! boundary-input definitions below.
//!
//! The JSON file is the source of truth for the
//! [`fp_fault_simulation_vectors`](../tests/fp_fault_simulation.rs)
//! test; this generator exists so reviewers can audit the
//! `(input, op) -> expected` recipe instead of staring at hex bytes,
//! and so the file can be regenerated when the boundary set evolves.
//!
//! Run:
//!
//! ```sh
//! cargo run --release --features expose-internals --example fp-fault-vectors-gen
//! ```
//!
//! Reference results are computed via `num-bigint` at unbounded
//! precision, so the generator is independent of `Fp` itself — but
//! the file the generator emits gets a self-check pass against
//! production `Fp` before being written.
//!
//! Methodology: vector-forge skill, `references/fault-simulation.md`.

use num_bigint::BigInt as NumBigInt;
use num_traits::{One, Signed, Zero};
use sqisign_selkie::{fields::fp::Fp, params::FP_ENCODED_BYTES};

/// `p = 5 · 2²⁴⁸ − 1`, the base prime, as an unbounded integer.
fn p() -> NumBigInt {
    (NumBigInt::from(5u32) << 248) - NumBigInt::one()
}

/// Reduces an unbounded integer to `[0, p)` and encodes it as 32 LE
/// bytes — the canonical wire form `Fp::from_bytes` expects.
fn num_to_fp_bytes(n: &NumBigInt) -> [u8; FP_ENCODED_BYTES] {
    let p = p();
    let mut r = n % &p;
    if r.is_negative() {
        r += &p;
    }
    let (_, mag) = r.to_bytes_le();
    let mut out = [0u8; FP_ENCODED_BYTES];
    out[..mag.len()].copy_from_slice(&mag);
    out
}

/// One arithmetic operation under test.
#[derive(Clone, Copy, Debug)]
enum Op {
    Add,
    Sub,
    Mul,
    Neg,
    Square,
}

impl Op {
    fn as_str(self) -> &'static str {
        match self {
            Op::Add => "add",
            Op::Sub => "sub",
            Op::Mul => "mul",
            Op::Neg => "neg",
            Op::Square => "square",
        }
    }
}

/// Boundary input as a symbolic name plus the unbounded integer. The
/// name appears in the generated comment to keep the vector human-
/// auditable; the integer drives the encoding.
struct BoundaryInput {
    name: &'static str,
    value: NumBigInt,
}

/// Returns the boundary inputs used to seed the `(a, b)` pairings
/// below. Grouped by fault class for easier review.
fn boundary_inputs() -> Vec<BoundaryInput> {
    let p = p();
    let one = NumBigInt::one();
    let two = NumBigInt::from(2u32);
    let zero = NumBigInt::zero();
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
            name: "2",
            value: two.clone(),
        },
        BoundaryInput {
            name: "p-1",
            value: &p - &one,
        },
        BoundaryInput {
            name: "p-2",
            value: &p - &two,
        },
        BoundaryInput {
            name: "(p-1)/2",
            value: (&p - &one) / 2u32,
        },
        BoundaryInput {
            name: "(p+1)/2",
            value: (&p + &one) / 2u32,
        },
        // Radix-2^51 limb fenceposts. A canonical Fp value at exactly
        // 2^(51k) − 1 saturates limbs 0..k-1; adding 1 forces a carry
        // through every limb up to k. Production's radix-51 add must
        // propagate this carry correctly.
        BoundaryInput {
            name: "2^51 - 1",
            value: (&one << 51) - &one,
        },
        BoundaryInput {
            name: "2^102 - 1",
            value: (&one << 102) - &one,
        },
        BoundaryInput {
            name: "2^153 - 1",
            value: (&one << 153) - &one,
        },
        BoundaryInput {
            name: "2^204 - 1",
            value: (&one << 204) - &one,
        },
        // Just inside / just outside the prime layout. p = 5·2^248 − 1,
        // so 2^248 = (p+1)/5 is the smallest 249-bit canonical Fp
        // value; 2^248 − 1 saturates the lower 248 bits.
        BoundaryInput {
            name: "2^248",
            value: &one << 248,
        },
        BoundaryInput {
            name: "2^248 - 1",
            value: (&one << 248) - &one,
        },
        // Multiplication-accumulator pressure: 2^127 squared produces
        // 2^254, which is at the top of the canonical Fp range (just
        // under 2^255 = R). Tests the high-limb path of the product.
        BoundaryInput {
            name: "2^124",
            value: &one << 124,
        },
        BoundaryInput {
            name: "2^127",
            value: &one << 127,
        },
    ]
}

/// Returns `(a, b, op, comment)` tuples whose expected outputs become
/// the vector set. Each tuple is chosen so its conceptual result
/// lands on a specific limb-boundary or modulus-crossing condition.
fn vector_recipes(inputs: &[BoundaryInput]) -> Vec<(usize, usize, Op, String)> {
    let by_name = |n: &str| {
        inputs
            .iter()
            .position(|x| x.name == n)
            .unwrap_or_else(|| panic!("missing boundary input {n}"))
    };

    let mut recipes: Vec<(usize, usize, Op, String)> = vec![
        // Canonical-maximum reduction: (p−1) + 1 must cross to 0;
        // (p−1) + (p−1) must subtract p once; (p−1) × (p−1) ≡ 1.
        (
            by_name("p-1"),
            by_name("1"),
            Op::Add,
            "(p-1) + 1 = 0 — final-reduction stress".into(),
        ),
        (
            by_name("p-1"),
            by_name("p-1"),
            Op::Add,
            "(p-1) + (p-1) = p-2 — one subtract-p needed".into(),
        ),
        (
            by_name("p-1"),
            by_name("p-1"),
            Op::Mul,
            "(p-1) * (p-1) = 1 — full Montgomery REDC stress".into(),
        ),
        (
            by_name("p-1"),
            by_name("2"),
            Op::Mul,
            "(p-1) * 2 = p-2 — one subtract-p in product".into(),
        ),
        (
            by_name("p-1"),
            by_name("0"),
            Op::Neg,
            "-(p-1) = 1 — negation at canonical max".into(),
        ),
        // Modulus crossing.
        (
            by_name("(p+1)/2"),
            by_name("(p-1)/2"),
            Op::Add,
            "(p+1)/2 + (p-1)/2 = p = 0 — exact-modulus reduction".into(),
        ),
        (
            by_name("(p-1)/2"),
            by_name("2"),
            Op::Mul,
            "((p-1)/2) * 2 = p-1 — at the canonical boundary".into(),
        ),
    ];

    // Radix-2^51 limb fenceposts: adding 1 forces a carry chain.
    for (max, label) in &[
        ("2^51 - 1", "limb 0 -> limb 1"),
        ("2^102 - 1", "limbs 0..1 -> limb 2"),
        ("2^153 - 1", "limbs 0..2 -> limb 3"),
        ("2^204 - 1", "limbs 0..3 -> limb 4"),
    ] {
        recipes.push((
            by_name(max),
            by_name("1"),
            Op::Add,
            format!("({max}) + 1 — carry propagation {label}"),
        ));
    }

    // 2^248 cluster: just inside the prime layout.
    recipes.push((
        by_name("2^248 - 1"),
        by_name("1"),
        Op::Add,
        "(2^248 - 1) + 1 = 2^248 — carry into the c=5 region of limb 4".into(),
    ));
    recipes.push((
        by_name("2^248"),
        by_name("2^248"),
        Op::Add,
        "2 * 2^248 mod p = (2p + 2)/5 — modulus crossing in the top region".into(),
    ));

    // Multiplication accumulator stress.
    recipes.push((
        by_name("2^124"),
        by_name("2^124"),
        Op::Mul,
        "2^124 * 2^124 = 2^248 mod p — single-bit set in high limb".into(),
    ));
    recipes.push((
        by_name("2^127"),
        by_name("2^127"),
        Op::Mul,
        "2^127 * 2^127 = 2^254 mod p — top of canonical Fp range".into(),
    ));
    recipes.push((
        by_name("p-1"),
        by_name("0"),
        Op::Square,
        "(p-1)^2 = 1 — square specialization vs general mul".into(),
    ));

    // Subtraction underflow: borrow triggers a modulus add.
    recipes.push((
        by_name("0"),
        by_name("1"),
        Op::Sub,
        "0 - 1 = p-1 — borrow triggers modulus add".into(),
    ));
    recipes.push((
        by_name("1"),
        by_name("2"),
        Op::Sub,
        "1 - 2 = p-1 — minimal-magnitude borrow".into(),
    ));

    recipes
}

/// Applies `op` to the unbounded integers and returns the result
/// reduced to `[0, p)`. This is the reference computation against
/// which `Fp` is asserted byte-equal.
fn apply_op(op: Op, a: &NumBigInt, b: &NumBigInt) -> NumBigInt {
    let p = p();
    let raw = match op {
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
        Op::Neg => -a,
        Op::Square => a * a,
    };
    let mut r = raw % &p;
    if r.is_negative() {
        r += &p;
    }
    r
}

/// Runs `op` through production `Fp` and returns the canonical byte
/// encoding. Used only as the in-generator self-check.
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

fn main() {
    let inputs = boundary_inputs();
    let recipes = vector_recipes(&inputs);

    let mut entries: Vec<String> = Vec::new();
    for (tc, (a_idx, b_idx, op, comment)) in recipes.iter().enumerate() {
        let a_num = &inputs[*a_idx].value;
        let b_num = &inputs[*b_idx].value;
        let a_bytes = num_to_fp_bytes(a_num);
        let b_bytes = num_to_fp_bytes(b_num);
        let expected_num = apply_op(*op, a_num, b_num);
        let expected_bytes = num_to_fp_bytes(&expected_num);

        let production = fp_op(*op, &a_bytes, &b_bytes);
        assert_eq!(
            production, expected_bytes,
            "vector {tc} ({comment}): production diverges from reference",
        );

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
  "algorithm": "SQIsign_248_Fp",
  "schema": "fp_fault_simulation_schema.json",
  "generatorVersion": "sqisign-selkie-0.0.1",
  "numberOfTests": {n},
  "header": [
    "Boundary-input vectors for Fp arithmetic (p = 5*2^248 - 1).",
    "Each vector exercises a specific limb-boundary or modulus-crossing",
    "condition where a carry / reduction / overflow fault would diverge.",
    "Reference results computed via num-bigint at unbounded precision.",
    "Methodology: vector-forge skill, references/fault-simulation.md."
  ],
  "notes": {{
    "FaultSimulation": {{
      "bugType": "FUNCTIONALITY",
      "description": "Boundary input crafted to detect a specific carry-propagation, modular-reduction, or accumulator-overflow fault. Any conformant Fp implementation must produce the byte-equal expected output."
    }}
  }},
  "testGroups": [
    {{
      "type": "FpArithmetic",
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
        .join("tests/vectors/fp_fault_simulation.json");
    std::fs::write(&path, json).expect("write fp_fault_simulation.json");
    eprintln!("wrote {n} vectors to {}", path.display());
}
