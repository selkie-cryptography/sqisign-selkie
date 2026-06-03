//! Regenerates `tests/vectors/fp2_fault_simulation.json` from the
//! boundary-input definitions below.
//!
//! The JSON file is the source of truth for the
//! [`fp2_fault_simulation_vectors`](../tests/fp2_fault_simulation.rs)
//! test; this generator exists so reviewers can audit the
//! `(input, op) -> expected` recipe instead of staring at hex bytes,
//! and so the file can be regenerated when the boundary set evolves.
//!
//! Run:
//!
//! ```sh
//! cargo run --release --features expose-internals --example fp2-fault-vectors-gen
//! ```
//!
//! Reference results are computed via two-component `num-bigint`
//! arithmetic at unbounded precision, so the generator is independent
//! of `Fp2` itself — but the file the generator emits gets a self-
//! check pass against production `Fp2` before being written.
//!
//! Methodology: vector-forge skill, `references/fault-simulation.md`.

use num_bigint::BigInt as NumBigInt;
use num_traits::{One, Signed, Zero};
use sqisign_selkie::{
    fields::fp2::Fp2,
    params::{FP_ENCODED_BYTES, FP2_ENCODED_BYTES},
};

/// `p = 5 · 2²⁴⁸ − 1`, the base prime, as an unbounded integer.
fn p() -> NumBigInt {
    (NumBigInt::from(5u32) << 248) - NumBigInt::one()
}

/// Reduces an unbounded integer to `[0, p)`.
fn reduce(n: NumBigInt) -> NumBigInt {
    let p = p();
    let mut r = n % &p;
    if r.is_negative() {
        r += &p;
    }
    r
}

/// Encodes one Fp component as 32 LE bytes.
fn fp_to_le(n: &NumBigInt) -> [u8; FP_ENCODED_BYTES] {
    let (_, mag) = reduce(n.clone()).to_bytes_le();
    let mut out = [0u8; FP_ENCODED_BYTES];
    out[..mag.len()].copy_from_slice(&mag);
    out
}

/// Encodes an Fp² element `a + bi` as 64 LE bytes: 32 for `a`, 32 for
/// `b`, matching [`Fp2::from_bytes`]'s contract.
fn fp2_to_le(a: &NumBigInt, b: &NumBigInt) -> [u8; FP2_ENCODED_BYTES] {
    let mut out = [0u8; FP2_ENCODED_BYTES];
    out[..FP_ENCODED_BYTES].copy_from_slice(&fp_to_le(a));
    out[FP_ENCODED_BYTES..].copy_from_slice(&fp_to_le(b));
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

/// Boundary input as a symbolic name plus the `(real, imag)` Fp²
/// component pair. The name appears in the generated comment to keep
/// the vector human-auditable; the integers drive the encoding.
struct BoundaryInput {
    name: &'static str,
    real: NumBigInt,
    imag: NumBigInt,
}

/// Returns the Fp² boundary inputs used to seed the `(a, b)` pairings
/// below.
fn boundary_inputs() -> Vec<BoundaryInput> {
    let p = p();
    let one = NumBigInt::one();
    let zero = NumBigInt::zero();
    let p_minus_one: NumBigInt = &p - &one;
    let two_127: NumBigInt = &one << 127;
    let half: NumBigInt = (&p - &one) / 2u32;
    vec![
        BoundaryInput {
            name: "0",
            real: zero.clone(),
            imag: zero.clone(),
        },
        BoundaryInput {
            name: "1",
            real: one.clone(),
            imag: zero.clone(),
        },
        BoundaryInput {
            name: "i",
            real: zero.clone(),
            imag: one.clone(),
        },
        BoundaryInput {
            name: "1+i",
            real: one.clone(),
            imag: one.clone(),
        },
        BoundaryInput {
            name: "1-i",
            real: one.clone(),
            imag: p_minus_one.clone(),
        },
        BoundaryInput {
            name: "p-1",
            real: p_minus_one.clone(),
            imag: zero.clone(),
        },
        BoundaryInput {
            name: "(p-1)i",
            real: zero.clone(),
            imag: p_minus_one.clone(),
        },
        BoundaryInput {
            name: "(p-1)+(p-1)i",
            real: p_minus_one.clone(),
            imag: p_minus_one.clone(),
        },
        BoundaryInput {
            name: "(p-1)+i",
            real: p_minus_one.clone(),
            imag: one.clone(),
        },
        BoundaryInput {
            name: "2^127·(1+i)",
            real: two_127.clone(),
            imag: two_127.clone(),
        },
        BoundaryInput {
            name: "((p-1)/2)·(1+i)",
            real: half.clone(),
            imag: half.clone(),
        },
    ]
}

/// Returns `(a, b, op, comment)` tuples whose expected outputs become
/// the vector set. Each tuple is chosen so its result lands on a
/// specific Fp²-formula-level boundary.
fn vector_recipes(inputs: &[BoundaryInput]) -> Vec<(usize, usize, Op, String)> {
    let by_name = |n: &str| {
        inputs
            .iter()
            .position(|x| x.name == n)
            .unwrap_or_else(|| panic!("missing boundary input {n}"))
    };

    vec![
        // Component-wise overflow on add: real, imag, both.
        (
            by_name("p-1"),
            by_name("1"),
            Op::Add,
            "(p-1) + 1 = 0 — real component overflows".into(),
        ),
        (
            by_name("(p-1)i"),
            by_name("i"),
            Op::Add,
            "(p-1)i + i = 0 — imag component overflows".into(),
        ),
        (
            by_name("(p-1)+(p-1)i"),
            by_name("1+i"),
            Op::Add,
            "(p-1)+(p-1)i + 1+i = 0 — both components overflow".into(),
        ),
        (
            by_name("(p-1)+i"),
            by_name("1-i"),
            Op::Add,
            "(p-1+i) + (1-i) = 0 — symmetric carry, asymmetric inputs".into(),
        ),
        (
            by_name("2^127·(1+i)"),
            by_name("2^127·(1+i)"),
            Op::Add,
            "limb-cross stress on both components simultaneously".into(),
        ),
        // Component-wise underflow on sub.
        (
            by_name("0"),
            by_name("p-1"),
            Op::Sub,
            "0 - (p-1) = 1 — real component underflows".into(),
        ),
        (
            by_name("0"),
            by_name("(p-1)i"),
            Op::Sub,
            "0 - (p-1)i = i — imag component underflows".into(),
        ),
        (
            by_name("0"),
            by_name("(p-1)+(p-1)i"),
            Op::Sub,
            "0 - ((p-1)+(p-1)i) = 1+i — both underflow".into(),
        ),
        // Negation at canonical max.
        (
            by_name("p-1"),
            by_name("0"),
            Op::Neg,
            "-(p-1) = 1 — real negation at canonical max".into(),
        ),
        (
            by_name("(p-1)i"),
            by_name("0"),
            Op::Neg,
            "-(p-1)i = i — imag negation at canonical max".into(),
        ),
        // The i² = -1 axiom. Sign flip in the `ac − bd` real-part
        // formula (i.e. mutating `−` to `+`) makes this `(1, 0)`
        // instead of `(p-1, 0)`. Component swap makes it `(0, p-1)`.
        (
            by_name("i"),
            by_name("i"),
            Op::Mul,
            "i * i = p-1 — the defining axiom of Fp²".into(),
        ),
        // 1 · i and i · 1 must both yield i; either order reveals a
        // component-swap mutation in the cross-term formula.
        (
            by_name("1"),
            by_name("i"),
            Op::Mul,
            "1 * i = i — identity preserves orientation".into(),
        ),
        (
            by_name("i"),
            by_name("1"),
            Op::Mul,
            "i * 1 = i — symmetric to above".into(),
        ),
        // (1+i)·(1+i) = 2i. Real part is `1·1 − 1·1 = 0`; imag is
        // `1·1 + 1·1 = 2`. `−`→`+` real-part mutation gives `(2, 2)`.
        (
            by_name("1+i"),
            by_name("1+i"),
            Op::Mul,
            "(1+i)*(1+i) = 2i — cross-term cancellation".into(),
        ),
        // (1+i)·(1-i) = 1 - i² = 2. Real `1·1 − 1·(p-1) = 2`; imag
        // `1·(p-1) + 1·1 = p ≡ 0`. The norm of (1+i).
        (
            by_name("1+i"),
            by_name("1-i"),
            Op::Mul,
            "(1+i)*(1-i) = 2 — Fp² norm of (1+i) via conjugate mul".into(),
        ),
        // ((p-1)+(p-1)i)² and the same as Mul produce identical
        // results; differing answers point at a bug in the square
        // specialization (`a² − b²`, `2ab`) vs general mul path.
        (
            by_name("(p-1)+(p-1)i"),
            by_name("(p-1)+(p-1)i"),
            Op::Mul,
            "saturated * saturated — full Fp²-mul cross-term stress".into(),
        ),
        // i · (a + bi) = -b + ai. Rotates by 90°. Swap of `a` / `b`
        // in the cross-term emerges as the wrong rotation direction.
        (
            by_name("i"),
            by_name("(p-1)+(p-1)i"),
            Op::Mul,
            "i * ((p-1)+(p-1)i) = 1 + (p-1)i — rotation by 90°".into(),
        ),
        // Pure-real squared via the cross-term path.
        (
            by_name("p-1"),
            by_name("p-1"),
            Op::Mul,
            "(p-1) * (p-1) = 1 — pure-real, no imag involvement".into(),
        ),
        // Pure-imag squared: (p-1)²·i² = -(p-1)² = -1 = p-1.
        (
            by_name("(p-1)i"),
            by_name("(p-1)i"),
            Op::Mul,
            "((p-1)i)*((p-1)i) = p-1 — pure-imag squared, i² hit".into(),
        ),
        // Limb stress via Fp² mul: 2¹²⁷(1+i) squared produces 2²⁵⁴
        // in each component. Propagates Fp's high-limb fault classes
        // through Fp² arithmetic.
        (
            by_name("2^127·(1+i)"),
            by_name("2^127·(1+i)"),
            Op::Mul,
            "(2^127·(1+i))² — Fp high-limb stress through Fp² mul".into(),
        ),
        // Square specialization checks via the dedicated `square()`
        // path; same inputs as mul, different formulas internally.
        (
            by_name("1+i"),
            by_name("0"),
            Op::Square,
            "(1+i)² = 2i via square() — matches mul cross-term".into(),
        ),
        (
            by_name("(p-1)+i"),
            by_name("0"),
            Op::Square,
            "((p-1)+i)² = (0, p-2) — square at the boundary".into(),
        ),
        (
            by_name("(p-1)+(p-1)i"),
            by_name("0"),
            Op::Square,
            "((p-1)+(p-1)i)² — saturated input through square()".into(),
        ),
    ]
}

/// Two-component reference: `(real, imag)` in `[0, p) × [0, p)`.
type Fp2Ref = (NumBigInt, NumBigInt);

/// Applies `op` to the unbounded `(a, b)` pairs and returns the result
/// reduced componentwise.
fn apply_op(op: Op, a: &Fp2Ref, b: &Fp2Ref) -> Fp2Ref {
    let (ar, ai) = a;
    let (br, bi) = b;
    match op {
        Op::Add => (reduce(ar + br), reduce(ai + bi)),
        Op::Sub => (reduce(ar - br), reduce(ai - bi)),
        // (ar + ai·i)(br + bi·i) = (ar·br − ai·bi) + (ar·bi + ai·br)·i
        Op::Mul => (reduce(ar * br - ai * bi), reduce(ar * bi + ai * br)),
        Op::Neg => (reduce(-ar), reduce(-ai)),
        // (a + bi)² = (a² − b²) + (2ab)i
        Op::Square => (reduce(ar * ar - ai * ai), reduce(2u32 * ar * ai)),
    }
}

/// Runs the same op through production `Fp2` and returns the
/// canonical byte encoding. Used only as the in-generator self-check:
/// every emitted vector is run through production before being
/// written, so a recipe bug surfaces here, not in CI.
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

fn main() {
    let inputs = boundary_inputs();
    let recipes = vector_recipes(&inputs);

    let mut entries: Vec<String> = Vec::new();
    for (tc, (a_idx, b_idx, op, comment)) in recipes.iter().enumerate() {
        let a = &inputs[*a_idx];
        let b = &inputs[*b_idx];
        let a_ref: Fp2Ref = (a.real.clone(), a.imag.clone());
        let b_ref: Fp2Ref = (b.real.clone(), b.imag.clone());
        let a_bytes = fp2_to_le(&a.real, &a.imag);
        let b_bytes = fp2_to_le(&b.real, &b.imag);
        let (er, ei) = apply_op(*op, &a_ref, &b_ref);
        let expected_bytes = fp2_to_le(&er, &ei);

        let production = fp2_op(*op, &a_bytes, &b_bytes);
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
  "algorithm": "SQIsign_248_Fp2",
  "schema": "fp2_fault_simulation_schema.json",
  "generatorVersion": "sqisign-selkie-0.0.1",
  "numberOfTests": {n},
  "header": [
    "Boundary-input vectors for Fp² arithmetic (Fp² = Fp[i] / (i² + 1)).",
    "Each vector exercises a specific Fp²-formula-level boundary: i² = -1,",
    "cross-term sign/swap in mul, square specialization vs general mul,",
    "componentwise overflow / underflow, limb stress propagated through Fp².",
    "Reference results computed via two-component num-bigint arithmetic.",
    "Methodology: vector-forge skill, references/fault-simulation.md."
  ],
  "notes": {{
    "FaultSimulation": {{
      "bugType": "FUNCTIONALITY",
      "description": "Boundary input crafted to detect a specific cross-term, sign, or component-swap fault in the Fp² formulas. Any conformant Fp² implementation must produce the byte-equal expected output."
    }}
  }},
  "testGroups": [
    {{
      "type": "Fp2Arithmetic",
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
        .join("tests/vectors/fp2_fault_simulation.json");
    std::fs::write(&path, json).expect("write fp2_fault_simulation.json");
    eprintln!("wrote {n} vectors to {}", path.display());
}
