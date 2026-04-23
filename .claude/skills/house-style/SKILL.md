---
name: house-style
description: Enforces sqisign-selkie-specific rules layered on top of the global rust-style skill. Use whenever writing, editing, reviewing, or refactoring Rust code in the sqisign-selkie crate; whenever implementing a spec algorithm or porting from the C reference; whenever touching the typed wrappers in src/curves, src/quaternions, src/surfaces, src/deuring, or src/keys. Covers constant-time requirements on secret data, spec-first implementation policy, NIST-I parameter discipline, project type conventions, and the divergence-documentation workflow. Depends on the rust-style skill for general Rust idioms.
---

# sqisign-selkie house style

This skill layers project-specific rules on top of the global
[`rust-style`] skill. Apply both — `rust-style` for general Rust
conventions (methods-on-types, From/TryFrom, RFC 1574 rustdoc, code
organization, commit policy) and the rules below for sqisign-selkie's
crypto-specific and spec-driven requirements.

When this skill is active, announce both: "Using rust-style and
house-style." Then load `rust-style` if it isn't already in context.

## Rule 1: Constant-time on secrets is non-negotiable

All code computing over secret or secret-derived values MUST be
constant-time. No exceptions. Variable-time is acceptable ONLY on truly
public data (e.g., verification of a public signature).

- "The C reference is variable-time" is NEVER an excuse.
- Before writing any algorithm: trace its callers back to signing/keygen
  (Algorithm 4.2). If any caller passes secret data, the algorithm is
  constant-time or it doesn't ship.
- Branch on secrets via `subtle::Choice` and `ConditionallySelectable`.
  Never via `if`, `match`, or short-circuit operators.
- Implement `ConditionallySelectable` on every type that may participate
  in CT branching.
- Variable-time code touching secrets must be tagged
  `// TODO(ct): Make constant-time before production use.` and cite which
  Algorithm 4.x line makes the input secret-derived.

### Forbidden trait impls on crypto types

Beyond what `rust-style` says about traits:

- **`PartialEq` / `Eq` / `ConstantTimeEq` on secret-key types** —
  comparing secrets is a code smell. Remove on sight.
- **`Default` / `Hash` / `AsRef<[u8]>` on crypto types** unless there
  is a concrete, justified caller. `Default` implies a meaningful zero
  (rare in crypto), `Hash` implies safe hash-map use (risky for
  secrets), `AsRef<[u8]>` exposes internals.
- **`PartialEq` / `Eq` on projective points via cross-multiplication**,
  not structural field comparison. On affine types and field elements,
  derive normally.

## Rule 2: Spec is the source of truth, C reference is the last resort

- Read the SQIsign v2 spec (`references/sqisign-20250707.pdf`) before
  implementing any algorithm. Read the surrounding prose, not just the
  pseudocode box — it defines types, invariants, and relationships
  between algorithms that pseudocode alone does not convey.
- Translate the spec's formulas to Rust. Do not copy the C reference's
  code structure, variable naming, or strategy. Their `t0`–`t4` temp
  chains produce unreadable Rust and obscure mathematical intent.
- Never invent operations the spec doesn't specify. No "extra"
  doublings or normalizations because they "should" be there. Invented
  steps break interop and are the hardest bugs to find.
- Consult the C ref only when: (1) the spec is ambiguous, (2) your
  spec-based code fails interop tests, or (3) you need an undocumented
  convention. Extract the mathematical idea, implement it from the
  spec's formulas in idiomatic Rust.
- Document every divergence from spec or C ref in rustdoc at the
  relevant code site, in `latex/spec-review.tex` if it's a spec gap,
  and in the paper if it's a bug.

## Rule 3: NIST-I only, fixed at compile time

- All parameters live in `src/params.rs`. Use them directly.
- Do not be generic over parameter sets. Specialize aggressively:
  field arithmetic is radix-51 Montgomery tuned to this prime, integer
  widths match Kim et al. (ePrint 2025/1649) worst-case bounds.
- Before adding a function parameter, ask: is this value fixed for
  NIST-I? If yes, reference the constant internally instead of exposing
  it. Passing a compile-time constant as an argument adds API surface
  for no generality.
- Use `TorsionExponent` for exponents that vary per signature but are
  bounded by f=248. Never bare `u32`.

## Rule 4: Project type conventions

Use these typed wrappers — never bare representations:

- `Lattice<N>` vs `HnfLattice<N>` — HNF guarantee in the type
- `Order<N>` vs `Lattice<N>` — only constructible by operations that
  produce orders (`right_order()`, `ExtremalOrder::order()`,
  `Order::from_lattice_unchecked`)
- `Coefficient` vs `AffineX` vs `RootOfUnity` — all wrap `Fp2`, none
  interchangeable
- `ProjectiveCoefficient`, `DoublingConstants`, `Curve` — three cached
  representations of the same Montgomery curve
- `TorsionExponent` — bounded `u32`, never bare `u32`
- `BigInt<4>` for storage, `BigInt<N>` for wider arithmetic (use
  `widen::<W>()` and `narrow_to::<T>()`)
- `IsogenyDegree` — positive odd `[u64; 4]`
- `Challenge`, `ChallengeMatrix`, `SecretKeyMatrix` — newtypes over
  scalar/matrix primitives that prevent cross-wiring
- Projective fields: uppercase `X`, `Z`. Affine: lowercase `x`, `y`.

## Rule 5: Documentation — project-specific headings

In addition to the RFC 1574 set covered by `rust-style`, sqisign-selkie
uses these additional standard headings:

- `# Constant-time` — explicit CT contract for any function that may
  touch secrets. Either:
  - "Constant-time on `<inputs>`." listing which inputs are protected,
    OR
  - "Variable-time. `TODO(ct)`: input is secret-derived via Algorithm
    4.2 line N."

  Functions on purely public data may omit this section.
- `# Divergences` — when this function diverges from the spec or C
  reference. State what diverges, why, and the resulting behavior.
  Add a corresponding entry to `latex/spec-review.tex` if it's a spec
  gap.

### Spec link style

Spec links use `#algorithm.X.Y`, `#section.X.Y`, `#subsection.X.Y.Z`,
or `#subsubsection.X.Y.Z.W` anchors (extracted via `pypdf`), never
`#page=N`. Use bare reference-style labels — no algorithm name prefix:

```rust
/// Implements [Alg. 3.13].
///
/// [Alg. 3.13]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.13
```

For section references use `[§X.Y]` / `[§X.Y.Z]` labels matching the
spec's numbering, and pick the anchor depth that matches the label
(`#section.X.Y` for two-deep, `#subsection.X.Y.Z` for three-deep,
`#subsubsection.X.Y.Z.W` for four-deep). A label like `[§8.5.5]`
pointing at `#section.8.5` is a bug — the click lands at the parent
section, not the subsection.

### Where to put divergences and references

- **Divergences from spec / C reference**: rustdoc at the relevant code
  site, plus a "Divergences from spec / C reference" section in the
  module-level `//!` doc. If it's a spec gap, also add to
  `latex/spec-review.tex` with severity, affected algorithm, current
  spec text, what goes wrong, and recommendation.
- **Bugs we fixed**: comments at the fix site, plus the paper's
  `§4 Bug Catalog` (`latex/main.tex`).
- **Never reference internal files** (`.claude/`, memory paths, agent
  files) from code comments or rustdoc.
- **Never reference `latex/` files from rustdoc.** Rustdoc readers
  don't have access to the LaTeX sources. Rustdoc divergence/bug
  sections must be self-contained. The LaTeX documents
  (`spec-review.tex`, `sections/04_bugs.tex`) are for the paper and
  spec authors, not for API consumers.
- **Never mention "house-style" in code, rustdoc, commit messages, or
  documentation.** The skill is internal tooling: code stands on its
  own merits, and rustdoc citing "per house-style Rule N" leaks the
  development process into the public artifact. State the reason the
  rule exists (e.g., "NIST-I is the only parameter set we target"),
  not that a rule exists. Same for `rust-style` and any other skill.

## Rule 6: Never panic after compilation

No runtime panics in production code paths. Once a build succeeds,
the compiled artifact must not `panic!`, `unwrap`, `expect`, index
out of bounds, or otherwise abort at runtime on any input — trusted
or adversarial. Cryptographic code that panics is a DoS channel,
potential side channel, and fails to uphold the "illegal states
unrepresentable" principle.

- **Prefer compile-time enforcement.** Use `const { assert!(...) }`,
  const generics, and typed wrappers to push checks into the type
  system. The assertion fires during monomorphization, not at
  runtime.
- **Return `Result` / `Option` for fallible runtime conditions.**
  Validation at system boundaries (parsing, deserialization, FFI)
  returns errors; internal code relies on type invariants.
- **`unwrap`/`expect` are acceptable only where the invariant is
  provably upheld by an adjacent explicit check or a type-level
  guarantee, and a comment states why.** The comment should
  reference the invariant, not just assert "cannot happen".
- **Exceptions:**
  - `#[cfg(test)]` code: `unwrap`/`expect`/`panic!` are fine in
    tests — they are the test assertion mechanism.
  - `#[cfg(debug_assertions)]` code: debug-only sanity checks may
    panic; they compile out of release builds.
  - `const fn` and `const { ... }` blocks: panicking there happens
    at compile time, which is exactly what we want.
  - Truly unreachable code after exhaustive matching on a closed
    enum: `unreachable!()` with a comment is acceptable, but
    prefer refactoring to make the case structurally impossible.
- **No `todo!()`, `unimplemented!()`, or bare `panic!()` in
  non-test code merged to `main`.** WIP goes on branches.
- **Array/slice indexing:** prefer `.get()` + `?`, iterator
  methods, or destructuring over bracket-indexing when the index
  is runtime-derived. Constant indices into fixed-size arrays are
  fine (bounds are checked at compile time for `[T; N]`).
- **Integer arithmetic:** use `checked_*` / `wrapping_*` /
  `saturating_*` explicitly when overflow is possible. Bare `+`,
  `-`, `*` panic on debug builds and wrap silently in release —
  both are wrong for crypto. For `BigInt<N>` fixed-precision ops,
  use the `ct_` family and `widen`/`narrow_to` to explicitly
  track the width budget.

This is enforced via code review. `cargo +nightly clippy
--all-targets --all-features -- -D warnings` catches many (but
not all) panic sites; reviewers check the rest.

## Rule 7: Keep benchmarks in sync

Benchmark files live in `benches/` (one per module: `field.rs`,
`bigint.rs`, `curves.rs`, `surfaces.rs`, `quaternions.rs`, `deuring.rs`,
`sqisign.rs`). They use `divan`.

When a change affects benchmarked code — renaming a type or method,
changing a function signature, adding a new public operation, or
removing an existing one:

- **Update the affected bench file** so it compiles and exercises the
  new API. A bench that doesn't compile is a broken CI gate.
- **Add a benchmark** for any new operation that is likely to be a
  performance-sensitive hotpath (field arithmetic, isogeny evaluation,
  lattice reduction, ideal-to-isogeny, signing, verification).
- **Remove benchmarks** for operations that no longer exist. Don't
  leave commented-out benches.
- **Run `cargo bench --bench <name> -- --test`** (divan dry-run) to
  verify the bench compiles and runs before committing.

## Rule 8: Negative and vulnerability test vectors

Edge-case, negative, and vulnerability test vectors live in
`tests/vectors/` as JSON in the
[C2SP/wycheproof](https://github.com/C2SP/wycheproof) format. The
test runner is `tests/wycheproof.rs` — a top-level integration test
that uses only the public API and is portable to other SQIsign
implementations.

Invalid vectors must be independently generated (by perturbation or
Sage), not derived from the C reference's behavior. Valid vectors
currently cite the C reference KAT file as provenance; the goal is
to replace them with Sage-generated vectors for upstream contribution
to C2SP/wycheproof.

### Algorithm naming

Use **`SQIsign_248`** as the `algorithm` field. The `248` is the
torsion exponent f, the fundamental parameter from which all others
derive (p = 5·2^248 − 1). This is stable across encoding changes and
security-level relabeling.

### What vectors cover

Vectors detect whether an implementation is **vulnerable**, not just
buggy. Categories span verification, signing, and key generation:

**Verification:**
- Wrong message, wrong pk, cross-vector pk swap, bit flips in every
  signature field, all-zero/all-ones signatures, truncated/extended
  signatures, out-of-range n_bt/r_rsp.
- Malformed curve attacks: E_aux not supersingular, E_aux = E_0
  (trivial curve), j-invariant 0 or 1728 (automorphisms).
- Torsion boundary cases: e_rsp = 0 (skip (2,2)-chain), n_bt = f−1
  (maximum backtracking), r_rsp at its maximum.
- Parsing robustness: non-canonical field encodings (≥ p), matrix
  entries exceeding the scalar bound, trailing garbage bytes, hint
  values outside valid range.

**Signing:**
- Signatures produced by our implementation must verify against the
  C reference (KAT round-trip).
- Side-channel probes: inputs crafted so that signing hits
  variable-time code paths differently depending on implementation
  choices (e.g., non-canonical square root branch, Cornacchia
  branching).
- Nonce bias: signatures over identical messages with different
  randomness must produce distinct commitment curves (no nonce
  reuse).
- Response phase edge cases: D_rsp requiring maximum lattice width,
  represent_integer needing many retries, degenerate ideals.

**Key generation:**
- Deterministic keygen from KAT seeds must produce matching (pk, sk)
  byte-for-byte.
- sk round-trip: parse → serialize → re-parse must be identical.
- pk embedded in sk must match standalone pk.
- Seeds that maximize retries, seeds that produce degenerate ideals,
  sk with norm = 1.
- Malformed sk: truncated, extended, non-canonical field elements,
  invalid ideal generator.

### Adding vectors

- Valid vectors come from the C reference KAT file. Invalid vectors
  are derived by perturbation (bit flips, field swaps, truncation)
  or crafted in Sage to target specific code paths.
- Each vector has `tcId`, `comment`, `msg` (hex), `sig` (hex),
  `result` ("valid"/"invalid"/"acceptable"), and `flags`.
- Flags are defined in the top-level `notes` dictionary with
  `bugType`, `description`, and optional `links`/`cves`.
- `numberOfTests` must match the actual count — the runner asserts
  this.
- Generate vectors programmatically (Python/Sage) rather than
  hand-editing hex. Document the generator in the `header`.

### When to add vectors

When fixing a bug or vulnerability in verification, signing,
parsing, or keygen, add a vector that would have caught it. When
adding a new rejection path, add a vector that exercises it. When a
published attack (e.g., SPA on Cornacchia, invalid curve) applies to
SQIsign, add vectors that detect whether an implementation is
vulnerable.

The goal is that `tests/vectors/*.json` is a standalone artifact
other implementations can consume to validate their own security.

### Running

```bash
cargo test --test wycheproof
```

## Rule 9: Run nightly fmt and -D warnings clippy

Per `rust-style` Rule 6, but specifically for this crate:

```bash
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-targets --all-features -- -D warnings
cargo test --lib
```

The `rustfmt.toml` uses nightly-only options (`imports_granularity`,
`wrap_comments`, `format_code_in_doc_comments`), so always use
`+nightly` for fmt. Always use `--all-targets --all-features -- -D
warnings` for clippy — that's the CI command.

## Rule 9: CI scripts in Rust

Write CI helper scripts for GitHub Actions in Rust (standalone `.rs`
files in `.github/scripts/`, compiled with `rustc -O` in the workflow)
once they exceed ~10 lines of inline bash in the workflow YAML. This
removes bash/Python/jq dependencies from CI and keeps the toolchain
homogeneous. All CI scripts live in `.github/scripts/` as `.rs` files.

[`rust-style`]: ../rust-style/SKILL.md
