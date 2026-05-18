# Fuzz targets

Coverage-guided fuzzing via [`cargo-fuzz`](https://rust-fuzz.github.io/book/cargo-fuzz.html)
(libFuzzer underneath). Each target lives at `fuzz_targets/<name>.rs` and is
registered as a `[[bin]]` in `Cargo.toml`. The matrix in
[`.github/workflows/fuzz.yml`](../.github/workflows/fuzz.yml) runs every
target for 10 minutes on each push to `main`.

## Targets

| Target | Input | Surface | Class |
| --- | --- | --- | --- |
| `fuzz_signature_parse` | `&[u8; 148]` | `Signature::from_bytes` | parse |
| `fuzz_verifying_key_parse` | `&[u8; 65]` | `VerifyingKey::from_bytes` | parse |
| `fuzz_signing_key_parse` | `&[u8; 353]` | `SigningKey::from_bytes` | parse |
| `fuzz_verify` | `pk \|\| sig \|\| msg` | `VerifyingKey::verify` (raw bytes) | verify |
| `fuzz_verify_typed` | KAT-anchored sig + msg | `VerifyingKey::verify` (structured) | verify |
| `fuzz_keygen_panic` | `&[u8; 48]` (DRBG seed) | `SigningKey::generate_derand` | slow / panic-on-input |
| `fuzz_sign_panic` | seed \|\| seed \|\| msg | keygen → `sign_derand` | slow / panic-on-input |

**Parse targets** are fast (millions of execs/min); their job is to assert
that `from_bytes` never panics on adversarial input. Crashes here usually
indicate a missing canonicality check.

**Verify targets** are slower (~100 execs/min). `fuzz_verify` feeds raw
bytes — most iters bounce at `Signature::from_bytes`. `fuzz_verify_typed`
anchors `vk` and `sig.curve_aux` to KAT 0 and XORs libFuzzer's input over
the rest of the signature, so every iter reaches the (2,2)-chain. Run
both: they cover complementary code regions.

**Slow / panic-on-input targets** (`fuzz_keygen_panic`, `fuzz_sign_panic`)
get only a handful of iterations per 10-min slot. The goal isn't coverage
saturation — it's that *any* panic libFuzzer catches is a real
panic-on-input bug. Worth running for hours when touching keygen or sign
internals.

## Running locally

```sh
# Build a target.
cargo +nightly fuzz build fuzz_verify

# Run for a fixed time.
cargo +nightly fuzz run fuzz_verify -- -max_total_time=600

# Run against the corpus only (no new mutations); good for sanity-checking
# that all seeds still parse after a parser change.
cargo +nightly fuzz run fuzz_verify -- -runs=0 corpus/fuzz_verify
```

## Seeding the corpus from KAT vectors

libFuzzer is much more effective starting from inputs that already reach
the deep code paths. The bundled `examples/seed-fuzz-corpus` writes
KAT-derived seeds into the relevant `fuzz/corpus/<target>/` directories:

```sh
cargo run --release --features expose-internals --example seed-fuzz-corpus
```

It seeds the parse targets, the two verify targets, and the
keygen/sign panic targets. Each KAT vector contributes one corpus file
per applicable target. The corpus is `.gitignore`'d, so seeding is
re-run per checkout.

## Corpus policy

`fuzz/corpus/` is `.gitignore`'d by default; libFuzzer's coverage-grown
corpus is *not* tracked. The exception is **crash inputs from CI**:
when a fuzz run finds a panic, the `Open PR on crash` step in
`fuzz.yml` force-adds the crashing input under `fuzz/corpus/<target>/`
in a separate branch and opens a PR. Once triaged into a regression
test, the file usually stays in-tree as a corpus seed.

So in this repo:

- Untracked corpus files: working / CI-cached set, regrown by libFuzzer.
- Tracked corpus files (mostly `crash-*` prefixed): triaged regressions
  from past CI runs that we want to keep replaying.

When you find a crash locally, copy the offending input into the matching
`corpus/<target>/` directory and commit it along with the fix.

## Adding a new fuzz target

1. Write `fuzz_targets/<name>.rs` with `fuzz_target!(|data| { … })`.
2. Add `[[bin]] name = "<name>"` to `Cargo.toml`.
3. Add `<name>` to the `matrix.target` list in `fuzz.yml`.
4. Add a row to the table above with what it covers and its class.
5. If KAT-derived seeds are meaningful, extend
   `examples/seed-fuzz-corpus.rs` to seed `corpus/<name>/`.
