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

## Seeding the corpus

libFuzzer is much more effective starting from inputs that already reach
the deep code paths. The bundled `examples/seed-fuzz-corpus` writes
seeds from two sources into the relevant `fuzz/corpus/<target>/`
directories:

| Source | Count | Where it lives |
| --- | --- | --- |
| NIST PQC KAT tuples | 100 | `keys::kat_data::KAT_VECTORS` (in-crate) |
| Wycheproof verify vectors | 52 | `tests/vectors/sqisign_verify.json` |
| Wycheproof extended-verify vectors | 24 | `tests/vectors/sqisign_verify_extended.json` |
| Wycheproof keygen vectors | 14 | `tests/vectors/sqisign_keygen.json` |
| Wycheproof sign vector | 1 | `tests/vectors/sqisign_sign.json` |

```sh
cargo run --release --features expose-internals --example seed-fuzz-corpus
```

The Wycheproof verify vectors include perturbed-invalid cases crafted
to land at parse / verify rejection boundaries — high-value seeds for
libFuzzer's mutation-around-known-good model. Each vector is routed
into the targets it makes sense for (e.g. sign vectors don't seed
`fuzz_sign_panic` because they don't carry the DRBG seed it needs).

Files are content-keyed by source + index (`kat-NNN`, `verify-NNN`,
`verify-ext-NNN`, `keygen-NNN`, `sign-NNN`). The corpus is
`.gitignore`'d so the seeder is re-run per checkout. Re-run it
manually after any change to `tests/vectors/sqisign_*.json`.

## Corpus policy

`fuzz/corpus/` is `.gitignore`'d; libFuzzer's coverage-grown corpus is
*not* tracked. Crash inputs from CI follow the same rule — they're not
auto-committed. Triage is manual:

1. CI's `Summarize crash candidates` step surfaces each crash as a
   base64-encoded heredoc in the run's Step Summary panel, with a
   Rust panic excerpt + reproducer command. The same blobs are also
   echoed into the workflow log under a `::group::Crash inputs`
   block, grep-able via `gh run view --log`.
2. A maintainer pastes the heredoc locally:
   ```sh
   base64 -d <<'EOF' > /tmp/crash-input
   QkFTRTY0X0VOQ09ERURfQllURVMK...
   EOF
   cargo +nightly fuzz run <target> /tmp/crash-input
   ```
3. If it's a real bug worth keeping as a regression seed, they copy
   it into `fuzz/corpus/<target>/` and force-add (the directory is
   gitignored):
   ```sh
   mv /tmp/crash-input fuzz/corpus/<target>/crash-<sha>
   git add -f fuzz/corpus/<target>/crash-<sha>
   git commit -m "fuzz: add crash input for <target> (<short-sha>)"
   ```

So in this repo:

- Untracked corpus files: KAT- / Wycheproof-derived seeds + libFuzzer's
  coverage-grown set, regrown locally / cached per CI run.
- Tracked corpus files (mostly `crash-*` prefixed): triaged regressions
  from past CI runs that we want to keep replaying.

When you find a crash locally, the same pattern applies: copy the
offending input into the matching `corpus/<target>/` directory and
force-add it along with the fix.

## Adding a new fuzz target

1. Write `fuzz_targets/<name>.rs` with `fuzz_target!(|data| { … })`.
2. Add `[[bin]] name = "<name>"` to `Cargo.toml`.
3. Add `<name>` to the `matrix.target` list in `fuzz.yml`.
4. Add a row to the table above with what it covers and its class.
5. If seeds from KAT or Wycheproof vectors are meaningful, extend
   `examples/seed-fuzz-corpus.rs` to seed `corpus/<name>/` from
   whichever source applies (look at `Seeder::seed_from_*_file` for
   the existing per-source helpers).
