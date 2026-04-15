# sqisign-selkie

A compact post-quantum signature scheme from quaternions and isogenies, in Rust.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/selkie-cryptography/.github/raw/main/assets/selkie-solid-white-on-transparent.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/selkie-cryptography/.github/raw/main/assets/selkie-solid-black-on-transparent.svg">
  <img width="25%" align="right" src="https://github.com/selkie-cryptography/.github/raw/main/assets/selkie-solid-black-on-transparent.svg" alt="Selkie logo">
</picture>

Implements [SQIsign][sqisign] as specified in the [v2.0.1
specification][spec] (2025-07-07), targeting the NIST-I parameter set
(p = 5 · 2²⁴⁸ − 1).

> **Status: work in progress.** Key generation and verification are
> functional. Signing is structurally complete with commitment phase
> working; the response phase is slow due to unoptimized quaternion
> arithmetic. All 100 C reference KAT verification vectors pass.
> Signing and key generation are **not yet constant-time** — the
> quaternion layer is variable-time with `TODO(ct)` markers throughout.
> **Do not use in production.**

## Example

<!-- TODO: remove no_run once signing completes in reasonable time -->
```rust,no_run
use sqisign_selkie::{SigningKey, VerifyingKey, SIGNATURE_BYTES};

// Generate a new signing key.
let sk = SigningKey::generate(&mut rand_core::OsRng)
    .expect("key generation failed");

// Sign a message.
let message = "Maighdean mhara mo mháithrín ard".as_bytes();
let signature = sk.sign(message, &mut rand_core::OsRng)
    .expect("signing failed");

// The verifying key can be extracted and serialized.
let vk_bytes = sk.verifying_key().to_bytes();

// Verify the signature.
let vk = VerifyingKey::from_bytes(&vk_bytes)
    .expect("invalid verifying key");
vk.verify(message, &signature)
    .expect("invalid signature");
```

## Design principles

**Spec-first.** Algorithms are translated directly from the SQIsign
v2.0.1 specification, not transliterated from the C reference
implementation. The C reference is consulted only to resolve spec
ambiguities and for interoperability testing.

**Type-safe.** Illegal states are unrepresentable at compile time.
Newtypes enforce domain constraints throughout: `Challenge`, `Scalar`,
`TorsionExponent`, `Coefficient`, `ProjectiveCoefficient`,
`DoublingConstants`, `Order<N>`, `HnfLattice<N>`, `IdealFactor`, and
others prevent cross-wiring of parameters that are bare integers in the
C reference.

**Fixed-precision quaternion arithmetic.** Where the C reference uses
GMP (arbitrary precision), we use fixed-width `BigInt<N>` generic over
the number of 64-bit limbs. Worst-case bounds from [Kim et al.][kim]
(ePrint 2025/1649) establish N=110 limbs (7,040 bits) as sufficient for
NIST-I.

# Building and Testing

```sh
cargo test --lib              # run the full test suite (~200 tests)
cargo +nightly fmt            # format
cargo clippy                  # lint
```

C reference cross-check tests are `#[ignore]` by default and fetch test
vectors from a pinned commit. Run them explicitly:

```sh
cargo test --lib -- --ignored
```

## Mutation testing

We use [cargo-mutants](https://mutants.rs/) to verify that our test
suite actually catches bugs, not just that it runs. Install with
`cargo install cargo-mutants`, then:

```sh
cargo mutants -- --lib                      # full run (slow)
cargo mutants --in-diff <(git diff main) -- --lib  # only changed code
```

Functions where mutation is meaningless (e.g., formatting, zeroize drop
glue, compile-time constants) are excluded in `.cargo/mutants.toml`.
Functions that would hang or loop forever under mutation are annotated
with `#[mutants::skip]` in the source.

CI runs incremental mutation testing on every PR (only changed code) and
a full sharded run weekly.

# Safety

The `sqisign-selkie` types are designed to make illegal states
unrepresentable. For example, any instance of a `ProjectiveXOnlyPoint`
is guaranteed to hold a point on the associated Montgomery curve, and
any instance of an `Fp` is guaranteed to hold a canonical element of
F_p.

Constant-time signing and key generation are a design goal, **not yet
achieved.** The quaternion arithmetic layer (lattice reduction, HNF,
Cornacchia, `IdealToIsogeny`) is currently variable-time on
secret-derived data. Every variable-time code path on secret data is
marked with a `TODO(ct)` comment documenting which spec algorithm line
makes the input secret-derived. Once end-to-end signing interoperability
is complete, we plan a six-phase CT hardening based on published
techniques (Kouider et al., Hanyecz et al., Basso et al., Kim et al.).

Variable-time code is only acceptable on truly public data (e.g.,
verification). We use the [`subtle` crate][subtle_doc] for conditional
moves and optimization barriers where CT has been implemented (field
arithmetic, curve operations).

Some functionality (e.g., batch inversion) requires heap allocation for
temporary buffers. All heap-allocated buffers of potentially secret data
are explicitly zeroed before release via the [`zeroize`
crate][zeroize-trait].

However, we do not attempt to zero stack data, for two reasons. First,
it's not possible to do so correctly: we don't have control over stack
allocations, so there's no way to know how much data to wipe. Second,
the correct place to start zeroing stack data is likely at the
entrypoints of the application, not at the entrypoints of library
functions.

The implementation is memory-safe and contains no `unsafe` code.

# Minimum Supported Rust Version

| Releases | MSRV   |
| :---     | :---   |
| 0.x      | 1.81.0 |

# License

Licensed under either of

- [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [MIT License](https://opensource.org/licenses/MIT)

at your option.

# About

*"In very ancient times some of the Clan Coneely, one of the early
septs of the county, were changed by "art magick" into seals; since
then no Coneely can kill a seal without afterwards having bad luck." --
Connemara Folk-Lore*

[sqisign]: https://sqisign.org/
[spec]: https://sqisign.org/spec/sqisign-20250707.pdf
[kim]: https://eprint.iacr.org/2025/1649
[subtle_doc]: https://docs.rs/subtle
[zeroize-trait]: https://docs.rs/zeroize/latest/zeroize/trait.Zeroize.html
