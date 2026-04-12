# sqisign-selkie

A compact post-quantum signature scheme from quaternions and isogenies, in Rust.

<img width="27%" align="right" src="https://user-images.githubusercontent.com/552961/197638905-f5144be3-a2f2-48c2-9ecb-26e4e34d8d8a.svg#gh-light-mode-only"/>
<img width="27%" align="right" src="https://user-images.githubusercontent.com/552961/197640007-f3f05dd1-c61c-4c16-bd04-d1813937ad47.svg#gh-dark-mode-only"/>

Implements [SQIsign][sqisign] as specified in the [v2.0.1
specification][spec] (2025-07-07), targeting the NIST-I parameter set
(p = 5 · 2²⁴⁸ − 1).

> **Status: work in progress.** Verification passes known-answer tests
> from the C reference implementation. Signing compiles and is
> structurally complete but has remaining arithmetic blockers. Key
> generation is not yet implemented. **Do not use in production.**

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

# Safety

The `sqisign-selkie` types are designed to make illegal states
unrepresentable. For example, any instance of a `ProjectiveXOnlyPoint`
is guaranteed to hold a point on the associated Montgomery curve, and
any instance of an `Fp` is guaranteed to hold a canonical element of
F_p.

All operations on secret or secret-derived data are intended to be
constant-time (no secret-dependent branches, no secret-dependent memory
accesses). Variable-time code is only acceptable on truly public data
(e.g., verification). Temporary variable-time code paths on secret data
are marked with `TODO(ct)` comments documenting which spec algorithm
line makes the input secret-derived. We use the [`subtle`
crate][subtle_doc] for conditional moves and optimization barriers.

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
