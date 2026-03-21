# sqisign-selkie

A compact post-quantum signature scheme from quaternions and isogenies, in Rust.

<img width="27%" align="right" src="https://user-images.githubusercontent.com/552961/197638905-f5144be3-a2f2-48c2-9ecb-26e4e34d8d8a.svg#gh-light-mode-only"/>
<img width="27%" align="right" src="https://user-images.githubusercontent.com/552961/197640007-f3f05dd1-c61c-4c16-bd04-d1813937ad47.svg#gh-dark-mode-only"/>

Implements [SQIsign](https://sqisign.org/) as specified in the
[v2.0.1 specification](https://sqisign.org/spec/sqisign-20250707.pdf),
targeting the NIST-I parameter set (p = 5·2²⁴⁸ − 1).

## Safety

The sqisign-selkie types are designed to make illegal states
unrepresentable. For example, any instance of a `MontgomeryPoint` is
guaranteed to hold a point on the associated Montgomery curve, and any
instance of an `Fp` is guaranteed to hold a canonical element of F_p.

All operations are implemented using constant-time logic (no
secret-dependent branches, no secret-dependent memory accesses), unless
specifically marked as being variable-time code. We believe that our
constant-time logic is lowered to constant-time assembly, at least on
x86_64 and aarch64 targets.

As an additional guard against possible future compiler optimizations,
the `subtle` crate places an optimization barrier before every
conditional move or assignment. More details can be found in the
[documentation for the `subtle` crate](https://docs.rs/subtle).

Some functionality (e.g., batch inversion) requires heap allocation for
temporary buffers. All heap-allocated buffers of potentially secret data
are explicitly zeroed before release.

However, we do not attempt to zero stack data, for two reasons. First,
it's not possible to do so correctly: we don't have control over stack
allocations, so there's no way to know how much data to wipe. Second,
because sqisign-selkie provides a mid-level API, the correct place to
start zeroing stack data is likely not at the entrypoints of
sqisign-selkie functions, but at the entrypoints of functions in other
crates.

The implementation is memory-safe, and contains no significant unsafe
code. The architecture-specific field arithmetic backends (aarch64,
x86_64) use unsafe internally for inline assembly. These are marked
unsafe only because they use `core::arch::asm!`, but each backend is
only compiled when targeting the appropriate architecture.

## About

*"In very ancient times some of the Clan Coneely, one of the early septs of the county, were changed
by "art magick" into seals; since then no Coneely can kill a seal without afterwards having bad
luck." - Connemara Folk-Lore*
