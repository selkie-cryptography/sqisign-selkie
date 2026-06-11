//! Deterministic profile benchmarks via gungraun (the renamed
//! iai-callgrind).
//!
//! Runs each bench under Valgrind/callgrind with cache + branch
//! simulation, so every run emits `Ir` (instructions), `EstimatedCycles`,
//! L1/LL cache misses, and branch mispredicts --- all deterministic
//! across CI runners (no timing noise).  Flamegraphs are NOT produced
//! here: callgrind output carries no stack traces, so gungraun's SVGs
//! rank functions by inclusive cost without real ancestry. The Profile
//! workflow's `flamegraphs` job samples genuine call stacks instead
//! (perf + cargo-flamegraph over `examples/profile_sign.rs`).
//!
//! Sharded under the `Profile` workflow (`.github/workflows/profile.yml`),
//! one matrix job per bench group / slow sqisign bench, so the wall-clock
//! cost of cache simulation is bounded by the longest single bench rather
//! than their sum.
//!
//! Requires Valgrind: `apt install valgrind` or `brew install valgrind`.
//! Run with: `cargo bench --bench instructions --features expose-internals`

mod common;

use std::hint::black_box;

use gungraun::{library_benchmark, library_benchmark_group, main};
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
use sqisign_selkie::fields::fp::arch::x86_64::mulx_adx::Fp64;
use sqisign_selkie::{
    curves::{
        Scalar,
        montgomery::{Curve, ProjectiveXOnlyPoint},
    },
    fields::{
        fp::{Fp, arch::generic::Fp51},
        fp2::Fp2,
    },
    params::BASIS_E0_P_X,
};

// --- Fp arithmetic ---
//
// Operands are built in `#[bench::case(...)]` setup -- the `from_bytes`
// Montgomery conversion runs OUTSIDE the measured region, so the
// reported Ir is the bare operation.  This matters: `from_bytes` cost
// differs per backend (it calls each backend's Mont mul), and folding
// it into the measured body earlier confounded the per-op counts (made
// Fp64 square look cheaper than Fp51's when invert showed the reverse).

#[library_benchmark]
#[bench::case(Fp::from_bytes(&[0x42; 32]), Fp::from_bytes(&[0x99; 32]))]
fn fp_mul(a: Fp, b: Fp) -> Fp {
    a * b
}

#[library_benchmark]
#[bench::case(Fp::from_bytes(&[0x42; 32]), Fp::from_bytes(&[0x99; 32]))]
fn fp_add(a: Fp, b: Fp) -> Fp {
    a + b
}

#[library_benchmark]
#[bench::case(Fp::from_bytes(&[0x42; 32]), Fp::from_bytes(&[0x99; 32]))]
fn fp_sub(a: Fp, b: Fp) -> Fp {
    a - b
}

#[library_benchmark]
#[bench::case(Fp::from_bytes(&[0x42; 32]))]
fn fp_square(a: Fp) -> Fp {
    a.square()
}

// Explicit Fp51 baseline benches.  `fp_*` above measures the dispatched
// `Fp` (Fp64 on x86_64+adx CI, Fp51 elsewhere); these always measure
// Fp51, so a single deterministic run yields the backend head-to-head
// (Ir + estimated cycles) immune to wall-clock runner noise.  Estimated
// cycles is a callgrind cache-model proxy, not a pipeline model, so it
// undercounts the asm path's ILP win -- read Ir for the op-count delta.

#[library_benchmark]
#[bench::case(Fp51::from_bytes(&[0x42; 32]), Fp51::from_bytes(&[0x99; 32]))]
fn fp51_mul(a: Fp51, b: Fp51) -> Fp51 {
    a * b
}

#[library_benchmark]
#[bench::case(Fp51::from_bytes(&[0x42; 32]))]
fn fp51_square(a: Fp51) -> Fp51 {
    a.square()
}

#[library_benchmark]
#[bench::case(Fp51::from_bytes(&[0x42; 32]))]
fn fp51_invert(a: Fp51) -> Fp51 {
    a.invert()
}

// Explicit Fp64 benches (x86_64 MULX/ADCX/ADOX backend).  On the CI
// runner the dispatched `fp_*` benches already resolve to Fp64 (adx is
// always on), so these are a labeled duplicate there; they exist so the
// dashboard names the backend explicitly and still reports Fp64 on any
// runner where the active `Fp` differs.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
#[library_benchmark]
#[bench::case(Fp64::from_bytes(&[0x42; 32]), Fp64::from_bytes(&[0x99; 32]))]
fn fp64_mul(a: Fp64, b: Fp64) -> Fp64 {
    a * b
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
#[library_benchmark]
#[bench::case(Fp64::from_bytes(&[0x42; 32]))]
fn fp64_square(a: Fp64) -> Fp64 {
    a.square()
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
#[library_benchmark]
#[bench::case(Fp64::from_bytes(&[0x42; 32]))]
fn fp64_invert(a: Fp64) -> Fp64 {
    a.invert()
}

// --- Fp2 arithmetic ---

#[library_benchmark]
#[bench::case(
    Fp2::new(Fp::from_bytes(&[0x42; 32]), Fp::from_bytes(&[0x11; 32])),
    Fp2::new(Fp::from_bytes(&[0x99; 32]), Fp::from_bytes(&[0x55; 32]))
)]
fn fp2_mul(a: Fp2, b: Fp2) -> Fp2 {
    a * b
}

// --- Curve operations ---

#[library_benchmark]
fn scalar_mul() -> ProjectiveXOnlyPoint {
    let p = black_box(ProjectiveXOnlyPoint::from_affine_x(
        BASIS_E0_P_X,
        &Curve::E0,
    ));
    let s = black_box(Scalar::from_limbs([
        0xDEAD_BEEF,
        0xCAFE_BABE,
        0x1234_5678,
        0x9ABC_DEF0,
    ]));
    p.scalar_mul(&s)
}

#[library_benchmark]
fn point_double() -> ProjectiveXOnlyPoint {
    let p = black_box(ProjectiveXOnlyPoint::from_affine_x(
        BASIS_E0_P_X,
        &Curve::E0,
    ));
    p.double()
}

// --- Parsing ---

#[library_benchmark]
fn vk_parse() {
    let bytes = black_box([0u8; 65]);
    let _ = black_box(sqisign_selkie::VerifyingKey::from_bytes(&bytes));
}

#[library_benchmark]
fn sig_parse() {
    let bytes = black_box([0u8; 148]);
    let _ = black_box(sqisign_selkie::Signature::from_bytes(&bytes));
}

#[library_benchmark]
fn sk_parse() {
    let sk_arr = common::kat0_sk_bytes();
    let _ = black_box(sqisign_selkie::SigningKey::from_bytes(&sk_arr));
}

// --- Variable-modulus Montgomery (MontReducer) ---
//
// `pow_mod` at width 18 drives `MontReducer::mul`/`square` over an
// 18-limb modulus -- the width Miller-Rabin uses during keygen's
// `random_prime_norm` (primality is 54% of keygen / 25% of sign per
// the flamegraph). On x86_64+adx+bmi2 the multiply routes through the
// `mont_mul_adx` dual-chain asm; elsewhere through the portable CIOS
// loop. Isolates the Montgomery cost so the ADX delta is measurable
// without the noise of the full keygen path. Exponent and modulus are
// fixed (a Fermat-style `base^(m-1) mod m`) so Ir is deterministic.
#[library_benchmark]
fn mont_pow_mod_w18() {
    use sqisign_selkie::quaternions::bigint::BigInt;

    // An 18-limb odd modulus (2^1088 - 9, prime-shaped; primality is
    // irrelevant -- only the Montgomery multiply chain is measured).
    let mut m_limbs = [u64::MAX; 18];
    m_limbs[0] = u64::MAX - 8;
    let m = BigInt::<18>::from_limbs(m_limbs);
    let base = BigInt::<18>::from_u64(3);
    let mut exp_limbs = m_limbs;
    exp_limbs[0] -= 1; // m - 1
    let exp = BigInt::<18>::from_limbs(exp_limbs);

    let _ = black_box(BigInt::<18>::pow_mod(
        black_box(&base),
        black_box(&exp),
        black_box(&m),
    ));
}

// --- Top-level operations ---

// KAT vector 0 verification: parse vk + sig, verify.
#[library_benchmark]
fn kat_verify() {
    let (vk, sig, msg) = common::kat0_vk_sig_msg();
    let _ = black_box(vk.verify(&msg, &sig));
}

// Deterministic keygen from KAT seed 0.
#[library_benchmark]
fn kat_keygen() {
    let seed = common::kat0_seed();
    let _ = black_box(sqisign_selkie::SigningKey::generate_derand(&seed));
}

// Deterministic sign with KAT key 0 — the response-phase flat profile
// that drives optimization targeting.
#[library_benchmark]
fn kat_sign() {
    let sk = common::kat0_signing_key();
    let msg = b"instructions benchmark message";
    let randomness = [0x42u8; 48];
    let _ = black_box(sk.sign_derand(msg, &randomness));
}

// Two arch variants: the gungraun group/main macros take plain idents
// (no `#[cfg]` on list entries), so the x86_64+adx build adds the
// explicit fp64_* benches via a separate group definition.  Both keep
// the name `field`, so `main!` and the `*::field::*` Profile shard are
// unchanged.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
library_benchmark_group!(
    name = field;
    benchmarks = fp_mul, fp_add, fp_sub, fp_square, fp2_mul, fp51_mul, fp51_square, fp51_invert,
        fp64_mul, fp64_square, fp64_invert
);

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
)))]
library_benchmark_group!(
    name = field;
    benchmarks = fp_mul, fp_add, fp_sub, fp_square, fp2_mul, fp51_mul, fp51_square, fp51_invert
);

library_benchmark_group!(
    name = bigint;
    benchmarks = mont_pow_mod_w18
);

library_benchmark_group!(
    name = curves;
    benchmarks = scalar_mul, point_double
);

library_benchmark_group!(
    name = parsing;
    benchmarks = vk_parse, sig_parse, sk_parse
);

library_benchmark_group!(
    name = sqisign;
    // Top-level sqisign keygen/sign/verify under Valgrind. keygen and sign are
    // the slow, high-value profiles.
    benchmarks = kat_verify, kat_sign, kat_keygen
);

// gungraun defaults `--cache-sim=yes --branch-sim=yes`, which is what we
// want here: every bench produces estimated_cycles, L1/LL cache misses,
// and branch mispredicts alongside Ir.  Cache simulation roughly doubles
// per-bench wall-clock; the workflow shards across benches to keep total
// wall-clock bounded.
main!(
    library_benchmark_groups = field,
    bigint,
    curves,
    parsing,
    sqisign
);
