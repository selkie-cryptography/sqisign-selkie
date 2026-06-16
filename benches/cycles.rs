//! rdtsc cycle benchmark: keygen / sign / verify in CPU cycles, so the
//! numbers line up unit-for-unit with the C reference's `cpucycles()`
//! (reported as megacycles).
//!
//! x86_64-only. Cycle counts come from `rdtsc` via Criterion's
//! [`criterion_cycles_per_byte::CyclesPerByte`] measurement. `rdtsc` is
//! the same unprivileged TSC read the C reference uses (and the only
//! cycle source available under Firecracker, which exposes no PMU), so
//! the two are comparable on the same machine. There is no unprivileged
//! cycle-counter read on aarch64 (the M-series virtual counter is a
//! fixed ~24 MHz timer, not cycles), so on every non-x86_64 target this
//! bench compiles to a skip stub.
//!
//! Deterministic off fixed seeds (the same the local profiler uses), so
//! the cycle counts are reproducible run to run.

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::hint::black_box;

    use criterion::{Criterion, criterion_group};
    use criterion_cycles_per_byte::CyclesPerByte;
    use sqisign_selkie::SigningKey;

    /// 48-byte AES-256-CTR-DRBG seed. Arbitrary but fixed, so keygen's
    /// rejection sampling always converges off the same DRBG stream.
    const SEED: [u8; 48] = [0x42; 48];
    /// Per-signature randomness; fixed for reproducibility.
    const SIG_RANDOMNESS: [u8; 48] = [0x17; 48];
    /// Message signed and verified in the hot loops.
    const MSG: &[u8] = b"sqisign-selkie cycles benchmark";

    /// Benchmarks keygen, sign, and verify under the rdtsc measurement,
    /// all in the `sqisign` group so the report ids match the C side.
    fn ops(c: &mut Criterion<CyclesPerByte>) {
        let mut group = c.benchmark_group("sqisign");
        // keygen and sign are slow and rejection-sampled; a handful of
        // samples is enough for a stable median and bounds wall-time.
        group.sample_size(20);

        group.bench_function("keygen", |b| {
            b.iter(|| SigningKey::generate_derand(black_box(&SEED)).expect("keygen"))
        });

        let sk = SigningKey::generate_derand(&SEED).expect("keygen");
        group.bench_function("sign", |b| {
            b.iter(|| {
                sk.sign_derand(black_box(MSG), black_box(&SIG_RANDOMNESS))
                    .expect("sign")
            })
        });

        let sig = sk.sign_derand(MSG, &SIG_RANDOMNESS).expect("sign");
        let vk = sk.verifying_key();
        group.bench_function("verify", |b| {
            b.iter(|| vk.verify(black_box(MSG), black_box(&sig)).expect("verify"))
        });

        group.finish();
    }

    criterion_group!(
        name = benches;
        config = Criterion::default().with_measurement(CyclesPerByte);
        targets = ops
    );

    /// Runs the Criterion group (it configures itself from the bench CLI
    /// args that `cargo bench` passes through).
    pub fn run() {
        benches();
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    imp::run();
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("cycles bench: x86_64-only (rdtsc); skipped on this target");
}
