//! Deterministic keygen / sign workload for local profiling on
//! Apple Silicon (samply, Instruments via cargo-instruments / xctrace)
//! and heap profiling (dhat). Self-contained: drives the `*_derand`
//! entry points off fixed seeds, so every run is byte-reproducible and
//! the only variance is the profiler's own sampling.
//!
//! This is for WALL-CLOCK / ALLOCATION profiling, which is all that is
//! available on arm64 macOS. The deterministic instruction counts the
//! perf decisions are graded on come from gungraun (valgrind/callgrind)
//! on x86 CI -- valgrind has no Apple Silicon port, so this cannot
//! reproduce those numbers. Use it to find hot frames and the relative
//! direction of a change; confirm committed wins on CI.
//!
//! Run via `scripts/profile.rs`, or directly:
//!   cargo build --profile profiling --example profile_sign
//!   ./target/profiling/examples/profile_sign verify 2000
//! Heap profile adds `--features dhat-heap` (writes dhat-heap.json).
//!
//! Args: `[keygen|sign|verify|all] [iters]`. With `--features dhat-heap`,
//! writes `dhat-heap.json` (view at https://nnethercote.github.io/dh_view/dh_view.html).

use std::hint::black_box;

use sqisign_selkie::SigningKey;

// dhat replaces the global allocator only when explicitly profiling the
// heap, so time profiles keep the real allocator and its costs.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// 48-byte AES256-CTR-DRBG seed (NIST SP 800-90A SEEDLEN). Arbitrary
/// but fixed, so the workload is deterministic; keygen rejection
/// sampling always converges off the DRBG stream.
const SEED: [u8; 48] = [0x42; 48];
/// Per-signature randomness; fixed for reproducibility.
const SIG_RANDOMNESS: [u8; 48] = [0x17; 48];
const MSG: &[u8] = b"sqisign-selkie profiling workload";

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("sign");

    // dhat reports totals, so one representative iteration suffices and
    // keeps its heavy instrumentation cheap. Time profilers want enough
    // samples: verify is ~250x cheaper per op than sign/keygen (~2.4ms
    // vs ~0.6s), so it needs far more iterations to fill a few seconds.
    let default_iters = if cfg!(feature = "dhat-heap") {
        1
    } else {
        match mode {
            "sign" => 30,
            "verify" => 2000,
            _ => 15,
        }
    };
    let iters: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_iters);

    // Inputs and outputs are black-boxed throughout so the loops are
    // not elided and the compiler cannot hoist the shared work out.
    match mode {
        "keygen" => {
            for i in 0..iters {
                // Perturb one seed byte per iteration so the run covers
                // a spread of keygen trajectories, still deterministic.
                let mut seed = SEED;
                seed[0] = i as u8;
                let sk = SigningKey::generate_derand(black_box(&seed)).expect("keygen");
                black_box(sk);
            }
        }
        "sign" => {
            // One key, then the sign hot path repeated.
            let sk = SigningKey::generate_derand(&SEED).expect("keygen");
            for _ in 0..iters {
                let sig = sk
                    .sign_derand(black_box(MSG), black_box(&SIG_RANDOMNESS))
                    .expect("sign");
                black_box(sig);
            }
        }
        "verify" => {
            // One key + signature, then the verify hot path repeated.
            // Verify is the public, already-constant-time path; this is
            // where field/curve wins show up.
            let sk = SigningKey::generate_derand(&SEED).expect("keygen");
            let sig = sk.sign_derand(MSG, &SIG_RANDOMNESS).expect("sign");
            let vk = sk.verifying_key();
            for _ in 0..iters {
                vk.verify(black_box(MSG), black_box(&sig)).expect("verify");
            }
        }
        "all" => {
            // End-to-end keygen -> sign -> verify per iteration; also the
            // mode that gives dhat the full secret-path allocation picture.
            for i in 0..iters {
                let mut seed = SEED;
                seed[0] = i as u8;
                let sk = SigningKey::generate_derand(black_box(&seed)).expect("keygen");
                let sig = sk
                    .sign_derand(black_box(MSG), black_box(&SIG_RANDOMNESS))
                    .expect("sign");
                sk.verifying_key()
                    .verify(black_box(MSG), black_box(&sig))
                    .expect("verify");
                black_box((sk, sig));
            }
        }
        other => {
            eprintln!(
                "unknown mode {other:?}; usage: profile_sign [keygen|sign|verify|all] [iters]"
            );
            std::process::exit(2);
        }
    }

    eprintln!("profile_sign: mode={mode} iters={iters} done");
}
