//! Differential cross-test against the C reference, beyond the 100
//! fixed NIST KAT vectors.
//!
//! The baked `KAT_VECTORS` pin byte-identity at exactly 100 seeds. This
//! harness drives the C reference (`the-sqisign`) and this crate over
//! *arbitrary* seeds via a prebuilt oracle and compares byte-for-byte.
//! Both sides thread one DRBG keygen -> sign (the NIST KAT consumption
//! pattern: sign reads the stream mid-flight after keygen), and sign the
//! same fixed 32-byte message (bytes 0x00..0x1f) the oracle compiles in.
//!
//! What it asserts vs reports:
//!
//! - **Asserts** the verifying key and secret key are byte-identical to the
//!   reference (the keygen byte-identity guarantee, extended from 100 seeds to
//!   unbounded), and that every produced signature verifies.
//! - **Reports** the rate at which the *signature* is byte-identical to the
//!   reference. SQIsign signing is randomized; this crate reproduces the
//!   reference's signature bytes only when the two consume the shared DRBG
//!   along the same rejection-sampling trajectory. That holds for a subset of
//!   inputs (e.g. baked KATs 0,1,2,7,8,12 but not 3,4,5,6,...), so signature
//!   byte-identity is partial and is surfaced here rather than asserted. The
//!   baked sign KATs only ever assert verification, which is why this gap was
//!   previously invisible.
//!
//! Requires the prebuilt oracle (`tools/cref_xtest/build.sh`); point
//! `SELKIE_XTEST_ORACLE` at it. `#[ignore]`d because it needs that
//! external binary and runs full keygen+sign per seed. Run with:
//!
//! ```text
//! ORACLE=$(tools/cref_xtest/build.sh)
//! SELKIE_XTEST_ORACLE=$ORACLE SELKIE_XTEST_COUNT=64 \
//!   cargo test --release --test cref_xtest --features expose-internals \
//!   -- --ignored --nocapture
//! ```

use std::{
    io::Write,
    process::{Command, Stdio},
};

use rand_core::RngCore;
use sqisign_selkie::{SigningKey, drbg::Aes256CtrDrbg};

/// Fixed cross-test message, identical to the oracle's compiled-in
/// default (`tools/cref_xtest/oracle.c`).
const XTEST_MSG: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];

/// One reference triple emitted by the oracle for a seed, or `Fail` if
/// the reference's keygen/sign exhausted its retry budget on that seed.
enum Reference {
    Ok {
        pk: Vec<u8>,
        sk: Vec<u8>,
        sig: Vec<u8>,
    },
    Fail,
}

/// Deterministic, reproducible seeds distinct from the NIST KAT stream
/// (entropy `0x00..0x2f`): drawn from a DRBG keyed with a fixed but
/// different entropy, so none coincide with the 100 baked vectors.
fn xtest_seeds(count: usize) -> Vec<[u8; 48]> {
    let mut seed_gen = Aes256CtrDrbg::new(&[0xA5; 48]);
    (0..count)
        .map(|_| {
            let mut seed = [0u8; 48];
            seed_gen.fill_bytes(&mut seed);
            seed
        })
        .collect()
}

/// Runs the oracle over `seeds`, returning one [`Reference`] per seed.
fn run_oracle(oracle: &str, seeds: &[[u8; 48]]) -> Vec<Reference> {
    let mut child = Command::new(oracle)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn oracle (is SELKIE_XTEST_ORACLE a built tools/cref_xtest/oracle?)");

    let mut stdin = child.stdin.take().expect("oracle stdin");
    let input: String = seeds
        .iter()
        .map(|s| format!("{}\n", hex::encode(s)))
        .collect();
    std::thread::spawn(move || {
        stdin
            .write_all(input.as_bytes())
            .expect("write seeds to oracle");
    });

    let output = child.wait_with_output().expect("oracle output");
    assert!(
        output.status.success(),
        "oracle exited with {}",
        output.status
    );

    let text = String::from_utf8(output.stdout).expect("oracle stdout is utf-8");
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(
        lines.len(),
        seeds.len(),
        "oracle emitted {} lines for {} seeds",
        lines.len(),
        seeds.len()
    );

    lines
        .iter()
        .map(|line| {
            if *line == "FAIL" {
                return Reference::Fail;
            }
            let mut fields = line.split_whitespace();
            let pk = hex::decode(fields.next().expect("pk field")).expect("pk hex");
            let sk = hex::decode(fields.next().expect("sk field")).expect("sk hex");
            let sig = hex::decode(fields.next().expect("sig field")).expect("sig hex");
            Reference::Ok { pk, sk, sig }
        })
        .collect()
}

#[test]
#[ignore = "needs SELKIE_XTEST_ORACLE=<tools/cref_xtest/oracle>; see module docs"]
fn cref_xtest_differential() {
    let oracle = std::env::var("SELKIE_XTEST_ORACLE")
        .expect("set SELKIE_XTEST_ORACLE to the built tools/cref_xtest/oracle");
    let count: usize = std::env::var("SELKIE_XTEST_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let seeds = xtest_seeds(count);
    let references = run_oracle(&oracle, &seeds);

    let mut sig_byte_eq = 0usize;
    let mut sig_mismatch_seeds = Vec::new();

    for (i, (seed, reference)) in seeds.iter().zip(references).enumerate() {
        // One DRBG, keygen then sign, mirroring the oracle's
        // randombytes_init(seed) -> keypair -> sign.
        let mut drbg = Aes256CtrDrbg::new(seed);
        let sk = match SigningKey::generate_with_rng(&mut drbg) {
            Ok(sk) => sk,
            Err(e) => {
                assert!(
                    matches!(reference, Reference::Fail),
                    "seed {i}: Rust keygen failed ({e:?}) but reference succeeded"
                );
                continue;
            }
        };

        let (ref_pk, ref_sk, ref_sig) = match reference {
            Reference::Ok { pk, sk, sig } => (pk, sk, sig),
            Reference::Fail => panic!("seed {i}: reference failed but Rust keygen succeeded"),
        };

        // Keygen byte-identity: asserted (the strong, unbounded guarantee).
        let vk = sk.verifying_key();
        assert_eq!(vk.to_bytes().as_slice(), ref_pk, "seed {i}: pk mismatch");
        assert_eq!(sk.to_bytes().as_slice(), ref_sk, "seed {i}: sk mismatch");

        // Sign: must verify (asserted); byte-identity vs the reference is
        // reported, not asserted (see module docs).
        let sig = sk
            .sign_with_rng(&XTEST_MSG, &mut drbg)
            .unwrap_or_else(|e| panic!("seed {i}: Rust sign failed: {e:?}"));
        vk.verify(&XTEST_MSG, &sig)
            .unwrap_or_else(|e| panic!("seed {i}: own signature failed to verify: {e:?}"));

        if sig.to_bytes().as_slice() == ref_sig.as_slice() {
            sig_byte_eq += 1;
        } else {
            sig_mismatch_seeds.push(i);
        }
    }

    println!(
        "cross-test {count} seeds: keygen byte-identical (pk+sk) and all signatures verify; \
         signature byte-identical {sig_byte_eq}/{count}"
    );
    if !sig_mismatch_seeds.is_empty() {
        let shown: Vec<_> = sig_mismatch_seeds.iter().take(16).collect();
        println!("  signature-divergent seed indices (first 16): {shown:?}");
    }
}
