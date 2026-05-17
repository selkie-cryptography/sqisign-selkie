#![no_main]
//! Fuzz the keygen→sign pipeline against an arbitrary keygen seed,
//! sign seed, and message. The composition must not panic: any
//! probabilistic failure should return `Err(SignatureError::KeyGenFailed)`
//! or `Err(SignatureError::SigningFailed)`.
//!
//! Input layout (96 + N bytes):
//!   [ keygen_seed (48 B) | sign_seed (48 B) | msg (rest) ]
//!
//! Slow per-iter (a single keygen+sign is ~10–60 s in fuzz-mode
//! release builds). Acceptable: panic-bugs are sparse but real (e.g.,
//! the KAT-29 retry-loop fix in the project's history was exactly
//! this class), and the input space is bounded enough that coverage
//! saturates within hours.

use libfuzzer_sys::fuzz_target;
use sqisign_selkie::SigningKey;

const SEED_LEN: usize = 48;
const MIN: usize = SEED_LEN * 2;

fuzz_target!(|data: &[u8]| {
    if data.len() < MIN {
        return;
    }
    let keygen_seed: &[u8; SEED_LEN] = data[..SEED_LEN].try_into().unwrap();
    let sign_seed: &[u8; SEED_LEN] = data[SEED_LEN..MIN].try_into().unwrap();
    let msg = &data[MIN..];

    let sk = match SigningKey::generate_derand(keygen_seed) {
        Ok(sk) => sk,
        Err(_) => return,
    };
    let _ = sk.sign_derand(msg, sign_seed);
});
