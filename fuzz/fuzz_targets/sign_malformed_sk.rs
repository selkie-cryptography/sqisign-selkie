#![no_main]
//! Fuzz the path where a SigningKey is parsed from arbitrary bytes
//! and then asked to sign. Threat model: a signing key file on disk
//! has been tampered with (or arrived from an untrusted source).
//! Parsing succeeds for many byte patterns that don't correspond to
//! an honest key; sign must surface tampering as `Err` or run to
//! `SigningFailed`, never panic.
//!
//! Sister target to `fuzz_sign_panic`, which exercises the honest
//! keygen→sign pipeline. This target instead injects arbitrary SK
//! bytes, which is the bug class the `prop_signing_key_from_bytes_never_panics`
//! property test only covers at parse time.
//!
//! Input layout (SIGNING_KEY_BYTES + 48 + N bytes):
//!   [ sk_bytes (353 B) | sign_seed (48 B) | msg (rest) ]
//!
//! Slow per-iter (parse + a single sign attempt is seconds to minutes,
//! since sign loops up to its retry cap on malformed inputs). Run for
//! hours when touching SK parse, sign, or anything in their call graph.

use libfuzzer_sys::fuzz_target;
use sqisign_selkie::{SIGNING_KEY_BYTES, SigningKey};

const SEED_LEN: usize = 48;
const MIN: usize = SIGNING_KEY_BYTES + SEED_LEN;

fuzz_target!(|data: &[u8]| {
    if data.len() < MIN {
        return;
    }
    let sk_bytes: &[u8; SIGNING_KEY_BYTES] = data[..SIGNING_KEY_BYTES].try_into().unwrap();
    let sign_seed: &[u8; SEED_LEN] = data[SIGNING_KEY_BYTES..MIN].try_into().unwrap();
    let msg = &data[MIN..];

    let sk = match SigningKey::from_bytes(sk_bytes) {
        Ok(sk) => sk,
        Err(_) => return,
    };
    let _ = sk.sign_derand(msg, sign_seed);
});
