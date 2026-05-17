#![no_main]
//! Fuzz `SigningKey::generate_derand` against an arbitrary 48-byte
//! AES-CTR-DRBG seed. The function is probabilistic-with-retries; any
//! seed it can't handle should return `Err(SignatureError::KeyGenFailed)`,
//! never panic.
//!
//! Slow per-iter (~5–60 s on most seeds; multi-minute on the worst —
//! see the `sign_kat_derand_NNN` slow-tail vectors). The slow exec/s
//! is acceptable here: every panic libFuzzer finds is a real bug, and
//! the seed space is small (48 bytes), so coverage saturates fast.

use libfuzzer_sys::fuzz_target;
use sqisign_selkie::SigningKey;

/// Seed length for the AES-CTR-DRBG used by `_derand` entry points
/// (KEYLEN + BLOCKLEN = 32 + 16). Kept private to the crate, so we
/// hard-code it here.
const SEED_LEN: usize = 48;

fuzz_target!(|data: &[u8]| {
    if let Ok(seed) = <&[u8; SEED_LEN]>::try_from(data) {
        let _ = SigningKey::generate_derand(seed);
    }
});
