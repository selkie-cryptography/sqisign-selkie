//! Top-level public-API benches: keygen, sign, verify, and the
//! parse / serialize round-trips that bracket them. Sub-step benches
//! (TorsionBasis, ChangeOfBasisMatrix, sample_from_ball, etc.) live
//! in `curves.rs`, `quaternions.rs`, and `deuring.rs` because they
//! need `expose-internals`.

mod common;

use sqisign_selkie::{Signature, SigningKey, VerifyingKey};

fn main() {
    divan::main();
}

// --- Verify ---

/// End-to-end `verify` on a pre-parsed KAT[0] vk + signature.
#[divan::bench]
fn verify(bencher: divan::Bencher) {
    let (vk, sig, msg) = common::kat0_vk_sig_msg();
    bencher.bench(|| vk.verify(divan::black_box(&msg), divan::black_box(&sig)));
}

/// Just the deserialize step that precedes `verify` in any wire-level
/// caller. Times two `from_bytes` calls back to back.
#[divan::bench]
fn verify_parse(bencher: divan::Bencher) {
    let vk_bytes = common::kat0_vk_bytes();
    let sig_bytes = common::kat0_sig_bytes();
    bencher.bench(|| {
        let _vk = VerifyingKey::from_bytes(divan::black_box(&vk_bytes));
        let _sig = Signature::from_bytes(divan::black_box(&sig_bytes));
    });
}

/// `VerifyingKey::to_bytes` followed by `from_bytes` — the wire
/// round-trip cost.
#[divan::bench]
fn vk_roundtrip(bencher: divan::Bencher) {
    let (vk, ..) = common::kat0_vk_sig_msg();
    bencher.bench(|| {
        let bytes = divan::black_box(&vk).to_bytes();
        VerifyingKey::from_bytes(divan::black_box(&bytes))
    });
}

/// `Signature::to_bytes` followed by `from_bytes`.
#[divan::bench]
fn sig_roundtrip(bencher: divan::Bencher) {
    let (_, sig, _) = common::kat0_vk_sig_msg();
    bencher.bench(|| {
        let bytes = divan::black_box(&sig).to_bytes();
        Signature::from_bytes(divan::black_box(&bytes))
    });
}

// --- Keygen ---

/// `SigningKey::generate` with `OsRng` — the path most production
/// callers take. Timing includes OS RNG cost.
#[divan::bench(sample_count = 10)]
fn keygen(bencher: divan::Bencher) {
    bencher.bench(|| {
        let mut rng = rand_core::OsRng;
        SigningKey::generate(&mut rng)
    });
}

/// `SigningKey::generate_derand` from the KAT[0] seed — reproducible
/// end-to-end keygen timing. Same DRBG path as `generate`, minus the
/// OS RNG cost.
#[divan::bench(sample_count = 10)]
fn keygen_derand(bencher: divan::Bencher) {
    let seed = common::kat0_seed();
    bencher.bench(|| SigningKey::generate_derand(divan::black_box(&seed)));
}

/// `SigningKey::to_bytes` round-trip on a parsed KAT[0] sk. Wire
/// serialize + deserialize.
#[divan::bench]
fn sk_roundtrip(bencher: divan::Bencher) {
    let sk = common::kat0_signing_key();
    bencher.bench(|| {
        let bytes = divan::black_box(&sk).to_bytes();
        SigningKey::from_bytes(divan::black_box(&bytes))
    });
}

// --- Sign ---

/// `SigningKey::sign` with `OsRng` on a pre-parsed KAT[0] sk.
#[divan::bench(sample_count = 10)]
fn sign(bencher: divan::Bencher) {
    let sk = common::kat0_signing_key();
    let msg = b"benchmark message";
    bencher.bench(|| {
        let mut rng = rand_core::OsRng;
        sk.sign(divan::black_box(msg.as_slice()), &mut rng)
    });
}

/// `SigningKey::sign_derand` with a fixed 48-byte randomness — the
/// reproducible sign path used by KATs.
#[divan::bench(sample_count = 10)]
fn sign_derand(bencher: divan::Bencher) {
    let sk = common::kat0_signing_key();
    let msg = b"benchmark message";
    let randomness = [0x42u8; 48];
    bencher.bench(|| {
        sk.sign_derand(
            divan::black_box(msg.as_slice()),
            divan::black_box(&randomness),
        )
    });
}
