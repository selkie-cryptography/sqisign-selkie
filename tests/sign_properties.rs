//! Sign-side property tests on the public API. Distinct from
//! `tests/wycheproof.rs` (input → output triples loadable by any
//! impl) because these are property assertions — e.g. "two seeds,
//! two signatures" — that don't fit the wycheproof schema cleanly.

use rand_core::OsRng;
use sqisign_selkie::{SignatureError, SigningKey, VERIFYING_KEY_BYTES, VerifyingKey};

/// Fresh signing key from `OsRng`. Retries on the rare probabilistic
/// `KeyGenFailed` so a flake doesn't fail the test.
fn fresh_sk() -> SigningKey {
    for _ in 0..4 {
        match SigningKey::generate(&mut OsRng) {
            Ok(sk) => return sk,
            Err(SignatureError::KeyGenFailed) => continue,
            Err(e) => panic!("unexpected error from SigningKey::generate: {e:?}"),
        }
    }
    panic!("SigningKey::generate failed 4× in a row");
}

/// Signing the same `(sk, msg)` with different seeds must produce
/// different signatures — nonce reuse leaks the secret key
/// (ePrint 2025/897).
#[test]
fn nonce_independence() {
    let sk = fresh_sk();
    let msg = b"nonce independence test message";

    let seed_a = [0x11u8; 48];
    let seed_b = [0x22u8; 48];

    let sig_a = sk.sign_derand(msg, &seed_a).expect("sign with seed_a");
    let sig_b = sk.sign_derand(msg, &seed_b).expect("sign with seed_b");

    assert_ne!(
        sig_a.to_bytes(),
        sig_b.to_bytes(),
        "two distinct DRBG seeds produced identical signatures — nonce reuse?"
    );

    let vk = sk.verifying_key();
    vk.verify(msg, &sig_a).expect("sig_a verifies under vk");
    vk.verify(msg, &sig_b).expect("sig_b verifies under vk");
}

/// Empty message round-trips through sign/verify.
#[test]
fn sign_empty_message_roundtrips() {
    let sk = fresh_sk();
    let seed = [0x33u8; 48];
    let sig = sk.sign_derand(b"", &seed).expect("sign empty msg");
    sk.verifying_key()
        .verify(b"", &sig)
        .expect("empty-msg signature verifies");
}

/// 64 KiB message round-trips — exercises SHAKE256 across many
/// absorb blocks (136 B each).
#[test]
fn sign_large_message_roundtrips() {
    let sk = fresh_sk();
    let seed = [0x44u8; 48];
    let msg = vec![0xA5u8; 65536];
    let sig = sk.sign_derand(&msg, &seed).expect("sign large msg");
    sk.verifying_key()
        .verify(&msg, &sig)
        .expect("large-msg signature verifies");
}

/// `sign_derand` is deterministic: same `(sk, msg, seed)` → same
/// signature bytes. Catches hidden RNG draws / non-deterministic
/// rejection-sampling state.
#[test]
fn sign_derand_is_deterministic() {
    let sk = fresh_sk();
    let msg = b"determinism test";
    let seed = [0x5Au8; 48];
    let sig_a = sk.sign_derand(msg, &seed).expect("sign 1");
    let sig_b = sk.sign_derand(msg, &seed).expect("sign 2");
    assert_eq!(
        sig_a.to_bytes(),
        sig_b.to_bytes(),
        "sign_derand non-deterministic"
    );
}

/// Sig binds to the message: same `(sk, seed)`, different msgs →
/// different signatures. Catches a missing-message-in-challenge-hash
/// regression.
#[test]
fn sig_binds_to_message() {
    let sk = fresh_sk();
    let seed = [0x6Bu8; 48];
    let sig_a = sk.sign_derand(b"message A", &seed).expect("sign a");
    let sig_b = sk.sign_derand(b"message B", &seed).expect("sign b");
    assert_ne!(
        sig_a.to_bytes(),
        sig_b.to_bytes(),
        "sig didn't bind to message"
    );
}

/// Sig binds to the key: `sk_a`'s sig must not verify under `vk_b`.
#[test]
fn sig_binds_to_key() {
    let sk_a = fresh_sk();
    let sk_b = fresh_sk();
    let msg = b"key-binding test";
    let seed = [0x7Cu8; 48];

    let sig_a = sk_a.sign_derand(msg, &seed).expect("sign with sk_a");
    sk_a.verifying_key()
        .verify(msg, &sig_a)
        .expect("sig_a verifies under vk_a");

    assert!(
        sk_b.verifying_key().verify(msg, &sig_a).is_err(),
        "sig_a accidentally verified under vk_b"
    );
}

/// Different seeds produce different keys. Asserts DRBG diversity at
/// the keygen entry point.
#[test]
fn distinct_seeds_distinct_keys() {
    let seed_a = [0x8Du8; 48];
    let seed_b = [0x9Eu8; 48];
    let sk_a = SigningKey::generate_derand(&seed_a).expect("keygen a");
    let sk_b = SigningKey::generate_derand(&seed_b).expect("keygen b");
    assert_ne!(
        sk_a.to_bytes(),
        sk_b.to_bytes(),
        "two seeds produced identical keys"
    );
}

/// `as_bytes` / `AsRef<[u8]>` / `TryFrom<&[u8]>` agree with
/// `to_bytes` / `from_bytes`; wrong-length input is rejected with
/// `InvalidLength`.
#[test]
fn verifying_key_byte_surface_roundtrips() {
    let vk_bytes: [u8; VERIFYING_KEY_BYTES] = fresh_sk().verifying_key().to_bytes();

    let vk = VerifyingKey::from_bytes(&vk_bytes).expect("vk parses");

    assert_eq!(
        vk.as_bytes(),
        &vk_bytes,
        "as_bytes matches from_bytes input"
    );
    assert_eq!(
        <VerifyingKey as AsRef<[u8]>>::as_ref(&vk),
        &vk_bytes[..],
        "AsRef<[u8]> matches"
    );

    let vk2 = VerifyingKey::try_from(&vk_bytes[..]).expect("try_from slice");
    assert_eq!(vk2.to_bytes(), vk_bytes);

    let short = &vk_bytes[..VERIFYING_KEY_BYTES - 1];
    match VerifyingKey::try_from(short) {
        Err(SignatureError::InvalidLength {
            expected, actual, ..
        }) => {
            assert_eq!(expected, VERIFYING_KEY_BYTES);
            assert_eq!(actual, VERIFYING_KEY_BYTES - 1);
        }
        other => panic!("expected InvalidLength, got {other:?}"),
    }
}
