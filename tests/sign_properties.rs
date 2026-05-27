//! Sign-side property tests on the public API. Distinct from
//! `tests/wycheproof.rs` (input → output triples loadable by any
//! impl) because these are property assertions — e.g. "two seeds,
//! two signatures" — that don't fit the wycheproof schema cleanly.

use proptest::prelude::*;
use rand_core::OsRng;
use sqisign_selkie::{
    SIGNATURE_BYTES, Signature, SignatureError, SigningKey, VERIFYING_KEY_BYTES, VerifyingKey,
};

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
/// different signatures — nonce reuse leaks the signing key
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

/// Helper: keygen, rejecting the proptest case on the rare
/// probabilistic flake (`KeyGenFailed`) but panicking on anything
/// unexpected so new error classes can't sneak past silently.
fn try_keygen(seed: &[u8; 48]) -> Result<SigningKey, TestCaseError> {
    match SigningKey::generate_derand(seed) {
        Ok(sk) => Ok(sk),
        Err(SignatureError::KeyGenFailed) => Err(TestCaseError::reject("keygen exhausted retries")),
        Err(e) => panic!("unexpected keygen error: {e:?}"),
    }
}

/// Helper: sign, rejecting the proptest case on `SigningFailed`,
/// panicking on anything unexpected.
fn try_sign(sk: &SigningKey, msg: &[u8], seed: &[u8; 48]) -> Result<Signature, TestCaseError> {
    match sk.sign_derand(msg, seed) {
        Ok(sig) => Ok(sig),
        Err(SignatureError::SigningFailed) => Err(TestCaseError::reject("sign exhausted retries")),
        Err(e) => panic!("unexpected sign error: {e:?}"),
    }
}

// Each case runs keygen + (sometimes) sign + verify — ~10–30 s in
// release builds. `#[ignore]`d so the default `cargo test` / nextest
// PR job stays under its 30-min timeout. Run explicitly via:
//
//     cargo nextest run --release \
//         --run-ignored=ignored-only \
//         -E 'test(/sig_under_wrong_key_rejects|flipped_sig_rejects/)'
//
// Bump case count for bug hunting with `PROPTEST_CASES=128 ...`. The
// fuzz target `fuzz/fuzz_targets/verify.rs` covers the same panic-on-
// adversarial-input surface continuously on main pushes.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// A signature made under `sk_a` must not verify under `sk_b`'s
    /// `vk`, and verify must return `Err(VerificationFailed)` rather
    /// than panic on the recovered-but-invalid (2,2)-isogeny kernel.
    /// Generalizes `sig_binds_to_key` over random seed pairs.
    #[test]
    #[ignore = "slow: keygen+sign+verify per case; run on demand or nightly"]
    fn sig_under_wrong_key_rejects(
        seed_a in any::<[u8; 48]>(),
        seed_b in any::<[u8; 48]>(),
        sign_seed in any::<[u8; 48]>(),
        msg in proptest::collection::vec(any::<u8>(), 0..128),
    ) {
        prop_assume!(seed_a != seed_b);
        let sk_a = try_keygen(&seed_a)?;
        let sk_b = try_keygen(&seed_b)?;
        let sig = try_sign(&sk_a, &msg, &sign_seed)?;
        prop_assert!(sk_a.verifying_key().verify(&msg, &sig).is_ok(),
            "sig didn't verify under its own vk");
        prop_assert!(sk_b.verifying_key().verify(&msg, &sig).is_err(),
            "sig accidentally verified under wrong vk");
    }

    /// A single bit-flip on a valid signature must not verify and
    /// must not panic on any code path. Sweeps the 1184-bit signature
    /// surface a few bits at a time.
    #[test]
    #[ignore = "slow: keygen+sign+verify per case; run on demand or nightly"]
    fn flipped_sig_rejects(
        seed in any::<[u8; 48]>(),
        sign_seed in any::<[u8; 48]>(),
        msg in proptest::collection::vec(any::<u8>(), 0..128),
        flip_bit in 0usize..(SIGNATURE_BYTES * 8),
    ) {
        let sk = try_keygen(&seed)?;
        let sig = try_sign(&sk, &msg, &sign_seed)?;

        let mut bytes = sig.to_bytes();
        bytes[flip_bit / 8] ^= 1 << (flip_bit % 8);

        // Parse rejection (`NonCanonical`, `NotSupersingular`, etc.)
        // is a valid outcome — that's the input filter doing its job.
        // Only follow through to `verify` when parse succeeds.
        if let Ok(sig_mut) = Signature::from_bytes(&bytes) {
            prop_assert!(sk.verifying_key().verify(&msg, &sig_mut).is_err(),
                "bit-flipped sig at bit {} accidentally verified", flip_bit);
        }
    }
}
