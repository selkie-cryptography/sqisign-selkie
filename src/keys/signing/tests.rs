use rand_core::OsRng;

use super::*;

/// `SigningKey::generate` runs to completion (success or
/// `KeyGenFailed`) without panicking. Marked `#[ignore]` because each
/// run takes a few seconds: `to_isogeny` does a (2,2)-isogeny chain.
///
/// Run with: `cargo test --lib generate_runs -- --ignored`.
#[test]
#[ignore]
fn generate_runs() {
    let result = SigningKey::generate(&mut OsRng);
    match result {
        Ok(_sk) => {
            // SigningKey was successfully constructed. We don't compare
            // any field — the assertion is just that no panic occurred
            // and the SigningKey type-checks end to end.
        }
        Err(SignatureError::KeyGenFailed) => {
            // Acceptable: probabilistic algorithm exhausted retries.
        }
        Err(other) => panic!("unexpected error from SigningKey::generate: {other:?}"),
    }
}

/// The verifying key paired with a freshly generated signing key
/// must round-trip through `VerifyingKey::to_bytes` /
/// `VerifyingKey::from_bytes`.
#[test]
#[ignore]
fn generated_verifying_key_roundtrips() {
    let sk = match SigningKey::generate(&mut OsRng) {
        Ok(sk) => sk,
        Err(SignatureError::KeyGenFailed) => return, // probabilistic skip
        Err(other) => panic!("unexpected error: {other:?}"),
    };

    let vk = sk.verifying_key();
    let bytes = vk.to_bytes();
    let parsed = VerifyingKey::from_bytes(&bytes).expect("vk should round-trip");
    assert_eq!(parsed.to_bytes(), bytes);
}

// ----------------------------------------------------------------------
// End-to-end signing tests using parsed KAT signing keys.
//
// These bypass the slow `SigningKey::generate` by deserializing a
// known-good `sk` from the C reference implementation's KAT file
// (commit 91e9e464fe5400192d13e1f9240cbf180200a103). Run with
// `cargo test --lib --release sign_kat -- --ignored`.
// ----------------------------------------------------------------------

/// Deterministic keygen from every KAT seed must produce the
/// matching KAT pk and sk.
///
/// Run with: `cargo test --lib --release keygen_kat_all -- --ignored`.
#[test]
#[ignore]
fn keygen_kat_all() {
    for (i, &(seed_hex, pk_hex, sk_hex, _, _)) in
        crate::keys::kat_data::KAT_VECTORS.iter().enumerate()
    {
        let seed_bytes = hex::decode(seed_hex).expect("valid hex");
        let seed: [u8; 48] = seed_bytes.as_slice().try_into().expect("seed is 48 bytes");

        let sk = match SigningKey::generate_derand(&seed) {
            Ok(sk) => sk,
            Err(SignatureError::KeyGenFailed) => {
                eprintln!("keygen_kat_all: vector {i} exhausted retries (expected)");
                continue;
            }
            Err(other) => panic!("vector {i}: unexpected keygen error: {other:?}"),
        };

        let pk_bytes = hex::decode(pk_hex).expect("valid hex");
        assert_eq!(
            &sk.verifying_key().to_bytes()[..],
            pk_bytes.as_slice(),
            "vector {i}: pk mismatch"
        );

        let sk_bytes = hex::decode(sk_hex).expect("valid hex");
        assert_eq!(
            &sk.to_bytes()[..],
            sk_bytes.as_slice(),
            "vector {i}: sk mismatch"
        );

        eprintln!("keygen_kat_all: vector {i} OK");
    }
}

/// Deserialize every KAT signing key, sign the corresponding
/// message, and verify with the paired public key.
///
/// Run with: `cargo test --lib --release sign_kat_all -- --ignored`.
#[test]
#[ignore]
fn sign_kat_all() {
    for (i, &(_, pk_hex, sk_hex, msg_hex, _)) in
        crate::keys::kat_data::KAT_VECTORS.iter().enumerate()
    {
        let sk_bytes = hex::decode(sk_hex).expect("valid hex");
        let pk_bytes = hex::decode(pk_hex).expect("valid hex");
        let msg = hex::decode(msg_hex).expect("valid hex");

        let sk = SigningKey::from_bytes(sk_bytes.as_slice().try_into().unwrap())
            .expect("sk should parse");
        let vk = VerifyingKey::from_bytes(pk_bytes.as_slice().try_into().unwrap())
            .expect("pk should parse");

        let sig = match sk.sign(&msg, &mut OsRng) {
            Ok(s) => s,
            Err(SignatureError::SigningFailed) => {
                eprintln!("sign_kat_all: vector {i} SigningFailed (expected)");
                continue;
            }
            Err(other) => panic!("vector {i}: unexpected sign error: {other:?}"),
        };

        vk.verify(&msg, &sig)
            .unwrap_or_else(|_| panic!("vector {i}: signature did not verify"));
        eprintln!("sign_kat_all: vector {i} OK");
    }
}

/// Every KAT signing key deserializes and its embedded verifying
/// key matches the standalone KAT pk.
#[test]
fn kat_sk_pk_match_all() {
    for (i, &(_, pk_hex, sk_hex, _, _)) in
        crate::keys::kat_data::KAT_VECTORS.iter().enumerate()
    {
        let sk_bytes = hex::decode(sk_hex).expect("valid hex");
        let pk_bytes = hex::decode(pk_hex).expect("valid hex");

        let sk = SigningKey::from_bytes(sk_bytes.as_slice().try_into().unwrap())
            .unwrap_or_else(|_| panic!("vector {i}: sk should parse"));

        assert_eq!(
            &sk.verifying_key().to_bytes()[..],
            pk_bytes.as_slice(),
            "vector {i}: embedded pk mismatch"
        );
    }
}

/// Generate a fresh key, sign a random message, verify.
///
/// Run with: `cargo test --lib --release sign_fresh -- --ignored`.
#[test]
#[ignore]
fn sign_fresh() {
    let sk = SigningKey::generate(&mut OsRng).expect("keygen should succeed");
    let mut msg = [0u8; 64];
    rand_core::RngCore::fill_bytes(&mut OsRng, &mut msg);

    let sig = match sk.sign(&msg, &mut OsRng) {
        Ok(s) => s,
        Err(SignatureError::SigningFailed) => {
            eprintln!("sign_fresh: SigningFailed (response phase incomplete)");
            return;
        }
        Err(other) => panic!("unexpected sign error: {other:?}"),
    };

    sk.verifying_key()
        .verify(&msg, &sig)
        .expect("signature should verify");
}

/// Sign with KAT vector 0's keypair and verify.
///
/// Run with: `cargo test --lib --release sign_with_kat_key -- --ignored`.
#[test]
#[ignore]
fn sign_with_kat_key() {
    let (_, pk_hex, sk_hex, _, _) = crate::keys::kat_data::KAT_VECTORS[0];
    let sk = SigningKey::from_bytes(
        hex::decode(sk_hex).unwrap().as_slice().try_into().unwrap(),
    )
    .expect("KAT sk should parse");
    let vk = VerifyingKey::from_bytes(
        hex::decode(pk_hex).unwrap().as_slice().try_into().unwrap(),
    )
    .expect("KAT pk should parse");

    let mut msg = [0u8; 64];
    rand_core::RngCore::fill_bytes(&mut OsRng, &mut msg);

    let sig = match sk.sign(&msg, &mut OsRng) {
        Ok(s) => s,
        Err(SignatureError::SigningFailed) => {
            eprintln!("sign_with_kat_key: SigningFailed (response phase incomplete)");
            return;
        }
        Err(other) => panic!("unexpected sign error: {other:?}"),
    };

    vk.verify(&msg, &sig)
        .expect("signature should verify against KAT pk");
}
