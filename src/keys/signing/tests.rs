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
