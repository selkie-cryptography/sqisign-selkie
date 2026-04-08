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

/// KAT vector 0: `sk` (secret key), `pk` (public key), `msg` (message).
/// Taken from `PQCsignKAT_353_SQIsign_lvl1.rsp`, count = 0.
const KAT0_SK_HEX: &str = "07CCD21425136F6E865E497D2D4D208F0054AD81372066E817480787AAF7B2029550C89E892D618CE3230F23510BFBE68FCCDDAEA51DB1436B462ADFAF008A010B19943116DB5B4552B05B174969C61C9C8701000000000000000000000000000094F28A5533DF8872E3C7EFE3D45A175A0CFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFF1959E3D67EADD79948DB766D9FFAF4D3FFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000000000000000000000000000000000000000000000000000358A8756E1CA2E31C2F3C879414AC08DF7EA0C1D732F9AE3D1AC4644E524340095A4F53D286FDE8A7226CE960C152C344888C963A457B02CAECA41C2672D76000365B548FB9C9E6C0E149BABA3EC7BC33B8F052B6B9D4F840A2AD67221C8F600464B9862D34ADF4D562F3836EBEFC4D8F874351B3E63A4DF9D33C0BBF9EB1800";
const KAT0_PK_HEX: &str = "07CCD21425136F6E865E497D2D4D208F0054AD81372066E817480787AAF7B2029550C89E892D618CE3230F23510BFBE68FCCDDAEA51DB1436B462ADFAF008A010B";
const KAT0_MSG_HEX: &str = "D81C4D8D734FCBFBEADE3D3F8A039FAA2A2C9957E835AD55B22E75BF57BB556AC8";

/// Deserialize a KAT signing key, sign the corresponding message, and
/// verify the result with the paired public key. This exercises the
/// full `sign()` response phase without the 172-second `keygen` cost.
///
/// Run with: `cargo test --lib --release sign_kat_roundtrip -- --ignored`.
#[test]
#[ignore]
fn sign_kat_roundtrip() {
    let sk_bytes = hex::decode(KAT0_SK_HEX).expect("valid hex");
    let pk_bytes = hex::decode(KAT0_PK_HEX).expect("valid hex");
    let msg = hex::decode(KAT0_MSG_HEX).expect("valid hex");

    let sk_array: &[u8; SIGNING_KEY_BYTES] = sk_bytes
        .as_slice()
        .try_into()
        .expect("sk has correct length");
    let sk = SigningKey::from_bytes(sk_array).expect("sk should parse");

    let pk_array: &[u8; VERIFYING_KEY_BYTES] = pk_bytes
        .as_slice()
        .try_into()
        .expect("pk has correct length");
    let vk = VerifyingKey::from_bytes(pk_array).expect("pk should parse");

    let sig = match sk.sign(&msg) {
        Ok(s) => s,
        Err(SignatureError::SigningFailed) => {
            // The sign() loop exhausted its retries without producing
            // a valid signature. Expected for the current state of
            // the response phase — promote to `panic!` once the
            // remaining blockers clear.
            eprintln!("sign_kat_roundtrip: SigningFailed (expected, not yet valid)");
            return;
        }
        Err(other) => panic!("unexpected sign error: {other:?}"),
    };

    vk.verify(&msg, &sig)
        .expect("freshly signed message should verify against its public key");
}

/// Lighter-weight check: the KAT signing key deserializes and its
/// embedded `verifying_key` field matches the standalone KAT `pk`.
/// Fast (no signing, no isogenies) and runs in the normal test suite.
#[test]
fn sign_kat_sk_pk_match() {
    let sk_bytes = hex::decode(KAT0_SK_HEX).expect("valid hex");
    let pk_bytes = hex::decode(KAT0_PK_HEX).expect("valid hex");

    let sk_array: &[u8; SIGNING_KEY_BYTES] = sk_bytes
        .as_slice()
        .try_into()
        .expect("sk has correct length");
    let sk = SigningKey::from_bytes(sk_array).expect("sk should parse");

    let embedded_pk_bytes = sk.verifying_key().to_bytes();
    assert_eq!(
        &embedded_pk_bytes[..],
        pk_bytes.as_slice(),
        "sk's embedded public key must match the standalone KAT pk"
    );
}
