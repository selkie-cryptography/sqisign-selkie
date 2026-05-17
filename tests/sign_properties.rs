//! Sign-side security properties not captured by the per-vector KAT
//! suite or by `tests/wycheproof.rs`.
//!
//! Each test uses only the public API
//! ([`SigningKey::sign_derand`] / [`VerifyingKey::verify`]) so the
//! cases double as documentation of the contract a downstream
//! cryptographic-protocol consumer can rely on.
//!
//! Why not `tests/wycheproof.rs`? Wycheproof entries are
//! `(input, expected_output)` triples loadable by any conformant
//! implementation; these are *property* assertions (e.g. signing the
//! same message twice with different randomness must produce distinct
//! signatures) that are clearer expressed in Rust than in the
//! C2SP/wycheproof JSON schema.

use sqisign_selkie::{
    SIGNING_KEY_BYTES, SignatureError, SigningKey, VERIFYING_KEY_BYTES, VerifyingKey,
};

/// KAT[0]'s signing key, copied from `tests/vectors/sqisign_keygen.json`
/// (which mirrors NIST's `PQCsignKAT_…rsp` line 1). Lets the
/// tests below skip the slow keygen step.
const KAT0_SK_HEX: &str = "07CCD21425136F6E865E497D2D4D208F0054AD81372066E817480787AAF7B2029550C89E892D618CE3230F23510BFBE68FCCDDAEA51DB1436B462ADFAF008A010B19943116DB5B4552B05B174969C61C9C8701000000000000000000000000000094F28A5533DF8872E3C7EFE3D45A175A0CFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFF1959E3D67EADD79948DB766D9FFAF4D3FFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000000000000000000000000000000000000000000000000000358A8756E1CA2E31C2F3C879414AC08DF7EA0C1D732F9AE3D1AC4644E524340095A4F53D286FDE8A7226CE960C152C344888C963A457B02CAECA41C2672D76000365B548FB9C9E6C0E149BABA3EC7BC33B8F052B6B9D4F840A2AD67221C8F600464B9862D34ADF4D562F3836EBEFC4D8F874351B3E63A4DF9D33C0BBF9EB1800";

fn kat0_sk() -> SigningKey {
    let bytes = hex::decode(KAT0_SK_HEX).expect("KAT[0] sk hex");
    let arr: &[u8; SIGNING_KEY_BYTES] = bytes.as_slice().try_into().expect("KAT[0] sk byte length");
    SigningKey::from_bytes(arr).expect("KAT[0] sk parses")
}

/// Signing the same `(sk, msg)` pair with two distinct random seeds
/// must produce two distinct signatures.
///
/// SQIsign signs by sampling a commitment isogeny seeded by the RNG;
/// the commitment curve `E_com` is then encoded in `sig.curve_aux`
/// (after a series of intermediate steps). If our implementation
/// silently reused the same nonce across calls — for any reason,
/// from a deterministic-rejection-sampling regression to a bad
/// RNG-threading bug — the same `(sk, msg)` would land on the same
/// `E_com` and emit identical signatures. That collision in turn
/// leaks the secret key (cf. [ePrint 2025/897]).
///
/// [ePrint 2025/897]: https://eprint.iacr.org/2025/897
#[test]
fn nonce_independence() {
    let sk = kat0_sk();
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

    // Both signatures must still verify under the same vk.
    let vk = sk.verifying_key();
    vk.verify(msg, &sig_a).expect("sig_a verifies under vk");
    vk.verify(msg, &sig_b).expect("sig_b verifies under vk");
}

/// `sign_derand` accepts the empty message and the resulting
/// signature verifies. SHAKE256 has no minimum-length requirement,
/// but a regression in the challenge-hash boundary handling could
/// in principle reject empty input.
#[test]
fn sign_empty_message_roundtrips() {
    let sk = kat0_sk();
    let seed = [0x33u8; 48];
    let sig = sk.sign_derand(b"", &seed).expect("sign empty msg");
    sk.verifying_key()
        .verify(b"", &sig)
        .expect("empty-msg signature verifies");
}

/// `sign_derand` accepts a 64 KiB message and the resulting signature
/// verifies under the same content. Exercises the SHAKE256 absorb
/// path across many blocks (block size 136 B; 64 KiB ≈ 481 blocks).
///
/// Distinct from the empty-message test because the absorb-then-squeeze
/// implementation could mishandle a long absorb without affecting the
/// short-message case.
#[test]
fn sign_large_message_roundtrips() {
    let sk = kat0_sk();
    let seed = [0x44u8; 48];
    let msg = vec![0xA5u8; 65536];
    let sig = sk.sign_derand(&msg, &seed).expect("sign large msg");
    sk.verifying_key()
        .verify(&msg, &sig)
        .expect("large-msg signature verifies");
}

/// `VerifyingKey`'s public accessors / standard-trait surface
/// (`as_bytes`, `AsRef<[u8]>`, `TryFrom<&[u8]>`) all round-trip
/// against the same byte representation as `to_bytes`/`from_bytes`.
/// These are the bytes-in / bytes-out boundary; a consumer wiring
/// SQIsign into a TLS handshake or storing it in a keystore relies
/// on each form agreeing.
#[test]
fn verifying_key_byte_surface_roundtrips() {
    let vk_bytes: [u8; VERIFYING_KEY_BYTES] = kat0_sk().verifying_key().to_bytes();

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

    // TryFrom<&[u8]> accepts exactly VERIFYING_KEY_BYTES.
    let vk2 = VerifyingKey::try_from(&vk_bytes[..]).expect("try_from slice");
    assert_eq!(vk2.to_bytes(), vk_bytes);

    // Wrong length is rejected with InvalidLength carrying the
    // expected/actual sizes.
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
