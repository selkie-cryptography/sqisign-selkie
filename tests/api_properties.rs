//! Property-based tests for the public API: serialization roundtrips
//! and panic-resistance.
//!
//! Run with: `cargo test --test api_properties --features expose-internals`

use proptest::prelude::*;
use sqisign_selkie::{
    SIGNATURE_BYTES, SIGNING_KEY_BYTES, Signature, SigningKey, VERIFYING_KEY_BYTES, VerifyingKey,
    keys::kat_data::KAT_VECTORS,
};

fn kat0_vk() -> VerifyingKey {
    let pk_bytes = hex::decode(KAT_VECTORS[0].1).expect("KAT 0 pk hex");
    let pk_arr: &[u8; VERIFYING_KEY_BYTES] = pk_bytes.as_slice().try_into().expect("pk length");
    VerifyingKey::from_bytes(pk_arr).expect("KAT 0 pk parses")
}

fn kat0_sig_bytes() -> [u8; SIGNATURE_BYTES] {
    let sm = hex::decode(KAT_VECTORS[0].4).expect("KAT 0 sm hex");
    sm[..SIGNATURE_BYTES].try_into().expect("sm has sig prefix")
}

fn arb_msg() -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(any::<u8>(), 0..=4096)
}

proptest! {
    /// Random 65 bytes → `VerifyingKey::from_bytes` → `to_bytes`
    /// roundtrip (if parsing succeeds).
    #[test]
    fn prop_verifying_key_roundtrip(bytes in any::<[u8; VERIFYING_KEY_BYTES]>()) {
        if let Ok(vk) = VerifyingKey::from_bytes(&bytes) {
            let rt = vk.to_bytes();
            let vk2 = VerifyingKey::from_bytes(&rt).expect("roundtrip must parse");
            prop_assert_eq!(vk, vk2);
        }
    }

    /// Random 148 bytes → `Signature::from_bytes` → `to_bytes`
    /// roundtrip (if parsing succeeds).
    #[test]
    fn prop_signature_roundtrip(bytes in any::<[u8; SIGNATURE_BYTES]>()) {
        if let Ok(sig) = Signature::from_bytes(&bytes) {
            let rt = sig.to_bytes();
            let sig2 = Signature::from_bytes(&rt).expect("roundtrip must parse");
            prop_assert_eq!(rt, sig2.to_bytes());
        }
    }

    /// `SigningKey::from_bytes` must return `Ok` or `Err` on any
    /// input, never panic. A signing key loaded from disk goes
    /// through this path; a corrupted file must surface as `Err`,
    /// not crash.
    #[test]
    fn prop_signing_key_from_bytes_never_panics(
        bytes in any::<[u8; SIGNING_KEY_BYTES]>(),
    ) {
        let _ = SigningKey::from_bytes(&bytes);
    }
}

proptest! {
    // Each case runs the verify chain; 256 cases ≈ a few seconds in
    // release. Keeps the test in the default suite without dragging.
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    /// `VerifyingKey::verify` must never panic — the API contract is
    /// `Result<(), SignatureError>`. Catches asserts hidden behind
    /// parser-accepted but verify-rejected signature shapes
    /// (regression for the `assert!(e >= 2)` chain panic).
    ///
    /// XOR-masking a real KAT signature gives a much higher parse
    /// rate than arbitrary 148 bytes, so each generated case is
    /// likely to reach the verify chain rather than bouncing at the
    /// parser.
    #[test]
    fn prop_verify_never_panics(
        mask in any::<[u8; SIGNATURE_BYTES]>(),
        msg in arb_msg(),
    ) {
        let vk = kat0_vk();
        let base = kat0_sig_bytes();
        let mut sig_bytes = [0u8; SIGNATURE_BYTES];
        for (i, b) in sig_bytes.iter_mut().enumerate() {
            *b = base[i] ^ mask[i];
        }
        if let Ok(sig) = Signature::from_bytes(&sig_bytes) {
            let _ = vk.verify(&msg, &sig);
        }
    }
}
