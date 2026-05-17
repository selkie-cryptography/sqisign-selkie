//! Property-based tests for the public API serialization roundtrips.
//!
//! Run with: `cargo test --test api_properties`

use proptest::prelude::*;
use sqisign_selkie::{SIGNATURE_BYTES, Signature, VERIFYING_KEY_BYTES, VerifyingKey};

proptest! {
    /// Random 65 bytes → VerifyingKey::from_bytes → to_bytes roundtrip
    /// (if parsing succeeds).
    #[test]
    fn prop_verifying_key_roundtrip(bytes in any::<[u8; VERIFYING_KEY_BYTES]>()) {
        if let Ok(vk) = VerifyingKey::from_bytes(&bytes) {
            let rt = vk.to_bytes();
            let vk2 = VerifyingKey::from_bytes(&rt).expect("roundtrip must parse");
            prop_assert_eq!(vk, vk2);
        }
    }

    /// Random 148 bytes → Signature::from_bytes → to_bytes roundtrip
    /// (if parsing succeeds).
    #[test]
    fn prop_signature_roundtrip(bytes in any::<[u8; SIGNATURE_BYTES]>()) {
        if let Ok(sig) = Signature::from_bytes(&bytes) {
            let rt = sig.to_bytes();
            let sig2 = Signature::from_bytes(&rt).expect("roundtrip must parse");
            prop_assert_eq!(rt, sig2.to_bytes());
        }
    }
}
