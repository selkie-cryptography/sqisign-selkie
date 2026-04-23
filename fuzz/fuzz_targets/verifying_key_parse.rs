#![no_main]
use libfuzzer_sys::fuzz_target;
use sqisign_selkie::{VERIFYING_KEY_BYTES, VerifyingKey};

fuzz_target!(|data: &[u8]| {
    if let Ok(arr) = <&[u8; VERIFYING_KEY_BYTES]>::try_from(data) {
        let _ = VerifyingKey::from_bytes(arr);
    }
});
