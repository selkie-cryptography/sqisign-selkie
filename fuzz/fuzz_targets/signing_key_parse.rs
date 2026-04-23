#![no_main]
use libfuzzer_sys::fuzz_target;
use sqisign_selkie::{SIGNING_KEY_BYTES, SigningKey};

fuzz_target!(|data: &[u8]| {
    if let Ok(arr) = <&[u8; SIGNING_KEY_BYTES]>::try_from(data) {
        // Parse must not panic. Ok or Err are both fine.
        let _ = SigningKey::from_bytes(arr);
    }
});
