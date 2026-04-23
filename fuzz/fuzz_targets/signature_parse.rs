#![no_main]
use libfuzzer_sys::fuzz_target;
use sqisign_selkie::{SIGNATURE_BYTES, Signature};

fuzz_target!(|data: &[u8]| {
    if let Ok(arr) = <&[u8; SIGNATURE_BYTES]>::try_from(data) {
        let _ = Signature::from_bytes(arr);
    }
});
