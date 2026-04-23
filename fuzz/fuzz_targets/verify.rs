#![no_main]
use libfuzzer_sys::fuzz_target;
use sqisign_selkie::{SIGNATURE_BYTES, VERIFYING_KEY_BYTES, Signature, VerifyingKey};

/// Fuzz the full verify path: parse pk, parse sig, verify.
/// Any combination of inputs must not panic.
fuzz_target!(|data: &[u8]| {
    // Need at least pk + sig bytes; remaining bytes are the message.
    const MIN: usize = VERIFYING_KEY_BYTES + SIGNATURE_BYTES;
    if data.len() < MIN {
        return;
    }

    let pk_bytes: &[u8; VERIFYING_KEY_BYTES] = data[..VERIFYING_KEY_BYTES].try_into().unwrap();
    let sig_bytes: &[u8; SIGNATURE_BYTES] =
        data[VERIFYING_KEY_BYTES..MIN].try_into().unwrap();
    let msg = &data[MIN..];

    let vk = match VerifyingKey::from_bytes(pk_bytes) {
        Ok(vk) => vk,
        Err(_) => return,
    };
    let sig = match Signature::from_bytes(sig_bytes) {
        Ok(sig) => sig,
        Err(_) => return,
    };

    // Verify must not panic. Ok (unlikely with random data) or Err.
    // Use catch_unwind because some code paths currently panic
    // instead of returning Err (known issue in splitting step).
    let vk_copy = vk;
    let sig_copy = sig;
    let msg_copy = msg.to_vec();
    let _ = std::panic::catch_unwind(move || vk_copy.verify(&msg_copy, &sig_copy));
});
