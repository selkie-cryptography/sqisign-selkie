//! The end-to-end usage example reproduced from the README.
//!
//! `cargo run --example readme --release` exercises the documented
//! happy path: generate a signing key, sign a message, serialize the
//! verifying key, deserialize it, verify the signature.

use sqisign_selkie::{SigningKey, VerifyingKey};

fn main() {
    let sk = SigningKey::generate(&mut rand_core::OsRng).expect("key generation failed");

    let message = "Maighdean mhara mo mháithrín ard".as_bytes();
    let signature = sk
        .sign(message, &mut rand_core::OsRng)
        .expect("signing failed");

    let vk_bytes = sk.verifying_key().to_bytes();

    let vk = VerifyingKey::from_bytes(&vk_bytes).expect("invalid verifying key");
    vk.verify(message, &signature).expect("invalid signature");

    println!("ok");
}
