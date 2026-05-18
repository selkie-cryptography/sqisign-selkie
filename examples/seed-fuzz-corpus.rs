//! Seed `fuzz/corpus/<target>/` directories with KAT-derived inputs.
//!
//! libFuzzer's coverage-guided exploration is dramatically more
//! effective when the starting corpus already contains inputs that
//! reach the deep code paths. With random bytes alone, ~99.999% of
//! iterations bounce at the relevant `from_bytes` call. Seeding with
//! real KAT bytes gives libFuzzer N parseable starting points; all of
//! its byte-level mutations then start one step away from the target's
//! deep code.
//!
//! Run before fuzzing:
//!
//! ```sh
//! cargo run --release --features expose-internals --example seed-fuzz-corpus
//! cd fuzz && cargo +nightly fuzz run <target>
//! ```
//!
//! The `expose-internals` feature flag is required because the seeder
//! reads `keys::kat_data::KAT_VECTORS`; the gating is enforced via
//! `required-features` on the `[[example]]` entry in the top-level
//! `Cargo.toml`.
//!
//! Output files are content-keyed by KAT index (`kat-000` … `kat-099`).
//! libFuzzer deduplicates on content hash, so re-running the seeder is
//! idempotent.

use std::{fs, path::Path};

use sqisign_selkie::{
    SIGNATURE_BYTES, SIGNING_KEY_BYTES, VERIFYING_KEY_BYTES, keys::kat_data::KAT_VECTORS,
};

/// `_derand` seed length in bytes (AES256-CTR-DRBG KEYLEN + BLOCKLEN).
const SEED_LEN: usize = 48;

/// Length of the XOR delta region of `fuzz_verify_typed`'s input — the
/// signature bytes after `curve_aux`, which the harness mutates.
const VERIFY_TYPED_MUT_LEN: usize = SIGNATURE_BYTES - 64;

fn write_seed(dir: &Path, i: usize, bytes: &[u8]) {
    let path = dir.join(format!("kat-{i:03}"));
    fs::write(&path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn ensure_dir(name: &str) -> std::path::PathBuf {
    let dir = Path::new("fuzz/corpus").join(name);
    fs::create_dir_all(&dir).expect("create fuzz corpus dir");
    dir
}

fn main() {
    let dir_sig_parse = ensure_dir("fuzz_signature_parse");
    let dir_vk_parse = ensure_dir("fuzz_verifying_key_parse");
    let dir_sk_parse = ensure_dir("fuzz_signing_key_parse");
    let dir_verify = ensure_dir("fuzz_verify");
    let dir_verify_typed = ensure_dir("fuzz_verify_typed");
    let dir_keygen_panic = ensure_dir("fuzz_keygen_panic");
    let dir_sign_panic = ensure_dir("fuzz_sign_panic");

    // KAT 0's signature is the XOR baseline for fuzz_verify_typed.
    let kat0_sig = {
        let sm0 = hex::decode(KAT_VECTORS[0].4).expect("KAT 0 sm hex");
        let mut s = [0u8; SIGNATURE_BYTES];
        s.copy_from_slice(&sm0[..SIGNATURE_BYTES]);
        s
    };

    for (i, &(seed_hex, pk_hex, sk_hex, msg_hex, sm_hex)) in KAT_VECTORS.iter().enumerate() {
        let seed = hex::decode(seed_hex).expect("seed hex decode");
        let pk = hex::decode(pk_hex).expect("pk hex decode");
        let sk = hex::decode(sk_hex).expect("sk hex decode");
        let msg = hex::decode(msg_hex).expect("msg hex decode");
        let sm = hex::decode(sm_hex).expect("sm hex decode");

        assert_eq!(seed.len(), SEED_LEN, "KAT[{i}] seed length");
        assert_eq!(pk.len(), VERIFYING_KEY_BYTES, "KAT[{i}] pk length");
        assert_eq!(sk.len(), SIGNING_KEY_BYTES, "KAT[{i}] sk length");
        assert!(
            sm.len() >= SIGNATURE_BYTES,
            "KAT[{i}] sm shorter than SIGNATURE_BYTES"
        );
        let sig = &sm[..SIGNATURE_BYTES];

        write_seed(&dir_sig_parse, i, sig);
        write_seed(&dir_vk_parse, i, &pk);
        write_seed(&dir_sk_parse, i, &sk);

        // fuzz_verify: pk || sig || msg.
        let mut verify_input = Vec::with_capacity(pk.len() + sig.len() + msg.len());
        verify_input.extend_from_slice(&pk);
        verify_input.extend_from_slice(sig);
        verify_input.extend_from_slice(&msg);
        write_seed(&dir_verify, i, &verify_input);

        // fuzz_verify_typed: bytes XOR'd onto KAT 0's sig from byte 64
        // onward, then a trailing message. KAT i's seed = delta to
        // recover KAT i's sig under the harness's XOR + KAT 0 msg.
        let mut typed_input = Vec::with_capacity(VERIFY_TYPED_MUT_LEN + msg.len());
        for j in 0..VERIFY_TYPED_MUT_LEN {
            typed_input.push(sig[64 + j] ^ kat0_sig[64 + j]);
        }
        typed_input.extend_from_slice(&msg);
        write_seed(&dir_verify_typed, i, &typed_input);

        // fuzz_keygen_panic: the 48-byte DRBG seed.
        write_seed(&dir_keygen_panic, i, &seed);

        // fuzz_sign_panic: keygen_seed || sign_seed || msg. Same seed
        // for both halves — matches the NIST PQC pattern where one
        // DRBG is threaded through keygen and signing.
        let mut sign_input = Vec::with_capacity(2 * SEED_LEN + msg.len());
        sign_input.extend_from_slice(&seed);
        sign_input.extend_from_slice(&seed);
        sign_input.extend_from_slice(&msg);
        write_seed(&dir_sign_panic, i, &sign_input);
    }

    let n = KAT_VECTORS.len();
    println!("seeded {n} KAT-derived files into each of:");
    for d in [
        &dir_sig_parse,
        &dir_vk_parse,
        &dir_sk_parse,
        &dir_verify,
        &dir_verify_typed,
        &dir_keygen_panic,
        &dir_sign_panic,
    ] {
        println!("  {}", d.display());
    }
}
