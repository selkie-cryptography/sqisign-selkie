//! Seed `fuzz/corpus/fuzz_verify/` with parseable `pk || sig || msg`
//! triples derived from the bundled 100-vector KAT set.
//!
//! libFuzzer's coverage-guided exploration is dramatically more
//! effective when the starting corpus already contains inputs that
//! reach the deep code paths. With random bytes alone, ~99.999% of
//! iterations bounce at [`Signature::from_bytes`] (`NotSupersingular`
//! on `curve_aux` is the dominant rejector). Seeding with real KAT
//! triples gives libFuzzer 100 inputs that *already* parse and verify;
//! all of its byte-level mutations then start one step away from
//! exercising the verify (2,2)-chain.
//!
//! Run before fuzzing:
//!
//! ```sh
//! cargo run --release --features expose-internals --example seed-fuzz-corpus
//! cd fuzz && cargo +nightly fuzz run fuzz_verify
//! ```
//!
//! The `expose-internals` feature flag is required because the
//! seeder reads `keys::kat_data::KAT_VECTORS`; the gating is
//! enforced via `required-features` on the `[[example]]` entry in
//! the top-level `Cargo.toml` so attempts without the flag fail
//! fast with a clear cargo message rather than a cryptic build
//! error.
//!
//! Output files are content-keyed by index (`kat-000` … `kat-099`).
//! libFuzzer deduplicates on content hash, so re-running the seeder
//! is idempotent.

use std::{fs, path::PathBuf};

use sqisign_selkie::{SIGNATURE_BYTES, keys::kat_data::KAT_VECTORS};

fn main() {
    let dest = PathBuf::from("fuzz/corpus/fuzz_verify");
    fs::create_dir_all(&dest).expect("create fuzz corpus dir");

    for (i, &(_seed_hex, pk_hex, _sk_hex, msg_hex, sm_hex)) in KAT_VECTORS.iter().enumerate() {
        let pk = hex::decode(pk_hex).expect("pk hex decode");
        let msg = hex::decode(msg_hex).expect("msg hex decode");
        let sm = hex::decode(sm_hex).expect("sm hex decode");

        // NIST PQC `sm` = sig || msg. Slice off the leading
        // SIGNATURE_BYTES to recover the bare signature.
        assert!(
            sm.len() >= SIGNATURE_BYTES,
            "KAT[{i}] sm shorter than SIGNATURE_BYTES"
        );
        let sig = &sm[..SIGNATURE_BYTES];

        // `fuzz_verify` reads its input as pk || sig || trailing-msg.
        let mut buf = Vec::with_capacity(pk.len() + sig.len() + msg.len());
        buf.extend_from_slice(&pk);
        buf.extend_from_slice(sig);
        buf.extend_from_slice(&msg);

        let path = dest.join(format!("kat-{i:03}"));
        fs::write(&path, &buf).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }

    println!(
        "seeded {} corpus files in {}",
        KAT_VECTORS.len(),
        dest.display()
    );
}
