//! Seed `fuzz/corpus/<target>/` directories with KAT- and
//! Wycheproof-derived inputs.
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
//! Sources:
//! - 100 NIST PQC KAT tuples from `keys::kat_data::KAT_VECTORS` (valid keypair
//!   + signature + message tuples).
//! - Wycheproof-format vectors at `tests/vectors/sqisign_*.json`, including the
//!   52 perturbed-or-valid verify cases and 24 extended-verify cases. These are
//!   high-value seeds for rejection-path coverage — many are already crafted to
//!   land one bit away from a parse boundary.
//!
//! Output files are content-keyed by source + index
//! (`kat-000` … `kat-099`, `verify-000` … `verify-051`, …). libFuzzer
//! deduplicates on content hash, so re-running the seeder is
//! idempotent.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use sqisign_selkie::{
    SIGNATURE_BYTES, SIGNING_KEY_BYTES, VERIFYING_KEY_BYTES, keys::kat_data::KAT_VECTORS,
};

/// `_derand` seed length in bytes (AES256-CTR-DRBG KEYLEN + BLOCKLEN).
const SEED_LEN: usize = 48;

/// Length of the XOR delta region of `fuzz_verify_typed`'s input — the
/// signature bytes after `curve_aux`, which the harness mutates.
const VERIFY_TYPED_MUT_LEN: usize = SIGNATURE_BYTES - 64;

/// Wycheproof verify-file schema. Each `[publicKey + tests]` group
/// shares one verifying key across N (msg, sig) tests, some valid
/// and some invalid. Mirrors `tests/wycheproof.rs::TestFile` so both
/// readers can consume the same JSON files. The JSON keeps the
/// upstream `publicKey` / `pk` names; we rename to `vk` at the
/// boundary per the project's verifying-key naming convention.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WycheproofVerifyFile {
    test_groups: Vec<WycheproofVerifyGroup>,
}

/// One verify group: a verifying key plus the tests that pair against
/// it. The JSON wire format uses `publicKey` / `pk`; we rename to
/// `vk` at the boundary to stay consistent with the rest of the
/// project (`VerifyingKey`, `SigningKey`).
#[derive(Deserialize)]
struct WycheproofVerifyGroup {
    #[serde(rename = "publicKey")]
    vk: WycheproofVerifyingKey,
    tests: Vec<WycheproofVerifyTest>,
}

/// Group-level verifying key. Hex-encoded.
#[derive(Deserialize)]
struct WycheproofVerifyingKey {
    #[serde(rename = "pk")]
    vk: String,
}

/// One verify vector. Both `msg` and `sig` are hex; invalid vectors
/// may have non-standard sig lengths.
#[derive(Deserialize)]
struct WycheproofVerifyTest {
    msg: String,
    sig: String,
}

/// Wycheproof keygen-file schema. Each test pins (seed, vk, sk).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WycheproofKeygenFile {
    test_groups: Vec<WycheproofKeygenGroup>,
}

#[derive(Deserialize)]
struct WycheproofKeygenGroup {
    tests: Vec<WycheproofKeygenTest>,
}

/// Keygen tests come in several shapes: some pin all three fields
/// (deterministic-keygen check), some pin only vk + sk (sk-embeds-vk
/// consistency check), some pin only a truncated/perturbed sk
/// (rejection-path check). All fields are optional; the seeder
/// writes each target only when its field is present.
#[derive(Deserialize)]
struct WycheproofKeygenTest {
    #[serde(default)]
    seed: Option<String>,
    #[serde(default, rename = "pk")]
    vk: Option<String>,
    #[serde(default)]
    sk: Option<String>,
}

/// Wycheproof sign-file schema. Each test pins (sk, vk, msg, sig).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WycheproofSignFile {
    test_groups: Vec<WycheproofSignGroup>,
}

#[derive(Deserialize)]
struct WycheproofSignGroup {
    tests: Vec<WycheproofSignTest>,
}

#[derive(Deserialize)]
struct WycheproofSignTest {
    sk: String,
    #[serde(rename = "pk")]
    vk: String,
    msg: String,
    sig: String,
}

/// Writes one seed file into `dir`. `key` becomes the filename;
/// libFuzzer dedupes on content hash, so collisions across sources
/// (e.g., `kat-000` and `verify-001` having the same bytes) are
/// harmless.
fn write_seed(dir: &Path, key: &str, bytes: &[u8]) {
    let path = dir.join(key);
    fs::write(&path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Creates (idempotent) `fuzz/corpus/<name>/` and returns the path.
fn ensure_dir(name: &str) -> PathBuf {
    let dir = Path::new("fuzz/corpus").join(name);
    fs::create_dir_all(&dir).expect("create fuzz corpus dir");
    dir
}

/// One-stop seeding context: every output corpus directory plus the
/// XOR baseline `fuzz_verify_typed` mutates against. Constructed once
/// in `main` and passed by reference to each per-source seed method.
struct Seeder {
    sig_parse: PathBuf,
    vk_parse: PathBuf,
    sk_parse: PathBuf,
    verify: PathBuf,
    verify_typed: PathBuf,
    keygen_panic: PathBuf,
    sign_panic: PathBuf,
    kat0_sig: [u8; SIGNATURE_BYTES],
}

impl Seeder {
    /// Sets up every `fuzz/corpus/<target>/` directory and captures
    /// KAT 0's signature as the XOR baseline for `fuzz_verify_typed`.
    fn new() -> Self {
        let kat0_sig = {
            let sm0 = hex::decode(KAT_VECTORS[0].4).expect("KAT 0 sm hex");
            let mut s = [0u8; SIGNATURE_BYTES];
            s.copy_from_slice(&sm0[..SIGNATURE_BYTES]);
            s
        };

        Self {
            sig_parse: ensure_dir("fuzz_signature_parse"),
            vk_parse: ensure_dir("fuzz_verifying_key_parse"),
            sk_parse: ensure_dir("fuzz_signing_key_parse"),
            verify: ensure_dir("fuzz_verify"),
            verify_typed: ensure_dir("fuzz_verify_typed"),
            keygen_panic: ensure_dir("fuzz_keygen_panic"),
            sign_panic: ensure_dir("fuzz_sign_panic"),
            kat0_sig,
        }
    }

    /// Returns the bytes to seed `fuzz_verify_typed` with for a given
    /// (sig, msg) pair, or `None` if the sig is the wrong length to
    /// XOR against `kat0_sig`. The harness reads exactly
    /// `VERIFY_TYPED_MUT_LEN` mutation bytes followed by an arbitrary
    /// message; a truncated sig can't produce a well-formed seed.
    fn verify_typed_bytes(&self, sig: &[u8], msg: &[u8]) -> Option<Vec<u8>> {
        if sig.len() < SIGNATURE_BYTES {
            return None;
        }

        let mut out = Vec::with_capacity(VERIFY_TYPED_MUT_LEN + msg.len());
        for j in 0..VERIFY_TYPED_MUT_LEN {
            out.push(sig[64 + j] ^ self.kat0_sig[64 + j]);
        }
        out.extend_from_slice(msg);
        Some(out)
    }

    /// Seeds the parse-, verify-, and verify-typed-targets from one
    /// Wycheproof verify-style file. Each group's `vk` seeds the
    /// verifying-key parser; each `(vk, sig, msg)` triple seeds
    /// `fuzz_verify`; sigs alone seed the signature parser; if the
    /// sig length is canonical, a `fuzz_verify_typed` XOR delta is
    /// also written.
    ///
    /// `tag` differentiates this file's seeds from others (e.g.
    /// `verify` vs `verify-ext`) so filenames don't collide across
    /// sources. Returns the count of tests seeded.
    fn seed_from_verify_file(&self, path: &str, tag: &str) -> usize {
        let file = load_verify_file(path);
        let mut count = 0;

        for group in &file.test_groups {
            let vk =
                hex::decode(&group.vk.vk).unwrap_or_else(|e| panic!("{path} group vk hex: {e}"));

            for test in &group.tests {
                let sig = hex::decode(&test.sig).unwrap_or_else(|e| panic!("{path} sig hex: {e}"));
                let msg = hex::decode(&test.msg).unwrap_or_else(|e| panic!("{path} msg hex: {e}"));

                let key = format!("{tag}-{count:03}");

                write_seed(&self.sig_parse, &key, &sig);
                write_seed(&self.vk_parse, &key, &vk);

                let mut verify_input = Vec::with_capacity(vk.len() + sig.len() + msg.len());
                verify_input.extend_from_slice(&vk);
                verify_input.extend_from_slice(&sig);
                verify_input.extend_from_slice(&msg);
                write_seed(&self.verify, &key, &verify_input);

                if let Some(typed) = self.verify_typed_bytes(&sig, &msg) {
                    write_seed(&self.verify_typed, &key, &typed);
                }

                count += 1;
            }
        }

        count
    }

    /// Seeds the keygen-, vk-parse-, and sk-parse-targets from one
    /// Wycheproof keygen-style file. Each test pins a 48-byte DRBG
    /// seed plus the expected vk/sk for it; the seed alone is the
    /// `fuzz_keygen_panic` input, while vk and sk seed their parsers.
    fn seed_from_keygen_file(&self, path: &str, tag: &str) -> usize {
        let file = load_keygen_file(path);
        let mut count = 0;

        for group in &file.test_groups {
            for test in &group.tests {
                let key = format!("{tag}-{count:03}");

                if let Some(seed_hex) = &test.seed {
                    let seed =
                        hex::decode(seed_hex).unwrap_or_else(|e| panic!("{path} seed hex: {e}"));
                    write_seed(&self.keygen_panic, &key, &seed);
                }

                if let Some(vk_hex) = &test.vk {
                    let vk = hex::decode(vk_hex).unwrap_or_else(|e| panic!("{path} vk hex: {e}"));
                    write_seed(&self.vk_parse, &key, &vk);
                }

                if let Some(sk_hex) = &test.sk {
                    let sk = hex::decode(sk_hex).unwrap_or_else(|e| panic!("{path} sk hex: {e}"));
                    write_seed(&self.sk_parse, &key, &sk);
                }

                count += 1;
            }
        }

        count
    }

    /// Seeds the parse-, verify-, and sk-parse-targets from one
    /// Wycheproof sign-style file. Sign vectors don't carry the
    /// keygen DRBG seed so `fuzz_sign_panic` is skipped; the (vk, sig,
    /// msg) triple still seeds `fuzz_verify` and the parsers.
    fn seed_from_sign_file(&self, path: &str, tag: &str) -> usize {
        let file = load_sign_file(path);
        let mut count = 0;

        for group in &file.test_groups {
            for test in &group.tests {
                let sk = hex::decode(&test.sk).unwrap_or_else(|e| panic!("{path} sk hex: {e}"));
                let vk = hex::decode(&test.vk).unwrap_or_else(|e| panic!("{path} vk hex: {e}"));
                let msg = hex::decode(&test.msg).unwrap_or_else(|e| panic!("{path} msg hex: {e}"));
                let sig = hex::decode(&test.sig).unwrap_or_else(|e| panic!("{path} sig hex: {e}"));

                let key = format!("{tag}-{count:03}");

                write_seed(&self.sig_parse, &key, &sig);
                write_seed(&self.vk_parse, &key, &vk);
                write_seed(&self.sk_parse, &key, &sk);

                let mut verify_input = Vec::with_capacity(vk.len() + sig.len() + msg.len());
                verify_input.extend_from_slice(&vk);
                verify_input.extend_from_slice(&sig);
                verify_input.extend_from_slice(&msg);
                write_seed(&self.verify, &key, &verify_input);

                if let Some(typed) = self.verify_typed_bytes(&sig, &msg) {
                    write_seed(&self.verify_typed, &key, &typed);
                }

                count += 1;
            }
        }

        count
    }

    /// Seeds every target from the bundled NIST PQC KAT tuple at
    /// index `i`. Strict length asserts here are intentional — KATs
    /// must be well-formed; corrupt KAT data should fail loudly.
    fn seed_from_kat(
        &self,
        i: usize,
        (seed_hex, vk_hex, sk_hex, msg_hex, sm_hex): (&str, &str, &str, &str, &str),
    ) {
        let seed = hex::decode(seed_hex).expect("seed hex decode");
        let vk = hex::decode(vk_hex).expect("vk hex decode");
        let sk = hex::decode(sk_hex).expect("sk hex decode");
        let msg = hex::decode(msg_hex).expect("msg hex decode");
        let sm = hex::decode(sm_hex).expect("sm hex decode");

        assert_eq!(seed.len(), SEED_LEN, "KAT[{i}] seed length");
        assert_eq!(vk.len(), VERIFYING_KEY_BYTES, "KAT[{i}] vk length");
        assert_eq!(sk.len(), SIGNING_KEY_BYTES, "KAT[{i}] sk length");
        assert!(
            sm.len() >= SIGNATURE_BYTES,
            "KAT[{i}] sm shorter than SIGNATURE_BYTES"
        );
        let sig = &sm[..SIGNATURE_BYTES];

        let key = format!("kat-{i:03}");

        write_seed(&self.sig_parse, &key, sig);
        write_seed(&self.vk_parse, &key, &vk);
        write_seed(&self.sk_parse, &key, &sk);

        // fuzz_verify: vk || sig || msg.
        let mut verify_input = Vec::with_capacity(vk.len() + sig.len() + msg.len());
        verify_input.extend_from_slice(&vk);
        verify_input.extend_from_slice(sig);
        verify_input.extend_from_slice(&msg);
        write_seed(&self.verify, &key, &verify_input);

        // fuzz_verify_typed: KAT sigs are full-length so the XOR is
        // always well-formed; unwrap matches the existing assert above.
        let typed = self
            .verify_typed_bytes(sig, &msg)
            .expect("KAT sig has SIGNATURE_BYTES length per the assert above");
        write_seed(&self.verify_typed, &key, &typed);

        // fuzz_keygen_panic: the 48-byte DRBG seed.
        write_seed(&self.keygen_panic, &key, &seed);

        // fuzz_sign_panic: keygen_seed || sign_seed || msg. Same seed
        // for both halves — matches the NIST PQC pattern where one
        // DRBG is threaded through keygen and signing.
        let mut sign_input = Vec::with_capacity(2 * SEED_LEN + msg.len());
        sign_input.extend_from_slice(&seed);
        sign_input.extend_from_slice(&seed);
        sign_input.extend_from_slice(&msg);
        write_seed(&self.sign_panic, &key, &sign_input);
    }

    /// Lists every destination directory in spawn order, for the
    /// run-summary printer in `main`.
    fn destinations(&self) -> [&Path; 7] {
        [
            &self.sig_parse,
            &self.vk_parse,
            &self.sk_parse,
            &self.verify,
            &self.verify_typed,
            &self.keygen_panic,
            &self.sign_panic,
        ]
    }
}

/// Loads + parses a Wycheproof verify-style JSON file. Panics on any
/// I/O or schema error — the seeder is a build step where stale or
/// corrupt vector files should fail loudly.
fn load_verify_file(path: &str) -> WycheproofVerifyFile {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// Loads + parses a Wycheproof keygen-style JSON file.
fn load_keygen_file(path: &str) -> WycheproofKeygenFile {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// Loads + parses a Wycheproof sign-style JSON file.
fn load_sign_file(path: &str) -> WycheproofSignFile {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

fn main() {
    let seeder = Seeder::new();

    for (i, &kat) in KAT_VECTORS.iter().enumerate() {
        seeder.seed_from_kat(i, kat);
    }

    let kat_count = KAT_VECTORS.len();

    // Wycheproof verify vectors. The bulk of the negative-path
    // coverage lives here — sigs and vks crafted to land at parse
    // and verify boundaries.
    let verify_count = seeder.seed_from_verify_file("tests/vectors/sqisign_verify.json", "verify");
    let verify_ext_count =
        seeder.seed_from_verify_file("tests/vectors/sqisign_verify_extended.json", "verify-ext");

    // Wycheproof keygen vectors: seeds + expected vk/sk.
    let keygen_count = seeder.seed_from_keygen_file("tests/vectors/sqisign_keygen.json", "keygen");

    // Wycheproof sign vectors: (sk, vk, msg, sig).
    let sign_count = seeder.seed_from_sign_file("tests/vectors/sqisign_sign.json", "sign");

    println!("seeded fuzz corpora:");
    println!("  {kat_count} KAT vectors (kat-NNN)");
    println!("  {verify_count} Wycheproof verify vectors (verify-NNN)");
    println!("  {verify_ext_count} Wycheproof extended-verify vectors (verify-ext-NNN)");
    println!("  {keygen_count} Wycheproof keygen vectors (keygen-NNN)");
    println!("  {sign_count} Wycheproof sign vectors (sign-NNN)");
    println!("destinations:");
    for d in seeder.destinations() {
        println!("  {}", d.display());
    }
}
