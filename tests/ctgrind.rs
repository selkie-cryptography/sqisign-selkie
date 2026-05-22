// Gated to Linux because crabgrind's `bindgen(libclang)` build step
// looks up `<valgrind/valgrind.h>` on the system include paths. The
// header is not present on a stock darwin / macOS install (Valgrind
// is effectively unmaintained on macOS since Big Sur, no homebrew
// formula ships the dev headers), so the build script panics at test
// time even though the call sites are guarded by `#[cfg(unix)]`.
// CI runs this suite on the Linux runner image where Valgrind is
// installed via `infra/runners/Dockerfile`.
#![cfg(target_os = "linux")]
//! Secret-dependent memory access tests using Valgrind memcheck.
//!
//! Marks secret inputs as "undefined" using Valgrind client requests,
//! then runs crypto operations. Valgrind will report errors if any
//! branch or memory access depends on the secret data.
//!
//! Run with:
//!   cargo test --test ctgrind --features expose-internals --no-run
//!   valgrind --tool=memcheck --error-exitcode=1 \
//!     target/debug/deps/ctgrind-* --test-threads=1

use core::ffi::c_void;

use crabgrind::memcheck::{self, MemState};
use sqisign_selkie::{
    fields::{fp::Fp, fp2::Fp2},
    keys::{Signature, SigningKey, kat_data::KAT_VECTORS},
};

/// AES256-CTR-DRBG seed length (matches `crate::drbg::SEEDLEN`,
/// which is `pub(crate)` and not reachable from an integration test).
const DRBG_SEED_LEN: usize = 48;

/// Mark a byte slice as "secret" (undefined) for Valgrind.
/// When not running under Valgrind, this is a no-op.
fn mark_secret(data: &[u8]) {
    let _ = memcheck::mark_memory(
        data.as_ptr() as *const c_void,
        data.len(),
        MemState::Undefined,
    );
}

/// Mark a byte slice as "public" (defined) for Valgrind.
fn mark_public(data: &[u8]) {
    let _ = memcheck::mark_memory(
        data.as_ptr() as *const c_void,
        data.len(),
        MemState::Defined,
    );
}

#[test]
fn fp_mul_secret_independent() {
    let a_bytes = [0x42u8; 32];
    let b_bytes = [0x99u8; 32];

    // Mark inputs as secret.
    mark_secret(&a_bytes);
    mark_secret(&b_bytes);

    let a = Fp::from_bytes(&a_bytes);
    let b = Fp::from_bytes(&b_bytes);

    // Multiplication must not branch on secret data.
    let result = a * b;

    // Mark result as public so the test framework can inspect it.
    let result_bytes = result.to_bytes();
    mark_public(&result_bytes);
}

#[test]
fn fp_add_secret_independent() {
    let a_bytes = [0x42u8; 32];
    let b_bytes = [0x99u8; 32];
    mark_secret(&a_bytes);
    mark_secret(&b_bytes);

    let a = Fp::from_bytes(&a_bytes);
    let b = Fp::from_bytes(&b_bytes);
    let result = a + b;

    let result_bytes = result.to_bytes();
    mark_public(&result_bytes);
}

#[test]
fn fp_sub_secret_independent() {
    let a_bytes = [0x42u8; 32];
    let b_bytes = [0x99u8; 32];
    mark_secret(&a_bytes);
    mark_secret(&b_bytes);

    let a = Fp::from_bytes(&a_bytes);
    let b = Fp::from_bytes(&b_bytes);
    let result = a - b;

    let result_bytes = result.to_bytes();
    mark_public(&result_bytes);
}

#[test]
fn fp2_mul_secret_independent() {
    let bytes = [0x42u8; 64];
    mark_secret(&bytes);

    let a = Fp2::new(
        Fp::from_bytes(bytes[..32].try_into().unwrap()),
        Fp::from_bytes(bytes[32..].try_into().unwrap()),
    );
    let b = Fp2::new(Fp::from_bytes(&[0x11; 32]), Fp::from_bytes(&[0x22; 32]));
    let result = a * b;

    // Consume result without inspecting it.
    std::hint::black_box(result);
}

#[test]
fn fp_ct_select_secret_independent() {
    use subtle::ConditionallySelectable;

    let a = Fp::from_bytes(&[0x42; 32]);
    let b = Fp::from_bytes(&[0x99; 32]);

    // The choice bit is secret.
    let mut choice_byte = 1u8;
    mark_secret(std::slice::from_ref(&choice_byte));
    let choice = subtle::Choice::from(choice_byte);

    let result = Fp::conditional_select(&a, &b, choice);

    let result_bytes = result.to_bytes();
    mark_public(&result_bytes);

    // Also mark the choice as public again for test cleanup.
    choice_byte = 0;
    mark_public(std::slice::from_ref(&choice_byte));
}

// --- Slow top-level CT tests -------------------------------------------------
//
// keygen and sign each take seconds; under Valgrind that becomes
// minutes. These tests are gated on the `CTGRIND_SLOW` env var so
// the regular ctgrind job stays fast; the weekly `ct` workflow sets
// the var. Returning early via `skip_unless_slow` is the
// CI-friendly form: the test still appears in `--list`, ctgrind-
// report still attributes errors to it, and a `pass` with zero
// errors when the var is unset is the right signal.

/// Returns `true` if the caller should skip — used to early-return
/// from the slow tests when `CTGRIND_SLOW` isn't set.
fn skip_unless_slow(name: &str) -> bool {
    if std::env::var_os("CTGRIND_SLOW").is_none() {
        eprintln!("[ctgrind] skipping {name}; set CTGRIND_SLOW=1 to enable");
        return true;
    }
    false
}

/// Decodes the first KAT vector's seed (48 bytes hex → 48 bytes).
fn kat0_seed() -> [u8; DRBG_SEED_LEN] {
    let hex_seed = KAT_VECTORS[0].0;
    let bytes = hex::decode(hex_seed).expect("KAT seed is hex");
    let mut out = [0u8; DRBG_SEED_LEN];
    out.copy_from_slice(&bytes);
    out
}

#[test]
fn keygen_secret_independent() {
    if skip_unless_slow("keygen_secret_independent") {
        return;
    }

    let mut seed = kat0_seed();
    mark_secret(&seed);

    let sk = SigningKey::generate_derand(&seed).expect("keygen on KAT 0 succeeds");

    // Declassify the verifying key so Valgrind doesn't complain about
    // its bytes leaving the test. The sk itself is dropped silently.
    let vk_bytes = sk.verifying_key().to_bytes();
    mark_public(&vk_bytes);

    // Re-defining the seed buffer so test-framework cleanup doesn't
    // touch tainted memory.
    seed.fill(0);
    mark_public(&seed);
}

#[test]
fn sign_secret_independent() {
    if skip_unless_slow("sign_secret_independent") {
        return;
    }

    // KAT 0: seed (secret) → sk; sk used to sign a public message.
    // Taint propagates: every `BigInt`/`Fp`/`Lattice` value derived
    // from `seed` is marked undefined by Valgrind, so any branch on
    // a secret-derived value lights up in the memcheck report.
    let mut seed = kat0_seed();
    mark_secret(&seed);
    let sk = SigningKey::generate_derand(&seed).expect("keygen on KAT 0 succeeds");

    let msg = hex::decode(KAT_VECTORS[0].3).expect("KAT msg is hex");

    // Sign uses a separate per-signature seed; mark it secret too —
    // the signing nonce is sensitive (a biased / leaked sig nonce
    // breaks the scheme).
    let mut sign_seed = [0u8; DRBG_SEED_LEN];
    sign_seed[0] = 0xA5; // arbitrary fixed value for reproducibility
    mark_secret(&sign_seed);

    let sig = sk.sign_derand(&msg, &sign_seed).expect("sign succeeds");

    let sig_bytes = sig.to_bytes();
    mark_public(&sig_bytes);
    seed.fill(0);
    sign_seed.fill(0);
    mark_public(&seed);
    mark_public(&sign_seed);
}

#[test]
fn verify_secret_independent() {
    if skip_unless_slow("verify_secret_independent") {
        return;
    }

    // Verify's inputs (vk, sig, msg) are all public per the spec, so
    // this is **not** a CT-on-secrets test. It exercises
    // *oracle-resistance*: an attacker submits crafted signatures
    // and observes timing/branch deltas to learn about either the
    // verifier's state or the structure of a hidden valid signature
    // they're trying to forge. Marking `sig` as undefined captures
    // any branch that depends on bits the attacker chose. Many such
    // branches are expected today (e.g. `n_bt` / `r_rsp` parsing) —
    // the value is tracking the count over time, not gating CI.
    let seed = kat0_seed();
    let sk = SigningKey::generate_derand(&seed).expect("keygen on KAT 0 succeeds");
    let vk = sk.verifying_key();

    let msg = hex::decode(KAT_VECTORS[0].3).expect("KAT msg is hex");
    let sign_seed = [0xA5u8; DRBG_SEED_LEN];
    let sig = sk.sign_derand(&msg, &sign_seed).expect("sign succeeds");

    let mut sig_bytes = sig.to_bytes();
    mark_secret(&sig_bytes);
    let sig_under_test = Signature::from_bytes(&sig_bytes).expect("our own signature round-trips");

    let _ = std::hint::black_box(vk.verify(&msg, &sig_under_test));

    sig_bytes.fill(0);
    mark_public(&sig_bytes);
}
