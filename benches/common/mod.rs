//! Shared fixtures for divan and gungraun benches.
//!
//! Anchored on KAT[0] (pinned to a specific C reference commit; see
//! [`C_REF_COMMIT`]) — every bench that pulls a fixture from here
//! measures byte-identical inputs across files and runs. The hex
//! strings live in [`sqisign_selkie::keys::kat_data::KAT_VECTORS`];
//! we read them through that one source of truth rather than
//! duplicating.
//!
//! All bench targets that include this module enable the
//! `expose-internals` feature in `Cargo.toml`, which is what makes
//! the `kat_data` module reachable.
//!
//! `mod common;` from any bench file pulls this in; the module is in
//! a subdirectory (`benches/common/mod.rs`) rather than a top-level
//! `benches/common.rs` so Cargo doesn't auto-discover it as a bench
//! binary target. Different benches use different subsets, so
//! dead-code warnings are silenced module-wide.

#![allow(dead_code)]

use sqisign_selkie::{
    SIGNATURE_BYTES, SIGNING_KEY_BYTES, Signature, SigningKey, VERIFYING_KEY_BYTES, VerifyingKey,
    fields::{fp::Fp, fp2::Fp2},
    keys::kat_data::KAT_VECTORS,
};

/// Pinned C reference commit the KAT fixtures track.
pub const C_REF_COMMIT: &str = "91e9e464fe5400192d13e1f9240cbf180200a103";

/// Precomputed x-coordinate of `P₀ − Q₀` on E₀, used by surface and
/// curve benches that need a propagated `PmQ` to seed differential
/// addition / Tate-pairing fixtures.
///
/// Same value as
/// `sqisign_selkie::deuring::precomputed::torsion_basis::E0_PMQ_X`,
/// which is `pub(crate)` and not reachable from benches; lives here
/// rather than in `params` because production code recomputes
/// `P − Q` on the fly via `projective_difference`.
pub const BASIS_E0_PMQ_X: Fp2 = Fp2::new(
    Fp::from_limbs([
        270480358487834,
        2072266045736319,
        1674191439884908,
        2200260875474967,
        6907110771017,
    ]),
    Fp::from_limbs([
        1752869285732728,
        495365606488051,
        1818143936964406,
        314346222928849,
        165077940050103,
    ]),
);

/// KAT[0] tuple: `(seed, pk, sk, msg, sm)`.
fn kat0() -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    KAT_VECTORS[0]
}

/// KAT[0] verifying key, signature, and message — the verify-side
/// fixture used by every "verify a known-good signature" bench.
pub fn kat0_vk_sig_msg() -> (VerifyingKey, Signature, Vec<u8>) {
    let (_, pk, _, _, sm) = kat0();
    let pk_bytes = hex::decode(pk).expect("KAT[0] pk hex");
    let sm_bytes = hex::decode(sm).expect("KAT[0] sm hex");

    let pk_arr: &[u8; VERIFYING_KEY_BYTES] = pk_bytes
        .as_slice()
        .try_into()
        .expect("pk byte length matches VERIFYING_KEY_BYTES");
    let sig_arr: &[u8; SIGNATURE_BYTES] = sm_bytes[..SIGNATURE_BYTES]
        .try_into()
        .expect("signed message has signature prefix");

    let vk = VerifyingKey::from_bytes(pk_arr).expect("KAT[0] vk parses");
    let sig = Signature::from_bytes(sig_arr).expect("KAT[0] sig parses");
    let msg = sm_bytes[SIGNATURE_BYTES..].to_vec();

    (vk, sig, msg)
}

/// KAT[0]'s 48-byte derand seed. Drives `generate_derand` for
/// reproducible end-to-end keygen timing across runs.
pub fn kat0_seed() -> [u8; 48] {
    let (seed, ..) = kat0();
    let bytes = hex::decode(seed).expect("KAT[0] seed hex");
    bytes.as_slice().try_into().expect("48-byte seed")
}

/// Fully-parsed KAT[0] [`SigningKey`] — much faster than running
/// `generate_derand`, since it skips the (2,2)-chain. Use for
/// sign-side benches where keygen is not what's being measured.
pub fn kat0_signing_key() -> SigningKey {
    let (_, _, sk, ..) = kat0();
    let sk_bytes = hex::decode(sk).expect("KAT[0] sk hex");
    let sk_arr: &[u8; SIGNING_KEY_BYTES] = sk_bytes
        .as_slice()
        .try_into()
        .expect("sk byte length matches SIGNING_KEY_BYTES");
    SigningKey::from_bytes(sk_arr).expect("KAT[0] sk parses")
}

/// KAT[0] verifying-key bytes — for parse-only benches where we
/// don't want the bench to include `from_bytes` validation.
pub fn kat0_vk_bytes() -> [u8; VERIFYING_KEY_BYTES] {
    let (_, pk, ..) = kat0();
    let bytes = hex::decode(pk).expect("KAT[0] pk hex");
    bytes.as_slice().try_into().expect("vk byte length")
}

/// KAT[0] signature bytes (signature prefix of the signed message).
pub fn kat0_sig_bytes() -> [u8; SIGNATURE_BYTES] {
    let (_, _, _, _, sm) = kat0();
    let bytes = hex::decode(sm).expect("KAT[0] sm hex");
    let arr: &[u8; SIGNATURE_BYTES] = bytes[..SIGNATURE_BYTES]
        .try_into()
        .expect("signed message has signature prefix");
    *arr
}

/// KAT[0] signing-key bytes.
pub fn kat0_sk_bytes() -> [u8; SIGNING_KEY_BYTES] {
    let (_, _, sk, ..) = kat0();
    let bytes = hex::decode(sk).expect("KAT[0] sk hex");
    bytes.as_slice().try_into().expect("sk byte length")
}
