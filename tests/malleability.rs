//! Targeted typed-component malleability checks on a valid signature.
//!
//! Sister to [`bit_flip_sweep`](../bit_flip_sweep/index.html). That
//! test sweeps every bit blindly; this one perturbs specific
//! semantic components — `n_bt` ↔ `r_rsp` swap, `chl` zeroing, hint
//! byte rewrites, `M_chl` field zeroing — to catch malleability that
//! a coarse bit sweep might miss (e.g., a flip that the parser
//! normalizes away).
//!
//! Every mutation must yield `Err(_)` (parse rejection or
//! `VerificationFailed`); none may panic. The malleability of a
//! mutation that *still verifies* is a forgery primitive — that's
//! the bug class we're hunting.
//!
//! KAT 0 byte layout (148 bytes total, NIST-I):
//!
//! ```text
//!   [0..64)   A_aux        (Fp2)
//!   [64]      n_bt         (= 0x00 for KAT 0)
//!   [65]      r_rsp        (= 0x01 for KAT 0)
//!   [66..)    M_chl        (4 entries * ceil(E_RSP/8) bytes)
//!   [130..146)  chl        (16 bytes = ceil(E_CHL=122/8))
//!   [146]     hint_aux
//!   [147]     hint_chl
//! ```
//!
//! All offsets above are inherent to the wire format defined in
//! `Signature::from_bytes` (`src/keys/mod.rs`).

use sqisign_selkie::{
    SIGNATURE_BYTES, Signature, VERIFYING_KEY_BYTES, VerifyingKey, keys::kat_data::KAT_VECTORS,
};

const CHL_OFFSET: usize = 130;
const CHL_LEN: usize = 16;
const HINT_AUX_OFFSET: usize = 146;
const HINT_CHL_OFFSET: usize = 147;

fn donor() -> (VerifyingKey, [u8; SIGNATURE_BYTES], Vec<u8>) {
    let (_seed, pk_hex, _sk, msg_hex, sig_hex) = KAT_VECTORS[0];
    let pk = hex::decode(pk_hex).unwrap();
    let sig = hex::decode(sig_hex).unwrap();
    let msg = hex::decode(msg_hex).unwrap();
    let pk_arr: [u8; VERIFYING_KEY_BYTES] = pk.as_slice().try_into().unwrap();
    let sig_arr: [u8; SIGNATURE_BYTES] = sig[..SIGNATURE_BYTES].try_into().unwrap();
    let vk = VerifyingKey::from_bytes(&pk_arr).expect("baseline pk parses");
    (vk, sig_arr, msg)
}

/// Verify that the unmodified donor signature actually verifies —
/// every test below asserts a mutation breaks verification, which is
/// only meaningful if the baseline succeeds.
#[test]
fn baseline_verifies() {
    let (vk, sig_bytes, msg) = donor();
    let sig = Signature::from_bytes(&sig_bytes).expect("baseline sig parses");
    vk.verify(&msg, &sig)
        .expect("baseline (vk, sig, msg) must verify");
}

/// Swap `n_bt` and `r_rsp`: KAT 0 has `(0, 1)`, swapped becomes
/// `(1, 0)` — selects a different branch class in verify (n_bt > 0
/// vs r_rsp > 0). Must reject.
#[test]
fn swap_n_bt_and_r_rsp_rejects() {
    let (vk, mut sig_bytes, msg) = donor();
    sig_bytes.swap(64, 65);
    let sig = match Signature::from_bytes(&sig_bytes) {
        Ok(s) => s,
        Err(_) => return,
    };
    assert!(
        vk.verify(&msg, &sig).is_err(),
        "swapping n_bt and r_rsp must invalidate the signature"
    );
}

/// `chl` zeroed: the challenge hash output is the binding between
/// `(vk, msg, codomain.j_invariant)`; zeroing it almost certainly
/// breaks the equality compare in verify's final step.
#[test]
fn zeroed_chl_rejects() {
    let (vk, mut sig_bytes, msg) = donor();
    for b in &mut sig_bytes[CHL_OFFSET..CHL_OFFSET + CHL_LEN] {
        *b = 0;
    }
    let sig = match Signature::from_bytes(&sig_bytes) {
        Ok(s) => s,
        Err(_) => return,
    };
    assert!(
        vk.verify(&msg, &sig).is_err(),
        "zeroed chl must invalidate the signature"
    );
}

/// `hint_aux` rewritten: torsion-basis recovery on `E_aux` consumes
/// this hint; a wrong hint picks a different basis, leading to
/// different chain output and (overwhelmingly) verify rejection.
#[test]
fn rewritten_hint_aux_rejects() {
    let (vk, mut sig_bytes, msg) = donor();
    sig_bytes[HINT_AUX_OFFSET] = sig_bytes[HINT_AUX_OFFSET].wrapping_add(1);
    let sig = match Signature::from_bytes(&sig_bytes) {
        Ok(s) => s,
        Err(_) => return,
    };
    assert!(
        vk.verify(&msg, &sig).is_err(),
        "rewritten hint_aux must invalidate the signature"
    );
}

/// `hint_chl` rewritten: same rationale, but for the `E_chl` basis.
#[test]
fn rewritten_hint_chl_rejects() {
    let (vk, mut sig_bytes, msg) = donor();
    sig_bytes[HINT_CHL_OFFSET] = sig_bytes[HINT_CHL_OFFSET].wrapping_add(1);
    let sig = match Signature::from_bytes(&sig_bytes) {
        Ok(s) => s,
        Err(_) => return,
    };
    assert!(
        vk.verify(&msg, &sig).is_err(),
        "rewritten hint_chl must invalidate the signature"
    );
}

/// Whole `M_chl` field zeroed: zero matrix can't be a valid change-
/// of-basis under any signature; verify must reject.
#[test]
fn zeroed_m_chl_rejects() {
    let (vk, mut sig_bytes, msg) = donor();
    for b in &mut sig_bytes[66..CHL_OFFSET] {
        *b = 0;
    }
    let sig = match Signature::from_bytes(&sig_bytes) {
        Ok(s) => s,
        Err(_) => return,
    };
    assert!(
        vk.verify(&msg, &sig).is_err(),
        "zeroed M_chl must invalidate the signature"
    );
}
