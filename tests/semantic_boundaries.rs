//! Cross-curve and semantic-boundary tests for verify.
//!
//! Hand-crafted negative inputs targeting:
//!
//!   * Degenerate Montgomery `A` values: `0`, `2`, `-2 mod p`. `A=0` is `E_0`
//!     (the SQIsign start curve, supersingular, valid as a parse); `A=2` and
//!     `A=-2` correspond to singular curves (Δ=0). All three must produce
//!     no-panic verifies, regardless of whether parse rejects.
//!   * `n_bt + r_rsp > e_rsp`: verify's `checked_sub` chain must catch this
//!     without integer underflow.
//!   * `n_bt` or `r_rsp` > `f=248`: parse must reject as `NonCanonical`.
//!   * `curve_aux == vk.curve`: weird but allowed-by-parse input; verify must
//!     reject without panic.
//!
//! Notable gap surfaced by this set: `SignatureError::NotSupersingular`
//! is defined in `errors.rs` but never returned from `VerifyingKey::from_bytes`
//! — non-supersingular `A` slips through parse and the chain has to
//! cope. Today these tests confirm no-panic; a future change should
//! make parse fail-fast with `NotSupersingular`.

use sqisign_selkie::{
    SIGNATURE_BYTES, Signature, SignatureError, VERIFYING_KEY_BYTES, VerifyingKey,
    keys::kat_data::KAT_VECTORS,
};

/// Decode the KAT 0 donor: (pk_bytes, sig_bytes, msg).
fn donor() -> ([u8; VERIFYING_KEY_BYTES], [u8; SIGNATURE_BYTES], Vec<u8>) {
    let (_seed, pk_hex, _sk, msg_hex, sig_hex) = KAT_VECTORS[0];
    let pk = hex::decode(pk_hex).unwrap();
    let sig = hex::decode(sig_hex).unwrap();
    let msg = hex::decode(msg_hex).unwrap();
    (
        pk.as_slice().try_into().unwrap(),
        sig[..SIGNATURE_BYTES].try_into().unwrap(),
        msg,
    )
}

/// `vk.A = 0` — the start curve `E_0: y² = x³ + x`. Supersingular for
/// `p = 5·2²⁴⁸ − 1`, so parse should succeed; verify with an unrelated
/// signature must reject without panic.
#[test]
fn vk_with_a_zero_does_not_panic() {
    let (_, sig_bytes, msg) = donor();
    let pk_bytes = [0u8; VERIFYING_KEY_BYTES];
    if let Ok(vk) = VerifyingKey::from_bytes(&pk_bytes) {
        let sig = Signature::from_bytes(&sig_bytes).expect("baseline sig parses");
        let _ = vk.verify(&msg, &sig);
    }
}

/// `vk.A = 2` — Montgomery curve `y² = x³ + 2x² + x`. Δ = 4(A²−4) = 0,
/// so the curve is singular. Parse must reject with
/// [`SignatureError::InvalidCurve`].
#[test]
fn vk_with_a_two_rejects_at_parse() {
    let mut pk_bytes = [0u8; VERIFYING_KEY_BYTES];
    pk_bytes[0] = 2;
    assert!(matches!(
        VerifyingKey::from_bytes(&pk_bytes),
        Err(SignatureError::InvalidCurve)
    ));
}

/// `vk.A = −2 mod p` — the other singular Montgomery curve. Real part
/// is `p − 2 = 5·2²⁴⁸ − 3`, encoded little-endian as
/// `FD FF…FF 04` (32 bytes); imaginary part is zero. Parse must
/// reject with [`SignatureError::InvalidCurve`].
#[test]
fn vk_with_a_negative_two_rejects_at_parse() {
    let mut pk_bytes = [0u8; VERIFYING_KEY_BYTES];
    pk_bytes[0] = 0xFD;
    for b in &mut pk_bytes[1..31] {
        *b = 0xFF;
    }
    pk_bytes[31] = 0x04;
    assert!(matches!(
        VerifyingKey::from_bytes(&pk_bytes),
        Err(SignatureError::InvalidCurve)
    ));
}

/// `sig.curve_aux.A = 2`: parse must reject with `InvalidCurve` —
/// same singular-curve check, on the signature side.
#[test]
fn sig_with_curve_aux_a_two_rejects_at_parse() {
    let (_, mut sig_bytes, _) = donor();
    for b in &mut sig_bytes[0..64] {
        *b = 0;
    }
    sig_bytes[0] = 2;
    assert!(matches!(
        Signature::from_bytes(&sig_bytes),
        Err(SignatureError::InvalidCurve)
    ));
}

/// `n_bt` large enough to drive the spec M_chl entry bound
/// `e_rsp − n_bt + 2` below the donor's actual entry width:
/// `Signature::from_bytes` must reject at parse with
/// [`SignatureError::NonCanonical`] rather than underflow or panic.
///
/// Each byte is individually within the per-byte bound (`n_bt`,
/// `r_rsp ≤ f = 248`), so the rejection comes from the composite
/// check enforced by [`ChallengeMatrix::parse`] against the
/// per-entry bound `e_rsp − n_bt + 2`.
#[test]
fn sig_with_n_bt_shrinks_m_chl_bound_rejects_at_parse() {
    let (_, mut sig_bytes, _) = donor();
    sig_bytes[64] = 200; // n_bt; donor M_chl entries exceed 2^(247+2-200) = 2^49
    sig_bytes[65] = 100; // r_rsp
    assert!(matches!(
        Signature::from_bytes(&sig_bytes),
        Err(SignatureError::NonCanonical)
    ));
}

/// `n_bt > f = 248`: `TorsionExponent::try_from(value)` must reject,
/// surfacing as `NonCanonical` from `Signature::from_bytes`.
#[test]
fn sig_with_n_bt_above_f_rejects_at_parse() {
    let (_, mut sig_bytes, _) = donor();
    sig_bytes[64] = 0xFF;
    assert!(matches!(
        Signature::from_bytes(&sig_bytes),
        Err(SignatureError::NonCanonical)
    ));
}

/// `r_rsp > f = 248`: same rejection path as `n_bt`.
#[test]
fn sig_with_r_rsp_above_f_rejects_at_parse() {
    let (_, mut sig_bytes, _) = donor();
    sig_bytes[65] = 0xFF;
    assert!(matches!(
        Signature::from_bytes(&sig_bytes),
        Err(SignatureError::NonCanonical)
    ));
}

/// `curve_aux ≡ vk.curve`: the auxiliary curve is meant to be a
/// distinct point in the supersingular isogeny graph from `E_pk`. A
/// signature where the encoded `A_aux` matches `vk`'s `A` should not
/// verify, and must not panic.
#[test]
fn sig_with_curve_aux_equal_vk_curve_rejects() {
    let (pk_bytes, mut sig_bytes, msg) = donor();
    sig_bytes[..64].copy_from_slice(&pk_bytes[..64]);
    let vk = VerifyingKey::from_bytes(&pk_bytes).unwrap();
    let sig = match Signature::from_bytes(&sig_bytes) {
        Ok(s) => s,
        Err(_) => return,
    };
    assert!(vk.verify(&msg, &sig).is_err());
}
