//! Cross-KAT verify rejection.
//!
//! For every distinct pair `(i, j)` from `KAT_VECTORS`,
//! `vk_i.verify(msg_j, sig_j)` must return `Err` and must not panic.
//! Each KAT pair is self-consistent (the per-KAT `verify_kat_NNN`
//! tests confirm `vk_i.verify(msg_i, sig_i) == Ok`); this test
//! covers the *other* 99·100 cells of the matrix, which together
//! assert that `verify` actually binds a signature to its
//! corresponding key.
//!
//! Run with:
//!
//! ```text
//! cargo test --test cross_kat_verify --features expose-internals
//! ```

use proptest::prelude::*;
use sqisign_selkie::{
    SIGNATURE_BYTES, Signature, VERIFYING_KEY_BYTES, VerifyingKey, keys::kat_data::KAT_VECTORS,
};

fn parse_vk(idx: usize) -> VerifyingKey {
    let (_seed, pk_hex, _sk, _msg, _sig) = KAT_VECTORS[idx];
    let bytes = hex::decode(pk_hex).expect("KAT pk hex");
    let arr: &[u8; VERIFYING_KEY_BYTES] = bytes.as_slice().try_into().expect("pk length");
    VerifyingKey::from_bytes(arr).expect("KAT pk parses")
}

fn parse_sig_and_msg(idx: usize) -> (Signature, Vec<u8>) {
    let (_seed, _pk, _sk, msg_hex, sig_hex) = KAT_VECTORS[idx];
    let sig_bytes = hex::decode(sig_hex).expect("KAT sig hex");
    let arr: [u8; SIGNATURE_BYTES] = sig_bytes[..SIGNATURE_BYTES]
        .try_into()
        .expect("sig prefix present");
    let sig = Signature::from_bytes(&arr).expect("KAT sig parses");
    let msg = hex::decode(msg_hex).expect("KAT msg hex");
    (sig, msg)
}

proptest! {
    // Each case is one verify against a mismatched KAT pair —
    // typically ~10–50 ms in release. 256 cases ≈ a few seconds.
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    /// `vk_i.verify(msg_j, sig_j)` must return `Err` for any
    /// `i != j` drawn from `KAT_VECTORS`. A success here is a
    /// forgery: the signature was minted under `sk_j`'s flow but
    /// accepted by `vk_i`'s verifier.
    #[test]
    fn vk_rejects_signature_from_other_kat(
        i in 0usize..KAT_VECTORS.len(),
        j in 0usize..KAT_VECTORS.len(),
    ) {
        prop_assume!(i != j);
        let vk_i = parse_vk(i);
        let (sig_j, msg_j) = parse_sig_and_msg(j);
        prop_assert!(
            vk_i.verify(&msg_j, &sig_j).is_err(),
            "cross-KAT verify accepted: vk_{i} accepted (sig, msg) from KAT {j}",
        );
    }

    /// Same matrix, but pair `vk_i` with `sig_j` under `msg_i`.
    /// Catches a class of forgery where the signature accidentally
    /// validates under any vk + any msg because some component of
    /// verify isn't actually consulting the inputs (e.g. a
    /// short-circuit `Ok(())`). The forged-message-with-other-sig
    /// shape is the canonical strong-unforgeability check.
    #[test]
    fn vk_rejects_signature_under_own_msg(
        i in 0usize..KAT_VECTORS.len(),
        j in 0usize..KAT_VECTORS.len(),
    ) {
        prop_assume!(i != j);
        let vk_i = parse_vk(i);
        let (sig_j, _msg_j) = parse_sig_and_msg(j);
        let (_sig_i, msg_i) = parse_sig_and_msg(i);
        prop_assert!(
            vk_i.verify(&msg_i, &sig_j).is_err(),
            "cross-KAT verify accepted: vk_{i} accepted sig from KAT {j} under msg from KAT {i}",
        );
    }
}
