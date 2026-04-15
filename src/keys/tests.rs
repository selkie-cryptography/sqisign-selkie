use super::*;

/// Known-good KAT vector 0 (pk, sm hex) from the C reference
/// implementation, used by the wire-format round-trip tests below.
const KAT0_PK: &str = "07CCD21425136F6E865E497D2D4D208F0054AD81372066E817480787AAF7B2029550C89E892D618CE3230F23510BFBE68FCCDDAEA51DB1436B462ADFAF008A010B";
const KAT0_SM: &str = "84228651F271B0F39F2F19F2E8718F31ED3365AC9E5CB303AFE663D0CFC11F0455D891B0CA6C7E653F9BA2667730BB77BEFE1B1A31828404284AF8FD7BAACC010001D974B5CA671FF65708D8B462A5A84A1443EE9B5FED7218767C9D85CEED04DB0A69A2F6EC3BE835B3B2624B9A0DF68837AD00BCACC27D1EC806A44840267471D86EFF3447018ADB0A6551EE8322AB30010202D81C4D8D734FCBFBEADE3D3F8A039FAA2A2C9957E835AD55B22E75BF57BB556AC8";

/// Helper: parse a NIST PQC KAT entry and verify the signature.
fn verify_kat(pk_hex: &str, sm_hex: &str) {
    let pk_bytes = hex::decode(pk_hex).unwrap();
    let sm_bytes = hex::decode(sm_hex).unwrap();

    // NIST PQC format: sm = signature || message.
    let sig_bytes: &[u8; SIGNATURE_BYTES] = sm_bytes[..SIGNATURE_BYTES]
        .try_into()
        .expect("sm shorter than SIGNATURE_BYTES");
    let msg = &sm_bytes[SIGNATURE_BYTES..];

    let vk = VerifyingKey::from_bytes(pk_bytes.as_slice().try_into().unwrap())
        .expect("public key should parse");
    let sig = Signature::from_bytes(sig_bytes).expect("signature should parse");

    vk.verify(msg, &sig).expect("signature should verify");
}

// ---------------- wire-format round-trip tests ----------------

/// Parse a KAT signature and re-serialize it: the bytes must be
/// byte-identical.
#[test]
fn signature_to_bytes_matches_kat_input() {
    let sm_bytes = hex::decode(KAT0_SM).unwrap();
    let sig_bytes: &[u8; SIGNATURE_BYTES] = sm_bytes[..SIGNATURE_BYTES].try_into().unwrap();
    let sig = Signature::from_bytes(sig_bytes).unwrap();
    assert_eq!(sig.to_bytes(), *sig_bytes);
}

/// Parse a KAT verifying key and re-serialize it: byte-identical.
#[test]
fn verifying_key_to_bytes_matches_kat_input() {
    let pk_bytes = hex::decode(KAT0_PK).unwrap();
    let pk_array: &[u8; VERIFYING_KEY_BYTES] = pk_bytes.as_slice().try_into().unwrap();
    let vk = VerifyingKey::from_bytes(pk_array).unwrap();
    assert_eq!(vk.to_bytes(), *pk_array);
}

/// Double round-trip: parse → serialize → re-parse → serialize again.
/// Guards against any stateful drift between the two calls.
#[test]
fn signature_double_roundtrip() {
    let sm_bytes = hex::decode(KAT0_SM).unwrap();
    let sig_bytes: &[u8; SIGNATURE_BYTES] = sm_bytes[..SIGNATURE_BYTES].try_into().unwrap();
    let sig1 = Signature::from_bytes(sig_bytes).unwrap();
    let bytes1 = sig1.to_bytes();
    let sig2 = Signature::from_bytes(&bytes1).unwrap();
    let bytes2 = sig2.to_bytes();
    assert_eq!(bytes1, bytes2);
    assert_eq!(bytes1, *sig_bytes);
}

// ---------------- wire-format error paths ----------------

/// Slices shorter than `SIGNATURE_BYTES` must produce
/// `InvalidLength`, not panic or `NonCanonical`.
#[test]
fn signature_try_from_short_slice_rejected() {
    let short = [0u8; SIGNATURE_BYTES - 1];
    match Signature::try_from(&short[..]) {
        Err(SignatureError::InvalidLength {
            expected, actual, ..
        }) => {
            assert_eq!(expected, SIGNATURE_BYTES);
            assert_eq!(actual, SIGNATURE_BYTES - 1);
        }
        other => panic!("expected InvalidLength, got {:?}", other),
    }
}

/// Slices longer than `SIGNATURE_BYTES` must also be rejected.
#[test]
fn signature_try_from_long_slice_rejected() {
    let long = [0u8; SIGNATURE_BYTES + 1];
    match Signature::try_from(&long[..]) {
        Err(SignatureError::InvalidLength {
            expected, actual, ..
        }) => {
            assert_eq!(expected, SIGNATURE_BYTES);
            assert_eq!(actual, SIGNATURE_BYTES + 1);
        }
        other => panic!("expected InvalidLength, got {:?}", other),
    }
}

/// `n_bt` greater than the bound f=248 must be rejected as
/// non-canonical rather than producing an invalid `TorsionExponent`.
#[test]
fn signature_out_of_range_n_bt_rejected() {
    let sm_bytes = hex::decode(KAT0_SM).unwrap();
    let mut sig_bytes = [0u8; SIGNATURE_BYTES];
    sig_bytes.copy_from_slice(&sm_bytes[..SIGNATURE_BYTES]);
    sig_bytes[64] = 249; // n_bt > TORSION_EVEN_POWER
    assert!(matches!(
        Signature::from_bytes(&sig_bytes),
        Err(SignatureError::NonCanonical)
    ));
}

/// Same check for `r_rsp`.
#[test]
fn signature_out_of_range_r_rsp_rejected() {
    let sm_bytes = hex::decode(KAT0_SM).unwrap();
    let mut sig_bytes = [0u8; SIGNATURE_BYTES];
    sig_bytes.copy_from_slice(&sm_bytes[..SIGNATURE_BYTES]);
    sig_bytes[65] = 255; // r_rsp way out of range
    assert!(matches!(
        Signature::from_bytes(&sig_bytes),
        Err(SignatureError::NonCanonical)
    ));
}

// ------------ ChallengeMatrix::from_bytes rejection ----------

/// `ChallengeMatrix::from_bytes` must reject data shorter than
/// `4 * comp_bytes`.
#[test]
fn challenge_matrix_from_bytes_too_short() {
    let comp_bytes = E_RSP.div_ceil(8) as usize;
    let short = vec![0u8; 4 * comp_bytes - 1];
    assert!(matches!(
        ChallengeMatrix::from_bytes(&short, comp_bytes),
        Err(SignatureError::NonCanonical)
    ));
}

/// `ChallengeMatrix::from_bytes` must reject `comp_bytes` larger
/// than `TORSION_2POWER_BYTES`.
#[test]
fn challenge_matrix_from_bytes_comp_too_wide() {
    let too_wide = TORSION_2POWER_BYTES + 1;
    let data = vec![0u8; 4 * too_wide];
    assert!(matches!(
        ChallengeMatrix::from_bytes(&data, too_wide),
        Err(SignatureError::NonCanonical)
    ));
}

/// `ChallengeMatrix::from_bytes` succeeds with valid-length data.
/// This guards against the bounds check being accidentally inverted
/// (e.g., `<` mutated to `>`).
#[test]
fn challenge_matrix_from_bytes_exact_length() {
    let comp_bytes = E_RSP.div_ceil(8) as usize;
    let data = vec![0u8; 4 * comp_bytes];
    assert!(ChallengeMatrix::from_bytes(&data, comp_bytes).is_ok());
}

// Wycheproof-style verification negatives live in tests/wycheproof.rs
// (top-level integration test, public API only, portable to other impls).

// ---------------- KAT verification ----------------

/// Verify all 100 hardcoded KAT vectors from the C reference
/// implementation (PQCsignKAT_353_SQIsign_lvl1.rsp).
#[test]
fn kat_verify_all() {
    for (i, &(_seed, pk, _sk, _msg, sm)) in kat_data::KAT_VECTORS.iter().enumerate() {
        verify_kat(pk, sm);
        if i == 0 {
            // Also check the inline constants match vector 0.
            assert_eq!(pk, KAT0_PK);
            assert_eq!(sm, KAT0_SM);
        }
    }
}

/// Cross-check all KAT vectors against the C reference implementation
/// at a pinned commit.
///
/// Fetches `PQCsignKAT_353_SQIsign_lvl1.rsp` from GitHub, parses
/// every (pk, sm) entry, and verifies each signature.
///
/// Run with: `cargo test c_ref_kat_cross_check -- --ignored`
#[test]
#[ignore]
fn c_ref_kat_cross_check() {
    const COMMIT: &str = "91e9e464fe5400192d13e1f9240cbf180200a103";
    let url = format!(
        "https://raw.githubusercontent.com/SQISign/the-sqisign/{}/KAT/PQCsignKAT_353_SQIsign_lvl1.rsp",
        COMMIT,
    );

    let body = reqwest::blocking::get(&url)
        .unwrap_or_else(|e| panic!("failed to fetch {url}: {e}"))
        .text()
        .unwrap();

    let mut pk = None;
    let mut count = 0u32;
    let mut verified = 0u32;

    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(val) = line.strip_prefix("pk = ") {
            pk = Some(val.to_string());
        } else if let Some(val) = line.strip_prefix("sm = ") {
            let pk_hex = pk
                .as_ref()
                .unwrap_or_else(|| panic!("sm line before pk at count {count}"));
            verify_kat(pk_hex, val);
            verified += 1;
        } else if line.starts_with("count = ") {
            count = line.strip_prefix("count = ").unwrap().parse().unwrap();
        }
    }

    assert!(
        verified >= 10,
        "expected at least 10 KAT vectors, verified {verified}"
    );

    // Also verify our hardcoded kat_data matches the fetched file.
    let mut fetched: Vec<(String, String, String, String, String)> = Vec::new();
    let mut cur_seed = String::new();
    let mut cur_pk = String::new();
    let mut cur_sk = String::new();
    let mut cur_msg = String::new();
    for line in body.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("seed = ") {
            cur_seed = v.to_string();
        } else if let Some(v) = line.strip_prefix("pk = ") {
            cur_pk = v.to_string();
        } else if let Some(v) = line.strip_prefix("sk = ") {
            cur_sk = v.to_string();
        } else if let Some(v) = line.strip_prefix("msg = ") {
            cur_msg = v.to_string();
        } else if let Some(v) = line.strip_prefix("sm = ") {
            fetched.push((
                cur_seed.clone(),
                cur_pk.clone(),
                cur_sk.clone(),
                cur_msg.clone(),
                v.to_string(),
            ));
        }
    }

    let kat = kat_data::KAT_VECTORS;
    assert_eq!(
        fetched.len(),
        kat.len(),
        "fetched {} vectors but kat_data has {}",
        fetched.len(),
        kat.len()
    );
    for (i, (f, k)) in fetched.iter().zip(kat.iter()).enumerate() {
        assert_eq!(f.0, k.0, "vector {i}: seed mismatch");
        assert_eq!(f.1, k.1, "vector {i}: pk mismatch");
        assert_eq!(f.2, k.2, "vector {i}: sk mismatch");
        assert_eq!(f.3, k.3, "vector {i}: msg mismatch");
        assert_eq!(f.4, k.4, "vector {i}: sm mismatch");
    }
}
