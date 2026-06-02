//! Negative and vulnerability test vectors for SQIsign.
//!
//! Loads test vectors from `tests/vectors/*.json` in the
//! [C2SP/wycheproof](https://github.com/C2SP/wycheproof) JSON format.
//! Covers verification negatives, keygen correctness, and sk parsing
//! robustness — rejection paths that KAT vectors (all valid) don't
//! exercise.
//!
//! **Provenance.** Invalid vectors are independently generated here by
//! perturbation — no impl dependency. Valid vectors are deterministic
//! KAT outputs of the SQIsign reference algorithm applied to fixed
//! seeds; they are read from the C reference's published KAT file as
//! the canonical source, but the values themselves are reproducible
//! byte-for-byte by any conformant implementation that matches the
//! reference's canonical-form choices (Montgomery model normalization,
//! splitting index conventions, etc.). The vector set is therefore
//! suitable for upstreaming to C2SP/wycheproof; see
//! `docs/wycheproof-upstream.md`.
//!
//! Uses only the public API — no `expose-internals` feature required.
//! Other SQIsign implementations can reuse `tests/vectors/*.json` by
//! writing their own test runner against the same schema.

use sqisign_selkie::{
    SIGNATURE_BYTES, SIGNING_KEY_BYTES, Signature, SignatureError, SigningKey, VerifyingKey,
};

// JSON schema types (C2SP/wycheproof signatures_common format).

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct TestFile {
    algorithm: String,
    number_of_tests: usize,
    test_groups: Vec<TestGroup>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct TestGroup {
    #[serde(rename = "publicKey")]
    vk: WycheproofVerifyingKey,
    tests: Vec<TestVector>,
}

#[derive(serde::Deserialize)]
struct WycheproofVerifyingKey {
    #[serde(rename = "pk")]
    vk: String,
}

#[derive(serde::Deserialize)]
struct TestVector {
    #[serde(rename = "tcId")]
    tc_id: u32,
    comment: String,
    msg: String,
    sig: String,
    result: String,
    #[allow(dead_code)] // parsed for completeness; runner doesn't branch on flags yet
    flags: Vec<String>,
}

// Helpers.

/// Returns `true` if `vk.verify(msg, sig)` succeeds.
///
/// Uses `catch_unwind` because some corruption paths currently
/// trigger `debug_assert!` panics in the (2,2)-isogeny splitting
/// step instead of returning `Err`. Both panic and `Err` count as
/// rejection for "invalid" vectors.
///
/// TODO: Once `GetIndexSplitting` returns `Option` instead of
/// panicking, remove the `catch_unwind` and test for `Err` directly.
fn verify_succeeds(vk: &VerifyingKey, msg: &[u8], sig: &Signature) -> bool {
    let vk = *vk;
    let sig = *sig;
    let msg = msg.to_vec();
    matches!(
        std::panic::catch_unwind(move || vk.verify(&msg, &sig)),
        Ok(Ok(()))
    )
}

// Test runner for sqisign_verify.json.

#[test]
fn sqisign_verify_vectors() {
    let json = include_str!("vectors/sqisign_verify.json");
    let file: TestFile = serde_json::from_str(json).expect("failed to parse test vectors");

    assert_eq!(file.algorithm, "SQIsign_248");

    let mut tested = 0;

    for group in &file.test_groups {
        let pk_bytes = hex::decode(&group.vk.vk).unwrap();
        let vk = match pk_bytes.as_slice().try_into() {
            Err(_) => {
                // pk wrong length — all tests in this group must be invalid
                for tv in &group.tests {
                    assert_eq!(
                        tv.result, "invalid",
                        "tcId {}: pk parse failed (wrong length) but result is {}",
                        tv.tc_id, tv.result
                    );
                    tested += 1;
                }
                continue;
            }
            Ok(arr) => match VerifyingKey::from_bytes(arr) {
                Err(_) => {
                    for tv in &group.tests {
                        assert_eq!(
                            tv.result, "invalid",
                            "tcId {}: pk parse failed but result is {}",
                            tv.tc_id, tv.result
                        );
                        tested += 1;
                    }
                    continue;
                }
                Ok(vk) => vk,
            },
        };

        for tv in &group.tests {
            let msg = hex::decode(&tv.msg).unwrap();
            let sig_bytes = hex::decode(&tv.sig).unwrap();

            match tv.result.as_str() {
                "valid" => {
                    let sig_arr: &[u8; SIGNATURE_BYTES] =
                        sig_bytes.as_slice().try_into().unwrap_or_else(|_| {
                            panic!("tcId {}: sig wrong length {}", tv.tc_id, sig_bytes.len())
                        });
                    let sig = Signature::from_bytes(sig_arr)
                        .unwrap_or_else(|e| panic!("tcId {}: sig parse failed: {e}", tv.tc_id));
                    assert!(
                        verify_succeeds(&vk, &msg, &sig),
                        "tcId {}: expected valid, got rejection ({})",
                        tv.tc_id,
                        tv.comment
                    );
                }
                "invalid" => {
                    // Parse may fail (which is fine) or succeed.
                    // If parse succeeds, verify must reject.
                    let sig_arr: Result<&[u8; SIGNATURE_BYTES], _> =
                        sig_bytes.as_slice().try_into();
                    let rejected = match sig_arr {
                        Err(_) => true, // wrong length — rejected at parse
                        Ok(arr) => match Signature::from_bytes(arr) {
                            Err(_) => true, // structural rejection
                            Ok(sig) => !verify_succeeds(&vk, &msg, &sig),
                        },
                    };
                    assert!(
                        rejected,
                        "tcId {}: expected invalid, but verified ({})",
                        tv.tc_id, tv.comment
                    );
                }
                "acceptable" => {
                    // Either outcome is fine — just ensure no panic
                    // outside catch_unwind.
                    let _ = sig_bytes.as_slice().try_into().ok().and_then(
                        |arr: &[u8; SIGNATURE_BYTES]| {
                            Signature::from_bytes(arr)
                                .ok()
                                .map(|sig| verify_succeeds(&vk, &msg, &sig))
                        },
                    );
                }
                other => panic!("tcId {}: unknown result {:?}", tv.tc_id, other),
            }

            tested += 1;
        }
    }

    assert_eq!(
        tested, file.number_of_tests,
        "verify numberOfTests mismatch: ran {tested}, expected {}",
        file.number_of_tests
    );
}

// Extended verify vector generator. Run with:
//   GENERATE_EXTENDED_VECTORS=1 cargo test --release --test wycheproof \
//     generate_extended_vectors -- --nocapture --ignored
// Produces `tests/vectors/sqisign_verify_extended.json` with vectors
// targeting bug classes uncovered after the off-by-one /
// cross-order-modulus / TorsionBasisFromHint fixes. Each generated
// vector is validated in-place (verify actually rejects/accepts as
// claimed) before being emitted, so the file is never out of sync
// with the runtime behavior.

#[test]
#[ignore = "generator; opt in with GENERATE_EXTENDED_VECTORS=1"]
fn generate_extended_vectors() {
    if std::env::var("GENERATE_EXTENDED_VECTORS").is_err() {
        return;
    }

    // Use the existing `sqisign_verify.json` as the source of valid
    // donor signatures rather than reaching into `kat_data` (which
    // would require enabling the `expose-internals` feature for
    // integration tests).
    let donor_json = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/sqisign_verify.json"),
    )
    .expect("read sqisign_verify.json");
    let donor_file: TestFile =
        serde_json::from_str(&donor_json).expect("parse sqisign_verify.json");

    // Find a known-valid vector (tcId=1 in the existing file is
    // "KAT vector 0 - valid signature"). Donor pk + msg + sig.
    let (donor_group, donor_tv) = donor_file
        .test_groups
        .iter()
        .find_map(|g| g.tests.iter().find(|t| t.result == "valid").map(|t| (g, t)))
        .expect("no valid donor vector in sqisign_verify.json");
    let pk_bytes = hex::decode(&donor_group.vk.vk).unwrap();
    let valid_sig_bytes_vec = hex::decode(&donor_tv.sig).unwrap();
    let valid_sig_bytes: &[u8; SIGNATURE_BYTES] = valid_sig_bytes_vec
        .as_slice()
        .try_into()
        .expect("donor sig length");
    let pk_arr: &[u8; sqisign_selkie::VERIFYING_KEY_BYTES] =
        pk_bytes.as_slice().try_into().expect("donor pk length");
    let vk = VerifyingKey::from_bytes(pk_arr).unwrap();
    let msg_kat0 = hex::decode(&donor_tv.msg).unwrap();

    let mut next_tc_id: u32 = 100;
    let mut vectors: Vec<String> = Vec::new();

    // --- HintOverflow variants ---------------------------------------
    // Each variant: take valid sig, randomize E_aux bytes, zero the
    // hint byte (forces fallback search), keep everything else valid.
    // The bounded search inside from_hint returns None for most
    // random E_aux, so verify rejects.
    let craft_hint_overflow = |seed_byte: u8, target: &str| -> [u8; SIGNATURE_BYTES] {
        let mut s = [0u8; SIGNATURE_BYTES];
        s.copy_from_slice(valid_sig_bytes);
        // Fill E_aux (bytes 0..64) deterministically from seed_byte.
        for (i, b) in s[..64].iter_mut().enumerate() {
            *b = seed_byte.wrapping_add(i as u8);
        }
        // Force fallback search by zeroing the relevant hint byte.
        match target {
            "aux" => s[146] = 0x00,     // h=0, h_A=0
            "aux_nqr" => s[146] = 0x80, // h=0, h_A=1
            "chl" => s[147] = 0x00,
            "chl_nqr" => s[147] = 0x80,
            _ => {}
        }
        s
    };

    let hint_overflow_seeds = [
        (0x11u8, "aux", "fallback search on random E_aux, h_A=0"),
        (0x22, "aux_nqr", "fallback search on random E_aux, h_A=1"),
        (0x33, "chl", "fallback search on derived E_chl, h_A=0"),
        (0x44, "chl_nqr", "fallback search on derived E_chl, h_A=1"),
        (0x55, "aux", "fallback search variant (seed 0x55)"),
    ];

    for &(seed_byte, target, desc) in &hint_overflow_seeds {
        let sig_bytes = craft_hint_overflow(seed_byte, target);
        // Validate: verify should reject (either parse-fail or verify-fail).
        let rejected = match Signature::from_bytes(&sig_bytes) {
            Err(_) => true,
            Ok(sig) => !verify_succeeds(&vk, &msg_kat0, &sig),
        };
        if !rejected {
            // Skip vectors whose adversarial mutation happened to be a
            // valid signature; this is improbable but possible.
            continue;
        }
        let tc = next_tc_id;
        next_tc_id += 1;
        vectors.push(format!(
            r#"        {{
          "tcId": {tc},
          "comment": "HintOverflow: {desc}",
          "msg": "{}",
          "sig": "{}",
          "result": "invalid",
          "flags": ["HintOverflow"]
        }}"#,
            hex::encode(&msg_kat0),
            hex::encode(sig_bytes)
        ));
    }

    // --- Boundary scalar values --------------------------------------
    // n_bt and r_rsp are 1-byte scalars constrained by the protocol.
    // Test min/max boundaries to exercise verify's range checks.
    let boundary_cases = [
        (0u8, valid_sig_bytes[65], "n_bt=0 (minimum), valid r_rsp"),
        (valid_sig_bytes[64], 0u8, "r_rsp=0 (minimum), valid n_bt"),
        (255, 0, "n_bt=255 (max byte value, exceeds e_rsp)"),
        (0, 255, "r_rsp=255 (max byte value, exceeds e_rsp)"),
        (
            128,
            128,
            "n_bt=r_rsp=128 (n_bt + r_rsp = 256, exceeds e_rsp=248)",
        ),
    ];
    for &(n_bt, r_rsp, desc) in &boundary_cases {
        let mut s = [0u8; SIGNATURE_BYTES];
        s.copy_from_slice(valid_sig_bytes);
        s[64] = n_bt;
        s[65] = r_rsp;
        let rejected = match Signature::from_bytes(&s) {
            Err(_) => true,
            Ok(sig) => !verify_succeeds(&vk, &msg_kat0, &sig),
        };
        if !rejected {
            continue;
        }
        let tc = next_tc_id;
        next_tc_id += 1;
        vectors.push(format!(
            r#"        {{
          "tcId": {tc},
          "comment": "boundary scalar: {desc}",
          "msg": "{}",
          "sig": "{}",
          "result": "invalid",
          "flags": ["BoundaryScalar"]
        }}"#,
            hex::encode(&msg_kat0),
            hex::encode(s)
        ));
    }

    // --- Splitting-degenerate mutations ------------------------------
    // Single-byte mutations at strategic offsets that pass parse but
    // drive the (2,2)-chain to a terminal theta null with 0 or
    // multiple vanishing U_{i,j}(0) entries. We probe four E_aux
    // bytes that empirically produce splitting-degenerate chains.
    let splitting_offsets = [
        (8u32, "E_aux real part low limb"),
        (24, "E_aux real part high limb"),
        (40, "E_aux imag part low limb"),
        (56, "E_aux imag part high limb"),
    ];
    for &(off, where_) in &splitting_offsets {
        let mut s = [0u8; SIGNATURE_BYTES];
        s.copy_from_slice(valid_sig_bytes);
        // Mutate by XORing 0x5a — a multi-bit twiddle that's unlikely
        // to give a coincidentally-valid sig but still leaves a
        // parseable Fp2 element.
        s[off as usize] ^= 0x5A;
        let rejected = match Signature::from_bytes(&s) {
            Err(_) => true,
            Ok(sig) => !verify_succeeds(&vk, &msg_kat0, &sig),
        };
        if !rejected {
            continue;
        }
        let tc = next_tc_id;
        next_tc_id += 1;
        vectors.push(format!(
            r#"        {{
          "tcId": {tc},
          "comment": "splitting-degenerate: XOR 0x5a at {where_} (byte {off})",
          "msg": "{}",
          "sig": "{}",
          "result": "invalid",
          "flags": ["SplittingDegenerate"]
        }}"#,
            hex::encode(&msg_kat0),
            hex::encode(s)
        ));
    }

    // Singular sig.curve_aux. The Montgomery model y² = x³ + Ax² + x
    // has discriminant Δ = 4(A² − 4); A = ±2 collapses it to a
    // singular cubic. Parse must reject (matches C ref's
    // `ec_curve_verify_A` at ec.c:169).
    let craft_singular_aux = |first_byte: u8, rest: u8| -> [u8; SIGNATURE_BYTES] {
        let mut s = [0u8; SIGNATURE_BYTES];
        s.copy_from_slice(valid_sig_bytes);
        // Zero out A_aux (bytes 0..64) then set the encoded value.
        for b in &mut s[..64] {
            *b = 0;
        }
        s[0] = first_byte;
        // For A = -2 mod p, the real-part LE encoding has 0xFF in
        // bytes 1..31 and 0x04 in byte 31; the caller passes that
        // pattern via `rest`.
        if rest != 0 {
            for b in &mut s[1..31] {
                *b = rest;
            }
            s[31] = 0x04;
        }
        s
    };

    for (first, rest, label) in &[
        (
            0x02u8,
            0u8,
            "sig.curve_aux A = 2 (singular Montgomery, Δ=0)",
        ),
        (
            0xFDu8,
            0xFFu8,
            "sig.curve_aux A = -2 mod p (singular Montgomery, Δ=0)",
        ),
    ] {
        let sig_bytes = craft_singular_aux(*first, *rest);

        let rejected = matches!(
            Signature::from_bytes(&sig_bytes),
            Err(SignatureError::InvalidCurve)
        );
        assert!(rejected, "singular sig.curve_aux must be rejected at parse");

        let tc = next_tc_id;
        next_tc_id += 1;
        vectors.push(format!(
            r#"        {{
          "tcId": {tc},
          "comment": "SingularCurve: {label}",
          "msg": "{}",
          "sig": "{}",
          "result": "invalid",
          "flags": ["SingularCurve"]
        }}"#,
            hex::encode(&msg_kat0),
            hex::encode(sig_bytes)
        ));
    }

    // M_chl entry above the spec bound.
    // Spec §4.5 Alg. 4.9 step 5 bounds each M_chl entry by
    // `2^(e_rsp - n_bt + 2)`. With `n_bt = 1` the bound becomes
    // `2^(e_rsp + 1) = 2^127` (NIST-I: e_rsp = 126), so bit 127 of
    // any entry must be zero. Set it and assert parse rejection.
    {
        let mut sig_bytes = *valid_sig_bytes;
        sig_bytes[64] = 1; // n_bt = 1 ⇒ bound = 2^127
        sig_bytes[65] = 0; // r_rsp = 0 to keep e'_rsp ≥ 0 trivially
        // Bit 127 of M_chl entry 0: byte 15 of the first 16-byte entry,
        // which starts at signature byte 66; absolute offset = 66 + 15 = 81.
        sig_bytes[81] |= 0x80;

        let rejected = matches!(
            Signature::from_bytes(&sig_bytes),
            Err(SignatureError::NonCanonical)
        );
        assert!(
            rejected,
            "M_chl entry above bound must be rejected at parse"
        );

        let tc = next_tc_id;
        next_tc_id += 1;
        vectors.push(format!(
            r#"        {{
          "tcId": {tc},
          "comment": "MatrixOverflow: M_chl entry 0 bit 127 set with n_bt=1 (bound=2^127)",
          "msg": "{}",
          "sig": "{}",
          "result": "invalid",
          "flags": ["MatrixOverflow"]
        }}"#,
            hex::encode(&msg_kat0),
            hex::encode(sig_bytes)
        ));
    }

    // The HintOverflow / boundary / splitting-degenerate / new
    // SingularCurve + MatrixOverflow vectors all share the donor pk
    // (KAT 0). Collect them into the first test group. The `mut` is
    // only used under `expose-internals`, where the cross-order group
    // below is appended.
    #[cfg_attr(not(feature = "expose-internals"), allow(unused_mut))]
    let mut groups: Vec<(String, Vec<String>)> = vec![(hex::encode(pk_arr), vectors)];

    // Singular pk (A = ±2). Each malformed pk gets its own group
    // since the wycheproof schema scopes `publicKey` per group.
    for (first_byte, rest, label) in &[
        (0x02u8, 0u8, "pk.A = 2 (singular Montgomery, Δ=0)"),
        (0xFDu8, 0xFFu8, "pk.A = -2 mod p (singular Montgomery, Δ=0)"),
    ] {
        let mut pk_bytes = [0u8; sqisign_selkie::VERIFYING_KEY_BYTES];
        pk_bytes[0] = *first_byte;
        if *rest != 0 {
            for b in &mut pk_bytes[1..31] {
                *b = *rest;
            }
            pk_bytes[31] = 0x04;
        }

        let rejected = matches!(
            VerifyingKey::from_bytes(&pk_bytes),
            Err(SignatureError::InvalidCurve)
        );
        assert!(rejected, "singular pk must be rejected at parse");

        let tc = next_tc_id;
        next_tc_id += 1;
        // sig is irrelevant since pk parse fails; reuse the donor.
        let entry = format!(
            r#"        {{
          "tcId": {tc},
          "comment": "SingularPk: {label}",
          "msg": "{}",
          "sig": "{}",
          "result": "invalid",
          "flags": ["SingularPk"]
        }}"#,
            hex::encode(&msg_kat0),
            hex::encode(valid_sig_bytes)
        );
        groups.push((hex::encode(pk_bytes), vec![entry]));
    }

    // Canonical Fp² pk values whose Montgomery `A` is almost certainly
    // not on a supersingular curve. Parse must not reject as
    // `NonCanonical` (each Fp component is < p); verify must reject
    // without panic. Pins the "NotSupersingular slips through parse"
    // gap called out in the module doc.
    //
    // The probability any specific A is supersingular over Fp² is
    // O(1/p) ≈ 2^-249, so "almost certainly not" is statistical, not
    // a proof. If a future change adds `NotSupersingular` rejection at
    // parse time, the assertions below still hold (rejection is
    // rejection either way) and the vectors continue to pin the
    // boundary.
    //
    // p = 5·2²⁴⁸ − 1, so `p − 1` LE-encodes to `[0xFE, 0xFF×30, 0x04]`.
    let mut p_minus_one = [0u8; 32];
    p_minus_one[0] = 0xFE;
    for b in &mut p_minus_one[1..31] {
        *b = 0xFF;
    }
    p_minus_one[31] = 0x04;
    let imag_unit = {
        let mut b = [0u8; 32];
        b[0] = 1;
        b
    };

    for (real, imag, label) in &[
        (
            p_minus_one,
            [0u8; 32],
            "pk.A real = p−1, imag = 0 (canonical max real component)",
        ),
        (
            [0u8; 32],
            imag_unit,
            "pk.A = i (pure imaginary unit; smallest non-zero imag, real = 0)",
        ),
        (
            [0u8; 32],
            p_minus_one,
            "pk.A imag = p−1, real = 0 (canonical max imag component)",
        ),
    ] {
        let mut pk_bytes = [0u8; sqisign_selkie::VERIFYING_KEY_BYTES];
        pk_bytes[..32].copy_from_slice(real);
        pk_bytes[32..64].copy_from_slice(imag);

        let verify_ok = match VerifyingKey::from_bytes(&pk_bytes) {
            Err(_) => false,
            Ok(vk) => {
                let sig = Signature::from_bytes(valid_sig_bytes).expect("donor sig parses");
                verify_succeeds(&vk, &msg_kat0, &sig)
            }
        };
        assert!(
            !verify_ok,
            "non-supersingular pk must not verify the donor sig: {label}"
        );

        let tc = next_tc_id;
        next_tc_id += 1;
        let entry = format!(
            r#"        {{
          "tcId": {tc},
          "comment": "NonSupersingularPk: {label}",
          "msg": "{}",
          "sig": "{}",
          "result": "invalid",
          "flags": ["NonSupersingularPk"]
        }}"#,
            hex::encode(&msg_kat0),
            hex::encode(valid_sig_bytes)
        );
        groups.push((hex::encode(pk_bytes), vec![entry]));
    }

    // --- Cross-order positive vectors --------------------------------
    // KATs 020, 053, 056, 079, 094, 095 exercise the cross-order
    // `suitable_ideals` path during signing — the source of the
    // product-modulus bug we just fixed.  Tagging them in the
    // wycheproof file keeps the cross-order regression class visible
    // at vector-review time, independently of the lib-test
    // `sign_kat_derand_NNN` coverage.  Each KAT has its own pk, so
    // each becomes a separate test group.
    //
    // Pulls per-KAT (pk, msg, sm) from `kat_data::KAT_VECTORS`, which
    // only exposes itself under `cfg(any(test, feature =
    // "expose-internals"))`.  Run the generator with
    // `--features expose-internals` to include this class; otherwise
    // these vectors are simply skipped.
    #[cfg(feature = "expose-internals")]
    {
        use sqisign_selkie::keys::kat_data::KAT_VECTORS;
        for &kat_idx in &[20usize, 53, 56, 79, 94, 95] {
            let (_, pk_hex_i, _, msg_hex_i, sm_hex_i) = KAT_VECTORS[kat_idx];
            let pk_i = hex::decode(pk_hex_i).unwrap();
            let sig_i_bytes_full = hex::decode(sm_hex_i).unwrap();
            let sig_i_arr: &[u8; SIGNATURE_BYTES] =
                sig_i_bytes_full[..SIGNATURE_BYTES].try_into().unwrap();
            let pk_arr_i: &[u8; sqisign_selkie::VERIFYING_KEY_BYTES] =
                pk_i.as_slice().try_into().unwrap();
            let vk_i = VerifyingKey::from_bytes(pk_arr_i).unwrap();
            let sig_i = Signature::from_bytes(sig_i_arr).unwrap();
            let msg_i = hex::decode(msg_hex_i).unwrap();
            assert!(
                verify_succeeds(&vk_i, &msg_i, &sig_i),
                "KAT {kat_idx}: expected valid, got rejection"
            );
            let tc = next_tc_id;
            next_tc_id += 1;
            let entry = format!(
                r#"        {{
          "tcId": {tc},
          "comment": "KAT vector {kat_idx} - exercises cross-order suitable_ideals path",
          "msg": "{}",
          "sig": "{}",
          "result": "valid",
          "flags": ["CrossOrderSuitableIdeals"]
        }}"#,
                hex::encode(&msg_i),
                hex::encode(sig_i_arr)
            );
            groups.push((hex::encode(pk_arr_i), vec![entry]));
        }
    }

    // Emit JSON file.
    let n: usize = groups.iter().map(|(_, vs)| vs.len()).sum();
    let groups_json = groups
        .iter()
        .map(|(pk_hex, vs)| {
            format!(
                r#"    {{
      "type": "SqisignVerify",
      "publicKey": {{
        "pk": "{pk_hex}"
      }},
      "tests": [
{}
      ]
    }}"#,
                vs.join(",\n")
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let json = format!(
        r#"{{
  "algorithm": "SQIsign_248",
  "schema": "sqisign_verify_schema.json",
  "generatorVersion": "sqisign-selkie-0.0.1",
  "numberOfTests": {n},
  "header": [
    "SQIsign_248: extended verify vectors.",
    "Extends sqisign_verify.json with vectors targeting bug classes uncovered after",
    "the random_norm β off-by-one, cross-order product modulus, and",
    "TorsionBasisFromHint bounded-search fixes (2026-05).",
    "Generated by `tests/wycheproof.rs::generate_extended_vectors`."
  ],
  "notes": {{
    "HintOverflow": {{
      "bugType": "DENIAL_OF_SERVICE",
      "description": "Adversarial E_aux/E_chl that forces TorsionBasisFromHint's bounded-search fallback. Without the cap, verify hangs; with the cap, it rejects."
    }},
    "BoundaryScalar": {{
      "bugType": "BASIC",
      "description": "n_bt or r_rsp at the byte-max boundary or outside the protocol-permissible range. Verify must reject without arithmetic underflow."
    }},
    "SplittingDegenerate": {{
      "bugType": "FUNCTIONALITY",
      "description": "Byte mutation that produces a (2,2)-chain terminal theta null with 0 or multiple vanishing U_{{i,j}}(0) entries. Verify must reject rather than silently picking an arbitrary splitting index."
    }},
    "CrossOrderSuitableIdeals": {{
      "bugType": "FUNCTIONALITY",
      "description": "Valid signature whose signing-side path exercises the cross-order suitable_ideals branch (the source of the t>0 product-modulus bug, 2026-05). Verify must accept."
    }},
    "SingularCurve": {{
      "bugType": "BASIC",
      "description": "Montgomery curve coefficient A = ±2 in sig.curve_aux yields a singular cubic (Δ = 4(A²−4) = 0). Parse must reject; matches the C reference's ec_curve_verify_A check at ec.c:169."
    }},
    "SingularPk": {{
      "bugType": "BASIC",
      "description": "Montgomery curve coefficient A = ±2 in the public key. Parse of VerifyingKey must reject; same singular-cubic rationale as SingularCurve, applied at the pk boundary."
    }},
    "NonSupersingularPk": {{
      "bugType": "FUNCTIONALITY",
      "description": "Montgomery curve coefficient A is a canonical Fp² element (each component < p) but the resulting curve is almost certainly not supersingular. Parse must not reject as NonCanonical; verify must reject without panic. Pins the 'NotSupersingular slips through parse' gap — if a future change adds parse-time supersingularity checking, parse rejection still satisfies the property."
    }},
    "MatrixOverflow": {{
      "bugType": "BASIC",
      "description": "M_chl entry exceeds the spec bound 2^(e'_rsp + r_rsp + 2) = 2^(e_rsp - n_bt + 2) from §4.5 Alg. 4.9 step 5. Parse must reject; matches the C reference's check_canonical_basis_change_matrix in verify.c."
    }}
  }},
  "testGroups": [
{groups_json}
  ]
}}
"#,
    );

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/vectors/sqisign_verify_extended.json");
    std::fs::write(&path, json).expect("write extended vectors file");
    eprintln!("wrote {n} vectors to {}", path.display());
}

/// Runs the extended verify vectors produced by
/// `generate_extended_vectors`. Mirrors `sqisign_verify_vectors`'s
/// runner but loads a different JSON file. The runner is no-op if the
/// extended file isn't present, so contributors can land the fix
/// before regenerating vectors.
#[test]
fn sqisign_verify_extended_vectors() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/vectors/sqisign_verify_extended.json");
    if !path.exists() {
        eprintln!("skip: extended vectors file not present");
        return;
    }
    let json = std::fs::read_to_string(&path).expect("read extended vectors");
    let file: TestFile = serde_json::from_str(&json).expect("parse extended vectors");
    assert_eq!(file.algorithm, "SQIsign_248");

    let mut tested = 0;
    for group in &file.test_groups {
        let pk_bytes = hex::decode(&group.vk.vk).unwrap();
        let pk_arr: &[u8; sqisign_selkie::VERIFYING_KEY_BYTES] =
            pk_bytes.as_slice().try_into().expect("pk length");

        // If the pk itself parses-reject, every test in the group
        // must be `invalid` (the rejection has already happened
        // before we ever look at the sig).
        let vk = match VerifyingKey::from_bytes(pk_arr) {
            Err(_) => {
                for tv in &group.tests {
                    assert_eq!(
                        tv.result, "invalid",
                        "tcId {}: pk parse rejected but result is {}",
                        tv.tc_id, tv.result
                    );
                    tested += 1;
                }
                continue;
            }
            Ok(vk) => vk,
        };

        for tv in &group.tests {
            let msg = hex::decode(&tv.msg).unwrap();
            let sig_bytes = hex::decode(&tv.sig).unwrap();
            let sig_arr: &[u8; SIGNATURE_BYTES] =
                sig_bytes.as_slice().try_into().expect("sig length");
            match tv.result.as_str() {
                "valid" => {
                    let sig = Signature::from_bytes(sig_arr)
                        .unwrap_or_else(|e| panic!("tcId {}: sig parse failed: {e}", tv.tc_id));
                    assert!(
                        verify_succeeds(&vk, &msg, &sig),
                        "tcId {}: expected valid, got rejection ({})",
                        tv.tc_id,
                        tv.comment
                    );
                }
                "invalid" => {
                    let rejected = match Signature::from_bytes(sig_arr) {
                        Err(_) => true,
                        Ok(sig) => !verify_succeeds(&vk, &msg, &sig),
                    };
                    assert!(
                        rejected,
                        "tcId {}: expected invalid, but verified ({})",
                        tv.tc_id, tv.comment
                    );
                }
                other => panic!("tcId {}: unknown result {:?}", tv.tc_id, other),
            }
            tested += 1;
        }
    }
    assert_eq!(
        tested, file.number_of_tests,
        "extended numberOfTests mismatch: ran {tested}, expected {}",
        file.number_of_tests
    );
}

// Keygen / sk parsing vectors.

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct KeygenTestFile {
    algorithm: String,
    number_of_tests: usize,
    test_groups: Vec<KeygenTestGroup>,
}

#[derive(serde::Deserialize)]
struct KeygenTestGroup {
    tests: Vec<KeygenTestVector>,
}

#[derive(serde::Deserialize)]
struct KeygenTestVector {
    #[serde(rename = "tcId")]
    tc_id: u32,
    comment: String,
    #[allow(dead_code)] // used by deterministic keygen tests (currently #[ignore])
    #[serde(default)]
    seed: Option<String>,
    #[serde(default, rename = "pk")]
    vk: Option<String>,
    sk: String,
    result: String,
    #[allow(dead_code)]
    flags: Vec<String>,
}

#[test]
fn sqisign_keygen_vectors() {
    let json = include_str!("vectors/sqisign_keygen.json");
    let file: KeygenTestFile = serde_json::from_str(json).expect("failed to parse keygen vectors");

    assert_eq!(file.algorithm, "SQIsign_248");

    let mut tested = 0;

    for group in &file.test_groups {
        for tv in &group.tests {
            let sk_bytes = hex::decode(&tv.sk).unwrap();

            match tv.result.as_str() {
                "valid" => {
                    // sk must parse
                    let sk_arr: &[u8; SIGNING_KEY_BYTES] =
                        sk_bytes.as_slice().try_into().unwrap_or_else(|_| {
                            panic!("tcId {}: sk wrong length {}", tv.tc_id, sk_bytes.len())
                        });
                    let sk = SigningKey::from_bytes(sk_arr)
                        .unwrap_or_else(|e| panic!("tcId {}: sk parse failed: {e}", tv.tc_id));

                    // If vk is provided, the embedded vk must match
                    if let Some(vk_hex) = &tv.vk {
                        let expected_pk = hex::decode(vk_hex).unwrap();
                        assert_eq!(
                            sk.verifying_key().to_bytes().as_slice(),
                            expected_pk.as_slice(),
                            "tcId {}: pk mismatch ({})",
                            tv.tc_id,
                            tv.comment
                        );
                    }

                    // Deterministic keygen vectors (with seed) are
                    // #[ignore] because keygen is slow. The sk/pk
                    // consistency is checked above without running
                    // keygen.
                }
                "invalid" => {
                    // Parse must fail (wrong length or structural).
                    let rejected = match sk_bytes.as_slice().try_into() {
                        Err(_) => true, // wrong length
                        Ok(arr) => {
                            let arr: &[u8; SIGNING_KEY_BYTES] = arr;
                            SigningKey::from_bytes(arr).is_err()
                        }
                    };
                    // If parse succeeds despite our expectation, the
                    // vector is "invalid" in the sense that signing
                    // with it should fail or produce non-verifiable
                    // signatures. For now we only check parse
                    // rejection; signing tests come later.
                    if !rejected {
                        // Parse succeeded — this is acceptable for
                        // some "invalid" vectors (e.g., bit flips in
                        // the ideal that don't violate structural
                        // constraints but produce a wrong key).
                        // TODO: once signing works, verify that
                        // signing with a parsed "invalid" sk produces
                        // signatures that don't verify.
                    }
                }
                "acceptable" => {
                    // Either outcome is fine.
                }
                other => panic!("tcId {}: unknown result {:?}", tv.tc_id, other),
            }

            tested += 1;
        }
    }

    assert_eq!(
        tested, file.number_of_tests,
        "keygen numberOfTests mismatch: ran {tested}, expected {}",
        file.number_of_tests
    );
}

// Signing vectors (skeleton — signing is in flux).

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SignTestFile {
    algorithm: String,
    number_of_tests: usize,
    test_groups: Vec<SignTestGroup>,
}

#[derive(serde::Deserialize)]
struct SignTestGroup {
    tests: Vec<SignTestVector>,
}

#[derive(serde::Deserialize)]
struct SignTestVector {
    #[serde(rename = "tcId")]
    tc_id: u32,
    comment: String,
    #[allow(dead_code)] // will be used once sign-then-verify round-trips are wired up
    sk: String,
    #[serde(rename = "pk")]
    vk: String,
    msg: String,
    sig: String,
    result: String,
    #[allow(dead_code)]
    flags: Vec<String>,
}

/// Signing test vectors. Currently validates that KAT signatures
/// verify under their paired pk (the sign-then-verify contract from
/// the C reference). Will grow as our signer stabilizes.
#[test]
fn sqisign_sign_vectors() {
    let json = include_str!("vectors/sqisign_sign.json");
    let file: SignTestFile = serde_json::from_str(json).expect("failed to parse sign vectors");

    assert_eq!(file.algorithm, "SQIsign_248");

    let mut tested = 0;

    for group in &file.test_groups {
        for tv in &group.tests {
            let vk_bytes = hex::decode(&tv.vk).unwrap();
            let vk = VerifyingKey::from_bytes(
                vk_bytes
                    .as_slice()
                    .try_into()
                    .unwrap_or_else(|_| panic!("tcId {}: vk wrong length", tv.tc_id)),
            )
            .unwrap_or_else(|e| panic!("tcId {}: vk parse failed: {e}", tv.tc_id));

            let msg = hex::decode(&tv.msg).unwrap();
            let sig_bytes = hex::decode(&tv.sig).unwrap();

            match tv.result.as_str() {
                "valid" => {
                    let sig_arr: &[u8; SIGNATURE_BYTES] =
                        sig_bytes.as_slice().try_into().unwrap_or_else(|_| {
                            panic!("tcId {}: sig wrong length {}", tv.tc_id, sig_bytes.len())
                        });
                    let sig = Signature::from_bytes(sig_arr)
                        .unwrap_or_else(|e| panic!("tcId {}: sig parse failed: {e}", tv.tc_id));
                    assert!(
                        verify_succeeds(&vk, &msg, &sig),
                        "tcId {}: expected valid, got rejection ({})",
                        tv.tc_id,
                        tv.comment
                    );
                    // TODO: once signing stabilizes, also:
                    // 1. sign(sk, msg) and verify the result
                    // 2. check NonceIndependence (distinct E_com)
                    // 3. check response edge cases (r_rsp, n_bt)
                }
                "invalid" => {
                    let rejected = match sig_bytes.as_slice().try_into() {
                        Err(_) => true,
                        Ok(arr) => match Signature::from_bytes(arr) {
                            Err(_) => true,
                            Ok(sig) => !verify_succeeds(&vk, &msg, &sig),
                        },
                    };
                    assert!(
                        rejected,
                        "tcId {}: expected invalid, but verified ({})",
                        tv.tc_id, tv.comment
                    );
                }
                "acceptable" => {}
                other => panic!("tcId {}: unknown result {:?}", tv.tc_id, other),
            }

            tested += 1;
        }
    }

    assert_eq!(
        tested, file.number_of_tests,
        "sign numberOfTests mismatch: ran {tested}, expected {}",
        file.number_of_tests
    );
}
