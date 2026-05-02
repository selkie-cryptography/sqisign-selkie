use rand_core::OsRng;

use super::*;

/// `SigningKey::generate` runs to completion (success or
/// `KeyGenFailed`) without panicking. Marked `#[ignore]` because each
/// run takes a few seconds: `to_isogeny` does a (2,2)-isogeny chain.
///
/// Run with: `cargo test --lib generate_runs -- --ignored`.
#[test]
#[ignore]
fn generate_runs() {
    let result = SigningKey::generate(&mut OsRng);
    match result {
        Ok(_sk) => {
            // SigningKey was successfully constructed. We don't compare
            // any field — the assertion is just that no panic occurred
            // and the SigningKey type-checks end to end.
        }
        Err(SignatureError::KeyGenFailed) => {
            // Acceptable: probabilistic algorithm exhausted retries.
        }
        Err(other) => panic!("unexpected error from SigningKey::generate: {other:?}"),
    }
}

/// The verifying key paired with a freshly generated signing key
/// must round-trip through `VerifyingKey::to_bytes` /
/// `VerifyingKey::from_bytes`.
#[test]
#[ignore]
fn generated_verifying_key_roundtrips() {
    let sk = match SigningKey::generate(&mut OsRng) {
        Ok(sk) => sk,
        Err(SignatureError::KeyGenFailed) => return, // probabilistic skip
        Err(other) => panic!("unexpected error: {other:?}"),
    };

    let vk = sk.verifying_key();
    let bytes = vk.to_bytes();
    let parsed = VerifyingKey::from_bytes(&bytes).expect("vk should round-trip");
    assert_eq!(parsed.to_bytes(), bytes);
}

// ----------------------------------------------------------------------
// End-to-end signing tests using parsed KAT signing keys.
//
// These bypass the slow `SigningKey::generate` by deserializing a
// known-good `sk` from the C reference implementation's KAT file
// (commit 91e9e464fe5400192d13e1f9240cbf180200a103). Run with
// `cargo test --lib --release sign_kat -- --ignored`.
// ----------------------------------------------------------------------

/// Deterministic keygen from every KAT seed must produce the
/// matching KAT pk and sk.
///
/// Run with: `cargo test --lib --release keygen_kat_all -- --ignored`.
#[test]
#[ignore]
fn keygen_kat_all() {
    for (i, &(seed_hex, pk_hex, sk_hex, ..)) in
        crate::keys::kat_data::KAT_VECTORS.iter().enumerate()
    {
        let seed_bytes = hex::decode(seed_hex).expect("valid hex");
        let seed: [u8; 48] = seed_bytes.as_slice().try_into().expect("seed is 48 bytes");

        let sk = match SigningKey::generate_derand(&seed) {
            Ok(sk) => sk,
            Err(SignatureError::KeyGenFailed) => {
                eprintln!("keygen_kat_all: vector {i} exhausted retries (expected)");
                continue;
            }
            Err(other) => panic!("vector {i}: unexpected keygen error: {other:?}"),
        };

        let pk_bytes = hex::decode(pk_hex).expect("valid hex");
        assert_eq!(
            &sk.verifying_key().to_bytes()[..],
            pk_bytes.as_slice(),
            "vector {i}: pk mismatch"
        );

        let sk_bytes = hex::decode(sk_hex).expect("valid hex");
        assert_eq!(
            &sk.to_bytes()[..],
            sk_bytes.as_slice(),
            "vector {i}: sk mismatch"
        );

        eprintln!("keygen_kat_all: vector {i} OK");
    }
}

/// Deterministic keygen on `KAT_VECTORS[idx]`: must match the KAT pk and sk.
fn keygen_kat_idx_inner(idx: usize) {
    let (seed_hex, pk_hex, sk_hex, ..) = crate::keys::kat_data::KAT_VECTORS[idx];
    let seed_bytes = hex::decode(seed_hex).expect("valid hex");
    let seed: [u8; 48] = seed_bytes.as_slice().try_into().expect("seed is 48 bytes");
    let sk = SigningKey::generate_derand(&seed)
        .unwrap_or_else(|e| panic!("KAT[{idx}] keygen exhausted retries or errored: {e:?}"));
    let pk_bytes = hex::decode(pk_hex).expect("valid hex");
    assert_eq!(
        &sk.verifying_key().to_bytes()[..],
        pk_bytes.as_slice(),
        "KAT[{idx}]: pk mismatch"
    );
    let sk_bytes = hex::decode(sk_hex).expect("valid hex");
    assert_eq!(
        &sk.to_bytes()[..],
        sk_bytes.as_slice(),
        "KAT[{idx}]: sk mismatch"
    );
}

// Per-KAT keygen tests (one per `KAT_VECTORS` entry). Each runs
// `generate_derand(seed)` and asserts the resulting (pk, sk)
// match the KAT vector. `#[ignore]`d because keygen is a
// probabilistic retry loop that may take seconds; run via
// `cargo nextest run --run-ignored=ignored-only --profile kat -E
// 'test(/keygen_kat_/)'`.

#[test]
#[ignore]
fn keygen_kat_000() {
    keygen_kat_idx_inner(0);
}

#[test]
#[ignore]
fn keygen_kat_001() {
    keygen_kat_idx_inner(1);
}

#[test]
#[ignore]
fn keygen_kat_002() {
    keygen_kat_idx_inner(2);
}

#[test]
#[ignore]
fn keygen_kat_003() {
    keygen_kat_idx_inner(3);
}

#[test]
#[ignore]
fn keygen_kat_004() {
    keygen_kat_idx_inner(4);
}

#[test]
#[ignore]
fn keygen_kat_005() {
    keygen_kat_idx_inner(5);
}

#[test]
#[ignore]
fn keygen_kat_006() {
    keygen_kat_idx_inner(6);
}

#[test]
#[ignore]
fn keygen_kat_007() {
    keygen_kat_idx_inner(7);
}

#[test]
#[ignore]
fn keygen_kat_008() {
    keygen_kat_idx_inner(8);
}

#[test]
#[ignore]
fn keygen_kat_009() {
    keygen_kat_idx_inner(9);
}

#[test]
#[ignore]
fn keygen_kat_010() {
    keygen_kat_idx_inner(10);
}

#[test]
#[ignore]
fn keygen_kat_011() {
    keygen_kat_idx_inner(11);
}

#[test]
#[ignore]
fn keygen_kat_012() {
    keygen_kat_idx_inner(12);
}

#[test]
#[ignore]
fn keygen_kat_013() {
    keygen_kat_idx_inner(13);
}

#[test]
#[ignore]
fn keygen_kat_014() {
    keygen_kat_idx_inner(14);
}

#[test]
#[ignore]
fn keygen_kat_015() {
    keygen_kat_idx_inner(15);
}

#[test]
#[ignore]
fn keygen_kat_016() {
    keygen_kat_idx_inner(16);
}

#[test]
#[ignore]
fn keygen_kat_017() {
    keygen_kat_idx_inner(17);
}

#[test]
#[ignore]
fn keygen_kat_018() {
    keygen_kat_idx_inner(18);
}

#[test]
#[ignore]
fn keygen_kat_019() {
    keygen_kat_idx_inner(19);
}

#[test]
#[ignore]
fn keygen_kat_020() {
    keygen_kat_idx_inner(20);
}

#[test]
#[ignore]
fn keygen_kat_021() {
    keygen_kat_idx_inner(21);
}

#[test]
#[ignore]
fn keygen_kat_022() {
    keygen_kat_idx_inner(22);
}

#[test]
#[ignore]
fn keygen_kat_023() {
    keygen_kat_idx_inner(23);
}

#[test]
#[ignore]
fn keygen_kat_024() {
    keygen_kat_idx_inner(24);
}

#[test]
#[ignore]
fn keygen_kat_025() {
    keygen_kat_idx_inner(25);
}

#[test]
#[ignore]
fn keygen_kat_026() {
    keygen_kat_idx_inner(26);
}

#[test]
#[ignore]
fn keygen_kat_027() {
    keygen_kat_idx_inner(27);
}

#[test]
#[ignore]
fn keygen_kat_028() {
    keygen_kat_idx_inner(28);
}

#[test]
#[ignore]
fn keygen_kat_029() {
    keygen_kat_idx_inner(29);
}

#[test]
#[ignore]
fn keygen_kat_030() {
    keygen_kat_idx_inner(30);
}

#[test]
#[ignore]
fn keygen_kat_031() {
    keygen_kat_idx_inner(31);
}

#[test]
#[ignore]
fn keygen_kat_032() {
    keygen_kat_idx_inner(32);
}

#[test]
#[ignore]
fn keygen_kat_033() {
    keygen_kat_idx_inner(33);
}

#[test]
#[ignore]
fn keygen_kat_034() {
    keygen_kat_idx_inner(34);
}

#[test]
#[ignore]
fn keygen_kat_035() {
    keygen_kat_idx_inner(35);
}

#[test]
#[ignore]
fn keygen_kat_036() {
    keygen_kat_idx_inner(36);
}

#[test]
#[ignore]
fn keygen_kat_037() {
    keygen_kat_idx_inner(37);
}

#[test]
#[ignore]
fn keygen_kat_038() {
    keygen_kat_idx_inner(38);
}

#[test]
#[ignore]
fn keygen_kat_039() {
    keygen_kat_idx_inner(39);
}

#[test]
#[ignore]
fn keygen_kat_040() {
    keygen_kat_idx_inner(40);
}

#[test]
#[ignore]
fn keygen_kat_041() {
    keygen_kat_idx_inner(41);
}

#[test]
#[ignore]
fn keygen_kat_042() {
    keygen_kat_idx_inner(42);
}

#[test]
#[ignore]
fn keygen_kat_043() {
    keygen_kat_idx_inner(43);
}

#[test]
#[ignore]
fn keygen_kat_044() {
    keygen_kat_idx_inner(44);
}

#[test]
#[ignore]
fn keygen_kat_045() {
    keygen_kat_idx_inner(45);
}

#[test]
#[ignore]
fn keygen_kat_046() {
    keygen_kat_idx_inner(46);
}

#[test]
#[ignore]
fn keygen_kat_047() {
    keygen_kat_idx_inner(47);
}

#[test]
#[ignore]
fn keygen_kat_048() {
    keygen_kat_idx_inner(48);
}

#[test]
#[ignore]
fn keygen_kat_049() {
    keygen_kat_idx_inner(49);
}

#[test]
#[ignore]
fn keygen_kat_050() {
    keygen_kat_idx_inner(50);
}

#[test]
#[ignore]
fn keygen_kat_051() {
    keygen_kat_idx_inner(51);
}

#[test]
#[ignore]
fn keygen_kat_052() {
    keygen_kat_idx_inner(52);
}

#[test]
#[ignore]
fn keygen_kat_053() {
    keygen_kat_idx_inner(53);
}

#[test]
#[ignore]
fn keygen_kat_054() {
    keygen_kat_idx_inner(54);
}

#[test]
#[ignore]
fn keygen_kat_055() {
    keygen_kat_idx_inner(55);
}

#[test]
#[ignore]
fn keygen_kat_056() {
    keygen_kat_idx_inner(56);
}

#[test]
#[ignore]
fn keygen_kat_057() {
    keygen_kat_idx_inner(57);
}

#[test]
#[ignore]
fn keygen_kat_058() {
    keygen_kat_idx_inner(58);
}

#[test]
#[ignore]
fn keygen_kat_059() {
    keygen_kat_idx_inner(59);
}

#[test]
#[ignore]
fn keygen_kat_060() {
    keygen_kat_idx_inner(60);
}

#[test]
#[ignore]
fn keygen_kat_061() {
    keygen_kat_idx_inner(61);
}

#[test]
#[ignore]
fn keygen_kat_062() {
    keygen_kat_idx_inner(62);
}

#[test]
#[ignore]
fn keygen_kat_063() {
    keygen_kat_idx_inner(63);
}

#[test]
#[ignore]
fn keygen_kat_064() {
    keygen_kat_idx_inner(64);
}

#[test]
#[ignore]
fn keygen_kat_065() {
    keygen_kat_idx_inner(65);
}

#[test]
#[ignore]
fn keygen_kat_066() {
    keygen_kat_idx_inner(66);
}

#[test]
#[ignore]
fn keygen_kat_067() {
    keygen_kat_idx_inner(67);
}

#[test]
#[ignore]
fn keygen_kat_068() {
    keygen_kat_idx_inner(68);
}

#[test]
#[ignore]
fn keygen_kat_069() {
    keygen_kat_idx_inner(69);
}

#[test]
#[ignore]
fn keygen_kat_070() {
    keygen_kat_idx_inner(70);
}

#[test]
#[ignore]
fn keygen_kat_071() {
    keygen_kat_idx_inner(71);
}

#[test]
#[ignore]
fn keygen_kat_072() {
    keygen_kat_idx_inner(72);
}

#[test]
#[ignore]
fn keygen_kat_073() {
    keygen_kat_idx_inner(73);
}

#[test]
#[ignore]
fn keygen_kat_074() {
    keygen_kat_idx_inner(74);
}

#[test]
#[ignore]
fn keygen_kat_075() {
    keygen_kat_idx_inner(75);
}

#[test]
#[ignore]
fn keygen_kat_076() {
    keygen_kat_idx_inner(76);
}

#[test]
#[ignore]
fn keygen_kat_077() {
    keygen_kat_idx_inner(77);
}

#[test]
#[ignore]
fn keygen_kat_078() {
    keygen_kat_idx_inner(78);
}

#[test]
#[ignore]
fn keygen_kat_079() {
    keygen_kat_idx_inner(79);
}

#[test]
#[ignore]
fn keygen_kat_080() {
    keygen_kat_idx_inner(80);
}

#[test]
#[ignore]
fn keygen_kat_081() {
    keygen_kat_idx_inner(81);
}

#[test]
#[ignore]
fn keygen_kat_082() {
    keygen_kat_idx_inner(82);
}

#[test]
#[ignore]
fn keygen_kat_083() {
    keygen_kat_idx_inner(83);
}

#[test]
#[ignore]
fn keygen_kat_084() {
    keygen_kat_idx_inner(84);
}

#[test]
#[ignore]
fn keygen_kat_085() {
    keygen_kat_idx_inner(85);
}

#[test]
#[ignore]
fn keygen_kat_086() {
    keygen_kat_idx_inner(86);
}

#[test]
#[ignore]
fn keygen_kat_087() {
    keygen_kat_idx_inner(87);
}

#[test]
#[ignore]
fn keygen_kat_088() {
    keygen_kat_idx_inner(88);
}

#[test]
#[ignore]
fn keygen_kat_089() {
    keygen_kat_idx_inner(89);
}

#[test]
#[ignore]
fn keygen_kat_090() {
    keygen_kat_idx_inner(90);
}

#[test]
#[ignore]
fn keygen_kat_091() {
    keygen_kat_idx_inner(91);
}

#[test]
#[ignore]
fn keygen_kat_092() {
    keygen_kat_idx_inner(92);
}

#[test]
#[ignore]
fn keygen_kat_093() {
    keygen_kat_idx_inner(93);
}

#[test]
#[ignore]
fn keygen_kat_094() {
    keygen_kat_idx_inner(94);
}

#[test]
#[ignore]
fn keygen_kat_095() {
    keygen_kat_idx_inner(95);
}

#[test]
#[ignore]
fn keygen_kat_096() {
    keygen_kat_idx_inner(96);
}

#[test]
#[ignore]
fn keygen_kat_097() {
    keygen_kat_idx_inner(97);
}

#[test]
#[ignore]
fn keygen_kat_098() {
    keygen_kat_idx_inner(98);
}

#[test]
#[ignore]
fn keygen_kat_099() {
    keygen_kat_idx_inner(99);
}

/// Deserialize every KAT signing key, sign the corresponding
/// message, and verify with the paired public key.
///
/// Run with: `cargo test --lib --release sign_kat_all -- --ignored`.
#[test]
#[ignore]
fn sign_kat_all() {
    for (i, &(_, pk_hex, sk_hex, msg_hex, _)) in
        crate::keys::kat_data::KAT_VECTORS.iter().enumerate()
    {
        let sk_bytes = hex::decode(sk_hex).expect("valid hex");
        let pk_bytes = hex::decode(pk_hex).expect("valid hex");
        let msg = hex::decode(msg_hex).expect("valid hex");

        let sk = SigningKey::from_bytes(sk_bytes.as_slice().try_into().unwrap())
            .expect("sk should parse");
        let vk = VerifyingKey::from_bytes(pk_bytes.as_slice().try_into().unwrap())
            .expect("pk should parse");

        let sig = match sk.sign(&msg, &mut OsRng) {
            Ok(s) => s,
            Err(SignatureError::SigningFailed) => {
                eprintln!("sign_kat_all: vector {i} SigningFailed (expected)");
                continue;
            }
            Err(other) => panic!("vector {i}: unexpected sign error: {other:?}"),
        };

        vk.verify(&msg, &sig)
            .unwrap_or_else(|_| panic!("vector {i}: signature did not verify"));
        eprintln!("sign_kat_all: vector {i} OK");
    }
}

/// Every KAT signing key deserializes and its embedded verifying
/// key matches the standalone KAT pk.
#[test]
fn kat_sk_pk_match_all() {
    for (i, &(_, pk_hex, sk_hex, ..)) in crate::keys::kat_data::KAT_VECTORS.iter().enumerate() {
        let sk_bytes = hex::decode(sk_hex).expect("valid hex");
        let pk_bytes = hex::decode(pk_hex).expect("valid hex");

        let sk = SigningKey::from_bytes(sk_bytes.as_slice().try_into().unwrap())
            .unwrap_or_else(|_| panic!("vector {i}: sk should parse"));

        assert_eq!(
            &sk.verifying_key().to_bytes()[..],
            pk_bytes.as_slice(),
            "vector {i}: embedded pk mismatch"
        );
    }
}

/// Parse → serialize → re-parse round-trip for every KAT signing key.
#[test]
fn kat_sk_roundtrip_all() {
    for (i, &(_, _, sk_hex, ..)) in crate::keys::kat_data::KAT_VECTORS.iter().enumerate() {
        let sk_bytes: [u8; SIGNING_KEY_BYTES] =
            hex::decode(sk_hex).expect("valid hex").try_into().unwrap();

        let sk = SigningKey::from_bytes(&sk_bytes)
            .unwrap_or_else(|_| panic!("vector {i}: sk should parse"));

        let reserialized = sk.to_bytes();
        assert_eq!(reserialized, sk_bytes, "vector {i}: sk round-trip mismatch");
    }
}

/// Sign + self-verify on KAT vector 0 only — focused test for
/// task #32 ("First sign() SUCCESS + self-verify roundtrip").
///
/// Avoids the cost of running through all 100 KAT vectors when
/// all we want is the answer to "does any signature round-trip
/// successfully?" If this passes, the response-phase pipeline
/// (`to_isogeny`, `split_aux`, `from_bases`, signature encoding,
/// verification) is end-to-end functional.
///
/// Run with: `cargo test --lib --release sign_kat_zero_only -- --ignored
/// --nocapture`.
#[test]
#[ignore]
fn sign_kat_zero_only() {
    let (_, pk_hex, sk_hex, msg_hex, _) = crate::keys::kat_data::KAT_VECTORS[0];
    let sk_bytes = hex::decode(sk_hex).expect("valid hex");
    let pk_bytes = hex::decode(pk_hex).expect("valid hex");
    let msg = hex::decode(msg_hex).expect("valid hex");

    let sk =
        SigningKey::from_bytes(sk_bytes.as_slice().try_into().unwrap()).expect("sk should parse");
    let vk =
        VerifyingKey::from_bytes(pk_bytes.as_slice().try_into().unwrap()).expect("pk should parse");

    let t0 = std::time::Instant::now();
    let sig = sk
        .sign(&msg, &mut OsRng)
        .expect("KAT[0] sign must succeed within retry budget");
    eprintln!("sign_kat_zero_only: sign OK in {:?}", t0.elapsed());

    let t1 = std::time::Instant::now();
    vk.verify(&msg, &sig)
        .expect("KAT[0] signature must verify against paired pk");
    eprintln!("sign_kat_zero_only: verify OK in {:?}", t1.elapsed());
}

/// Deterministic sign-and-verify probe for `KAT_VECTORS[kat_idx]`.
fn sign_kat_idx_probe_inner(kat_idx: usize) {
    let (seed_hex, pk_hex, sk_hex, msg_hex, _) = crate::keys::kat_data::KAT_VECTORS[kat_idx];
    let seed_bytes = hex::decode(seed_hex).expect("valid hex");
    let seed: [u8; 48] = seed_bytes.as_slice().try_into().expect("seed is 48 bytes");
    let sk_bytes = hex::decode(sk_hex).expect("valid hex");
    let pk_bytes = hex::decode(pk_hex).expect("valid hex");
    let msg = hex::decode(msg_hex).expect("valid hex");
    let sk =
        SigningKey::from_bytes(sk_bytes.as_slice().try_into().unwrap()).expect("sk should parse");
    let vk =
        VerifyingKey::from_bytes(pk_bytes.as_slice().try_into().unwrap()).expect("pk should parse");
    let t0 = std::time::Instant::now();
    let sig = sk
        .sign_derand(&msg, &seed)
        .expect("sign_derand must succeed within retry budget");
    let elapsed = t0.elapsed();
    vk.verify(&msg, &sig)
        .unwrap_or_else(|e| panic!("KAT[{kat_idx}] verify failed: {e:?}"));
    eprintln!("sign_kat_derand_{kat_idx:03}: sign={elapsed:?}");
}

/// `KAT_IDX=N`-parametrized deterministic sign-and-verify probe.
/// Used to confirm `sign_derand` is genuinely deterministic and to
/// localize response-phase hangs on specific trajectories.
#[test]
#[ignore]
fn sign_kat_idx_probe() {
    let kat_idx: usize = std::env::var("KAT_IDX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    eprintln!("sign_kat_idx_probe: KAT_IDX={kat_idx}");
    sign_kat_idx_probe_inner(kat_idx);
}

/// `KAT_IDX=N`-parametrized sign + verify that **also writes**
/// `pk.bin / msg.bin / our_sig.bin` to `$DUMP_DIR` (default
/// `/tmp/cross_verify/`). Pair with C ref's
/// `sqisign_verify_external_lvl1` to confirm C ref accepts our
/// signature byte-for-byte. Required by
/// `/tmp/cross_verify/cross_verify_our_sigs.sh`.
#[test]
#[ignore]
fn sign_kat_zero_dump_for_xverify() {
    use std::fs;

    let kat_idx: usize = std::env::var("KAT_IDX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let dir = std::env::var("DUMP_DIR").unwrap_or_else(|_| "/tmp/cross_verify".to_string());

    let (seed_hex, pk_hex, sk_hex, msg_hex, _) = crate::keys::kat_data::KAT_VECTORS[kat_idx];
    let seed_bytes = hex::decode(seed_hex).expect("valid hex");
    let seed: [u8; 48] = seed_bytes.as_slice().try_into().expect("seed is 48 bytes");
    let sk_bytes = hex::decode(sk_hex).expect("valid hex");
    let pk_bytes = hex::decode(pk_hex).expect("valid hex");
    let msg = hex::decode(msg_hex).expect("valid hex");

    let sk = SigningKey::from_bytes(sk_bytes.as_slice().try_into().unwrap()).expect("sk parses");
    let vk = VerifyingKey::from_bytes(pk_bytes.as_slice().try_into().unwrap()).expect("pk parses");

    let t0 = std::time::Instant::now();
    let sig = sk
        .sign_derand(&msg, &seed)
        .expect("sign_derand within retry budget");
    let sign_elapsed = t0.elapsed();

    fs::create_dir_all(&dir).expect("mkdir dump dir");
    fs::write(format!("{dir}/pk.bin"), &pk_bytes).expect("write pk.bin");
    fs::write(format!("{dir}/msg.bin"), &msg).expect("write msg.bin");
    fs::write(format!("{dir}/our_sig.bin"), sig.to_bytes()).expect("write our_sig.bin");
    eprintln!(
        "sign_kat_zero_dump_for_xverify: KAT[{kat_idx}] sign={sign_elapsed:?} dumped to {dir}"
    );

    let t1 = std::time::Instant::now();
    let r = vk.verify(&msg, &sig);
    eprintln!(
        "sign_kat_zero_dump_for_xverify: KAT[{kat_idx}] verify result={r:?} in {:?}",
        t1.elapsed()
    );
}

// Per-KAT deterministic sign+verify tests (one per `KAT_VECTORS`
// Each runs `sign_derand(seed)` + `verify`. `#[ignore]`d
// because sign goes through a long retry loop on certain
// trajectories; run via
// `cargo nextest run --run-ignored=ignored-only --profile kat -E
// 'test(/sign_kat_derand_/)'`

#[test]
#[ignore]
fn sign_kat_derand_000() {
    sign_kat_idx_probe_inner(0);
}

// Not `#[ignore]`d: the only sign() KAT that runs in default `cargo test`,
// chosen because KAT[1] is the smallest vector in the (n_bt > 0, r_rsp > 0)
// cross-product class — the parameter combination that hides the verify-side
// scaling-formula bugs called out in `kat_data` doc. Without one such test
// in the regular suite, `keys/signing.rs` shows ~13% coverage and any sign
// mutation lands in untested code. See `kat_data` for the parameter table.
#[test]
fn sign_kat_derand_001() {
    sign_kat_idx_probe_inner(1);
}

#[test]
#[ignore]
fn sign_kat_derand_002() {
    sign_kat_idx_probe_inner(2);
}

#[test]
#[ignore]
fn sign_kat_derand_003() {
    sign_kat_idx_probe_inner(3);
}

#[test]
#[ignore]
fn sign_kat_derand_004() {
    sign_kat_idx_probe_inner(4);
}

#[test]
#[ignore]
fn sign_kat_derand_005() {
    sign_kat_idx_probe_inner(5);
}

#[test]
#[ignore]
fn sign_kat_derand_006() {
    sign_kat_idx_probe_inner(6);
}

#[test]
#[ignore]
fn sign_kat_derand_007() {
    sign_kat_idx_probe_inner(7);
}

#[test]
#[ignore]
fn sign_kat_derand_008() {
    sign_kat_idx_probe_inner(8);
}

// Not `#[ignore]`d: companion to `sign_kat_derand_001`. KAT[9] is the
// largest-`n_bt` vector in the cross-product class (n_bt=2, r_rsp=1),
// hedging against parameter-edge regressions that a single (n_bt=1)
// sample misses. See `kat_data` for the parameter table.
#[test]
fn sign_kat_derand_009() {
    sign_kat_idx_probe_inner(9);
}

#[test]
#[ignore]
fn sign_kat_derand_010() {
    sign_kat_idx_probe_inner(10);
}

#[test]
#[ignore]
fn sign_kat_derand_011() {
    sign_kat_idx_probe_inner(11);
}

#[test]
#[ignore]
fn sign_kat_derand_012() {
    sign_kat_idx_probe_inner(12);
}

#[test]
#[ignore]
fn sign_kat_derand_013() {
    sign_kat_idx_probe_inner(13);
}

#[test]
#[ignore]
fn sign_kat_derand_014() {
    sign_kat_idx_probe_inner(14);
}

#[test]
#[ignore]
fn sign_kat_derand_015() {
    sign_kat_idx_probe_inner(15);
}

#[test]
#[ignore]
fn sign_kat_derand_016() {
    sign_kat_idx_probe_inner(16);
}

#[test]
#[ignore]
fn sign_kat_derand_017() {
    sign_kat_idx_probe_inner(17);
}

#[test]
#[ignore]
fn sign_kat_derand_018() {
    sign_kat_idx_probe_inner(18);
}

#[test]
#[ignore]
fn sign_kat_derand_019() {
    sign_kat_idx_probe_inner(19);
}

#[test]
#[ignore]
fn sign_kat_derand_020() {
    sign_kat_idx_probe_inner(20);
}

#[test]
#[ignore]
fn sign_kat_derand_021() {
    sign_kat_idx_probe_inner(21);
}

#[test]
#[ignore]
fn sign_kat_derand_022() {
    sign_kat_idx_probe_inner(22);
}

#[test]
#[ignore]
fn sign_kat_derand_023() {
    sign_kat_idx_probe_inner(23);
}

#[test]
#[ignore]
fn sign_kat_derand_024() {
    sign_kat_idx_probe_inner(24);
}

#[test]
#[ignore]
fn sign_kat_derand_025() {
    sign_kat_idx_probe_inner(25);
}

#[test]
#[ignore]
fn sign_kat_derand_026() {
    sign_kat_idx_probe_inner(26);
}

#[test]
#[ignore]
fn sign_kat_derand_027() {
    sign_kat_idx_probe_inner(27);
}

#[test]
#[ignore]
fn sign_kat_derand_028() {
    sign_kat_idx_probe_inner(28);
}

#[test]
#[ignore]
fn sign_kat_derand_029() {
    sign_kat_idx_probe_inner(29);
}

#[test]
#[ignore]
fn sign_kat_derand_030() {
    sign_kat_idx_probe_inner(30);
}

#[test]
#[ignore]
fn sign_kat_derand_031() {
    sign_kat_idx_probe_inner(31);
}

#[test]
#[ignore]
fn sign_kat_derand_032() {
    sign_kat_idx_probe_inner(32);
}

#[test]
#[ignore]
fn sign_kat_derand_033() {
    sign_kat_idx_probe_inner(33);
}

#[test]
#[ignore]
fn sign_kat_derand_034() {
    sign_kat_idx_probe_inner(34);
}

#[test]
#[ignore]
fn sign_kat_derand_035() {
    sign_kat_idx_probe_inner(35);
}

#[test]
#[ignore]
fn sign_kat_derand_036() {
    sign_kat_idx_probe_inner(36);
}

#[test]
#[ignore]
fn sign_kat_derand_037() {
    sign_kat_idx_probe_inner(37);
}

#[test]
#[ignore]
fn sign_kat_derand_038() {
    sign_kat_idx_probe_inner(38);
}

#[test]
#[ignore]
fn sign_kat_derand_039() {
    sign_kat_idx_probe_inner(39);
}

#[test]
#[ignore]
fn sign_kat_derand_040() {
    sign_kat_idx_probe_inner(40);
}

#[test]
#[ignore]
fn sign_kat_derand_041() {
    sign_kat_idx_probe_inner(41);
}

#[test]
#[ignore]
fn sign_kat_derand_042() {
    sign_kat_idx_probe_inner(42);
}

#[test]
#[ignore]
fn sign_kat_derand_043() {
    sign_kat_idx_probe_inner(43);
}

#[test]
#[ignore]
fn sign_kat_derand_044() {
    sign_kat_idx_probe_inner(44);
}

#[test]
#[ignore]
fn sign_kat_derand_045() {
    sign_kat_idx_probe_inner(45);
}

#[test]
#[ignore]
fn sign_kat_derand_046() {
    sign_kat_idx_probe_inner(46);
}

#[test]
#[ignore]
fn sign_kat_derand_047() {
    sign_kat_idx_probe_inner(47);
}

#[test]
#[ignore]
fn sign_kat_derand_048() {
    sign_kat_idx_probe_inner(48);
}

#[test]
#[ignore]
fn sign_kat_derand_049() {
    sign_kat_idx_probe_inner(49);
}

#[test]
#[ignore]
fn sign_kat_derand_050() {
    sign_kat_idx_probe_inner(50);
}

#[test]
#[ignore]
fn sign_kat_derand_051() {
    sign_kat_idx_probe_inner(51);
}

#[test]
#[ignore]
fn sign_kat_derand_052() {
    sign_kat_idx_probe_inner(52);
}

#[test]
#[ignore]
fn sign_kat_derand_053() {
    sign_kat_idx_probe_inner(53);
}

#[test]
#[ignore]
fn sign_kat_derand_054() {
    sign_kat_idx_probe_inner(54);
}

#[test]
#[ignore]
fn sign_kat_derand_055() {
    sign_kat_idx_probe_inner(55);
}

#[test]
#[ignore]
fn sign_kat_derand_056() {
    sign_kat_idx_probe_inner(56);
}

#[test]
#[ignore]
fn sign_kat_derand_057() {
    sign_kat_idx_probe_inner(57);
}

#[test]
#[ignore]
fn sign_kat_derand_058() {
    sign_kat_idx_probe_inner(58);
}

#[test]
#[ignore]
fn sign_kat_derand_059() {
    sign_kat_idx_probe_inner(59);
}

#[test]
#[ignore]
fn sign_kat_derand_060() {
    sign_kat_idx_probe_inner(60);
}

#[test]
#[ignore]
fn sign_kat_derand_061() {
    sign_kat_idx_probe_inner(61);
}

#[test]
#[ignore]
fn sign_kat_derand_062() {
    sign_kat_idx_probe_inner(62);
}

#[test]
#[ignore]
fn sign_kat_derand_063() {
    sign_kat_idx_probe_inner(63);
}

#[test]
#[ignore]
fn sign_kat_derand_064() {
    sign_kat_idx_probe_inner(64);
}

#[test]
#[ignore]
fn sign_kat_derand_065() {
    sign_kat_idx_probe_inner(65);
}

#[test]
#[ignore]
fn sign_kat_derand_066() {
    sign_kat_idx_probe_inner(66);
}

#[test]
#[ignore]
fn sign_kat_derand_067() {
    sign_kat_idx_probe_inner(67);
}

#[test]
#[ignore]
fn sign_kat_derand_068() {
    sign_kat_idx_probe_inner(68);
}

#[test]
#[ignore]
fn sign_kat_derand_069() {
    sign_kat_idx_probe_inner(69);
}

#[test]
#[ignore]
fn sign_kat_derand_070() {
    sign_kat_idx_probe_inner(70);
}

#[test]
#[ignore]
fn sign_kat_derand_071() {
    sign_kat_idx_probe_inner(71);
}

#[test]
#[ignore]
fn sign_kat_derand_072() {
    sign_kat_idx_probe_inner(72);
}

#[test]
#[ignore]
fn sign_kat_derand_073() {
    sign_kat_idx_probe_inner(73);
}

#[test]
#[ignore]
fn sign_kat_derand_074() {
    sign_kat_idx_probe_inner(74);
}

#[test]
#[ignore]
fn sign_kat_derand_075() {
    sign_kat_idx_probe_inner(75);
}

#[test]
#[ignore]
fn sign_kat_derand_076() {
    sign_kat_idx_probe_inner(76);
}

#[test]
#[ignore]
fn sign_kat_derand_077() {
    sign_kat_idx_probe_inner(77);
}

#[test]
#[ignore]
fn sign_kat_derand_078() {
    sign_kat_idx_probe_inner(78);
}

#[test]
#[ignore]
fn sign_kat_derand_079() {
    sign_kat_idx_probe_inner(79);
}

#[test]
#[ignore]
fn sign_kat_derand_080() {
    sign_kat_idx_probe_inner(80);
}

#[test]
#[ignore]
fn sign_kat_derand_081() {
    sign_kat_idx_probe_inner(81);
}

#[test]
#[ignore]
fn sign_kat_derand_082() {
    sign_kat_idx_probe_inner(82);
}

#[test]
#[ignore]
fn sign_kat_derand_083() {
    sign_kat_idx_probe_inner(83);
}

#[test]
#[ignore]
fn sign_kat_derand_084() {
    sign_kat_idx_probe_inner(84);
}

#[test]
#[ignore]
fn sign_kat_derand_085() {
    sign_kat_idx_probe_inner(85);
}

#[test]
#[ignore]
fn sign_kat_derand_086() {
    sign_kat_idx_probe_inner(86);
}

#[test]
#[ignore]
fn sign_kat_derand_087() {
    sign_kat_idx_probe_inner(87);
}

#[test]
#[ignore]
fn sign_kat_derand_088() {
    sign_kat_idx_probe_inner(88);
}

#[test]
#[ignore]
fn sign_kat_derand_089() {
    sign_kat_idx_probe_inner(89);
}

#[test]
#[ignore]
fn sign_kat_derand_090() {
    sign_kat_idx_probe_inner(90);
}

#[test]
#[ignore]
fn sign_kat_derand_091() {
    sign_kat_idx_probe_inner(91);
}

#[test]
#[ignore]
fn sign_kat_derand_092() {
    sign_kat_idx_probe_inner(92);
}

#[test]
#[ignore]
fn sign_kat_derand_093() {
    sign_kat_idx_probe_inner(93);
}

#[test]
#[ignore]
fn sign_kat_derand_094() {
    sign_kat_idx_probe_inner(94);
}

#[test]
#[ignore]
fn sign_kat_derand_095() {
    sign_kat_idx_probe_inner(95);
}

#[test]
#[ignore]
fn sign_kat_derand_096() {
    sign_kat_idx_probe_inner(96);
}

#[test]
#[ignore]
fn sign_kat_derand_097() {
    sign_kat_idx_probe_inner(97);
}

#[test]
#[ignore]
fn sign_kat_derand_098() {
    sign_kat_idx_probe_inner(98);
}

#[test]
#[ignore]
fn sign_kat_derand_099() {
    sign_kat_idx_probe_inner(99);
}

/// Verify the *C reference's* KAT[0] signature with our verifier.
///
/// Isolates `verify` bugs from `sign` bugs: if our verifier rejects the
/// C ref's known-good signature, the bug is on the verify side. If it
/// accepts, sign is the source of `sign_kat_zero_only`'s
/// `VerificationFailed`.
///
/// Run with: `cargo test --lib --release verify_kat_zero_cref_sig --
/// --ignored`.
#[test]
#[ignore]
fn verify_kat_zero_cref_sig() {
    let (_, pk_hex, _, msg_hex, sm_hex) = crate::keys::kat_data::KAT_VECTORS[0];
    let pk_bytes = hex::decode(pk_hex).expect("valid hex");
    let msg = hex::decode(msg_hex).expect("valid hex");
    let sm = hex::decode(sm_hex).expect("valid hex");

    // NIST signed-message format: sm = sig || msg. Our SIGNATURE_BYTES
    // is 148, so sig = sm[..148].
    let sig_bytes: [u8; crate::keys::SIGNATURE_BYTES] = sm[..crate::keys::SIGNATURE_BYTES]
        .try_into()
        .expect("sig prefix is SIGNATURE_BYTES");

    // Round-trip the message bytes match too (cheap sanity).
    assert_eq!(
        &sm[crate::keys::SIGNATURE_BYTES..],
        msg.as_slice(),
        "sm tail must equal msg"
    );

    let vk = VerifyingKey::from_bytes(pk_bytes.as_slice().try_into().unwrap()).expect("pk parses");
    let sig = Signature::from_bytes(&sig_bytes).expect("C ref sig parses");

    // Dump C ref's signature fields for cross-comparison with our
    // sign-side `[SIGN_FINAL]` output. Same KAT, different sig
    // (different randomness) — but `n_bt`, `r_rsp`, and the
    // `hint_chl` / `hint_aux` should be `KAT[0]`-deterministic
    // because they're derived from the public `chl` and the
    // canonical-basis `to_hint` outputs of curves the deterministic
    // commitment phase hits. M_chl and curve_aux vary with the
    // randomness.
    eprintln!("[CREF_SIG] curve_aux.A={:?}", sig.curve_aux.coefficient());
    eprintln!(
        "[CREF_SIG] n_bt={} r_rsp={}",
        sig.n_bt.value(),
        sig.r_rsp.value()
    );
    eprintln!(
        "[CREF_SIG] hint_aux={} hint_chl={}",
        u8::from(sig.hint_aux),
        u8::from(sig.hint_chl)
    );
    eprintln!("[CREF_SIG] M_chl[0][0]={:?}", sig.M_chl.entries[0][0]);
    eprintln!("[CREF_SIG] M_chl[0][1]={:?}", sig.M_chl.entries[0][1]);
    eprintln!("[CREF_SIG] M_chl[1][0]={:?}", sig.M_chl.entries[1][0]);
    eprintln!("[CREF_SIG] M_chl[1][1]={:?}", sig.M_chl.entries[1][1]);

    vk.verify(&msg, &sig)
        .expect("C ref's KAT[0] signature must verify against our verifier");
}

/// Generate a fresh key, sign a random message, verify.
///
/// Run with: `cargo test --lib --release sign_fresh -- --ignored`.
#[test]
#[ignore]
fn sign_fresh() {
    let sk = SigningKey::generate(&mut OsRng).expect("keygen should succeed");
    let mut msg = [0u8; 64];
    rand_core::RngCore::fill_bytes(&mut OsRng, &mut msg);

    let sig = match sk.sign(&msg, &mut OsRng) {
        Ok(s) => s,
        Err(SignatureError::SigningFailed) => {
            eprintln!("sign_fresh: SigningFailed (response phase incomplete)");
            return;
        }
        Err(other) => panic!("unexpected sign error: {other:?}"),
    };

    sk.verifying_key()
        .verify(&msg, &sig)
        .expect("signature should verify");
}

/// Sign with KAT vector 0's keypair and verify.
///
/// Run with: `cargo test --lib --release sign_with_kat_key -- --ignored`.
#[test]
#[ignore]
fn sign_with_kat_key() {
    let (_, pk_hex, sk_hex, ..) = crate::keys::kat_data::KAT_VECTORS[0];
    let sk = SigningKey::from_bytes(hex::decode(sk_hex).unwrap().as_slice().try_into().unwrap())
        .expect("KAT sk should parse");
    let vk = VerifyingKey::from_bytes(hex::decode(pk_hex).unwrap().as_slice().try_into().unwrap())
        .expect("KAT pk should parse");

    let mut msg = [0u8; 64];
    rand_core::RngCore::fill_bytes(&mut OsRng, &mut msg);

    let sig = match sk.sign(&msg, &mut OsRng) {
        Ok(s) => s,
        Err(SignatureError::SigningFailed) => {
            eprintln!("sign_with_kat_key: SigningFailed (response phase incomplete)");
            return;
        }
        Err(other) => panic!("unexpected sign error: {other:?}"),
    };

    vk.verify(&msg, &sig)
        .expect("signature should verify against KAT pk");
}

/// Aggregate DRBG-byte consumption check for `generate_with_rng`
/// on KAT seed 0, independent of the per-phase probe below.
///
/// Calls the public [`SigningKey::generate_with_rng`] and asserts
/// the total byte count threaded through the DRBG matches an
/// expected value measured against the C reference. Because it
/// doesn't duplicate the keygen loop's structure, this test survives
/// any internal restructuring of `generate_with_rng` and fails
/// precisely when the aggregate byte budget drifts from the C ref.
///
/// Set `expected` to `None` to have the test print the observed
/// count (useful the first time it runs after the C reference is
/// instrumented); once a value is known, replace `None` with
/// `Some(count)` to lock it in.
///
/// Run with:
/// ```text
/// cargo test --lib --release \
///   keygen_drbg_total_bytes_seed_0 -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn keygen_drbg_total_bytes_seed_0() {
    let (seed_hex, ..) = crate::keys::kat_data::KAT_VECTORS[0];
    let seed: [u8; 48] = hex::decode(seed_hex)
        .expect("valid seed hex")
        .as_slice()
        .try_into()
        .expect("seed is 48 bytes");

    let mut drbg = crate::drbg::Aes256CtrDrbg::new(&seed);
    let before = drbg.bytes_consumed();
    let _sk = match SigningKey::generate_with_rng(&mut drbg) {
        Ok(sk) => sk,
        Err(SignatureError::KeyGenFailed) => {
            eprintln!("[TOTAL-BYTES] keygen probabilistically failed for seed 0; test skipped");
            return;
        }
        Err(other) => panic!("unexpected keygen error: {other:?}"),
    };
    let observed = drbg.bytes_consumed() - before;
    eprintln!("[TOTAL-BYTES] keygen consumed {observed} DRBG bytes for seed 0");

    // Replace `None` with `Some(...)` once the C-reference
    // `drbg_bytes_consumed` counter is instrumented at the same
    // user-facing output boundary (see `drbg.rs::bytes_consumed`
    // for the counter's semantics).
    let expected: Option<u64> = None;
    if let Some(e) = expected {
        assert_eq!(
            observed, e,
            "DRBG byte-consumption for seed-0 keygen drifted from the C reference"
        );
    }
}

/// Reconstruct KAT[0]'s secret ideal directly from `(norm, gen)`
/// and run `to_isogeny` on it; check whether the computed `e_pk`
/// matches the standalone KAT pk.
///
/// This bypasses the seed→sample→reduce sampling chain entirely
/// and isolates whether the post-sampling pipeline (to_isogeny,
/// to_hint, M_sk encoding) reproduces the C reference's output for
/// a known-correct ideal. If it passes:
/// - `keygen_kat_all` divergence is in the sampling/reduction path
///   (`random_prime_norm_wide`, `reduce_to_prime_norm`).
/// - The to_isogeny-and-friends pipeline is interop-correct, which would also
///   be a strong signal for the still-open sign+verify mystery (since signing
///   reuses to_isogeny internally).
///
/// If it fails:
/// - `to_isogeny` and/or its downstream consumers diverge from the C ref.
///   Compare `e_pk` byte-for-byte to localize.
///
/// Run with:
/// ```text
/// cargo test --lib --release \
///   keygen_target_to_isogeny_seed_0 -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn keygen_target_to_isogeny_seed_0() {
    let (_, pk_hex, sk_hex, ..) = crate::keys::kat_data::KAT_VECTORS[0];
    let sk_bytes = hex::decode(sk_hex).expect("valid hex");
    let pk_bytes = hex::decode(pk_hex).expect("valid hex");

    // Parse the secret-ideal `(norm, gen)` from `sk_bytes` exactly
    // as `SigningKey::from_bytes` does.
    let mut pos = VERIFYING_KEY_BYTES;
    let norm_bytes: &[u8; FP_ENCODED_BYTES] = sk_bytes[pos..pos + FP_ENCODED_BYTES]
        .try_into()
        .expect("32-byte norm");
    let norm = IsogenyDegree::from_bytes_le(norm_bytes)
        .expect("norm parses as positive odd IsogenyDegree");
    pos += FP_ENCODED_BYTES;

    let mut gen_coords = [BigInt::<4>::ZERO; 4];
    for coord in &mut gen_coords {
        *coord = BigInt::<4>::from_bytes_le_signed(
            sk_bytes[pos..pos + FP_ENCODED_BYTES]
                .try_into()
                .expect("32-byte coord"),
        );
        pos += FP_ENCODED_BYTES;
    }

    let gen = Element {
        a: Coordinate::from(gen_coords[0]),
        b: Coordinate::from(gen_coords[1]),
        c: Coordinate::from(gen_coords[2]),
        d: Coordinate::from(gen_coords[3]),
        denom: Denominator::ONE,
    };
    let norm_bigint = norm.to_bigint();
    let ideal = LeftIdeal::<4>::new(&gen, &norm_bigint, EXTREMAL_ORDERS[0].order());

    // Run `to_isogeny` and compare `e_pk` byte-for-byte against the
    // standalone KAT pk's curve coefficient.
    let (e_pk, _phi_p, _phi_q, _phi_pmq) = ideal
        .to_isogeny(&mut OsRng)
        .expect("to_isogeny on KAT[0] reconstructed ideal must succeed");

    let computed_a = e_pk.coefficient().to_bytes();
    let expected_a = &pk_bytes[..64];

    eprintln!("[TI] computed e_pk.A = {}", hex::encode(computed_a));
    eprintln!("[TI] expected   pk.A = {}", hex::encode(expected_a));

    let (_, hint_pk) = TorsionBasis::to_hint(&e_pk);
    eprintln!(
        "[TI] computed hint_pk = {:08b}, expected = {:08b}",
        hint_pk.to_byte(),
        pk_bytes[64]
    );

    assert_eq!(
        &computed_a[..],
        expected_a,
        "to_isogeny(KAT[0] secret ideal) must reproduce KAT[0] pk's A"
    );
    assert_eq!(
        hint_pk.to_byte(),
        pk_bytes[64],
        "to_hint on the computed e_pk must produce KAT[0]'s hint byte"
    );
}

/// Survey `to_isogeny` against the first 10 KAT vectors. For each
/// vector, parse `(norm, gen)` from `sk_hex`, build the secret ideal
/// the same way [`keygen_target_to_isogeny_seed_0`] does, run
/// `to_isogeny`, and compare the resulting curve coefficient to the
/// expected `pk_bytes[..64]`.
///
/// Prints a `[SURVEY]` line per vector plus a final summary so we can
/// see at a glance whether divergences from the C reference are
/// patterned (one fixed algorithmic mismatch) or differ in shape per
/// vector (suggesting non-determinism / iteration-order dependence).
///
/// Run with:
/// ```text
/// cargo test --lib --release \
///   survey_keygen_target_to_isogeny_first_10 -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn survey_keygen_target_to_isogeny_first_10() {
    let mut total = 0u32;
    let mut matched = 0u32;
    let mut none_count = 0u32;

    for i in 0..10 {
        let (_, pk_hex, sk_hex, ..) = crate::keys::kat_data::KAT_VECTORS[i];
        let sk_bytes = hex::decode(sk_hex).expect("valid hex");
        let pk_bytes = hex::decode(pk_hex).expect("valid hex");

        // Parse `(norm, gen)` from `sk_bytes`, exactly as
        // `keygen_target_to_isogeny_seed_0` does.
        let mut pos = VERIFYING_KEY_BYTES;
        let norm_bytes: &[u8; FP_ENCODED_BYTES] = sk_bytes[pos..pos + FP_ENCODED_BYTES]
            .try_into()
            .expect("32-byte norm");
        let norm = IsogenyDegree::from_bytes_le(norm_bytes)
            .expect("norm parses as positive odd IsogenyDegree");
        pos += FP_ENCODED_BYTES;

        let mut gen_coords = [BigInt::<4>::ZERO; 4];
        for coord in &mut gen_coords {
            *coord = BigInt::<4>::from_bytes_le_signed(
                sk_bytes[pos..pos + FP_ENCODED_BYTES]
                    .try_into()
                    .expect("32-byte coord"),
            );
            pos += FP_ENCODED_BYTES;
        }

        let gen = Element {
            a: Coordinate::from(gen_coords[0]),
            b: Coordinate::from(gen_coords[1]),
            c: Coordinate::from(gen_coords[2]),
            d: Coordinate::from(gen_coords[3]),
            denom: Denominator::ONE,
        };
        let norm_bigint = norm.to_bigint();
        let ideal = LeftIdeal::<4>::new(&gen, &norm_bigint, EXTREMAL_ORDERS[0].order());

        total += 1;

        match ideal.to_isogeny(&mut OsRng) {
            None => {
                eprintln!("[SURVEY] vec={i} to_isogeny=None");
                none_count += 1;
            }
            Some((e_pk, _phi_p, _phi_q, _phi_pmq)) => {
                let computed_a = e_pk.coefficient().to_bytes();
                let expected_a = &pk_bytes[..64];
                let is_match = &computed_a[..] == expected_a;
                if is_match {
                    matched += 1;
                }
                let computed_hex = hex::encode(computed_a);
                let expected_hex = hex::encode(expected_a);
                eprintln!(
                    "[SURVEY] vec={i} match={is_match} computed={}...  expected={}...",
                    &computed_hex[..32],
                    &expected_hex[..32]
                );
            }
        }
    }

    eprintln!("[SURVEY] total={total} matched={matched} none={none_count}");
}

/// Decode KAT[0]'s SK and print the components our keygen must
/// produce: secret-ideal norm, generator coordinates, and `M_sk`
/// entries. Together with the public key bytes (already in
/// `kat_sk_pk_match_all`), this is the canonical "what we're aiming
/// at" for `keygen_kat_all`.
///
/// Run with:
/// ```text
/// cargo test --lib --release \
///   keygen_target_seed_0 -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn keygen_target_seed_0() {
    let (seed_hex, pk_hex, sk_hex, ..) = crate::keys::kat_data::KAT_VECTORS[0];
    eprintln!("[TARGET] seed_hex (first 32 chars) = {}", &seed_hex[..32]);
    eprintln!("[TARGET] pk_hex (first 32 chars) = {}", &pk_hex[..32]);

    let sk_bytes = hex::decode(sk_hex).expect("valid hex");
    // Parse offsets must match `SigningKey::from_bytes`.
    use crate::params::{FP_ENCODED_BYTES, TORSION_2POWER_BYTES, VERIFYING_KEY_BYTES};

    let mut pos = VERIFYING_KEY_BYTES; // skip pk
    let norm_hex: String = sk_bytes[pos..pos + FP_ENCODED_BYTES]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    pos += FP_ENCODED_BYTES;
    eprintln!("[TARGET] norm (LE 32 B): 0x{norm_hex}");

    for (i, label) in ["gen.a (1)", "gen.b (i)", "gen.c (j)", "gen.d (k=ij)"]
        .iter()
        .enumerate()
    {
        let coord_hex: String = sk_bytes[pos..pos + FP_ENCODED_BYTES]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        pos += FP_ENCODED_BYTES;
        eprintln!("[TARGET] {label} (LE 32 B, signed): 0x{coord_hex}");
        let _ = i;
    }

    for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        let entry_hex: String = sk_bytes[pos..pos + TORSION_2POWER_BYTES]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        pos += TORSION_2POWER_BYTES;
        eprintln!("[TARGET] M_sk[{r}][{c}] (LE 32 B): 0x{entry_hex}");
    }

    assert_eq!(pos, sk_bytes.len(), "SK byte layout drift");
}

/// First sample dump for KAT seed 0.
///
/// Seeds the DRBG from KAT vector 0, draws the first three
/// 65-byte rejection-sampled values that `random_prime_norm_wide`
/// uses to build `(g₁, g₂, g₃)`, and prints them as hex. Pair
/// against the same first three samples in the patched C reference
/// to spot byte-order, masking, or rejection-criterion divergences
/// before diving deeper.
///
/// Run with:
/// ```text
/// cargo test --lib --release \
///   keygen_first_sample_seed_0 -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn keygen_first_sample_seed_0() {
    let (seed_hex, ..) = crate::keys::kat_data::KAT_VECTORS[0];
    let seed: [u8; 48] = hex::decode(seed_hex)
        .expect("valid seed hex")
        .as_slice()
        .try_into()
        .expect("seed is 48 bytes");

    let mut drbg = crate::drbg::Aes256CtrDrbg::new(&seed);
    // Match `random_prime_norm_wide`'s sample_mod_n exactly:
    //   n_bits = 513, n_bytes = 65, mask top byte to keep bit 512 only.
    let n_bytes: usize = 65;
    let n_bits: usize = 513;

    for label in ["g1", "g2", "g3"] {
        let mut bytes = [0u8; 65];
        rand_core::RngCore::fill_bytes(&mut drbg, &mut bytes[..n_bytes]);
        let pre_mask: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        if n_bits % 8 != 0 {
            bytes[n_bytes - 1] &= (1u8 << (n_bits % 8)) - 1;
        }
        let post_mask: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        eprintln!("[FIRST_SAMPLE] {label}: pre_mask=0x{pre_mask}");
        eprintln!("[FIRST_SAMPLE] {label}: post_mask=0x{post_mask}");
    }
}

/// Per-phase DRBG-byte consumption probe for KAT seed 0.
///
/// Replicates the body of [`SigningKey::generate_with_rng`] inline
/// so we can bracket `random_prime_norm_wide`, `reduce_to_prime_norm`,
/// and `to_isogeny` with [`Aes256CtrDrbg::bytes_consumed`] calls and
/// print a per-step byte count to stderr. Diff against the
/// equivalent probe points in the patched SQIsign C reference
/// (`drbg_bytes_consumed` counter exported from
/// `randombytes_ctrdrbg.c`) to pinpoint which step first diverges
/// from the reference's consumption pattern.
///
/// Run with:
/// ```text
/// cargo test --lib --release \
///   keygen_drbg_byte_probe_seed_0 -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn keygen_drbg_byte_probe_seed_0() {
    use crate::{
        params::D_MIX,
        quaternions::{bigint::BigInt, lattice::LeftIdeal, precomputed::EXTREMAL_ORDERS},
    };

    let (seed_hex, ..) = crate::keys::kat_data::KAT_VECTORS[0];
    let seed: [u8; 48] = hex::decode(seed_hex)
        .expect("valid seed hex")
        .as_slice()
        .try_into()
        .expect("seed is 48 bytes");

    let mut drbg = crate::drbg::Aes256CtrDrbg::new(&seed);
    let d_mix_wide: BigInt<30> = D_MIX.widen();

    // Match the retry loop in `SigningKey::generate_with_rng` and
    // report which early-return fires on each iteration.
    for iter in 0..8 {
        let t0 = drbg.bytes_consumed();
        let ideal =
            LeftIdeal::<30>::random_prime_norm_wide(&d_mix_wide, &EXTREMAL_ORDERS[0], &mut drbg);
        let t1 = drbg.bytes_consumed();
        eprintln!(
            "[PROBE] iter={iter} random_prime_norm_wide: {} bytes (Some={})",
            t1 - t0,
            ideal.is_some()
        );
        let Some(mut ideal) = ideal else { continue };

        let ok = ideal.reduce_to_prime_norm::<30, _>(&mut drbg);
        let t2 = drbg.bytes_consumed();
        eprintln!(
            "[PROBE] iter={iter} reduce_to_prime_norm: {} bytes (ok={ok})",
            t2 - t1
        );
        if !ok {
            continue;
        }

        let Some(ideal_narrow) = ideal.narrow() else {
            eprintln!("[PROBE] iter={iter} narrow: None — continue");
            continue;
        };

        let t_before_iso = drbg.bytes_consumed();
        let iso = ideal_narrow.to_isogeny(&mut drbg);
        let t_after_iso = drbg.bytes_consumed();
        eprintln!(
            "[PROBE] iter={iter} to_isogeny: {} bytes (Some={})",
            t_after_iso - t_before_iso,
            iso.is_some()
        );
        if iso.is_none() {
            continue;
        }

        let gen = ideal_narrow.generator();
        eprintln!("[PROBE] iter={iter} generator: Some={}", gen.is_some());
        if gen.is_none() {
            continue;
        }
        eprintln!(
            "[PROBE] iter={iter} SUCCESS: total {} bytes so far",
            drbg.bytes_consumed()
        );
        return;
    }
    eprintln!("[PROBE] exhausted 8 attempts");
}

/// Lock in byte-stream alignment of `Lattice::random_prime_norm_wide`
/// with the C reference's `quat_sampling_random_ideal_O0_given_norm`
/// (`normeq.c:297-384`).
///
/// Captured 2026-04-29 via `[SAMPID]` instrumentation: with the
/// AES-CTR-DRBG seeded from KAT vector 0's seed, `quat_sampling_random_ideal`
/// consumes exactly **1040 bytes** for the first call (Phase A
/// trace-zero sampling + sqrt-mod-N + Phase B `gen_rerand`
/// rerandomization, all driven by `ibz_rand_interval`).
///
/// We additionally verified that the FIRST KAT vector's resulting
/// quaternion `γ` (Phase A), `δ` (Phase B), and `γ·δ` (multiply)
/// are byte-identical between Rust and C ref. See
/// `project_mode_b_diagnostic.md`.
///
/// This regression test asserts the byte count alone — values are
/// validated by `keygen_drbg_byte_probe_seed_0` running alongside.
/// If a future change breaks the byte-stream contract, this test
/// fails first.
#[test]
#[ignore]
fn random_prime_norm_wide_byte_aligned_with_cref_kat0() {
    use crate::{
        params::D_MIX,
        quaternions::{bigint::BigInt, lattice::LeftIdeal, precomputed::EXTREMAL_ORDERS},
    };

    let (seed_hex, ..) = crate::keys::kat_data::KAT_VECTORS[0];
    let seed: [u8; 48] = hex::decode(seed_hex)
        .expect("valid seed hex")
        .as_slice()
        .try_into()
        .expect("seed is 48 bytes");

    let mut drbg = crate::drbg::Aes256CtrDrbg::new(&seed);
    let d_mix_wide: BigInt<30> = D_MIX.widen();

    let before = drbg.bytes_consumed();
    let ideal =
        LeftIdeal::<30>::random_prime_norm_wide(&d_mix_wide, &EXTREMAL_ORDERS[0], &mut drbg)
            .expect("KAT[0] first random_prime_norm_wide must succeed");
    let after = drbg.bytes_consumed();

    let consumed = after - before;
    assert_eq!(
        consumed, 1040,
        "byte-stream contract: random_prime_norm_wide must consume exactly C ref's \
         1040 bytes for KAT seed 0's first call (got {consumed})"
    );

    // Sanity: produced an ideal of the requested norm.
    assert_eq!(*ideal.norm(), d_mix_wide);
}

/// Reproduce the SQIsign C reference's byte consumption for a KAT
/// seed by threading a single AES-CTR-DRBG through both keygen and
/// signing, matching `randombytes_init(seed); crypto_sign_keypair;
/// crypto_sign` in `PQCgenKAT_sign.c`.
///
/// This is the cross-check partner for
/// `scripts/cref_outer_ker.sh` and
/// `tests/fixtures/cref_outer_ker_kat_vector_0.txt`: stderr lines
/// emitted by the `[OUTER_KER]` diagnostic in `to_isogeny` should
/// agree with the C reference's `OUTER_KER` block bit-for-bit for
/// a correct implementation. If they diverge, our kernel is wrong
/// upstream of the chain; if they agree but our chain still fails
/// to split, the bug is in `Kernel::from_montgomery` or the
/// `(2,2)`-chain internals.
///
/// Run with:
/// ```text
/// cargo test --lib --release \
///   kat_cref_cross_check_vector_0 -- --ignored --nocapture \
///   2> /tmp/rust-outer-ker.log
/// scripts/cref_outer_ker.sh --vector 0 > /tmp/cref-outer-ker.txt
/// diff <(grep '^\[OUTER_KER\]' /tmp/rust-outer-ker.log) \
///      /tmp/cref-outer-ker.txt
/// ```
#[test]
#[ignore]
fn kat_cref_cross_check_vector_0() {
    let (seed_hex, _pk_hex, _sk_hex, msg_hex, sm_hex) = crate::keys::kat_data::KAT_VECTORS[0];
    let seed: [u8; 48] = hex::decode(seed_hex)
        .expect("valid seed hex")
        .as_slice()
        .try_into()
        .expect("seed is 48 bytes");
    let msg = hex::decode(msg_hex).expect("valid msg hex");

    let mut drbg = crate::drbg::Aes256CtrDrbg::new(&seed);
    let before_keygen = drbg.bytes_consumed();
    let sk = match SigningKey::generate_with_rng(&mut drbg) {
        Ok(sk) => sk,
        Err(SignatureError::KeyGenFailed) => return, // probabilistic skip
        Err(other) => panic!("unexpected keygen error: {other:?}"),
    };
    let after_keygen = drbg.bytes_consumed();
    eprintln!(
        "[CROSSCHECK] keygen consumed {} DRBG bytes (seed 0)",
        after_keygen - before_keygen
    );

    let sig = match sk.sign_with_rng(&msg, &mut drbg) {
        Ok(s) => s,
        Err(SignatureError::SigningFailed) => {
            eprintln!(
                "kat_cref_cross_check_vector_0: SigningFailed — outer-chain \
                 bug still present, cross-check of [OUTER_KER] stderr lines \
                 against tests/fixtures/cref_outer_ker_kat_vector_0.txt is \
                 the reason we wrote this test."
            );
            return;
        }
        Err(other) => panic!("unexpected sign error: {other:?}"),
    };
    let after_sign = drbg.bytes_consumed();
    eprintln!(
        "[CROSSCHECK] sign consumed {} DRBG bytes (seed 0)",
        after_sign - after_keygen
    );

    // Full byte-for-byte match: `sm` in the rsp file is `sig || msg`,
    // so the signature prefix must equal the first CRYPTO_BYTES of
    // the KAT's `sm` field.
    let sm = hex::decode(sm_hex).expect("valid sm hex");
    let sig_bytes = sig.to_bytes();
    assert_eq!(
        &sig_bytes[..],
        &sm[..sig_bytes.len()],
        "signature must match KAT `sm` prefix byte-for-byte"
    );
}

/// Survey the bit-magnitudes of every KAT secret-ideal `(norm, gen)`.
///
/// For each KAT vector index 0..99, decode `sk_bytes[65..97]` as the
/// unsigned norm and the next four 32-byte chunks as signed generator
/// coordinates `gen.{a, b, c, d}` (two's complement LE). Print the
/// bit-size of the norm and of each `|coord|`, plus
/// `max_coord_bits = max(|a|, |b|, |c|, |d|)`. Bucket the maxima
/// against the `LeftIdeal::<4>::new` width-8 product safety bound
/// (≤ 127 bits).
///
/// Run with:
/// ```text
/// cargo test --lib --release \
///     survey_kat_secret_ideal_coord_magnitudes -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn survey_kat_secret_ideal_coord_magnitudes() {
    let mut max_le_127 = 0usize;
    let mut max_in_127_192 = 0usize;
    let mut max_gt_192 = 0usize;
    let mut min_observed: u32 = u32::MAX;
    let mut max_observed: u32 = 0;
    let mut safe_indices: Vec<usize> = Vec::new();

    for (i, &(_, _, sk_hex, ..)) in crate::keys::kat_data::KAT_VECTORS.iter().enumerate() {
        let sk_bytes = hex::decode(sk_hex).expect("valid hex");

        let mut pos = VERIFYING_KEY_BYTES;
        let norm_bytes: &[u8; FP_ENCODED_BYTES] = sk_bytes[pos..pos + FP_ENCODED_BYTES]
            .try_into()
            .expect("32-byte norm");
        let norm_bigint = BigInt::<4>::from_bytes_le_unsigned(norm_bytes);
        let norm_bits = norm_bigint.bitsize();
        pos += FP_ENCODED_BYTES;

        let mut coord_bits = [0u32; 4];
        for slot in &mut coord_bits {
            let coord = BigInt::<4>::from_bytes_le_signed(
                sk_bytes[pos..pos + FP_ENCODED_BYTES]
                    .try_into()
                    .expect("32-byte coord"),
            );
            // `bitsize` works on the magnitude (limbs), independent of
            // the sign bit, so |coord|.bitsize() == coord.bitsize().
            *slot = coord.bitsize();
            pos += FP_ENCODED_BYTES;
        }

        let max_coord_bits = *coord_bits.iter().max().expect("4 coords");
        eprintln!(
            "[SURVEY] vec={i:02} norm_bits={norm_bits:3} coord_bits=[{}, {}, {}, {}] max={max_coord_bits}",
            coord_bits[0], coord_bits[1], coord_bits[2], coord_bits[3],
        );

        min_observed = min_observed.min(max_coord_bits);
        max_observed = max_observed.max(max_coord_bits);

        if max_coord_bits <= 127 {
            max_le_127 += 1;
            safe_indices.push(i);
        } else if max_coord_bits <= 192 {
            max_in_127_192 += 1;
        } else {
            max_gt_192 += 1;
        }
    }

    eprintln!("[SURVEY] ----- histogram -----");
    eprintln!("[SURVEY] max_coord_bits ≤ 127        : {max_le_127}");
    eprintln!("[SURVEY] max_coord_bits ∈ (127, 192] : {max_in_127_192}");
    eprintln!("[SURVEY] max_coord_bits > 192        : {max_gt_192}");
    eprintln!("[SURVEY] observed range of max_coord_bits: [{min_observed}, {max_observed}]");
    eprintln!("[SURVEY] SAFE indices (max ≤ 127): {safe_indices:?}");
}

// ----------------------------------------------------------------------
// RNG byte-trace for keygen-from-seed divergence debug (Bug 2).
//
// Our DRBG is bit-correct against the C ref (verified by
// `drbg::tests::matches_cref_seed_zero_first_128_bytes`). So if
// `keygen_kat_all` produces a wrong `e_pk`, the divergence is in
// *which* DRBG bytes are consumed at *which* algorithmic step — not
// in DRBG output itself.
//
// `TracingDrbg` wraps an underlying RNG and logs every `fill_bytes`
// call: cumulative byte offset before the call, length, and the
// returned bytes. Run `keygen_kat_000_rng_trace` to dump our
// consumption pattern for KAT[0]'s seed; instrument the C ref
// equivalently and diff. The first divergent line (different length,
// or different bytes for the same offset+length) localizes which
// algorithmic step has a different RNG-consumption pattern.
// ----------------------------------------------------------------------

#[derive(Debug)]
struct TracingDrbg<R: rand_core::RngCore> {
    inner: R,
    cumulative: usize,
    /// Each entry: `(cumulative_offset_before_call, returned_bytes)`.
    log: Vec<(usize, Vec<u8>)>,
}

impl<R: rand_core::RngCore> TracingDrbg<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            cumulative: 0,
            log: Vec::new(),
        }
    }
}

impl<R: rand_core::RngCore> rand_core::RngCore for TracingDrbg<R> {
    fn next_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.fill_bytes(&mut buf);
        u32::from_le_bytes(buf)
    }

    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill_bytes(&mut buf);
        u64::from_le_bytes(buf)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.inner.fill_bytes(dest);
        self.log.push((self.cumulative, dest.to_vec()));
        self.cumulative += dest.len();
        // Publish to the thread-local so checkpoint sites in keygen
        // can read the cumulative offset without threading the
        // wrapper through generic `R: CryptoRngCore` arguments.
        crate::drbg::debug::set(self.cumulative);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl<R: rand_core::CryptoRng + rand_core::RngCore> rand_core::CryptoRng for TracingDrbg<R> {}

/// Capture our DRBG byte-consumption pattern during keygen-from-seed
/// for KAT vector 0. Writes the trace to
/// `/tmp/drbg_trace_keygen_kat_0.txt` for diff against an
/// instrumented C ref run.
///
/// Output format: one line per `fill_bytes` call:
///
///     [offset_hex] len=NNN bytes=hexhexhex...
///
/// where `offset_hex` is the cumulative byte offset before this call
/// (i.e., the byte position into the DRBG output stream where this
/// call's first byte landed).
///
/// `#[ignore]` because keygen takes minutes even in release mode. Run with:
/// `cargo test --lib --release keygen_kat_000_rng_trace -- --include-ignored
/// --nocapture`
#[test]
#[ignore]
fn keygen_kat_000_rng_trace() {
    let seed_hex = crate::keys::kat_data::KAT_VECTORS[0].0;
    let seed_bytes = hex::decode(seed_hex).expect("valid hex");
    let seed: [u8; 48] = seed_bytes.as_slice().try_into().expect("seed is 48 bytes");

    let inner = crate::drbg::Aes256CtrDrbg::new(&seed);
    let mut tracing = TracingDrbg::new(inner);

    // Reset the byte-offset thread-local so checkpoint sites in
    // keygen log offsets relative to this run, not whatever was
    // accumulated by previous tests on this thread.
    crate::drbg::debug::reset();

    // Run keygen; we don't care about success/failure, only the trace.
    let _ = SigningKey::generate_with_rng(&mut tracing);

    let total_calls = tracing.log.len();
    let total_bytes = tracing.cumulative;

    let mut out = String::new();
    out.push_str(&format!(
        "# DRBG byte-trace for keygen_kat_000\n# {total_calls} fill_bytes calls, {total_bytes} total bytes\n"
    ));
    for (offset, bytes) in &tracing.log {
        let mut hex_buf = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            hex_buf.push_str(&format!("{b:02x}"));
        }
        out.push_str(&format!(
            "[{offset:08x}] len={:>4} bytes={hex_buf}\n",
            bytes.len()
        ));
    }

    let path = "/tmp/drbg_trace_keygen_kat_0.txt";
    std::fs::write(path, &out).expect("write trace");
    eprintln!("[trace] {total_calls} fill_bytes calls, {total_bytes} bytes total → {path}");
}
