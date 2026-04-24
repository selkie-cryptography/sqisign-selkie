//! Allocation budget tests.
//!
//! Measures heap allocations per operation and asserts they stay
//! within a budget. Writes `alloc-report.json` for the CI dashboard.
//!
//! Run with: `cargo test --test alloc_budget --features expose-internals`

use dhat::Profiler;
use sqisign_selkie::{SIGNATURE_BYTES, Signature, VERIFYING_KEY_BYTES, VerifyingKey};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

struct OpResult {
    name: &'static str,
    allocs: u64,
    bytes: u64,
    budget: u64,
}

fn measure(name: &'static str, budget: u64, f: impl FnOnce()) -> OpResult {
    let _profiler = Profiler::builder().testing().build();
    f();
    let stats = dhat::HeapStats::get();
    let allocs = stats.total_blocks;
    let bytes = stats.total_bytes;
    eprintln!("{name}: {allocs} allocs, {bytes} bytes (budget: {budget})");
    assert!(
        allocs <= budget,
        "{name}: {allocs} allocations exceeds budget of {budget}",
    );
    OpResult {
        name,
        allocs,
        bytes,
        budget,
    }
}

/// Single test that measures all operations and writes the report.
#[test]
fn alloc_budgets() {
    let mut results = Vec::new();

    // vk_parse
    let vk_bytes = [0u8; VERIFYING_KEY_BYTES];
    results.push(measure("vk_parse", 50, || {
        let _ = VerifyingKey::from_bytes(&vk_bytes);
    }));

    // sig_parse
    let sig_bytes = [0u8; SIGNATURE_BYTES];
    results.push(measure("sig_parse", 20, || {
        let _ = Signature::from_bytes(&sig_bytes);
    }));

    // verify
    let pk_hex = "07CCD21425136F6E865E497D2D4D208F0054AD81372066E817480787AAF7B2029550C89E892D618CE3230F23510BFBE68FCCDDAEA51DB1436B462ADFAF008A010B";
    let sm_hex = "84228651F271B0F39F2F19F2E8718F31ED3365AC9E5CB303AFE663D0CFC11F0455D891B0CA6C7E653F9BA2667730BB77BEFE1B1A31828404284AF8FD7BAACC010001D974B5CA671FF65708D8B462A5A84A1443EE9B5FED7218767C9D85CEED04DB0A69A2F6EC3BE835B3B2624B9A0DF68837AD00BCACC27D1EC806A44840267471D86EFF3447018ADB0A6551EE8322AB30010202D81C4D8D734FCBFBEADE3D3F8A039FAA2A2C9957E835AD55B22E75BF57BB556AC8";
    let pk_bytes = hex::decode(pk_hex).unwrap();
    let sm_bytes = hex::decode(sm_hex).unwrap();
    let sig_arr: &[u8; SIGNATURE_BYTES] = sm_bytes[..SIGNATURE_BYTES].try_into().unwrap();
    let msg = &sm_bytes[SIGNATURE_BYTES..];
    let vk = VerifyingKey::from_bytes(pk_bytes.as_slice().try_into().unwrap()).unwrap();
    let sig = Signature::from_bytes(sig_arr).unwrap();
    results.push(measure("verify", 500, || {
        let _ = vk.verify(msg, &sig);
    }));

    // keygen (deterministic from KAT seed)
    results.push(measure("keygen", 5000, || {
        let seed_hex = "061550234D158C5EC95595FE04EF7A25767F2E24CC2BC479D09D86DC9ABCFDE7056A8C266F9EF97ED08541DBD2E1FFA1";
        let seed_bytes = hex::decode(seed_hex).unwrap();
        let seed: [u8; 48] = seed_bytes.as_slice().try_into().unwrap();
        let _ = sqisign_selkie::SigningKey::generate_derand(&seed);
    }));

    // sign (with parsed KAT key)
    let sk_hex = "07CCD21425136F6E865E497D2D4D208F0054AD81372066E817480787AAF7B2029550C89E892D618CE3230F23510BFBE68FCCDDAEA51DB1436B462ADFAF008A010B19943116DB5B4552B05B174969C61C9C8701000000000000000000000000000094F28A5533DF8872E3C7EFE3D45A175A0CFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFF1959E3D67EADD79948DB766D9FFAF4D3FFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000000000000000000000000000000000000000000000000000358A8756E1CA2E31C2F3C879414AC08DF7EA0C1D732F9AE3D1AC4644E524340095A4F53D286FDE8A7226CE960C152C344888C963A457B02CAECA41C2672D76000365B548FB9C9E6C0E149BABA3EC7BC33B8F052B6B9D4F840A2AD67221C8F600464B9862D34ADF4D562F3836EBEFC4D8F874351B3E63A4DF9D33C0BBF9EB1800";
    let sk_bytes2 = hex::decode(sk_hex).unwrap();
    let sk =
        sqisign_selkie::SigningKey::from_bytes(sk_bytes2.as_slice().try_into().unwrap()).unwrap();
    results.push(measure("sign", 5000, || {
        let seed = [0x42u8; 48];
        let _ = sk.sign_derand(b"alloc budget test", &seed);
    }));

    // Write JSON report.
    use std::io::Write;
    if let Ok(mut f) = std::fs::File::create("alloc-report.json") {
        let _ = write!(f, "{{\"operations\":[");
        for (i, r) in results.iter().enumerate() {
            if i > 0 {
                let _ = write!(f, ",");
            }
            let _ = write!(
                f,
                "{{\"name\":\"{}\",\"allocs\":{},\"bytes\":{},\"budget\":{}}}",
                r.name, r.allocs, r.bytes, r.budget
            );
        }
        let _ = write!(f, "]}}");
    }
}
