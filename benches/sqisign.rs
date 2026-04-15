use sqisign_selkie::{SIGNATURE_BYTES, Signature, SigningKey, VerifyingKey};

fn main() {
    divan::main();
}

const KAT0_PK: &str = "07CCD21425136F6E865E497D2D4D208F0054AD81372066E817480787AAF7B2029550C89E892D618CE3230F23510BFBE68FCCDDAEA51DB1436B462ADFAF008A010B";
const KAT0_SM: &str = "84228651F271B0F39F2F19F2E8718F31ED3365AC9E5CB303AFE663D0CFC11F0455D891B0CA6C7E653F9BA2667730BB77BEFE1B1A31828404284AF8FD7BAACC010001D974B5CA671FF65708D8B462A5A84A1443EE9B5FED7218767C9D85CEED04DB0A69A2F6EC3BE835B3B2624B9A0DF68837AD00BCACC27D1EC806A44840267471D86EFF3447018ADB0A6551EE8322AB30010202D81C4D8D734FCBFBEADE3D3F8A039FAA2A2C9957E835AD55B22E75BF57BB556AC8";

fn parse_kat() -> (VerifyingKey, Signature, Vec<u8>) {
    let pk_bytes = hex::decode(KAT0_PK).unwrap();
    let sm_bytes = hex::decode(KAT0_SM).unwrap();
    let sig_bytes: &[u8; SIGNATURE_BYTES] = sm_bytes[..SIGNATURE_BYTES].try_into().unwrap();
    let msg = sm_bytes[SIGNATURE_BYTES..].to_vec();
    let vk = VerifyingKey::from_bytes(pk_bytes.as_slice().try_into().unwrap()).unwrap();
    let sig = Signature::from_bytes(sig_bytes).unwrap();
    (vk, sig, msg)
}

#[divan::bench]
fn verify(bencher: divan::Bencher) {
    let (vk, sig, msg) = parse_kat();
    bencher.bench(|| vk.verify(divan::black_box(&msg), divan::black_box(&sig)));
}

#[divan::bench(sample_count = 10)]
fn keygen(bencher: divan::Bencher) {
    bencher.bench(|| {
        let mut rng = rand_core::OsRng;
        SigningKey::generate(&mut rng)
    });
}

#[divan::bench(sample_count = 10)]
fn sign(bencher: divan::Bencher) {
    let sk = SigningKey::generate(&mut rand_core::OsRng).unwrap();
    let msg = b"benchmark message";
    bencher.bench(|| {
        let mut rng = rand_core::OsRng;
        sk.sign(divan::black_box(msg.as_slice()), &mut rng)
    });
}
