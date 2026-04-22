use sqisign_selkie::fields::{fp::Fp, fp2::Fp2};

fn main() {
    divan::main();
}

#[divan::bench]
fn fp_mul(bencher: divan::Bencher) {
    let a = Fp::from_small(17);
    let b = Fp::from_small(42);
    bencher.bench(|| divan::black_box(a) * divan::black_box(b));
}

#[divan::bench]
fn fp_square(bencher: divan::Bencher) {
    let a = Fp::from_small(17);
    bencher.bench(|| divan::black_box(&a).square());
}

#[divan::bench]
fn fp_invert(bencher: divan::Bencher) {
    let a = Fp::from_small(17);
    bencher.bench(|| divan::black_box(&a).invert());
}

#[divan::bench]
fn fp_sqrt(bencher: divan::Bencher) {
    let a = Fp::from_small(17).square();
    bencher.bench(|| divan::black_box(&a).sqrt());
}

#[divan::bench]
fn fp2_mul(bencher: divan::Bencher) {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let b = Fp2::new(Fp::from_small(11), Fp::from_small(13));
    bencher.bench(|| divan::black_box(a) * divan::black_box(b));
}

#[divan::bench]
fn fp2_square(bencher: divan::Bencher) {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    bencher.bench(|| divan::black_box(&a).square());
}

#[divan::bench]
fn fp2_invert(bencher: divan::Bencher) {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    bencher.bench(|| divan::black_box(&a).invert());
}

#[divan::bench]
fn fp2_sqrt(bencher: divan::Bencher) {
    let a = Fp2::new(Fp::from_small(3), Fp::from_small(7)).square();
    bencher.bench(|| divan::black_box(&a).sqrt());
}
