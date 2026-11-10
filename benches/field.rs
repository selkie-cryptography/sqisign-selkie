#[cfg(target_arch = "aarch64")]
use sqisign_selkie::fields::fp::arch::aarch64::neon::{Fp29, Fp29x4};
#[cfg(target_arch = "x86_64")]
use sqisign_selkie::fields::fp::arch::x86_64::avx2::Fp26;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use sqisign_selkie::fields::fp::arch::x86_64::avx2::Fp26x4;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
use sqisign_selkie::fields::fp::arch::x86_64::mulx_adx::Fp64;
use sqisign_selkie::{
    fields::{
        fp::{Fp, arch::generic::Fp55},
        fp2::Fp2,
    },
    params::FP_ENCODED_BYTES,
};

fn main() {
    divan::main();
}

/// A canonical (< p) field-element encoding filled with `fill`.
fn fp_bytes(fill: u8, top: u8) -> [u8; FP_ENCODED_BYTES] {
    let mut b = [fill; FP_ENCODED_BYTES];
    b[FP_ENCODED_BYTES - 1] = top;
    b
}

fn fp_bytes_a() -> [u8; FP_ENCODED_BYTES] {
    fp_bytes(17, 0x11)
}

fn fp_bytes_b() -> [u8; FP_ENCODED_BYTES] {
    fp_bytes(42, 0x22)
}

fn fp_a() -> Fp {
    Fp::from_bytes(&fp_bytes_a())
}

fn fp_b() -> Fp {
    Fp::from_bytes(&fp_bytes_b())
}

// --- Fp ---

#[divan::bench]
fn fp_add(bencher: divan::Bencher) {
    let (a, b) = (fp_a(), fp_b());
    bencher.bench(|| divan::black_box(&a) + divan::black_box(&b));
}

#[divan::bench]
fn fp_sub(bencher: divan::Bencher) {
    let (a, b) = (fp_a(), fp_b());
    bencher.bench(|| divan::black_box(&a) - divan::black_box(&b));
}

#[divan::bench]
fn fp_mul(bencher: divan::Bencher) {
    let (a, b) = (fp_a(), fp_b());
    bencher.bench(|| divan::black_box(&a) * divan::black_box(&b));
}

#[divan::bench]
fn fp_square(bencher: divan::Bencher) {
    let a = fp_a();
    bencher.bench(|| divan::black_box(&a).square());
}

#[divan::bench]
fn fp_sum_of_2_products(bencher: divan::Bencher) {
    let (a, b) = (fp_a(), fp_b());
    bencher.bench(|| {
        Fp::sum_of_2_products(
            divan::black_box(&a),
            divan::black_box(&b),
            divan::black_box(&b),
            divan::black_box(&a),
        )
    });
}

#[divan::bench]
fn fp_invert(bencher: divan::Bencher) {
    let a = fp_a();
    bencher.bench(|| divan::black_box(&a).invert());
}

#[divan::bench]
fn fp_sqrt(bencher: divan::Bencher) {
    let a = fp_a().square();
    bencher.bench(|| divan::black_box(&a).sqrt());
}

#[divan::bench]
fn fp_is_square(bencher: divan::Bencher) {
    let a = fp_a();
    bencher.bench(|| divan::black_box(&a).is_square());
}

#[divan::bench]
fn fp_to_bytes(bencher: divan::Bencher) {
    let a = fp_a();
    bencher.bench(|| divan::black_box(a).to_bytes());
}

#[divan::bench]
fn fp_from_bytes(bencher: divan::Bencher) {
    let bytes = fp_a().to_bytes();
    bencher.bench(|| Fp::from_bytes(divan::black_box(&bytes)));
}

// --- Fp2 ---

fn fp2_a() -> Fp2 {
    Fp2::new(fp_a(), fp_b())
}

fn fp2_b() -> Fp2 {
    Fp2::new(fp_b(), fp_a())
}

#[divan::bench]
fn fp2_add(bencher: divan::Bencher) {
    let (a, b) = (fp2_a(), fp2_b());
    bencher.bench(|| divan::black_box(&a) + divan::black_box(&b));
}

#[divan::bench]
fn fp2_mul(bencher: divan::Bencher) {
    let (a, b) = (fp2_a(), fp2_b());
    bencher.bench(|| divan::black_box(&a) * divan::black_box(&b));
}

#[divan::bench]
fn fp2_square(bencher: divan::Bencher) {
    let a = fp2_a();
    bencher.bench(|| divan::black_box(&a).square());
}

#[divan::bench]
fn fp2_invert(bencher: divan::Bencher) {
    let a = fp2_a();
    bencher.bench(|| divan::black_box(&a).invert());
}

#[divan::bench]
fn fp2_sqrt(bencher: divan::Bencher) {
    let a = fp2_a().square();
    bencher.bench(|| divan::black_box(&a).sqrt());
}

#[divan::bench]
fn fp2_is_square(bencher: divan::Bencher) {
    let a = fp2_a();
    bencher.bench(|| divan::black_box(&a).is_square());
}

// --- Explicit backend benches ---
//
// `fp_*` above measures the *active* `Fp`, whichever backend the
// dispatcher picked.  The benches below hit each backend explicitly on
// the same hardware so the per-op delta between layouts is visible.

#[divan::bench]
fn fp55_mul(bencher: divan::Bencher) {
    let a = Fp55::from_bytes(&fp_bytes_a());
    let b = Fp55::from_bytes(&fp_bytes_b());
    bencher.bench(|| divan::black_box(&a) * divan::black_box(&b));
}

#[divan::bench]
fn fp55_square(bencher: divan::Bencher) {
    let a = Fp55::from_bytes(&fp_bytes_a());
    bencher.bench(|| divan::black_box(&a).square());
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
#[divan::bench]
fn fp64_mul(bencher: divan::Bencher) {
    let a = Fp64::from_bytes(&fp_bytes_a());
    let b = Fp64::from_bytes(&fp_bytes_b());
    bencher.bench(|| divan::black_box(a) * divan::black_box(b));
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
#[divan::bench]
fn fp64_square(bencher: divan::Bencher) {
    let a = Fp64::from_bytes(&fp_bytes_a());
    bencher.bench(|| divan::black_box(&a).square());
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
#[divan::bench]
fn fp64_sum_of_2_products(bencher: divan::Bencher) {
    let a = Fp64::from_bytes(&fp_bytes_a());
    let b = Fp64::from_bytes(&fp_bytes_b());
    bencher.bench(|| {
        Fp64::sum_of_2_products(
            divan::black_box(&a),
            divan::black_box(&b),
            divan::black_box(&b),
            divan::black_box(&a),
        )
    });
}

/// Four independent scalar `Fp::mul`s: the baseline a batch path must beat.
#[divan::bench]
fn fp_mul_4_independent(bencher: divan::Bencher) {
    let a = [fp_a(), fp_b(), fp_a().square(), fp_b().square()];
    let b = [fp_b(), fp_a(), fp_b().square(), fp_a().square()];
    bencher.bench(|| {
        let aa = divan::black_box(&a);
        let bb = divan::black_box(&b);
        [aa[0] * bb[0], aa[1] * bb[1], aa[2] * bb[2], aa[3] * bb[3]]
    });
}

#[cfg(target_arch = "aarch64")]
#[divan::bench]
fn fp29_mul_scalar(bencher: divan::Bencher) {
    let a = Fp29::from_bytes(&fp_bytes_a());
    let b = Fp29::from_bytes(&fp_bytes_b());
    bencher.bench(|| divan::black_box(&a) * divan::black_box(&b));
}

/// One `Fp29x4::mul`: four independent products in one NEON
/// Karatsuba-decomposed Montgomery multiplication.
#[cfg(target_arch = "aarch64")]
#[divan::bench]
fn fp29x4_mul_neon(bencher: divan::Bencher) {
    let a = Fp29::from_bytes(&fp_bytes_a());
    let b = Fp29::from_bytes(&fp_bytes_b());
    let a4 = Fp29x4::from_scalars(&[a, b, a.square(), b.square()]);
    let b4 = Fp29x4::from_scalars(&[b, a, b.square(), a.square()]);
    bencher.bench(|| divan::black_box(&a4).mul(divan::black_box(&b4)));
}

#[cfg(target_arch = "aarch64")]
#[divan::bench]
fn fp29x4_square_neon(bencher: divan::Bencher) {
    let a = Fp29::from_bytes(&fp_bytes_a());
    let b = Fp29::from_bytes(&fp_bytes_b());
    let a4 = Fp29x4::from_scalars(&[a, b, a.square(), b.square()]);
    bencher.bench(|| divan::black_box(&a4).square());
}

#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn fp26_mul_scalar(bencher: divan::Bencher) {
    let a = Fp26::from_bytes(&fp_bytes_a());
    let b = Fp26::from_bytes(&fp_bytes_b());
    bencher.bench(|| divan::black_box(&a) * divan::black_box(&b));
}

/// One `Fp26x4::mul`: four independent Montgomery products in one AVX2
/// schoolbook CIOS.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[divan::bench]
fn fp26x4_mul_avx2(bencher: divan::Bencher) {
    let a = Fp26::from_bytes(&fp_bytes_a());
    let b = Fp26::from_bytes(&fp_bytes_b());
    let a4 = Fp26x4::from_scalars(&[a, b, a.square(), b.square()]);
    let b4 = Fp26x4::from_scalars(&[b, a, b.square(), a.square()]);
    bencher.bench(|| divan::black_box(&a4).mul(divan::black_box(&b4)));
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[divan::bench]
fn fp26x4_square_avx2(bencher: divan::Bencher) {
    let a = Fp26::from_bytes(&fp_bytes_a());
    let b = Fp26::from_bytes(&fp_bytes_b());
    let a4 = Fp26x4::from_scalars(&[a, b, a.square(), b.square()]);
    bencher.bench(|| divan::black_box(&a4).square());
}
