//! Build script for sqisign-selkie.
//!
//! When the `ffi-cref-lll` feature is enabled, compiles a small C shim
//! that exposes mini-GMP and DPE primitives from the SQIsign C
//! reference for bit-equality property testing of our pure-Rust LLL
//! port. The C ref source tree must be available; default location is
//! `~/src/github.com/SQISign/the-sqisign`, override with
//! `SQISIGN_C_REF_PATH`.
//!
//! When the feature is off (the common case), this script is a no-op
//! and we have zero C dependencies.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_FFI_CREF_LLL");
    println!("cargo:rerun-if-env-changed=SQISIGN_C_REF_PATH");

    if std::env::var_os("CARGO_FEATURE_FFI_CREF_LLL").is_none() {
        return;
    }

    let cref = std::env::var("SQISIGN_C_REF_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("HOME must be set");
        format!("{home}/src/github.com/SQISign/the-sqisign")
    });
    let cref = std::path::PathBuf::from(cref);
    println!("cargo:rerun-if-changed={}", cref.display());

    let mini_gmp = cref.join("src/mini-gmp/mini-gmp.c");
    let mini_gmp_extra = cref.join("src/mini-gmp/mini-gmp-extra.c");
    let mini_gmp_inc = cref.join("src/mini-gmp");
    let dpe_inc = cref.join("src/quaternion/ref/generic/internal_quaternion_headers");
    let tutil_inc = cref.join("src/common/generic/include");

    for p in [&mini_gmp, &mini_gmp_extra] {
        assert!(
            p.exists(),
            "ffi-cref-lll: required C source missing: {}",
            p.display()
        );
    }

    // mini-gmp.c expects GMP_LIMB_BITS as a compile-time define
    // (matching the_sqisign's gmpconfig.cmake which sets it from
    // sizeof(mp_limb_t) * 8). We hard-code 64 since both Selkie and
    // C ref target 64-bit limbs on aarch64-darwin / x86_64-linux-gnu;
    // assert here to fail loudly if someone enables this on a 32-bit
    // build that mini-gmp would treat differently.
    assert_eq!(
        std::mem::size_of::<usize>(),
        8,
        "ffi-cref-lll currently assumes 64-bit limbs (GMP_LIMB_BITS=64); 32-bit targets need a different config"
    );

    cc::Build::new()
        .define("MINI_GMP", None)
        .define("GMP_LIMB_BITS", "64")
        .define("RADIX_64", None) // for tutil.h
        .include(&mini_gmp_inc)
        .include(&dpe_inc)
        .include(&tutil_inc)
        .include("src/ffi") // for our shim header
        .file(&mini_gmp)
        .file(&mini_gmp_extra)
        .file("src/ffi/cref_dpe_shim.c")
        .warnings(false) // mini-gmp emits a few benign warnings on macOS
        .compile("cref_lll_ffi");

    println!("cargo:rustc-link-lib=static=cref_lll_ffi");
}
