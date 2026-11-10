//! Symbol shims for the `llvm-mca` static-analysis tool, not backend
//! code.
//!
//! These exist only so `.github/scripts/mca.rs` can point `llvm-mca` at
//! the hot field kernels.  Those kernels are `#[inline]`, so they fold
//! into callers and leave no standalone symbol in `--emit asm` output.
//! Each wrapper here is `#[inline(never)] #[no_mangle] extern "C"`,
//! emitting one clean `mca_<name>` symbol whose body is the kernel plus
//! a tiny prologue/epilogue; the tool slices that symbol out and feeds
//! it to `llvm-mca`.
//!
//! The whole module is gated behind the `mca` feature (off by default)
//! and never compiles into a normal build.  Inputs are raw Montgomery
//! limbs passed through [`core::hint::black_box`] so the optimizer
//! can't fold the kernel away; the limbs aren't validated, since the
//! pipeline model only cares about the instruction stream.  The `Fp64`
//! shims need `bmi2` + `adx` (the MULX + ADCX/ADOX kernels); the `Fp55`
//! shims are always available on x86_64.
//!
//! Raw-array-by-value signatures aren't FFI-safe, but these are never
//! called across an FFI boundary, so the lint is suppressed
//! module-wide.
#![allow(improper_ctypes_definitions)]

use crate::fields::fp::arch::generic::Fp55;

/// `Fp55` (radix-55) Montgomery squaring.
#[no_mangle]
#[inline(never)]
pub extern "C" fn mca_fp55_square(a: [u64; 6]) -> [u64; 6] {
    let a = core::hint::black_box(Fp55::from_limbs(a));
    a.square().0
}

/// `Fp55` (radix-55) Montgomery multiplication.
#[no_mangle]
#[inline(never)]
pub extern "C" fn mca_fp55_mul(a: [u64; 6], b: [u64; 6]) -> [u64; 6] {
    let a = core::hint::black_box(Fp55::from_limbs(a));
    let b = core::hint::black_box(Fp55::from_limbs(b));
    (&a * &b).0
}

/// `Fp64` shims, gated on `bmi2` + `adx` (the MULX + ADCX/ADOX kernels
/// these wrap).
#[cfg(all(target_feature = "bmi2", target_feature = "adx"))]
mod adx {
    use crate::fields::fp::arch::x86_64::mulx_adx::Fp64;

    /// `Fp64` (radix-2^64) MULX + ADCX/ADOX Montgomery multiplication.
    #[no_mangle]
    #[inline(never)]
    pub extern "C" fn mca_fp64_mul(a: [u64; 6], b: [u64; 6]) -> [u64; 6] {
        let a = core::hint::black_box(Fp64::from_raw(a));
        let b = core::hint::black_box(Fp64::from_raw(b));
        (a * b).0
    }

    /// `Fp64` (radix-2^64) Montgomery squaring.
    #[no_mangle]
    #[inline(never)]
    pub extern "C" fn mca_fp64_square(a: [u64; 6]) -> [u64; 6] {
        let a = core::hint::black_box(Fp64::from_raw(a));
        a.square().0
    }

    /// `Fp64` fused two-product kernel, the per-coordinate `Fp2` product.
    #[no_mangle]
    #[inline(never)]
    pub extern "C" fn mca_fp64_sum_of_2_products(
        a1: [u64; 6],
        b1: [u64; 6],
        a2: [u64; 6],
        b2: [u64; 6],
    ) -> [u64; 6] {
        let a1 = core::hint::black_box(Fp64::from_raw(a1));
        let b1 = core::hint::black_box(Fp64::from_raw(b1));
        let a2 = core::hint::black_box(Fp64::from_raw(a2));
        let b2 = core::hint::black_box(Fp64::from_raw(b2));
        Fp64::sum_of_2_products(&a1, &b1, &a2, &b2).0
    }
}
