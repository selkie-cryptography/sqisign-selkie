//! Symbol shims for the `llvm-mca` static-analysis tool -- NOT backend
//! code.
//!
//! These exist only so [`.github/scripts/mca.rs`] can point `llvm-mca`
//! at the hot field kernels.  Those kernels ([`Fp64::mul_montgomery`],
//! [`Fp64::square`], the [`Fp2`] product) are `#[inline]`, so they fold
//! into callers and leave no standalone symbol in `--emit asm` output --
//! invisible to a static pipeline model.  Each wrapper here is
//! `#[inline(never)] #[no_mangle] extern "C"`, emitting one clean
//! `mca_<name>` symbol whose body is the kernel plus a tiny prologue/
//! epilogue; the tool slices that symbol out and feeds it to `llvm-mca`.
//!
//! The whole module is gated behind the `mca` feature (off by default)
//! and never compiles into a normal build.  Inputs are raw Montgomery
//! limbs passed through [`core::hint::black_box`] so the optimizer can't
//! fold the kernel away; the limbs aren't validated, since the pipeline
//! model only cares about the instruction stream, not the result.  The
//! `Fp64` / `Fp2` shims need `bmi2` + `adx` (the MULX + ADCX/ADOX
//! kernels); the `Fp51` shim is always available on x86_64.
//!
//! Raw-array-by-value signatures aren't FFI-safe, but these are never
//! called across an FFI boundary -- they exist only to emit a symbol --
//! so the lint is suppressed module-wide.
#![allow(improper_ctypes_definitions)]

use crate::fields::fp::arch::generic::Fp51;

// `mca_fp51_mul` is intentionally absent: `Fp51`'s `Mul` impl is not
// `#[inline]`, so a shim around it codegens to a bare `call` and
// `llvm-mca` would model the call, not the schoolbook reduction.
// Adding `#[inline]` there is a production change to the active
// aarch64/portable `Fp` and out of scope for this tool.  `Fp51`'s
// `square` *is* inlinable, so the comparison square is provided.

/// `Fp51` (radix-51) Montgomery squaring.
#[no_mangle]
#[inline(never)]
pub extern "C" fn mca_fp51_square(a: [u64; 5]) -> [u64; 5] {
    let a = core::hint::black_box(Fp51::from_limbs(a));
    a.square().0
}

/// `Fp64` / `Fp2` shims, gated on `bmi2` + `adx` (the MULX + ADCX/ADOX
/// kernels these wrap).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
mod adx {
    use crate::fields::{fp::arch::x86_64::mulx_adx::Fp64, fp2::Fp2};

    /// `Fp64` (radix-2^64) MULX + ADCX/ADOX Montgomery multiplication.
    #[no_mangle]
    #[inline(never)]
    pub extern "C" fn mca_fp64_mul(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        let a = core::hint::black_box(Fp64::from_raw(a));
        let b = core::hint::black_box(Fp64::from_raw(b));
        (a * b).0
    }

    /// `Fp64` (radix-2^64) Montgomery squaring.
    #[no_mangle]
    #[inline(never)]
    pub extern "C" fn mca_fp64_square(a: [u64; 4]) -> [u64; 4] {
        let a = core::hint::black_box(Fp64::from_raw(a));
        a.square().0
    }

    /// `Fp2` multiplication (two `Fp64` real/imag limbs in, two out).
    ///
    /// Computes the Algorithm 8.1 coefficients directly via the two
    /// `Fp64` primitives rather than `Fp2 * Fp2`: the `Mul for Fp2` impl
    /// is not `#[inline]`, so wrapping it leaves only a `call` in the
    /// shim body -- invisible to the pipeline model.  Inlining the
    /// primitives here puts the real instruction stream in front of
    /// `llvm-mca`.
    #[no_mangle]
    #[inline(never)]
    pub extern "C" fn mca_fp2_mul(a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        let a = core::hint::black_box(fp2_from_raw(a));
        let b = core::hint::black_box(fp2_from_raw(b));
        let re = Fp64::difference_of_2_products(&a.a, &b.a, &a.b, &b.b);
        let im = Fp64::sum_of_2_products(&a.a, &b.b, &a.b, &b.a);
        fp2_to_raw(Fp2::new(re, im))
    }

    /// Packs `Fp2` (real, imag) into eight raw `Fp64` limbs.
    fn fp2_to_raw(x: Fp2) -> [u64; 8] {
        let (re, im) = (x.a.0, x.b.0);
        [re[0], re[1], re[2], re[3], im[0], im[1], im[2], im[3]]
    }

    /// Unpacks eight raw `Fp64` limbs into an `Fp2`.
    fn fp2_from_raw(x: [u64; 8]) -> Fp2 {
        let re = Fp64::from_raw([x[0], x[1], x[2], x[3]]);
        let im = Fp64::from_raw([x[4], x[5], x[6], x[7]]);
        Fp2::new(re, im)
    }
}
