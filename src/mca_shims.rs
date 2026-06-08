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
    use crate::fields::fp::arch::x86_64::mulx_adx::Fp64;

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

    /// `Fp64` 4x4 -> 8-limb product, no reduction.  `mca_fp64_mul` minus
    /// this is the cost of one Montgomery reduction -- the per-coordinate
    /// saving a fused `fp2` would capture.
    #[no_mangle]
    #[inline(never)]
    pub extern "C" fn mca_fp64_mul_wide(a: [u64; 4], b: [u64; 4]) -> [u64; 8] {
        let a = core::hint::black_box(a);
        let b = core::hint::black_box(b);
        Fp64::mul_wide_adx(&a, &b)
    }

    // The fused `fp2` coordinate kernel (`Fp64::sum_of_products_packed`)
    // is intentionally absent: it is register-dense (a 5-limb rotating
    // accumulator over two products) and exceeds the GPR file when
    // inlined into a `#[no_mangle] extern "C"` shim, though it fits in
    // its production caller.  Its cost is tracked by the `field`
    // instruction-count benches instead.
}
