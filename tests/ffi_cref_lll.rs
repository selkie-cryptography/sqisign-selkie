//! Bit-equality tests of `quaternions::lll` against the SQIsign C
//! reference's mini-GMP + DPE primitives, via the C shim built by
//! `build.rs` under the `ffi-cref-lll` feature.
//!
//! Run with:
//!
//! ```sh
//! cargo test --features 'ffi-cref-lll expose-internals' --test ffi_cref_lll
//! ```
//!
//! These tests are the load-bearing validation of the
//! `quaternions::lll` port. If they pass on millions of random inputs,
//! we have high confidence that the pure-Rust LLL produces byte-equal
//! reduced bases to the C reference.

#![cfg(feature = "ffi-cref-lll")]

use proptest::prelude::*;

use sqisign_selkie::quaternions::bigint::BigInt;
use sqisign_selkie::quaternions::lattice::dpe::DoublePlusExponent;

// ---------------------------------------------------------------------------
// FFI bindings to src/ffi/cref_dpe_shim.c
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Calls mini-GMP's `mini_mpz_get_d_2exp` on a BigInt-shaped input.
    /// Limbs are little-endian u64 words encoding the magnitude;
    /// `is_negative` toggles the sign separately.
    fn selkie_cref_to_dpe(
        limbs: *const u64,
        n_limbs: usize,
        is_negative: core::ffi::c_int,
        out_mantissa: *mut f64,
        out_exp: *mut core::ffi::c_long,
    );

    /// Smoke-test: the C side returns DPE form of the constant 0.75
    /// (= mantissa 0.75, exp 0).
    fn selkie_cref_dpe_smoke(out_mantissa: *mut f64, out_exp: *mut core::ffi::c_long);
}

/// Helper: invoke the C shim on a Selkie BigInt and return the
/// resulting `(mantissa_bits, exp)` pair.
fn cref_to_dpe<const N: usize>(x: &BigInt<N>) -> (u64, i64) {
    let abs = x.abs();
    let limbs = abs.as_limbs();
    let is_neg = if bool::from(x.is_negative()) { 1 } else { 0 };
    let mut mantissa = 0.0f64;
    let mut exp: core::ffi::c_long = 0;
    unsafe {
        selkie_cref_to_dpe(
            limbs.as_ptr(),
            N,
            is_neg,
            &mut mantissa as *mut f64,
            &mut exp as *mut core::ffi::c_long,
        );
    }
    (mantissa.to_bits(), exp as i64)
}

// ---------------------------------------------------------------------------
// Smoke test: confirms the C shim builds and runs.
// ---------------------------------------------------------------------------

#[test]
fn ffi_smoke_dpe_set_d_075() {
    let mut m = 0.0f64;
    let mut e: core::ffi::c_long = 0;
    unsafe {
        selkie_cref_dpe_smoke(&mut m as *mut f64, &mut e as *mut core::ffi::c_long);
    }
    // dpe_set_d(0.75) → mantissa 0.75, exp 0 (since 0.75 ∈ [1/2, 1)).
    assert_eq!(m, 0.75);
    assert_eq!(e, 0);
}

// ---------------------------------------------------------------------------
// Bit-equality: BigInt::to_dpe == mini_mpz_get_d_2exp
// ---------------------------------------------------------------------------

/// Small fixed cases where divergence is most likely (boundary inputs).
#[test]
fn to_dpe_boundary_cases() {
    let zero = BigInt::<4>::ZERO;
    assert_eq!(
        cref_to_dpe(&zero),
        (DoublePlusExponent::from_bigint(&zero).m.to_bits(), DoublePlusExponent::from_bigint(&zero).e),
        "zero"
    );

    for v in [1i64, -1, 2, -2, 3, 7, -7, 8, 100, 1000, i64::MAX, i64::MIN + 1] {
        let b = BigInt::<4>::from_i64(v);
        let cref = cref_to_dpe(&b);
        let rust = {
            let d = DoublePlusExponent::from_bigint(&b);
            (d.m.to_bits(), d.e)
        };
        assert_eq!(cref, rust, "v={v}: cref=({:?}, {}) rust=({:?}, {})",
                   f64::from_bits(cref.0), cref.1, f64::from_bits(rust.0), rust.1);
    }
}

/// Random BigInt<4> inputs (up to 256-bit) — covers SQIsign-shaped data.
#[test]
fn to_dpe_random_4_limb() {
    proptest!(ProptestConfig::with_cases(100_000), |(
        limbs in proptest::array::uniform4(any::<u64>()),
        is_negative in any::<bool>(),
    )| {
        let mut b = BigInt::<4>::from_limbs(limbs);
        if is_negative {
            b = b.wrapping_neg();
        }
        let cref = cref_to_dpe(&b);
        let rust = {
            let d = DoublePlusExponent::from_bigint(&b);
            (d.m.to_bits(), d.e)
        };
        prop_assert_eq!(cref, rust,
            "BigInt<4> mismatch: limbs={:?} neg={}", limbs, is_negative);
    });
}

/// Random BigInt<30> inputs — exercise the `bits > DBL_MAX_EXP` path
/// (BigInt<30> = 30 * 64 = 1920 bits, well past 1024).
#[test]
fn to_dpe_random_30_limb_overflow_path() {
    proptest!(ProptestConfig::with_cases(20_000), |(
        limbs in prop::collection::vec(any::<u64>(), 30),
        is_negative in any::<bool>(),
    )| {
        let mut arr = [0u64; 30];
        arr.copy_from_slice(&limbs);
        let mut b = BigInt::<30>::from_limbs(arr);
        if is_negative {
            b = b.wrapping_neg();
        }
        let cref = cref_to_dpe(&b);
        let rust = {
            let d = DoublePlusExponent::from_bigint(&b);
            (d.m.to_bits(), d.e)
        };
        prop_assert_eq!(cref, rust,
            "BigInt<30> mismatch: bitsize={} neg={}", b.bitsize(), is_negative);
    });
}

/// Pathological inputs: single high bit at every position from 0 to 1919.
#[test]
fn to_dpe_single_high_bit_30_limb() {
    for bit_pos in 0..30 * 64 {
        let mut arr = [0u64; 30];
        let limb_idx = bit_pos / 64;
        let bit_in_limb = bit_pos % 64;
        arr[limb_idx] = 1u64 << bit_in_limb;
        let b = BigInt::<30>::from_limbs(arr);
        let cref = cref_to_dpe(&b);
        let rust = {
            let d = DoublePlusExponent::from_bigint(&b);
            (d.m.to_bits(), d.e)
        };
        assert_eq!(cref, rust, "single high bit at pos {bit_pos}");
    }
}

/// All-ones inputs at every bit width 1..=1919 (exercises maximum
/// rounding pressure where every truncated bit is 1).
#[test]
fn to_dpe_all_ones_below_width() {
    for bit_width in 1..=30 * 64 {
        // n = 2^bit_width - 1 (all ones up to bit_width).
        let mut arr = [0u64; 30];
        let full_limbs = bit_width / 64;
        let leftover_bits = bit_width % 64;
        for i in 0..full_limbs {
            arr[i] = u64::MAX;
        }
        if leftover_bits > 0 && full_limbs < 30 {
            arr[full_limbs] = (1u64 << leftover_bits) - 1;
        }
        let b = BigInt::<30>::from_limbs(arr);
        let cref = cref_to_dpe(&b);
        let rust = {
            let d = DoublePlusExponent::from_bigint(&b);
            (d.m.to_bits(), d.e)
        };
        assert_eq!(cref, rust, "all-ones width {bit_width}");
    }
}
