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
use sqisign_selkie::quaternions::lattice::NrdBasis;
use sqisign_selkie::quaternions::lattice::dpe::DoublePlusExponent;
use sqisign_selkie::quaternions::linear::{Matrix, Vector};

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

    fn selkie_cref_dpe_mul(
        a_m: f64,
        a_e: core::ffi::c_long,
        b_m: f64,
        b_e: core::ffi::c_long,
        out_m: *mut f64,
        out_e: *mut core::ffi::c_long,
    );
    fn selkie_cref_dpe_add(
        a_m: f64,
        a_e: core::ffi::c_long,
        b_m: f64,
        b_e: core::ffi::c_long,
        out_m: *mut f64,
        out_e: *mut core::ffi::c_long,
    );
    fn selkie_cref_dpe_sub(
        a_m: f64,
        a_e: core::ffi::c_long,
        b_m: f64,
        b_e: core::ffi::c_long,
        out_m: *mut f64,
        out_e: *mut core::ffi::c_long,
    );
    fn selkie_cref_dpe_div(
        a_m: f64,
        a_e: core::ffi::c_long,
        b_m: f64,
        b_e: core::ffi::c_long,
        out_m: *mut f64,
        out_e: *mut core::ffi::c_long,
    );
    fn selkie_cref_dpe_cmp(
        a_m: f64,
        a_e: core::ffi::c_long,
        b_m: f64,
        b_e: core::ffi::c_long,
    ) -> core::ffi::c_int;
    fn selkie_cref_dpe_cmp_d(
        a_m: f64,
        a_e: core::ffi::c_long,
        d: f64,
    ) -> core::ffi::c_int;
    fn selkie_cref_dpe_round(
        a_m: f64,
        a_e: core::ffi::c_long,
        out_m: *mut f64,
        out_e: *mut core::ffi::c_long,
    );
    fn selkie_cref_dpe_get_z(
        a_m: f64,
        a_e: core::ffi::c_long,
        out_limbs: *mut u64,
        n_limbs: usize,
        out_is_negative: *mut core::ffi::c_int,
        out_actual_limbs: *mut usize,
    );
}

/// Pair of `DoublePlusExponent`-equivalent inputs, returned as raw
/// `(f64 bits, i64 exp)` by both the Rust-side compute and the C-side
/// FFI shim. Equality on this tuple is bit-exactness.
type DpeBits = (u64, i64);

fn dpe_bits(d: DoublePlusExponent) -> DpeBits {
    (d.m.to_bits(), d.e)
}

fn cref_call<F>(f: F) -> DpeBits
where
    F: FnOnce(*mut f64, *mut core::ffi::c_long),
{
    let mut m = 0.0f64;
    let mut e: core::ffi::c_long = 0;
    f(&mut m as *mut f64, &mut e as *mut core::ffi::c_long);
    (m.to_bits(), e as i64)
}

fn cref_mul(a: DoublePlusExponent, b: DoublePlusExponent) -> DpeBits {
    cref_call(|out_m, out_e| unsafe {
        selkie_cref_dpe_mul(a.m, a.e as core::ffi::c_long, b.m, b.e as core::ffi::c_long, out_m, out_e);
    })
}
fn cref_add(a: DoublePlusExponent, b: DoublePlusExponent) -> DpeBits {
    cref_call(|out_m, out_e| unsafe {
        selkie_cref_dpe_add(a.m, a.e as core::ffi::c_long, b.m, b.e as core::ffi::c_long, out_m, out_e);
    })
}
fn cref_sub(a: DoublePlusExponent, b: DoublePlusExponent) -> DpeBits {
    cref_call(|out_m, out_e| unsafe {
        selkie_cref_dpe_sub(a.m, a.e as core::ffi::c_long, b.m, b.e as core::ffi::c_long, out_m, out_e);
    })
}
fn cref_div(a: DoublePlusExponent, b: DoublePlusExponent) -> DpeBits {
    cref_call(|out_m, out_e| unsafe {
        selkie_cref_dpe_div(a.m, a.e as core::ffi::c_long, b.m, b.e as core::ffi::c_long, out_m, out_e);
    })
}
fn cref_round(a: DoublePlusExponent) -> DpeBits {
    cref_call(|out_m, out_e| unsafe {
        selkie_cref_dpe_round(a.m, a.e as core::ffi::c_long, out_m, out_e);
    })
}
fn cref_cmp(a: DoublePlusExponent, b: DoublePlusExponent) -> i32 {
    unsafe {
        selkie_cref_dpe_cmp(a.m, a.e as core::ffi::c_long, b.m, b.e as core::ffi::c_long)
    }
}
fn cref_cmp_d(a: DoublePlusExponent, d: f64) -> i32 {
    unsafe { selkie_cref_dpe_cmp_d(a.m, a.e as core::ffi::c_long, d) }
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

// ---------------------------------------------------------------------------
// DPE arithmetic op bit-equality vs C ref
// ---------------------------------------------------------------------------

/// Strategy: build a normalized DoublePlusExponent from a random
/// BigInt<4>. Returns both the Rust-side value and inputs in a form
/// that can also be passed to the C shim.
fn arb_dpe() -> impl Strategy<Value = DoublePlusExponent> {
    (proptest::array::uniform4(any::<u64>()), any::<bool>())
        .prop_map(|(limbs, neg)| {
            let mut b = BigInt::<4>::from_limbs(limbs);
            if neg { b = b.wrapping_neg(); }
            DoublePlusExponent::from(&b)
        })
}

/// Multiplication is the simplest op (mantissa product, exp sum,
/// normalize). LLVM has no way to fuse this with anything else
/// because the operands come from struct loads — no FMA risk.
#[test]
fn dpe_mul_matches_cref() {
    proptest!(ProptestConfig::with_cases(20_000), |(a in arb_dpe(), b in arb_dpe())| {
        let rust = dpe_bits(a * b);
        let c = cref_mul(a, b);
        prop_assert_eq!(rust, c,
            "mul: a=({:?},{}) b=({:?},{})",
            f64::from_bits(a.m.to_bits()), a.e,
            f64::from_bits(b.m.to_bits()), b.e);
    });
}

/// Division — same risk profile as mul.
#[test]
fn dpe_div_matches_cref() {
    proptest!(ProptestConfig::with_cases(20_000), |(a in arb_dpe(), b in arb_dpe())| {
        // Skip zero divisor (UB / NaN).
        if b.m == 0.0 {
            return Ok(());
        }
        let rust = dpe_bits(a / b);
        let c = cref_div(a, b);
        prop_assert_eq!(rust, c,
            "div: a=({:?},{}) b=({:?},{})",
            f64::from_bits(a.m.to_bits()), a.e,
            f64::from_bits(b.m.to_bits()), b.e);
    });
}

/// Addition has the highest bit-equality risk: it does exponent
/// alignment via `dpe_scale` (table-lookup multiply) before adding
/// mantissas. Different operation order in our impl could diverge.
#[test]
fn dpe_add_matches_cref() {
    proptest!(ProptestConfig::with_cases(20_000), |(a in arb_dpe(), b in arb_dpe())| {
        let rust = dpe_bits(a + b);
        let c = cref_add(a, b);
        prop_assert_eq!(rust, c,
            "add: a=({:?},{}) b=({:?},{})",
            f64::from_bits(a.m.to_bits()), a.e,
            f64::from_bits(b.m.to_bits()), b.e);
    });
}

/// Subtraction — same path as add (calls sub via `add(neg)`).
#[test]
fn dpe_sub_matches_cref() {
    proptest!(ProptestConfig::with_cases(20_000), |(a in arb_dpe(), b in arb_dpe())| {
        let rust = dpe_bits(a - b);
        let c = cref_sub(a, b);
        prop_assert_eq!(rust, c,
            "sub: a=({:?},{}) b=({:?},{})",
            f64::from_bits(a.m.to_bits()), a.e,
            f64::from_bits(b.m.to_bits()), b.e);
    });
}

/// Comparison must return identical sign of result. C ref returns
/// {-1, 0, +1, or any signed int with same sign}; we compare via
/// signum.
#[test]
fn dpe_cmp_matches_cref() {
    proptest!(ProptestConfig::with_cases(20_000), |(a in arb_dpe(), b in arb_dpe())| {
        let rust = match a.partial_cmp(&b) {
            Some(core::cmp::Ordering::Less) => -1,
            Some(core::cmp::Ordering::Equal) => 0,
            Some(core::cmp::Ordering::Greater) => 1,
            None => panic!("DPE partial_cmp returned None"),
        };
        let c = cref_cmp(a, b).signum();
        prop_assert_eq!(rust, c,
            "cmp: a=({:?},{}) b=({:?},{})", a.m, a.e, b.m, b.e);
    });
}

/// Compare against an arbitrary f64.
#[test]
fn dpe_cmp_d_matches_cref() {
    proptest!(ProptestConfig::with_cases(20_000), |(a in arb_dpe(), d in any::<f64>())| {
        // Skip NaN/inf — neither side has defined semantics for those.
        if !d.is_finite() {
            return Ok(());
        }
        let rust = {
            let d_dpe = DoublePlusExponent::from_f64(d);
            match a.partial_cmp(&d_dpe) {
                Some(core::cmp::Ordering::Less) => -1,
                Some(core::cmp::Ordering::Equal) => 0,
                Some(core::cmp::Ordering::Greater) => 1,
                None => panic!("None"),
            }
        };
        let c = cref_cmp_d(a, d).signum();
        prop_assert_eq!(rust, c, "cmp_d: a=({:?},{}) d={}", a.m, a.e, d);
    });
}

/// `to_bigint` (= `dpe_get_z`) is the load-bearing op for L²'s
/// basis update: the size-reduction integer `X = round(u[k][i])`
/// is derived this way and used to update the integer basis. Any
/// bit-difference here changes the output reduced basis byte-for-byte.
#[test]
fn dpe_to_bigint_matches_cref() {
    proptest!(ProptestConfig::with_cases(20_000), |(a in arb_dpe())| {
        // Rust side.
        let rust: BigInt<4> = a.to_bigint();

        // C side: into raw limbs + sign.
        let mut out_limbs = [0u64; 4];
        let mut is_neg: core::ffi::c_int = 0;
        let mut actual: usize = 0;
        unsafe {
            selkie_cref_dpe_get_z(
                a.m,
                a.e as core::ffi::c_long,
                out_limbs.as_mut_ptr(),
                4,
                &mut is_neg as *mut core::ffi::c_int,
                &mut actual as *mut usize,
            );
        }
        // C may write fewer than 4 limbs if the magnitude is short;
        // remaining are already zero. Reconstruct sign-magnitude.
        let mut c_bigint = BigInt::<4>::from_limbs(out_limbs);
        if is_neg != 0 {
            c_bigint = c_bigint.wrapping_neg();
        }

        prop_assert_eq!(rust, c_bigint,
            "to_bigint: a.m={:?} a.e={}", a.m, a.e);
    });
}

/// dpe_round writes back into a dpe_t — the rounded integer in
/// canonical (mantissa, exp) form. Used to derive the basis-update
/// integer X in L²'s size reduction. Bit-equality matters.
#[test]
fn dpe_round_matches_cref() {
    proptest!(ProptestConfig::with_cases(20_000), |(a in arb_dpe())| {
        // Compute Rust-side round-then-pack-as-dpe.
        let rounded_f = a.round(); // returns f64 (existing API)
        let rust = dpe_bits(DoublePlusExponent::from_f64(rounded_f));
        let c = cref_round(a);
        prop_assert_eq!(rust, c, "round: a=({:?},{})", a.m, a.e);
    });
}

// ---------------------------------------------------------------------------
// quat_lll_core black-box bit-equality
// ---------------------------------------------------------------------------

const LLL_STRIDE: usize = 64; // 4096-bit per matrix entry — covers any SQIsign-shaped Gram

unsafe extern "C" {
    fn selkie_cref_quat_lll_core(
        gram_limbs: *mut u64,
        gram_signs: *mut core::ffi::c_int,
        basis_limbs: *mut u64,
        basis_signs: *mut core::ffi::c_int,
        stride: usize,
    );
}

/// Encode a Selkie `Matrix<N>` (with N ≤ LLL_STRIDE) as a flat
/// (limbs, signs) buffer pair laid out for the C shim.
fn encode_mat<const N: usize>(m: &Matrix<N>) -> (Vec<u64>, Vec<core::ffi::c_int>) {
    assert!(N <= LLL_STRIDE);
    let mut limbs = vec![0u64; 16 * LLL_STRIDE];
    let mut signs = vec![0 as core::ffi::c_int; 16];
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            let entry = m[i][j];
            let abs = entry.abs();
            let src = abs.as_limbs();
            let dst = &mut limbs[idx * LLL_STRIDE..idx * LLL_STRIDE + N];
            dst.copy_from_slice(src);
            signs[idx] = if bool::from(entry.is_negative()) { 1 } else { 0 };
        }
    }
    (limbs, signs)
}

/// Decode the flat C-side output back into a Selkie `Matrix<N>`.
fn decode_mat<const N: usize>(limbs: &[u64], signs: &[core::ffi::c_int]) -> Matrix<N> {
    let mut m = Matrix::<N>::ZERO;
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            let src = &limbs[idx * LLL_STRIDE..idx * LLL_STRIDE + N];
            let mut arr = [0u64; N];
            arr.copy_from_slice(src);
            let mut entry = BigInt::<N>::from_limbs(arr);
            if signs[idx] != 0 {
                entry = entry.wrapping_neg();
            }
            m[i][j] = entry;
        }
    }
    m
}

/// Run C ref's `quat_lll_core` on `(gram, basis)`. Returns the
/// reduced `(gram, basis)` pair.
fn cref_quat_lll_core<const N: usize>(
    gram: &Matrix<N>,
    basis: &Matrix<N>,
) -> (Matrix<N>, Matrix<N>) {
    let (mut gram_limbs, mut gram_signs) = encode_mat(gram);
    let (mut basis_limbs, mut basis_signs) = encode_mat(basis);
    unsafe {
        selkie_cref_quat_lll_core(
            gram_limbs.as_mut_ptr(),
            gram_signs.as_mut_ptr(),
            basis_limbs.as_mut_ptr(),
            basis_signs.as_mut_ptr(),
            LLL_STRIDE,
        );
    }
    (
        decode_mat(&gram_limbs, &gram_signs),
        decode_mat(&basis_limbs, &basis_signs),
    )
}

/// Wide-input variant: random ~120-bit entries (closer to SQIsign's
/// post-class-gram Gram entry sizes). `#[ignore]`d because random
/// non-orthogonal wide bases drive `l2_reduce` into many size-reduction
/// iterations — single-case wall time ~1 min, total suite would be
/// 20 min+. Run explicitly with `--ignored` when stress-testing.
#[test]
#[ignore]
fn lll_matches_cref_random_wide() {
    proptest!(ProptestConfig::with_cases(20), |(
        seeds in proptest::array::uniform4(proptest::array::uniform2(any::<u64>())),
    )| {
        // Build a random 4-column basis with ~128-bit entries.
        let bigint_cols: [Vector<8>; 4] = core::array::from_fn(|j| {
            let s = seeds[j];
            let lo = s[0]; let hi = s[1];
            // Each coord = same seed shifted around to vary entries
            // independently per row.
            Vector::new(
                BigInt::<8>::from_limbs([lo,        hi.wrapping_add(1), 0, 0, 0, 0, 0, 0]),
                BigInt::<8>::from_limbs([lo.wrapping_add(2), hi.wrapping_add(3), 0, 0, 0, 0, 0, 0]),
                BigInt::<8>::from_limbs([lo.wrapping_add(4), hi.wrapping_add(5), 0, 0, 0, 0, 0, 0]),
                BigInt::<8>::from_limbs([lo.wrapping_add(6), hi.wrapping_add(7), 0, 0, 0, 0, 0, 0]),
            )
        });
        let basis_mat = Matrix::from_columns(&bigint_cols);
        let nrd = NrdBasis::new(bigint_cols);
        let gram = *nrd.gram();

        let rust_reduced = nrd.l2_reduce();
        let rust_basis = Matrix::from_columns(rust_reduced.cols());
        let rust_gram = *rust_reduced.gram();

        let (cref_gram, cref_basis) = cref_quat_lll_core(&gram, &basis_mat);

        prop_assert_eq!(rust_basis, cref_basis, "wide basis mismatch");
        prop_assert_eq!(rust_gram, cref_gram, "wide gram mismatch");
    });
}

/// Parse one line of the fixture format produced by
/// `reduce_to_prime_norm`'s `SELKIE_DUMP_PRELLL_GRAM` instrumentation:
/// `<tag>[i][j] sign=<0|1> limbs=[0xHEX,0xHEX,...]`
fn parse_fixture_line(line: &str) -> Option<((usize, usize), BigInt<30>)> {
    let after_bracket = line.find('[')?;
    let i: usize = line[after_bracket + 1..line[after_bracket + 1..].find(']')? + after_bracket + 1]
        .parse()
        .ok()?;
    let rest = &line[after_bracket + 1..];
    let second_bracket = rest.find('[')? + after_bracket + 1;
    let j_end = line[second_bracket + 1..].find(']')? + second_bracket + 1;
    let j: usize = line[second_bracket + 1..j_end].parse().ok()?;

    let sign_pos = line.find("sign=")?;
    let sign: u32 = line[sign_pos + 5..sign_pos + 6].parse().ok()?;

    let limbs_start = line.find("limbs=[")? + "limbs=[".len();
    let limbs_end = line.rfind(']')?;
    let limbs_str = &line[limbs_start..limbs_end];

    let mut limbs = [0u64; 30];
    for (k, tok) in limbs_str.split(',').filter(|s| !s.is_empty()).enumerate() {
        let s = tok.trim().trim_start_matches("0x");
        if k < 30 {
            limbs[k] = u64::from_str_radix(s, 16).ok()?;
        }
    }
    let mut v = BigInt::<30>::from_limbs(limbs);
    if sign == 1 {
        v = v.wrapping_neg();
    }
    Some(((i, j), v))
}

/// Real-world LLL bit-equality on the KAT[0] pre-LLL class Gram —
/// the actual Gram `reduce_to_prime_norm` feeds to `l2_reduce` during
/// keygen for KAT seed 0. Captured via the
/// `SELKIE_DUMP_PRELLL_GRAM=1` instrumentation in
/// `quaternions::lattice::LeftIdeal::reduce_to_prime_norm`.
///
/// If this passes, Selkie's `l2_reduce` is bit-exact with C ref's
/// `quat_lll_core` even on production-shaped Grams (~512-bit entries,
/// not just toy random ones), which means the 11/100 keygen
/// byte-equal limit is NOT in the LLL — it's in rejection sampling
/// or post-LLL ideal construction.
#[test]
fn lll_matches_cref_kat0_real_gram() {
    let fixture = std::fs::read_to_string("tests/fixtures/kat0_prelll_gram.txt")
        .expect("KAT[0] pre-LLL Gram fixture must exist");

    let mut gram = Matrix::<30>::ZERO;
    let mut basis = Matrix::<30>::ZERO;
    for line in fixture.lines() {
        let line = line.trim();
        if !line.starts_with("g[") && !line.starts_with("c[") {
            continue;
        }
        let ((i, j), v) = parse_fixture_line(line).expect("parse fixture line");
        if line.starts_with("g[") {
            gram[i][j] = v;
        } else {
            // c[col][row] in dump → basis[row][col] in row-major.
            basis[j][i] = v;
        }
    }

    // Selkie side: reconstruct an NrdBasis with the recorded
    // (cols, gram) and reduce.
    let cols: [Vector<30>; 4] = core::array::from_fn(|j| {
        Vector::new(basis[0][j], basis[1][j], basis[2][j], basis[3][j])
    });
    let nrd = NrdBasis::from_cols_and_gram(cols, gram);
    let rust_reduced = nrd.l2_reduce();
    let rust_basis = Matrix::from_columns(rust_reduced.cols());
    let rust_gram = *rust_reduced.gram();

    // C ref side.
    let (cref_gram, cref_basis) = cref_quat_lll_core(&gram, &basis);

    assert_eq!(rust_basis, cref_basis, "real KAT[0] Gram: basis mismatch");
    assert_eq!(rust_gram, cref_gram, "real KAT[0] Gram: gram mismatch");
}

/// Smoke test: identity Gram + identity basis → still identity.
#[test]
fn lll_smoke_identity() {
    let g = Matrix::<8>::IDENTITY;
    let b = Matrix::<8>::IDENTITY;
    let (g_red, b_red) = cref_quat_lll_core(&g, &b);
    assert_eq!(g_red, Matrix::<8>::IDENTITY);
    assert_eq!(b_red, Matrix::<8>::IDENTITY);
}

/// Random small Gram → run both Selkie and C ref, expect byte-equal
/// reduced (gram, basis). This is the load-bearing test.
///
/// `arb_small_gram_basis` builds a random integer basis and computes
/// its Gram via Selkie's `NrdBasis::compute_gram` (so the Gram is
/// guaranteed to be the inner-product matrix of the basis — what
/// `quat_lll_core` expects).
#[test]
fn lll_matches_cref_random_small() {
    proptest!(ProptestConfig::with_cases(500), |(
        cols in proptest::array::uniform4(proptest::array::uniform4(1i64..=200_000_000)),
    )| {
        // Build a random 4-column basis with small entries.
        let bigint_cols: [Vector<8>; 4] = core::array::from_fn(|j| {
            Vector::new(
                BigInt::<8>::from_i64(cols[j][0]),
                BigInt::<8>::from_i64(cols[j][1]),
                BigInt::<8>::from_i64(cols[j][2]),
                BigInt::<8>::from_i64(cols[j][3]),
            )
        });
        let basis_mat = Matrix::from_columns(&bigint_cols);
        let nrd = NrdBasis::new(bigint_cols);
        let gram = *nrd.gram();

        // Selkie side.
        let rust_reduced = nrd.l2_reduce();
        let rust_basis = Matrix::from_columns(rust_reduced.cols());
        let rust_gram = *rust_reduced.gram();

        // C ref side.
        let (cref_gram, cref_basis) = cref_quat_lll_core(&gram, &basis_mat);

        prop_assert_eq!(rust_basis, cref_basis,
            "basis mismatch on input cols={:?}", cols);
        prop_assert_eq!(rust_gram, cref_gram,
            "gram mismatch on input cols={:?}", cols);
    });
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
