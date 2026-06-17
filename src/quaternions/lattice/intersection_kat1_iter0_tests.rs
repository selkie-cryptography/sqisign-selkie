//! Diagnostic regression test: KAT-1 sign iter 0 intersection.
//!
//! Captures the two lattices passed to `intersection_via_kernel::<500>`
//! (a.k.a. `lattice_hom_chall_to_com` in C-ref) on iter 0 of
//! `sk.sign_derand(msg, seed)` for `KAT_VECTORS[1]`, then runs both
//! `intersection_via_kernel` and `intersection_via_dual_sum_dual` on
//! them and compares outputs.
//!
//! Rationale: empirically, all 8 in-iter rejection samples on iter 0
//! fail the post-backtracking divisibility check, suggesting either
//! the kernel-method basis has a structural distribution skew (same
//! Z-module as C-ref's, different basis) or the kernel method is
//! producing a different Z-module entirely. This test discriminates.
//!
//! Captured 2026-05-09 from commit `e0f2ca3` via
//! `SELKIE_DUMP_INTER=1 cargo test sign_kat_derand_001 -- --ignored
//! --nocapture`.

use super::{HnfLattice, Lattice};
use crate::quaternions::{
    algebra::{Coordinate, Denominator, Element},
    bigint::BigInt,
    linear::{Matrix, Vector},
};

const N_RESP: usize = 30;

#[allow(clippy::type_complexity)]
fn i_chl_sk_lat() -> Lattice<N_RESP> {
    let basis = Matrix::<N_RESP>::from_rows(
        Vector::new(
            BigInt::from_sign_and_limbs(
                0,
                [
                    0,
                    0,
                    0,
                    4755801206503243776,
                    13980318085585502755,
                    1278114601137767694,
                    13,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
            BigInt::from_sign_and_limbs(0, [0; 30]),
            BigInt::from_sign_and_limbs(
                0,
                [
                    10570939321768069668,
                    10950436989368459512,
                    336700417856746014,
                    8081656857070734465,
                    7105814799960192001,
                    1920917671859675317,
                    3,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    16801572591457864103,
                    11283576233110893943,
                    7988007179148279652,
                    5076483508874332546,
                    14734896877763483574,
                    14455304422266373679,
                    3,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
        ),
        Vector::new(
            BigInt::from_sign_and_limbs(0, [0; 30]),
            BigInt::from_sign_and_limbs(
                0,
                [
                    0,
                    0,
                    0,
                    4755801206503243776,
                    13980318085585502755,
                    1278114601137767694,
                    13,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    1645171482251687513,
                    7163167840598657672,
                    10458736894561271963,
                    18126061771338462845,
                    17692165281531570796,
                    5269554252580945630,
                    9,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    10570939321768069668,
                    10950436989368459512,
                    336700417856746014,
                    8081656857070734465,
                    7105814799960192001,
                    1920917671859675317,
                    3,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
        ),
        Vector::new(
            BigInt::from_sign_and_limbs(0, [0; 30]),
            BigInt::from_sign_and_limbs(0, [0; 30]),
            BigInt::from_sign_and_limbs(
                0,
                [
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0,
                ],
            ),
            BigInt::from_sign_and_limbs(0, [0; 30]),
        ),
        Vector::new(
            BigInt::from_sign_and_limbs(0, [0; 30]),
            BigInt::from_sign_and_limbs(0, [0; 30]),
            BigInt::from_sign_and_limbs(0, [0; 30]),
            BigInt::from_sign_and_limbs(
                0,
                [
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0,
                ],
            ),
        ),
    );
    let denom = BigInt::<N_RESP>::from_sign_and_limbs(
        0,
        [
            2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0,
        ],
    );
    Lattice::<N_RESP>::new(basis, denom)
}

#[allow(clippy::type_complexity)]
fn i_com_conj_lat() -> Lattice<N_RESP> {
    let basis = Matrix::<N_RESP>::from_rows(
        Vector::new(
            BigInt::from_sign_and_limbs(
                0,
                [
                    293079610235691842,
                    13602688638206152386,
                    3345,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
            BigInt::from_sign_and_limbs(0, [0; 30]),
            BigInt::from_sign_and_limbs(
                0,
                [
                    4439216597309828578,
                    11149545067280553749,
                    1292,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    16546990024834791637,
                    2673781900770453680,
                    1517,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
        ),
        Vector::new(
            BigInt::from_sign_and_limbs(1, [0; 30]),
            BigInt::from_sign_and_limbs(
                0,
                [
                    293079610235691842,
                    13602688638206152386,
                    3345,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    16546990024834791637,
                    2673781900770453680,
                    1517,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    14300607086635414880,
                    2453143570925598636,
                    2053,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
        ),
        Vector::new(
            BigInt::from_sign_and_limbs(1, [0; 30]),
            BigInt::from_sign_and_limbs(0, [0; 30]),
            BigInt::from_sign_and_limbs(
                0,
                [
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0,
                ],
            ),
            BigInt::from_sign_and_limbs(0, [0; 30]),
        ),
        Vector::new(
            BigInt::from_sign_and_limbs(1, [0; 30]),
            BigInt::from_sign_and_limbs(0, [0; 30]),
            BigInt::from_sign_and_limbs(0, [0; 30]),
            BigInt::from_sign_and_limbs(
                0,
                [
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0,
                ],
            ),
        ),
    );
    let denom = BigInt::<N_RESP>::from_sign_and_limbs(
        0,
        [
            2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0,
        ],
    );
    Lattice::<N_RESP>::new(basis, denom)
}

/// Sanity-print bit-sizes of the two captured input lattices.
#[test]
fn dump_input_sizes() {
    let l1 = i_chl_sk_lat();
    let l2 = i_com_conj_lat();
    let max_entry_bits = |lat: &Lattice<N_RESP>| -> u32 {
        let mut m = 0u32;
        for r in 0..4 {
            for c in 0..4 {
                let b = lat.basis()[r][c].bitsize();
                if b > m {
                    m = b;
                }
            }
        }
        m
    };
    eprintln!(
        "i_chl_sk_lat: max entry bits = {}, denom bits = {}",
        max_entry_bits(&l1),
        l1.denom().bitsize()
    );
    eprintln!(
        "i_com_conj_lat: max entry bits = {}, denom bits = {}",
        max_entry_bits(&l2),
        l2.denom().bitsize()
    );
}

/// Diagnostic: cross-check `intersection_via_kernel::<500>` and
/// `intersection_via_dual_sum_dual::<W>` on the captured KAT-1 iter 0
/// inputs, using three independent correctness criteria:
///
/// 1. **Containment in inputs.** Every basis col of the result must live in
///    BOTH input lattices.
/// 2. **Cross-method containment.** Determines which (if either) output is a
///    strict sublattice of the other.
/// 3. **Lattice-index identity.** `covol(L1) · covol(L2) = covol(L_int) ·
///    covol(L_sum)` uniquely identifies the *true* L1 ∩ L2 among sublattices of
///    both inputs.
///
/// State as of 2026-05-09 (after the modular-HNF sum fix in DSD):
///
/// | method                                  | (1) contained        | (2) vs kernel | (3) identity |
/// |-----------------------------------------|----------------------|---------------|--------------|
/// | `intersection_via_kernel::<500>`        | ✓                    | (self)        | ✓ TRUE       |
/// | `intersection_via_dual_sum_dual::<60>`  | overflow (cols outside) | n/a        | n/a          |
/// | `intersection_via_dual_sum_dual::<100>` | overflow (cols outside) | n/a        | ✗ overflow   |
/// | `intersection_via_dual_sum_dual::<200>` | ✓                    | matches kernel | ✓ TRUE     |
///
/// Findings (post-fix):
/// - `intersection_via_kernel::<500>` is correct on these inputs.
/// - `intersection_via_dual_sum_dual::<200>` is now correct after switching the
///   sum step from non-modular to modular HNF (mirroring C-ref's
///   `quat_lattice_add` recipe with `modulus = gcd(det1, det2)`).
/// - Smaller widths (W=60, 100) silently overflow during the modular HNF
///   intermediates; the captured inputs need W >= ~200 to be accommodated.
///
/// Underlying root cause (left as TODO): `Matrix::from_hnf_columns`
/// (non-modular HNF) produces a strict superlattice of the true Z-module
/// span when given >4 columns with large per-column common factors. Mod-HNF
/// gets the right answer because the explicit modulus generators it appends
/// (`D · e_i`) drive the canonical pivot gcds; non-mod HNF lacks that and
/// silently wrong-answers. Fix the underlying `from_hnf_columns` separately;
/// for now `Lattice::sum`'s other callers happen not to trip the bug.
#[test]
fn intersection_results_contained_in_inputs() {
    let l1 = i_chl_sk_lat();
    let l2 = i_com_conj_lat();
    let l1_hnf: HnfLattice<N_RESP> = HnfLattice::from(l1).canonicalize();
    let l2_hnf: HnfLattice<N_RESP> = HnfLattice::from(l2).canonicalize();

    let inter_k = l1
        .intersection_via_kernel::<500>(&l2)
        .expect("kernel intersection at W=500 must succeed");
    // W=200 — the smallest tested width that handles KAT-1 iter 0
    // inputs without silent overflow under the modular-HNF sum.
    let inter_d = l1
        .intersection_via_dual_sum_dual::<200>(&l2)
        .expect("dual-sum-dual intersection at W=200 must succeed");

    let basis_col_to_element = |hnf: &HnfLattice<N_RESP>, c: usize| -> Element<N_RESP> {
        let denom = Denominator::<N_RESP>::new(*hnf.denom()).expect("denom > 0");
        Element::<N_RESP>::new(
            Coordinate::<N_RESP>::from(hnf.basis()[0][c]),
            Coordinate::<N_RESP>::from(hnf.basis()[1][c]),
            Coordinate::<N_RESP>::from(hnf.basis()[2][c]),
            Coordinate::<N_RESP>::from(hnf.basis()[3][c]),
            denom,
        )
    };

    let report = |name: &str, inter: &HnfLattice<N_RESP>| {
        eprintln!("--- {name} ---");
        let mut all_in_l1 = true;
        let mut all_in_l2 = true;
        for c in 0..4 {
            let e = basis_col_to_element(inter, c);
            let in_l1 = l1_hnf.contains(&e).is_some();
            let in_l2 = l2_hnf.contains(&e).is_some();
            eprintln!(
                "  col[{c}]: in L1={}, in L2={}, zero?={}",
                in_l1,
                in_l2,
                e.is_zero()
            );
            if !in_l1 && !e.is_zero() {
                all_in_l1 = false;
            }
            if !in_l2 && !e.is_zero() {
                all_in_l2 = false;
            }
        }
        eprintln!("  → all-in-L1={all_in_l1}, all-in-L2={all_in_l2}");
        (all_in_l1, all_in_l2)
    };

    let (k_in_l1, k_in_l2) = report("intersection_via_kernel<500>", &inter_k);
    let (d_in_l1, d_in_l2) = report("intersection_via_dual_sum_dual<200>", &inter_d);

    // Probe DSD<60> separately to document the silent-overflow boundary.
    let inter_d60 = l1
        .intersection_via_dual_sum_dual::<60>(&l2)
        .expect("dual-sum-dual<60> should still narrow");
    let (d60_in_l1, d60_in_l2) = report("intersection_via_dual_sum_dual<60>", &inter_d60);

    // Cross-method containment: is DSD<100>'s lattice contained in
    // kernel<500>'s, or vice versa? (If both are correct → equal,
    // and both inclusions hold. If one is a strict sublattice → only
    // one inclusion holds.)
    if let Some(d100) = l1.intersection_via_dual_sum_dual::<100>(&l2) {
        let kernel_canon = inter_k.canonicalize();
        let d100_canon = d100.canonicalize();
        eprintln!("--- cross-method containment ---");
        let mut d100_in_kernel = true;
        for c in 0..4 {
            let e = basis_col_to_element(&d100_canon, c);
            if !e.is_zero() && kernel_canon.contains(&e).is_none() {
                eprintln!("  DSD<100>.col[{c}] is NOT in kernel<500>");
                d100_in_kernel = false;
            }
        }
        let mut kernel_in_d100 = true;
        for c in 0..4 {
            let e = basis_col_to_element(&kernel_canon, c);
            if !e.is_zero() && d100_canon.contains(&e).is_none() {
                eprintln!("  kernel<500>.col[{c}] is NOT in DSD<100>");
                kernel_in_d100 = false;
            }
        }
        eprintln!("  DSD<100> ⊆ kernel<500>? {d100_in_kernel}");
        eprintln!("  kernel<500> ⊆ DSD<100>? {kernel_in_d100}");
    }

    // Probe DSD at additional widths to localize the silent-overflow
    // boundary. Avoid W=500 here — `[BigInt<500>; 4×4]` exhausts the
    // default 2 MB test thread stack on macOS.
    if let Some(d100) = l1.intersection_via_dual_sum_dual::<100>(&l2) {
        report("intersection_via_dual_sum_dual<100>", &d100);
    } else {
        eprintln!("dsd<100>: None");
    }
    if let Some(d200) = l1.intersection_via_dual_sum_dual::<200>(&l2) {
        report("intersection_via_dual_sum_dual<200>", &d200);
    } else {
        eprintln!("dsd<200>: None");
    }

    // Lattice-index identity:
    //   covol(L1) · covol(L2) = covol(L_int) · covol(L_sum)
    // ⇔ |det(B1)| · |det(B2)| · d_int^4 · d_sum^4 = |det(B_int)| · |det(B_sum)| ·
    // d1^4 · d2^4
    //
    // Whichever candidate L_int satisfies this identity (within the
    // available width) is the *true* intersection L1 ∩ L2.
    //
    // We need to compare two large products. Both fit in BigInt<N_RESP>=30
    // limbs comfortably for these specific KAT-1 inputs, but compute in
    // BigInt<60> for headroom.
    {
        // Widen-by-copying helper.
        let widen = |x: &BigInt<N_RESP>| -> BigInt<60> { x.widen::<60>() };

        let l1_w: Lattice<60> = {
            let mut basis = Matrix::<60>::ZERO;
            for r in 0..4 {
                for c in 0..4 {
                    basis[r][c] = widen(&l1.basis()[r][c]);
                }
            }
            Lattice::<60>::new(basis, widen(l1.denom()))
        };
        let l2_w: Lattice<60> = {
            let mut basis = Matrix::<60>::ZERO;
            for r in 0..4 {
                for c in 0..4 {
                    basis[r][c] = widen(&l2.basis()[r][c]);
                }
            }
            Lattice::<60>::new(basis, widen(l2.denom()))
        };
        let sum_hnf = l1_w.sum(&l2_w);
        let det_b1 = l1_w.basis().det().abs();
        let det_b2 = l2_w.basis().det().abs();
        let det_sum = sum_hnf.basis().det().abs();
        let d1_w = *l1_w.denom();
        let d2_w = *l2_w.denom();
        let d_sum = *sum_hnf.denom();
        eprintln!("--- lattice-index identity ---");
        eprintln!(
            "det(B1) bits = {}, det(B2) bits = {}, det(Bsum) bits = {}",
            det_b1.bitsize(),
            det_b2.bitsize(),
            det_sum.bitsize()
        );
        eprintln!(
            "d1 = {} bits, d2 = {} bits, d_sum = {} bits",
            d1_w.bitsize(),
            d2_w.bitsize(),
            d_sum.bitsize()
        );

        let pow4 = |x: BigInt<60>| -> BigInt<60> {
            let x2 = x.ct_mul(&x);
            x2.ct_mul(&x2)
        };

        let check_identity = |label: &str, det_int: BigInt<N_RESP>, d_int: BigInt<N_RESP>| {
            let det_int_w = widen(&det_int);
            let d_int_w = widen(&d_int);
            let lhs = det_b1
                .ct_mul(&det_b2)
                .ct_mul(&pow4(d_int_w))
                .ct_mul(&pow4(d_sum));
            let rhs = det_int_w
                .ct_mul(&det_sum)
                .ct_mul(&pow4(d1_w))
                .ct_mul(&pow4(d2_w));
            let ok = lhs == rhs;
            eprintln!(
                "  {label}: covol identity {} (lhs bits {}, rhs bits {})",
                if ok { "HOLDS ✓" } else { "FAILS ✗" },
                lhs.bitsize(),
                rhs.bitsize()
            );
            ok
        };

        let kernel_canon = inter_k.canonicalize();
        let det_k = kernel_canon.basis().det().abs();
        let d_k = *kernel_canon.denom();
        check_identity("kernel<500>", det_k, d_k);

        if let Some(d100) = l1.intersection_via_dual_sum_dual::<100>(&l2) {
            let dc = d100.canonicalize();
            let det_d = dc.basis().det().abs();
            let d_d = *dc.denom();
            check_identity("DSD<100>    ", det_d, d_d);
        }
        if let Some(d200) = l1.intersection_via_dual_sum_dual::<200>(&l2) {
            let dc = d200.canonicalize();
            let det_d = dc.basis().det().abs();
            let d_d = *dc.denom();
            check_identity("DSD<200>    ", det_d, d_d);
        }
    }

    // Definitive assertions:
    //   - kernel<500> must produce a true sublattice of L1 ∩ L2.
    //   - DSD<200> must too (the post-fix correctness check).
    //   - DSD<60> is documented to silently overflow on these inputs — fail-loud
    //     assertion fires only if the boundary moved.
    assert!(
        k_in_l1 && k_in_l2,
        "intersection_via_kernel<500> produced a result NOT contained in both inputs"
    );
    assert!(
        d_in_l1 && d_in_l2,
        "intersection_via_dual_sum_dual<200> produced a result NOT contained in both inputs \
         — modular-HNF sum fix may have regressed."
    );
    assert!(
        !d60_in_l1 || !d60_in_l2,
        "EXPECTED FAILURE on iter-0 inputs: dual_sum_dual<60> is supposed to silently overflow. \
         If this fires, the silent-overflow boundary moved and the docs should be narrowed."
    );

    eprintln!();
    eprintln!("Summary:");
    eprintln!(
        "  kernel<500>:        valid intersection? {}",
        k_in_l1 && k_in_l2
    );
    eprintln!(
        "  dual_sum_dual<200>: valid intersection? {}",
        d_in_l1 && d_in_l2
    );
}

/// Step-by-step trace of `intersection_via_dual_sum_dual` on KAT-1
/// iter 0 inputs, comparing each intermediate's covolume to the
/// expected value.
///
/// Expected covolumes (with denom counted as `det / d^4`, so values
/// are bits of `|det(B)| / d^4`, possibly negative):
///
/// - covol(L1) = det 776 - 4·denom 1 = 772
/// - covol(L2) = 280 - 4 = 276
/// - covol(L_int) = 911 (kernel-derived ground truth)
/// - covol(L_sum) = 137 (computed via index identity)
/// - covol(dual(L1)) = -772
/// - covol(dual(L2)) = -276
/// - covol(dual(L1) + dual(L2)) = -911
/// - covol(dual(dual(L1) + dual(L2))) = 911 = covol(L_int)
///
/// We don't directly assert; we print and inspect.
#[test]
fn dsd_step_by_step() {
    const W: usize = 200;
    let l1 = i_chl_sk_lat();
    let l2 = i_com_conj_lat();

    // Widen.
    let widen_lat = |lat: &Lattice<N_RESP>| -> Lattice<W> {
        let mut basis = Matrix::<W>::ZERO;
        for r in 0..4 {
            for c in 0..4 {
                basis[r][c] = lat.basis()[r][c].widen::<W>();
            }
        }
        Lattice::<W>::new(basis, lat.denom().widen::<W>())
    };
    let l1_w = widen_lat(&l1);
    let l2_w = widen_lat(&l2);

    let report_lat = |label: &str, lat: &Lattice<W>| {
        let det = lat.basis().det().abs();
        let denom = *lat.denom();
        eprintln!(
            "  {label}: |det(B)| = {} bits, denom = {} bits  → covol = {} - 4·{} = {} bits",
            det.bitsize(),
            denom.bitsize(),
            det.bitsize(),
            denom.bitsize(),
            det.bitsize() as i64 - 4 * (denom.bitsize() as i64),
        );
    };

    eprintln!("--- inputs ---");
    report_lat("L1   ", &l1_w);
    report_lat("L2   ", &l2_w);

    eprintln!("--- step 1: dual ---");
    let d1 = l1_w.dual();
    let d2 = l2_w.dual();
    report_lat("dual(L1)", &d1);
    report_lat("dual(L2)", &d2);

    eprintln!("--- step 1.5: gcd inside d1, d2 representations ---");
    let lat_internal_gcd = |lat: &Lattice<W>| -> u32 {
        let mut g = lat.denom().abs();
        for r in 0..4 {
            for c in 0..4 {
                g = g.gcd(&lat.basis()[r][c].abs());
                if g == BigInt::<W>::ONE {
                    return 0;
                }
            }
        }
        g.bitsize()
    };
    eprintln!(
        "  gcd(d1.denom, d1.basis entries) = {} bits",
        lat_internal_gcd(&d1)
    );
    eprintln!(
        "  gcd(d2.denom, d2.basis entries) = {} bits",
        lat_internal_gcd(&d2)
    );

    eprintln!("--- step 2: sum of duals ---");
    let sum_hnf = d1.sum(&d2);
    let sum_lat: Lattice<W> = sum_hnf.into();
    report_lat("sum (raw d1, d2)    ", &sum_lat);

    // Hypothesis test: sum CANONICAL representations of d1, d2 (i.e.,
    // factor out gcd inside each before summing). If sum is sensitive
    // to representation choice, this should give a different (and
    // larger) lattice.
    let canonicalize_lat = |lat: &Lattice<W>| -> Lattice<W> {
        let mut g = lat.denom().abs();
        for r in 0..4 {
            for c in 0..4 {
                g = g.gcd(&lat.basis()[r][c].abs());
                if g == BigInt::<W>::ONE {
                    return *lat;
                }
            }
        }
        if g == BigInt::<W>::ONE {
            return *lat;
        }
        let mut basis = *lat.basis();
        for r in 0..4 {
            for c in 0..4 {
                let (q, _) = basis[r][c].vt_div_rem(&g);
                basis[r][c] = q;
            }
        }
        let (denom_new, _) = lat.denom().vt_div_rem(&g);
        Lattice::<W>::new(basis, denom_new)
    };
    let d1_canon = canonicalize_lat(&d1);
    let d2_canon = canonicalize_lat(&d2);
    eprintln!(
        "  d1_canon: |det| = {} bits, denom = {} bits",
        d1_canon.basis().det().abs().bitsize(),
        d1_canon.denom().bitsize()
    );
    eprintln!(
        "  d2_canon: |det| = {} bits, denom = {} bits",
        d2_canon.basis().det().abs().bitsize(),
        d2_canon.denom().bitsize()
    );
    let sum_canon_hnf = d1_canon.sum(&d2_canon);
    let sum_canon_lat: Lattice<W> = sum_canon_hnf.into();
    report_lat("sum (canon inputs)  ", &sum_canon_lat);

    eprintln!("--- step 3: dual of sum (pre-gcd-reduce) ---");
    let result_pre = sum_lat.dual();
    report_lat("dual(sum) pre", &result_pre);

    // Apply DSD's first-pass gcd reduction manually so we can
    // inspect the result.
    let mut basis_w = result_pre.basis;
    let denom_w = result_pre.denom;
    let mut g = denom_w.abs();
    for r in 0..4 {
        for c in 0..4 {
            g = g.gcd(&basis_w[r][c].abs());
            if g == BigInt::<W>::ONE {
                break;
            }
        }
        if g == BigInt::<W>::ONE {
            break;
        }
    }
    eprintln!("  first-pass gcd: {} bits", g.bitsize());
    let denom_reduced = if g == BigInt::<W>::ONE {
        denom_w
    } else {
        for r in 0..4 {
            for c in 0..4 {
                let (q, _) = basis_w[r][c].vt_div_rem(&g);
                basis_w[r][c] = q;
            }
        }
        let (q, _) = denom_w.vt_div_rem(&g);
        q
    };
    let reduced_lat = Lattice::<W>::new(basis_w, denom_reduced);
    eprintln!("--- step 4: after first gcd-reduce ---");
    report_lat("dual(sum) reduced", &reduced_lat);

    eprintln!("--- step 5: HNF + canonicalize ---");
    let result_hnf = reduced_lat.hnf();
    let result_hnf = result_hnf.canonicalize();
    let result_lat: Lattice<W> = result_hnf.into();
    report_lat("HNF reduced", &result_lat);

    eprintln!("--- expected ---");
    eprintln!("  L_int (kernel-derived ground truth): covol = 911 bits (det 915, denom 2)");

    // Now run the full DSD pipeline with CANONICAL inputs at every
    // dual step. If the bug is in `Lattice::sum`'s handling of
    // non-canonical (basis, denom) representations, this should give
    // the right answer.
    eprintln!("--- full DSD path with canonicalize-after-each-dual ---");
    let d1c = canonicalize_lat(&d1);
    let d2c = canonicalize_lat(&d2);
    let sum_c_hnf = d1c.sum(&d2c);
    let sum_c_lat: Lattice<W> = sum_c_hnf.into();
    let result_c_pre = sum_c_lat.dual();
    let result_c = canonicalize_lat(&result_c_pre);
    report_lat("DSD-canon final", &result_c);

    // Most aggressive variant: canonicalize the SUM before taking
    // the second dual.
    eprintln!("--- full DSD path with canonicalize-sum-too ---");
    let sum_cc = canonicalize_lat(&sum_c_lat);
    let result_cc_pre = sum_cc.dual();
    let result_cc = canonicalize_lat(&result_cc_pre);
    report_lat("DSD-cc final", &result_cc);

    // Hypothesis: the bug is that non-modular HNF on 8 cols loses
    // some content. Try the C-ref recipe: modular HNF with modulus
    // gcd(det of scaled L1 cols, det of scaled L2 cols).
    eprintln!("--- DSD with modular-HNF sum (C-ref recipe) ---");
    {
        // Mimic the C-ref `quat_lattice_add` exactly:
        //   1. Scale L2.basis by L1.denom → tmp_b. det1 = |det(tmp_b)|.
        //   2. Scale L1.basis by L2.denom → tmp_a. det2 = |det(tmp_a)|.
        //   3. modulus = gcd(det1, det2).
        //   4. HNF mod modulus over the 8 cols.
        //   5. Result denom = L1.denom · L2.denom; reduce.
        let scale_basis = |basis: &Matrix<W>, s: BigInt<W>| -> Matrix<W> {
            let mut out = Matrix::<W>::ZERO;
            for r in 0..4 {
                for c in 0..4 {
                    out[r][c] = basis[r][c].ct_mul(&s);
                }
            }
            out
        };
        let tmp_a = scale_basis(d1.basis(), *d2.denom()); // d1.basis · d2.denom
        let tmp_b = scale_basis(d2.basis(), *d1.denom()); // d2.basis · d1.denom
        let det1 = tmp_a.det().abs();
        let det2 = tmp_b.det().abs();
        let modulus = det1.gcd(&det2);
        eprintln!(
            "  det1 = {} bits, det2 = {} bits, gcd = {} bits",
            det1.bitsize(),
            det2.bitsize(),
            modulus.bitsize()
        );
        let cols_a = tmp_a.columns();
        let cols_b = tmp_b.columns();
        let all_cols = [
            cols_a[0], cols_a[1], cols_a[2], cols_a[3], cols_b[0], cols_b[1], cols_b[2], cols_b[3],
        ];
        let common_denom = d1.denom().ct_mul(d2.denom());
        // Need W' such that mod-HNF intermediates fit. Modular HNF
        // entries are bounded by `modulus` (~ 2x its bitsize for
        // 2-product), so width ~2·modulus_bits + slack should fit.
        let sum_basis_mod = Matrix::<W>::from_hnf_columns_mod::<W>(&all_cols, &modulus);
        let sum_mod_lat = Lattice::<W>::new(sum_basis_mod, common_denom);
        report_lat("sum (mod-HNF)        ", &sum_mod_lat);

        let result_pre = sum_mod_lat.dual();
        let result_canon = canonicalize_lat(&result_pre);
        report_lat("DSD-mod final", &result_canon);

        // Try: non-modular HNF on the SAME 8 cols + 4 modulus cols.
        // If this matches modular HNF's answer, the bug is just that
        // non-modular HNF on 8 cols (without the implicit modulus
        // generators) doesn't produce the canonical sum HNF.
        let mut all_cols_with_mod = vec![
            cols_a[0], cols_a[1], cols_a[2], cols_a[3], cols_b[0], cols_b[1], cols_b[2], cols_b[3],
        ];
        for i in 0..4 {
            let mut e_i = Vector::<W>::ZERO;
            e_i[i] = modulus;
            all_cols_with_mod.push(e_i);
        }
        let sum_basis_12 = Matrix::<W>::from_hnf_columns(&all_cols_with_mod);
        let sum_12_lat = Lattice::<W>::new(sum_basis_12, common_denom);
        report_lat("sum (non-mod, 12 cols)", &sum_12_lat);
    }
}

/// Sanity: does `Lattice::sum` produce the correct answer on simple
/// `2·Z^4 + 3·Z^4 = Z^4`? If yes, the bug from `dsd_step_by_step` is
/// triggered by something specific to our KAT-1 input shape; if no,
/// it's a general bug in `Lattice::sum` / `Matrix::from_hnf_columns`.
#[test]
fn sum_simple_2zn_plus_3zn() {
    let basis_2 = Matrix::<4>::from_columns(&[
        Vector::new(BigInt::from(2), BigInt::ZERO, BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::from(2), BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::from(2), BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ZERO, BigInt::from(2)),
    ]);
    let basis_3 = Matrix::<4>::from_columns(&[
        Vector::new(BigInt::from(3), BigInt::ZERO, BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::from(3), BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::from(3), BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ZERO, BigInt::from(3)),
    ]);
    let l1 = Lattice::<4>::new(basis_2, BigInt::ONE);
    let l2 = Lattice::<4>::new(basis_3, BigInt::ONE);
    let sum = l1.sum(&l2);
    let det = sum.basis().det().abs();
    let denom = *sum.denom();
    eprintln!(
        "2Z^4 + 3Z^4: |det| = {} bits = {:?}, denom = {} bits = {:?}",
        det.bitsize(),
        det,
        denom.bitsize(),
        denom
    );
    assert_eq!(det, BigInt::<4>::ONE, "expected covol = 1 (= Z^4)");
}

/// Same sanity, but with non-trivial off-diagonal structure mimicking
/// the dual outputs.
#[test]
fn sum_off_diagonal_simple() {
    // L1 = upper-triangular [(2, 1, 0, 0), (0, 3, 0, 0), (0, 0, 1, 0), (0, 0, 0,
    // 1)]
    let basis_a = Matrix::<4>::from_columns(&[
        Vector::new(BigInt::from(2), BigInt::ZERO, BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::from(1), BigInt::from(3), BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ONE, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ZERO, BigInt::ONE),
    ]);
    // L2 = upper-triangular [(5, 0, 0, 0), (0, 7, 0, 0), (0, 0, 1, 0), (0, 0, 0,
    // 1)]
    let basis_b = Matrix::<4>::from_columns(&[
        Vector::new(BigInt::from(5), BigInt::ZERO, BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::from(7), BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ONE, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ZERO, BigInt::ONE),
    ]);
    let l1 = Lattice::<4>::new(basis_a, BigInt::ONE);
    let l2 = Lattice::<4>::new(basis_b, BigInt::ONE);
    let sum = l1.sum(&l2);
    let det = sum.basis().det().abs();
    let denom = *sum.denom();
    // L1 + L2 should be Z^4 since gcd(2,5)=1, gcd(3,7)=1, etc.
    eprintln!(
        "off-diag sum: |det| = {} = {:?}, denom = {:?}",
        det.bitsize(),
        det,
        denom
    );
}
