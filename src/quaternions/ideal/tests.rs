//! Tests for [`crate::quaternions::ideal`].

use rand_core::OsRng;

use super::super::{
    bigint::BigInt,
    lattice::{ExtremalOrder, Lattice, LeftIdeal, NrdBasis},
    linear::Vector,
    precomputed::{EXTREMAL_ORDERS, P_WIDE},
};
use crate::{drbg::Aes256CtrDrbg, params::QUAT_PRIME_COFACTOR};

/// Returns the reduced norm `a² + b² + p·(c² + d²)` of a quaternion
/// basis column `[a, b, c, d]` (numerators over the lattice denom),
/// computed at [`BigInt<8>`] so it cannot truncate.
///
/// The naive `BigInt<4>` form overflows: `p ≈ 2^250`, so a column
/// with `c² + d² ≥ 52` (i.e. `|c|` or `|d| ≳ 8`, common in a reduced
/// O_0-ideal basis) pushes `p·(c² + d²)` past 256 bits and silently
/// loses the high limb. That made the divisibility check below
/// spuriously fail on the unlucky `OsRng` draws that produce such a
/// basis — a flaky test, not a flaky `random_norm`. Widening every
/// operand to 8 limbs (512 bits) leaves ~250 bits of headroom.
fn nrd_column_wide(col: &Vector<4>) -> BigInt<8> {
    let a = col[0].widen::<8>();
    let b = col[1].widen::<8>();
    let c = col[2].widen::<8>();
    let d = col[3].widen::<8>();
    let p8 = P_WIDE;
    a.ct_mul(&a)
        .ct_add(&b.ct_mul(&b))
        .ct_add(&p8.ct_mul(&c.ct_mul(&c).ct_add(&d.ct_mul(&d))))
}

/// Miller-Rabin width check: `pow_mod_w<W>` requires
/// `64·W ≥ 2·bits(modulus) − 1`. For 379-bit moduli (aux-path
/// `represent_integer`) we need `W ≥ 12`. Running at `W = 9`
/// silently truncates every Miller-Rabin exponentiation and
/// makes every input look composite, so `represent_integer`
/// never finds a witness. This test confirms the issue by
/// comparing `is_probable_prime_w::<9>` and
/// `is_probable_prime_w::<20>` across a range of small
/// candidates built on top of a 379-bit base: the
/// 20-limb width is well above the safe threshold while the
/// 9-limb width is below it, and the two disagree on roughly
/// half the inputs.
#[test]
fn primality_w9_vs_w20_at_aux_magnitude() {
    // Base = 2^378 (a 379-bit even number). Try (base + k)
    // for small k and check that the wider width finds at
    // least one prime that the narrower width misses. If we
    // don't see any disagreement, either the width math is
    // fine at this bit width or the Rng seed is unlucky —
    // in either case, a valid workflow should never rely on
    // `W = 9` for 379-bit inputs.
    let mut base = [0u64; 8];
    base[5] = 1u64 << (378 - 64 * 5);
    let base_b = BigInt::<8>::from_limbs(base);
    assert_eq!(base_b.bitsize(), 379);
    let mut disagreements = 0;
    for k in 1..200i64 {
        let cand = base_b.ct_add(&BigInt::<8>::from_i64(2 * k - 1));
        // `cand` is an odd 379-bit integer.
        let wide_says_prime = cand.is_probable_prime_w::<20>(4);
        let narrow_says_prime = cand.is_probable_prime_w::<9>(4);
        if wide_says_prime && !narrow_says_prime {
            disagreements += 1;
        }
    }
    assert!(
        disagreements > 0,
        "expected `is_probable_prime_w::<9>` to miss primes that `<20>` accepts at 379 bits"
    );
}

/// Timing benchmark for `represent_integer` at aux-path
/// magnitudes (`M ≈ 2^377`). Run with `--nocapture` to read the
/// elapsed time printed on success.
#[test]
fn represent_integer_aux_magnitude() {
    let order = ExtremalOrder::<8>::from(EXTREMAL_ORDERS[0]);
    // Simulate the aux-path input: `m = QUAT_PRIME_COFACTOR`
    // (~2^251) times a 126-bit `aux_norm`. Use a fixed odd
    // 126-bit value so the measurement is reproducible.
    let m4 = QUAT_PRIME_COFACTOR;
    let m_wide = BigInt::<8>::from_limbs({
        let mut limbs = [0u64; 8];
        limbs[..4].copy_from_slice(m4.as_limbs());
        limbs
    });
    let aux_norm = BigInt::<8>::from_limbs({
        let mut limbs = [0u64; 8];
        limbs[0] = 0x1234_5678_9ABC_DEF1;
        limbs[1] = 0xFEDC_BA98_7654_3211;
        limbs
    });
    let mn = m_wide.ct_mul(&aux_norm);
    let t0 = std::time::Instant::now();
    let gamma = order.represent_integer(&mn, false, &mut OsRng);
    let elapsed = t0.elapsed();
    crate::selkie_trace!(
        "[aux-RI] mn.bits={} elapsed={elapsed:?} ok={}",
        mn.bitsize(),
        gamma.is_some()
    );
    assert!(gamma.is_some(), "represent_integer(~2^377) must find a γ");
}

#[test]
fn is_probable_prime_small() {
    let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
    for &p in &primes {
        assert!(
            BigInt::<4>::from_u64(p).is_probable_prime(6),
            "{p} should be prime"
        );
    }
    let composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 25, 27];
    for &c in &composites {
        assert!(
            !BigInt::<4>::from_u64(c).is_probable_prime(6),
            "{c} should not be prime"
        );
    }
}

#[test]
fn legendre_symbol() {
    assert_eq!(
        BigInt::<4>::legendre(&BigInt::from(1i64), &BigInt::from(7i64)),
        1
    );
    assert_eq!(
        BigInt::<4>::legendre(&BigInt::from(2i64), &BigInt::from(7i64)),
        1
    );
    assert_eq!(
        BigInt::<4>::legendre(&BigInt::from(3i64), &BigInt::from(7i64)),
        -1
    );
    assert_eq!(
        BigInt::<4>::legendre(&BigInt::from(4i64), &BigInt::from(7i64)),
        1
    );
    assert_eq!(
        BigInt::<4>::legendre(&BigInt::from(0i64), &BigInt::from(7i64)),
        0
    );
}

#[test]
fn random_ideal_prime_norm() {
    let n = BigInt::<4>::from_u64(7);
    let result = LeftIdeal::random_prime_norm(&n, &EXTREMAL_ORDERS[0]);
    if let Some(ideal) = result {
        assert_eq!(*ideal.norm(), n, "ideal norm should be N");
    }
}

#[test]
fn represent_integer_q1() {
    let p: BigInt<8> = P_WIDE;
    let order = ExtremalOrder::<8>::from(EXTREMAL_ORDERS[0]);
    let m = p.ct_add(&BigInt::TWO);

    let result = order.represent_integer(&m, false, &mut OsRng);
    if let Some(gamma) = result {
        // `represent_integer` returns `Element<8>`; compute nrd at a
        // wide width (coords reach ~2^259 for q ≥ 5 orders, so the
        // square + p factor needs ~770 bits).
        let (nrd_num, nrd_den) = gamma.norm_w::<16>();
        let (nrd, rem) = nrd_num.vt_div_rem(&nrd_den);
        assert!(bool::from(rem.is_zero()), "nrd not integer");
        assert_eq!(nrd, m.widen::<16>(), "nrd(γ) should equal M");
    }
}

/// Tries all seven extremal orders with NIST-I p.
#[test]
fn represent_integer_any_order_verifies_norm() {
    let p: BigInt<8> = P_WIDE;
    let m = p.ct_add(&BigInt::TWO);

    let result = ExtremalOrder::<8>::represent_integer_any(&m, &mut OsRng);
    if let Some(gamma) = result {
        let (nrd_num, nrd_den) = gamma.norm_w::<16>();
        let (nrd, rem) = nrd_num.vt_div_rem(&nrd_den);
        assert!(bool::from(rem.is_zero()), "nrd not integer");
        assert_eq!(nrd, m.widen::<16>(), "nrd(γ) should equal M");
    }
}

/// Two DRBG instances seeded identically must drive
/// `represent_integer` to the *same* γ. Locks in determinism
/// of the search loop (no `OsRng` leak, no
/// implementation-specific iteration order beyond the
/// `(z, t)` byte stream).
///
/// Required precondition for KAT byte-match: the C ref's
/// `quat_represent_integer` is deterministic given the DRBG
/// state, so ours must be too. Until our byte-stream is
/// validated against C ref ground truth, this test only
/// asserts self-consistency, not interop.
#[test]
fn represent_integer_deterministic_under_same_drbg_seed() {
    // Pick a target large enough that the search loop runs.
    // M = p + 2 (smallest valid input ≥ p that's odd).
    let p: BigInt<8> = P_WIDE;
    let m = p.ct_add(&BigInt::TWO);
    let order = ExtremalOrder::<8>::from(EXTREMAL_ORDERS[0]);

    let seed = [0x42u8; 48];
    let mut d1 = Aes256CtrDrbg::new(&seed);
    let mut d2 = Aes256CtrDrbg::new(&seed);

    let g1 = order.represent_integer(&m, false, &mut d1);
    let g2 = order.represent_integer(&m, false, &mut d2);

    // Both runs must agree on outcome (Some/None) and, when
    // Some, on the actual quaternion element returned.
    match (g1, g2) {
        (None, None) => {
            // Acceptable: search exhausted on both sides.
        }
        (Some(a), Some(b)) => {
            assert_eq!(
                a.a.as_bigint(),
                b.a.as_bigint(),
                "γ.a divergence between same-seed runs"
            );
            assert_eq!(a.b.as_bigint(), b.b.as_bigint(), "γ.b divergence");
            assert_eq!(a.c.as_bigint(), b.c.as_bigint(), "γ.c divergence");
            assert_eq!(a.d.as_bigint(), b.d.as_bigint(), "γ.d divergence");
            assert_eq!(
                a.denom.as_bigint(),
                b.denom.as_bigint(),
                "γ.denom divergence"
            );
        }
        _ => panic!("same-seed DRBGs disagreed on Some/None outcome"),
    }
}

#[test]
fn gram_matrix_nrd_identity_basis() {
    // Standard basis {1, i, j, k} has Gram matrix diag(1, 1, p, p).
    let cols: [Vector<8>; 4] = [
        Vector::new(BigInt::ONE, BigInt::ZERO, BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ONE, BigInt::ZERO, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ONE, BigInt::ZERO),
        Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ZERO, BigInt::ONE),
    ];
    let nrd = NrdBasis::new(cols);
    let gram = nrd.gram();
    let p: BigInt<8> = P_WIDE;

    assert_eq!(gram[0][0], BigInt::ONE, "nrd(1) = 1");
    assert_eq!(gram[1][1], BigInt::ONE, "nrd(i) = 1");
    assert_eq!(gram[2][2], p, "nrd(j) = p");
    assert_eq!(gram[3][3], p, "nrd(k) = p");

    // Off-diagonal entries should be zero.
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                assert!(
                    bool::from(gram[i][j].is_zero()),
                    "G[{i}][{j}] should be zero for orthogonal basis"
                );
            }
        }
    }
}

/// `random_prime_norm(N)` produces a valid O_0-ideal: every basis
/// element has nrd divisible by N (equivalently, by
/// N · denom²  at the integer-column level).
///
/// This test previously exposed a bug where `LeftIdeal<4>::new`
/// used `Element::mul` (GCD-normalized) while assuming a
/// uniform `order.denom · alpha.denom` for `o_alpha_denom`. The
/// denom mismatch let the HNF produce lattices containing the
/// unit element — impossible in a proper ideal of norm > 1.
/// Fixed by using `mul_direct` (no GCD normalization), so every
/// column is scaled consistently relative to the stored denom.
#[test]
fn random_prime_norm_lattice_actually_has_norm() {
    let n = BigInt::<4>::from_u64(7);
    let Some(ideal) = LeftIdeal::random_prime_norm(&n, &EXTREMAL_ORDERS[0]) else {
        return;
    };

    let lat: Lattice<4> = (*ideal.lattice()).into();
    let denom = *lat.denom();
    let denom_sq = denom.ct_mul(&denom);
    let n_times_denom_sq = n.ct_mul(&denom_sq);

    for j in 0..4 {
        let col = lat.basis().columns()[j];
        let nrd_col_8 = nrd_column_wide(&col);
        let divisor_8: BigInt<8> = n_times_denom_sq.widen();
        let (_, rem) = nrd_col_8.vt_div_rem(&divisor_8);
        assert!(
            bool::from(rem.is_zero()),
            "random_prime_norm(7) basis[{j}] nrd not divisible by 7·denom² — \
             lattice is not an O_0-ideal of norm 7",
        );
    }
}

/// `random_norm(N)` produces a valid O_0-ideal for composite N:
/// every basis element has nrd divisible by N.
///
/// The `random_norm` correctness fix (Task #28): compute γ·β at
/// `Element<8>` (`Element<4>::mul` truncates once product coords
/// reach ~2^388, since γ has coords ~2^129 from `represent_integer`),
/// then reduce each numerator coord mod `N · denom` to fit back in
/// `BigInt<4>`. The reduction preserves the ideal `O·α + O·N` since
/// the difference lives in `N · Z<1,i,j,k> ⊂ N · O_0 = O · N`.
///
/// The check itself uses [`nrd_column_wide`]: an earlier version
/// computed the reduced norm in `BigInt<4>`, which overflows once a
/// reduced basis column has `c² + d² ≥ 52` (since `p ≈ 2^250`). That
/// is common across `OsRng` draws, so the test flaked (passed only on
/// draws whose reduced basis kept the j/k coords small) while
/// `random_norm` was always correct.
#[test]
fn random_norm_lattice_actually_has_norm() {
    let n = BigInt::<4>::from_u64(143);

    // Draw many ideals: the invariant must hold on *every* `OsRng`
    // trajectory, not just a lucky one. With the old `BigInt<4>` nrd
    // check this loop tripped within a handful of iterations (most
    // reduced bases have a column with `c² + d² ≥ 52`).
    let mut built = 0usize;
    for _ in 0..300 {
        let Some(ideal) = LeftIdeal::random_norm(&n, &EXTREMAL_ORDERS[0], &mut OsRng) else {
            continue;
        };
        built += 1;

        let lat: Lattice<4> = (*ideal.lattice()).into();
        let denom = *lat.denom();
        let denom_sq = denom.ct_mul(&denom);
        let divisor_8: BigInt<8> = n.ct_mul(&denom_sq).widen();

        for j in 0..4 {
            let col = lat.basis().columns()[j];
            let nrd_col_8 = nrd_column_wide(&col);
            let (_, rem) = nrd_col_8.vt_div_rem(&divisor_8);
            assert!(
                bool::from(rem.is_zero()),
                "random_norm(143) basis[{j}] nrd not divisible by 143·denom² — \
                 lattice is not an O_0-ideal of norm 143"
            );
        }
    }
    // `random_norm(143)` should succeed often enough that the loop
    // exercised real draws; if it never built one, the test is vacuous.
    assert!(built > 0, "random_norm(143) never produced an ideal");
}

/// `smallest_equiv_with_delta` returns an ideal with norm
/// strictly smaller than the input (since LLL finds a short
/// basis vector) and a `δ` that lies in the original ideal.
/// Rather than re-prove containment (which would reconstruct
/// the `I · δ̄ / nrd(I)` identity), this test checks the norm
/// shrinkage and that the returned tuple is consistent with
/// [`Self::smallest_equiv`] (same reduced ideal).
#[test]
fn smallest_equiv_with_delta_consistent() {
    let n = BigInt::<4>::from_u64(13);
    let Some(ideal) = LeftIdeal::random_prime_norm(&n, &EXTREMAL_ORDERS[0]) else {
        return;
    };

    let (reduced_with_delta, _delta) = match ideal.smallest_equiv_with_delta() {
        Some(r) => r,
        None => return,
    };
    let reduced_only = ideal
        .smallest_equiv()
        .expect("smallest_equiv succeeded above");

    assert_eq!(
        reduced_with_delta.norm(),
        reduced_only.norm(),
        "both smallest_equiv variants must produce the same reduced norm",
    );
    // The reduced norm should be bounded by the input norm
    // (generically much smaller — O(√p) vs O(p)).
    assert!(
        reduced_with_delta.norm().bitsize() <= ideal.norm().bitsize(),
        "reduced norm should not exceed the original",
    );
}

/// `smallest_equiv_narrow<W>` on a small-prime ideal: no panic and
/// the result, if any, has non-trivial odd norm.
#[test]
fn smallest_equiv_narrow_basic() {
    let n = BigInt::<4>::from_u64(13);
    let Some(ideal) = LeftIdeal::random_prime_norm(&n, &EXTREMAL_ORDERS[0]) else {
        return;
    };
    let Some(reduced) = ideal.smallest_equiv_narrow::<8>() else {
        return;
    };

    let rn = *reduced.norm();
    assert!(!bool::from(rn.is_zero()), "reduced norm must be nonzero");
    assert_ne!(rn, BigInt::<4>::ONE, "reduced norm must be > 1 (non-unit)");
    assert!(
        !bool::from(rn.is_even()),
        "smallest_equiv_narrow must return an odd-norm equivalent (parent_norm even breaks invmod downstream)"
    );
}

/// `smallest_equiv_narrow` on a ~64-bit-prime ideal — exercises
/// the L2-on-class-gram path where DPE precision matters.
#[test]
fn smallest_equiv_narrow_larger_norm() {
    // ~64-bit prime; large enough that the class-gram entries
    // scale with `N(I) ≈ 2^64` but still inside `BigInt<4>` for
    // the input ideal.
    let n = BigInt::<4>::from_u64(0xFFFF_FFFF_FFFF_FFC5); // 2^64 - 59 (prime)
    let Some(ideal) = LeftIdeal::random_prime_norm(&n, &EXTREMAL_ORDERS[0]) else {
        return;
    };
    let Some(reduced) = ideal.smallest_equiv_narrow::<8>() else {
        return;
    };

    let rn = *reduced.norm();
    assert!(!bool::from(rn.is_zero()));
    assert_ne!(rn, BigInt::<4>::ONE);
    assert!(!bool::from(rn.is_even()));
    // LLL on the class gram should typically shrink an ~2^64 norm
    // to ~2^(p/2) = 2^126. We just check it doesn't grow.
    assert!(
        reduced.norm().bitsize() <= ideal.norm().bitsize() + 2,
        "reduced norm shouldn't be materially larger than input"
    );
}
