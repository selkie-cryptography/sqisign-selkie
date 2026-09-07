use super::*;
use crate::{
    curves::{
        BasisHint, ChangeOfBasisMatrix, TorsionBasis,
        montgomery::{Coefficient, Curve, ProjectiveXOnlyPoint},
    },
    fields::fp2::Fp2,
    params,
    quaternions::bigint::BigInt,
};

/// Builds the E₀ torsion basis from params.
fn e0_basis() -> TorsionBasis {
    let curve = Curve::E0;
    let p = ProjectiveXOnlyPoint::from_affine_x(params::BASIS_E0_P_X, &curve);
    let q = ProjectiveXOnlyPoint::from_affine_x(params::BASIS_E0_Q_X, &curve);
    let pmq = ProjectiveXOnlyPoint::from_affine_x(params::BASIS_E0_PMQ_X, &curve);
    TorsionBasis::from_propagated(p, q, pmq)
}

#[test]
fn tate_pairing_is_root_of_unity() {
    let basis = e0_basis();
    let e = TorsionExponent::FULL; // 248

    let zeta = TorsionBasis::from_propagated(basis.P, basis.Q, basis.PmQ).tate(e);

    // ζ ≠ 1 (non-degenerate pairing on a basis).
    assert_ne!(
        zeta,
        RootOfUnity::ONE,
        "pairing of basis should be non-trivial"
    );

    // ζ^{2^e} = 1 (it's a 2^e-th root of unity).
    let should_be_one = zeta.square_n(e.value());
    assert_eq!(should_be_one, RootOfUnity::ONE, "ζ^(2^e) should equal 1");
}

/// On the full E_0 basis (order 2^TORSION_EVEN_POWER), ord(ζ) is
/// the full 2^e exponent — i.e., ζ is primitive. Sister test to
/// [`tate_pairing_primitive_on_reduced_basis`] documenting the
/// contract our existing test only loosely checks.
#[test]
fn tate_pairing_primitive_on_full_basis() {
    let basis = e0_basis();
    let e = TorsionExponent::FULL;
    let zeta = TorsionBasis::from_propagated(basis.P, basis.Q, basis.PmQ).tate(e);

    let ord = (0..=e.value() + 4).find(|&k| zeta.square_n(k) == RootOfUnity::ONE);
    assert_eq!(ord, Some(e.value()), "ord(ζ) must equal 2^{}", e.value());
}

/// `TorsionBasis::cross_pairings`'s canonical reference `w[0]`
/// must be a primitive 2^e-th root of unity. The other four
/// outputs only have full order if all four basis-coefficient
/// dlogs are odd, which conflicts with the `(R, S)` basis
/// constraint `det = ad − bc` being odd — so at least one of
/// `w[1..4]` will land in a proper subgroup of μ_{2^e}. The
/// `cross_pairings_dlog_roundtrip` test below verifies the
/// reduced-precision dlogs still reconstruct the input matrix
/// correctly.
#[test]
fn cross_pairings_canonical_primitive() {
    let pq_full = e0_basis();
    let e_red: u32 = 128;
    let scale = TorsionExponent::FULL.value() - e_red;
    let scale_scalar = Scalar::from_limbs(*(BigInt::<4>::ONE << scale).as_limbs());

    let one = Scalar::from_u64(1);
    let two = Scalar::from_u64(2);
    let r_full = pq_full.biscalar_mul(&two, &one, TorsionExponent::FULL);
    let s_full = pq_full.biscalar_mul(&one, &one, TorsionExponent::FULL);
    // R − S = (2P + Q) − (P + Q) = P.
    let reduced = TorsionBasis::from_propagated(
        &scale_scalar * &r_full,
        &scale_scalar * &s_full,
        &scale_scalar * &pq_full.P,
    );

    let e = TorsionExponent::try_from(e_red).expect("128 valid");
    let ws = pq_full
        .cross_pairings(&reduced, e)
        .expect("cross_pairings must succeed on E_0 bases");

    assert_ne!(ws[0], RootOfUnity::ONE, "w[0] must not be trivial");
    let ord = (0..=e.value() + 4).find(|&k| ws[0].square_n(k) == RootOfUnity::ONE);
    assert_eq!(
        ord,
        Some(e.value()),
        "w[0] must be primitive 2^{}",
        e.value()
    );
}

/// `cross_pairings` followed by `RootOfUnity::dlog` must reconstruct
/// the original matrix coefficients of the reduced basis when
/// expressed in the full-order canonical basis.
///
/// Setup: `R = 3P + 5Q`, `S = 3P + 4Q` at full order, then doubled
/// `e_diff` times to land at order 2^e. `R − S = Q` falls out by
/// construction. Expected dlogs (per the C ref's `r1·P + r2·Q = R`
/// contract): `(r1, r2) = (3, 5)`, `(s1, s2) = (3, 4)`.
#[test]
fn cross_pairings_dlog_roundtrip() {
    let pq_full = e0_basis();
    let e_red: u32 = 128;
    let scale = TorsionExponent::FULL.value() - e_red;
    let scale_scalar = Scalar::from_limbs(*(BigInt::<4>::ONE << scale).as_limbs());

    let alpha = Scalar::from_u64(3);
    let beta = Scalar::from_u64(5);
    let gamma = Scalar::from_u64(3);
    let delta = Scalar::from_u64(4);
    let r_full = pq_full.biscalar_mul(&alpha, &beta, TorsionExponent::FULL);
    let s_full = pq_full.biscalar_mul(&gamma, &delta, TorsionExponent::FULL);
    // R − S = (3P + 5Q) − (3P + 4Q) = Q.
    let reduced = TorsionBasis::from_propagated(
        &scale_scalar * &r_full,
        &scale_scalar * &s_full,
        &scale_scalar * &pq_full.PmQ,
    );

    let e = TorsionExponent::try_from(e_red).expect("128 valid");
    let ws = pq_full
        .cross_pairings(&reduced, e)
        .expect("cross_pairings must succeed on E_0 bases");

    // Per C ref's `tate_dlog_partial` (biextension.c:720):
    //   r2 = dlog_w0(w[1]),  r1 = dlog_w0(w[2]),
    //   s2 = dlog_w0(w[3]),  s1 = dlog_w0(w[4]).
    let r2 = ws[0].dlog(&ws[1], e);
    let r1 = ws[0].dlog(&ws[2], e);
    let s2 = ws[0].dlog(&ws[3], e);
    let s1 = ws[0].dlog(&ws[4], e);

    assert_eq!(r1, alpha, "r1 must equal α = 3");
    assert_eq!(r2, beta, "r2 must equal β = 5");
    assert_eq!(s1, gamma, "s1 must equal γ = 3");
    assert_eq!(s2, delta, "s2 must equal δ = 4");
}

/// Documents a known limitation of [`TorsionBasis::tate`]: its symmetric
/// cubical-ladder formulation produces ord(ζ) =
/// 2^(2·e − TORSION_EVEN_POWER) when both inputs are pre-reduced
/// to order 2^e (rather than primitively used at the curve's
/// 2^TORSION_EVEN_POWER torsion). For e=128, e_full=248 this is
/// 2^8, which collapses any change-of-basis matrix at e_cob ≤ 124.
///
/// Sign and verify route around this by using
/// [`TorsionBasis::cross_pairings`] (asymmetric ladder, full-order
/// canonical × reduced-order target) which produces a primitive
/// 2^e-th root. See [`from_bases`] for the change-of-basis call
/// site.
///
/// Fixing the symmetric path would require restructuring the
/// cubical ladder to track the path-dependent `xq^k` factor that
/// differential_add accumulates — an open task tracked separately.
/// This test asserts the *current* (broken) behavior so a
/// downstream fix to [`TorsionBasis::tate`] flags as a regression.
///
/// [`from_bases`]: crate::curves::ChangeOfBasisMatrix::from_bases
#[test]
fn tate_pairing_subprimitive_on_reduced_basis() {
    let basis = e0_basis();
    let e_full = TorsionExponent::FULL.value();
    let e_red: u32 = 128;
    let scale = e_full - e_red;
    let scale_scalar = Scalar::from_limbs(*(BigInt::<4>::ONE << scale).as_limbs());

    let r = &scale_scalar * &basis.P;
    let s = &scale_scalar * &basis.PmQ;
    let rs = &scale_scalar * &basis.Q;
    let reduced = TorsionBasis::from_propagated(r, s, rs);

    let e = TorsionExponent::try_from(e_red).expect("128 is a valid TorsionExponent");
    let zeta = TorsionBasis::from_propagated(reduced.P, reduced.Q, reduced.PmQ).tate(e);
    assert_ne!(zeta, RootOfUnity::ONE);

    let ord = (0..=e.value() + 4).find(|&k| zeta.square_n(k) == RootOfUnity::ONE);
    let expected = 2 * e_red - e_full;
    assert_eq!(
        ord,
        Some(expected),
        "current symmetric TorsionBasis::tate on reduced bases gives \
         ord(ζ) = 2^(2·e − e_full) = 2^{expected}; if this assertion \
         fires, the symmetric path was fixed and `cross_pairings` may \
         no longer be necessary"
    );
}

#[test]
fn dlog_round_trip_large() {
    let basis = e0_basis();
    let e = TorsionExponent::FULL;
    let zeta = TorsionBasis::from_propagated(basis.P, basis.Q, basis.PmQ).tate(e);

    // dlog with full exponent: ζ^42 should round-trip.
    let zeta42 = zeta.pow(42);
    let k = zeta.dlog(&zeta42, e);
    assert_eq!(k, Scalar::from_u64(42), "dlog(ζ^42) should be 42");
}

/// Dlog must round-trip for exponents that exceed `u32::MAX`.
///
/// `M_chl` / `M_sk` entries reach `2^126`, so the dlog must
/// preserve all 126 bits. An earlier implementation truncated
/// `k'` (the low half of the dlog recursion) to `u32` before
/// calling `pow`, which produced a fixed root of unity for any
/// `k' ≥ 2^32` — every cross-pairing dlog collapsed to the same
/// value and `from_bases` returned a constant matrix.
#[test]
fn dlog_round_trip_above_u32() {
    let basis = e0_basis();
    let e = TorsionExponent::FULL; // 248
    let zeta = TorsionBasis::from_propagated(basis.P, basis.Q, basis.PmQ).tate(e);

    // Pick a value with bits set above 2^32 so any `as u32` cast
    // would lose information.
    let k_in = Scalar::from_limbs([0x1234_5678_9ABC_DEF0, 0x55, 0, 0]);
    let zeta_k = zeta.pow_scalar(&k_in);
    let k_out = zeta.dlog(&zeta_k, e);
    assert_eq!(
        k_out, k_in,
        "dlog(ζ^k) must round-trip for k with bits above 2^64"
    );
}

/// Verifies Tate pairing bilinearity (P+Q convention).
///
/// `T([2]P, Q, [2]P+Q) == T(P, Q, P+Q)^2`. The third argument
/// is the SUM, computed via `differential_add(P, Q, P-Q)`.
///
/// Currently fails: our cubical Tate implementation does not
/// satisfy this relation. The output is always a 2^e-th root of
/// unity (verified by `tate_pairing_is_root_of_unity`), so the
/// algorithm produces "a pairing-like value" — but it does not
/// scale linearly under doubling of the first argument. This
/// blocks Weil-pairing-based codomain disambiguation in
/// `LeftIdeal::to_isogeny` (which the C reference uses to pick
/// `codomain.E1` vs `codomain.E2` in `dim2id2iso.c:1148-1178`).
#[test]
fn tate_bilinear_in_first_arg_with_sum() {
    let basis = e0_basis();
    let e = TorsionExponent::FULL;

    // Compute P+Q from (P, Q, P-Q): differential_add(P, Q, P-Q) = P + Q.
    let ppq = basis.P.differential_add(&basis.PmQ, &basis.Q);
    let t_pq = TorsionBasis::from_propagated(basis.P, ppq, basis.PmQ).tate(e);
    let t_pq_squared = t_pq.square_n(1);

    let p2 = basis.P.double();
    // [2]P + Q via differential_add([2]P, Q, [2]P-Q).
    // [2]P-Q from differential_add(P, P-Q, Q).
    let two_p_minus_q = basis.P.differential_add(&basis.Q, &basis.PmQ);
    let two_p_plus_q = p2.differential_add(&basis.PmQ, &two_p_minus_q);

    let t_2p_q = TorsionBasis::from_propagated(p2, two_p_plus_q, basis.PmQ).tate(e);
    assert_eq!(
        t_2p_q, t_pq_squared,
        "Tate bilinearity (P+Q form): T([2]P, Q, [2]P+Q) should equal T(P, Q, P+Q)^2"
    );
}

/// Verifies Tate pairing bilinearity (P-Q convention).
///
/// `T([2]P, Q, [2]P-Q) == T(P, Q, P-Q)^2`. Same root cause as
/// [`tate_bilinear_in_first_arg_with_sum`] — fails for both
/// sum and difference conventions of the third argument.
#[test]
fn tate_bilinear_in_first_arg_with_diff() {
    let basis = e0_basis();
    let e = TorsionExponent::FULL;

    let t_pq = TorsionBasis::from_propagated(basis.P, basis.Q, basis.PmQ).tate(e);
    let t_pq_squared = t_pq.square_n(1);

    let p2 = basis.P.double();
    // [2]P - Q via differential_add(P, P-Q, Q).
    let two_p_minus_q = basis.P.differential_add(&basis.Q, &basis.PmQ);

    let t_2p_q = TorsionBasis::from_propagated(p2, two_p_minus_q, basis.PmQ).tate(e);
    assert_eq!(
        t_2p_q, t_pq_squared,
        "Tate bilinearity (P-Q form): T([2]P, Q, [2]P-Q) should equal T(P, Q, P-Q)^2"
    );
}

/// Verifies Tate pairing antisymmetry — the property
/// `from_bases` relies on for the cross-pairing dlog.
///
/// Specifically: `t(P, Q) · t(Q, P) == 1` so that
/// `ζ_2 = 1/t(target.P, full.P)` correctly recovers
/// `ζ^{coefficient of P in target.P}`.
#[test]
fn tate_antisymmetric() {
    let basis = e0_basis();
    let e = TorsionExponent::FULL;

    // P+Q for the third arg per `tate_pairing`'s convention.
    let ppq = basis.P.differential_add(&basis.PmQ, &basis.Q);

    let t_pq = TorsionBasis::from_propagated(basis.P, ppq, basis.PmQ).tate(e);
    let t_qp = TorsionBasis::from_propagated(basis.PmQ, ppq, basis.P).tate(e);
    let product = t_pq.as_fp2() * t_qp.as_fp2();
    assert_eq!(
        product,
        Fp2::ONE,
        "Tate antisymmetry: t(P, Q) · t(Q, P) should equal 1"
    );
}

/// Verifies Weil pairing antisymmetry on the natural basis form:
/// `basis.weil(e) * basis_swapped.weil(e) == 1`, where `basis_swapped`
/// is `(Q, PmQ, P)` (the role of P and Q swapped, with x-symmetric
/// `PmQ` shared between both bases).
#[test]
fn weil_antisymmetric() {
    let basis = e0_basis();
    let e = TorsionExponent::FULL;

    let basis_swapped = TorsionBasis {
        P: basis.Q,
        PmQ: basis.PmQ,
        Q: basis.P,
    };
    let w_pq = basis.weil(e);
    let w_qp = basis_swapped.weil(e);
    let product = w_pq.as_fp2() * w_qp.as_fp2();
    assert_eq!(
        product,
        Fp2::ONE,
        "Weil antisymmetry: basis.weil() · basis_swapped.weil() should equal 1"
    );
}

/// `basis.weil(e)` produces a non-trivial 2^e-th root of unity on
/// a full-order E_0 basis. Direct smoke test of the natural method
/// shape used by sign-side and verify-side production callers.
#[test]
fn weil_natural_form_root_of_unity() {
    let basis = e0_basis();
    let e = TorsionExponent::FULL;

    let w = basis.weil(e);
    assert_ne!(w, RootOfUnity::ONE, "weil on a basis must be non-trivial");
    assert_eq!(
        w.square_n(e.value()),
        RootOfUnity::ONE,
        "weil output must be a 2^e-th root of unity"
    );
}

#[test]
fn dlog_round_trip() {
    let basis = e0_basis();

    // Use the full-order pairing which is guaranteed primitive.
    let e = TorsionExponent::FULL; // 248
    let zeta = TorsionBasis::from_propagated(basis.P, basis.Q, basis.PmQ).tate(e);
    assert_ne!(zeta, RootOfUnity::ONE);

    // Use a small exponent for the dlog test by squaring down.
    // ζ' = ζ^{2^{248-10}} is a primitive 2^10-th root.
    let e_small = TorsionExponent::try_from(10).unwrap();
    let zeta_small = zeta.square_n(248 - 10);
    assert_ne!(zeta_small, RootOfUnity::ONE);
    assert_eq!(zeta_small.square_n(10), RootOfUnity::ONE);

    // ζ'^7 should dlog back to 7.
    let zeta7 = zeta_small.pow(7);
    let k = zeta_small.dlog(&zeta7, e_small);
    assert_eq!(k, Scalar::from_u64(7), "dlog(ζ'^7) should be 7");
}

/// `mul(from_bases(A, B), A)` must equal `B` in x-only sense
/// even when `B` is constructed via a unimodular matrix on `A`.
///
/// This guards against breaking the weaker (downstream-relevant)
/// invariant when fixing the stronger `from_bases_independent_bases`
/// invariant. The chain consumer of `M_chl` only needs x-only
/// equality.
#[test]
fn from_bases_x_only_roundtrip_unimodular() {
    let basis_a = e0_basis();
    let e = TorsionExponent::FULL;
    let m_ab = [
        [Scalar::from_u64(2), Scalar::from_u64(1)],
        [Scalar::from_u64(1), Scalar::from_u64(1)],
    ];
    let p_b = basis_a.biscalar_mul(&m_ab[0][0], &m_ab[1][0], e);
    let q_b = basis_a.biscalar_mul(&m_ab[0][1], &m_ab[1][1], e);
    let k = e.value();
    let pmq_a_scalar = m_ab[0][0].sub_mod2k(&m_ab[0][1], k);
    let pmq_c_scalar = m_ab[1][0].sub_mod2k(&m_ab[1][1], k);
    let pmq_b = basis_a.biscalar_mul(&pmq_a_scalar, &pmq_c_scalar, e);
    let basis_b = TorsionBasis::from_propagated(p_b, q_b, pmq_b);

    let recovered =
        ChangeOfBasisMatrix::from_bases(&basis_a, &basis_b, e).expect("dlog should succeed");
    let applied = recovered.mul(&basis_a);
    assert_eq!(
        applied.P, basis_b.P,
        "x-only: applied.P must equal basis_b.P"
    );
    assert_eq!(
        applied.PmQ, basis_b.PmQ,
        "x-only: applied.PmQ must equal basis_b.PmQ"
    );
    assert_eq!(
        applied.Q, basis_b.Q,
        "x-only: applied.Q must equal basis_b.Q"
    );
}

/// Scaling by `2^k` via `Scalar * Point` matches scaling via
/// `k` repeated doublings — in x-only sense.
///
/// Signing-side reduces basis via `Scalar(2^k) * point`. Verify
/// reduces basis via a doubling loop. Both paths must produce
/// the same affine x for the post-scaling basis to match across
/// signing and verify.
#[test]
fn scalar_mul_pow2_matches_doubling() {
    let basis = e0_basis();
    let target = basis.P;
    let k_bits = 60u32; // arbitrary, exercise the multi-bit ladder

    let scale_scalar = Scalar::from_limbs(*(BigInt::<4>::ONE << k_bits).as_limbs());
    let via_mul = &scale_scalar * &target;

    let mut via_double = target;
    for _ in 0..k_bits {
        via_double = via_double.double();
    }

    assert_eq!(
        via_mul, via_double,
        "[2^k]P via scalar mul must equal [2^k]P via repeated doubling (x-only)"
    );
}

/// `biscalar_mul(k, l).x == biscalar_mul(-k, -l).x`.
///
/// Underlies a lot of x-only reasoning: scalars `k` and `-k`
/// (= `2^e − k`) produce points that differ only in y-sign, so
/// the affine x must be identical. If this property fails, the
/// "x-only equivalence" arguments throughout `from_bases` and
/// the bench/verify path break down.
#[test]
fn biscalar_mul_negation_x_only() {
    let basis = e0_basis();
    let e = TorsionExponent::FULL;

    let k = Scalar::from_u64(13);
    let l = Scalar::from_u64(7);
    let neg_k = Scalar::ZERO.sub_mod2k(&k, e.value());
    let neg_l = Scalar::ZERO.sub_mod2k(&l, e.value());

    let p_pos = basis.biscalar_mul(&k, &l, e);
    let p_neg = basis.biscalar_mul(&neg_k, &neg_l, e);

    assert_eq!(
        p_pos, p_neg,
        "biscalar_mul([k, l]) and biscalar_mul([-k, -l]) must give same x-only point"
    );
}

/// `from_hint(curve, to_hint(curve).hint)` must reproduce
/// `to_hint(curve).basis` exactly.
///
/// In the signing pipeline, signing computes
/// `(basis, hint) = to_hint(e_chl_final)` and embeds `hint` in
/// the signature. Verify reconstructs the canonical basis via
/// `from_hint(curve_chl, hint)`. For the matrix `M_chl` that
/// signing computes (against signing's `basis`) to apply
/// correctly on verify's reconstructed basis, the two bases must
/// be identical.
///
/// This test asserts the round-trip on `E_0`.  If it fails,
/// signing's `det_chl_scaled` and verify's `basis_chl_scaled`
/// would disagree, and the matrix `M_chl` — even if computed
/// correctly relative to signing's basis — would map to a
/// different basis when applied on verify, producing the
/// observed `transformed.P != post-M_chl.P` mismatch.
#[test]
fn to_hint_from_hint_roundtrip_e0() {
    // E_0: A = 0 by NIST-I convention.
    let curve = Curve::from(Coefficient::ZERO);

    let (basis_via_to, hint) =
        TorsionBasis::to_hint(&curve).expect("test: to_hint failed on honest curve");
    let basis_via_from = TorsionBasis::from_hint(&curve, BasisHint::from_byte(hint.to_byte()))
        .expect("test: from_hint failed on honest curve");

    assert_eq!(
        basis_via_to.P, basis_via_from.P,
        "to_hint/from_hint round-trip must match on E_0: R differs"
    );
    assert_eq!(
        basis_via_to.PmQ, basis_via_from.PmQ,
        "to_hint/from_hint round-trip must match on E_0: S differs"
    );
    assert_eq!(
        basis_via_to.Q, basis_via_from.Q,
        "to_hint/from_hint round-trip must match on E_0: RS differs"
    );
}

/// `from_bases` must recover the *exact* matrix entries used to
/// build the target — not just an x-only-equivalent.
///
/// `mul(M, source).x == mul(-M, source).x` for x-only points, so a
/// matrix-level round-trip via `mul` catches transposes but
/// silently accepts negated entries. Verify-side downstream
/// consumers (the `(2,2)`-isogeny chain) need the actual scalar
/// values, not just x-only equivalents — so this test asserts
/// `recovered.entries == m_known` directly.
#[test]
fn from_bases_recovers_known_entries() {
    let source = e0_basis();
    let e = TorsionExponent::FULL;

    let m_known = [
        [Scalar::from_u64(3), Scalar::from_u64(5)],
        [Scalar::from_u64(7), Scalar::from_u64(11)],
    ];
    // Target with matrix M applied via the convention `from_bases`
    // expects (column-major: column j of M = coefficients of
    // target.basis[j] in source).
    let p_target = source.biscalar_mul(&m_known[0][0], &m_known[1][0], e);
    let q_target = source.biscalar_mul(&m_known[0][1], &m_known[1][1], e);
    let k = e.value();
    let pmq_a = m_known[0][0].sub_mod2k(&m_known[0][1], k);
    let pmq_c = m_known[1][0].sub_mod2k(&m_known[1][1], k);
    let pmq_target = source.biscalar_mul(&pmq_a, &pmq_c, e);
    let target = TorsionBasis::from_propagated(p_target, q_target, pmq_target);

    let recovered =
        ChangeOfBasisMatrix::from_bases(&source, &target, e).expect("dlog should succeed");

    assert_eq!(
        recovered.entries[0][0], m_known[0][0],
        "entries[0][0]: got {:?}, expected {:?}",
        recovered.entries[0][0], m_known[0][0]
    );
    assert_eq!(
        recovered.entries[0][1], m_known[0][1],
        "entries[0][1]: got {:?}, expected {:?}",
        recovered.entries[0][1], m_known[0][1]
    );
    assert_eq!(
        recovered.entries[1][0], m_known[1][0],
        "entries[1][0]: got {:?}, expected {:?}",
        recovered.entries[1][0], m_known[1][0]
    );
    assert_eq!(
        recovered.entries[1][1], m_known[1][1],
        "entries[1][1]: got {:?}, expected {:?}",
        recovered.entries[1][1], m_known[1][1]
    );
}

/// Round-trip: `from_bases(A, B) · A == B`.
///
/// Catches transposes between the column-major storage produced by
/// [`ChangeOfBasisMatrix::from_bases`] and the column-applied
/// semantics of `mul`. Without this, `from_bases` and `mul` can be
/// internally inconsistent and only fail in the full sign + verify
/// round-trip — which costs minutes per attempt.
#[test]
fn from_bases_mul_roundtrip() {
    let source = e0_basis();
    let e = TorsionExponent::FULL;

    // Build a target = M · source for a known M with small entries
    // (still applied modulo 2^e via biscalar).
    let m_known = [
        [Scalar::from_u64(3), Scalar::from_u64(5)],
        [Scalar::from_u64(7), Scalar::from_u64(11)],
    ];
    let p_target = source.biscalar_mul(&m_known[0][0], &m_known[1][0], e);
    let q_target = source.biscalar_mul(&m_known[0][1], &m_known[1][1], e);
    // Difference via biscalar to avoid a sqrt branch flip.
    let k = e.value();
    let pmq_a = m_known[0][0].sub_mod2k(&m_known[0][1], k);
    let pmq_c = m_known[1][0].sub_mod2k(&m_known[1][1], k);
    let pmq_target = source.biscalar_mul(&pmq_a, &pmq_c, e);
    let target = TorsionBasis::from_propagated(p_target, q_target, pmq_target);

    // Recover M via from_bases, then check that mul reproduces target.
    let recovered =
        ChangeOfBasisMatrix::from_bases(&source, &target, e).expect("dlog should succeed");
    let applied = recovered.mul(&source);

    assert_eq!(
        applied.P, target.P,
        "from_bases + mul round-trip must reproduce target.P"
    );
    assert_eq!(
        applied.PmQ, target.PmQ,
        "from_bases + mul round-trip must reproduce target.PmQ"
    );
}

/// `from_bases` round-trip with matrix entries that exceed `2^32`.
///
/// `m_known` is chosen so that the recovered dlogs straddle the
/// 32-bit boundary in the recursion's intermediate `k'` value.
/// Catches the historic bug where `dlog` cast the recursive
/// `k'` to `u32` and silently truncated for `k' > u32::MAX`,
/// collapsing every `M_chl` cross-pairing to a fixed root of
/// unity (\S\ref{sec:dlog-truncation} in the bug catalog).
#[test]
fn from_bases_mul_roundtrip_above_u32() {
    let source = e0_basis();
    let e = TorsionExponent::FULL;

    // Entries straddling 2^32 in different limbs to force the
    // dlog recursion's `k'` past u32::MAX at multiple levels.
    let m_known = [
        [
            Scalar::from_limbs([0xDEAD_BEEF_1234_5678, 0x1234_5678_ABCD_EF01, 0, 0]),
            Scalar::from_limbs([0xFEDC_BA98_7654_3210, 0xCAFE_0000_0000_0001, 0, 0]),
        ],
        [
            Scalar::from_limbs([0x0123_4567_89AB_CDEF, 0xFACE_FACE_FACE_FACE, 0, 0]),
            Scalar::from_limbs([0xA5A5_A5A5_A5A5_A5A5, 0x5A5A_5A5A_5A5A_5A5A, 0, 0]),
        ],
    ];

    let p_target = source.biscalar_mul(&m_known[0][0], &m_known[1][0], e);
    let q_target = source.biscalar_mul(&m_known[0][1], &m_known[1][1], e);
    let k = e.value();
    let pmq_a = m_known[0][0].sub_mod2k(&m_known[0][1], k);
    let pmq_c = m_known[1][0].sub_mod2k(&m_known[1][1], k);
    let pmq_target = source.biscalar_mul(&pmq_a, &pmq_c, e);
    let target = TorsionBasis::from_propagated(p_target, q_target, pmq_target);

    let recovered =
        ChangeOfBasisMatrix::from_bases(&source, &target, e).expect("dlog should succeed");
    let applied = recovered.mul(&source);

    assert_eq!(
        applied.P, target.P,
        "from_bases + mul round-trip with k > u32 must reproduce target.P"
    );
    assert_eq!(
        applied.PmQ, target.PmQ,
        "from_bases + mul round-trip with k > u32 must reproduce target.PmQ"
    );
}

/// Exercise the exact `from_bases_invert → mul → from_bases` chain
/// `keys::signing` uses to build `m1` and `m_chl`. Catches breakage
/// in the inverse-direction matrix or in the `transformed = m1·b`
/// → `from_bases(canonical, transformed)` second hop, both of
/// which the simpler `from_bases_mul_roundtrip` test misses.
///
/// Setup mirrors signing:
/// 1. `canonical_a` = E_0 basis at full order (proxy for `det_aux`).
/// 2. `reduced_a` = M_known applied to `canonical_a` via biscalar at `e_cob`
///    (proxy for `basis_aux`). Order 2^e_cob.
/// 3. `m1 = from_bases_invert(canonical_a, reduced_a, e_cob)` — expects
///    `m1·reduced_a = canonical_a` at 2^e_cob precision.
/// 4. Apply `m1` to a different "shared" reduced basis via `mul`; the result is
///    `transformed`.
/// 5. `m_chl = from_bases(canonical_a, transformed, e_cob)` should succeed (all
///    lifts work) and reproduce a known relationship.
///
/// Step 5 is the load-bearing assertion: if `transformed` (= the
/// mul output) has lift-incompatible (R, S, RS), the second
/// `from_bases` returns `None` and signing drops. This is the
/// failure mode observed when running `sign_kat_zero_only`.
#[test]
fn from_bases_invert_mul_then_from_bases_chain() {
    let canonical = e0_basis();
    let e_red: u32 = 128;
    let scale = TorsionExponent::FULL.value() - e_red;
    let scale_scalar = Scalar::from_limbs(*(BigInt::<4>::ONE << scale).as_limbs());
    let e = TorsionExponent::try_from(e_red).expect("128 valid");

    // m_known: det = 3·5 − 4·1 = 11 (odd) → invertible mod 2^128.
    let m_known = [
        [Scalar::from_u64(3), Scalar::from_u64(4)],
        [Scalar::from_u64(1), Scalar::from_u64(5)],
    ];
    // Build reduced basis at order 2^e_cob: biscalar at FULL order,
    // then scale down by 2^scale.
    let p_full = canonical.biscalar_mul(&m_known[0][0], &m_known[1][0], TorsionExponent::FULL);
    let q_full = canonical.biscalar_mul(&m_known[0][1], &m_known[1][1], TorsionExponent::FULL);
    let k = TorsionExponent::FULL.value();
    let pmq_a = m_known[0][0].sub_mod2k(&m_known[0][1], k);
    let pmq_c = m_known[1][0].sub_mod2k(&m_known[1][1], k);
    let pmq_full = canonical.biscalar_mul(&pmq_a, &pmq_c, TorsionExponent::FULL);
    let reduced = TorsionBasis::from_propagated(
        &scale_scalar * &p_full,
        &scale_scalar * &q_full,
        &scale_scalar * &pmq_full,
    );

    // Step 3: m1 = inverse direction. m1·reduced ≡ canonical (at 2^e_cob).
    let m1 = ChangeOfBasisMatrix::from_bases_invert(&canonical, &reduced, e)
        .expect("from_bases_invert must succeed on full + reduced E_0 bases");

    // Step 4: apply m1 to a reduced-order basis. Sign uses
    // `basis_chl` here (response-phase output, at order 2^e_cob).
    // We proxy with `reduced` (also order 2^e_cob) — the matrix
    // application then yields `transformed` at order 2^e_cob,
    // matching the sign-side call shape that m_chl consumes.
    let transformed = m1.mul(&reduced);

    // Step 5: second from_bases. This is the call sign currently
    // sees fail with `m_chl None`. transformed ≡ m1·reduced
    // ≡ m1·M·canonical = M^(-1)·M·canonical = canonical (mod 2^e).
    // So m_chl reconstructs the identity matrix at 2^e.
    let m_chl = ChangeOfBasisMatrix::from_bases(&canonical, &transformed, e)
        .expect("from_bases on transformed must succeed (lift consistency)");

    // m_chl·canonical = transformed = canonical → m_chl == identity.
    assert_eq!(
        m_chl.entries[0][0],
        Scalar::from_u64(1),
        "m_chl[0][0] must be 1"
    );
    assert_eq!(m_chl.entries[0][1], Scalar::ZERO, "m_chl[0][1] must be 0");
    assert_eq!(m_chl.entries[1][0], Scalar::ZERO, "m_chl[1][0] must be 0");
    assert_eq!(
        m_chl.entries[1][1],
        Scalar::from_u64(1),
        "m_chl[1][1] must be 1"
    );
}
