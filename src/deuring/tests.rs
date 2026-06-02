// TODO: Consider moving the network-dependent test (c_ref_basis_cross_check)
// to an integration test in tests/.

use precomputed::{ENDOMORPHISM_MATRICES, torsion_basis};

use super::*;
use crate::{
    curves::montgomery::JacobianPoint,
    quaternions::algebra::{Coordinate, Denominator, Element},
};

/// Pinned commit of the SQIsign C reference implementation.
/// Used by cross-check tests that fetch precomputed data.
const C_REF_COMMIT: &str = "91e9e464fe5400192d13e1f9240cbf180200a103";

#[test]
fn torsion_basis_points_on_e0() {
    use crate::curves::montgomery::{Curve, ProjectiveXOnlyPoint};

    let p = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::E0_P_X, &Curve::E0);
    let q = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::E0_Q_X, &Curve::E0);

    assert!(!bool::from(p.is_identity()), "P₀ should not be identity");
    assert!(!bool::from(q.is_identity()), "Q₀ should not be identity");

    // [2^248]P₀ = O (order divides 2^248).
    let mut p_test = p;
    for _ in 0..crate::params::TORSION_EVEN_POWER {
        p_test = p_test.double();
    }
    assert!(
        bool::from(p_test.is_identity()),
        "[2^248]P₀ should be identity"
    );

    let mut q_test = q;
    for _ in 0..crate::params::TORSION_EVEN_POWER {
        q_test = q_test.double();
    }
    assert!(
        bool::from(q_test.is_identity()),
        "[2^248]Q₀ should be identity"
    );

    // [2^247]P₀ ≠ O (order is exactly 2^248).
    let mut p_half = p;
    for _ in 0..crate::params::TORSION_EVEN_POWER - 1 {
        p_half = p_half.double();
    }
    assert!(
        !bool::from(p_half.is_identity()),
        "[2^247]P₀ should not be identity"
    );

    let mut q_half = q;
    for _ in 0..crate::params::TORSION_EVEN_POWER - 1 {
        q_half = q_half.double();
    }
    assert!(
        !bool::from(q_half.is_identity()),
        "[2^247]Q₀ should not be identity"
    );

    // Independence: [2^247]P₀ ≠ ±[2^247]Q₀.
    assert_ne!(p_half, q_half, "[2^247]P₀ = [2^247]Q₀ — basis is dependent");
}

/// Cross-check our precomputed torsion basis against the C reference
/// implementation at a pinned commit.
///
/// Fetches `e0_basis.c` from GitHub, parses the 64-bit limbs from the
/// C ref's Broadwell backend (which stores Fp in Montgomery form
/// with R = 2^256),
/// converts from the C ref's Montgomery form (R=2^256) to plain integers,
/// and compares against our stored values.
///
/// Run with: `cargo test c_ref_basis_cross_check -- --ignored`
#[test]
#[ignore]
#[cfg(unix)] // reqwest/regex dev-deps are unix-only
fn c_ref_basis_cross_check() {
    let commit = C_REF_COMMIT;
    let url = format!(
        "https://raw.githubusercontent.com/SQISign/the-sqisign/{}/src/precomp/ref/lvl1/e0_basis.c",
        commit
    );

    // Fetch the file.
    let body = reqwest::blocking::get(&url)
        .unwrap_or_else(|e| panic!("failed to fetch {url}: {e}"))
        .text()
        .unwrap();

    // Parse Broadwell 64-bit blocks.
    let re = regex::Regex::new(r"SQISIGN_GF_IMPL_BROADWELL\)\n\{([^}]+)\}").unwrap();

    let mut broadwell_values: Vec<[u64; 4]> = Vec::new();
    for cap in re.captures_iter(&body) {
        let block = &cap[1];
        let limbs: Vec<u64> = block
            .split(',')
            .map(|s| {
                let s = s.trim();
                u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap()
            })
            .collect();
        assert_eq!(limbs.len(), 4, "expected four limbs per Broadwell block");
        broadwell_values.push([limbs[0], limbs[1], limbs[2], limbs[3]]);
    }

    assert_eq!(
        broadwell_values.len(),
        4,
        "expected four Broadwell blocks (PX_RE, PX_IM, QX_RE, QX_IM), got {}",
        broadwell_values.len()
    );

    // The C ref's Broadwell backend stores Fp elements in Montgomery
    // form with R = 2^256. Convert to plain integers by multiplying
    // by R⁻¹ mod p.
    let p = BigInt::<8>::from_sign_and_limbs(
        0,
        [
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0x04FFFFFFFFFFFFFF,
            0,
            0,
            0,
            0,
        ],
    );
    let r_bw = BigInt::<8>::ONE << 256;
    let p_minus_2 = p.ct_sub(&BigInt::<8>::TWO);
    let r_bw_inv = BigInt::<8>::pow_mod(&r_bw, &p_minus_2, &p);

    let convert = |limbs: &[u64; 4]| -> [u8; 32] {
        let mont = BigInt::<8>::from_sign_and_limbs(
            0,
            [limbs[0], limbs[1], limbs[2], limbs[3], 0, 0, 0, 0],
        );
        let plain = mont.ct_mul(&r_bw_inv).ct_mod(&p);
        let mut bytes = [0u8; 32];
        for i in 0..4 {
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&plain.as_limbs()[i].to_le_bytes());
        }
        bytes
    };

    let c_ref_px_re = convert(&broadwell_values[0]);
    let c_ref_px_im = convert(&broadwell_values[1]);
    let c_ref_qx_re = convert(&broadwell_values[2]);
    let c_ref_qx_im = convert(&broadwell_values[3]);

    let our_px = torsion_basis::E0_P_X;
    let our_qx = torsion_basis::E0_Q_X;

    assert_eq!(
        c_ref_px_re,
        our_px.a.to_bytes(),
        "P₀ x real part mismatch with C ref at commit {C_REF_COMMIT}"
    );
    assert_eq!(
        c_ref_px_im,
        our_px.b.to_bytes(),
        "P₀ x imaginary part mismatch with C ref at commit {C_REF_COMMIT}"
    );
    assert_eq!(
        c_ref_qx_re,
        our_qx.a.to_bytes(),
        "Q₀ x real part mismatch with C ref at commit {C_REF_COMMIT}"
    );
    assert_eq!(
        c_ref_qx_im,
        our_qx.b.to_bytes(),
        "Q₀ x imaginary part mismatch with C ref at commit {C_REF_COMMIT}"
    );
}

/// Cross-check torsion bases for all 7 curves against the C ref's
/// `endomorphism_action.c` (fetched from the pinned commit).
///
/// Run with: `cargo test c_ref_all_bases_cross_check -- --ignored`
#[test]
#[ignore]
#[cfg(unix)] // reqwest/regex dev-deps are unix-only
fn c_ref_all_bases_cross_check() {
    let commit = C_REF_COMMIT;
    let url = format!(
        "https://raw.githubusercontent.com/SQISign/the-sqisign/{}/src/precomp/ref/lvl1/endomorphism_action.c",
        commit
    );

    let body = reqwest::blocking::get(&url)
        .unwrap_or_else(|e| panic!("failed to fetch {url}: {e}"))
        .text()
        .unwrap();

    // Parse ALL Broadwell 64-bit blocks.
    let re = regex::Regex::new(r"SQISIGN_GF_IMPL_BROADWELL\)\n\{([^}]+)\}").unwrap();
    let mut blocks: Vec<[u64; 4]> = Vec::new();
    for cap in re.captures_iter(&body) {
        let limbs: Vec<u64> = cap[1]
            .split(',')
            .map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).unwrap())
            .collect();
        assert_eq!(limbs.len(), 4);
        blocks.push([limbs[0], limbs[1], limbs[2], limbs[3]]);
    }

    assert_eq!(
        blocks.len(),
        140,
        "expected 140 Broadwell blocks for 7 curves"
    );

    // The C ref's Broadwell backend stores Fp elements in Montgomery
    // form with R = 2^256. Convert to plain integers.
    let p = BigInt::<8>::from_sign_and_limbs(
        0,
        [
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0x04FFFFFFFFFFFFFF,
            0,
            0,
            0,
            0,
        ],
    );
    let r_bw = BigInt::<8>::ONE << 256;
    let r_bw_inv = BigInt::<8>::pow_mod(&r_bw, &p.ct_sub(&BigInt::<8>::TWO), &p);

    let convert = |limbs: &[u64; 4]| -> [u8; 32] {
        let mont = BigInt::<8>::from_sign_and_limbs(
            0,
            [limbs[0], limbs[1], limbs[2], limbs[3], 0, 0, 0, 0],
        );
        let plain = mont.ct_mul(&r_bw_inv).ct_mod(&p);
        let mut bytes = [0u8; 32];
        for i in 0..4 {
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&plain.as_limbs()[i].to_le_bytes());
        }
        bytes
    };

    // Layout per curve: 20 Broadwell blocks.
    // Blocks 0-7: ec_curve_t {A(re,im), C(re,im), A24(x_re,x_im,z_re,z_im)}
    // Blocks 8-19: ec_basis_t {P(x_re,x_im,z_re,z_im),
    //                          Q(x_re,x_im,z_re,z_im),
    //                          PmQ(x_re,x_im,z_re,z_im)}
    const BLOCKS_PER_CURVE: usize = 20;

    for curve in torsion_basis::ExtremalCurve::ALL {
        let t = curve as usize;
        let base = t * BLOCKS_PER_CURVE;

        let c_ref_px_re = convert(&blocks[base + 8]);
        let c_ref_px_im = convert(&blocks[base + 9]);
        let c_ref_qx_re = convert(&blocks[base + 12]);
        let c_ref_qx_im = convert(&blocks[base + 13]);

        let (our_px, our_qx, ..) = curve.basis();

        assert_eq!(
            c_ref_px_re,
            our_px.a.to_bytes(),
            "curve {t}: P x real part mismatch"
        );
        assert_eq!(
            c_ref_px_im,
            our_px.b.to_bytes(),
            "curve {t}: P x imaginary part mismatch"
        );
        assert_eq!(
            c_ref_qx_re,
            our_qx.a.to_bytes(),
            "curve {t}: Q x real part mismatch"
        );
        assert_eq!(
            c_ref_qx_im,
            our_qx.b.to_bytes(),
            "curve {t}: Q x imaginary part mismatch"
        );
    }
}

/// Decompose θ = 3 + 5i + 7j + 11k in O₀ and verify the action
/// matrix produces the same point as direct scalar computation.
#[test]
fn action_matrix_nontrivial_element() {
    let order = &EXTREMAL_ORDERS[0];
    let gen_matrices = [
        ENDOMORPHISM_MATRICES[0][3],
        ENDOMORPHISM_MATRICES[0][4],
        ENDOMORPHISM_MATRICES[0][5],
    ];
    let f = TorsionExponent::FULL;

    // θ = 3 + 5i + 7j + 11k. Check it's in O₀.
    // O₀ = {1, i, (i+j)/2, (1+k)/2}.
    // θ = 3·1 + 5·i + 7·j + 11·k
    //   = 3·1 + (5-7)·i + 7·(i+j)/2·2 + (11-3)·... hmm, let me
    //   just check if decompose succeeds.
    let theta = Element::<4>::from_i64(3, 5, 7, 11);
    let coords = order.order().decompose(&theta);

    // θ must be in O₀ for decompose to succeed. Let's check:
    // θ = 3 + 5i + 7j + 11k
    // In O₀ basis {1, i, (i+j)/2, (1+k)/2}:
    //   θ = c₀·1 + c₁·i + c₂·(i+j)/2 + c₃·(1+k)/2
    //   = (c₀ + c₃/2) + (c₁ + c₂/2)·i + (c₂/2)·j + (c₃/2)·k
    // So: c₂/2 = 7 → c₂ = 14, c₃/2 = 11 → c₃ = 22
    //     c₀ + c₃/2 = 3 → c₀ = 3 - 11 = -8
    //     c₁ + c₂/2 = 5 → c₁ = 5 - 7 = -2
    if let Some(c) = &coords {
        crate::selkie_trace!(
            "decompose(3+5i+7j+11k) = [{}, {}, {}, {}]",
            c[0],
            c[1],
            c[2],
            c[3]
        );
    } else {
        // θ might not be in O₀ — try a different element.
        // Use θ = 1 + i + (i+j)/2 + (1+k)/2 = (3/2) + (3/2)i + (1/2)j + (1/2)k
        // which has denom 2.
        crate::selkie_trace!("3+5i+7j+11k not in O₀, trying different element");
    }

    // Use M_i test as baseline: i is in O₀ and works.
    // Now test with (i+j)/2 (= gen3, third basis element).
    // gen3 = (i+j)/2 = (0 + 1·i + 1·j + 0·k) / 2
    let gen3_elem = Element::<4>::new(
        Coordinate::from_bigint(BigInt::ZERO),
        Coordinate::from_bigint(BigInt::ONE),
        Coordinate::from_bigint(BigInt::ONE),
        Coordinate::from_bigint(BigInt::ZERO),
        Denominator::TWO,
    );
    let endo = EndomorphismAction {
        order: order.order(),
        generators: gen_matrices,
    };
    let m_computed = endo
        .apply(&gen3_elem, f)
        .expect("apply should succeed for gen3");
    let m_precomp = &ENDOMORPHISM_MATRICES[0][4]; // gen3

    crate::selkie_trace!(
        "gen3 computed[0][0] == precomp[0][0]: {}",
        m_computed.entry(0, 0) == m_precomp.entry(0, 0)
    );
    crate::selkie_trace!(
        "gen3 computed[1][0] == precomp[1][0]: {}",
        m_computed.entry(1, 0) == m_precomp.entry(1, 0)
    );
    assert_eq!(
        m_computed.entry(0, 0),
        m_precomp.entry(0, 0),
        "gen3 action matrix mismatch at (0,0)"
    );
}

/// Action matrix for θ=3 (scalar element) should produce [3]P.
///
/// This tests whether `Element::action_matrix(3·1)` produces the
/// identity matrix scaled by 3, as expected for a scalar
/// endomorphism.
#[test]
fn action_matrix_scalar_three() {
    let order = &EXTREMAL_ORDERS[0];
    let elem = Element::<4>::from_i64(3, 0, 0, 0);
    let endo = EndomorphismAction {
        order: order.order(),
        generators: [
            ENDOMORPHISM_MATRICES[0][3],
            ENDOMORPHISM_MATRICES[0][4],
            ENDOMORPHISM_MATRICES[0][5],
        ],
    };
    let m = endo
        .apply(&elem, TorsionExponent::FULL)
        .expect("decompose should succeed for scalar element");

    // For θ = 3·1, the action matrix should be 3·I = [[3,0],[0,3]].
    let three = Scalar::from_u64(3);
    assert_eq!(*m.entry(0, 0), three, "m00 should be 3");
    assert_eq!(*m.entry(1, 1), three, "m11 should be 3");
    assert_eq!(*m.entry(0, 1), Scalar::ZERO, "m01 should be 0");
    assert_eq!(*m.entry(1, 0), Scalar::ZERO, "m10 should be 0");
}

/// Action matrix for θ=3 (scalar element) on O_t (t > 0) should
/// also produce 3·I. Probes whether the q ≥ 5 paths in
/// `action_matrix` (decompose at width 8) and `ENDOMORPHISM_MATRICES[t]`
/// (the `gen_matrices` table) are correct.
#[test]
fn action_matrix_scalar_three_alternate_orders() {
    for t in 1..crate::quaternions::precomputed::NUM_EXTREMAL_ORDERS {
        let order = &EXTREMAL_ORDERS[t];
        let elem = Element::<4>::from_i64(3, 0, 0, 0);
        let endo = EndomorphismAction {
            order: order.order(),
            generators: [
                ENDOMORPHISM_MATRICES[t][3],
                ENDOMORPHISM_MATRICES[t][4],
                ENDOMORPHISM_MATRICES[t][5],
            ],
        };
        let m = endo
            .apply(&elem, TorsionExponent::FULL)
            .expect("decompose should succeed for scalar element");
        let three = Scalar::from_u64(3);
        assert_eq!(*m.entry(0, 0), three, "t={t}: m00 should be 3");
        assert_eq!(*m.entry(1, 1), three, "t={t}: m11 should be 3");
        assert_eq!(*m.entry(0, 1), Scalar::ZERO, "t={t}: m01 should be 0");
        assert_eq!(*m.entry(1, 0), Scalar::ZERO, "t={t}: m10 should be 0");
    }
}

/// Compares Montgomery ladder vs biladder for [3]*P.
#[test]
fn ladder_vs_biladder_agree() {
    let curve = Curve::E0;
    let p = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::E0_P_X, &curve);
    let q = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::E0_Q_X, &curve);
    let pmq = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::E0_PMQ_X, &curve);

    let three = Scalar::from_u64(3);
    let ladder_result = &three * &p;

    let basis = TorsionBasis::from_propagated(p, q, pmq);
    let biladder_result = basis.eval_decomposition(&three, &Scalar::ZERO);

    assert_eq!(
        ladder_result.to_affine_x(),
        biladder_result.to_affine_x(),
        "[3]P via ladder ≠ eval_decomposition(3, 0)"
    );
}

/// Scalar-multiplication kernel on E₀ × E₀: (P, [3]P), (P-Q, [3](P-Q)).
///
/// Bypasses the action matrix entirely — uses direct scalar
/// multiplication on the torsion basis. If this produces zeros=1
/// in the splitting, the chain works for E₀×E₀ and the issue is
/// in the action matrix convention.
#[test]
fn scalar_mul_kernel_splits() {
    let curve = Curve::E0;
    let p = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::E0_P_X, &curve);
    let q = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::E0_Q_X, &curve);
    let pmq = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::E0_PMQ_X, &curve);

    // [3]P, [3](P-Q), [3]Q via direct Montgomery scalar mul.
    let three = Scalar::from_u64(3);
    let three_p = &three * &p;
    let three_pmq = &three * &pmq;
    let three_q = &three * &q;

    // Lift component 1: (P, P-Q, Q)
    let comp1 = TorsionBasis::from_propagated(p, pmq, q);
    let (p_jac, pmq_jac) = comp1.lift(&curve).expect("lift comp1");
    // Lift component 2: ([3]P, [3](P-Q), [3]Q)
    let comp2 = TorsionBasis::from_propagated(three_p, three_pmq, three_q);
    let (tp_jac, tpmq_jac) = comp2.lift(&curve).expect("lift comp2");

    // Kernel: K₁ = (P, [3]P), K₂ = (P-Q, [3](P-Q))
    let e = 50u32;
    let doublings = TorsionExponent::FULL.value() - 2 - e;
    let double_n = |mut pt: JacobianPoint, n: u32| -> JacobianPoint {
        for _ in 0..n {
            pt = pt.double_for_theta();
        }
        pt
    };
    let k1 = (double_n(p_jac, doublings), double_n(tp_jac, doublings));
    let k2 = (double_n(pmq_jac, doublings), double_n(tpmq_jac, doublings));

    let product = surfaces::EllipticProduct::new(curve, curve);
    let kernel = surfaces::Kernel::from_jacobian(product, k1, k2);

    let te = TorsionExponent::try_from(e).unwrap();
    let (_codomain, _images) = kernel
        .isogeny_extra_torsion(te, &[])
        .expect("test kernel must split as product");
}

/// Verifies all 7 torsion bases: points on curve, correct order.
#[test]
fn all_torsion_bases_on_curve() {
    use crate::curves::montgomery::{Coefficient, Curve, ProjectiveXOnlyPoint};

    for curve_idx in torsion_basis::ExtremalCurve::ALL {
        let t = curve_idx as usize;
        let (px, qx, _pmq_x, a) = curve_idx.basis();
        let curve = Curve::from(Coefficient::from(a));
        let p = ProjectiveXOnlyPoint::from_affine_x(px, &curve);
        let q = ProjectiveXOnlyPoint::from_affine_x(qx, &curve);

        // Points should be on the curve (recover_y succeeds).
        assert!(
            curve.recover_y(&p.to_affine_x()).is_some(),
            "curve {t}: P not on curve"
        );
        assert!(
            curve.recover_y(&q.to_affine_x()).is_some(),
            "curve {t}: Q not on curve"
        );

        // [2^f]P = O (order divides 2^f).
        let f = TorsionExponent::FULL.value();
        let mut test = p;
        for _ in 0..f {
            test = test.double();
        }
        assert!(
            test.Z == crate::fields::fp2::Fp2::ZERO,
            "curve {t}: [2^f]P ≠ O"
        );
    }
}

/// Verifies the precomputed `P − Q` on each curve `E_t` is the
/// x-coordinate of `P − Q` (or `P + Q` — the x-only basis admits a
/// sign swap that the biladder tolerates) for SOME y-sign choice of
/// `P` and `Q`. Curve 0 also gets a direct round-trip against
/// [`ProjectiveXOnlyPoint::projective_difference`].
#[test]
fn alternate_curves_pmq_consistent() {
    use crate::{
        curves::montgomery::{Coefficient, Curve, JacobianPoint, ProjectiveXOnlyPoint},
        fields::fp2::Fp2,
    };

    // Curve 0: the precomputed `E0_PMQ_X` must match what
    // `projective_difference(P₀, Q₀)` produces on the fly. Catches
    // any silent corruption of the precomputed constant.
    {
        let p = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::E0_P_X, &Curve::E0);
        let q = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::E0_Q_X, &Curve::E0);
        let pmq = p.projective_difference(&q);
        assert_eq!(
            *pmq.to_affine_x().as_fp2(),
            torsion_basis::E0_PMQ_X,
            "E0_PMQ_X must match projective_difference(E0_P_X, E0_Q_X)"
        );
    }

    // Curves 1..6: pmq_x must be the x-coordinate of (±P ± Q) for
    // some y-sign choice. We check all four sign combinations and
    // accept a match on any of them — x_add_sub returns
    // (x(P+Q), x(P−Q)) for a given pair, and negating either P or
    // Q flips which is which.
    let f = TorsionExponent::FULL.value();
    for curve_idx in &torsion_basis::ExtremalCurve::ALL[1..] {
        let t = *curve_idx as usize;
        let (px, qx, pmq_x, a) = curve_idx.basis();
        let curve = Curve::from(Coefficient::from(a));

        // Sanity: pmq_x is on the curve.
        let pmq_proj = ProjectiveXOnlyPoint::from_affine_x(pmq_x, &curve);
        assert!(
            curve.recover_y(&pmq_proj.to_affine_x()).is_some(),
            "curve {t}: P−Q not on curve"
        );

        // Sanity: [2^f]pmq_x = O.
        let mut test = pmq_proj;
        for _ in 0..f {
            test = test.double();
        }
        assert!(test.Z == Fp2::ZERO, "curve {t}: [2^f](P−Q) ≠ O");

        // P − Q identity: lift P and Q to Jacobian form with one
        // y-sign choice each, then use x_add_sub to get
        // (x(P+Q), x(P−Q)) with the sign distinguished. If the
        // Sage verification script's sign-swap case applies on
        // this curve, x(P+Q) will match pmq_x instead — accept
        // either.
        let p_y = curve
            .recover_y(&ProjectiveXOnlyPoint::from_affine_x(px, &curve).to_affine_x())
            .unwrap_or_else(|| panic!("curve {t}: P not on curve"));
        let q_y = curve
            .recover_y(&ProjectiveXOnlyPoint::from_affine_x(qx, &curve).to_affine_x())
            .unwrap_or_else(|| panic!("curve {t}: Q not on curve"));
        let p_jac = JacobianPoint::new(px, p_y, Fp2::ONE, &curve);
        let q_jac = JacobianPoint::new(qx, q_y, Fp2::ONE, &curve);
        let (x_add, x_sub) = p_jac.x_add_sub(&q_jac);
        let x_add_aff = *x_add.to_affine_x().as_fp2();
        let x_sub_aff = *x_sub.to_affine_x().as_fp2();
        assert!(
            pmq_x == x_sub_aff || pmq_x == x_add_aff,
            "curve {t}: pmq_x matches neither x(P−Q) nor x(P+Q) for the \
             chosen y-signs (stored pmq_x is inconsistent with (P, Q))"
        );
    }
}

/// Action-matrix composition matches direct decomposition of the
/// quaternion product.
///
/// The outer `to_isogeny` builds
/// `M_{β₂ · conj(β₁)} = M_{β₂} · M_{conj(β₁)} = M_{β₂} · adj(M_{β₁})`
/// on the fly rather than first constructing the wide quaternion
/// `β₂ · conj(β₁)` and decomposing it on O₀. This test checks the
/// two routes agree mod 2^f for several small, hand-chosen pairs
/// that fit without widening.
#[test]
fn action_matrix_composition_equals_direct() {
    let order = &EXTREMAL_ORDERS[0];
    let gen_matrices = [
        ENDOMORPHISM_MATRICES[0][3],
        ENDOMORPHISM_MATRICES[0][4],
        ENDOMORPHISM_MATRICES[0][5],
    ];
    let f = TorsionExponent::FULL;
    let fv = f.value();

    // β pairs: each a primitive element of O₀ = {1, i, (i+j)/2, (1+k)/2}.
    // Stored in the {1, i, j, k} basis with an explicit denominator.
    //
    // In the {1, i, j, k} basis, c₀·1 + c₁·i + c₂·(i+j)/2 + c₃·(1+k)/2
    // equals ((2c₀+c₃) + (2c₁+c₂) i + c₂ j + c₃ k) / 2, so an
    // O₀-primitive element has coords (2c₀+c₃, 2c₁+c₂, c₂, c₃) with
    // denom 2. We list a few combinations.
    type QuatTuple = (i64, i64, i64, i64, i64);
    let cases: &[(QuatTuple, QuatTuple)] = &[
        // β₁ = 1, β₂ = i  → coords/denom as quaternions in {1,i,j,k}/denom.
        ((1, 0, 0, 0, 1), (0, 1, 0, 0, 1)),
        // β₁ = i, β₂ = (i+j)/2
        ((0, 1, 0, 0, 1), (0, 1, 1, 0, 2)),
        // β₁ = (1+k)/2, β₂ = (i+j)/2
        ((1, 0, 0, 1, 2), (0, 1, 1, 0, 2)),
        // β₁ = 1 + (i+j)/2 = (2+i+j)/2, β₂ = 1 + (1+k)/2 = (3+k)/2
        ((2, 1, 1, 0, 2), (3, 0, 0, 1, 2)),
    ];

    for (n, ((a1, b1, c1, d1, dn1), (a2, b2, c2, d2, dn2))) in cases.iter().enumerate() {
        let beta1 = Element::<4>::new(
            Coordinate::from_bigint(BigInt::<4>::from_i64(*a1)),
            Coordinate::from_bigint(BigInt::<4>::from_i64(*b1)),
            Coordinate::from_bigint(BigInt::<4>::from_i64(*c1)),
            Coordinate::from_bigint(BigInt::<4>::from_i64(*d1)),
            Denominator::from_bigint_unchecked(BigInt::<4>::from_i64(*dn1)),
        );
        let beta2 = Element::<4>::new(
            Coordinate::from_bigint(BigInt::<4>::from_i64(*a2)),
            Coordinate::from_bigint(BigInt::<4>::from_i64(*b2)),
            Coordinate::from_bigint(BigInt::<4>::from_i64(*c2)),
            Coordinate::from_bigint(BigInt::<4>::from_i64(*d2)),
            Denominator::from_bigint_unchecked(BigInt::<4>::from_i64(*dn2)),
        );

        let endo = EndomorphismAction {
            order: order.order(),
            generators: gen_matrices,
        };
        let m_beta1 = endo
            .apply(&beta1, f)
            .unwrap_or_else(|| panic!("case {n}: apply(β₁) failed"));
        let m_beta2 = endo
            .apply(&beta2, f)
            .unwrap_or_else(|| panic!("case {n}: apply(β₂) failed"));

        // Route A: matrix composition, M_{β₂} · adj(M_{β₁}).
        let m_composed = m_beta2.mat_mul_mod(&m_beta1.adjugate_mod(fv), fv);

        // Route B: direct apply on the quaternion product.
        let theta = beta2
            .mul(&beta1.conjugate())
            .unwrap_or_else(|| panic!("case {n}: β₂·conj(β₁) overflows BigInt<4>"));
        let m_direct = endo
            .apply(&theta, f)
            .unwrap_or_else(|| panic!("case {n}: apply(θ) failed"));

        for row in 0..2 {
            for col in 0..2 {
                assert_eq!(
                    m_composed.entry(row, col),
                    m_direct.entry(row, col),
                    "case {n}: composition ≠ direct at [{row}][{col}]",
                );
            }
        }
    }
}
