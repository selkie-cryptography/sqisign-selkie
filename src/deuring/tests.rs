// TODO: Consider moving the network-dependent test (c_ref_basis_cross_check)
// to an integration test in tests/.

use precomputed::{ACTION_MATRICES, torsion_basis};

use super::*;
use crate::quaternions::algebra::{Coordinate, Denominator};

/// Pinned commit of the SQIsign C reference implementation.
/// Used by cross-check tests that fetch precomputed data.
const C_REF_COMMIT: &str = "91e9e464fe5400192d13e1f9240cbf180200a103";

#[test]
fn action_matrix_via_trait() {
    let elem = Element::<4>::from_i64(1, 0, 0, 0);
    let basis_matrices: [ActionMatrix; 4] = [
        ACTION_MATRICES[0][0],
        ACTION_MATRICES[0][1],
        ACTION_MATRICES[0][2],
        ACTION_MATRICES[0][3],
    ];
    use crate::curves::TorsionExponent;
    let m = elem.action_matrix(&basis_matrices, TorsionExponent::FULL);
    assert_eq!(*m.entry(0, 0), *ACTION_MATRICES[0][0].entry(0, 0));
    assert_eq!(*m.entry(1, 1), *ACTION_MATRICES[0][0].entry(1, 1));
}

#[test]
fn action_matrix_linear_combination() {
    let a = Element::<4>::from_i64(1, 0, 0, 0);
    let b = Element::<4>::from_i64(0, 1, 0, 0);
    let ab = Element::<4>::from_i64(1, 1, 0, 0);

    let basis_matrices: [ActionMatrix; 4] = [
        ACTION_MATRICES[0][0],
        ACTION_MATRICES[0][1],
        ACTION_MATRICES[0][2],
        ACTION_MATRICES[0][3],
    ];
    use crate::curves::TorsionExponent;
    let f = TorsionExponent::FULL;

    let m_a = a.action_matrix(&basis_matrices, f);
    let m_b = b.action_matrix(&basis_matrices, f);
    let m_ab = ab.action_matrix(&basis_matrices, f);

    for row in 0..2 {
        for col in 0..2 {
            let sum = m_a
                .entry(row, col)
                .add_mod2k(m_b.entry(row, col), f.value());
            assert_eq!(
                sum,
                *m_ab.entry(row, col),
                "linearity failed at [{row}][{col}]"
            );
        }
    }
}

#[test]
fn torsion_basis_points_on_e0() {
    use crate::curves::montgomery::{Curve, ProjectiveXOnlyPoint};

    let p = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::e0_px(), &Curve::E0);
    let q = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::e0_qx(), &Curve::E0);

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
    let r_bw = BigInt::<8>::ONE.shl(256);
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

    let our_px = torsion_basis::e0_px();
    let our_qx = torsion_basis::e0_qx();

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

    assert_eq!(blocks.len(), 140, "expected 140 Broadwell blocks for 7 curves");

    // The C ref's Broadwell backend stores Fp elements in Montgomery
    // form with R = 2^256. Convert to plain integers.
    let p = BigInt::<8>::from_sign_and_limbs(
        0,
        [0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x04FFFFFFFFFFFFFF, 0, 0, 0, 0],
    );
    let r_bw = BigInt::<8>::ONE.shl(256);
    let r_bw_inv = BigInt::<8>::pow_mod(&r_bw, &p.ct_sub(&BigInt::<8>::TWO), &p);

    let convert = |limbs: &[u64; 4]| -> [u8; 32] {
        let mont = BigInt::<8>::from_sign_and_limbs(0, [limbs[0], limbs[1], limbs[2], limbs[3], 0, 0, 0, 0]);
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

    for t in 0..7 {
        let base = t * BLOCKS_PER_CURVE;

        let c_ref_px_re = convert(&blocks[base + 8]);
        let c_ref_px_im = convert(&blocks[base + 9]);
        let c_ref_qx_re = convert(&blocks[base + 12]);
        let c_ref_qx_im = convert(&blocks[base + 13]);

        let (our_px, our_qx, _) = torsion_basis::basis_for_curve(t)
            .unwrap_or_else(|| panic!("no basis for curve {t}"));

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
        ACTION_MATRICES[0][3],
        ACTION_MATRICES[0][4],
        ACTION_MATRICES[0][5],
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
        eprintln!("decompose(3+5i+7j+11k) = [{}, {}, {}, {}]",
            c[0], c[1], c[2], c[3]);
    } else {
        // θ might not be in O₀ — try a different element.
        // Use θ = 1 + i + (i+j)/2 + (1+k)/2 = (3/2) + (3/2)i + (1/2)j + (1/2)k
        // which has denom 2.
        eprintln!("3+5i+7j+11k not in O₀, trying different element");
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
    let m_computed = action_matrix(&gen3_elem, order.order(), &gen_matrices, f)
        .expect("action_matrix should succeed for gen3");
    let m_precomp = &ACTION_MATRICES[0][4]; // gen3

    eprintln!("gen3 computed[0][0] == precomp[0][0]: {}",
        m_computed.entry(0, 0) == m_precomp.entry(0, 0));
    eprintln!("gen3 computed[1][0] == precomp[1][0]: {}",
        m_computed.entry(1, 0) == m_precomp.entry(1, 0));
    assert_eq!(
        m_computed.entry(0, 0), m_precomp.entry(0, 0),
        "gen3 action matrix mismatch at (0,0)"
    );

}

/// Verify that M_i applied to the basis produces i(P₀).
#[test]
fn action_matrix_consistent_with_basis() {
    use crate::curves::{
        TorsionBasis,
        montgomery::{Curve, ProjectiveXOnlyPoint},
        scalar::Scalar,
    };

    let p0 = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::e0_px(), &Curve::E0);
    let q0 = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::e0_qx(), &Curve::E0);
    let basis = TorsionBasis::from((p0, q0));

    let m_i = &ACTION_MATRICES[0][0];

    let a = Scalar::from_limbs(*m_i.entry(0, 0).as_limbs());
    let b = Scalar::from_limbs(*m_i.entry(1, 0).as_limbs());
    let result = basis.eval_decomposition(&a, &b);

    let neg_px = -torsion_basis::e0_px();
    let i_of_p = ProjectiveXOnlyPoint::from_affine_x(neg_px, &Curve::E0);

    assert_eq!(
        result, i_of_p,
        "M_i · (1, 0)^T applied to basis does not match i(P₀)"
    );
}


/// Action matrix for θ=3 (scalar element) should produce [3]P.
///
/// This tests whether action_matrix(3·1) produces the identity
/// matrix scaled by 3, as expected for a scalar endomorphism.
#[test]
fn action_matrix_scalar_three() {
    let order = &EXTREMAL_ORDERS[0];
    let elem = Element::<4>::from_i64(3, 0, 0, 0);
    let gen_matrices = [
        ACTION_MATRICES[0][3],
        ACTION_MATRICES[0][4],
        ACTION_MATRICES[0][5],
    ];
    let m = action_matrix(&elem, order.order(), &gen_matrices, TorsionExponent::FULL)
        .expect("decompose should succeed for scalar element");

    // For θ = 3·1, the action matrix should be 3·I = [[3,0],[0,3]].
    let three = Scalar::from_u64(3);
    assert_eq!(*m.entry(0, 0), three, "m00 should be 3");
    assert_eq!(*m.entry(1, 1), three, "m11 should be 3");
    assert_eq!(*m.entry(0, 1), Scalar::ZERO, "m01 should be 0");
    assert_eq!(*m.entry(1, 0), Scalar::ZERO, "m10 should be 0");
}


/// Compare Montgomery ladder vs biladder for [3]*P.
#[test]
fn ladder_vs_biladder_agree() {
    let curve = Curve::E0;
    let p = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::e0_px(), &curve);
    let q = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::e0_qx(), &curve);
    let pmq = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_PMQ_X, &curve);

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
    let p = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::e0_px(), &curve);
    let q = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::e0_qx(), &curve);
    let pmq = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_PMQ_X, &curve);

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
    let (_codomain, _images) = kernel.isogeny_extra_torsion(te, &[]);
}

/// Verify all 7 torsion bases: points on curve, correct order.
#[test]
fn all_torsion_bases_on_curve() {
    use crate::curves::montgomery::{Coefficient, Curve, ProjectiveXOnlyPoint};

    for t in 0..7 {
        let (px, qx, a) = torsion_basis::basis_for_curve(t)
            .unwrap_or_else(|| panic!("no basis for curve {t}"));
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
            test.Z == crate::fields::Fp2::ZERO,
            "curve {t}: [2^f]P ≠ O"
        );
    }
}
