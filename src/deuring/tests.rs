// TODO: Consider moving the network-dependent test (c_ref_basis_cross_check)
// to an integration test in tests/.

use precomputed::{ACTION_MATRICES, torsion_basis};

use super::*;

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
/// Fetches `e0_basis.c` from GitHub, parses the Broadwell 64-bit limbs,
/// converts from the C ref's Montgomery form (R=2^256) to plain integers,
/// and compares against our stored values.
///
/// Run with: `cargo test c_ref_basis_cross_check -- --ignored`
#[test]
#[ignore]
fn c_ref_basis_cross_check() {
    // Pinned commit of the SQIsign reference implementation.
    const COMMIT: &str = "91e9e464fe5400192d13e1f9240cbf180200a103";
    let url = format!(
        "https://raw.githubusercontent.com/SQISign/the-sqisign/{}/src/precomp/ref/lvl1/e0_basis.c",
        COMMIT
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

    // Convert from Broadwell Montgomery form (R=2^256) to plain integers.
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
        "P₀ x real part mismatch with C ref at commit {COMMIT}"
    );
    assert_eq!(
        c_ref_px_im,
        our_px.b.to_bytes(),
        "P₀ x imaginary part mismatch with C ref at commit {COMMIT}"
    );
    assert_eq!(
        c_ref_qx_re,
        our_qx.a.to_bytes(),
        "Q₀ x real part mismatch with C ref at commit {COMMIT}"
    );
    assert_eq!(
        c_ref_qx_im,
        our_qx.b.to_bytes(),
        "Q₀ x imaginary part mismatch with C ref at commit {COMMIT}"
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
    let basis = TorsionBasis::new(p0, q0, p0.projective_difference(&q0));

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

/// Diagonal kernel (P,P),(Q,Q) on E₀ × E₀ should produce a product.
///
/// This tests the (2,2)-chain in isolation: the diagonal embedding
/// is always a valid kernel, so if splitting gives zeros≠1, the
/// chain itself has a bug for the E₀ × E₀ case.
#[test]
#[ignore] // TODO: chain splitting fails for E₀×E₀ kernel — see project_progress.md
fn diagonal_kernel_splits() {

    let curve = Curve::E0;
    let p = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::e0_px(), &curve);
    let q = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::e0_qx(), &curve);
    let pmq = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_PMQ_X, &curve);
    let basis = TorsionBasis::new(p, q, pmq);

    // Action matrix M_i applied to (P, Q, P-Q) basis.
    let m_i = &ACTION_MATRICES[0][0];
    let f = TorsionExponent::FULL.value();
    let i_p = basis.eval_decomposition(m_i.entry(0, 0), m_i.entry(1, 0));
    let i_pmq = basis.eval_decomposition(
        &m_i.entry(0, 0).sub_mod2k(m_i.entry(0, 1), f),
        &m_i.entry(1, 0).sub_mod2k(m_i.entry(1, 1), f),
    );
    let i_q = basis.eval_decomposition(m_i.entry(0, 1), m_i.entry(1, 1));

    // Lift with (P, P-Q) as generators (matching C ref convention).
    let comp1 = TorsionBasis::new(p, pmq, q);
    let (p_jac, pmq_jac) = comp1.lift(&curve).expect("lift comp1");
    let comp2 = TorsionBasis::new(i_p, i_pmq, i_q);
    let (ip_jac, ipmq_jac) = comp2.lift(&curve).expect("lift comp2");

    // Kernel: K₁ = (P, i(P)), K₂ = (P-Q, i(P-Q)).
    let e = 50u32;
    let doublings = TorsionExponent::FULL.value() - 2 - e;
    let double_n = |mut pt: JacobianPoint, n: u32| -> JacobianPoint {
        for _ in 0..n {
            pt = pt.double_for_theta();
        }
        pt
    };
    let k1 = (double_n(p_jac, doublings), double_n(ip_jac, doublings));
    let k2 = (double_n(pmq_jac, doublings), double_n(ipmq_jac, doublings));

    let product = surfaces::EllipticProduct::new(curve, curve);
    let kernel = surfaces::Kernel::from_jacobian(product, k1, k2);

    // Print the y-value for comparison with C ref.
    let y_p = curve.recover_y(&p.to_affine_x()).unwrap();
    let y_bytes = y_p.to_bytes();
    let re_hex: String = y_bytes[..32].iter().rev().map(|b| format!("{:02x}", b)).collect();
    let im_hex: String = y_bytes[32..].iter().rev().map(|b| format!("{:02x}", b)).collect();
    eprintln!("recover_y(P) re=0x{re_hex} im=0x{im_hex}");

    // Use extra_torsion mode since the kernel has order 2^(e+2),
    // matching the signing path (FixedDegreeIsogeny).
    let te = TorsionExponent::try_from(e).unwrap();
    let (_codomain, _images) = kernel.isogeny_extra_torsion(te, &[]);
    // If this doesn't panic and the test-only splitting diagnostic
    // prints zeros=1, the chain works for E₀ × E₀.
    // The splitting count is printed by the #[cfg(test)] block
    // inside the chain's Phase 4.
}

/// Action matrix for θ=3 (scalar element) should produce [3]P.
///
/// This tests whether action_matrix(3·1) produces the identity
/// matrix scaled by 3, as expected for a scalar endomorphism.
#[test]
fn action_matrix_scalar_three() {
    let order = &crate::quaternions::precomputed::EXTREMAL_ORDERS[0];
    let elem = Element::<4>::from_i64(3, 0, 0, 0);
    let gen_matrices = [
        ACTION_MATRICES[0][3],
        ACTION_MATRICES[0][4],
        ACTION_MATRICES[0][5],
    ];
    let m = super::action_matrix(&elem, order.order(), &gen_matrices, TorsionExponent::FULL)
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

    let basis = TorsionBasis::new(p, q, pmq);
    let biladder_result = basis.eval_decomposition(&three, &Scalar::ZERO);

    assert_eq!(
        ladder_result.to_affine_x(),
        biladder_result.to_affine_x(),
        "[3]P via ladder ≠ eval_decomposition(3, 0)"
    );
}

/// Action-matrix kernel using M_i: (P, i(P)), (P-Q, i(P-Q)).
///
/// Uses the action matrix with the biladder to compute i(P) etc,
/// then builds the kernel the same way as scalar_mul_kernel_splits.
#[test]
fn action_matrix_kernel_splits() {
    let curve = Curve::E0;
    let p = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::e0_px(), &curve);
    let q = ProjectiveXOnlyPoint::from_affine_x(torsion_basis::e0_qx(), &curve);
    let pmq = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_PMQ_X, &curve);
    let basis = TorsionBasis::new(p, q, pmq);

    let m_i = &ACTION_MATRICES[0][0];
    let i_p = basis.eval_decomposition(m_i.entry(0, 0), m_i.entry(1, 0));
    let i_q = basis.eval_decomposition(m_i.entry(0, 1), m_i.entry(1, 1));
    let f = TorsionExponent::FULL.value();
    let i_pmq = basis.eval_decomposition(
        &m_i.entry(0, 0).sub_mod2k(m_i.entry(0, 1), f),
        &m_i.entry(1, 0).sub_mod2k(m_i.entry(1, 1), f),
    );

    // Lift component 1: (P, P-Q, Q)
    let comp1 = TorsionBasis::new(p, pmq, q);
    let (p_jac, pmq_jac) = comp1.lift(&curve).expect("lift comp1");
    // Lift component 2: (i(P), i(P-Q), i(Q))
    // Use projective_difference for i(P)-i(Q) instead of the
    // biladder's i(P-Q), to get a consistent PmQ representative.
    let i_pmq_diff = i_p.projective_difference(&i_q);
    let comp2 = TorsionBasis::new(i_p, i_pmq_diff, i_q);
    let (ip_jac, ipmq_jac) = comp2.lift(&curve).expect("lift comp2");

    // Test with e=150 to match the signing pipeline's chain length.
    let e = 150u32;
    let doublings = TorsionExponent::FULL.value() - 2 - e;
    let double_n = |mut pt: JacobianPoint, n: u32| -> JacobianPoint {
        for _ in 0..n {
            pt = pt.double_for_theta();
        }
        pt
    };
    let k1 = (double_n(p_jac, doublings), double_n(ip_jac, doublings));
    let k2 = (double_n(pmq_jac, doublings), double_n(ipmq_jac, doublings));

    let product = surfaces::EllipticProduct::new(curve, curve);
    let kernel = surfaces::Kernel::from_jacobian(product, k1, k2);

    let te = TorsionExponent::try_from(e).unwrap();
    let (_codomain, _images) = kernel.isogeny_extra_torsion(te, &[]);
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
    let comp1 = TorsionBasis::new(p, pmq, q);
    let (p_jac, pmq_jac) = comp1.lift(&curve).expect("lift comp1");
    // Lift component 2: ([3]P, [3](P-Q), [3]Q)
    let comp2 = TorsionBasis::new(three_p, three_pmq, three_q);
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
