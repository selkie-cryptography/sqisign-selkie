// TODO: Consider moving the network-dependent test (c_ref_basis_cross_check)
// to an integration test in tests/.

use super::*;
use precomputed::torsion_basis;
use precomputed::ACTION_MATRICES;

#[test]
fn action_matrix_via_trait() {
    let elem = Element::from_i64(1, 0, 0, 0);
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
    let a = Element::from_i64(1, 0, 0, 0);
    let b = Element::from_i64(0, 1, 0, 0);
    let ab = Element::from_i64(1, 1, 0, 0);

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

    let modulus = BigInt::<4>::ONE.shl(f.value());
    for row in 0..2 {
        for col in 0..2 {
            let sum = m_a
                .entry(row, col)
                .ct_add(m_b.entry(row, col))
                .ct_mod(&modulus);
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
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x04ffffffffffffff,
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
    use crate::curves::montgomery::{Curve, ProjectiveXOnlyPoint};
    use crate::curves::scalar::Scalar;
    use crate::curves::TorsionBasis;

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
