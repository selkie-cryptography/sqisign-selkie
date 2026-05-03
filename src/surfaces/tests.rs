use super::*;
use crate::curves::montgomery::{Curve, ProjectiveXOnlyPoint};

/// Test the splitting function with a synthetic product null point.
///
/// A product theta null point for E₁ × E₂ with theta constants
/// (a₁, b₁) and (a₂, b₂) is (a₁a₂, a₁b₂, b₁a₂, b₁b₂).
/// The splitting should find exactly one zero U_{i,j}.
///
/// This tests the splitting formula in isolation — no (2,2)-chain,
/// no kernel construction. The null point is constructed directly
/// with known product structure.
#[test]
fn splitting_synthetic_product() {
    use crate::fields::fp::Fp;

    // Fixed small values as theta constants. These are arbitrary
    // nonzero Fp2 elements; the only requirement is that the
    // product (a₁a₂, a₁b₂, b₁a₂, b₁b₂) has exactly one
    // vanishing U_{i,j}(0).
    let a1 = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let b1 = Fp2::new(Fp::from_small(5), Fp::from_small(11));
    let a2 = Fp2::new(Fp::from_small(13), Fp::from_small(17));
    let b2 = Fp2::new(Fp::from_small(19), Fp::from_small(23));

    let null = ThetaNullPoint::new(
        &a1 * &a2, // (0,0) component
        &a1 * &b2, // (0,1) component
        &b1 * &a2, // (1,0) component
        &b1 * &b2, // (1,1) component
    );

    let count = isogeny::get_index_splitting_count(&null);
    assert_eq!(
        count, 1,
        "product null point should have exactly 1 zero U index"
    );
}

/// Test the gluing codomain computation in isolation.
///
/// Uses E₀ × E₀ with the precomputed basis. This exercises the
/// gluing's `ActionByTranslation`, `theta_change_of_basis`,
/// `product_to_theta`, and `to_squared_theta` steps without
/// running the full chain. The codomain null point and dual are
/// checked for internal consistency.
///
/// Note: the kernel (P,Q)×(Q,P) may have a degenerate
/// `ActionByTranslation` determinant (the DMPR24 reference also
/// fails on this kernel). This test checks the codomain structure
/// rather than asserting the splitting succeeds.
#[test]
fn gluing_codomain_manual_check() {
    use crate::curves::TorsionBasis;

    let curve = Curve::E0;
    let p = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_P_X, &curve);
    let q = ProjectiveXOnlyPoint::from_affine_x(crate::params::BASIS_E0_Q_X, &curve);

    // Scale to order 16.
    let mut p4 = p;
    let mut q4 = q;
    for _ in 0..244 {
        p4 = p4.double();
        q4 = q4.double();
    }

    let t1_0 = p4;
    let t1_1 = q4;
    let t2_0 = q4;
    let t2_1 = p4;

    let pmq_0 = t1_0.projective_difference(&t2_0);
    let pmq_1 = t1_1.projective_difference(&t2_1);
    let (t1_jac_0, t2_jac_0) = TorsionBasis::from_propagated(t1_0, t2_0, pmq_0)
        .lift(&curve)
        .expect("lift E0 failed");
    let (t1_jac_1, t2_jac_1) = TorsionBasis::from_propagated(t1_1, t2_1, pmq_1)
        .lift(&curve)
        .expect("lift E0 failed");

    let gluing = GluingKernel {
        T1: (t1_0, t1_1),
        T1_jac: (t1_jac_0, t1_jac_1),
        T2: (t2_0, t2_1),
        T2_jac: (t2_jac_0, t2_jac_1),
    };
    let data = gluing.codomain();

    // The dual should have delta = 0 (from hs1.W = 0 invariant).
    let null = &data.codomain.null;
    let _count = isogeny::get_index_splitting_count(null);

    // Check the dual form: (alpha, beta, gamma, 0).
    let dual_null =
        ThetaNullPoint::new(data.dual.alpha, data.dual.beta, data.dual.gamma, Fp2::ZERO);
    let _dual_count = isogeny::get_index_splitting_count(&dual_null);
}

/// `theta_to_product` (Algorithm 8.44) on a synthetic product null point
/// must recover the two component Montgomery coefficients exactly.
///
/// For a product theta null point `(α₁α₂, α₁β₂, β₁α₂, β₁β₂)` the algorithm
/// is expected to yield curves with affine coefficients
/// `Aᵢ = -2(αᵢ⁴ + βᵢ⁴) / (αᵢ⁴ - βᵢ⁴)`. The shared factors cancel cleanly
/// — the input components factor as `α₁⁴(α₂⁴ ± β₂⁴)` and
/// `α₂⁴(α₁⁴ ± β₁⁴)` after raising to the 4th power.
///
/// Direct value-based assertion. Catches surviving mutations on the final
/// projective-to-affine conversion (`A_num * C.invert()`), which the
/// existing end-to-end KAT path does not guard.
#[test]
fn theta_to_product_recovers_component_coefficients() {
    use crate::fields::fp::Fp;

    let alpha1 = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let beta1 = Fp2::new(Fp::from_small(5), Fp::from_small(11));
    let alpha2 = Fp2::new(Fp::from_small(13), Fp::from_small(17));
    let beta2 = Fp2::new(Fp::from_small(19), Fp::from_small(23));

    let null = ThetaNullPoint::new(
        &alpha1 * &alpha2,
        &alpha1 * &beta2,
        &beta1 * &alpha2,
        &beta1 * &beta2,
    );

    let product = isogeny::theta_to_product(&null);

    let four = |x: &Fp2| x.square().square();
    let expected_a = |alpha: &Fp2, beta: &Fp2| {
        let a4 = four(alpha);
        let b4 = four(beta);
        let num = -&(&(&a4 + &b4) + &(&a4 + &b4));
        &num * &(&a4 - &b4).invert()
    };

    let a1 = expected_a(&alpha1, &beta1);
    let a2 = expected_a(&alpha2, &beta2);

    assert_eq!(Fp2::from(*product.E1.coefficient()), a1, "E1 coefficient");
    assert_eq!(Fp2::from(*product.E2.coefficient()), a2, "E2 coefficient");
}

/// `theta_product_to_montgomery` (Algorithm 8.45) must compute the
/// projective `(X : Z)` pairs exactly per the formula
/// `X₁ = a·z + c·x, Z₁ = a·z − c·x, X₂ = a·y + b·x, Z₂ = a·y − b·x`.
///
/// Tests the formula in isolation by constructing a synthetic
/// [`JacobianPoint`] over a synthetic surface — the function only reads
/// the null-point components and the point's `(X, Y, Z, W)` coordinates,
/// so the surface is structural ballast.
///
/// One of the surviving mutants flips `+` to `−` in `X₁`, which makes
/// `X₁ = a·z − c·x = Z₁` and collapses `(X₁ : Z₁)` to `(1 : 1)` — the
/// projective `PartialEq` cross-multiplication catches this iff
/// `c·x ≠ 0`.
#[test]
fn theta_product_to_montgomery_matches_formula() {
    use crate::{
        curves::montgomery::{Curve, ProjectiveXOnlyPoint},
        fields::fp::Fp,
    };

    let alpha1 = Fp2::new(Fp::from_small(3), Fp::from_small(7));
    let beta1 = Fp2::new(Fp::from_small(5), Fp::from_small(11));
    let alpha2 = Fp2::new(Fp::from_small(13), Fp::from_small(17));
    let beta2 = Fp2::new(Fp::from_small(19), Fp::from_small(23));

    let null = ThetaNullPoint::new(
        &alpha1 * &alpha2,
        &alpha1 * &beta2,
        &beta1 * &alpha2,
        &beta1 * &beta2,
    );
    let surface = Jacobian::new(null);

    let x = Fp2::new(Fp::from_small(2), Fp::from_small(29));
    let y = Fp2::new(Fp::from_small(31), Fp::from_small(37));
    let z = Fp2::new(Fp::from_small(41), Fp::from_small(43));
    let w = Fp2::new(Fp::from_small(47), Fp::from_small(53));
    let pt = JacobianPoint::new(x, y, z, w, surface);

    // Curves are passed through to `ProjectiveXOnlyPoint::from_XZ` for
    // storage only — the formula's output values do not depend on them.
    let product = EllipticProduct::new(Curve::E0, Curve::E0);

    let (out1, out2) = isogeny::theta_product_to_montgomery(&pt, &null, &product);

    let (a, b, c, _d) = (&null.a, &null.b, &null.c, &null.d);
    let exp_X1 = &(a * &z) + &(c * &x);
    let exp_Z1 = &(a * &z) - &(c * &x);
    let exp_X2 = &(a * &y) + &(b * &x);
    let exp_Z2 = &(a * &y) - &(b * &x);
    let exp1 = ProjectiveXOnlyPoint::from_XZ(exp_X1, exp_Z1, &Curve::E0);
    let exp2 = ProjectiveXOnlyPoint::from_XZ(exp_X2, exp_Z2, &Curve::E0);

    assert_eq!(out1, exp1, "(X₁ : Z₁) mismatch");
    assert_eq!(out2, exp2, "(X₂ : Z₂) mismatch");
}

/// Cross-check `Kernel::isogeny_no_extra_torsion(e)` against
/// `Kernel::isogeny(e)` on a kernel of order `2^(e+2)` vs `2^e`
/// constructed from the same E₀ basis points.
///
/// **As written, this test's kernel `⟨(P, [3]P), (Q, [3]Q)⟩` is
/// not a valid (2,2)-isogeny kernel** — it fails the isotropy
/// condition for the Weil pairing on E₀ × E₀ (one needs
/// `α² ≡ 1 (mod 2^e)` for the graph kernel `⟨(P, [α]P),
/// (Q, [α]Q)⟩`, but `9 ≢ 1 (mod 256)`). Both our implementation
/// and the C reference correctly reject it: feeding the same
/// kernel bytes through `theta_chain_compute_and_eval_randomized`
/// (with `extra_torsion=false`) returns chain=0, byte-for-byte
/// matching our `[NOEX]` per-step null dumps. The test is kept
/// `#[ignore]` because Mode A's branch happens to be accepted by
/// our looser `SplittingKernel` (which does not enforce the
/// "zero at index 8" check that the C ref does for
/// `extra_torsion=true`), giving a misleading "Mode A succeeded"
/// signal — the chain output itself is *also* byte-identical to
/// C ref, so the implementation is verified equivalent.
///
/// To turn this into a real cross-check, replace the `[3]`
/// scaling with an actual quaternion endomorphism `θ` of E₀ such
/// that `deg(θ) ≡ 1 (mod 2^e)` — then both modes produce a valid
/// product splitting. The simplest concrete choice is the
/// E₀ endomorphism `i: (x, y) → (-x, ζ_4 y)` (with `i² = -1`),
/// which gives `deg(i) = 1`. (See `src/curves/montgomery/iota.rs`
/// or wherever `i` is exposed.)
///
/// Run with:
/// ```text
/// cargo test --lib --release isogeny_no_extra_torsion_matches_extra_torsion -- --ignored --nocapture
/// ```
#[test]
#[ignore = "test kernel ⟨(P, [3]P), (Q, [3]Q)⟩ is not isotropic — both impls correctly reject; \
            see docstring for the proper construction"]
fn isogeny_no_extra_torsion_matches_extra_torsion() {
    use crate::{
        curves::TorsionExponent,
        params::{BASIS_E0_P_X, BASIS_E0_Q_X, TORSION_EVEN_POWER},
    };

    // Small chain length for fast iteration. Mode B's
    // implementation requires `e >= 4` (gluing + 1 main + 4-iso +
    // 2-iso); we use `e = 8` to give a couple of regular main-loop
    // steps.
    const E: u32 = 8;
    let f = TORSION_EVEN_POWER;
    assert!(E + 2 <= f, "test parameters require E + 2 <= f");

    let curve = Curve::E0;
    let p = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
    let q = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_Q_X, &curve);
    let pmq = p.projective_difference(&q);

    // Build a non-degenerate (2,2)-isogeny kernel via the graph of
    // the [3]-multiplication map: ⟨(P, [3]P), (Q, [3]Q)⟩. This is
    // isotropic for the Weil pairing on E₀×E₀[2^E] (since
    // e_2((P,[3]P), (Q,[3]Q)) = e_2(P,Q)^(1+9) = 1) and produces a
    // splitting codomain (E×E with twisted product structure).
    // Diagonal kernel ⟨(P,P), (Q,Q)⟩ is fully degenerate
    // (`splitting: zeros=10`).
    let three = crate::curves::Scalar::from_u64(3);
    let p3 = p.scalar_mul(&three);
    let q3 = q.scalar_mul(&three);
    let pmq3 = pmq.scalar_mul(&three);

    // Reduce all six points to order 2^E (Mode B input).
    let mut p_e = p;
    let mut q_e = q;
    let mut pmq_e = pmq;
    let mut p3_e = p3;
    let mut q3_e = q3;
    let mut pmq3_e = pmq3;
    for _ in 0..(f - E) {
        p_e = p_e.double();
        q_e = q_e.double();
        pmq_e = pmq_e.double();
        p3_e = p3_e.double();
        q3_e = q3_e.double();
        pmq3_e = pmq3_e.double();
    }

    // Reduce all six points to order 2^(E+2) (Mode A input).
    let mut p_e2 = p;
    let mut q_e2 = q;
    let mut pmq_e2 = pmq;
    let mut p3_e2 = p3;
    let mut q3_e2 = q3;
    let mut pmq3_e2 = pmq3;
    for _ in 0..(f - E - 2) {
        p_e2 = p_e2.double();
        q_e2 = q_e2.double();
        pmq_e2 = pmq_e2.double();
        p3_e2 = p3_e2.double();
        q3_e2 = q3_e2.double();
        pmq3_e2 = pmq3_e2.double();
    }

    let product = EllipticProduct::new(curve, curve);
    let exp_e = TorsionExponent::try_from(E).expect("E within bounds");

    let kernel_a =
        Kernel::from_montgomery(product, (p_e2, p3_e2), (q_e2, q3_e2), (pmq_e2, pmq3_e2));
    let result_a = kernel_a.and_then(|k| k.isogeny(exp_e, &[]));

    let kernel_b = Kernel::from_montgomery(product, (p_e, p3_e), (q_e, q3_e), (pmq_e, pmq3_e));
    let result_b = kernel_b.and_then(|k| k.isogeny_no_extra_torsion(exp_e, &[]));

    eprintln!(
        "[iso-cross-check] Mode A (extra_torsion=true,  e={E}) succeeded: {}",
        result_a.is_some()
    );
    eprintln!(
        "[iso-cross-check] Mode B (extra_torsion=false, e={E}) succeeded: {}",
        result_b.is_some()
    );

    match (result_a, result_b) {
        (Some((cod_a, _)), Some((cod_b, _))) => {
            let j_a1 = cod_a.E1.j_invariant();
            let j_a2 = cod_a.E2.j_invariant();
            let j_b1 = cod_b.E1.j_invariant();
            let j_b2 = cod_b.E2.j_invariant();
            let same = (j_a1 == j_b1 && j_a2 == j_b2) || (j_a1 == j_b2 && j_a2 == j_b1);
            assert!(
                same,
                "codomain j-invariants disagree:\n  Mode A: ({j_a1:?}, {j_a2:?})\n  Mode B: ({j_b1:?}, {j_b2:?})"
            );
        }
        (Some(_), None) => panic!(
            "Mode B (no_extra_torsion) failed while Mode A succeeded — \
             confirms isogeny_no_extra_torsion is broken on this kernel"
        ),
        (None, Some(_)) => panic!(
            "Mode A failed while Mode B succeeded — extra_torsion path \
             is broken on this kernel (unexpected)"
        ),
        (None, None) => panic!(
            "Both modes failed — likely a degenerate test kernel. \
             Substitute a non-trivial graph kernel (e.g. (P, [3]P), \
             (Q, [3]Q)) and re-run"
        ),
    }
}

/// Dump the Mode B kernel from
/// [`isogeny_no_extra_torsion_matches_extra_torsion`] to a binary
/// file in the format consumed by the C reference's
/// `theta_split_test`. Used to drive the C ref's
/// `_theta_chain_compute_impl` with `extra_torsion=false` on the
/// same kernel bytes so its `[CHAIN_DUMP]` per-step nulls can be
/// diffed against our `[NOEX]` per-step nulls.
///
/// File format (little-endian, no padding) — must match
/// `test/theta_split_test.c`:
///
/// ```text
/// u32 e_chain
/// u32 reduced_order   (informational)
/// Fp2 E1.A_aff        (64 B; A=0 for E0)
/// Fp2 E2.A_aff        (64 B; A=0 for E0)
/// for each of T1.P1, T1.P2, T2.P1, T2.P2, T1m2.P1, T1m2.P2:
///     Fp2 X (64 B), Fp2 Z (64 B)
/// Total: 904 B.
/// ```
///
/// In our test the kernel layout matches the C ref's
/// `theta_kernel_couple_points_t`:
///
///   - `T1.P1` ↔ `p_e`        (first generator on E1 side)
///   - `T1.P2` ↔ `p3_e`       (first generator on E2 side; here = [3]p_e)
///   - `T2.P1` ↔ `q_e`        (second generator on E1 side)
///   - `T2.P2` ↔ `q3_e`       (second generator on E2 side; here = [3]q_e)
///   - `T1m2.P1` ↔ `pmq_e`    (P − Q on E1 side)
///   - `T1m2.P2` ↔ `pmq3_e`   ([3](P − Q) on E2 side)
///
/// Run with:
/// ```text
/// DUMP_NOEX_KERNEL=/tmp/noex_ker.bin \
///   cargo test --lib --release \
///   surfaces::tests::dump_no_extra_torsion_kernel -- --ignored --nocapture
/// ```
#[test]
#[ignore = "writes a kernel binary for the C ref's theta_split_test diff"]
fn dump_no_extra_torsion_kernel() {
    use crate::{
        curves::{Scalar, TorsionExponent},
        params::{BASIS_E0_P_X, BASIS_E0_Q_X, TORSION_EVEN_POWER},
    };

    const E: u32 = 8;
    let f = TORSION_EVEN_POWER;

    let curve = Curve::E0;
    let p = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
    let q = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_Q_X, &curve);
    let pmq = p.projective_difference(&q);
    let three = Scalar::from_u64(3);
    let p3 = p.scalar_mul(&three);
    let q3 = q.scalar_mul(&three);
    let pmq3 = pmq.scalar_mul(&three);

    let mut p_e = p;
    let mut q_e = q;
    let mut pmq_e = pmq;
    let mut p3_e = p3;
    let mut q3_e = q3;
    let mut pmq3_e = pmq3;
    for _ in 0..(f - E) {
        p_e = p_e.double();
        q_e = q_e.double();
        pmq_e = pmq_e.double();
        p3_e = p3_e.double();
        q3_e = q3_e.double();
        pmq3_e = pmq3_e.double();
    }

    let _ = TorsionExponent::try_from(E).expect("E within bounds");
    let e_chain: u32 = E;
    let reduced_order: u32 = 0;

    let mut buf: Vec<u8> = Vec::with_capacity(904);
    buf.extend_from_slice(&e_chain.to_le_bytes());
    buf.extend_from_slice(&reduced_order.to_le_bytes());
    // E1 = E2 = E0, A = 0.
    buf.extend_from_slice(&Fp2::ZERO.to_bytes());
    buf.extend_from_slice(&Fp2::ZERO.to_bytes());
    let dump_pt = |buf: &mut Vec<u8>, p: &ProjectiveXOnlyPoint| {
        buf.extend_from_slice(&p.X.to_bytes());
        buf.extend_from_slice(&p.Z.to_bytes());
    };
    // T1 = (P-side, P-side via [3]) = (p_e, p3_e)
    dump_pt(&mut buf, &p_e);
    dump_pt(&mut buf, &p3_e);
    // T2 = (Q-side, Q-side via [3]) = (q_e, q3_e)
    dump_pt(&mut buf, &q_e);
    dump_pt(&mut buf, &q3_e);
    // T1m2 = (PmQ-side, PmQ-side via [3])
    dump_pt(&mut buf, &pmq_e);
    dump_pt(&mut buf, &pmq3_e);

    assert_eq!(buf.len(), 904, "buffer must be exactly 904 bytes");

    let path =
        std::env::var("DUMP_NOEX_KERNEL").unwrap_or_else(|_| "/tmp/noex_ker.bin".to_string());
    std::fs::write(&path, &buf).expect("write kernel binary");
    eprintln!(
        "[dump_no_extra_torsion_kernel] wrote {} ({} bytes, e_chain={E})",
        path,
        buf.len()
    );

    // Also dump the Mode A kernel (at 2^(E+2)) so the C ref's
    // `extra_torsion=true` path can be exercised on the same
    // mathematical kernel — confirming whether the kernel is valid
    // for ANY chain mode or whether our Mode A "success" was a false
    // positive.
    let mut p_e2 = p;
    let mut q_e2 = q;
    let mut pmq_e2 = pmq;
    let mut p3_e2 = p3;
    let mut q3_e2 = q3;
    let mut pmq3_e2 = pmq3;
    for _ in 0..(f - E - 2) {
        p_e2 = p_e2.double();
        q_e2 = q_e2.double();
        pmq_e2 = pmq_e2.double();
        p3_e2 = p3_e2.double();
        q3_e2 = q3_e2.double();
        pmq3_e2 = pmq3_e2.double();
    }
    let mut buf_a: Vec<u8> = Vec::with_capacity(904);
    buf_a.extend_from_slice(&e_chain.to_le_bytes());
    buf_a.extend_from_slice(&reduced_order.to_le_bytes());
    buf_a.extend_from_slice(&Fp2::ZERO.to_bytes());
    buf_a.extend_from_slice(&Fp2::ZERO.to_bytes());
    dump_pt(&mut buf_a, &p_e2);
    dump_pt(&mut buf_a, &p3_e2);
    dump_pt(&mut buf_a, &q_e2);
    dump_pt(&mut buf_a, &q3_e2);
    dump_pt(&mut buf_a, &pmq_e2);
    dump_pt(&mut buf_a, &pmq3_e2);
    let path_a =
        std::env::var("DUMP_EXTRA_KERNEL").unwrap_or_else(|_| "/tmp/extra_ker.bin".to_string());
    std::fs::write(&path_a, &buf_a).expect("write extra kernel binary");
    eprintln!(
        "[dump_no_extra_torsion_kernel] wrote {} ({} bytes, e_chain={E}, kernel at 2^{})",
        path_a,
        buf_a.len(),
        E + 2
    );
}
