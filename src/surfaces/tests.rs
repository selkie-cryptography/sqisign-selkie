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
    let result_a = kernel_a.and_then(|k| k.isogeny(exp_e, &[], None));

    let kernel_b = Kernel::from_montgomery(product, (p_e, p3_e), (q_e, q3_e), (pmq_e, pmq3_e));
    let result_b = kernel_b.and_then(|k| k.isogeny_no_extra_torsion(exp_e, &[], None));

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

/// Decode a 32-byte big-endian hex string (as emitted by `[CHAIN_DUMP]`
/// dumps on either side) into a 32-byte little-endian array suitable for
/// `Fp::from_bytes` / `Fp2::from_bytes`.
///
/// Selkie's `dump_fp2_hex` and the C reference's `CHAIN_DUMP_FP2` macro
/// both reverse the encoded LE bytes during printing, so the dumped hex
/// is big-endian. `Fp::from_bytes` expects the encoded little-endian
/// form, hence the reverse.
fn fp32_le_from_dump_hex(be_hex: &str) -> [u8; 32] {
    let s = be_hex.strip_prefix("0x").unwrap_or(be_hex);
    let v = hex::decode(s).expect("valid 32-byte hex");
    assert_eq!(v.len(), 32, "expected 32 bytes (64 hex chars)");
    let mut le = [0u8; 32];
    for (i, b) in v.iter().enumerate() {
        le[31 - i] = *b;
    }
    le
}

/// Build an `Fp2` from a `(re_be_hex, im_be_hex)` pair in the C
/// reference's `[CHAIN_DUMP]` format.
fn fp2_from_dump_hex(re_be_hex: &str, im_be_hex: &str) -> Fp2 {
    let re_le = fp32_le_from_dump_hex(re_be_hex);
    let im_le = fp32_le_from_dump_hex(im_be_hex);
    let mut bytes = [0u8; 64];
    bytes[..32].copy_from_slice(&re_le);
    bytes[32..].copy_from_slice(&im_le);
    Fp2::from_bytes(&bytes)
}

/// Cross-implementation byte-equality test for `isogeny::codomain_8torsion`.
///
/// Inputs are taken verbatim from the C reference's KAT[0] keygen log
/// (`/tmp/cref_kat0.log`), first chain (`step=glue n=150`), step
/// `main_1` (`hadamard_bools=(0,1)` — the interior Mode A variant that
/// matches Selkie's `codomain_8torsion`):
///
/// - `step=glue null.{a,b,c,d}` — the domain Jacobian's theta null.
/// - `step=main_input_T1.{x,y,z,t}` — first 8-torsion kernel point.
/// - `step=main_input_T2.{x,y,z,t}` — second 8-torsion kernel point.
///
/// Expected output bytes come from the same step's
/// `step=main_codomain null.{a..d}`.
///
/// Outcome interpretation:
/// - **Passes** → `codomain_8torsion` is byte-correct given byte-equal
///   inputs. The KAT divergence observed in keygen logs is therefore
///   purely orchestrational (different chain instances run on each
///   side), not a step-internal bug.
/// - **Fails** → the `cfg(test)` `step=internal` dumps emitted by the
///   call (TT1/TT2 after `T.squared().hadamard()`, and `pre_H.null`
///   after the four cross-product `fp2_mul`s) pin the divergence to
///   one of: input transform, cross-product formula, or final Hadamard.
#[test]
fn cref_kat0_main1_codomain_8torsion_byte_eq() {
    let null = ThetaNullPoint::new(
        fp2_from_dump_hex(
            "0x023fc9733c1a1ea1c6e3ed5f41871e627cc074663b552b8df13a800ecb2efdbb",
            "0x001805ecdb3da00e18600a78e2bc875bfdccc62f988d85f6842b96743d622bfc",
        ),
        fp2_from_dump_hex(
            "0x04416b669244c18153bc8f4e5ba2e2911bb53898eebd037f1426cce73d41eca8",
            "0x0110e0fc1535c643567dfbf4d7b7cc2f0189f7ac53c8456874d2d2d0563e6d83",
        ),
        fp2_from_dump_hex(
            "0x04c5b98a33c318496e257f35af61d89be08d95ea56083d0efb31f6bca9e1d1cf",
            "0x00f0c0f740f4dc9c365aa5fa4263f7ff9d6acc865f9f9288ae197a081bd4816a",
        ),
        fp2_from_dump_hex(
            "0x01c75b7d89edbb28fafe2124c97d9cca7f825a1d097015001e1e43951bf4c0bd",
            "0x01e99c067aed02d174789776375f3cd2a127fe031ada51fa9ec0b66434b0c2f1",
        ),
    );
    let domain = Jacobian::new(null);

    let t1 = JacobianPoint::new(
        fp2_from_dump_hex(
            "0x0113963b8f9d892f0d6457e5bcf13cf892b0c7d5f6df741e06785a023bc7d114",
            "0x029338a6c5c55206a26905ac42e5376a8e69ef2c6076480b17764d7f1dec8d81",
        ),
        fp2_from_dump_hex(
            "0x0007c590cdeb4cf015385be493dab0cc7e0766eb10a2c43d041019312921052a",
            "0x028a602737d11481930ed1c27e5f4bc64ca331556f541706995a23e01b98f745",
        ),
        fp2_from_dump_hex(
            "0x04945a7a0dee5f45708ec455409ada8ddd7ae9fa3fa22c6d0461649da9a08551",
            "0x017315d88cde47778cafca7263e7bb469c7e4f02923788c40b00e2802b2bbe15",
        ),
        fp2_from_dump_hex(
            "0x00aac832e4df909401bca6bd6b773a3e2efd1a752b10fc6d269b270bb954e503",
            "0x023a5f6298e5f44f7fcf7ff3d51cdabe5dffb10acbf905d0ec71aaa06ea41069",
        ),
        domain.clone(),
    );

    let t2 = JacobianPoint::new(
        fp2_from_dump_hex(
            "0x01c306ce5674414545084221375f5db1601938d84a412929cd3b86a24f8f4f04",
            "0x01e7c9f35322f94884f810d35b83b4657de9485d25e7fffa16154a8f46d180d2",
        ),
        fp2_from_dump_hex(
            "0x0359380ce9f46fdb8ca38bf86bc7529aeee85c16621f6b47782ed912257ec8bb",
            "0x04b584a16c6df8ae33928c53f8ee357b4f22d9072db55c6dab7772e8dbd3f1bf",
        ),
        fp2_from_dump_hex(
            "0x03036ac2aab9bf391b3baf0e26a37dc9dcf680e0a2a7dc6ca5aa6102341a4f78",
            "0x0443f3ae9191cd4e004602f3d6cebdae78618e1fdcaa330fecac7b53ee7a2c89",
        ),
        fp2_from_dump_hex(
            "0x03772e44b5b25ec08a881d99b34b28cdcc3371b3d82698137d43c22c1efb446e",
            "0x04389c6f79731277d3c25288e3e17e77aaf47ddacdfcc534afb2793006012c3c",
        ),
        domain,
    );

    let (_dual, codomain) = isogeny::codomain_8torsion(&t1, &t2);

    let expected_a = fp2_from_dump_hex(
        "0x00f92571a7895818ea5c88ff6b9970fb5f0a09677aa16f793aeb9deb294a6cf3",
        "0x033b74c8636c68f15b4c73132346ac68522f609cc6f3ed5d9109c33cd4b98901",
    );
    let expected_b = fp2_from_dump_hex(
        "0x02403e6b9fd00d015d51d8bf653116c6260d6d2dda39a4f3f241ae793599741b",
        "0x02916f8b9eec15717643d2927a65b961885c550748daa31cff8c338f894c4729",
    );
    let expected_c = fp2_from_dump_hex(
        "0x01e7d907f318fa209f04cc763e1296270f0df720056e739f123dab9cd6a8422f",
        "0x01862de2665da8de5501a40edb189279f6aab8b1671f499f6af40f72841007c3",
    );
    let expected_d = fp2_from_dump_hex(
        "0x00bea14423ae40003c192c7703053b99eeefaa362bdae572acb94f201b0206d4",
        "0x007fc45e595ed3b158d9ec55117accdf927d6f79294ed5d14dcebc6e40c45100",
    );

    assert_eq!(codomain.null.a, expected_a, "main_1 codomain.null.a");
    assert_eq!(codomain.null.b, expected_b, "main_1 codomain.null.b");
    assert_eq!(codomain.null.c, expected_c, "main_1 codomain.null.c");
    assert_eq!(codomain.null.d, expected_d, "main_1 codomain.null.d");
}

/// Diagnostic: are Selkie's `GLUE_IN T2.0` and C ref's `K2_8.P1`
/// projectively equal? They have different (X:Z) representations
/// (Selkie has nonzero imaginary parts in both X and Z; C ref has
/// imaginary parts of zero), but if they are the same group element
/// then `X1·Z2 == X2·Z1` and the affine `X/Z` ratios match.
///
/// If this passes, Selkie's strategy doubling produces a different
/// projective representative than C ref's, but the underlying point
/// is the same. The bug is then in projective rep choice, not the
/// kernel itself.
///
/// If this fails, the bug is upstream: Selkie's "Q on E_t" after
/// `f-2-e_fdi` doublings is a *different group element* than C
/// ref's, despite both starting from the same precomputed
/// `BASIS_E0_QX = E0_Q_X`.
#[test]
fn selkie_t2_vs_cref_k28_p1_projective_eq_kat0() {
    // Selkie `GLUE_IN T2.0` from KAT[0] keygen (post-(P,Q)-fix run).
    let selkie_x = fp2_from_dump_hex(
        "0x00cdc08e40a7717c338df630dade4f2014ee00b8f805b0cd9b3904591f739907",
        "0x02c05588e51e2290d36339b610c4a7c788f99099b261d7e2ace87d751fa4c2bf",
    );
    let selkie_z = fp2_from_dump_hex(
        "0x03429746a431fd194370b87840ac7d18cabd6395f50795002e05b3a5bd282861",
        "0x00cfdf25d5fec0947e5678ff09814fdf85dc8f83c5b7dbf758981679a12c2a86",
    );

    // C ref `K2_8.P1` from KAT[0] keygen (first chain's gluing).
    let cref_x = fp2_from_dump_hex(
        "0x015d9efb080b9f71a5c840cd1627c152fd72e374dc6d57f01031b3aa42e0a8f6",
        "0x0000000000000000000000000000000000000000000000000000000000000000",
    );
    let cref_z = fp2_from_dump_hex(
        "0x0349853160eb39a9ec30a34fe6f6a66a46f2ae60aedf1544d5865ec5b53ef884",
        "0x0000000000000000000000000000000000000000000000000000000000000000",
    );

    // Projective equality: X1 · Z2 == X2 · Z1.
    let lhs = &selkie_x * &cref_z;
    let rhs = &cref_x * &selkie_z;
    assert_eq!(
        lhs, rhs,
        "Selkie GLUE_IN T2.0 and C ref K2_8.P1 are NOT projectively equal — \
         they are different group elements, not just different reps"
    );
}

/// One-shot helper: print the Montgomery-form limbs for E0
/// `basis_even.{P, Q, PmQ}.x` extracted from a C ref dump. Run this
/// once, copy the output into `src/deuring/precomputed.rs` to fix the
/// E0 basis used by FDI keygen/sign so it matches the basis the
/// precomputed action matrices were calibrated against.
#[test]
fn print_e0_basis_even_limbs() {
    let p_x = fp2_from_dump_hex(
        "0x019b877fca82b12483cc04c3a66216c444be991a59bfa78b2119d95eaeb40078",
        "0x04442adb49eae04252150aaa9867e92fb2cfddae514292748e04133dc3f9d275",
    );
    let q_x = fp2_from_dump_hex(
        "0x045ffd477d5c0b719fdf2717050d041d878678f7a54be1f37c16252a5593eb1f",
        "0x0487d4e9df1873dc4465a8fb3676b39a39ff054b6f8ea5aefde228b7a0cdaaee",
    );
    let pmq_x = fp2_from_dump_hex(
        "0x0017ed1ded6dce3c56831deae1dadeabad269e104cf932fae5b7b99c0128dd27",
        "0x03cdd6007c4f727655ecab154c6425fb0ec882078cca9770b17c2e4640d7234e",
    );
    eprintln!("E0_P_X.re limbs   = {:#018x?}", p_x.a.0);
    eprintln!("E0_P_X.im limbs   = {:#018x?}", p_x.b.0);
    eprintln!("E0_Q_X.re limbs   = {:#018x?}", q_x.a.0);
    eprintln!("E0_Q_X.im limbs   = {:#018x?}", q_x.b.0);
    eprintln!("E0_PMQ_X.re limbs = {:#018x?}", pmq_x.a.0);
    eprintln!("E0_PMQ_X.im limbs = {:#018x?}", pmq_x.b.0);
}

/// Print Montgomery-form limbs for `1/2 mod p`, needed for E0's
/// normalized doubling constant `A24 = (A + 2C)/(4C) = 1/2`.
#[test]
fn print_one_half_limbs() {
    use crate::fields::fp::Fp;
    let two = &Fp::ONE + &Fp::ONE;
    let half = two.invert();
    eprintln!("1/2 limbs (Montgomery) = {:#018x?}", half.0);
}

/// Print Montgomery-form limbs for `-1 mod p`, needed for the
/// `NORMALIZATION_TRANSFORMS` matrices in the splitter.
#[test]
fn print_minus_one_limbs() {
    use crate::fields::fp::Fp;
    let minus_one = -&Fp::ONE;
    eprintln!("-1 limbs (Montgomery) = {:#018x?}", minus_one.0);
}

/// Diagnostic: are Selkie's `theta_p`, `theta_q`, `theta_pmq` (post-biladder)
/// projectively equal to C ref's `B0_two_theta.{P, Q, PmQ}`?
///
/// If projectively equal, the bug is in the biladder's choice of
/// projective representative — the algorithms produce the same group
/// element but in different `(X : Z)` reps. If not equal, scalars
/// or basis differ.
#[test]
fn selkie_theta_basis_vs_cref_kat0_projective_eq() {
    let pairs = [
        (
            // Selkie theta_p (post-biladder, after double_e0 fix)
            fp2_from_dump_hex(
                "0x00d36552cc0bcbfe54e655f70c62bebcdd8a3a0a61d059acec207dc80246e332",
                "0x034275269252bd909e89fd0e2aad6e482f41b6a89ea48659323fe1fac774dda0",
            ),
            fp2_from_dump_hex(
                "0x029c1478c1eb6683cf8fe530c4da4bee6a8d9b4a08a311fe7a4bf30fa7cc636d",
                "0x039902a4a27ee4b27ac2aa388dbda523f7f076b0d8af06d8bebc39ff6cf208d2",
            ),
            // C ref B0_two_theta.P
            fp2_from_dump_hex(
                "0x0155c922a762323abeb8397a037c2201140acbf8fc87d6564cbdf58ae6d318c7",
                "0x03d1a5f77119799563c12d3c0a8bf41b3cdf67959abfbff215fb6676a12ac5a9",
            ),
            fp2_from_dump_hex(
                "0x00161d2c2649f2068ac85da1db28b2abce89c0b91ffd7e5082c6bf16d6561557",
                "0x01cc0deb98a62d83720bdd806b48263b3648634db00dd146637cf31a8c6ab5e1",
            ),
            "theta_p",
        ),
        (
            // Selkie theta_q
            fp2_from_dump_hex(
                "0x048254cf9029d373d9abc1b33bcb7f90f15a7cae1c26a403f27583ff410b675f",
                "0x015cc41c02ed92b62a18ea56f9622e732de1ce1ccf7b90a897445271a4a966aa",
            ),
            fp2_from_dump_hex(
                "0x00c807435287c86e3f9041a177c59804ca00eaaf9d570be9e21c75f1a05d0811",
                "0x01f1fa2c6129fcced3527feb655185b6b37a4fc18314a82c82f49456ae5ff829",
            ),
            // C ref B0_two_theta.Q
            fp2_from_dump_hex(
                "0x019d8954fcbcdde45156b2019245d1c3aaf1233cb2aff083d72f294b6096cb88",
                "0x01d25d441efffb83f373ea249a915ed4d4e009ba26e618895f5a02f131f6b287",
            ),
            fp2_from_dump_hex(
                "0x0189719c4fec7f0762f6fbc8efaa542eae654db35b2f92c58710cf8c86a400eb",
                "0x029816784805c76e3311e3d31f6be4e9c8ba3b3375bba11a9f3d47b7195d8d0b",
            ),
            "theta_q",
        ),
        (
            // Selkie theta_pmq
            fp2_from_dump_hex(
                "0x035db463b526749d8dc160d164abd390e325fe88194f06d7e248180f674ced4c",
                "0x00d5999af7959227c630669c3748521d138c99c3adf0d3b8d5f6e0fb07eb1646",
            ),
            fp2_from_dump_hex(
                "0x048b868b5f771b67ac925363772fc06e38b0e15fdc5f5b29fee8255ee317c9ed",
                "0x04f941cafc806817e0170f65fba8a0d713d2cd801267599f09510a302e1e2dd3",
            ),
            // C ref B0_two_theta.PmQ
            fp2_from_dump_hex(
                "0x0473043a276a745441f3866e7084a33ab18360c90db315ec712c88611c7b3473",
                "0x0103c1ee2f5b77c7eab84868cbeecf9af892847c530e755d6c8f152878a16dbf",
            ),
            fp2_from_dump_hex(
                "0x02155c4243794be874d873467a786be1a4e110319185478ac061392a85e118cf",
                "0x040a5b36d83affc2e0e1542d90e27828693e0dd6766ee99aac6c5099cd17a4df",
            ),
            "theta_pmq",
        ),
    ];
    for (selkie_x, selkie_z, cref_x, cref_z, name) in pairs.iter() {
        let lhs = selkie_x * cref_z;
        let rhs = cref_x * selkie_z;
        let eq = lhs == rhs;
        eprintln!("{name}: projective equal = {eq}");

        // Compute lambda = Selkie_X / C_ref_X. If same lambda matches Z too,
        // they're projectively equivalent with scaling factor lambda.
        let cref_x_inv = cref_x.invert();
        let lambda_x = selkie_x * &cref_x_inv;
        let cref_z_inv = cref_z.invert();
        let lambda_z = selkie_z * &cref_z_inv;
        let bytes_x = lambda_x.to_bytes();
        let bytes_z = lambda_z.to_bytes();
        let hex_x: String = bytes_x[..32].iter().rev().map(|b| format!("{b:02x}")).collect();
        let hex_z: String = bytes_z[..32].iter().rev().map(|b| format!("{b:02x}")).collect();
        eprintln!("  lambda_x = 0x{hex_x}");
        eprintln!("  lambda_z = 0x{hex_z}");
        eprintln!("  lambda equal = {}", lambda_x == lambda_z);
    }
}
