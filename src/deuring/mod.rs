//! The Deuring correspondence: converting between ideals and isogenies.
//!
//! This module bridges the quaternion algebra world ([`crate::quaternions`])
//! and the elliptic curve world ([`crate::curves`], [`crate::surfaces`]).
//! The Deuring correspondence establishes a bijection between left ideals
//! of a maximal order O ⊂ B_{p,∞} and separable isogenies from E where
//! End(E) ≅ O.
//!
//! The main algorithms are:
//! - [`IdealToIsogeny`][Alg. 3.13] (Algorithm 3.13): convert an O₀-ideal to an
//!   isogeny and its images on torsion points
//! - [`FixedDegreeIsogeny`][Alg. 3.15] (Algorithm 3.15): compute an isogeny of
//!   prescribed degree from precomputed data
//! - [`SuitableIdeals`][Alg. 3.16] (Algorithm 3.16): decompose an ideal into
//!   pieces with coprime odd norms
//!
//! See [§3.2] of the SQIsign specification.
//!
//! [§3.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.3.2
//! [Alg. 3.13]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.13
//! [Alg. 3.15]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.15
//! [Alg. 3.16]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.16

pub(crate) mod endomorphism;
pub(crate) mod precomputed;

use endomorphism::EndomorphismAction;
use precomputed::ENDOMORPHISM_MATRICES;

use crate::{
    curves::{
        TorsionBasis, TorsionExponent,
        isogeny::IsogenyDegree,
        montgomery::{Curve, ProjectiveXOnlyPoint},
        scalar::Scalar,
    },
    params::QUAT_REPRES_BOUND_INPUT,
    quaternions::{
        bigint::BigInt,
        lattice::{ExtremalOrder, LeftIdeal},
        precomputed::{CONNECTING_IDEAL_NORMS, EXTREMAL_ORDERS},
    },
    surfaces,
};

/// Computes a u-isogeny from E_t via a (2,2)-isogeny chain.
///
/// [FixedDegreeIsogeny][Alg. 3.15] (Algorithm [3.15][Alg. 3.15]):
/// given a positive odd [`IsogenyDegree`] u < 2^{f−2} and the
/// precomputed data for curve E_t, finds an endomorphism θ of E_t
/// with nrd(θ) = u·(2^{e_FDI} − u), then computes a
/// (2^{e_FDI}, 2^{e_FDI})-isogeny on E_t × E_t whose projection
/// gives the u-isogeny φ: E_t → E.
///
/// Returns `(E, φ(P_t), φ(Q_t))`.
///
/// # Side-channel considerations
///
/// WARNING: Not constant-time — calls RepresentInteger (variable-time
/// search) and the (2,2)-chain length depends on e_FDI.
///
/// TODO(ct): Make constant-time before production use. Called on
/// secret-derived (u, t) values during signing (Algorithm 4.2 via
/// IdealToIsogeny).
///
/// [Alg. 3.15]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.15
fn fixed_degree_isogeny<R: rand_core::RngCore>(
    order: &'static ExtremalOrder<4>,
    u: &IsogenyDegree,
    rng: &mut R,
) -> Option<(
    Curve,
    ProjectiveXOnlyPoint,
    ProjectiveXOnlyPoint,
    ProjectiveXOnlyPoint,
)> {
    let f = TorsionExponent::FULL;
    let p_bits = 251u32; // ⌈log₂(p)⌉ for NIST-I

    // Look up the precomputed curve and torsion basis for this order.
    // `EXTREMAL_ORDERS.len() == ExtremalCurve::ALL.len()` is enforced
    // at compile time in `precomputed.rs`, so the `try_from` below
    // cannot fail for any `t` returned by `.position()`.
    let t = EXTREMAL_ORDERS.iter().position(|o| o.q() == order.q())?;
    let curve_idx = precomputed::torsion_basis::ExtremalCurve::try_from(t)
        .expect("EXTREMAL_ORDERS length matches ExtremalCurve::ALL by compile-time assert");

    let (px, qx, pmq_x, a_coeff) = curve_idx.basis();
    let curve_t = if curve_idx == precomputed::torsion_basis::ExtremalCurve::E0 {
        Curve::E0
    } else {
        Curve::from(crate::curves::montgomery::Coefficient::from(a_coeff))
    };
    let p_t = ProjectiveXOnlyPoint::from_affine_x(px, &curve_t);
    let q_t = ProjectiveXOnlyPoint::from_affine_x(qx, &curve_t);
    let pmq_t = ProjectiveXOnlyPoint::from_affine_x(pmq_x, &curve_t);
    // The biladder's three-point ladder requires a PmQ whose
    // projective representative is consistent with the precomputed
    // action matrices. Computing PmQ via `projective_difference`
    // picks a different representative that causes the
    // Okeya-Sakurai lift to recover the wrong y-sign.
    let basis_t = TorsionBasis::from_propagated(p_t, q_t, pmq_t);
    // CT linear-scan over all seven candidate rows. `curve_idx` is
    // derived from the secret signing-key ideal's right order
    // (Algorithm 4.2, line 7 `I_sig_response → right_order`), so the
    // lookup must not depend on the index through memory access.
    let endo_t = EndomorphismAction {
        order: order.order(),
        generators: curve_idx.gen_matrices(),
    };

    // Step 1: e_FDI = min(f − 2, ⌈log₂(p)⌉ − ⌈log₂(u)⌉ + QUAT_repres_bound_input)
    let e_fdi = core::cmp::min(
        f.value() - 2,
        p_bits.saturating_sub(u.bit_length()) + QUAT_REPRES_BOUND_INPUT,
    );

    // Step 2: θ ← RepresentInteger(u·(2^{e_FDI} − u), O_t, true)
    let u_wide = u.to_bigint_wide();
    let two_e_fdi = BigInt::<8>::ONE << e_fdi;
    // If u ≥ 2^{e_FDI}, the product is non-positive — no solution.
    if u_wide >= two_e_fdi {
        return None;
    }
    let m = u_wide.ct_mul(&two_e_fdi.ct_sub(&u_wide));
    let order_wide = ExtremalOrder::<8>::from(*order);
    let theta = match order_wide.represent_integer(&m, true, rng) {
        Some(t) => t,
        None => {
            return None;
        }
    };

    // Step 3: M_θ via order-basis decomposition.
    let m_theta = endo_t.apply(&theta, f)?;
    // Step 3.5: Multiply M_θ entries by u⁻¹ mod 2^{e_FDI+2}.
    //
    // θ has norm u·(2^{e_FDI} − u), so the endomorphism θ/u has
    // degree 2^{e_FDI} − u, coprime to 2. The kernel of the
    // (2,2)-chain is generated by (θ/u)(basis), not θ(basis).
    // Since the action matrix is linear in θ, multiplying matrix
    // entries by u⁻¹ is equivalent to applying (θ/u) to the
    // basis. The modulus 2^{e_FDI+2} suffices because the kernel
    // lives in E[2^{e_FDI}] and the extra +2 accounts for the
    // gluing's order-4 structure.
    //
    // # Divergences
    //
    // This step is not in the spec pseudocode (Algorithm 3.15)
    // but is present in the C ref (dim2id2iso.c lines 133–143).
    // Without it, the ActionByTranslation determinant is
    // degenerate and the (2,2)-chain always fails.
    let u_scalar = u.to_scalar();
    let u_inv = u_scalar
        .inv_mod2k(e_fdi + 2)
        .expect("u is odd by IsogenyDegree invariant");
    let m00 = m_theta.entry(0, 0).mul_mod2k(&u_inv, e_fdi + 2);
    let m01 = m_theta.entry(0, 1).mul_mod2k(&u_inv, e_fdi + 2);
    let m10 = m_theta.entry(1, 0).mul_mod2k(&u_inv, e_fdi + 2);
    let m11 = m_theta.entry(1, 1).mul_mod2k(&u_inv, e_fdi + 2);

    // Step 4–5: Match the C ref's flow (dim2id2iso.c:121–160):
    //
    // 1. Double Montgomery basis from order 2^f to order 2^(e_fdi+2)
    // 2. Apply θ/u via three biladder calls at e_fdi+2 precision
    // 3. Lift both bases to Jacobian
    //
    // The C ref doubles BEFORE the biladder (line 121), then applies
    // at precision `length + HD_extra_torsion` (line 148). This is
    // critical: the scalars are mod 2^(e_fdi+2), so the basis must
    // have matching order. Using full-precision biladder on the full
    // 2^f-order basis computes the WRONG endomorphism because the
    // extra zero bits at positions e_fdi+2..f change the group
    // element (verified empirically).
    let doublings = f.value() - 2 - e_fdi;
    let mut doubled_p = basis_t.P;
    let mut doubled_q = basis_t.PmQ;
    let mut doubled_pmq = basis_t.Q;
    for _ in 0..doublings {
        doubled_p = doubled_p.double();
        doubled_q = doubled_q.double();
        doubled_pmq = doubled_pmq.double();
    }
    // Do NOT normalize the doubled basis points. The C reference's
    // biladder (`xDBLMUL`, `ec.c:384`) operates on the unnormalized
    // post-doubling basis (`tmp_bas` in `matrix_application_even_basis`,
    // `id2iso.c:103`), and the subsequent `lift_basis` (`basis.c:121`)
    // normalizes only `P.z` to 1 before calling `lift_basis_normalized`.
    // [`TorsionBasis::lift`] normalizes its `R` component internally via
    // `R.Z.invert()`, so we get the same lift behavior as C ref without
    // having to pre-normalize here. Pre-normalizing `doubled_p` would
    // make Selkie's biladder run on a different projective `(X : Z)`
    // representative than C ref, producing a `theta_q`/`theta_pmq` rep
    // that flows into the Okeya-Sakurai lift differently and ultimately
    // gives a byte-different (but projectively-equal) `K2_8.P2`.
    let doubled_basis = TorsionBasis::from_propagated(doubled_p, doubled_q, doubled_pmq);

    let endo_bits = TorsionExponent::try_from(e_fdi + 2).ok()?;
    let theta_p = doubled_basis.biscalar_mul(&m00, &m10, endo_bits);
    let theta_q = doubled_basis.biscalar_mul(&m01, &m11, endo_bits);
    let theta_pmq = doubled_basis.biscalar_mul(
        &m00.sub_mod2k(&m01, e_fdi + 2),
        &m10.sub_mod2k(&m11, e_fdi + 2),
        endo_bits,
    );

    // Lift using (P, Q) kernel generators (with PmQ as the difference
    // point), matching the C reference's `copy_bases_to_kernel`
    // (`hd.c:82-93`). The lifted Jacobian pairs `(p_jac_1, p_jac_2)`
    // and `(q_jac_1, q_jac_2)` form the (2,2)-isogeny kernel
    // generators `K1 = (P, θP)` and `K2 = (Q, θQ)` on the product
    // `E_t × E_t`.
    let comp1 = TorsionBasis::from_propagated(doubled_p, doubled_q, doubled_pmq);
    let (p_jac_1, q_jac_1) = comp1.lift(&curve_t)?;
    let comp2 = TorsionBasis::from_propagated(theta_p, theta_q, theta_pmq);
    let (p_jac_2, q_jac_2) = comp2.lift(&curve_t)?;

    // No Jacobian doubling needed — basis was doubled before biladder.
    let k1_jac = (p_jac_1, p_jac_2);
    let k2_jac = (q_jac_1, q_jac_2);

    // Step 6: (2,2)-isogeny chain on E_t × E_t.
    let product = surfaces::EllipticProduct::new(curve_t, curve_t);
    let kernel = surfaces::Kernel::from_jacobian(product, k1_jac, k2_jac);

    // Push all three basis points (P, Q, PmQ) through the chain,
    // matching the C ref (dim2id2iso.c:903-905). The outer chain
    // needs all three for `apply_scaled` — using only P and Q forces
    // `projective_difference` to recompute PmQ, giving an inconsistent
    // projective representative that breaks the Okeya-Sakurai lift.
    let zero = ProjectiveXOnlyPoint::identity(&curve_t);
    let (codomain, images) = kernel.isogeny_extra_torsion(
        TorsionExponent::try_from(e_fdi).ok()?,
        &[(basis_t.P, zero), (basis_t.PmQ, zero), (basis_t.Q, zero)],
    )?;

    let e_out = &codomain.E1;
    let p_out = images[0].0;
    let q_out = images[1].0;
    let pmq_out = images[2].0;
    if e_out.recover_y(&p_out.to_affine_x()).is_none()
        || e_out.recover_y(&q_out.to_affine_x()).is_none()
    {
        return None;
    }

    Some((codomain.E1, p_out, q_out, pmq_out))
}

/// [IdealToIsogeny][Alg. 3.13] (Algorithm [3.13][Alg. 3.13]).
///
/// Given a left O₀-ideal I, computes the codomain E_I of the
/// corresponding isogeny φ_I : E₀ → E_I, and the images
/// φ_I(P₀) and φ_I(Q₀) of the torsion basis.
///
/// Internally decomposes I via [SuitableIdeals][Alg. 3.16],
/// computes two odd-degree isogenies via [FixedDegreeIsogeny][Alg. 3.15],
/// and combines them through a (2,2)-isogeny chain.
///
/// # Side-channel considerations
///
/// WARNING: Not constant-time — calls SuitableIdeals and
/// RepresentInteger (both variable-time).
///
/// TODO(ct): Make constant-time before production use. Called on
/// secret-derived ideals during signing (Algorithm 4.2).
///
/// [Alg. 3.13]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.13
/// [Alg. 3.15]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.15
/// [Alg. 3.16]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.16
impl<const N: usize> LeftIdeal<N> {
    /// Computes the isogeny corresponding to this ideal.
    ///
    /// Returns `(E_I, φ_I(P₀), φ_I(Q₀), φ_I(P₀ − Q₀))` or `None` if
    /// the decomposition or chain fails probabilistically.
    ///
    /// # Why a `PmQ` is returned
    ///
    /// Downstream consumers (`SplitAuxiliaryIsogeny`, even-response
    /// chain) form additional kernels from these basis points. The
    /// (2,2)-chain's `lift_basis` (Okeya-Sakurai y-recovery) requires
    /// a `PmQ` projective representative consistent with the chain's
    /// own evaluation history of `P` and `Q`. Recomputing
    /// `projective_difference(P, Q)` after the chain picks a sqrt
    /// branch that is not aligned with the chain's internal lift,
    /// which manifests as `splitting_index_count() = 0` at the next
    /// (2,2)-chain's terminal theta null. Pushing `PmQ` through the
    /// chain alongside `P` and `Q`, and propagating it through the
    /// final `M_{β₁}` matrix application via `biscalar_mul`, keeps
    /// every downstream kernel splittable.
    ///
    /// # Divergences
    ///
    /// The SQIsign v2.0.1 spec (Algorithm 3.13) describes
    /// `IdealToIsogeny` as returning only `(E_I, φ_I(P₀), φ_I(Q₀))`
    /// and does not specify how the projective representative of
    /// `φ_I(P₀ − Q₀)` is to be constructed for downstream consumers.
    /// The C reference threads `PmQ` through every isogeny it
    /// computes (a `theta_couple_curve_with_basis_t` always carries
    /// `B.PmQ`). Implementing the spec without that thread produces
    /// a build that fails interoperability against C-reference KAT
    /// vectors at the response phase even when every other invariant
    /// matches.
    pub fn to_isogeny<R: rand_core::RngCore>(
        self,
        rng: &mut R,
    ) -> Option<(
        Curve,
        ProjectiveXOnlyPoint,
        ProjectiveXOnlyPoint,
        ProjectiveXOnlyPoint,
    )> {
        let norm = *self.norm();
        self.to_isogeny_with_norm(&norm, rng)
    }

    /// [`to_isogeny`](Self::to_isogeny) with an explicit norm override
    /// for the post-matrix scaling factor.
    ///
    /// `to_isogeny` uses `self.norm()` for both the suitable-ideals
    /// decomposition and the post-matrix `1/(nrd(I)·d₁)` scaling.
    /// When the caller has pre-reduced the ideal (replacing `I` with
    /// `δ⁻¹ · I` to fit a width budget), `self.norm()` is the
    /// *reduced* norm — but the kernel-isotropy condition consumed
    /// downstream by `SplitAuxiliaryIsogeny` requires the *original*
    /// norm in the scaling formula. The C reference handles this
    /// implicitly: `find_uv` makes a local copy of the input and
    /// reduces it internally, so `lideal->norm` (used at the
    /// post-matrix scaling step) keeps the original value even
    /// though the enumeration uses the reduced lattice.
    ///
    /// This entry point lets the caller supply the original norm
    /// while still passing the pre-reduced ideal as `self`.
    pub fn to_isogeny_with_norm<R: rand_core::RngCore>(
        self,
        original_norm: &BigInt<N>,
        rng: &mut R,
    ) -> Option<(
        Curve,
        ProjectiveXOnlyPoint,
        ProjectiveXOnlyPoint,
        ProjectiveXOnlyPoint,
    )> {
        const { assert!(N <= 8, "to_isogeny supports N ≤ 8") };
        let f = TorsionExponent::FULL;

        // Step 1: Decompose via SuitableIdeals.
        #[cfg(test)]
        let _t0 = std::time::Instant::now();
        let sui = self.suitable_ideals()?;
        // Cross-check dump against C ref's [KEYGEN_PROBE] in
        // `id2iso/ref/lvlx/dim2id2iso.c:846`. Same format/order so a
        // diff localizes whether `(s,t,u,v,β₁,β₂,d₁,d₂)` matches —
        // i.e., whether `reduce_to_prime_norm` produced the same
        // reduced ideal as the C ref despite same DRBG byte stream.
        // Steps 2–3: degrees (already in sui.factor1.degree, sui.factor2.degree).
        let d1 = &sui.factor1.degree;
        let _d2 = &sui.factor2.degree;

        // Step 4: E_u, φ_u(P_s), φ_u(Q_s) ← FixedDegreeIsogeny(s, u)
        #[cfg(test)]
        let _t1 = std::time::Instant::now();
        let u_deg = IsogenyDegree::new_odd(*sui.u.as_limbs())?;
        let (e_u, phi_u_p, phi_u_q, phi_u_pmq) =
            fixed_degree_isogeny(sui.factor1.order, &u_deg, rng)?;
        // Step 5: E_v, φ_v(P_t), φ_v(Q_t) ← FixedDegreeIsogeny(t, v)
        #[cfg(test)]
        let _t2 = std::time::Instant::now();
        let v_deg = IsogenyDegree::new_odd(*sui.v.as_limbs())?;
        let (e_v, phi_v_p, phi_v_q, phi_v_pmq) =
            fixed_degree_isogeny(sui.factor2.order, &v_deg, rng)?;
        // Step 6: second component of the outer kernel.
        //
        // # Divergences
        //
        // The spec (Algorithm 3.13, line 6) writes
        // `(1/(nrd(I) · nrd(J_t))) · M_{β₁⁻¹ · β₂}` applied to
        // `(φ_v(P_t), φ_v(Q_t))`, paired with `[d₁] φ_u(P_s)` on the
        // first component (line 7). The C reference
        // (`dim2id2iso.c:881-1014`) instead builds
        // `θ = β₂ · conj(β₁)` as a quaternion, multiplies its coords
        // by `invmod(d₁ · nrd(connecting ideal), 2^f)`, and applies
        // it on the second component — with `φ_u(P_s)` (no `[d₁]`)
        // on the first.
        //
        // The two conventions differ at the quaternion level
        // (`conj(β₁)·β₂` vs `β₂·conj(β₁)` — quaternions do not
        // commute), in the scaling factor (spec `1/(nrd(I)²·d₁)` vs
        // C ref `1/(nrd(I)·d₁)`), and in whether `[d₁]` appears on
        // the first component. We follow the C ref because it is
        // the implementation whose signatures verify against the
        // published KAT vectors.
        //
        // Using the identity `M_{conj(β)} ≡ adj(M_β) (mod 2^f)` —
        // both have determinant `nrd(β)`, and
        // `M_β · M_{conj(β)} = nrd(β) · I` — we assemble
        //
        //   M_θ = M_{β₂} · M_{conj(β₁)} = M_{β₂} · adj(M_{β₁}).
        //
        // For s = t = 0: `nrd(J_t) = 1` and
        // `nrd(β₁) = d₁ · nrd(parent_ideal)`, so the total scalar
        // applied to `M_θ` before acting on the basis is
        // `1 / (nrd(parent_ideal) · d₁)`.
        //
        // For `t > 0` the spec and C reference ( `dim2id2iso.c:885-889`)
        // add a factor of `nrd(J_t)` to the denominator to account
        // for the pushforward `β_2 ∈ J_t · self`: β_2 is scaled by
        // `nrd(J_t)` relative to its O_0-representative, and the
        // matrix action inherits that scale. The factor for `s` does
        // not appear — β_1 enters through `fixed_degree_isogeny(s, u)`
        // on the first component, which already handles the per-order
        // embedding. See [§3.1.7.2].
        //
        // For `t = 0`, `CONNECTING_IDEAL_NORMS[0] = 1` (J_0 is O₀
        // itself), so multiplying by it is a no-op. The gate on
        // `t_index > 0` matches C ref's `find_uv` (`dim2id2iso.c:651`)
        // skipping the extra `nrd(J_t)` factor when both sides came
        // from the special order.
        //
        // [§3.1.7.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.7.2
        let modulus = BigInt::<4>::ONE << f.value();
        let s_index = EXTREMAL_ORDERS
            .iter()
            .position(|o| o.q() == sui.factor1.order.q())?;
        let t_index = EXTREMAL_ORDERS
            .iter()
            .position(|o| o.q() == sui.factor2.order.q())?;
        // After `suitable_ideals`'s cross-order post-processing
        // (`ideal.rs` `(j_i != 0) ? beta_i = conj(delta · beta_i)`),
        // beta_i ∈ I ⊆ O₀ regardless of which alternate order it
        // came from. Always decompose on O₀ for cross-order pairs,
        // matching C ref's `endomorphism_application_even_basis(...,
        // index_alternate_curve=0, ...)` at `dim2id2iso.c:1054, 1156`.
        // For the pure-special path (s = t = 0) decompose on O₀ too
        // — equivalent.
        let cross_order = s_index != 0 || t_index != 0;
        let endo_index_s = if cross_order { 0 } else { s_index };
        let endo_index_t = if cross_order { 0 } else { t_index };
        let endo_s = EndomorphismAction {
            order: if cross_order {
                EXTREMAL_ORDERS[0].order()
            } else {
                sui.factor1.order.order()
            },
            generators: [
                ENDOMORPHISM_MATRICES[endo_index_s][3],
                ENDOMORPHISM_MATRICES[endo_index_s][4],
                ENDOMORPHISM_MATRICES[endo_index_s][5],
            ],
        };
        let endo_t = EndomorphismAction {
            order: if cross_order {
                EXTREMAL_ORDERS[0].order()
            } else {
                sui.factor2.order.order()
            },
            generators: [
                ENDOMORPHISM_MATRICES[endo_index_t][3],
                ENDOMORPHISM_MATRICES[endo_index_t][4],
                ENDOMORPHISM_MATRICES[endo_index_t][5],
            ],
        };
        let m_beta1 = endo_s.apply(&sui.factor1.beta, f)?;
        let m_beta2 = endo_t.apply(&sui.factor2.beta, f)?;
        let m_beta1_adj = m_beta1.adjugate_mod(f.value());
        let m_prod = m_beta2.mat_mul_mod(&m_beta1_adj, f.value());

        // scale = 1 / (nrd(I) · d₁) mod 2^f (for s = t = 0).
        //
        // The spec formula uses `nrd(I)` for the caller-supplied
        // ideal, not the reduced equivalent `parent_ideal` that β
        // was enumerated from. The C reference (`dim2id2iso.c:885`,
        // `ibz_mul(&theta.denom, &theta.denom, &lideal->norm)`)
        // matches: `lideal` is the original input, not the reduced
        // copy used internally by `find_uv`. Using the reduced
        // ideal's norm here breaks whenever `nrd(parent_ideal)` is
        // even (the reduced norm is `nrd(δ)/nrd(I)` for an
        // LLL-first vector δ ∈ I, and that ratio can be even even
        // for odd-prime `nrd(I)`), which leaves
        // `invmod(parent_norm · d₁, 2^f)` undefined.
        //
        // The translation between β's reduced-ideal provenance
        // (where `nrd(β) = d · nrd(parent_ideal)`) and the
        // original-ideal scaling is absorbed by the identity
        // `parent_ideal = I · δ̄/nrd(I)` — conjugate multiplication
        // by δ in the quaternion algebra shifts `nrd(β)` by
        // `nrd(δ)/nrd(I)`, and the resulting matrix equation
        // collapses back to the spec's `1/nrd(I)` scaling modulo
        // 2^f.
        // Reduce the original norm mod 2^f; the scaling factor is
        // computed mod 2^f, so the value always fits in `BigInt<4>`
        // even when `original_norm` itself spans more limbs.
        let modulus_8: BigInt<8> = modulus.widen();
        let original_norm_mod_f_8 = original_norm.widen::<8>().ct_mod(&modulus_8);
        let parent_norm: BigInt<4> = original_norm_mod_f_8
            .narrow_to::<4>()
            .expect("nrd(I) mod 2^248 fits in BigInt<4>");
        let d1_big = BigInt::<4>::from_sign_and_limbs(0, *d1.limbs());
        let scale_denom = {
            let base = parent_norm.ct_mul(&d1_big).ct_mod(&modulus);
            if t_index > 0 {
                base.ct_mul(&CONNECTING_IDEAL_NORMS[t_index])
                    .ct_mod(&modulus)
            } else {
                base
            }
        };
        let scale_inv = scale_denom.invert_mod(&modulus)?;
        let s = Scalar::from(scale_inv);
        let fv = f.value();
        let s00 = s.mul_mod2k(m_prod.entry(0, 0), fv);
        let s01 = s.mul_mod2k(m_prod.entry(0, 1), fv);
        let s10 = s.mul_mod2k(m_prod.entry(1, 0), fv);
        let s11 = s.mul_mod2k(m_prod.entry(1, 1), fv);
        let fdi_v_basis = TorsionBasis::from_propagated(phi_v_p, phi_v_q, phi_v_pmq);
        let p_step6 = fdi_v_basis.eval_decomposition(&s00, &s10);
        let q_step6 = fdi_v_basis.eval_decomposition(&s01, &s11);
        let pmq_step6 =
            fdi_v_basis.eval_decomposition(&s00.sub_mod2k(&s01, fv), &s10.sub_mod2k(&s11, fv));

        // Steps 7–8: Build kernel points on E_u × E_v.
        //
        // # Divergences
        //
        // The spec (Algorithm 3.13, line 7) writes `[d₁] φ_u(P_s)`
        // for the first component. The C reference
        // (`dim2id2iso.c:949-951`) uses `φ_u(P_s)` directly — no
        // `[d₁]` multiplication. The two are equivalent for
        // splitting (the spec's `[d₁]` factor cancels against the
        // extra `1/d₁` implicit in the spec's second-component
        // scaling of `1/(nrd(I)²·d₁)` vs. the C ref's
        // `1/(nrd(parent)·d₁)`). We follow the C ref's convention
        // on both sides of the kernel since it is the
        // implementation whose signatures verify against the
        // published KAT vectors.
        let mut kp_first = phi_u_p;
        let mut kq_first = phi_u_q;
        let mut kpmq_first = phi_u_pmq;
        let mut kp_second = p_step6;
        let mut kq_second = q_step6;
        let mut kpmq_second = pmq_step6;

        // # Divergences
        //
        // The spec (Algorithm 3.13, between lines 8 and 9) says
        // to double the kernel "to reduce to 2^sui.e-torsion"
        // before passing it to Algorithm 8.47. The natural
        // reading -- double by `f - sui.e` -- is wrong.
        //
        // Algorithm 8.47 as described in the spec, and as
        // implemented by our [`Kernel::isogeny`], runs a chain
        // of `sui.e` 8-torsion isogenies. The chain's internal
        // penultimate and ultimate steps fold the kernel's
        // 4-torsion and 2-torsion residues into the last two
        // `(2,2)`-isogenies via the `hadamard_bool` mechanism
        // of Algorithm 8.41. This requires the kernel to enter
        // the chain with order `2^(sui.e + 2)` -- two torsion
        // bits above the kernel subgroup.
        //
        // The C reference implements two chain variants
        // (`theta_isogenies.c:1088`) keyed on `extra_torsion`.
        // With `extra_torsion = true`, the chain matches the
        // spec's uniform 8-torsion formulation and needs a
        // `2^(sui.e + 2)`-torsion kernel. With
        // `extra_torsion = false`, the chain runs 2 fewer
        // 8-torsion steps and adds dedicated 4-torsion and
        // 2-torsion steps at the tail, consuming only a
        // `2^sui.e`-torsion kernel. `dim2id2iso.c:1128` calls
        // the outer chain with `extra_torsion = false`, while
        // `dim2id2iso.c:181` (Algorithm 3.15) uses
        // `extra_torsion = true`.
        //
        // Our chain only supports the `extra_torsion = true`
        // variant, so we pad the kernel by leaving 2 extra
        // torsion bits instead of doubling down to
        // `2^sui.e` exactly. The `try_find_uv` filter
        // `v_2(gcd(u, v)) >= 2` ensures `sui.e <= f - 2`, so
        // `scale = f - sui.e - 2 >= 0` is always well-defined.
        //
        // Without this compensation the chain runs to
        // completion but produces a degenerate theta null: the
        // penultimate/ultimate steps double past identity, and
        // the splitting routine
        // (`ThetaNullPoint::splitting_index_count`) sees either 0
        // or 10 vanishing `U_{i,j}(0)`
        // coordinates rather than the unique 1 that identifies
        // a genuine elliptic product. Downstream consumers
        // accept the malformed codomain silently, producing
        // wrong signing keys that fail only against KAT
        // vectors.
        //
        // Dispatch on `sui.e` vs `f`:
        //
        //   * `sui.e ≤ f − 2` → extra-torsion path. Pad the kernel by `f − sui.e − 2`
        //     doublings, leaving 2 spare torsion bits for the chain's
        //     penultimate/ultimate hadamard absorption. Calls [`Kernel::isogeny`] (=
        //     `isogeny_extra_torsion`).
        //
        //   * `sui.e ∈ {f − 1, f}` → no-extra-torsion path. Pad by `f − sui.e`
        //     doublings so the kernel is exactly `2^sui.e`. Calls
        //     [`Kernel::isogeny_no_extra_torsion`], which runs `sui.e − 2` main
        //     8-torsion steps then a dedicated 4-isogeny + 2-isogeny tail. Mirrors the
        //     C reference's `extra_torsion=false` mode at `dim2id2iso.c:1128`.
        // Always dispatch the outer chain to Mode B
        // (`isogeny_no_extra_torsion`), matching C ref's
        // `theta_chain_compute_and_eval_randomized(.., extra_torsion=false,
        // .)` at `dim2id2iso.c:1128`. C ref uses Mode B for the outer
        // chain regardless of available torsion; we previously
        // dispatched to Mode A when `sui.e ≤ f − 2` and got a different
        // codomain rep. With the `double_unnormalized` curve fix landed,
        // Mode B is byte-stable on FDI codomains and matches C ref.
        let no_extra_torsion = true;
        let scale = f.value().checked_sub(sui.e.value())?;
        // Outer-chain kernel-prep doubling, mirroring C ref's
        // `double_couple_point_iter(&ker.T1, TORSION_EVEN_POWER - exp,
        // ...)` (`dim2id2iso.c:1196-1198`). C ref calls `ec_dbl` on a
        // freshly returned `Fu_codomain` / `Fv_codomain` curve whose
        // `is_A24_computed_and_normalized` flag is `false`, so it
        // dispatches to the un-normalized `xDBL` (`ec.c:234`). To
        // produce the same `(X : Z)` byte-rep we use
        // [`ProjectiveXOnlyPoint::double_unnormalized`], which reads
        // the curve's projective `(A : C)` from
        // `EllipticProduct::from(&ThetaNullPoint)` (preserved
        // un-divided since the rewrite) and applies the `xDBL`
        // formula directly. Without this
        // un-normalized doubling, our `(X : Z)` drifts from C ref by
        // a `4C^scale` projective scalar; `lift_basis` absorbs the
        // drift on the P side (R normalizes to Z=1) but the
        // Okeya-Sakurai recovery on Q propagates it, breaking
        // byte-equality at the outer-chain `step=glue null` for any
        // KAT with `scale > 0`.
        for _ in 0..scale {
            kp_first = kp_first.double_unnormalized();
            kp_second = kp_second.double_unnormalized();
            kq_first = kq_first.double_unnormalized();
            kq_second = kq_second.double_unnormalized();
            kpmq_first = kpmq_first.double_unnormalized();
            kpmq_second = kpmq_second.double_unnormalized();
        }

        // Diagnostics: kernel order and curve-membership checks, plus
        // affine x-coordinates in the exact format the SQIsign C
        // reference prints (see `dim2id2iso.c:1036-1107`). Callers
        // with access to the C reference's stderr can diff the
        // `OUTER_KER` lines here against the fixture in
        // `tests/fixtures/cref_outer_ker_kat_vector_*.txt` to locate
        // which stage first diverges. The on-curve and exact-order
        // checks mirror the C reference's
        // `test_point_order_twof(..., exp)` assertions at
        // `dim2id2iso.c:1109-1110`.
        // Step 9: (2,2)-isogeny chain on E_u × E_v.
        let product = surfaces::EllipticProduct::new(e_u, e_v);
        let kernel = surfaces::Kernel::from_montgomery(
            product,
            (kp_first, kp_second),
            (kq_first, kq_second),
            (kpmq_first, kpmq_second),
        )?;

        // Chain exponent is `sui.e`. The C ref calls
        // `theta_chain_compute_and_eval_randomized(exp=sui.e, ...,
        // extra_torsion=false)` here
        // (`dim2id2iso.c:1128`) — full-length chain, no extra
        // torsion bits, kernel of order exactly 2^sui.e. The chain
        // produces a `(2^sui.e, 2^sui.e)`-isogeny of length sui.e.
        //
        // The earlier version of this call used
        // `isogeny_extra_torsion(chain_e = sui.e − 2)`, expecting
        // kernel 2^(chain_e + 2) = 2^sui.e with two spare
        // torsion bits for the internal double-and-add. That only
        // produced a `(2^(sui.e − 2), 2^(sui.e − 2))`-isogeny — four
        // bits too small — so the codomain landed at
        // `A / [2²](P, Q)` rather than `A / (P, Q)`, and the
        // splitting check at the end of the chain found zero
        // candidate indices.
        let chain_e = sui.e;
        let zero_v = ProjectiveXOnlyPoint::identity(&e_v);
        // Push `PmQ` through the chain alongside `P` and `Q`. The
        // chain's evaluator (`gluing_eval_point_special_case` →
        // `theta_isogeny_eval`) preserves the projective rep
        // semantics required by the next chain's `lift_basis`. See
        // the doc comment on this function for why we cannot recover
        // `PmQ` after the fact via `projective_difference`.
        let chain_pts = &[(phi_u_p, zero_v), (phi_u_q, zero_v), (phi_u_pmq, zero_v)];
        // The outer chain in keygen and signing must consume four bytes
        // of randomness at its splitting step to mirror the C reference's
        // `theta_chain_compute_and_eval_randomized` (`dim2id2iso.c:1261`),
        // which calls `splitting_compute(..., randomize=true)`. The four
        // bytes pick a uniform index in `[0, 6)` selecting one of six
        // `NORMALIZATION_TRANSFORMS` matrices that re-randomizes the
        // post-splitter projective representative. Without this, the
        // public output curve is a deterministic function of secret
        // kernel structure (a side-channel leak) and bytes diverge from
        // the C reference's published KAT vectors. FDI's internal chain
        // (`isogeny_extra_torsion`, called above) is the only chain
        // where C ref calls the non-randomized
        // `theta_chain_compute_and_eval`.
        //
        // The Mode A branch (else) does NOT yet match the C reference
        // bytewise: C ref always uses `extra_torsion=false` (Mode B)
        // for the outer chain regardless of available torsion. An
        // attempt to unify by routing both branches through Mode B
        // with `scale = f - sui.e` regressed the KAT pass count from
        // 35 to 33; Mode B in `isogeny_inner_no_extra_torsion`
        // appears to compute something subtly different from the
        // C reference's `_theta_chain_compute_impl(..., false, true)`
        // for the case where the kernel does NOT come from a chain
        // already short on torsion. TODO: instrument step-by-step to
        // localize the divergence.
        let (codomain, images) = if no_extra_torsion {
            kernel.isogeny_no_extra_torsion(chain_e, chain_pts, Some(rng))?
        } else {
            kernel.isogeny(chain_e, chain_pts, None)?
        };

        // Steps 10–13: Pick correct output curve via Weil-pairing
        // disambiguation.
        //
        // The (2,2)-chain on `E_u × E_v` splits as a product of two
        // curves; one is `E_I` (what we want), the other an auxiliary
        // unrelated to the response isogeny. Following the C reference
        // (`dim2id2iso.c:1148-1178`):
        //
        //   w_s     = e_{2^f}(canonical basis on E_s)
        //   w_chain = e_{2^f}(images on chosen codomain side)
        //   E_I = side where `w_chain == w_s^(d₁ · u² mod 2^f)`
        //
        // The exponent `d₁·u²` accounts for the composite isogeny
        // `phi_u (degree u) ∘ chain restriction (degree d₁·u)` from
        // E_s to the correct codomain side. Without this disambiguation
        // a 50/50 wrong-side selection produces a basis on the
        // auxiliary curve, and downstream `SplitAuxiliaryIsogeny`
        // reports `splitting_index_count() = 0` because its kernel is
        // malformed — not a (2,2)-isotropic subgroup of the intended
        // product.
        let s_idx_for_disamb = EXTREMAL_ORDERS
            .iter()
            .position(|o| o.q() == sui.factor1.order.q())?;
        let s_curve_idx = precomputed::torsion_basis::ExtremalCurve::try_from(s_idx_for_disamb)
            .expect("EXTREMAL_ORDERS length matches ExtremalCurve::ALL by compile-time assert");
        let (s_px, s_qx, s_pmq_x, s_a_coeff) = s_curve_idx.basis();
        let s_curve = if s_curve_idx == precomputed::torsion_basis::ExtremalCurve::E0 {
            Curve::E0
        } else {
            Curve::from(crate::curves::montgomery::Coefficient::from(s_a_coeff))
        };
        let s_p = ProjectiveXOnlyPoint::from_affine_x(s_px, &s_curve);
        let s_q = ProjectiveXOnlyPoint::from_affine_x(s_qx, &s_curve);
        let s_pmq = ProjectiveXOnlyPoint::from_affine_x(s_pmq_x, &s_curve);
        // Tate-based Weil pairing expects P+Q as third arg; compute
        // it via differential addition from (P, Q, P-Q).
        let s_ppq = s_p.differential_add(&s_q, &s_pmq);
        let w_s = crate::curves::pairing::weil_pairing(&s_p, &s_q, &s_ppq, f);

        // Expected: w_s^(d₁ · u² mod 2^f).
        let d1_big = BigInt::<4>::from_sign_and_limbs(0, *d1.limbs());
        let u_sq = sui.u.ct_mul(&sui.u);
        let exp_disamb = d1_big.ct_mul(&u_sq).ct_mod(&modulus);
        let expected = w_s.pow_scalar(&Scalar::from(exp_disamb));

        // Compute Weil pairing on codomain.E1 side using propagated
        // PmQ; convert to P+Q via differential addition.
        let ppq_e1 = images[0].0.differential_add(&images[1].0, &images[2].0);
        let w1 = crate::curves::pairing::weil_pairing(&images[0].0, &images[1].0, &ppq_e1, f);

        let matched_e1 = w1 == expected;
        let (e_i, p_chain, q_chain, pmq_chain) = if matched_e1 {
            (codomain.E1, images[0].0, images[1].0, images[2].0)
        } else {
            (codomain.E2, images[0].1, images[1].1, images[2].1)
        };

        // Step 14: [P_I, Q_I]^T ← (1/(u·d₁)) M_{β₁} [P_I, Q_I]^T
        let s_index = EXTREMAL_ORDERS
            .iter()
            .position(|o| o.q() == sui.factor1.order.q())?;
        let endo_s = EndomorphismAction {
            order: sui.factor1.order.order(),
            generators: [
                ENDOMORPHISM_MATRICES[s_index][3],
                ENDOMORPHISM_MATRICES[s_index][4],
                ENDOMORPHISM_MATRICES[s_index][5],
            ],
        };
        let m_beta1 = endo_s.apply(&sui.factor1.beta, f)?;

        // scalar = 1/(u · d₁) mod 2^f
        let ud1 = sui
            .u
            .ct_mul(&BigInt::<4>::from_sign_and_limbs(0, *d1.limbs()));
        let ud1_inv = ud1.invert_mod(&modulus)?;

        // Apply `(1/(u·d₁)) M_{β₁}` to the basis `(P, Q, PmQ)` using
        // biscalar multiplication for all three points so the
        // resulting `PmQ` is a propagated projective rep — never the
        // sqrt-branch result of `projective_difference(P', Q')`.
        //
        // # Mathematical invariant (verified)
        //
        // The output basis `(P_I, Q_I, PmQ_I)` satisfies
        //   `e_{2^f}(P_I, Q_I) = e_{2^f}(canonical basis on E_s)^nrd(self)`
        // where `self` is the input ideal. Derivation:
        //   chain Weil = e_E_s^(d₁·u²)  (from disambiguation step)
        //   M_{β₁} det ≡ nrd(β₁) = d₁·nrd(self) mod 2^f
        //   scaling by 1/(u·d₁) multiplies det by 1/(u·d₁)²
        //   final det = nrd(self)/(u²·d₁)
        //   total Weil = e_E_s^(d₁·u² · nrd(self)/(u²·d₁)) = e_E_s^nrd(self)
        //
        // This invariant matters for downstream `SplitAuxiliaryIsogeny`:
        // the response-phase kernel's isotropy condition
        // `e_E_com · q⁻² · e_E_aux' = 1` only holds when
        // `nrd(I_inter) = q · nrd(I_com) · nrd(I_aux)` literally —
        // i.e., when `self` here is the input ideal's *original* norm,
        // not a reduced equivalent.
        let chain_basis = TorsionBasis::from_propagated(p_chain, q_chain, pmq_chain);
        let (p_i, q_i, pmq_i) = m_beta1.apply_scaled_basis(&ud1_inv, &chain_basis, f);

        Some((e_i, p_i, q_i, pmq_i))
    }
}

// TODO: Complete KernelDecomposedToIdeal (Algorithm 3.17, partially
// implemented above as kernel_to_ideal).

#[cfg(test)]
mod tests;
