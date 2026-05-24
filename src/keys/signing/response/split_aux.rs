//! Sign-side kernel for [SplitAuxiliaryIsogeny][Alg. 4.5] — the
//! `(2,2)`-isogeny `φ_aux × φ^odd_rsp` on the elliptic product
//! `E_com × E'_aux`.
//!
//! The kernel is the primary subject of the algorithm: it
//! uniquely determines the isogeny. Construction is split from
//! execution to match the project's [`LeftIdeal::to_isogeny`] /
//! [`surfaces::Kernel::isogeny`] convention:
//! [`SplitAuxiliaryKernel::from_bases`] builds the kernel from
//! full-torsion bases on the two input curves;
//! [`SplitAuxiliaryKernel::isogeny`] runs the chain and returns the
//! codomain `F_1 × F_2` with the propagated commitment-side basis
//! on each component.
//!
//! [Alg. 4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.5
//! [`LeftIdeal::to_isogeny`]: crate::quaternions::lattice::LeftIdeal::to_isogeny

use rand_core::CryptoRngCore;

use crate::{
    curves::{
        TorsionBasis, TorsionExponent,
        montgomery::{Curve, ProjectiveXOnlyPoint},
        scalar::Scalar,
    },
    params::TORSION_EVEN_POWER,
    quaternions::bigint::BigInt,
    surfaces,
};

/// The `(2,2)`-kernel of [SplitAuxiliaryIsogeny][Alg. 4.5].
///
/// Of order `2^(e'_rsp + 2)` on the elliptic product
/// `E_com × E'_aux`. Constructed via [`Self::from_bases`] from
/// full-torsion bases on the two input curves; run via
/// [`Self::isogeny`].
///
/// [Alg. 4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.5
pub(crate) struct SplitAuxiliaryKernel {
    inner: surfaces::Kernel,
    /// Commitment-side basis (reduced to order `2^(e'_rsp + 2 + r_rsp)`)
    /// paired with the zero point on `E'_aux`, pushed through the
    /// chain so the codomain basis on the challenge side carries a
    /// propagated `PmQ`.
    pushed: [surfaces::ProductPoint; 3],
    e_prime: TorsionExponent,
}

/// The codomain `F_1 × F_2` returned by
/// [`SplitAuxiliaryKernel::isogeny`].
///
/// - `challenge_basis` is the propagated image basis on `F_1 = E_chl`, the
///   codomain of the odd response isogeny `φ^odd_rsp : E_com → E_chl`. `F_1`
///   itself is accessible as `challenge_basis.P.curve()`.
/// - `auxiliary_curve` and `auxiliary_basis` are the codomain `F_2 = E_aux` and
///   the propagated image basis under the auxiliary isogeny `φ_aux : E'_aux →
///   E_aux`. The curve is kept as a separate field because downstream consumers
///   (`to_hint`) take it separately from the basis.
///
/// Each propagated basis is
/// `(φ((P, P − Q, Q)))` of the commitment-side input basis.
pub(crate) struct SplitCodomain {
    /// `φ^odd_rsp((P, P − Q, Q))` — the propagated image basis on
    /// `F_1 = E_chl`.
    pub challenge_basis: TorsionBasis,
    /// `F_2 = E_aux`, the auxiliary-side codomain.
    pub auxiliary_curve: Curve,
    /// `φ_aux((P, P − Q, Q))` — the propagated image basis on
    /// `F_2`.
    pub auxiliary_basis: TorsionBasis,
}

impl SplitAuxiliaryKernel {
    /// Constructs the kernel from full-torsion bases on `E_com` and
    /// `E'_aux`, plus the response-isogeny degree inverse `q_rsp`
    /// and the chain exponents.
    ///
    /// Reduces both bases to order `2^(e'_rsp + r_rsp + 2)`, scales
    /// the auxiliary-side basis by
    /// `q_rsp^{-1} mod 2^(e'_rsp + r_rsp + 2)` (so the kernel
    /// aligns with the response-isogeny degree), then doubles by
    /// `r_rsp` to land at the chain's required `2^(e'_rsp + 2)`-
    /// torsion.
    ///
    /// The `PmQ` slot in each input basis is the propagated `P − Q`
    /// projective rep, NOT a value freshly recomputed via
    /// `projective_difference`. The propagated value MUST come from
    /// the same chain that produced `(P, Q)` — typically
    /// [`LeftIdeal::to_isogeny`]'s fourth return value. A
    /// `projective_difference`-recomputed `PmQ` picks a sqrt branch
    /// that is not aligned with the chain's evaluation history; the
    /// resulting kernel produces a terminal theta null with
    /// `splitting_index_count() = 0`.
    ///
    /// # Divergences
    ///
    /// The spec ([Alg. 4.5] lines 1–4) describes a kernel with
    /// torsion `2^(e' + 2)` directly (via `[2^(f-e'-2)]` reductions
    /// on each side), but does not spell out (a) the basis-reduction
    /// step that the C reference performs before invoking the chain
    /// or (b) that downstream consumers of `IdealToIsogeny`'s output
    /// require a `PmQ` whose projective rep is propagated alongside
    /// `P` and `Q`, not recomputed via `projective_difference`.
    /// Both are required for interoperability with the C reference's
    /// KAT vectors.
    ///
    /// [Alg. 4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.5
    /// [`LeftIdeal::to_isogeny`]: crate::quaternions::lattice::LeftIdeal::to_isogeny
    pub(crate) fn from_bases(
        commitment: TorsionBasis,
        auxiliary: TorsionBasis,
        q_rsp: BigInt<4>,
        e_prime: TorsionExponent,
        r_rsp: TorsionExponent,
    ) -> Option<Self> {
        let f = TORSION_EVEN_POWER;
        let e_prime_val = e_prime.value();
        let r_val = r_rsp.value();
        let e1 = *commitment.P.curve();
        let e2 = *auxiliary.P.curve();
        let (p1, q1, pmq1) = (commitment.P, commitment.Q, commitment.PmQ);
        let (p2, q2, pmq2) = (auxiliary.P, auxiliary.Q, auxiliary.PmQ);

        // Kernel construction follows the C reference's
        // `compute_dim2_isogeny_challenge` (sign.c:578-590, 240-256).
        // The C ref reduces the bases on both curves to order
        // `2^(reduced_order) = 2^(e_prime + 2 + r)` BEFORE forming
        // the kernel, then forms `(P1_red, q_inv · P2_red)` and
        // doubles by `r`. After the `r`-doubling the kernel has
        // order `2^(e_prime + 2)`, exactly what the (2,2)-chain of
        // length `e_prime` requires (the chain's gluing step needs
        // 8-torsion at the bottom of the strategy, and the chain
        // length `e_prime` implies a kernel of order
        // `2^(e_prime + 2)`).
        //
        // Earlier attempts that fed raw `2^f`-torsion bases into
        // the kernel left `2^(f - r - e_prime - 2)` extra torsion
        // above the chain's expected order. Our
        // `surfaces::Kernel::isogeny` always computes exactly `e`
        // doubling-down steps before gluing, so any extra torsion
        // shifts the strategy bottom away from the gluing's
        // required 8-torsion level —
        // `ThetaNullPoint::splitting_index_count` returns 0 and
        // the chain returns `None`.
        //
        // The kernel `PmQ` projective reps come from the inputs'
        // `PmQ` slots, scaled alongside `P` and `Q` to keep the
        // projective history aligned. Recomputing via
        // `projective_difference` on post-scaling kernel points
        // picks a sqrt branch that the chain's `lift_basis` then
        // rejects, again surfacing as
        // `splitting_index_count() = 0`.

        // C ref: reduced_order = pow_dim2_deg_resp + HD_extra_torsion +
        //                        sig->two_resp_length
        //                      = e_prime + 2 + r
        let reduced_order = match e_prime_val
            .checked_add(2)
            .and_then(|x| x.checked_add(r_val))
        {
            Some(o) if o <= f => o,
            _ => {
                return None;
            }
        };
        let reduce_steps = f - reduced_order;

        // Reduce all six basis points (P, Q, PmQ on each curve) from
        // order `2^f` to order
        // `2^reduced_order = 2^(e_prime + 2 + r)`. Matches the C
        // reference's `ec_dbl_iter_basis` which doubles all of
        // `(B.P, B.Q, B.PmQ)` together to keep the projective rep
        // history consistent.
        let mut p1_red = p1;
        let mut q1_red = q1;
        let mut pmq1_red = pmq1;
        let mut p2_red = p2;
        let mut q2_red = q2;
        let mut pmq2_red = pmq2;
        for _ in 0..reduce_steps {
            p1_red = p1_red.double();
            q1_red = q1_red.double();
            pmq1_red = pmq1_red.double();
            p2_red = p2_red.double();
            q2_red = q2_red.double();
            pmq2_red = pmq2_red.double();
        }

        // q_inv ← q^{-1} (mod 2^reduced_order). C ref uses
        // `degree_resp_inv = degree_odd_resp^{-1} mod 2^(reduced_order)`
        // (computed in compute_random_aux_norm_and_helpers).
        let q_scalar = Scalar::from_limbs(*q_rsp.as_limbs());
        let q_inv = q_scalar.inv_mod2k(reduced_order)?;

        // Kernel:
        //   T1   = (P1_red,     q_inv · P2_red)
        //   T2   = (Q1_red,     q_inv · Q2_red)
        //   T1m2 = (P1mQ1_red,  q_inv · P2mQ2_red)   [propagated PmQ]
        // Then double all three pairs by `r`. After the `r`-doubling
        // the kernel has order exactly `2^(e_prime + 2)`. The PmQ
        // points are scaled by the same `q_inv · 2^r` as P and Q,
        // so the projective rep stays aligned with the chain's
        // evaluator.
        let p2_qinv = &q_inv * &p2_red;
        let q2_qinv = &q_inv * &q2_red;
        let pmq2_qinv = &q_inv * &pmq2_red;

        let two_r_scalar = Scalar::from_limbs(*(BigInt::<4>::ONE << r_val).as_limbs());
        let p1_ker = &two_r_scalar * &p1_red;
        let q1_ker = &two_r_scalar * &q1_red;
        let pmq1_ker = &two_r_scalar * &pmq1_red;
        let p2_ker = &two_r_scalar * &p2_qinv;
        let q2_ker = &two_r_scalar * &q2_qinv;
        let pmq2_ker = &two_r_scalar * &pmq2_qinv;

        let product = surfaces::EllipticProduct::new(e1, e2);
        let inner = surfaces::Kernel::from_montgomery(
            product,
            (p1_ker, p2_ker),
            (q1_ker, q2_ker),
            (pmq1_ker, pmq2_ker),
        )?;

        // Pushed points: the REDUCED bases (order
        // `2^(e_prime + 2 + r)`) with zero on E'_aux, matching
        // sign.c:262-269. After the chain these become the
        // canonical bases on the codomain components. The third
        // pushed point (PmQ) ensures the codomain basis carries a
        // propagated `P − Q` — recomputing it via
        // `projective_difference` downstream picks a sqrt branch
        // that breaks `lift_basis` y-recovery.
        let zero_e2 = ProjectiveXOnlyPoint::identity(&e2);
        let pushed = [(p1_red, zero_e2), (q1_red, zero_e2), (pmq1_red, zero_e2)];

        Some(Self {
            inner,
            pushed,
            e_prime,
        })
    }

    /// Runs the `(2,2)`-chain to compute `φ_aux × φ^odd_rsp` and
    /// returns the labeled codomain pair.
    ///
    /// C ref dispatches the chain through
    /// `theta_chain_compute_and_eval_randomized` (`sign.c:274`),
    /// the `randomize = true` variant of `splitting_compute`. The
    /// randomized splitter consumes 4 RNG bytes to pick a level-2
    /// normalization matrix index in `[0, 6)`; without
    /// `Some(rng)` here our codomain lands on a different (but
    /// isomorphic) Montgomery model than C ref's, basis points
    /// pushed through diverge, and downstream `compute_even_response`
    /// produces a different `j(E_chl_3)`.
    ///
    /// # Codomain labelling
    ///
    /// The `(2,2)`-chain's codomain decomposes as
    /// `codomain.E1 × codomain.E2` per the Kani matrix structure.
    /// The C reference labels these `Eaux2_Echall2.E1 = E_aux_2`
    /// and `Eaux2_Echall2.E2 = E_chall_2` based on which input went
    /// where during kernel construction — but the SQIsign spec
    /// invariant requires `j(E_chl_2) = j(challenge_iso(E_pk))`,
    /// which matches the matching j-invariant for the downstream
    /// challenge isogeny. Empirically our `surfaces::Kernel::isogeny`
    /// returns `codomain.E1` as the "challenge side" and
    /// `codomain.E2` as the "auxiliary side" — opposite of the C
    /// reference's E1/E2 labelling (the chain implementations
    /// differ in which factor they label "first"). [`SplitCodomain`]
    /// swaps the labels back so callers see `challenge_basis` on
    /// the curve with j-invariant matching `challenge_iso(E_pk)`,
    /// and `auxiliary_curve` / `auxiliary_basis` on the auxiliary
    /// side.
    pub(crate) fn isogeny<R: CryptoRngCore>(self, rng: &mut R) -> Option<SplitCodomain> {
        let (codomain, images) = self.inner.isogeny(self.e_prime, &self.pushed, Some(rng))?;
        let auxiliary_curve = codomain.E2;
        let p_chl = images[0].0;
        let q_chl = images[1].0;
        let pmq_chl = images[2].0;
        let p_aux = images[0].1;
        let q_aux = images[1].1;
        let pmq_aux = images[2].1;
        Some(SplitCodomain {
            challenge_basis: TorsionBasis::from_propagated(p_chl, pmq_chl, q_chl),
            auxiliary_curve,
            auxiliary_basis: TorsionBasis::from_propagated(p_aux, pmq_aux, q_aux),
        })
    }
}
