//! Step-level (2,2)-isogeny primitives: codomain and evaluator pairs.
//!
//! Each `codomain_*` / `eval_*` pair computes one chain step from
//! kernel-data inputs. Pairs come in three variants per kernel-data
//! type:
//!
//! - **Standard** (`codomain_X` + `eval`): applies the final Hadamard
//!   transform; output codomain is in theta form. Used for chain-interior
//!   steps.
//! - **`_no_hadamard`** (`codomain_X_no_hadamard` + `eval_no_outer_hadamard`):
//!   skips the final Hadamard; output is in dual form. Used for the penultimate
//!   step when a splitting follows.
//! - **`_ultimate`** (`codomain_X_ultimate` + `eval_ultimate`): applies only
//!   the outer Hadamard; compensates for the prior `_no_hadamard` step. Used
//!   for the ultimate step before a splitting.
//!
//! The `surfaces::Kernel::isogeny_inner` chain machine picks the
//! variant per step based on its position in the chain.

use crate::surfaces::{DualThetaNullPoint, Jacobian, JacobianPoint, ThetaNullPoint, hadamard4};

pub(crate) fn codomain_8torsion(
    T1: &JacobianPoint,
    T2: &JacobianPoint,
) -> (DualThetaNullPoint, Jacobian) {
    let hs1 = T1.squared().hadamard();
    let hs2 = T2.squared().hadamard();

    let xawb = &hs1.X * &hs2.Y;
    let zaxb = &hs2.X * &hs1.Y;

    let alpha = &hs2.X * &xawb;
    let beta = &hs2.Y * &zaxb;
    let gamma = &hs2.Z * &xawb;
    let delta = &hs2.W * &zaxb;

    let zgwd = &hs2.Z * &hs2.W;
    let alpha_inv = &hs1.Y * &zgwd;
    let beta_inv = &hs1.X * &zgwd;
    let gamma_inv = delta;
    let delta_inv = gamma;

    let dual = DualThetaNullPoint {
        alpha,
        beta,
        gamma,
        delta,
        alpha_inv,
        beta_inv,
        gamma_inv,
        delta_inv,
    };
    let null_B = hadamard_null(&dual);
    (dual, Jacobian::new(null_B))
}

/// Codomain from 4-torsion (Algorithm 8.32).
pub(super) fn codomain_4torsion(
    T1: &JacobianPoint,
    domain: &Jacobian,
) -> (DualThetaNullPoint, Jacobian) {
    // Line 1: (xαβ, _, xγδ, _) ← H ∘ S(T₁')
    let hs = T1.squared().hadamard();

    // Line 2: (α², β², γ², δ²) ← H ∘ S(0_A)
    let (a2, b2, g2, d2) = hadamard4(
        &domain.null.a.square(),
        &domain.null.b.square(),
        &domain.null.c.square(),
        &domain.null.d.square(),
    );

    // Lines 3–4: square roots.
    let ab = (&a2 * &b2).sqrt();
    let ag = (&a2 * &g2).sqrt();

    // Lines 5–8: recover (α, β, γ, δ).
    let beta = &(&ab * &ag) * &hs.Z;
    let delta_inv = &beta * &hs.X;
    let beta_mul = &beta * &hs.X;
    let xgd_ab_a2 = &(&hs.Z * &ab) * &a2;
    let _delta = &xgd_ab_a2 * &(&ab * &a2);
    let alpha = &(&hs.X * &ab) * &a2;
    let gamma = &alpha * &g2;
    let delta_final = &alpha * &d2;

    // Projective inverses.
    let alpha_inv = &hs.X * &d2;
    let beta_inv = &alpha * &b2;
    let gamma_inv_val = &delta_inv * &b2;

    let dual = DualThetaNullPoint {
        alpha,
        beta: beta_mul,
        gamma,
        delta: delta_final,
        alpha_inv,
        beta_inv,
        gamma_inv: gamma_inv_val,
        delta_inv,
    };
    let null_B = hadamard_null(&dual);
    (dual, Jacobian::new(null_B))
}

/// Codomain from null point only (Algorithm 8.33).
pub(super) fn codomain_from_null(domain: &Jacobian) -> (DualThetaNullPoint, Jacobian) {
    let (a2, b2, g2, d2) = hadamard4(
        &domain.null.a.square(),
        &domain.null.b.square(),
        &domain.null.c.square(),
        &domain.null.d.square(),
    );

    let alpha = a2;
    let beta = (&a2 * &b2).sqrt();
    let gamma = (&a2 * &g2).sqrt();
    let delta = (&a2 * &d2).sqrt();

    let ab = &alpha * &beta;
    let gd = &gamma * &delta;
    let alpha_inv = &ab * &d2;
    let beta_inv = &ab * &g2;
    let gamma_inv = &gd * &b2;
    let delta_inv = &gd * &a2;

    let dual = DualThetaNullPoint {
        alpha,
        beta,
        gamma,
        delta,
        alpha_inv,
        beta_inv,
        gamma_inv,
        delta_inv,
    };
    let null_B = hadamard_null(&dual);
    (dual, Jacobian::new(null_B))
}

/// Codomain from 8-torsion WITHOUT final Hadamard (dual form).
///
/// Same as [`codomain_8torsion`] but omits the Hadamard transform on
/// the codomain null point, corresponding to `hadamard_bool_2=0` in
/// the C reference. Used for the penultimate and ultimate chain steps
/// so the splitting step receives the codomain in dual form.
pub(crate) fn codomain_8torsion_no_hadamard(
    T1: &JacobianPoint,
    T2: &JacobianPoint,
) -> (DualThetaNullPoint, Jacobian) {
    let hs1 = T1.squared().hadamard();
    let hs2 = T2.squared().hadamard();

    let xawb = &hs1.X * &hs2.Y;
    let zaxb = &hs2.X * &hs1.Y;

    let alpha = &hs2.X * &xawb;
    let beta = &hs2.Y * &zaxb;
    let gamma = &hs2.Z * &xawb;
    let delta = &hs2.W * &zaxb;

    let zgwd = &hs2.Z * &hs2.W;
    let alpha_inv = &hs1.Y * &zgwd;
    let beta_inv = &hs1.X * &zgwd;
    let gamma_inv = delta;
    let delta_inv = gamma;

    let dual = DualThetaNullPoint {
        alpha,
        beta,
        gamma,
        delta,
        alpha_inv,
        beta_inv,
        gamma_inv,
        delta_inv,
    };
    // NO hadamard_null here — codomain stays in dual form.
    let null = ThetaNullPoint::new(dual.alpha, dual.beta, dual.gamma, dual.delta);
    (dual, Jacobian::new(null))
}

/// Evaluate: normal interior step (`hadamard_bool_1=0, hadamard_bool_2=1`).
///
/// The C reference's `theta_isogeny_eval` with bool_1=0, bool_2=1
/// computes: `H(precomp · to_squared_theta(P))` where
/// `to_squared_theta(P) = H(P²)`.
///
/// However, the `precomp` (alpha_inv etc.) stored in our
/// `DualThetaNullPoint` already incorporates the coordinate
/// relationships such that the eval formula is simply
/// `H(precomp · P²)` — the inner Hadamard is absorbed into how
/// the precomputation relates to the codomain.
pub(crate) fn eval(
    P: &JacobianPoint,
    dual: &DualThetaNullPoint,
    codomain: &Jacobian,
) -> JacobianPoint {
    let t = P
        .squared()
        .hadamard()
        .scale(
            &dual.alpha_inv,
            &dual.beta_inv,
            &dual.gamma_inv,
            &dual.delta_inv,
        )
        .hadamard();
    JacobianPoint::new(t.X, t.Y, t.Z, t.W, codomain.clone())
}

/// Evaluate: penultimate step (`hadamard_bool_1=0, hadamard_bool_2=0`).
///
/// Formula: `precomp · H(P²)` — no outer Hadamard.
pub(crate) fn eval_no_outer_hadamard(
    P: &JacobianPoint,
    dual: &DualThetaNullPoint,
    codomain: &Jacobian,
) -> JacobianPoint {
    let t = P.squared().hadamard().scale(
        &dual.alpha_inv,
        &dual.beta_inv,
        &dual.gamma_inv,
        &dual.delta_inv,
    );
    JacobianPoint::new(t.X, t.Y, t.Z, t.W, codomain.clone())
}

/// Evaluate: ultimate step (`hadamard_bool_1=1, hadamard_bool_2=0`).
///
/// Formula: `precomp · H(H(P)²)` — extra Hadamard on input, no outer.
pub(crate) fn eval_ultimate(
    P: &JacobianPoint,
    dual: &DualThetaNullPoint,
    codomain: &Jacobian,
) -> JacobianPoint {
    let t = P.hadamard().squared().hadamard().scale(
        &dual.alpha_inv,
        &dual.beta_inv,
        &dual.gamma_inv,
        &dual.delta_inv,
    );
    JacobianPoint::new(t.X, t.Y, t.Z, t.W, codomain.clone())
}

/// Codomain from 8-torsion: ultimate step (`hadamard_bool_1=1,
/// hadamard_bool_2=0`).
///
/// Same cross-product formulas as the normal 8-torsion codomain, but
/// applies Hadamard to each kernel point BEFORE `to_squared_theta`,
/// and omits the final Hadamard on the codomain. This compensates for
/// the penultimate step having produced a dual-form codomain.
///
/// C reference: `theta_isogeny_compute` with `hadamard_bool_1=1,
/// hadamard_bool_2=0` (theta_isogenies.c:636-644, 692-694).
pub(crate) fn codomain_8torsion_ultimate(
    T1: &JacobianPoint,
    T2: &JacobianPoint,
) -> (DualThetaNullPoint, Jacobian) {
    // bool_1=1: Hadamard before to_squared_theta
    let hs1 = T1.hadamard().squared().hadamard();
    let hs2 = T2.hadamard().squared().hadamard();

    let xawb = &hs1.X * &hs2.Y;
    let zaxb = &hs2.X * &hs1.Y;

    let alpha = &hs2.X * &xawb;
    let beta = &hs2.Y * &zaxb;
    let gamma = &hs2.Z * &xawb;
    let delta = &hs2.W * &zaxb;

    let zgwd = &hs2.Z * &hs2.W;
    let alpha_inv = &hs1.Y * &zgwd;
    let beta_inv = &hs1.X * &zgwd;
    let gamma_inv = delta;
    let delta_inv = gamma;

    let dual = DualThetaNullPoint {
        alpha,
        beta,
        gamma,
        delta,
        alpha_inv,
        beta_inv,
        gamma_inv,
        delta_inv,
    };
    // bool_2=0: NO final Hadamard — codomain stays in dual form.
    let null = ThetaNullPoint::new(dual.alpha, dual.beta, dual.gamma, dual.delta);
    (dual, Jacobian::new(null))
}

/// Codomain from 4-torsion: penultimate step (`hadamard_bool_1=0,
/// hadamard_bool_2=0`).
///
/// Same as [`codomain_4torsion`] (Algorithm 8.32) but omits the final
/// Hadamard on the codomain null point. The codomain stays in dual
/// form, matching the C reference's `theta_isogeny_compute_4(..., 0, 0)`
/// call at `theta_isogenies.c:1258` — the dedicated penultimate step
/// in an `extra_torsion=false` chain.
///
/// Pair with [`eval_no_outer_hadamard`] for the matching evaluator.
pub(crate) fn codomain_4torsion_no_hadamard(
    T1: &JacobianPoint,
    domain: &Jacobian,
) -> (DualThetaNullPoint, Jacobian) {
    // TT1 = H(S(T1_4)). For a dual-form 4-torsion point, only the X
    // and Z components are nonzero (Y = W = 0), but we don't depend
    // on that explicitly here.
    let tt1 = T1.squared().hadamard();
    // tt1.X, tt1.Y, tt1.Z, tt1.W ↔ C ref's TT1.x, TT1.y, TT1.z, TT1.t.

    // TT2 = H(S(domain.null)). Maps to C ref's TT2.{x,y,z,t}.
    let (tt2_x, tt2_y, tt2_z, tt2_t) = hadamard4(
        &domain.null.a.square(),
        &domain.null.b.square(),
        &domain.null.c.square(),
        &domain.null.d.square(),
    );

    let sqaabb = (&tt2_x * &tt2_y).sqrt();
    let sqaacc = (&tt2_x * &tt2_z).sqrt();

    // Codomain null point — direct transcription of C ref's
    // `theta_isogeny_compute_4(0, 0)` body
    // (`theta_isogenies.c:789–802`):
    //
    //   null.x = TT1.x · TT2.x · sqaacc
    //   null.y = TT1.x · sqaabb · sqaacc
    //   null.z = TT1.x · TT2.x · TT2.z
    //   null.t = TT1.z · sqaabb · TT2.x
    let null_x = &(&tt1.X * &tt2_x) * &sqaacc;
    let null_y = &(&tt1.X * &sqaabb) * &sqaacc;
    let null_z = &(&tt1.X * &tt2_x) * &tt2_z;
    let null_t = &(&tt1.Z * &sqaabb) * &tt2_x;

    // Precomputation for evaluation
    // (`theta_isogenies.c:790, 804–810`):
    //
    //   precomp.x = TT1.x · TT2.t · TT2.z · TT2.y
    //   precomp.y = TT1.x · TT2.t · TT2.z · sqaabb
    //   precomp.z = TT1.x · TT2.t · TT2.y · sqaacc
    //   precomp.t = sqaabb · sqaacc · TT1.z · TT2.y
    let xt = &tt1.X * &tt2_t;
    let xtz = &xt * &tt2_z;
    let xty = &xt * &tt2_y;
    let precomp_x = &xtz * &tt2_y;
    let precomp_y = &xtz * &sqaabb;
    let precomp_z = &xty * &sqaacc;
    let sab_sac_z = &(&sqaabb * &sqaacc) * &tt1.Z;
    let precomp_t = &sab_sac_z * &tt2_y;

    let dual = DualThetaNullPoint {
        alpha: null_x,
        beta: null_y,
        gamma: null_z,
        delta: null_t,
        alpha_inv: precomp_x,
        beta_inv: precomp_y,
        gamma_inv: precomp_z,
        delta_inv: precomp_t,
    };
    // hadamard_bool_2 = 0 → codomain stays in dual form (no final
    // Hadamard on the null point).
    let null = ThetaNullPoint::new(dual.alpha, dual.beta, dual.gamma, dual.delta);
    (dual, Jacobian::new(null))
}

/// Codomain from 2-torsion (null only): ultimate step
/// (`hadamard_bool_1=1, hadamard_bool_2=0`).
///
/// Same shape as [`codomain_from_null`] (Algorithm 8.33) but applies
/// Hadamard to the domain null before squaring (compensating for the
/// previous step's dual-form output) and omits the final Hadamard on
/// the codomain.
///
/// C reference: `theta_isogeny_compute_2(..., 1, 0)` at
/// `theta_isogenies.c:1266` — the dedicated ultimate step in an
/// `extra_torsion=false` chain.
///
/// Pair with [`eval_ultimate`] for the matching evaluator.
pub(crate) fn codomain_2torsion_ultimate(domain: &Jacobian) -> (DualThetaNullPoint, Jacobian) {
    // bool_1 = 1: TT2 = H(S(H(domain.null))) = to_squared_theta of
    // hadamarded null. Maps to C ref's TT2.{x,y,z,t} = AA, BB, CC, DD.
    let (na, nb, nc, nd) = hadamard4(
        &domain.null.a,
        &domain.null.b,
        &domain.null.c,
        &domain.null.d,
    );
    let (tt2_x, tt2_y, tt2_z, tt2_t) =
        hadamard4(&na.square(), &nb.square(), &nc.square(), &nd.square());

    // Codomain null — direct from C ref's
    // `theta_isogeny_compute_2(1, 0)` body
    // (`theta_isogenies.c:860–867`):
    //
    //   null.x = TT2.x                        // AA
    //   null.y = sqrt(TT2.x · TT2.y)          // sqrt(AA·BB)
    //   null.z = sqrt(TT2.x · TT2.z)          // sqrt(AA·CC)
    //   null.t = sqrt(TT2.x · TT2.t)          // sqrt(AA·DD)
    let alpha = tt2_x;
    let beta = (&tt2_x * &tt2_y).sqrt();
    let gamma = (&tt2_x * &tt2_z).sqrt();
    let delta = (&tt2_x * &tt2_t).sqrt();

    // Precomputation (`theta_isogenies.c:869–877`):
    //
    //   precomp.x = TT2.y · TT2.z · TT2.t          // BB·CC·DD
    //   precomp.y = TT2.z · TT2.t · null.y         // CC·DD·sqrt(AA·BB)
    //   precomp.z = TT2.y · TT2.t · null.z         // BB·DD·sqrt(AA·CC)
    //   precomp.t = TT2.y · TT2.z · null.t         // BB·CC·sqrt(AA·DD)
    let zt = &tt2_z * &tt2_t;
    let yt = &tt2_y * &tt2_t;
    let yz = &tt2_y * &tt2_z;

    let alpha_inv = &yz * &tt2_t; // BB·CC·DD
    let beta_inv = &zt * &beta; // CC·DD·sqrt(AA·BB)
    let gamma_inv = &yt * &gamma; // BB·DD·sqrt(AA·CC)
    let delta_inv = &yz * &delta; // BB·CC·sqrt(AA·DD)

    let dual = DualThetaNullPoint {
        alpha,
        beta,
        gamma,
        delta,
        alpha_inv,
        beta_inv,
        gamma_inv,
        delta_inv,
    };
    // bool_2=0: codomain stays in dual form (no final Hadamard).
    let null = ThetaNullPoint::new(dual.alpha, dual.beta, dual.gamma, dual.delta);
    (dual, Jacobian::new(null))
}

/// Codomain theta null point from dual via Hadamard.
fn hadamard_null(dual: &DualThetaNullPoint) -> ThetaNullPoint {
    let (a, b, c, d) = hadamard4(&dual.alpha, &dual.beta, &dual.gamma, &dual.delta);
    ThetaNullPoint::new(a, b, c, d)
}
