//! Per-step (2,2)-isogeny kernels.
//!
//! Each kernel type holds the kernel data for one chain step,
//! parameterized by torsion level (8, 4, or 2) and chain position
//! (interior, penultimate, ultimate). Each has an
//! [`isogeny`][EightTorsionStepKernel::isogeny] method that consumes the
//! kernel and returns a [`StepIsogeny`] — paralleling
//! [`Curve::isomorphism`](crate::curves::montgomery::Curve::isomorphism)
//! at the per-step layer. The returned `StepIsogeny` exposes
//! [`eval`][StepIsogeny::eval] to push points through (same agency
//! as [`Isomorphism::eval`](crate::curves::montgomery::Isomorphism::eval)).
//!
//! The Hadamard-flag variant (`Standard` / `NoOuter` / `Ultimate`)
//! is private to [`StepIsogeny`]: it's set at step construction
//! and threads through the `eval` formula, so the three former
//! free `eval` / `eval_no_outer_hadamard` / `eval_ultimate` shapes
//! collapse into a single [`StepIsogeny::eval`].

use crate::surfaces::{DualThetaNullPoint, Jacobian, JacobianPoint, ThetaNullPoint, hadamard4};

/// Which Hadamard transforms fire in the step's codomain + eval
/// formulas. Selected by chain position; private to this module.
///
/// Matches the C reference's `(hadamard_bool_1, hadamard_bool_2)`
/// flag pair.
#[derive(Copy, Clone, Debug)]
enum HadamardVariant {
    /// `(0, 1)` — chain-interior step.
    Standard,
    /// `(0, 0)` — penultimate step before a splitting.
    NoOuter,
    /// `(1, 0)` — ultimate step before a splitting.
    Ultimate,
}

/// Kernel of an 8-torsion (2,2)-isogeny step.
///
/// Defined by a pair of order-8 points `(T₁, T₂)` on the domain
/// surface; the variant fixes the chain position and selects the
/// Hadamard flag pair.
pub(crate) enum EightTorsionStepKernel<'a> {
    /// Chain-interior step (`(0, 1)`).
    Interior {
        /// First 8-torsion kernel point.
        T1: &'a JacobianPoint,
        /// Second 8-torsion kernel point.
        T2: &'a JacobianPoint,
    },
    /// Penultimate step before a splitting (`(0, 0)`); codomain stays
    /// in dual form.
    Penultimate {
        /// First 8-torsion kernel point.
        T1: &'a JacobianPoint,
        /// Second 8-torsion kernel point.
        T2: &'a JacobianPoint,
    },
    /// Ultimate step before a splitting (`(1, 0)`); applies the input
    /// Hadamard that the prior penultimate step skipped.
    Ultimate {
        /// First 8-torsion kernel point.
        T1: &'a JacobianPoint,
        /// Second 8-torsion kernel point.
        T2: &'a JacobianPoint,
    },
}

impl EightTorsionStepKernel<'_> {
    /// Computes the step's isogeny. Returns a [`StepIsogeny`] usable
    /// for repeated [`StepIsogeny::eval`] calls.
    ///
    /// Parallels [`Curve::isomorphism`](crate::curves::montgomery::Curve::isomorphism)
    /// at the per-step layer.
    pub(crate) fn isogeny(self) -> StepIsogeny {
        let (T1, T2, variant) = match self {
            Self::Interior { T1, T2 } => (T1, T2, HadamardVariant::Standard),
            Self::Penultimate { T1, T2 } => (T1, T2, HadamardVariant::NoOuter),
            Self::Ultimate { T1, T2 } => (T1, T2, HadamardVariant::Ultimate),
        };

        // Inner Hadamard flag (bool_1):
        //   Standard / NoOuter → no inner Hadamard
        //   Ultimate           → Hadamard before to_squared_theta
        let (hs1, hs2) = match variant {
            HadamardVariant::Standard | HadamardVariant::NoOuter => {
                (T1.squared().hadamard(), T2.squared().hadamard())
            }
            HadamardVariant::Ultimate => (
                T1.hadamard().squared().hadamard(),
                T2.hadamard().squared().hadamard(),
            ),
        };

        // Cross-product formulas; identical across the three variants.
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

        // Outer Hadamard flag (bool_2):
        //   Standard → final Hadamard → codomain in theta form
        //   NoOuter / Ultimate → no final Hadamard → codomain in dual form
        let null = match variant {
            HadamardVariant::Standard => ThetaNullPoint::from(&dual),
            HadamardVariant::NoOuter | HadamardVariant::Ultimate => {
                ThetaNullPoint::new(dual.alpha, dual.beta, dual.gamma, dual.delta)
            }
        };

        StepIsogeny {
            codomain: Jacobian::new(null),
            dual,
            variant,
        }
    }
}

/// Kernel of a 4-torsion (2,2)-isogeny step — penultimate step in an
/// `extra_torsion=false` chain tail.
///
/// Defined by a single 4-torsion point and the domain surface. Uses
/// the Hadamard flag pair `(0, 0)`; the codomain stays in dual form
/// for the matching ultimate step or splitting consumer.
///
/// Implements [Alg. 8.32].
///
/// [Alg. 8.32]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.32
pub(crate) struct FourTorsionStepKernel<'a> {
    /// The 4-torsion kernel point `T₁` on the domain.
    pub(crate) T1: &'a JacobianPoint,
    /// The domain surface, read for its null point.
    pub(crate) domain: &'a Jacobian,
}

impl FourTorsionStepKernel<'_> {
    /// Computes the step's isogeny. Returns a [`StepIsogeny`] usable
    /// for repeated [`StepIsogeny::eval`] calls.
    pub(crate) fn isogeny(self) -> StepIsogeny {
        let Self { T1, domain } = self;
        // TT1 = H(S(T1_4)). For a dual-form 4-torsion point, only the
        // X and Z components are nonzero (Y = W = 0), but we don't
        // depend on that explicitly here.
        let tt1 = T1.squared().hadamard();

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
        // hadamard_bool_2 = 0 → codomain in dual form.
        let null = ThetaNullPoint::new(dual.alpha, dual.beta, dual.gamma, dual.delta);
        StepIsogeny {
            codomain: Jacobian::new(null),
            dual,
            variant: HadamardVariant::NoOuter,
        }
    }
}

/// Kernel of a 2-torsion (2,2)-isogeny step — ultimate step in an
/// `extra_torsion=false` chain tail.
///
/// Defined only by the domain surface (no kernel-point data left at
/// this point in the chain). Uses the Hadamard flag pair `(1, 0)`;
/// the codomain stays in dual form for the splitting consumer.
///
/// Implements [Alg. 8.33].
///
/// [Alg. 8.33]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.8.33
pub(crate) struct TwoTorsionStepKernel<'a> {
    /// The domain surface.
    pub(crate) domain: &'a Jacobian,
}

impl TwoTorsionStepKernel<'_> {
    /// Computes the step's isogeny. Returns a [`StepIsogeny`] usable
    /// for repeated [`StepIsogeny::eval`] calls.
    pub(crate) fn isogeny(self) -> StepIsogeny {
        let domain = self.domain;
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
        // bool_2=0 → codomain in dual form.
        let null = ThetaNullPoint::new(dual.alpha, dual.beta, dual.gamma, dual.delta);
        StepIsogeny {
            codomain: Jacobian::new(null),
            dual,
            variant: HadamardVariant::Ultimate,
        }
    }
}

/// A computed (2,2)-isogeny over one chain step. Owns the codomain
/// surface and the data needed to evaluate points through it.
///
/// Same agency direction as
/// [`Isomorphism::eval`](crate::curves::montgomery::Isomorphism::eval)
/// and [`Kernel::isogeny`](crate::surfaces::Kernel::isogeny) — the
/// isogeny owns its data; points pass through via [`eval`][Self::eval].
pub(crate) struct StepIsogeny {
    /// The codomain surface.
    pub(crate) codomain: Jacobian,
    /// Dual theta null point + projective inverses for the eval
    /// formula.
    dual: DualThetaNullPoint,
    /// Which Hadamard variant the eval formula uses.
    variant: HadamardVariant,
}

impl StepIsogeny {
    /// The codomain surface, by reference.
    pub(crate) fn codomain(&self) -> &Jacobian {
        &self.codomain
    }

    /// Consumes this isogeny and returns the codomain by value, so
    /// the caller avoids an extra `Jacobian` clone when it doesn't
    /// need to push any more points.
    pub(crate) fn into_codomain(self) -> Jacobian {
        self.codomain
    }

    /// Evaluates a point under this isogeny.
    ///
    /// Same shape as
    /// [`Isomorphism::eval`](crate::curves::montgomery::Isomorphism::eval):
    /// the isogeny is computed; the point passes through.
    ///
    /// `#[inline]` so LLVM's loop-invariant code motion can hoist
    /// the variant match out of hot orchestrator loops where
    /// `self` doesn't change across iterations.
    #[inline]
    pub(crate) fn eval(&self, p: &JacobianPoint) -> JacobianPoint {
        let scale = |q: &JacobianPoint| {
            q.scale(
                &self.dual.alpha_inv,
                &self.dual.beta_inv,
                &self.dual.gamma_inv,
                &self.dual.delta_inv,
            )
        };
        let t = match self.variant {
            // `H(precomp · P²·H)` — the inner Hadamard is absorbed
            // into the precomputation.
            HadamardVariant::Standard => scale(&p.squared().hadamard()).hadamard(),
            // `precomp · H(P²)` — no outer Hadamard.
            HadamardVariant::NoOuter => scale(&p.squared().hadamard()),
            // `precomp · H(H(P)²)` — extra Hadamard on input.
            HadamardVariant::Ultimate => scale(&p.hadamard().squared().hadamard()),
        };
        JacobianPoint::new(t.X, t.Y, t.Z, t.W, self.codomain.clone())
    }
}
