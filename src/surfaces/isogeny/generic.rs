//! Generic kernel types `GenericKernel{4,2}` for the chain-tail
//! (2,2)-isogeny steps in an `extra_torsion=false` chain. Each kernel
//! type carries the kernel data appropriate to its torsion order — a
//! 4-torsion point or just the domain Jacobian.
//!
//! The standard chain-interior step uses the 8-torsion kernel data
//! directly via [`step::codomain_8torsion`](super::step::codomain_8torsion);
//! it does not get its own kernel struct because the chain orchestrator
//! consumes the 8-torsion pair and the result without an intervening
//! kernel object.

use super::step::{
    codomain_2torsion_ultimate, codomain_4torsion_no_hadamard, eval_no_outer_hadamard,
    eval_ultimate,
};
use crate::surfaces::{Jacobian, JacobianPoint};

/// Kernel of a generic (2,2)-isogeny specified by a single 4-torsion
/// point on a Jacobian, plus the domain's theta null point.
///
/// Used at the penultimate step of a chain when only 4-torsion data
/// remains. Requires two square roots.
///
/// See [§8.5.3], Algorithm 8.32.
///
/// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
pub(crate) struct GenericKernel4 {
    /// 4-torsion point T₁' on the domain Jacobian.
    pub T1: JacobianPoint,
}

impl GenericKernel4 {
    /// Penultimate-step (2,2)-isogeny for an `extra_torsion=false`
    /// chain tail.
    ///
    /// Uses [`codomain_4torsion_no_hadamard`] (flag `(0, 0)`) and
    /// [`eval_no_outer_hadamard`]. The codomain is left in dual form
    /// for the matching ultimate step or splitting consumer.
    ///
    /// [`codomain_4torsion_no_hadamard`]: super::step::codomain_4torsion_no_hadamard
    /// [`eval_no_outer_hadamard`]: super::step::eval_no_outer_hadamard
    ///
    /// C reference: `theta_isogeny_compute_4(..., 0, 0)` at
    /// `theta_isogenies.c:1258`.
    pub(crate) fn isogeny_penultimate(
        &self,
        domain: &Jacobian,
        pts: &[JacobianPoint],
    ) -> (Jacobian, Vec<JacobianPoint>) {
        let (dual, codomain) = codomain_4torsion_no_hadamard(&self.T1, domain);
        let images = pts
            .iter()
            .map(|p| eval_no_outer_hadamard(p, &dual, &codomain))
            .collect();
        (codomain, images)
    }
}

/// Kernel of a generic (2,2)-isogeny specified only by the domain
/// Jacobian's theta null point (2-torsion kernel generators).
///
/// Used at the final generic step of a chain. Requires three square
/// roots.
///
/// See [§8.5.3], Algorithm 8.33.
///
/// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
pub(crate) struct GenericKernel2;

impl GenericKernel2 {
    /// Ultimate-step (2,2)-isogeny for an `extra_torsion=false`
    /// chain tail.
    ///
    /// Uses [`codomain_2torsion_ultimate`] (flag `(1, 0)`) and
    /// [`eval_ultimate`]. The codomain is left in dual form for the
    /// splitting consumer.
    ///
    /// [`codomain_2torsion_ultimate`]: super::step::codomain_2torsion_ultimate
    /// [`eval_ultimate`]: super::step::eval_ultimate
    ///
    /// C reference: `theta_isogeny_compute_2(..., 1, 0)` at
    /// `theta_isogenies.c:1266`.
    pub(crate) fn isogeny_ultimate(
        domain: &Jacobian,
        pts: &[JacobianPoint],
    ) -> (Jacobian, Vec<JacobianPoint>) {
        let (dual, codomain) = codomain_2torsion_ultimate(domain);
        let images = pts
            .iter()
            .map(|p| eval_ultimate(p, &dual, &codomain))
            .collect();
        (codomain, images)
    }
}
