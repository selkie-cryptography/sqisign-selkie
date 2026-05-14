//! Generic kernel types `GenericKernel{8,4,2}`: the chain-interior
//! (2,2)-isogeny steps that map `Jacobian → Jacobian`. Each kernel
//! type carries the kernel data appropriate to its order (an
//! 8-torsion pair, a 4-torsion point, or just the domain).

use super::step::{
    codomain_2torsion_ultimate, codomain_4torsion, codomain_4torsion_no_hadamard,
    codomain_8torsion, codomain_from_null, eval, eval_no_outer_hadamard, eval_ultimate,
};
use crate::surfaces::{Jacobian, JacobianPoint};

pub(crate) struct GenericKernel8 {
    /// 8-torsion point T₁'' on the domain Jacobian.
    pub T1: JacobianPoint,
    /// 8-torsion point T₂'' on the domain Jacobian.
    pub T2: JacobianPoint,
}

impl GenericKernel8 {
    /// Compute the generic (2,2)-isogeny and push points through.
    ///
    /// Returns the codomain Jacobian and the images of `pts`.
    ///
    /// Implements `GenericCodomainWith8Torsion` + `GenericEval`
    /// ([§8.5.3], Algorithm 8.30; [§8.5.4], Algorithm 8.34).
    ///
    /// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
    /// [§8.5.4]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.4
    pub(crate) fn isogeny(&self, pts: &[JacobianPoint]) -> (Jacobian, Vec<JacobianPoint>) {
        let (dual, codomain) = codomain_8torsion(&self.T1, &self.T2);
        let images = pts.iter().map(|p| eval(p, &dual, &codomain)).collect();
        (codomain, images)
    }
}

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
    /// Compute the generic (2,2)-isogeny and push points through.
    ///
    /// Implements `GenericCodomainWith4Torsion` + `GenericEval`
    /// ([§8.5.3], Algorithm 8.32).
    ///
    /// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
    pub(crate) fn isogeny(
        &self,
        domain: &Jacobian,
        pts: &[JacobianPoint],
    ) -> (Jacobian, Vec<JacobianPoint>) {
        let (dual, codomain) = codomain_4torsion(&self.T1, domain);
        let images = pts.iter().map(|p| eval(p, &dual, &codomain)).collect();
        (codomain, images)
    }

    /// Penultimate-step (2,2)-isogeny for an `extra_torsion=false`
    /// chain tail.
    ///
    /// Uses [`codomain_4torsion_no_hadamard`] (flag `(0, 0)`) and
    /// [`eval_no_outer_hadamard`]. The codomain is left in dual
    /// form for the matching ultimate step or splitting consumer.
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
    /// Compute the generic (2,2)-isogeny and push points through.
    ///
    /// Implements `GenericCodomain` + `GenericEval`
    /// ([§8.5.3], Algorithm 8.33).
    ///
    /// [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
    pub(crate) fn isogeny(
        domain: &Jacobian,
        pts: &[JacobianPoint],
    ) -> (Jacobian, Vec<JacobianPoint>) {
        let (dual, codomain) = codomain_from_null(domain);
        let images = pts.iter().map(|p| eval(p, &dual, &codomain)).collect();
        (codomain, images)
    }

    /// Ultimate-step (2,2)-isogeny for an `extra_torsion=false`
    /// chain tail.
    ///
    /// Uses [`codomain_2torsion_ultimate`] (flag `(1, 0)`) and
    /// [`eval_ultimate`]. The codomain is left in dual form for the
    /// splitting consumer.
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
