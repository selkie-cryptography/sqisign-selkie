//! (2,2)-isogeny kernel types and their isogeny computations.
//!
//! Each step of a (2,2)-isogeny chain is defined by its kernel data:
//!
//! | Kernel | Domain → Codomain | Data |
//! |---|---|---|
//! | [`GluingKernel`] | [`EllipticProduct`] → [`Jacobian`] | Two 8-torsion `ProductPoint` |
//! | [`GenericKernel4`] | [`Jacobian`] → [`Jacobian`] | One 4-torsion `JacobianPoint` (penultimate-only) |
//! | [`GenericKernel2`] | [`Jacobian`] → [`Jacobian`] | Domain Jacobian only (ultimate-only) |
//! | [`SplittingKernel`] | [`Jacobian`] → [`EllipticProduct`] | Domain Jacobian only |
//!
//! Chain-interior steps consume 8-torsion data via
//! [`EightTorsionStep`] in the [`step`] submodule; the penultimate
//! and ultimate steps of an `extra_torsion=false` chain tail use
//! [`FourTorsionStep`] and [`TwoTorsionStep`] respectively. Each
//! returns a [`StepIsogeny`] (parallel to
//! [`Isomorphism`](crate::curves::montgomery::Isomorphism)) that
//! exposes `eval` for point pushing. The orchestrator
//! [`Kernel::isogeny`] in [`crate::surfaces`] picks the right
//! kernel per chain position.
//!
//! See [§2.4.1] and [§8.5.3] through [§8.5.8].
//!
//! [§2.4.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.4.1
//! [§8.5.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.3
//! [§8.5.8]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.8
//!
//! [`EllipticProduct`]: super::EllipticProduct
//! [`Jacobian`]: super::Jacobian
//! [`Kernel::isogeny`]: super::Kernel::isogeny

mod gluing;
mod splitting;
pub(crate) mod step;

pub(crate) use gluing::GluingKernel;
pub(crate) use splitting::SplittingKernel;
#[cfg(test)]
pub(crate) use splitting::{get_index_splitting_count, theta_product_to_montgomery};
pub(crate) use step::{
    EightTorsionStepKernel, FourTorsionStepKernel, StepIsogeny, TwoTorsionStepKernel,
};
