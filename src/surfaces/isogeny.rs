//! (2,2)-isogeny kernel types and their isogeny computations.
//!
//! Each step of a (2,2)-isogeny chain is defined by its kernel data:
//!
//! | Kernel | Domain → Codomain | Data |
//! |---|---|---|
//! | [`GluingKernel`] | [`EllipticProduct`] → [`Jacobian`] | Two 8-torsion `ProductPoint` |
//! | [`EightTorsionStepKernel`] | [`Jacobian`] → [`Jacobian`] | Two 8-torsion `JacobianPoint` (chain-interior) |
//! | [`FourTorsionStepKernel`] | [`Jacobian`] → [`Jacobian`] | One 4-torsion `JacobianPoint` (penultimate-only) |
//! | [`TwoTorsionStepKernel`] | [`Jacobian`] → [`Jacobian`] | Domain Jacobian only (ultimate-only) |
//! | [`SplittingKernel`] | [`Jacobian`] → [`EllipticProduct`] | Domain Jacobian only |
//!
//! Each step kernel exposes `.isogeny() -> `[`StepIsogeny`], which
//! in turn exposes `eval` for point pushing — parallel to
//! [`Curve::isomorphism`](crate::curves::montgomery::Curve::isomorphism)
//! returning an
//! [`Isomorphism`](crate::curves::montgomery::Isomorphism) with
//! its own `eval`. The orchestrator [`Kernel::isogeny`] in
//! [`crate::surfaces`] picks the right step kernel per chain
//! position.
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
pub(crate) use step::{
    EightTorsionStepKernel, FourTorsionStepKernel, StepIsogeny, TwoTorsionStepKernel,
};
