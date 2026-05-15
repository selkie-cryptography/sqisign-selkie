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
//! Chain-interior steps consume 8-torsion data directly via the
//! [`step`] primitives ([`codomain_8torsion`]/[`eval`] and the
//! `_no_hadamard` / `_ultimate` variants for the penultimate and
//! ultimate steps before a splitting). The orchestrator
//! [`Kernel::isogeny`] in [`crate::surfaces`] picks the right
//! primitive per chain position.
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

mod generic;
mod gluing;
mod splitting;
pub(crate) mod step;

pub(crate) use generic::{GenericKernel2, GenericKernel4};
pub(crate) use gluing::GluingKernel;
pub(crate) use splitting::SplittingKernel;
#[cfg(test)]
pub(crate) use splitting::{get_index_splitting_count, theta_product_to_montgomery};
pub(crate) use step::{
    codomain_8torsion, codomain_8torsion_no_hadamard, codomain_8torsion_ultimate, eval,
    eval_no_outer_hadamard, eval_ultimate,
};
