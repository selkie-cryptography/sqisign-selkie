//! (2,2)-isogeny kernel types and their isogeny computations.
//!
//! Each step of a (2,2)-isogeny chain is defined by its kernel data.
//! The kernel type determines the domain, codomain, and algorithm:
//!
//! | Kernel | Domain → Codomain | Data |
//! |---|---|---|
//! | [`GluingKernel`] | [`EllipticProduct`] → [`Jacobian`] | Two 8-torsion `ProductPoint` |
//! | `GenericKernel8` | [`Jacobian`] → [`Jacobian`] | Two 8-torsion `JacobianPoint` |
//! | [`GenericKernel4`] | [`Jacobian`] → [`Jacobian`] | One 4-torsion `JacobianPoint` |
//! | [`GenericKernel2`] | [`Jacobian`] → [`Jacobian`] | Domain Jacobian only |
//! | [`SplittingKernel`] | [`Jacobian`] → [`EllipticProduct`] | Domain Jacobian only |
//!
//! The top-level [`Kernel::isogeny`] in [`crate::surfaces`] orchestrates
//! a full chain by constructing the appropriate kernel type at each
//! step.
//!
//! The step-level codomain and evaluator primitives — used directly by
//! the chain orchestrator when step kind (interior, penultimate, or
//! ultimate) needs to vary per chain position — live in
//! [`step`].
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
pub(crate) use splitting::get_index_splitting_count;
#[cfg(test)]
pub(crate) use splitting::{theta_product_to_montgomery, theta_to_product};
pub(crate) use step::{
    codomain_8torsion, codomain_8torsion_no_hadamard, codomain_8torsion_ultimate, eval,
    eval_no_outer_hadamard, eval_ultimate,
};
