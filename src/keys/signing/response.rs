//! Sign-side response-phase kernels.
//!
//! Houses the wrappers for the two `(2,2)`- and small-chain kernels
//! invoked during response computation (lines 13-37 of
//! [Algorithm 4.2][Alg. 4.2]):
//!
//! - [`split_aux::SplitAuxiliaryKernel`] — the `(2,2)`-isogeny `φ_aux ×
//!   φ^odd_rsp` on `E_com × E'_aux` from [SplitAuxiliaryIsogeny][Alg. 4.5].
//! - [`even_response::EvenResponseKernel`] — the small even isogeny `φ^even_rsp
//!   : E_chl → E` from [ComputeEvenNonBacktrackingResponse][Alg. 4.6].
//!
//! The third response-phase algorithm,
//! [ComputeChallengeIsogeny][Alg. 4.7], hangs off the
//! [`Challenge`][crate::keys::Challenge] type as
//! [`Challenge::to_isogeny`][crate::keys::Challenge::to_isogeny]
//! since it operates on a challenge value rather than a kernel
//! point — it stays in `crate::keys`.
//!
//! [Alg. 4.2]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.2
//! [Alg. 4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.5
//! [Alg. 4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.6
//! [Alg. 4.7]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.7

pub(crate) mod even_response;
pub(crate) mod split_aux;
