//! Precomputed constants used by the (2,2)-isogeny machinery in
//! [`crate::surfaces`].
//!
//! Currently holds the [`NORMALIZATION_TRANSFORMS`] used to randomize
//! the projective representative emitted by the chain's splitting
//! step. Other surface-level precomputed data (e.g. theta change-of-
//! basis matrices) may move here as it accumulates.

use crate::{fields::fp2::Fp2, surfaces::GluingMatrix};

/// Six 4×4 matrices over `F_{p²}` used to randomize the splitting
/// isomorphism's projective representative.
///
/// Each matrix is the tensor product `M ⊗ M` of one of six 2×2
/// matrices that form coset representatives of the symplectic
/// modular group quotient `Γ / Γ⁰(4)` (sufficient generators for
/// the equivalence class of theta nulls when working in the
/// Montgomery model). The 6-element subset is taken from a fuller
/// 24-element set (representing `Γ / Γ(2, 4)`) and is precomputed
/// by `precompute_hd_splitting.sage` in the C reference.
///
/// # Why this exists
///
/// The output of the (2,2)-isogeny chain is a level-2 theta null
/// point on a product abelian surface `E₁ × E₂`. Multiple
/// projective representatives exist for the same abstract surface,
/// related by the symplectic group action. The deterministic
/// `From<&ThetaNullPoint> for GluingMatrix` (Algorithm 8.42) picks one specific
/// representative — but that choice is a function of the kernel,
/// and so leaks information about the kernel through the
/// representative actually emitted. Pre-multiplying the splitting
/// matrix by a uniformly random `M_k` from this list re-randomizes
/// the representative within its equivalence class, breaking the
/// leak.
///
/// # Divergence from spec
///
/// The SQIsign v2 spec's Algorithm 8.42 specifies only the
/// deterministic matrix. The randomization is an
/// implementation-level countermeasure introduced by the C
/// reference (`precomp/.../hd_splitting_transforms.c`,
/// `hd/ref/lvlx/theta_isogenies.c:1043-1059`). We mirror it
/// because:
///
/// 1. It is a real side-channel hardening — without it, the public output of
///    keygen / sign is a deterministic function of secret kernel structure, not
///    just the secret kernel itself.
/// 2. The published KAT vectors are produced *with* this randomization applied,
///    so byte-equality with KATs requires us to mirror the same RNG-byte
///    consumption pattern and matrix selection as the C reference.
pub(crate) const NORMALIZATION_TRANSFORMS: [GluingMatrix; 6] = [
    // Index 0 — identity: `M = I₂ ⊗ I₂`, no transform.
    GluingMatrix([
        [Fp2::ONE, Fp2::ZERO, Fp2::ZERO, Fp2::ZERO],
        [Fp2::ZERO, Fp2::ONE, Fp2::ZERO, Fp2::ZERO],
        [Fp2::ZERO, Fp2::ZERO, Fp2::ONE, Fp2::ZERO],
        [Fp2::ZERO, Fp2::ZERO, Fp2::ZERO, Fp2::ONE],
    ]),
    // Index 1 — anti-diagonal swap: `M = S ⊗ S` where `S = [[0,1],[1,0]]`.
    GluingMatrix([
        [Fp2::ZERO, Fp2::ZERO, Fp2::ZERO, Fp2::ONE],
        [Fp2::ZERO, Fp2::ZERO, Fp2::ONE, Fp2::ZERO],
        [Fp2::ZERO, Fp2::ONE, Fp2::ZERO, Fp2::ZERO],
        [Fp2::ONE, Fp2::ZERO, Fp2::ZERO, Fp2::ZERO],
    ]),
    // Index 2 — `H ⊗ H` with `H = [[1, 1], [1, -1]]`: a Hadamard-like
    // permutation of the level-2 theta basis.
    GluingMatrix([
        [Fp2::ONE, Fp2::ONE, Fp2::ONE, Fp2::ONE],
        [Fp2::ONE, Fp2::MINUS_ONE, Fp2::ONE, Fp2::MINUS_ONE],
        [Fp2::ONE, Fp2::ONE, Fp2::MINUS_ONE, Fp2::MINUS_ONE],
        [Fp2::ONE, Fp2::MINUS_ONE, Fp2::MINUS_ONE, Fp2::ONE],
    ]),
    // Index 3 — `H' ⊗ H'` with `H' = [[-1, 1], [1, 1]]`.
    GluingMatrix([
        [Fp2::ONE, Fp2::MINUS_ONE, Fp2::MINUS_ONE, Fp2::ONE],
        [Fp2::MINUS_ONE, Fp2::MINUS_ONE, Fp2::ONE, Fp2::ONE],
        [Fp2::MINUS_ONE, Fp2::ONE, Fp2::MINUS_ONE, Fp2::ONE],
        [Fp2::ONE, Fp2::ONE, Fp2::ONE, Fp2::ONE],
    ]),
    // Index 4 — `K ⊗ K` with `K = [[i, 1], [1, i]]`.
    GluingMatrix([
        [Fp2::MINUS_ONE, Fp2::I, Fp2::I, Fp2::ONE],
        [Fp2::I, Fp2::MINUS_ONE, Fp2::ONE, Fp2::I],
        [Fp2::I, Fp2::ONE, Fp2::MINUS_ONE, Fp2::I],
        [Fp2::ONE, Fp2::I, Fp2::I, Fp2::MINUS_ONE],
    ]),
    // Index 5 — `K' ⊗ K'` with `K' = [[1, i], [i, 1]]`.
    GluingMatrix([
        [Fp2::ONE, Fp2::I, Fp2::I, Fp2::MINUS_ONE],
        [Fp2::I, Fp2::ONE, Fp2::MINUS_ONE, Fp2::I],
        [Fp2::I, Fp2::MINUS_ONE, Fp2::ONE, Fp2::I],
        [Fp2::MINUS_ONE, Fp2::I, Fp2::I, Fp2::ONE],
    ]),
];
