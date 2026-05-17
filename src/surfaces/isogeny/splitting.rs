//! `SplittingKernel`: the final (2,2)-isogeny step of a chain,
//! `Jacobian → EllipticProduct`. The domain Jacobian's theta null
//! point has product structure, allowing recovery of the component
//! curves and a return to the elliptic-curve setting. Includes the
//! `SplittingIndex` machinery, the [`GluingMatrix`]-from-null
//! normaliser (Algorithm 8.42, [`From<&ThetaNullPoint> for
//! GluingMatrix`]), and the [`EllipticProduct`]-from-null /
//! `theta_product_to_montgomery` final conversions.
//!
//! See [§8.5.7] and [§8.5.8].
//!
//! [§8.5.7]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.7
//! [§8.5.8]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.8

use crate::{
    curves::montgomery::{Curve, ProjectiveXOnlyPoint},
    fields::fp2::Fp2,
    surfaces::{
        EllipticProduct, GluingMatrix, Jacobian, JacobianPoint, ProductPoint, ThetaNullPoint,
        precomputed::NORMALIZATION_TRANSFORMS,
    },
};

/// Kernel of a splitting (2,2)-isogeny Φₑ : Aₑ₋₁ → E₃ × E₄.
///
/// The final step of a (2,2)-isogeny chain. The domain Jacobian's
/// theta null point has product structure, allowing recovery of the
/// component curves E₃, E₄.
///
/// See [§8.5.7].
///
/// [§8.5.7]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.7
pub(crate) struct SplittingKernel {
    /// The domain Jacobian (whose null point has product structure).
    pub domain: Jacobian,
}

impl SplittingKernel {
    /// Compute the splitting and push points through.
    ///
    /// Returns the codomain product E₃ × E₄ and the images of `pts`
    /// converted back to Montgomery coordinates, or [`None`] if the
    /// chain's final theta null does not admit a product splitting
    /// (i.e., [`ThetaNullPoint::splitting_index_count`] is not
    /// exactly 1).
    ///
    /// Implements `SplittingIsomorphism` + `ThetaToProduct` +
    /// `ThetaProductPointToMontgomery` ([§8.5.7]).
    ///
    /// # Divergences
    ///
    /// The spec's Algorithm 8.42 says to "find the unique index
    /// such that `U_{i,j}(0) = 0`" without stating that exactly
    /// one must vanish. In practice the splitting-index count is
    /// a load-bearing invariant: a malformed input (e.g., an
    /// upstream chain that was fed a kernel short on torsion)
    /// produces a terminal theta null where either 0 or 10 of the
    /// ten `U_{i,j}(0)` coordinates vanish, and the splitting
    /// machinery then silently picks a wrong branch and emits
    /// bad curves. Treating `count != 1` as an explicit error
    /// is the difference between "signing key is wrong but
    /// keygen looks successful" and "keygen retries with a fresh
    /// random ideal." The SQIsign v2 spec review flags this as
    /// a recommended spec clarification (§\textsc{SplittingIsomorphism}
    /// must handle malformed input).
    ///
    /// [§8.5.7]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.8.5.7
    pub(crate) fn isogeny(
        &self,
        pts: &[JacobianPoint],
        randomize: Option<&mut dyn rand_core::RngCore>,
    ) -> Option<(EllipticProduct, Vec<ProductPoint>)> {
        // Exactly one of the 10 `U_{i,j}(0)` coordinates must
        // vanish for the chain's terminal theta null to
        // correspond to a product of elliptic curves. Anything
        // else is a malformed chain output (see Divergences
        // above).
        let split_count = self.domain.null.splitting_index_count();
        if split_count != 1 {
            return None;
        }

        // 1. SplittingIsomorphism: find the matrix M (Algorithm 8.42).
        let mut M = GluingMatrix::from(&self.domain.null);

        // Apply a random level-2 normalization matrix when the caller
        // requested randomization. See `NORMALIZATION_TRANSFORMS` for
        // the underlying construction. This does not change the
        // *abstract* product surface — both `M` and
        // `NORMALIZATION_TRANSFORMS[idx] · M` send the level-2 theta
        // null to a product theta null on the same `E₁ × E₂` — but it
        // does randomize the specific projective representative the
        // caller observes. Mirrors C reference's
        // `splitting_compute(...)`'s `randomize=true` branch
        // (`theta_isogenies.c:1043-1059`).
        if let Some(rng) = randomize {
            let idx = sample_normalization_index(rng) as usize;
            M = &NORMALIZATION_TRANSFORMS[idx] * &M;
        }

        // 2. Apply M to the null point to get product theta structure.
        let product_null = M.apply_null(&self.domain.null);

        // 3. ThetaToProduct: recover (A₁:C₁), (A₂:C₂) (Algorithm 8.44).
        let product = EllipticProduct::from(&product_null);

        // 4. ThetaProductPointToMontgomery for each point (Algorithm 8.45).
        let images = pts
            .iter()
            .map(|p| {
                // Apply M to point, then convert to Montgomery.
                let mp = M.apply_point(p);
                theta_product_to_montgomery(&mp, &product_null, &product)
            })
            .collect();

        Some((product, images))
    }
}

/// Sample a uniform index in `[0, 6)` for selecting one of the six
/// [`NORMALIZATION_TRANSFORMS`] matrices.
///
/// Reads four bytes from `rng`, parses them as a little-endian
/// `u32`, and rejects-and-resamples any value `≥ 6 · ⌊2³² / 6⌋ =
/// 4_294_967_292` to obtain an unbiased `mod 6` reduction.
///
/// Mirrors the C reference's `sample_random_index`
/// (`theta_isogenies.c:980`) — same byte-stream consumption pattern
/// (4 bytes, little-endian, rejection-sample threshold `4_294_967_292`)
/// so a deterministic DRBG seeded identically on both
/// implementations selects the same index. The C reference also
/// uses a constant-time `mod 6` trick (Granlund–Möller); we use the
/// plain `% 6` since this code path is variable-time on public data
/// (the chain output is public; the secret signing key has already
/// been fully consumed by the kernel).
///
/// [`NORMALIZATION_TRANSFORMS`]: crate::surfaces::precomputed::NORMALIZATION_TRANSFORMS
fn sample_normalization_index<R: rand_core::RngCore + ?Sized>(rng: &mut R) -> u8 {
    loop {
        let mut buf = [0u8; 4];
        rng.fill_bytes(&mut buf);
        let seed = u32::from_le_bytes(buf);
        if seed < 4_294_967_292u32 {
            return (seed % 6) as u8;
        }
        // Resample on the rare seed in `[4_294_967_292, 2³²)` —
        // approximately `4 / 2³² ≈ 10⁻⁹` chance per draw.
    }
}

#[derive(Clone, Copy)]
enum SplittingIndex {
    /// (i, j) = (0, 0)
    I00,
    /// (i, j) = (0, 1)
    I01,
    /// (i, j) = (0, 2)
    I02,
    /// (i, j) = (0, 3)
    I03,
    /// (i, j) = (1, 0)
    I10,
    /// (i, j) = (1, 2)
    I12,
    /// (i, j) = (2, 0)
    I20,
    /// (i, j) = (2, 1)
    I21,
    /// (i, j) = (3, 0)
    I30,
    /// (i, j) = (3, 3)
    I33,
}

/// All valid splitting indices, in the order checked by Algorithm 8.41.
const SPLITTING_INDICES: [(usize, usize, SplittingIndex); 10] = [
    (0, 0, SplittingIndex::I00),
    (0, 1, SplittingIndex::I01),
    (0, 2, SplittingIndex::I02),
    (0, 3, SplittingIndex::I03),
    (1, 0, SplittingIndex::I10),
    (1, 2, SplittingIndex::I12),
    (2, 0, SplittingIndex::I20),
    (2, 1, SplittingIndex::I21),
    (3, 0, SplittingIndex::I30),
    (3, 3, SplittingIndex::I33),
];

/// χ function for `GetIndexSplitting` (Algorithm 8.41).
fn chi(i: usize, j: usize) -> i8 {
    match (i, j) {
        (0, 0) | (0, 1) | (0, 2) | (0, 3) | (1, 0) | (1, 2) | (2, 0) | (2, 1) | (3, 0) | (3, 3) => {
            1
        }
        (1, 1) | (1, 3) | (2, 2) | (2, 3) | (3, 1) | (3, 2) => -1,
        _ => 0,
    }
}

impl ThetaNullPoint {
    /// Counts how many of the 10 `U_{i,j}(0)` coordinates vanish at
    /// this theta null point.
    ///
    /// For a chain that ends at a product of elliptic curves, exactly
    /// one `U_{i,j}(0)` is zero (it identifies which product
    /// decomposition applies). Any other count — particularly 0 —
    /// signals that the codomain is not a product and the splitting
    /// machinery will produce bad output if applied.
    pub(crate) fn splitting_index_count(&self) -> u32 {
        let coords = [&self.a, &self.b, &self.c, &self.d];
        let mut count = 0u32;
        for &(i, j, _idx) in &SPLITTING_INDICES {
            let mut U = Fp2::ZERO;
            for t in 0..4 {
                let chi_val = chi(i, t) as i64;
                if chi_val != 0 {
                    let term = coords[j ^ t] * coords[t];
                    if chi_val > 0 {
                        U = &U + &term;
                    } else {
                        U = &U - &term;
                    }
                }
            }
            if U == Fp2::ZERO {
                count += 1;
            }
        }
        count
    }
}

/// Test alias kept so diagnostic callers in `surfaces::mod` continue
/// to compile with the original spec name.
#[cfg(test)]
pub(crate) fn get_index_splitting_count(null: &ThetaNullPoint) -> u32 {
    null.splitting_index_count()
}

/// Find the splitting index such that U_{i,j}(0) = 0
/// (Algorithm 8.41).
fn get_index_splitting(null: &ThetaNullPoint) -> SplittingIndex {
    let coords = [&null.a, &null.b, &null.c, &null.d];

    let mut count = 0;
    let mut result = SplittingIndex::I00;
    for &(i, j, idx) in &SPLITTING_INDICES {
        let mut U = Fp2::ZERO;
        for t in 0..4 {
            let chi_val = chi(i, t) as i64;
            if chi_val != 0 {
                let term = coords[j ^ t] * coords[t];
                if chi_val > 0 {
                    U = &U + &term;
                } else {
                    U = &U - &term;
                }
            }
        }
        if U == Fp2::ZERO {
            count += 1;
            result = idx;
        }
    }
    debug_assert!(
        count == 1,
        "GetIndexSplitting: expected exactly one zero index, found {count}"
    );
    result
}

impl From<&ThetaNullPoint> for GluingMatrix {
    /// Compute `SplittingIsomorphism` (Algorithm 8.42).
    ///
    /// Returns the 4×4 matrix `M` whose action on `null` recovers the
    /// product theta structure. Defined only when the input is a
    /// terminal theta null (exactly one vanishing `U_{i,j}(0)`); the
    /// caller is expected to gate with
    /// [`ThetaNullPoint::splitting_index_count`].
    fn from(null: &ThetaNullPoint) -> Self {
        let idx = get_index_splitting(null);
        let one = Fp2::ONE;
        let neg = -&one;
        let zero = Fp2::ZERO;

        // The matrices for each (i,j) case come from Algorithm 8.42.
        // For simplicity, only implement the cases that arise in
        // Isogeny22Chain (the spec guarantees (i,j) = (0,0) or (1,1)
        // for SQIsign's chain via Algorithm 8.47).
        use SplittingIndex::*;
        GluingMatrix(match idx {
            I00 => {
                // C reference: SPLITTING_TRANSFORMS[0] for (i,j) = (0,0).
                // Uses i = sqrt(-1) in Fp2.
                let i_val = Fp2::I;
                let neg_i = -&i_val;
                [
                    [one, i_val, one, i_val],
                    [one, neg_i, neg, i_val],
                    [one, i_val, neg, neg_i],
                    [neg, i_val, neg, i_val],
                ]
            }
            I10 => [
                [one, one, one, one],
                [one, neg, neg, one],
                [one, one, neg, neg],
                [neg, one, neg, one],
            ],
            I20 => [
                [one, one, one, one],
                [one, neg, one, neg],
                [one, neg, neg, one],
                [neg, neg, one, one],
            ],
            I30 => [
                [one, one, one, one],
                [one, neg, one, neg],
                [one, one, neg, neg],
                [neg, one, one, neg],
            ],
            I01 => [
                [one, zero, zero, zero],
                [zero, zero, zero, one],
                [zero, zero, one, zero],
                [zero, neg, zero, zero],
            ],
            I21 => [
                [one, one, one, one],
                [one, neg, one, neg],
                [one, neg, neg, one],
                [one, one, neg, neg],
            ],
            I02 => [
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, zero, one],
                [zero, zero, neg, zero],
            ],
            I12 => [
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, zero, one],
                [zero, zero, one, zero],
            ],
            I03 => [
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, one, zero],
                [zero, zero, zero, neg],
            ],
            I33 => [
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, one, zero],
                [zero, zero, zero, one],
            ],
        })
    }
}

impl From<&ThetaNullPoint> for EllipticProduct {
    /// Recover the component Montgomery curves from a product theta
    /// null point (Algorithm 8.44).
    ///
    /// Constructs each component curve via
    /// `Curve::from(ProjectiveCoefficient)`, preserving the un-reduced
    /// `(A : C)` form that comes out of the formulas
    /// (`A = -2(x⁴ + z⁴)`, `C = x⁴ − z⁴`). The cached
    /// `DoublingConstants` is still normalized to `(A₂₄/C₂₄ : 1)`, so
    /// `.double()` produces the normalized `xDBL_A24` representative
    /// used in most downstream code paths.
    ///
    /// The original `(A : C)` is still readable via
    /// `curve.projective`. Code paths that need to byte-match C-ref's
    /// un-normalized `xDBL` (e.g., the outer-chain prep doublings in
    /// `to_isogeny`, where C-ref skips `ec_curve_normalize_A24`) use
    /// [`ProjectiveXOnlyPoint::double_unnormalized`] instead.
    ///
    /// Defined only when `null` has product theta structure (`ad =
    /// bc`); callers must gate by
    /// [`ThetaNullPoint::splitting_index_count`] returning `1`.
    fn from(null: &ThetaNullPoint) -> Self {
        use crate::curves::montgomery::ProjectiveCoefficient;

        let (a, b, c, d) = (&null.a, &null.b, &null.c, &null.d);

        // Check product structure: ad == bc.
        debug_assert!(
            &(a * d) == &(b * c),
            "ThetaToProduct: not a product theta structure"
        );

        let x = a.square().square(); // a⁴
        let y = b.square().square(); // b⁴
        let z = c.square().square(); // c⁴

        // (A₂ : C₂) for E₂: A₂ = -2(x + y), C₂ = x - y
        let pc2 = ProjectiveCoefficient {
            A: -&(&(&x + &y) + &(&x + &y)),
            C: &x - &y,
        };

        // (A₁ : C₁) for E₁: A₁ = -2(x + z), C₁ = x - z
        let pc1 = ProjectiveCoefficient {
            A: -&(&(&x + &z) + &(&x + &z)),
            C: &x - &z,
        };

        EllipticProduct::new(Curve::from(pc1), Curve::from(pc2))
    }
}

/// Convert a theta point with product structure to Montgomery
/// coordinates on each component (Algorithm 8.45).
pub(crate) fn theta_product_to_montgomery(
    P: &JacobianPoint,
    null: &ThetaNullPoint,
    product: &EllipticProduct,
) -> ProductPoint {
    let (a, b, c, _d) = (&null.a, &null.b, &null.c, &null.d);
    #[allow(unused_variables)]
    let (x, y, z, w) = (&P.X, &P.Y, &P.Z, &P.W);

    // Algorithm 8.45:
    // X₁ = a·z + c·x,  Z₁ = a·z − c·x
    // X₂ = a·y + b·x,  Z₂ = a·y − b·x
    let X1 = &(a * z) + &(c * x);
    let Z1 = &(a * z) - &(c * x);
    let X2 = &(a * y) + &(b * x);
    let Z2 = &(a * y) - &(b * x);

    (
        ProjectiveXOnlyPoint::from_XZ(X1, Z1, &product.E1),
        ProjectiveXOnlyPoint::from_XZ(X2, Z2, &product.E2),
    )
}
