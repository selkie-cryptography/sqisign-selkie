//! `SuitableIdeals` ([Alg. 3.16]) and supporting machinery: short-vector
//! enumeration over the seven extremal orders, the multi-order
//! `(β_s, β_t)` pair search, and the `u·d₁ + v·d₂ = 2^e` solver.
//!
//! [Alg. 3.16]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.16

use core::ops::Deref;

use super::super::{
    algebra::{Coordinate, Denominator, Element},
    bigint::BigInt,
    lattice::{ExtremalOrder, Lattice, LeftIdeal, NrdBasis},
    linear::{Matrix, Vector},
    precomputed::{EXTREMAL_ORDERS, NUM_EXTREMAL_ORDERS, P_WIDE},
};
use crate::curves::{TorsionExponent, isogeny::IsogenyDegree};

/// One factor of a [`SuitableIdealResult`] decomposition.
///
/// Bundles a short-vector element β ∈ J_t · I together with the
/// extremal order it came from and its degree d = nrd(β) / nrd(J_t · I).
/// Grouping these prevents accidentally pairing one factor's degree
/// with another factor's element.
///
/// Generic over the parent-ideal storage width `N`: callers with a
/// narrowed ideal use `IdealFactor<4>`; the response phase passes
/// the un-reduced intersection at `IdealFactor<N>` for some
/// `N ≥ 8` so that downstream `to_isogeny` scaling reads the
/// original (un-reduced) `nrd(parent_ideal)`.
pub(crate) struct IdealFactor<const N: usize> {
    /// The extremal order that produced this factor.
    pub(crate) order: &'static ExtremalOrder<4>,
    /// The short vector β.
    pub(crate) beta: ShortVector,
    /// Degree `d = nrd(β) / nrd(parent_ideal)`.
    pub(crate) degree: IsogenyDegree,
    /// The ideal β was enumerated from: for `t = 0` this is the
    /// caller-supplied ideal `I` (possibly replaced by its smallest
    /// equivalent), for `t > 0` it will be `J_t · I`.
    ///
    /// Carrying the parent ideal here keeps the invariant
    /// `nrd(β) = degree · nrd(parent_ideal)` local to the factor,
    /// so downstream scaling formulas needing `nrd(J_t · I)` read
    /// it through [`LeftIdeal::norm`] on this field rather than
    /// reaching for the caller-supplied ideal whose norm no longer
    /// matches β after the reduction step.
    // reason: read-side wiring is documented but not yet hooked up
    // downstream; the field is assigned/copied so the invariant
    // travels with each factor.
    #[allow(dead_code)]
    pub(crate) parent_ideal: LeftIdeal<N>,
}

/// Result of [Alg. 3.16][Alg. 3.16] (SuitableIdeals).
///
/// Contains integers u, v and exponent e such that
/// u · d₁ + v · d₂ = 2^e with gcd(u · d₁, v · d₂) = 1 and e ≤ f,
/// where d₁ and d₂ are the degrees of [`factor1`](Self::factor1)
/// and [`factor2`](Self::factor2).
///
/// [Alg. 3.16]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.16
pub(crate) struct SuitableIdealResult<const N: usize> {
    /// Odd positive integer u.
    // TODO: Replace with a positive-integer newtype.
    pub(crate) u: BigInt<4>,
    /// Positive integer v.
    // TODO: Replace with a positive-integer newtype.
    pub(crate) v: BigInt<4>,
    /// Exponent e ≤ f.
    pub(crate) e: TorsionExponent,
    /// First factor (β₁, d₁, order index s).
    pub(crate) factor1: IdealFactor<N>,
    /// Second factor (β₂, d₂, order index t).
    pub(crate) factor2: IdealFactor<N>,
}

/// A candidate from the short-vector enumeration, retaining its
/// degree both as a membership invariant and as the sort key.
///
/// A single enumeration batch uses a fixed parent ideal, so
/// `nrd(parent_ideal)` is constant across every candidate and
/// sorting by `degree = nrd(β) / nrd(parent_ideal)` gives the same
/// ordering as sorting by `nrd(β)` itself. Promoted to an
/// [`IdealFactor`] once a viable pair is selected in
/// [`try_find_uv`].
struct ShortVectorCandidate {
    /// Quaternion element β, a linear combination of the reduced basis.
    elem: Element<4>,
    /// Degree: `nrd(β) / nrd(parent_ideal)`, a positive odd integer.
    degree: IsogenyDegree,
}

/// One per-order enumeration batch: the extremal order the batch
/// belongs to, the ideal the short vectors were enumerated from
/// (`I` for `t = 0`, typically `J_t · I` or an equivalent for
/// `t > 0`), and the resulting candidate list sorted by degree.
///
/// Passing this through [`try_find_uv`] keeps the per-factor
/// `(order, parent_ideal)` pair local to each β; in the
/// multi-order search the two factors come from different batches.
struct ShortVectorBatch<const N: usize> {
    /// The extremal order that produced this batch.
    order: &'static ExtremalOrder<4>,
    /// The ideal the short vectors live in.
    parent_ideal: LeftIdeal<N>,
}

/// A quaternion element that is a short vector in some ideal lattice.
///
/// Wraps an [`Element<4>`] with a stronger contract than a bare
/// algebra element: a `ShortVector` was produced by the
/// short-vector enumeration inside [`LeftIdeal::suitable_ideals`]
/// and lives in a specific [`LeftIdeal<4>`] (tracked by the
/// consumer via [`IdealFactor::parent_ideal`]) with norm on the
/// order of `√nrd(parent_ideal)`.
///
/// Construction is private to this module — there is no public
/// constructor — so a `ShortVector` in hand is a proof-by-type
/// that the enumeration has already validated the underlying
/// element.
///
/// The inner [`Element<4>`] is reachable through [`Deref`], so any
/// site that wants `&Element<4>` (for example `action_matrix` in
/// the deuring module) works by deref coercion.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShortVector(Element<4>);

impl Deref for ShortVector {
    type Target = Element<4>;

    fn deref(&self) -> &Element<4> {
        &self.0
    }
}

/// Canonicalize the L2-reduced basis of an ideal in the **special
/// order** O₀ (the t=0 case in `suitable_ideals`'s multi-order loop).
///
/// Mirrors C ref's `post_LLL_basis_treatment(_, _, _, is_special_order=true)`
/// in `the-sqisign/src/id2iso/ref/lvlx/dim2id2iso.c:213-268`.
///
/// L2 reduction returns *some* reduced basis; the choice between
/// equivalent reduced bases (related by signed column permutations)
/// is implementation-defined. C ref imposes a deterministic
/// canonicalization on the special order so its enumerated short
/// vectors and the eventual `(β_s, β_t)` pair are reproducible. Two
/// independent L2 implementations agree on the abstract reduced
/// basis but may differ in the L2-output column order; without this
/// canonicalization step, the downstream
/// `enumerate_short_vectors` and `try_find_uv` produce a *different*
/// `(β_s, β_t)` than C ref does for the same input ideal — even when
/// L2 is bit-exact.
///
/// The canonicalization has two phases:
///
/// 1. **Column swap.** If `gram[0][0] == gram[2][2]`, swap col 1 ↔ col 2. Else
///    if `gram[0][0] == gram[3][3]`, swap col 1 ↔ col 3. Else if `gram[1][1] ==
///    gram[3][3]`, swap col 1 ↔ col 2 (same swap as the first case). The Gram
///    is updated correspondingly via the symmetric permutation `P^T G P`.
/// 2. **Sign flip.** If `basis[0][0] != basis[1][1]`, negate col 1. If
///    `basis[0][2] != basis[1][3]`, negate col 3. Each negation flips the sign
///    of the corresponding row + column of Gram.
///
/// The non-special-order branch (`is_special_order=false`) of the
/// C ref is empty, so this function is only called for `t = 0`.
fn post_lll_basis_treatment_special<const W: usize>(
    cols: &mut [Vector<W>; 4],
    gram: &mut Matrix<W>,
) {
    use crate::quaternions::linear::Vector;

    // Helper: swap columns `a` and `b` in the basis (cols are stored
    // column-major, so swapping `cols[a]` ↔ `cols[b]` swaps the cols).
    let swap_cols = |cols: &mut [Vector<W>; 4], a: usize, b: usize| {
        cols.swap(a, b);
    };
    // Helper: apply the symmetric permutation `P^T G P` for swapping
    // cols `a` ↔ `b` to the Gram matrix in place. This swaps row a ↔
    // row b AND col a ↔ col b. Equivalent to swapping `gram[a][k]` ↔
    // `gram[b][k]` for each k, then `gram[k][a]` ↔ `gram[k][b]` for
    // each k. The combined effect: every off-diagonal entry indexed
    // by (a, k) or (k, a) flips with its counterpart at (b, k) or
    // (k, b); the diagonal entries `gram[a][a]` and `gram[b][b]` swap.
    let swap_gram_rows_cols = |gram: &mut Matrix<W>, a: usize, b: usize| {
        for k in 0..4 {
            let tmp = gram[a][k];
            gram[a][k] = gram[b][k];
            gram[b][k] = tmp;
        }
        for k in 0..4 {
            let tmp = gram[k][a];
            gram[k][a] = gram[k][b];
            gram[k][b] = tmp;
        }
    };
    // Helper: negate column `j` of basis + corresponding row/col of
    // Gram (sign change is equivalent to multiplying both sides by
    // -1, which leaves diagonal entries unchanged but flips signs of
    // off-diagonal entries).
    let negate_col = |cols: &mut [Vector<W>; 4], gram: &mut Matrix<W>, j: usize| {
        let v = cols[j];
        cols[j] = Vector::new(
            v[0].wrapping_neg(),
            v[1].wrapping_neg(),
            v[2].wrapping_neg(),
            v[3].wrapping_neg(),
        );
        for k in 0..4 {
            gram[j][k] = gram[j][k].wrapping_neg();
        }
        for k in 0..4 {
            gram[k][j] = gram[k][j].wrapping_neg();
        }
    };

    // Phase 1: column reorder based on Gram diagonal patterns.
    if gram[0][0] == gram[2][2] {
        swap_cols(cols, 1, 2);
        swap_gram_rows_cols(gram, 1, 2);
    } else if gram[0][0] == gram[3][3] {
        swap_cols(cols, 1, 3);
        swap_gram_rows_cols(gram, 1, 3);
    } else if gram[1][1] == gram[3][3] {
        swap_cols(cols, 1, 2);
        swap_gram_rows_cols(gram, 1, 2);
    }

    // Phase 2: sign-flip cols based on basis-entry equality checks.
    if cols[0][0] != cols[1][1] {
        negate_col(cols, gram, 1);
    }
    if cols[2][0] != cols[3][1] {
        negate_col(cols, gram, 3);
    }
}

/// Enumerate the C-reference's filtered half-cube of integer
/// 4-tuples in `[-m, m]⁴`.
///
/// Returns the tuples in the same deterministic order as the C
/// reference's `enumerate_hypercube` (`dim2id2iso.c:270-376`).
/// Pinning the candidate order pins the first-success
/// `(β_s, β_t)` choice in [`try_find_uv`], which in turn pins
/// `e_pk` for KAT byte match.
///
/// Filters applied, in order:
///
/// * Half-cube iteration: walk only `x ≤ 0`, breaking each inner loop when the
///   leading-zero suffix would cross into the positive half. `±v` and `v` give
///   the same Gram-form value, so keeping just one representative halves the
///   candidate pool.
/// * Skip all-even tuples: `2·v` has Gram-form value `4·G(v, v)`, never smaller
///   than `G(v, v)` itself.
/// * Skip all-mult-of-3 tuples for the same reason.
/// * When `gram_has_i_symmetry` is set — i.e., the L2-reduced basis is `(γ, iγ,
///   β, iβ)` so that `G[0][0] = G[1][1]` and `G[2][2] = G[3][3]` — keep only
///   the `i`-orbit representative with the smallest lex rank in the `dim = 2m +
///   1` hypercube layout.
pub(super) fn enumerate_hypercube(m: i64, gram_has_i_symmetry: bool) -> Vec<[i64; 4]> {
    debug_assert!(m > 0);

    let dim = 2 * m + 1;
    let dim2 = dim * dim;
    let dim3 = dim2 * dim;

    let cap = (dim as usize).pow(4);
    let mut out = Vec::with_capacity(cap);

    for x in -m..=0 {
        for y in -m..=m {
            if x == 0 && y > 0 {
                break;
            }
            for z in -m..=m {
                if x == 0 && y == 0 && z > 0 {
                    break;
                }
                for w in -m..=m {
                    if x == 0 && y == 0 && z == 0 && w >= 0 {
                        break;
                    }

                    if (x | y | z | w) & 1 == 0 {
                        continue;
                    }
                    if x.rem_euclid(3) == 0
                        && y.rem_euclid(3) == 0
                        && z.rem_euclid(3) == 0
                        && w.rem_euclid(3) == 0
                    {
                        continue;
                    }

                    if gram_has_i_symmetry {
                        let check1 = (m + w) + dim * (m + z) + dim2 * (m + y) + dim3 * (m + x);
                        let check2 = (m - z) + dim * (m + w) + dim2 * (m - x) + dim3 * (m + y);
                        let check3 = (m + z) + dim * (m - w) + dim2 * (m + x) + dim3 * (m - y);
                        if !(check1 <= check2 && check1 <= check3) {
                            continue;
                        }
                    }

                    out.push([x, y, z, w]);
                }
            }
        }
    }

    out
}

impl<const W: usize> NrdBasis<W> {
    /// Enumerate non-zero lattice vectors within the box \[-m, m\]⁴
    /// from an L2-reduced basis, compute their degrees, and sort by
    /// norm.
    ///
    /// For NIST-I with m = 2, this produces up to
    /// (2·2+1)⁴ − 1 = 624 non-zero vectors.
    ///
    /// Generic over the working-width `W`. Use `W = 8` for typical
    /// commitment ideals (~2^133 norm); use `W = 16` for the wide
    /// response-phase intersection `I_inter ≈ 2^385` whose Gram
    /// matrix entries reach ~2^770.
    ///
    /// # Mirror of C reference's off-by-one
    ///
    /// The very last step here drops the last enumerated vector
    /// (`vectors.pop()`) before sorting. This is a deliberate
    /// reproduction of an off-by-one bug in the C reference's
    /// `enumerate_hypercube` (`dim2id2iso.c:449`,
    /// `return count - 1;`): after pushing N vectors into
    /// `small_vecs`, the C reference reports `count - 1` to the
    /// caller, so the last-enumerated vector is silently dropped
    /// before `qsort`. The KAT vectors shipped with SQIsign were
    /// generated by the C reference, so byte-equality with the KATs
    /// requires we drop the same vector here.
    ///
    /// Without the drop, our sorted batch contains one more entry
    /// than C ref's, shifting all sort indices by one and causing
    /// [`try_find_uv`] to pick a different `(β_s, β_t)` pair. On
    /// NIST-I that flips ~6 keygen KATs from passing to failing
    /// (and vice-versa), but, more importantly, makes the published
    /// `(u, v)` decomposition byte-different from C ref's, which
    /// then propagates through every downstream stage
    /// (`fixed_degree_isogeny`, the inner chain, the outer chain,
    /// the splitter) so the resulting verifying key fails to match the
    /// KAT vector at the byte level.
    fn enumerate_short_vectors(
        &self,
        ideal_norm: &BigInt<W>,
        lattice_denom: &BigInt<W>,
    ) -> Vec<ShortVectorCandidate> {
        let m = crate::params::FINDUV_BOX_SIZE;
        let denom_sq = lattice_denom.ct_mul(lattice_denom);
        let divisor = ideal_norm.ct_mul(&denom_sq);

        // Verify the lattice denom fits in BigInt<4> — we only check
        // existence here (not actually used below) to mirror the
        // pre-generic invariant.
        let Some(den_4) = lattice_denom.narrow_to::<4>() else {
            return Vec::new();
        };

        let width = (2 * m + 1) as usize;
        let mut vectors = Vec::with_capacity(width.pow(4) - 1);

        let need_remove_symmetry =
            self.gram()[0][0] == self.gram()[1][1] && self.gram()[3][3] == self.gram()[2][2];

        for [x_i, y_i, z_i, w_i] in enumerate_hypercube(m, need_remove_symmetry) {
            let x = [
                BigInt::<W>::from_i64(x_i),
                BigInt::<W>::from_i64(y_i),
                BigInt::<W>::from_i64(z_i),
                BigInt::<W>::from_i64(w_i),
            ];

            // nrd(β) · denom² = Σ x_i x_j G_{ij}.
            let nrd_scaled = self.eval_quadratic_form(&x);

            if bool::from(nrd_scaled.is_zero()) || bool::from(nrd_scaled.is_negative()) {
                continue;
            }

            // degree = nrd_scaled / (nrd(I) · denom²).
            let (degree_wide, rem) = nrd_scaled.div_rem(&divisor);
            if !bool::from(rem.is_zero()) {
                continue;
            }
            if bool::from(degree_wide.is_zero()) {
                continue;
            }
            let Some(degree_4) = degree_wide.narrow_to::<4>() else {
                continue;
            };
            let Some(degree) = IsogenyDegree::new_odd(*degree_4.as_limbs()) else {
                continue;
            };

            // β = Σ x_k · col_k.
            let coords: [BigInt<W>; 4] = core::array::from_fn(|row| {
                (0..4).fold(BigInt::<W>::ZERO, |acc, k| {
                    acc.ct_add(&x[k].ct_mul(&self.cols()[k][row]))
                })
            });

            // Narrow coordinates to BigInt<4>. After L2 reduction
            // with small coefficients this should always succeed.
            let narrow: [Option<BigInt<4>>; 4] =
                core::array::from_fn(|i| coords[i].narrow_to::<4>());
            let [Some(a), Some(b), Some(c), Some(d)] = narrow else {
                continue;
            };

            vectors.push(ShortVectorCandidate {
                elem: Element::<4>::new(
                    Coordinate::from_bigint(a),
                    Coordinate::from_bigint(b),
                    Coordinate::from_bigint(c),
                    Coordinate::from_bigint(d),
                    Denominator::from_bigint_unchecked(den_4),
                ),
                degree,
            });
        }

        let _ = width; // capacity hint only

        // Mirror C ref's `enumerate_hypercube` off-by-one at
        // `dim2id2iso.c:449` (`return count - 1;`). C ref kept `count`
        // vectors during enumeration but reports only `count - 1` to
        // the caller, so the last-enumerated vector is silently
        // dropped before `qsort`. The KAT vectors were generated by
        // the C reference, so byte-equality requires we drop the same
        // vector. (Without this drop, our sorted batch contains one
        // more entry than C ref's, shifting all subsequent indices
        // and causing `try_find_uv` to pick a different `(β_s, β_t)`
        // for many KATs.) See the function-level docs for the full
        // context.
        vectors.pop();
        debug_assert!(
            vectors.len() < width.pow(4),
            "post-pop vector count consistent with hypercube bound"
        );

        // Sort enumerated vectors by `degree` (= norm) ascending,
        // matching C ref's `qsort(small_vecs_and_norms, ...,
        // compare_vec_by_norm)` in `dim2id2iso.c:634`. Ties broken
        // by original enumeration order (Rust's `sort_by` is stable,
        // matching C ref's explicit `idx` tiebreaker).
        //
        // Without this sort, both impls produce the same set of
        // short vectors but in different orders → `try_find_uv`
        // selects a different first valid `(β_s, β_t)` pair → the
        // entire downstream Deuring correspondence diverges → keygen
        // vk bytes don't match C ref.
        vectors.sort_by(|a, b| {
            let al = a.degree.limbs();
            let bl = b.degree.limbs();
            // Compare from MSB to LSB (limbs are little-endian).
            for i in (0..al.len()).rev() {
                match al[i].cmp(&bl[i]) {
                    core::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            core::cmp::Ordering::Equal
        });

        // Stable sort by `degree`: `nrd(parent_ideal)` is constant
        // across this batch, so ordering by degree matches ordering
        // by `nrd(β)` exactly — no `f64` precision concerns. A
        // stable sort preserves insertion order for equal-degree
        // candidates, giving deterministic pair selection in
        // [`try_find_uv`]; determinism matters for signing because a
        // non-deterministic pair choice could leak information about
        // which short vectors matched.
        vectors.sort_by_key(|c| c.degree);
        vectors
    }
}

/// Try to find coprime odd degrees and matching `u`, `v` from a
/// pair of short-vector candidates enumerated from potentially
/// different extremal orders.
///
/// `(order1, parent_ideal1)` is the (extremal order, ideal) pair
/// that produced `sv1`; `(order2, parent_ideal2)` is the same for
/// `sv2`. For the `t = 0` single-order search both pairs are
/// identical; for the multi-order search they select different
/// rows of `ENDOMORPHISM_MATRICES` and different `nrd(J_t · I)` scaling
/// factors downstream. Carrying each β's origin through to the
/// resulting [`IdealFactor`] keeps the `nrd(β) = degree ·
/// nrd(parent_ideal)` invariant local to each factor.
///
/// Returns `None` when the pair fails any of the SuitableIdeals
/// conditions: non-coprime degrees, no solution to
/// `u·d₁ + v·d₂ = 2^f` with `u, v > 0`, or an exponent that would
/// push `e` out of range.
fn try_find_uv<const N: usize>(
    sv1: &ShortVectorCandidate,
    sv2: &ShortVectorCandidate,
    batch1: &ShortVectorBatch<N>,
    batch2: &ShortVectorBatch<N>,
    two_f: &BigInt<8>,
    f: TorsionExponent,
) -> Option<SuitableIdealResult<N>> {
    let d1 = &sv1.degree;
    let d2 = &sv2.degree;

    // Oddness is guaranteed by IsogenyDegree construction.
    // Check gcd(d₁, d₂) = 1 via widened BigInt.
    let d1_w = d1.to_bigint_wide();
    let d2_w = d2.to_bigint_wide();
    if d1_w.gcd(&d2_w) != BigInt::<8>::ONE {
        return None;
    }

    // Enumerate positive-integer solutions `(u, v)` to
    // `u·d₁ + v·d₂ = 2^f` along the line, starting at the smallest
    // `v` and walking `v += d₁` (correspondingly `u -= d₂`) — matches
    // C ref's `find_uv_from_lists` enumeration direction
    // (`dim2id2iso.c:404-429`):
    //
    //     v = (n · d₂⁻¹) mod d₁    (smallest non-negative v)
    //     while v < n / d₂:
    //         u = (n − v·d₂) / d₁
    //         if accept(u, v): return (u, v)
    //         v += d₁
    //
    // Earlier versions started at the smallest `u` and walked
    // `v -= d₁`, but that's the *opposite* enumeration order: it
    // accepts `(small u, large v)` first while C ref accepts
    // `(large u, small v)` first. With the same accept criterion
    // both directions terminate, but they pick different `(u, v)` —
    // so on KAT-aligned DRBG the chosen `(β_s, β_t)` diverges.
    let d2_inv = d2_w.invert_mod(&d1_w)?;
    let v0 = two_f.ct_mul(&d2_inv).ct_mod(&d1_w);
    let mut v = v0;
    let mut u = {
        let vd2 = v.ct_mul(&d2_w);
        if vd2 >= *two_f {
            return None;
        }
        let (u, rem) = two_f.ct_sub(&vd2).div_rem(&d1_w);
        if !bool::from(rem.is_zero()) || bool::from(u.is_negative()) {
            return None;
        }
        u
    };

    // Note: an earlier version "balance-biased" the start by jumping
    // to `u ≈ 2^{f/2}` to give `represent_integer` a larger search
    // space. C reference (`find_uv_from_lists` in `dim2id2iso.c:382`)
    // walks the full line from the natural starting point with no
    // such bias, and the chosen `(u, v)` matters for interop with
    // KAT vectors — different starting points → different first
    // success → different `(β_s, β_t)` selection → different chain
    // codomain → different `e_pk`. Removed.
    let _ = f;

    // Cap the line-walk. Without a cap, the `MIN_U_ODD_BITS`
    // filter below can force the loop to step `(u, v)` forward
    // by `(+d_2, -d_1)` for pathological `d_1, d_2` pairs where
    // no step in the visible portion of the line has `u_odd ≥
    // 2^{MIN_U_ODD_BITS}`. For `d_1 = 1` and `v_0 ≈ 2^f`, the
    // natural termination condition `v ≤ d_1` requires
    // `~2^{f}` steps — unbounded in practice. Give up after
    // `MAX_WALK_STEPS` and let the caller try the next pair.
    const MAX_WALK_STEPS: u32 = 10_000;
    let mut steps = 0u32;
    loop {
        steps += 1;
        if steps > MAX_WALK_STEPS {
            return None;
        }
        if !bool::from(u.is_zero()) && !bool::from(v.is_zero()) {
            // Factor out the 2-adic part of `gcd(u, v)`, matching the
            // C reference (`dim2id2iso.c:833`). The spec writes
            // `v_2(u)` in Algorithm 3.16 line 14, but that is only
            // equivalent to `v_2(gcd(u, v))` when `v_2(u) ≤ v_2(v)`.
            let e_val = u.gcd(&v).trailing_zeros();
            // No `e_val < 2` rejection: matches the C reference's
            // `find_uv_from_lists` (`dim2id2iso.c:397-483`), which
            // returns the first `(u, v)` from the line walk and
            // dispatches downstream on `e = f − v_2(gcd(u, v))`.
            // [`LeftIdeal::to_isogeny`] now selects between
            // [`surfaces::Kernel::isogeny`] (kernel `2^(e+2)`,
            // `e ≤ f − 2`) and
            // [`surfaces::Kernel::isogeny_no_extra_torsion`] (kernel
            // `2^e`, `e ∈ {f − 1, f}`) based on `sui.e`.
            // Mirror C ref's `(ibz_get(u) != 0 && ibz_get(v) != 0)`
            // guard at `dim2id2iso.c:421`. `ibz_get` returns an
            // `int32_t` packed as `(sign_bit << 31) | (low_31_bits)`
            // (`intbig.c:399-410`); for non-negative `u, v` (which
            // is the case on this enumeration line) this is just
            // the low 31 bits, so `ibz_get(u) == 0` ⟺ `v_2(u) >= 31`.
            // Per the C ref comment, the filter "removes weird
            // cases where u, v have big power of two".
            //
            // Without this filter, our line walk would accept
            // `(u, v)` pairs that C ref rejects, leading to
            // different `(β_s, β_t)` selections than C ref on
            // KAT-aligned DRBG sequences.
            if u.trailing_zeros() >= 31 || v.trailing_zeros() >= 31 {
                // Advance `v += d₁`, `u -= d₂`. Stop if u would go
                // non-positive.
                v = v.ct_add(&d1_w);
                if u <= d2_w {
                    return None;
                }
                u = u.ct_sub(&d2_w);
                continue;
            }
            if let Ok(e) = TorsionExponent::try_from(f.value() - e_val) {
                if let (Some(u_narrow), Some(v_narrow)) =
                    ((u >> e_val).narrow(), (v >> e_val).narrow())
                {
                    return Some(SuitableIdealResult {
                        u: u_narrow,
                        v: v_narrow,
                        e,
                        factor1: IdealFactor {
                            order: batch1.order,
                            beta: ShortVector(sv1.elem),
                            degree: *d1,
                            parent_ideal: batch1.parent_ideal,
                        },
                        factor2: IdealFactor {
                            order: batch2.order,
                            beta: ShortVector(sv2.elem),
                            degree: *d2,
                            parent_ideal: batch2.parent_ideal,
                        },
                    });
                }
            }
        }

        // Advance to the next solution on the line: `v += d₁`,
        // `u -= d₂`. Stop when `u` would go non-positive (matches
        // C ref's `while (cmp < 0)` exit at v >= n/d₂).
        v = v.ct_add(&d1_w);
        if u <= d2_w {
            return None;
        }
        u = u.ct_sub(&d2_w);
    }
}

impl<const N: usize> LeftIdeal<N> {
    /// [Alg. 3.16]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.16
    pub(crate) fn suitable_ideals(&self) -> Option<SuitableIdealResult<N>> {
        // Internal arithmetic widens to 8 limbs for the L2 reduction
        // and short-vector enumeration. Inputs with `N > 8` would
        // require a wider `NrdBasis`; reject at compile time.
        const { assert!(N <= 8, "suitable_ideals supports N ≤ 8") };
        let f = TorsionExponent::FULL;
        let two_f = BigInt::<8>::ONE << f.value();

        // Phase 1: for each of the seven extremal orders O_t, build
        // the corresponding ideal in which β is enumerated:
        //
        // - `t = 0`: the caller-supplied ideal `self`. Short vectors β ∈ `self` have
        //   `nrd(β) = degree · nrd(self)`.
        // - `t > 0`: the pushforward `J_t · self` (see [§3.1.6.1][§3.1.6.1]), where
        //   `J_t = connecting_ideal(t)` is the precomputed left-O_0 ideal with
        //   right-order O_t. `pushforward` returns the left-O_t ideal `J_t^{-1} · (J_t
        //   ∩ self)`, which has the same norm as `self` but lives in a different
        //   lattice — β's enumerated here act on `E_t` via
        //   `ENDOMORPHISM_MATRICES[t][*]`.
        //
        // The C reference (`dim2id2iso.c:535-609`) does an equivalent
        // multi-order L2 reduction + enumeration via
        // `conj(I_reduced) · J_t`, which produces the same degrees
        // and a right-O_t lattice; the pushforward shape here
        // matches the spec literal and avoids the delta-transport
        // step. See [§3.1.7.2].
        //
        // [§3.1.6.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.6.1
        // [§3.1.7.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.7.2

        // Per-order enumeration: compute an L2-reduced basis and
        // short-vector list for each extremal order, then stage a
        // `ShortVectorBatch` pointing at the matching parent ideal.
        let mut batches: [Option<ShortVectorBatch<N>>; NUM_EXTREMAL_ORDERS] = Default::default();
        let mut short_vecs_per_order: [Vec<ShortVectorCandidate>; NUM_EXTREMAL_ORDERS] =
            Default::default();

        // C ref's multi-order recipe (`dim2id2iso.c:617-666`):
        // 1. After LLL on self, let δ = first post-L² basis col.
        // 2. k = nrd(δ_num) / (denom_self² · N(self)) — integer "abstract norm" of the
        //    equivalent ideal class.
        // 3. reduced_id.lattice = self.lattice · conj(δ)/N(self) (a fractional
        //    left-O_0-ideal in B; integer-col covolume at denom 4N is 64·N^5·k).
        // 4. conj_reduced_id = conjugate(reduced_id).
        // 5. ideal[t] = conj_reduced_id · J_t. Norm = k · N(J_t). Integer-col covolume
        //    at denom 8N is 1024·N^4·(k·N_J)².
        //
        // We pass these *exact* covolume formulas as modular-HNF
        // moduli rather than the `det(first 4 cols)` heuristic, which
        // would give a strict multiple of canonical covolume and
        // embed the lattice in a coarser sublattice (failing
        // enumerate's divisor check).
        const W2: usize = 60;
        // (conj_lat, k_norm, n_self, denom_self, conj_delta) — extended
        // to also carry `denom_self_w2` and `conj_delta` for the
        // cross-order beta post-processing below (C ref's
        // `dim2id2iso.c:651-672` `quat_alg_mul(beta, &delta, beta)
        // + quat_alg_normalize + quat_alg_conj` for `j != 0`).
        let mut conj_reduced_state: Option<(
            Lattice<W2>,
            BigInt<W2>,
            BigInt<W2>,
            BigInt<W2>,
            Element<W2>,
        )> = None;

        for t in 0..NUM_EXTREMAL_ORDERS {
            // Build the parent ideal for this order.
            let parent_ideal_t = if t == 0 {
                *self
            } else {
                let Some((ref conj_lat, ref k_norm, ref n_self_w2, ..)) = conj_reduced_state else {
                    continue;
                };
                let j_t = LeftIdeal::<4>::connecting(t)
                    .expect("t < NUM_EXTREMAL_ORDERS by loop bound")
                    .widen::<W2>();
                let j_t_lat: Lattice<W2> = (*j_t.lattice()).into();
                let j_t_norm: BigInt<W2> = *j_t.norm();
                let prod_norm = k_norm.ct_mul(&j_t_norm);
                // Use `Lattice::product`'s built-in `det(first 4 cols)`
                // modulus rather than the precomputed
                // `1024 · N^4 · (k·N_J)²` covolume formula. The precomputed
                // formula was 4× the canonical covolume in the default
                // (denom_self=2, denom_J=2) case — and Selkie's
                // `from_hnf_columns_mod` constructs the lattice
                // `⟨input cols⟩ + D·Z^4`, so a too-large `D` not in the
                // lattice yields a coarser sublattice containing scalars
                // like `(2, 0, 0, 0)`. That degenerate generator, fed
                // through `class_gram = 2·nrd/(denom²·N)`, zeros row/col 0
                // and L²-LLL never terminates (KAT 053 t=5 hang;
                // C-ref's `quat_lll_core` also hangs on the same input).
                //
                // C-ref's `quat_lideal_lideal_mul_reduced` (`lll/lll_applications.c:38`)
                // calls `quat_lattice_mul`, which uses `|det(first 4
                // generators)|` (`lattice.c:231-233`) as the HNF modulus —
                // a value guaranteed to live in the lattice. `Lattice::product`
                // mirrors that recipe.
                let _ = n_self_w2;
                let _ = j_t_norm;
                let prod_lat = conj_lat.product(&j_t_lat).reduce_denom();
                let parent_o0 = *EXTREMAL_ORDERS[0].widen::<W2>().order();
                let ideal_w2 = LeftIdeal::<W2>::from_parts(prod_lat, prod_norm, parent_o0);
                match ideal_w2.narrow_to::<N>() {
                    Some(p) => p,
                    None => continue,
                }
            };

            // Widen the parent ideal's lattice to width 16 for L2
            // reduction. Width 16 fits the response-phase
            // intersection ideal (`nrd(I) ≈ 2^385`, lattice cols
            // ~2^385) — its Gram matrix entries can reach ~2^770,
            // which overflows the previous width-8 working space.
            // For the commitment path (norm ~2^133) width 16 has
            // ample slack.
            const W: usize = 16;
            let lattice_n: Lattice<N> = (*parent_ideal_t.lattice()).into();
            let cols_n = lattice_n.basis().columns();
            let cols_w: [Vector<W>; 4] = core::array::from_fn(|j| cols_n[j].widen());
            let denom_w: BigInt<W> = (*lattice_n.denom()).widen();
            let norm_w: BigInt<W> = (*parent_ideal_t.norm()).widen();

            // Feed L2 the *class gram* (= 2·nrd_bilinear / (denom²·ideal_norm))
            // rather than the raw NRD gram, matching C ref's
            // `quat_lideal_class_gram` + `quat_lll_core` pipeline
            // (`the-sqisign/src/quaternion/ref/generic/ideal.c:237` →
            // `lll/lll_applications.c:11-25`). L2 is algebraically
            // scale-invariant in its decisions, but the DPE
            // floating-point representation has finite precision —
            // borderline swap/size-reduction decisions can flip when
            // the gram values are at different magnitudes. Running
            // L2 on the same class gram C ref runs it on guarantees
            // bit-exact decision agreement.
            //
            // After L2, recompute the NRD gram from the post-L2
            // cols so `enumerate_short_vectors` (which uses NRD form
            // + divisor = ideal_norm·denom²) keeps working unchanged.
            let nrd_pre = NrdBasis::new(cols_w);
            let class_gram = {
                let two = BigInt::<W>::from_u64(2);
                let denom_sq = denom_w.ct_mul(&denom_w);
                let class_divisor = denom_sq.ct_mul(&norm_w);
                let mut g = Matrix::<W>::ZERO;
                for i in 0..4 {
                    for j in 0..4 {
                        let traced = nrd_pre.gram()[i][j].ct_mul(&two);
                        let (q, _rem) = traced.div_rem(&class_divisor);
                        g[i][j] = q;
                    }
                }
                g
            };
            let class_basis = NrdBasis::from_cols_and_gram(cols_w, class_gram).l2_reduce();

            let post_l2_cols = *class_basis.cols();
            let nrd_basis = NrdBasis::new(post_l2_cols);

            // Apply C ref's `post_LLL_basis_treatment` for the
            // "special" order t=0 only. Canonicalizes the L2-reduced
            // basis by column swaps + sign flips based on Gram
            // diagonal patterns. See
            // `the-sqisign/src/id2iso/ref/lvlx/dim2id2iso.c:213-268`.
            let nrd_basis = if t == 0 {
                let mut cols = *nrd_basis.cols();
                let mut gram = *nrd_basis.gram();
                post_lll_basis_treatment_special::<W>(&mut cols, &mut gram);
                NrdBasis::from_cols_and_gram(cols, gram)
            } else {
                nrd_basis
            };
            short_vecs_per_order[t] = nrd_basis.enumerate_short_vectors(&norm_w, &denom_w);
            batches[t] = Some(ShortVectorBatch {
                order: &EXTREMAL_ORDERS[t],
                parent_ideal: parent_ideal_t,
            });

            // After t=0 L², populate conj_reduced_state for t > 0.
            if t == 0 {
                let cols_t0 = post_l2_cols;
                let denom_t0_w2: BigInt<W2> = denom_w.widen();
                let norm_t0_w2: BigInt<W2> = norm_w.widen();

                let p_w2: BigInt<W2> = {
                    let p8 = P_WIDE;
                    let mut limbs = [0u64; W2];
                    limbs[..8].copy_from_slice(p8.as_limbs());
                    BigInt::from_sign_and_limbs(0, limbs)
                };

                let dx: BigInt<W2> = cols_t0[0][0].widen();
                let dy: BigInt<W2> = cols_t0[0][1].widen();
                let dz: BigInt<W2> = cols_t0[0][2].widen();
                let dw: BigInt<W2> = cols_t0[0][3].widen();

                // nrd(δ_num) = dx² + dy² + p·(dz² + dw²).
                let nrd_delta_num = dx
                    .ct_mul(&dx)
                    .ct_add(&dy.ct_mul(&dy))
                    .ct_add(&p_w2.ct_mul(&dz.ct_mul(&dz).ct_add(&dw.ct_mul(&dw))));

                // k = nrd(δ_num) / (denom² · N(self)). Integer for valid input.
                let denom_sq = denom_t0_w2.ct_mul(&denom_t0_w2);
                let div = denom_sq.ct_mul(&norm_t0_w2);
                let (k_norm, rem) = nrd_delta_num.div_rem(&div);
                if !bool::from(rem.is_zero()) {
                    continue;
                }

                // conj(δ) as Element<W2> with denom = denom_self · N(self).
                let conj_delta = Element::<W2>::new(
                    Coordinate::from_bigint(dx),
                    Coordinate::from_bigint(dy.wrapping_neg()),
                    Coordinate::from_bigint(dz.wrapping_neg()),
                    Coordinate::from_bigint(dw.wrapping_neg()),
                    Denominator::from_bigint_unchecked(denom_t0_w2.ct_mul(&norm_t0_w2)),
                );

                let self_lat_w2: Lattice<W2> = {
                    let widened = self.widen::<W2>();
                    (*widened.lattice()).into()
                };

                // modulus_inner = (4N)^4·k²/4 = 64·N^4·k²
                // (integer-col covolume of reduced_id at denom 4N).
                let n_sq = norm_t0_w2.ct_mul(&norm_t0_w2);
                let n4 = n_sq.ct_mul(&n_sq);
                let k_sq = k_norm.ct_mul(&k_norm);
                let modulus_inner = BigInt::<W2>::from_u64(64).ct_mul(&n4).ct_mul(&k_sq);

                let reduced_id_lat_hnf = self_lat_w2
                    .alg_elem_mul_with_modulus(&conj_delta, &modulus_inner)
                    .reduce_denom();
                let reduced_id_lat: Lattice<W2> = reduced_id_lat_hnf.into();
                conj_reduced_state = Some((
                    reduced_id_lat.conjugate(),
                    k_norm,
                    norm_t0_w2,
                    denom_t0_w2,
                    conj_delta,
                ));
            }
        }

        // Phase 2: iterate (s, t) pairs with `t ≥ s` (matching the
        // C ref's diag filter) and search for a viable (β_s, β_t).
        // First success wins. `try_find_uv` handles all the coprime
        // and `u·d_s + v·d_t = 2^e` checks.
        let mut _pairs_tried = 0u64;
        for s in 0..NUM_EXTREMAL_ORDERS {
            let Some(batch_s) = batches[s].as_ref() else {
                continue;
            };
            for t in s..NUM_EXTREMAL_ORDERS {
                let Some(batch_t) = batches[t].as_ref() else {
                    continue;
                };
                let svs_s = &short_vecs_per_order[s];
                let svs_t = &short_vecs_per_order[t];

                let same_batch = s == t;
                for (i, sv1) in svs_s.iter().enumerate() {
                    // When s == t we iterate the upper-triangular
                    // half to avoid pairing a candidate with itself
                    // or double-counting (β_a, β_b) / (β_b, β_a).
                    // When s != t both orderings are distinct
                    // enumerations so we iterate the full cross
                    // product.
                    let inner_start = if same_batch { i } else { 0 };
                    for sv2 in &svs_t[inner_start..] {
                        _pairs_tried += 1;
                        if let Some(result) = try_find_uv(sv1, sv2, batch_s, batch_t, &two_f, f) {
                            // Cross-order beta post-processing
                            // (C ref `dim2id2iso.c:651-672`):
                            //
                            //   if (j_i != 0) {
                            //       beta_i = delta · beta_i  (quaternion mul)
                            //       beta_i = normalize(beta_i)
                            //       beta_i = conj(beta_i)
                            //   }
                            //
                            // where `delta` is `conj_delta` with denom
                            // adjusted to `denom_self · k_norm`. This
                            // maps `beta_i ∈ conj(reduced_id) · J_t`
                            // (where it was enumerated) back to `O_t`
                            // (where downstream `action_matrix(O_t)`
                            // expects it). Without this, the
                            // alternate-order action_matrix call
                            // returns None and keygen retries with
                            // wrong DRBG offset.
                            //
                            // No transformation when both `s == 0`
                            // and `t == 0` (the special-order path
                            // already produces β in O_0).
                            let mut result = result;
                            if let (
                                true,
                                Some((_, k_norm, _n_self_w2, denom_self_w2, conj_delta)),
                            ) = (s != 0 || t != 0, conj_reduced_state.as_ref())
                            {
                                // delta_pp: same coords as conj_delta,
                                // denom = denom_self · k_norm
                                // (instead of denom_self · n_self).
                                let mut delta_pp = *conj_delta;
                                let denom_pp = denom_self_w2.ct_mul(k_norm);
                                delta_pp.denom = Denominator::from_bigint_unchecked(denom_pp);

                                let transform = |beta4: &Element<4>| -> Option<Element<4>> {
                                    let beta_w = Element::<W2>::new(
                                        Coordinate::from_bigint(beta4.a.as_bigint().widen::<W2>()),
                                        Coordinate::from_bigint(beta4.b.as_bigint().widen::<W2>()),
                                        Coordinate::from_bigint(beta4.c.as_bigint().widen::<W2>()),
                                        Coordinate::from_bigint(beta4.d.as_bigint().widen::<W2>()),
                                        Denominator::from_bigint_unchecked(
                                            BigInt::<4>::from(beta4.denom).widen::<W2>(),
                                        ),
                                    );
                                    let prod = delta_pp.mul_direct(&beta_w);
                                    let mut prod = prod;
                                    prod.normalize();
                                    let conjugated = prod.conjugate();
                                    conjugated.narrow_to::<4>()
                                };

                                if s != 0 {
                                    let new_beta = transform(&result.factor1.beta.0)?;
                                    result.factor1.beta = ShortVector(new_beta);
                                }
                                if t != 0 {
                                    let new_beta = transform(&result.factor2.beta.0)?;
                                    result.factor2.beta = ShortVector(new_beta);
                                }
                            }
                            return Some(result);
                        }
                    }
                }
            }
        }

        None
    }
}
