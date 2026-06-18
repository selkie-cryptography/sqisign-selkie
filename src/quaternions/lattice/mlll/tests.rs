//! Tests for [`super::Generators`] MLLL reduction.
//!
//! The identity smoke exercises the no-insertion path. The
//! `mlll_preserves_lattice_*` tests are differential checks against the
//! trusted, canonical HNF: MLLL must return a basis of the *same* lattice its
//! generators span, so `HNF(output) == HNF(inputs)`. These exercise the
//! deep-insertion and `b_κ=0` rank-deficiency paths on redundant inputs.

use super::*;
use crate::quaternions::linear::Matrix;

/// Returns whether every column of `inputs` lies in the integer span of the
/// four columns `out`: solve `out·x = v` via `x = adj(out)·v / det(out)`, which
/// is integral iff `det(out)` divides every component of `adj(out)·v`. This is
/// the definitive lattice-preservation check, independent of HNF.
fn output_spans_inputs<const N: usize>(out: &[Vector<N>; 4], inputs: &[Vector<N>]) -> bool {
    let bo = Matrix::<N>::from_columns(out);
    let det = bo.det();
    if bool::from(det.is_zero()) {
        return false;
    }
    let adj = bo.adjugate();
    inputs.iter().all(|v| {
        let x = adj.eval(v);
        (0..4).all(|c| bool::from(x[c].vt_div_rem(&det).1.is_zero()))
    })
}

/// Returns a small signed `BigInt<16>` in `[-9, 9]` from `rng`.
fn small16(rng: &mut Lcg) -> BigInt<16> {
    BigInt::from((rng.next_u64() % 19) as i64 - 9)
}

/// MLLL must preserve the lattice: every input generator lies in the span of
/// the reduced output. Stresses dense, imbalanced two-sublattice 8-column sets
/// (a `2^40`-scaled dense sublattice concatenated with a small dense one) — the
/// structure real dual-sums have.
#[test]
fn mlll_preserves_span_dense_imbalanced() {
    type Vw = Vector<16>;
    let big = BigInt::<16>::ONE << 40;
    let mut rng = Lcg(0x1234_5678_9ABC_DEF0);

    for _ in 0..80 {
        let mut cols = [Vw::ZERO; 8];
        for slot in cols.iter_mut().take(4) {
            *slot = Vw::new(
                small16(&mut rng).ct_mul(&big),
                small16(&mut rng).ct_mul(&big),
                small16(&mut rng).ct_mul(&big),
                small16(&mut rng).ct_mul(&big),
            );
        }
        for slot in cols.iter_mut().skip(4) {
            *slot = Vw::new(
                small16(&mut rng),
                small16(&mut rng),
                small16(&mut rng),
                small16(&mut rng),
            );
        }

        let out = Generators::<16, 8>::new(cols).mlll_reduce();

        assert!(
            output_spans_inputs(&out, &cols),
            "MLLL output does not span its inputs — span lost"
        );
    }
}

/// Width for the test generators.
type W = BigInt<8>;

/// Vector counterpart of [`W`].
type V = Vector<8>;

/// Builds a width-8 integer from an `i64`.
fn w(v: i64) -> W {
    W::from(v)
}

/// Asserts MLLL returns a basis of the same lattice its `G` generators span,
/// by comparing the canonical HNF of the output against that of the inputs.
fn assert_mlll_preserves_lattice<const N: usize, const G: usize>(cols: [Vector<N>; G]) {
    let inputs = cols.to_vec();
    let expected = Matrix::<N>::from_hnf_columns(&inputs);

    let reduced = Generators::<N, G>::new(cols).mlll_reduce();
    let got = Matrix::<N>::from_hnf_columns(&reduced);

    assert_eq!(
        got, expected,
        "MLLL output (G={G}) spans a different lattice than its inputs"
    );
}

/// The standard already-reduced basis `{1, i, j, k}` is a fixed point of
/// MLLL: no size-reduction, no insertion, nrd-Gram diagonal `[1, 1, p, p]`.
#[test]
fn mlll_identity_g4_is_fixed_point() {
    let cols = [
        V::new(w(1), w(0), w(0), w(0)),
        V::new(w(0), w(1), w(0), w(0)),
        V::new(w(0), w(0), w(1), w(0)),
        V::new(w(0), w(0), w(0), w(1)),
    ];

    let reduced = Generators::<8, 4>::new(cols).mlll_reduce();
    let gram = Generators::<8, 4>::new(reduced).gram;

    let p: W = crate::quaternions::precomputed::P_WIDE;
    assert_eq!(gram[0][0], W::ONE, "nrd(b0) != 1");
    assert_eq!(gram[1][1], W::ONE, "nrd(b1) != 1");
    assert_eq!(gram[2][2], p, "nrd(b2) != p");
    assert_eq!(gram[3][3], p, "nrd(b3) != p");
}

/// Five generators of `Z⟨1, i, j, k⟩` with one dependent (`1 + i`): the
/// dependent generator must collapse, leaving the standard lattice.
#[test]
fn mlll_preserves_lattice_g5_redundant() {
    assert_mlll_preserves_lattice([
        V::new(w(1), w(0), w(0), w(0)),
        V::new(w(0), w(1), w(0), w(0)),
        V::new(w(0), w(0), w(1), w(0)),
        V::new(w(0), w(0), w(0), w(1)),
        V::new(w(1), w(1), w(0), w(0)),
    ]);
}

/// Eight generators of `Z⟨1, i, j, k⟩` (the four basis vectors plus four cyclic
/// pairwise sums): four dependent generators must collapse.
#[test]
fn mlll_preserves_lattice_g8_redundant() {
    assert_mlll_preserves_lattice([
        V::new(w(1), w(0), w(0), w(0)),
        V::new(w(0), w(1), w(0), w(0)),
        V::new(w(0), w(0), w(1), w(0)),
        V::new(w(0), w(0), w(0), w(1)),
        V::new(w(1), w(1), w(0), w(0)),
        V::new(w(0), w(1), w(1), w(0)),
        V::new(w(0), w(0), w(1), w(1)),
        V::new(w(1), w(0), w(0), w(1)),
    ]);
}

/// A non-trivial lattice `2Z × 3Z × 5Z × 7Z` given by five generators
/// (one dependent), to check insertion/collapse on a coarser lattice.
#[test]
fn mlll_preserves_lattice_g5_scaled() {
    assert_mlll_preserves_lattice([
        V::new(w(2), w(0), w(0), w(0)),
        V::new(w(0), w(3), w(0), w(0)),
        V::new(w(0), w(0), w(5), w(0)),
        V::new(w(0), w(0), w(0), w(7)),
        V::new(w(2), w(3), w(0), w(0)),
    ]);
}

/// Skewed, non-orthogonal generators with a large off-diagonal coefficient,
/// forcing genuine size-reduction and at least one deep insertion before the
/// dependent generator collapses.
#[test]
fn mlll_preserves_lattice_g6_skewed() {
    assert_mlll_preserves_lattice([
        V::new(w(1), w(0), w(0), w(0)),
        V::new(w(50), w(1), w(0), w(0)),
        V::new(w(0), w(0), w(1), w(0)),
        V::new(w(0), w(0), w(13), w(1)),
        V::new(w(50), w(1), w(1), w(0)),
        V::new(w(1), w(0), w(13), w(1)),
    ]);
}

/// Imbalanced 8 generators = two *distinct* rank-4 sublattice bases
/// concatenated (`2^40·Z⁴` and a unimodular spanning set of `Z⁴`) — the
/// structure real dual-sums have, unlike the redundant-single-lattice cases
/// above. Their span is `Z⁴`.
#[test]
fn mlll_preserves_lattice_g8_imbalanced() {
    let big = W::ONE << 40;
    assert_mlll_preserves_lattice([
        V::new(big, w(0), w(0), w(0)),
        V::new(w(0), big, w(0), w(0)),
        V::new(w(0), w(0), big, w(0)),
        V::new(w(0), w(0), w(0), big),
        V::new(w(1), w(0), w(0), w(0)),
        V::new(w(1), w(1), w(0), w(0)),
        V::new(w(1), w(0), w(1), w(0)),
        V::new(w(1), w(0), w(0), w(1)),
    ]);
}

/// Minimal SplitMix64-style PRNG for deterministic, reproducible test inputs.
struct Lcg(u64);

impl Lcg {
    /// Returns the next pseudo-random word.
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Returns a small signed coordinate in `[-9, 9]`.
    fn small(&mut self) -> W {
        W::from((self.next_u64() % 19) as i64 - 9)
    }

    /// Returns a signed `~2^bits`-magnitude integer (fills `bits/64 + 1`
    /// limbs).
    fn wide<const M: usize>(&mut self, bits: u32) -> BigInt<M> {
        let n = (bits / 64) as usize + 1;
        let mut limbs = [0u64; M];
        for slot in limbs.iter_mut().take(n.min(M)) {
            *slot = self.next_u64();
        }
        let v = BigInt::from_sign_and_limbs(0, limbs);
        if self.next_u64() & 1 == 1 {
            v.wrapping_neg()
        } else {
            v
        }
    }
}

/// Differential sweep: 100 random rank-4 lattices, each given by 8 generators
/// (4 random independent + 4 random integer combinations), must round-trip
/// through MLLL to the same lattice (HNF-equal). Catches insertion/collapse
/// edge cases the structured tests miss.
#[test]
fn mlll_preserves_lattice_random_g8() {
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);

    for _ in 0..100 {
        let base: [V; 4] =
            core::array::from_fn(|_| V::new(rng.small(), rng.small(), rng.small(), rng.small()));

        if bool::from(Matrix::from_columns(&base).det().is_zero()) {
            continue;
        }

        let mut cols = [V::ZERO; 8];
        cols[0..4].copy_from_slice(&base);
        for slot in cols[4..].iter_mut() {
            let a = (rng.next_u64() % 4) as usize;
            let b = (rng.next_u64() % 4) as usize;
            *slot = base[a] + base[b];
        }

        assert_mlll_preserves_lattice(cols);
    }
}

/// The real ideal-multiplication generator count: 16 generators (4 wide random
/// independent + 12 integer combinations) of a rank-4 lattice round-trip
/// through MLLL with multi-limb coordinates.
#[test]
fn mlll_preserves_lattice_g16_wide() {
    type Wide = Vector<32>;

    let mut rng = Lcg(0x00C0_FFEE_1234_5678);
    let base: [Wide; 4] = core::array::from_fn(|_| {
        Wide::new(rng.wide(120), rng.wide(120), rng.wide(120), rng.wide(120))
    });

    let mut cols = [Wide::ZERO; 16];
    cols[0..4].copy_from_slice(&base);
    for slot in cols[4..].iter_mut() {
        let a = (rng.next_u64() % 4) as usize;
        let b = (rng.next_u64() % 4) as usize;
        *slot = base[a] + base[b];
    }

    assert_mlll_preserves_lattice(cols);
}
