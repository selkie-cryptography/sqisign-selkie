//! Width-4 specializations of [`LeftIdeal<4>`][super::LeftIdeal]:
//! the [`new`][LeftIdeal::new] constructor at storage width 4,
//! the [`connecting`][LeftIdeal::connecting] precomputed
//! connecting-ideal lookup,
//! the [`random_prime_norm`][LeftIdeal::random_prime_norm] and
//! [`random_norm`][LeftIdeal::random_norm] sampling routines, and the
//! [`generator`][LeftIdeal::generator] small-generator finder
//! ([Alg. 3.8]).
//!
//! [Alg. 3.8]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.8

use core::array;

use rand_core::{OsRng, RngCore};

use super::{
    super::{
        super::{
            algebra::{Coordinate, Denominator, Element},
            bigint::BigInt,
            linear::{Matrix, Vector},
            precomputed::{
                CONNECTING_IDEAL_NORMS, CONNECTING_IDEAL_X, CONNECTING_IDEAL_Y,
                NUM_EXTREMAL_ORDERS, STANDARD_ORDER,
            },
        },
        ExtremalOrder, HnfLattice, Lattice, Order,
    },
    LeftIdeal,
};

impl LeftIdeal<4> {
    /// Returns the precomputed connecting ideal `J_t` for extremal
    /// order index `t`, or `None` if `t >= NUM_EXTREMAL_ORDERS`.
    ///
    /// For `t = 0` returns `O₀` itself (norm 1, `x = y = 1`), which
    /// lets the alternate-order search in
    /// [`LeftIdeal::suitable_ideals`] treat `t = 0` uniformly with
    /// `t > 0`.
    ///
    /// The basis HNF leading entries are `2·N` (not `N`): at denom 2
    /// this gives the affine elements `α₀ = N` and `α₁ = N·i`. Halving
    /// these would represent `N/2`, which is not in `O₀` for an
    /// odd-norm ideal. Compare with [`CONNECTING_IDEAL_NORMS`]
    /// (which stores the reduced norm `N`).
    ///
    /// The constants `(N, x, y)` come from the precomputed tables in
    /// [`crate::quaternions::precomputed`]; the
    /// [`From<Lattice<4>>`](HnfLattice) conversion to
    /// [`HnfLattice<4>`] is idempotent here — computing the HNF
    /// again just re-validates it.
    ///
    /// See [§3.1.7.2] of the spec for the connecting-ideal construction.
    ///
    /// [§3.1.7.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.7.2
    #[doc(alias = "connecting_ideal")]
    #[must_use]
    pub fn connecting(t: usize) -> Option<Self> {
        if t >= NUM_EXTREMAL_ORDERS {
            return None;
        }
        let norm = CONNECTING_IDEAL_NORMS[t];
        let x = CONNECTING_IDEAL_X[t];
        let y = CONNECTING_IDEAL_Y[t];
        // Columns are basis vectors (α₀, α₁, α₂, α₃) in the {1, i, j, k}
        // basis; rows are components. Pre-denom basis:
        //   α₀ = 2N,    α₁ = 2N·i,    α₂ = x·i + j,   α₃ = y + k,
        // after dividing by denom = 2 yields (N, N·i, (x·i + j)/2, (y + k)/2).
        // For t = 0 (N = 1) this is the standard order basis
        // (1, i, (i+j)/2, (1+k)/2).
        let two_norm = norm.ct_add(&norm);
        let basis = Matrix::from_rows(
            Vector::new(two_norm, BigInt::ZERO, BigInt::ZERO, y),
            Vector::new(BigInt::ZERO, two_norm, x, BigInt::ZERO),
            Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ONE, BigInt::ZERO),
            Vector::new(BigInt::ZERO, BigInt::ZERO, BigInt::ZERO, BigInt::ONE),
        );
        let denom = BigInt::<4>::from_u64(2);
        let hnf = HnfLattice::from(Lattice::new(basis, denom));
        Some(Self::from_parts(hnf, norm, *STANDARD_ORDER.order()))
    }

    /// Creates the left ideal I = O⟨α, N⟩ = Oα + ON at width 4.
    ///
    /// Uses [`Element<4>::mul`] which widens to `BigInt<8>`
    /// internally, so this is the safe choice for small moduli
    /// (up to ~128 bits) where direct multiplication would overflow.
    /// For wider widths, use
    /// [`LeftIdeal::from_generator`](LeftIdeal::from_generator).
    ///
    /// See [§3.1.6.1] of the spec.
    ///
    /// [§3.1.6.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.6.1
    pub fn new(alpha: &Element<4>, norm: &BigInt<4>, order: &Order<4>) -> Self {
        // Compute Oα: multiply each basis element of O by α, at
        // `BigInt<8>` throughout.
        //
        // `mul_direct` (no GCD normalization) is used instead of
        // `mul` because the post-sum HNF needs every column to be
        // expressed at the SAME `denom`. `Element::mul` normalizes
        // each product independently — if different products reduce
        // by different GCDs, the stored numerators end up at
        // inconsistent scales while `o_alpha_denom` is computed
        // uniformly as `order.denom · alpha.denom`. The mismatch
        // silently corrupts the lattice. `mul_direct` keeps every
        // product's denom at exactly `order.denom · alpha.denom`
        // with numerators scaled accordingly.
        //
        // Width safety: product coordinates reach `|p · c · d| ≤
        // 2^250 · 2^N · 2^N` for α of coord magnitude ~2^N. For
        // narrow-path callers (`N < 2^127`), the product fits in
        // `BigInt<8>` (512 bits) with margin: 2^(250+127+127) =
        // 2^504 < 2^512. `BigInt<4>` (256 bits) would overflow
        // already at N > 2^3.
        // Widen α and the order's basis to BigInt<12> (= 768 bits).
        //
        // Width-12 is required for KAT-shaped α: every NIST-I KAT
        // secret-ideal generator has |coord| in [2^134, 2^140] (see
        // `survey_kat_secret_ideal_coord_magnitudes`). The largest
        // intermediate is `|p · c · d| ≈ 2^(250 + 140 + 140) = 2^530`
        // and the modulus `64 · N² · p · denom²` ≈ 2^538 — both
        // overflow BigInt<8> (= 512 bits). Width 12 (= 768 bits)
        // gives ~230 bits of headroom and absorbs the post-HNF
        // canonicalization without truncation.
        let alpha_w = Element::<12>::new(
            Coordinate::from_bigint(alpha.a.as_bigint().widen::<12>()),
            Coordinate::from_bigint(alpha.b.as_bigint().widen::<12>()),
            Coordinate::from_bigint(alpha.c.as_bigint().widen::<12>()),
            Coordinate::from_bigint(alpha.d.as_bigint().widen::<12>()),
            Denominator::from_bigint_unchecked(alpha.denom.as_bigint().widen::<12>()),
        );
        let order_basis_cols_4 = order.basis().columns();
        let widen_col_4_to_w = |col: &Vector<4>| -> Vector<12> {
            Vector::new(
                col[0].widen::<12>(),
                col[1].widen::<12>(),
                col[2].widen::<12>(),
                col[3].widen::<12>(),
            )
        };
        let order_denom_w: BigInt<12> = order.denom().widen();

        let mut o_alpha_cols_w = [Vector::<12>::ZERO; 4];
        for (j, col) in o_alpha_cols_w.iter_mut().enumerate() {
            let basis_col_w = widen_col_4_to_w(&order_basis_cols_4[j]);
            let basis_j_w = Element::<12>::new(
                Coordinate::from_bigint(basis_col_w[0]),
                Coordinate::from_bigint(basis_col_w[1]),
                Coordinate::from_bigint(basis_col_w[2]),
                Coordinate::from_bigint(basis_col_w[3]),
                Denominator::from_bigint_unchecked(order_denom_w),
            );
            let product = basis_j_w.mul_direct(&alpha_w);
            *col = Vector::new(
                *product.a.as_bigint(),
                *product.b.as_bigint(),
                *product.c.as_bigint(),
                *product.d.as_bigint(),
            );
        }
        let o_alpha_denom_w = order_denom_w.vt_mul(&alpha.denom.as_bigint().widen::<12>());

        // Compute ON: scale each basis vector of O by N, at `BigInt<12>`.
        let norm_w: BigInt<12> = norm.widen();
        let mut o_n_cols_w: [Vector<12>; 4] =
            array::from_fn(|j| widen_col_4_to_w(&order_basis_cols_4[j]));
        for col in &mut o_n_cols_w {
            for row in 0..4 {
                col[row] = col[row].vt_mul(&norm_w);
            }
        }
        let o_n_denom_w = order_denom_w;

        // Use [`Lattice::sum_mod`] at `BigInt<12>` to avoid
        // coefficient blow-up inside the XGCD pivot reduction of
        // classical HNF. The wider variant
        // [`LeftIdeal::random_prime_norm_wide`] uses `sum_mod::<44>`
        // at much larger ideal sizes; the narrow path needs only
        // enough headroom to absorb the modulus and its squared
        // products.
        //
        // `sum_mod` requires the two lattices to share a denom.
        // `o_alpha_denom = order.denom · alpha.denom` (may be
        // larger than 1 for alpha.denom > 1), while
        // `o_n_denom = order.denom`. When they're equal (the
        // common case for alpha.denom = 1), sum directly. When
        // different, scale to a common denom.
        let scale_cols_in_place_w = |cols: &mut [Vector<12>; 4], s: &BigInt<12>| {
            for col in cols.iter_mut() {
                for row in 0..4 {
                    col[row] = col[row].vt_mul(s);
                }
            }
        };
        let (common_denom_w, o_alpha_cols_w, o_n_cols_w) = if o_alpha_denom_w == o_n_denom_w {
            (o_alpha_denom_w, o_alpha_cols_w, o_n_cols_w)
        } else {
            let mut o_a = o_alpha_cols_w;
            let mut o_b = o_n_cols_w;
            scale_cols_in_place_w(&mut o_a, &o_n_denom_w);
            scale_cols_in_place_w(&mut o_b, &o_alpha_denom_w);
            (o_alpha_denom_w.vt_mul(&o_n_denom_w), o_a, o_b)
        };
        let o_alpha_w = Lattice::<12>::new(Matrix::from_columns(&o_alpha_cols_w), common_denom_w);
        let o_n_w = Lattice::<12>::new(Matrix::from_columns(&o_n_cols_w), common_denom_w);
        // Modulus `64 · N² · p · common_denom²` ≈ 2^538 for KAT-shaped
        // α (norm ≈ 2^140, p ≈ 2^250, denom ≤ 2). Width 12 holds it
        // with margin; sum_mod's working width 24 holds the squared
        // intermediates xgcd produces during HNF reduction.
        let modulus_w: BigInt<12> = {
            let n_w: BigInt<12> = norm.widen();
            let n_sq = n_w.vt_mul(&n_w);
            let p_w: BigInt<12> = crate::quaternions::precomputed::P_WIDE.widen::<12>();
            let denom_sq = common_denom_w.vt_mul(&common_denom_w);
            BigInt::<12>::from_u64(64)
                .vt_mul(&n_sq)
                .vt_mul(&p_w)
                .vt_mul(&denom_sq)
        };
        let lattice_w = o_alpha_w
            .sum_mod::<24>(&o_n_w, &modulus_w)
            .expect("denoms share common_denom_w by construction");

        // Canonicalize: compute the GCD of every basis entry and
        // the denom, then divide through. The common-denom
        // rescaling above multiplied the denom by
        // `o_n_denom_8 = order.denom`, which leaves a factor of 2
        // (or similar) shared across every basis column and the
        // denom. Without this GCD step, the lattice is represented
        // at 2× its minimal denom, making `basis[0] / denom` come
        // out as `7/2` instead of `7/1` (the latter being an
        // actual O_0 element) and the HNF diagonal encoding the
        // same lattice at a coarser grain. See the analogous step
        // in `smallest_equiv` (lattice.rs, shortly after `hnf_8`).
        let basis_cols_w = lattice_w.basis().columns();
        let denom_w = lattice_w.denom();
        let mut g: BigInt<12> = denom_w.abs();
        for col in &basis_cols_w {
            for row in 0..4 {
                if !bool::from(col[row].is_zero()) {
                    g = g.gcd(&col[row].abs());
                }
            }
        }
        // Narrow back to `HnfLattice<4>`. The HNF entries of a
        // proper ideal with `nrd ≤ N^2` fit in `BigInt<4>` for
        // `N < 2^128`; `narrow_to` returns `None` if this fails, which
        // indicates either a miscomputation upstream or a caller
        // passing α with coordinates outside `[−N, N)`.
        let basis_4 = {
            let mut m = Matrix::<4>::ZERO;
            for (j, col) in basis_cols_w.iter().enumerate() {
                for row in 0..4 {
                    let (q, _) = col[row].div_rem(&g);
                    m[row][j] = q
                        .narrow_to::<4>()
                        .expect("LeftIdeal<4>::new basis entry does not fit in BigInt<4>");
                }
            }
            m
        };
        let denom_4 = {
            let (q, _) = denom_w.div_rem(&g);
            q.narrow_to::<4>()
                .expect("LeftIdeal<4>::new denom does not fit in BigInt<4>")
        };
        let lattice = HnfLattice::from(Lattice::new(basis_4, denom_4));

        Self {
            lattice,
            norm: *norm,
            parent_order: *order,
        }
    }

    /// Constructs a random left ideal of a given prime norm.
    ///
    /// [Alg. 3.10] from the spec (prime case).
    ///
    /// WARNING: Not constant-time — brute-force search with
    /// data-dependent Legendre symbol and modular sqrt.
    ///
    /// TODO(ct): Make constant-time before production use. The norm
    /// argument may be secret-derived during signing (Algorithm 4.2
    /// line 23, where the norm depends on α_rsp).
    ///
    /// [Alg. 3.10]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.10
    pub fn random_prime_norm(n: &BigInt<4>, order: &ExtremalOrder<4>) -> Option<Self> {
        // Algorithm 3.10, prime case: sample γ = g₁i + g₂j + g₃ij
        // with g₁, g₂, g₃ uniform in [0, N-1], check Legendre
        // symbol, then adjust with modular sqrt.
        let n_bits = n.bitsize() as usize;
        let n_bytes = n_bits.div_ceil(8);

        for _ in 0..10_000 {
            // Sample g₁, g₂, g₃ uniform in [0, N-1] via rejection.
            let sample_mod_n = || -> BigInt<4> {
                loop {
                    let mut bytes = [0u8; 32];
                    OsRng.fill_bytes(&mut bytes[..n_bytes]);
                    // Mask top byte to avoid bias.
                    if !n_bits.is_multiple_of(8) {
                        bytes[n_bytes - 1] &= (1u8 << (n_bits % 8)) - 1;
                    }
                    let val = BigInt::<4>::from_bytes_le_unsigned(&bytes[..n_bytes]);
                    // Reject if val >= N.
                    if val.bitsize() <= n.bitsize() && val.ct_mod(n) == val {
                        return val; // val < N
                    }
                }
            };

            let g1 = sample_mod_n();
            let g2 = sample_mod_n();
            let g3 = sample_mod_n();

            // γ = g₁i + g₂j + g₃ij  (a = 0, denom = 1)
            let gamma = Element::<4>::new(
                Coordinate::ZERO,
                Coordinate::from_bigint(g1),
                Coordinate::from_bigint(g2),
                Coordinate::from_bigint(g3),
                Denominator::ONE,
            );
            let (nrd_num, nrd_den) = gamma.norm();

            // Narrow norm to BigInt<4>.
            let nrd_num_4: subtle::CtOption<BigInt<4>> = nrd_num.into();
            let nrd_den_4: subtle::CtOption<BigInt<4>> = nrd_den.into();
            if !bool::from(nrd_num_4.is_some()) || !bool::from(nrd_den_4.is_some()) {
                continue;
            }
            let nrd_num_4 = nrd_num_4.unwrap();
            let nrd_den_4 = nrd_den_4.unwrap();
            let (nrd_val, rem) = nrd_num_4.div_rem(&nrd_den_4);
            if !bool::from(rem.is_zero()) {
                continue;
            }

            // Check Legendre(-nrd(γ), N) = 1.
            let neg_nrd = n.ct_sub(&nrd_val.ct_mod(n));
            if BigInt::<4>::legendre(&neg_nrd, n) != 1 {
                continue;
            }

            // γ ← γ + √(-nrd(γ)) mod N
            let sqrt = match BigInt::<4>::modular_sqrt(&neg_nrd, n) {
                Some(s) => s,
                None => continue,
            };
            let gamma_adjusted = Element::<4>::new(
                Coordinate::from_bigint(sqrt),
                Coordinate::from_bigint(g1),
                Coordinate::from_bigint(g2),
                Coordinate::from_bigint(g3),
                Denominator::ONE,
            );

            return Some(Self::new(&gamma_adjusted, n, order.order()));
        }

        None
    }

    /// Constructs a random left ideal of a given (not necessarily prime) norm.
    ///
    /// [Alg. 3.10][Alg. 3.10] from the spec (non-prime case).
    /// Uses [`ExtremalOrder::represent_integer`] to find γ with
    /// nrd(γ) = m·N, then samples random β with gcd(nrd(β), N) = 1.
    ///
    /// # Divergences
    ///
    /// The `gcd(nrd(β), N) = 1` check runs at `BigInt<8>` rather
    /// than narrowing down to `BigInt<4>` first. An earlier version
    /// of this function narrowed `nrd(β) ≈ p·N² ≈ 2^505` (for
    /// response-phase `N ≈ 2^126`) into `BigInt<4>`, which always
    /// failed silently and rejected every one of the 10,000
    /// samples. `random_norm` then always returned `None`, signing
    /// quietly ran out of its 1000-iteration retry budget, and
    /// failed as `SigningFailed`.
    ///
    /// WARNING: Not constant-time.
    ///
    /// TODO(ct): Make constant-time before production use.
    ///
    /// [Alg. 3.10]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.10
    pub fn random_norm<R: RngCore>(
        n: &BigInt<4>,
        order: &ExtremalOrder<4>,
        rng: &mut R,
    ) -> Option<Self> {
        // m = QUAT_prime_cofactor (precomputed prime ≈ p)
        let m4 = crate::params::QUAT_PRIME_COFACTOR;
        let m = BigInt::<8>::from_limbs({
            let mut limbs = [0u64; 8];
            limbs[..4].copy_from_slice(m4.as_limbs());
            limbs
        });

        // Line 10: γ ← GeneralizedRepresentInteger(mN, i, O₀, false)
        let n_wide = BigInt::<8>::from_limbs({
            let mut limbs = [0u64; 8];
            limbs[..4].copy_from_slice(n.as_limbs());
            limbs
        });
        let mn = m.vt_mul(&n_wide);
        let order_wide = ExtremalOrder::<8>::from(*order);
        let gamma = order_wide.represent_integer(&mn, false, rng)?;

        // Lines 11-14: sample β = x + yi + zj + wij with gcd(nrd(β), N) = 1.
        //
        // Match C-ref's `ibz_rand_interval(out, 1, N)` byte-for-byte
        // (`src/quaternion/ref/generic/intbig.c:413`). C-ref masks the
        // top limb to `ceil(log2(N - 1))` bits, rejects `tmp > N - 1`,
        // then returns `tmp + 1`, mapping accepted tmps in `[0, N-1]`
        // to results in `[1, N]`. The earlier Selkie code rejected
        // `val == 0` and returned `val ∈ [1, N-1]` — same DRBG bytes
        // but each accepted sample one less than C-ref's, which then
        // compounded through `γ·β` into a different `i_aux` lattice
        // (KAT 39 iter 0 byte-diff vs `[I_AUX_CREF]`).
        let n_minus_1 = n.ct_sub(&BigInt::<4>::ONE);
        let bmina_bits = n_minus_1.bitsize() as usize;
        let bmina_bytes = bmina_bits.div_ceil(8);
        let mut sample_in_range = || -> BigInt<4> {
            loop {
                let mut bytes = [0u8; 32];
                rng.fill_bytes(&mut bytes[..bmina_bytes]);
                if !bmina_bits.is_multiple_of(8) {
                    bytes[bmina_bytes - 1] &= (1u8 << (bmina_bits % 8)) - 1;
                }
                let tmp = BigInt::<4>::from_bytes_le_unsigned(&bytes[..bmina_bytes]);
                // Reject when `tmp > N - 1`, matching C-ref's
                // `mpz_cmp(tmp, bmina) <= 0` accept condition.
                if tmp > n_minus_1 {
                    continue;
                }
                return tmp.ct_add(&BigInt::<4>::ONE);
            }
        };

        for _ in 0..10_000 {
            let x = sample_in_range();
            let y = sample_in_range();
            let z = sample_in_range();
            let w = sample_in_range();

            let beta = Element::<4>::new(
                Coordinate::from_bigint(x),
                Coordinate::from_bigint(y),
                Coordinate::from_bigint(z),
                Coordinate::from_bigint(w),
                Denominator::ONE,
            );

            // Check gcd(nrd(β), N) = 1. `nrd(β) = x² + y² + p(z² +
            // w²)` is about `p · N² ≈ 2^505` for N ~ 2^126, so it
            // doesn't narrow to `BigInt<4>`. Compute the gcd at
            // `BigInt<8>` against a widened N, then narrow the
            // (always small) gcd back to check for 1. An earlier
            // version of this function narrowed nrd before the
            // gcd check, which silently rejected every sample for
            // any N > ~2^64 and made `random_norm` return `None`
            // after 10_000 futile iterations.
            let (nrd_num, nrd_den) = beta.norm();
            let (nrd_val_wide, rem) = nrd_num.div_rem(&nrd_den);
            if !bool::from(rem.is_zero()) {
                continue;
            }
            let n_wide: BigInt<8> = n.widen();
            let gcd_wide = nrd_val_wide.gcd(&n_wide);
            if gcd_wide != BigInt::<8>::ONE {
                continue;
            }

            // Line 15: J' ← ideal generated by γβ and N
            //
            // `Element<4>::mul` would silently truncate: γ has
            // coords ~√(m·N) ≈ 2^129, β has coords < N ≈ 2^8, and
            // the quaternion product components reach ~p · 2^137 ≈
            // 2^388 — far beyond `BigInt<4>`'s 256-bit budget. In
            // release builds `from_wide` just drops the top limbs,
            // producing an `α` with wrong nrd and hence an invalid
            // O_0-ideal.
            //
            // Fix: compute γ·β at `Element<8>`, then reduce each
            // coordinate modulo `N`. The ideal `O·α + O·N` is
            // unchanged by `α → α mod N` because `N · Z<1,i,j,k> ⊂
            // N · O_0 = O · N` (since `O_0 ⊃ Z<1,i,j,k>`), so any
            // coordinate shift by a multiple of N lives in `O·N`
            // and is absorbed by the sum. The reduced α has
            // coords `< N`, trivially fitting in `BigInt<4>`.
            let widen_elem_4_to_8 = |e: &Element<4>| -> Element<8> {
                Element::<8>::new(
                    Coordinate::from_bigint(e.a.as_bigint().widen::<8>()),
                    Coordinate::from_bigint(e.b.as_bigint().widen::<8>()),
                    Coordinate::from_bigint(e.c.as_bigint().widen::<8>()),
                    Coordinate::from_bigint(e.d.as_bigint().widen::<8>()),
                    Denominator::from_bigint_unchecked(e.denom.as_bigint().widen::<8>()),
                )
            };
            // `gamma` is already `Element<8>` (RepresentInteger returns
            // width 8 so q ≥ 5 orders' ~2^259 coordinates survive);
            // only `beta` needs widening.
            let beta_8 = widen_elem_4_to_8(&beta);
            let gamma_beta_8 = gamma.mul_direct(&beta_8);

            // Reduce each numerator coord mod `N · denom`. For
            // α = (a, b, c, d) / denom, subtracting `k · N · denom`
            // from `a` changes α by `k · N`, which lives in
            // `N · Z<1,i,j,k> ⊂ N · O_0 = O · N` and is absorbed
            // by the sum `O · α + O · N`. Reducing mod `N` alone
            // (without the `· denom` factor) would leave a
            // half-integer residue for `denom = 2` and push α out
            // of O_0 entirely.
            let n_times_denom_8 = n.widen::<8>().vt_mul(gamma_beta_8.denom.as_bigint());
            let reduce_coord = |c: &BigInt<8>| -> BigInt<4> {
                let r = c.ct_mod(&n_times_denom_8);
                r.narrow_to::<4>()
                    .expect("coord reduced mod N·denom fits in BigInt<4>")
            };
            let denom_4 = gamma_beta_8
                .denom
                .as_bigint()
                .narrow_to::<4>()
                .expect("product denom = γ.denom·β.denom fits in BigInt<4>");
            let gamma_beta = Element::<4>::new(
                Coordinate::from_bigint(reduce_coord(gamma_beta_8.a.as_bigint())),
                Coordinate::from_bigint(reduce_coord(gamma_beta_8.b.as_bigint())),
                Coordinate::from_bigint(reduce_coord(gamma_beta_8.c.as_bigint())),
                Coordinate::from_bigint(reduce_coord(gamma_beta_8.d.as_bigint())),
                Denominator::from_bigint_unchecked(denom_4),
            );

            // Build the ideal `O_0·α + O_0·N` from the reduced
            // generator. The first `β` accepted by the pre-product
            // `gcd(nrd(β), N) = 1` check is the one returned, matching
            // the C reference's `quat_sampling_random_ideal_O0_given_norm`
            // (`quaternion/ref/generic/normeq.c`), whose rerandomization
            // loop only checks `gcd(nrd(gen_rerand), norm) = 1` before
            // committing.
            //
            // # Divergence (byte-interop)
            //
            // An earlier version recomputed `gcd(nrd(α mod N·denom)/N, N)`
            // and a per-HNF-column nrd-divisibility predicate after
            // reduction, resampling `β` on failure. Those predicates are
            // mathematically vacuous: `α → α mod (N·denom)` shifts `α` by
            // a member of `N·O_0` (since `1, i, j, k ∈ O_0`), so
            // `O_0·α + O_0·N` is unchanged and always has norm exactly
            // `N` — yet `nrd(α mod N·denom)/N (mod N)` is arbitrary and
            // frequently shares a factor with composite `N`. The spurious
            // rejection consumed extra DRBG bytes resampling `β`, landing
            // on a different (still valid) ideal than the C reference and
            // producing byte-different `E_aux`/`M_chl`/`hint_aux`.
            return Some(Self::new(&gamma_beta, n, order.order()));
        }

        None
    }

    /// Finds a primitive generator γ of this ideal.
    ///
    /// [Alg. 3.8] from the spec.
    ///
    /// WARNING: Not constant-time — bounded brute-force search with
    /// data-dependent norm checks and GCD.
    ///
    /// TODO(ct): Make constant-time before production use. Called on
    /// secret-derived ideals via IdealToKernel during signing.
    ///
    /// [Alg. 3.8]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.8
    pub fn generator(&self) -> Option<Element<4>> {
        let basis = self.lattice.basis();
        let n_i = &self.norm;

        const MAX_NORM: i64 = 1000;
        let mut n: i64 = 0;
        while n < MAX_NORM {
            n += 1;
            let mut a = -n;
            while a <= n {
                let rem_a = n - a.abs();
                let mut b = -rem_a;
                while b <= rem_a {
                    let rem_b = rem_a - b.abs();
                    let mut c = -rem_b;
                    while c <= rem_b {
                        let d = rem_b - c.abs();
                        for &d_val in &[d, -d] {
                            if a.abs() + b.abs() + c.abs() + d_val.abs() != n {
                                continue;
                            }

                            let a_big = BigInt::<4>::from_i64(a);
                            let b_big = BigInt::<4>::from_i64(b);
                            let c_big = BigInt::<4>::from_i64(c);
                            let d_big = BigInt::<4>::from_i64(d_val);

                            let g = a_big
                                .abs()
                                .gcd(&b_big.abs())
                                .gcd(&c_big.abs())
                                .gcd(&d_big.abs());
                            if g != BigInt::ONE {
                                continue;
                            }

                            let mut gamma_coords = [BigInt::<4>::ZERO; 4];
                            for row in 0..4 {
                                gamma_coords[row] = a_big
                                    .vt_mul(&basis[row][0])
                                    .ct_add(&b_big.vt_mul(&basis[row][1]))
                                    .ct_add(&c_big.vt_mul(&basis[row][2]))
                                    .ct_add(&d_big.vt_mul(&basis[row][3]));
                            }
                            let gamma = Element::<4>::new(
                                Coordinate::from_bigint(gamma_coords[0]),
                                Coordinate::from_bigint(gamma_coords[1]),
                                Coordinate::from_bigint(gamma_coords[2]),
                                Coordinate::from_bigint(gamma_coords[3]),
                                Denominator::from_bigint_unchecked(*self.lattice.denom()),
                            );

                            let (nrd_num, nrd_den) = gamma.norm();
                            // Widen n_i to BigInt<8> for division.
                            let n_i_wide: BigInt<8> = (*n_i).into();
                            let (q, rem) = nrd_num.div_rem(&nrd_den.vt_mul(&n_i_wide));
                            if !bool::from(rem.is_zero()) {
                                continue;
                            }
                            if q.gcd(&n_i_wide) == BigInt::<8>::ONE {
                                return Some(gamma);
                            }
                        }
                        c += 1;
                    }
                    b += 1;
                }
                a += 1;
            }
        }
        None
    }

    // KernelToIdeal (Algorithm 3.17) is defined as
    // TorsionBasis::kernel_to_ideal() in curves/mod.rs.
}
