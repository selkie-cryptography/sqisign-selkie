//! Width-30 specializations of [`LeftIdeal<30>`][super::LeftIdeal]:
//! [`from_generator_mod_hnf`][LeftIdeal::from_generator_mod_hnf] (used
//! by signing's response phase to construct the commitment-response
//! ideal modulo a precomputed HNF modulus) and
//! [`random_prime_norm_wide`][LeftIdeal::random_prime_norm_wide]
//! (commitment-phase prime-norm ideal sampler).

use core::array;

use rand_core::RngCore;

use super::{
    super::{
        super::{
            algebra::{Coordinate, Denominator, Element},
            bigint::BigInt,
            linear::{Matrix, Vector},
        },
        ExtremalOrder, Lattice, Order,
    },
    LeftIdeal,
};

impl LeftIdeal<30> {
    /// Constructs a left ideal `I = O⟨α, N⟩` at storage width
    /// `N = 30` via modular HNF, avoiding the classical HNF
    /// coefficient blow-up that corrupts the generic
    /// [`from_generator`][Self::from_generator] path at this
    /// width.
    ///
    /// This is the response-phase analogue of the construction
    /// path used by [`random_prime_norm_wide`][Self::random_prime_norm_wide]
    /// for the commitment ideal. At `N = 30` the incoming
    /// generator `α_rsp` from the sampling step has coordinates
    /// up to ≈ 2^1400 bits; the classical HNF inside
    /// [`Lattice::sum`] will silently overflow on products of
    /// these entries, whereas [`Lattice::sum_mod`] bounds every
    /// intermediate by the per-call modulus
    /// `D = 4 · d⁴ · norm² · p`.
    ///
    /// Because the modulus depends on `norm` (which varies per
    /// signature — it is `q_rsp · D_MIX` with `q_rsp` sampled
    /// each iteration), it is computed at call time rather than
    /// precomputed as a const.
    ///
    /// # Returns
    ///
    /// `None` if `α.denom` differs from `order.denom()` (which
    /// would put the two sub-lattices at mismatched denominators
    /// and force a rescale that is incompatible with the chosen
    /// `sum_mod` width budget). The sole current caller
    /// (`sign_derand`'s response phase) always passes `α` with
    /// denom `1`, and the order is `O₀` with denom `2`, so the
    /// denoms match by construction. The check is defensive.
    pub fn from_generator_mod_hnf(
        alpha: &Element<30>,
        norm: &BigInt<30>,
        order: &Order<30>,
    ) -> Option<Self> {
        // Compute Oα: multiply each basis element of O by α.
        let mut o_alpha_cols = [Vector::<30>::ZERO; 4];
        for (j, o_alpha_col) in o_alpha_cols.iter_mut().enumerate() {
            let basis_j = order.basis_elem(j);
            let product = basis_j.mul_direct(alpha);
            *o_alpha_col = Vector::new(
                *product.a.as_bigint(),
                *product.b.as_bigint(),
                *product.c.as_bigint(),
                *product.d.as_bigint(),
            );
        }
        let o_alpha_denom = order.denom().ct_mul(alpha.denom.as_bigint());
        // Apply classical HNF to o_alpha (= 4 mul_direct cols),
        // mirroring C-ref's `quat_lattice_alg_elem_mul` which calls
        // `quat_lattice_hnf` after the multiplication. Without this,
        // o_alpha is the raw mul cols; the downstream `sum_mod_cref`
        // sees DIFFERENT inputs than C-ref's `quat_lattice_add` which
        // gets HNF-reduced o_alpha. Classical HNF at width 30
        // overflows for our shape; widen to W=60 to compute, then
        // narrow back.
        let o_alpha_cols_hnf: [Vector<30>; 4] = {
            let widened: [Vector<60>; 4] = array::from_fn(|i| {
                let v = &o_alpha_cols[i];
                Vector::<60>::new(
                    v[0].widen::<60>(),
                    v[1].widen::<60>(),
                    v[2].widen::<60>(),
                    v[3].widen::<60>(),
                )
            });
            // Use C-ref's `quat_lattice_hnf` recipe: modular HNF with
            // mod = |det| of the input matrix, NOT classical HNF.
            // Selkie's classical `Matrix::hnf()` produces a valid
            // upper-triangular HNF but with off-diagonal entries
            // (cols 2,3 rows 0,1) that differ from C-ref's modular-HNF
            // result, even though both bases describe the same lattice.
            // Tested on KAT-1: byte-mismatch with C-ref's `lideal_com_resp`
            // is gone once we use HNF mod with mod=|det|.
            let det_w = Matrix::from_columns(&widened).det().abs();
            // Use the constant-modulus variant (= old Selkie path)
            // — for 4-col input this should match canonical HNF.
            let hnf_w = Matrix::from_hnf_columns_mod::<60>(&widened, &det_w);
            let cols_w: [Vector<60>; 4] = hnf_w.columns();
            array::from_fn(|i| {
                Vector::<30>::new(
                    cols_w[i][0]
                        .narrow_to::<30>()
                        .expect("o_alpha hnf col 0 fits in 30"),
                    cols_w[i][1]
                        .narrow_to::<30>()
                        .expect("o_alpha hnf col 1 fits in 30"),
                    cols_w[i][2]
                        .narrow_to::<30>()
                        .expect("o_alpha hnf col 2 fits in 30"),
                    cols_w[i][3]
                        .narrow_to::<30>()
                        .expect("o_alpha hnf col 3 fits in 30"),
                )
            })
        };
        let o_alpha = Lattice::new(Matrix::from_columns(&o_alpha_cols_hnf), o_alpha_denom);

        // Mod-HNF bounding modulus `D = 4 · d⁴ · norm² · p`.
        //
        // For the sign response phase, `norm = q_rsp · D_MIX`
        // with `q_rsp ≤ 2^126` and `D_MIX ≈ 2^513`, so
        // `norm ≲ 2^640` and `D ≲ 2^(2 + 4 + 1280 + 256) =
        // 2^1542`. This fits comfortably in `BigInt<30>` (1920
        // bits).
        let p_wide: BigInt<30> = {
            let p8: BigInt<8> = crate::quaternions::precomputed::P_WIDE;
            let mut limbs = [0u64; 30];
            limbs[..8].copy_from_slice(p8.as_limbs());
            BigInt::from_sign_and_limbs(0, limbs)
        };
        // With the shared denom = order_denom · α_denom, the
        // modulus needs to account for both: D = 4·d_total⁴·norm²·p.
        let d_total = o_alpha_denom;
        let d_sq = d_total.ct_mul(&d_total);
        let d_fourth = d_sq.ct_mul(&d_sq);
        let norm_sq = norm.ct_mul(norm);
        let four = BigInt::<30>::from_u64(4);
        // Mod-HNF modulus uses the spec/Selkie historic formula
        // `4 · d⁴ · norm² · p`.  An earlier diagnostic toggle compared
        // this against C-ref's `quat_lattice_add` `gcd(det1, det2)`
        // recipe; the two produced the same Z-module for the sign
        // path and the toggle has been retired.
        let modulus = four.ct_mul(&d_fourth).ct_mul(&norm_sq).ct_mul(&p_wide);

        // ON denom may differ from o_alpha denom — pre-scale ON
        // basis so both share `o_alpha_denom` for `sum_mod` (which
        // assumes equal denoms).
        let alpha_d = *alpha.denom.as_bigint();
        let mut o_n_for_sum = order.basis().columns();
        for col in &mut o_n_for_sum {
            for row in 0..4 {
                col[row] = col[row].ct_mul(norm).ct_mul(&alpha_d);
            }
        }
        let o_n_scaled = Lattice::new(Matrix::from_columns(&o_n_for_sum), o_alpha_denom);
        let lattice = o_alpha.sum_mod::<60>(&o_n_scaled, &modulus)?;

        Some(Self {
            lattice,
            norm: *norm,
            parent_order: *order,
        })
    }

    /// Constructs a random left ideal of a given prime norm (wide version).
    ///
    /// For the commitment phase (Algorithm 4.2 line 4), the norm D_MIX
    /// = 2^512 + 75 is 513 bits. This method stores the resulting
    /// ideal at `BigInt<30>` (1920 bits) so that:
    /// - Column entries `p·g_i ≈ 2^769` fit without truncation.
    /// - The downstream `reduce_to_prime_norm` gram computation `c^T·G·c ≈
    ///   2^1806` fits without overflow.
    ///
    /// After construction, call `reduce_to_prime_norm` to get a small
    /// prime norm, then `narrow_to::<4>()` to convert to `LeftIdeal<4>`
    /// for `to_isogeny`.
    ///
    /// [Alg. 3.10][Alg. 3.10] from the spec (prime case).
    ///
    /// WARNING: Not constant-time.
    ///
    /// [Alg. 3.10]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.10
    pub fn random_prime_norm_wide<R: RngCore>(
        n: &BigInt<30>,
        order: &ExtremalOrder<4>,
        rng: &mut R,
    ) -> Option<Self> {
        let p_wide: BigInt<30> = {
            let p8: BigInt<8> = crate::quaternions::precomputed::P_WIDE;
            let mut limbs = [0u64; 30];
            limbs[..8].copy_from_slice(p8.as_limbs());
            BigInt::from_sign_and_limbs(0, limbs)
        };

        let zero_big = BigInt::<30>::ZERO;
        let one_big = BigInt::<30>::ONE;
        let n_minus_one = n.ct_sub(&one_big);

        for _ in 0..10_000 {
            // Phase A: trace-zero quaternion γ = a + g₁·i + g₂·j +
            // g₃·k with nrd(γ) ≡ 0 (mod N). Sample (g₁, g₂, g₃) ∈
            // [0, N − 1] via [`BigInt::rand_interval`] (matches
            // C ref's `ibz_rand_interval(0, n−1)` byte-for-byte),
            // compute disc = −nrd mod N, and recover a via
            // `sqrt mod N` after a Legendre check.
            let g1 = BigInt::<30>::rand_interval(rng, &zero_big, &n_minus_one);
            let g2 = BigInt::<30>::rand_interval(rng, &zero_big, &n_minus_one);
            let g3 = BigInt::<30>::rand_interval(rng, &zero_big, &n_minus_one);

            // nrd(γ) = g₁² + p(g₂² + g₃²) for γ = g₁i + g₂j + g₃ij
            // in the quaternion algebra B_{p,∞} = (-1, -p). With
            // g_i < 2^513 and p ≈ 2^256 the result is ≈ 2^1282 bits,
            // well within `BigInt<30>` (1920 bits).
            let g1_sq = g1.ct_mul(&g1);
            let g2_sq = g2.ct_mul(&g2);
            let g3_sq = g3.ct_mul(&g3);
            let nrd = g1_sq.ct_add(&p_wide.ct_mul(&g2_sq.ct_add(&g3_sq)));

            let nrd_mod = nrd.ct_mod(n);
            let neg_nrd = n.ct_sub(&nrd_mod);

            // Check Legendre(-nrd(γ), N) = 1. The `_w::<30>` variants
            // keep the primality/sqrt arithmetic at the storage width
            // (well above the 1026-bit `pow_mod` requirement).
            if BigInt::<30>::legendre_w::<30>(&neg_nrd, n) != 1 {
                continue;
            }

            // a = √(-nrd(γ)) mod N.
            let a = match BigInt::<30>::modular_sqrt_w::<30>(&neg_nrd, n) {
                Some(s) => s,
                None => continue,
            };

            // Phase B: rerandomize the principal ideal class by
            // sampling δ = (d₀, d₁, d₂, d₃) with gcd(nrd(δ), N) = 1
            // and replacing γ ← γ · δ. Mirrors C ref's
            // `quat_sampling_random_ideal_O0_given_norm`
            // (`normeq.c:297-384`). Without this step the resulting
            // ideal lattice differs from C ref's by a multiplicative
            // δ-twist, so the downstream `reduce_to_prime_norm`
            // basis (and every byte after it) diverges.
            let delta_coords: Option<[BigInt<30>; 4]> = (0..1000).find_map(|_| {
                let d0 = BigInt::<30>::rand_interval(rng, &one_big, n);
                let d1 = BigInt::<30>::rand_interval(rng, &one_big, n);
                let d2 = BigInt::<30>::rand_interval(rng, &one_big, n);
                let d3 = BigInt::<30>::rand_interval(rng, &one_big, n);
                let nrd_d = d0
                    .ct_mul(&d0)
                    .ct_add(&d1.ct_mul(&d1))
                    .ct_add(&p_wide.ct_mul(&d2.ct_mul(&d2).ct_add(&d3.ct_mul(&d3))));
                let nrd_d_mod = nrd_d.ct_mod(n);
                if nrd_d_mod.gcd(n) == BigInt::<30>::ONE {
                    Some([d0, d1, d2, d3])
                } else {
                    None
                }
            });
            let Some([d0, d1, d2, d3]) = delta_coords else {
                continue;
            };

            // γ · δ via [`Element::mul_direct`] at width 30.
            // Inputs use ≤ 9 limbs (513 bits); products fit the
            // N/2 = 15-limb precondition. Output coords reach
            // ~2^1285 (still well within 30 limbs).
            let gamma_elem = Element::<30>::new(
                Coordinate::from_bigint(a),
                Coordinate::from_bigint(g1),
                Coordinate::from_bigint(g2),
                Coordinate::from_bigint(g3),
                Denominator::from_bigint_unchecked(one_big),
            );
            let delta_elem = Element::<30>::new(
                Coordinate::from_bigint(d0),
                Coordinate::from_bigint(d1),
                Coordinate::from_bigint(d2),
                Coordinate::from_bigint(d3),
                Denominator::from_bigint_unchecked(one_big),
            );
            let new_gen = gamma_elem.mul_direct(&delta_elem);
            let a = *new_gen.a.as_bigint();
            let g1 = *new_gen.b.as_bigint();
            let g2 = *new_gen.c.as_bigint();
            let g3 = *new_gen.d.as_bigint();

            // Construct I = O₀⟨γ, N⟩ as a lattice.
            //
            // Precompute the 4 products of basis quaternions with γ:
            //   1·γ = ( a,    g₁,   g₂,   g₃)
            //   i·γ = (-g₁,   a,   -g₃,   g₂)
            //   j·γ = (-pg₂,  pg₃,  a,   -g₁)
            //   k·γ = (-pg₃, -pg₂,  g₁,   a )
            //
            // For B_{p,∞} = (-1,-p): i²=-1, j²=-p, k=ij. These
            // `pg_i` products reach ≈ 2^769 and required the 1920-bit
            // storage width.
            let pg2 = p_wide.ct_mul(&g2);
            let pg3 = p_wide.ct_mul(&g3);
            let prod_1 = [a, g1, g2, g3];
            let prod_i = [g1.wrapping_neg(), a, g3.wrapping_neg(), g2];
            let prod_j = [pg2.wrapping_neg(), pg3, a, g1.wrapping_neg()];
            let prod_k = [pg3.wrapping_neg(), pg2.wrapping_neg(), g1, a];

            // For each order basis element e = (e₀,e₁,e₂,e₃)/denom,
            // compute e·γ = (e₀·(1·γ) + e₁·(i·γ) + e₂·(j·γ) + e₃·(k·γ))/denom.
            let order_wide = ExtremalOrder::<30>::from(*order);
            let order_lat = order_wide.order();
            let order_denom = *order_lat.denom();

            let mut o_alpha_cols = [Vector::<30>::ZERO; 4];
            for (col, o_alpha_col) in o_alpha_cols.iter_mut().enumerate() {
                let e = [
                    order_lat.basis()[0][col],
                    order_lat.basis()[1][col],
                    order_lat.basis()[2][col],
                    order_lat.basis()[3][col],
                ];
                for row in 0..4 {
                    o_alpha_col[row] = e[0]
                        .ct_mul(&prod_1[row])
                        .ct_add(&e[1].ct_mul(&prod_i[row]))
                        .ct_add(&e[2].ct_mul(&prod_j[row]))
                        .ct_add(&e[3].ct_mul(&prod_k[row]));
                }
            }

            // O₀·N: scale each order basis column by N.
            let mut o_n_cols = [Vector::<30>::ZERO; 4];
            for (col, o_n_col) in o_n_cols.iter_mut().enumerate() {
                for row in 0..4 {
                    o_n_col[row] = order_lat.basis()[row][col].ct_mul(n);
                }
            }

            // I = O₀·γ + O₀·N, as a Z-lattice sum.
            //
            // We use [`Lattice::sum_mod`] rather than the classical
            // [`Lattice::sum`] because at `BigInt<30>` the
            // classical HNF path inside `sum` suffers from
            // coefficient blow-up: intermediate xgcd products
            // exceed the 1920-bit storage budget and silently
            // truncate, collapsing the basis to something unrelated
            // to the intended ideal. Modular HNF bounds every
            // intermediate by the precomputed NIST-I constant
            // [`D_HNF_MODULUS_COMMITMENT`][crate::quaternions::precomputed::D_HNF_MODULUS_COMMITMENT]
            // = 4 · d⁴ · D_MIX² · p, keeping everything within the
            // `W = 44` working width. See the rustdoc on
            // [`Matrix::from_hnf_columns_mod`] and the "Fixed-Precision HNF"
            // rationale (spec gap: Algorithm 3.2 doesn't discuss
            // fixed-precision adaptations).
            let o_alpha = Lattice::new(Matrix::from_columns(&o_alpha_cols), order_denom);
            let o_n = Lattice::new(Matrix::from_columns(&o_n_cols), order_denom);

            // `o_alpha` and `o_n` share `order_denom`, so
            // `sum_mod` returns `Some` by construction here; the
            // `?` is defensive against any future refactor that
            // changes one of the two denominators.
            let lattice = o_alpha.sum_mod::<44>(
                &o_n,
                &crate::quaternions::precomputed::D_HNF_MODULUS_COMMITMENT,
            )?;
            return Some(LeftIdeal {
                lattice,
                norm: *n,
                parent_order: *order_lat,
            });
        }

        None
    }
}
