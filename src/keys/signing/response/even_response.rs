//! Sign-side kernel for [ComputeEvenNonBacktrackingResponse][Alg. 4.6] —
//! the even response isogeny `φ^even_rsp : E_chl → E` of degree
//! `2^r_rsp`, with kernel determined by the response quaternion `α`
//! acting on the `2^r_rsp`-torsion of `E_chl`.
//!
//! Two-phase construction mirroring
//! [`crate::quaternions::lattice::LeftIdeal::to_isogeny`]:
//! [`EvenResponseKernel::from_quaternion`] picks a small generator of
//! `O₀·conj(α) + O₀·(2^r_rsp)`, decomposes it on the endomorphism
//! ring of `E₀`, and builds the kernel point `K = [s]P + [t]Q` on
//! the reduced basis; [`EvenResponseKernel::isogeny`] runs the
//! `r_rsp`-isogeny chain and returns the codomain with the
//! propagated basis.
//!
//! [Alg. 4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.6

use subtle::{Choice, ConditionallySelectable};

use crate::{
    curves::{
        TorsionBasis, TorsionExponent, isogeny::Kernel as CurveKernel,
        montgomery::ProjectiveXOnlyPoint, scalar::Scalar,
    },
    deuring::{endomorphism::EndomorphismAction, precomputed::ENDOMORPHISM_MATRICES},
    quaternions::{
        algebra::Element, bigint::BigInt, lattice::LeftIdeal, precomputed::EXTREMAL_ORDERS,
    },
};

/// The kernel of [ComputeEvenNonBacktrackingResponse][Alg. 4.6].
///
/// Of order `2^r_rsp` on `E_chl`. Constructed via
/// [`Self::from_quaternion`] from the response quaternion `α` and
/// the challenge basis on `E_chl`; run via [`Self::isogeny`].
///
/// [Alg. 4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.6
pub(crate) struct EvenResponseKernel {
    /// Kernel generator `K = [s]P + [t]Q` on `E_chl`.
    K: ProjectiveXOnlyPoint,
    /// `2^r_rsp`-torsion basis on `E_chl`, propagated through the isogeny.
    basis: TorsionBasis,
    /// Chain length `r_rsp` of the response isogeny.
    r_rsp: TorsionExponent,
}

/// Codomain of [`EvenResponseKernel::isogeny`].
///
/// `basis` is the propagated image
/// `(φ^even_rsp(P), φ^even_rsp(P − Q), φ^even_rsp(Q))` of the
/// input challenge basis on `E_chl`. The codomain curve
/// `E = φ^even_rsp(E_chl)` is accessible as `basis.P.curve()`.
pub(crate) struct EvenResponseCodomain {
    /// Image basis `(φ^even_rsp(P), φ^even_rsp(P − Q), φ^even_rsp(Q))`.
    pub basis: TorsionBasis,
}

impl EvenResponseKernel {
    /// Constructs the kernel from the response quaternion `α` and
    /// the challenge basis on `E_chl`.
    ///
    /// `alpha` is the response quaternion narrowed to width 4
    /// (coordinate-wise reduced mod `2^r_rsp`, then narrowed). It
    /// has the same residue mod `2^r_rsp · O₀` as the wide
    /// response, so generates the same left ideal
    /// `I = O₀·conj(α) + O₀·(2^r_rsp)`.
    ///
    /// # Side-channel considerations
    ///
    /// WARNING: Not yet fully constant-time. Called on the secret
    /// response quaternion `α` during signing
    /// ([Algorithm 4.2][Alg. 4.2] line 27).  Variable-time operations
    /// on `α`-derived values:
    /// - [`LeftIdeal::new`] on `conj(α) + 2^r_rsp`
    /// - [`LeftIdeal::generator`] (`ideal.generator()`)
    /// - [`BigInt::gcd`] on the `M_α` column-0 entries `s0, t0` — timing leak
    ///   only, no longer a direct bit leak (the column selection below is now
    ///   branch-free).
    ///
    /// The previous 1-bit parity branch
    /// `if g0_is_even { col 1 } else { col 0 }` is now a
    /// constant-time [`subtle::ConditionallySelectable`] swap; the
    /// kernel scalars are picked obliviously on the parity of
    /// `gcd(s0, t0)`.
    ///
    /// TODO(ct): close the residual `gcd` timing leak by routing
    /// through a constant-time `gcd` (Bernstein--Yang divstep).
    ///
    /// # Divergences
    ///
    /// The 3-point ladder used for `K = [s]P_red + [t]Q_red` runs
    /// on the *reduced* basis (`P, Q, PmQ` doubled `e_prime + 2`
    /// times to order `2^r_rsp`), NOT on the full-order basis with
    /// scalars `s · 2^(e_prime + 2), t · 2^(e_prime + 2)`. The two
    /// are mathematically equivalent — same kernel subgroup — but
    /// produce byte-different projective `(X : Z)` representatives
    /// of the same abstract point, and the chain's
    /// [`CurveKernel::isogeny_small`] is sensitive to the projective
    /// rep (different `(X : Z)` → different chain codomain rep).
    /// Mirrors C-ref's `compute_small_chain_isogeny_signature`
    /// (sign.c:627-632: `ec_dbl_iter_basis` then
    /// `ec_biscalar_mul_ibz_vec`).
    ///
    /// `α` is conjugated before constructing the ideal to match
    /// C-ref's sign.c:401, which conjugates `resp_quat` in place
    /// before calling `quat_lideal_create(lideal_resp_two,
    /// resp_quat, 2^r_rsp, O₀)`. A small generator of the resulting
    /// `O₀·conj(α) + O₀·(2^r_rsp)` is then chosen via
    /// [`LeftIdeal::generator`] — the kernel of `α | E[2^r_rsp]` is
    /// NOT invariant under `α → α · u` for non-trivial units
    /// `u ∈ O_R(I)^×` (which include `±i` in `O₀`), so picking a
    /// different generator yields a different kernel subgroup and
    /// thus a different `j(E_chl_3)`.
    ///
    /// [Alg. 4.2]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.2
    pub(crate) fn from_quaternion(
        alpha: &Element<4>,
        basis: TorsionBasis,
        e_prime: TorsionExponent,
        r_rsp: TorsionExponent,
    ) -> Option<Self> {
        let e_prime_val = e_prime.value();
        let r_rsp_val = r_rsp.value();

        // Step 1: I = O₀·conj(α) + O₀·(2^r_rsp)
        let norm = BigInt::<4>::ONE << r_rsp_val;
        let alpha_for_ideal = alpha.conjugate();
        let ideal = LeftIdeal::new(&alpha_for_ideal, &norm, EXTREMAL_ORDERS[0].order());

        // Step 2: (s, t) ← IdealToKernel(I).
        //
        // Inline C-ref's `id2iso_ideal_to_kernel_dlogs_even`
        // (id2iso.c:29-86).
        let alpha_for_kernel = match ideal.generator() {
            Some(g) => g,
            None => {
                #[cfg(test)]
                eprintln!("[EvenResponseKernel::from_quaternion] DROP: ideal.generator() None");
                return None;
            }
        };

        let alpha_conj = alpha_for_kernel.conjugate();
        let endo_e0 = EndomorphismAction {
            order: EXTREMAL_ORDERS[0].order(),
            generators: [
                ENDOMORPHISM_MATRICES[0][3], // gen2 = action of i
                ENDOMORPHISM_MATRICES[0][4], // gen3 = action of (i+j)/2
                ENDOMORPHISM_MATRICES[0][5], // gen4 = action of (1+k)/2
            ],
        };

        let m_alpha = endo_e0.apply(&alpha_conj, TorsionExponent::try_from(r_rsp_val).ok()?)?;
        let modulus = BigInt::<4>::ONE << r_rsp_val;
        // Pick the column of `M_α` whose gcd is odd.  Both columns are
        // computed unconditionally and the selection is a branch-free
        // [`ConditionallySelectable`] swap so the column choice does
        // not leak the parity of `gcd(s0, t0)` (a derived bit of α).
        //
        // The `gcd` call is still variable-time on its inputs; that
        // residual leak is a separate `TODO(ct)` (Bernstein--Yang
        // divstep replacement of `BigInt::gcd`).
        let s0 = BigInt::<4>::from(*m_alpha.entry(0, 0)).vt_mod(&modulus);
        let t0 = BigInt::<4>::from(*m_alpha.entry(1, 0)).vt_mod(&modulus);
        let s1 = BigInt::<4>::from(*m_alpha.entry(0, 1)).vt_mod(&modulus);
        let t1 = BigInt::<4>::from(*m_alpha.entry(1, 1)).vt_mod(&modulus);
        let g0 = s0.gcd(&t0);
        let pick_col1 = Choice::from(((g0.as_limbs()[0] & 1) ^ 1) as u8);
        let s = BigInt::conditional_select(&s0, &s1, pick_col1);
        let t = BigInt::conditional_select(&t0, &t1, pick_col1);

        // Step 3: double the basis down to order 2^r_rsp, then
        // compute K = [s]P_red + [t]Q_red on the reduced basis.
        //
        // The `from_propagated(Pr, Qr, Rr)` call below places
        // `Q_red` in the `.PmQ` field and `PmQ_red` in the `.Q`
        // field — a deliberate field-label swap. `biscalar_mul`
        // computes `[m]·.P + [n]·.PmQ` with `.Q` as the
        // differential; under the swap this becomes
        // `[s]·P + [t]·Q` with `x(P − Q) = PmQ_red` as the
        // differential — the spec K. (See the
        // `id2iso_ideal_to_kernel_dlogs_even` comment above for
        // why `(s, t)` are scalars on the `(P, Q)` basis under
        // C-ref's matrix encoding.)
        let shift = e_prime_val + 2;
        let basis_reduced = {
            let mut Pr = basis.P;
            let mut Qr = basis.Q;
            let mut Rr = basis.PmQ;
            for _ in 0..shift {
                Pr = Pr.double();
                Qr = Qr.double();
                Rr = Rr.double();
            }
            TorsionBasis::from_propagated(Pr, Qr, Rr)
        };

        let s_scalar = Scalar::from(s);
        let t_scalar = Scalar::from(t);
        let K = basis_reduced.biscalar_mul(
            &s_scalar,
            &t_scalar,
            TorsionExponent::try_from(r_rsp_val).ok()?,
        );

        Some(Self { K, basis, r_rsp })
    }

    /// Runs the `r_rsp`-isogeny chain `φ^even_rsp` and returns the
    /// codomain with the propagated basis image.
    ///
    /// The input basis is pushed through with `PmQ` as the third
    /// evaluated point, so the codomain basis carries a propagated
    /// `P − Q`. Downstream consumers ([`Challenge::to_isogeny`]
    /// and [`ChangeOfBasisMatrix::from_bases`]) require `PmQ`
    /// consistent with `P` and `Q`'s evaluation history — never
    /// recompute it via
    /// [`ProjectiveXOnlyPoint::projective_difference`] on the
    /// output basis.
    ///
    /// [`Challenge::to_isogeny`]: crate::keys::Challenge::to_isogeny
    /// [`ChangeOfBasisMatrix::from_bases`]: crate::curves::ChangeOfBasisMatrix::from_bases
    pub(crate) fn isogeny(self) -> Option<EvenResponseCodomain> {
        let isogeny_res = CurveKernel::new(self.K).isogeny_small(
            self.r_rsp,
            &[self.basis.P, self.basis.Q, self.basis.PmQ],
            true,
        );
        let (_, images) = match isogeny_res {
            Ok(r) => r,
            Err(_e) => {
                #[cfg(test)]
                eprintln!(
                    "[EvenResponseKernel::isogeny] DROP: isogeny_small err: {_e:?}, r_rsp={}",
                    self.r_rsp.value()
                );
                return None;
            }
        };

        Some(EvenResponseCodomain {
            basis: TorsionBasis::from_propagated(images[0], images[2], images[1]),
        })
    }
}
