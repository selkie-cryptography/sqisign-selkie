//! SQIsign verifying (public) keys.
//!
//! See [§4.5] (verification) and [§4.6] (binary format).
//!
//! [§4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
//! [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6

use subtle::ConstantTimeEq;

use crate::{
    curves::{
        BasisHint, TorsionBasis, TorsionExponent, VerifyingKeyHint,
        isogeny::Kernel as CurveKernel,
        montgomery::{Coefficient, Curve},
    },
    fields::fp2::Fp2,
    hash,
    keys::{Challenge, Signature, SignatureError, VERIFYING_KEY_BYTES},
    params::{E_RSP, TORSION_EVEN_POWER},
    surfaces,
};

/// An SQIsign verifying (public) key.
///
/// Contains the public curve E_pk (encoded as its Montgomery coefficient A)
/// and a hint for deterministic torsion basis generation.
///
/// # Wire format (65 bytes, NIST-I)
///
/// ```text
/// [    A ∈ F_{p²}    | hint_pk ]
/// [     64 bytes      |  1 B    ]
/// ```
#[derive(Copy, Clone, Debug)]
pub struct VerifyingKey {
    /// The public curve.
    pub(crate) curve: Curve,
    /// Hint for deterministic torsion basis generation on E_pk.
    pub(crate) hint: VerifyingKeyHint,
    /// Cached canonical byte encoding.
    pub(crate) bytes: [u8; VERIFYING_KEY_BYTES],
}

impl VerifyingKey {
    /// Deserialize a verifying key from a fixed-length byte array.
    ///
    /// Rejects with [`SignatureError::InvalidCurve`] when the encoded
    /// coefficient yields a singular Montgomery model (`A = ±2`),
    /// matching the C reference's `ec_curve_verify_A` parse-time
    /// check. Supersingularity beyond non-singularity is *not*
    /// checked here — the spec ([§4.5], Algorithm 4.9 steps 3–4)
    /// allows that check to be a byproduct of the verify chain
    /// rather than an explicit parse-time test, and that's what we
    /// rely on. See `docs/spec-compliance.md` (TODO) for the full
    /// table.
    ///
    /// [§4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
    pub fn from_bytes(bytes: &[u8; VERIFYING_KEY_BYTES]) -> Result<VerifyingKey, SignatureError> {
        let a_bytes: &[u8; 64] = bytes[..64]
            .try_into()
            .map_err(|_| SignatureError::NonCanonical)?;
        let A = Fp2::from_bytes(a_bytes);
        let coefficient = Coefficient::from(A);
        if coefficient.is_singular() {
            return Err(SignatureError::InvalidCurve);
        }
        let hint = VerifyingKeyHint::from(bytes[64]);
        let curve = Curve::from(coefficient);

        Ok(VerifyingKey {
            curve,
            hint,
            bytes: *bytes,
        })
    }

    /// Serialize this verifying key to bytes.
    pub fn to_bytes(&self) -> [u8; VERIFYING_KEY_BYTES] {
        self.bytes
    }

    /// View as a byte slice.
    pub fn as_bytes(&self) -> &[u8; VERIFYING_KEY_BYTES] {
        &self.bytes
    }

    /// The public curve E_pk.
    pub fn curve(&self) -> &Curve {
        &self.curve
    }

    /// Verify a signature on a message.
    ///
    /// Corresponds to `SQIsign.Verify` ([§4.5], Algorithm 4.9).
    /// See the doc comment on this method in the source for a
    /// step-by-step breakdown.
    ///
    /// [§4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
    pub fn verify(&self, msg: &[u8], sig: &Signature) -> Result<(), SignatureError> {
        let f = TORSION_EVEN_POWER;
        let e_rsp = E_RSP;

        // Algorithm 4.9, line 6–7: compute e'_rsp.
        // https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
        let e_rsp_prime = e_rsp
            .checked_sub(sig.n_bt.value())
            .and_then(|v| v.checked_sub(sig.r_rsp.value()))
            .ok_or(SignatureError::VerificationFailed)?;

        // Line 8: torsion basis on E_pk from hint_pk.
        // `from_hint` performs a bounded x-coordinate search whose
        // failure on adversarial curves is treated as a verification
        // failure (rather than an infinite loop) — see
        // `find_na_x_coord` / `find_nqr_factor` in `curves/mod.rs` and
        // the wycheproof verify `tcId = 49` ("fuzz crash: integer
        // overflow in find_na_x_coord") regression test.
        let basis_pk =
            TorsionBasis::from_hint(&self.curve, BasisHint::from_byte(u8::from(self.hint)))
                .ok_or(SignatureError::VerificationFailed)?;

        // Line 9: challenge isogeny.
        // Compute kernel: P_pk + [chl]Q_pk, then [2^n_bt] of that.
        // https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
        let kernel_gen = basis_pk.scalar_mul_add(sig.chl.as_ref());
        let mut K_chl = kernel_gen;
        for _ in 0..sig.n_bt.value() {
            K_chl = K_chl.double();
        }

        let (curve_chl, _) = CurveKernel::new(K_chl).isogeny(
            TorsionExponent::try_from(f - sig.n_bt.value())
                .map_err(|_| SignatureError::VerificationFailed)?,
            &[],
        );

        // Lines 10–11: torsion bases on E_aux and E_chl.
        let basis_aux =
            TorsionBasis::from_hint(&sig.curve_aux, BasisHint::from_byte(u8::from(sig.hint_aux)))
                .ok_or(SignatureError::VerificationFailed)?;
        let basis_chl =
            TorsionBasis::from_hint(&curve_chl, BasisHint::from_byte(u8::from(sig.hint_chl)))
                .ok_or(SignatureError::VerificationFailed)?;

        // Algorithm 4.9 line 11:
        // Scale aux basis: double f − e'_rsp − 2 times.
        // Scale all three points to preserve PmQ (never recompute via sqrt).
        let mut P_aux = basis_aux.P;
        let mut Q_aux = basis_aux.PmQ;
        let mut PmQ_aux = basis_aux.Q;
        for _ in 0..(f - e_rsp_prime - 2) {
            P_aux = P_aux.double();
            Q_aux = Q_aux.double();
            PmQ_aux = PmQ_aux.double();
        }

        // Algorithm 4.9 line 13:
        // Scale chl basis: double f − e'_rsp − r_rsp − 2 times.
        //
        // Scale ALL THREE basis points (R, S, RS) so that the
        // TorsionBasis carries the correct PmQ through the matrix
        // application. The C reference doubles B.P, B.Q, and B.PmQ
        // together via `ec_dbl_iter_basis`. Recomputing PmQ via
        // `projective_difference` after scaling would give a
        // different point (the sqrt picks a different branch).
        let mut P_chl = basis_chl.P;
        let mut Q_chl = basis_chl.PmQ;
        let mut PmQ_chl = basis_chl.Q;
        for _ in 0..(f - e_rsp_prime - sig.r_rsp.value() - 2) {
            P_chl = P_chl.double();
            Q_chl = Q_chl.double();
            PmQ_chl = PmQ_chl.double();
        }

        // Line 14: apply M_chl.
        let basis_chl_scaled = TorsionBasis::from_propagated(P_chl, Q_chl, PmQ_chl);
        let basis_chl_transformed = &sig.M_chl * &basis_chl_scaled;
        let (mut P_chl, mut Q_chl, mut PmQ_chl) = (
            basis_chl_transformed.P,
            basis_chl_transformed.PmQ,
            basis_chl_transformed.Q,
        );

        // Lines 15–20: even response isogeny.
        //
        // Push ALL THREE basis points (P, Q, PmQ) through the
        // isogeny. The C reference does this explicitly in
        // `two_response_isogeny_verify`. Never recompute PmQ via
        // projective_difference — the sqrt would pick a different
        // branch.
        let mut curve_chl = curve_chl;
        if sig.r_rsp.value() > 0 {
            let kernel_pt = if sig.M_chl.first_column_even() {
                Q_chl
            } else {
                P_chl
            };
            let mut K = kernel_pt;
            for _ in 0..(e_rsp_prime + 2) {
                K = K.double();
            }
            let (new_curve, images) = CurveKernel::new(K)
                .isogeny_small(
                    TorsionExponent::try_from(sig.r_rsp.value())
                        .map_err(|_| SignatureError::VerificationFailed)?,
                    &[P_chl, Q_chl, PmQ_chl],
                    false,
                )
                .map_err(|_| SignatureError::VerificationFailed)?;
            curve_chl = new_curve;
            P_chl = images[0];
            Q_chl = images[1];
            PmQ_chl = images[2];
        }

        // Lines 21–23: if e'_rsp = 0, skip (2,2)-isogeny.
        if e_rsp_prime == 0 {
            // With no (2,2)-chain to run, there's no implicit
            // supersingularity certificate from "chain succeeded" —
            // the chain's correct completion is what the spec
            // (§4.5) cites as the byproduct check on every other
            // verify branch. Mirror C ref's explicit
            // `ec_is_basis_four_torsion(B_chall_can, E_chall)` from
            // `verify.c:226`: assert that the canonical basis on
            // E_chl actually spans `E_chl[4]`. Equivalently: P, Q
            // both have order exactly 4, and `2P ≠ 2Q` (their
            // 2-torsion images generate independent subgroups).
            //
            // Without this check, a non-supersingular E_chl whose
            // basis happens to collapse to lower torsion would
            // proceed straight to the j-invariant compare and be
            // rejected by the final hash check — but the chain has
            // no opportunity to fail, so the implementation-level
            // signal the spec relies on never fires.
            let two_p = P_chl.double();
            let two_q = Q_chl.double();
            let four_p = two_p.double();
            let four_q = two_q.double();
            let p_has_order_4 =
                bool::from(four_p.is_identity()) && !bool::from(two_p.is_identity());
            let q_has_order_4 =
                bool::from(four_q.is_identity()) && !bool::from(two_q.is_identity());
            let independent = two_p != two_q;
            if !(p_has_order_4 && q_has_order_4 && independent) {
                return Err(SignatureError::VerificationFailed);
            }

            let j = curve_chl.j_invariant();
            let chl_prime = hash::hash(self, &j, msg);
            return if sig.chl == Challenge::from(chl_prime) {
                Ok(())
            } else {
                Err(SignatureError::VerificationFailed)
            };
        }

        // Lines 24–26: (2,2)-isogeny chain.
        //
        // Pass Montgomery points to from_montgomery, which lifts to
        // Jacobian internally. The chain does all Phase 1 doublings
        // in Jacobian to produce the correct (x, z²) representative
        // for the gluing step.
        //
        // Use the PmQ that was computed by the matrix application and
        // pushed through the even response isogeny — NEVER recompute
        // via projective_difference (ambiguous square root).
        let product = surfaces::EllipticProduct::new(curve_chl, sig.curve_aux);

        let kernel = surfaces::Kernel::from_montgomery(
            product,
            (P_chl, P_aux),
            (Q_chl, Q_aux),
            (PmQ_chl, PmQ_aux),
        )
        .ok_or(SignatureError::VerificationFailed)?;

        // Reject (vk, sig) inputs whose recovered kernel is not
        // isotropic for the 2^(e+2)-Weil pairing — i.e. cannot be
        // the kernel of any legitimate (2,2)-isogeny. Inside
        // `Kernel::isogeny` this same condition is a debug-only
        // invariant (sign-side kernels are isotropic by construction);
        // verify promotes it to a release-build rejection so a bogus
        // signature fails fast instead of running the chain on garbage
        // and rejecting on the final j-invariant compare.
        let e = TorsionExponent::try_from(e_rsp_prime)
            .map_err(|_| SignatureError::VerificationFailed)?;
        let kernel_e = TorsionExponent::try_from(e_rsp_prime + 2)
            .map_err(|_| SignatureError::VerificationFailed)?;
        if !kernel.is_isotropic(kernel_e) {
            return Err(SignatureError::VerificationFailed);
        }

        // Verification uses the deterministic
        // `theta_chain_compute_and_eval_verify` in C ref —
        // `randomize=false`. No RNG consumed.
        let (codomain, _) = kernel
            .isogeny(e, &[], None)
            .ok_or(SignatureError::VerificationFailed)?;

        // Lines 29–30: recompute challenge.
        let j_com = codomain.E1.j_invariant();
        let chl_prime = hash::hash(self, &j_com, msg);

        if sig.chl == Challenge::from(chl_prime) {
            Ok(())
        } else {
            Err(SignatureError::VerificationFailed)
        }
    }
}

impl AsRef<[u8]> for VerifyingKey {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl TryFrom<&[u8]> for VerifyingKey {
    type Error = SignatureError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        let bytes: &[u8; VERIFYING_KEY_BYTES] =
            bytes
                .try_into()
                .map_err(|_| SignatureError::InvalidLength {
                    name: "VerifyingKey",
                    expected: VERIFYING_KEY_BYTES,
                    actual: bytes.len(),
                })?;
        VerifyingKey::from_bytes(bytes)
    }
}

impl ConstantTimeEq for VerifyingKey {
    fn ct_eq(&self, other: &VerifyingKey) -> subtle::Choice {
        self.bytes.ct_eq(&other.bytes)
    }
}

impl PartialEq for VerifyingKey {
    fn eq(&self, other: &VerifyingKey) -> bool {
        self.ct_eq(other).into()
    }
}

impl Eq for VerifyingKey {}
