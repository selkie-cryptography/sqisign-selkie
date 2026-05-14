//! SQIsign verifying (public) keys.
//!
//! See [§4.5] (verification) and [§4.6] (binary format).
//!
//! [§4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
//! [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6

use subtle::ConstantTimeEq;

#[cfg(test)]
use crate::curves::montgomery::ProjectiveXOnlyPoint;
#[cfg(test)]
use crate::curves::scalar::Scalar;
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
    pub fn from_bytes(bytes: &[u8; VERIFYING_KEY_BYTES]) -> Result<VerifyingKey, SignatureError> {
        let a_bytes: &[u8; 64] = bytes[..64]
            .try_into()
            .map_err(|_| SignatureError::NonCanonical)?;
        let A = Fp2::from_bytes(a_bytes);
        let hint = VerifyingKeyHint::from(bytes[64]);
        let curve = Curve::from(Coefficient::from(A));

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

        #[cfg(test)]
        {
            let fp2_hex_short = |fp2val: &Fp2| {
                let bytes = fp2val.to_bytes();
                let re: String = bytes[..32]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                let im: String = bytes[32..]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                format!("0x{re}+i*0x{im}")
            };
            crate::selkie_trace!("VK: j(E_pk)={}", fp2_hex_short(&self.curve.j_invariant()));
            crate::selkie_trace!("VK: n_bt={} r_rsp={}", sig.n_bt.value(), sig.r_rsp.value());
        }

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
        #[cfg(test)]
        crate::selkie_trace!("VK: sig.chl = {:?}", sig.chl.as_ref());

        #[cfg(test)]
        {
            let fp2_hex_short = |fp2val: &Fp2| {
                let bytes = fp2val.to_bytes();
                let re: String = bytes[..32]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                format!("0x{re}")
            };
            // Check all three basis points
            let R_aff = &basis_pk.P.X * &basis_pk.P.Z.invert();
            let S_aff = &basis_pk.PmQ.X * &basis_pk.PmQ.Z.invert();
            let RS_aff = &basis_pk.Q.X * &basis_pk.Q.Z.invert();
            crate::selkie_trace!("VK: R(=P) affine = {}", fp2_hex_short(&R_aff));
            crate::selkie_trace!("VK: S(=P-Q) affine = {}", fp2_hex_short(&S_aff));
            crate::selkie_trace!("VK: RS(=Q) affine = {}", fp2_hex_short(&RS_aff));
        }

        let kernel_gen = basis_pk.scalar_mul_add(sig.chl.as_ref());
        let mut K_chl = kernel_gen;
        for _ in 0..sig.n_bt.value() {
            K_chl = K_chl.double();
        }

        #[cfg(test)]
        {
            let fp2_hex_short = |fp2val: &Fp2| {
                let bytes = fp2val.to_bytes();
                let re: String = bytes[..32]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                let im: String = bytes[32..]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                format!("0x{re}+i*0x{im}")
            };
            crate::selkie_trace!("VK: basis_pk.P.X={}", fp2_hex_short(&basis_pk.P.X));
            crate::selkie_trace!("VK: basis_pk.P.Z={}", fp2_hex_short(&basis_pk.P.Z));
            crate::selkie_trace!("VK: K_chl.X={}", fp2_hex_short(&K_chl.X));
            crate::selkie_trace!("VK: K_chl.Z={}", fp2_hex_short(&K_chl.Z));
            let k_aff = &K_chl.X * &K_chl.Z.invert();
            crate::selkie_trace!("VK: K_chl affine={}", fp2_hex_short(&k_aff));
        }

        let (curve_chl, _) = CurveKernel::new(K_chl).isogeny(
            TorsionExponent::try_from(f - sig.n_bt.value())
                .map_err(|_| SignatureError::VerificationFailed)?,
            &[],
        );

        #[cfg(test)]
        {
            let fp2_hex_short = |fp2val: &Fp2| {
                let bytes = fp2val.to_bytes();
                let re: String = bytes[..32]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                let im: String = bytes[32..]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                format!("0x{re}+i*0x{im}")
            };
            crate::selkie_trace!(
                "VK: j(E_chl) after challenge isogeny = {}",
                fp2_hex_short(&curve_chl.j_invariant())
            );
        }

        // Lines 10–11: torsion bases on E_aux and E_chl.
        let basis_aux =
            TorsionBasis::from_hint(&sig.curve_aux, BasisHint::from_byte(u8::from(sig.hint_aux)))
                .ok_or(SignatureError::VerificationFailed)?;
        let basis_chl =
            TorsionBasis::from_hint(&curve_chl, BasisHint::from_byte(u8::from(sig.hint_chl)))
                .ok_or(SignatureError::VerificationFailed)?;

        #[cfg(test)]
        {
            let fp2_hex = |fp2val: &Fp2| -> String {
                let bytes = fp2val.to_bytes();
                let re: String = bytes[..32]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                let im: String = bytes[32..]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                format!("0x{re}+i*0x{im}")
            };
            let aff = |p: &ProjectiveXOnlyPoint| -> String { fp2_hex(&(&p.X * &p.Z.invert())) };
            crate::selkie_trace!("TRACE basis_chl.P aff={}", aff(&basis_chl.P));
            crate::selkie_trace!("TRACE basis_chl.PmQ aff={}", aff(&basis_chl.PmQ));
            crate::selkie_trace!("TRACE basis_aux.P aff={}", aff(&basis_aux.P));
        }

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

        #[cfg(test)]
        {
            let fp2_short = |v: &Fp2| -> String {
                let bytes = v.to_bytes();
                let r: String = bytes[..8]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                format!("0x{r}")
            };
            let aff = |p: &ProjectiveXOnlyPoint| -> String { fp2_short(&(&p.X * &p.Z.invert())) };
            crate::selkie_trace!(
                "VERIFY post-M_chl: R={}, S={}, RS={}, R==S={}, R==RS={}",
                aff(&P_chl),
                aff(&Q_chl),
                aff(&PmQ_chl),
                P_chl == Q_chl,
                P_chl == PmQ_chl,
            );
            let dump = |s: &Scalar| -> String {
                let l = s.as_limbs();
                format!("{:016x}_{:016x}_{:016x}_{:016x}", l[3], l[2], l[1], l[0])
            };
            crate::selkie_trace!("VERIFY M_chl[0][0]={}", dump(&sig.M_chl.entries[0][0]));
            crate::selkie_trace!("VERIFY M_chl[0][1]={}", dump(&sig.M_chl.entries[0][1]));
            crate::selkie_trace!("VERIFY M_chl[1][0]={}", dump(&sig.M_chl.entries[1][0]));
            crate::selkie_trace!("VERIFY M_chl[1][1]={}", dump(&sig.M_chl.entries[1][1]));
            crate::selkie_trace!(
                "VERIFY basis_chl_scaled (det_chl): R={}, S={}, RS={}",
                aff(&basis_chl_scaled.P),
                aff(&basis_chl_scaled.PmQ),
                aff(&basis_chl_scaled.Q),
            );
        }

        // Lines 15–20: even response isogeny.
        //
        // Push ALL THREE basis points (P, Q, PmQ) through the
        // isogeny. The C reference does this explicitly in
        // `two_response_isogeny_verify`. Never recompute PmQ via
        // projective_difference — the sqrt would pick a different
        // branch.
        #[cfg(test)]
        {
            let fp2_hex = |fp2val: &Fp2| -> String {
                let bytes = fp2val.to_bytes();
                let re: String = bytes[..32]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                format!("0x{re}")
            };
            crate::selkie_trace!(
                "EVEN_RSP: r_rsp={}, first_col_even={}",
                sig.r_rsp.value(),
                sig.M_chl.first_column_even()
            );
            crate::selkie_trace!("EVEN_RSP: P_chl X_re={}", fp2_hex(&P_chl.X));
            crate::selkie_trace!(
                "EVEN_RSP: j(curve_chl) before={}",
                fp2_hex(&curve_chl.j_invariant())
            );
        }
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

            #[cfg(test)]
            {
                let fp2_hex = |fp2val: &Fp2| -> String {
                    let bytes = fp2val.to_bytes();
                    let re: String = bytes[..32]
                        .iter()
                        .rev()
                        .map(|b| format!("{:02x}", b))
                        .collect();
                    format!("0x{re}")
                };
                crate::selkie_trace!(
                    "EVEN_RSP: j(curve_chl) after={}",
                    fp2_hex(&curve_chl.j_invariant())
                );
            }
        }

        // Lines 21–23: if e'_rsp = 0, skip (2,2)-isogeny.
        if e_rsp_prime == 0 {
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

        #[cfg(test)]
        {
            let fp2_hex = |fp2val: &Fp2| {
                let bytes = fp2val.to_bytes();
                let re: String = bytes[..32]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                let im: String = bytes[32..]
                    .iter()
                    .rev()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                format!("0x{re}+i*0x{im}")
            };
            crate::selkie_trace!(
                "VERIFY: e_rsp_prime={e_rsp_prime} n_bt={} r_rsp={}",
                sig.n_bt.value(),
                sig.r_rsp.value()
            );
            crate::selkie_trace!("VERIFY: curve_chl j={}", fp2_hex(&curve_chl.j_invariant()));
            let aux_A = *sig.curve_aux.coefficient().as_fp2();
            crate::selkie_trace!("VERIFY: curve_aux A={}", fp2_hex(&aux_A));
            crate::selkie_trace!(
                "VERIFY: j(sig.curve_aux)={}",
                fp2_hex(&sig.curve_aux.j_invariant())
            );
            crate::selkie_trace!("VERIFY: P_chl.X={}", fp2_hex(&P_chl.X));
            crate::selkie_trace!("VERIFY: P_chl.Z={}", fp2_hex(&P_chl.Z));
            crate::selkie_trace!("VERIFY: Q_chl.X={}", fp2_hex(&Q_chl.X));
            crate::selkie_trace!("VERIFY: Q_chl.Z={}", fp2_hex(&Q_chl.Z));
            crate::selkie_trace!("VERIFY: P_aux.X={}", fp2_hex(&P_aux.X));
            crate::selkie_trace!("VERIFY: P_aux.Z={}", fp2_hex(&P_aux.Z));
        }

        let kernel = surfaces::Kernel::from_montgomery(
            product,
            (P_chl, P_aux),
            (Q_chl, Q_aux),
            (PmQ_chl, PmQ_aux),
        )
        .ok_or(SignatureError::VerificationFailed)?;
        let (codomain, _) = kernel
            .isogeny(
                TorsionExponent::try_from(e_rsp_prime)
                    .map_err(|_| SignatureError::VerificationFailed)?,
                &[],
                // Verification uses the deterministic
                // `theta_chain_compute_and_eval_verify` in C ref —
                // `randomize=false`. No RNG consumed.
                None,
            )
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
