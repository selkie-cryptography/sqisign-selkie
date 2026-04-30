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
            eprintln!("VK: j(E_pk)={}", fp2_hex_short(&self.curve.j_invariant()));
            eprintln!("VK: n_bt={} r_rsp={}", sig.n_bt.value(), sig.r_rsp.value());
        }

        // --- Algorithm 4.9, line 6–7: compute e'_rsp ---
        // https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
        let e_rsp_prime = e_rsp
            .checked_sub(sig.n_bt.value())
            .and_then(|v| v.checked_sub(sig.r_rsp.value()))
            .ok_or(SignatureError::VerificationFailed)?;

        // --- Line 8: torsion basis on E_pk from hint_pk ---
        let basis_pk = TorsionBasis::from_hint(&self.curve, BasisHint::from_byte(u8::from(self.hint)))
            .ok_or(SignatureError::VerificationFailed)?;

        // --- Line 9: challenge isogeny ---
        // Compute kernel: P_pk + [chl]Q_pk, then [2^n_bt] of that.
        // https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
        #[cfg(test)]
        eprintln!("VK: sig.chl = {:?}", sig.chl.as_ref());

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
            let R_aff = &basis_pk.R.X * &basis_pk.R.Z.invert();
            let S_aff = &basis_pk.S.X * &basis_pk.S.Z.invert();
            let RS_aff = &basis_pk.RS.X * &basis_pk.RS.Z.invert();
            eprintln!("VK: R(=P) affine = {}", fp2_hex_short(&R_aff));
            eprintln!("VK: S(=P-Q) affine = {}", fp2_hex_short(&S_aff));
            eprintln!("VK: RS(=Q) affine = {}", fp2_hex_short(&RS_aff));
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
            eprintln!("VK: basis_pk.R.X={}", fp2_hex_short(&basis_pk.R.X));
            eprintln!("VK: basis_pk.R.Z={}", fp2_hex_short(&basis_pk.R.Z));
            eprintln!("VK: K_chl.X={}", fp2_hex_short(&K_chl.X));
            eprintln!("VK: K_chl.Z={}", fp2_hex_short(&K_chl.Z));
            let k_aff = &K_chl.X * &K_chl.Z.invert();
            eprintln!("VK: K_chl affine={}", fp2_hex_short(&k_aff));
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
            eprintln!(
                "VK: j(E_chl) after challenge isogeny = {}",
                fp2_hex_short(&curve_chl.j_invariant())
            );
        }

        // --- Lines 10–11: torsion bases on E_aux and E_chl ---
        let basis_aux = TorsionBasis::from_hint(&sig.curve_aux, BasisHint::from_byte(u8::from(sig.hint_aux)))
            .ok_or(SignatureError::VerificationFailed)?;
        let basis_chl = TorsionBasis::from_hint(&curve_chl, BasisHint::from_byte(u8::from(sig.hint_chl)))
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
            eprintln!("TRACE basis_chl.R aff={}", aff(&basis_chl.R));
            eprintln!("TRACE basis_chl.S aff={}", aff(&basis_chl.S));
            eprintln!("TRACE basis_aux.R aff={}", aff(&basis_aux.R));
        }

        // Algorithm 4.9 line 11:
        // Scale aux basis: double f − e'_rsp − 2 times.
        // Scale all three points to preserve PmQ (never recompute via sqrt).
        let aux_doubles = f
            .checked_sub(e_rsp_prime)
            .and_then(|v| v.checked_sub(2))
            .ok_or(SignatureError::VerificationFailed)?;
        let mut P_aux = basis_aux.R;
        let mut Q_aux = basis_aux.S;
        let mut PmQ_aux = basis_aux.RS;
        for _ in 0..aux_doubles {
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
        let chl_doubles = f
            .checked_sub(e_rsp_prime)
            .and_then(|v| v.checked_sub(sig.r_rsp.value()))
            .and_then(|v| v.checked_sub(2))
            .ok_or(SignatureError::VerificationFailed)?;
        let mut P_chl = basis_chl.R;
        let mut Q_chl = basis_chl.S;
        let mut PmQ_chl = basis_chl.RS;
        for _ in 0..chl_doubles {
            P_chl = P_chl.double();
            Q_chl = Q_chl.double();
            PmQ_chl = PmQ_chl.double();
        }

        // --- Line 14: apply M_chl ---
        let basis_chl_scaled = TorsionBasis::from_propagated(P_chl, Q_chl, PmQ_chl);
        let basis_chl_transformed = &sig.M_chl * &basis_chl_scaled;
        let (mut P_chl, mut Q_chl, mut PmQ_chl) = (
            basis_chl_transformed.R,
            basis_chl_transformed.S,
            basis_chl_transformed.RS,
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
            eprintln!(
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
            eprintln!("VERIFY M_chl[0][0]={}", dump(&sig.M_chl.entries[0][0]));
            eprintln!("VERIFY M_chl[0][1]={}", dump(&sig.M_chl.entries[0][1]));
            eprintln!("VERIFY M_chl[1][0]={}", dump(&sig.M_chl.entries[1][0]));
            eprintln!("VERIFY M_chl[1][1]={}", dump(&sig.M_chl.entries[1][1]));
            eprintln!(
                "VERIFY basis_chl_scaled (det_chl): R={}, S={}, RS={}",
                aff(&basis_chl_scaled.R),
                aff(&basis_chl_scaled.S),
                aff(&basis_chl_scaled.RS),
            );
        }

        // --- Lines 15–20: even response isogeny ---
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
            eprintln!(
                "EVEN_RSP: r_rsp={}, first_col_even={}",
                sig.r_rsp.value(),
                sig.M_chl.first_column_even()
            );
            eprintln!("EVEN_RSP: P_chl X_re={}", fp2_hex(&P_chl.X));
            eprintln!(
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
                eprintln!(
                    "EVEN_RSP: j(curve_chl) after={}",
                    fp2_hex(&curve_chl.j_invariant())
                );
            }
        }

        // --- Lines 21–23: if e'_rsp = 0, skip (2,2)-isogeny ---
        if e_rsp_prime == 0 {
            let j = curve_chl.j_invariant();
            let chl_prime = hash::hash(self, &j, msg);
            return if sig.chl == Challenge::from(chl_prime) {
                Ok(())
            } else {
                Err(SignatureError::VerificationFailed)
            };
        }

        // --- Lines 24–26: (2,2)-isogeny chain ---
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
            // Match the C reference's `VERIFY_KER ...` format from
            // `compute_commitment_curve_verify` (verify.c) so a
            // line-by-line diff catches the first divergence between
            // the two impls' chain inputs on KAT[0].
            //
            // C ref's `T1 = (B_chall_can.P, B_aux_can.P)`,
            // `T2 = (..., .Q)`, `T1m2 = (..., .PmQ)`. After the
            // common (P, P-Q, Q) shuffle in `to_hint`, our
            // (P_chl, Q_chl, PmQ_chl) and (P_aux, Q_aux, PmQ_aux)
            // line up with the C ref's basis fields directly.
            let dump_fp2 = |label: &str, v: &Fp2| {
                let bytes = v.to_bytes();
                let re: String = bytes[..32].iter().rev().map(|b| format!("{:02x}", b)).collect();
                let im: String = bytes[32..].iter().rev().map(|b| format!("{:02x}", b)).collect();
                eprintln!("VERIFY_KER {label}_re=0x{re}");
                eprintln!("VERIFY_KER {label}_im=0x{im}");
            };
            let dump_pt = |label: &str, p: &ProjectiveXOnlyPoint| {
                dump_fp2(&format!("{label}_X"), &p.X);
                dump_fp2(&format!("{label}_Z"), &p.Z);
                let aff = &p.X * &p.Z.invert();
                dump_fp2(&format!("{label}_aff"), &aff);
            };

            eprintln!("VERIFY_KER: pow={e_rsp_prime}");
            // Curves: print affine A coefficient.
            let e1_a = *curve_chl.coefficient().as_fp2();
            let e2_a = *sig.curve_aux.coefficient().as_fp2();
            dump_fp2("E1_A_aff", &e1_a);
            dump_fp2("E2_A_aff", &e2_a);

            // Six chain-input points: T1 = (P_chl, P_aux),
            // T2 = (Q_chl, Q_aux), T1m2 = (PmQ_chl, PmQ_aux).
            dump_pt("T1_P1",   &P_chl);
            dump_pt("T1_P2",   &P_aux);
            dump_pt("T2_P1",   &Q_chl);
            dump_pt("T2_P2",   &Q_aux);
            dump_pt("T1m2_P1", &PmQ_chl);
            dump_pt("T1m2_P2", &PmQ_aux);
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
            )
            .ok_or(SignatureError::VerificationFailed)?;
        #[cfg(test)]
        eprintln!("VERIFY_KER chain_splits=1");

        // --- Lines 29–30: recompute challenge ---
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
