//! SQIsign verifying (public) keys.
//!
//! See [§4.5] (verification) and [§4.6] (binary format).
//!
//! [§4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
//! [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6

use subtle::ConstantTimeEq;

use crate::curves::montgomery::Curve;
use crate::fields::Fp2;
use crate::curves::VerifyingKeyHint;
use crate::keys::{Signature, SignatureError, SIGNATURE_BYTES, VERIFYING_KEY_BYTES};

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
        let curve = Curve::new(A.into());

        Ok(VerifyingKey { curve, hint, bytes: *bytes })
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
    pub fn verify(
        &self,
        msg: &[u8],
        sig: &Signature,
    ) -> Result<(), SignatureError> {
        use crate::curves::{TorsionBasis, TorsionExponent};
        use crate::curves::isogeny::Kernel as CurveKernel;
        use crate::hash;
        use crate::params::{E_RSP, TORSION_EVEN_POWER};
        use crate::surfaces;

        let f = TORSION_EVEN_POWER;
        let e_rsp = E_RSP;

        // --- Algorithm 4.9, line 6–7: compute e'_rsp ---
        // https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
        let e_rsp_prime = e_rsp
            .checked_sub(sig.n_bt)
            .and_then(|v| v.checked_sub(sig.r_rsp))
            .ok_or(SignatureError::VerificationFailed)?;

        // --- Line 8: torsion basis on E_pk from hint_pk ---
        let basis_pk = TorsionBasis::from_hint(
            &self.curve,
            crate::curves::BasisHint::from_byte(u8::from(self.hint)),
        );

        // --- Line 9: challenge isogeny ---
        // Compute kernel: P_pk + [chl]Q_pk, then [2^n_bt] of that.
        // https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
        let kernel_gen = basis_pk.ladder3pt(&sig.chl);
        let mut K_chl = kernel_gen;
        for _ in 0..sig.n_bt {
            K_chl = K_chl.double();
        }
        let (curve_chl, _) = CurveKernel::new(K_chl)
            .isogeny(TorsionExponent::new(f - sig.n_bt), &[]);

        // --- Lines 10–11: torsion bases on E_aux and E_chl ---
        let basis_aux = TorsionBasis::from_hint(
            &sig.curve_aux,
            crate::curves::BasisHint::from_byte(u8::from(sig.hint_aux)),
        );
        let basis_chl = TorsionBasis::from_hint(
            &curve_chl,
            crate::curves::BasisHint::from_byte(u8::from(sig.hint_chl)),
        );

        // Scale bases to the correct order.
        // (P_aux, Q_aux) ← [2^(f − e'_rsp + 2)] ...
        let mut P_aux = basis_aux.R;
        let mut Q_aux = basis_aux.S;
        for _ in 0..(f - e_rsp_prime + 2) {
            P_aux = P_aux.double();
            Q_aux = Q_aux.double();
        }

        // (P_chl, Q_chl) ← [2^(f − e_rsp − r_rsp − 2)] ...
        let mut P_chl = basis_chl.R;
        let mut Q_chl = basis_chl.S;
        for _ in 0..(f - e_rsp - sig.r_rsp - 2) {
            P_chl = P_chl.double();
            Q_chl = Q_chl.double();
        }

        // --- Line 14: apply M_chl ---
        // TODO: parse M_chl and apply 2×2 matrix to (P_chl, Q_chl).

        // --- Lines 15–20: even response isogeny ---
        if sig.r_rsp > 0 {
            // TODO: compute even response via TwoIsogenyChainSmall.
        }

        // --- Lines 21–23: if e'_rsp = 0, skip (2,2)-isogeny ---
        if e_rsp_prime == 0 {
            let j = curve_chl.j_invariant();
            let chl_prime = hash::hash(self, &j, msg);
            return if sig.chl == chl_prime {
                Ok(())
            } else {
                Err(SignatureError::VerificationFailed)
            };
        }

        // --- Lines 24–26: (2,2)-isogeny chain ---
        let product = surfaces::EllipticProduct::new(curve_chl, sig.curve_aux);
        let kernel = surfaces::Kernel::new(
            product,
            (P_chl, P_aux),
            (Q_chl, Q_aux),
        );
        let (codomain, _) = kernel.isogeny(
            TorsionExponent::new(e_rsp_prime),
            &[],
        );

        // --- Lines 29–30: recompute challenge ---
        let j_com = codomain.E1.j_invariant();
        let chl_prime = hash::hash(self, &j_com, msg);

        if sig.chl == chl_prime {
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
            bytes.try_into().map_err(|_| SignatureError::InvalidLength {
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
