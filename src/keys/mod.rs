//! SQIsign key types and signature.
//!
//! See [§4.3] (key generation), [§4.5] (verification), and [§4.6]
//! (binary format) of the SQIsign spec.
//!
//! [§4.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.3
//! [§4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
//! [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6

/// Known-answer test vectors from the C reference implementation.
#[cfg(any(test, feature = "expose-internals"))]
pub mod kat_data;
mod signing;
#[cfg(test)]
mod tests;
mod verifying;

use core::ops::Mul;

pub use signing::SigningKey;
pub use verifying::VerifyingKey;

pub use crate::params::{SIGNATURE_BYTES, SIGNING_KEY_BYTES, VERIFYING_KEY_BYTES};
use crate::{
    curves::{
        AuxiliaryHint, ChallengeHint, TorsionBasis, TorsionExponent,
        montgomery::{Coefficient, Curve},
        scalar::Scalar,
    },
    fields::fp2::Fp2,
    hash::CHALLENGE_BYTES,
    params::{E_RSP, TORSION_2POWER_BYTES},
};

/// A challenge value: an integer in [0, 2^e_chl) produced by HASH.
///
/// Wraps a [`Scalar`] since the challenge is used as a multiplier
/// in the 3-point ladder: `P + [ch]Q`.
///
/// [`Scalar`]: crate::curves::scalar::Scalar
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Challenge(Scalar);

impl Challenge {
    /// Derive the challenge via Fiat-Shamir: hash the public key,
    /// commitment curve, and message.
    ///
    /// `chl ← HASH(pk ∥ j(E_com) ∥ msg)` ([Alg. 4.2][Alg. 4.2], line 10).
    ///
    /// [Alg. 4.2]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.2
    pub(crate) fn derive(pk: &VerifyingKey, e_com: &Curve, msg: &[u8]) -> Self {
        let j_com = e_com.j_invariant();
        crate::hash::hash(pk, &j_com, msg).into()
    }

    /// The underlying scalar.
    pub fn as_scalar(&self) -> &Scalar {
        &self.0
    }
}

impl AsRef<Scalar> for Challenge {
    fn as_ref(&self) -> &Scalar {
        &self.0
    }
}

impl From<Challenge> for Scalar {
    fn from(c: Challenge) -> Self {
        c.0
    }
}

impl From<[u8; CHALLENGE_BYTES]> for Challenge {
    /// Decode from little-endian bytes.
    fn from(bytes: [u8; CHALLENGE_BYTES]) -> Self {
        let mut limbs = [0u64; 4];
        for (i, chunk) in bytes.chunks(8).enumerate() {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            limbs[i] = u64::from_le_bytes(buf);
        }
        Self(Scalar::from_limbs(limbs))
    }
}

/// A SQIsign signature.
///
/// Produced by [`SigningKey::sign`], or deserialized from
/// [`SIGNATURE_BYTES`] = 148 bytes via [`Signature::from_bytes`],
/// `TryFrom<&[u8; SIGNATURE_BYTES]>`, or `TryFrom<&[u8]>`.
/// Serialized via [`Signature::to_bytes`].
///
/// Deserialization validates structural integrity but does not
/// verify the signature — call [`VerifyingKey::verify`] for that.
///
/// # Wire format (148 bytes, NIST-I)
///
/// ```text
/// [ E_aux (64) | n_bt (1) | r_rsp (1) | M_chl (64) | chl (16) | hint_aux (1) | hint_chl (1) ]
/// ```
///
/// See [§4.6] for encoding details.
///
/// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
#[derive(Copy, Clone, Debug)]
pub struct Signature {
    /// The auxiliary curve E_aux.
    pub(crate) curve_aux: Curve,
    /// Number of backtracking steps n_bt.
    pub(crate) n_bt: TorsionExponent,
    /// Largest integer r_rsp such that 2^n_bt divides the response
    /// isogeny degree.
    pub(crate) r_rsp: TorsionExponent,
    /// Change-of-basis matrix M_chl.
    pub(crate) M_chl: ChallengeMatrix,
    /// The challenge chl (122 bits for NIST-I).
    pub(crate) chl: Challenge,
    /// Hint for torsion basis on E_aux.
    pub(crate) hint_aux: AuxiliaryHint,
    /// Hint for torsion basis on E_chl.
    pub(crate) hint_chl: ChallengeHint,
}

impl Signature {
    /// Parse and validate a signature from its byte encoding.
    ///
    /// Returns an error if the encoding is structurally invalid.
    /// Does **not** verify the signature — use
    /// [`VerifyingKey::verify`] for that.
    ///
    /// See [§4.6] for the encoding format.
    ///
    /// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
    pub fn from_bytes(bytes: &[u8; SIGNATURE_BYTES]) -> Result<Signature, SignatureError> {
        let sig = bytes;

        // E_aux: Montgomery coefficient A ∈ F_{p²} (64 bytes).
        let A_aux = Fp2::from_bytes(
            sig[..64]
                .try_into()
                .map_err(|_| SignatureError::NonCanonical)?,
        );
        let curve_aux = Curve::from(Coefficient::from(A_aux));

        // n_bt, r_rsp: 1 byte each, bounded by f=248.
        let n_bt =
            TorsionExponent::try_from(sig[64] as u32).map_err(|_| SignatureError::NonCanonical)?;
        let r_rsp =
            TorsionExponent::try_from(sig[65] as u32).map_err(|_| SignatureError::NonCanonical)?;

        // M_chl: 2×2 matrix, each component ⌈(e_rsp+7)/8⌉ bytes.
        let comp_bytes = E_RSP.div_ceil(8) as usize;
        let m_offset = 66;
        let M_chl =
            ChallengeMatrix::from_bytes(&sig[m_offset..m_offset + 4 * comp_bytes], comp_bytes)?;

        // chl: ⌈e_chl/8⌉ bytes.
        let chl_offset = m_offset + 4 * comp_bytes;
        let mut chl = [0u8; CHALLENGE_BYTES];
        chl.copy_from_slice(&sig[chl_offset..chl_offset + CHALLENGE_BYTES]);

        // hints: 1 byte each.
        let hint_offset = chl_offset + CHALLENGE_BYTES;
        let hint_aux = AuxiliaryHint::from(sig[hint_offset]);
        let hint_chl = ChallengeHint::from(sig[hint_offset + 1]);

        Ok(Signature {
            curve_aux,
            n_bt,
            r_rsp,
            M_chl,
            chl: chl.into(),
            hint_aux,
            hint_chl,
        })
    }

    /// Serialize this signature to its wire format.
    ///
    /// See [§4.6] for the encoding format.
    ///
    /// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
    pub fn to_bytes(&self) -> [u8; SIGNATURE_BYTES] {
        let mut bytes = [0u8; SIGNATURE_BYTES];

        // E_aux: 64 bytes.
        bytes[..64].copy_from_slice(&self.curve_aux.coefficient().to_bytes());

        // n_bt, r_rsp: 1 byte each.
        bytes[64] = self.n_bt.value() as u8;
        bytes[65] = self.r_rsp.value() as u8;

        // M_chl: 4 × comp_bytes, each entry as LE bytes.
        let comp_bytes = E_RSP.div_ceil(8) as usize;
        let m_offset = 66;
        let e = &self.M_chl.entries;
        let scalars = [e[0][0], e[0][1], e[1][0], e[1][1]];
        for (idx, s) in scalars.iter().enumerate() {
            let s_bytes = s.to_le_bytes();
            bytes[m_offset + idx * comp_bytes..m_offset + (idx + 1) * comp_bytes]
                .copy_from_slice(&s_bytes[..comp_bytes]);
        }

        // chl: CHALLENGE_BYTES.
        let chl_offset = m_offset + 4 * comp_bytes;
        let chl_bytes = self.chl.as_scalar().to_le_bytes();
        bytes[chl_offset..chl_offset + CHALLENGE_BYTES]
            .copy_from_slice(&chl_bytes[..CHALLENGE_BYTES]);

        // hints: 1 byte each.
        let hint_offset = chl_offset + CHALLENGE_BYTES;
        bytes[hint_offset] = u8::from(self.hint_aux);
        bytes[hint_offset + 1] = u8::from(self.hint_chl);

        bytes
    }
}

impl TryFrom<&[u8; SIGNATURE_BYTES]> for Signature {
    type Error = SignatureError;

    fn try_from(bytes: &[u8; SIGNATURE_BYTES]) -> Result<Self, Self::Error> {
        Signature::from_bytes(bytes)
    }
}

impl TryFrom<&[u8]> for Signature {
    type Error = SignatureError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        let bytes: &[u8; SIGNATURE_BYTES] =
            bytes
                .try_into()
                .map_err(|_| SignatureError::InvalidLength {
                    name: "Signature",
                    expected: SIGNATURE_BYTES,
                    actual: bytes.len(),
                })?;
        Signature::from_bytes(bytes)
    }
}

/// The 2×2 change-of-basis matrix M_chl from a signature.
///
/// Encodes a linear transformation on a torsion basis:
/// given (P, Q), computes (\[a\]P + \[b\]Q, \[c\]P + \[d\]Q) where
/// M = ((a, b), (c, d)). Each entry is a little-endian integer
/// stored in a fixed-size array of [`TORSION_2POWER_BYTES`] = 32
/// bytes (zero-padded).
///
/// The matrix is applied to a [`TorsionBasis`] via [`apply`](Self::apply),
/// which uses the biscalar Montgomery ladder internally.
///
/// See [§4.5] (Algorithm 4.9, line 14) and [§4.6].
///
/// [§4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
/// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
/// [`TORSION_2POWER_BYTES`]: crate::params::TORSION_2POWER_BYTES
/// The change-of-basis matrix M_chl from a signature (public).
#[derive(Copy, Clone, Debug)]
pub struct ChallengeMatrix(pub(crate) crate::curves::ChangeOfBasisMatrix);

impl core::ops::Deref for ChallengeMatrix {
    type Target = crate::curves::ChangeOfBasisMatrix;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ChallengeMatrix {
    /// Compute the challenge matrix from two torsion bases via the
    /// Tate pairing ([Alg. 2.5][Alg. 2.5]).
    ///
    /// [Alg. 2.5]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.2.5
    pub(crate) fn encode(
        full_basis: &TorsionBasis,
        target_basis: &TorsionBasis,
        e: TorsionExponent,
    ) -> Self {
        Self(crate::curves::ChangeOfBasisMatrix::from_bases(
            full_basis,
            target_basis,
            e,
        ))
    }
}

impl From<crate::curves::ChangeOfBasisMatrix> for ChallengeMatrix {
    fn from(m: crate::curves::ChangeOfBasisMatrix) -> Self {
        Self(m)
    }
}

impl ChallengeMatrix {
    /// Parse from raw signature bytes.
    ///
    /// `data` contains 4 × `comp_bytes` bytes: a, b, c, d
    /// concatenated in little-endian order. Each is zero-padded
    /// to [`TORSION_2POWER_BYTES`].
    ///
    /// [`TORSION_2POWER_BYTES`]: crate::params::TORSION_2POWER_BYTES
    pub(crate) fn from_bytes(
        data: &[u8],
        comp_bytes: usize,
    ) -> Result<ChallengeMatrix, SignatureError> {
        if data.len() < 4 * comp_bytes || comp_bytes > TORSION_2POWER_BYTES {
            return Err(SignatureError::NonCanonical);
        }
        let parse_scalar = |offset: usize| -> Scalar {
            let mut buf = [0u8; TORSION_2POWER_BYTES];
            buf[..comp_bytes].copy_from_slice(&data[offset..offset + comp_bytes]);
            let mut limbs = [0u64; 4];
            for (i, chunk) in buf.chunks_exact(8).enumerate() {
                limbs[i] = u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
            }
            Scalar::from_limbs(limbs)
        };
        let e = TorsionExponent::try_from((comp_bytes * 8) as u32)
            .map_err(|_| SignatureError::NonCanonical)?;
        Ok(ChallengeMatrix(crate::curves::ChangeOfBasisMatrix {
            entries: [
                [parse_scalar(0), parse_scalar(comp_bytes)],
                [parse_scalar(2 * comp_bytes), parse_scalar(3 * comp_bytes)],
            ],
            e,
        }))
    }

    /// Whether both first-column entries (a, c) are even.
    ///
    /// Used during verification to select the kernel point for the even
    /// response isogeny ([Alg. 4.9][Alg. 4.9], line 16): if both
    /// are even, the kernel comes from Q; otherwise from P.
    ///
    /// The C reference checks `mat[0][0]` and `mat[1][0]`
    /// (`two_response_isogeny_verify` in `verify.c`), which are our
    /// entries `a` and `c` — the **first column** of the matrix.
    /// The matrix is applied by columns (see the `Mul` impl), so the
    /// first column determines the first output point R'.
    ///
    /// [Alg. 4.9]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.9
    pub(crate) fn first_column_even(&self) -> bool {
        self.entries[0][0].as_limbs()[0] & 1 == 0 && self.entries[1][0].as_limbs()[0] & 1 == 0
    }
}

/// Multiply a [`ChallengeMatrix`] by a [`TorsionBasis`], producing
/// a new [`TorsionBasis`] with transformed generators:
/// R' = \[a\]R + \[b\]S, S' = \[c\]R + \[d\]S, and R'−S' computed
/// via [`ProjectiveXOnlyPoint::projective_difference`].
///
/// Uses the biscalar Montgomery ladder ([§8.2], Algorithm 8.8).
///
/// [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
impl Mul<&TorsionBasis> for &ChallengeMatrix {
    type Output = TorsionBasis;

    fn mul(self, basis: &TorsionBasis) -> TorsionBasis {
        self.0.mul(basis)
    }
}

/// Errors that can occur when parsing or verifying keys and signatures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignatureError {
    /// The input bytes had an invalid length.
    InvalidLength {
        /// Name of the type being deserialized.
        name: &'static str,
        /// Expected length in bytes.
        expected: usize,
        /// Actual length in bytes.
        actual: usize,
    },
    /// The encoded public key curve is not supersingular.
    NotSupersingular,
    /// The encoded field element is not canonical (>= p).
    NonCanonical,
    /// The verification equation was not satisfied.
    VerificationFailed,
    /// Key generation failed (probabilistic algorithm exhausted retries).
    KeyGenFailed,
    /// Signing failed (probabilistic algorithm exhausted retries).
    SigningFailed,
}

impl core::fmt::Display for SignatureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SignatureError::InvalidLength {
                name,
                expected,
                actual,
            } => {
                write!(f, "{name} must be {expected} bytes, got {actual}")
            }
            SignatureError::NotSupersingular => write!(f, "curve is not supersingular"),
            SignatureError::NonCanonical => write!(f, "non-canonical field element encoding"),
            SignatureError::VerificationFailed => write!(f, "signature verification failed"),
            SignatureError::KeyGenFailed => write!(f, "key generation failed"),
            SignatureError::SigningFailed => write!(f, "signing failed"),
        }
    }
}

impl core::error::Error for SignatureError {}
