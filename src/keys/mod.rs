//! SQIsign key types and signature.
//!
//! See [§4.3] (key generation), [§4.5] (verification), and [§4.6]
//! (binary format) of the SQIsign spec.
//!
//! [§4.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.3
//! [§4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
//! [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6

mod signing;
mod verifying;

pub use signing::SigningKey;
pub use verifying::VerifyingKey;

pub use crate::params::{SIGNATURE_BYTES, SIGNING_KEY_BYTES, VERIFYING_KEY_BYTES};

use crate::curves::montgomery::Curve;
use crate::curves::{AuxiliaryHint, ChallengeHint};
use crate::fields::fp2::Fp2;
use crate::hash::CHALLENGE_BYTES;
use crate::params::E_RSP;

/// A parsed, validated SQIsign signature.
///
/// Constructed from [`SIGNATURE_BYTES`] = 148 raw bytes via
/// [`Signature::from_bytes`] or `TryFrom<&[u8]>`. Parsing validates
/// structural integrity (correct field lengths) but does not verify
/// the signature — call [`VerifyingKey::verify`] for that.
///
/// # Wire format (148 bytes, NIST-I)
///
/// ```text
/// [ E_aux (64 B) | n_bt (1) | r_rsp (1) | M_chl (64 B) | chl (16 B) | hint_aux (1) | hint_chl (1) ]
/// ```
///
/// See [§4.6] for encoding details.
///
/// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
#[derive(Clone, Debug)]
pub struct Signature {
    /// The auxiliary curve E_aux.
    pub(crate) curve_aux: Curve,
    /// Number of backtracking steps n_bt.
    pub(crate) n_bt: u32,
    /// Largest integer r_rsp such that 2^n_bt divides the response
    /// isogeny degree.
    pub(crate) r_rsp: u32,
    /// Change-of-basis matrix M_chl (2×2, each component up to 16 bytes
    /// for NIST-I). Stored as 4 little-endian byte arrays.
    pub(crate) M_chl: [[u8; 32]; 4],
    /// The challenge chl (122 bits for NIST-I).
    pub(crate) chl: [u8; CHALLENGE_BYTES],
    /// Hint for torsion basis on E_aux.
    pub(crate) hint_aux: AuxiliaryHint,
    /// Hint for torsion basis on E_chl.
    pub(crate) hint_chl: ChallengeHint,
    /// The raw bytes (cached for re-serialization).
    bytes: [u8; SIGNATURE_BYTES],
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
            sig[..64].try_into().map_err(|_| SignatureError::NonCanonical)?,
        );
        let curve_aux = Curve::new(A_aux.into());

        // n_bt, r_rsp: 1 byte each.
        let n_bt = sig[64] as u32;
        let r_rsp = sig[65] as u32;

        // M_chl: 2×2 matrix, each component ⌈(e_rsp+7)/8⌉ bytes.
        let comp_bytes = ((E_RSP + 7) / 8) as usize;
        let m_offset = 66;
        let mut M_chl = [[0u8; 32]; 4];
        for i in 0..4 {
            let start = m_offset + i * comp_bytes;
            M_chl[i][..comp_bytes].copy_from_slice(&sig[start..start + comp_bytes]);
        }

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
            chl,
            hint_aux,
            hint_chl,
            bytes: *bytes,
        })
    }

    /// Serialize this signature to bytes.
    pub fn to_bytes(&self) -> [u8; SIGNATURE_BYTES] {
        self.bytes
    }

    /// View this signature as a byte slice.
    pub fn as_bytes(&self) -> &[u8; SIGNATURE_BYTES] {
        &self.bytes
    }
}

impl AsRef<[u8]> for Signature {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl TryFrom<&[u8]> for Signature {
    type Error = SignatureError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        let bytes: &[u8; SIGNATURE_BYTES] =
            bytes.try_into().map_err(|_| SignatureError::InvalidLength {
                name: "Signature",
                expected: SIGNATURE_BYTES,
                actual: bytes.len(),
            })?;
        Signature::from_bytes(bytes)
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
            SignatureError::InvalidLength { name, expected, actual } => {
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
