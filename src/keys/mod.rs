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
        isogeny::Kernel,
        montgomery::{Coefficient, Curve},
        scalar::Scalar,
    },
    fields::fp2::Fp2,
    hash::CHALLENGE_BYTES,
    params::{E_RSP, TORSION_2POWER_BYTES, TORSION_EVEN_POWER},
};

/// Wire-format offsets for the [`Signature`] encoding at the NIST-I
/// parameter set (148 bytes):
///
/// ```text
/// [ A_aux (64) | n_bt (1) | r_rsp (1) | M_chl (64) | chl (16) | hint_aux (1) | hint_chl (1) ]
/// ```
const M_CHL_COMP_BYTES: usize = E_RSP.div_ceil(8) as usize;
const M_CHL_BYTES: usize = 4 * M_CHL_COMP_BYTES;
const A_AUX_BYTES: usize = 64;
const M_CHL_OFFSET: usize = A_AUX_BYTES + 2;
const CHL_OFFSET: usize = M_CHL_OFFSET + M_CHL_BYTES;
const HINT_OFFSET: usize = CHL_OFFSET + CHALLENGE_BYTES;

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

    /// Computes the challenge isogeny on `basis_pk`, then maps
    /// `pre_iso` onto the codomain via the canonical isomorphism.
    ///
    /// `basis_pk` is the verifying-key torsion basis on `E_pk`.
    /// `pre_iso` is the post-response basis on a curve `E'` with
    /// `j(E') = j(E_chl)`. Returns the challenge codomain `E_chl`
    /// and the propagated image of `pre_iso` on `E_chl`.
    ///
    /// # Returns
    ///
    /// `None` if `j(pre_iso.curve()) ≠ j(E_chl)`, or if the
    /// isomorphism `E' → E_chl` is degenerate (λ_x = 0 or
    /// λ_z = 0).
    ///
    /// # Divergences
    ///
    /// The j-invariant check before [`Curve::isomorphism`] mirrors
    /// the C reference's debug assertion in
    /// `compute_challenge_codomain_signature`. Without it,
    /// [`Curve::isomorphism`] silently computes a degenerate
    /// λ_x / λ_z and produces a bogus iso evaluation —
    /// downstream basis points pushed through it are
    /// mathematically meaningless, the signature serializes, and
    /// verify rejects at the `(2,2)`-chain step.
    ///
    /// The [`Isomorphism::eval`] is applied to `pre_iso.PmQ`
    /// alongside `P` and `Q`. Never recompute the codomain `PmQ`
    /// via [`ProjectiveXOnlyPoint::projective_difference`]:
    /// downstream consumers ([`ChangeOfBasisMatrix::from_bases`])
    /// call [`TorsionBasis::lift`], and the sqrt branch chosen by
    /// `projective_difference` is fragile.
    ///
    /// Implements [ComputeChallengeIsogeny][Alg. 4.7] (Algorithm 4.7).
    ///
    /// [Alg. 4.7]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.7
    /// [`Curve::isomorphism`]: crate::curves::montgomery::Curve::isomorphism
    /// [`Isomorphism::eval`]: crate::curves::montgomery::Isomorphism::eval
    /// [`ProjectiveXOnlyPoint::projective_difference`]: crate::curves::montgomery::ProjectiveXOnlyPoint::projective_difference
    /// [`TorsionBasis::lift`]: crate::curves::TorsionBasis::lift
    /// [`ChangeOfBasisMatrix::from_bases`]: crate::curves::ChangeOfBasisMatrix::from_bases
    pub(crate) fn to_isogeny(
        self,
        basis_pk: &TorsionBasis,
        pre_iso: &TorsionBasis,
        n_bt: TorsionExponent,
    ) -> Option<(Curve, TorsionBasis)> {
        // Line 1: E_chl ← TwoIsogenyChain([2^n_bt]·(P + [chl]·Q), E_pk, f − n_bt)
        let mut kernel_point = basis_pk.scalar_mul_add(self.as_ref());
        for _ in 0..n_bt.value() {
            kernel_point = kernel_point.double();
        }
        let e_chain = TorsionExponent::try_from(TORSION_EVEN_POWER - n_bt.value()).ok()?;
        let (curve_chl, _) = Kernel::new(kernel_point).isogeny(e_chain, &[]);

        // Line 2: (P_chl, Q_chl, P_chl − Q_chl) ← IsomorphismMontgomeryCurves(E', P, Q,
        // P−Q, E_chl)
        let e_prime = pre_iso.P.curve();
        if e_prime.j_invariant() != curve_chl.j_invariant() {
            return None;
        }
        let iso = e_prime.isomorphism(&curve_chl)?;
        let basis_chl = TorsionBasis::from_propagated(
            iso.eval(&pre_iso.P),
            iso.eval(&pre_iso.PmQ),
            iso.eval(&pre_iso.Q),
        );

        // Line 3
        Some((curve_chl, basis_chl))
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

        // E_aux: Montgomery coefficient A ∈ F_{p²}.
        // Rejects A = ±2 (singular Montgomery model) — matches the
        // C reference's `ec_curve_verify_A` (`ec.c:169`).
        let a_aux_buf: &[u8; A_AUX_BYTES] = sig[..A_AUX_BYTES]
            .try_into()
            .expect("A_AUX_BYTES fits within SIGNATURE_BYTES");
        let A_aux = Fp2::from_bytes(a_aux_buf);
        let coefficient_aux = Coefficient::from(A_aux);
        if coefficient_aux.is_singular() {
            return Err(SignatureError::InvalidCurve);
        }
        let curve_aux = Curve::from(coefficient_aux);

        // n_bt, r_rsp: 1 byte each, bounded by f=248.
        let n_bt = TorsionExponent::try_from(sig[A_AUX_BYTES] as u32)
            .map_err(|_| SignatureError::NonCanonical)?;
        let r_rsp = TorsionExponent::try_from(sig[A_AUX_BYTES + 1] as u32)
            .map_err(|_| SignatureError::NonCanonical)?;

        // M_chl. Spec §4.5 Alg. 4.9 step 5 bounds each entry by
        // `2^(e_rsp − n_bt + 2)`; `checked_*` also catches
        // `n_bt > e_rsp + 2` (step 7 `e'_rsp ≥ 0`) at parse.
        let m_chl_bound = E_RSP
            .checked_add(2)
            .and_then(|v| v.checked_sub(n_bt.value()))
            .and_then(|v| TorsionExponent::try_from(v).ok())
            .ok_or(SignatureError::NonCanonical)?;
        let m_chl_buf: &[u8; M_CHL_BYTES] = sig[M_CHL_OFFSET..M_CHL_OFFSET + M_CHL_BYTES]
            .try_into()
            .expect("M_chl region fits within SIGNATURE_BYTES");
        let M_chl = ChallengeMatrix::parse(m_chl_buf, m_chl_bound)?;

        // chl: CHALLENGE_BYTES.
        let mut chl = [0u8; CHALLENGE_BYTES];
        chl.copy_from_slice(&sig[CHL_OFFSET..CHL_OFFSET + CHALLENGE_BYTES]);

        // hints: 1 byte each.
        let hint_aux = AuxiliaryHint::from(sig[HINT_OFFSET]);
        let hint_chl = ChallengeHint::from(sig[HINT_OFFSET + 1]);

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

        // E_aux.
        bytes[..A_AUX_BYTES].copy_from_slice(&self.curve_aux.coefficient().to_bytes());

        // n_bt, r_rsp: 1 byte each.
        bytes[A_AUX_BYTES] = self.n_bt.value() as u8;
        bytes[A_AUX_BYTES + 1] = self.r_rsp.value() as u8;

        // M_chl: 4 × M_CHL_COMP_BYTES, each entry as LE bytes.
        let e = &self.M_chl.entries;
        let scalars = [e[0][0], e[0][1], e[1][0], e[1][1]];
        for (idx, s) in scalars.iter().enumerate() {
            let s_bytes = s.to_le_bytes();
            let start = M_CHL_OFFSET + idx * M_CHL_COMP_BYTES;
            bytes[start..start + M_CHL_COMP_BYTES].copy_from_slice(&s_bytes[..M_CHL_COMP_BYTES]);
        }

        // chl: CHALLENGE_BYTES.
        let chl_bytes = self.chl.as_scalar().to_le_bytes();
        bytes[CHL_OFFSET..CHL_OFFSET + CHALLENGE_BYTES]
            .copy_from_slice(&chl_bytes[..CHALLENGE_BYTES]);

        // hints: 1 byte each.
        bytes[HINT_OFFSET] = u8::from(self.hint_aux);
        bytes[HINT_OFFSET + 1] = u8::from(self.hint_chl);

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
/// The matrix is applied to a [`TorsionBasis`] via its `Mul`
/// implementation (`&matrix * &basis`), which uses the biscalar
/// Montgomery ladder internally.
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

impl From<crate::curves::ChangeOfBasisMatrix> for ChallengeMatrix {
    fn from(m: crate::curves::ChangeOfBasisMatrix) -> Self {
        Self(m)
    }
}

impl ChallengeMatrix {
    /// Parse the wire encoding of M_chl and validate the spec entry bound.
    ///
    /// `data` is the `4 × M_CHL_COMP_BYTES`-byte M_chl region of a
    /// signature: entries `a, b, c, d` concatenated in little-endian
    /// order, each padded to a full [`Scalar`]. `bound` is the
    /// algebraic upper-bound exponent: every entry must satisfy
    /// `entry < 2^bound` ([§4.5][§4.5] Algorithm 4.9 step 5, where
    /// the bound is `e'_rsp + r_rsp + 2 = e_rsp − n_bt + 2`).
    ///
    /// On success, the constructed value carries `bound` in its
    /// underlying [`e`] field so downstream uses (including
    /// [`ChangeOfBasisMatrix::mul`]) read the algebraic bound rather
    /// than the looser encoding bit count. Out-of-bound entries
    /// return [`SignatureError::NonCanonical`].
    ///
    /// [§4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.5
    /// [`e`]: crate::curves::ChangeOfBasisMatrix
    /// [`ChangeOfBasisMatrix::mul`]: crate::curves::ChangeOfBasisMatrix::mul
    pub(crate) fn parse(
        data: &[u8; M_CHL_BYTES],
        bound: TorsionExponent,
    ) -> Result<ChallengeMatrix, SignatureError> {
        let parse_scalar = |offset: usize| -> Scalar {
            let mut buf = [0u8; TORSION_2POWER_BYTES];
            buf[..M_CHL_COMP_BYTES].copy_from_slice(&data[offset..offset + M_CHL_COMP_BYTES]);
            let mut limbs = [0u64; 4];
            for (i, chunk) in buf.chunks_exact(8).enumerate() {
                limbs[i] = u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
            }
            Scalar::from_limbs(limbs)
        };

        let matrix = crate::curves::ChangeOfBasisMatrix {
            entries: [
                [parse_scalar(0), parse_scalar(M_CHL_COMP_BYTES)],
                [
                    parse_scalar(2 * M_CHL_COMP_BYTES),
                    parse_scalar(3 * M_CHL_COMP_BYTES),
                ],
            ],
            e: bound,
        };
        if !matrix.entries_below_pow2(bound.value()) {
            return Err(SignatureError::NonCanonical);
        }

        Ok(ChallengeMatrix(matrix))
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

/// Multiply a `ChallengeMatrix` by a `TorsionBasis`, producing
/// a new `TorsionBasis` with transformed generators:
/// `R' = [a]R + [b]S`, `S' = [c]R + [d]S`, and `R'−S'` computed
/// via `ProjectiveXOnlyPoint::projective_difference`.
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
    /// The Montgomery curve coefficient describes a singular curve
    /// (`A == ±2`, i.e. discriminant `Δ = 4(A² − 4) = 0`). Such
    /// coefficients are rejected at parse — both for `VerifyingKey`'s
    /// `A` and `Signature`'s `A_aux` — to match the C reference's
    /// `ec_curve_verify_A` (`ec.c:169`).
    InvalidCurve,
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
            SignatureError::InvalidCurve => {
                write!(f, "singular Montgomery curve coefficient (A = ±2)")
            }
            SignatureError::VerificationFailed => write!(f, "signature verification failed"),
            SignatureError::KeyGenFailed => write!(f, "key generation failed"),
            SignatureError::SigningFailed => write!(f, "signing failed"),
        }
    }
}

impl core::error::Error for SignatureError {}
