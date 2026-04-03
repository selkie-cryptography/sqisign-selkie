//! SQIsign signing (secret) keys.
//!
//! A signing key contains the secret isogeny φ_sk : E₀ → E_pk represented
//! as a quaternion ideal I_sk, plus the change-of-basis matrix M_sk and the
//! public key.
//!
//! See [§4.3] (key generation), [§4.4] (signing), and [§4.6] (binary format).
//!
//! [§4.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.3
//! [§4.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.4
//! [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6

#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::keys::verifying::VerifyingKey;
use crate::keys::{Signature, SignatureError, SIGNING_KEY_BYTES, VERIFYING_KEY_BYTES};
use crate::params::{FP_ENCODED_BYTES, TORSION_2POWER_BYTES};
use crate::quaternions::algebra::{Coordinate, Denominator, Element};
use crate::quaternions::bigint::BigInt;
use crate::quaternions::lattice::LeftIdeal;
use crate::quaternions::precomputed::EXTREMAL_ORDERS;

/// An SQIsign signing (secret) key.
///
/// Contains the secret ideal I_sk, the change-of-basis matrix M_sk, and
/// the corresponding verifying key. The signing key always holds a cached
/// copy of the verifying key to prevent the signing oracle attack
/// described in [MystenLabs/ed25519-unsafe-libs].
///
/// # Wire format (353 bytes, NIST-I)
///
/// ```text
/// [  pk (65 B) | norm (32 B) | gen[0..3] (4×32 B) | M_sk (4×32 B) ]
/// ```
///
/// `norm` and `gen[0..3]` define the secret ideal I_sk = O₀⟨gen, norm⟩.
/// `gen[i]` are signed (two's complement LE); `norm` and M_sk entries
/// are unsigned LE.
///
/// See [§4.6] for full encoding details.
///
/// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
/// [MystenLabs/ed25519-unsafe-libs]: https://github.com/MystenLabs/ed25519-unsafe-libs
#[derive(Clone)]
pub struct SigningKey {
    /// The corresponding verifying (public) key, cached for safety.
    verifying_key: VerifyingKey,
    /// The secret ideal I_sk (left O₀-ideal).
    ideal: LeftIdeal<4>,
    /// Change-of-basis matrix M_sk: 2×2 over Z, stored as
    /// \[\[m00, m01\], \[m10, m11\]\] with entries mod 2^f.
    mat_sk: [[BigInt<4>; 2]; 2],
}

impl SigningKey {
    /// Generate a new random signing key.
    ///
    /// Corresponds to `SQIsign.KeyGen` ([§4.3], Algorithm 4.1):
    ///
    /// 1. Sample a random secret left O₀-ideal I_sk of prime norm
    ///    D_mix via `RandomIdealGivenNorm` ([§3.1.6])
    /// 2. Reduce to a smaller equivalent ideal via
    ///    `RandomEquivalentPrimeIdeal` ([§3.1.6])
    /// 3. Translate I_sk to the secret isogeny φ_sk : E₀ → E_pk
    ///    and evaluate on the torsion basis via `IdealToIsogeny` ([§3.2.3])
    /// 4. Generate the deterministic torsion basis hint for E_pk
    ///    via `TorsionBasisToHint` ([§2.2.3])
    /// 5. Compute the change-of-basis matrix M_sk via
    ///    `ChangeOfBasis` ([§2.2.5])
    /// 6. Encode pk = (E_pk, hint_pk) and sk = (pk, I_sk, M_sk)
    ///
    /// This is a probabilistic algorithm: steps 2–3 may fail, in which
    /// case a fresh ideal is sampled. Expected to succeed after O(1)
    /// attempts.
    ///
    /// [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
    /// [§2.2.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
    /// [§3.1.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.3.1
    /// [§3.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.3.2
    /// [§4.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.3
    pub fn generate(
        _rng: &mut impl rand_core::CryptoRngCore,
    ) -> Result<SigningKey, SignatureError> {
        // Requires: quaternion algebra, id2iso, pairings.
        todo!()
    }

    /// Construct a signing key from its byte representation.
    ///
    /// The first [`VERIFYING_KEY_BYTES`] bytes must be a valid verifying
    /// key. The remaining bytes encode the secret ideal I_sk and the
    /// change-of-basis matrix M_sk.
    pub fn from_bytes(bytes: &[u8; SIGNING_KEY_BYTES]) -> Result<SigningKey, SignatureError> {
        let vk_bytes: &[u8; VERIFYING_KEY_BYTES] = bytes[..VERIFYING_KEY_BYTES]
            .try_into()
            .map_err(|_| SignatureError::NonCanonical)?;
        let verifying_key = VerifyingKey::from_bytes(vk_bytes)?;

        let mut pos = VERIFYING_KEY_BYTES;

        // Parse I_sk: norm (32 bytes unsigned) + generator coords (4 × 32 bytes signed).
        let norm = BigInt::<4>::from_bytes_le_unsigned(
            bytes[pos..pos + FP_ENCODED_BYTES].try_into().unwrap(),
        );
        pos += FP_ENCODED_BYTES;

        let mut gen_coords = [BigInt::<4>::ZERO; 4];
        for coord in &mut gen_coords {
            *coord = BigInt::<4>::from_bytes_le_signed(
                bytes[pos..pos + FP_ENCODED_BYTES].try_into().unwrap(),
            );
            pos += FP_ENCODED_BYTES;
        }

        // Reconstruct I_sk = O₀⟨gen, norm⟩.
        // The C ref skips the denominator in encoding (it's coprime to
        // norm, so the ideal is the same). Denominator = 1.
        let gen = Element {
            a: Coordinate::from(gen_coords[0]),
            b: Coordinate::from(gen_coords[1]),
            c: Coordinate::from(gen_coords[2]),
            d: Coordinate::from(gen_coords[3]),
            denom: Denominator::ONE,
        };
        let ideal = LeftIdeal::new(&gen, &norm, EXTREMAL_ORDERS[0].order());

        // Parse M_sk: 4 × 32 bytes unsigned, row-major [[m00, m01], [m10, m11]].
        let mut mat_sk = [[BigInt::<4>::ZERO; 2]; 2];
        for row in &mut mat_sk {
            for entry in row.iter_mut() {
                *entry = BigInt::<4>::from_bytes_le_unsigned(
                    bytes[pos..pos + TORSION_2POWER_BYTES].try_into().unwrap(),
                );
                pos += TORSION_2POWER_BYTES;
            }
        }
        debug_assert_eq!(pos, SIGNING_KEY_BYTES);

        Ok(SigningKey {
            verifying_key,
            ideal,
            mat_sk,
        })
    }

    /// Serialize this signing key to bytes.
    pub fn to_bytes(&self) -> [u8; SIGNING_KEY_BYTES] {
        // TODO: encode from parsed fields (ideal + mat_sk + vk).
        todo!("SigningKey::to_bytes")
    }

    /// Get the verifying key corresponding to this signing key.
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.verifying_key
    }

    /// Sign a message, producing a detached signature.
    ///
    /// Corresponds to `SQIsign.Sign` ([§4.4], Algorithm 4.2):
    ///
    /// **Commitment** (lines 4–9):
    /// 1. Sample a random commitment ideal I_com of norm D_mix
    /// 2. Translate to the commitment isogeny φ_com : E₀ → E_com
    ///    via `IdealToIsogeny` ([§3.2.3])
    ///
    /// **Challenge** (line 10):
    /// 3. Compute chl ← HASH(pk ‖ j(E_com) ‖ msg)
    ///
    /// **Response** (lines 11–38):
    /// 4. Convert chl to the challenge ideal I_chl via M_sk and
    ///    `KernelDecomposedToIdeal` ([§3.2.6])
    /// 5. Sample response quaternion α_rsp from the intersection
    ///    lattice via `RandomEquivalentQuaternion` ([§4.4.3])
    /// 6. Compute backtracking via
    ///    `ComputeBacktrackingAndNormalize` ([§4.4.3])
    /// 7. Compute the response isogeny, split into odd and even
    ///    parts, using `SplitAuxiliaryIsogeny` ([§4.4.3]) or
    ///    `IdealToIsogeny` depending on e'_rsp
    /// 8. Compute the challenge isogeny via
    ///    `ComputeChallengeIsogeny` ([§4.4.2])
    /// 9. Encode σ = (E_aux, n_bt, r_rsp, M_chl, chl, hint_aux, hint_chl)
    ///
    /// This is probabilistic: several sub-algorithms may fail,
    /// requiring a restart with fresh randomness.
    ///
    /// [§3.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.3.2
    /// [§3.2.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.3.2
    /// [§4.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.4
    /// [§4.4.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.4
    /// [§4.4.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.4
    pub fn sign(&self, _msg: &[u8]) -> Result<Signature, SignatureError> {
        // Requires: quaternion algebra, id2iso, pairings, SHAKE256.
        todo!()
    }
}

impl From<&SigningKey> for VerifyingKey {
    fn from(sk: &SigningKey) -> VerifyingKey {
        *sk.verifying_key()
    }
}

impl TryFrom<&[u8; SIGNING_KEY_BYTES]> for SigningKey {
    type Error = SignatureError;

    fn try_from(bytes: &[u8; SIGNING_KEY_BYTES]) -> Result<Self, Self::Error> {
        SigningKey::from_bytes(bytes)
    }
}

impl TryFrom<&[u8]> for SigningKey {
    type Error = SignatureError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        let bytes: &[u8; SIGNING_KEY_BYTES] =
            bytes
                .try_into()
                .map_err(|_| SignatureError::InvalidLength {
                    name: "SigningKey",
                    expected: SIGNING_KEY_BYTES,
                    actual: bytes.len(),
                })?;
        SigningKey::from_bytes(bytes)
    }
}

impl core::fmt::Debug for SigningKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SigningKey")
            .field("verifying_key", &self.verifying_key)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "zeroize")]
impl Drop for SigningKey {
    fn drop(&mut self) {
        // TODO: zeroize ideal and mat_sk fields.
    }
}

#[cfg(feature = "zeroize")]
impl ZeroizeOnDrop for SigningKey {}
