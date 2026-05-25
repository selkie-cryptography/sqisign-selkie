//! SQIsign signing (secret) keys.
//!
//! A signing key contains the secret isogeny φ_sk : E₀ → E_pk represented
//! as a quaternion ideal I_sk, plus the change-of-basis matrix M_sk and the
//! verifying key.
//!
//! See [§4.3] (key generation), [§4.4] (signing), and [§4.6] (binary format).
//!
//! [§4.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.3
//! [§4.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.4
//! [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6

#[cfg(feature = "zeroize")]
use zeroize::ZeroizeOnDrop;

#[cfg(test)]
mod tests;

mod response;

use core::{
    fmt::{self, Debug},
    ops::Deref,
};

use crate::{
    curves::{
        AuxiliaryHint, BasisHint, ChallengeHint, ChangeOfBasisMatrix, TorsionBasis,
        TorsionExponent, VerifyingKeyHint, isogeny::IsogenyDegree, scalar::Scalar,
    },
    keys::{
        Challenge, ChallengeMatrix, SIGNING_KEY_BYTES, Signature, SignatureError,
        VERIFYING_KEY_BYTES, verifying::VerifyingKey,
    },
    params::{D_MIX, E_RSP, FP_ENCODED_BYTES, TORSION_2POWER_BYTES, TORSION_EVEN_POWER},
    quaternions::{
        algebra::{Coordinate, Denominator, Element},
        bigint::BigInt,
        lattice::{Lattice, LeftIdeal},
        precomputed::EXTREMAL_ORDERS,
    },
};

/// Wire-format offsets for the [`SigningKey`] encoding at the NIST-I
/// parameter set (353 bytes):
///
/// ```text
/// [ pk (65) | norm (32) | gen[0..4] (4·32) | M_sk (4·32) ]
/// ```
/// Byte offset of the ideal-norm field in the wire-format signing key.
const SK_NORM_OFFSET: usize = VERIFYING_KEY_BYTES;
/// Byte offset of the ideal generator field in the wire-format signing key.
const SK_GEN_OFFSET: usize = SK_NORM_OFFSET + FP_ENCODED_BYTES;
/// Number of quaternion coordinates in the encoded generator.
const SK_GEN_COORDS: usize = 4;
/// Byte length of the encoded ideal generator.
const SK_GEN_BYTES: usize = SK_GEN_COORDS * FP_ENCODED_BYTES;
/// Byte offset of the change-of-basis matrix `M_sk` in the signing key.
const SK_MSK_OFFSET: usize = SK_GEN_OFFSET + SK_GEN_BYTES;
/// Number of entries in the encoded `M_sk` matrix.
const SK_MSK_ENTRIES: usize = 4;
/// Byte length of the encoded `M_sk` matrix.
const SK_MSK_BYTES: usize = SK_MSK_ENTRIES * TORSION_2POWER_BYTES;
// Static checks: any layout drift surfaces here at compile time.
const _: () = assert!(SK_MSK_OFFSET + SK_MSK_BYTES == SIGNING_KEY_BYTES);

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
    /// Generator α of I_sk where I_sk = O₀⟨α, nrd(I_sk)⟩.
    ///
    /// The wire format ([§4.6]) encodes α's {1,i,j,k} coordinates
    /// directly. We store it alongside the HNF lattice because
    /// recovering α from HNF requires a brute-force search
    /// ([`LeftIdeal::generator`]).
    ///
    /// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
    ideal_gen: Element<4>,
    /// Change-of-basis matrix M_sk from (φ_sk(P₀), φ_sk(Q₀)) to B_pk.
    mat_sk: SigningKeyMatrix,
}

/// The secret change-of-basis matrix `M_sk` (part of the signing key).
///
/// `M_sk` is the 2×2 matrix mod `2^f` (full even torsion) such that
/// `(φ_sk(P₀), φ_sk(Q₀)) = M_sk · (P_pk, Q_pk)` ([Algorithm 4.1][Alg. 4.1],
/// line 9). The "full torsion" is intrinsic — every constructor
/// ([`SigningKeyMatrix::new`], [`SigningKeyMatrix::from_bases`],
/// [`From<ChangeOfBasisMatrix>`]) fixes the torsion exponent at
/// [`TorsionExponent::FULL`].
///
/// No `PartialEq` / `Eq` / `ConstantTimeEq`: comparing signing key
/// material is a code smell.
///
/// [Alg. 4.1]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.1
#[derive(Clone)]
pub(crate) struct SigningKeyMatrix(ChangeOfBasisMatrix);

impl SigningKeyMatrix {
    /// Constructs a [`SigningKeyMatrix`] from raw 2×2 scalar entries.
    ///
    /// Always uses the full torsion exponent `f` ([`TORSION_EVEN_POWER`])
    /// since `M_sk` entries are mod `2^f`.
    pub(crate) fn new(entries: [[Scalar; 2]; 2]) -> Self {
        Self(ChangeOfBasisMatrix {
            entries,
            e: TorsionExponent::FULL,
        })
    }

    /// Computes `M_sk` via the Tate pairing ([Algorithm 2.5][Alg. 2.5]).
    ///
    /// `M_sk ← ChangeOfBasis_{2^f}(E_pk, (φ_sk(P₀), φ_sk(Q₀)), (P_pk,
    /// Q_pk))` per [Algorithm 4.1][Alg. 4.1] line 9. The torsion
    /// exponent is fixed at [`TorsionExponent::FULL`] (`2^f`) — see the
    /// type-level documentation on [`SigningKeyMatrix`].
    ///
    /// Returns `None` when [`ChangeOfBasisMatrix::from_bases`] cannot
    /// invert the source basis matrix mod `2^f` — the caller (keygen)
    /// retries with a fresh ideal.
    ///
    /// [Alg. 2.5]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.2.5
    /// [Alg. 4.1]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.1
    pub(crate) fn from_bases(
        full_basis: &TorsionBasis,
        target_basis: &TorsionBasis,
    ) -> Option<Self> {
        Some(Self(ChangeOfBasisMatrix::from_bases(
            full_basis,
            target_basis,
            TorsionExponent::FULL,
        )?))
    }
}

impl Deref for SigningKeyMatrix {
    type Target = ChangeOfBasisMatrix;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<ChangeOfBasisMatrix> for SigningKeyMatrix {
    fn from(m: ChangeOfBasisMatrix) -> Self {
        Self(m)
    }
}

impl SigningKey {
    /// Constructs a [`SigningKey`] from validated components.
    ///
    /// Enforces the invariant that `ideal_gen` generates `ideal`:
    /// `LeftIdeal::new(&ideal_gen, &norm, order)` must produce the
    /// same HNF lattice as `ideal`. All construction paths go through
    /// this method.
    ///
    /// # Constant-time
    ///
    /// The `debug_assert` reconstructs the ideal from `ideal_gen` and
    /// compares lattice bases. This leaks timing information in debug
    /// builds but compiles out in release.
    fn from_parts(
        verifying_key: VerifyingKey,
        ideal: LeftIdeal<4>,
        ideal_gen: Element<4>,
        mat_sk: SigningKeyMatrix,
    ) -> Self {
        debug_assert!(
            {
                let reconstructed =
                    LeftIdeal::new(&ideal_gen, ideal.norm(), EXTREMAL_ORDERS[0].order());
                reconstructed.lattice().basis() == ideal.lattice().basis()
            },
            "ideal_gen must generate the same ideal"
        );
        Self {
            verifying_key,
            ideal,
            ideal_gen,
            mat_sk,
        }
    }

    /// Generates a new random signing key.
    ///
    /// Corresponds to `SQIsign.KeyGen` ([§4.3], Algorithm 4.1):
    ///
    /// 1. Sample a random secret left O₀-ideal I_sk of prime norm D_mix via
    ///    `RandomIdealGivenNorm` ([§3.1.6])
    /// 2. Reduce to a smaller equivalent ideal via `RandomEquivalentPrimeIdeal`
    ///    ([§3.1.6])
    /// 3. Translate I_sk to the secret isogeny φ_sk : E₀ → E_pk and evaluate on
    ///    the torsion basis via `IdealToIsogeny` ([§3.2.3])
    /// 4. Generate the deterministic torsion basis hint for E_pk via
    ///    `TorsionBasisToHint` ([§2.2.3])
    /// 5. Compute the change-of-basis matrix M_sk via `ChangeOfBasis`
    ///    ([§2.2.5])
    /// 6. Encode pk = (E_pk, hint_pk) and sk = (pk, I_sk, M_sk)
    ///
    /// This is a probabilistic algorithm: steps 2–3 may fail, in which
    /// case a fresh ideal is sampled. Expected to succeed after O(1)
    /// attempts.
    ///
    /// [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.2.3
    /// [§2.2.5]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.2.2.5
    /// [§3.1.6]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.3.1.6
    /// [§3.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.3.2.3
    /// [§4.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.3
    ///
    /// # Constant-time
    ///
    /// Variable-time. `TODO(ct)`: every step touches the secret ideal
    /// `I_sk`. `random_prime_norm_wide` is rejection sampling,
    /// `reduce_to_prime_norm` is non-CT LLL, and `to_isogeny` is
    /// non-CT (variable-time SuitableIdeals/RepresentInteger). Must be
    /// hardened before production use.
    pub fn generate<R: rand_core::CryptoRngCore>(
        rng: &mut R,
    ) -> Result<SigningKey, SignatureError> {
        // Sample a 48-byte seed from the caller's RNG, then hand it
        // off to [`generate_derand`], which instantiates its own
        // AES256-CTR-DRBG from it.
        let mut randomness = [0u8; crate::drbg::SEEDLEN];
        rng.fill_bytes(&mut randomness);
        Self::generate_derand(&randomness)
    }

    /// Generates a signing key deterministically from a 48-byte seed.
    ///
    /// The seed instantiates an AES256-CTR-DRBG (NIST SP 800-90A) that
    /// drives every random sampling step in key generation. Passing
    /// the same seed yields the same key; useful for KATs and
    /// reproducible tests.
    pub fn generate_derand(
        randomness: &[u8; crate::drbg::SEEDLEN],
    ) -> Result<SigningKey, SignatureError> {
        let mut drbg = crate::drbg::Aes256CtrDrbg::new(randomness);
        Self::generate_with_rng(&mut drbg)
    }

    /// Generates a signing key from a caller-owned RNG.
    ///
    /// Same algorithm as [`SigningKey::generate_derand`], but the
    /// caller supplies the RNG instead of this method instantiating
    /// its own AES-CTR-DRBG. Used by the keygen-then-sign cross-check
    /// path, which threads a single DRBG through both phases so its
    /// byte consumption pattern matches the SQIsign C reference.
    pub fn generate_with_rng<R: rand_core::CryptoRngCore>(
        rng: &mut R,
    ) -> Result<SigningKey, SignatureError> {
        // Bound the retry loop. Each iteration may fail in
        // reduce_to_prime_norm, narrow, or to_isogeny.
        for _iter in 0..1000 {
            // Line 2: I_sk ← RandomIdealGivenNorm(D_mix, true).
            // D_MIX = 2^512 + 75. The ideal is stored at `BigInt<30>`
            // so that `p · g_i ≈ 2^769` entries and the downstream
            // raw gram `c^T·G·c ≈ 2^1806` fit without truncation.
            let d_mix_wide: BigInt<30> = D_MIX.widen();
            let mut i_sk = match LeftIdeal::<30>::random_prime_norm_wide(
                &d_mix_wide,
                &EXTREMAL_ORDERS[0],
                rng,
            ) {
                Some(i) => i,
                None => continue,
            };

            // Line 4: I_sk ← RandomEquivalentPrimeIdeal(I_sk).
            // `reduce_to_prime_norm` operates at the ideal's storage
            // width `N=30`; `PRIME_W=30` keeps the internal pow_mod
            // in the Miller-Rabin check well above the 1026-bit
            // bound for a 513-bit modulus.
            if !i_sk.reduce_to_prime_norm::<30, _>(rng) {
                continue;
            }
            let i_sk_narrow = match i_sk.narrow() {
                Some(i) => i,
                None => continue,
            };

            // Line 5: E_pk, φ_sk(P₀), φ_sk(Q₀) ← IdealToIsogeny(I_sk).
            let (e_pk, phi_p, phi_q, phi_pmq) = match i_sk_narrow.to_isogeny(rng) {
                Some(r) => r,
                None => continue,
            };

            // Line 8: (P_pk, Q_pk), hint_pk ← TorsionBasisToHint(E_pk).
            let (basis_pk, basis_hint) = match TorsionBasis::to_hint(&e_pk) {
                Some(r) => r,
                None => continue,
            };

            // Line 9: M_sk ← ChangeOfBasis_{2^f}(E_pk, (φ_sk(P₀), φ_sk(Q₀)), (P_pk, Q_pk)).
            let eval_basis = TorsionBasis::from_propagated(phi_p, phi_pmq, phi_q);
            let Some(mat_sk) = SigningKeyMatrix::from_bases(&eval_basis, &basis_pk) else {
                continue; // basis lift failed — retry with fresh ideal
            };

            // Assemble the verifying key with its cached byte form.
            let hint_pk = VerifyingKeyHint::from(basis_hint.to_byte());
            let mut vk_bytes = [0u8; VERIFYING_KEY_BYTES];
            vk_bytes[..64].copy_from_slice(&e_pk.coefficient().to_bytes());
            vk_bytes[64] = u8::from(hint_pk);
            let verifying_key = VerifyingKey {
                curve: e_pk,
                hint: hint_pk,
                bytes: vk_bytes,
            };

            let Some(ideal_gen) = i_sk_narrow.generator() else {
                continue; // generator too large for brute-force recovery — retry
            };

            return Ok(Self::from_parts(
                verifying_key,
                i_sk_narrow,
                ideal_gen,
                mat_sk,
            ));
        }
        Err(SignatureError::KeyGenFailed)
    }

    /// Constructs a signing key from its byte representation.
    ///
    /// The first [`VERIFYING_KEY_BYTES`] bytes must be a valid verifying
    /// key. The remaining bytes encode the secret ideal I_sk and the
    /// change-of-basis matrix M_sk.
    pub fn from_bytes(bytes: &[u8; SIGNING_KEY_BYTES]) -> Result<SigningKey, SignatureError> {
        let vk_bytes: &[u8; VERIFYING_KEY_BYTES] = bytes[..VERIFYING_KEY_BYTES]
            .try_into()
            .expect("layout const: bytes spans VERIFYING_KEY_BYTES");
        let verifying_key = VerifyingKey::from_bytes(vk_bytes)?;

        // I_sk norm (32 bytes unsigned, positive odd).
        let norm_bytes: &[u8; FP_ENCODED_BYTES] = bytes
            [SK_NORM_OFFSET..SK_NORM_OFFSET + FP_ENCODED_BYTES]
            .try_into()
            .expect("layout const: bytes spans norm region");
        let norm = IsogenyDegree::from_bytes_le(norm_bytes).ok_or(SignatureError::NonCanonical)?;

        // I_sk generator coords (4 × 32 bytes signed two's complement LE).
        let mut gen_coords = [BigInt::<4>::ZERO; 4];
        for (i, coord) in gen_coords.iter_mut().enumerate() {
            let start = SK_GEN_OFFSET + i * FP_ENCODED_BYTES;
            let chunk: &[u8; FP_ENCODED_BYTES] = bytes[start..start + FP_ENCODED_BYTES]
                .try_into()
                .expect("layout const: bytes spans gen coord i");
            *coord = BigInt::<4>::from_bytes_le_signed(chunk);
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
        let norm_bigint = norm.to_bigint();
        let ideal = LeftIdeal::new(&gen, &norm_bigint, EXTREMAL_ORDERS[0].order());

        // M_sk: 4 × 32 bytes unsigned, row-major [[m00, m01], [m10, m11]].
        // Wire format encodes C ref's M_sk (basis_pk in NORMAL `(P, Q, P−Q)`
        // slot semantics on the eval side). Our internal `M_sk` uses the
        // SWAPPED convention from `from_propagated`/`from_hint` (matches our
        // 3-pt ladder semantics; see `from_hint` and `scalar_mul_add` docs).
        // The two are related by `internal = T · wire` where
        // `T = [[1, 1], [0, −1]]`, `T = T⁻¹`. Apply T at the byte boundary.
        let mut wire = [[Scalar::ZERO; 2]; 2];
        for (i, row) in wire.iter_mut().enumerate() {
            for (j, entry) in row.iter_mut().enumerate() {
                let idx = i * 2 + j;
                let start = SK_MSK_OFFSET + idx * TORSION_2POWER_BYTES;
                let chunk: &[u8; TORSION_2POWER_BYTES] = bytes[start..start + TORSION_2POWER_BYTES]
                    .try_into()
                    .expect("layout const: bytes spans M_sk entry");
                let b = BigInt::<4>::from_bytes_le_unsigned(chunk);
                *entry = Scalar::from(b);
            }
        }
        let f = TORSION_EVEN_POWER;
        let entries = [
            [
                wire[0][0].add_mod2k(&wire[1][0], f),
                wire[0][1].add_mod2k(&wire[1][1], f),
            ],
            [
                Scalar::ZERO.sub_mod2k(&wire[1][0], f),
                Scalar::ZERO.sub_mod2k(&wire[1][1], f),
            ],
        ];
        let mat_sk = SigningKeyMatrix::new(entries);

        Ok(Self::from_parts(verifying_key, ideal, gen, mat_sk))
    }

    /// Serializes this signing key to bytes.
    ///
    /// Layout: `[pk (65 B) | norm (32 B) | gen[0..3] (4×32 B) | M_sk (4×32
    /// B)]`.
    ///
    /// `gen[i]` are the {1,i,j,k} coordinates of the ideal generator α
    /// where I_sk = O₀⟨α, norm⟩, encoded as signed (two's complement)
    /// little-endian. `norm` and M_sk entries are unsigned LE.
    ///
    /// # Constant-time
    ///
    /// Variable-time. TODO(ct): the two's complement negation branches
    /// on the sign of secret generator coordinates. The signing key is
    /// secret-derived (Algorithm 4.1).
    pub fn to_bytes(&self) -> [u8; SIGNING_KEY_BYTES] {
        let mut out = [0u8; SIGNING_KEY_BYTES];

        // pk.
        out[..VERIFYING_KEY_BYTES].copy_from_slice(&self.verifying_key.to_bytes());

        // I_sk norm (FP_ENCODED_BYTES, unsigned LE).
        let norm = self.ideal.norm();
        for (i, limb) in norm.as_limbs().iter().enumerate() {
            let start = SK_NORM_OFFSET + i * 8;
            out[start..start + 8].copy_from_slice(&limb.to_le_bytes());
        }

        // I_sk generator coords (4 × FP_ENCODED_BYTES, signed two's
        // complement LE).
        let gen = &self.ideal_gen;
        let coords = [
            gen.a.as_bigint(),
            gen.b.as_bigint(),
            gen.c.as_bigint(),
            gen.d.as_bigint(),
        ];
        for (idx, coord) in coords.iter().enumerate() {
            let coord_start = SK_GEN_OFFSET + idx * FP_ENCODED_BYTES;
            let is_neg = bool::from(coord.is_negative()) && !bool::from(coord.is_zero());
            for (i, limb) in coord.as_limbs().iter().enumerate() {
                let start = coord_start + i * 8;
                out[start..start + 8].copy_from_slice(&limb.to_le_bytes());
            }
            if is_neg {
                // Two's complement on the 32-byte block: flip, +1.
                let block = &mut out[coord_start..coord_start + FP_ENCODED_BYTES];
                for b in block.iter_mut() {
                    *b = !*b;
                }
                let mut carry = 1u16;
                for b in block.iter_mut() {
                    carry += *b as u16;
                    *b = carry as u8;
                    carry >>= 8;
                }
            }
        }

        // M_sk (4 × TORSION_2POWER_BYTES, unsigned LE, row-major).
        // Convert internal (swapped-slot convention, see `from_bytes`)
        // to wire format (C ref's normal-slot convention) via
        // `wire = T · internal`, `T = [[1, 1], [0, −1]]`.
        let f = TORSION_EVEN_POWER;
        let m = &self.mat_sk.entries;
        let wire = [
            [
                m[0][0].add_mod2k(&m[1][0], f),
                m[0][1].add_mod2k(&m[1][1], f),
            ],
            [
                Scalar::ZERO.sub_mod2k(&m[1][0], f),
                Scalar::ZERO.sub_mod2k(&m[1][1], f),
            ],
        ];
        for (i, row) in wire.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                let idx = i * 2 + j;
                let start = SK_MSK_OFFSET + idx * TORSION_2POWER_BYTES;
                out[start..start + TORSION_2POWER_BYTES].copy_from_slice(&entry.to_le_bytes());
            }
        }

        out
    }

    /// Returns the verifying key corresponding to this signing key.
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.verifying_key
    }

    /// Signs a message, producing a detached signature.
    ///
    /// Implements [SQIsign.Sign][Alg. 4.2] ([Algorithm 4.2][Alg. 4.2]):
    ///
    /// **Commitment** (lines 4–9):
    /// 1. Sample a random commitment ideal `I_com` of norm `D_mix`.
    /// 2. Translate to the commitment isogeny `φ_com: E₀ → E_com` via
    ///    `IdealToIsogeny` ([§3.2.3]).
    ///
    /// **Challenge** (line 10):
    ///
    /// 3. Compute `chl ← HASH(pk ‖ j(E_com) ‖ msg)`.
    ///
    /// **Response** (lines 11–38):
    ///
    /// 4. Convert `chl` to the challenge ideal `I_chl` via `M_sk` and
    ///    `KernelDecomposedToIdeal` ([§3.2.6]).
    /// 5. Sample response quaternion `α_rsp` from the intersection lattice via
    ///    `RandomEquivalentQuaternion` ([§4.4.3]).
    /// 6. Compute backtracking via `ComputeBacktrackingAndNormalize`
    ///    ([§4.4.3]).
    /// 7. Compute the response isogeny, split into odd and even parts, using
    ///    `SplitAuxiliaryIsogeny` ([§4.4.3]) or `IdealToIsogeny` depending on
    ///    `e'_rsp`.
    /// 8. Compute the challenge isogeny via `ComputeChallengeIsogeny`
    ///    ([§4.4.2]).
    /// 9. Encode `σ = (E_aux, n_bt, r_rsp, M_chl, chl, hint_aux, hint_chl)`.
    ///
    /// This is probabilistic: several sub-algorithms may fail,
    /// requiring a restart with fresh randomness.
    ///
    /// [§3.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.3.2.3
    /// [§3.2.6]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.3.2.6
    /// [§4.4]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.4
    /// [§4.4.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.4.4.2
    /// [§4.4.3]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.4.4.3
    /// [Alg. 4.2]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.2
    pub fn sign<R: rand_core::CryptoRngCore>(
        &self,
        msg: &[u8],
        rng: &mut R,
    ) -> Result<Signature, SignatureError> {
        let mut randomness = [0u8; crate::drbg::SEEDLEN];
        rng.fill_bytes(&mut randomness);
        self.sign_derand(msg, &randomness)
    }

    /// Signs a message deterministically from a 48-byte seed.
    ///
    /// The seed instantiates an AES256-CTR-DRBG (NIST SP 800-90A) that
    /// drives every random sampling step in the commitment and
    /// response phases. For a fixed `(self, msg, randomness)` triple
    /// the output is deterministic, so this is the entry point used
    /// for KATs and reproducible tests. Different signing keys (or
    /// different messages) yield different signatures even when
    /// seeded with the same 48 bytes.
    pub fn sign_derand(
        &self,
        msg: &[u8],
        randomness: &[u8; crate::drbg::SEEDLEN],
    ) -> Result<Signature, SignatureError> {
        let mut drbg = crate::drbg::Aes256CtrDrbg::new(randomness);
        self.sign_with_rng(msg, &mut drbg)
    }

    /// Signs using a caller-owned RNG.
    ///
    /// Same algorithm as [`SigningKey::sign_derand`], but the RNG is
    /// provided by the caller rather than instantiated from a 48-byte
    /// seed. Pair with [`SigningKey::generate_with_rng`] on the same
    /// DRBG instance to reproduce the SQIsign C reference's
    /// byte-consumption pattern (one DRBG seeded via
    /// `randombytes_init`, consumed by `crypto_sign_keypair` and
    /// then `crypto_sign` in order).
    pub fn sign_with_rng<R: rand_core::CryptoRngCore>(
        &self,
        msg: &[u8],
        rng: &mut R,
    ) -> Result<Signature, SignatureError> {
        // Status: response phase operates at `LeftIdeal<N_RESP>`
        // (= 22 limbs, enough for the 1399-bit sampling radius).
        //
        // - Intersection is `I_com ∩ (I_sk · I_chl)` — corrected from the earlier bug
        //   where it was `I_sk ∩ (I_sk · I_chl)`.
        // - Sampling uses the full spec radius `D²_mix · 2^{e_rsp + f + 1}`.
        // - `d_rsp = nrd(α_rsp) / (D²_mix · 2^{f - n_bt})` is computed at wide width
        //   with the full `D²_mix` division.
        // - `I_com,rsp = O₀·α_rsp + O₀·(q_rsp · D_mix)` is built via
        //   [`LeftIdeal::from_generator`] at `LeftIdeal<N_RESP>` with the full ~513-bit
        //   norm, then reduced to a prime-norm equivalent and narrowed to
        //   `LeftIdeal<4>` for `to_isogeny`.
        // - `compute_even_response` is invoked with `α_rsp mod 2^r_rsp` (since that
        //   generates the same even-response ideal), narrowed to `Element<4>`.
        //
        // Known remaining issues:
        //
        // - `q_rsp` is still extracted as a single `u64` from the odd part of `d_rsp`.
        //   The spec allows `q_rsp` up to `D_rsp ≈ 2^126`, so a `u64` is insufficient
        //   in general. TODO: widen `q_rsp` to `BigInt<4>`.
        // - Not yet tested end-to-end against KATs — correctness of the wide-width
        //   response phase depends on every step above and needs integration testing.

        let f = TORSION_EVEN_POWER;
        let e_rsp = E_RSP;

        // Line 1: Parse sk
        let e_pk = self.verifying_key.curve();

        // Line 2: basis on E_pk. For an honestly-generated signing
        // key, `from_hint`'s bounded search is guaranteed to succeed
        // — but a tampered or otherwise-malformed `self.bytes` could
        // present a curve / hint pair the search can't satisfy. Treat
        // that as `SigningFailed` rather than panic.
        let basis_pk = TorsionBasis::from_hint(
            e_pk,
            BasisHint::from_byte(u8::from(self.verifying_key.hint)),
        )
        .ok_or(SignatureError::SigningFailed)?;

        // Line 3: while true do
        //
        // Debug short-cut: when `SELKIE_MAX_SIGN_ITERS=N` is set, abort
        // after the Nth iter regardless of progress. Used to keep
        // debugging cycles tight while diagnostics are noisy.
        #[cfg(test)]
        let max_iters: u32 = std::env::var("SELKIE_MAX_SIGN_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);
        #[cfg(not(test))]
        let max_iters: u32 = 1000;
        for _iter in 0..max_iters {
            #[cfg(test)]
            let _iter_start = std::time::Instant::now();

            // Commitment (lines 4–9).

            // Line 4: I_com ← RandomIdealGivenNorm(D_mix, true).
            // Stored at `BigInt<30>` for the same reason as keygen:
            // `p·g_i` and raw gram entries exceed narrower widths.
            let d_mix_wide: BigInt<30> = D_MIX.widen();
            let mut i_com = match LeftIdeal::<30>::random_prime_norm_wide(
                &d_mix_wide,
                &EXTREMAL_ORDERS[0],
                rng,
            ) {
                Some(i) => i,
                None => {
                    continue;
                }
            };

            // Lines 5–6: RandomEquivalentPrimeIdeal.
            if !i_com.reduce_to_prime_norm::<30, _>(rng) {
                continue;
            }

            // Narrow to LeftIdeal<4> for to_isogeny.
            let i_com_narrow = match i_com.narrow() {
                Some(i) => i,
                None => {
                    continue;
                }
            };

            // Line 7: E_com, P_com, Q_com ← IdealToIsogeny(I_com)
            //
            // The fourth return value is the propagated `PmQ_com`,
            // needed by `SplitAuxiliaryKernel::from_bases` so the
            // response (2,2)-chain's `lift_basis` sees a projective
            // rep consistent with the chain's evaluation history of
            // `P_com` and `Q_com`.
            let (e_com, p_com, q_com, pmq_com) = match i_com_narrow.to_isogeny(rng) {
                Some(r) => r,
                None => {
                    continue;
                }
            };

            // Challenge (line 10).
            let chl = Challenge::derive(&self.verifying_key, &e_com, msg);

            // Response (lines 11–38).

            // Line 11: (c₁, c₂) ← M_sk · (1, chl)
            //
            // `kernel_to_ideal` mirrors C-ref's
            // `id2iso_kernel_dlogs_to_ideal_even`, which expects
            // (c₁, c₂) computed from C-ref's *wire-format* M_sk
            // (normal P/Q slot convention). Our internal M_sk is
            // stored in the swapped-slot convention
            // (matching scalar_mul_add's ladder3pt semantics — see
            // `from_bytes`), so for this kernel-to-ideal computation
            // we must undo the T = [[1, 1], [0, −1]] transform to
            // recover the wire M_sk. `T = T⁻¹`, so wire = T · internal.
            let chl_scalar: Scalar = chl.into();
            let m = &self.mat_sk.entries;
            let m_wire = [
                [
                    m[0][0].add_mod2k(&m[1][0], f),
                    m[0][1].add_mod2k(&m[1][1], f),
                ],
                [
                    Scalar::ZERO.sub_mod2k(&m[1][0], f),
                    Scalar::ZERO.sub_mod2k(&m[1][1], f),
                ],
            ];
            let c1 = m_wire[0][0].add_mod2k(&m_wire[0][1].mul_mod2k(&chl_scalar, f), f);
            let c2 = m_wire[1][0].add_mod2k(&m_wire[1][1].mul_mod2k(&chl_scalar, f), f);

            // Line 12: I'_chl ← KernelDecomposedToIdeal(c₁, c₂)
            let c1_big = BigInt::<4>::from(c1);
            let c2_big = BigInt::<4>::from(c2);
            let i_chl_prime =
                match TorsionBasis::kernel_to_ideal(&c1_big, &c2_big, TorsionExponent::FULL) {
                    Some(ideal) => ideal,
                    None => {
                        continue;
                    }
                };

            const N_RESP: usize = 30;
            let i_sk_w = self.ideal.widen::<N_RESP>();
            let i_com_w = i_com.widen::<N_RESP>();
            let i_chl_prime_w = i_chl_prime.widen::<N_RESP>();

            // Line 14: α_rsp ← RandomEquivalentQuaternion(I_com ∩ I_sk · I_chl)
            //
            // The spec (Algorithm 4.3) uses a sampling radius of
            //   B = D_rsp · D²_mix · 2^{f+1} ≈ 2^1399   (NIST-I).

            // The product I_sk · I_chl has entries up to ~1663 bits.
            // The intersection via dual→sum→dual cubes the entry
            // size through 3×3 subdeterminants, requiring ~5000-bit
            // The product I_sk · I_chl has entries up to ~1775 bits.
            // The dual→sum→dual intersection cubes entry sizes
            // through 3×3 subdeterminants — incompatible with fixed
            // width. Use the stacked-kernel intersection instead,
            // which needs ~4× the entry size for intermediates.
            //
            // Hadamard bound on kernel vectors: ~4 × (denom + entry)
            // ≈ 4 × 3438 ≈ 13752 bits. After B₁·a: ~15527 bits
            // ≈ 243 limbs. But the HNF of the result is bounded by
            // the ideal norms (~381 bits), so narrowing succeeds.
            // Line 14: sample from (I_chl' ∩ I_sk) ∩ I̅_com.
            //
            // # Divergences
            //
            // The spec (Algorithm 4.2, line 14) writes
            // `I̅_com ∩ I_sk · I_chl`. The C ref (sign.c:81-85)
            // computes:
            //   1. I_chl_secret = I_chl' ∩ I_sk (intersection)
            //   2. conjugate I_com → I̅_com
            //   3. intersect I_chl_secret with I̅_com
            //
            // Note: the C ref uses intersection (∩) of I_chl'
            // with I_sk, not the ideal product (·). The spec's
            // notation `I_sk · I_chl` is ambiguous between
            // product and intersection; the C ref uses
            // intersection throughout. We match the C ref.
            let i_chl_lat = Lattice::<N_RESP>::from(*i_chl_prime_w.lattice());
            let i_sk_lat = Lattice::<N_RESP>::from(*i_sk_w.lattice());
            #[cfg(test)]
            let _t_int = std::time::Instant::now();
            // W=60: entries start at ~60 limbs (d*B products); xgcd
            // elimination may grow them. W=120 is the safe Hadamard
            // bound but 4x slower. W=60 is adequate in practice —
            // validate by completing a full signing round-trip.
            let i_chl_sk = match i_chl_lat.intersection_via_kernel::<150>(&i_sk_lat) {
                Some(l) => l,
                None => {
                    continue;
                }
            };
            let i_com_conj = i_com_w.lattice().conjugate();
            let i_chl_sk_lat = Lattice::<N_RESP>::from(i_chl_sk);
            let i_com_conj_lat = Lattice::<N_RESP>::from(i_com_conj);

            #[cfg(test)]
            let _t_int2 = std::time::Instant::now();
            // C-ref's `quat_lattice_intersect` (lattice.c:127) uses the
            // dual-sum-dual identity `L1 ∩ L2 = (L1* + L2*)*`. The
            // kernel-of-quotient method produces the same covolume but
            // a different Z-module when inputs have very imbalanced
            // magnitudes (KAT-1 sign iter 0: ~520-bit i_chl_sk vs
            // ~140-bit conj(I_com)). For byte-equality with C-ref we
            // must use the dual-sum-dual path here.
            let intersection = {
                let compact = i_chl_sk_lat.compact_intersection::<128>(&i_com_conj_lat);
                #[cfg(test)]
                {
                    let hnf = i_chl_sk_lat.intersection_via_dual_sum_dual::<500>(&i_com_conj_lat);
                    assert_eq!(
                        compact, hnf,
                        "compact_intersection::<128> != dual_sum_dual::<500>"
                    );
                }
                match compact {
                    Some(l) => l,
                    None => {
                        continue;
                    }
                }
            };
            let intersection_lat = Lattice::<N_RESP>::from(intersection);

            // Sampling radius — C-ref formula, not spec.
            //
            // # Divergences
            //
            // Spec (Algorithm 4.3) uses `radius = D_rsp · D²_mix ·
            // 2^{f+1}` ≈ 2^1398 at NIST-I, assuming `I_com` still
            // has its original `D_mix` norm. After line 5's
            // `RandomEquivalentPrimeIdeal`, `I_com`'s norm is a
            // prime `N(I_com) ≪ D_mix`, so the spec radius is way
            // too large — α ends up with `nrd` on the order of
            // `2^1398`, giving an odd-part `q_rsp` that doesn't fit
            // in `BigInt<4>`.
            //
            // The C ref (`sign.c:53-59`) uses
            //   `bound = (2^{SQIsign_response_length} − 1) ·
            //            N(I_chl_secret) · N(I_com)`
            // ≈ 2^(e_rsp) · 2^f · N(I_sk) · N(I_com) ≈ 2^626 at
            // NIST-I — small enough that `q_rsp ≤ 2^126` fits.
            // `radius_crf = (2^e_rsp − 1) · lattice_content`
            // Use the *actual* lattice covolume-derived norm as
            // `lattice_content`. The formula `N(I_com) · 2^f ·
            // N(I_sk)` matches the formal intersection norm only
            // when the three pairwise intersections are exactly
            // coprime and `intersection_via_kernel` produces
            // exactly the mathematical intersection. In practice,
            // `intersection_via_kernel` (or one of its inputs) can
            // produce a sublattice off by a small denom factor —
            // refreshing from the actual lattice covolume gives
            // the value `nrd(α)` is divisible by, which is what
            // the downstream divisibility check needs.
            // `intersection_lat = I_chl_secret ∩ conj(I_com)` is the
            // lattice C-ref calls `lattice_hom_chall_to_com`. It is
            // NOT a left O0-ideal (conj(I_com) is a *right* ideal,
            // and the intersection is closed under neither side
            // alone), so `LeftIdeal::refresh_norm` -- which assumes
            // left-ideal structure and checks `[O0:I] = N(I)^2` --
            // spuriously rejects on most iterations and produces a
            // wrong norm on others.
            //
            // C-ref (`sign.c:140`) instead computes
            //   `lattice_content = N(I_chl_secret) · N(I_com)`
            // directly from the constituent ideal norms, with
            //   `N(I_chl_secret) = N(I_chl) · N(I_sk) = 2^f · N(I_sk)`
            // since `I_chl` (norm `2^f`) and `I_sk` (odd prime norm)
            // are coprime. We mirror that here -- product of the
            // three known norms.
            let n_chl: BigInt<N_RESP> = *i_chl_prime_w.norm();
            let n_sk: BigInt<N_RESP> = *i_sk_w.norm();
            let n_com: BigInt<N_RESP> = *i_com_w.norm();
            let lattice_content_r: BigInt<N_RESP> = n_chl.vt_mul(&n_sk).vt_mul(&n_com);
            let two_to_e_rsp: BigInt<N_RESP> = BigInt::<N_RESP>::ONE << e_rsp;
            let two_e_rsp_minus_one = two_to_e_rsp.vt_sub(&BigInt::<N_RESP>::ONE);
            let radius = two_e_rsp_minus_one.vt_mul(&lattice_content_r);
            // The intersection lattice has entries up to ~1920 bits
            // (BigInt<30>). The gram computation squares these:
            // ~3840 bits ≈ 60 limbs. Use W=64 for margin.
            //
            // In-iter rejection sampling: Selkie's
            // `intersection_via_kernel` and C-ref's
            // `quat_lattice_intersect` produce the same lattice as a
            // Z-module but different basis representations. Sample
            // distributions therefore differ, and Selkie's α from
            // the kernel-method basis often fails the post-
            // backtracking divisibility check `nrd(α/tmp) %
            // (lc_pre/2^bt) == 0` even though α is genuinely in the
            // abstract intersection. Resample up to a small number
            // of times within one iter before giving up.
            #[cfg(test)]
            let _t_sample = std::time::Instant::now();

            let mut sample_result: Option<(_, u32, BigInt<N_RESP>, BigInt<N_RESP>)> = None;
            for _try in 0..8u32 {
                let alpha_try = match intersection_lat.sample_from_ball::<64, _>(&radius, rng) {
                    Some(a) => a,
                    None => continue,
                };
                let (alpha_norm, n_bt_try) = alpha_try.compute_backtracking();
                let (num_w, den_sq_w) = alpha_norm.norm_w::<N_RESP>();
                let (q1, r1) = num_w.vt_div_rem(&den_sq_w);
                if !bool::from(r1.is_zero()) {
                    continue;
                }
                let lc_post: BigInt<N_RESP> = lattice_content_r >> n_bt_try;
                let (_q2, r2) = q1.vt_div_rem(&lc_post);
                if !bool::from(r2.is_zero()) {
                    continue;
                }
                sample_result = Some((alpha_norm, n_bt_try, num_w, den_sq_w));
                break;
            }

            let (alpha_rsp_w, n_bt, nrd_num_w, nrd_den_w) = match sample_result {
                Some(t) => t,
                None => {
                    continue;
                }
            };
            // Lines 16–20: degree computations — C-ref formula.
            //
            // # Divergences
            //
            // The spec (Algorithm 4.2, line 16) uses
            //   `d_rsp = nrd(α) / (D²_mix · 2^{f-n_bt})`
            // which assumes `I_com` still has its original `D_mix`
            // norm. But line 5 replaces `I_com` with an equivalent
            // prime-norm ideal via `RandomEquivalentPrimeIdeal`, so
            // `nrd(α)` is a multiple of `N(I_com_reduced)` — not
            // `D_mix²`. The C ref (`sign.c:144-148`) uses
            //   `lattice_content = N(I_chl_secret) · N(I_com)`
            //   `degree_full_resp = nrd(α) / lattice_content`
            // which is the correct relation given `α ∈ conj(I_com) ∩
            // I_chl_secret` (left-ideal intersection of coprime-norm
            // ideals). Under the `RandomEquivalentPrimeIdeal`
            // transformation this replaces `D_mix² · 2^{f-n_bt}`
            // with `N(I_com) · N(I_chl_secret)`.
            //
            // `N(I_chl_secret) = N(I_chl) · N(I_sk) = 2^f · N(I_sk)`
            // since `I_chl` (norm `2^f`) and `I_sk` (odd prime norm)
            // are coprime.
            // Use the actual lattice covolume-derived norm (computed
            // earlier via `intersection_ideal.refresh_norm`). The
            // formula `N(I_com)·2^f·N(I_sk)` matches the formal
            // intersection norm only when the inputs are exactly
            // coprime AND `intersection_via_kernel` produces the
            // mathematically exact intersection. In practice the
            // refresh-derived value is what `nrd(α)` is divisible
            // by, so use it for the divisibility check too (not
            // just for the radius).
            //
            // Mirror C-ref's `compute_backtracking_signature`
            // (`sign.c:166-169`): after `quat_alg_make_primitive`
            // divides α's coordinates by their gcd `tmp`, divide
            // `lattice_content` by `2^backtracking` where
            // `backtracking = ν_2(tmp)`. Without this, the
            // divisibility check `nrd(α) ÷ lattice_content` fails
            // whenever `n_bt > 0`, because `nrd(α_primitive) =
            // nrd(α) / tmp²` shrinks faster than `lattice_content`
            // does — the missing `2^backtracking` factor is the gap.
            let lattice_content: BigInt<N_RESP> = lattice_content_r >> n_bt;

            let d_rsp_wide = {
                let (q1, r1) = nrd_num_w.vt_div_rem(&nrd_den_w);
                if !bool::from(r1.is_zero()) {
                    continue;
                }
                let (q2, r2) = q1.vt_div_rem(&lattice_content);
                if !bool::from(r2.is_zero()) {
                    continue;
                }
                q2
            };
            let r_rsp_val = d_rsp_wide.trailing_zeros();
            let d_rsp_shifted = d_rsp_wide >> r_rsp_val;
            // q_rsp = d_rsp / 2^r_rsp (odd part). For NIST-I the
            // response-degree odd part is bounded by `D_rsp ≈ 2^126`,
            // so it fits in `BigInt<4>` (256 bits) with room to
            // spare. Narrow from the wide working width.
            let q_rsp: BigInt<4> = match d_rsp_shifted.narrow_to::<4>() {
                Some(q) => q,
                None => {
                    continue;
                }
            };
            let e_rsp_prime = e_rsp - r_rsp_val - n_bt;

            let n_bt_te =
                TorsionExponent::try_from(n_bt).map_err(|_| SignatureError::SigningFailed)?;
            let r_rsp =
                TorsionExponent::try_from(r_rsp_val).map_err(|_| SignatureError::SigningFailed)?;
            let e_rsp_prime_te = TorsionExponent::try_from(e_rsp_prime)
                .map_err(|_| SignatureError::SigningFailed)?;

            // Line 19: I_com,rsp = O₀⟨ᾱ_rsp, N(I_com)·q_rsp⟩.
            //
            // # Divergences
            //
            // The spec writes `O₀·α_rsp + O₀·(q_rsp·D_mix)`.
            // The C ref (sign.c:165-169):
            //   1. Conjugates α_rsp (ᾱ_rsp)
            //   2. Uses norm = N(I_com) · q_rsp (reduced norm, not the original D_mix)
            //
            // We match the C ref. N(I_com) is the prime norm from
            // reduce_to_prime_norm (~2^133), not D_mix (~2^513).
            let o0_w = EXTREMAL_ORDERS[0].widen::<N_RESP>();
            let alpha_rsp_conj = alpha_rsp_w.conjugate();
            let q_rsp_wide: BigInt<N_RESP> = q_rsp.widen();
            let i_com_norm_w: BigInt<N_RESP> = i_com.norm().widen();
            let i_com_rsp_norm_w = i_com_norm_w.vt_mul(&q_rsp_wide);
            // KNOWN BUG: this `from_generator_mod_hnf` produces a
            // lattice whose canonical HNF differs from C-ref's
            // `quat_lideal_create` output for the same (α, N) inputs.
            // Diagnosed via `[I_COM_RESP_CREF]` byte-diff vs Selkie's
            // i_com_rsp dump (`SELKIE_DUMP_I_COM_RSP=1`): same norm,
            // same diagonal, different off-diagonal cols ⇒ different
            // lattices. The `from_generator` (classical-HNF) variant
            // overflows at width 30 (`Lattice::sum` Hadamard bound
            // exceeds the working width when α has denom > 1 and norm
            // ≈ 2^260). Suspect bug source is in either `mul_direct`
            // (`O·α` step) or `sum_mod`'s HNF reduction; needs a
            // bytewise diff against C-ref's `quat_lattice_alg_elem_mul`
            // and `quat_lattice_add` to localize.
            let mut i_com_rsp_w = match LeftIdeal::<30>::from_generator_mod_hnf(
                &alpha_rsp_conj,
                &i_com_rsp_norm_w,
                o0_w.order(),
            ) {
                Some(i) => i,
                None => {
                    continue;
                }
            };
            // Match the C ref (`quat_lideal_norm`): derive the stored
            // norm from the lattice covolume rather than trusting the
            // passed-in `i_com_rsp_norm_w`. With only `α` primitive
            // in the 2-adic sense, the true `n(I)` can differ from
            // `N(I_com) · q_rsp` by odd content in `α`, and downstream
            // `smallest_equiv_narrow` rejects valid δ when `self.norm`
            // is inflated.
            if i_com_rsp_w.refresh_norm::<120>().is_none() {
                continue;
            }
            // Narrow `i_com_rsp` to `LeftIdeal<8>` for the intersection
            // step. Earlier the response phase reduced this to width 4
            // via `smallest_equiv_narrow::<120>()`, but the reduction
            // replaces `I_com_rsp` with `δ⁻¹·I_com_rsp` and the
            // resulting `I_inter` is no longer the same ideal as the
            // C reference's `lideal_aux_resp_com`. The downstream
            // kernel-isotropy condition in `SplitAuxiliaryIsogeny`
            // breaks (`splitting_index_count() = 0` on every input).
            // Keeping width 8 fits `~2^257` norms with room to spare.
            let i_com_rsp = match i_com_rsp_w.narrow_to::<8>() {
                Some(i) => i,
                None => {
                    continue;
                }
            };

            // Lines 21–33: compute response isogeny
            let (mut p_chl, mut q_chl, mut pmq_chl);
            let curve_aux;
            let p_aux;
            let q_aux;
            let pmq_aux;

            if e_rsp_prime > 0 {
                // Lines 22–27: auxiliary isogeny path.
                //
                // After reduce_to_prime_norm, I_com_rsp has a small
                // prime norm (~2^15). The intersection with I_aux
                // (norm ~2^126) produces a ~141-bit norm ideal,
                // within FixedDegreeIsogeny's bound (< 2^246).
                let aux_norm = (BigInt::<4>::ONE << e_rsp_prime).vt_sub(&q_rsp);
                let i_aux = match LeftIdeal::<4>::random_norm(&aux_norm, &EXTREMAL_ORDERS[0], rng) {
                    Some(i) => i,
                    None => {
                        continue;
                    }
                };
                // Line 24: E_aux, P_aux, Q_aux ← IdealToIsogeny(I_{com,rsp} ∩ I_aux)
                //
                // Use the true lattice intersection via
                // `intersection_via_kernel` — the same method used
                // for the sampling intersection above. This matches
                // the C reference (`quat_lideal_inter` →
                // `quat_lattice_intersect`).
                //
                // `Lattice::product` is NOT equivalent here: as a
                // Z-module it generates `{α·β : α ∈ I_1, β ∈ I_2}`
                // but this is the **two-sided** ideal product, not
                // the left-ideal intersection — even when the
                // norms are coprime. For the Deuring correspondence
                // we need the left-ideal intersection, which
                // corresponds to the pullback isogeny.
                //
                // Widen inputs to `BigInt<8>` so that intersection
                // HNF entries (up to `~n(I_1)·n(I_2) ≈ 2^374`,
                // about 6 limbs) fit, then `smallest_equiv_narrow`
                // produces the final `LeftIdeal<4>`.
                // `i_com_rsp` is already at `LeftIdeal<8>` (un-reduced
                // — see comment at line 956). Widen `i_aux` to match
                // and intersect at width 8.
                let i_aux_w: LeftIdeal<8> = i_aux.widen::<8>();
                let i_com_rsp_lat_w: Lattice<8> = (*i_com_rsp.lattice()).into();
                let i_aux_lat_w: Lattice<8> = (*i_aux_w.lattice()).into();
                #[cfg(test)]
                let _t_inter = std::time::Instant::now();
                // Use dual-sum-dual intersection — `intersection_via_kernel`
                // produces a wrong lattice for sign's i_inter inputs
                // (i_com_rsp norm ≈ 2^263 ∩ i_aux norm ≈ 2^122),
                // diagnosed via `[I_INTER_CREF]` byte-diff vs C-ref's
                // `quat_lideal_inter`. Same-norm, same-diagonal HNF, but
                // off-diagonal cols disagree — distinct canonical HNFs =
                // distinct lattices. dual-sum-dual matches C-ref.
                let inter_hnf_w8 = {
                    let compact = i_com_rsp_lat_w.compact_intersection::<64>(&i_aux_lat_w);
                    #[cfg(test)]
                    {
                        let hnf = i_com_rsp_lat_w.intersection_via_dual_sum_dual::<200>(&i_aux_lat_w);
                        assert_eq!(
                            compact, hnf,
                            "compact_intersection::<64> != dual_sum_dual::<200>"
                        );
                    }
                    match compact {
                        Some(h) => h,
                        None => {
                            continue;
                        }
                    }
                };
                let inter_norm_w8: BigInt<8> = i_com_rsp
                    .norm()
                    .widen::<8>()
                    .vt_mul(&i_aux.norm().widen::<8>());
                let o0_w8 = EXTREMAL_ORDERS[0].widen::<8>();
                let mut i_inter_w =
                    LeftIdeal::<8>::from_parts(inter_hnf_w8, inter_norm_w8, *o0_w8.order());
                #[cfg(test)]
                let _t_refresh = std::time::Instant::now();
                if i_inter_w.refresh_norm::<40>().is_none() {
                    continue;
                }
                if *i_inter_w.norm() == BigInt::<8>::ONE {
                    continue;
                }
                // Pass the un-reduced `i_inter_w` (at width 8) directly
                // to `to_isogeny`. The C reference's
                // `dim2id2iso_arbitrary_isogeny_evaluation` is invoked
                // on the un-reduced `lideal_aux_resp_com`; `find_uv`
                // reduces internally on a local copy, but `lideal->norm`
                // (used in the post-matrix `1/(nrd(I)·d₁)` scaling)
                // remains the original. Pre-reducing here would replace
                // `I_inter` with a δ⁻¹·equivalent and the resulting
                // `(u, v, β₁, β₂)` decomposition differs from what the
                // C reference produces — `SplitAuxiliaryIsogeny`'s
                // kernel-isotropy condition then fails with
                // `splitting_index_count() = 0`.
                let (_e_aux_prime, p_aux_prime, q_aux_prime, pmq_aux_prime) =
                    match i_inter_w.to_isogeny(rng) {
                        Some(r) => r,
                        None => {
                            continue;
                        }
                    };

                let basis_com = TorsionBasis::from_propagated(p_com, pmq_com, q_com);
                let basis_aux_prime =
                    TorsionBasis::from_propagated(p_aux_prime, pmq_aux_prime, q_aux_prime);
                let kernel = match response::split_aux::SplitAuxiliaryKernel::from_bases(
                    basis_com,
                    basis_aux_prime,
                    q_rsp,
                    e_rsp_prime_te,
                    r_rsp,
                ) {
                    Some(k) => k,
                    None => continue,
                };
                let split = match kernel.isogeny(rng) {
                    Some(s) => s,
                    None => continue,
                };
                curve_aux = split.auxiliary_curve;
                p_aux = split.auxiliary_basis.P;
                q_aux = split.auxiliary_basis.Q;
                pmq_aux = split.auxiliary_basis.PmQ;
                p_chl = split.challenge_basis.P;
                q_chl = split.challenge_basis.Q;
                pmq_chl = split.challenge_basis.PmQ;
            } else {
                // Lines 28–31: direct path
                let (ec, pc, qc, pc_pmq) = match i_com_narrow.to_isogeny(rng) {
                    Some(r) => r,
                    None => {
                        continue;
                    }
                };
                p_chl = pc;
                q_chl = qc;
                pmq_chl = pc_pmq;
                curve_aux = ec;
                p_aux = p_chl;
                q_aux = q_chl;
                pmq_aux = pmq_chl;
            }

            // Lines 34–35: even response.
            //
            // `compute_even_response` constructs the ideal
            // `I = O₀·conj(α) + O₀·(2^r_rsp)` internally. Two elements
            // that differ by a member of `2^r_rsp · O₀` generate the
            // same ideal, so `α_rsp` can be replaced with any element
            // congruent to it mod `2^r_rsp · O₀` before narrowing to
            // `Element<4>`.
            //
            // # Divergences
            //
            // The reduction MUST be taken in O₀'s integral basis, not
            // coordinate-wise in the standard `(1, i, j, k)` basis.
            // O₀ properly contains `Z⟨1, i, j, k⟩` (it has the
            // half-integral generators `(i + j)/2` and `(1 + k)/2`),
            // so reducing the raw `(1, i, j, k)` coordinates mod
            // `2^r_rsp` lands α in a different coset of `2^r_rsp · O₀`
            // and yields a different ideal `O₀·conj(α) + O₀·2^r_rsp`,
            // hence a different small generator (the kernel of
            // `α | E[2^r_rsp]` depends on the specific generator, not
            // just the ideal) and a different `j(E_chl)`. The C
            // reference (`sign.c` -> `quat_lideal_create`) builds the
            // ideal from the full-width α, which is congruent to the
            // O₀-basis reduction but not to the standard-basis one.
            // Decompose α in O₀'s basis, reduce those coordinates mod
            // `2^r_rsp`, and reconstruct `α' = Σ cᵢ·bᵢ` with
            // `α' ≡ α (mod 2^r_rsp · O₀)`.
            if r_rsp_val > 0 {
                let two_to_r: BigInt<N_RESP> = BigInt::<N_RESP>::ONE << r_rsp_val;
                let o0_lat: Lattice<N_RESP> = *o0_w.order().lattice();
                let o0_coords = match o0_lat.decompose(&alpha_rsp_w) {
                    Some(c) => c,
                    None => {
                        continue;
                    }
                };
                // Canonicalize each O₀-coordinate to `[0, 2^r_rsp)`.
                // `vt_mod` truncates toward zero, so a negative
                // coordinate gives a negative remainder — add the
                // modulus to land in the canonical range.
                let reduce_o0 = |c: &BigInt<N_RESP>| -> BigInt<N_RESP> {
                    let r = c.vt_mod(&two_to_r);
                    if bool::from(r.is_negative()) {
                        r.vt_add(&two_to_r)
                    } else {
                        r
                    }
                };
                let mut num = [BigInt::<N_RESP>::ZERO; 4];
                for (j, cj) in o0_coords.iter().enumerate() {
                    let cj_red = reduce_o0(cj);
                    let col = o0_lat.basis_elem(j);
                    num[0] = num[0].vt_add(&cj_red.vt_mul(col.a.as_bigint()));
                    num[1] = num[1].vt_add(&cj_red.vt_mul(col.b.as_bigint()));
                    num[2] = num[2].vt_add(&cj_red.vt_mul(col.c.as_bigint()));
                    num[3] = num[3].vt_add(&cj_red.vt_mul(col.d.as_bigint()));
                }
                let reduced_w = Element::<N_RESP>::new(
                    Coordinate::from_bigint(num[0]),
                    Coordinate::from_bigint(num[1]),
                    Coordinate::from_bigint(num[2]),
                    Coordinate::from_bigint(num[3]),
                    Denominator::from_bigint_unchecked(*o0_lat.denom()),
                );
                let alpha_narrow = match reduced_w.narrow_to::<4>() {
                    Some(a) => a,
                    None => {
                        continue;
                    }
                };
                let basis_pre = TorsionBasis::from_propagated(p_chl, pmq_chl, q_chl);
                let kernel = match response::even_response::EvenResponseKernel::from_quaternion(
                    &alpha_narrow,
                    basis_pre,
                    e_rsp_prime_te,
                    r_rsp,
                ) {
                    Some(k) => k,
                    None => continue,
                };
                let response = match kernel.isogeny() {
                    Some(r) => r,
                    None => continue,
                };
                p_chl = response.basis.P;
                q_chl = response.basis.Q;
                pmq_chl = response.basis.PmQ;
            }

            // Line 36: ComputeChallengeIsogeny
            let pre_iso_chl = TorsionBasis::from_propagated(p_chl, pmq_chl, q_chl);
            let (
                e_chl_final,
                TorsionBasis {
                    P: p_chl_final,
                    PmQ: pmq_chl_final,
                    Q: q_chl_final,
                },
            ) = match chl.to_isogeny(&basis_pk, &pre_iso_chl, n_bt_te) {
                Some(r) => r,
                None => continue,
            };
            // Line 37: SetChangeOfBasisMatrix (Algorithm 4.8, inlined).
            // TODO: refactor into ChallengeMatrix::from_response_endpoints()
            // that takes (E_aux, E_chl, P_aux, Q_aux, P_chl, Q_chl, e)
            // and returns (ChallengeMatrix, AuxiliaryHint, ChallengeHint).
            let (det_aux, hint_aux_raw) = match TorsionBasis::to_hint(&curve_aux) {
                Some(r) => r,
                None => continue,
            };
            let (det_chl, hint_chl_raw) = match TorsionBasis::to_hint(&e_chl_final) {
                Some(r) => r,
                None => continue,
            };

            // Matrix exponent: e_rsp' + r_rsp + 2 (HD extra torsion).
            // The dlog and matrix entries are at this exponent.
            let e_cob = TorsionExponent::try_from(e_rsp_prime + r_rsp_val + 2)
                .map_err(|_| SignatureError::SigningFailed)?;

            // C-ref-shuffled basis convention: `(R, S, RS) = (P, P−Q,
            // Q)` — matches what `TorsionBasis::from_hint` returns
            // and what verify expects when applying `M_chl`. Using
            // the naive `(P, Q, P−Q)` here produces an `M_chl` that
            // verify rejects (post-application bases collapse).
            let basis_aux = TorsionBasis::from_propagated(p_aux, pmq_aux, q_aux);
            // m1 = "coords of det_aux in basis basis_aux at 2^e_cob".
            // Compute via the inverse direction (from_bases_invert):
            // the cubical Tate's asymmetric ladder requires the
            // canonical / full-order side as the *first* arg, so we
            // pass `det_aux` (full order from `to_hint`) as canonical
            // and `basis_aux` (reduced order 2^e_cob) as reduced.
            // The forward call returns "coords of basis_aux in
            // det_aux"; we invert to get the direction `mul` consumes.
            let m1 = match ChangeOfBasisMatrix::from_bases_invert(&det_aux, &basis_aux, e_cob) {
                Some(m) => m,
                None => {
                    continue;
                }
            };

            // C-ref-shuffled basis convention: `(P, P−Q, Q)`. See
            // `basis_aux` above for the rationale.
            let basis_chl = TorsionBasis::from_propagated(p_chl_final, pmq_chl_final, q_chl_final);
            let transformed = m1.mul(&basis_chl);

            // m_chl = "coords of transformed in det_chl at 2^e_cob".
            // C ref's `change_of_basis_matrix_tate` (non-invert):
            // canonical = det_chl (full order), reduced = transformed
            // (order 2^e_cob from m1's matrix application).
            let m_chl = match ChangeOfBasisMatrix::from_bases(&det_chl, &transformed, e_cob) {
                Some(m) => m,
                None => {
                    continue;
                }
            };

            // Line 38: assemble signature.
            let hint_aux = AuxiliaryHint::from(hint_aux_raw.to_byte());
            let hint_chl = ChallengeHint::from(hint_chl_raw.to_byte());

            // Convert ChangeOfBasisMatrix → ChallengeMatrix for Signature.
            let sig_matrix = ChallengeMatrix::from(m_chl);

            return Ok(Signature {
                curve_aux,
                n_bt: n_bt_te,
                r_rsp,
                M_chl: sig_matrix,
                chl,
                hint_aux,
                hint_chl,
            });
        }

        Err(SignatureError::SigningFailed)
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

impl Debug for SigningKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
