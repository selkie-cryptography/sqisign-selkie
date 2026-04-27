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
use zeroize::ZeroizeOnDrop;

#[cfg(test)]
mod tests;

use crate::{
    curves::{
        AuxiliaryHint, BasisHint, ChallengeHint, ChangeOfBasisMatrix, TorsionBasis,
        TorsionExponent, VerifyingKeyHint,
        isogeny::{IsogenyDegree, Kernel},
        montgomery::{Curve, ProjectiveXOnlyPoint},
        scalar::Scalar,
    },
    deuring,
    keys::{
        Challenge, ChallengeMatrix, SIGNING_KEY_BYTES, Signature, SignatureError,
        VERIFYING_KEY_BYTES, verifying::VerifyingKey,
    },
    params::{D_MIX, E_RSP, FP_ENCODED_BYTES, TORSION_2POWER_BYTES, TORSION_EVEN_POWER},
    quaternions::{
        algebra::{Coordinate, Denominator, Element},
        bigint::BigInt,
        lattice::{HnfLattice, Lattice, LeftIdeal},
        precomputed::EXTREMAL_ORDERS,
    },
    surfaces,
};

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
    mat_sk: SecretKeyMatrix,
}

/// The secret change-of-basis matrix M_sk (part of the signing key).
///
/// No `PartialEq`/`Eq`/`ConstantTimeEq`: comparing secret key
/// material is a code smell.
#[derive(Clone)]
pub(crate) struct SecretKeyMatrix(ChangeOfBasisMatrix);

impl core::ops::Deref for SecretKeyMatrix {
    type Target = ChangeOfBasisMatrix;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl SecretKeyMatrix {
    /// Construct from raw 2×2 scalar entries. Always uses the full
    /// torsion exponent f = [`TORSION_EVEN_POWER`] since M_sk entries
    /// are mod 2^f.
    pub(crate) fn new(entries: [[Scalar; 2]; 2]) -> Self {
        Self(ChangeOfBasisMatrix {
            entries,
            e: TorsionExponent::FULL,
        })
    }

    /// Compute M_sk via the Tate pairing ([Algorithm 2.5][Alg. 2.5]).
    ///
    /// Used in key generation ([Algorithm 4.1][Alg. 4.1], line 9):
    /// `M_sk ← ChangeOfBasis_{2^f}(E_pk, (φ_sk(P₀), φ_sk(Q₀)), (P_pk, Q_pk))`.
    ///
    /// [Alg. 2.5]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.2.5
    /// [Alg. 4.1]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.1
    pub(crate) fn encode(full_basis: &TorsionBasis, target_basis: &TorsionBasis) -> Self {
        Self(ChangeOfBasisMatrix::from_bases(
            full_basis,
            target_basis,
            TorsionExponent::FULL,
        ))
    }
}

impl From<ChangeOfBasisMatrix> for SecretKeyMatrix {
    fn from(m: ChangeOfBasisMatrix) -> Self {
        Self(m)
    }
}

impl SigningKey {
    /// Construct a `SigningKey` from validated components.
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
        mat_sk: SecretKeyMatrix,
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

    /// Generate a new random signing key.
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
    /// [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
    /// [§2.2.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
    /// [§3.1.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.3.1
    /// [§3.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.3.2
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

    /// Derandomized keygen from a 48-byte seed. The seed is used to
    /// instantiate an AES256-CTR-DRBG (NIST SP 800-90A), which in
    /// turn drives every random sampling step in key generation.
    /// Passing the same seed yields the same key; useful for KATs
    /// and reproducible tests.
    pub fn generate_derand(
        randomness: &[u8; crate::drbg::SEEDLEN],
    ) -> Result<SigningKey, SignatureError> {
        let mut drbg = crate::drbg::Aes256CtrDrbg::new(randomness);
        Self::generate_with_rng(&mut drbg)
    }

    /// Key generation driven by a caller-owned RNG.
    ///
    /// Same algorithm as [`SigningKey::generate_derand`], but the
    /// caller supplies the RNG instead of this method instantiating
    /// its own AES-CTR-DRBG. Used by the keygen-then-sign cross-check
    /// path, which threads a single DRBG through both phases so its
    /// byte consumption pattern matches the SQIsign C reference.
    pub(crate) fn generate_with_rng<R: rand_core::CryptoRngCore>(
        rng: &mut R,
    ) -> Result<SigningKey, SignatureError> {
        // Bound the retry loop. Each iteration may fail in
        // reduce_to_prime_norm, narrow, or to_isogeny.
        for _ in 0..1000 {
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
            let (e_pk, phi_p, phi_q, phi_pmq) = match i_sk_narrow.to_isogeny() {
                Some(r) => r,
                None => continue,
            };

            // Line 8: (P_pk, Q_pk), hint_pk ← TorsionBasisToHint(E_pk).
            let (basis_pk, basis_hint) = TorsionBasis::to_hint(&e_pk);

            // Line 9: M_sk ← ChangeOfBasis_{2^f}(E_pk, (φ_sk(P₀), φ_sk(Q₀)), (P_pk, Q_pk)).
            let eval_basis = TorsionBasis::from_propagated(phi_p, phi_pmq, phi_q);
            let mat_sk = SecretKeyMatrix::encode(&eval_basis, &basis_pk);

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

        // Parse I_sk: norm (32 bytes unsigned, positive odd) + generator
        // coords (4 × 32 bytes signed).
        let norm_bytes: &[u8; 32] = bytes[pos..pos + FP_ENCODED_BYTES]
            .try_into()
            .map_err(|_| SignatureError::NonCanonical)?;
        let norm = IsogenyDegree::from_bytes_le(norm_bytes).ok_or(SignatureError::NonCanonical)?;
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
        let norm_bigint = norm.to_bigint();
        let ideal = LeftIdeal::new(&gen, &norm_bigint, EXTREMAL_ORDERS[0].order());

        // Parse M_sk: 4 × 32 bytes unsigned, row-major [[m00, m01], [m10, m11]].
        let mut entries = [[Scalar::ZERO; 2]; 2];
        for row in &mut entries {
            for entry in row.iter_mut() {
                let b = BigInt::<4>::from_bytes_le_unsigned(
                    bytes[pos..pos + TORSION_2POWER_BYTES].try_into().unwrap(),
                );
                *entry = Scalar::from(b);
                pos += TORSION_2POWER_BYTES;
            }
        }
        let mat_sk = SecretKeyMatrix::new(entries);
        debug_assert_eq!(pos, SIGNING_KEY_BYTES);

        Ok(Self::from_parts(verifying_key, ideal, gen, mat_sk))
    }

    /// Serialize this signing key to bytes.
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
        use crate::params::{FP_ENCODED_BYTES, TORSION_2POWER_BYTES};

        let mut out = [0u8; SIGNING_KEY_BYTES];
        let mut pos = 0;

        // pk (65 bytes).
        out[..VERIFYING_KEY_BYTES].copy_from_slice(&self.verifying_key.to_bytes());
        pos += VERIFYING_KEY_BYTES;

        // norm (32 bytes, unsigned LE).
        let norm = self.ideal.norm();
        for limb in norm.as_limbs() {
            out[pos..pos + 8].copy_from_slice(&limb.to_le_bytes());
            pos += 8;
        }

        // gen[0..3] (4 × 32 bytes, signed two's complement LE).
        let gen = &self.ideal_gen;
        let coords = [
            gen.a.as_bigint(),
            gen.b.as_bigint(),
            gen.c.as_bigint(),
            gen.d.as_bigint(),
        ];
        for coord in &coords {
            let is_neg = bool::from(coord.is_negative()) && !bool::from(coord.is_zero());
            // Write magnitude as LE bytes.
            for limb in coord.as_limbs() {
                out[pos..pos + 8].copy_from_slice(&limb.to_le_bytes());
                pos += 8;
            }
            if is_neg {
                // Two's complement: negate the 32-byte block.
                // Flip all bits, then add 1.
                let block = &mut out[pos - FP_ENCODED_BYTES..pos];
                for b in block.iter_mut() {
                    *b = !*b;
                }
                // Add 1 with carry.
                let mut carry = 1u16;
                for b in block.iter_mut() {
                    carry += *b as u16;
                    *b = carry as u8;
                    carry >>= 8;
                }
            }
        }

        // M_sk (4 × 32 bytes, unsigned LE, row-major).
        for row in &self.mat_sk.entries {
            for entry in row {
                out[pos..pos + TORSION_2POWER_BYTES].copy_from_slice(&entry.to_le_bytes());
                pos += TORSION_2POWER_BYTES;
            }
        }

        debug_assert_eq!(pos, SIGNING_KEY_BYTES);
        out
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
    /// 2. Translate to the commitment isogeny φ_com : E₀ → E_com via
    ///    `IdealToIsogeny` ([§3.2.3])
    ///
    /// **Challenge** (line 10):
    /// 3. Compute chl ← HASH(pk ‖ j(E_com) ‖ msg)
    ///
    /// **Response** (lines 11–38):
    /// 4. Convert chl to the challenge ideal I_chl via M_sk and
    ///    `KernelDecomposedToIdeal` ([§3.2.6])
    /// 5. Sample response quaternion α_rsp from the intersection lattice via
    ///    `RandomEquivalentQuaternion` ([§4.4.3])
    /// 6. Compute backtracking via `ComputeBacktrackingAndNormalize` ([§4.4.3])
    /// 7. Compute the response isogeny, split into odd and even parts, using
    ///    `SplitAuxiliaryIsogeny` ([§4.4.3]) or `IdealToIsogeny` depending on
    ///    e'_rsp
    /// 8. Compute the challenge isogeny via `ComputeChallengeIsogeny`
    ///    ([§4.4.2])
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
    /// Sign a message.
    ///
    /// Implements [SQIsign.Sign][Alg. 4.2] ([Algorithm 4.2][Alg. 4.2]).
    ///
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

    /// Derandomized sign from a 48-byte seed. The seed is used to
    /// instantiate an AES256-CTR-DRBG (NIST SP 800-90A), which in
    /// turn drives every random sampling step in the commitment
    /// and response phases. For a fixed `(self, msg, randomness)`
    /// triple the output is deterministic, so this is the entry
    /// point used for KATs and reproducible tests. Different
    /// signing keys (or different messages) yield different
    /// signatures even when seeded with the same 48 bytes.
    pub fn sign_derand(
        &self,
        msg: &[u8],
        randomness: &[u8; crate::drbg::SEEDLEN],
    ) -> Result<Signature, SignatureError> {
        let mut drbg = crate::drbg::Aes256CtrDrbg::new(randomness);
        self.sign_with_rng(msg, &mut drbg)
    }

    /// Signing driven by a caller-owned RNG.
    ///
    /// Same algorithm as [`SigningKey::sign_derand`], but the RNG is
    /// provided by the caller rather than instantiated from a 48-byte
    /// seed. Pair with [`SigningKey::generate_with_rng`] on the same
    /// DRBG instance to reproduce the SQIsign C reference's
    /// byte-consumption pattern (one DRBG seeded via
    /// `randombytes_init`, consumed by `crypto_sign_keypair` and
    /// then `crypto_sign` in order).
    pub(crate) fn sign_with_rng<R: rand_core::CryptoRngCore>(
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

        // Line 2: basis on E_pk
        let basis_pk = TorsionBasis::from_hint(
            e_pk,
            BasisHint::from_byte(u8::from(self.verifying_key.hint)),
        );

        // Line 3: while true do
        for _iter in 0..1000 {
            #[cfg(test)]
            let _iter_start = std::time::Instant::now();

            // --- Commitment (lines 4–9) ---

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
                    #[cfg(test)]
                    eprintln!("[sign {_iter}] DROP: random_prime_norm_wide None");
                    continue;
                }
            };

            // Lines 5–6: RandomEquivalentPrimeIdeal.
            if !i_com.reduce_to_prime_norm::<30, _>(rng) {
                #[cfg(test)]
                eprintln!("[sign {_iter}] DROP: reduce_to_prime_norm false");
                continue;
            }

            // Narrow to LeftIdeal<4> for to_isogeny.
            let i_com_narrow = match i_com.narrow() {
                Some(i) => i,
                None => {
                    #[cfg(test)]
                    eprintln!("[sign {_iter}] DROP: i_com.narrow None");
                    continue;
                }
            };

            // Line 7: E_com, P_com, Q_com ← IdealToIsogeny(I_com)
            //
            // The fourth return value is the propagated `PmQ_com`,
            // needed by `split_auxiliary_isogeny` so the response
            // (2,2)-chain's `lift_basis` sees a projective rep
            // consistent with the chain's evaluation history of
            // `P_com` and `Q_com`.
            #[cfg(test)]
            eprintln!("[sign {_iter}] commitment to_isogeny...");
            let (e_com, p_com, q_com, pmq_com) = match i_com_narrow.to_isogeny() {
                Some(r) => {
                    #[cfg(test)]
                    eprintln!("[sign {_iter}] commitment OK ({:?})", _iter_start.elapsed());
                    r
                }
                None => {
                    #[cfg(test)]
                    eprintln!("[sign {_iter}] DROP: commitment i_com_narrow.to_isogeny None");
                    continue;
                }
            };

            // --- Challenge (line 10) ---
            let chl = Challenge::derive(&self.verifying_key, &e_com, msg);

            // --- Response (lines 11–38) ---

            // Line 11: (c₁, c₂) ← M_sk · (1, chl)
            let chl_scalar: Scalar = chl.into();
            let m = &self.mat_sk.entries;
            let c1 = m[0][0].add_mod2k(&m[0][1].mul_mod2k(&chl_scalar, f), f);
            let c2 = m[1][0].add_mod2k(&m[1][1].mul_mod2k(&chl_scalar, f), f);

            // Line 12: I'_chl ← KernelDecomposedToIdeal(c₁, c₂)
            let c1_big = BigInt::<4>::from(c1);
            let c2_big = BigInt::<4>::from(c2);
            let i_chl_prime =
                match TorsionBasis::kernel_to_ideal(&c1_big, &c2_big, TorsionExponent::FULL) {
                    Some(ideal) => ideal,
                    None => {
                        #[cfg(test)]
                        eprintln!("[sign {_iter}] DROP: kernel_to_ideal None");
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
                    #[cfg(test)]
                    eprintln!("[sign {_iter}] DROP: intersection_via_kernel 1 None");
                    continue;
                }
            };
            #[cfg(test)]
            eprintln!(
                "[sign {_iter}] intersection 1: {:?} (cumul {:?})",
                _t_int.elapsed(),
                _iter_start.elapsed()
            );

            let i_com_conj = i_com_w.lattice().conjugate();
            let i_chl_sk_lat = Lattice::<N_RESP>::from(i_chl_sk);
            let i_com_conj_lat = Lattice::<N_RESP>::from(i_com_conj);

            #[cfg(test)]
            let _t_int2 = std::time::Instant::now();
            let intersection = match i_chl_sk_lat.intersection_via_kernel::<150>(&i_com_conj_lat) {
                Some(l) => l,
                None => {
                    #[cfg(test)]
                    eprintln!("[sign {_iter}] DROP: intersection_via_kernel 2 None");
                    continue;
                }
            };
            #[cfg(test)]
            eprintln!(
                "[sign {_iter}] intersection 2: {:?} (cumul {:?})",
                _t_int2.elapsed(),
                _iter_start.elapsed()
            );
            let intersection_lat = Lattice::<N_RESP>::from(intersection);

            #[cfg(test)]
            {
                let cols = intersection_lat.basis().columns();
                eprintln!(
                    "[sign {_iter}] intersection_lat: denom bits={}, col bits=[{},{},{},{}]/[{},{},{},{}]/[{},{},{},{}]/[{},{},{},{}]",
                    intersection_lat.denom().bitsize(),
                    cols[0][0].bitsize(),
                    cols[0][1].bitsize(),
                    cols[0][2].bitsize(),
                    cols[0][3].bitsize(),
                    cols[1][0].bitsize(),
                    cols[1][1].bitsize(),
                    cols[1][2].bitsize(),
                    cols[1][3].bitsize(),
                    cols[2][0].bitsize(),
                    cols[2][1].bitsize(),
                    cols[2][2].bitsize(),
                    cols[2][3].bitsize(),
                    cols[3][0].bitsize(),
                    cols[3][1].bitsize(),
                    cols[3][2].bitsize(),
                    cols[3][3].bitsize(),
                );
            }

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
            let o0_full = EXTREMAL_ORDERS[0].widen::<N_RESP>();
            let mut intersection_ideal = LeftIdeal::<N_RESP>::from_parts(
                HnfLattice::from(intersection_lat),
                BigInt::<N_RESP>::ZERO,
                *o0_full.order(),
            );
            if intersection_ideal.refresh_norm::<120>().is_none() {
                #[cfg(test)]
                eprintln!("[sign {_iter}] DROP: intersection_ideal.refresh_norm None");
                continue;
            }
            let lattice_content_r: BigInt<N_RESP> = *intersection_ideal.norm();
            let two_to_e_rsp: BigInt<N_RESP> = BigInt::<N_RESP>::ONE.shl(e_rsp);
            let two_e_rsp_minus_one = two_to_e_rsp.ct_sub(&BigInt::<N_RESP>::ONE);
            let radius = two_e_rsp_minus_one.ct_mul(&lattice_content_r);
            #[cfg(test)]
            eprintln!(
                "[sign {_iter}] radius: bits={}, lattice_content bits={}, e_rsp={}",
                radius.bitsize(),
                lattice_content_r.bitsize(),
                e_rsp,
            );

            // The intersection lattice has entries up to ~1920 bits
            // (BigInt<30>). The gram computation squares these:
            // ~3840 bits ≈ 60 limbs. Use W=64 for margin.
            #[cfg(test)]
            let _t_sample = std::time::Instant::now();
            let alpha_rsp_w = match intersection_lat.sample_from_ball::<64>(&radius) {
                Some(a) => {
                    #[cfg(test)]
                    eprintln!(
                        "[sign {_iter}] sample: {:?} (cumul {:?})",
                        _t_sample.elapsed(),
                        _iter_start.elapsed()
                    );
                    a
                }
                None => {
                    #[cfg(test)]
                    eprintln!("[sign {_iter}] DROP: sample_from_ball None");
                    continue;
                }
            };

            // Line 15: α_rsp, n_bt ← ComputeBacktrackingAndNormalize(α_rsp).
            // Keep `alpha_rsp_w` at `Element<N_RESP>` for the wide
            // degree-computation and ideal construction below.
            //
            // Earlier attempts also primitivized α's odd integer
            // content here (matching the C ref's
            // `quat_alg_make_primitive`). That was wrong: the spec
            // formula `d_rsp = nrd(α) / (D²_mix · 2^{f-n_bt})` holds
            // only for the *un-primitivized* α. Dividing α by an odd
            // `g` shrinks `nrd(α)` by `g²`; if `g` shares a factor
            // with `D_mix` (513-bit prime, so `g ≥ D_mix` occurs for
            // α coordinates with that magnitude), the division is no
            // longer exact. `refresh_norm` below derives the true
            // `n(I)` from the lattice covolume, so we do not need
            // primitivization to align the stored norm with the
            // actual ideal.
            let (alpha_rsp_w, n_bt) = alpha_rsp_w.compute_backtracking();
            let (nrd_num_w, nrd_den_w) = alpha_rsp_w.norm_w::<N_RESP>();

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
            let lattice_content: BigInt<N_RESP> = lattice_content_r;

            let d_rsp_wide = {
                let (q1, r1) = nrd_num_w.div_rem(&nrd_den_w);
                if !bool::from(r1.is_zero()) {
                    #[cfg(test)]
                    eprintln!("[sign {_iter}] DROP: nrd not exact by denom²");
                    continue;
                }
                let (q2, r2) = q1.div_rem(&lattice_content);
                if !bool::from(r2.is_zero()) {
                    #[cfg(test)]
                    {
                        let g = q1.gcd(&lattice_content);
                        eprintln!(
                            "[sign {_iter}] DROP: nrd not divisible by N(I_com)·N(I_chl_sec): \
                             q1 bits={}, lattice_content bits={}, rem bits={}, gcd(q1, lc) bits={}",
                            q1.bitsize(),
                            lattice_content.bitsize(),
                            r2.bitsize(),
                            g.bitsize(),
                        );
                    }
                    continue;
                }
                q2
            };
            let r_rsp_val = d_rsp_wide.trailing_zeros();
            let d_rsp_shifted = d_rsp_wide.shr(r_rsp_val);
            // q_rsp = d_rsp / 2^r_rsp (odd part). For NIST-I the
            // response-degree odd part is bounded by `D_rsp ≈ 2^126`,
            // so it fits in `BigInt<4>` (256 bits) with room to
            // spare. Narrow from the wide working width.
            let q_rsp: BigInt<4> = match d_rsp_shifted.narrow_to::<4>() {
                Some(q) => q,
                None => {
                    #[cfg(test)]
                    eprintln!("[sign {_iter}] DROP: q_rsp narrow_to::<4>() None");
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
            let i_com_rsp_norm_w = i_com_norm_w.ct_mul(&q_rsp_wide);
            let mut i_com_rsp_w = match LeftIdeal::<30>::from_generator_mod_hnf(
                &alpha_rsp_conj,
                &i_com_rsp_norm_w,
                o0_w.order(),
            ) {
                Some(i) => i,
                None => {
                    #[cfg(test)]
                    eprintln!("[sign {_iter}] DROP: from_generator_mod_hnf None");
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
                #[cfg(test)]
                eprintln!("[sign {_iter}] DROP: refresh_norm None");
                continue;
            }
            // The response ideal has norm ~2^257 which may exceed
            // BigInt<4>. Always reduce via `smallest_equiv_narrow`
            // to ensure the norm is small enough that downstream
            // operations (e.g. `i_com_rsp.norm() · i_aux.norm()`
            // in the intersection below) don't silently truncate
            // at BigInt<4>. A bare `narrow()` path keeps norms up
            // to 2^256, and multiplying by aux_norm ~2^126 gives
            // ~2^382 which wraps mod 2^256 and corrupts the
            // intersection ideal's stored norm. The smallest-
            // equiv reduction brings the norm down to ~√p ≈ 2^126,
            // so the product fits in BigInt<4>.
            let i_com_rsp = match i_com_rsp_w.smallest_equiv_narrow::<120>() {
                Some(i) => i,
                None => {
                    #[cfg(test)]
                    eprintln!("[sign {_iter}] DROP: smallest_equiv_narrow::<120> None");
                    continue;
                }
            };

            // Lines 21–33: compute response isogeny
            let (mut e_chl, mut p_chl, mut q_chl);
            let curve_aux;
            let p_aux;
            let q_aux;

            if e_rsp_prime > 0 {
                // Lines 22–27: auxiliary isogeny path.
                //
                // After reduce_to_prime_norm, I_com_rsp has a small
                // prime norm (~2^15). The intersection with I_aux
                // (norm ~2^126) produces a ~141-bit norm ideal,
                // within FixedDegreeIsogeny's bound (< 2^246).
                let aux_norm = BigInt::<4>::ONE.shl(e_rsp_prime).ct_sub(&q_rsp);
                #[cfg(test)]
                eprintln!(
                    "[sign {_iter}] aux_norm step starting, aux_norm bits={} (cumul {:?})",
                    aux_norm.bitsize(),
                    _iter_start.elapsed()
                );
                let i_aux = match LeftIdeal::<4>::random_norm(&aux_norm, &EXTREMAL_ORDERS[0]) {
                    Some(i) => i,
                    None => {
                        #[cfg(test)]
                        eprintln!("[sign {_iter}] DROP: i_aux random_norm None");
                        continue;
                    }
                };
                #[cfg(test)]
                eprintln!(
                    "[sign {_iter}] i_aux done, norm={} bits (cumul {:?})",
                    i_aux.norm().bitsize(),
                    _iter_start.elapsed()
                );

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
                let i_com_rsp_w: LeftIdeal<8> = i_com_rsp.widen::<8>();
                let i_aux_w: LeftIdeal<8> = i_aux.widen::<8>();
                let i_com_rsp_lat_w: Lattice<8> = (*i_com_rsp_w.lattice()).into();
                let i_aux_lat_w: Lattice<8> = (*i_aux_w.lattice()).into();
                #[cfg(test)]
                let _t_inter = std::time::Instant::now();
                let inter_hnf_w8 =
                    match i_com_rsp_lat_w.intersection_via_kernel::<150>(&i_aux_lat_w) {
                        Some(h) => h,
                        None => {
                            #[cfg(test)]
                            eprintln!("[sign {_iter}] DROP: i_inter intersection_via_kernel None");
                            continue;
                        }
                    };
                #[cfg(test)]
                eprintln!(
                    "[sign {_iter}] i_inter intersection_via_kernel: {:?} (cumul {:?})",
                    _t_inter.elapsed(),
                    _iter_start.elapsed()
                );
                let inter_norm_w8: BigInt<8> = i_com_rsp
                    .norm()
                    .widen::<8>()
                    .ct_mul(&i_aux.norm().widen::<8>());
                let o0_w8 = EXTREMAL_ORDERS[0].widen::<8>();
                let mut i_inter_w =
                    LeftIdeal::<8>::from_parts(inter_hnf_w8, inter_norm_w8, *o0_w8.order());
                #[cfg(test)]
                let _t_refresh = std::time::Instant::now();
                if i_inter_w.refresh_norm::<40>().is_none() {
                    #[cfg(test)]
                    eprintln!("[sign {_iter}] DROP: i_inter.refresh_norm None");
                    continue;
                }
                #[cfg(test)]
                eprintln!(
                    "[sign {_iter}] i_inter refresh_norm OK: {:?}, norm={} bits (cumul {:?})",
                    _t_refresh.elapsed(),
                    i_inter_w.norm().bitsize(),
                    _iter_start.elapsed()
                );
                if *i_inter_w.norm() == BigInt::<8>::ONE {
                    #[cfg(test)]
                    eprintln!("[sign {_iter}] DROP: i_inter collapsed to O_0");
                    continue;
                }
                #[cfg(test)]
                let _t_smeq = std::time::Instant::now();
                let i_inter = match i_inter_w.smallest_equiv_narrow::<32>() {
                    Some(i) => i,
                    None => {
                        #[cfg(test)]
                        eprintln!("[sign {_iter}] DROP: i_inter.smallest_equiv_narrow None");
                        continue;
                    }
                };
                #[cfg(test)]
                eprintln!(
                    "[sign {_iter}] i_inter smallest_equiv_narrow OK: {:?}, norm={} bits (cumul {:?})",
                    _t_smeq.elapsed(),
                    i_inter.norm().bitsize(),
                    _iter_start.elapsed()
                );
                #[cfg(test)]
                eprintln!(
                    "[sign {_iter}] response to_isogeny... (cumul {:?})",
                    _iter_start.elapsed()
                );
                let (e_aux_prime, p_aux_prime, q_aux_prime, pmq_aux_prime) =
                    match i_inter.to_isogeny() {
                        Some(r) => {
                            #[cfg(test)]
                            eprintln!(
                                "[sign {_iter}] response to_isogeny OK (cumul {:?})",
                                _iter_start.elapsed()
                            );
                            r
                        }
                        None => {
                            #[cfg(test)]
                            eprintln!("[sign {_iter}] DROP: i_inter.to_isogeny() None");
                            continue;
                        }
                    };

                let split = match split_auxiliary_isogeny(
                    &e_com,
                    &e_aux_prime,
                    &p_com,
                    &q_com,
                    &pmq_com,
                    &p_aux_prime,
                    &q_aux_prime,
                    &pmq_aux_prime,
                    q_rsp,
                    e_rsp_prime_te,
                    r_rsp,
                ) {
                    Some(r) => r,
                    None => {
                        #[cfg(test)]
                        eprintln!("[sign {_iter}] DROP: split_auxiliary_isogeny None");
                        continue;
                    }
                };
                curve_aux = split.0;
                p_aux = split.1;
                q_aux = split.2;
                e_chl = split.3;
                p_chl = split.4;
                q_chl = split.5;
            } else {
                // Lines 28–31: direct path
                let (ec, pc, qc, _pmq_chl) = match i_com_narrow.to_isogeny() {
                    Some(r) => r,
                    None => {
                        #[cfg(test)]
                        eprintln!(
                            "[sign {_iter}] DROP: direct-path i_com_narrow.to_isogeny() None"
                        );
                        continue;
                    }
                };
                e_chl = ec;
                p_chl = pc;
                q_chl = qc;
                curve_aux = e_chl;
                p_aux = p_chl;
                q_aux = q_chl;
            }

            // Lines 34–35: even response.
            //
            // `compute_even_response` constructs the ideal
            // `I = O₀·α + O₀·(2^r_rsp)` internally. Two elements that
            // differ by a member of `2^r_rsp · O₀` generate the same
            // ideal, so we can replace `α_rsp` with
            // `α_rsp mod (2^r_rsp · O₀)` — coordinate-wise reduction
            // mod `2^r_rsp` — and then narrow to `Element<4>`.
            // For `r_rsp ≤ 126`, the reduced coordinates fit easily.
            if r_rsp_val > 0 {
                let two_to_r: BigInt<N_RESP> = BigInt::<N_RESP>::ONE.shl(r_rsp_val);
                let mod_coord = |c: &BigInt<N_RESP>| c.ct_mod(&two_to_r);
                let reduced_w = Element::<N_RESP>::new(
                    Coordinate::from_bigint(mod_coord(alpha_rsp_w.a.as_bigint())),
                    Coordinate::from_bigint(mod_coord(alpha_rsp_w.b.as_bigint())),
                    Coordinate::from_bigint(mod_coord(alpha_rsp_w.c.as_bigint())),
                    Coordinate::from_bigint(mod_coord(alpha_rsp_w.d.as_bigint())),
                    alpha_rsp_w.denom,
                );
                let alpha_narrow = match reduced_w.narrow_to::<4>() {
                    Some(a) => a,
                    None => {
                        #[cfg(test)]
                        eprintln!("[sign {_iter}] DROP: reduced_w.narrow_to::<4>() None");
                        continue;
                    }
                };
                let (ec, pc, qc) = match deuring::compute_even_response(
                    &e_chl,
                    &p_chl,
                    &q_chl,
                    &alpha_narrow,
                    e_rsp_prime_te,
                    r_rsp,
                ) {
                    Some(r) => r,
                    None => {
                        #[cfg(test)]
                        eprintln!("[sign {_iter}] DROP: compute_even_response None");
                        continue;
                    }
                };
                e_chl = ec;
                p_chl = pc;
                q_chl = qc;
            }

            // Line 36: ComputeChallengeIsogeny
            let (e_chl_final, p_chl_final, q_chl_final) =
                match compute_challenge_isogeny(&basis_pk, &chl, &e_chl, &p_chl, &q_chl, n_bt_te) {
                    Some(r) => r,
                    None => {
                        #[cfg(test)]
                        eprintln!("[sign {_iter}] DROP: compute_challenge_isogeny None");
                        continue;
                    }
                };

            // Line 37: SetChangeOfBasisMatrix (Algorithm 4.8, inlined).
            // TODO: refactor into ChallengeMatrix::from_response_endpoints()
            // that takes (E_aux, E_chl, P_aux, Q_aux, P_chl, Q_chl, e)
            // and returns (ChallengeMatrix, AuxiliaryHint, ChallengeHint).
            let (det_aux, hint_aux_raw) = TorsionBasis::to_hint(&curve_aux);
            let (det_chl, hint_chl_raw) = TorsionBasis::to_hint(&e_chl_final);

            let e_cob = TorsionExponent::try_from(e_rsp_prime + r_rsp_val)
                .map_err(|_| SignatureError::SigningFailed)?;
            let scale = f - e_cob.value() - 2;
            let scale_scalar = Scalar::from_limbs(*BigInt::<4>::ONE.shl(scale).as_limbs());

            let det_aux_scaled = TorsionBasis::from_propagated(
                &scale_scalar * &det_aux.R,
                &scale_scalar * &det_aux.S,
                &scale_scalar * &det_aux.RS,
            );
            let det_chl_scaled = TorsionBasis::from_propagated(
                &scale_scalar * &det_chl.R,
                &scale_scalar * &det_chl.S,
                &scale_scalar * &det_chl.RS,
            );

            let basis_aux = TorsionBasis::from((p_aux, q_aux));
            let m1 = ChangeOfBasisMatrix::from_bases(&basis_aux, &det_aux_scaled, e_cob);

            let basis_chl = TorsionBasis::from((p_chl_final, q_chl_final));
            let transformed = m1.mul(&basis_chl);
            let m_chl = ChangeOfBasisMatrix::from_bases(&det_chl_scaled, &transformed, e_cob);

            // Line 38: assemble signature.
            let hint_aux = AuxiliaryHint::from(hint_aux_raw.to_byte());
            let hint_chl = ChallengeHint::from(hint_chl_raw.to_byte());

            // Convert ChangeOfBasisMatrix → ChallengeMatrix for Signature.
            let sig_matrix = ChallengeMatrix::from(m_chl);

            #[cfg(test)]
            eprintln!("[sign {_iter}] SUCCESS (cumul {:?})", _iter_start.elapsed());
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

// ---------------------------------------------------------------------------
// Challenge isogeny (Algorithm 4.7)
// ---------------------------------------------------------------------------

/// Compute the challenge isogeny and map points through the isomorphism.
///
/// Given a basis (P, Q) on curve E and a challenge `chl`, computes the
/// isogeny with kernel ⟨[2^n](P + [chl]Q)⟩ of degree 2^(f−n), producing
/// the challenge curve E''. Then maps P', Q' from E' (which has the
/// same j-invariant as E'') onto E'' via
/// [`Isomorphism`](crate::curves::montgomery::Isomorphism).
///
/// Implements [ComputeChallengeIsogeny][Alg. 4.7] ([Algorithm 4.7][Alg. 4.7]).
///
/// [Alg. 4.7]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.7
pub(crate) fn compute_challenge_isogeny(
    basis: &TorsionBasis,
    chl: &Challenge,
    e_prime: &Curve,
    p_prime: &ProjectiveXOnlyPoint,
    q_prime: &ProjectiveXOnlyPoint,
    n_bt: TorsionExponent,
) -> Option<(Curve, ProjectiveXOnlyPoint, ProjectiveXOnlyPoint)> {
    // Line 1: E'' ← TwoIsogenyChain([2^n](P + [ch]Q), E, f-n)
    let mut kernel_point = basis.scalar_mul_add(chl.as_ref());
    for _ in 0..n_bt.value() {
        kernel_point = kernel_point.double();
    }
    let e_chain = TorsionExponent::try_from(TORSION_EVEN_POWER - n_bt.value()).ok()?;
    let (curve_chl, _) = Kernel::new(kernel_point).isogeny(e_chain, &[]);

    // Line 2: P'', Q'' ← IsomorphismMontgomeryCurves(E', P', Q', E'')
    let iso = e_prime.isomorphism(&curve_chl)?;
    let p_chl = iso.eval(p_prime);
    let q_chl = iso.eval(q_prime);

    // Line 3
    Some((curve_chl, p_chl, q_chl))
}

// ---------------------------------------------------------------------------
// Split auxiliary isogeny (Algorithm 4.5)
// ---------------------------------------------------------------------------

/// Compute the split auxiliary isogeny via a (2,2)-isogeny chain.
///
/// Takes the commitment curve E₁ (= E_com) and auxiliary curve E₂
/// (= E'_aux) with torsion points, and computes the (2,2)-isogeny
/// chain that splits the response isogeny into odd and even parts.
///
/// `pmq1` and `pmq2` are the propagated `P − Q` projective reps on
/// each curve. They MUST come from the same chain that produced
/// `(p1, q1)` and `(p2, q2)` — typically [`LeftIdeal::to_isogeny`]'s
/// fourth return value. Recomputing them via `projective_difference`
/// at this site picks a sqrt branch that is not aligned with the
/// chain's evaluation history, and the resulting kernel produces a
/// terminal theta null with `count_splitting_indices = 0`.
///
/// Returns `(E_aux, P_aux, Q_aux, E_chl, P_chl, Q_chl)`.
///
/// Implements [SplitAuxiliaryIsogeny][Alg. 4.5] ([Algorithm 4.5][Alg. 4.5]).
///
/// [Alg. 4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.5
/// [`LeftIdeal::to_isogeny`]: crate::quaternions::lattice::LeftIdeal::to_isogeny
#[allow(clippy::too_many_arguments)]
pub(crate) fn split_auxiliary_isogeny(
    e1: &Curve,
    e2: &Curve,
    p1: &ProjectiveXOnlyPoint,
    q1: &ProjectiveXOnlyPoint,
    pmq1: &ProjectiveXOnlyPoint,
    p2: &ProjectiveXOnlyPoint,
    q2: &ProjectiveXOnlyPoint,
    pmq2: &ProjectiveXOnlyPoint,
    q_rsp: BigInt<4>,
    e_prime: TorsionExponent,
    r_rsp: TorsionExponent,
) -> Option<(
    Curve,
    ProjectiveXOnlyPoint,
    ProjectiveXOnlyPoint,
    Curve,
    ProjectiveXOnlyPoint,
    ProjectiveXOnlyPoint,
)> {
    let f = TORSION_EVEN_POWER;
    let e_prime_val = e_prime.value();
    let r_val = r_rsp.value();

    // Kernel construction follows the C reference's
    // `compute_dim2_isogeny_challenge` (sign.c:578-590, 240-256).
    // The C ref reduces the bases on both curves to order
    // `2^(reduced_order) = 2^(e_prime + 2 + r)` BEFORE forming
    // the kernel, then forms `(P1_red, q_inv · P2_red)` and doubles
    // by `r`. After the `r`-doubling the kernel has order
    // `2^(e_prime + 2)`, exactly what the (2,2)-chain of length
    // `e_prime` requires (the chain's gluing step needs 8-torsion
    // at the bottom of the strategy, and the chain length `e_prime`
    // implies a kernel of order `2^(e_prime + 2)`).
    //
    // Earlier attempts that fed raw `2^f`-torsion bases into the
    // kernel left `2^(f - r - e_prime - 2)` extra torsion above
    // the chain's expected order. Our `Kernel::isogeny` always
    // computes exactly `e` doubling-down steps before gluing, so
    // any extra torsion shifts the strategy bottom away from the
    // gluing's required 8-torsion level — `count_splitting_indices`
    // returns 0 and the chain returns `None`.
    //
    // The kernel `PmQ` projective reps come from `pmq1`/`pmq2`
    // inputs, scaled alongside `P` and `Q` to keep the projective
    // history aligned. Recomputing via `projective_difference` on
    // post-scaling kernel points picks a sqrt branch that the
    // chain's `lift_basis` then rejects, again surfacing as
    // `count_splitting_indices = 0`.
    //
    // # Divergences
    //
    // The spec (Algorithm 4.5 lines 1–4) describes a kernel with
    // torsion `2^(e' + 2)` directly (via `[2^(f-e'-2)]` reductions
    // on each side), but does not spell out (a) the basis-reduction
    // step that the C reference performs before invoking the chain
    // or (b) that downstream consumers of `IdealToIsogeny`'s output
    // require a `PmQ` whose projective rep is propagated alongside
    // `P` and `Q`, not recomputed via `projective_difference`. Both
    // are required for interoperability with the C reference's KAT
    // vectors.

    // C ref: reduced_order = pow_dim2_deg_resp + HD_extra_torsion +
    //                        sig->two_resp_length
    //                      = e_prime + 2 + r
    let reduced_order = match e_prime_val
        .checked_add(2)
        .and_then(|x| x.checked_add(r_val))
    {
        Some(o) if o <= f => o,
        _ => {
            #[cfg(test)]
            eprintln!(
                "[split_aux] reduced_order out of range: e_prime={e_prime_val}, r={r_val}, f={f}"
            );
            return None;
        }
    };
    let reduce_steps = f - reduced_order;

    // Reduce all six basis points (P, Q, PmQ on each curve) from
    // order `2^f` to order `2^reduced_order = 2^(e_prime + 2 + r)`.
    // Matches the C reference's [`ec_dbl_iter_basis`][c-ref] which
    // doubles all of `(B.P, B.Q, B.PmQ)` together to keep the
    // projective rep history consistent.
    //
    // [c-ref]: https://github.com/SQIsign/the-sqisign/blob/91e9e464fe5400192d13e1f9240cbf180200a103/src/signature/ref/lvlx/sign.c#L579-L580
    let mut p1_red = *p1;
    let mut q1_red = *q1;
    let mut pmq1_red = *pmq1;
    let mut p2_red = *p2;
    let mut q2_red = *q2;
    let mut pmq2_red = *pmq2;
    for _ in 0..reduce_steps {
        p1_red = p1_red.double();
        q1_red = q1_red.double();
        pmq1_red = pmq1_red.double();
        p2_red = p2_red.double();
        q2_red = q2_red.double();
        pmq2_red = pmq2_red.double();
    }

    // q_inv ← q^{-1} (mod 2^reduced_order). C ref uses
    // `degree_resp_inv = degree_odd_resp^{-1} mod 2^(reduced_order)`
    // (computed in compute_random_aux_norm_and_helpers).
    let q_scalar = Scalar::from_limbs(*q_rsp.as_limbs());
    let q_inv = match q_scalar.inv_mod2k(reduced_order) {
        Some(v) => v,
        None => {
            #[cfg(test)]
            eprintln!(
                "[split_aux] q.inv_mod2k None: q_rsp parity={:?}, reduced_order={reduced_order}",
                if bool::from(q_rsp.is_even()) {
                    "even"
                } else {
                    "odd"
                }
            );
            return None;
        }
    };

    // Kernel:
    //   T1 = (P1_red,         q_inv · P2_red)
    //   T2 = (Q1_red,         q_inv · Q2_red)
    //   T1m2 = (P1mQ1_red,    q_inv · P2mQ2_red)   [propagated PmQ]
    // Then double all three pairs by `r`. After the `r`-doubling
    // the kernel has order exactly `2^(e_prime + 2)`. The PmQ
    // points are scaled by the same `q_inv · 2^r` as P and Q, so
    // the projective rep stays aligned with the chain's evaluator.
    let p2_qinv = &q_inv * &p2_red;
    let q2_qinv = &q_inv * &q2_red;
    let pmq2_qinv = &q_inv * &pmq2_red;

    let two_r_scalar = Scalar::from_limbs(*BigInt::<4>::ONE.shl(r_val).as_limbs());
    let p1_ker = &two_r_scalar * &p1_red;
    let q1_ker = &two_r_scalar * &q1_red;
    let pmq1_ker = &two_r_scalar * &pmq1_red;
    let p2_ker = &two_r_scalar * &p2_qinv;
    let q2_ker = &two_r_scalar * &q2_qinv;
    let pmq2_ker = &two_r_scalar * &pmq2_qinv;

    // (2,2)-isogeny chain on E_com × E_aux.
    let product = surfaces::EllipticProduct::new(*e1, *e2);
    let kernel = match surfaces::Kernel::from_montgomery(
        product,
        (p1_ker, p2_ker),
        (q1_ker, q2_ker),
        (pmq1_ker, pmq2_ker),
    ) {
        Some(k) => k,
        None => {
            #[cfg(test)]
            eprintln!("[split_aux] Kernel::from_montgomery None");
            return None;
        }
    };

    // Pushed points: the REDUCED bases (order `2^(e_prime + 2 + r)`)
    // with zero on E_aux, matching sign.c:262-269. After the chain
    // these become the canonical bases on the codomain components.
    let zero_e2 = ProjectiveXOnlyPoint::identity(e2);
    let e_chain = e_prime;
    let (codomain, images) = match kernel.isogeny(e_chain, &[(p1_red, zero_e2), (q1_red, zero_e2)])
    {
        Some(r) => r,
        None => {
            #[cfg(test)]
            eprintln!(
                "[split_aux] kernel.isogeny None: e_chain={}, reduced_order={reduced_order}",
                e_chain.value()
            );
            return None;
        }
    };

    // Line 6: return F₁, S₁, R₁, F₂, S₂, R₂
    // The codomain is F₁ × F₂; images are (S₁,S₂) and (R₁,R₂).
    let curve_aux = codomain.E1;
    let curve_chl = codomain.E2;
    let p_aux = images[0].0;
    let q_aux = images[1].0;
    let p_chl = images[0].1;
    let q_chl = images[1].1;

    Some((curve_aux, p_aux, q_aux, curve_chl, p_chl, q_chl))
}
