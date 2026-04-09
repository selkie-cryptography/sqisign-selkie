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
        isogeny::Kernel,
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
        lattice::{Lattice, LeftIdeal},
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
        let rng = &mut drbg;
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
            let (e_pk, phi_p, phi_q) = match i_sk_narrow.to_isogeny() {
                Some(r) => r,
                None => continue,
            };

            // Line 8: (P_pk, Q_pk), hint_pk ← TorsionBasisToHint(E_pk).
            let (basis_pk, basis_hint) = TorsionBasis::to_hint(&e_pk);

            // Line 9: M_sk ← ChangeOfBasis_{2^f}(E_pk, (φ_sk(P₀), φ_sk(Q₀)), (P_pk, Q_pk)).
            let phi_pmq = phi_p.projective_difference(&phi_q);
            let eval_basis = TorsionBasis::new(phi_p, phi_pmq, phi_q);
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

            return Ok(SigningKey {
                verifying_key,
                ideal: i_sk_narrow,
                mat_sk,
            });
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

        // Parse I_sk: norm (32 bytes unsigned) + generator coords (4 × 32 bytes
        // signed).
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
        let rng = &mut drbg;
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
            eprintln!("sign iter {_iter}");
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
                    eprintln!("  skip: rand");
                    continue;
                }
            };
            eprintln!("  rand ok");

            // Lines 5–6: RandomEquivalentPrimeIdeal.
            if !i_com.reduce_to_prime_norm::<30, _>(rng) {
                eprintln!("  skip: reduce");
                continue;
            }
            eprintln!(
                "  reduce ok, norm_limbs[0..4]={:x?}",
                &i_com.norm().as_limbs()[..4]
            );

            // Narrow to LeftIdeal<4> for to_isogeny.
            let i_com_narrow = match i_com.narrow() {
                Some(i) => i,
                None => {
                    eprintln!("  skip: narrow");
                    continue;
                }
            };
            eprintln!("  narrow ok");

            // Line 7: E_com, P_com, Q_com ← IdealToIsogeny(I_com)
            let (e_com, p_com, q_com) = match i_com_narrow.to_isogeny() {
                Some(r) => r,
                None => {
                    eprintln!("  skip: to_isogeny");
                    continue;
                }
            };
            eprintln!("  to_isogeny ok");

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
                    None => continue,
                };

            // Line 13: I_chl ← [I_sk]_* I'_chl
            let o_r_sk = self.ideal.right_order();
            let i_chl = self.ideal.pushforward(&i_chl_prime, &o_r_sk);

            // Line 14: α_rsp ← RandomEquivalentQuaternion(I_com ∩ I_sk · I_chl)
            //
            // The spec (Algorithm 4.3) uses a sampling radius of
            //   B = D_rsp · D²_mix · 2^{f+1} ≈ 2^1399   (NIST-I).
            //
            // We widen the three ideals to a common `LeftIdeal<22>`
            // (1408 bits) to accommodate the radius, compute the
            // intersection `I_com ∩ (I_sk · I_chl)` at that width, and
            // sample with intermediate width W=44.
            const N_RESP: usize = 30;
            let i_sk_w = self.ideal.widen::<N_RESP>();
            let i_chl_w = i_chl.widen::<N_RESP>();
            let i_com_w = i_com.widen::<N_RESP>();

            let i_sk_i_chl = i_sk_w.lattice().product(i_chl_w.lattice());
            let intersection = i_com_w.lattice().intersection(&i_sk_i_chl);
            let intersection_lat = Lattice::<N_RESP>::from(intersection);

            // Radius: D_rsp · D²_mix · 2^{f+1}, computed at BigInt<22>.
            // D_rsp = 2^e_rsp.
            let d_mix_22 = D_MIX.widen::<N_RESP>();
            let d_mix_sq = d_mix_22.ct_mul(&d_mix_22);
            let radius = d_mix_sq.shl(e_rsp + f + 1);

            let alpha_rsp_w = match intersection_lat.sample_from_ball::<44>(&radius) {
                Some(a) => a,
                None => continue,
            };

            // Line 15: α_rsp, n_bt ← ComputeBacktrackingAndNormalize(α_rsp).
            // Keep `alpha_rsp_w` at `Element<N_RESP>` for the wide
            // degree-computation and ideal construction below.
            let (alpha_rsp_w, n_bt) = alpha_rsp_w.compute_backtracking();
            let (nrd_num_w, nrd_den_w) = alpha_rsp_w.norm_w::<N_RESP>();

            // Lines 16–20: degree computations.
            //
            // d_rsp = nrd(α_rsp) / (D²_mix · 2^{f-n_bt})
            // r_rsp = DyadicValuation(d_rsp)
            // q_rsp = d_rsp / 2^r_rsp
            // e'_rsp = e_rsp - r_rsp - n_bt
            //
            // `nrd_num_w / nrd_den_w` is the norm at `BigInt<N_RESP>`,
            // computed above before narrowing α. The division by
            // `D²_mix · 2^{f-n_bt}` must also be done at the wider
            // width because `D²_mix ~ 2^1024` exceeds `BigInt<4>`.
            let d_rsp_wide = {
                let (q1, _) = nrd_num_w.div_rem(&nrd_den_w);
                let q2 = q1.shr(f - n_bt);
                let (q3, _) = q2.div_rem(&d_mix_sq);
                q3
            };
            let r_rsp_val = d_rsp_wide.trailing_zeros();
            let d_rsp_shifted = d_rsp_wide.shr(r_rsp_val);
            // q_rsp = d_rsp / 2^r_rsp (odd part). For NIST-I the
            // response-degree odd part is bounded by `D_rsp ≈ 2^126`,
            // so it fits in `BigInt<4>` (256 bits) with room to
            // spare. Narrow from the wide working width.
            let q_rsp: BigInt<4> = match d_rsp_shifted.narrow_to::<4>() {
                Some(q) => q,
                None => continue,
            };
            let e_rsp_prime = e_rsp - r_rsp_val - n_bt;

            let n_bt_te =
                TorsionExponent::try_from(n_bt).map_err(|_| SignatureError::SigningFailed)?;
            let r_rsp =
                TorsionExponent::try_from(r_rsp_val).map_err(|_| SignatureError::SigningFailed)?;
            let e_rsp_prime_te = TorsionExponent::try_from(e_rsp_prime)
                .map_err(|_| SignatureError::SigningFailed)?;

            // Line 19: I_com,rsp = O₀·α_rsp + O₀·(q_rsp · D_mix).
            //
            // Built at the wide width `LeftIdeal<N_RESP>` via
            // [`LeftIdeal::from_generator`]. The norm `q_rsp · D_mix`
            // has ~513 + log2(q_rsp) bits and does not fit in
            // `BigInt<4>`, but does fit in `BigInt<N_RESP>`.
            // After construction we reduce to an equivalent
            // prime-norm ideal and narrow to `LeftIdeal<4>` for the
            // downstream `to_isogeny` call.
            let o0_w = EXTREMAL_ORDERS[0].widen::<N_RESP>();
            let q_rsp_wide: BigInt<N_RESP> = q_rsp.widen();
            let i_com_rsp_norm_w = q_rsp_wide.ct_mul(&d_mix_22);
            let mut i_com_rsp_w =
                LeftIdeal::from_generator(&alpha_rsp_w, &i_com_rsp_norm_w, o0_w.order());
            if !i_com_rsp_w.reduce_to_prime_norm::<44, _>(rng) {
                continue;
            }
            let i_com_rsp = match i_com_rsp_w.narrow() {
                Some(i) => i,
                None => continue,
            };

            // Lines 21–33: compute response isogeny
            let (mut e_chl, mut p_chl, mut q_chl);
            let curve_aux;
            let p_aux;
            let q_aux;

            if e_rsp_prime > 0 {
                // Lines 22–27: auxiliary isogeny path
                let aux_norm = BigInt::<4>::ONE.shl(e_rsp_prime).ct_sub(&q_rsp);
                let i_aux = match LeftIdeal::<4>::random_norm(&aux_norm, &EXTREMAL_ORDERS[0]) {
                    Some(i) => i,
                    None => continue,
                };

                // Line 24: E_aux, P_aux, Q_aux ← IdealToIsogeny(I_{com,rsp} ∩ I_aux)
                //
                // The intersection of two O₀-ideals with coprime norms N₁, N₂
                // is an O₀-ideal of norm N₁·N₂. `i_com_rsp.norm()` is the
                // narrowed prime norm after reduction.
                let inter_lattice = i_com_rsp.lattice().intersection(i_aux.lattice());
                let inter_norm = i_com_rsp.norm().ct_mul(i_aux.norm());
                let i_inter =
                    LeftIdeal::from_parts(inter_lattice, inter_norm, *EXTREMAL_ORDERS[0].order());
                let (e_aux_prime, p_aux_prime, q_aux_prime) = match i_inter.to_isogeny() {
                    Some(r) => r,
                    None => continue,
                };

                let split = match split_auxiliary_isogeny(
                    &e_com,
                    &e_aux_prime,
                    &p_com,
                    &q_com,
                    &p_aux_prime,
                    &q_aux_prime,
                    q_rsp,
                    e_rsp_prime_te,
                    r_rsp,
                ) {
                    Some(r) => r,
                    None => continue,
                };
                curve_aux = split.0;
                p_aux = split.1;
                q_aux = split.2;
                e_chl = split.3;
                p_chl = split.4;
                q_chl = split.5;
            } else {
                // Lines 28–31: direct path
                let (ec, pc, qc) = match i_com_narrow.to_isogeny() {
                    Some(r) => r,
                    None => continue,
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
                    None => continue,
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
                    None => continue,
                };
                e_chl = ec;
                p_chl = pc;
                q_chl = qc;
            }

            // Line 36: ComputeChallengeIsogeny
            let (e_chl_final, p_chl_final, q_chl_final) =
                match compute_challenge_isogeny(&basis_pk, &chl, &e_chl, &p_chl, &q_chl, n_bt_te) {
                    Some(r) => r,
                    None => continue,
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

            let det_aux_scaled = TorsionBasis::new(
                &scale_scalar * &det_aux.R,
                &scale_scalar * &det_aux.S,
                &scale_scalar * &det_aux.RS,
            );
            let det_chl_scaled = TorsionBasis::new(
                &scale_scalar * &det_chl.R,
                &scale_scalar * &det_chl.S,
                &scale_scalar * &det_chl.RS,
            );

            let basis_aux = TorsionBasis::new(p_aux, q_aux, p_aux.projective_difference(&q_aux));
            let m1 = ChangeOfBasisMatrix::from_bases(&basis_aux, &det_aux_scaled, e_cob);

            let basis_chl = TorsionBasis::new(
                p_chl_final,
                q_chl_final,
                p_chl_final.projective_difference(&q_chl_final),
            );
            let transformed = m1.mul(&basis_chl);
            let m_chl = ChangeOfBasisMatrix::from_bases(&det_chl_scaled, &transformed, e_cob);

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
/// Returns `(E_aux, P_aux, Q_aux, E_chl, P_chl, Q_chl)`.
///
/// Implements [SplitAuxiliaryIsogeny][Alg. 4.5] ([Algorithm 4.5][Alg. 4.5]).
///
/// [Alg. 4.5]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.4.5
#[allow(clippy::too_many_arguments)]
pub(crate) fn split_auxiliary_isogeny(
    e1: &Curve,
    e2: &Curve,
    p1: &ProjectiveXOnlyPoint,
    q1: &ProjectiveXOnlyPoint,
    p2: &ProjectiveXOnlyPoint,
    q2: &ProjectiveXOnlyPoint,
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

    // Line 1: P''₁, Q''₁ ← [2^{f-e'-r-2}]P₁, [2^{f-e'-r-2}]Q₁
    let scale1 = f - e_prime_val - r_val - 2;
    let scale1_scalar = Scalar::from_limbs(*BigInt::<4>::ONE.shl(scale1).as_limbs());
    let p1_double_prime = &scale1_scalar * p1;
    let q1_double_prime = &scale1_scalar * q1;

    // Line 2: P'₁, Q'₁ ← [2^r]P''₁, [2^r]Q''₁
    let mut p1_prime = p1_double_prime;
    let mut q1_prime = q1_double_prime;
    for _ in 0..r_val {
        p1_prime = p1_prime.double();
        q1_prime = q1_prime.double();
    }

    // Line 3: q_inv ← q^{-1} (mod 2^{f-e'-2})
    let mod_bits = f - e_prime_val - 2;
    let q_scalar = Scalar::from_limbs(*q_rsp.as_limbs());
    let q_inv = q_scalar.inv_mod2k(mod_bits)?;

    // Line 4: P'₂, Q'₂ ← [q_inv·2^{f-e'-2}]P₂, [q_inv·2^{f-e'-2}]Q₂
    let shift_scalar = Scalar::from_limbs(*BigInt::<4>::ONE.shl(mod_bits).as_limbs());
    let scale2 = q_inv.mul_mod2k(&shift_scalar, f);
    let p2_prime = &scale2 * p2;
    let q2_prime = &scale2 * q2;

    // Line 5: (2,2)-isogeny chain
    // Kernel: ((P'₁, P'₂), (Q'₁, Q'₂))
    // Torsion to push through: {(P''₁, 0_{E₂}), (Q''₁, 0_{E₂})}
    let product = surfaces::EllipticProduct::new(*e1, *e2);
    let pmq1_prime = p1_prime.projective_difference(&q1_prime);
    let pmq2_prime = p2_prime.projective_difference(&q2_prime);
    let kernel = surfaces::Kernel::from_montgomery(
        product,
        (p1_prime, p2_prime),
        (q1_prime, q2_prime),
        (pmq1_prime, pmq2_prime),
    )?;

    let zero_e2 = ProjectiveXOnlyPoint::identity(e2);
    let e_chain = TorsionExponent::try_from(e_prime_val + r_val).ok()?;
    let (codomain, images) = kernel.isogeny(
        e_chain,
        &[(p1_double_prime, zero_e2), (q1_double_prime, zero_e2)],
    );

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
