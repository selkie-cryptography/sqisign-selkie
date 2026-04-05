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

use crate::{
    curves::{
        TorsionBasis, TorsionExponent,
        isogeny::Kernel,
        montgomery::{Curve, ProjectiveXOnlyPoint},
        scalar::Scalar,
    },
    keys::{
        Challenge, SIGNING_KEY_BYTES, Signature, SignatureError, VERIFYING_KEY_BYTES,
        verifying::VerifyingKey,
    },
    params::{FP_ENCODED_BYTES, TORSION_2POWER_BYTES, TORSION_EVEN_POWER},
    quaternions::{
        algebra::{Coordinate, Denominator, Element},
        bigint::BigInt,
        lattice::LeftIdeal,
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
    /// Change-of-basis matrix M_sk: 2×2 over Z, stored as
    /// \[\[m00, m01\], \[m10, m11\]\] with entries mod 2^f.
    mat_sk: [[Scalar; 2]; 2],
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
        let mut mat_sk = [[Scalar::ZERO; 2]; 2];
        for row in &mut mat_sk {
            for entry in row.iter_mut() {
                let b = BigInt::<4>::from_bytes_le_unsigned(
                    bytes[pos..pos + TORSION_2POWER_BYTES].try_into().unwrap(),
                );
                *entry = Scalar::from(b);
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
    pub fn sign(&self, msg: &[u8]) -> Result<Signature, SignatureError> {
        // TODO: remaining issues before sign() produces valid signatures:
        //   - Degree computations (lines 16-20) are placeholders
        //   - I_com,rsp computation (line 19/24) not implemented
        //   - Signature encoding to bytes (line 38)
        //   - Right order of I_sk for pushforward (line 13)

        use crate::{
            curves::{BasisHint, ChangeOfBasisMatrix, TorsionBasis},
            params::{D_MIX, E_RSP},
            quaternions::lattice::LeftIdeal as LeftIdeal8,
        };

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
        for _ in 0..1000 {
            // --- Commitment (lines 4–9) ---

            // Line 4: I_com ← RandomIdealGivenNorm(D_mix, true)
            // D_MIX is BigInt<9> (513 bits); widen to BigInt<8> for the
            // wide ideal path. This loses the top bit — TODO: use BigInt<9>
            // when random_prime_norm_wide supports it.
            let d_mix_wide = BigInt::<8>::from_limbs({
                let mut limbs = [0u64; 8];
                let d = D_MIX.as_limbs();
                limbs[..8.min(d.len())].copy_from_slice(&d[..8.min(d.len())]);
                limbs
            });
            let mut i_com =
                match LeftIdeal8::<8>::random_prime_norm_wide(&d_mix_wide, &EXTREMAL_ORDERS[0]) {
                    Some(i) => i,
                    None => continue,
                };

            // Lines 5–6: RandomEquivalentPrimeIdeal
            if !i_com.reduce_to_prime_norm() {
                continue;
            }

            // Narrow to LeftIdeal<4> for to_isogeny.
            let i_com_narrow = match i_com.narrow() {
                Some(i) => i,
                None => continue,
            };

            // Line 7: E_com, P_com, Q_com ← IdealToIsogeny(I_com)
            let (e_com, p_com, q_com) = match i_com_narrow.to_isogeny() {
                Some(r) => r,
                None => continue,
            };

            // --- Challenge (line 10) ---
            let chl = Challenge::derive(&self.verifying_key, &e_com, msg);

            // --- Response (lines 11–38) ---

            // Line 11: (c₁, c₂) ← M_sk · (1, chl)
            let chl_scalar: Scalar = chl.into();
            let c1 = self.mat_sk[0][0].add_mod2k(&self.mat_sk[0][1].mul_mod2k(&chl_scalar, f), f);
            let c2 = self.mat_sk[1][0].add_mod2k(&self.mat_sk[1][1].mul_mod2k(&chl_scalar, f), f);

            // Line 12: I'_chl ← KernelDecomposedToIdeal(c₁, c₂)
            let c1_big = BigInt::<4>::from(c1);
            let c2_big = BigInt::<4>::from(c2);
            let i_chl_prime =
                match TorsionBasis::kernel_to_ideal(&c1_big, &c2_big, TorsionExponent::FULL) {
                    Some(ideal) => ideal,
                    None => continue,
                };

            // Line 13: I_chl ← [I_sk]_* I'_chl
            // TODO: use actual O_R(I_sk) instead of O₀.
            let i_chl = self
                .ideal
                .pushforward(&i_chl_prime, EXTREMAL_ORDERS[0].order());

            // Line 14: α_rsp ← RandomEquivalentQuaternion(I_com ∩ I_sk · I_chl)
            let i_sk_i_chl = self.ideal.lattice().product(&i_chl.lattice());
            let intersection = self.ideal.lattice().intersection(&i_sk_i_chl);
            let intersection_lat = crate::quaternions::lattice::Lattice::<4>::from(intersection);
            // TODO: compute proper radius D_rsp · D²_mix · 2^{f+1}
            let radius = BigInt::<4>::ONE.shl(f);
            let alpha_rsp = match intersection_lat.sample_from_ball(&radius) {
                Some(a) => a,
                None => continue,
            };

            // Line 15: α_rsp, n_bt ← ComputeBacktrackingAndNormalize(α_rsp)
            let (alpha_rsp, n_bt) = alpha_rsp.compute_backtracking();

            // Lines 16–20: degree computations
            // TODO: compute properly from nrd(α_rsp)
            let r_rsp_val = 0u32;
            let q_rsp: u64 = 1;
            let e_rsp_prime = e_rsp - r_rsp_val - n_bt;

            let r_rsp =
                TorsionExponent::try_from(r_rsp_val).map_err(|_| SignatureError::SigningFailed)?;
            let n_bt_te =
                TorsionExponent::try_from(n_bt).map_err(|_| SignatureError::SigningFailed)?;
            let e_rsp_prime_te = TorsionExponent::try_from(e_rsp_prime)
                .map_err(|_| SignatureError::SigningFailed)?;

            // Lines 21–33: compute response isogeny
            let (mut e_chl, mut p_chl, mut q_chl);
            let curve_aux;
            let p_aux;
            let q_aux;

            if e_rsp_prime > 0 {
                // Lines 22–27: auxiliary isogeny path
                let aux_norm = BigInt::<4>::ONE
                    .shl(e_rsp_prime)
                    .ct_sub(&BigInt::<4>::from_u64(q_rsp));
                let i_aux = match LeftIdeal::<4>::random_norm(&aux_norm, &EXTREMAL_ORDERS[0]) {
                    Some(i) => i,
                    None => continue,
                };

                // TODO: compute I_com,rsp = O₀·α_rsp + O₀(q_rsp·D_mix)
                let (e_aux_prime, p_aux_prime, q_aux_prime) = match i_aux.to_isogeny() {
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

            // Lines 34–35: even response
            if r_rsp_val > 0 {
                let (ec, pc, qc) = match crate::deuring::compute_even_response(
                    &e_chl,
                    &p_chl,
                    &q_chl,
                    &alpha_rsp,
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

            // Line 37: SetChangeOfBasisMatrix (inlined)
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
            let transformed = m1.mul(&basis_chl, e_cob);
            let m_chl = ChangeOfBasisMatrix::from_bases(&det_chl_scaled, &transformed, e_cob);

            // Line 38: assemble signature
            let hint_aux = crate::curves::AuxiliaryHint::from(hint_aux_raw.to_byte());
            let hint_chl = crate::curves::ChallengeHint::from(hint_chl_raw.to_byte());

            // TODO: encode signature to wire format (148 bytes).
            let _ = (curve_aux, n_bt_te, r_rsp, m_chl, chl, hint_aux, hint_chl);
            return Err(SignatureError::SigningFailed);
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
    q_rsp: u64,
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
    let q_scalar = Scalar::from_u64(q_rsp);
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
