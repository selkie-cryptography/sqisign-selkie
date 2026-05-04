//! NIST-I parameter set constants for SQIsign-353.
//!
//! All scheme parameters are derived from the prime p and the security
//! parameter λ. See [§4.2] (parameters) and [§5.2] (parameter sets).
//!
//! [§4.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.2
//! [§5.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.5.2

use crate::quaternions::bigint::{BigInt, MontCtx};

/// Security parameter λ = 128.
pub const SECURITY_BITS: u32 = 128;

/// The prime p = 5 · 2^248 − 1.
///
/// The cofactor c = 5 and the 2-valuation f = 248, so p = c · 2^f − 1.
/// p ≡ 3 (mod 4), which gives us i² = −1 in F_{p²}.
pub const COFACTOR: u64 = 5;

/// The 2-valuation f of p + 1: the largest integer such that 2^f divides p + 1.
///
/// This is the exponent of the full even torsion subgroup E[2^f].
/// f ≈ 2λ = 248 for NIST-I.
pub const TORSION_EVEN_POWER: u32 = 248;

/// e_rsp = ⌈log₂(√p)⌉, the bit-length of the response isogeny degree.
///
/// For NIST-I: ⌈251/2⌉ = 126. (SQIsign_response_length in the C reference.)
pub const E_RSP: u32 = 126;

/// e_chl = f − e_rsp, the bit-length of the challenge space.
///
/// For NIST-I: 248 − 126 = 122.
pub const E_CHL: u32 = TORSION_EVEN_POWER - E_RSP;

/// Number of iterations of SHAKE256 used to build the hash function HASH.
///
/// HASH = SHAKE256_{122} ∘ SHAKE256_{256}^{∘63} for NIST-I.
pub const HASH_ITERATIONS: u32 = 64;

/// Number of bytes to encode an element of F_p.
pub const FP_ENCODED_BYTES: usize = 32;

/// Number of bytes to encode an element of F_{p²}.
pub const FP2_ENCODED_BYTES: usize = 2 * FP_ENCODED_BYTES;

/// Number of bytes to encode a Montgomery curve coefficient A ∈ F_{p²}.
pub const CURVE_ENCODED_BYTES: usize = FP2_ENCODED_BYTES;

/// Number of bytes for each component of the change-of-basis matrix
/// (integers mod 2^f).
pub const TORSION_2POWER_BYTES: usize = 32;

/// Public (verifying) key size in bytes.
pub const VERIFYING_KEY_BYTES: usize = 65;

/// Secret (signing) key size in bytes.
pub const SIGNING_KEY_BYTES: usize = 353;

/// Signature size in bytes.
pub const SIGNATURE_BYTES: usize = 148;

// ---------------------------------------------------------------------------
// SuitableIdeals / id2iso parameters
// ---------------------------------------------------------------------------

/// Half-width of the enumeration box in [Alg. 3.16][Alg. 3.16]
/// (SuitableIdeals).
///
/// For NIST-I: m = 2 + ⌊(⌈log₂ p⌉ − f) / 4⌋ = 2 + ⌊3/4⌋ = 2.
/// The algorithm enumerates (2m+1)⁴ − 1 = 624 non-zero vectors per order.
///
/// [Alg. 3.16]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.16
pub const FINDUV_BOX_SIZE: i64 = 2;

/// Bound on the RepresentInteger search window.
///
/// Controls the dimension of the isogeny kernel needed for
/// [Alg. 3.15][Alg. 3.15] (FixedDegreeIsogeny) to succeed.
///
/// [Alg. 3.15]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.15
pub const QUAT_REPRES_BOUND_INPUT: u32 = 20;

/// Sampling bound for `LeftIdeal::reduce_to_prime_norm` (Algorithm 3.9).
///
/// Random coefficients are sampled from \[−bound, bound\]⁴. Matches
/// the C reference's `QUAT_equiv_bound_coeff` (`precomp/ref/lvl1/
/// include/quaternion_constants.h:4`, value `64`). Must match the C
/// ref byte-for-byte: a smaller bound changes the per-iteration
/// rejection rate AND the integer values produced from a given DRBG
/// byte stream, which desyncs every downstream sample and produces
/// a different reduced ideal — and therefore a different `e_pk` —
/// from the same KAT seed.
pub const EQUIV_BOUND_COEFF: u32 = 64;

/// Number of Miller-Rabin rounds for primality testing in ideal
/// reduction (Algorithm 3.9). Matches the C reference's
/// `QUAT_primality_num_iter` (= 32).
pub const PRIMALITY_NUM_ITER: u32 = 32;

/// Precomputed prime cofactor for [`RandomIdealGivenNorm`][Alg. 3.10]
/// (non-prime case).
///
/// The smallest prime of the same bit size as p, used as the
/// multiplier m in `GeneralizedRepresentInteger(mN, ...)`.
/// For NIST-I: `QUAT_prime_cofactor = 2^251 + 65`.
///
/// [Alg. 3.10]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.10
pub const QUAT_PRIME_COFACTOR: BigInt<4> = BigInt::from_limbs([0x41, 0, 0, 0x0800_0000_0000_0000]);

/// Commitment/secret isogeny degree D_mix (= `COM_DEGREE` in the C ref).
///
/// The smallest prime > 2^{4λ} = 2^{512}. For NIST-I: `D_mix = 2^512 + 75`.
/// This is 513 bits, so it requires `BigInt<9>` (576 bits). Our ideal
/// infrastructure uses `BigInt<4>` and `BigInt<8>`, so `random_prime_norm`
/// needs to be widened to accept larger norms for the commitment ideal.
///
/// See [§4.2.1] of the spec.
///
/// [§4.2.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsection.4.2.1
pub const D_MIX: BigInt<9> = BigInt::from_limbs([0x4B, 0, 0, 0, 0, 0, 0, 0, 1]);

/// `D_mix` widened to 18 limbs — the working width for `pow_mod_w::<18>`
/// chains. Same value as [`D_MIX`], zero-padded.
pub const D_MIX_W18: BigInt<18> = BigInt::from_limbs([
    0x4B, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]);

/// Precomputed Montgomery context for `D_mix` at width 18 — built at
/// compile time, eliminates the runtime `MontCtx::new` cost (Newton
/// iter for `n_inv` + 128·N doublings for `R²`) for every pow_mod call
/// against `D_mix`.
///
/// Use directly via `D_MIX_W18_MONT.pow(base, exp)`, or pass to any
/// `*_with_ctx` API that accepts a `&MontCtx<18>`.
pub const D_MIX_W18_MONT: MontCtx<18> = MontCtx::const_new(&D_MIX_W18);

// ---------------------------------------------------------------------------
// Precomputed E₀ basis points
// ---------------------------------------------------------------------------

use crate::fields::{fp::Fp, fp2::Fp2};

/// x-coordinate of the first basis point P₀ of E₀\[2^f\],
/// where f = [`TORSION_EVEN_POWER`].
///
/// (P₀, Q₀) generates E₀\[2^f\] on E₀ : y² = x³ + x.
/// Stored in Montgomery form, radix-51 representation.
///
/// See [Appendix B].
///
/// [Appendix B]: https://sqisign.org/spec/sqisign-20250707.pdf#appendix.B
pub const BASIS_E0_P_X: Fp2 = Fp2::new(
    Fp::from_limbs([
        0x0005BCAB12000C08,
        0x000452654B56D052,
        0x00026F81B5190A0A,
        0x00036CFD66A361EB,
        0x000012726610D11B,
    ]),
    Fp::from_limbs([
        0x0006B96065C83EFC,
        0x00029DA1D4A82CD9,
        0x000190797AB98BDF,
        0x0006841AA6EEEE05,
        0x0001377C5431166,
    ]),
);

/// x-coordinate of the second basis point Q₀ of E₀\[2^f\],
/// where f = [`TORSION_EVEN_POWER`].
///
/// (P₀, Q₀) generates E₀\[2^f\] on E₀ : y² = x³ + x.
/// Stored in Montgomery form, radix-51 representation.
///
/// See [Appendix B].
///
/// [Appendix B]: https://sqisign.org/spec/sqisign-20250707.pdf#appendix.B
pub const BASIS_E0_Q_X: Fp2 = Fp2::new(
    Fp::from_limbs([
        0x00021DD55B97832F,
        0x000210F2D30B26AD,
        0x00000680BCFCF6396,
        0x00027B318EC126A7,
        0x000004FFBA5956012,
    ]),
    Fp::from_limbs([
        0x00074590149117E3,
        0x0004982EDEFCC606,
        0x0002AE3DB0CC6884,
        0x0007D0384872F5EC,
        0x000004FBB0FCB5A52,
    ]),
);

/// x-coordinate of the difference point P₀ − Q₀ of E₀\[2^f\],
/// where f = [`TORSION_EVEN_POWER`].
///
/// Precomputed from `projective_difference(P₀, Q₀)` on E₀.
/// Stored in Montgomery form, radix-51 representation.
pub const BASIS_E0_PMQ_X: Fp2 = Fp2::new(
    Fp::from_limbs([
        270480358487834,
        2072266045736319,
        1674191439884908,
        2200260875474967,
        6907110771017,
    ]),
    Fp::from_limbs([
        1752869285732728,
        495365606488051,
        1818143936964406,
        314346222928849,
        165077940050103,
    ]),
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Check that x³ + x is a square in F_{p²} (i.e., (x, ·) is on E₀).
    fn is_on_e0(x: &Fp2) -> bool {
        let x2 = x.square();
        let rhs = &(&x2 + &Fp2::ONE) * x;
        bool::from(rhs.is_square())
    }

    #[test]
    fn basis_e0_p_is_on_curve() {
        assert!(is_on_e0(&BASIS_E0_P_X), "P₀ x-coordinate is not on E₀");
    }

    #[test]
    fn basis_e0_q_is_on_curve() {
        assert!(is_on_e0(&BASIS_E0_Q_X), "Q₀ x-coordinate is not on E₀");
    }

    #[test]
    fn basis_e0_points_are_distinct() {
        assert_ne!(BASIS_E0_P_X, BASIS_E0_Q_X);
    }

    #[test]
    fn basis_e0_pmq_is_on_curve() {
        assert!(is_on_e0(&BASIS_E0_PMQ_X), "P₀−Q₀ x-coordinate is not on E₀");
    }

    #[test]
    fn basis_e0_pmq_matches_projective_difference() {
        use crate::curves::montgomery::{Curve, ProjectiveXOnlyPoint};
        let curve = Curve::E0;
        let p = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &curve);
        let q = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_Q_X, &curve);
        let pmq = p.projective_difference(&q);
        let x = pmq.to_affine_x();
        assert_eq!(*x.as_fp2(), BASIS_E0_PMQ_X);
    }
}
