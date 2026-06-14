//! Precomputed quaternion data for the NIST-I parameter set.
//!
//! Contains the seven p-extremal maximal orders and their associated
//! elements, as described in [§3.1.7.2] of the SQIsign specification.
//! These are used by `RepresentInteger` ([Alg. 3.12]) to solve
//! norm equations in the quaternion algebra B_{p,∞} = (-1, -p)_Q
//! where p = 5 · 2²⁴⁸ − 1.
//!
//! Values match the C reference's `quaternion_data.c` for `lvl1` and
//! are independently checked by `scripts/precomp/verify_precomputed.py`
//! (SageMath). Run `cd scripts/precomp && sage verify_precomputed.py`
//! to re-verify.
//!
//! [§3.1.7.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.7.2
//! [Alg. 3.12]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.12

use super::{bigint::BigInt, lattice::ExtremalOrder};

/// Number of precomputed extremal orders.
pub const NUM_EXTREMAL_ORDERS: usize = 7;

/// The prime p = 5 · 2²⁴⁸ − 1 as a [`BigInt<4>`]. Only the quaternion
/// reduced-norm test helpers consume this `BigInt<4>` form of `p`; the
/// production norm paths carry `p` through other constants.
#[cfg(test)]
pub const P: BigInt<4> = BigInt::from_limbs([
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0x04FFFFFFFFFFFFFF,
]);

/// The prime p = 5 · 2²⁴⁸ − 1 as a [`BigInt<8>`] for arithmetic
/// that needs wider intermediates.
pub const P_WIDE: BigInt<8> = BigInt::from_limbs([
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0x04FFFFFFFFFFFFFF,
    0,
    0,
    0,
    0,
]);

/// Modulus `D = 4 · d⁴ · D_MIX² · p` for mod-HNF on commitment
/// ideals (see [`Matrix::from_hnf_columns_mod`]).
///
/// For NIST-I with d = 2 (O₀'s stored basis denominator),
/// D_MIX = 2⁵¹² + 75, and p = 5 · 2²⁴⁸ − 1, this evaluates to a
/// 1281-bit integer that is a positive multiple of the integer-
/// column covolume of the lattice `d · I` for any commitment
/// ideal I = O₀⟨γ, D_MIX⟩. That is exactly what
/// [`Matrix::from_hnf_columns_mod`]'s bounding modulus needs.
///
/// Stored as a literal [`BigInt<30>`] — matching the working
/// width of `LeftIdeal<30>::random_prime_norm_wide` — because
/// `BigInt::ct_mul` is not `const fn`, so we cannot write
/// `4 * d⁴ * D_MIX² * p` as a compile-time expression. The limbs
/// below were precomputed in Python; a `#[test]` in this module
/// re-derives them at runtime and asserts equality, guarding
/// against silent drift if the parameter set ever changes.
///
/// [`Matrix::from_hnf_columns_mod`]: crate::quaternions::linear::Matrix::from_hnf_columns_mod
pub const D_HNF_MODULUS_COMMITMENT: BigInt<30> = BigInt::from_limbs([
    0xFFFFFFFFFFFA81C0,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0x3FFFFFFFFFFFFFFF,
    0x0000000000001B77,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0xFFFFFFFFFFFFDA80,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0x7FFFFFFFFFFFFFFF,
    0x00000000000000BB,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0xFFFFFFFFFFFFFFC0,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0x3FFFFFFFFFFFFFFF,
    0x0000000000000001,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
]);

/// The seven precomputed p-extremal maximal orders for NIST-I.
///
/// Each order has a distinguished element z with z² = −q (small q),
/// and an element t = j with nrd(t) = p. The q values are:
/// `[1, 5, 17, 37, 41, 53, 97]`.
///
/// `EXTREMAL_ORDERS[0]` is the standard order O₀ with basis
/// `{1, i, (i+j)/2, (1+k)/2}` and q = 1.
///
/// All data is `(sign, [u64; 4])` — see [`ExtremalOrder::from_raw_limbs`]
/// for the data format and how it maps to the C reference's layout.
#[rustfmt::skip]
pub const EXTREMAL_ORDERS: [ExtremalOrder<4>; NUM_EXTREMAL_ORDERS] = [
    // Order 0 (q = 1): the standard order O₀.
    // Basis: {1, i, (i+j)/2, (1+k)/2}, z = i, t = j.
    ExtremalOrder::from_raw_limbs(
        [
            [(0, [2, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [1, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (0, [2, 0, 0, 0]), (0, [1, 0, 0, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [1, 0, 0, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [1, 0, 0, 0])],
        ],
        [(0, [0, 0, 0, 0]), (0, [2, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [0, 0, 0, 0])],
        1,
    ),
    // Order 1 (q = 5).
    ExtremalOrder::from_raw_limbs(
        [
            [(0, [0, 0x1000000000000000, 0, 0]), (0, [0, 0, 0, 0]), (0, [0, 0x0800000000000000, 0, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (1, [1, 0, 0, 0]), (0, [0, 0, 0, 0]), (1, [0, 0, 0, 0x0080000000000000])],
            [(0, [0, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [0, 0x0800000000000000, 0, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (1, [1, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [0, 0, 0, 0])],
        ],
        [(0, [0, 0, 0, 0]), (1, [1, 0, 0, 0]), (0, [0, 0, 0, 0]), (1, [1, 0, 0, 0])],
        5,
    ),
    // Order 2 (q = 17).
    ExtremalOrder::from_raw_limbs(
        [
            [(0, [0xF5F27A647B8578D4, 0xB8746101369629B9, 0, 0]), (0, [0, 0, 0, 0]), (0, [0xFAF93D323DC2BC6A, 0x5C3A30809B4B14DC, 0, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (0, [0x95AD2AD56FA47D47, 0xC89877E749BE8A4B, 1, 0]), (0, [0, 0, 0, 0]), (0, [0x3E355E2970603F47, 0x78DD10AE2A1BD950, 0, 0x0280000000000000])],
            [(0, [0, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [0xFAF93D323DC2BC6A, 0x5C3A30809B4B14DC, 0, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (0, [0x11, 0, 0, 0]), (0, [0, 0, 0, 0]), (1, [0xB19426E828EE3FE7, 0x0D6DE568AF586D7A, 0, 0])],
        ],
        [(0, [0, 0, 0, 0]), (0, [0x95AD2AD56FA47D47, 0xC89877E749BE8A4B, 1, 0]), (0, [0, 0, 0, 0]), (0, [0x11, 0, 0, 0])],
        17,
    ),
    // Order 3 (q = 37).
    ExtremalOrder::from_raw_limbs(
        [
            [(0, [0x3C6FA8E67715E5E2, 0x17949BEC872B9078, 0, 0]), (0, [0, 0, 0, 0]), (0, [0x9E37D4733B8AF2F1, 0x0BCA4DF64395C83C, 0, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (0, [0xD59D8F4E5F28AFF9, 0xB2CBC61BD37EF3F9, 0, 0]), (0, [0, 0, 0, 0]), (1, [0x6D86CF9EFD858949, 0x16ED7F44E09E115D, 0, 0x00C0000000000000])],
            [(0, [0, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [0x9E37D4733B8AF2F1, 0x0BCA4DF64395C83C, 0, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (0, [0x25, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [0xBD312454CA3A0E7F, 0x002172F0CB4CE562, 0, 0])],
        ],
        [(0, [0, 0, 0, 0]), (1, [0xB034808274C8307A, 0x09AB399AC43A4E8A, 0, 0]), (0, [0, 0, 0, 0]), (0, [4, 0, 0, 0])],
        37,
    ),
    // Order 4 (q = 41).
    ExtremalOrder::from_raw_limbs(
        [
            [(0, [0xDE33C5116DEEAFA2, 0x2DF94F97C89EC8CE, 0, 0]), (0, [0, 0, 0, 0]), (0, [0x6F19E288B6F757D1, 0x16FCA7CBE44F6467, 0, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (0, [0xD17AA943DA6BDD36, 0x44D44B0C564CE307, 0, 0]), (0, [0, 0, 0, 0]), (1, [0xA0A2047CC4063A03, 0x6CEE07961DF46DBC, 0xC7CE0C7CE0C7CE0C, 0x007CE0C7CE0C7CE0])],
            [(0, [0, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [0x6F19E288B6F757D1, 0x16FCA7CBE44F6467, 0, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (1, [8, 0, 0, 0]), (0, [0, 0, 0, 0]), (1, [0xD9F82148A1E2188F, 0x00D6E1B21A072E79, 0, 0])],
        ],
        [(0, [0, 0, 0, 0]), (0, [0xD17AA943DA6BDD36, 0x44D44B0C564CE307, 0, 0]), (0, [0, 0, 0, 0]), (1, [8, 0, 0, 0])],
        41,
    ),
    // Order 5 (q = 53).
    ExtremalOrder::from_raw_limbs(
        [
            [(0, [0x380014F2025B96A4, 0x7BBEAB7F79584E7C, 1, 0]), (0, [0, 0, 0, 0]), (0, [0x1C000A79012DCB52, 0xBDDF55BFBCAC273E, 0, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (1, [0x4BA119E7333973E3, 0xDBD0EE6227026EBC, 7, 0]), (0, [0, 0, 0, 0]), (0, [0x09F01D923DD0CA33, 0x83F7E395AFE92F81, 0xFFFFFFFFFFFFFFFC, 0x027FFFFFFFFFFFFF])],
            [(0, [0, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [0x1C000A79012DCB52, 0xBDDF55BFBCAC273E, 0, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (0, [0x35, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [0x87F571C0F93CEB73, 0x12FAB9CBCB3C667A, 0, 0])],
        ],
        [(0, [0, 0, 0, 0]), (1, [0x4BA119E7333973E3, 0xDBD0EE6227026EBC, 7, 0]), (0, [0, 0, 0, 0]), (0, [0x35, 0, 0, 0])],
        53,
    ),
    // Order 6 (q = 97).
    ExtremalOrder::from_raw_limbs(
        [
            [(0, [0xE2B97B9E55AF7FFA, 0xC227F76B578CA7AF, 0xF, 0]), (0, [0, 0, 0, 0]), (0, [0xF15CBDCF2AD7BFFD, 0xE113FBB5ABC653D7, 7, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (1, [0xA2EF1CE7F02B0D16, 0x066759632C56054B, 0x6F, 0]), (0, [0, 0, 0, 0]), (0, [0x84AC06EA9D3BF0AB, 0xD021882BDDE962E5, 0xFFFFFFFFFFFFFFE2, 0x13FFFFFFFFFFFFFF])],
            [(0, [0, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [0xF15CBDCF2AD7BFFD, 0xE113FBB5ABC653D7, 7, 0]), (0, [0, 0, 0, 0])],
            [(0, [0, 0, 0, 0]), (0, [0x308, 0, 0, 0]), (0, [0, 0, 0, 0]), (0, [0x077013F15C4A1F37, 0x9281DA3156007183, 0, 0])],
        ],
        [(0, [0, 0, 0, 0]), (1, [0xA2EF1CE7F02B0D16, 0x066759632C56054B, 0x6F, 0]), (0, [0, 0, 0, 0]), (0, [0x308, 0, 0, 0])],
        97,
    ),
];

/// The standard order O₀ (alias for `EXTREMAL_ORDERS[0]`).
pub const STANDARD_ORDER: &ExtremalOrder<4> = &EXTREMAL_ORDERS[0];

/// Connecting ideal data for each extremal order.
///
/// `CONNECTING_IDEAL_NORMS[t]` is the reduced norm of the connecting
/// ideal I_t (a left O₀-ideal with right order conjugate to O_t).
/// "Reduced norm" here matches C ref's `quat_lideal_norm`
/// (`quaternion/ref/generic/ideal.c:7`), which is `sqrt([O_0 : I])`
/// computed from the lattice index. This equals Sage's
/// `QuaternionFractionalIdeal.norm()`.
///
/// All ideals have HNF basis of the form:
///
/// ```text
///   [2N, 0, 0, 0]
///   [0, 2N, 0, 0]    / denom = 2
///   [0, x,  1, 0]
///   [y, 0,  0, 1]
/// ```
///
/// (Pre-denom diagonal `(2N, 2N, 1, 1)` divided by 2 gives the affine
/// elements `(N, N·i, (x·i + j)/2, (y + k)/2)`. The reduced norm is
/// `N`, half the leading basis entry.)
///
/// For t=0 the connecting ideal is O₀ itself (norm 1, x=1, y=1).
///
/// Originally extracted via the Sage script
/// `scripts/precomp/extract_connecting_ideals.sage`, which returns
/// the leading basis entry rather than the reduced norm — these are
/// off by a factor of 2 for the denom-2 representation. Values below
/// were halved to match C ref's labeled `CONNECTING_IDEALS[t].norm`.
pub const CONNECTING_IDEAL_NORMS: [BigInt<4>; NUM_EXTREMAL_ORDERS] = [
    BigInt::from_sign_and_limbs(
        0,
        [
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=1
    BigInt::from_sign_and_limbs(
        0,
        [
            0x0000000000000001,
            0x3000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=5
    BigInt::from_sign_and_limbs(
        0,
        [
            0xBFC80ABDC339FAFF,
            0x3C7A53236805E962,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=17
    BigInt::from_sign_and_limbs(
        0,
        [
            0x1E37D4733B8AF2F1,
            0x0BCA4DF64395C83C,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=37
    BigInt::from_sign_and_limbs(
        0,
        [
            0x6F19E288B6F757D1,
            0x16FCA7CBE44F6467,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=41
    BigInt::from_sign_and_limbs(
        0,
        [
            0xA951773BAACA0CF9,
            0x59A410C3A2E4FA2C,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=53
    BigInt::from_sign_and_limbs(
        0,
        [
            0xE818B56BB3E7D51D,
            0x14CB6C2975E50380,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=97
];

/// The `x` entry (row 2, col 1) of each connecting ideal's HNF basis.
pub const CONNECTING_IDEAL_X: [BigInt<4>; NUM_EXTREMAL_ORDERS] = [
    BigInt::from_sign_and_limbs(
        0,
        [
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=1
    BigInt::from_sign_and_limbs(
        0,
        [
            0x0000000000000001,
            0x5000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=5
    BigInt::from_sign_and_limbs(
        0,
        [
            0x99333EA38647F719,
            0x73436F08E8DE6A21,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=17
    BigInt::from_sign_and_limbs(
        0,
        [
            0x81469E8C3C1E604B,
            0x0F44A68AB45218B7,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=37
    BigInt::from_sign_and_limbs(
        0,
        [
            0x083DF746C4E35E07,
            0x1F9F9C4344C15354,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=41
    BigInt::from_sign_and_limbs(
        0,
        [
            0x34AE63E0BF193E1F,
            0xB125DFED38597C14,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=53
    BigInt::from_sign_and_limbs(
        0,
        [
            0x13C97CEB9024A7C5,
            0x1507E56D3D1459C0,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=97
];

/// The `y` entry (row 3, col 0) of each connecting ideal's HNF basis.
pub const CONNECTING_IDEAL_Y: [BigInt<4>; NUM_EXTREMAL_ORDERS] = [
    BigInt::from_sign_and_limbs(
        0,
        [
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=1
    BigInt::from_sign_and_limbs(
        0,
        [
            0x0000000000000001,
            0x1000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=5
    BigInt::from_sign_and_limbs(
        0,
        [
            0xE65CD6D8002BFEE5,
            0x05B1373DE72D68A3,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=17
    BigInt::from_sign_and_limbs(
        0,
        [
            0xBB290A5A3AF78597,
            0x084FF561D2D977C0,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=37
    BigInt::from_sign_and_limbs(
        0,
        [
            0xD5F5CDCAA90B519B,
            0x0E59B35483DD757A,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=41
    BigInt::from_sign_and_limbs(
        0,
        [
            0x1DF48A96967ADBD3,
            0x0222419A0D707845,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=53
    BigInt::from_sign_and_limbs(
        0,
        [
            0xBC67EDEBD7AB0275,
            0x148EF2E5AEB5AD41,
            0x0000000000000000,
            0x0000000000000000,
        ],
    ), // q=97
];

#[cfg(test)]
mod tests {
    use super::{
        super::{
            algebra::{Coordinate, Denominator, Element},
            lattice::LeftIdeal,
        },
        *,
    };

    #[test]
    fn standard_order_q_is_one() {
        assert_eq!(STANDARD_ORDER.q(), 1);
    }

    /// Every connecting ideal builds without panicking and
    /// round-trips its stored norm through
    /// [`LeftIdeal::from_parts`]. Parent-order equality is checked
    /// via the underlying `Lattice` basis since `Order` does not
    /// derive `PartialEq`.
    #[test]
    fn connecting_ideal_norm_and_parent_roundtrip() {
        for (t, expected_norm) in CONNECTING_IDEAL_NORMS.iter().enumerate() {
            let j = LeftIdeal::<4>::connecting(t).expect("t < NUM_EXTREMAL_ORDERS by iter bound");
            assert_eq!(*j.norm(), *expected_norm, "J_{t} norm mismatch");
            assert_eq!(
                j.parent_order().basis(),
                STANDARD_ORDER.order().basis(),
                "J_{t} parent-order basis mismatch",
            );
        }
    }

    /// Re-derive `D_HNF_MODULUS_COMMITMENT = 4 · d⁴ · D_MIX² · p`
    /// at runtime and assert the precomputed limbs still match the
    /// NIST-I parameter set. Catches silent drift if D_MIX or p
    /// ever change without the literal being regenerated.
    #[test]
    fn d_hnf_modulus_commitment_matches_formula() {
        // d = 2, d⁴ = 16, 4 · d⁴ = 64.
        let d_mix: BigInt<30> = crate::params::D_MIX.widen();
        let p: BigInt<30> = {
            let mut limbs = [0u64; 30];
            limbs[..8].copy_from_slice(P_WIDE.as_limbs());
            BigInt::from_sign_and_limbs(0, limbs)
        };
        let d_mix_sq = d_mix.ct_mul(&d_mix);
        let sixty_four = BigInt::<30>::from_u64(64);
        let expected = sixty_four.ct_mul(&d_mix_sq).ct_mul(&p);
        assert_eq!(expected, D_HNF_MODULUS_COMMITMENT);
    }

    #[test]
    fn all_q_values() {
        let expected = [1, 5, 17, 37, 41, 53, 97];
        for (i, &q) in expected.iter().enumerate() {
            assert_eq!(EXTREMAL_ORDERS[i].q(), q, "order {i} has wrong q");
        }
    }

    #[test]
    fn z_squared_is_neg_q_for_all_orders() {
        for (idx, order) in EXTREMAL_ORDERS.iter().enumerate() {
            let q = order.q();
            let z = order.z();
            let z_sq = z
                .mul(z)
                .expect("z² of an extremal-order generator fits in BigInt<4>")
                .normalized();

            assert_eq!(
                z_sq.a,
                Coordinate::from_i64(-(q as i64)),
                "order {idx} (q={q}): z².a != -{q}"
            );
            assert_eq!(z_sq.b, Coordinate::ZERO, "order {idx}: z².b != 0");
            assert_eq!(z_sq.c, Coordinate::ZERO, "order {idx}: z².c != 0");
            assert_eq!(z_sq.d, Coordinate::ZERO, "order {idx}: z².d != 0");
            assert_eq!(z_sq.denom, Denominator::ONE, "order {idx}: z².denom != 1");
        }
    }

    #[test]
    fn t_is_j_for_all_orders() {
        for (i, order) in EXTREMAL_ORDERS.iter().enumerate() {
            assert_eq!(*order.t(), Element::J, "order {i}: t != j");
        }
    }

    /// Cross-check precomputed constants against an independent SageMath
    /// recomputation. Requires `sage` on PATH.
    ///
    /// Run with: `cargo test sage_cross_check -- --ignored`
    #[test]
    #[ignore]
    fn sage_cross_check() {
        use std::process::Command;

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let script_dir = format!("{manifest_dir}/scripts/precomp");

        let output = Command::new("sage")
            .arg("verify_precomputed.py")
            .current_dir(&script_dir)
            .output()
            .expect("failed to run `sage verify_precomputed.py` — is SageMath installed?");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        assert!(
            output.status.success(),
            "sage verify_precomputed.py failed.\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );

        let json: serde_json::Value = serde_json::from_str(stdout.trim())
            .unwrap_or_else(|e| panic!("failed to parse Sage JSON output: {e}\nstdout:\n{stdout}"));

        let sage_orders = json["orders"]
            .as_array()
            .expect("orders should be an array");
        assert_eq!(
            sage_orders.len(),
            NUM_EXTREMAL_ORDERS,
            "Sage computed {} orders, expected {NUM_EXTREMAL_ORDERS}",
            sage_orders.len()
        );

        for (i, sage_order) in sage_orders.iter().enumerate() {
            let sage_q = sage_order["q"].as_u64().unwrap() as u32;
            let rust_q = EXTREMAL_ORDERS[i].q();
            assert_eq!(
                sage_q, rust_q,
                "order {i}: q mismatch (sage={sage_q}, rust={rust_q})"
            );
        }
    }
}
