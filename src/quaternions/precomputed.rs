//! Precomputed quaternion data for the NIST-I parameter set.
//!
//! Contains the seven p-extremal maximal orders and their associated
//! elements, as described in [§3.1.7.2] of the SQIsign specification.
//! These are used by `RepresentInteger` ([Algorithm 3.12]) to solve
//! norm equations in the quaternion algebra B_{p,∞} = (-1, -p)_Q
//! where p = 5 · 2²⁴⁸ − 1.
//!
//! Values are generated from the SQIsign reference implementation's
//! precomputation scripts and match `quaternion_data.c` for `lvl1`.
//! Run `cd scripts/precomp && sage verify_precomputed.py` to
//! independently verify these constants against a SageMath recomputation.
//!
//! TODO: Reimplement the extremal order computation in pure Rust so
//! we can cross-check the Sage and Rust computations in CI without
//! requiring a SageMath installation. Once both paths exist, CI should
//! run both and fail the build if they disagree.
//!
//! [§3.1.7.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.7.2
//! [Algorithm 3.12]: https://sqisign.org/spec/sqisign-20250707.pdf#algorithm.3.12

use super::{
    algebra::{Coordinate, Denominator, Element},
    bigint::BigInt,
    lattice::{ExtremalOrder, Lattice},
    linear::{Matrix, Vector},
};

/// Number of precomputed extremal orders.
pub const NUM_EXTREMAL_ORDERS: usize = 7;

/// The prime p = 5 · 2²⁴⁸ − 1 as a [`BigInt<4>`].
pub const P: BigInt<4> = BigInt::from_sign_and_limbs(
    0,
    [
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x04FFFFFFFFFFFFFF,
    ],
);

/// The prime p = 5 · 2²⁴⁸ − 1 as a [`BigInt<8>`] for arithmetic
/// that needs wider intermediates.
pub const P_WIDE: BigInt<8> = BigInt::from_sign_and_limbs(
    0,
    [
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x04FFFFFFFFFFFFFF,
        0,
        0,
        0,
        0,
    ],
);

/// The seven precomputed p-extremal maximal orders for NIST-I.
///
/// Each order has a distinguished element z with z² = −q (small q),
/// and an element t = j with nrd(t) = p. The q values are:
/// `[1, 5, 17, 37, 41, 53, 97]`.
///
/// `EXTREMAL_ORDERS[0]` is the standard order O₀ with basis
/// `{1, i, (i+j)/2, (1+k)/2}` and q = 1.
pub const EXTREMAL_ORDERS: [ExtremalOrder<4>; NUM_EXTREMAL_ORDERS] = [
    // Order 0 (q = 1): the standard order O₀.
    // Basis: {1, i, (i+j)/2, (1+k)/2}, z = i, t = j.
    ExtremalOrder::new(
        Lattice::new(
            Matrix::from_rows(
                Vector::new(
                    BigInt::from_u64(2),
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::from_u64(1),
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::from_u64(2),
                    BigInt::from_u64(1),
                    BigInt::ZERO,
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::from_u64(1),
                    BigInt::ZERO,
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::from_u64(1),
                ),
            ),
            BigInt::from_u64(2),
        ),
        Element::new(
            Coordinate::ZERO,
            Coordinate::from_bigint(BigInt::from_u64(2)),
            Coordinate::ZERO,
            Coordinate::ZERO,
            Denominator::TWO,
        ),
        Element::J,
        1,
    ),
    // Order 1 (q = 5).
    ExtremalOrder::new(
        Lattice::new(
            Matrix::from_rows(
                Vector::new(
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x0000000000000000,
                            0x1000000000000000,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x0000000000000000,
                            0x0800000000000000,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::MINUS_ONE,
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        1,
                        [
                            0x0000000000000000,
                            0x0000000000000000,
                            0x0000000000000000,
                            0x0080000000000000,
                        ],
                    ),
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x0000000000000000,
                            0x0800000000000000,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                ),
                Vector::new(BigInt::ZERO, BigInt::MINUS_ONE, BigInt::ZERO, BigInt::ZERO),
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    0x0000000000000000,
                    0x1000000000000000,
                    0x0000000000000000,
                    0x0000000000000000,
                ],
            ),
        ),
        Element::new(
            Coordinate::ZERO,
            Coordinate::from_sign_and_limbs(
                1,
                [
                    0x0000000000000001,
                    0x0000000000000000,
                    0x0000000000000000,
                    0x0000000000000000,
                ],
            ),
            Coordinate::ZERO,
            Coordinate::from_sign_and_limbs(
                1,
                [
                    0x0000000000000001,
                    0x0000000000000000,
                    0x0000000000000000,
                    0x0000000000000000,
                ],
            ),
            Denominator::from_limbs([
                0x0000000000000000,
                0x1000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ]),
        ),
        Element::J,
        5,
    ),
    // Order 2 (q = 17).
    ExtremalOrder::new(
        Lattice::new(
            Matrix::from_rows(
                Vector::new(
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0xF5F27A647B8578D4,
                            0xB8746101369629B9,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0xFAF93D323DC2BC6A,
                            0x5C3A30809B4B14DC,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x95AD2AD56FA47D47,
                            0xC89877E749BE8A4B,
                            0x0000000000000001,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x3E355E2970603F47,
                            0x78DD10AE2A1BD950,
                            0x0000000000000000,
                            0x0280000000000000,
                        ],
                    ),
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0xFAF93D323DC2BC6A,
                            0x5C3A30809B4B14DC,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x0000000000000011,
                            0x0000000000000000,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        1,
                        [
                            0xB19426E828EE3FE7,
                            0x0D6DE568AF586D7A,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                ),
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    0xF5F27A647B8578D4,
                    0xB8746101369629B9,
                    0x0000000000000000,
                    0x0000000000000000,
                ],
            ),
        ),
        Element::new(
            Coordinate::ZERO,
            Coordinate::from_sign_and_limbs(
                0,
                [
                    0x95AD2AD56FA47D47,
                    0xC89877E749BE8A4B,
                    0x0000000000000001,
                    0x0000000000000000,
                ],
            ),
            Coordinate::ZERO,
            Coordinate::from_sign_and_limbs(
                0,
                [
                    0x0000000000000011,
                    0x0000000000000000,
                    0x0000000000000000,
                    0x0000000000000000,
                ],
            ),
            Denominator::from_limbs([
                0xF5F27A647B8578D4,
                0xB8746101369629B9,
                0x0000000000000000,
                0x0000000000000000,
            ]),
        ),
        Element::J,
        17,
    ),
    // Order 3 (q = 37).
    ExtremalOrder::new(
        Lattice::new(
            Matrix::from_rows(
                Vector::new(
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x3C6FA8E67715E5E2,
                            0x17949BEC872B9078,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x1E37D4733B8AF2F1,
                            0x0BCA4DF64395C83C,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        1,
                        [
                            0xB034808274C8307A,
                            0x09AB399AC43A4E8A,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x3D25CA466BC9954F,
                            0x04F5822946ED431B,
                            0xEB3E45306EB3E453,
                            0x0045306EB3E45306,
                        ],
                    ),
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x1E37D4733B8AF2F1,
                            0x0BCA4DF64395C83C,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x0000000000000004,
                            0x0000000000000000,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0xBD312454CA3A0E7F,
                            0x002172F0CB4CE562,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                ),
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    0x3C6FA8E67715E5E2,
                    0x17949BEC872B9078,
                    0x0000000000000000,
                    0x0000000000000000,
                ],
            ),
        ),
        Element::new(
            Coordinate::ZERO,
            Coordinate::from_sign_and_limbs(
                1,
                [
                    0xB034808274C8307A,
                    0x09AB399AC43A4E8A,
                    0x0000000000000000,
                    0x0000000000000000,
                ],
            ),
            Coordinate::ZERO,
            Coordinate::from_sign_and_limbs(
                0,
                [
                    0x0000000000000004,
                    0x0000000000000000,
                    0x0000000000000000,
                    0x0000000000000000,
                ],
            ),
            Denominator::from_limbs([
                0x3C6FA8E67715E5E2,
                0x17949BEC872B9078,
                0x0000000000000000,
                0x0000000000000000,
            ]),
        ),
        Element::J,
        37,
    ),
    // Order 4 (q = 41).
    ExtremalOrder::new(
        Lattice::new(
            Matrix::from_rows(
                Vector::new(
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0xDE33C5116DEEAFA2,
                            0x2DF94F97C89EC8CE,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x6F19E288B6F757D1,
                            0x16FCA7CBE44F6467,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0xD17AA943DA6BDD36,
                            0x44D44B0C564CE307,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        1,
                        [
                            0xA0A2047CC4063A03,
                            0x6CEE07961DF46DBC,
                            0xC7CE0C7CE0C7CE0C,
                            0x007CE0C7CE0C7CE0,
                        ],
                    ),
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x6F19E288B6F757D1,
                            0x16FCA7CBE44F6467,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        1,
                        [
                            0x0000000000000008,
                            0x0000000000000000,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        1,
                        [
                            0xD9F82148A1E2188F,
                            0x00D6E1B21A072E79,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                ),
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    0xDE33C5116DEEAFA2,
                    0x2DF94F97C89EC8CE,
                    0x0000000000000000,
                    0x0000000000000000,
                ],
            ),
        ),
        Element::new(
            Coordinate::ZERO,
            Coordinate::from_sign_and_limbs(
                0,
                [
                    0xD17AA943DA6BDD36,
                    0x44D44B0C564CE307,
                    0x0000000000000000,
                    0x0000000000000000,
                ],
            ),
            Coordinate::ZERO,
            Coordinate::from_sign_and_limbs(
                1,
                [
                    0x0000000000000008,
                    0x0000000000000000,
                    0x0000000000000000,
                    0x0000000000000000,
                ],
            ),
            Denominator::from_limbs([
                0xDE33C5116DEEAFA2,
                0x2DF94F97C89EC8CE,
                0x0000000000000000,
                0x0000000000000000,
            ]),
        ),
        Element::J,
        41,
    ),
    // Order 5 (q = 53).
    ExtremalOrder::new(
        Lattice::new(
            Matrix::from_rows(
                Vector::new(
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x380014F2025B96A4,
                            0x7BBEAB7F79584E7C,
                            0x0000000000000001,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x1C000A79012DCB52,
                            0xBDDF55BFBCAC273E,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        1,
                        [
                            0x4BA119E7333973E3,
                            0xDBD0EE6227026EBC,
                            0x0000000000000007,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x09F01D923DD0CA33,
                            0x83F7E395AFE92F81,
                            0xFFFFFFFFFFFFFFFC,
                            0x027FFFFFFFFFFFFF,
                        ],
                    ),
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x1C000A79012DCB52,
                            0xBDDF55BFBCAC273E,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x0000000000000035,
                            0x0000000000000000,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x87F571C0F93CEB73,
                            0x12FAB9CBCB3C667A,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                ),
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    0x380014F2025B96A4,
                    0x7BBEAB7F79584E7C,
                    0x0000000000000001,
                    0x0000000000000000,
                ],
            ),
        ),
        Element::new(
            Coordinate::ZERO,
            Coordinate::from_sign_and_limbs(
                1,
                [
                    0x4BA119E7333973E3,
                    0xDBD0EE6227026EBC,
                    0x0000000000000007,
                    0x0000000000000000,
                ],
            ),
            Coordinate::ZERO,
            Coordinate::from_sign_and_limbs(
                0,
                [
                    0x0000000000000035,
                    0x0000000000000000,
                    0x0000000000000000,
                    0x0000000000000000,
                ],
            ),
            Denominator::from_limbs([
                0x380014F2025B96A4,
                0x7BBEAB7F79584E7C,
                0x0000000000000001,
                0x0000000000000000,
            ]),
        ),
        Element::J,
        53,
    ),
    // Order 6 (q = 97).
    ExtremalOrder::new(
        Lattice::new(
            Matrix::from_rows(
                Vector::new(
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0xE2B97B9E55AF7FFA,
                            0xC227F76B578CA7AF,
                            0x000000000000000F,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0xF15CBDCF2AD7BFFD,
                            0xE113FBB5ABC653D7,
                            0x0000000000000007,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        1,
                        [
                            0xA2EF1CE7F02B0D16,
                            0x066759632C56054B,
                            0x000000000000006F,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x84AC06EA9D3BF0AB,
                            0xD021882BDDE962E5,
                            0xFFFFFFFFFFFFFFE2,
                            0x13FFFFFFFFFFFFFF,
                        ],
                    ),
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0xF15CBDCF2AD7BFFD,
                            0xE113FBB5ABC653D7,
                            0x0000000000000007,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x0000000000000308,
                            0x0000000000000000,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x077013F15C4A1F37,
                            0x9281DA3156007183,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                ),
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    0xE2B97B9E55AF7FFA,
                    0xC227F76B578CA7AF,
                    0x000000000000000F,
                    0x0000000000000000,
                ],
            ),
        ),
        Element::new(
            Coordinate::ZERO,
            Coordinate::from_sign_and_limbs(
                1,
                [
                    0xA2EF1CE7F02B0D16,
                    0x066759632C56054B,
                    0x000000000000006F,
                    0x0000000000000000,
                ],
            ),
            Coordinate::ZERO,
            Coordinate::from_sign_and_limbs(
                0,
                [
                    0x0000000000000308,
                    0x0000000000000000,
                    0x0000000000000000,
                    0x0000000000000000,
                ],
            ),
            Denominator::from_limbs([
                0xE2B97B9E55AF7FFA,
                0xC227F76B578CA7AF,
                0x000000000000000F,
                0x0000000000000000,
            ]),
        ),
        Element::J,
        97,
    ),
];

/// The standard order O₀ (alias for `EXTREMAL_ORDERS[0]`).
pub const STANDARD_ORDER: &ExtremalOrder<4> = &EXTREMAL_ORDERS[0];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_order_q_is_one() {
        assert_eq!(STANDARD_ORDER.q(), 1);
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
        // For each extremal order, z² = -q. Element::mul uses BigInt<8>
        // internally, so this works even for large coordinates.
        for (idx, order) in EXTREMAL_ORDERS.iter().enumerate() {
            let q = order.q();
            let z = order.z();
            let z_sq = z.mul(z).normalized();

            // z² should be the scalar -q.
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

        // Parse the JSON output from the Sage script.
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

        // The q values are the primary cross-check: they confirm that
        // Sage independently computed the same set of valid primes from p.
        for (i, sage_order) in sage_orders.iter().enumerate() {
            let sage_q = sage_order["q"].as_u64().unwrap() as u32;
            let rust_q = EXTREMAL_ORDERS[i].q();
            assert_eq!(
                sage_q, rust_q,
                "order {i}: q mismatch (sage={sage_q}, rust={rust_q})"
            );
        }

        // TODO: Compare full basis matrices and z/t elements once we
        // have BigInt parsing from decimal strings and quaternion
        // arithmetic over BigInt<8> for intermediate products.
    }
}
