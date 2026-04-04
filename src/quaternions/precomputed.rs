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
        0xffffffffffffffff,
        0xffffffffffffffff,
        0xffffffffffffffff,
        0x04ffffffffffffff,
    ],
);

/// The prime p = 5 · 2²⁴⁸ − 1 as a [`BigInt<8>`] for arithmetic
/// that needs wider intermediates.
pub const P_WIDE: BigInt<8> = BigInt::from_sign_and_limbs(
    0,
    [
        0xffffffffffffffff,
        0xffffffffffffffff,
        0xffffffffffffffff,
        0x04ffffffffffffff,
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
                            0xf5f27a647b8578d4,
                            0xb8746101369629b9,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0xfaf93d323dc2bc6a,
                            0x5c3a30809b4b14dc,
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
                            0x95ad2ad56fa47d47,
                            0xc89877e749be8a4b,
                            0x0000000000000001,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x3e355e2970603f47,
                            0x78dd10ae2a1bd950,
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
                            0xfaf93d323dc2bc6a,
                            0x5c3a30809b4b14dc,
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
                            0xb19426e828ee3fe7,
                            0x0d6de568af586d7a,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                ),
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    0xf5f27a647b8578d4,
                    0xb8746101369629b9,
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
                    0x95ad2ad56fa47d47,
                    0xc89877e749be8a4b,
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
                0xf5f27a647b8578d4,
                0xb8746101369629b9,
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
                            0x3c6fa8e67715e5e2,
                            0x17949bec872b9078,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x1e37d4733b8af2f1,
                            0x0bca4df64395c83c,
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
                            0xb034808274c8307a,
                            0x09ab399ac43a4e8a,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x3d25ca466bc9954f,
                            0x04f5822946ed431b,
                            0xeb3e45306eb3e453,
                            0x0045306eb3e45306,
                        ],
                    ),
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x1e37d4733b8af2f1,
                            0x0bca4df64395c83c,
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
                            0xbd312454ca3a0e7f,
                            0x002172f0cb4ce562,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                ),
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    0x3c6fa8e67715e5e2,
                    0x17949bec872b9078,
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
                    0xb034808274c8307a,
                    0x09ab399ac43a4e8a,
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
                0x3c6fa8e67715e5e2,
                0x17949bec872b9078,
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
                            0xde33c5116deeafa2,
                            0x2df94f97c89ec8ce,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x6f19e288b6f757d1,
                            0x16fca7cbe44f6467,
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
                            0xd17aa943da6bdd36,
                            0x44d44b0c564ce307,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        1,
                        [
                            0xa0a2047cc4063a03,
                            0x6cee07961df46dbc,
                            0xc7ce0c7ce0c7ce0c,
                            0x007ce0c7ce0c7ce0,
                        ],
                    ),
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x6f19e288b6f757d1,
                            0x16fca7cbe44f6467,
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
                            0xd9f82148a1e2188f,
                            0x00d6e1b21a072e79,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                ),
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    0xde33c5116deeafa2,
                    0x2df94f97c89ec8ce,
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
                    0xd17aa943da6bdd36,
                    0x44d44b0c564ce307,
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
                0xde33c5116deeafa2,
                0x2df94f97c89ec8ce,
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
                            0x380014f2025b96a4,
                            0x7bbeab7f79584e7c,
                            0x0000000000000001,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x1c000a79012dcb52,
                            0xbddf55bfbcac273e,
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
                            0x4ba119e7333973e3,
                            0xdbd0ee6227026ebc,
                            0x0000000000000007,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x09f01d923dd0ca33,
                            0x83f7e395afe92f81,
                            0xfffffffffffffffc,
                            0x027fffffffffffff,
                        ],
                    ),
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x1c000a79012dcb52,
                            0xbddf55bfbcac273e,
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
                            0x87f571c0f93ceb73,
                            0x12fab9cbcb3c667a,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                ),
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    0x380014f2025b96a4,
                    0x7bbeab7f79584e7c,
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
                    0x4ba119e7333973e3,
                    0xdbd0ee6227026ebc,
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
                0x380014f2025b96a4,
                0x7bbeab7f79584e7c,
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
                            0xe2b97b9e55af7ffa,
                            0xc227f76b578ca7af,
                            0x000000000000000f,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0xf15cbdcf2ad7bffd,
                            0xe113fbb5abc653d7,
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
                            0xa2ef1ce7f02b0d16,
                            0x066759632c56054b,
                            0x000000000000006f,
                            0x0000000000000000,
                        ],
                    ),
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0x84ac06ea9d3bf0ab,
                            0xd021882bdde962e5,
                            0xffffffffffffffe2,
                            0x13ffffffffffffff,
                        ],
                    ),
                ),
                Vector::new(
                    BigInt::ZERO,
                    BigInt::ZERO,
                    BigInt::from_sign_and_limbs(
                        0,
                        [
                            0xf15cbdcf2ad7bffd,
                            0xe113fbb5abc653d7,
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
                            0x077013f15c4a1f37,
                            0x9281da3156007183,
                            0x0000000000000000,
                            0x0000000000000000,
                        ],
                    ),
                ),
            ),
            BigInt::from_sign_and_limbs(
                0,
                [
                    0xe2b97b9e55af7ffa,
                    0xc227f76b578ca7af,
                    0x000000000000000f,
                    0x0000000000000000,
                ],
            ),
        ),
        Element::new(
            Coordinate::ZERO,
            Coordinate::from_sign_and_limbs(
                1,
                [
                    0xa2ef1ce7f02b0d16,
                    0x066759632c56054b,
                    0x000000000000006f,
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
                0xe2b97b9e55af7ffa,
                0xc227f76b578ca7af,
                0x000000000000000f,
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
