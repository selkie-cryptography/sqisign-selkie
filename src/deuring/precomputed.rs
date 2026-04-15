//! Precomputed endomorphism action matrices for the NIST-I parameter set.
//!
//! Six 2×2 matrices per curve (seven curves total), representing the
//! action of quaternion order basis elements on the torsion basis
//! E_t[2^f] where f = 248.
//!
//! Generated from the SQIsign reference implementation's
//! `endomorphism_action.c` for `lvl1`.

#![allow(unused)]

use super::endomorphism::ActionMatrix;
use crate::quaternions::bigint::BigInt;

pub mod torsion_basis {
    //! Torsion basis x-coordinates for all 7 extremal order curves.
    //!
    //! Each curve E_t has a canonical 2^f-torsion basis (P_t, Q_t)
    //! with f = 248. Stored as 32-byte little-endian plain integers
    //! (NOT Montgomery form). Use `Fp::from_bytes` to convert.
    //!
    //! Extracted from the C ref's `endomorphism_action.c` (pinned
    //! commit 91e9e464) and independently verified by Sage (see
    //! `scripts/precomp/verify_torsion_bases.sage`).

    use crate::fields::{fp::Fp, fp2::Fp2};

    /// Torsion basis and curve data for curve `t`.
    ///
    /// Returns `(px, qx, a)` where `px` and `qx` are the
    /// x-coordinates of P_t and Q_t, and `a` is the Montgomery
    /// coefficient A (with A_im = 0 for all curves except E₀
    /// which has A = 0).
    pub fn basis_for_curve(t: usize) -> Option<(Fp2, Fp2, Fp2)> {
        match t {
            0 => Some((e0_px(), e0_qx(), Fp2::ZERO)),
            1 => Some((e1_px(), e1_qx(), e1_a())),
            2 => Some((e2_px(), e2_qx(), e2_a())),
            3 => Some((e3_px(), e3_qx(), e3_a())),
            4 => Some((e4_px(), e4_qx(), e4_a())),
            5 => Some((e5_px(), e5_qx(), e5_a())),
            6 => Some((e6_px(), e6_qx(), e6_a())),
            _ => None,
        }
    }

    // ---- Curve 0 (E₀: y² = x³ + x, A = 0) ----

    /// P₀ x-coordinate.
    pub fn e0_px() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0x78, 0x00, 0xB4, 0xAE, 0x5E, 0xD9, 0x19, 0x21, 0x8B, 0xA7, 0xBF, 0x59, 0x1A, 0x99,
                0xBE, 0x44, 0xC4, 0x16, 0x62, 0xA6, 0xC3, 0x04, 0xCC, 0x83, 0x24, 0xB1, 0x82, 0xCA,
                0x7F, 0x87, 0x9B, 0x01,
            ]),
            b: Fp::from_bytes(&[
                0x75, 0xD2, 0xF9, 0xC3, 0x3D, 0x13, 0x04, 0x8E, 0x74, 0x92, 0x42, 0x51, 0xAE, 0xDD,
                0xCF, 0xB2, 0x2F, 0xE9, 0x67, 0x98, 0xAA, 0x0A, 0x15, 0x52, 0x42, 0xE0, 0xEA, 0x49,
                0xDB, 0x2A, 0x44, 0x04,
            ]),
        }
    }

    /// Q₀ x-coordinate.
    pub fn e0_qx() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0x1F, 0xEB, 0x93, 0x55, 0x2A, 0x25, 0x16, 0x7C, 0xF3, 0xE1, 0x4B, 0xA5, 0xF7, 0x78,
                0x86, 0x87, 0x1D, 0x04, 0x0D, 0x05, 0x17, 0x27, 0xDF, 0x9F, 0x71, 0x0B, 0x5C, 0x7D,
                0x47, 0xFD, 0x5F, 0x04,
            ]),
            b: Fp::from_bytes(&[
                0xEE, 0xAA, 0xCD, 0xA0, 0xB7, 0x28, 0xE2, 0xFD, 0xAE, 0xA5, 0x8E, 0x6F, 0x4B, 0x05,
                0xFF, 0x39, 0x9A, 0xB3, 0x76, 0x36, 0xFB, 0xA8, 0x65, 0x44, 0xDC, 0x73, 0x18, 0xDF,
                0xE9, 0xD4, 0x87, 0x04,
            ]),
        }
    }

    // ---- Curve 1 ----

    /// P_1 x-coordinate.
    pub fn e1_px() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0xD0, 0xBE, 0x12, 0x78, 0x9D, 0x0B, 0xEE, 0x6B, 0x82, 0xF2, 0x7C, 0x6B, 0x24, 0xAF,
                0x04, 0x49, 0xE4, 0x19, 0xDB, 0x20, 0xC4, 0xDA, 0x50, 0x2A, 0xC5, 0x73, 0xA6, 0x9C,
                0xD0, 0x5A, 0x25, 0x03,
            ]),
            b: Fp::from_bytes(&[
                0x4D, 0x14, 0xC8, 0x24, 0x7F, 0x45, 0x6A, 0x13, 0xCB, 0x35, 0xBA, 0x9D, 0xA8, 0x61,
                0xC6, 0x9C, 0xE4, 0x8D, 0xAB, 0x60, 0x1E, 0x1B, 0x7B, 0xF3, 0xFE, 0xB0, 0xC8, 0x09,
                0x67, 0x9A, 0x96, 0x03,
            ]),
        }
    }

    /// Q_1 x-coordinate.
    pub fn e1_qx() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0x0F, 0x06, 0xD1, 0xD0, 0x0C, 0x66, 0xC1, 0x38, 0x1E, 0x29, 0xF9, 0xCF, 0x4A, 0x64,
                0xCE, 0x87, 0xB5, 0xA7, 0x5F, 0x7B, 0x14, 0x5A, 0x4B, 0xA3, 0x44, 0xFF, 0xF5, 0x51,
                0x9C, 0x8C, 0x91, 0x02,
            ]),
            b: Fp::from_bytes(&[
                0x64, 0xE6, 0x03, 0x7E, 0x91, 0x7F, 0x3E, 0xF1, 0x59, 0xC1, 0x75, 0x28, 0xA2, 0x93,
                0x8D, 0x9A, 0xB2, 0xF1, 0x34, 0x0D, 0xAD, 0x42, 0xCC, 0x04, 0x5D, 0x97, 0x50, 0x08,
                0xD7, 0x93, 0x21, 0x01,
            ]),
        }
    }

    /// Montgomery coefficient A for curve 1.
    pub fn e1_a() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0xF8, 0x67, 0x64, 0x85, 0xAF, 0xAE, 0x2D, 0xD3, 0x4B, 0xD9, 0x8C, 0xB0, 0xE9, 0x53,
                0xAB, 0xC1, 0xA7, 0x56, 0xE2, 0x3D, 0xDA, 0x2F, 0x6B, 0xC0, 0xB2, 0xD2, 0x93, 0x16,
                0x56, 0xA8, 0x6F, 0x00,
            ]),
            b: Fp::ZERO,
        }
    }

    // ---- Curve 2 ----

    /// P_2 x-coordinate.
    pub fn e2_px() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0xC5, 0x23, 0x27, 0xBB, 0xA0, 0xFA, 0x22, 0xDC, 0x7E, 0x52, 0xAE, 0xD2, 0x7B, 0x7E,
                0x20, 0x53, 0x77, 0xA5, 0x5C, 0x69, 0x1D, 0x91, 0x6C, 0xC7, 0x52, 0xFF, 0x73, 0xD4,
                0x4D, 0x77, 0x6B, 0x01,
            ]),
            b: Fp::from_bytes(&[
                0x25, 0xEC, 0x95, 0x2D, 0x87, 0x67, 0x9A, 0x41, 0xF5, 0x10, 0x4C, 0x12, 0xA7, 0xDA,
                0xB8, 0x7D, 0x35, 0x61, 0x63, 0x2F, 0xDD, 0x9C, 0x8D, 0x1A, 0x60, 0xAA, 0xA7, 0xB7,
                0xB2, 0xC6, 0xCB, 0x03,
            ]),
        }
    }

    /// Q_2 x-coordinate.
    pub fn e2_qx() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0xA5, 0x1A, 0x18, 0x75, 0x8B, 0xD7, 0x08, 0x1E, 0x8C, 0xA0, 0x8B, 0x14, 0x55, 0xBA,
                0x19, 0x1D, 0x4B, 0xF1, 0x18, 0x71, 0xAA, 0x76, 0x68, 0x04, 0x6C, 0xC8, 0x0F, 0x79,
                0x53, 0x40, 0xCB, 0x04,
            ]),
            b: Fp::from_bytes(&[
                0x75, 0x44, 0xB7, 0x0D, 0x17, 0x89, 0x3D, 0x0F, 0x73, 0xBF, 0xE0, 0x2D, 0xB4, 0xC2,
                0x25, 0xE3, 0x0A, 0x96, 0x3C, 0x05, 0xEF, 0x46, 0x56, 0x82, 0x44, 0x29, 0x67, 0x18,
                0x2F, 0x3B, 0x68, 0x04,
            ]),
        }
    }

    /// Montgomery coefficient A for curve 2.
    pub fn e2_a() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0x2C, 0x17, 0x90, 0xAE, 0x2B, 0xB0, 0x24, 0x7C, 0xC0, 0x09, 0x5E, 0x81, 0x2A, 0x15,
                0xCA, 0x7D, 0xDE, 0xC1, 0x2C, 0x18, 0xC5, 0xF7, 0x15, 0xBF, 0x5E, 0x49, 0xCC, 0x1D,
                0x85, 0xA8, 0x2F, 0x01,
            ]),
            b: Fp::ZERO,
        }
    }

    // ---- Curve 3 ----

    /// P_3 x-coordinate.
    pub fn e3_px() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0x64, 0x5A, 0x70, 0x1E, 0x2C, 0x09, 0x07, 0x70, 0x7C, 0xD6, 0x1E, 0x9E, 0x7B, 0xB8,
                0x14, 0x52, 0x53, 0x5D, 0x2E, 0xC8, 0x6F, 0x51, 0xE8, 0x89, 0xBC, 0x2A, 0xF4, 0x5F,
                0x71, 0x61, 0xCC, 0x03,
            ]),
            b: Fp::from_bytes(&[
                0x34, 0xD6, 0xDD, 0xD1, 0xF5, 0x29, 0xE8, 0x37, 0x06, 0x98, 0x2A, 0xAB, 0xC6, 0xAF,
                0x1A, 0x55, 0x9C, 0x22, 0x11, 0x49, 0xA9, 0xB8, 0x35, 0x85, 0xFF, 0x53, 0xD3, 0x25,
                0xD7, 0x84, 0xAC, 0x01,
            ]),
        }
    }

    /// Q_3 x-coordinate.
    pub fn e3_qx() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0x42, 0x1C, 0xCE, 0xEE, 0x34, 0x4D, 0x05, 0x3B, 0x29, 0xAE, 0x18, 0xDD, 0x27, 0xE6,
                0x8B, 0x3A, 0x0F, 0x28, 0x4C, 0xC4, 0xD8, 0xD0, 0x4B, 0x10, 0xE3, 0xE5, 0xDB, 0x66,
                0xB3, 0x66, 0x74, 0x02,
            ]),
            b: Fp::from_bytes(&[
                0x27, 0x4D, 0xC8, 0xB2, 0x5E, 0x15, 0xCE, 0xB0, 0x95, 0x4D, 0x0C, 0x9D, 0xAB, 0x4A,
                0x5C, 0x95, 0xCE, 0x44, 0x54, 0x72, 0x0C, 0xEA, 0x6C, 0xA4, 0x08, 0x3A, 0xC8, 0x54,
                0xF0, 0xF0, 0x13, 0x04,
            ]),
        }
    }

    /// Montgomery coefficient A for curve 3.
    pub fn e3_a() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0x59, 0xF4, 0x23, 0x6A, 0x8E, 0x97, 0xA6, 0xB2, 0xC4, 0xA7, 0x0C, 0x6C, 0xDB, 0x7B,
                0xA5, 0xFB, 0x8B, 0xAF, 0xE2, 0x65, 0x47, 0x5B, 0x73, 0x58, 0xA6, 0x9A, 0x88, 0xA7,
                0xDA, 0x96, 0x53, 0x01,
            ]),
            b: Fp::ZERO,
        }
    }

    // ---- Curve 4 ----

    /// P_4 x-coordinate.
    pub fn e4_px() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0x2C, 0xDA, 0x94, 0x5D, 0x78, 0x17, 0x3F, 0x20, 0xF6, 0x07, 0x49, 0x24, 0xA4, 0x08,
                0x48, 0x44, 0x34, 0x81, 0x80, 0x8F, 0xE3, 0xAF, 0x26, 0x15, 0xA4, 0x3B, 0x33, 0x59,
                0x0C, 0xF4, 0xA4, 0x02,
            ]),
            b: Fp::from_bytes(&[
                0x11, 0x1D, 0x71, 0x47, 0x98, 0xC9, 0xBB, 0x4B, 0x18, 0x82, 0xFF, 0xD4, 0x24, 0x6F,
                0x8D, 0x47, 0xDF, 0x73, 0xB2, 0xD5, 0xC0, 0xCC, 0xA9, 0xBA, 0xF1, 0xB3, 0xAC, 0x66,
                0xD9, 0x2A, 0x63, 0x03,
            ]),
        }
    }

    /// Q_4 x-coordinate.
    pub fn e4_qx() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0xD1, 0xDC, 0x14, 0x28, 0xF3, 0x16, 0xC8, 0x2F, 0xB3, 0x95, 0xC1, 0x78, 0xA9, 0x53,
                0x4B, 0xDE, 0xAC, 0x22, 0x60, 0x8D, 0xFB, 0x68, 0xAA, 0xC5, 0xE3, 0x0C, 0x25, 0x44,
                0x45, 0x5D, 0x79, 0x04,
            ]),
            b: Fp::from_bytes(&[
                0x9B, 0x08, 0xE1, 0xAB, 0x9A, 0xFC, 0x88, 0x36, 0x96, 0x1E, 0x3A, 0x09, 0x69, 0xA4,
                0x3D, 0x45, 0x5E, 0x9E, 0xE5, 0x6D, 0x68, 0x9A, 0xAE, 0x28, 0x24, 0x37, 0xAD, 0xB8,
                0x12, 0x9E, 0xED, 0x02,
            ]),
        }
    }

    /// Montgomery coefficient A for curve 4.
    pub fn e4_a() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0xAD, 0xAF, 0xE7, 0xBA, 0xBC, 0x5C, 0xF2, 0x20, 0xFE, 0x67, 0x81, 0xBA, 0xDA, 0xA5,
                0x91, 0x8E, 0x08, 0xD7, 0xD6, 0xFB, 0x98, 0xBE, 0xCE, 0x1D, 0x37, 0x31, 0x4D, 0x98,
                0xEE, 0x5F, 0x49, 0x00,
            ]),
            b: Fp::ZERO,
        }
    }

    // ---- Curve 5 ----

    /// P_5 x-coordinate.
    pub fn e5_px() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0x53, 0xB2, 0xE0, 0x7E, 0x9B, 0xBD, 0x54, 0x39, 0x3F, 0x25, 0x7D, 0x96, 0x3A, 0x33,
                0x28, 0x3A, 0xF6, 0x6C, 0xB2, 0x42, 0x0C, 0x06, 0xDE, 0xB9, 0x06, 0xCD, 0xC7, 0x22,
                0x47, 0xDA, 0x91, 0x03,
            ]),
            b: Fp::from_bytes(&[
                0xD8, 0xB5, 0x25, 0x1C, 0x9E, 0xF1, 0xF0, 0x38, 0xA2, 0x86, 0xD0, 0x58, 0xFC, 0xF8,
                0x86, 0xAB, 0xC0, 0xCB, 0xA4, 0x11, 0x45, 0xBC, 0x52, 0xB0, 0x02, 0x6C, 0xD6, 0xB9,
                0x46, 0x1B, 0x91, 0x00,
            ]),
        }
    }

    /// Q_5 x-coordinate.
    pub fn e5_qx() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0x06, 0x31, 0xE9, 0x91, 0x62, 0x17, 0xAB, 0x4D, 0x09, 0x64, 0xB3, 0xD0, 0x67, 0xD4,
                0x80, 0x6F, 0x41, 0x33, 0xD7, 0x8C, 0x26, 0x41, 0x2E, 0xB5, 0x1B, 0xBC, 0x69, 0xF5,
                0xD5, 0x9D, 0x98, 0x03,
            ]),
            b: Fp::from_bytes(&[
                0xED, 0xA2, 0xB8, 0x8D, 0x4E, 0x35, 0xE6, 0x83, 0x11, 0x55, 0x57, 0x2B, 0x27, 0x61,
                0xED, 0xC2, 0xD9, 0x8B, 0x2E, 0x5A, 0x00, 0xB8, 0xEA, 0xC4, 0x71, 0x3B, 0x29, 0x53,
                0x85, 0xF8, 0x74, 0x02,
            ]),
        }
    }

    /// Montgomery coefficient A for curve 5.
    pub fn e5_a() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0xFA, 0xC4, 0xF0, 0x0C, 0xF0, 0x78, 0x6B, 0x94, 0x27, 0xF3, 0x09, 0xFF, 0x30, 0x10,
                0x10, 0xA6, 0x47, 0x44, 0x9D, 0xD0, 0x65, 0x12, 0x9D, 0xFF, 0xE2, 0xB3, 0x01, 0xCF,
                0x9E, 0x83, 0x82, 0x00,
            ]),
            b: Fp::ZERO,
        }
    }

    // ---- Curve 6 ----

    /// P_6 x-coordinate.
    pub fn e6_px() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0x26, 0xA7, 0x9F, 0x42, 0x7A, 0xBC, 0xD9, 0x00, 0xFD, 0x9D, 0x98, 0x3B, 0x7E, 0x3D,
                0x16, 0x0D, 0x5D, 0x1D, 0x5A, 0xC8, 0x82, 0xD3, 0xFA, 0xB2, 0x0D, 0x54, 0x3E, 0x66,
                0xB8, 0x95, 0x63, 0x02,
            ]),
            b: Fp::from_bytes(&[
                0x30, 0xAB, 0x68, 0x8D, 0x27, 0x42, 0x2C, 0xA6, 0x3A, 0x1C, 0x16, 0xAD, 0x00, 0xD0,
                0xD2, 0x0F, 0x14, 0xD3, 0x77, 0xB9, 0xA9, 0x99, 0x8A, 0x5F, 0x98, 0x5D, 0x7B, 0xCD,
                0x17, 0xDC, 0xB3, 0x02,
            ]),
        }
    }

    /// Q_6 x-coordinate.
    pub fn e6_qx() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0x20, 0xAC, 0x83, 0xC0, 0x88, 0x1C, 0x9C, 0x08, 0xE3, 0x68, 0x0D, 0x7B, 0xC1, 0x74,
                0xB5, 0xD8, 0x19, 0x1F, 0x42, 0xBC, 0x86, 0xD9, 0x08, 0x53, 0x55, 0xE5, 0x60, 0xC9,
                0x8F, 0x80, 0x8F, 0x00,
            ]),
            b: Fp::from_bytes(&[
                0xCB, 0xEB, 0x9B, 0xCC, 0x6B, 0x39, 0x9D, 0xFE, 0xF6, 0x4D, 0x6E, 0x02, 0xDB, 0x2F,
                0x9F, 0x7C, 0x93, 0x89, 0x42, 0x38, 0x8B, 0xED, 0x14, 0x1F, 0xFD, 0x9E, 0x1E, 0xC7,
                0x96, 0x95, 0xDD, 0x01,
            ]),
        }
    }

    /// Montgomery coefficient A for curve 6.
    pub fn e6_a() -> Fp2 {
        Fp2 {
            a: Fp::from_bytes(&[
                0xBD, 0x17, 0x91, 0xE1, 0xE5, 0x53, 0xE0, 0x9F, 0x43, 0x0B, 0x51, 0xB5, 0x93, 0x13,
                0xC6, 0x37, 0x44, 0x0D, 0x09, 0x7D, 0xB1, 0xAF, 0x17, 0xCF, 0xCE, 0xFF, 0x9F, 0x92,
                0xDD, 0xCB, 0xA2, 0x00,
            ]),
            b: Fp::ZERO,
        }
    }
}

/// Endomorphism action matrices for each curve.
///
/// Six 2×2 matrices per curve: action of `i`, `j`, `k`, `gen2`,
/// `gen3`, `gen4` on the torsion basis `E_t[2^f]`.
///
/// # Basis convention
///
/// These matrices encode the endomorphism in the standard
/// `(P, Q)` basis. For a matrix `M = [[m00, m01], [m10, m11]]`:
///
/// - Column 0: `θ(P) = [m00]·P + [m10]·Q`
/// - Column 1: `θ(Q) = [m01]·P + [m11]·Q`
///
/// Use with `eval_decomposition` on a `TorsionBasis(P, Q, P−Q)`.
///
/// Verified by `action_matrix_consistent_with_basis` (x-only
/// check against known endomorphism) and `action_matrix_scalar_three`
/// (decomposition of scalar elements produces the identity matrix).
pub const ACTION_MATRICES: [[ActionMatrix; 6]; 7] = [
    // Curve 0
    [
        ActionMatrix::from_limbs(
            [
                0xC5D3BDA21B5456DB,
                0x74759780861DDD06,
                0x7F9D34B241AF33D1,
                0x00CAB471AA8C7F8C,
            ],
            [
                0x7BFB7D32048B7D7A,
                0xA955918263D89BD3,
                0x76BF6861034403E1,
                0x00574AE3EEB45CD0,
            ],
            [
                0x856FD6493698444F,
                0x189CAFDF498F41DB,
                0xF7E00BFFE50BCB5B,
                0x001535DAA88B47F9,
            ],
            [
                0x3A2C425DE4ABA925,
                0x8B8A687F79E222F9,
                0x8062CB4DBE50CC2E,
                0x00354B8E55738073,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x36BAD5FD54900ABF,
                0x00D14EEA4A59DA0F,
                0x914606F6A7AEA3F0,
                0x007DA2D2CDE65004,
            ],
            [
                0x611DBDE3B7878680,
                0x0819C9EC8B68A95F,
                0xBD7B5E31F73E2361,
                0x0068240040D72B45,
            ],
            [
                0x1F0C9E126D204277,
                0x563F9D1CF854977F,
                0xE829AF54C2ED00DB,
                0x00CA7BE80D8304FB,
            ],
            [
                0xC9452A02AB6FF541,
                0xFF2EB115B5A625F0,
                0x6EB9F90958515C0F,
                0x00825D2D3219AFFB,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xB19C16401AF2231B,
                0xF39A683EE470F713,
                0x904EC26E7A543289,
                0x004455FC6A0CD5A6,
            ],
            [
                0x55D2DE69B685AD7A,
                0x925F591684E85675,
                0x83917C511CB68C0A,
                0x00CD96CE11D1FFCE,
            ],
            [
                0x959B1B9279BD3724,
                0x64A727D46F18B3EC,
                0x664BADE78C7E9B4B,
                0x00486A1DA287A6D9,
            ],
            [
                0x4E63E9BFE50DDCE5,
                0x0C6597C11B8F08EC,
                0x6FB13D9185ABCD76,
                0x00BBAA0395F32A59,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xC5D3BDA21B5456DB,
                0x74759780861DDD06,
                0x7F9D34B241AF33D1,
                0x00CAB471AA8C7F8C,
            ],
            [
                0x7BFB7D32048B7D7A,
                0xA955918263D89BD3,
                0x76BF6861034403E1,
                0x00574AE3EEB45CD0,
            ],
            [
                0x856FD6493698444F,
                0x189CAFDF498F41DB,
                0xF7E00BFFE50BCB5B,
                0x001535DAA88B47F9,
            ],
            [
                0x3A2C425DE4ABA925,
                0x8B8A687F79E222F9,
                0x8062CB4DBE50CC2E,
                0x00354B8E55738073,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xFE4749CFB7F230CD,
                0xBAA37335683BDB8A,
                0x88719DD474AEEBE0,
                0x00242BA23C3967C8,
            ],
            [
                0x6E8C9D8ADE0981FD,
                0x58B7ADB777A0A299,
                0x1A1D63497D4113A1,
                0x00DFB77217C5C40B,
            ],
            [
                0x523E3A2DD1DC4363,
                0x376E267E20F1ECAD,
                0xF004DDAA53FC661B,
                0x006FD8E15B07267A,
            ],
            [
                0x01B8B630480DCF33,
                0x455C8CCA97C42475,
                0x778E622B8B51141F,
                0x00DBD45DC3C69837,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xD8CE0B200D79118E,
                0xF9CD341F72387B89,
                0x482761373D2A1944,
                0x00222AFE35066AD3,
            ],
            [
                0xAAE96F34DB42D6BD,
                0x492FAC8B42742B3A,
                0x41C8BE288E5B4605,
                0x0066CB6708E8FFE7,
            ],
            [
                0x4ACD8DC93CDE9B92,
                0xB25393EA378C59F6,
                0xB325D6F3C63F4DA5,
                0x0024350ED143D36C,
            ],
            [
                0x2731F4DFF286EE73,
                0x0632CBE08DC78476,
                0xB7D89EC8C2D5E6BB,
                0x00DDD501CAF9952C,
            ],
        ),
    ],
    // Curve 1
    [
        ActionMatrix::from_limbs(
            [
                0xE4058CEBA8DCEF13,
                0x3BBE28ACFDA5E2F5,
                0x5F5CB0FFEE9141E5,
                0x0095EF671E331920,
            ],
            [
                0xB1B6FBCE9E936B6E,
                0x6BCD20AE14B880BB,
                0xCEB3C4A7FEFFB7F4,
                0x00E9E00365BFD874,
            ],
            [
                0x523646B6C98847FF,
                0x7D56D563EC049694,
                0xE1958B0AC48F6833,
                0x00DB58B2E957B64E,
            ],
            [
                0x1BFA7314572310ED,
                0xC441D753025A1D0A,
                0xA0A34F00116EBE1A,
                0x006A1098E1CCE6DF,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xB19C16401AF2231B,
                0xF39A683EE470F713,
                0x904EC26E7A543289,
                0x004455FC6A0CD5A6,
            ],
            [
                0x55D2DE69B685AD7A,
                0x925F591684E85675,
                0x83917C511CB68C0A,
                0x00CD96CE11D1FFCE,
            ],
            [
                0x959B1B9279BD3724,
                0x64A727D46F18B3EC,
                0x664BADE78C7E9B4B,
                0x00486A1DA287A6D9,
            ],
            [
                0x4E63E9BFE50DDCE5,
                0x0C6597C11B8F08EC,
                0x6FB13D9185ABCD76,
                0x00BBAA0395F32A59,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xE776F94C38F88D79,
                0x867742422D2E2BDF,
                0x8EE7A2E31736DDF0,
                0x00A4BB554BB152AC,
            ],
            [
                0xD49DC0B8E2806774,
                0x7A5DC53F25773B88,
                0x3ED5D6B24CFB3032,
                0x00FC85B1584C27B8,
            ],
            [
                0xADB9D25B25CFC139,
                0x4E7A8867AA20BD39,
                0xACFC412AA81F8B24,
                0x00201D50AB0CEE2D,
            ],
            [
                0x188906B3C7077287,
                0x7988BDBDD2D1D420,
                0x71185D1CE8C9220F,
                0x005B44AAB44EAD53,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xE4058CEBA8DCEF13,
                0x3BBE28ACFDA5E2F5,
                0x5F5CB0FFEE9141E5,
                0x0095EF671E331920,
            ],
            [
                0xB1B6FBCE9E936B6E,
                0x6BCD20AE14B880BB,
                0xCEB3C4A7FEFFB7F4,
                0x00E9E00365BFD874,
            ],
            [
                0x523646B6C98847FF,
                0x7D56D563EC049694,
                0xE1958B0AC48F6833,
                0x00DB58B2E957B64E,
            ],
            [
                0x1BFA7314572310ED,
                0xC441D753025A1D0A,
                0xA0A34F00116EBE1A,
                0x006A1098E1CCE6DF,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xD8CE0B200D79118E,
                0xF9CD341F72387B89,
                0x482761373D2A1944,
                0x00222AFE35066AD3,
            ],
            [
                0xAAE96F34DB42D6BD,
                0x492FAC8B42742B3A,
                0x41C8BE288E5B4605,
                0x0066CB6708E8FFE7,
            ],
            [
                0x4ACD8DC93CDE9B92,
                0xB25393EA378C59F6,
                0xB325D6F3C63F4DA5,
                0x0024350ED143D36C,
            ],
            [
                0x2731F4DFF286EE73,
                0x0632CBE08DC78476,
                0xB7D89EC8C2D5E6BB,
                0x00DDD501CAF9952C,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x994175298B307029,
                0x4553E3D77B3F2BE8,
                0xC80BB49C7BEF7065,
                0x00181ECE950CFA3E,
            ],
            [
                0x7C8285E892CEB399,
                0x64F18924B18686EB,
                0x419631655E9A0D93,
                0x003155D501585E79,
            ],
            [
                0xAA0C720929F8DA47,
                0x517C6E1939C9FC22,
                0x1EDC20FCCFA4C94E,
                0x00DF85F0396DE0D0,
            ],
            [
                0x66BE8AD674CF8FD7,
                0xBAAC1C2884C0D417,
                0x37F44B6384108F9A,
                0x00E7E1316AF305C1,
            ],
        ),
    ],
    // Curve 2
    [
        ActionMatrix::from_limbs(
            [
                0xE75D52B3A5945FF1,
                0xD9767D25D267DD09,
                0x10BF9AAEC1A80BC5,
                0x0070AE848DE3E894,
            ],
            [
                0xA6D796C0B9E011E6,
                0xF4C52F4404B6EE81,
                0xEBB65B93E75D4597,
                0x00163084C08E59C6,
            ],
            [
                0x479F60313463F41D,
                0x404E1E6B159F6FE7,
                0xAD84C1F788A8E302,
                0x00AB3D6631758B50,
            ],
            [
                0x18A2AD4C5A6BA00F,
                0x268982DA2D9822F6,
                0xEF4065513E57F43A,
                0x008F517B721C176B,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xB19C16401AF2231B,
                0xF39A683EE470F713,
                0x904EC26E7A543289,
                0x004455FC6A0CD5A6,
            ],
            [
                0x55D2DE69B685AD7A,
                0x925F591684E85675,
                0x83917C511CB68C0A,
                0x00CD96CE11D1FFCE,
            ],
            [
                0x959B1B9279BD3724,
                0x64A727D46F18B3EC,
                0x664BADE78C7E9B4B,
                0x00486A1DA287A6D9,
            ],
            [
                0x4E63E9BFE50DDCE5,
                0x0C6597C11B8F08EC,
                0x6FB13D9185ABCD76,
                0x00BBAA0395F32A59,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x56C4C7EF1FBEFFC3,
                0x1CED36AAFA5C2834,
                0x0E528890A31D9076,
                0x00C298BCEF6887D2,
            ],
            [
                0x8109B5C6D7404098,
                0xC0D081C23A8C0299,
                0x656A89969243E848,
                0x004CB9F56C998C87,
            ],
            [
                0xC5FE55B5FEED712B,
                0x7814177577A9E867,
                0x397386B173B14780,
                0x00B001FA7F0B797A,
            ],
            [
                0xA93B3810E041003D,
                0xE312C95505A3D7CB,
                0xF1AD776F5CE26F89,
                0x003D67431097782D,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xE75D52B3A5945FF1,
                0xD9767D25D267DD09,
                0x10BF9AAEC1A80BC5,
                0x0070AE848DE3E894,
            ],
            [
                0xA6D796C0B9E011E6,
                0xF4C52F4404B6EE81,
                0xEBB65B93E75D4597,
                0x00163084C08E59C6,
            ],
            [
                0x479F60313463F41D,
                0x404E1E6B159F6FE7,
                0xAD84C1F788A8E302,
                0x00AB3D6631758B50,
            ],
            [
                0x18A2AD4C5A6BA00F,
                0x268982DA2D9822F6,
                0xEF4065513E57F43A,
                0x008F517B721C176B,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xD8CE0B200D79118E,
                0xF9CD341F72387B89,
                0x482761373D2A1944,
                0x00222AFE35066AD3,
            ],
            [
                0xAAE96F34DB42D6BD,
                0x492FAC8B42742B3A,
                0x41C8BE288E5B4605,
                0x0066CB6708E8FFE7,
            ],
            [
                0x4ACD8DC93CDE9B92,
                0xB25393EA378C59F6,
                0xB325D6F3C63F4DA5,
                0x0024350ED143D36C,
            ],
            [
                0x2731F4DFF286EE73,
                0x0632CBE08DC78476,
                0xB7D89EC8C2D5E6BB,
                0x00DDD501CAF9952C,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x7E74CC3F1BD65D2B,
                0xD6D49F84FBA04FEA,
                0x4F4E68B188D142A1,
                0x002EB13BA60C13EC,
            ],
            [
                0x649AAA1694487B4F,
                0x8DF1D3FD3BC2E4B4,
                0xE8968CAA4078931E,
                0x007C166EBED5DEEC,
            ],
            [
                0x2B40D32D51AA101D,
                0xB323257AC5F807BA,
                0x2C3DDC8F20C59BDE,
                0x009917B954A64E7B,
            ],
            [
                0x818B33C0E429A2D5,
                0x292B607B045FB015,
                0xB0B1974E772EBD5E,
                0x00D14EC459F3EC13,
            ],
        ),
    ],
    // Curve 3
    [
        ActionMatrix::from_limbs(
            [
                0x415C44557ED2323F,
                0xCC1176EF42825876,
                0x340547291142BDAB,
                0x00C57C1F17791155,
            ],
            [
                0x8A694FACA958C9CE,
                0x8C191A17999731E1,
                0x8113C0EB68C7D118,
                0x003FC94EF8C862FD,
            ],
            [
                0x127C8EA0ED4741CB,
                0x3826CE8CF74C69D5,
                0xE695056B6BF33DA2,
                0x00D0581784ACC45C,
            ],
            [
                0xBEA3BBAA812DCDC1,
                0x33EE8910BD7DA789,
                0xCBFAB8D6EEBD4254,
                0x003A83E0E886EEAA,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x4E63E9BFE50DDCE5,
                0x0C6597C11B8F08EC,
                0x6FB13D9185ABCD76,
                0x00BBAA0395F32A59,
            ],
            [
                0xAA2D2196497A5286,
                0x6DA0A6E97B17A98A,
                0x7C6E83AEE34973F5,
                0x00326931EE2E0031,
            ],
            [
                0x6A64E46D8642C8DC,
                0x9B58D82B90E74C13,
                0x99B45218738164B4,
                0x00B795E25D785926,
            ],
            [
                0xB19C16401AF2231B,
                0xF39A683EE470F713,
                0x904EC26E7A543289,
                0x004455FC6A0CD5A6,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xB6E0C901BE7A7363,
                0xD659BC42779D6A56,
                0x923A12F438683476,
                0x0003D5F954126FA8,
            ],
            [
                0x014B5C144FD4EDB4,
                0x04BB9C11D6BEF702,
                0x5085B159DE259F10,
                0x001DC4D36F42B0A9,
            ],
            [
                0x6B01C7ED4974E873,
                0x20D6C641F3D4AFFB,
                0x451E9F012D69CA22,
                0x00DEB43BEF65FA05,
            ],
            [
                0x491F36FE41858C9D,
                0x29A643BD886295A9,
                0x6DC5ED0BC797CB89,
                0x00FC2A06ABED9057,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x415C44557ED2323F,
                0xCC1176EF42825876,
                0x340547291142BDAB,
                0x00C57C1F17791155,
            ],
            [
                0x8A694FACA958C9CE,
                0x8C191A17999731E1,
                0x8113C0EB68C7D118,
                0x003FC94EF8C862FD,
            ],
            [
                0x127C8EA0ED4741CB,
                0x3826CE8CF74C69D5,
                0xE695056B6BF33DA2,
                0x00D0581784ACC45C,
            ],
            [
                0xBEA3BBAA812DCDC1,
                0x33EE8910BD7DA789,
                0xCBFAB8D6EEBD4254,
                0x003A83E0E886EEAA,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x2731F4DFF286EE73,
                0x0632CBE08DC78476,
                0xB7D89EC8C2D5E6BB,
                0x00DDD501CAF9952C,
            ],
            [
                0x551690CB24BD2943,
                0xB6D05374BD8BD4C5,
                0xBE3741D771A4B9FA,
                0x00993498F7170018,
            ],
            [
                0xB5327236C321646E,
                0x4DAC6C15C873A609,
                0x4CDA290C39C0B25A,
                0x00DBCAF12EBC2C93,
            ],
            [
                0xD8CE0B200D79118E,
                0xF9CD341F72387B89,
                0x482761373D2A1944,
                0x00222AFE35066AD3,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x490C34A61036CA5B,
                0x1C171590771BC0ED,
                0xDB988054977E4855,
                0x00FE77221175B26F,
            ],
            [
                0xB0CBAE0A821CF543,
                0x2340D2BF5A80642D,
                0xDACE4E38E1CE0C8F,
                0x00059BC4807E3445,
            ],
            [
                0x39A70FD4C3DC6E85,
                0xB0FE0B83777B5158,
                0x53DC103F45B355DE,
                0x00012B18B6CB27E2,
            ],
            [
                0xB6F3CB59EFC935A5,
                0xE3E8EA6F88E43F12,
                0x24677FAB6881B7AA,
                0x000188DDEE8A4D90,
            ],
        ),
    ],
    // Curve 4
    [
        ActionMatrix::from_limbs(
            [
                0x206AB453D052900D,
                0xFB21C57931F2E61D,
                0xF9C1F38F02BBC870,
                0x00EB58D147F183AA,
            ],
            [
                0x9E04EBC3A5E8727E,
                0x8EA968E038D7F1EB,
                0x82C048EB83318F77,
                0x0054F2213583B0A3,
            ],
            [
                0x34E9C9B349E4DBA9,
                0xCFB0CAE0D8767AB9,
                0x1E302C9826B36177,
                0x00713BDC53CC4A38,
            ],
            [
                0xDF954BAC2FAD6FF3,
                0x04DE3A86CE0D19E2,
                0x063E0C70FD44378F,
                0x0014A72EB80E7C55,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xB19C16401AF2231B,
                0xF39A683EE470F713,
                0x904EC26E7A543289,
                0x004455FC6A0CD5A6,
            ],
            [
                0x55D2DE69B685AD7A,
                0x925F591684E85675,
                0x83917C511CB68C0A,
                0x00CD96CE11D1FFCE,
            ],
            [
                0x959B1B9279BD3724,
                0x64A727D46F18B3EC,
                0x664BADE78C7E9B4B,
                0x00486A1DA287A6D9,
            ],
            [
                0x4E63E9BFE50DDCE5,
                0x0C6597C11B8F08EC,
                0x6FB13D9185ABCD76,
                0x00BBAA0395F32A59,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x89600CFE1B002417,
                0x222CC00F42D2662E,
                0xBCAC863F278B7671,
                0x001B4C5A6E5EDB9C,
            ],
            [
                0xEF865A2DD92B21E8,
                0x6B378AE01483F492,
                0xF4EC69C57B907F78,
                0x00F8829616602FB9,
            ],
            [
                0xBC0E9AEA5DC538FF,
                0xC311447B775DBEA5,
                0x162A15FDB63AF01C,
                0x00C52A0D9DEFAB76,
            ],
            [
                0x769FF301E4FFDBE9,
                0xDDD33FF0BD2D99D1,
                0x435379C0D874898E,
                0x00E4B3A591A12463,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x206AB453D052900D,
                0xFB21C57931F2E61D,
                0xF9C1F38F02BBC870,
                0x00EB58D147F183AA,
            ],
            [
                0x9E04EBC3A5E8727E,
                0x8EA968E038D7F1EB,
                0x82C048EB83318F77,
                0x0054F2213583B0A3,
            ],
            [
                0x34E9C9B349E4DBA9,
                0xCFB0CAE0D8767AB9,
                0x1E302C9826B36177,
                0x00713BDC53CC4A38,
            ],
            [
                0xDF954BAC2FAD6FF3,
                0x04DE3A86CE0D19E2,
                0x063E0C70FD44378F,
                0x0014A72EB80E7C55,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xD8CE0B200D79118E,
                0xF9CD341F72387B89,
                0x482761373D2A1944,
                0x00222AFE35066AD3,
            ],
            [
                0xAAE96F34DB42D6BD,
                0x492FAC8B42742B3A,
                0x41C8BE288E5B4605,
                0x0066CB6708E8FFE7,
            ],
            [
                0x4ACD8DC93CDE9B92,
                0xB25393EA378C59F6,
                0xB325D6F3C63F4DA5,
                0x0024350ED143D36C,
            ],
            [
                0x2731F4DFF286EE73,
                0x0632CBE08DC78476,
                0xB7D89EC8C2D5E6BB,
                0x00DDD501CAF9952C,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x731AACBF269320F0,
                0xF8361BCDD8CEB0F3,
                0xD3AAD60444D58469,
                0x003F9086CDC34AA8,
            ],
            [
                0x9534932FE55ACC11,
                0x614F2956AF432895,
                0x025560C6E4B24E84,
                0x0013C61ED70DBB14,
            ],
            [
                0x54E4D24F450D28D6,
                0xADBE9B71081D6E67,
                0x4B684EBF61481088,
                0x00E5E1A665C8829E,
            ],
            [
                0x8CE55340D96CDF10,
                0x07C9E43227314F0C,
                0x2C5529FBBB2A7B96,
                0x00C06F79323CB557,
            ],
        ),
    ],
    // Curve 5
    [
        ActionMatrix::from_limbs(
            [
                0xCD7513E0493127CB,
                0x9FF95A913DE76846,
                0xB97226ECA6D6A270,
                0x003F52FCE4B80B44,
            ],
            [
                0x1D16CA745A382D7E,
                0xAFC28C2916742547,
                0x79572C7348562349,
                0x00AD04D33C3E67E1,
            ],
            [
                0xCF15321E27FBD8D7,
                0x7ED75FBD6F8EFBD3,
                0xB73C593758D6F394,
                0x002264ACE0270BFB,
            ],
            [
                0x328AEC1FB6CED835,
                0x6006A56EC21897B9,
                0x468DD91359295D8F,
                0x00C0AD031B47F4BB,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x4E63E9BFE50DDCE5,
                0x0C6597C11B8F08EC,
                0x6FB13D9185ABCD76,
                0x00BBAA0395F32A59,
            ],
            [
                0xAA2D2196497A5286,
                0x6DA0A6E97B17A98A,
                0x7C6E83AEE34973F5,
                0x00326931EE2E0031,
            ],
            [
                0x6A64E46D8642C8DC,
                0x9B58D82B90E74C13,
                0x99B45218738164B4,
                0x00B795E25D785926,
            ],
            [
                0xB19C16401AF2231B,
                0xF39A683EE470F713,
                0x904EC26E7A543289,
                0x004455FC6A0CD5A6,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xEBB2CD7F6DC794DF,
                0x0C882825811DB290,
                0xC8C37D64959AD514,
                0x00321EEA106A16A9,
            ],
            [
                0x19FE1464E778E08C,
                0x4AF0A98F1D24EF25,
                0x95EA2B828E4D70D3,
                0x000F1695C2673277,
            ],
            [
                0x4BC9E2A4B6E1F1DF,
                0x9A383FCA6365DC85,
                0x1984CA7FED030EE2,
                0x008BC1731EFEE709,
            ],
            [
                0x144D328092386B21,
                0xF377D7DA7EE24D6F,
                0x373C829B6A652AEB,
                0x00CDE115EF95E956,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xCD7513E0493127CB,
                0x9FF95A913DE76846,
                0xB97226ECA6D6A270,
                0x003F52FCE4B80B44,
            ],
            [
                0x1D16CA745A382D7E,
                0xAFC28C2916742547,
                0x79572C7348562349,
                0x00AD04D33C3E67E1,
            ],
            [
                0xCF15321E27FBD8D7,
                0x7ED75FBD6F8EFBD3,
                0xB73C593758D6F394,
                0x002264ACE0270BFB,
            ],
            [
                0x328AEC1FB6CED835,
                0x6006A56EC21897B9,
                0x468DD91359295D8F,
                0x00C0AD031B47F4BB,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x2731F4DFF286EE73,
                0x0632CBE08DC78476,
                0xB7D89EC8C2D5E6BB,
                0x00DDD501CAF9952C,
            ],
            [
                0x551690CB24BD2943,
                0xB6D05374BD8BD4C5,
                0xBE3741D771A4B9FA,
                0x00993498F7170018,
            ],
            [
                0xB5327236C321646E,
                0x4DAC6C15C873A609,
                0x4CDA290C39C0B25A,
                0x00DBCAF12EBC2C93,
            ],
            [
                0xD8CE0B200D79118E,
                0xF9CD341F72387B89,
                0x482761373D2A1944,
                0x00222AFE35066AD3,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x01EE9A31F187D647,
                0x9418BFEDEEC2193B,
                0xAE854C740E9A15B6,
                0x00423F1226773EBE,
            ],
            [
                0x7BBF416C99BE68FF,
                0xA8FF7682609FD44F,
                0xC0768E77EC03A6BB,
                0x00F5A15296767873,
            ],
            [
                0xA38EBD23F7DA3739,
                0x01A76B0E76908CF9,
                0x51015AC7A2BD77F0,
                0x00952D3AA9223AAE,
            ],
            [
                0xFE1165CE0E7829B9,
                0x6BE74012113DE6C4,
                0x517AB38BF165EA49,
                0x00BDC0EDD988C141,
            ],
        ),
    ],
    // Curve 6
    [
        ActionMatrix::from_limbs(
            [
                0xC57273DEB1867177,
                0xFE177031C0EE9802,
                0xED41E2A741C5BC2E,
                0x001EF5BC9FF91CBF,
            ],
            [
                0x75C232B6BAE3726A,
                0x382AD1726E79E003,
                0x6A39A56379628A51,
                0x00A0F6F0C9109CDD,
            ],
            [
                0xBD8754E69FA9246B,
                0x25FA64701C7015B5,
                0x7EB5A6E989403F5C,
                0x0016A8DF54A16109,
            ],
            [
                0x3A8D8C214E798E89,
                0x01E88FCE3F1167FD,
                0x12BE1D58BE3A43D1,
                0x00E10A436006E340,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xB19C16401AF2231B,
                0xF39A683EE470F713,
                0x904EC26E7A543289,
                0x004455FC6A0CD5A6,
            ],
            [
                0x55D2DE69B685AD7A,
                0x925F591684E85675,
                0x83917C511CB68C0A,
                0x00CD96CE11D1FFCE,
            ],
            [
                0x959B1B9279BD3724,
                0x64A727D46F18B3EC,
                0x664BADE78C7E9B4B,
                0x00486A1DA287A6D9,
            ],
            [
                0x4E63E9BFE50DDCE5,
                0x0C6597C11B8F08EC,
                0x6FB13D9185ABCD76,
                0x00BBAA0395F32A59,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0x9CBE086C2B021975,
                0x737ED9A7B1C37576,
                0xF9BF7652A2454DE1,
                0x0008EAA1DC2C4BF8,
            ],
            [
                0x337C717746BCEE88,
                0x3366B65740DC92B6,
                0x114640EB2B986C8A,
                0x00E3A22FB00AE116,
            ],
            [
                0x40B9E24864D3F28D,
                0xCF3582EA82BB5141,
                0x6E88D71F0003FAF0,
                0x00CC6B9EF4C97AC9,
            ],
            [
                0x6341F793D4FDE68B,
                0x8C8126584E3C8A89,
                0x064089AD5DBAB21E,
                0x00F7155E23D3B407,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xC57273DEB1867177,
                0xFE177031C0EE9802,
                0xED41E2A741C5BC2E,
                0x001EF5BC9FF91CBF,
            ],
            [
                0x75C232B6BAE3726A,
                0x382AD1726E79E003,
                0x6A39A56379628A51,
                0x00A0F6F0C9109CDD,
            ],
            [
                0xBD8754E69FA9246B,
                0x25FA64701C7015B5,
                0x7EB5A6E989403F5C,
                0x0016A8DF54A16109,
            ],
            [
                0x3A8D8C214E798E89,
                0x01E88FCE3F1167FD,
                0x12BE1D58BE3A43D1,
                0x00E10A436006E340,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xD8CE0B200D79118E,
                0xF9CD341F72387B89,
                0x482761373D2A1944,
                0x00222AFE35066AD3,
            ],
            [
                0xAAE96F34DB42D6BD,
                0x492FAC8B42742B3A,
                0x41C8BE288E5B4605,
                0x0066CB6708E8FFE7,
            ],
            [
                0x4ACD8DC93CDE9B92,
                0xB25393EA378C59F6,
                0xB325D6F3C63F4DA5,
                0x0024350ED143D36C,
            ],
            [
                0x2731F4DFF286EE73,
                0x0632CBE08DC78476,
                0xB7D89EC8C2D5E6BB,
                0x00DDD501CAF9952C,
            ],
        ),
        ActionMatrix::from_limbs(
            [
                0xC440BCC48AD184A0,
                0x784E15C646EA94E1,
                0x2BEE0630D26F0190,
                0x003CE06193CE74B1,
            ],
            [
                0xA69BFA45DAFE1E2B,
                0xA0F9927DF670B77E,
                0x5F229607C897CCB5,
                0x00D9F781086747BF,
            ],
            [
                0x19BD02ADBD3F2AA2,
                0xD17E3FFF3B95E6F0,
                0x3B21DA467888F3A6,
                0x00C43E505301CC57,
            ],
            [
                0x3BBF433B752E7B60,
                0x87B1EA39B9156B1E,
                0xD411F9CF2D90FE6F,
                0x00C31F9E6C318B4E,
            ],
        ),
    ],
];
