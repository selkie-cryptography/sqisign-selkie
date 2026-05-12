//! Precomputed endomorphism action matrices for the NIST-I parameter set.
//!
//! Six 2×2 matrices per curve (seven curves total), representing the
//! action of quaternion order basis elements on the torsion basis
//! E_t[2^f] where f = 248.
//!
//! Generated from the SQIsign reference implementation's
//! `endomorphism_action.c` for `lvl1`.

use super::endomorphism::ActionMatrix;

pub mod torsion_basis {
    //! Torsion basis x-coordinates for all 7 extremal order curves.
    //!
    //! Each curve E_t has a canonical 2^f-torsion basis (P_t, Q_t)
    //! with f = 248. Stored as radix-51 Montgomery-form limbs, so the
    //! constants are usable in `const` contexts and agree byte-for-byte
    //! with the `BASIS_E0_*` items in [`crate::params`].
    //!
    //! Extracted from the C reference's `endomorphism_action.c`
    //! (pinned commit 91e9e464) and independently verified by Sage
    //! (see `scripts/precomp/verify_torsion_bases.sage`). The plain
    //! integer form lives in `scripts/precomp/torsion_bases.json`;
    //! `scripts/precomp/gen_torsion_consts.py` converts each
    //! coordinate `v` to `(v · R) mod p` with R = 2^255 and splits it
    //! into 5 × 51-bit limbs.
    //!
    //! The `PmQ` coordinate for every curve uses the SAME projective
    //! representative as the C reference's
    //! `CURVES_WITH_ENDOMORPHISMS[t].basis_even.PmQ`. Deriving it
    //! instead via `projective_difference(P, Q)` produces a different
    //! representative that flips the Okeya-Sakurai y-sign and breaks
    //! the downstream biladder.

    use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

    use super::{ACTION_MATRICES, ActionMatrix};
    use crate::fields::{fp::Fp, fp2::Fp2};

    /// Index of one of the seven extremal-order curves.
    ///
    /// Values beyond `0..7` are unrepresentable, so callers cannot
    /// pass a curve number that doesn't have precomputed data. Use
    /// [`TryFrom<usize>`] to construct at the boundary with an
    /// unbounded index (e.g. the result of `Iterator::position`).
    ///
    /// Discriminants match the `EXTREMAL_ORDERS` / `ACTION_MATRICES`
    /// array indexing; [`ExtremalCurve::as_index`] exposes the
    /// `usize` form for those lookups.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[repr(u8)]
    pub enum ExtremalCurve {
        /// E₀: y² = x³ + x (the base curve, A = 0).
        E0 = 0,
        /// The first alternate extremal-order curve.
        E1 = 1,
        /// The second alternate extremal-order curve.
        E2 = 2,
        /// The third alternate extremal-order curve.
        E3 = 3,
        /// The fourth alternate extremal-order curve.
        E4 = 4,
        /// The fifth alternate extremal-order curve.
        E5 = 5,
        /// The sixth alternate extremal-order curve.
        E6 = 6,
    }

    impl ExtremalCurve {
        /// Every extremal curve in canonical order, for iteration.
        pub const ALL: [Self; 7] = [
            Self::E0,
            Self::E1,
            Self::E2,
            Self::E3,
            Self::E4,
            Self::E5,
            Self::E6,
        ];

        /// Position in the `EXTREMAL_ORDERS` / `ACTION_MATRICES`
        /// arrays.
        pub const fn as_index(self) -> usize {
            self as usize
        }

        /// Torsion basis and Montgomery coefficient.
        ///
        /// Returns `(px, qx, pmq_x, a)` where `px`, `qx`, and `pmq_x`
        /// are the x-coordinates of `P_t`, `Q_t`, and `P_t − Q_t`, and
        /// `a` is the Montgomery coefficient A (with `A_im = 0` for
        /// all curves, and `a = 0` for [`Self::E0`]).
        ///
        /// # Constant-time
        ///
        /// Constant-time on `self`. During signing, `self` is derived
        /// from the secret right order of the response ideal
        /// (Algorithm 4.2, line 7 `I_sig_response → right_order`), so
        /// this method must not branch on it. Implemented as a
        /// linear-scan `conditional_select` over all seven
        /// candidates: every invocation touches every row and the
        /// discriminant is used only through `Choice`-valued
        /// comparisons.
        pub fn basis(self) -> (Fp2, Fp2, Fp2, Fp2) {
            const BASES: [(Fp2, Fp2, Fp2, Fp2); 7] = [
                (E0_P_X, E0_Q_X, E0_PMQ_X, Fp2::ZERO),
                (E1_P_X, E1_Q_X, E1_PMQ_X, E1_A),
                (E2_P_X, E2_Q_X, E2_PMQ_X, E2_A),
                (E3_P_X, E3_Q_X, E3_PMQ_X, E3_A),
                (E4_P_X, E4_Q_X, E4_PMQ_X, E4_A),
                (E5_P_X, E5_Q_X, E5_PMQ_X, E5_A),
                (E6_P_X, E6_Q_X, E6_PMQ_X, E6_A),
            ];
            let idx = self as u8;
            let (mut px, mut qx, mut pmq, mut a) = BASES[0];
            // Skip i=0 since it seeds the accumulator.
            for (i, (p, q, pm, aa)) in BASES.iter().enumerate().skip(1) {
                let matches: Choice = (i as u8).ct_eq(&idx);
                px = Fp2::conditional_select(&px, p, matches);
                qx = Fp2::conditional_select(&qx, q, matches);
                pmq = Fp2::conditional_select(&pmq, pm, matches);
                a = Fp2::conditional_select(&a, aa, matches);
            }
            (px, qx, pmq, a)
        }

        /// The three generator action matrices for this curve:
        /// `ACTION_MATRICES[self.as_index()][3..6]`.
        ///
        /// # Constant-time
        ///
        /// Constant-time on `self`. Same CT contract as
        /// [`Self::basis`]: linear-scan `conditional_select` over all
        /// seven curves, so the index is never used to dereference
        /// secret-dependent memory.
        pub fn gen_matrices(self) -> [ActionMatrix; 3] {
            let idx = self as u8;
            let seed = &ACTION_MATRICES[0];
            let mut out = [seed[3], seed[4], seed[5]];
            for (i, row) in ACTION_MATRICES.iter().enumerate().skip(1) {
                let matches: Choice = (i as u8).ct_eq(&idx);
                out[0] = ActionMatrix::conditional_select(&out[0], &row[3], matches);
                out[1] = ActionMatrix::conditional_select(&out[1], &row[4], matches);
                out[2] = ActionMatrix::conditional_select(&out[2], &row[5], matches);
            }
            out
        }
    }

    /// Keep the enum in lockstep with `EXTREMAL_ORDERS`. Fires at
    /// monomorphization if either side grows or shrinks without the
    /// other.
    const _: () = assert!(
        crate::quaternions::precomputed::NUM_EXTREMAL_ORDERS == ExtremalCurve::ALL.len(),
        "ExtremalCurve variants must match NUM_EXTREMAL_ORDERS",
    );

    /// Reported by [`ExtremalCurve::try_from`] when the candidate
    /// index is outside `0..7`. Carries the bad value so diagnostics
    /// don't lose it.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct InvalidCurveIndex(pub usize);

    impl core::fmt::Display for InvalidCurveIndex {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "invalid extremal-curve index: {} (must be 0..7)", self.0)
        }
    }

    impl core::error::Error for InvalidCurveIndex {}

    impl TryFrom<usize> for ExtremalCurve {
        type Error = InvalidCurveIndex;

        fn try_from(t: usize) -> Result<Self, Self::Error> {
            match t {
                0 => Ok(Self::E0),
                1 => Ok(Self::E1),
                2 => Ok(Self::E2),
                3 => Ok(Self::E3),
                4 => Ok(Self::E4),
                5 => Ok(Self::E5),
                6 => Ok(Self::E6),
                _ => Err(InvalidCurveIndex(t)),
            }
        }
    }

    // ---- Curve 0 (E₀: y² = x³ + x, A = 0) ----

    /// P_0 x-coordinate.
    pub(crate) const E0_P_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x5BCAB12000C08,
            0x452654B56D052,
            0x26F81B5190A0A,
            0x36CFD66A361EB,
            0x012726610D11B,
        ]),
        Fp::from_limbs([
            0x6B96065C83EFC,
            0x29DA1D4A82CD9,
            0x190797AB98BDF,
            0x6841AA6EEEE05,
            0x01377C5431166,
        ]),
    );

    /// Q_0 x-coordinate.
    pub(crate) const E0_Q_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x21DD55B97832F,
            0x210F2D30B26AD,
            0x0680BCFCF6396,
            0x27B318EC126A7,
            0x04FFBA5956012,
        ]),
        Fp::from_limbs([
            0x74590149117E3,
            0x4982EDEFCC606,
            0x2AE3DB0CC6884,
            0x7D0384872F5EC,
            0x04FBB0FCB5A52,
        ]),
    );

    /// P_0 − Q_0 x-coordinate (for consistent Okeya-Sakurai lift).
    pub(crate) const E0_PMQ_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x0F6001DAFB71A,
            0x75CB70989457F,
            0x5F2AB120F726C,
            0x7D12027E55817,
            0x006482FE24949,
        ]),
        Fp::from_limbs([
            0x63A39AF1D2179,
            0x1C2884B0237F3,
            0x675979F836736,
            0x11DE56EF443D1,
            0x0462333FA18B7,
        ]),
    );

    // ---- Curve 1 ----

    /// P_1 x-coordinate.
    pub(crate) const E1_P_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x5F6259B797B43,
            0x157F63B3AF2F9,
            0x7A3F4EA01DFA8,
            0x1DBB73E23680A,
            0x018914DC770B9,
        ]),
        Fp::from_limbs([
            0x08CB6E0CED492,
            0x05F20AC237154,
            0x7D25B71E8F3DD,
            0x4BF5FC15B1E6E,
            0x01DC3D80FA781,
        ]),
    );

    /// Q_1 x-coordinate.
    pub(crate) const E1_Q_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x567AE7B4D67F3,
            0x5CCB6E9FA4F37,
            0x176489CB8F4EA,
            0x6A1C3C481062B,
            0x02C142D4FEFFE,
        ]),
        Fp::from_limbs([
            0x4C1BFCD30A39F,
            0x21B126AB96A61,
            0x060ADD76BD4A7,
            0x4A6A3D02240A9,
            0x01F52F1A6E758,
        ]),
    );

    /// P_1 − Q_1 x-coordinate (for consistent Okeya-Sakurai lift).
    pub(crate) const E1_PMQ_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x023FAD1A2013B,
            0x5E4194AF99678,
            0x34468FAB3BF1B,
            0x76E4E3F5B18C0,
            0x0432503DA9000,
        ]),
        Fp::from_limbs([
            0x34C912D2B3900,
            0x014D40850DCBE,
            0x672A3EAB48FFE,
            0x2B790AFFECF8C,
            0x002BA92928EAB,
        ]),
    );

    /// Montgomery coefficient A for curve 1.
    pub(crate) const E1_A: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x177F3BD3D98CF,
            0x568291DBF7092,
            0x755DCB3DE2190,
            0x423388F314FE4,
            0x002A6F0241FB7,
        ]),
        Fp::from_limbs([
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
        ]),
    );

    // ---- Curve 2 ----

    /// P_2 x-coordinate.
    pub(crate) const E2_P_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x11012B71D2D54,
            0x76EFAA195F3A3,
            0x6A89621403297,
            0x0F05F07417877,
            0x0058BAFBA5332,
        ]),
        Fp::from_limbs([
            0x3F3EAF5646A2D,
            0x6A0F369773854,
            0x15A15657D2442,
            0x667BA47D7DBF8,
            0x002D784590C43,
        ]),
    );

    /// Q_2 x-coordinate.
    pub(crate) const E2_Q_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x3F45882691098,
            0x6A82534F3934F,
            0x6C6EAD870B0EE,
            0x5669ED2BBB8DA,
            0x02B9A1F281940,
        ]),
        Fp::from_limbs([
            0x41BE7C586D896,
            0x22C68CB09CA5E,
            0x03C045ADBE77B,
            0x506845058C043,
            0x02D2B7E8D71DB,
        ]),
    );

    /// P_2 − Q_2 x-coordinate (for consistent Okeya-Sakurai lift).
    pub(crate) const E2_PMQ_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x42B9C93C44402,
            0x461426DB46E24,
            0x6D7AAB066DC8C,
            0x0BF26F540D0B8,
            0x04F6E2764CC0C,
        ]),
        Fp::from_limbs([
            0x072F03D7912CD,
            0x43AA6E7AF9E21,
            0x679AA18A05871,
            0x14C0756AFFA95,
            0x02ABCBD62F832,
        ]),
    );

    /// Montgomery coefficient A for curve 2.
    pub(crate) const E2_A: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x4D12B0E68B79F,
            0x337935267F3A8,
            0x380BF65840877,
            0x4BCC119304135,
            0x035DA6E9613A8,
        ]),
        Fp::from_limbs([
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
        ]),
    );

    // ---- Curve 3 ----

    /// P_3 x-coordinate.
    pub(crate) const E3_P_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x5B79CA4D5D6E0,
            0x39395E18E3349,
            0x75887BA6EB031,
            0x7D3B20412639B,
            0x013CF1BCCB9DD,
        ]),
        Fp::from_limbs([
            0x76561C962386E,
            0x6F0884CE0B2E6,
            0x20DD8220AACB5,
            0x19375E2D543A7,
            0x04DA1583C8553,
        ]),
    );

    /// Q_3 x-coordinate.
    pub(crate) const E3_Q_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x4854B149C6D0C,
            0x7904EFA1D89AA,
            0x343394A9E5C0F,
            0x68D9D640AD69D,
            0x02D711F0AF96F,
        ]),
        Fp::from_limbs([
            0x3BCAB7A6E1D94,
            0x6C35A91DF0293,
            0x1B51F6EF1B777,
            0x06E9F0BB3D284,
            0x0464E4D547390,
        ]),
    );

    /// P_3 − Q_3 x-coordinate (for consistent Okeya-Sakurai lift).
    pub(crate) const E3_PMQ_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x376C342849596,
            0x657B69DCED4B6,
            0x44B159AEB5ECA,
            0x54B8ABF1BDBFE,
            0x0202393A746E4,
        ]),
        Fp::from_limbs([
            0x260478AD25E9B,
            0x21652ECC55014,
            0x728048F1594DA,
            0x06B5EB728D6D3,
            0x03F305DB59A7F,
        ]),
    );

    /// Montgomery coefficient A for curve 3.
    pub(crate) const E3_A: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x0C17103986F53,
            0x6268EE5A8A215,
            0x11304CB0EFE57,
            0x3846C2AF6C518,
            0x02F57C43F40F7,
        ]),
        Fp::from_limbs([
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
        ]),
    );

    // ---- Curve 4 ----

    /// P_4 x-coordinate.
    pub(crate) const E4_P_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x58C095BAF6ADA,
            0x0741CE646CD96,
            0x5007B4E8336A8,
            0x5010EBBFE93F9,
            0x01B2013C1EB92,
        ]),
        Fp::from_limbs([
            0x75C0724E94E91,
            0x77664D380F258,
            0x0FB261C9EF941,
            0x749554A3CD77C,
            0x01B77C23DE11F,
        ]),
    );

    /// Q_4 x-coordinate.
    pub(crate) const E4_Q_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x71850CEE2E1CA,
            0x1826B78A3CC19,
            0x00DDEBF5154AA,
            0x696AEEBA62D78,
            0x008953BA03B47,
        ]),
        Fp::from_limbs([
            0x2DC44634DA928,
            0x4EA539513E1B6,
            0x5728C1BB241C3,
            0x3686F2152057E,
            0x02F6351277B8B,
        ]),
    );

    /// P_4 − Q_4 x-coordinate (for consistent Okeya-Sakurai lift).
    pub(crate) const E4_PMQ_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x4C38023BA1341,
            0x0EE167E7A402B,
            0x7CBAE09CD7AEE,
            0x442BF312E4537,
            0x00658D9F7AB76,
        ]),
        Fp::from_limbs([
            0x370F1DB4D5016,
            0x4E773FEECB28A,
            0x0427C305FFBE2,
            0x687AB9F2E04CB,
            0x01FEAA39F031C,
        ]),
    );

    /// Montgomery coefficient A for curve 4.
    pub(crate) const E4_A: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x14612B0C4C481,
            0x7219E19939CA1,
            0x2BC69D2A0A8BD,
            0x5F4B0BCBAD964,
            0x025664A8D484E,
        ]),
        Fp::from_limbs([
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
        ]),
    );

    // ---- Curve 5 ----

    /// P_5 x-coordinate.
    pub(crate) const E5_P_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x6292649AB6EC5,
            0x514C3AA63EAA8,
            0x42B95B0DCE14A,
            0x05617E6B3D022,
            0x0262A0B6AD948,
        ]),
        Fp::from_limbs([
            0x0296936F8959C,
            0x7829B486D8303,
            0x51E4D11693064,
            0x3559DBC9D0DAE,
            0x0282BA45C8A46,
        ]),
    );

    /// Q_5 x-coordinate.
    pub(crate) const E5_Q_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x0BD0E9751B3DF,
            0x29BD7A6842BBD,
            0x61480930054F6,
            0x7C90F1CDB870A,
            0x010FC8988A92C,
        ]),
        Fp::from_limbs([
            0x6EE415F437E26,
            0x2244AA9D1A613,
            0x437F0B45EF3A9,
            0x749D8893337B5,
            0x00E5A6EEB752B,
        ]),
    );

    /// P_5 − Q_5 x-coordinate (for consistent Okeya-Sakurai lift).
    pub(crate) const E5_PMQ_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x085579E1B8722,
            0x632525E90080B,
            0x35539378E8D10,
            0x47389416F49D3,
            0x011C1E7BBB047,
        ]),
        Fp::from_limbs([
            0x367F0F2E5527C,
            0x763BFB94F5016,
            0x70DF1A057BFDC,
            0x42460F20B8757,
            0x04A07F8D23DD8,
        ]),
    );

    /// Montgomery coefficient A for curve 5.
    pub(crate) const E5_A: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x27E67B1AD4C35,
            0x4C9B9707EA7BE,
            0x54E830F39A013,
            0x02661741EB0D4,
            0x040D297B19C53,
        ]),
        Fp::from_limbs([
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
        ]),
    );

    // ---- Curve 6 ----

    /// P_6 x-coordinate.
    pub(crate) const E6_P_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x472A0432A50A5,
            0x2584EC65CCF85,
            0x5A5586BA27EFF,
            0x248F2F0F9BD37,
            0x042892709FD53,
        ]),
        Fp::from_limbs([
            0x03727BDAAB80D,
            0x229E05A5546F4,
            0x4BAD4D3212000,
            0x79E6087AEE2DF,
            0x042F9BFAF2BC8,
        ]),
    );

    /// Q_6 x-coordinate.
    pub(crate) const E6_Q_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x140E00D2AD002,
            0x3235E1C701B8D,
            0x272D7237BC84D,
            0x44426D7AD2303,
            0x0459A7FA89B08,
        ]),
        Fp::from_limbs([
            0x4246142CAC789,
            0x1A160F97CC85D,
            0x43707CB72DFF1,
            0x30E5AA57A2936,
            0x02C228AD830FE,
        ]),
    );

    /// P_6 − Q_6 x-coordinate (for consistent Okeya-Sakurai lift).
    pub(crate) const E6_PMQ_X: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x519B1A003883D,
            0x356E25ED579A9,
            0x6B2A143D80555,
            0x1039D06C01EAD,
            0x00A3C331E0448,
        ]),
        Fp::from_limbs([
            0x45DDC052CDEF3,
            0x20A40813439EF,
            0x52630BAF0E697,
            0x4B49649819137,
            0x014D0E0CFB056,
        ]),
    );

    /// Montgomery coefficient A for curve 6.
    pub(crate) const E6_A: Fp2 = Fp2::new(
        Fp::from_limbs([
            0x1FD635B4F2C83,
            0x3DDD0240B9934,
            0x53881AFE8D4A1,
            0x723F462627973,
            0x0147962843332,
        ]),
        Fp::from_limbs([
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
            0x0000000000000,
        ]),
    );
}

/// Endomorphism action matrices for each curve.
///
/// Six 2×2 matrices per curve: action of `i`, `j`, `k`, `gen2`,
/// `gen3`, `gen4` on the torsion basis `E_t[2^f]`.
///
/// # Basis convention
///
/// Generated by the SQIsign C reference's `endomorphism_action.c`
/// for `lvl1` and consumed by [`biscalar_mul`] / [`scalar_mul_add`]
/// without transformation. C-ref's
/// [`ec_biscalar_mul`](https://github.com/SQISign/the-sqisign/blob/main/src/ec/ref/lvlx/ec.c)
/// operates on the slots of `ec_basis_t` positionally (`[m]·B.P +
/// [n]·B.Q`), and [`basis.c:422-425`](https://github.com/SQISign/the-sqisign/blob/main/src/ec/ref/lvlx/basis.c)
/// stores `B.Q = x(P − Q)` and `B.PmQ = Q` per the spec's permuted
/// `(x_P, x_{P−Q}, x_Q)` layout. So for a matrix
/// `M = [[m00, m01], [m10, m11]]`:
///
/// - Column 0: `θ(P) = [m00]·P + [m10]·(P − Q)`
/// - Column 1: `θ(P − Q) = [m01]·P + [m11]·(P − Q)`
///
/// i.e., the matrices encode endomorphisms in the **`(P, P − Q)`
/// basis**, not the textbook `(P, Q)` basis. Selkie's
/// [`TorsionBasis`] mirrors C-ref's slot layout, so passing column 0
/// to [`biscalar_mul`] reproduces `θ(P)` correctly. See the doc
/// comment on `<TorsionBasis as From<(P, Q)>>::from` for the
/// underlying memory layout.
///
/// Verified by `action_matrix_consistent_with_basis` (x-only
/// check against known endomorphism) and `action_matrix_scalar_three`
/// (decomposition of scalar elements produces the identity matrix).
///
/// [`biscalar_mul`]: crate::curves::TorsionBasis::biscalar_mul
/// [`scalar_mul_add`]: crate::curves::TorsionBasis::scalar_mul_add
/// [`TorsionBasis`]: crate::curves::TorsionBasis
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
