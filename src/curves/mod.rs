//! Elliptic curves, points, and isogenies between them.
//!
//! This module provides:
//! - [`montgomery`]: Montgomery curves and x-only projective point arithmetic
//! - [`TorsionBasis`]: generators of torsion subgroups, used to define
//!   isogeny kernels
//! - [`two_isogeny`], [`four_isogeny`]: individual isogeny steps
//! - [`chain`]: chains of isogenies of degree 2^e
//! - Torsion basis hints and ladders ([§2.2.3], [§8.2])
//!
//! [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
//! [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2

pub mod montgomery;
pub mod isogeny;

use subtle::ConditionallySelectable;

use crate::curves::montgomery::{Curve, MontgomeryPoint};
use crate::params::TORSION_EVEN_POWER;

/// An exponent e such that 2^e divides the torsion group order.
///
/// Always satisfies 0 ≤ e ≤ [`TORSION_EVEN_POWER`]. Used to specify
/// the degree 2^e of isogeny chains and torsion subgroups.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct TorsionExponent(u32);

impl TorsionExponent {
    /// The full torsion exponent f = [`TORSION_EVEN_POWER`].
    pub const FULL: TorsionExponent = TorsionExponent(TORSION_EVEN_POWER);

    /// Construct from a raw value, panicking if out of range.
    pub fn new(e: u32) -> TorsionExponent {
        assert!(e <= TORSION_EVEN_POWER, "torsion exponent {e} exceeds f = {TORSION_EVEN_POWER}");
        TorsionExponent(e)
    }

    /// Construct from a raw value, returning `None` if out of range.
    pub fn try_new(e: u32) -> Option<TorsionExponent> {
        if e <= TORSION_EVEN_POWER {
            Some(TorsionExponent(e))
        } else {
            None
        }
    }

    /// The raw exponent value.
    pub fn value(self) -> u32 {
        self.0
    }

    /// Subtract, returning `None` if the result would be negative.
    pub fn checked_sub(self, rhs: u32) -> Option<TorsionExponent> {
        self.0.checked_sub(rhs).and_then(TorsionExponent::try_new)
    }
}

impl From<TorsionExponent> for u32 {
    fn from(e: TorsionExponent) -> u32 {
        e.0
    }
}

// ---------------------------------------------------------------------------
// Basis hints
// ---------------------------------------------------------------------------

/// A 1-byte hint for deterministic torsion basis reconstruction.
///
/// Encodes a pair (h_A, h) where h_A is a quadratic residuosity flag
/// (1 bit, stored in the LSB) and h is a 7-bit index used to find a
/// suitable x-coordinate for the first basis point.
///
/// See [§2.2.3], [§4.6].
///
/// [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
/// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct BasisHint(u8);

impl BasisHint {
    /// The quadratic residuosity flag h_A (0 or 1).
    fn h_A(self) -> u8 {
        self.0 & 1
    }

    /// The 7-bit index h.
    fn h(self) -> u8 {
        self.0 >> 1
    }

    /// Construct from the (h_A, h) pair.
    fn new(h_A: u8, h: u8) -> BasisHint {
        debug_assert!(h_A <= 1);
        debug_assert!(h < 128);
        BasisHint((h << 1) | (h_A & 1))
    }

    /// The raw byte representation.
    fn to_byte(self) -> u8 {
        self.0
    }

    /// Construct from a raw byte.
    pub(crate) fn from_byte(b: u8) -> BasisHint {
        BasisHint(b)
    }
}

/// Hint for the verifying key torsion basis on E_pk.
///
/// Serialized as part of the [verifying key][§4.6] (1 byte).
///
/// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct VerifyingKeyHint(BasisHint);

/// Hint for the auxiliary curve torsion basis on E_aux.
///
/// Serialized as part of the [signature][§4.6] (1 byte).
///
/// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AuxiliaryHint(BasisHint);

/// Hint for the challenge curve torsion basis on E_chl.
///
/// Serialized as part of the [signature][§4.6] (1 byte).
///
/// [§4.6]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.6
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ChallengeHint(BasisHint);

impl From<u8> for VerifyingKeyHint {
    fn from(b: u8) -> Self { VerifyingKeyHint(BasisHint::from_byte(b)) }
}

impl From<VerifyingKeyHint> for u8 {
    fn from(h: VerifyingKeyHint) -> u8 { h.0.to_byte() }
}

impl From<u8> for AuxiliaryHint {
    fn from(b: u8) -> Self { AuxiliaryHint(BasisHint::from_byte(b)) }
}

impl From<AuxiliaryHint> for u8 {
    fn from(h: AuxiliaryHint) -> u8 { h.0.to_byte() }
}

impl From<u8> for ChallengeHint {
    fn from(b: u8) -> Self { ChallengeHint(BasisHint::from_byte(b)) }
}

impl From<ChallengeHint> for u8 {
    fn from(h: ChallengeHint) -> u8 { h.0.to_byte() }
}

// ---------------------------------------------------------------------------
// Torsion basis
// ---------------------------------------------------------------------------

/// An x-only basis (R, S) of a torsion subgroup E\[m\], stored as
/// the projective triple (R, S, R−S).
///
/// The third point R−S is required for differential addition, which
/// is the only way to compute P + Q in x-only Montgomery arithmetic.
/// This triple is the minimum information needed to compute arbitrary
/// linear combinations \[a\]R + \[b\]S via `LadderBiscalar`.
///
/// See [§2.2.3] (torsion subgroups and deterministic basis computation).
///
/// [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
#[derive(Copy, Clone, Debug)]
pub struct TorsionBasis {
    /// First basis element R.
    pub R: MontgomeryPoint,
    /// Second basis element S.
    pub S: MontgomeryPoint,
    /// Difference R − S (needed for differential addition).
    pub RS: MontgomeryPoint,
}

impl TorsionBasis {
    /// Construct a basis from its three components.
    pub fn new(R: MontgomeryPoint, S: MontgomeryPoint, RS: MontgomeryPoint) -> TorsionBasis {
        TorsionBasis { R, S, RS }
    }

    /// Compute R + \[m\]S from this basis.
    ///
    /// Given the basis (R, S, R−S), uses the three-point Montgomery
    /// ladder to compute R + \[m\]S. The scalar m is given as a
    /// little-endian bit slice (LSB first). Constant-time in the value
    /// of m.
    ///
    /// This is the primary way to compute an isogeny kernel generator
    /// from a torsion basis and a scalar.
    ///
    /// Implements `Ladder3pt` ([§8.2], Algorithm 8.7).
    ///
    /// [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
    pub fn ladder3pt(&self, m_bits_le: &[u8]) -> MontgomeryPoint {
        use crate::curves::montgomery::differential_add_and_double;

        // Initialize: X₀ ← Q, X₁ ← P, X₂ ← P−Q
        let mut x0 = self.S;
        let mut x1 = self.R;
        let mut x2 = self.RS;

        // Process bits from LSB to MSB.
        for &bit in m_bits_le.iter() {
            let swap = subtle::Choice::from(bit & 1);
            MontgomeryPoint::conditional_swap(&mut x1, &mut x2, swap);
            differential_add_and_double(&mut x0, &mut x1, &x2);
            MontgomeryPoint::conditional_swap(&mut x1, &mut x2, swap);
        }
        x1
    }

    /// Compute \[m\]R + \[n\]S from this basis.
    ///
    /// Uses the biscalar Montgomery ladder with scalar recoding.
    /// Both scalars are given as little-endian byte slices of equal
    /// length. Constant-time in the values of m and n.
    ///
    /// Implements `LadderBiscalar` ([§8.2], Algorithm 8.8).
    ///
    /// [§8.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.8.2
    pub fn ladder_biscalar(&self, m: &[u8], n: &[u8], kbits: usize) -> MontgomeryPoint {
        use crate::curves::montgomery::differential_add_and_double;
        use subtle::ConditionallySelectable;

        let P = &self.R;
        let Q = &self.S;
        let PmQ = &self.RS;
        let curve = P.curve();

        // --- Recoding stage ---
        // Determine sigma based on parity of m and n.
        let bit_m0 = m[0] & 1;
        let bit_n0 = n[0] & 1;
        let mask_m: u8 = 0u8.wrapping_sub(bit_m0);
        let mask_n: u8 = 0u8.wrapping_sub(bit_n0);

        // sigma = (0,1) if both same parity, else the even one gets sigma=1
        let evens = (bit_m0 ^ 1) + (bit_n0 ^ 1);
        let m_evens: u8 = 0u8.wrapping_sub(evens & 1);
        let mut sigma0: u8 = (bit_m0 ^ 1) & m_evens;
        let mut sigma1: u8 = ((bit_n0 ^ 1) & m_evens) | (1 & !m_evens);

        // Convert even scalars to odd (subtract 1).
        let mut m_t = [0u8; 32];
        let mut n_t = [0u8; 32];
        let m_len = m.len().min(32);
        let n_len = n.len().min(32);
        m_t[..m_len].copy_from_slice(&m[..m_len]);
        n_t[..n_len].copy_from_slice(&n[..n_len]);

        // Subtract 1 from even scalars (constant-time).
        sub_one_ct(&mut m_t, mask_m ^ 0xff); // subtract if m was even
        sub_one_ct(&mut n_t, mask_n ^ 0xff); // subtract if n was even

        // Compute recoding bits r[2i] and r[2i+1].
        let mut r = vec![0u8; 2 * kbits];
        let mut pre_sigma = 0u8;
        for i in 0..kbits {
            // Swap m_t and n_t if sigma changed.
            let swap_mask = 0u8.wrapping_sub(sigma0 ^ pre_sigma);
            swap_bytes_ct(&mut m_t, &mut n_t, swap_mask);

            let bs1_ip1: u8;
            let bs2_ip1: u8;
            if i == kbits - 1 {
                bs1_ip1 = 0;
                bs2_ip1 = 0;
            } else {
                bs1_ip1 = shr1_ct(&mut m_t);
                bs2_ip1 = shr1_ct(&mut n_t);
            }
            let bs1_i = m_t[0] & 1;
            let bs2_i = n_t[0] & 1;

            r[2 * i] = bs1_i ^ bs1_ip1;
            r[2 * i + 1] = bs2_i ^ bs2_ip1;

            // Update sigma if r[2i+1] = 1.
            pre_sigma = sigma0;
            let flip = 0u8.wrapping_sub(r[2 * i + 1]);
            let tmp = (sigma0 & !flip) | (sigma1 & flip);
            sigma1 = (sigma1 & !flip) | (sigma0 & flip);
            sigma0 = tmp;
        }

        // --- Evaluation stage ---
        let mut R0 = MontgomeryPoint::identity(curve);
        let sigma0_choice = subtle::Choice::from(sigma0 & 1);
        let mut R1 = MontgomeryPoint::conditional_select(P, Q, sigma0_choice);
        let mut R2 = MontgomeryPoint::conditional_select(Q, P, sigma0_choice);

        let mut D1 = R1;
        let mut D2 = R2;

        // R2 ← xADD(R1, R2, P−Q)
        R2 = R1.differential_add(&R2, PmQ);

        let mut F1 = R2;
        let mut F2 = *PmQ;

        // Main loop: process bits from MSB to LSB.
        for i in (0..kbits).rev() {
            let h = r[2 * i] + r[2 * i + 1]; // h ∈ {0, 1, 2}

            // T0 ← R_{⌊h/2⌋}, then double it.
            let h_bit0 = subtle::Choice::from(h & 1);
            let h_bit1 = subtle::Choice::from((h >> 1) & 1);
            let mut T0 = MontgomeryPoint::conditional_select(&R0, &R1, h_bit0);
            T0 = MontgomeryPoint::conditional_select(&T0, &R2, h_bit1);
            T0 = T0.double();

            // T1 and T2 depend on r[2i+1].
            let r_bit = subtle::Choice::from(r[2 * i + 1] & 1);
            let T1_a = MontgomeryPoint::conditional_select(&R0, &R1, r_bit);
            let T1_b = MontgomeryPoint::conditional_select(&R1, &R2, r_bit);

            // Swap DIFF1a/DIFF1b based on r[2i+1].
            MontgomeryPoint::conditional_swap(&mut D1, &mut D2, r_bit);
            let T1 = T1_a.differential_add(&T1_b, &D1);
            let T2 = R0.differential_add(&R2, &F1);

            // Swap DIFF2a/DIFF2b if h is odd.
            MontgomeryPoint::conditional_swap(&mut F1, &mut F2, h_bit0);

            R0 = T0;
            R1 = T1;
            R2 = T2;
        }

        // Output: select based on parity of original scalars.
        let mut result = MontgomeryPoint::conditional_select(&R0, &R1, subtle::Choice::from(m_evens & 1));
        let both_odd = subtle::Choice::from(bit_m0 & bit_n0);
        result = MontgomeryPoint::conditional_select(&result, &R2, both_odd);

        result
    }

    /// Deterministically generate a torsion basis for E_A\[2^e\] from a
    /// curve and a hint, where e = [`TORSION_EVEN_POWER`].
    ///
    /// The hint encodes (h_A, h) where h_A indicates the quadratic
    /// residuosity of A, and h is used to quickly find a valid
    /// x-coordinate for the first basis point.
    ///
    /// Implements `TorsionBasisFromHint` ([§2.2.3], Algorithm 2.2).
    ///
    /// [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
    /// [`TORSION_EVEN_POWER`]: crate::params::TORSION_EVEN_POWER
    pub(crate) fn from_hint(curve: &Curve, hint: BasisHint) -> TorsionBasis {
        use crate::fields::fp::Fp;
        use crate::fields::fp2::Fp2;

        let e = crate::params::TORSION_EVEN_POWER;
        let A = Fp2::from(*curve.coefficient().as_fp2());

        // Special case: A = 0 (the starting curve E₀).
        // Use precomputed basis points and compute the difference.
        if A == Fp2::ZERO {
            let P = MontgomeryPoint::from_affine_x(crate::params::BASIS_E0_P_X, curve);
            let Q = MontgomeryPoint::from_affine_x(crate::params::BASIS_E0_Q_X, curve);
            let PmQ = P.projective_difference(&Q);
            return TorsionBasis { R: P, S: PmQ, RS: Q };
        }

        let h_A = hint.h_A();
        let h = hint.h();

        // Compute x(P) from the hint.
        let x_P = if h == 0 {
            // Rare fallback: hint didn't fit in 7 bits.
            // Must search from scratch (starting at 128).
            if h_A == 0 {
                // A is NQR: search for n*A on the curve.
                find_na_x_coord(&A, curve, 128)
            } else {
                // A is QR: search for -A/(1+i*b) on the curve.
                find_nqr_factor(&A, curve, 128)
            }
        } else if h_A == 0 {
            // A is NQR: x(P) = h * A
            &A * &Fp2::from_fp(Fp::from_small(h as u32))
        } else {
            // A is QR: x(P) = -A / (1 + i*h)
            let z = Fp2::new(Fp::ONE, Fp::from_small(h as u32));
            &(-&A) * &z.invert()
        };

        let x_Q = -&(&A + &x_P); // x(Q) = -x(P) - A

        let mut P = MontgomeryPoint::from_affine_x(x_P, curve);
        let mut Q = MontgomeryPoint::from_affine_x(x_Q, curve);

        // Clear odd cofactor to get points of order 2^e.
        // Multiply by (p+1)/2^e = cofactor.
        P = clear_cofactor(&P);
        Q = clear_cofactor(&Q);

        // Compute P−Q and arrange so Q is above (0,0).
        let PmQ = P.projective_difference(&Q);

        // The C reference swaps: basis = (P, PmQ, Q) so that
        // the second generator is above (0,0).
        TorsionBasis {
            R: P,
            S: PmQ,
            RS: Q,
        }
    }

    /// Generate a torsion basis for E_A\[2^e\] and its associated hint,
    /// where e = [`TORSION_EVEN_POWER`].
    ///
    /// Implements `TorsionBasisToHint` ([§2.2.3], Algorithm 2.1).
    ///
    /// [§2.2.3]: https://sqisign.org/spec/sqisign-20250707.pdf#section.2.2
    /// [`TORSION_EVEN_POWER`]: crate::params::TORSION_EVEN_POWER
    pub(crate) fn to_hint(curve: &Curve) -> (TorsionBasis, BasisHint) {
        use crate::fields::fp2::Fp2;

        let e = crate::params::TORSION_EVEN_POWER;
        let A = Fp2::from(*curve.coefficient().as_fp2());

        if A == Fp2::ZERO {
            // E₀ has no hint — the basis is precomputed.
            let P = MontgomeryPoint::from_affine_x(crate::params::BASIS_E0_P_X, curve);
            let Q = MontgomeryPoint::from_affine_x(crate::params::BASIS_E0_Q_X, curve);
            let PmQ = P.projective_difference(&Q);
            let basis = TorsionBasis { R: P, S: PmQ, RS: Q };
            return (basis, BasisHint::from_byte(0));
        }

        let h_A = bool::from(A.is_square());

        let (x_P, h) = if !h_A {
            // A is NQR: find n such that n*A is on the curve.
            let (x, hint) = find_na_x_coord_with_hint(&A, curve);
            (x, hint)
        } else {
            // A is QR: find b such that -A/(1+i*b) is on the curve.
            let (x, hint) = find_nqr_factor_with_hint(&A, curve);
            (x, hint)
        };

        let x_Q = -&(&A + &x_P);

        let mut P = MontgomeryPoint::from_affine_x(x_P, curve);
        let mut Q = MontgomeryPoint::from_affine_x(x_Q, curve);

        P = clear_cofactor(&P);
        Q = clear_cofactor(&Q);

        let PmQ = P.projective_difference(&Q);

        let basis = TorsionBasis {
            R: P,
            S: PmQ,
            RS: Q,
        };

        let hint_byte = BasisHint::new(h_A as u8, h);
        (basis, hint_byte)
    }
}

// ---------------------------------------------------------------------------
// Helper functions for torsion basis generation
// ---------------------------------------------------------------------------

/// Subtract 1 from a little-endian byte array, conditionally.
/// `mask` is 0xff to subtract, 0x00 to skip.
fn sub_one_ct(a: &mut [u8], mask: u8) {
    let mut borrow: u16 = (mask & 1) as u16;
    for byte in a.iter_mut() {
        let diff = (*byte as u16).wrapping_sub(borrow);
        *byte = diff as u8;
        borrow = (diff >> 8) & 1;
    }
}

/// Shift a little-endian byte array right by 1. Returns the shifted-out LSB.
fn shr1_ct(a: &mut [u8]) -> u8 {
    let lsb = a[0] & 1;
    let len = a.len();
    for i in 0..len - 1 {
        a[i] = (a[i] >> 1) | (a[i + 1] << 7);
    }
    a[len - 1] >>= 1;
    lsb
}

/// Conditionally swap two byte arrays. `mask` is 0xff to swap, 0x00 to skip.
fn swap_bytes_ct(a: &mut [u8], b: &mut [u8], mask: u8) {
    for (ai, bi) in a.iter_mut().zip(b.iter_mut()) {
        let diff = (*ai ^ *bi) & mask;
        *ai ^= diff;
        *bi ^= diff;
    }
}

/// Check if x³ + Ax² + x is a square in F_{p²} (i.e., (x, ·) is on E_A).
fn is_on_curve(x: &crate::fields::fp2::Fp2, A: &crate::fields::fp2::Fp2) -> bool {
    let t = &(x + A) * x; // x² + Ax
    let t = &(&t + &crate::fields::fp2::Fp2::ONE) * x; // x³ + Ax² + x
    bool::from(t.is_square())
}

/// Find n such that n*A is a valid x-coordinate on E_A. Returns x(P).
fn find_na_x_coord(
    A: &crate::fields::fp2::Fp2,
    _curve: &Curve,
    start: u8,
) -> crate::fields::fp2::Fp2 {
    let mut x = &crate::fields::fp2::Fp2::from_fp(crate::fields::fp::Fp::from_small(start as u32)) * A;
    let mut n = start;
    while !is_on_curve(&x, A) || bool::from(x.is_square()) {
        x = &x + A;
        n += 1;
    }
    x
}

/// Find n*A and return (x, hint).
fn find_na_x_coord_with_hint(
    A: &crate::fields::fp2::Fp2,
    _curve: &Curve,
) -> (crate::fields::fp2::Fp2, u8) {
    let mut x = *A;
    let mut n: u8 = 1;
    while !is_on_curve(&x, A) || bool::from(x.is_square()) {
        x = &x + A;
        n += 1;
    }
    let hint = if n < 128 { n } else { 0 };
    (x, hint)
}

/// Find b such that -A/(1+i*b) is a valid NQR x-coordinate on E_A.
fn find_nqr_factor(
    A: &crate::fields::fp2::Fp2,
    _curve: &Curve,
    start: u8,
) -> crate::fields::fp2::Fp2 {
    use crate::fields::fp::Fp;
    use crate::fields::fp2::Fp2;

    let mut n = start;
    loop {
        let z = Fp2::new(Fp::ONE, Fp::from_small(n as u32));
        let x = &(-A) * &z.invert();
        if is_on_curve(&x, A) && !bool::from(x.is_square()) {
            return x;
        }
        n += 1;
    }
}

/// Find -A/(1+i*b) and return (x, hint).
fn find_nqr_factor_with_hint(
    A: &crate::fields::fp2::Fp2,
    _curve: &Curve,
) -> (crate::fields::fp2::Fp2, u8) {
    use crate::fields::fp::Fp;
    use crate::fields::fp2::Fp2;

    let mut n: u8 = 1;
    loop {
        let z = Fp2::new(Fp::ONE, Fp::from_small(n as u32));
        let x = &(-A) * &z.invert();
        if is_on_curve(&x, A) && !bool::from(x.is_square()) {
            let hint = if n < 128 { n } else { 0 };
            return (x, hint);
        }
        n += 1;
    }
}

/// Clear the odd cofactor: multiply P by (p+1)/2^f to get a point of
/// order dividing 2^f, then double (TORSION_EVEN_POWER − e) times.
fn clear_cofactor(P: &MontgomeryPoint) -> MontgomeryPoint {
    // (p+1)/2^f = c = 5 for NIST-I.
    // This is a public, small scalar multiplication.
    P * crate::params::COFACTOR
}

