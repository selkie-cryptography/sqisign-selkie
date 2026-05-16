//! Maximal orders in the quaternion algebra `B_{p,∞}`.
//!
//! [`Order`] is the general type of a maximal order, expressed as a
//! [`Lattice`] with a type-level guarantee of the order invariant.
//! [`ExtremalOrder`] is the p-extremal specialization carrying a
//! small-discriminant quadratic subring witness.
//!
//! See [§3.1.5.1] and [§3.1.7.2] of the SQIsign specification.
//!
//! [§3.1.5.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.1
//! [§3.1.7.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.7.2

use super::{
    super::{
        algebra::{Coordinate, Denominator, Element},
        bigint::BigInt,
        linear::{Matrix, Vector},
    },
    Lattice,
};

/// A maximal order in B_{p,∞}.
///
/// An order is a lattice that is also a subring of B_{p,∞} (closed under
/// multiplication, contains 1). This newtype over [`Lattice`] enforces
/// the order invariant at the type level: values are only constructed by
/// operations that guarantee the result is an order:
///
/// - [`ExtremalOrder::order`] — precomputed extremal orders
/// - [`LeftIdeal::right_order`] — O_R(I) = I⁻¹ · I
/// - [`Order::from_lattice_unchecked`] — internal use when the lattice is known
///   to be an order (e.g., narrowing after `reduce_to_prime_norm`)
///
/// Implements [`Deref<Target = Lattice<N>>`](core::ops::Deref) so all
/// lattice methods are available transparently. Use `From<Order<N>>` to
/// unwrap into the underlying [`Lattice`].
///
/// See [§3.1.5.1] of the SQIsign specification.
///
/// [`LeftIdeal::right_order`]: super::LeftIdeal::right_order
/// [§3.1.5.1]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.5.1
#[derive(Clone)]
pub struct Order<const N: usize>(Lattice<N>);

impl<const N: usize> Order<N> {
    /// Constructs an order from a lattice that is known to be an order.
    ///
    /// # Safety (logical)
    ///
    /// The caller must ensure the lattice is actually a maximal order
    /// (closed under multiplication, contains 1). This is not checked.
    pub(crate) const fn from_lattice_unchecked(lattice: Lattice<N>) -> Self {
        Self(lattice)
    }

    /// Returns the underlying lattice.
    #[inline]
    pub const fn lattice(&self) -> &Lattice<N> {
        &self.0
    }
}

impl<const N: usize> core::ops::Deref for Order<N> {
    type Target = Lattice<N>;

    #[inline]
    fn deref(&self) -> &Lattice<N> {
        &self.0
    }
}

impl<const N: usize> Copy for Order<N> where BigInt<N>: Copy {}

/// Unwrap an order into its underlying lattice.
impl<const N: usize> From<Order<N>> for Lattice<N> {
    fn from(order: Order<N>) -> Self {
        order.0
    }
}

impl<const N: usize> core::fmt::Debug for Order<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Order({:?})", self.0)
    }
}

/// A p-extremal maximal order in B_{p,∞}.
///
/// These are maximal orders containing `j` and a distinguished quadratic
/// subring `Z[ω]` of small discriminant, such that `j` and `Z[ω]` are
/// orthogonal. The element `z` with `z² = -q` generates the quadratic
/// subring, and `t` is an element of norm `p` orthogonal to `z`.
///
/// See [§3.1.7.2] of the SQIsign specification.
///
/// [§3.1.7.2]: https://sqisign.org/spec/sqisign-20250707.pdf#subsubsection.3.1.7.2
#[derive(Clone)]
pub struct ExtremalOrder<const N: usize> {
    /// The order.
    order: Order<N>,
    /// Element z with z² = -q (small discriminant).
    z: Element<4>,
    /// Element t with nrd(t) = p, orthogonal to z.
    t: Element<4>,
    /// The absolute value |z²| (a small positive integer).
    q: u32,
}

impl<const N: usize> ExtremalOrder<N> {
    /// Creates an extremal order from typed components.
    ///
    /// The lattice must be a maximal order (closed under multiplication,
    /// contains 1). This is not checked — the lattice is wrapped in
    /// [`Order`] unconditionally. Prefer
    /// [`from_raw_limbs`](ExtremalOrder::from_raw_limbs) for constructing from
    /// precomputed raw data.
    #[inline]
    pub const fn new(order: Lattice<N>, z: Element<4>, t: Element<4>, q: u32) -> Self {
        Self {
            order: Order::from_lattice_unchecked(order),
            z,
            t,
            q,
        }
    }

    /// Returns the maximal order as an [`Order`].
    #[inline]
    pub const fn order(&self) -> &Order<N> {
        &self.order
    }

    /// Returns the element z (z² = -q).
    #[inline]
    pub const fn z(&self) -> &Element<4> {
        &self.z
    }

    /// Returns the element t (nrd(t) = p).
    #[inline]
    pub const fn t(&self) -> &Element<4> {
        &self.t
    }

    /// Returns q = |z²|.
    #[inline]
    pub const fn q(&self) -> u32 {
        self.q
    }
}

impl ExtremalOrder<4> {
    /// Construct from raw sign+limbs data, matching the C reference's
    /// `quat_p_extremal_maximal_order_t` layout.
    ///
    /// # Data format
    ///
    /// All integer values are `(sign, [u64; 4])` where `sign = 0` means
    /// non-negative and `sign = 1` means negative. The `[u64; 4]` array
    /// holds the absolute value in little-endian 64-bit limbs. This
    /// matches GMP's `_mp_size` (sign) + `_mp_d` (limbs) representation
    /// used by the C reference's `quaternion_data.c`.
    ///
    /// - `basis`: 4×4 matrix of the order's lattice basis in HNF, expressed in
    ///   the `{1, i, j, k}` basis. Columns divided by the lattice denominator
    ///   give elements of B_{p,∞}.
    /// - `z`: the element z with z² = −q, as four coordinates `[a, b, c, d]` in
    ///   the `{1, i, j, k}` basis.
    /// - `q`: the absolute value |z²|.
    ///
    /// Both the lattice denominator and the z denominator are deduced
    /// from `basis[0][0]` (the top-left HNF entry), which equals both
    /// for all NIST-I extremal orders. All orders have `t = j`.
    ///
    /// # Divergence from internal `Element` representation
    ///
    /// The z data here stores all four coordinates explicitly, matching
    /// the C reference's `quat_alg_elem_t` (which always has four
    /// coordinates + denominator). In practice, for all NIST-I extremal
    /// orders, z has the form `(0, b, 0, d)/denom` — the `a` and `c`
    /// coordinates are zero. However, this constructor does not assume
    /// that: it passes all four coordinates through to `Element::new`,
    /// so the Sage precomputation script can output z in the same
    /// format as the C reference without special-casing.
    pub const fn from_raw_limbs(
        basis: [[(u64, [u64; 4]); 4]; 4],
        z: [(u64, [u64; 4]); 4],
        q: u32,
    ) -> Self {
        const fn bi(sl: (u64, [u64; 4])) -> BigInt<4> {
            BigInt::from_sign_and_limbs(sl.0, sl.1)
        }
        // Both lattice denom and z denom = basis[0][0] (all 7 NIST-I orders).
        let denom = bi(basis[0][0]);
        Self::new(
            Lattice::new(
                Matrix::from_rows(
                    Vector::new(
                        bi(basis[0][0]),
                        bi(basis[0][1]),
                        bi(basis[0][2]),
                        bi(basis[0][3]),
                    ),
                    Vector::new(
                        bi(basis[1][0]),
                        bi(basis[1][1]),
                        bi(basis[1][2]),
                        bi(basis[1][3]),
                    ),
                    Vector::new(
                        bi(basis[2][0]),
                        bi(basis[2][1]),
                        bi(basis[2][2]),
                        bi(basis[2][3]),
                    ),
                    Vector::new(
                        bi(basis[3][0]),
                        bi(basis[3][1]),
                        bi(basis[3][2]),
                        bi(basis[3][3]),
                    ),
                ),
                denom,
            ),
            Element::new(
                Coordinate::from_bigint(bi(z[0])),
                Coordinate::from_bigint(bi(z[1])),
                Coordinate::from_bigint(bi(z[2])),
                Coordinate::from_bigint(bi(z[3])),
                Denominator::from_bigint_unchecked(denom),
            ),
            Element::J,
            q,
        )
    }
}

impl<const N: usize> Copy for ExtremalOrder<N> where BigInt<N>: Copy {}

impl ExtremalOrder<4> {
    /// Widen to `ExtremalOrder<M>` by zero-extending all `BigInt<4>`
    /// limbs in the lattice basis and denominator to `BigInt<M>`.
    ///
    /// The z and t elements remain at width 4 (they are always small).
    /// Used when lattice arithmetic needs wider intermediates (e.g.,
    /// `LeftIdeal<8>` for `represent_integer`, `LeftIdeal<9>` for
    /// D_MIX commitment).
    #[must_use]
    pub fn widen<const M: usize>(&self) -> ExtremalOrder<M> {
        let basis4 = self.order().basis();
        let denom4 = self.order().denom();

        let mut basis_m = Matrix::<M>::ZERO;
        for row in 0..4 {
            for col in 0..4 {
                basis_m[row][col] = basis4[row][col].widen::<M>();
            }
        }
        let order_lat = Lattice::new(basis_m, denom4.widen::<M>());

        ExtremalOrder::new(order_lat, *self.z(), *self.t(), self.q())
    }
}

impl From<ExtremalOrder<4>> for ExtremalOrder<8> {
    fn from(order: ExtremalOrder<4>) -> Self {
        order.widen()
    }
}

impl From<ExtremalOrder<4>> for ExtremalOrder<30> {
    fn from(order: ExtremalOrder<4>) -> Self {
        order.widen()
    }
}

impl<const N: usize> core::fmt::Debug for ExtremalOrder<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ExtremalOrder(q={}, z={:?}, t={:?})",
            self.q, self.z, self.t
        )
    }
}
