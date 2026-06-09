//! 4-element integer vectors over [`BigInt`].
//!
//! Used for quaternion-algebra element coordinates in the basis
//! `{1, i, j, ij}` and for lattice basis column vectors.

use core::{
    fmt,
    ops::{Add, Index, IndexMut, Neg, Sub},
};

use crate::quaternions::bigint::BigInt;

/// A 4-element vector of [`BigInt<N>`] values.
///
/// Used for quaternion algebra element coordinates in the basis
/// `{1, i, j, ij}` and for lattice basis column vectors.
#[derive(Clone)]
pub struct Vector<const N: usize>([BigInt<N>; 4]);

impl<const N: usize> Vector<N> {
    /// The zero vector.
    pub const ZERO: Self = Self([BigInt::ZERO; 4]);

    /// Creates a vector from four elements.
    #[inline]
    pub const fn new(a: BigInt<N>, b: BigInt<N>, c: BigInt<N>, d: BigInt<N>) -> Self {
        Self([a, b, c, d])
    }

    /// Dot product: `sum_i self[i]*other[i]`.
    pub fn dot(&self, other: &Self) -> BigInt<N> {
        let mut acc = self.0[0].ct_mul(&other.0[0]);
        acc = acc.ct_add(&self.0[1].ct_mul(&other.0[1]));
        acc = acc.ct_add(&self.0[2].ct_mul(&other.0[2]));
        acc = acc.ct_add(&self.0[3].ct_mul(&other.0[3]));
        acc
    }

    /// Widen each component from `BigInt<N>` to `BigInt<W>`.
    ///
    /// Sign-extends per [`BigInt::widen`]. Requires `W ≥ N`.
    #[inline]
    pub fn widen<const W: usize>(self) -> Vector<W> {
        Vector::new(
            self.0[0].widen::<W>(),
            self.0[1].widen::<W>(),
            self.0[2].widen::<W>(),
            self.0[3].widen::<W>(),
        )
    }
}

impl<const N: usize> Copy for Vector<N> where BigInt<N>: Copy {}

impl<const N: usize> Index<usize> for Vector<N> {
    type Output = BigInt<N>;
    #[inline]
    fn index(&self, idx: usize) -> &BigInt<N> {
        &self.0[idx]
    }
}

impl<const N: usize> IndexMut<usize> for Vector<N> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut BigInt<N> {
        &mut self.0[idx]
    }
}

impl<const N: usize> Add for Vector<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0].ct_add(&rhs.0[0]),
            self.0[1].ct_add(&rhs.0[1]),
            self.0[2].ct_add(&rhs.0[2]),
            self.0[3].ct_add(&rhs.0[3]),
        ])
    }
}

impl<const N: usize> Sub for Vector<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0].ct_sub(&rhs.0[0]),
            self.0[1].ct_sub(&rhs.0[1]),
            self.0[2].ct_sub(&rhs.0[2]),
            self.0[3].ct_sub(&rhs.0[3]),
        ])
    }
}

impl<const N: usize> Neg for Vector<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self([
            self.0[0].wrapping_neg(),
            self.0[1].wrapping_neg(),
            self.0[2].wrapping_neg(),
            self.0[3].wrapping_neg(),
        ])
    }
}

impl<const N: usize> PartialEq for Vector<N> {
    fn eq(&self, other: &Self) -> bool {
        self.0[0] == other.0[0]
            && self.0[1] == other.0[1]
            && self.0[2] == other.0[2]
            && self.0[3] == other.0[3]
    }
}

impl<const N: usize> Eq for Vector<N> {}

/// Widen: zero-extend a four-element vector to eight-limb elements.
impl From<Vector<4>> for Vector<8> {
    fn from(v: Vector<4>) -> Self {
        Self::new(v[0].into(), v[1].into(), v[2].into(), v[3].into())
    }
}

impl<const N: usize> fmt::Debug for Vector<N> {
    #[cfg_attr(test, mutants::skip)] // formatting, not correctness
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Vector([{}, {}, {}, {}])",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}
