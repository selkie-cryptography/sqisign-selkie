//! Constant-time selection on [`BigInt<N>`][super::BigInt]: the
//! [`ConditionallySelectable`] impl and the private limb-level
//! [`mag_select`](BigInt::mag_select) helper that backs it (and
//! other branchless dataflow inside the crate, e.g.
//! [`ct_add`](BigInt::ct_add)'s sign-and-magnitude merge).

use subtle::{Choice, ConditionallySelectable};

use super::{BigInt, ct_select_u64};

impl<const N: usize> BigInt<N> {
    /// Constant-time conditional select on limb arrays.
    pub(super) fn mag_select(a: &[u64; N], b: &[u64; N], choice: u64) -> [u64; N] {
        let mut result = [0u64; N];
        let mut i = 0;
        while i < N {
            result[i] = ct_select_u64(a[i], b[i], choice);
            i += 1;
        }
        result
    }
}

impl<const N: usize> ConditionallySelectable for BigInt<N> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let c = choice.unwrap_u8() as u64;
        Self {
            sign: ct_select_u64(a.sign, b.sign, c),
            limbs: Self::mag_select(&a.limbs, &b.limbs, c),
        }
    }
}
