//! Differential proptests for `BigInt<N>` against `num-bigint`.
//!
//! Algebraic-identity proptests (associativity, distributivity, etc.)
//! catch internal inconsistencies but cannot detect a uniformly wrong
//! answer that still satisfies the laws. The oracle catches that
//! class: a second, unrelated implementation of the same math, queried
//! on every operation.
//!
//! `num-bigint` is dynamically sized, so each comparison reduces
//! `BigInt::<N>::op(a, b)` to the same `N·64`-bit window — the
//! reference is computed in unbounded precision and then truncated to
//! match our fixed-width output mod `2^(N·64)`, with sign preserved
//! by the sign+magnitude representation.
//!
//! Today the oracle is `num-bigint` everywhere. A second GMP-backed
//! oracle (via `rug`) is planned for the Linux CI runner behind a
//! feature flag; until then the pure-Rust oracle covers the common bug
//! classes (sign handling, carry propagation, Euclidean vs truncated
//! remainder, gcd termination, shift overflow).

use num_bigint::{BigInt as NumBigInt, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, Zero};
use proptest::prelude::*;
use sqisign_selkie::quaternions::bigint::BigInt;

/// Random `BigInt<4>` from a sign bit and four `u64` limbs.
fn arb_bigint4() -> impl Strategy<Value = BigInt<4>> {
    (any::<bool>(), any::<[u64; 4]>())
        .prop_map(|(neg, limbs)| BigInt::from_sign_and_limbs(if neg { 1 } else { 0 }, limbs))
}

/// Small `BigInt<4>` that fits in `i32`, so multiplication doesn't wrap.
fn arb_small_bigint4() -> impl Strategy<Value = BigInt<4>> {
    any::<i32>().prop_map(|v| BigInt::from_i64(v as i64))
}

/// Lifts a `BigInt<N>` into unbounded-precision `num_bigint::BigInt`.
///
/// Sign+magnitude → `Sign` + little-endian limb bytes. Zero
/// canonicalises to `Sign::NoSign` regardless of our `sign` field.
fn to_num<const N: usize>(a: BigInt<N>) -> NumBigInt {
    let mut bytes = [0u8; 8 * 8];
    assert!(N <= 8, "oracle helper sized for N <= 8");
    for (i, limb) in a.as_limbs().iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    let magnitude = NumBigInt::from_bytes_le(Sign::Plus, &bytes[..N * 8]);
    if a.is_negative().into() {
        -magnitude
    } else {
        magnitude
    }
}

/// Projects a `num_bigint::BigInt` into `BigInt<N>` by truncating the
/// magnitude to `N·64` bits and reattaching the sign — mirrors what
/// our wrapping arithmetic does on overflow.
fn from_num<const N: usize>(n: &NumBigInt) -> BigInt<N> {
    let (_, mut bytes) = n.abs().to_bytes_le();
    bytes.resize(N * 8, 0);
    let mut limbs = [0u64; N];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        *limb = u64::from_le_bytes(buf);
    }
    BigInt::from_sign_and_limbs(if n.is_negative() { 1 } else { 0 }, limbs)
}

/// Reduces an unbounded-precision integer into the same fixed-width
/// window our arithmetic lives in: truncate to `N·64` magnitude bits,
/// preserve the sign. Matches the wrap semantics of `BigInt::add` /
/// `BigInt::mul` on overflow.
fn truncate<const N: usize>(n: &NumBigInt) -> NumBigInt {
    let modulus = NumBigInt::one() << (N as u32 * 64);
    let mag = n.abs() % &modulus;
    if n.is_negative() && !mag.is_zero() {
        -mag
    } else {
        mag
    }
}

/// `BigInt::ZERO` may carry a stale `sign == 1` bit through some
/// operations; canonicalise before comparison so equality reflects
/// the mathematical value, not the in-memory representation.
fn canon<const N: usize>(mut a: BigInt<N>) -> BigInt<N> {
    a.normalize();
    a
}

proptest! {
    #[test]
    fn oracle_roundtrip_4(a in arb_bigint4()) {
        let lifted = to_num(a);
        let landed: BigInt<4> = from_num(&lifted);
        prop_assert_eq!(canon(a), canon(landed));
    }

    #[test]
    fn oracle_add(a in arb_bigint4(), b in arb_bigint4()) {
        let ours = canon(a + b);
        let theirs: BigInt<4> = from_num(&truncate::<4>(&(to_num(a) + to_num(b))));
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_sub(a in arb_bigint4(), b in arb_bigint4()) {
        let ours = canon(a - b);
        let theirs: BigInt<4> = from_num(&truncate::<4>(&(to_num(a) - to_num(b))));
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_neg(a in arb_bigint4()) {
        let ours = canon(-a);
        let theirs: BigInt<4> = from_num(&(-to_num(a)));
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_abs(a in arb_bigint4()) {
        let ours = canon(a.abs());
        let theirs: BigInt<4> = from_num(&to_num(a).abs());
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_mul_small(a in arb_small_bigint4(), b in arb_small_bigint4()) {
        // Inputs fit in `i32`, so the product fits in `i64 ≪ 2^256`:
        // no wrap, our output must equal the unbounded product.
        let ours = canon(a * b);
        let theirs: BigInt<4> = from_num(&(to_num(a) * to_num(b)));
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_mul_wrap(a in arb_bigint4(), b in arb_bigint4()) {
        // Full-width inputs do wrap mod `2^256`; verify the wrap matches.
        let ours = canon(a * b);
        let theirs: BigInt<4> = from_num(&truncate::<4>(&(to_num(a) * to_num(b))));
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_div_rem(a in arb_small_bigint4(), b in arb_small_bigint4()) {
        prop_assume!(!bool::from(b.is_zero()));
        let (q_ours, r_ours) = a.div_rem(&b);

        // `BigInt::div_rem` is Euclidean (`r >= 0`); `num_bigint` is
        // truncated. Lift the quotient and remainder by hand.
        let na = to_num(a);
        let nb = to_num(b);
        let (mut q_ref, mut r_ref) = na.div_rem(&nb);
        if r_ref.is_negative() {
            if nb.is_negative() {
                q_ref += 1;
                r_ref -= &nb;
            } else {
                q_ref -= 1;
                r_ref += &nb;
            }
        }

        prop_assert_eq!(canon(q_ours), from_num::<4>(&q_ref));
        prop_assert_eq!(canon(r_ours), from_num::<4>(&r_ref));
    }

    #[test]
    fn oracle_gcd(a in arb_small_bigint4(), b in arb_small_bigint4()) {
        // `num-integer::gcd` is non-negative and `gcd(0, 0) == 0`,
        // matching our convention.
        let ours = canon(a.gcd(&b));
        let theirs: BigInt<4> = from_num(&to_num(a).gcd(&to_num(b)));
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_shl_small_k(a in arb_bigint4(), k in 0u32..=63) {
        let ours = canon(a << k);
        let theirs: BigInt<4> = from_num(&truncate::<4>(&(to_num(a) << k)));
        prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn oracle_shr_small_k(a in arb_bigint4(), k in 0u32..=63) {
        // Arithmetic shift on a sign+magnitude `BigInt` rounds the
        // magnitude toward zero, then reattaches the sign. The
        // reference is `|a| >> k` with the original sign restored.
        let ours = canon(a >> k);
        let na = to_num(a);
        let shifted = na.abs() >> k;
        let signed = if na.is_negative() && !shifted.is_zero() {
            -shifted
        } else {
            shifted
        };
        prop_assert_eq!(ours, from_num::<4>(&signed));
    }
}
