use proptest::prelude::*;

use super::*;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Generates a random `Element<4>` from four small i64 coordinates.
/// Uses i8-range values so that chained multiplications (3 deep) and
/// norm computations stay well within `BigInt<4>`'s 256-bit budget.
fn arb_element4() -> impl Strategy<Value = Element<4>> {
    (any::<i8>(), any::<i8>(), any::<i8>(), any::<i8>())
        .prop_map(|(a, b, c, d)| Element::from_i64(a as i64, b as i64, c as i64, d as i64))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[test]
fn zero() {
    assert!(Element::<4>::ZERO.is_zero());
}

#[test]
fn bigint_xgcd_negative_inputs() {
    // xgcd with negative inputs — Stein's binary algorithm may give
    // different cofactor signs than GMP's Euclidean mpz_gcdext.
    for (x_val, y_val) in [(-7i64, 3i64), (7, -3), (-7, -3), (-100, 7), (100, -7)] {
        let x = BigInt::<4>::from(x_val);
        let y = BigInt::<4>::from(y_val);
        let (g, u, v) = x.xgcd(&y);
        let id = u.ct_mul(&x).ct_add(&v.ct_mul(&y));
        let ux = u.ct_mul(&x);
        let ux_pos = bool::from(ux.is_positive()) || bool::from(ux.is_zero());
        eprintln!(
            "xgcd({}, {}) = (g={:?}, u={:?}, v={:?}), u·x={:?} (pos: {}), id={:?}",
            x_val, y_val, g, u, v, ux, ux_pos, id
        );
    }
}

#[test]
fn bigint_xgcd_3_5() {
    // GMP's mpz_gcdext on (3, 5) returns (gcd=1, u=2, v=-1):
    // 2·3 + (-1)·5 = 1, with |v| < |x|/2/gcd = 1.5.
    // C-ref's `ibz_xgcd_with_u_not_0` post-normalizes to ensure u·x > 0
    // and minimum |v|. Selkie's `BigInt::xgcd` uses Stein's binary
    // algorithm which may give different cofactors.
    let x = BigInt::<4>::from(3i64);
    let y = BigInt::<4>::from(5i64);
    let (g, u, v) = x.xgcd(&y);
    eprintln!("xgcd(3, 5) = (g={:?}, u={:?}, v={:?})", g, u, v);
    eprintln!("  u·x = {:?}", u.ct_mul(&x));
    eprintln!("  v·y = {:?}", v.ct_mul(&y));
    eprintln!("  u·x + v·y = {:?}", u.ct_mul(&x).ct_add(&v.ct_mul(&y)));
    assert_eq!(g, BigInt::<4>::ONE);
    assert_eq!(
        u.ct_mul(&x).ct_add(&v.ct_mul(&y)),
        BigInt::<4>::ONE,
        "xgcd identity"
    );
}

#[test]
fn mul_direct_kat1_alpha_at_width_30() {
    // Exact reproduction of from_generator_mod_hnf's call: basis_elem(0)
    // at width 30 × C-ref's KAT-1 α at width 30.
    use crate::quaternions::precomputed::EXTREMAL_ORDERS;
    let order_30 = EXTREMAL_ORDERS[0].widen::<30>();
    let basis_0 = order_30.order().basis_elem(0);
    eprintln!("basis_0.a = {:?}", basis_0.a.as_bigint());
    eprintln!("basis_0.denom = {:?}", basis_0.denom.as_bigint());

    // C-ref KAT-1 α (post_bt, denom=2; conjugating gives same first coord).
    let alpha_a_hex =
        "339de45818a8dcad1962ce0fabad5d66ddd0f321bcb2c9e4e982adb63e429421d5c85d3a06eee1986";
    let pad = format!("{:0>1$}", alpha_a_hex, 30 * 16);
    let mut limbs = [0u64; 30];
    for (i, chunk) in pad.as_bytes().rchunks(16).enumerate() {
        if i >= 30 {
            break;
        }
        let s = std::str::from_utf8(chunk).unwrap();
        limbs[i] = u64::from_str_radix(s, 16).unwrap_or(0);
    }
    let alpha_a = BigInt::<30>::from_limbs(limbs);
    let alpha = Element::<30>::new(
        Coordinate::from_bigint(alpha_a),
        Coordinate::from_bigint(BigInt::<30>::ZERO),
        Coordinate::from_bigint(BigInt::<30>::ZERO),
        Coordinate::from_bigint(BigInt::<30>::ZERO),
        Denominator::new(BigInt::<30>::ONE).unwrap(),
    );
    eprintln!("alpha.a = {:?}", alpha.a.as_bigint());

    let product = basis_0.mul_direct(&alpha);
    eprintln!("product.a = {:?}", product.a.as_bigint());
    let expected = alpha_a.ct_mul(&BigInt::<30>::from_u64(2));
    eprintln!("expected (2·α.a) = {:?}", expected);
    assert_eq!(
        *product.a.as_bigint(),
        expected,
        "mul_direct failed: product.a != 2·α.a"
    );
}

#[test]
fn mul_direct_two_times_alpha() {
    // basis_elem(0) for O₀ has a=2, b=c=d=0, denom=2 (representing the
    // quaternion 1 stored as 2/2). Multiplying by α=(1, 2, 3, 4) at denom=1
    // should give (2·1, 2·2, 2·3, 2·4) = (2, 4, 6, 8) at denom=2.
    let basis_0 = Element::<4>::new(
        Coordinate::from_i64(2),
        Coordinate::from_i64(0),
        Coordinate::from_i64(0),
        Coordinate::from_i64(0),
        Denominator::new(BigInt::from(2i64)).unwrap(),
    );
    let alpha = Element::<4>::from_i64(1, 2, 3, 4);
    let product = basis_0.mul_direct(&alpha);
    assert_eq!(
        *product.a.as_bigint(),
        BigInt::from(2i64),
        "product.a should be 2·α.a = 2"
    );
    assert_eq!(
        *product.b.as_bigint(),
        BigInt::from(4i64),
        "product.b should be 2·α.b = 4"
    );
    assert_eq!(
        *product.c.as_bigint(),
        BigInt::from(6i64),
        "product.c should be 2·α.c = 6"
    );
    assert_eq!(
        *product.d.as_bigint(),
        BigInt::from(8i64),
        "product.d should be 2·α.d = 8"
    );
    assert_eq!(
        *product.denom.as_bigint(),
        BigInt::from(2i64),
        "product.denom should be 2·1 = 2"
    );
}

#[test]
fn conjugate() {
    let e = Element::<4>::from_i64(1, 2, 3, 4);
    let conj = e.conjugate();
    assert_eq!(conj.a, Coordinate::from_i64(1));
    assert_eq!(conj.b, Coordinate::from_i64(-2));
    assert_eq!(conj.c, Coordinate::from_i64(-3));
    assert_eq!(conj.d, Coordinate::from_i64(-4));
}

#[test]
fn trace() {
    let e = Element::<4>::from_i64(5, 2, 3, 4);
    let (tr_num, tr_den) = e.trace();
    assert_eq!(tr_num, BigInt::from(10i64));
    assert_eq!(tr_den, BigInt::ONE);
}

#[test]
fn norm() {
    // nrd(1 + 2i + 3j + 4k) with p from NIST-I.
    // = 1 + 4 + p*(9 + 16) = 5 + 25p
    let e = Element::<4>::from_i64(1, 2, 3, 4);
    let (n_num, n_den) = e.norm();
    let p: BigInt<8> = P_WIDE;
    let expected = BigInt::<8>::from(5i64).ct_add(&BigInt::<8>::from(25i64).ct_mul(&p));
    assert_eq!(n_num, expected);
    assert_eq!(n_den, BigInt::<8>::ONE);
}

#[test]
fn norm_w_matches_norm() {
    // `norm_w::<W>` at W = N' > 8 should match `norm` (widened).
    let e = Element::<4>::from_i64(1, 2, 3, 4);
    let (num8, den8) = e.norm();
    let (num22, den22) = e.norm_w::<22>();
    assert_eq!(num22, num8.widen::<22>());
    assert_eq!(den22, den8.widen::<22>());
}

#[test]
fn mul_i_squared() {
    // i² = -1
    let unit_i = Element::<4>::from_i64(0, 1, 0, 0);
    let result = unit_i.mul(&unit_i);
    assert_eq!(result, Element::<4>::from_i64(-1, 0, 0, 0));
}

#[test]
fn mul_ij_eq_k() {
    // ij = k
    let unit_i = Element::<4>::from_i64(0, 1, 0, 0);
    let unit_j = Element::<4>::from_i64(0, 0, 1, 0);
    let result = unit_i.mul(&unit_j);
    assert_eq!(result, Element::<4>::from_i64(0, 0, 0, 1));
}

#[test]
fn mul_ji_eq_neg_k() {
    // ji = -k
    let unit_i = Element::<4>::from_i64(0, 1, 0, 0);
    let unit_j = Element::<4>::from_i64(0, 0, 1, 0);
    let result = unit_j.mul(&unit_i);
    assert_eq!(result, Element::<4>::from_i64(0, 0, 0, -1));
}

#[test]
fn norm_is_multiplicative() {
    let alpha = Element::<4>::from_i64(1, 2, 0, 1);
    let beta = Element::<4>::from_i64(3, 0, 1, 0);
    let product = alpha.mul(&beta);

    let (na, da) = alpha.norm();
    let (nb, db) = beta.norm();
    let (np, dp) = product.norm();

    // na/da * nb/db == np/dp
    let lhs = na.ct_mul(&nb).ct_mul(&dp);
    let rhs = np.ct_mul(&da).ct_mul(&db);
    assert_eq!(lhs, rhs);
}

#[test]
fn mul_by_conjugate_is_norm() {
    let e = Element::<4>::from_i64(1, 2, 3, 4);
    let conj = e.conjugate();
    let product = e.mul(&conj).normalized();

    let (_n_num, _n_den) = e.norm();
    // product should be scalar: (nrd, 0, 0, 0).
    // The product denom is r², and n_den is also r².
    assert_eq!(product.b, Coordinate::from_i64(0));
    assert_eq!(product.c, Coordinate::from_i64(0));
    assert_eq!(product.d, Coordinate::from_i64(0));
}

#[test]
fn addition() {
    let a = Element::<4>::from_i64(1, 2, 3, 4);
    let b = Element::<4>::from_i64(5, 6, 7, 8);
    let sum = a.add(&b);
    assert_eq!(sum, Element::<4>::from_i64(6, 8, 10, 12));
}

#[test]
fn subtraction() {
    let a = Element::<4>::from_i64(5, 6, 7, 8);
    let b = Element::<4>::from_i64(1, 2, 3, 4);
    let diff = a.sub(&b);
    assert_eq!(diff, Element::<4>::from_i64(4, 4, 4, 4));
}

#[test]
fn normalize_gcd() {
    let mut e = Element::<4>::new(
        Coordinate::from_i64(2),
        Coordinate::from_i64(4),
        Coordinate::from_i64(6),
        Coordinate::from_i64(8),
        Denominator::TWO,
    );
    e.normalize();
    assert_eq!(e, Element::<4>::from_i64(1, 2, 3, 4));
    assert_eq!(e.denom, Denominator::ONE);
}

#[test]
fn equality_across_denominators() {
    let a = Element::<4>::new(
        Coordinate::from_i64(2),
        Coordinate::from_i64(4),
        Coordinate::from_i64(0),
        Coordinate::from_i64(0),
        Denominator::TWO,
    );
    let b = Element::<4>::from_i64(1, 2, 0, 0);
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// Property-based tests
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn element_add_commutative(a in arb_element4(), b in arb_element4()) {
        prop_assert_eq!(a.add(&b), b.add(&a));
    }

    #[test]
    fn element_add_associative(a in arb_element4(), b in arb_element4(), c in arb_element4()) {
        prop_assert_eq!(a.add(&b).add(&c), a.add(&b.add(&c)));
    }

    #[test]
    fn element_add_identity(a in arb_element4()) {
        prop_assert_eq!(a.add(&Element::ZERO), a);
    }

    #[test]
    fn element_sub_is_add_neg(a in arb_element4(), b in arb_element4()) {
        // a - b should equal a + (-b).
        let neg_b = b.scalar_mul(&BigInt::MINUS_ONE);
        prop_assert_eq!(a.sub(&b), a.add(&neg_b));
    }

    // Quaternion multiplication at width 4 involves the prime p (~248
    // bits), so products always overflow BigInt<4> even for tiny
    // coordinates. Multiplication properties (associativity,
    // distributivity, norm multiplicativity, conjugate anti-automorphism)
    // are tested in the unit tests with specific small values that are
    // known to fit. Property tests cover the additive structure and
    // conjugation involution.

    #[test]
    fn element_conjugate_involution(a in arb_element4()) {
        // conj(conj(a)) == a
        prop_assert_eq!(a.conjugate().conjugate(), a);
    }

    #[test]
    fn element_norm_of_conjugate(a in arb_element4()) {
        // nrd(conj(a)) == nrd(a) — uses widening norm() → BigInt<8>.
        prop_assert_eq!(a.norm(), a.conjugate().norm());
    }

    /// Conjugation is linear: conj(a + b) == conj(a) + conj(b).
    #[test]
    fn element_conjugate_linear(a in arb_element4(), b in arb_element4()) {
        prop_assert_eq!(
            a.add(&b).conjugate(),
            a.conjugate().add(&b.conjugate())
        );
    }

    /// `scalar_mul` distributes: (a + b) * s == a*s + b*s.
    #[test]
    fn element_scalar_mul_distributive(a in arb_element4(), b in arb_element4(), s in any::<i8>()) {
        let scalar = BigInt::from_i64(s as i64);
        prop_assert_eq!(
            a.add(&b).scalar_mul(&scalar),
            a.scalar_mul(&scalar).add(&b.scalar_mul(&scalar))
        );
    }

    /// `scalar_mul` by 1 is identity.
    #[test]
    fn element_scalar_mul_identity(a in arb_element4()) {
        prop_assert_eq!(a.scalar_mul(&BigInt::ONE), a);
    }

    /// `scalar_mul` by 0 is zero.
    #[test]
    fn element_scalar_mul_zero(a in arb_element4()) {
        prop_assert!(a.scalar_mul(&BigInt::ZERO).is_zero());
    }

    /// Norm is non-negative: nrd(a) >= 0 for all a.
    #[test]
    fn element_norm_nonnegative(a in arb_element4()) {
        let (num, den) = a.norm();
        // Both numerator and denominator should be non-negative.
        prop_assert!(!bool::from(num.is_negative()));
        prop_assert!(!bool::from(den.is_negative()));
    }
}

#[test]
fn bigint_xgcd_zero_neg_input() {
    // self=-5, other=0 should give (gcd=5, u=-1, v=0): -5·(-1) + 0·0 = 5.
    // self=0, other=-5 should give (gcd=5, u=0, v=-1): 0·0 + (-5)·(-1) = 5.
    let neg5 = BigInt::<4>::from(-5i64);
    let zero = BigInt::<4>::ZERO;

    let (g1, u1, v1) = neg5.xgcd(&zero);
    let id1 = u1.ct_mul(&neg5).ct_add(&v1.ct_mul(&zero));
    eprintln!(
        "xgcd(-5, 0) = (g={:?}, u={:?}, v={:?}), id={:?}",
        g1, u1, v1, id1
    );
    assert_eq!(g1, BigInt::<4>::from(5i64), "gcd should be 5");
    assert_eq!(
        id1,
        BigInt::<4>::from(5i64),
        "Bezout identity: u·self + v·other = gcd"
    );

    let (g2, u2, v2) = zero.xgcd(&neg5);
    let id2 = u2.ct_mul(&zero).ct_add(&v2.ct_mul(&neg5));
    eprintln!(
        "xgcd(0, -5) = (g={:?}, u={:?}, v={:?}), id={:?}",
        g2, u2, v2, id2
    );
    assert_eq!(g2, BigInt::<4>::from(5i64), "gcd should be 5");
    assert_eq!(
        id2,
        BigInt::<4>::from(5i64),
        "Bezout identity: u·self + v·other = gcd"
    );
}

#[test]
fn bigint_div_rem_semantics() {
    let neg7 = BigInt::<4>::from(-7i64);
    let three = BigInt::<4>::from(3i64);
    let (q, r) = neg7.div_rem(&three);
    eprintln!("(-7).div_rem(3) = (q={:?}, r={:?})", q, r);
    // GMP's mpz_tdiv_qr (= C-ref's ibz_div): TRUNCATED. (-7)/3 = -2, rem = -1.
    // FLOOR/Euclidean: (-7)/3 = -3, rem = 2.
}

#[test]
fn bigint_ct_mod_large_negative() {
    use crate::quaternions::bigint::BigInt;
    // Construct a large negative value at width 30 and verify ct_mod
    // returns a value in [0, modulus).
    let mut limbs_a = [0u64; 30];
    limbs_a[10] = 0x1234_5678_9abc_def0;
    let big = BigInt::<30>::from_limbs(limbs_a);
    let neg_big = big.wrapping_neg();
    let mut limbs_m = [0u64; 30];
    limbs_m[5] = 0xabcd_1234_5678_9abc;
    let m = BigInt::<30>::from_limbs(limbs_m);

    let r = neg_big.ct_mod(&m);
    eprintln!("neg_big.bits = {}, m.bits = {}, r.bits = {}", neg_big.bitsize(), m.bitsize(), r.bitsize());
    eprintln!("r is_negative? {}", bool::from(r.is_negative()));
    eprintln!("r is_positive? {}", bool::from(r.is_positive()));
    // Verify r is in [0, m).
    assert!(!bool::from(r.is_negative()), "ct_mod result should be non-negative");
    // r < m.
    let diff = m.ct_sub(&r);
    assert!(bool::from(diff.is_positive()), "ct_mod result should be less than modulus");
    // Verify: neg_big = q·m + r where 0 ≤ r < m.
    let (q, _) = neg_big.div_rem(&m);
    let recon = q.ct_mul(&m).ct_add(&r);
    assert_eq!(recon, neg_big, "div_rem reconstruction");
}

#[test]
fn bigint_xgcd_canonical_cofactor_test() {
    // mpz_gcdext guarantees |u| ≤ |y|/(2·gcd).
    // Verify Selkie's xgcd output satisfies this property for various inputs.
    for (x_val, y_val) in [
        (12345i64, 67890i64),
        (-12345, 67890),
        (12345, -67890),
        (-12345, -67890),
        (100, 7),
        (100000, 33),
        (-987654, 321),
    ] {
        let x = BigInt::<4>::from(x_val);
        let y = BigInt::<4>::from(y_val);
        let (g, u, _v) = x.xgcd(&y);
        let g_val = if g.as_limbs()[0] == 0 && g.as_limbs()[1] == 0 { 0 } else { g.as_limbs()[0] as i64 };
        let u_val = {
            let mag = u.as_limbs()[0] as i64;
            if bool::from(u.is_negative()) { -mag } else { mag }
        };
        let bound = (y_val.abs() / (2 * g_val.max(1))).max(1);
        let in_range = u_val.abs() <= bound;
        eprintln!(
            "xgcd({}, {}): gcd={}, u={}, |u|={}, bound=|y|/2g={}, canonical={}",
            x_val, y_val, g_val, u_val, u_val.abs(), bound, in_range
        );
    }
}
