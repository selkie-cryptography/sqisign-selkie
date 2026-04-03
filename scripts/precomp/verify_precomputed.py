#!/usr/bin/env sage
"""Recompute extremal order data and output as JSON for verification
by the Rust test harness.

Usage:
    cd scripts/precomp && sage verify_precomputed.py

Adapted from the SQIsign reference implementation:
https://github.com/SQISign/the-sqisign

Original authors: the SQIsign team (Maria Corte-Real Santos, Joy Hall,
Michael Meyer, Giacomo Pope, Damien Robert, et al.)
Licensed under the Apache License, Version 2.0.
"""

import json
import sys

from sage.all import *

from sage.misc.banner import require_version
if not require_version(10, 5, print_message=True):
    exit('')

from parameters import p, num_orders as num

Quat1, (i,j,k) = QuaternionAlgebra(-1, -p).objgens()
assert Quat1.discriminant() == p

O0mat = matrix([list(g) for g in [Quat1(1), i, (i+j)/2, (1+k)/2]])
O0 = Quat1.quaternion_order(list(O0mat))

orders = [(1, O0mat, i)]

q = ZZ(1)
while len(orders) < num:
    q = next_prime(q)
    if q % 4 != 1:
        continue

    Quatq, (ii,jj,kk) = QuaternionAlgebra(-q, -p).objgens()
    if Quatq.discriminant() != p:
        continue

    x, y = QuadraticForm(QQ, 2, [1,0,p]).solve(q)
    gamma = x + j*y
    assert gamma.reduced_norm() == q
    ims1 = [Quat1(1), i*gamma, j, k*gamma]
    assert ims1[1]**2 == -q
    iso1q = ~matrix(map(list, ims1))

    r = min(map(ZZ, Mod(-p, 4*q).sqrt(all=True)))
    basq = [Quatq(1), ii, (1 + jj) / 2, (r + jj) * ii / 2 / q]

    Oq = Quatq.quaternion_order(basq)
    assert Oq.discriminant() == p

    mat1 = matrix(map(list, basq)) * ~iso1q
    O1 = Quat1.quaternion_order(list(mat1))
    assert O1.discriminant() == p
    assert j in O1

    orders.append((q, mat1, ims1[1]))

# Output JSON
result = {"p": str(hex(p)), "orders": []}

for q_val, mat, z_elt in orders:
    denoms = [entry.denominator() for row in mat for entry in row]
    denom = lcm(denoms)
    int_basis = [[str(ZZ(entry * denom)) for entry in row] for row in mat]

    z_coords = list(z_elt)
    z_denoms = [c.denominator() for c in z_coords]
    z_denom = lcm(z_denoms)
    z_int = [str(ZZ(c * z_denom)) for c in z_coords]

    result["orders"].append({
        "q": int(q_val),
        "basis_denom": str(denom),
        "basis": int_basis,
        "z_denom": str(z_denom),
        "z_coord": z_int,
        "t_denom": "1",
        "t_coord": ["0", "0", "1", "0"],
    })

# Also compute the torsion basis for E₀.
# E₀: y² = x³ + x over F_{p²}.
Fp = GF(p)
Rpoly = PolynomialRing(Fp, 'x')
xvar = Rpoly.gen()
Fp2 = GF(p**2, 'ii', modulus=xvar**2+1)
ii = Fp2.gen()
E0 = EllipticCurve(Fp2, [1, 0])  # y² = x³ + x

# The order of E₀ over F_{p²} is (p+1)² for supersingular curves
# with p ≡ 3 mod 4.
f = (p + 1).valuation(2)
cofactor = (p + 1) // 2**f

# Generate the torsion basis deterministically.
# Find two linearly independent points of full 2**f order.
def find_torsion_point(E, cofactor, f, start=1):
    """Find a point of exact order 2**f on E."""
    for seed in range(start, start + 10000):
        # Deterministic x-coordinate from seed
        x_val = Fp2(seed)
        try:
            P = E.lift_x(x_val)
        except ValueError:
            continue
        # Clear cofactor
        P = cofactor * P
        if P == E(0):
            continue
        # Check full order: [2^{f-1}]P ≠ O
        Q = P
        for _ in range(f - 1):
            Q = 2 * Q
        if Q != E(0):
            return P, seed
    return None, None

P, seed_p = find_torsion_point(E0, cofactor, f, start=1)
assert P is not None, "Failed to find P"
assert (2**f) * P == E0(0), "P does not have order dividing 2**f"

# For Q, use x-coordinates with nonzero imaginary part.
Q = None
seed_q = None
for s1 in range(1, 100):
    for s2 in range(1, 100):
        x_val = Fp2(s1) + s2 * ii
        try:
            Qc = E0.lift_x(x_val)
        except ValueError:
            continue
        Qc = cofactor * Qc
        if Qc == E0(0):
            continue
        # Check full order
        Qtest = Qc
        for _ in range(f - 1):
            Qtest = 2 * Qtest
        if Qtest == E0(0):
            continue
        Q = Qc
        seed_q = (s1, s2)
        break
    if Q is not None:
        break

assert Q is not None, "Failed to find Q"
assert (2**f) * Q == E0(0), "Q does not have order dividing 2**f"

# Verify independence: [2^{f-1}]P ≠ ±[2^{f-1}]Q.
# Since P has real x and Q has imaginary x, they're in different
# Frobenius eigenspaces and therefore independent.
P_half = P
Q_half = Q
for _ in range(f - 1):
    P_half = 2 * P_half
    Q_half = 2 * Q_half
assert P_half != Q_half and P_half != -Q_half, "P and Q are not independent"

# Extract x-coordinates as (re, im) in F_p
def fp2_to_parts(z):
    """Extract real and imaginary parts of z ∈ F_{p²}."""
    coeffs = z.polynomial().padded_list(2)
    return str(ZZ(coeffs[0])), str(ZZ(coeffs[1]))

px_re, px_im = fp2_to_parts(P[0])
qx_re, qx_im = fp2_to_parts(Q[0])

result["torsion_basis"] = {
    "f": int(f),
    "cofactor": int(cofactor),
    "P_x_re": px_re,
    "P_x_im": px_im,
    "Q_x_re": qx_re,
    "Q_x_im": qx_im,
    "P_seed": int(seed_p),
    "Q_seed": str(seed_q),
}

json.dump(result, sys.stdout)
print()
