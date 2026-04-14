#!/usr/bin/env sage
"""Verify extracted torsion basis data for all 7 extremal curves.

Usage:
    cd scripts/precomp && sage verify_torsion_bases.sage

Reads torsion_bases.json (from extract_torsion_bases.py) and verifies:
1. Each point is on its curve
2. Each point has exact order 2^f
3. P and Q are linearly independent (Weil pairing is primitive)
4. P-Q matches the stored value
"""

import json
import sys
from sage.all import *

with open("torsion_bases.json") as f:
    data = json.load(f)

p = ZZ(data["p"], 0)  # auto-detect base from 0x prefix
assert p == 5 * 2**248 - 1

Fp2 = GF(p**2, "i", modulus=[1, 0, 1])
i = Fp2.gen()

f = 248  # torsion exponent for NIST-I
cofactor = (p + 1) // 2**f
assert cofactor * 2**f == p + 1

num_ok = 0
num_fail = 0

for curve_data in data["curves"]:
    idx = curve_data["index"]
    A_re = ZZ(curve_data["A_re"], 0)
    A_im = ZZ(curve_data["A_im"], 0)
    A = Fp2(A_re) + i * Fp2(A_im)

    # Montgomery curve: y² = x³ + Ax² + x
    E = EllipticCurve(Fp2, [0, A, 0, 1, 0])
    E.set_order((p + 1) ** 2)

    def parse_point(prefix):
        x_re = ZZ(curve_data[f"{prefix}_x_re"], 0)
        x_im = ZZ(curve_data[f"{prefix}_x_im"], 0)
        z_re = ZZ(curve_data[f"{prefix}_z_re"], 0)
        z_im = ZZ(curve_data[f"{prefix}_z_im"], 0)
        x = Fp2(x_re) + i * Fp2(x_im)
        z = Fp2(z_re) + i * Fp2(z_im)
        if z == 0:
            return E(0)  # point at infinity
        return E.lift_x(x / z)

    try:
        P = parse_point("P")
        Q = parse_point("Q")
        PmQ = parse_point("PmQ")
    except (ValueError, TypeError) as e:
        print(f"Curve {idx} (A={A}): FAIL — point not on curve: {e}")
        num_fail += 1
        continue

    # Check order
    P.set_order(multiple=p + 1)
    Q.set_order(multiple=p + 1)

    ok = True
    if P.order() % 2**f != 0:
        print(f"Curve {idx}: FAIL — P order {P.order()} not divisible by 2^{f}")
        ok = False
    else:
        P_full = (P.order() // 2**f) * P
        P_full.set_order(2**f)
        if (2 ** (f - 1)) * P_full == E(0):
            print(f"Curve {idx}: FAIL — P has order < 2^{f}")
            ok = False
        else:
            P = P_full

    if Q.order() % 2**f != 0:
        print(f"Curve {idx}: FAIL — Q order {Q.order()} not divisible by 2^{f}")
        ok = False
    else:
        Q_full = (Q.order() // 2**f) * Q
        Q_full.set_order(2**f)
        if (2 ** (f - 1)) * Q_full == E(0):
            print(f"Curve {idx}: FAIL — Q has order < 2^{f}")
            ok = False
        else:
            Q = Q_full

    if ok:
        # Check linear independence
        e_pq = P.weil_pairing(Q, 2**f)
        if e_pq ** (2 ** (f - 1)) != -1:
            print(f"Curve {idx}: FAIL — P and Q not independent")
            ok = False

    if ok:
        # Check P-Q
        PmQ_computed = P - Q
        if PmQ[0] != PmQ_computed[0]:
            # x-only: might be P+Q instead of P-Q (sign ambiguity)
            PpQ_computed = P + Q
            if PmQ[0] == PpQ_computed[0]:
                print(f"Curve {idx}: NOTE — stored PmQ is actually P+Q (sign swap)")
            else:
                print(f"Curve {idx}: FAIL — PmQ x-coord doesn't match P-Q or P+Q")
                ok = False

    if ok:
        print(f"Curve {idx} (A_re=0x{A_re:064x}): OK")
        num_ok += 1
    else:
        num_fail += 1

print(f"\n{num_ok} OK, {num_fail} FAIL out of {len(data['curves'])} curves")
if num_fail > 0:
    sys.exit(1)
