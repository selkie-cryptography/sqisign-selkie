#!/usr/bin/env sage
"""Extract connecting ideals for all seven extremal orders.

Usage:
    cd scripts/precomp && sage extract_connecting_ideals.sage > connecting_ideals.json

Outputs JSON with the basis matrix, denominator, and *reduced norm*
of each connecting ideal I_t (a left O₀-ideal of odd norm with right
order conjugate to O_t).

The "norm" field matches the C reference's `quat_lideal_norm`
(`quaternion/ref/generic/ideal.c:7`), which is `sqrt([O₀ : I])`
computed from the lattice index. This is the same as Sage's
`QuaternionFractionalIdeal.norm()`.
"""

import json
import sys
import io

from sage.all import *

# maxorders.py prints to stdout; suppress it
_stdout = sys.stdout
sys.stdout = io.StringIO()
from maxorders import orders
sys.stdout = _stdout

result = []

for q, iso1q, mat1, ii, idl1, gamma in orders:
    denom = ZZ(idl1.denominator())
    int_mat = (idl1 * denom).change_ring(ZZ)

    # Reduced norm via lattice index, matching C ref's
    # `quat_lideal_norm` (`quaternion/ref/generic/ideal.c:7`):
    #
    #   N(I)² = [O₀ : I] = (covol(I) / covol(O₀))
    #         = (|det(I_int)| / |det(O₀_int)|) · (denom_O₀ / denom_I)⁴
    #
    # For the standard order {1, i, (i+j)/2, (1+k)/2} at denom 2,
    # |det(O₀_int)| = 4. With `denom_I = denom_O₀ = 2` (always true
    # for left-O₀ ideals here), this simplifies to N² = |det(I_int)|/4.
    # Equivalent to Sage's `I.norm()`.
    #
    # An earlier version of this script returned `abs(int_mat.row_space()
    # ∩ Z·(1,0,0,0)).basis()[0][0]` — the leading first-coordinate entry
    # of the HNF row space. For our HNF shape that entry equals
    # `denom · reduced_norm`, so the value was uniformly off by a
    # factor of `denom = 2`. C ref's `CONNECTING_IDEALS[t].norm`
    # stores the reduced norm, not the leading entry.
    det_O0_int = ZZ(4)  # |det(O₀ at denom 2)|; basis (1,i,(i+j)/2,(1+k)/2)
    O0_denom = ZZ(2)
    idx_num = abs(int_mat.det()) * (O0_denom ** 4)
    idx_den = det_O0_int * (denom ** 4)
    assert idx_num % idx_den == 0, "lattice index must be integral"
    idx = idx_num // idx_den
    norm = isqrt(idx)
    assert norm * norm == idx, "lattice index must be a perfect square (= norm²)"

    entry = {
        "q": int(q),
        "denom": str(denom),
        "norm": str(norm),
        "basis": [[str(int_mat[r][c]) for c in range(4)] for r in range(4)],
        "gamma_denom": str(gamma.denominator()),
        "gamma": [str(ZZ(g * gamma.denominator())) for g in gamma],
    }
    result.append(entry)

json.dump(result, sys.stdout, indent=2)
print()
