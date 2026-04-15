#!/usr/bin/env sage
"""Extract connecting ideals for all seven extremal orders.

Usage:
    cd scripts/precomp && sage extract_connecting_ideals.sage > connecting_ideals.json

Outputs JSON with the basis matrix, denominator, and norm of each
connecting ideal I_t (a left O₀-ideal of odd norm with right order
conjugate to O_t).
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

    # Compute norm: intersect the row space with the (1,0,0,0) axis
    norm_row = int_mat.row_space(ZZ).intersection(
        (ZZ**4).submodule([[1, 0, 0, 0]])
    ).basis()[0]
    norm = ZZ(abs(norm_row[0]))

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
