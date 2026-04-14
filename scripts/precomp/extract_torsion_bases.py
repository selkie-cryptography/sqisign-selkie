#!/usr/bin/env python3
"""Extract torsion basis data for ALL 7 extremal order curves from the
SQIsign C reference implementation.

Usage:
    python3 extract_torsion_bases.py > torsion_bases.json

Parses endomorphism_action.c from the C ref's Broadwell backend
(which stores Fp elements in Montgomery form with R = 2^256) and
converts to plain integers.

Output JSON structure:
  { "p": "0x...",
    "curves": [
      { "index": 0,
        "A_re": "0x...", "A_im": "0x...",
        "P_x_re": "0x...", "P_x_im": "0x...",
        "P_z_re": "0x...", "P_z_im": "0x...",
        "Q_x_re": "0x...", ...,
        "PmQ_x_re": "0x...", ... },
      ...
    ] }
"""

import json
import re
import sys
import os

# NIST-I prime
p = 5 * 2**248 - 1

# The C ref's Broadwell backend uses Montgomery form with R = 2^256.
R_BW = 2**256
R_BW_INV = pow(R_BW, p - 2, p)

NUM_CURVES = 7

# Layout of one curve_with_endomorphism_ring_t in Broadwell blocks:
#   ec_curve_t curve: A(re, im), C(re, im) = 4 blocks
#   ec_basis_t basis_even: P(x_re, x_im, z_re, z_im),
#                          Q(x_re, x_im, z_re, z_im),
#                          PmQ(x_re, x_im, z_re, z_im) = 12 blocks
#   ibz_mat_2x2_t action_gen2, action_gen3, action_gen4: NOT Fp blocks
#   bool: 1 byte (not a Broadwell block)
#
# Total Fp blocks per curve: 4 + 12 = 16
# Action matrices are ibz_t (big integers), stored separately.
# Total blocks: 7 * 16 = 112 for curve+basis, rest are ibz matrices.

# Actually, the ibz_mat_2x2_t entries are also stored as Broadwell
# blocks (4 limbs each), but they're plain integers not Montgomery.
# Each matrix: 4 entries, each 1 block = 4 blocks per matrix.
# 3 matrices per curve = 12 blocks.
# Total per curve: 16 (Fp) + 12 (ibz) = 28 blocks? But some ibz
# entries might be multi-block.
#
# Let's just count: 140 total blocks / 7 curves = 20 blocks per curve.
# That's 4 (curve) + 12 (basis) + 4 (something else).
# The bool is not a block. Let's check: 20 = 4 + 12 + 4? The action
# matrices might be stored differently.
#
# Actually, looking at the C struct more carefully:
# - ec_curve_t has just A (Fp2 = 2 blocks) and C (Fp2 = 2 blocks) = 4
# - ec_basis_t has P, Q, PmQ each (x, z) with x and z each Fp2 = 12
# - Action matrices are ibz_mat_2x2_t where ibz_t might be a single
#   Broadwell block (4 limbs = 256 bits) or might be GMP.
#
# For Broadwell, ibz_t is likely stored as a digit_t[NWORDS_ORDER]
# which is 4 uint64 = 1 block. So 3 matrices * 4 entries = 12 blocks.
# Total: 4 + 12 + 12 = 28. But 140 / 7 = 20. Hmm.
#
# Let me just parse by examining the actual block values.

BLOCKS_PER_CURVE = 20  # Empirical: 140 / 7


def from_montgomery(val):
    """Convert from Montgomery form (R = 2^256) to plain integer."""
    return (val * R_BW_INV) % p


def extract_broadwell_blocks(content):
    """Extract all Broadwell 64-bit limb blocks from the C source."""
    blocks = []
    in_broadwell = False
    for line in content.split("\n"):
        stripped = line.strip()
        if "SQISIGN_GF_IMPL_BROADWELL" in stripped:
            in_broadwell = True
            continue
        if in_broadwell:
            match = re.match(
                r"\{(0x[0-9a-fA-F]+(?:,\s*0x[0-9a-fA-F]+)*)\}", stripped
            )
            if match:
                limbs = [int(x.strip(), 16) for x in match.group(1).split(",")]
                val = sum(limb << (64 * i) for i, limb in enumerate(limbs))
                blocks.append(val)
                in_broadwell = False
            elif stripped.startswith("#"):
                in_broadwell = False
    return blocks


def to_hex(val):
    return f"0x{val:064x}"


def parse_curves(blocks):
    """Parse Broadwell blocks into per-curve torsion basis data."""
    curves = []

    for curve_idx in range(NUM_CURVES):
        base = curve_idx * BLOCKS_PER_CURVE
        b = blocks[base:]

        # Blocks 0-7: ec_curve_t {A(re, im), C(re, im), A24.x(re, im), A24.z(re, im)}
        A_re = from_montgomery(b[0])
        A_im = from_montgomery(b[1])
        C_re = from_montgomery(b[2])
        C_im = from_montgomery(b[3])
        # blocks 4-7: A24 (skip)

        # Blocks 8-19: ec_basis_t {P(x_re, x_im, z_re, z_im),
        #                          Q(x_re, x_im, z_re, z_im),
        #                          PmQ(x_re, x_im, z_re, z_im)}
        P_x_re = from_montgomery(b[8])
        P_x_im = from_montgomery(b[9])
        P_z_re = from_montgomery(b[10])
        P_z_im = from_montgomery(b[11])

        Q_x_re = from_montgomery(b[12])
        Q_x_im = from_montgomery(b[13])
        Q_z_re = from_montgomery(b[14])
        Q_z_im = from_montgomery(b[15])

        PmQ_x_re = from_montgomery(b[16])
        PmQ_x_im = from_montgomery(b[17])
        PmQ_z_re = from_montgomery(b[18])
        PmQ_z_im = from_montgomery(b[19])

        curves.append({
            "index": curve_idx,
            "A_re": to_hex(A_re),
            "A_im": to_hex(A_im),
            "C_re": to_hex(C_re),
            "C_im": to_hex(C_im),
            "P_x_re": to_hex(P_x_re),
            "P_x_im": to_hex(P_x_im),
            "P_z_re": to_hex(P_z_re),
            "P_z_im": to_hex(P_z_im),
            "Q_x_re": to_hex(Q_x_re),
            "Q_x_im": to_hex(Q_x_im),
            "Q_z_re": to_hex(Q_z_re),
            "Q_z_im": to_hex(Q_z_im),
            "PmQ_x_re": to_hex(PmQ_x_re),
            "PmQ_x_im": to_hex(PmQ_x_im),
            "PmQ_z_re": to_hex(PmQ_z_re),
            "PmQ_z_im": to_hex(PmQ_z_im),
        })

    return curves


def main():
    local_path = os.path.expanduser(
        "~/src/github.com/SQISign/the-sqisign"
        "/src/precomp/ref/lvl1/endomorphism_action.c"
    )
    with open(local_path) as f:
        content = f.read()

    blocks = extract_broadwell_blocks(content)
    assert len(blocks) == 140, f"Expected 140 blocks, got {len(blocks)}"

    curves = parse_curves(blocks)

    # Sanity check: E₀ has A=0, C=1
    e0 = curves[0]
    assert e0["A_re"] == to_hex(0), f"E₀ A_re should be 0, got {e0['A_re']}"
    assert e0["A_im"] == to_hex(0), f"E₀ A_im should be 0, got {e0['A_im']}"
    assert e0["C_re"] == to_hex(1), f"E₀ C_re should be 1, got {e0['C_re']}"
    assert e0["C_im"] == to_hex(0), f"E₀ C_im should be 0, got {e0['C_im']}"

    # Sanity check: all basis Z coordinates should be 1
    # (the C ref normalizes the basis to affine form before storing)
    for c in curves:
        assert c["P_z_re"] == to_hex(1), \
            f"Curve {c['index']} P_z_re should be 1"
        assert c["P_z_im"] == to_hex(0), \
            f"Curve {c['index']} P_z_im should be 0"

    result = {"p": to_hex(p), "curves": curves}
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
