#!/usr/bin/env python3
"""Extract E₀ torsion basis x-coordinates from the SQIsign C reference
implementation at a pinned commit, and output as JSON for the Rust
cross-check test.

Usage:
    python3 extract_e0_basis.py

Fetches the raw e0_basis.c file from the pinned commit on GitHub and
parses the 64-bit limbs from the C ref's Broadwell backend (which
stores Fp elements in Montgomery form with R = 2^256). Outputs the
plain integer x-coordinates.

The pinned commit ensures reproducibility regardless of upstream changes.
"""

import json
import re
import sys
import urllib.request

# Pinned commit of the SQIsign reference implementation.
COMMIT = "91e9e464fe5400192d13e1f9240cbf180200a103"
URL = f"https://raw.githubusercontent.com/SQISign/the-sqisign/{COMMIT}/src/precomp/ref/lvl1/e0_basis.c"

# NIST-I prime
p = 5 * 2**248 - 1

# The C ref's Broadwell backend uses Montgomery form with R = 2^256.
R_BW = 2**256
R_BW_INV = pow(R_BW, p - 2, p)


def fetch_file(url):
    """Fetch a file from a URL."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")


def extract_broadwell_blocks(content):
    """Extract all Broadwell 64-bit limb arrays from C source."""
    blocks = re.findall(
        r'SQISIGN_GF_IMPL_BROADWELL\)\n\{([^}]+)\}', content
    )
    values = []
    for block in blocks:
        limbs = [int(x.strip(), 16) for x in block.split(",")]
        # Reconstruct the 256-bit value from four LE limbs.
        val = sum(limb << (64 * i) for i, limb in enumerate(limbs))
        values.append(val)
    return values


def broadwell_to_plain(mont_val):
    """Convert from Montgomery form (R = 2^256) to plain integer."""
    return (mont_val * R_BW_INV) % p


def int_to_le_bytes_hex(val):
    """Convert an integer to a hex string of 32 LE bytes."""
    return val.to_bytes(32, "little").hex()


def main():
    content = fetch_file(URL)
    values = extract_broadwell_blocks(content)

    if len(values) != 4:
        print(f"Expected 4 Broadwell blocks, got {len(values)}", file=sys.stderr)
        sys.exit(1)

    # Order: PX_RE, PX_IM, QX_RE, QX_IM
    names = ["P_x_re", "P_x_im", "Q_x_re", "Q_x_im"]
    result = {"commit": COMMIT, "url": URL}

    for i, name in enumerate(names):
        plain = broadwell_to_plain(values[i])
        result[name] = str(plain)
        result[name + "_bytes_le"] = int_to_le_bytes_hex(plain)

    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
