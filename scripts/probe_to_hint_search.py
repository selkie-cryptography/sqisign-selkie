#!/usr/bin/env sage
# Empirical probe of `TorsionBasis::to_hint`'s bounded x-coord search.
#
# For each random Montgomery coefficient A in Fp^2 (with p the NIST-I
# SQIsign prime), measure how many tries `find_na_x_coord_with_hint`
# (NQR branch) or `find_nqr_factor_with_hint` (QR branch) consumes
# before its predicate succeeds, and compare against the Rust constant
# `FIND_X_COORD_MAX_TRIES = 2^16`.
#
# Run: sage scripts/probe_to_hint_search.py [N_SAMPLES]
#
# Output: per-branch min/max/mean/p99 over N random A, plus the worst
# A as hex for the Rust runner.

import sys
from sage.all import GF, PolynomialRing, ZZ, randint, sqrt

P = 5 * 2**248 - 1
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
MAX_TRIES = 1 << 16

Fp = GF(P)
Fp2 = GF(P**2, name='i', modulus=PolynomialRing(Fp, 'x').gen()**2 + 1)
i_gen = Fp2.gen()


# Phase 1: targeted "structured" candidates likely to bias the predicate.
def structured_candidates():
    cands = []
    # Small Fp scalars (not singular: skip 0, ±2).
    for n in [1, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 1 << 64, 1 << 128, 1 << 240]:
        cands.append(("scalar n=" + str(n), Fp2(n)))
    # Imaginary axis (pure i, scaled).
    for n in [1, 2, 3, 4, 5, 10, 100]:
        cands.append((f"i*{n}", i_gen * n))
    # Mixed: 1 + i*n.
    for n in [1, 2, 3, 4, 5, 10, 100]:
        cands.append((f"1+i*{n}", Fp2(1) + i_gen * n))
    # j = 0 curve (A² = 3): if 3 is a square mod p, A = ±√3 in Fp.
    three = Fp(3)
    if three.is_square():
        A = three.sqrt()
        cands.append(("j=0 (A=sqrt(3))", Fp2(A)))
    else:
        # 3 non-square in Fp: sqrt(3) lives in Fp2. Build it.
        A = Fp2(3).sqrt()
        cands.append(("j=0 (A=sqrt(3) in Fp2)", A))
    # j = 8000 curve (A² = 2). Pre-curated CM-ish picks.
    two = Fp(2)
    if two.is_square():
        cands.append(("A=sqrt(2)", Fp2(two.sqrt())))
    # A from quadratic twist of a small curve.
    cands.append(("A=p-1 (i.e. -1)", Fp2(P - 1)))
    return cands


def rand_fp2():
    return Fp2(randint(0, P - 1) + randint(0, P - 1) * i_gen)


def is_on_curve(x, A):
    t = x * (x * (x + A) + 1)
    return t.is_square()


def find_na_tries(A):
    """NQR branch: x_n = n*A, predicate is_on_curve(x_n, A) && !x_n.is_square()."""
    for n in range(1, MAX_TRIES + 1):
        x = n * A
        if is_on_curve(x, A) and not x.is_square():
            return n
    return None


def find_nqr_tries(A):
    """QR branch: x_n = -A / (1 + i*n), same predicate."""
    for n in range(1, MAX_TRIES + 1):
        z = Fp2(1) + i_gen * n
        x = -A / z
        if is_on_curve(x, A) and not x.is_square():
            return n
    return None


def fp2_hex(z):
    c = z.polynomial().coefficients(sparse=False)
    a = ZZ(c[0]) if len(c) > 0 else 0
    b = ZZ(c[1]) if len(c) > 1 else 0
    return f"a=0x{a:064x}, b=0x{b:064x}"


nqr_tries = []  # branch when A is non-square
qr_tries = []   # branch when A is square
nqr_worst = (0, None)
qr_worst = (0, None)
failures = []

print("=== phase 1: structured candidates ===")
for label, A in structured_candidates():
    if A == 0 or A == Fp2(2) or A == Fp2(P - 2):
        print(f"  {label}: skipped (A in {{0, ±2}})")
        continue
    branch = "QR" if A.is_square() else "NQR"
    n = find_nqr_tries(A) if A.is_square() else find_na_tries(A)
    print(f"  {label}: branch={branch} tries={n}")
    if n is None:
        failures.append((branch, A))

print()
print(f"=== phase 2: {N_SAMPLES} random samples ===")
for k in range(N_SAMPLES):
    A = rand_fp2()
    if A == 0:
        continue
    if A.is_square():
        n = find_nqr_tries(A)
        if n is None:
            failures.append(("QR", A))
            continue
        qr_tries.append(n)
        if n > qr_worst[0]:
            qr_worst = (n, A)
    else:
        n = find_na_tries(A)
        if n is None:
            failures.append(("NQR", A))
            continue
        nqr_tries.append(n)
        if n > nqr_worst[0]:
            nqr_worst = (n, A)
    if (k + 1) % max(1, N_SAMPLES // 20) == 0:
        print(f"  [{k + 1}/{N_SAMPLES}] nqr_max={nqr_worst[0]} qr_max={qr_worst[0]}", flush=True)


def stats(label, ts):
    if not ts:
        print(f"{label}: no samples")
        return
    ts_sorted = sorted(ts)
    n = len(ts_sorted)
    p99_idx = max(0, n * 99 // 100 - 1)
    print(
        f"{label}: n={n} min={ts_sorted[0]} max={ts_sorted[-1]} "
        f"mean={sum(ts_sorted) / n:.2f} p99={ts_sorted[p99_idx]}"
    )


print()
print("=== results ===")
stats("NQR branch (A non-square, find_na_x_coord)", nqr_tries)
stats("QR  branch (A square,     find_nqr_factor)", qr_tries)
print()
print(f"worst-case NQR: tries={nqr_worst[0]}, A: {fp2_hex(nqr_worst[1]) if nqr_worst[1] is not None else 'n/a'}")
print(f"worst-case QR:  tries={qr_worst[0]}, A: {fp2_hex(qr_worst[1]) if qr_worst[1] is not None else 'n/a'}")
print()
print(f"failures (search exhausted within {MAX_TRIES} tries): {len(failures)}")
for branch, A in failures:
    print(f"  {branch}: A = {fp2_hex(A)}")
