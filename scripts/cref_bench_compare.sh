#!/usr/bin/env bash
# Local same-machine wall-clock comparison of the SQIsign C reference
# against this crate, NIST-I (lvl1).
#
# Builds the C reference's `benchmark` app against a prebuilt C-ref tree
# and runs it in wall-clock milliseconds, then runs this crate's divan
# keygen/sign/verify bench, and prints both plus a parsed median-ms table.
#
# Wall-clock, not cycles: the C-ref `bench.h` cycle counter on Apple
# Silicon is the `kpc` interface, which needs root. Compiling `benchmark.c`
# with `-DNO_CYCLE_COUNTER` (and without `-DTARGET_ARM64`) routes it
# through `clock_gettime`, so `bench.h` reports milliseconds with no
# privilege. This matches the unit divan reports, making the two columns
# directly comparable.
#
# The fair peer is `GMP=mini`: mini-gmp + ref GF is portable C with no
# per-arch asm, the same tier as this crate's own bignum/field. `GMP=system`
# instead links GMP's hand-tuned asm (a much faster, unfair baseline).
#
# Assumes the C reference is checked out and built. Build it once with, e.g.:
#   cmake -S "$SQISIGN_C_REF" -B "$SQISIGN_C_REF/build-minigmp-ref" \
#     -DCMAKE_BUILD_TYPE=Release -DGMP_LIBRARY=MINI -DSQISIGN_BUILD_TYPE=ref
#   cmake --build "$SQISIGN_C_REF/build-minigmp-ref" -j
#
# Env:
#   SQISIGN_C_REF   C-ref checkout (default ~/src/github.com/SQISign/the-sqisign)
#   GMP             mini | system            (default mini)
#   ITERS           benchmark iterations     (default 30)
#   PROFDATA        optional .profdata path; builds this crate with that PGO profile
#
# Usage:
#   scripts/cref_bench_compare.sh
#   GMP=system ITERS=50 scripts/cref_bench_compare.sh
#   PROFDATA=/tmp/pgo.profdata scripts/cref_bench_compare.sh
set -euo pipefail

ref="${SQISIGN_C_REF:-$HOME/src/github.com/SQISign/the-sqisign}"
gmp="${GMP:-mini}"
iters="${ITERS:-30}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
crate="$(cd "$here/.." && pwd)"

case "$gmp" in
  mini)   build="$ref/build-minigmp-ref"; label="mini-gmp + ref GF (no asm, fair peer)"; gmp_link="";;
  system) build="$ref/build";             label="system GMP + ref GF (GMP asm)";          gmp_link="-L/opt/homebrew/lib -lgmp";;
  *) echo "GMP must be 'mini' or 'system', got '$gmp'" >&2; exit 1;;
esac

[ -d "$ref" ]   || { echo "C reference not found at $ref (set SQISIGN_C_REF)" >&2; exit 1; }
[ -d "$build" ] || { echo "C reference not built at $build (see header for cmake)" >&2; exit 1; }

# Build the C-ref benchmark for wall-clock ms (see header).
out="$(mktemp -d)/cref_bench_lvl1"
inc=(
  "$ref/include" "$ref/src/nistapi/lvl1" "$ref/src/signature/ref/include"
  "$ref/src/quaternion/ref/generic/include" "$ref/src/precomp/ref/lvl1/include"
  "$ref/src/mp/ref/generic/include" "$ref/src/ec/ref/include" "$ref/src/gf/ref/include"
  "$ref/src/common/generic/include" "$ref/src/mini-gmp"
)
inc_flags=(); for i in "${inc[@]}"; do inc_flags+=("-I$i"); done

libs=(
  "$build/src/libsqisign_lvl1_test_nistapi.a"
  "$build/src/libsqisign_lvl1_test.a"
  "$build/src/signature/ref/lvl1/libsqisign_signature_lvl1.a"
  "$build/src/verification/ref/lvl1/libsqisign_verification_lvl1.a"
  "$build/src/id2iso/ref/lvl1/libsqisign_id2iso_lvl1.a"
  "$build/src/quaternion/ref/generic/libsqisign_quaternion_generic.a"
  "$build/src/hd/ref/lvl1/libsqisign_hd_lvl1.a"
  "$build/src/ec/ref/lvl1/libsqisign_ec_lvl1.a"
  "$build/src/gf/ref/lvl1/libsqisign_gf_lvl1.a"
  "$build/src/mp/ref/generic/libsqisign_mp_generic.a"
  "$build/src/precomp/ref/lvl1/libsqisign_precomp_lvl1.a"
)
[ "$gmp" = mini ] && libs+=("$build/libGMP.a")
libs+=("$build/src/common/generic/libsqisign_common_test.a")

for l in "${libs[@]}"; do
  [ -f "$l" ] || { echo "missing C-ref archive: $l (rebuild $build)" >&2; exit 1; }
done

cc -O3 -std=c11 -w \
  -DENABLE_SIGN -DSQISIGN_VARIANT=lvl1 -DSQISIGN_BUILD_TYPE_REF \
  -DNO_CYCLE_COUNTER -DSQISIGN_TEST_REPS="$iters" -DNDEBUG \
  "${inc_flags[@]}" "$ref/apps/benchmark.c" -o "$out" "${libs[@]}" $gmp_link -lm

echo "### C reference -- $label, $iters iters (wall-clock ms)"
cref_out="$("$out" --iterations="$iters" 2>/dev/null)"
echo "$cref_out" | grep -E "keypair|sign |verify "

echo
echo "### sqisign-selkie (this crate)${PROFDATA:+ -- PGO ($PROFDATA)}, divan"
rustflags="--cfg aes_armv8"
[ -n "${PROFDATA:-}" ] && rustflags="$rustflags -Cprofile-use=$PROFDATA"
rust_out="$(cd "$crate" && RUSTFLAGS="$rustflags" cargo bench --bench sqisign \
  --features expose-internals,vartime -- 'keygen_derand|sign_derand|verify' 2>/dev/null)"
echo "$rust_out" | grep -E "keygen_derand|sign_derand|verify "

echo
echo "### median wall-clock, same machine (ms)"
CREF="$cref_out" RUST="$rust_out" LABEL="$label" python3 - <<'PY'
import os, re

def to_ms(tok):
    m = re.match(r"([\d.]+)\s*(ms|us|µs|ns|s)", tok)
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2)
    return v * {"s": 1e3, "ms": 1.0, "us": 1e-3, "µs": 1e-3, "ns": 1e-6}[u]

# C-ref rows: "  keypair  | average X | stddev Y | median Z | min ...".
cref = {}
for line in os.environ["CREF"].splitlines():
    for key, name in (("keypair", "keygen"), ("sign", "sign"), ("verify", "verify")):
        if re.match(rf"\s*{key}\b", line):
            mm = re.search(r"median\s+([\d.]+)", line)
            if mm:
                cref[name] = float(mm.group(1))

# divan rows: "name  fastest | slowest | median | ..." with unicode bars.
rust = {}
for line in os.environ["RUST"].splitlines():
    for key, name in (("keygen_derand", "keygen"), ("sign_derand", "sign"), ("verify", "verify")):
        if re.search(rf"\b{key}\b", line):
            vals = re.findall(r"[\d.]+\s*(?:ms|us|µs|ns|s)\b", line)
            if len(vals) >= 3:  # fastest, slowest, median, ...
                rust.setdefault(name, to_ms(vals[2]))

print(f"{'op':<8}{'C ref':>12}{'selkie':>12}   {'C ref = '+os.environ['LABEL']}")
for name in ("keygen", "sign", "verify"):
    c, r = cref.get(name), rust.get(name)
    if c is None or r is None:
        print(f"{name:<8}{'?':>12}{'?':>12}")
        continue
    ratio = r / c
    verdict = f"selkie {1/ratio:.2f}x faster" if ratio < 1 else f"selkie {ratio:.2f}x slower"
    print(f"{name:<8}{c:>10.2f}ms{r:>10.2f}ms   {verdict}")
PY
