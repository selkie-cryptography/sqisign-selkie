/* C shim exposing the SQIsign C reference's mini-GMP + DPE primitives
 * as a stable C ABI for Selkie's FFI bit-equality tests.
 *
 * Compiled by build.rs only when the `ffi-cref-lll` Cargo feature is
 * enabled. Never built in production.
 *
 * The shim:
 *   1. Wraps `mini_mpz_get_d_2exp` so we can pass it raw little-endian
 *      limbs (matching Selkie's BigInt<N> layout) without exposing
 *      mpz_t through FFI.
 *   2. Surfaces dpe_t arithmetic individually so we can property-test
 *      each op separately.
 */

#include <stdint.h>
#include <string.h>

/* mini-gmp-extra.h includes mini-gmp.h (which defines __MINI_GMP_H__,
 * the trigger dpe.h checks) and typedefs mp_exp_t — both required
 * before dpe.h can be parsed under MINI_GMP. */
#include "mini-gmp-extra.h"
#include "dpe.h"

/* ------------------------------------------------------------------------- */
/* mpz_t builder from raw little-endian limbs.                               */
/* ------------------------------------------------------------------------- */

/* Build an mpz_t from `n_limbs` u64 limbs (little-endian, two's-complement
 * if `is_negative` is set: the limbs encode the magnitude only; sign is
 * applied separately, matching Selkie's BigInt sign-magnitude layout). */
static void
selkie_mpz_from_limbs(mpz_t out, const uint64_t *limbs, size_t n_limbs, int is_negative)
{
    /* mpz_import expects byte data; convert via byte buffer. */
    uint8_t buf[8 * 64]; /* up to 64 u64 limbs == 512 bytes */
    if (n_limbs * 8 > sizeof(buf)) { /* defensive */
        mpz_set_ui(out, 0);
        return;
    }
    /* little-endian limbs → little-endian bytes. */
    for (size_t i = 0; i < n_limbs; i++) {
        for (int b = 0; b < 8; b++)
            buf[i * 8 + b] = (uint8_t)(limbs[i] >> (8 * b));
    }
    mpz_import(out, n_limbs * 8, -1 /* LE words */, 1 /* word size 1B */, 0 /* native endian byte */, 0, buf);
    if (is_negative)
        mpz_neg(out, out);
}

/* ------------------------------------------------------------------------- */
/* Public shim functions                                                     */
/* ------------------------------------------------------------------------- */

/* Compute mini_mpz_get_d_2exp on a BigInt-shaped input. Writes the
 * returned mantissa (raw f64 bits) and exponent through the out
 * pointers. */
void
selkie_cref_to_dpe(const uint64_t *limbs,
                   size_t n_limbs,
                   int is_negative,
                   double *out_mantissa,
                   long *out_exp)
{
    mpz_t op;
    mpz_init(op);
    selkie_mpz_from_limbs(op, limbs, n_limbs, is_negative);
    *out_mantissa = mini_mpz_get_d_2exp(out_exp, op);
    mpz_clear(op);
}

/* Smoke-test the FFI build: just returns a known DPE value. */
void
selkie_cref_dpe_smoke(double *out_mantissa, long *out_exp)
{
    dpe_t x;
    dpe_init(x);
    dpe_set_d(x, 0.75);
    *out_mantissa = DPE_MANT(x);
    *out_exp = (long)DPE_EXP(x);
    dpe_clear(x);
}

/* ------------------------------------------------------------------------- */
/* Per-op DPE shims for bit-equality property tests.                          */
/*                                                                           */
/* All ops take inputs as (raw mantissa bits, exp) pairs and return outputs  */
/* the same way. The Rust side reconstructs the f64 mantissa via             */
/* f64::from_bits(...).                                                       */
/* ------------------------------------------------------------------------- */

static inline void
selkie_dpe_pack(dpe_t out, double mantissa, long exp)
{
    DPE_MANT(out) = mantissa;
    DPE_EXP(out) = (DPE_EXP_T)exp;
}

void
selkie_cref_dpe_mul(double a_m, long a_e, double b_m, long b_e,
                    double *out_m, long *out_e)
{
    dpe_t a, b, r;
    dpe_init(a); dpe_init(b); dpe_init(r);
    selkie_dpe_pack(a, a_m, a_e);
    selkie_dpe_pack(b, b_m, b_e);
    dpe_mul(r, a, b);
    *out_m = DPE_MANT(r);
    *out_e = (long)DPE_EXP(r);
    dpe_clear(a); dpe_clear(b); dpe_clear(r);
}

void
selkie_cref_dpe_add(double a_m, long a_e, double b_m, long b_e,
                    double *out_m, long *out_e)
{
    dpe_t a, b, r;
    dpe_init(a); dpe_init(b); dpe_init(r);
    selkie_dpe_pack(a, a_m, a_e);
    selkie_dpe_pack(b, b_m, b_e);
    dpe_add(r, a, b);
    *out_m = DPE_MANT(r);
    *out_e = (long)DPE_EXP(r);
    dpe_clear(a); dpe_clear(b); dpe_clear(r);
}

void
selkie_cref_dpe_sub(double a_m, long a_e, double b_m, long b_e,
                    double *out_m, long *out_e)
{
    dpe_t a, b, r;
    dpe_init(a); dpe_init(b); dpe_init(r);
    selkie_dpe_pack(a, a_m, a_e);
    selkie_dpe_pack(b, b_m, b_e);
    dpe_sub(r, a, b);
    *out_m = DPE_MANT(r);
    *out_e = (long)DPE_EXP(r);
    dpe_clear(a); dpe_clear(b); dpe_clear(r);
}

void
selkie_cref_dpe_div(double a_m, long a_e, double b_m, long b_e,
                    double *out_m, long *out_e)
{
    dpe_t a, b, r;
    dpe_init(a); dpe_init(b); dpe_init(r);
    selkie_dpe_pack(a, a_m, a_e);
    selkie_dpe_pack(b, b_m, b_e);
    dpe_div(r, a, b);
    *out_m = DPE_MANT(r);
    *out_e = (long)DPE_EXP(r);
    dpe_clear(a); dpe_clear(b); dpe_clear(r);
}

int
selkie_cref_dpe_cmp(double a_m, long a_e, double b_m, long b_e)
{
    dpe_t a, b;
    dpe_init(a); dpe_init(b);
    selkie_dpe_pack(a, a_m, a_e);
    selkie_dpe_pack(b, b_m, b_e);
    int r = dpe_cmp(a, b);
    dpe_clear(a); dpe_clear(b);
    return r;
}

int
selkie_cref_dpe_cmp_d(double a_m, long a_e, double d)
{
    dpe_t a;
    dpe_init(a);
    selkie_dpe_pack(a, a_m, a_e);
    int r = dpe_cmp_d(a, d);
    dpe_clear(a);
    return r;
}

void
selkie_cref_dpe_round(double a_m, long a_e, double *out_m, long *out_e)
{
    dpe_t a, r;
    dpe_init(a); dpe_init(r);
    selkie_dpe_pack(a, a_m, a_e);
    dpe_round(r, a);
    *out_m = DPE_MANT(r);
    *out_e = (long)DPE_EXP(r);
    dpe_clear(a); dpe_clear(r);
}

/* ------------------------------------------------------------------------- */
/* quat_lll_core shim — runs the C ref's full L² LLL on a 4×4 Gram + basis.  */
/* ------------------------------------------------------------------------- */

/* `quat_lll_core` is declared in `lll_internals.h`, but that header
 * pulls in `quaternion.h` → `intbig.h` → `sqisign_namespace.h` plus a
 * pile of transitive .c implementations (`finit.c` etc.). To avoid
 * pulling in the rest of the C ref's quaternion library just for the
 * matrix init/finalize helpers, we declare `quat_lll_core` directly
 * with the underlying `mpz_t[4][4]` type — equivalent under the
 * typedef `ibz_mat_4x4_t = mpz_t[4][4]` (`quaternion.h:52`). */
extern void quat_lll_core(mpz_t (*G)[4][4], mpz_t (*basis)[4][4]);

/* Matrix layout: row-major, 16 entries per matrix. Each entry is a
 * fixed-width little-endian bigint of `stride` u64 limbs plus a
 * separate sign byte.
 *
 * Run quat_lll_core in place: G and basis are both read-write.
 * On entry, *_limbs contain the input matrices encoded as above.
 * On exit, *_limbs contain the reduced matrices.
 *
 * `stride` must be large enough to hold every Gram entry the LLL
 * produces (covolume-bounded, so stride=64 = 4096 bits is safe for
 * any SQIsign-sized lattice).
 */
void
selkie_cref_quat_lll_core(uint64_t *gram_limbs,
                          int *gram_signs,
                          uint64_t *basis_limbs,
                          int *basis_signs,
                          size_t stride)
{
    mpz_t G[4][4], B[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mpz_init(G[i][j]);
            mpz_init(B[i][j]);
        }
    }

    /* Decode inputs. */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int idx = i * 4 + j;
            const uint64_t *gl = gram_limbs + (size_t)idx * stride;
            const uint64_t *bl = basis_limbs + (size_t)idx * stride;
            selkie_mpz_from_limbs(G[i][j], gl, stride, gram_signs[idx]);
            selkie_mpz_from_limbs(B[i][j], bl, stride, basis_signs[idx]);
        }
    }

    /* Run LLL. */
    quat_lll_core(&G, &B);

    /* Encode outputs. */
    mpz_t tmp;
    mpz_init(tmp);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int idx = i * 4 + j;
            uint64_t *gl = gram_limbs + (size_t)idx * stride;
            uint64_t *bl = basis_limbs + (size_t)idx * stride;
            for (size_t k = 0; k < stride; k++) {
                gl[k] = 0;
                bl[k] = 0;
            }
            /* G[i][j] sign + magnitude. */
            gram_signs[idx] = (mpz_sgn(G[i][j]) < 0) ? 1 : 0;
            mpz_abs(tmp, G[i][j]);
            if (mpz_sgn(tmp) != 0) {
                size_t count = 0;
                mpz_export(gl, &count, -1, sizeof(uint64_t), 0, 0, tmp);
                /* count ≤ stride by precondition; otherwise truncated. */
            }
            /* B[i][j] sign + magnitude. */
            basis_signs[idx] = (mpz_sgn(B[i][j]) < 0) ? 1 : 0;
            mpz_abs(tmp, B[i][j]);
            if (mpz_sgn(tmp) != 0) {
                size_t count = 0;
                mpz_export(bl, &count, -1, sizeof(uint64_t), 0, 0, tmp);
            }
        }
    }
    mpz_clear(tmp);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mpz_clear(G[i][j]);
            mpz_clear(B[i][j]);
        }
    }
}

/* dpe_get_z(out_mpz, x): converts a DPE to an integer (round-to-nearest).
 * We take a DPE input and write the resulting magnitude as little-endian
 * limbs into out_limbs[], plus a sign flag. n_limbs is the buffer size;
 * actual_limbs returns how many limbs were used (any leading zero limbs
 * not counted). */
void
selkie_cref_dpe_get_z(double a_m, long a_e,
                      uint64_t *out_limbs, size_t n_limbs,
                      int *out_is_negative,
                      size_t *out_actual_limbs)
{
    dpe_t a;
    dpe_init(a);
    selkie_dpe_pack(a, a_m, a_e);
    mpz_t z;
    mpz_init(z);
    dpe_get_z(z, a);
    *out_is_negative = (mpz_sgn(z) < 0) ? 1 : 0;

    /* Take absolute value, then export little-endian limbs. */
    mpz_t zabs;
    mpz_init(zabs);
    mpz_abs(zabs, z);
    /* Zero out output buffer first. */
    for (size_t i = 0; i < n_limbs; i++) out_limbs[i] = 0;
    if (mpz_sgn(zabs) == 0) {
        *out_actual_limbs = 0;
    } else {
        size_t count = 0;
        mpz_export(out_limbs, &count, -1 /* LSB first */, sizeof(uint64_t),
                   0 /* native endian */, 0 /* full words */, zabs);
        *out_actual_limbs = count;
    }
    mpz_clear(zabs);
    mpz_clear(z);
    dpe_clear(a);
}
