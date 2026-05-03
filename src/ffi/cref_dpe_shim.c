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
