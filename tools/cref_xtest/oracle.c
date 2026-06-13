// SQIsign-Selkie cross-test oracle.
//
// Drives the C reference (the-sqisign) over arbitrary seeds so the Rust
// implementation can be differential-tested byte-for-byte beyond the 100
// fixed NIST KAT vectors. Reads one `seed_hex [msg_hex]` line per case
// from stdin (seed is 96 hex chars; the message is optional and defaults
// to bytes 0x00..0x1f). For each, reproduces the NIST KAT per-vector
// protocol (PQCgenKAT_sign.c):
//
//     randombytes_init(seed, NULL, 256);
//     crypto_sign_keypair(pk, sk);
//     crypto_sign(sm, &smlen, msg, msg_len, sk);
//
// and writes one output line `pk_hex sk_hex sig_hex`, where sig_hex is
// the first SIGNATURE_BYTES of sm (sqisign.c lays out sm = signature ||
// message). One DRBG is threaded keygen -> sign, exactly as the KAT
// framework and the Rust harness (Aes256CtrDrbg threaded through
// generate_with_rng then sign_with_rng) do, so sign reads the DRBG
// mid-stream after keygen.

#include "api.h"
#include "rng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEED_BYTES 48
#define DEFAULT_MSG_LEN 32
#define MAX_MSG_BYTES 4096

// Parses `len` hex bytes from `hex` into `out`. Returns 0 on success,
// -1 on a short/non-hex run.
static int
parse_hex(const char *hex, unsigned char *out, int len)
{
    for (int i = 0; i < len; i++) {
        unsigned int byte;
        if (sscanf(hex + 2 * i, "%2x", &byte) != 1)
            return -1;
        out[i] = (unsigned char) byte;
    }
    return 0;
}

static void
print_hex(const unsigned char *a, size_t len)
{
    for (size_t i = 0; i < len; i++)
        printf("%02x", a[i]);
}

int
main(void)
{
    unsigned char seed[SEED_BYTES];
    unsigned char msg[MAX_MSG_BYTES];
    unsigned char pk[CRYPTO_PUBLICKEYBYTES];
    unsigned char sk[CRYPTO_SECRETKEYBYTES];
    unsigned char sm[CRYPTO_BYTES + MAX_MSG_BYTES];
    unsigned long long smlen;

    // Input lines: `seed_hex [msg_hex]`. Allow generous slack for a long
    // optional message hex field.
    char line[2 * MAX_MSG_BYTES + 256];
    while (fgets(line, sizeof(line), stdin) != NULL) {
        if (parse_hex(line, seed, SEED_BYTES) != 0)
            continue;

        // Optional message: hex after a space; default 0x00..0x1f.
        int msg_len = DEFAULT_MSG_LEN;
        const char *sp = strchr(line, ' ');
        if (sp != NULL) {
            const char *mhex = sp + 1;
            int mhex_len = (int) strspn(mhex, "0123456789abcdefABCDEF");
            msg_len = mhex_len / 2;
            if (msg_len > MAX_MSG_BYTES || parse_hex(mhex, msg, msg_len) != 0) {
                puts("FAIL");
                fflush(stdout);
                continue;
            }
        } else {
            for (int i = 0; i < DEFAULT_MSG_LEN; i++)
                msg[i] = (unsigned char) i;
        }

        randombytes_init(seed, NULL, 256);

        // One "FAIL" line per failing seed keeps stdout aligned with the
        // input seeds for the batch harness; the Rust side asserts it
        // also fails on that seed.
        if (crypto_sign_keypair(pk, sk) != 0) {
            puts("FAIL");
            fflush(stdout);
            continue;
        }

        if (crypto_sign(sm, &smlen, msg, msg_len, sk) != 0) {
            puts("FAIL");
            fflush(stdout);
            continue;
        }

        print_hex(pk, CRYPTO_PUBLICKEYBYTES);
        putchar(' ');
        print_hex(sk, CRYPTO_SECRETKEYBYTES);
        putchar(' ');
        print_hex(sm, CRYPTO_BYTES);
        putchar('\n');
        fflush(stdout);
    }

    return 0;
}
