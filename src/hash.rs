//! The SQIsign challenge hash function HASH.
//!
//! HASH produces a challenge value `chl` from the public key, a
//! j-invariant, and a message. For NIST-I:
//!
//! ```text
//! HASH = SHAKE256_{e_chl} ∘ SHAKE256_{256}^{∘(HASH_ITERATIONS−1)}
//! ```
//!
//! where e_chl = [`E_CHL`] = 122 bits and
//! [`HASH_ITERATIONS`] = 64.
//!
//! The repeated hashing ("grinding") compensates for the challenge
//! space being slightly smaller than the security parameter λ.
//!
//! See [§4.2.1] and [§10.2.5].
//!
//! [§4.2.1]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.2
//! [§10.2.5]: https://sqisign.org/spec/sqisign-20250707.pdf#section.10.2
//! [`E_CHL`]: crate::params::E_CHL
//! [`HASH_ITERATIONS`]: crate::params::HASH_ITERATIONS

use crate::{
    fields::fp2::Fp2,
    keys::VerifyingKey,
    params::{E_CHL, FP2_ENCODED_BYTES, HASH_ITERATIONS, SECURITY_BITS},
};

/// Number of bytes in the intermediate SHAKE256 output.
///
/// For NIST-I: 2λ/8 = 2·128/8 = 32 bytes = 256 bits, which is
/// exactly byte-aligned. This means no intermediate masking is
/// needed (see [`hash`] # Security).
const INTERMEDIATE_BYTES: usize = (2 * SECURITY_BITS as usize + 7) / 8;

/// Number of bytes in the final challenge output (⌈e_chl/8⌉ = 16).
pub(crate) const CHALLENGE_BYTES: usize = (E_CHL as usize + 7) / 8;

/// Bit mask for the top byte of the final challenge to ensure
/// the output is < 2^e_chl.
///
/// For NIST-I: e_chl = 122, 122 % 8 = 2, so mask = 0b11 = 0x03.
const CHALLENGE_TOP_MASK: u8 = {
    let bits = E_CHL as usize % 8;
    if bits == 0 {
        0xff
    } else {
        (1u8 << bits) - 1
    }
};

// Compile-time check: intermediate output is byte-aligned for NIST-I,
// so no intermediate masking is required. If this assertion fails,
// intermediate masking must be added back. See:
// https://github.com/SQIsign/the-sqisign/blob/dd133d7aca576c361a270c8e6434832535b42ecc/src/verification/ref/lvlx/common.c
const _: () = assert!(
    (2 * SECURITY_BITS as usize) % 8 == 0,
    "intermediate SHAKE256 output is not byte-aligned; \
     intermediate masking must be added for this parameter set"
);

/// Compute HASH(pk ‖ j(E) ‖ msg).
///
/// Hashes the verifying key, a j-invariant (of a commitment or
/// challenge curve), and a message into a challenge `chl` of
/// [`E_CHL`] = 122 bits (returned as [`CHALLENGE_BYTES`] = 16 bytes
/// with upper bits masked to zero).
///
/// All inputs are public — this function does not need to be
/// constant-time in the input values.
///
/// # Security
///
/// For NIST-I, the intermediate SHAKE256 output is exactly 256 bits
/// (= 2λ), which is byte-aligned. Therefore, no masking of
/// intermediate values is required — the full 32-byte output of each
/// SHAKE256 squeeze is fed directly into the next iteration.
///
/// The C reference implementation masks intermediate values to 2λ
/// bits after each squeeze. For NIST-I this mask is all-ones (a
/// no-op), but for a parameter set where 2λ is not a multiple of 8,
/// the mask would be necessary. If this code is adapted to a
/// different parameter set, the compile-time assertion above will
/// fail, and intermediate masking must be added back. See the C
/// reference [`hash_to_challenge`][c-ref] in
/// `verification/ref/lvlx/common.c`.
///
/// [c-ref]: https://github.com/SQIsign/the-sqisign/blob/dd133d7aca576c361a270c8e6434832535b42ecc/src/verification/ref/lvlx/common.c
///
/// See [§4.2.1] and [§4.4.2].
///
/// [§4.2.1]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.2
/// [§4.4.2]: https://sqisign.org/spec/sqisign-20250707.pdf#section.4.4
/// [`E_CHL`]: crate::params::E_CHL
pub(crate) fn hash(pk: &VerifyingKey, j: &Fp2, msg: &[u8]) -> [u8; CHALLENGE_BYTES] {
    // Serialize: j(pk) ‖ j(E) ‖ msg
    // https://sqisign.org/spec/sqisign-20250707.pdf#section.4.4
    let j_pk = pk.curve().j_invariant();
    let j_pk_bytes = j_pk.to_bytes();
    let j_bytes = j.to_bytes();

    // Initial SHAKE256: absorb all input, squeeze 32 bytes.
    let mut buf = [0u8; INTERMEDIATE_BYTES];
    {
        let mut input = Vec::with_capacity(2 * FP2_ENCODED_BYTES + msg.len());
        input.extend_from_slice(&j_pk_bytes);
        input.extend_from_slice(&j_bytes);
        input.extend_from_slice(msg);
        libcrux_sha3::shake256_ema(&mut buf, &input);
    }
    // No intermediate masking needed: 2λ = 256 is byte-aligned.

    // Grind: iterate SHAKE256 (HASH_ITERATIONS − 2) more times.
    // C reference: for (int i = 2; i < HASH_ITERATIONS; i++)
    for _ in 2..HASH_ITERATIONS {
        let input = buf;
        libcrux_sha3::shake256_ema(&mut buf, &input);
    }

    // Final SHAKE256: squeeze e_chl = 122 bits.
    let mut chl = [0u8; CHALLENGE_BYTES];
    libcrux_sha3::shake256_ema(&mut chl, &buf);
    chl[CHALLENGE_BYTES - 1] &= CHALLENGE_TOP_MASK;

    chl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants() {
        assert_eq!(INTERMEDIATE_BYTES, 32);
        assert_eq!(CHALLENGE_BYTES, 16);
        // e_chl = 122, 122 % 8 = 2, mask = 0b11 = 0x03
        assert_eq!(CHALLENGE_TOP_MASK, 0x03);
    }
}
