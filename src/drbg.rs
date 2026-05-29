//! AES256-CTR-DRBG per [NIST SP 800-90A][sp80090a], §10.2.1 (no
//! derivation function, no reseed counter, no additional input).
//!
//! # Why 48 bytes of seed
//!
//! AES-256 has `keylen = 32` and `blocklen = 16`, so SP 800-90A
//! §10.2.1 Table 3 gives `seedlen = keylen + blocklen = 48` bytes
//! (384 bits). The NIST PQC test harness [`PQCgenKAT_sign.c`][kat]
//! passes a 48-byte `entropy_input` to `randombytes_init` for every
//! test vector — this is the `seed = ...` line in each KAT `.rsp`
//! file — and the SQIsign C reference implementation seeds its
//! AES256-CTR-DRBG the same way. Using `SEEDLEN = 48` here is what
//! lets `generate_derand` / `sign_derand` consume those KAT seeds
//! byte-for-byte and produce the same output as the C reference.
//!
//! The SQIsign spec ([Chapter 8][ch8]) specifies AES256-CTR-DRBG as
//! the pseudo-random number generator for optimized implementations.
//! Exposed as an [`RngCore`] so the generic sampling code in
//! `quaternions::lattice` can drive it directly.
//!
//! [sp80090a]: https://doi.org/10.6028/NIST.SP.800-90Ar1
//! [kat]: https://csrc.nist.gov/Projects/post-quantum-cryptography/post-quantum-cryptography-standardization/example-files
//! [ch8]: https://sqisign.org/spec/sqisign-20250707.pdf#chapter.8

use aes::{
    Aes256,
    cipher::{BlockEncrypt, KeyInit, generic_array::GenericArray},
};
use rand_core::{CryptoRng, Error, RngCore};

/// AES-256 key length in bytes.
const KEYLEN: usize = 32;
/// AES block length in bytes.
const BLOCKLEN: usize = 16;

/// Size in bytes of the 48-byte seed consumed by
/// [`Aes256CtrDrbg::new`]. Matches `crypto_bytes` in the NIST PQC
/// test harness (`randombytes_init` entropy_input).
pub(crate) const SEEDLEN: usize = KEYLEN + BLOCKLEN;

/// AES256-CTR-DRBG state: 32-byte Key + 16-byte V counter.
pub(crate) struct Aes256CtrDrbg {
    /// 32-byte AES-256 key (`Key` in SP 800-90A §10.2.1).
    key: [u8; KEYLEN],
    /// 16-byte counter (`V` in SP 800-90A §10.2.1).
    v: [u8; BLOCKLEN],
    /// Total bytes delivered to callers via `fill` since
    /// instantiation. Test-only probe for diffing byte consumption
    /// against the SQIsign C reference.
    ///
    /// Counts user-facing output, NOT internal AES block generation
    /// (which is always rounded up to a multiple of `BLOCKLEN = 16`
    /// per NIST SP 800-90A §10.2.1.5.2). Matches the C reference's
    /// `drbg_bytes_consumed` counter only when that counter is also
    /// instrumented at the user-facing output boundary; if the C
    /// side counts AES blocks instead, diffs will not be right on
    /// any odd-length `fill`.
    #[cfg(test)]
    consumed: u64,
}

impl Aes256CtrDrbg {
    /// Returns the total number of output bytes delivered to callers
    /// via `fill` since instantiation.
    ///
    /// See [`Self::consumed`] for the counter's precise semantics.
    #[cfg(test)]
    pub(crate) fn bytes_consumed(&self) -> u64 {
        self.consumed
    }
}

impl Aes256CtrDrbg {
    /// Instantiate the DRBG from 48 bytes of entropy input.
    ///
    /// Matches `randombytes_init(entropy_input, NULL, 256)` in the
    /// NIST reference code: start from an all-zero Key/V, then run
    /// CTR_DRBG_Update with `entropy_input` as the provided data.
    pub(crate) fn new(seed: &[u8; SEEDLEN]) -> Self {
        let mut d = Self {
            key: [0u8; KEYLEN],
            v: [0u8; BLOCKLEN],
            #[cfg(test)]
            consumed: 0,
        };
        d.update(Some(seed));
        d
    }

    /// CTR_DRBG_Update (SP 800-90A §10.2.1.2).
    fn update(&mut self, provided_data: Option<&[u8; SEEDLEN]>) {
        let mut temp = [0u8; SEEDLEN];
        let cipher = Aes256::new(GenericArray::from_slice(&self.key));
        for i in 0..(SEEDLEN / BLOCKLEN) {
            Self::increment_v(&mut self.v);
            let mut block = *GenericArray::from_slice(&self.v);
            cipher.encrypt_block(&mut block);
            temp[i * BLOCKLEN..(i + 1) * BLOCKLEN].copy_from_slice(&block);
        }
        if let Some(pd) = provided_data {
            for i in 0..SEEDLEN {
                temp[i] ^= pd[i];
            }
        }
        self.key.copy_from_slice(&temp[..KEYLEN]);
        self.v.copy_from_slice(&temp[KEYLEN..]);
    }

    /// CTR_DRBG_Generate (SP 800-90A §10.2.1.5), no additional input.
    fn randombytes(&mut self, out: &mut [u8]) {
        let cipher = Aes256::new(GenericArray::from_slice(&self.key));
        let mut i = 0;
        while i < out.len() {
            Self::increment_v(&mut self.v);
            let mut block = *GenericArray::from_slice(&self.v);
            cipher.encrypt_block(&mut block);
            let take = BLOCKLEN.min(out.len() - i);
            out[i..i + take].copy_from_slice(&block[..take]);
            i += take;
        }
        self.update(None);
        #[cfg(test)]
        {
            self.consumed += out.len() as u64;
        }
    }

    /// Big-endian increment of the 16-byte V counter.
    fn increment_v(v: &mut [u8; BLOCKLEN]) {
        for j in (0..BLOCKLEN).rev() {
            if v[j] == 0xFF {
                v[j] = 0;
            } else {
                v[j] = v[j].wrapping_add(1);
                return;
            }
        }
    }
}

impl RngCore for Aes256CtrDrbg {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0u8; 4];
        self.randombytes(&mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0u8; 8];
        self.randombytes(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.randombytes(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.randombytes(dest);
        Ok(())
    }
}

impl CryptoRng for Aes256CtrDrbg {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-answer cross-check against the SQIsign C reference.
    ///
    /// Seed the DRBG with the 48-byte entropy input from KAT vector
    /// 0 of `PQCsignKAT_353_SQIsign_lvl1.req` (commit pinned in
    /// `tests/fixtures/`) and draw 128 bytes. The expected prefix
    /// was captured with the C reference's `randombytes_init +
    /// randombytes` on the same seed; any divergence here means our
    /// DRBG — or the byte-at-a-time ordering in `randombytes` — is
    /// out of step with the reference, and every downstream KAT
    /// cross-check would drift for reasons unrelated to the
    /// SQIsign algorithm.
    #[test]
    fn matches_cref_seed_zero_first_128_bytes() {
        const SEED: [u8; SEEDLEN] = [
            0x06, 0x15, 0x50, 0x23, 0x4D, 0x15, 0x8C, 0x5E, 0xC9, 0x55, 0x95, 0xFE, 0x04, 0xEF,
            0x7A, 0x25, 0x76, 0x7F, 0x2E, 0x24, 0xCC, 0x2B, 0xC4, 0x79, 0xD0, 0x9D, 0x86, 0xDC,
            0x9A, 0xBC, 0xFD, 0xE7, 0x05, 0x6A, 0x8C, 0x26, 0x6F, 0x9E, 0xF9, 0x7E, 0xD0, 0x85,
            0x41, 0xDB, 0xD2, 0xE1, 0xFF, 0xA1,
        ];
        const EXPECTED_HEX: &str = "\
            7c9935a0b07694aa0c6d10e4db6b1add\
            2fd81a25ccb148032dcd739936737f2d\
            b505d7cfad1b497499323c8686325e47\
            92f267aafa3f87ca60d01cb54f29202a\
            3e784ccb7ebcdcfd45542b7f6af77874\
            2e0f4479175084aa488b3b74340678aa\
            38e22e9628b0a161fdeb0bd252173b9c\
            4e4cd0dbbd9cd3f10ef5fe5e4b034745";
        let mut drbg = Aes256CtrDrbg::new(&SEED);
        let mut buf = [0u8; 128];
        drbg.randombytes(&mut buf);
        let got: String = buf.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, EXPECTED_HEX);
    }
}
