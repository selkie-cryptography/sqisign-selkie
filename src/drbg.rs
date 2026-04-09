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

const KEYLEN: usize = 32;
const BLOCKLEN: usize = 16;

/// Size in bytes of the 48-byte seed consumed by
/// [`Aes256CtrDrbg::new`]. Matches `crypto_bytes` in the NIST PQC
/// test harness (`randombytes_init` entropy_input).
pub(crate) const SEEDLEN: usize = KEYLEN + BLOCKLEN;

/// AES256-CTR-DRBG state: 32-byte Key + 16-byte V counter.
pub(crate) struct Aes256CtrDrbg {
    key: [u8; KEYLEN],
    v: [u8; BLOCKLEN],
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
