//! Uniform random sampling on [`BigInt<N>`][super::BigInt] via
//! rejection on a top-bit-aligned mask.

use rand_core::RngCore;

use super::BigInt;

impl<const N: usize> BigInt<N> {
    /// Sample a uniform integer in `[a, b]` with rejection sampling on
    /// a top-bit-aligned mask, matching the C reference's
    /// `ibz_rand_interval`
    /// (`quaternion/ref/generic/intbig.c:413-473`).
    ///
    /// This is the byte-stream contract `quat_represent_integer` uses
    /// to draw `z` and `t`. Replicating it byte-for-byte is required
    /// for KAT byte-match against the C reference: with the same DRBG
    /// state, each call must consume the same number of bytes per
    /// rejection cycle.
    ///
    /// Per call: computes `bmina = b − a`, then loops drawing
    /// `ceil(bit_length(bmina) / 8)` bytes from `rng`, masking the top
    /// byte to exactly `bit_length(bmina)` bits, decoding little-
    /// endian, and accepting iff the value is `≤ bmina`. On accept,
    /// returns `bmina_value + a`. On `a == b`, returns `a` immediately
    /// without consuming any randomness.
    ///
    /// # Constant-time
    ///
    /// Variable-time. Used only for `represent_integer`'s `(z, t)`
    /// sampling, which the C reference also runs in variable time.
    /// `TODO(ct)`: revisit when the keygen path is hardened against
    /// timing side-channels (see `project_ct_plan.md`).
    ///
    /// # Panics
    ///
    /// Debug-asserts `a ≤ b`.
    pub fn rand_interval<R: RngCore>(rng: &mut R, a: &Self, b: &Self) -> Self {
        debug_assert!(a <= b, "rand_interval: a must be ≤ b");
        let bmina = b.ct_sub(a);
        if bool::from(bmina.is_zero()) {
            return *a;
        }
        let len_bits = bmina.bitsize();
        let len_bytes = len_bits.div_ceil(8) as usize;

        // Top-byte mask: keep only the low `len_bits % 8` bits of the
        // last sampled byte. When `len_bits` is byte-aligned, the byte
        // is unmasked.
        let top_byte_bits = (len_bits % 8) as u8;
        let top_mask: u8 = if top_byte_bits == 0 {
            0xFF
        } else {
            (1u8 << top_byte_bits) - 1
        };

        let mut buf = vec![0u8; len_bytes];
        loop {
            rng.fill_bytes(&mut buf);
            buf[len_bytes - 1] &= top_mask;
            // Decode little-endian into `BigInt<N>` limbs.
            let mut limbs = [0u64; N];
            for (i, chunk) in buf.chunks(8).enumerate() {
                if i >= N {
                    break;
                }
                let mut bytes = [0u8; 8];
                bytes[..chunk.len()].copy_from_slice(chunk);
                limbs[i] = u64::from_le_bytes(bytes);
            }
            let val = Self::from_limbs(limbs);
            if val <= bmina {
                return val.ct_add(a);
            }
        }
    }
}
