//! MULX + dual ADCX/ADOX asm leaves for [`super::super::generic`]'s
//! radix-51 `Fp`.
//!
//! Not a standalone backend — these are asm replacements for the hot
//! leaves in `generic` ([`Fp::mul`], [`Fp::square`],
//! [`Fp::sum_of_2_products`], [`Fp::difference_of_2_products`]), called
//! when the build has both BMI2 (for MULX) and ADX (for ADCX/ADOX)
//! available.  Storage layout, public surface, and constants live in
//! `generic`; only the inner arithmetic is replaced.
//!
//! Mirrors the layering in the SQIsign C reference:
//! `src/gf/broadwell/lvl1/fp_asm.S` holds the asm (`fp_mul`, `fp_sqr`,
//! `fp2_mul_c0`, `fp2_mul_c1`, `fp_add`, `fp_sub`); the surrounding
//! `fp.c` wraps them and provides higher-level Fp behavior in C.
//!
//! [`Fp::mul`]: super::super::generic::Fp51::mul
//! [`Fp::square`]: super::super::generic::Fp51::square
//! [`Fp::sum_of_2_products`]: super::super::generic::Fp51::sum_of_2_products
//! [`Fp::difference_of_2_products`]: super::super::generic::Fp51::difference_of_2_products

// Contents land in subsequent commits:
//   - fp_mul    (Montgomery 5x5 schoolbook with interleaved P4 reduction)
//   - fp_sqr    (symmetric 5x5, ~30% fewer muls than fp_mul)
//   - fp2_mul_c0 / fp2_mul_c1  (fused Alg 8.1 sum/difference of 2 products)
