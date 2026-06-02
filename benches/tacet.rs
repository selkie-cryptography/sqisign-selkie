//! Tacet constant-time tests with attacker-model-aware analysis.
//!
//! Complements the dudect bench with exploitability assessments per
//! attacker model. Each test reports leak probability and effect size
//! rather than a raw t-statistic.
//!
//! Run with: `cargo bench --bench tacet --features expose-internals`
//!
//! Inputs are passed as byte arrays (which impl Hash, as required by
//! InputPair) and converted to crypto types inside the operation
//! closure. The conversion cost is identical for both classes.

use rand_core::RngCore;
use sqisign_selkie::{
    curves::{
        Scalar,
        montgomery::{Curve, ProjectiveXOnlyPoint},
    },
    fields::{fp::Fp, fp2::Fp2},
    params::{BASIS_E0_P_X, BASIS_E0_Q_X},
};
use tacet::{AttackerModel, Outcome, TimingOracle, helpers::InputPair};

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut buf = [0u8; N];
    rand_core::OsRng.fill_bytes(&mut buf);
    buf
}

const MODELS: &[(&str, AttackerModel)] = &[
    ("shared_hw", AttackerModel::SharedHardware),
    ("pq_sentinel", AttackerModel::PostQuantumSentinel),
    ("adjacent", AttackerModel::AdjacentNetwork),
];

fn report(name: &str, model_name: &str, outcome: &Outcome) {
    match outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            println!("PASS  {name:<20} [{model_name:<12}] leak_prob={leak_probability:.4}");
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            println!(
                "FAIL  {name:<20} [{model_name:<12}] leak_prob={leak_probability:.4} exploit={exploitability:?}"
            );
        }
        Outcome::Inconclusive { reason, .. } => {
            println!("SKIP  {name:<20} [{model_name:<12}] inconclusive: {reason:?}");
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            println!("SKIP  {name:<20} [{model_name:<12}] unmeasurable: {recommendation}");
        }
        _ => {
            println!("????  {name:<20} [{model_name:<12}]");
        }
    }
}

fn main() {
    println!("tacet constant-time analysis");
    println!("============================\n");

    // --- Fp mul: zero vs random ---
    for &(mname, model) in MODELS {
        let outcome = TimingOracle::for_attacker(model).test(
            InputPair::new(|| [0u8; 64], random_bytes::<64>),
            |bytes| {
                let a = Fp::from_bytes(bytes[..32].try_into().unwrap());
                let b = Fp::from_bytes(bytes[32..].try_into().unwrap());
                let _ = std::hint::black_box(a * b);
            },
        );
        report("fp_mul", mname, &outcome);
    }

    // --- Fp add: zero vs random ---
    for &(mname, model) in MODELS {
        let outcome = TimingOracle::for_attacker(model).test(
            InputPair::new(|| [0u8; 64], random_bytes::<64>),
            |bytes| {
                let a = Fp::from_bytes(bytes[..32].try_into().unwrap());
                let b = Fp::from_bytes(bytes[32..].try_into().unwrap());
                let _ = std::hint::black_box(a + b);
            },
        );
        report("fp_add", mname, &outcome);
    }

    // --- Fp sub: equal vs random ---
    for &(mname, model) in MODELS {
        let outcome = TimingOracle::for_attacker(model).test(
            InputPair::new(
                || {
                    let a: [u8; 32] = random_bytes();
                    let mut out = [0u8; 64];
                    out[..32].copy_from_slice(&a);
                    out[32..].copy_from_slice(&a); // equal → result is 0
                    out
                },
                random_bytes::<64>,
            ),
            |bytes| {
                let a = Fp::from_bytes(bytes[..32].try_into().unwrap());
                let b = Fp::from_bytes(bytes[32..].try_into().unwrap());
                let _ = std::hint::black_box(a - b);
            },
        );
        report("fp_sub", mname, &outcome);
    }

    // --- Fp2 mul: zero vs random ---
    for &(mname, model) in MODELS {
        let outcome = TimingOracle::for_attacker(model).test(
            InputPair::new(|| [0u8; 128], random_bytes::<128>),
            |bytes| {
                let a = Fp2::new(
                    Fp::from_bytes(bytes[..32].try_into().unwrap()),
                    Fp::from_bytes(bytes[32..64].try_into().unwrap()),
                );
                let b = Fp2::new(
                    Fp::from_bytes(bytes[64..96].try_into().unwrap()),
                    Fp::from_bytes(bytes[96..].try_into().unwrap()),
                );
                let _ = std::hint::black_box(a * b);
            },
        );
        report("fp2_mul", mname, &outcome);
    }

    // --- Fp ct_select: choice 0 vs 1 ---
    for &(mname, model) in MODELS {
        let outcome =
            TimingOracle::for_attacker(model).test(InputPair::new(|| 0u8, || 1u8), |&choice| {
                let a = Fp::from_bytes(&[0x42; 32]);
                let b = Fp::from_bytes(&[0x99; 32]);
                let _ = std::hint::black_box(subtle::ConditionallySelectable::conditional_select(
                    &a,
                    &b,
                    subtle::Choice::from(choice),
                ));
            });
        report("fp_ct_select", mname, &outcome);
    }

    // --- Fp ct_eq: equal vs unequal ---
    for &(mname, model) in MODELS {
        let outcome = TimingOracle::for_attacker(model).test(
            InputPair::new(
                || [0x42u8; 64], // a == b
                || {
                    let mut out = [0x42u8; 64];
                    out[32] = 0x99; // a != b
                    out
                },
            ),
            |bytes| {
                let a = Fp::from_bytes(bytes[..32].try_into().unwrap());
                let b = Fp::from_bytes(bytes[32..].try_into().unwrap());
                use subtle::ConstantTimeEq;
                let _ = std::hint::black_box(a.ct_eq(&b));
            },
        );
        report("fp_ct_eq", mname, &outcome);
    }

    // --- Scalar mul: scalar 1 vs random ---
    let p = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &Curve::E0);
    for &(mname, model) in MODELS {
        let outcome = TimingOracle::for_attacker(model).test(
            InputPair::new(|| [0u8; 32], random_bytes::<32>),
            |bytes| {
                let mut limbs = [0u64; 4];
                for (i, chunk) in bytes.chunks(8).enumerate() {
                    limbs[i] = u64::from_le_bytes(chunk.try_into().unwrap());
                }
                // Force scalar 1 for baseline (all bytes zero → limbs[0]=1)
                if limbs == [0; 4] {
                    limbs[0] = 1;
                }
                let s = Scalar::from_limbs(limbs);
                let _ = std::hint::black_box(p.scalar_mul(&s));
            },
        );
        report("scalar_mul", mname, &outcome);
    }

    // --- Point double: P₀ vs Q₀ ---
    let q = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_Q_X, &Curve::E0);
    for &(mname, model) in MODELS {
        let outcome =
            TimingOracle::for_attacker(model).test(InputPair::new(|| false, || true), |&use_q| {
                let pt = if use_q { q } else { p };
                let _ = std::hint::black_box(pt.double());
            });
        report("point_double", mname, &outcome);
    }

    // TODO: Add keygen and sign once they're fast enough for tacet's
    // adaptive sampling (~seconds per invocation currently).
    // keygen: Left = all-zero seed, Right = random seed
    // sign:   Left = all-zero message, Right = random message
    //
    // verify is intentionally NOT planned here: its inputs (vk, sig,
    // msg) are all public per the spec, so CT-on-secrets isn't the
    // property. The dudect `verify` bench measures an oracle-resistance
    // property as a separate opt-in (`DUDECT_ORACLE=1`); tacet would
    // duplicate that without testing anything required by the spec.

    println!("\ntacet analysis complete");
}
