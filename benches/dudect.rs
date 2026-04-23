//! DudeCT constant-time tests.
//!
//! For each function, we define two input classes (Left and Right) that
//! we suspect might take different amounts of time. If the function is
//! truly constant-time, the t-statistic should stay small (|t| < 4.5).
//!
//! Run with: `cargo bench --bench dudect --features expose-internals`

use dudect_bencher::{BenchRng, Class, CtRunner, ctbench_main, rand::RngExt};
use sqisign_selkie::{
    curves::{
        Scalar,
        montgomery::{Curve, ProjectiveXOnlyPoint},
    },
    fields::{fp::Fp, fp2::Fp2},
    params::{BASIS_E0_P_X, BASIS_E0_Q_X},
};

// --- Field arithmetic ---

/// Fp multiplication: Left = (0, 0), Right = (random, random).
fn fp_mul(runner: &mut CtRunner, rng: &mut BenchRng) {
    let mut inputs = Vec::new();
    let mut classes = Vec::new();

    for _ in 0..100_000 {
        if rng.random::<bool>() {
            inputs.push((Fp::ZERO, Fp::ZERO));
            classes.push(Class::Left);
        } else {
            let a = Fp::from_bytes(&rng.random::<[u8; 32]>());
            let b = Fp::from_bytes(&rng.random::<[u8; 32]>());
            inputs.push((a, b));
            classes.push(Class::Right);
        }
    }

    for (class, (a, b)) in classes.into_iter().zip(inputs) {
        runner.run_one(class, || {
            let _ = std::hint::black_box(a * b);
        });
    }
}

/// Fp addition: Left = (0, 0), Right = (random, random).
fn fp_add(runner: &mut CtRunner, rng: &mut BenchRng) {
    let mut inputs = Vec::new();
    let mut classes = Vec::new();

    for _ in 0..100_000 {
        if rng.random::<bool>() {
            inputs.push((Fp::ZERO, Fp::ZERO));
            classes.push(Class::Left);
        } else {
            let a = Fp::from_bytes(&rng.random::<[u8; 32]>());
            let b = Fp::from_bytes(&rng.random::<[u8; 32]>());
            inputs.push((a, b));
            classes.push(Class::Right);
        }
    }

    for (class, (a, b)) in classes.into_iter().zip(inputs) {
        runner.run_one(class, || {
            let _ = std::hint::black_box(a + b);
        });
    }
}

/// Fp subtraction: Left = (x, x) (result = 0), Right = (random, random).
fn fp_sub(runner: &mut CtRunner, rng: &mut BenchRng) {
    let mut inputs = Vec::new();
    let mut classes = Vec::new();

    for _ in 0..100_000 {
        if rng.random::<bool>() {
            let a = Fp::from_bytes(&rng.random::<[u8; 32]>());
            inputs.push((a, a));
            classes.push(Class::Left);
        } else {
            let a = Fp::from_bytes(&rng.random::<[u8; 32]>());
            let b = Fp::from_bytes(&rng.random::<[u8; 32]>());
            inputs.push((a, b));
            classes.push(Class::Right);
        }
    }

    for (class, (a, b)) in classes.into_iter().zip(inputs) {
        runner.run_one(class, || {
            let _ = std::hint::black_box(a - b);
        });
    }
}

/// Fp squaring: Left = square of 0, Right = square of random.
fn fp_square(runner: &mut CtRunner, rng: &mut BenchRng) {
    let mut inputs = Vec::new();
    let mut classes = Vec::new();

    for _ in 0..100_000 {
        if rng.random::<bool>() {
            inputs.push(Fp::ZERO);
            classes.push(Class::Left);
        } else {
            inputs.push(Fp::from_bytes(&rng.random::<[u8; 32]>()));
            classes.push(Class::Right);
        }
    }

    for (class, a) in classes.into_iter().zip(inputs) {
        runner.run_one(class, || {
            let _ = std::hint::black_box(a * a);
        });
    }
}

/// Fp2 multiplication: Left = (0, 0), Right = (random, random).
fn fp2_mul(runner: &mut CtRunner, rng: &mut BenchRng) {
    let mut inputs = Vec::new();
    let mut classes = Vec::new();

    for _ in 0..100_000 {
        if rng.random::<bool>() {
            inputs.push((Fp2::ZERO, Fp2::ZERO));
            classes.push(Class::Left);
        } else {
            let a = Fp2::new(
                Fp::from_bytes(&rng.random::<[u8; 32]>()),
                Fp::from_bytes(&rng.random::<[u8; 32]>()),
            );
            let b = Fp2::new(
                Fp::from_bytes(&rng.random::<[u8; 32]>()),
                Fp::from_bytes(&rng.random::<[u8; 32]>()),
            );
            inputs.push((a, b));
            classes.push(Class::Right);
        }
    }

    for (class, (a, b)) in classes.into_iter().zip(inputs) {
        runner.run_one(class, || {
            let _ = std::hint::black_box(a * b);
        });
    }
}

// --- subtle trait impls ---

/// Fp conditional select: Left = select(0, a), Right = select(1, a).
fn fp_ct_select(runner: &mut CtRunner, rng: &mut BenchRng) {
    let mut inputs = Vec::new();
    let mut classes = Vec::new();

    for _ in 0..100_000 {
        let a = Fp::from_bytes(&rng.random::<[u8; 32]>());
        let b = Fp::from_bytes(&rng.random::<[u8; 32]>());
        if rng.random::<bool>() {
            inputs.push((a, b, subtle::Choice::from(0)));
            classes.push(Class::Left);
        } else {
            inputs.push((a, b, subtle::Choice::from(1)));
            classes.push(Class::Right);
        }
    }

    for (class, (a, b, choice)) in classes.into_iter().zip(inputs) {
        runner.run_one(class, || {
            let _ = std::hint::black_box(subtle::ConditionallySelectable::conditional_select(
                &a, &b, choice,
            ));
        });
    }
}

/// Fp constant-time equality: Left = (a, a), Right = (a, b).
fn fp_ct_eq(runner: &mut CtRunner, rng: &mut BenchRng) {
    let mut inputs = Vec::new();
    let mut classes = Vec::new();

    for _ in 0..100_000 {
        let a = Fp::from_bytes(&rng.random::<[u8; 32]>());
        if rng.random::<bool>() {
            inputs.push((a, a));
            classes.push(Class::Left);
        } else {
            let b = Fp::from_bytes(&rng.random::<[u8; 32]>());
            inputs.push((a, b));
            classes.push(Class::Right);
        }
    }

    for (class, (a, b)) in classes.into_iter().zip(inputs) {
        runner.run_one(class, || {
            use subtle::ConstantTimeEq;
            let _ = std::hint::black_box(a.ct_eq(&b));
        });
    }
}

// --- Curve operations ---

/// Scalar multiplication on E₀: Left = scalar of 1, Right = random scalar.
/// The Montgomery ladder must be constant-time in the scalar.
fn scalar_mul(runner: &mut CtRunner, rng: &mut BenchRng) {
    let p = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &Curve::E0);

    let mut inputs = Vec::new();
    let mut classes = Vec::new();

    for _ in 0..10_000 {
        if rng.random::<bool>() {
            inputs.push(Scalar::ONE);
            classes.push(Class::Left);
        } else {
            let limbs: [u64; 4] = rng.random();
            inputs.push(Scalar::from_limbs(limbs));
            classes.push(Class::Right);
        }
    }

    for (class, s) in classes.into_iter().zip(inputs) {
        runner.run_one(class, || {
            let _ = std::hint::black_box(p.scalar_mul(&s));
        });
    }
}

/// Point doubling: Left = P₀, Right = Q₀. Must be constant-time
/// regardless of which point is doubled.
fn point_double(runner: &mut CtRunner, rng: &mut BenchRng) {
    let p = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_P_X, &Curve::E0);
    let q = ProjectiveXOnlyPoint::from_affine_x(BASIS_E0_Q_X, &Curve::E0);

    let mut inputs = Vec::new();
    let mut classes = Vec::new();

    for _ in 0..100_000 {
        if rng.random::<bool>() {
            inputs.push(p);
            classes.push(Class::Left);
        } else {
            inputs.push(q);
            classes.push(Class::Right);
        }
    }

    for (class, pt) in classes.into_iter().zip(inputs) {
        runner.run_one(class, || {
            let _ = std::hint::black_box(pt.double());
        });
    }
}

// --- Keygen and signing ---
// These are slow (seconds per invocation), so we use far fewer samples
// than the field-level tests. DudeCT still works — it just needs more
// wall-clock time to reach statistical significance.

/// Keygen: Left = seed of all zeros, Right = random seed.
/// A variable-time keygen might branch on the seed value.
fn keygen(runner: &mut CtRunner, rng: &mut BenchRng) {
    use sqisign_selkie::SigningKey;

    let mut inputs = Vec::new();
    let mut classes = Vec::new();

    // Only 20 samples — each keygen takes ~5 seconds.
    for _ in 0..20 {
        let mut seed = [0u8; 48];
        if rng.random::<bool>() {
            // Left: all-zero seed
            classes.push(Class::Left);
        } else {
            // Right: random seed
            rng.fill(&mut seed[..]);
            classes.push(Class::Right);
        }
        inputs.push(seed);
    }

    for (class, seed) in classes.into_iter().zip(inputs) {
        runner.run_one(class, || {
            let _ = std::hint::black_box(SigningKey::generate_derand(&seed));
        });
    }
}

/// Signing: Left = all-zero message, Right = random message.
/// A variable-time sign might branch on message content (it shouldn't —
/// the message only enters via the hash).
fn sign(runner: &mut CtRunner, rng: &mut BenchRng) {
    use sqisign_selkie::SigningKey;

    let sk_hex = sqisign_selkie::keys::kat_data::KAT_VECTORS[0].2;
    let sk_bytes = hex::decode(sk_hex).unwrap();
    let sk = SigningKey::from_bytes(sk_bytes.as_slice().try_into().unwrap()).unwrap();

    let mut inputs = Vec::new();
    let mut classes = Vec::new();

    // Only 20 samples — each sign takes seconds.
    for _ in 0..20 {
        if rng.random::<bool>() {
            // Left: all-zero message
            inputs.push([0u8; 64]);
            classes.push(Class::Left);
        } else {
            // Right: random message
            inputs.push(rng.random::<[u8; 64]>());
            classes.push(Class::Right);
        }
    }

    for (class, msg) in classes.into_iter().zip(inputs) {
        runner.run_one(class, || {
            let mut rng = rand_core::OsRng;
            let _ = std::hint::black_box(sk.sign(&msg, &mut rng));
        });
    }
}

ctbench_main!(
    fp_mul,
    fp_add,
    fp_sub,
    fp_square,
    fp2_mul,
    fp_ct_select,
    fp_ct_eq,
    scalar_mul,
    point_double,
    keygen,
    sign
);
