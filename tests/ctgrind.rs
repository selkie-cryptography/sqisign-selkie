//! Secret-dependent memory access tests using Valgrind memcheck.
//!
//! Marks secret inputs as "undefined" using Valgrind client requests,
//! then runs crypto operations. Valgrind will report errors if any
//! branch or memory access depends on the secret data.
//!
//! Run with:
//!   cargo test --test ctgrind --features expose-internals --no-run
//!   valgrind --tool=memcheck --error-exitcode=1 \
//!     target/debug/deps/ctgrind-* --test-threads=1

use core::ffi::c_void;

use crabgrind::memcheck::{self, MemState};
use sqisign_selkie::fields::{fp::Fp, fp2::Fp2};

/// Mark a byte slice as "secret" (undefined) for Valgrind.
/// When not running under Valgrind, this is a no-op.
fn mark_secret(data: &[u8]) {
    let _ = memcheck::mark_memory(
        data.as_ptr() as *const c_void,
        data.len(),
        MemState::Undefined,
    );
}

/// Mark a byte slice as "public" (defined) for Valgrind.
fn mark_public(data: &[u8]) {
    let _ = memcheck::mark_memory(
        data.as_ptr() as *const c_void,
        data.len(),
        MemState::Defined,
    );
}

#[test]
fn fp_mul_secret_independent() {
    let a_bytes = [0x42u8; 32];
    let b_bytes = [0x99u8; 32];

    // Mark inputs as secret.
    mark_secret(&a_bytes);
    mark_secret(&b_bytes);

    let a = Fp::from_bytes(&a_bytes);
    let b = Fp::from_bytes(&b_bytes);

    // Multiplication must not branch on secret data.
    let result = a * b;

    // Mark result as public so the test framework can inspect it.
    let result_bytes = result.to_bytes();
    mark_public(&result_bytes);
}

#[test]
fn fp_add_secret_independent() {
    let a_bytes = [0x42u8; 32];
    let b_bytes = [0x99u8; 32];
    mark_secret(&a_bytes);
    mark_secret(&b_bytes);

    let a = Fp::from_bytes(&a_bytes);
    let b = Fp::from_bytes(&b_bytes);
    let result = a + b;

    let result_bytes = result.to_bytes();
    mark_public(&result_bytes);
}

#[test]
fn fp_sub_secret_independent() {
    let a_bytes = [0x42u8; 32];
    let b_bytes = [0x99u8; 32];
    mark_secret(&a_bytes);
    mark_secret(&b_bytes);

    let a = Fp::from_bytes(&a_bytes);
    let b = Fp::from_bytes(&b_bytes);
    let result = a - b;

    let result_bytes = result.to_bytes();
    mark_public(&result_bytes);
}

#[test]
fn fp2_mul_secret_independent() {
    let bytes = [0x42u8; 64];
    mark_secret(&bytes);

    let a = Fp2::new(
        Fp::from_bytes(bytes[..32].try_into().unwrap()),
        Fp::from_bytes(bytes[32..].try_into().unwrap()),
    );
    let b = Fp2::new(Fp::from_bytes(&[0x11; 32]), Fp::from_bytes(&[0x22; 32]));
    let result = a * b;

    // Consume result without inspecting it.
    std::hint::black_box(result);
}

#[test]
fn fp_ct_select_secret_independent() {
    use subtle::ConditionallySelectable;

    let a = Fp::from_bytes(&[0x42; 32]);
    let b = Fp::from_bytes(&[0x99; 32]);

    // The choice bit is secret.
    let mut choice_byte = 1u8;
    mark_secret(std::slice::from_ref(&choice_byte));
    let choice = subtle::Choice::from(choice_byte);

    let result = Fp::conditional_select(&a, &b, choice);

    let result_bytes = result.to_bytes();
    mark_public(&result_bytes);

    // Also mark the choice as public again for test cleanup.
    choice_byte = 0;
    mark_public(std::slice::from_ref(&choice_byte));
}

// TODO: Add keygen, sign, verify under Valgrind memcheck once they're
// fast enough. Currently each takes seconds, and Valgrind adds ~20x
// overhead making these impractical (~minutes per test).
//
// keygen: mark seed as secret, run generate_derand, check no
//         branch or memory access depends on seed
// sign:   mark sk fields as secret, run sign, check no branch or
//         memory access depends on secret key material
// verify: mark sig as secret, run verify, check no branch or memory
//         access depends on the signature (prevents oracle attacks)
