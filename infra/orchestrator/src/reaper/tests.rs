//! Tests for the pure reaper helpers. `Reaper::sweep` is exercised in
//! the deployed environment.

use std::time::{Duration, SystemTime};

use serde_json::json;

use super::{newest_spawn_digest, reap_candidates};
use crate::fly::Machine;

/// Build a [`Machine`] via serde so tests exercise the same parse path
/// as a live Fly response.
fn machine(id: &str, created_at: &str, digest: &str) -> Machine {
    serde_json::from_value(json!({
        "id": id,
        "name": id,
        "created_at": created_at,
        "image_ref": { "digest": digest },
    }))
    .expect("test fixture deserializes")
}

/// `2026-05-15T17:36:00Z`. Anchor for `age`-based assertions.
fn anchor_now() -> SystemTime {
    // 20588 days × 86400 = 1_778_803_200 (2026-05-15T00:00:00Z);
    //   + 17h 36m = 63_360 seconds → 1_778_866_560.
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_778_866_560)
}

#[test]
fn newest_spawn_digest_picks_newest_created_at() {
    let ms = vec![
        machine("a", "2026-05-15T17:00:00Z", "sha256:old"),
        machine("b", "2026-05-15T17:36:00Z", "sha256:new"),
        machine("c", "2026-05-15T17:20:00Z", "sha256:old"),
    ];
    assert_eq!(newest_spawn_digest(&ms), Some("sha256:new"));
}

#[test]
fn newest_spawn_digest_empty_returns_none() {
    assert_eq!(newest_spawn_digest(&[]), None);
}

#[test]
fn newest_spawn_digest_skips_unparseable_created_at() {
    let ms = vec![
        machine("a", "garbage", "sha256:should_skip"),
        machine("b", "2026-05-15T17:20:00Z", "sha256:keep"),
    ];
    assert_eq!(newest_spawn_digest(&ms), Some("sha256:keep"));
}

#[test]
fn reap_candidates_skips_canonical_digest() {
    let ms = vec![
        machine("a", "2026-05-15T16:00:00Z", "sha256:new"),
        machine("b", "2026-05-15T17:36:00Z", "sha256:new"),
    ];
    let candidates = reap_candidates(&ms, "sha256:new", anchor_now(), Duration::from_secs(60));
    assert!(candidates.is_empty(), "got {candidates:?}");
}

#[test]
fn reap_candidates_skips_too_young_stale_machine() {
    let ms = vec![machine("dry", "2026-05-15T17:31:00Z", "sha256:old")];
    let candidates = reap_candidates(
        &ms,
        "sha256:new",
        anchor_now(),
        Duration::from_secs(30 * 60),
    );
    assert!(candidates.is_empty(), "got {candidates:?}");
}

#[test]
fn reap_candidates_destroys_stale_and_old() {
    let ms = vec![
        machine("fresh", "2026-05-15T17:30:00Z", "sha256:new"),
        machine("zombie", "2026-05-15T15:51:00Z", "sha256:old"),
    ];
    let candidates = reap_candidates(
        &ms,
        "sha256:new",
        anchor_now(),
        Duration::from_secs(30 * 60),
    );
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].0.0, "zombie");
    assert_eq!(candidates[0].2, "sha256:old");
}

#[test]
fn reap_candidates_destroys_lone_stale_machine() {
    // The case the registry path exists to handle.
    let ms = vec![machine("zombie", "2026-05-15T15:51:00Z", "sha256:old")];
    let candidates = reap_candidates(
        &ms,
        "sha256:new",
        anchor_now(),
        Duration::from_secs(30 * 60),
    );
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].0.0, "zombie");
}

#[test]
fn reap_candidates_skips_machine_with_unparseable_created_at() {
    let ms = vec![machine("broken", "not-a-date", "sha256:old")];
    let candidates = reap_candidates(
        &ms,
        "sha256:new",
        anchor_now(),
        Duration::from_secs(30 * 60),
    );
    assert!(candidates.is_empty(), "got {candidates:?}");
}
