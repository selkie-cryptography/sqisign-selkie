//! Unit tests for the pure reaper helpers. The `Reaper::sweep`
//! integration path needs a real Fly Machines API and is exercised
//! in deployed environments rather than here.

use std::time::{Duration, SystemTime};

use serde_json::json;

use super::{canonical_digest, reap_candidates};
use crate::fly::Machine;

/// Construct a [`Machine`] for tests by deserializing a minimal JSON
/// payload. Goes through serde so the test exercises the same code
/// path as a live Fly response.
fn machine(id: &str, created_at: &str, digest: &str) -> Machine {
    serde_json::from_value(json!({
        "id": id,
        "created_at": created_at,
        "image_ref": { "digest": digest },
    }))
    .expect("test fixture deserializes")
}

/// Builds a [`SystemTime`] for `2026-05-15T17:36:00Z`. Anchor for
/// `age`-based assertions across the test file.
fn anchor_now() -> SystemTime {
    // 20588 days × 86400 = 1_778_803_200 (2026-05-15T00:00:00Z);
    //   + 17h 36m = 63_360 seconds → 1_778_866_560.
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_778_866_560)
}

#[test]
fn canonical_digest_picks_newest_created_at() {
    let ms = vec![
        machine("a", "2026-05-15T17:00:00Z", "sha256:old"),
        machine("b", "2026-05-15T17:36:00Z", "sha256:new"),
        machine("c", "2026-05-15T17:20:00Z", "sha256:old"),
    ];
    assert_eq!(canonical_digest(&ms), Some("sha256:new"));
}

#[test]
fn canonical_digest_empty_returns_none() {
    assert_eq!(canonical_digest(&[]), None);
}

#[test]
fn canonical_digest_single_returns_only_digest() {
    let ms = vec![machine("a", "2026-05-15T17:00:00Z", "sha256:solo")];
    assert_eq!(canonical_digest(&ms), Some("sha256:solo"));
}

#[test]
fn canonical_digest_skips_unparseable_created_at() {
    let ms = vec![
        machine("a", "garbage", "sha256:should_skip"),
        machine("b", "2026-05-15T17:20:00Z", "sha256:keep"),
    ];
    assert_eq!(canonical_digest(&ms), Some("sha256:keep"));
}

#[test]
fn reap_candidates_skips_canonical_digest() {
    // Newest Machine is on `:new`. Even if it were old, it's never
    // a reap candidate — it defines the baseline.
    let ms = vec![
        machine("old_new", "2026-05-15T16:00:00Z", "sha256:new"),
        machine("newest", "2026-05-15T17:36:00Z", "sha256:new"),
    ];
    let candidates = reap_candidates(&ms, anchor_now(), Duration::from_secs(60));
    assert!(candidates.is_empty(), "got {candidates:?}");
}

#[test]
fn reap_candidates_skips_too_young_stale_machine() {
    // `dry` is stale-digest but only 5 min old; `max_age` 30 min →
    // skip, it might still be running its one job.
    let ms = vec![
        machine("newest", "2026-05-15T17:35:00Z", "sha256:new"),
        machine("dry", "2026-05-15T17:31:00Z", "sha256:old"),
    ];
    let candidates = reap_candidates(&ms, anchor_now(), Duration::from_secs(30 * 60));
    assert!(candidates.is_empty(), "got {candidates:?}");
}

#[test]
fn reap_candidates_destroys_stale_and_old() {
    // `zombie` is old (1h45m) AND stale-digest → reap.
    let ms = vec![
        machine("newest", "2026-05-15T17:30:00Z", "sha256:new"),
        machine("zombie", "2026-05-15T15:51:00Z", "sha256:old"),
    ];
    let candidates = reap_candidates(&ms, anchor_now(), Duration::from_secs(30 * 60));
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].0.0, "zombie");
    assert_eq!(candidates[0].2, "sha256:old");
}

#[test]
fn reap_candidates_skips_machine_with_unparseable_created_at() {
    // A stale-digest Machine whose `created_at` won't parse is
    // skipped: we have no age to compare against, so the conservative
    // call is to leave it alone for a future sweep.
    let ms = vec![
        machine("newest", "2026-05-15T17:30:00Z", "sha256:new"),
        machine("broken", "not-a-date", "sha256:old"),
    ];
    let candidates = reap_candidates(&ms, anchor_now(), Duration::from_secs(30 * 60));
    assert!(candidates.is_empty(), "got {candidates:?}");
}
