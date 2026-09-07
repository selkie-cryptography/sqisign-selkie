//! Unit tests for [`crate::reconciler`]'s spawn ledger: the hold
//! schedule, allow/record round-trips, pruning of finished jobs, and
//! the fleet-failure count.

use super::*;

#[test]
fn hold_doubles_from_one_interval_and_caps() {
    assert_eq!(SpawnLedger::hold(1), SWEEP_INTERVAL * 2);
    assert_eq!(SpawnLedger::hold(2), SWEEP_INTERVAL * 4);
    assert_eq!(SpawnLedger::hold(4), SWEEP_INTERVAL * 16);
    assert_eq!(SpawnLedger::hold(5), MAX_BACKOFF);
    assert_eq!(SpawnLedger::hold(40), MAX_BACKOFF);
}

#[test]
fn first_spawn_is_immediate_and_later_ones_wait() {
    let mut ledger = SpawnLedger::default();
    let t0 = Instant::now();
    assert!(ledger.allows(7, t0));

    let (attempt, hold) = ledger.record(7, t0);
    assert_eq!(attempt, 1);
    assert_eq!(hold, SWEEP_INTERVAL * 2);
    assert!(!ledger.allows(7, t0 + SWEEP_INTERVAL));
    assert!(ledger.allows(7, t0 + hold));

    let (attempt, hold) = ledger.record(7, t0 + hold);
    assert_eq!(attempt, 2);
    assert_eq!(hold, SWEEP_INTERVAL * 4);
}

#[test]
fn retain_drops_jobs_that_left_the_queue() {
    let mut ledger = SpawnLedger::default();
    let t0 = Instant::now();
    ledger.record(1, t0);
    ledger.record(2, t0);

    ledger.retain(&HashSet::from([2]));

    assert!(ledger.allows(1, t0));
    assert!(!ledger.allows(2, t0));
}

#[test]
fn respawned_counts_jobs_spawned_more_than_once() {
    let mut ledger = SpawnLedger::default();
    let t0 = Instant::now();
    ledger.record(1, t0);
    ledger.record(2, t0);
    ledger.record(2, t0 + MAX_BACKOFF);

    assert_eq!(ledger.respawned(), 1);
}
