//! Periodic GC for leaked runner Machines.
//!
//! Each runner Machine is spawned with `auto_destroy: true`, so in the
//! happy path it self-destroys when its JIT-bound runner finishes its
//! one job and exits. Two failure modes leak Machines anyway:
//!
//! - **Runner crashes without a clean exit.** `auto_destroy` never fires; the
//!   Machine sits idle forever on whatever image it booted with.
//! - **Image rollover.** A new `:latest` is pushed; existing Machines keep
//!   running their job on the old image. In the normal case `auto_destroy`
//!   reaps them on exit, but a crash here leaves a Machine that is BOTH idle
//!   AND on a stale image, ready to fail any future job somehow routed to it.
//!
//! The reaper handles both: any Machine whose `image_ref.digest`
//! differs from the most-recently-spawned Machine's digest AND whose
//! age exceeds `max_age` is force-destroyed. The "newest digest"
//! baseline adapts across rollovers — as soon as the orchestrator
//! spawns the first Machine on the new image, every older Machine on
//! the previous digest becomes a reap candidate once it ages past
//! `max_age` (chosen to comfortably exceed the longest single job).

#[cfg(test)]
mod tests;

use std::time::{Duration, SystemTime};

use anyhow::Result;
use tracing::{info, warn};

use crate::fly::{FlyClient, Machine, MachineId};

/// Background sweeper that periodically force-destroys stale-image,
/// old-enough Machines in the bound runner app.
#[derive(Debug)]
pub struct Reaper {
    fly: FlyClient,
    interval: Duration,
    max_age: Duration,
}

impl Reaper {
    /// Construct a reaper bound to `fly`, waking every `interval` and
    /// reaping stale-digest Machines older than `max_age`.
    pub fn new(fly: FlyClient, interval: Duration, max_age: Duration) -> Self {
        Self {
            fly,
            interval,
            max_age,
        }
    }

    /// Run forever: sweep, sleep `interval`, repeat. Errors in any
    /// single sweep are logged and the loop continues — a transient
    /// Fly API failure should not take the reaper down.
    pub async fn run(self) {
        info!(
            interval_s = self.interval.as_secs(),
            max_age_s = self.max_age.as_secs(),
            "reaper started"
        );
        loop {
            if let Err(e) = self.sweep().await {
                warn!(error = format!("{e:#}"), "reaper sweep failed");
            }
            tokio::time::sleep(self.interval).await;
        }
    }

    /// Execute one sweep: list Machines, classify, destroy survivors.
    async fn sweep(&self) -> Result<()> {
        let machines = self.fly.list_machines().await?;
        let now = SystemTime::now();
        let candidates = reap_candidates(&machines, now, self.max_age);

        if candidates.is_empty() {
            return Ok(());
        }

        for (id, age, digest) in &candidates {
            info!(
                machine = %id.0,
                age_s = age.as_secs(),
                digest = %digest,
                "reaping stale-image Machine"
            );
            if let Err(e) = self.fly.destroy_machine(id).await {
                warn!(
                    machine = %id.0,
                    error = format!("{e:#}"),
                    "destroy failed"
                );
            }
        }
        Ok(())
    }
}

/// Classify the Machines a sweep should destroy.
///
/// Pure function for testability: takes the snapshot and policy in,
/// returns the (id, age, digest) of each survivor. Decoupled from
/// [`FlyClient`] so unit tests don't need a mock HTTP server.
///
/// Returns an empty list if there's no canonical digest to compare
/// against (zero or one Machine), or if every Machine matches the
/// canonical digest.
fn reap_candidates(
    machines: &[Machine],
    now: SystemTime,
    max_age: Duration,
) -> Vec<(&MachineId, Duration, &str)> {
    let Some(canonical) = canonical_digest(machines) else {
        return Vec::new();
    };

    machines
        .iter()
        .filter(|m| m.image_ref.digest != canonical)
        .filter_map(|m| {
            let age = m.age(now).ok()?;
            (age >= max_age).then_some((&m.id, age, m.image_ref.digest.as_str()))
        })
        .collect()
}

/// Digest of the most-recently-created Machine in `machines`, or
/// `None` if the slice is empty or no Machine has a parseable
/// `created_at`.
///
/// Using "newest spawn's digest" as the canonical image lets the
/// reaper auto-adapt across rollovers without an out-of-band
/// "current `:latest`" lookup. The first new-image Machine to be
/// spawned after a deploy immediately becomes the baseline; every
/// older Machine on the previous digest is then a reap candidate
/// once it ages past `max_age`.
fn canonical_digest(machines: &[Machine]) -> Option<&str> {
    machines
        .iter()
        .filter_map(|m| m.created_at_systemtime().ok().map(|t| (t, m)))
        .max_by_key(|(t, _)| *t)
        .map(|(_, m)| m.image_ref.digest.as_str())
}
