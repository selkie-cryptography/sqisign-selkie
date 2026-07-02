//! Periodic GC for leaked runner Machines.
//!
//! Runner Machines spawn with `auto_destroy: true`, so the happy path
//! is self-cleanup on runner exit. Failures that leak Machines anyway:
//!
//! - **Runner crash without clean exit.** `auto_destroy` never fires.
//! - **Image rollover + crash.** Old-image Machine survives because its runner
//!   didn't exit; will fail any job routed to it next.
//! - **Runner never receives a job.** Job-stealing or a cancelled-while-queued
//!   job leaves a registered runner idle forever (`run.sh --jitconfig` has no
//!   idle timeout), on the canonical image, with a missed/failed `completed`
//!   webhook as the only other cleanup path.
//!
//! Strategy, two rules per sweep:
//!
//! 1. **Stale image**: resolve the canonical `:latest` digest from the
//!    registry, force-destroy any Machine whose digest differs AND whose age
//!    exceeds `max_age`. Asking the registry beats an in-list heuristic: a lone
//!    stale Machine compared only against its peers becomes its own baseline
//!    and is never reaped. Falls back to "newest spawn's digest" if the
//!    registry call fails.
//! 2. **Hard age**: force-destroy any Machine older than `hard_max_age`,
//!    regardless of digest. `hard_max_age` must exceed the longest workflow job
//!    timeout, so a Machine past it cannot be running a legitimate job.

#[cfg(test)]
mod tests;

use std::time::{Duration, SystemTime};

use anyhow::Result;
use tracing::{info, warn};

use crate::fly::{FlyClient, Machine, MachineId};

/// Background sweeper that periodically force-destroys stale-image
/// and over-hard-age Machines.
#[derive(Debug)]
pub struct Reaper {
    fly: FlyClient,
    interval: Duration,
    max_age: Duration,
    hard_max_age: Duration,
}

impl Reaper {
    /// Construct a reaper waking every `interval`, reaping stale-digest
    /// Machines older than `max_age` and any Machine older than
    /// `hard_max_age`.
    pub fn new(
        fly: FlyClient,
        interval: Duration,
        max_age: Duration,
        hard_max_age: Duration,
    ) -> Self {
        Self {
            fly,
            interval,
            max_age,
            hard_max_age,
        }
    }

    /// Sweep-sleep-repeat forever. Sweep errors are logged and the
    /// loop continues.
    pub async fn run(self) {
        info!(
            interval_s = self.interval.as_secs(),
            max_age_s = self.max_age.as_secs(),
            hard_max_age_s = self.hard_max_age.as_secs(),
            "reaper started"
        );
        loop {
            if let Err(e) = self.sweep().await {
                warn!(error = format!("{e:#}"), "reaper sweep failed");
            }
            tokio::time::sleep(self.interval).await;
        }
    }

    /// One sweep: list, resolve canonical, classify, destroy. A
    /// failed canonical resolve disables only the stale-image rule
    /// for the sweep; the hard-age rule needs no digest.
    async fn sweep(&self) -> Result<()> {
        let machines = self.fly.list_machines().await?;
        let canonical = self.canonical_digest(&machines).await;
        let now = SystemTime::now();
        let candidates = reap_candidates(
            &machines,
            canonical.as_deref(),
            now,
            self.max_age,
            self.hard_max_age,
        );

        if candidates.is_empty() {
            return Ok(());
        }

        for (id, age, digest, reason) in &candidates {
            info!(
                machine = %id.0,
                age_s = age.as_secs(),
                digest = %digest,
                canonical = canonical.as_deref().unwrap_or("<unresolved>"),
                reason,
                "reaping leaked Machine"
            );
            if let Err(e) = self.fly.destroy_machine(id).await {
                warn!(machine = %id.0, error = format!("{e:#}"), "destroy failed");
            }
        }
        Ok(())
    }

    /// Registry digest first, in-list fallback on failure.
    async fn canonical_digest(&self, machines: &[Machine]) -> Option<String> {
        match self.fly.resolve_latest_digest().await {
            Ok(d) => Some(d),
            Err(e) => {
                warn!(
                    error = format!("{e:#}"),
                    "registry digest resolve failed; falling back to newest-spawn digest"
                );
                newest_spawn_digest(machines).map(str::to_owned)
            }
        }
    }
}

/// Pure classifier: given snapshot + canonical digest + policy,
/// return the Machines to destroy, each with a reason string for the
/// sweep log. `canonical: None` (registry and fallback both failed)
/// disables the stale-image rule; the hard-age rule always applies.
fn reap_candidates<'a>(
    machines: &'a [Machine],
    canonical: Option<&str>,
    now: SystemTime,
    max_age: Duration,
    hard_max_age: Duration,
) -> Vec<(&'a MachineId, Duration, &'a str, &'static str)> {
    machines
        .iter()
        .filter_map(|m| {
            let age = m.age(now).ok()?;

            let stale = canonical.is_some_and(|c| m.image_ref.digest != c) && age >= max_age;
            let over_hard_age = age >= hard_max_age;

            let reason = match (stale, over_hard_age) {
                (true, _) => "stale image",
                (_, true) => "over hard max age",
                (false, false) => return None,
            };
            Some((&m.id, age, m.image_ref.digest.as_str(), reason))
        })
        .collect()
}

/// Fallback canonical when the registry is unreachable. A lone stale
/// Machine becomes its own baseline here — the registry path exists
/// to handle that case.
fn newest_spawn_digest(machines: &[Machine]) -> Option<&str> {
    machines
        .iter()
        .filter_map(|m| m.created_at_systemtime().ok().map(|t| (t, m)))
        .max_by_key(|(t, _)| *t)
        .map(|(_, m)| m.image_ref.digest.as_str())
}
