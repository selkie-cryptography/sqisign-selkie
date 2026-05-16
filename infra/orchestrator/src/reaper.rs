//! Periodic GC for leaked runner Machines.
//!
//! Runner Machines spawn with `auto_destroy: true`, so the happy path
//! is self-cleanup on runner exit. Failures that leak Machines anyway:
//!
//! - **Runner crash without clean exit.** `auto_destroy` never fires.
//! - **Image rollover + crash.** Old-image Machine survives because its runner
//!   didn't exit; will fail any job routed to it next.
//!
//! Strategy: resolve canonical `:latest` digest from the registry,
//! force-destroy any Machine whose digest differs AND whose age
//! exceeds `max_age`. Asking the registry beats an in-list heuristic:
//! a lone stale Machine compared only against its peers becomes its
//! own baseline and is never reaped. Falls back to "newest spawn's
//! digest" if the registry call fails.

#[cfg(test)]
mod tests;

use std::time::{Duration, SystemTime};

use anyhow::Result;
use tracing::{info, warn};

use crate::fly::{FlyClient, Machine, MachineId};

/// Background sweeper that periodically force-destroys stale-image,
/// old-enough Machines.
#[derive(Debug)]
pub struct Reaper {
    fly: FlyClient,
    interval: Duration,
    max_age: Duration,
}

impl Reaper {
    /// Construct a reaper waking every `interval`, reaping stale-digest
    /// Machines older than `max_age`.
    pub fn new(fly: FlyClient, interval: Duration, max_age: Duration) -> Self {
        Self {
            fly,
            interval,
            max_age,
        }
    }

    /// Sweep-sleep-repeat forever. Sweep errors are logged and the
    /// loop continues.
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

    /// One sweep: list, resolve canonical, classify, destroy.
    async fn sweep(&self) -> Result<()> {
        let machines = self.fly.list_machines().await?;
        let Some(canonical) = self.canonical_digest(&machines).await else {
            return Ok(());
        };
        let now = SystemTime::now();
        let candidates = reap_candidates(&machines, &canonical, now, self.max_age);

        if candidates.is_empty() {
            return Ok(());
        }

        for (id, age, digest) in &candidates {
            info!(
                machine = %id.0,
                age_s = age.as_secs(),
                digest = %digest,
                canonical = %canonical,
                "reaping stale-image Machine"
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
/// return the Machines to destroy.
fn reap_candidates<'a>(
    machines: &'a [Machine],
    canonical: &str,
    now: SystemTime,
    max_age: Duration,
) -> Vec<(&'a MachineId, Duration, &'a str)> {
    machines
        .iter()
        .filter(|m| m.image_ref.digest != canonical)
        .filter_map(|m| {
            let age = m.age(now).ok()?;
            (age >= max_age).then_some((&m.id, age, m.image_ref.digest.as_str()))
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
