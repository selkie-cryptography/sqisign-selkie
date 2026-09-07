//! Periodic reconcile loop: spawn runners for `queued` `fly` jobs that
//! have no live runner Machine.
//!
//! The webhook path spawns one runner per `queued` event. Two failure
//! modes strand jobs with no runner, and the webhook path never retries:
//!
//! - **Missed/lost webhook.** An orchestrator restart (deploy) or a GitHub
//!   delivery gap drops the `queued` event; no runner is spawned.
//! - **Runner job-stealing.** Runners register with labels, not a job id, so
//!   GitHub hands a registering ephemeral runner *any* matching queued job. A
//!   runner spawned for job A legitimately takes job B; when more same-label
//!   jobs queue near-simultaneously than runners overlap in time, some jobs are
//!   left queued while their would-be runners have already taken other jobs and
//!   exited.
//!
//! Strategy (mirrors [`crate::reaper::Reaper`]): every `interval`, list
//! queued `fly` jobs from the Actions API and the live runner Machines,
//! then spawn a runner for each queued job whose id has no Machine named
//! `fly-{job_id}-*`. The Machine-name check makes spawns idempotent, so a
//! job already being served is never double-spawned -- the loop only fills
//! genuine gaps.
//!
//! A job that is still queued after the loop spawned for it lost that
//! runner: the Machine exited without taking the job (deprecated runner
//! version, broken image, registration failure) or took another job of
//! the same labels. Each further spawn for that job waits twice as long
//! as the last ([`SpawnLedger`]), so a fleet that cannot take jobs costs
//! a few Machines an hour instead of one per stranded job per sweep.

use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use anyhow::Result;
use tracing::{error, info, warn};

use crate::{
    fly::{FlyClient, Machine, RunnerSizes, SpawnOutcome},
    github::GitHubAppClient,
};

#[cfg(test)]
mod tests;

/// Seconds between reconcile sweeps. A fixed implementation detail, not
/// an operator knob: polling GitHub once a minute is negligible API load,
/// and the loop is a backstop to the webhook path, not the primary spawn
/// trigger. Promote to `runners.toml` only if a real need to tune arises.
const SWEEP_INTERVAL: Duration = Duration::from_secs(60);

/// Longest hold between two spawns for the same job. Holds double from
/// two sweep intervals up to this.
const MAX_BACKOFF: Duration = Duration::from_secs(30 * 60);

/// Stranded jobs respawned at least once at or above this count mean
/// runners are exiting without taking any job at all, not losing the
/// odd job to stealing.
const FLEET_FAILURE_THRESHOLD: usize = 3;

/// Spawn history for jobs the reconciler has spawned a runner for and
/// that are still queued.
#[derive(Debug, Default)]
struct SpawnLedger {
    attempts: HashMap<u64, Attempt>,
}

/// One job's spawn count and the earliest instant another spawn may go.
#[derive(Debug, Clone, Copy)]
struct Attempt {
    count: u32,
    not_before: Instant,
}

impl SpawnLedger {
    /// Hold after the `count`-th spawn: `SWEEP_INTERVAL * 2^count`,
    /// capped at [`MAX_BACKOFF`].
    fn hold(count: u32) -> Duration {
        SWEEP_INTERVAL
            .checked_mul(1u32 << count.min(16))
            .unwrap_or(MAX_BACKOFF)
            .min(MAX_BACKOFF)
    }

    /// Whether a spawn for `job_id` may go at `now`.
    fn allows(&self, job_id: u64, now: Instant) -> bool {
        self.attempts
            .get(&job_id)
            .is_none_or(|a| now >= a.not_before)
    }

    /// Records a spawn for `job_id` at `now`. Returns the attempt number,
    /// 1 for the first, and the hold before the next one.
    fn record(&mut self, job_id: u64, now: Instant) -> (u32, Duration) {
        let count = self.attempts.get(&job_id).map_or(0, |a| a.count) + 1;
        let hold = Self::hold(count);
        self.attempts.insert(
            job_id,
            Attempt {
                count,
                not_before: now + hold,
            },
        );
        (count, hold)
    }

    /// Forgets jobs that are no longer queued.
    fn retain(&mut self, queued: &HashSet<u64>) {
        self.attempts.retain(|id, _| queued.contains(id));
    }

    /// Number of queued jobs whose runner vanished at least once.
    fn respawned(&self) -> usize {
        self.attempts.values().filter(|a| a.count >= 2).count()
    }
}

/// Background loop that spawns runners for queued jobs the webhook path
/// missed or that lost their runner to job-stealing.
#[derive(Debug)]
pub struct Reconciler {
    github: GitHubAppClient,
    fly: FlyClient,
    sizes: RunnerSizes,
    repo: String,
    ledger: SpawnLedger,
}

impl Reconciler {
    /// Construct a reconciler polling `repo`'s queued `fly` jobs on a
    /// fixed interval.
    pub fn new(github: GitHubAppClient, fly: FlyClient, sizes: RunnerSizes, repo: String) -> Self {
        Self {
            github,
            fly,
            sizes,
            repo,
            ledger: SpawnLedger::default(),
        }
    }

    /// Sweep-sleep-repeat forever. Sweep errors are logged and the loop
    /// continues.
    pub async fn run(mut self) {
        info!(
            interval_s = SWEEP_INTERVAL.as_secs(),
            repo = %self.repo,
            "reconciler started"
        );
        loop {
            if let Err(e) = self.sweep().await {
                warn!(error = format!("{e:#}"), "reconciler sweep failed");
            }
            tokio::time::sleep(SWEEP_INTERVAL).await;
        }
    }

    /// One sweep: list queued jobs, subtract those that already have a
    /// runner Machine or are in backoff, spawn the remainder.
    async fn sweep(&mut self) -> Result<()> {
        let queued = self.github.list_queued_fly_jobs(&self.repo).await?;
        let queued_ids: HashSet<u64> = queued.iter().map(|j| j.id).collect();
        self.ledger.retain(&queued_ids);
        if queued.is_empty() {
            return Ok(());
        }

        let machines = self.fly.list_machines().await?;
        let served: HashSet<u64> = machines
            .iter()
            .filter_map(Machine::spawned_job_id)
            .collect();

        // No per-sweep cap: spawns are bounded by the dedup against live
        // Machines, the per-job backoff, and Fly's org-wide machine
        // limit, which `spawn_runner` backs off on.
        let now = Instant::now();
        let (missing, held): (Vec<_>, Vec<_>) = queued
            .into_iter()
            .filter(|j| !served.contains(&j.id))
            .partition(|j| self.ledger.allows(j.id, now));
        if !held.is_empty() {
            info!(
                count = held.len(),
                "reconciler holding stranded jobs in backoff"
            );
        }
        let respawned = self.ledger.respawned();
        if respawned >= FLEET_FAILURE_THRESHOLD {
            error!(
                jobs = respawned,
                "runners exit without taking jobs; check the runner image and version"
            );
        }
        if missing.is_empty() {
            return Ok(());
        }
        info!(
            count = missing.len(),
            "reconciler spawning runners for stranded jobs"
        );

        for job in missing {
            let size = self
                .sizes
                .from_labels(&job.labels)
                .unwrap_or_else(|| self.sizes.default_size());
            let name = Machine::runner_name(job.id);
            let label_refs: Vec<&str> = job.labels.iter().map(String::as_str).collect();

            let jit = match self.github.mint_jit_config(&name, &label_refs).await {
                Ok(j) => j,
                Err(e) => {
                    warn!(
                        error = format!("{e:#}"),
                        job_id = job.id,
                        "reconciler JIT mint failed"
                    );
                    continue;
                }
            };
            match self.fly.spawn_runner(&name, size, &jit).await {
                Ok(SpawnOutcome::Spawned(id)) => {
                    let (attempt, hold) = self.ledger.record(job.id, now);
                    if attempt == 1 {
                        info!(machine = ?id, job_id = job.id, job = %job.name, "reconciler spawned runner")
                    } else {
                        warn!(
                            machine = ?id,
                            job_id = job.id,
                            job = %job.name,
                            attempt,
                            next_hold_s = hold.as_secs(),
                            "runner vanished without taking its job; respawned with backoff"
                        )
                    }
                }
                Ok(SpawnOutcome::AtCapacity) => {
                    // The org is at its machine limit. Stop the sweep:
                    // the remaining jobs stay queued on GitHub and the
                    // next sweep drains them as finishing runners free
                    // slots. Avoids hammering the API at the cap.
                    info!(
                        "reconciler at machine-limit capacity; deferring remaining jobs to next sweep"
                    );
                    break;
                }
                Err(e) => {
                    warn!(
                        error = format!("{e:#}"),
                        job_id = job.id,
                        "reconciler spawn failed"
                    )
                }
            }
        }

        Ok(())
    }
}
