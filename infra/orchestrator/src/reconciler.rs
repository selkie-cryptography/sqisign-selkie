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

use std::{collections::HashSet, time::Duration};

use anyhow::Result;
use tracing::{info, warn};

use crate::{
    fly::{FlyClient, Machine, RunnerSizes, SpawnOutcome},
    github::GitHubAppClient,
};

/// Seconds between reconcile sweeps. A fixed implementation detail, not
/// an operator knob: polling GitHub once a minute is negligible API load,
/// and the loop is a backstop to the webhook path, not the primary spawn
/// trigger. Promote to `runners.toml` only if a real need to tune arises.
const SWEEP_INTERVAL: Duration = Duration::from_secs(60);

/// Background loop that spawns runners for queued jobs the webhook path
/// missed or that lost their runner to job-stealing.
#[derive(Debug)]
pub struct Reconciler {
    github: GitHubAppClient,
    fly: FlyClient,
    sizes: RunnerSizes,
    repo: String,
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
        }
    }

    /// Sweep-sleep-repeat forever. Sweep errors are logged and the loop
    /// continues.
    pub async fn run(self) {
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
    /// runner Machine, spawn the remainder.
    async fn sweep(&self) -> Result<()> {
        let queued = self.github.list_queued_fly_jobs(&self.repo).await?;
        if queued.is_empty() {
            return Ok(());
        }

        let machines = self.fly.list_machines().await?;
        let served: HashSet<u64> = machines
            .iter()
            .filter_map(Machine::spawned_job_id)
            .collect();

        // No per-sweep cap: spawns are already bounded by the dedup above
        // (only queued jobs lacking a runner) and, in turn, by Fly's
        // org-wide machine limit, which `spawn_runner` backs off on.
        let missing: Vec<_> = queued
            .into_iter()
            .filter(|j| !served.contains(&j.id))
            .collect();
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
                    info!(machine = ?id, job_id = job.id, job = %job.name, "reconciler spawned runner")
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
