//! Unit tests for [`crate::fly::RunnerSizes`] label resolution and
//! actionlint rendering. The Fly API client itself is exercised by
//! the orchestrator's integration paths; this module covers the
//! pure label-resolution logic that does not require a live API.

use super::*;

/// Builds a small fixture mirroring the production layout: one
/// `shared` default, two `performance` opt-ups, plus a size that
/// pins explicit memory / rootfs overrides so the default-fallback
/// paths on [`MachineSize`] have something to assert against.
/// Enough surface area to cover the explicit-tier-wins and
/// default-fallback paths without dragging in every real tier.
fn fixture() -> Config {
    toml::from_str(
        r#"
        default = "shared-4x"
        [sizes."perf-8x"]
        cpu_kind = "performance"
        cpus = 8
        memory_mb = 32768
        rootfs_gb = 50
        [sizes."perf-2x"]
        cpu_kind = "performance"
        cpus = 2
        [sizes."shared-4x"]
        cpu_kind = "shared"
        cpus = 4
        "#,
    )
    .unwrap()
}

/// Sugar so tests read against the size table directly without
/// going through `fixture().sizes` at every call site.
fn fixture_sizes() -> RunnerSizes {
    fixture().sizes
}

/// Convenience: `&[&str]` literal into the `Vec<String>` shape
/// [`RunnerSizes::from_labels`] consumes.
fn labels(slice: &[&str]) -> Vec<String> {
    slice.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn explicit_tier_wins_over_x64_default() {
    let sizes = fixture_sizes();

    let s = sizes
        .from_labels(&labels(&["self-hosted", "fly", "linux", "x64", "perf-8x"]))
        .unwrap();

    assert_eq!(s.cpu_kind, "performance");
    assert_eq!(s.cpus, 8);
}

#[test]
fn explicit_tier_wins_regardless_of_label_order() {
    let sizes = fixture_sizes();

    let s = sizes
        .from_labels(&labels(&["perf-2x", "x64", "fly"]))
        .unwrap();

    assert_eq!(s.cpu_kind, "performance");
    assert_eq!(s.cpus, 2);
}

#[test]
fn plain_x64_resolves_to_default() {
    let sizes = fixture_sizes();

    let s = sizes
        .from_labels(&labels(&["self-hosted", "fly", "linux", "x64"]))
        .unwrap();

    assert_eq!(s.cpu_kind, "shared");
    assert_eq!(s.cpus, 4);
}

#[test]
fn no_fly_label_returns_none() {
    let sizes = fixture_sizes();

    assert!(
        sizes
            .from_labels(&labels(&["self-hosted", "linux", "arm64"]))
            .is_none()
    );
}

#[test]
fn rejects_default_naming_unknown_size() {
    let config: Config = toml::from_str(
        r#"
        default = "perf-99x"
        [sizes."shared-2x"]
        cpu_kind = "shared"
        cpus = 2
        "#,
    )
    .expect("TOML itself is well-formed; only the cross-field check should fail");

    // Parse succeeds; the validation lives in `Config::load`. Reproduce its
    // check directly so we don't depend on touching the filesystem.
    assert!(!config.sizes.sizes.contains_key(&config.sizes.default));
}

#[test]
fn machine_size_falls_back_to_per_cpu_memory_when_unset() {
    let sizes = fixture_sizes();
    let perf_2x = sizes.from_labels(&labels(&["perf-2x"])).unwrap();

    // `perf-2x` declares no memory_mb in the fixture, so the
    // accessor must compute `cpus * DEFAULT_MEMORY_MB_PER_CPU`.
    assert_eq!(perf_2x.memory_mb(), 2 * DEFAULT_MEMORY_MB_PER_CPU);
}

#[test]
fn machine_size_honors_explicit_memory_override() {
    let sizes = fixture_sizes();
    let perf_8x = sizes.from_labels(&labels(&["perf-8x"])).unwrap();

    // The fixture pins perf-8x at `memory_mb = 32768`; the accessor
    // must surface the explicit value, not the per-cpu default.
    assert_eq!(perf_8x.memory_mb(), 32_768);
}

#[test]
fn machine_size_rootfs_falls_back_to_default() {
    let sizes = fixture_sizes();
    let shared_4x = sizes.from_labels(&labels(&["shared-4x"])).unwrap();

    assert_eq!(shared_4x.rootfs_gb(), DEFAULT_ROOTFS_GB);
}

#[test]
fn machine_size_rootfs_honors_explicit_override() {
    let sizes = fixture_sizes();
    let perf_8x = sizes.from_labels(&labels(&["perf-8x"])).unwrap();

    assert_eq!(perf_8x.rootfs_gb(), 50);
}

#[test]
fn reaper_config_falls_back_to_defaults_when_section_omitted() {
    let config: Config = toml::from_str(
        r#"
        default = "shared-4x"
        [sizes."shared-4x"]
        cpu_kind = "shared"
        cpus = 4
        "#,
    )
    .unwrap();

    assert_eq!(config.reaper.sweep_interval_secs, DEFAULT_REAPER_SWEEP_SECS,);
    assert_eq!(config.reaper.max_age_secs, DEFAULT_REAPER_MAX_AGE_SECS);
}

#[test]
fn reaper_config_picks_up_explicit_values() {
    let config: Config = toml::from_str(
        r#"
        default = "shared-4x"
        [sizes."shared-4x"]
        cpu_kind = "shared"
        cpus = 4
        [reaper]
        sweep_interval_secs = 120
        max_age_secs = 3600
        "#,
    )
    .unwrap();

    assert_eq!(config.reaper.sweep_interval_secs, 120);
    assert_eq!(config.reaper.max_age_secs, 3600);
}

#[test]
fn actionlint_yaml_emits_fly_and_every_size_label() {
    let yaml = fixture_sizes().actionlint_yaml();

    assert!(yaml.contains("- fly\n"), "expected `fly` marker label");
    for label in ["perf-8x", "perf-2x", "shared-4x"] {
        assert!(
            yaml.contains(&format!("- {label}\n")),
            "expected `{label}` in {yaml}",
        );
    }
}

/// Builds a [`Machine`] with the given `name`; other fields stubbed.
fn named_machine(name: &str) -> Machine {
    serde_json::from_value(serde_json::json!({
        "id": "0123456789abcd",
        "name": name,
        "created_at": "2026-06-20T00:00:00Z",
        "image_ref": { "digest": "sha256:test" },
    }))
    .expect("test fixture deserializes")
}

#[test]
fn spawned_job_id_parses_runner_names() {
    assert_eq!(
        named_machine("fly-82509019233-1980abc").spawned_job_id(),
        Some(82509019233)
    );
    assert_eq!(named_machine("fly-123").spawned_job_id(), Some(123));
}

#[test]
fn spawned_job_id_ignores_non_runner_names() {
    assert_eq!(named_machine("summer-dream-9663").spawned_job_id(), None);
    assert_eq!(named_machine("fly-abc-1980").spawned_job_id(), None);
}

#[test]
fn runner_name_round_trips_through_spawned_job_id() {
    let name = Machine::runner_name(82_509_019_233);

    assert_eq!(
        named_machine(&name).spawned_job_id(),
        Some(82_509_019_233),
        "runner_name must be parseable by spawned_job_id: {name}"
    );
}
