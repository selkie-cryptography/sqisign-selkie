## Summary

<!-- What changed and why, in a few sentences. -->

## Doc surfaces

Tick each surface that this change affects and that you updated in
the same PR (or state why it is unaffected).

- [ ] `README.md`: public type names, status claims, test and KAT counts, quickstart
- [ ] `latex/sections/`: status (`09_status`), design (`03_architecture`), bug catalog (`04_bugs`), perf and CT (`10_post_baseline`)
- [ ] Spec review in `latex/`: new spec gaps or divergences
- [ ] Rustdoc: renamed or retyped items, `# Divergences` at changed call sites
- [ ] `.github/site/`: structural status

## Checks

- [ ] `cargo +nightly fmt --all`, `cargo +nightly clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test --all-features`
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items`
