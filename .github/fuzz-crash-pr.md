Fuzzing found crashing inputs for `__TARGET__`.

**Commit:** __SHA__
**Run:** __RUN_URL__

### Crashing inputs

__CRASH_LIST__

### Reproduce

```bash
cargo +nightly fuzz run __TARGET__ fuzz/corpus/__TARGET__/
```

Adding these inputs to the corpus ensures they are re-tested on every
future fuzz run as regression tests.
