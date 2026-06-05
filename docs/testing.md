---
icon: simple/pytest
---

# Testing

All tests should pass before merging to `main`.

## Quick Tests

These are the tests run by the default `ci` workflow on GitHub Actions.

```bash
pytest tests/ -v -s -m "not slow"
```

## Slow Tests

```bash
pytest tests/ -v -s -m "slow"
```

## Benchmarks

Benchmark tests are useful for release checks and solver-performance investigations. They are intentionally kept out of the default CI path and are run separately from the fast CI workflow.

```bash
pytest tests/ -v -m "slow" --benchmark-columns='mean, min, max, stddev, rounds'
```

To save a local benchmark baseline:

```bash
pytest tests/ -v -m benchmark --benchmark-autosave
```
