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

Benchmarks measure runtime performance: they help detect efficiency regressions and
quantify the performance impact of solver or numerical-code changes relative to a
previous revision. They are not correctness tests. Benchmark tests are intentionally
kept out of the default CI path and run separately from the fast CI workflow.

```bash
pytest tests/ -v -m "slow" --benchmark-columns='mean, min, max, stddev, rounds'
```

For a reproducible comparison, run the same benchmark test harness against both the
current source tree and a baseline Git revision. This isolates implementation changes
from changes to the benchmark itself.

The comparison script automates this procedure, including temporary-worktree
management, result naming, and the final report. Supply a baseline ref and a pytest
`-k` expression selecting the stable benchmark subset:

```bash
uv run --extra test python scripts/benchmark/compare_revision.py \
  --baseline v1.0.0 \
  --k 'compute_al_residual_flat_from_decision_vars_warm_perf or compute_al_residual_flat_from_decision_vars_constraint_heavy_warm_perf or jacobian_al_residual_flat_autodiff_constraint_heavy_warm_perf or newton_solve_stationarity_start_metrics_constraint_heavy_warm_perf'
```

Use `--repeat N` to collect repeated paired samples, `--storage PATH` to retain data
outside the default `.benchmarks` directory, and `--allow-dirty` only when the current
uncommitted source is intentionally being benchmarked. Run the script with `--help`
for all options.

Run comparisons on the same, otherwise-idle machine. Avoid comparing results from
different operating systems, CPU models, Python versions, or dependency environments;
those differences can easily dominate a small code-performance change.

### Manual Comparison Procedure

From the current checkout, create an isolated worktree for the baseline release:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
BASELINE_REF="v1.0.0"
BASELINE_DIR="${TMPDIR:-/tmp}/pydgens-v100"

git worktree add "$BASELINE_DIR" "$BASELINE_REF"
```

Use one Python environment for both runs. With `uv`, the test extra supplies
`pytest` and `pytest-benchmark`:

```bash
cd "$REPO_ROOT"
uv sync --extra test

PYTHON="$REPO_ROOT/.venv/bin/python"
PYTEST_BENCHMARK="$REPO_ROOT/.venv/bin/pytest-benchmark"
```

Choose the stable benchmark subset to compare and a persistent local result store.
The quoted expression is a pytest `-k` selector; add or remove test names as the
benchmark suite evolves.

```bash
BENCH_K='compute_al_residual_flat_from_decision_vars_warm_perf or compute_al_residual_flat_from_decision_vars_constraint_heavy_warm_perf or jacobian_al_residual_flat_autodiff_constraint_heavy_warm_perf or newton_solve_stationarity_start_metrics_constraint_heavy_warm_perf'
BENCH_STORE="file://${TMPDIR:-/tmp}/pydgens-benchmarks"
CURRENT_ID="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
BASELINE_ID="$(git -C "$BASELINE_DIR" rev-parse --short HEAD)"
```

First, run and save results for the current revision:

```bash
PYTHONPATH="$REPO_ROOT/src" \
  "$PYTHON" -m pytest "$REPO_ROOT/tests/test_alsolver.py" \
  -k "$BENCH_K" \
  --benchmark-only \
  --benchmark-save="alsolver-${CURRENT_ID}" \
  --benchmark-storage="$BENCH_STORE"
```

Next, run that *same current test file* against the baseline source tree. Keeping the
test harness fixed ensures that any timing difference reflects the code under test,
rather than a changed benchmark setup.

```bash
cd "$BASELINE_DIR"

PYTHONPATH="$BASELINE_DIR/src" \
  "$PYTHON" -m pytest "$REPO_ROOT/tests/test_alsolver.py" \
  -k "$BENCH_K" \
  --benchmark-only \
  --benchmark-save="alsolver-${BASELINE_ID}" \
  --benchmark-storage="$BENCH_STORE"
```

Compare the saved samples by benchmark name:

```bash
"$PYTEST_BENCHMARK" \
  --storage="$BENCH_STORE" \
  compare \
  --group-by=name \
  --columns=min,mean,median,ops
```

Treat a small difference as inconclusive when repeated runs overlap or show high
variation. For a meaningful claim, repeat the paired comparison, report the median
or mean alongside variability, and retain the saved benchmark data. Remove the
baseline worktree when it is no longer needed:

```bash
git -C "$REPO_ROOT" worktree remove "$BASELINE_DIR"
```
