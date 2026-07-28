---
icon: material/chart-bar
---

# Profiling

Profiling explains where a single run spends time or allocates memory. It is
different from [benchmarking](testing.md#benchmarks): use profiling to find a
candidate optimization and benchmarking to measure whether that optimization
improved performance relative to a baseline.

PYDGENS provides optional profiling dependencies. Install them into the same
Python environment used to run the example or tests:

```bash
python -m pip install -e ".[profile]"
```

Profile representative, focused workloads rather than the entire test suite.
The commands below use the unicycle example; replace it with the example or
test that exercises the code path of interest. Profilers add overhead, so do
not use their elapsed times as benchmark results.

## Scalene: CPU and Memory Hotspots

[Scalene](https://github.com/plasma-umass/scalene) is useful for a broad
line-level view of CPU and memory activity. The command below profiles the
unicycle example and saves its data outside the source tree:

```bash
mkdir -p .profiles
scalene run --profile-all \
  --outfile .profiles/unicycle-scalene.json \
  src/pydgens/examples/unicycle.py
```

Open the interactive report or render a terminal summary:

```bash
scalene view .profiles/unicycle-scalene.json
scalene view --cli .profiles/unicycle-scalene.json
```

`--profile-all` includes code beyond the target script. For a faster CPU-only
pass, replace it with `--cpu-only`. To profile a focused pytest workload,
Scalene can run pytest as a module:

```bash
scalene run -m pytest tests/test_alsolver.py -k 'test_name'
```

## Line Profiler: One Function at a Time

[Line Profiler](https://kernprof.readthedocs.io/en/latest/index.html) measures
execution time line by line for functions you explicitly mark. It has higher
overhead than a sampling profiler, so use it after another tool has identified
a small area to investigate.

Temporarily decorate the function of interest. Importing the decorator keeps
the module runnable when it is not being profiled:

```python
from line_profiler import profile


@profile
def function_to_investigate(...):
    ...
```

Run the module through `kernprof` with line-by-line profiling enabled. The
`-v` flag prints the report and `-o` saves it for later inspection:

```bash
mkdir -p .profiles
PYTHONPATH=src kernprof -l -v \
  -o .profiles/unicycle.lprof \
  -m pydgens.examples.unicycle
```

View a saved report again with:

```bash
python -m line_profiler .profiles/unicycle.lprof
```

Remove temporary decorators before committing unless they are intentionally
part of the maintained profiling setup.

## PyInstrument: Call-Tree Overview

[PyInstrument](https://pyinstrument.readthedocs.io/en/stable/guide.html) is a
sampling profiler that produces a call-tree overview with relatively low
overhead. It can profile a script or module directly, without modifying the
source code:

```bash
pyinstrument -m pydgens.examples.unicycle
```

Save an interactive HTML report for sharing or later inspection:

```bash
mkdir -p .profiles
pyinstrument \
  --renderer html \
  --outfile .profiles/unicycle-pyinstrument.html \
  -m pydgens.examples.unicycle
```

To profile selected tests, run pytest through PyInstrument:

```bash
pyinstrument -m pytest tests/test_alsolver.py -k 'test_name'
```

For a very short workload, PyInstrument may collect too few samples. In that
case, profile a representative loop or use a smaller sampling interval, while
recognizing that smaller intervals add overhead.

## Interpreting Results

Treat profiler output as evidence for where to investigate, not as a direct
performance claim. JAX compilation and asynchronous execution can make a
first-run profile look very different from a steady-state solve. After making
an optimization, use the [benchmark comparison procedure](testing.md#benchmarks)
to compare the same workload against a baseline under controlled conditions.
