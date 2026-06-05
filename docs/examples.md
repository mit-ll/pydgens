---
icon: material/file-document-check
---

# Examples

Examples live in [`src/pydgens/examples/`](../src/pydgens/examples/). These scripts are useful both as smoke tests and as starting points for new game formulations.

## Tug-of-War

Compute a feedback Nash equilibrium for a simple linear-quadratic game:

```bash
python src/pydgens/examples/tug_o_war.py
```

## Two-Player Unicycle

Run the iterative linear-quadratic solver on a nonlinear game where two players control a shared unicycle system:

```bash
python src/pydgens/examples/run_unicycle1.py
```

## Double-Integrator Lady-Bandit-Guard

Run a linear-quadratic target-guarding example with double-integrator vehicles:

```bash
python src/pydgens/examples/run_doubleint_lqlbg.py --cfg C1_001
```

The next public-docs pass should add saved plots or animations for these examples so users can see expected solver outputs before running the scripts locally.
