# Solver Layer

This directory contains numerical algorithms for computing equilibria, policies, trajectories, and related solution objects.

Solvers are expected to operate (almost) exclusively on objects from the ir/ layer rather than directly on high-level frontend/specification objects.

In general:

- ir/ defines executable numerical representations
- solvers/ implements algorithms operating on IRs

Solver implementations should strive to:
- preserve JAX transform compatibility (jit, grad, vmap, etc.)
- minimize dynamic Python-side control flow inside traced regions
- separate setup/preprocessing from execution-critical kernels
- make execution assumptions explicit through IR types

The solver layer is algorithm-oriented rather than user-API-oriented.