# IR Layer

This directory contains the library's intermediate representations (IRs): low-level, execution-oriented objects used by solvers and numerical routines.

IR objects are intended to be:

- JAX-friendly
- jit-safe
- functional paradigms rather than object oriented
- structurally static
- minimal in semantic richness
- optimized for numerical execution rather than user ergonomics

In general, objects in spec/ or other user-facing modules are lowered into IR objects before entering solver pipelines.

The IR layer should avoid:
- high-level user APIs
- rich semantic validation
- dynamically branching behavior
- setup-oriented convenience abstractions

A useful mental model is:

text semantic specification -> intermediate representation (IR) -> solver execution 

Solvers should (almost) exclusively operate on IR objects rather than frontend/specification objects.