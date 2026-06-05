# Frontend Layer

This directory contains the library's user-facing API layer.

The frontend layer is responsible for:
- semantic problem specification
- high-level modeling abstractions
- convenience/helper APIs
- validation and setup logic
- lowering frontend objects into IR objects
- orchestrating calls into solver pipelines

Objects in this layer prioritize:
- expressiveness
- usability
- semantic clarity
- flexible composition 
- object oriented over functional paradigms

over strict execution-oriented constraints.

Unlike the ir/ layer, frontend objects and functions are not necessarily expected to be:
- jit-safe
- transform-safe
- optimized for traced execution

A typical flow is:

frontend objects/functions -> IR objects -> solvers 

The frontend layer may depend on ir/ and solvers/, but lower layers should avoid depending on the frontend layer.