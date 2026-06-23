---
icon: material/bookshelf
---

# References

This page collects papers, derivations, and related implementations that are
useful for understanding the solver families and examples in PYDGENS.

Other documentation pages should link to these headings rather than repeating
full citation details inline.

## Dynamic Noncooperative Game Theory

Başar, Tamer, and Geert Jan Olsder. _Dynamic Noncooperative Game Theory_.
Society for Industrial and Applied Mathematics, 1998.

Used for:

- finite-horizon dynamic game background
- feedback Nash concepts
- linear-quadratic dynamic games

## Feedback LQ Nash Derivation

Fridovich-Keil, David, et al. "Feedback LQ Nash" derivation.

- [Derivation PDF](https://github.com/HJReachability/ilqgames/blob/master/derivations/feedback_lq_nash.pdf)

Used for:

- feedback linear-quadratic Nash recursion conventions
- implementation cross-checks for LQ solver behavior

## iLQGames

Fridovich-Keil, David, Ellis Ratner, Lasse Peters, Anca D. Dragan, and
Claire J. Tomlin. "Efficient Iterative Linear-Quadratic Approximations for
Nonlinear Multi-Player General-Sum Differential Games." _2020 IEEE
International Conference on Robotics and Automation (ICRA)_. IEEE, 2020.

- [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9197129)
- [arXiv](https://arxiv.org/abs/1909.04694)
- [C++ implementation](https://github.com/HJReachability/ilqgames)
- [Julia implementation](https://github.com/JuliaGameTheoreticPlanning/iLQGames.jl)

Used for:

- iterative linear-quadratic game approximations
- local feedback Nash solves for nonlinear games
- inspiration for iLQ examples and solver conventions

## ALGAMES

Le Cleac'h, Simon, Mac Schwager, and Zachary Manchester. "ALGAMES: A Fast
Augmented Lagrangian Solver for Constrained Dynamic Games." _Autonomous
Robots_ 46, no. 1 (2022): 201-215.

- [arXiv](https://arxiv.org/abs/2104.08452)

Used for:

- augmented-Lagrangian approaches to constrained dynamic games
- local open-loop Nash trajectories with constraints
- inspiration for constrained examples and AL solver development

## Smooth Game Theory

Fridovich-Keil, David. "Smooth Game Theory." 2024.

- [Notes PDF](https://clearoboticslab.github.io/documents/smooth_game_theory.pdf)

Used for:

- general smooth game-theoretic background
- differential game notation and equilibrium concepts
- future frontend/solver documentation
