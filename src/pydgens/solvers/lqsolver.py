# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Linear Quadratic Game Solver
#
# Ref: 
# - Basar and Olsder, Sec 6.2, Corollary 6.1 (althought it is lacking some terms, i.e. r_ij)
# - https://github.com/HJReachability/ilqgames/blob/master/derivations/feedback_lq_nash.pdf

import jax
import jax.numpy as jnp

from jax import lax
from functools import partial

from pydgens.ir.gametypes import LinearQuadraticGameType1
from pydgens.ir.strategytypes import FixedStepAffineStrategies
from pydgens.utils.utils import is_block_diagonal


def _starts_ends(u_splits):
    sizes  = [int(s) for s in u_splits]
    starts = [0]
    for s in sizes[:-1]:
        starts.append(starts[-1] + s)
    ends = [a + b for a, b in zip(starts, sizes)]
    return tuple(starts), tuple(ends)

@partial(jax.jit, static_argnums=(8, 9))  # u_starts, u_ends are Python tuples -> static
def _solve_lqgame_feedback_core(A, B, Q, q, R, r, Qf, qf, u_starts, u_ends):
    """
    Solve the stage-indexed finite-horizon LQ feedback Nash recursion.

    Array shapes:
      A: (nsteps, nx, nx)   B: (nsteps, nx, nu)
      Q: (nsteps, N, nx, nx) q: (nsteps, N, nx)
      R: (nsteps, N, nu, nu) r: (nsteps, N, nu)
      Qf: (N, nx, nx)         qf: (N, nx)

    Returns:
      P: (nsteps, nu, nx)
      alpha: (nsteps, nu)

    Indexing convention:
      Let K = nsteps = number of control stages.

      Then the finite-horizon trajectory is
        x[0], x[1], ..., x[K]        # K+1 state nodes
        u[0], u[1], ..., u[K-1]      # K control stages

      and the dynamics are
        x[k+1] = A[k] x[k] + B[k] u[k],   for k = 0, ..., K-1.

      The running-cost arrays Q, q, R, r are also stage-indexed, so Q[k] and
      R[k] belong to the same stage as u[k].

    Cost convention:
      For each player i we solve the running-cost-only game

        J_i =
            0.5 x_K^T Qf_i x_K + qf_i^T x_K
          + sum_{k=0}^{K-1} [
            0.5 x_k^T Q_{k,i} x_k + q_{k,i}^T x_k
          + 0.5 u_k^T R_{k,i} u_k + r_{k,i}^T u_k
        ].

      Under this convention there is:
      - a running state cost on the initial state x[0] through the k=0 term
      - an explicit terminal state cost on x[K] via Qf, qf
      - no control at the terminal node x[K]

      Therefore the dynamic-programming boundary condition is
        V_{i,K}(x[K]) = 0.5 x_K^T Qf_i x_K + qf_i^T x_K,
      which implies
        Z_K = Qf
        zeta_K = qf.
    """
    nsteps, nx, nu = A.shape[0], A.shape[1], B.shape[2]
    N = Q.shape[1]

    # Backward-recursion boundary condition.
    #
    # The final state is x[K] and the last control stage is k = K-1. The
    # explicit terminal-state cost stored in Qf, qf defines the boundary:
    #
    #   V_{i,K}(x[K]) = 0.5 x_K^T Qf_i x_K + qf_i^T x_K
    #   Z_K = Qf
    #   zeta_K = qf
    #
    # Note on iLQGames.jl:
    # In iLQGames.jl v0.2.7/v0.2.8, `horizon(g) == length(player_costs(g))`,
    # and the solver still computes/stores a strategy at that final index.
    # However, it initializes Z, zeta from `last(player_costs(g))` instead of
    # from an explicit terminal field. Under the conventions used here, that
    # only makes sense if the last stored cost slice is intended to represent a
    # terminal state cost rather than the last running-cost stage.
    Z0    = Qf
    zeta0 = qf

    def assemble_S_YP_Ya(A_t, B_t, R_t, r_t, Z, zeta):
        """
        Build S (nu,nu), YP (nu,nx), Ya (nu,) with a few large ops:
          - stack B_i blocks and B_i^T @ Z_i rows
          - block-diagonal from R_t
        """
        # Build B_i blocks (Python tuple is fine; length N is static)
        B_blocks = tuple(B_t[:, s:e] for (s, e) in zip(u_starts, u_ends))  # each (nx, m_i)

        # Stack BZ rows: concat along control axis -> (nu, nx)
        BZ_rows = tuple(Bb.T @ Z[i] for i, Bb in enumerate(B_blocks))      # each (m_i, nx)
        BZ = jnp.concatenate(BZ_rows, axis=0)                               # (nu, nx)

        # Big matmuls (dominant, fast ops)
        S_base = BZ @ B_t                                                   # (nu, nu)
        YP     = BZ @ A_t                                                   # (nu, nx)

        # Block-diagonal R
        R_blk = jnp.zeros_like(S_base)
        for i, (s, e) in enumerate(zip(u_starts, u_ends)):
            # If you must check block structure, do it OUTSIDE jit; see wrapper.
            R_blk = R_blk.at[s:e, s:e].set(R_t[i, s:e, s:e])

        # Ya by stacking each block contribution
        Ya_parts = tuple(B_blocks[i].T @ zeta[i] + r_t[i, s:e]
                         for i, (s, e) in enumerate(zip(u_starts, u_ends)))  # each (m_i,)
        Ya = jnp.concatenate(Ya_parts, axis=0)                                # (nu,)

        S = S_base + R_blk
        return S, YP, Ya

    def step(carry, t_idx):
        Z, zeta = carry
        A_t, B_t = A[t_idx], B[t_idx]
        Q_t, q_t, R_t, r_t = Q[t_idx], q[t_idx], R[t_idx], r[t_idx]

        # Assemble linear system in ~3 big ops
        S, YP, Ya = assemble_S_YP_Ya(A_t, B_t, R_t, r_t, Z, zeta)
        Y = jnp.concatenate([YP, Ya[:, None]], axis=1)  # (nu, nx+1)

        # One dense solve with multiple RHS (general, since S may be indefinite)
        X = jax.scipy.linalg.solve(S, Y)                # (nu, nx+1)
        P_t     = X[:, :-1]                             # (nu, nx)
        alpha_t = X[:, -1]                              # (nu,)

        # Update intermediate terms
        F    = A_t - B_t @ P_t
        beta = -B_t @ alpha_t

        # Per-player Z/zeta updates (N is small; these are cheap)
        Z_new, zeta_new = Z, zeta
        for i in range(N):
            Zi    = Z[i]
            zeta_i = q_t[i] + P_t.T @ (R_t[i] @ alpha_t - r_t[i]) + F.T @ (zeta[i] + Zi @ beta)
            Z_i    = Q_t[i] + P_t.T @ R_t[i] @ P_t + F.T @ Zi @ F
            zeta_new = zeta_new.at[i].set(zeta_i)
            Z_new    = Z_new.at[i].set(Z_i)

        return (Z_new, zeta_new), (P_t, alpha_t)

    # scan backward over k = nsteps-1 ... 0
    (_, _), (P_seq_rev, a_seq_rev) = lax.scan(step, (Z0, zeta0), jnp.arange(nsteps - 1, -1, -1))
    # flip stage dimension back to 0...nsteps-1
    P     = jnp.flip(P_seq_rev, axis=0)
    alpha = jnp.flip(a_seq_rev, axis=0)
    return P, alpha


def solve_lqgame_feedback(lqgame, check_block_diag: bool = True):
    """
    Compute the feedback Nash equilibrium of a finite-horizon, stage-indexed LQ game.

    This solver assumes the LQ game is parameterized by running-cost arrays
    defined on control intervals/stages, not on time-grid nodes. Concretely,
    if ``K = nsteps = nt - 1``, then for each player i the solved objective is

        J_i = sum_{k=0}^{K-1} ell_{i,k}(x_k, u_k),

    with no terminal cost term at x_K.

    Consequences of that convention:
    - the initial state x_0 appears in the stage-k=0 running cost
    - the terminal state x_K = x[nt-1] may carry an explicit terminal state
      cost through ``Qf`` and ``qf``
    - the final feedback strategy entry corresponds to stage K-1 and is zero
      only when there is no terminal state cost and no final-stage affine
      control term

    Wrapper with cheap checks outside JIT; core backward pass is a single
    JIT+scan.

    Parameters
    - lqgame : LinearQuadraticGameType1
        game object encoding a linear quadratic game
    - check_block_diag : bool
        If true, it will ensure that cost quadratic control cost matrix (R) is block diagonal
        Setting to False will significantly reduce computation time, but solution is invalid
        if matrix is not actually block-diagonal
    Returns
    - strat : FixedStepAffineStrategies
        strategy object encoding feedback Nash equilibrium of game
    Ref:
    - Basar and Olsder, Sec 6.2, Corollary 6.1 (althought it is lacking some terms, i.e. r_ij)
    - https://github.com/HJReachability/ilqgames/blob/master/derivations/feedback_lq_nash.pdf

    """
    if not isinstance(lqgame, LinearQuadraticGameType1):
        raise TypeError(
            f"lqgame must be LinearQuadraticGameType1, got {type(LinearQuadraticGameType1)}"
        )

    g = lqgame
    if g.nsteps == 0:
        return FixedStepAffineStrategies(
            tg=g.tg,
            P=jnp.zeros((0, g.nu, g.nx), dtype=g.cs.A.dtype),
            alpha=jnp.zeros((0, g.nu), dtype=g.cs.A.dtype),
        )

    # Optional: do block-diagonal checks ONCE on host (kept out of hot path)
    if check_block_diag:
        for k in range(g.cs.nsteps):
            for i in range(g.N):
                if not is_block_diagonal(R=g.R[k, i], u_splits=g.u_splits):
                    raise ValueError(f"Non-block-diagonal R at (k={k}, i={i})")

    u_starts, u_ends = _starts_ends(g.u_splits)

    P, alpha = _solve_lqgame_feedback_core(
        g.cs.A, g.cs.B, g.Q, g.q, g.R, g.r, g.Qf, g.qf, u_starts, u_ends
    )
    return FixedStepAffineStrategies(tg=g.tg, P=P, alpha=alpha)

def solve_lqgame_feedback_old(
    lqgame:LinearQuadraticGameType1,
    check_block_diag: bool = True,
    Z0: jnp.ndarray | None = None,
    zeta0: jnp.ndarray | None = None,
    ):
    """ Compute feedback Nash equilibrium strategies of N-player, 
    K-stage linear-quadratic game

    Notes
    - This is a non-optimize, not-jax-friendly (i.e. conditional code execution)
        implementation that is to be used to help validate the optimized, jax-friendly
        implementation solve_lqgame_feedback
    - K = number of control stages (i.e. tg.nsteps, T in feedback_lq_nash.pdf). 
        K = nt - 1, where nt is the number of time nodes

    Parameters
    - lqgame : LinearQuadraticGameType1
        game object encoding a linear quadratic game
    - check_block_diag : bool
        If true, it will ensure that cost quadratic control cost matrix (R) is block diagonal
        Setting to False will significantly reduce computation time, but solution is invalid
        if matrix is not actually block-diagonal
    - Z0, zeta0 : jnp.ndarray
        initial recursion states. Defaults to ``lqgame.Qf`` and ``lqgame.qf``.
        These parameters remain configurable to test alternate boundary
        conventions; in particular, to cross validate against iLQGames.jl.

    Returns
    - strat : FixedStepAffineStrategies
        strategy object encoding feedback Nash equilibrium of game

    Ref:
    - Basar and Olsder, Sec 6.2, Corollary 6.1 (althought it is lacking some terms, i.e. r_ij)
    - https://github.com/HJReachability/ilqgames/blob/master/derivations/feedback_lq_nash.pdf
    """

    # type-check the linear quadratic game to ensure it is a discrete-time
    # control system. This solver is not valid for a continuous-time linear system
    if not isinstance(lqgame, LinearQuadraticGameType1):
        raise TypeError(f"lqgame must be of type LinearQuadraticGameType1, got {type(LinearQuadraticGameType1)}")

    # unpack for brevity
    g = lqgame
    nt, nx, nu, N = g.cs.nt, g.cs.nx, g.cs.nu, g.N

    # Precompute control block starts/ends ONCE on host (static Python ints -> fast simple slices)
    # This prevents dynamic slice bookkeeping in inner loops.
    u_sizes  = list(map(int, g.u_splits))
    u_starts = [0]
    for s in u_sizes[:-1]:
        u_starts.append(u_starts[-1] + s)
    u_ends   = [a+b for a,b in zip(u_starts, u_sizes)]

    # Allocate outputs
    P     = jnp.zeros((nt-1, nu, nx), dtype=g.cs.A.dtype)
    alpha = jnp.zeros((nt-1, nu),     dtype=g.cs.A.dtype)

    # Backward recursion state
    if Z0 is None:
        Z = g.Qf
    else:
        Z = Z0
        
    if zeta0 is None:
        zeta = g.qf
    else:
        zeta = zeta0

    # Backward in time: note range stop is -1 to include t=0
    for t in range(nt-2, -1, -1):
        # ---- Hoist once-per-tensor indexing (avoid mixing ints & slices later) ----
        A_t = g.cs.A[t]        # (nx, nx)
        B_t = g.cs.B[t]        # (nx, nu)

        # Per-player tensors at time t; we keep them with leading player axis to slice later
        # Shapes:
        #   Q_t:   (N, nx, nx)
        #   q_t:   (N, nx)
        #   R_t:   (N, nu, nu)
        #   r_t:   (N, nu)
        Q_t = g.Q[t]
        q_t = g.q[t]
        R_t = g.R[t]
        r_t = g.r[t]

        # Build linear system S @ [P | alpha] = [YP | Ya]
        S  = jnp.zeros((nu, nu), dtype=A_t.dtype)
        YP = jnp.zeros((nu, nx), dtype=A_t.dtype)
        Ya = jnp.zeros((nu,),    dtype=A_t.dtype)

        # Fill block rows; keep only simple slices here
        row_start = 0
        for i in range(N):
            row_end = u_ends[i]
            m_i     = u_sizes[i]

            # Cheap check of block-diagonality without advanced indexing:
            # Pull the i-th player block matrix once (no slicing yet)
            R_ti = R_t[i]  # (nu, nu)
            if check_block_diag and not is_block_diagonal(R=R_ti, u_splits=g.u_splits):
                raise ValueError(f"Non-block-diagonal control cost quadratic term R for player {i} at time {t}. Got {R_ti}")

            # Slice columns for this player's controls ONCE; no mixed int+slice
            B_i  = B_t[:, row_start:row_end]                     # (nx, m_i)
            r_i  = r_t[i, row_start:row_end]                     # (m_i,)
            R_ii = R_ti[row_start:row_end, row_start:row_end]    # (m_i, m_i)

            # Precompute reused term
            BZ_i = B_i.T @ Z[i]                                  # (m_i, nx)

            # Row-block writes (use .at[...] with plain slices only)
            S   = S.at[row_start:row_end, :].set(BZ_i @ B_t)
            YP  = YP.at[row_start:row_end, :].set(BZ_i @ A_t)
            Ya  = Ya.at[row_start:row_end].set(B_i.T @ zeta[i] + r_i)

            # Add block-diagonal R_ii
            S   = S.at[row_start:row_end, row_start:row_end].add(R_ii)

            row_start = row_end

        # Solve in one shot
        Y = jnp.concatenate([YP, Ya[:, None]], axis=1)  # (nu, nx+1)
        X = jax.scipy.linalg.solve(S, Y)                                  # (nu, nx+1)
        P_t     = X[:, :-1]
        alpha_t = X[:, -1]
        P     = P.at[t].set(P_t)
        alpha = alpha.at[t].set(alpha_t)

        # Update intermediates
        F    = A_t - B_t @ P_t
        beta = -B_t @ alpha_t

        # Per-player updates; again, avoid mixed indexing
        # Use locals R_ti, r_ti, Q_ti to avoid g.R[t,i,...] style accesses
        for i in range(N):
            R_ti = R_t[i]
            r_ti = r_t[i]
            Q_ti = Q_t[i]

            # Note: P_t.T @ (R_ti @ alpha_t - r_ti) uses only matmuls and simple indexing
            zeta_i_new = (
                q_t[i]
                + P_t.T @ (R_ti @ alpha_t - r_ti)
                + F.T @ (zeta[i] + Z[i] @ beta)
            )
            Z_i_new = Q_ti + P_t.T @ R_ti @ P_t + F.T @ Z[i] @ F

            zeta = zeta.at[i].set(zeta_i_new)
            Z    = Z.at[i].set(Z_i_new)

    return FixedStepAffineStrategies(tg=lqgame.tg, P=P, alpha=alpha)
