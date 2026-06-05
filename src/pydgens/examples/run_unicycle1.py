# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

######################################################################
# RUNNABLE EXAMPLE ANALYSIS SCRIPT
######################################################################
import logging
import jax.numpy as jnp
from jax import profiler

from pydgens.examples.unicycle1 import Unicycle1
from pydgens.solvers.ilqsolver import solve_ilqgame_feedback

if __name__ == "__main__":

    # create logger to pass to solver
    logging.basicConfig(level=logging.INFO) # global config of root logger, to be run once
    logger = logging.getLogger("ilq_solver")
    logger.setLevel(logging.DEBUG)

    # start code profiler to identify bottlenecks in code
    profiler.start_trace("/tmp/jax_trace")

    # instantiate game wrapper object
    uni1 = Unicycle1(nt=34, dt=0.1)

    # define initial state
    x0 = jnp.array([4.0, 4.0, 0.0, 0.0])

    # solve for feedback Nash strategy of all players
    conv, nl_traj, nl_strat = solve_ilqgame_feedback(uni1.game, x0, logger=logger)

    # stop code profiler to identify bottlenecks in code
    profiler.stop_trace()

    # print results
    print(f"Results:")
    print(f"--- converged: {conv}")
    print(f"--- trajectory: {nl_traj}")
    print(f"--- strategy: {nl_strat}")

