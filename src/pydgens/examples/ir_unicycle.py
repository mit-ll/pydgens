# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# based on minimal_example.jl in https://github.com/JuliaGameTheoreticPlanning/iLQGames.jl/blob/v0.2.7/examples/minimal_example.jl

import logging
import jax.numpy as jnp

from jax import profiler

from pydgens.ir.timetypes import TimeGrid
from pydgens.ir.systemtypes import SampledContinuousSystemType1
from pydgens.ir.costtypes import PlayerCostSpecContinuous
from pydgens.ir.gametypes import NonlinearGameType1
from pydgens.solvers.ilqsolver import solve_ilqgame_feedback

class Unicycle:
    '''
    2-player control of a unicycle system where 
    player-1 wants the system near the origin
    and player-2 wants the system near 1 m/s speed

    The game dynamics are parameterized by:
    - nx (int) = 4: dimension of joint game state vector
    - nu (int) = 2: dimension of joint game control vector 
    - x_t (jnp.ndarray size (n,)): is the joint state vector,
        - x_t[0] = px : x-position of unicycle at time t [m]
        - x_t[1] = py : y-position of unicycle time t [m]
        - x_t[2] = th : heading angle (theta) at time t [rad]
        - x_t[3] = vt : total velocity (linear speed) at time t [m/s]
    - u_t (jnp.ndarray size (m,)): is the joint control vector,
        - u_t[0] = dth_B : rate of change heading (theta dot) at time t [rad/s]
        - u_t[1] = dvt_B : linear acceleration (vt dot) at time t [m/s/s]
    '''

    def __init__(self, 
        nt: int=20, 
        dt: float=0.1):
        """
        # Args:
        - nt (int) : number of time nodes
        - dt (float): size of each time step [sec]
        """

        # compose time characteristics
        tg = TimeGrid(nt=nt, dt=dt)

        # Unpack parameters for ease of use
        N = 2
        nx = 4
        nu = 2

        # encode the size of each player's subvector in the joint control vector u
        u_splits = [1, 1]

        # encode each players cost function
        costfns = [
            lambda t, x, u: x[0]**2 + x[1]**2 + u[0]**2,
            lambda t, x, u: (x[3] - 1)**2 + u[1]**2
        ]
        costs = [PlayerCostSpecContinuous(running=costfns[i]) for i in range(N)]

        # encode 4D unicycle dynamics
        dynamics = lambda t, x, u: jnp.array([x[3]*jnp.cos(x[2]), x[3]*jnp.sin(x[2]), u[0], u[1]])

        # compose control system from dynamics
        cs = SampledContinuousSystemType1(
            tg = tg,
            nx = nx,
            nu = nu,
            dynamics=dynamics
        )

        # compose game from control system and costs
        self.game = NonlinearGameType1(
            cs = cs,
            N = N,
            costs = costs,
            u_splits = jnp.asarray(u_splits)
        )

def main():
    # create logger to pass to solver
    logging.basicConfig(level=logging.INFO) # global config of root logger, to be run once
    logger = logging.getLogger("ilq_solver")
    logger.setLevel(logging.DEBUG)

    # start code profiler to identify bottlenecks in code
    profiler.start_trace("/tmp/jax_trace")

    # instantiate game wrapper object
    uni1 = Unicycle(nt=34, dt=0.1)

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

if __name__ == "__main__":
    main()