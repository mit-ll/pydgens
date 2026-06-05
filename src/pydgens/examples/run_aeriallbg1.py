# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

######################################################################
# RUNNABLE EXAMPLE ANALYSIS SCRIPT
######################################################################

# libraries needed for the running the solver and analysis example on the class
import logging
import argparse
import json
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from pydgens.examples.aeriallbg1 import AerialLBG1_C1
from pydgens.ilqsolver import solve_ilqgame_feedback
from pydgens.strategytypes import FixedStepAffineStrategies
from pydgens.systemtypes import propagate_system_trajectory
from pydgens.timetypes import compute_ts

def run_example(config):
    """ main function for setting up, solving and analyzing Lady-Bandit-Guard Target Guarding problem 
    """

    # pop out the solver config so that it is not passed to game instantiation
    solver_cfg = config.pop("solver_cfg")
    logger = logging.getLogger(solver_cfg.pop("logger_name"))
    logger.setLevel(logging.DEBUG)

    # formulate Nonlinear Target Guarding game and initial state
    x0 = jnp.array(config.pop("init_state"))
    lbg1 = AerialLBG1_C1(**config)

    # define initial strategy
    P = jnp.broadcast_to(jnp.eye(lbg1.game.nu, lbg1.game.nx), (lbg1.game.nt, lbg1.game.nu, lbg1.game.nx))
    alpha = jnp.broadcast_to(jnp.ones(lbg1.game.nu), (lbg1.game.nt, lbg1.game.nu))
    init_strat = FixedStepAffineStrategies(tg=lbg1.game.tg, P=P, alpha=alpha)

    # get initial trajectory from initial state and strategy
    # init_traj = get_game_trajectory(x0=x0, strategy=init_strat, dynamics_eom=lbg1.dynamics, dt=lbg1.dt)
    init_traj = propagate_system_trajectory(lbg1.game.cs, x0=x0, strategy=init_strat)

    # solve for feedback Nash strategy of all players
    conv, nash_traj, nash_strat = solve_ilqgame_feedback(lbg1.game, x0, init_traj=init_traj, init_strat=init_strat, logger=logger, **solver_cfg)

    plot_analysis(lbg1, nash_traj, init_traj)

def heading_alignment_angle(px1, py1, vx1, vy1, px2, py2):
    """
    Compute the angle between velocity vector v1 and the relative position vector (p2 - p1)
    at each time step.

    Parameters:
        px1, py1: jnp.ndarray of shape (T,) — position of object 1
        vx1, vy1: jnp.ndarray of shape (T,) — velocity of object 1
        px2, py2: jnp.ndarray of shape (T,) — position of object 2

    Returns:
        gamma: jnp.ndarray of shape (T,) — angle in radians between v1 and (p2 - p1), 
                in the range [-π, π]
    """
    # Relative position vector from object 1 to object 2
    dx = px2 - px1
    dy = py2 - py1

    # Velocity vector of object 1
    vx = vx1
    vy = vy1

    # Compute angles
    angle_v = jnp.arctan2(vy, vx)
    angle_r = jnp.arctan2(dy, dx)

    # Compute the angle difference (wrapped to [-π, π])
    gamma = angle_r - angle_v
    gamma = (gamma + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return gamma

def plot_analysis(lbg1, traj, traj0):
    px_b, py_b = traj.xs[:, lbg1.PARAMS.GAME_STATE.I_BANDIT_PX], traj.xs[:, lbg1.PARAMS.GAME_STATE.I_BANDIT_PY]
    px0_b, py0_b = traj0.xs[:, lbg1.PARAMS.GAME_STATE.I_BANDIT_PX], traj0.xs[:, lbg1.PARAMS.GAME_STATE.I_BANDIT_PY]
    th_b, vt_b = traj.xs[:, lbg1.PARAMS.GAME_STATE.I_BANDIT_TH], traj.xs[:, lbg1.PARAMS.GAME_STATE.I_BANDIT_VT]
    # ax_b, ay_b = traj.us[:, game.PARAMS.GAME_CTRL.I_BANDIT_AX], traj.us[:, game.PARAMS.GAME_CTRL.I_BANDIT_AY]
    dth_b, dvt_b = traj.us[:, lbg1.PARAMS.GAME_CTRL.I_BANDIT_DTH], traj.us[:, lbg1.PARAMS.GAME_CTRL.I_BANDIT_DVT]

    px_l, py_l = traj.xs[:, lbg1.PARAMS.GAME_STATE.I_LADY_PX], traj.xs[:, lbg1.PARAMS.GAME_STATE.I_LADY_PY]
    px0_l, py0_l = traj0.xs[:, lbg1.PARAMS.GAME_STATE.I_LADY_PX], traj0.xs[:, lbg1.PARAMS.GAME_STATE.I_LADY_PY]
    th_l, vt_l = traj.xs[:, lbg1.PARAMS.GAME_STATE.I_LADY_TH], traj.xs[:, lbg1.PARAMS.GAME_STATE.I_LADY_VT]
    # ax_l, ay_l = traj.us[:, game.PARAMS.GAME_CTRL.I_LADY_AX], traj.us[:, game.PARAMS.GAME_CTRL.I_LADY_AY]
    dth_l, dvt_l = traj.us[:, lbg1.PARAMS.GAME_CTRL.I_LADY_DTH], traj.us[:, lbg1.PARAMS.GAME_CTRL.I_LADY_DVT]

    px_g, py_g = traj.xs[:, lbg1.PARAMS.GAME_STATE.I_GUARD_PX], traj.xs[:, lbg1.PARAMS.GAME_STATE.I_GUARD_PY]
    px0_g, py0_g = traj0.xs[:, lbg1.PARAMS.GAME_STATE.I_GUARD_PX], traj0.xs[:, lbg1.PARAMS.GAME_STATE.I_GUARD_PY]
    th_g, vt_g = traj.xs[:, lbg1.PARAMS.GAME_STATE.I_GUARD_TH], traj.xs[:, lbg1.PARAMS.GAME_STATE.I_GUARD_VT]
    # ax_g, ay_g = traj.us[:, game.PARAMS.GAME_CTRL.I_GUARD_AX], traj.us[:, game.PARAMS.GAME_CTRL.I_GUARD_AY]
    dth_g, dvt_g = traj.us[:, lbg1.PARAMS.GAME_CTRL.I_GUARD_DTH], traj.us[:, lbg1.PARAMS.GAME_CTRL.I_GUARD_DVT]
    # time = jnp.arange(traj.xs.shape[0])
    # time = traj.ts
    time = compute_ts(tg=lbg1.game.tg)

    # compute velocity vectors
    vx_b, vy_b = vt_b * jnp.cos(th_b), vt_b * jnp.sin(th_b)
    vx_l, vy_l = vt_l * jnp.cos(th_l), vt_l * jnp.sin(th_l)
    vx_g, vy_g = vt_g * jnp.cos(th_g), vt_g * jnp.sin(th_g)

    # compute acceleration vectors
    ax_b = dvt_b * jnp.cos(th_b) - vt_b * dth_b * jnp.sin(th_b)
    ay_b = dvt_b * jnp.sin(th_b) + vt_b * dth_b * jnp.sin(th_b)
    ax_l = dvt_l * jnp.cos(th_l) - vt_l * dth_l * jnp.sin(th_l)
    ay_l = dvt_l * jnp.sin(th_l) + vt_l * dth_l * jnp.sin(th_l)
    ax_g = dvt_g * jnp.cos(th_g) - vt_g * dth_g * jnp.sin(th_g)
    ay_g = dvt_g * jnp.sin(th_g) + vt_g * dth_g * jnp.sin(th_g)


    gamma_bl = heading_alignment_angle(
        px1 = px_b, py1 = py_b,
        vx1 = vx_b, vy1 = vy_b,
        px2 = px_l, py2 = py_l) 
    gamma_gb = heading_alignment_angle(
        px1 = px_g, py1 = py_g,
        vx1 = vx_g, vy1 = vy_g,
        px2 = px_b, py2 = py_b) 
    
    dist_bl = jnp.sqrt((px_l-px_b)**2 + (py_l-py_b)**2)
    dist_gb = jnp.sqrt((px_b-px_g)**2 + (py_b-py_g)**2)
    dist_lt = jnp.sqrt((lbg1.px_target-px_l)**2 + (lbg1.py_target-py_l)**2)

    vel_b = jnp.sqrt(vx_b**2+ vy_b**2)
    vel_l = jnp.sqrt(vx_l**2+ vy_l**2)
    vel_g = jnp.sqrt(vx_g**2+ vy_g**2)

    acc_b = jnp.sqrt(ax_b**2+ ay_b**2)
    acc_l = jnp.sqrt(ax_l**2+ ay_l**2)
    acc_g = jnp.sqrt(ax_g**2+ ay_g**2)

    # distances = compute_distance(xi)
    # speed1, speed2 = compute_speeds(xi)

    fig, axs = plt.subplots(2, 3, figsize=(12, 10))

    # [0,0] - Trajectories
    ax = axs[0, 0]
    ax.plot(px_b, py_b, label="Bandit init", marker='o')
    ax.plot(px0_b, py0_b, label="Bandit", linestyle='--', color='blue', marker=None)
    ax.plot(px_l, py_l, label="Lady", marker='x')
    ax.plot(px0_l, py0_l, label="Lady init", linestyle='--', color='orange', marker=None)
    ax.plot(px_g, py_g, label="Guard", marker='^')
    ax.plot(px0_g, py0_g, label="Guard init", linestyle='--', color='green', marker=None)
    ax.scatter(px_b[0], py_b[0], color='blue', marker='s', label='Bandit Start')
    ax.scatter(px_l[0], py_l[0], color='orange', marker='s', label='Lady Start')
    ax.scatter(px_g[0], py_g[0], color='green', marker='s', label='Guard Start')
    # ax.scatter(px_b[-1], py_b[-1], color='blue', marker='*', s=100, label='Player 1 End')
    # ax.scatter(px_l[-1], py_l[-1], color='orange', marker='*', s=100, label='Player 2 End')
    ax.set_title("Player Trajectories")
    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")
    # ax.axis("equal")
    ax.grid(True)
    ax.legend()

    # [0,1] - Pointing angle
    ax = axs[0, 1]
    ax.plot(time, gamma_bl*180/jnp.pi, label="Bandit-Lady")
    ax.plot(time, gamma_gb*180/jnp.pi, label="Guard-Bandit")
    ax.set_title("Heading Alignment Angles Between Players")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Angle (deg)")
    ax.grid(True)
    ax.legend()

    # [1,0] - Distance
    ax = axs[1, 0]
    ax.plot(time, dist_bl, label="Bandit-Lady")
    ax.plot(time, dist_gb, label="Guard-Bandit")
    ax.plot(time, dist_lt, label="Lady-Target")
    ax.set_title("Distance Between Players")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Distance [m]")
    ax.grid(True)
    ax.legend()

    # # [1,1] - Speeds
    ax = axs[1, 1]
    ax.plot(time, vel_b, label="Bandit")
    ax.plot(time, vel_l, label="Lday")
    ax.plot(time, vel_g, label="Guard")
    ax.set_title("Player Speeds Over Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Velocity Magnitude [m/s]")
    ax.grid(True)
    ax.legend()

    # # [1,1] - Acceleration
    ax = axs[1, 2]
    ax.plot(time, acc_b, label="Bandit")
    ax.plot(time, acc_l, label="Lday")
    ax.plot(time, acc_g, label="Guard")
    ax.set_title("Player Acceleration Magnitudes Over Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Acceleration Magnitude [m/s/s]")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

def load_config(config_name: str, config_path: str = None):
    """
    Load configuration values from a JSON file by configuration name using pathlib.

    Args:
        config_name (str): The key in the JSON config file to load.
        config_path (str): Path to the JSON config file.

    Returns:
        dict: Configuration dictionary with keys
            that map to AerialLBG1_C1_P1 init args, e.g. 'T', 'dt', 'px_target', 'py_target', etc.
    """
    if config_path is None:
        # Make default path relative to this file, not the current working directory
        config_path = Path(__file__).parent / "aeriallbg1_cfg.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with config_path.open('r') as f:
        configs = json.load(f)

    if config_name not in configs:
        raise ValueError(f"Configuration '{config_name}' not found in {config_path.name}")

    config = configs[config_name]

    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration from JSON.")
    parser.add_argument("--cfg", type=str, required=True,
                        help="Name of the config block in config.json to use.")
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # global config of root logger, to be run once
    args = parse_args()
    config = load_config(args.cfg)
    run_example(config)