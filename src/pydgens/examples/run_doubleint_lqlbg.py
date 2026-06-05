# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

######################################################################
# RUNNABLE EXAMPLE ANALYSIS SCRIPT
######################################################################

# libraries needed for the running the solver and analysis example on the class
import argparse
import json
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from pydgens.examples.doubleint import DoubleInt_LQLBG_C1, DoubleInt_LQLBG_C2
from pydgens.lqsolver import solve_lqgame_feedback as solve_lqgame_feedback


def run_example(config):
    """ main function for setting up, solving and analyzing Lady-Bandit-Guard Target Guarding problem 
    """

    # formulate LQ LBG game wrapper
    xi0 = jnp.array(config.pop("init_aux_state"))
    cost_type = config.pop("cost_type")
    if cost_type.lower() == "c1":
        lqlbg = DoubleInt_LQLBG_C1(**config)
    elif cost_type.lower() == "c2":
        lqlbg = DoubleInt_LQLBG_C2(**config)
    else:
        raise ValueError(f"No DoubleInt_LQLBG class defined for cost-type {cost_type}")

    # solve for feedback Nash strategy of all players in 
    # auxiliary linear dynamics
    st = solve_lqgame_feedback(lqlbg.game)

    # propagate auxiliary state trajectory based upon
    # equilibrium strategy auxiliary control 
    xi = jnp.zeros((lqlbg.game.nt+1,lqlbg.PARAMS.GAME_AUX_STATE.NX))
    mu = jnp.zeros((lqlbg.game.nt+1,lqlbg.PARAMS.GAME_AUX_CTRL.NU))
    xi = xi.at[0].set(xi0)
    for t in range(lqlbg.game.nt):
        mu = mu.at[t].set(- st.P[t] @ xi[t] - st.alpha[t])
        xi = xi.at[t+1].set(lqlbg.game.A[t] @ xi[t] + lqlbg.game.B[t] @ mu[t])

    # plot results for analysis
    plot_analysis(lqlbg, xi, mu)


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

def plot_analysis(lqlbg, xi, mu):
    px_b, py_b = xi[:, lqlbg.PARAMS.GAME_AUX_STATE.I_BANDIT_PX], xi[:, lqlbg.PARAMS.GAME_AUX_STATE.I_BANDIT_PY]
    vx_b, vy_b = xi[:, lqlbg.PARAMS.GAME_AUX_STATE.I_BANDIT_VX], xi[:, lqlbg.PARAMS.GAME_AUX_STATE.I_BANDIT_VY]
    ax_b, ay_b = mu[:, lqlbg.PARAMS.GAME_AUX_CTRL.I_BANDIT_AX], mu[:, lqlbg.PARAMS.GAME_AUX_CTRL.I_BANDIT_AY]
    px_l, py_l = xi[:, lqlbg.PARAMS.GAME_AUX_STATE.I_LADY_PX], xi[:, lqlbg.PARAMS.GAME_AUX_STATE.I_LADY_PY]
    vx_l, vy_l = xi[:, lqlbg.PARAMS.GAME_AUX_STATE.I_LADY_VX], xi[:, lqlbg.PARAMS.GAME_AUX_STATE.I_LADY_VY]
    ax_l, ay_l = mu[:, lqlbg.PARAMS.GAME_AUX_CTRL.I_LADY_AX], mu[:, lqlbg.PARAMS.GAME_AUX_CTRL.I_LADY_AY]
    px_g, py_g = xi[:, lqlbg.PARAMS.GAME_AUX_STATE.I_GUARD_PX], xi[:, lqlbg.PARAMS.GAME_AUX_STATE.I_GUARD_PY]
    vx_g, vy_g = xi[:, lqlbg.PARAMS.GAME_AUX_STATE.I_GUARD_VX], xi[:, lqlbg.PARAMS.GAME_AUX_STATE.I_GUARD_VY]
    ax_g, ay_g = mu[:, lqlbg.PARAMS.GAME_AUX_CTRL.I_GUARD_AX], mu[:, lqlbg.PARAMS.GAME_AUX_CTRL.I_GUARD_AY]
    time = jnp.arange(xi.shape[0])

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
    dist_lt = jnp.sqrt((lqlbg.cfg.px_target-px_l)**2 + (lqlbg.cfg.py_target-py_l)**2)

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
    ax.plot(px_b, py_b, label="Bandit", marker='o')
    ax.plot(px_l, py_l, label="Lady", marker='x')
    ax.plot(px_g, py_g, label="Guard", marker='^')
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
    ax.plot(time, vel_l, label="Lady")
    ax.plot(time, vel_g, label="Guard")
    ax.set_title("Player Speeds Over Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Velocity Magnitude [m/s]")
    ax.grid(True)
    ax.legend()

    # # [1,1] - Acceleration
    ax = axs[1, 2]
    ax.plot(time, acc_b, label="Bandit")
    ax.plot(time, acc_l, label="Lady")
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
        dict: Configuration dictionary with keys 'T', 'dt', 'px', 'py'.
    """
    if config_path is None:
        # Make default path relative to this file, not the current working directory
        config_path = Path(__file__).parent / "doubleint_lqlbg.json"
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
    args = parse_args()
    config = load_config(args.cfg)
    run_example(config)