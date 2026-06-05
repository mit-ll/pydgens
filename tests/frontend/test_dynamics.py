# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax
import jax.numpy as jnp

import pydgens.ir.systemtypes as irsys

# public api under test
import pydgens as pdg


def test_linear_dynamics_public_api_returns_linear_continuous_system():
    dyn = pdg.linear_dynamics(
        A=jnp.array([[0.0]]),
        B=jnp.array([[1.0, 1.0]]),
    )

    assert isinstance(dyn, pdg.dynamics.LTIContinuousSystem)


def test_nonlinear_dynamics_public_api_returns_nonlinear_continuous_system():
    dyn = pdg.nonlinear_dynamics(
        nx=2,
        nu=1,
        dynamics=lambda t, x, u: jnp.array([
            x[1],
            -x[0] + u[0],
        ]),
    )

    assert isinstance(dyn, pdg.dynamics.NonlinearContinuousSystem)


def test_nonlinear_dynamics_infers_dimensions_from_explicit_arguments():
    dyn = pdg.nonlinear_dynamics(
        nx=4,
        nu=2,
        dynamics=lambda t, x, u: jnp.array([
            x[3] * jnp.cos(x[2]),
            x[3] * jnp.sin(x[2]),
            u[0],
            u[1],
        ]),
    )

    assert dyn.nx == 4
    assert dyn.nu == 2


def test_nonlinear_dynamics_evaluates_dxdt():
    dyn = pdg.nonlinear_dynamics(
        nx=2,
        nu=1,
        dynamics=lambda t, x, u: jnp.array([
            t + x[0] ** 2 + u[0],
            x[1] - 2.0 * u[0],
        ]),
    )

    x = jnp.array([3.0, -1.0])
    u = jnp.array([2.0])

    dxdt = dyn.evaluate(0.5, x, u)

    expected = jnp.array([
        11.5,
        -5.0,
    ])
    assert jnp.allclose(dxdt, expected)


def test_nonlinear_dynamics_forwards_time_argument():
    dyn = pdg.nonlinear_dynamics(
        nx=2,
        nu=1,
        dynamics=lambda t, x, u: jnp.array([
            t + x[0] + u[0],
            x[1] - u[0],
        ]),
    )

    x = jnp.array([1.0, 2.0])
    u = jnp.array([3.0])

    dxdt_0 = dyn.evaluate(0.0, x, u)
    dxdt_1 = dyn.evaluate(10.0, x, u)

    assert not jnp.allclose(dxdt_0, dxdt_1)


def test_nonlinear_dynamics_rejects_nonpositive_nx():
    with pytest.raises(ValueError, match="`nx` must be a positive integer"):
        pdg.nonlinear_dynamics(
            nx=0,
            nu=1,
            dynamics=lambda t, x, u: x,
        )


def test_nonlinear_dynamics_rejects_nonpositive_nu():
    with pytest.raises(ValueError, match="`nu` must be a positive integer"):
        pdg.nonlinear_dynamics(
            nx=1,
            nu=0,
            dynamics=lambda t, x, u: x,
        )


def test_nonlinear_dynamics_rejects_nondynamic_callable():
    with pytest.raises(TypeError, match="`dynamics` must be callable"):
        pdg.nonlinear_dynamics(
            nx=2,
            nu=1,
            dynamics=3.14,
        )


def test_nonlinear_dynamics_lowers_to_sampled_continuous_system_ir():
    dyn = pdg.nonlinear_dynamics(
        nx=2,
        nu=1,
        dynamics=lambda t, x, u: jnp.array([
            t + x[1] + u[0],
            -x[0],
        ]),
    )
    tg = pdg.time_grid(nt=5, dt=0.2)

    ir_sys = dyn.to_ir(tg=tg)

    assert isinstance(ir_sys, irsys.SampledContinuousSystemType1)
    assert ir_sys.tg == tg
    assert ir_sys.nx == 2
    assert ir_sys.nu == 1

    x = jnp.array([1.0, 2.0])
    u = jnp.array([3.0])
    assert jnp.allclose(
        ir_sys.dynamics(7.5, x, u),
        dyn.evaluate(7.5, x, u),
    )


def test_linear_dynamics_infers_dimensions():
    dyn = pdg.linear_dynamics(
        A=jnp.array([[0.0, 1.0], [-2.0, -3.0]]),
        B=jnp.array([[0.0], [1.0]]),
    )

    assert dyn.nx == 2
    assert dyn.nu == 1


def test_linear_dynamics_evaluates_dxdt():
    dyn = pdg.linear_dynamics(
        A=jnp.array([[0.0, 1.0], [-2.0, -3.0]]),
        B=jnp.array([[0.0], [1.0]]),
    )

    t = 0.0
    x = jnp.array([1.0, 2.0])
    u = jnp.array([3.0])

    dxdt = dyn.evaluate(t, x, u)

    expected = dyn.A @ x + dyn.B @ u
    assert jnp.allclose(dxdt, expected)


def test_linear_dynamics_time_argument_does_not_affect_time_invariant_system():
    dyn = pdg.linear_dynamics(
        A=jnp.array([[0.0]]),
        B=jnp.array([[2.0]]),
    )

    x = jnp.array([1.0])
    u = jnp.array([3.0])

    dxdt_0 = dyn.evaluate(0.0, x, u)
    dxdt_1 = dyn.evaluate(10.0, x, u)

    assert jnp.allclose(dxdt_0, dxdt_1)


def test_linear_dynamics_rejects_non_matrix_A():
    with pytest.raises(ValueError, match="A.*2D"):
        pdg.linear_dynamics(
            A=jnp.array([1.0, 2.0]),
            B=jnp.array([[1.0], [2.0]]),
        )


def test_linear_dynamics_rejects_non_square_A():
    with pytest.raises(ValueError, match="A.*square"):
        pdg.linear_dynamics(
            A=jnp.ones((2, 3)),
            B=jnp.ones((2, 1)),
        )


def test_linear_dynamics_rejects_non_matrix_B():
    with pytest.raises(ValueError, match="B.*2D"):
        pdg.linear_dynamics(
            A=jnp.eye(2),
            B=jnp.array([1.0, 2.0]),
        )


def test_linear_dynamics_rejects_B_with_wrong_state_dimension():
    with pytest.raises(ValueError, match="B.*shape"):
        pdg.linear_dynamics(
            A=jnp.eye(2),
            B=jnp.ones((3, 1)),
        )


def test_linear_dynamics_accepts_numpy_like_inputs():
    dyn = pdg.linear_dynamics(
        A=[[0.0, 1.0], [-1.0, 0.0]],
        B=[[0.0], [1.0]],
    )

    assert isinstance(dyn.A, jnp.ndarray)
    assert isinstance(dyn.B, jnp.ndarray)
    assert dyn.nx == 2
    assert dyn.nu == 1


# ---------------------------------------------------------------------
# Abstract base class behavior
# ---------------------------------------------------------------------


def test_dynamical_system_is_abstract():

    with pytest.raises(TypeError):
        pdg.dynamics.DynamicalSystem()


def test_continuous_system_is_abstract():

    with pytest.raises(TypeError):
        pdg.dynamics.ContinuousSystem()


def test_linear_continuous_system_is_abstract():

    A = jnp.eye(2)
    B = jnp.ones((2, 1))

    with pytest.raises(TypeError):
        pdg.dynamics.LinearContinuousSystem(A=A, B=B)


# ---------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------


def test_lti_constructs():

    A = jnp.eye(2)
    B = jnp.ones((2, 1))

    sys = pdg.dynamics.LTIContinuousSystem(A=A, B=B)

    assert sys.nx == 2
    assert sys.nu == 1


def test_A_must_be_2d():

    A = jnp.array([1.0, 2.0])
    B = jnp.ones((2, 1))

    with pytest.raises(ValueError, match="`A` must be a 2D array"):
        pdg.dynamics.LTIContinuousSystem(A=A, B=B)


def test_B_must_be_2d():

    A = jnp.eye(2)
    B = jnp.array([1.0, 2.0])

    with pytest.raises(ValueError, match="`B` must be a 2D array"):
        pdg.dynamics.LTIContinuousSystem(A=A, B=B)


def test_A_must_be_square():

    A = jnp.ones((2, 3))
    B = jnp.ones((2, 1))

    with pytest.raises(ValueError, match="`A` must be square"):
        pdg.dynamics.LTIContinuousSystem(A=A, B=B)


def test_B_shape_must_match_A():

    A = jnp.eye(3)
    B = jnp.ones((2, 1))

    with pytest.raises(ValueError, match="`B` must have shape"):
        pdg.dynamics.LTIContinuousSystem(A=A, B=B)


# ---------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------


def test_nx_property():

    A = jnp.eye(4)
    B = jnp.ones((4, 2))

    sys = pdg.dynamics.LTIContinuousSystem(A=A, B=B)

    assert sys.nx == 4


def test_nu_property():

    A = jnp.eye(4)
    B = jnp.ones((4, 3))

    sys = pdg.dynamics.LTIContinuousSystem(A=A, B=B)

    assert sys.nu == 3


# ---------------------------------------------------------------------
# Dynamics evaluation
# ---------------------------------------------------------------------


def test_evaluate():

    A = jnp.array([
        [1.0, 0.0],
        [0.0, 2.0],
    ])

    B = jnp.array([
        [1.0],
        [3.0],
    ])

    sys = pdg.dynamics.LTIContinuousSystem(A=A, B=B)

    x = jnp.array([2.0, 4.0])
    u = jnp.array([5.0])

    out = sys.evaluate(
        t=0.0,
        x=x,
        u=u,
    )

    expected = A @ x + B @ u

    assert jnp.allclose(out, expected)


def test_time_argument_is_ignored():

    A = jnp.eye(2)
    B = jnp.ones((2, 1))

    sys = pdg.dynamics.LTIContinuousSystem(A=A, B=B)

    x = jnp.array([1.0, 2.0])
    u = jnp.array([3.0])

    out1 = sys.evaluate(0.0, x, u)
    out2 = sys.evaluate(100.0, x, u)

    assert jnp.allclose(out1, out2)


# ---------------------------------------------------------------------
# Inheritance / MRO behavior
# ---------------------------------------------------------------------


def test_inheritance_relationships():

    A = jnp.eye(2)
    B = jnp.ones((2, 1))

    sys = pdg.dynamics.LTIContinuousSystem(A=A, B=B)

    assert isinstance(sys, pdg.dynamics.LTIContinuousSystem)
    assert isinstance(sys, pdg.dynamics.LinearContinuousSystem)
    assert isinstance(sys, pdg.dynamics.LinearSystem)
    assert isinstance(sys, pdg.dynamics.ContinuousSystem)
    assert isinstance(sys, pdg.dynamics.DynamicalSystem)


def test_mro_order():

    mro = pdg.dynamics.LTIContinuousSystem.__mro__

    assert mro.index(pdg.dynamics.ContinuousSystem) < mro.index(pdg.dynamics.LinearSystem)


# ---------------------------------------------------------------------
# Cooperative kwargs forwarding behavior
# ---------------------------------------------------------------------


def test_linear_system_receives_kwargs_through_mro():
    """
    Ensure A/B kwargs successfully propagate through
    ContinuousSystem -> LinearSystem initialization.
    """

    A = jnp.eye(2)
    B = jnp.ones((2, 1))

    sys = pdg.dynamics.LTIContinuousSystem(A=A, B=B)

    assert jnp.allclose(sys.A, A)
    assert jnp.allclose(sys.B, B)


# ---------------------------------------------------------------------
# Post-construction mutation validation
# ---------------------------------------------------------------------

def test_setting_valid_A_succeeds():

    sys = pdg.dynamics.LTIContinuousSystem(
        A=jnp.eye(2),
        B=jnp.ones((2, 1)),
    )

    new_A = 2.0 * jnp.eye(2)

    sys.A = new_A

    assert jnp.allclose(sys.A, new_A)


def test_setting_invalid_A_ndim_raises():

    sys = pdg.dynamics.LTIContinuousSystem(
        A=jnp.eye(2),
        B=jnp.ones((2, 1)),
    )

    with pytest.raises(ValueError, match="`A` must be a 2D array"):

        sys.A = jnp.array([1.0, 2.0])


def test_setting_non_square_A_raises():

    sys = pdg.dynamics.LTIContinuousSystem(
        A=jnp.eye(2),
        B=jnp.ones((2, 1)),
    )

    with pytest.raises(ValueError, match="`A` must be square"):

        sys.A = jnp.ones((2, 3))


def test_setting_valid_B_succeeds():

    sys = pdg.dynamics.LTIContinuousSystem(
        A=jnp.eye(2),
        B=jnp.ones((2, 1)),
    )

    new_B = jnp.zeros((2, 5))

    sys.B = new_B

    assert jnp.allclose(sys.B, new_B)


def test_setting_invalid_B_ndim_raises():

    sys = pdg.dynamics.LTIContinuousSystem(
        A=jnp.eye(2),
        B=jnp.ones((2, 1)),
    )

    with pytest.raises(ValueError, match="`B` must be a 2D array"):

        sys.B = jnp.array([1.0, 2.0])


def test_setting_B_with_wrong_nx_raises():

    sys = pdg.dynamics.LTIContinuousSystem(
        A=jnp.eye(3),
        B=jnp.ones((3, 1)),
    )

    with pytest.raises(ValueError, match="`B` must have shape"):

        sys.B = jnp.ones((2, 1))


def test_setting_A_with_incompatible_shape_raises():

    sys = pdg.dynamics.LTIContinuousSystem(
        A=jnp.eye(3),
        B=jnp.ones((3, 1)),
    )

    with pytest.raises(ValueError, match="`B` must have shape"):

        sys.A = jnp.eye(2)


def test_failed_assignment_does_not_mutate_state():

    original_A = jnp.eye(3)

    sys = pdg.dynamics.LTIContinuousSystem(
        A=original_A,
        B=jnp.ones((3, 1)),
    )

    with pytest.raises(ValueError):

        sys.A = jnp.ones((2, 3))

    assert jnp.allclose(sys.A, original_A)


# ---------------------------------------------------------------------
# IR Transformations
# ---------------------------------------------------------------------

def test_discretize_lti_euler_scalar_integrator():
    A = jnp.array([[0.0]])
    B = jnp.array([[1.0]])
    dt = 0.1

    A_d, B_d = pdg.dynamics._discretize_lti_euler(A, B, dt)

    assert jnp.allclose(A_d, jnp.array([[1.0]]))
    assert jnp.allclose(B_d, jnp.array([[0.1]]))


def test_discretize_lti_euler_matches_formula():
    A = jnp.array([[0.0, 1.0], [-2.0, -3.0]])
    B = jnp.array([[0.0], [1.0]])
    dt = 0.2

    A_d, B_d = pdg.dynamics._discretize_lti_euler(A, B, dt)

    expected_A_d = jnp.eye(2) + dt * A
    expected_B_d = dt * B

    assert jnp.allclose(A_d, expected_A_d)
    assert jnp.allclose(B_d, expected_B_d)


def test_discretize_lti_zoh_scalar_integrator():
    A = jnp.array([[0.0]])
    B = jnp.array([[1.0]])
    dt = 0.1

    A_d, B_d = pdg.dynamics._discretize_lti_zoh(A, B, dt)

    assert jnp.allclose(A_d, jnp.array([[1.0]]))
    assert jnp.allclose(B_d, jnp.array([[0.1]]))


def test_discretize_lti_zoh_zero_control_matrix():
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.zeros((2, 1))
    dt = 0.1

    A_d, B_d = pdg.dynamics._discretize_lti_zoh(A, B, dt)

    expected_A_d = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    expected_B_d = jnp.zeros((2, 1))

    assert jnp.allclose(A_d, expected_A_d)
    assert jnp.allclose(B_d, expected_B_d)


def test_discretize_lti_zoh_double_integrator():
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    dt = 0.1

    A_d, B_d = pdg.dynamics._discretize_lti_zoh(A, B, dt)

    expected_A_d = jnp.array([[1.0, dt], [0.0, 1.0]])
    expected_B_d = jnp.array([[0.5 * dt**2], [dt]])

    assert jnp.allclose(A_d, expected_A_d)
    assert jnp.allclose(B_d, expected_B_d)


def test_discretize_lti_zoh_is_jittable():
    """
    NOTE: this test is likely unnecessary now that
    discretize_lti_zoh has been moved to frontend/
    and not expected to be jittable
    """
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    dt = 0.1

    jit_fn = jax.jit(pdg.dynamics._discretize_lti_zoh)
    A_d, B_d = jit_fn(A, B, dt)

    expected_A_d = jnp.array([[1.0, dt], [0.0, 1.0]])
    expected_B_d = jnp.array([[0.5 * dt**2], [dt]])

    assert jnp.allclose(A_d, expected_A_d)
    assert jnp.allclose(B_d, expected_B_d)


def test_discretize_lti_dynamics_returns_linear_discrete_system():
    tg = pdg.time_grid(nt=3, dt=0.1)
    dynamics = pdg.linear_dynamics(
        A=jnp.array([[0.0]]),
        B=jnp.array([[1.0]]),
    )

    system = dynamics.discretize_to_ir(
        tg=tg,
        method="euler",
    )

    assert isinstance(system, irsys.LinearDiscreteSystemType1)


def test_discretize_lti_dynamics_broadcasts_over_transitions():
    tg = pdg.time_grid(nt=4, dt=0.1)
    dynamics = pdg.linear_dynamics(
        A=jnp.array([[0.0]]),
        B=jnp.array([[1.0]]),
    )

    system = dynamics.discretize_to_ir(
        tg=tg,
        method="euler",
    )

    assert system.A.shape == (tg.nt-1, 1, 1)
    assert system.B.shape == (tg.nt-1, 1, 1)

    assert jnp.allclose(system.A[:, 0, 0], jnp.ones(tg.nt-1))
    assert jnp.allclose(system.B[:, 0, 0], tg.dt * jnp.ones(tg.nt-1))


def test_discretize_lti_dynamics_uses_zoh_by_default():
    tg = pdg.time_grid(nt=3, dt=0.1)
    dynamics = pdg.linear_dynamics(
        A=jnp.array([[0.0, 1.0], [0.0, 0.0]]),
        B=jnp.array([[0.0], [1.0]]),
    )

    system = dynamics.discretize_to_ir(tg=tg)

    expected_A = jnp.array([[1.0, tg.dt], [0.0, 1.0]])
    expected_B = jnp.array([[0.5 * tg.dt**2], [tg.dt]])

    assert jnp.allclose(system.A[0], expected_A)
    assert jnp.allclose(system.B[0], expected_B)


def test_discretize_lti_dynamics_rejects_unknown_method():
    tg = pdg.time_grid(nt=3, dt=0.1)
    dynamics = pdg.linear_dynamics(
        A=jnp.array([[0.0]]),
        B=jnp.array([[1.0]]),
    )

    with pytest.raises(ValueError, match="Unknown LTI discretization method"):
        dynamics.discretize_to_ir(
            tg=tg,
            method="bad_method",
        )
