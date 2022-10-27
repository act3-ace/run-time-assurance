"""This module implements RTA methods for the multiagent inspection problem with 3D CWH dynamics models
"""
from collections import OrderedDict
from typing import Union

import jax.numpy as jnp
import numpy as np
import scipy
from safe_autonomy_dynamics.base_models import BaseLinearODESolverDynamics
from safe_autonomy_dynamics.cwh.point_model import M_DEFAULT, N_DEFAULT, generate_cwh_matrices

from run_time_assurance.constraint import (
    ConstraintMagnitudeStateLimit,
    ConstraintModule,
    ConstraintStrengthener,
    PolynomialConstraintStrengthener,
)
from run_time_assurance.rta import ExplicitASIFModule
from run_time_assurance.state import RTAStateWrapper

NUM_DEPUTIES_DEFAULT = 5  # Number of deputies for inspection problem
CHIEF_RADIUS_DEFAULT = 5  # chief radius of collision [m] (collision freedom)
DEPUTY_RADIUS_DEFAULT = 5  # deputy radius of collision [m] (collision freedom)
V0_DEFAULT = 0.2  # maximum docking speed [m/s] (dynamic velocity constraint)
V1_COEF_DEFAULT = 4  # velocity constraint slope [-] (dynamic velocity constraint)
R_MAX_DEFAULT = 1000  # max distance from chief [m] (translational keep out zone)
THETA_DEFAULT = jnp.pi / 6  # sun avoidance angle [rad] (translational keep out zone)
U_MAX_DEFAULT = 1  # Max thrust [N] (10. avoid actuation saturation)
X_VEL_LIMIT_DEFAULT = 2  # Maximum velocity limit [m/s] (Avoid aggressive maneuvering)
Y_VEL_LIMIT_DEFAULT = 2  # Maximum velocity limit [m/s] (Avoid aggressive maneuvering)
Z_VEL_LIMIT_DEFAULT = 2  # Maximum velocity limit [m/s] (Avoid aggressive maneuvering)


class Inspection3dState(RTAStateWrapper):
    """RTA state for inspection 3d RTA (only current deputy's state)"""

    @property
    def x(self) -> float:
        """Getter for x position"""
        return self.vector[0]

    @x.setter
    def x(self, val: float):
        """Setter for x position"""
        self.vector[0] = val

    @property
    def y(self) -> float:
        """Getter for y position"""
        return self.vector[1]

    @y.setter
    def y(self, val: float):
        """Setter for y position"""
        self.vector[1] = val

    @property
    def z(self) -> float:
        """Getter for z position"""
        return self.vector[2]

    @z.setter
    def z(self, val: float):
        """Setter for z position"""
        self.vector[2] = val

    @property
    def x_dot(self) -> float:
        """Getter for x velocity component"""
        return self.vector[3]

    @x_dot.setter
    def x_dot(self, val: float):
        """Setter for x velocity component"""
        self.vector[3] = val

    @property
    def y_dot(self) -> float:
        """Getter for y velocity component"""
        return self.vector[4]

    @y_dot.setter
    def y_dot(self, val: float):
        """Setter for y velocity component"""
        self.vector[4] = val

    @property
    def z_dot(self) -> float:
        """Getter for z velocity component"""
        return self.vector[5]

    @z_dot.setter
    def z_dot(self, val: float):
        """Setter for z velocity component"""
        self.vector[5] = val


class InspectionRTA(ExplicitASIFModule):
    """
    Implements Explicit Optimization RTA for the 3d Inspection problem

    Utilizes Explicit Active Set Invariance Filter algorithm

    Parameters
    ----------
    num_deputies : float, optional
        number of deputies in simulation, by default NUM_DEPUTIES_DEFAULT
    m : float, optional
        mass in kg of spacecraft, by default M_DEFAULT
    n : float, optional
        orbital mean motion in rad/s of current Hill's reference frame, by default N_DEFAULT
    chief_radius : float, optional
        radius of collision for chief spacecraft, by default CHIEF_RADIUS_DEFAULT
    deputy_radius : float, optional
        radius of collision for each deputy spacecraft, by default DEPUTY_RADIUS_DEFAULT
    v0 : float, optional
        Maximum safe docking velocity in m/s, by default V0_DEFAULT
        v0 of v_limit = v0 + v1*n*||r||
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r||
    r_max : float, optional
        maximum relative distance from chief, by default R_MAX_DEFAULT
    theta : float, optional
        sun avoidance angle (theta_EZ), by default THETA_DEFAULT
    x_vel_limit : float, optional
        max velocity magnitude in the x direction, by default X_VEL_LIMIT_DEFAULT
    y_vel_limit : float, optional
        max velocity magnitude in the y direction, by default Y_VEL_LIMIT_DEFAULT
    z_vel_limit : float, optional
        max velocity magnitude in the z direction, by default Z_VEL_LIMIT_DEFAULT
    control_bounds_high : float, optional
        upper bound of allowable control. Pass a list for element specific limit. By default U_MAX_DEFAULT
    control_bounds_low : float, optional
        lower bound of allowable control. Pass a list for element specific limit. By default -U_MAX_DEFAULT
    """

    def __init__(
        self,
        *args,
        num_deputies=NUM_DEPUTIES_DEFAULT,
        m: float = M_DEFAULT,
        n: float = N_DEFAULT,
        chief_radius: float = CHIEF_RADIUS_DEFAULT,
        deputy_radius: float = DEPUTY_RADIUS_DEFAULT,
        v0: float = V0_DEFAULT,
        v1_coef: float = V1_COEF_DEFAULT,
        r_max: float = R_MAX_DEFAULT,
        theta: float = THETA_DEFAULT,
        x_vel_limit: float = X_VEL_LIMIT_DEFAULT,
        y_vel_limit: float = Y_VEL_LIMIT_DEFAULT,
        z_vel_limit: float = Z_VEL_LIMIT_DEFAULT,
        control_bounds_high: Union[float, list, np.ndarray, jnp.ndarray] = U_MAX_DEFAULT,
        control_bounds_low: Union[float, list, np.ndarray, jnp.ndarray] = -U_MAX_DEFAULT,
        **kwargs
    ):
        self.num_deputies = num_deputies
        self.m = m
        self.n = n
        self.chief_radius = chief_radius
        self.deputy_radius = deputy_radius
        self.v0 = v0
        self.v1_coef = v1_coef
        self.v1 = self.v1_coef * self.n
        self.r_max = r_max
        self.theta = theta
        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit
        self.z_vel_limit = z_vel_limit

        self.e_hat = jnp.array([1, 0, 0])
        self.u_max = U_MAX_DEFAULT
        self.a_max = self.u_max / self.m - (3 * self.n**2 + 2 * self.n * self.v1) * self.r_max - 2 * self.n * self.v0
        A, B = generate_cwh_matrices(self.m, self.n, mode="3d")
        self.A = jnp.array(A)
        self.B = jnp.array(B)

        self.dynamics = BaseLinearODESolverDynamics(A=A, B=B, integration_method="RK45")

        A_n = jnp.copy(self.A)
        for _ in range(self.num_deputies - 1):
            A_n = scipy.linalg.block_diag(A_n, jnp.copy(self.A))
        self.A_n = jnp.array(A_n)

        B_n = jnp.copy(self.B)
        for _ in range(self.num_deputies - 1):
            B_n = jnp.vstack((B_n, jnp.zeros(self.B.shape)))
        self.B_n = jnp.array(B_n)

        self.control_dim = self.B.shape[1]

        super().__init__(
            *args, control_dim=self.control_dim, control_bounds_high=control_bounds_high, control_bounds_low=control_bounds_low, **kwargs
        )

    def _setup_constraints(self) -> OrderedDict:
        constraint_dict = OrderedDict(
            [
                ('rel_vel', ConstraintCWHRelativeVelocity(v0=self.v0, v1=self.v1, buffer=1e-4)),
                ('chief_collision', ConstraintCWHChiefCollision(collision_radius=self.chief_radius + self.deputy_radius, a_max=self.a_max)),
                ('sun', ConstraintCWHSunAvoidance(a_max=self.a_max, theta=self.theta, e_hat=self.e_hat)),
                (
                    'x_vel',
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.x_vel_limit, state_index=3, alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]), buffer=0.001
                    )
                ),
                (
                    'y_vel',
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.y_vel_limit, state_index=4, alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]), buffer=0.001
                    )
                ),
                (
                    'z_vel',
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.z_vel_limit, state_index=5, alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]), buffer=0.001
                    )
                )
            ]
        )
        for i in range(self.num_deputies - 1):
            constraint_dict[f'deputy_collision_{i+1}'] = ConstraintCWHDeputyCollision(
                collision_radius=self.deputy_radius * 2, a_max=self.a_max, deputy=i + 1
            )
        return constraint_dict

    def _pred_state(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        pass

    def state_transition_system(self, state: jnp.ndarray) -> jnp.ndarray:
        return self.A_n @ state

    def state_transition_input(self, state: jnp.ndarray) -> jnp.ndarray:
        return self.B_n


class ConstraintCWHRelativeVelocity(ConstraintModule):
    """CWH dynamic velocity constraint

    Parameters
    ----------
    v0: float
        NMT safety constraint velocity upper bound constatnt component where ||v|| <= v0 + v1*distance. m/s
    v1: float
        NMT safety constraint velocity upper bound distance proportinality coefficient where
        ||v|| <= v0 + v1*distance. 1/s
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.05, 0, 0.5])
    """

    def __init__(self, v0: float, v1: float, alpha: ConstraintStrengthener = None, **kwargs):
        self.v0 = v0
        self.v1 = v1

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.01, 0, 0.001])
        super().__init__(alpha=alpha, **kwargs)

    def _compute(self, state: jnp.ndarray) -> float:
        h = (self.v0 + self.v1 * jnp.linalg.norm(state[0:3])) - jnp.linalg.norm(state[3:6])
        return h


class ConstraintCWHChiefCollision(ConstraintModule):
    """CWH chief collision avoidance constraint

    Parameters
    ----------
    collision_radius: float
        radius of collision for chief spacecraft. m
    a_max: float
        Maximum braking acceleration for deputy spacecraft. m/s^2
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.005, 0, 0.05])
    """

    def __init__(self, collision_radius: float, a_max: float, alpha: ConstraintStrengthener = None, **kwargs):
        self.collision_radius = collision_radius
        self.a_max = a_max

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.01, 0, 0.001])
        super().__init__(alpha=alpha, **kwargs)

    def _compute(self, state: jnp.ndarray) -> float:
        delta_p = state[0:3]
        delta_v = state[3:6]
        mag_delta_p = jnp.linalg.norm(delta_p)
        h = jnp.sqrt(2 * self.a_max * (mag_delta_p - self.collision_radius)) + delta_p.T @ delta_v / mag_delta_p
        return h

    def _phi(self, state: jnp.ndarray) -> float:
        delta_p = state[0:3]
        return jnp.linalg.norm(delta_p) - self.collision_radius


class ConstraintCWHDeputyCollision(ConstraintModule):
    """CWH deputy collision avoidance constraint

    Parameters
    ----------
    collision_radius: float
        radius of collision for deputy spacecraft. m
    a_max: float
        Maximum braking acceleration for each deputy spacecraft. m/s^2
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.005, 0, 0.05])
    """

    def __init__(self, collision_radius: float, a_max: float, deputy: float, alpha: ConstraintStrengthener = None, **kwargs):
        self.collision_radius = collision_radius
        self.a_max = a_max
        self.deputy = deputy

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.01, 0, 0.001])
        super().__init__(alpha=alpha, **kwargs)

    def _compute(self, state: jnp.ndarray) -> float:
        delta_p = state[0:3] - state[int(self.deputy * 6):int(self.deputy * 6 + 3)]
        delta_v = state[3:6] - state[int(self.deputy * 6 + 3):int(self.deputy * 6 + 6)]
        mag_delta_p = jnp.linalg.norm(delta_p)
        h = jnp.sqrt(4 * self.a_max * (mag_delta_p - self.collision_radius)) + delta_p.T @ delta_v / mag_delta_p
        return h

    def _phi(self, state: jnp.ndarray) -> float:
        delta_p = state[0:3] - state[int(self.deputy * 6):int(self.deputy * 6 + 3)]
        return jnp.linalg.norm(delta_p) - self.collision_radius


class ConstraintCWHSunAvoidance(ConstraintModule):
    """CWH sun avoidance constraint
    Assumes deputy is always pointing at chief, and sun is not moving

    Parameters
    ----------
    a_max: float
        Maximum braking acceleration for each deputy spacecraft. m/s^2
    theta: float
        sun avoidance angle (theta_EZ). radians
    e_hat: float
        Normal vector pointing from chief to sun.
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.01, 0, 0.05])
    """

    def __init__(self, a_max: float, theta: float, e_hat: jnp.ndarray, alpha: ConstraintStrengthener = None, **kwargs):
        self.a_max = a_max
        self.theta = theta
        self.e_hat = e_hat
        self.e_hat_vel = jnp.array([0, 0, 0])

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.001, 0, 0.001])
        super().__init__(alpha=alpha, **kwargs)

    def _compute(self, state: jnp.ndarray) -> float:
        p = state[0:3]
        p_es = p - jnp.dot(p, self.e_hat) * self.e_hat
        a = jnp.cos(self.theta) * (jnp.linalg.norm(p_es) - jnp.tan(self.theta) * jnp.dot(p, self.e_hat))
        p_pr = p + a * jnp.sin(self.theta) * self.e_hat + a * jnp.cos(self.theta
                                                                      ) * (jnp.dot(p, self.e_hat) * self.e_hat - p) / jnp.linalg.norm(p_es)

        h = jnp.sqrt(2 * self.a_max * jnp.linalg.norm(p - p_pr)
                     ) + jnp.dot(p - p_pr, state[3:6] - self.e_hat_vel) / jnp.linalg.norm(p - p_pr)
        return h

    def _phi(self, state: jnp.ndarray) -> float:
        h = jnp.arccos(jnp.dot(state[0:3], self.e_hat) / (jnp.linalg.norm(state[0:3]) * jnp.linalg.norm(self.e_hat))) - self.theta
        return h
