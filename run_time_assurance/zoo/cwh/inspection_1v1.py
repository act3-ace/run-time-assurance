"""This module implements RTA methods for the single-agent inspection problem with 3D CWH dynamics models
"""
from collections import OrderedDict
from typing import Union

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.experimental.ode import odeint
from safe_autonomy_dynamics.cwh.point_model import M_DEFAULT, N_DEFAULT, generate_cwh_matrices

from run_time_assurance.constraint import (
    ConstraintMagnitudeStateLimit,
    ConstraintModule,
    ConstraintStrengthener,
    PolynomialConstraintStrengthener,
)
from run_time_assurance.rta import ExplicitASIFModule
from run_time_assurance.state import RTAStateWrapper

CHIEF_RADIUS_DEFAULT = 5  # chief radius of collision [m] (collision freedom)
DEPUTY_RADIUS_DEFAULT = 5  # deputy radius of collision [m] (collision freedom)
V0_DEFAULT = 0.2  # maximum docking speed [m/s] (dynamic velocity constraint)
V1_COEF_DEFAULT = 2  # velocity constraint slope [-] (dynamic velocity constraint)
R_MAX_DEFAULT = 1000  # max distance from chief [m] (translational keep out zone)
FOV_DEFAULT = 60 * jnp.pi / 180  # sun avoidance angle [rad] (translational keep out zone)
U_MAX_DEFAULT = 1  # Max thrust [N] (10. avoid actuation saturation)
VEL_LIMIT_DEFAULT = 1  # Maximum velocity limit [m/s] (Avoid aggressive maneuvering)
FUEL_LIMIT_DEFAULT = 1  # kg
GRAVITY = 9.81  # m/s^2
SPECIFIC_IMPULSE_DEFAULT = 220  # s


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

    @property
    def sun_angle(self) -> float:
        """Getter for sun_angle component"""
        return self.vector[6]

    @sun_angle.setter
    def sun_angle(self, val: float):
        """Setter for sun_angle component"""
        self.vector[6] = val

    @property
    def m_fuel(self) -> float:
        """Getter for fuel mass component"""
        return self.vector[7]

    @m_fuel.setter
    def m_fuel(self, val: float):
        """Setter for fuel mass component"""
        self.vector[7] = val


class InspectionRTA(ExplicitASIFModule):
    """
    Implements Explicit Optimization RTA for the 3d Inspection problem

    Utilizes Explicit Active Set Invariance Filter algorithm

    Parameters
    ----------
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
    fov : float, optional
        sensor field of view, by default FOV_DEFAULT
    vel_limit : float, optional
        max velocity magnitude, by default VEL_LIMIT_DEFAULT
    fuel_limit : float, optional
        maximum fuel used limit, by default FUEL_LIMIT_DEFAULT
    control_bounds_high : float, optional
        upper bound of allowable control. Pass a list for element specific limit. By default U_MAX_DEFAULT
    control_bounds_low : float, optional
        lower bound of allowable control. Pass a list for element specific limit. By default -U_MAX_DEFAULT
    """

    def __init__(
        self,
        *args,
        m: float = M_DEFAULT,
        n: float = N_DEFAULT,
        chief_radius: float = CHIEF_RADIUS_DEFAULT,
        deputy_radius: float = DEPUTY_RADIUS_DEFAULT,
        v0: float = V0_DEFAULT,
        v1_coef: float = V1_COEF_DEFAULT,
        r_max: float = R_MAX_DEFAULT,
        fov: float = FOV_DEFAULT,
        vel_limit: float = VEL_LIMIT_DEFAULT,
        fuel_limit: float = FUEL_LIMIT_DEFAULT,
        gravity: float = GRAVITY,
        isp: float = SPECIFIC_IMPULSE_DEFAULT,
        control_bounds_high: Union[float, list, np.ndarray, jnp.ndarray] = U_MAX_DEFAULT,
        control_bounds_low: Union[float, list, np.ndarray, jnp.ndarray] = -U_MAX_DEFAULT,
        **kwargs
    ):
        self.m = m
        self.n = n
        self.chief_radius = chief_radius
        self.deputy_radius = deputy_radius
        self.v0 = v0
        self.v1_coef = v1_coef
        self.v1 = self.v1_coef * self.n
        self.r_max = r_max
        self.fov = fov
        self.vel_limit = vel_limit
        self.fuel_limit = fuel_limit
        self.gravity = gravity
        self.isp = isp

        self.u_max = U_MAX_DEFAULT
        self.a_max = self.u_max / self.m - (3 * self.n**2 + 2 * self.n * self.v1) * self.r_max - 2 * self.n * self.v0
        A, B = generate_cwh_matrices(self.m, self.n, mode="3d")
        self.A = jnp.array(A)
        self.B = jnp.array(B)

        self.control_dim = self.B.shape[1]

        self.pred_state_fn = jit(self._pred_state, static_argnames=['step_size'])

        super().__init__(
            *args, control_dim=self.control_dim, control_bounds_high=control_bounds_high, control_bounds_low=control_bounds_low, **kwargs
        )

    def _setup_constraints(self) -> OrderedDict:
        constraint_dict = OrderedDict(
            [
                ('chief_collision', ConstraintCWHChiefCollision(collision_radius=self.chief_radius + self.deputy_radius, a_max=self.a_max)),
                ('rel_vel', ConstraintCWHRelativeVelocity(v0=self.v0, v1=self.v1, buffer=1e-4)),
                ('sun', ConstraintCWHSunAvoidance(a_max=self.a_max, fov=self.fov, sun_vel=jnp.array([0, 0, -self.n]))),
                ('r_max', ConstraintCWHMaxDistance(r_max=self.r_max, a_max=self.a_max)),
                (
                    'PSM',
                    ConstraintPassivelySafeManeuver(
                        collision_radius=self.chief_radius + self.deputy_radius, m=self.m, n=self.n, dt=1, steps=100
                    )
                ),
                (
                    'x_vel',
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.vel_limit, state_index=3, alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]), buffer=0.001
                    )
                ),
                (
                    'y_vel',
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.vel_limit, state_index=4, alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]), buffer=0.001
                    )
                ),
                (
                    'z_vel',
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.vel_limit, state_index=5, alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]), buffer=0.001
                    )
                ),
            ]
        )
        return constraint_dict

    def _pred_state(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        sol = odeint(self.compute_state_dot, state, jnp.linspace(0., step_size, 11), control)
        return sol[-1, :]

    def compute_state_dot(self, x, t, u):
        """Computes state dot for ODE integration
        """
        xd = self.A @ x[0:6] + self.B @ u + 0 * t
        fuel = jnp.sum(jnp.abs(u)) / (self.gravity * self.isp)
        return jnp.concatenate((xd, jnp.array([-self.n, fuel])))

    def state_transition_system(self, state: jnp.ndarray) -> jnp.ndarray:
        xd = self.A @ state[0:6]
        return jnp.concatenate((xd, jnp.array([-self.n, 0])))

    def state_transition_input(self, state: jnp.ndarray) -> jnp.ndarray:
        return jnp.vstack((self.B, jnp.zeros((2, 3))))


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


class ConstraintCWHSunAvoidance(ConstraintModule):
    """CWH sun avoidance constraint
    Assumes deputy is always pointing at chief, and sun is not moving

    Parameters
    ----------
    a_max: float
        Maximum braking acceleration for each deputy spacecraft. m/s^2
    fov: float
        sensor field of view. radians
    sun_vel: float
        sun velocity vector
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.01, 0, 0.05])
    """

    def __init__(self, a_max: float, fov: float, sun_vel: float, alpha: ConstraintStrengthener = None, **kwargs):
        self.a_max = a_max
        self.fov = fov
        self.sun_vel = sun_vel

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.001, 0, 0.001])
        super().__init__(alpha=alpha, **kwargs)

    def _compute(self, state: jnp.ndarray) -> float:
        p = state[0:3]
        theta = self.fov / 2
        r_s_hat = -jnp.array([jnp.cos(state[6]), jnp.sin(state[6]), 0.])
        p_es = p - jnp.dot(p, r_s_hat) * r_s_hat
        a = jnp.cos(theta) * (jnp.linalg.norm(p_es) - jnp.tan(theta) * jnp.dot(p, r_s_hat))
        p_pr = p + a * jnp.sin(theta) * r_s_hat + a * jnp.cos(theta) * (jnp.dot(p, r_s_hat) * r_s_hat - p) / jnp.linalg.norm(p_es)

        h = jnp.sqrt(2 * self.a_max * jnp.linalg.norm(p - p_pr)) + jnp.dot(p - p_pr, state[3:6] - self.sun_vel) / jnp.linalg.norm(p - p_pr)
        return h

    def _phi(self, state: jnp.ndarray) -> float:
        r_s_hat = jnp.array([jnp.cos(state[6]), jnp.sin(state[6]), 0.])
        r_b_hat = -state[0:3] / jnp.linalg.norm(state[0:3])
        h = jnp.arccos(jnp.dot(r_s_hat, r_b_hat)) - self.fov / 2
        return h


class ConstraintPassivelySafeManeuver(ConstraintModule):
    """Passively Safe Maneuver Constraint
    Assures that deputy will never collide with chief if there is a fault and u=0

    Parameters
    ----------
    collision_radius: float
        radius of collision for deputy spacecraft. m
    m: float
        mass of deputy. kg
    n: float
        mean motion. rad/sec
    dt: float
        time step for integration. sec
    steps: int
        total number of steps to integrate over
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.01, 0, 0.001])
    """

    def __init__(self, collision_radius: float, m: float, n: float, dt: float, steps: int, alpha: ConstraintStrengthener = None):
        self.n = n
        self.collision_radius = collision_radius
        self.dt = dt
        self.steps = steps
        A, _ = generate_cwh_matrices(m, n, mode="3d")
        self.A = jnp.array(A)

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.01, 0, 0.001])
        super().__init__(alpha=alpha)

    def _compute(self, state: jnp.ndarray) -> float:
        vmapped_get_future_state = vmap(self.get_future_state, (None, 0), 0)
        phi_array = vmapped_get_future_state(state, jnp.linspace(self.dt, self.steps * self.dt, self.steps))
        return jnp.min(phi_array)

    def get_future_state(self, state, t):
        """Gets future state using closed form CWH dynamics (http://www.ae.utexas.edu/courses/ase366k/cw_equations.pdf)
        """
        x = (4 - 3 * jnp.cos(self.n * t)) * state[0] + jnp.sin(self.n * t
                                                               ) * state[3] / self.n + 2 / self.n * (1 - jnp.cos(self.n * t)) * state[4]
        y = 6 * (jnp.sin(self.n * t) - self.n * t) * state[0] + state[
            1] - 2 / self.n * (1 - jnp.cos(self.n * t)) * state[3] + (4 * jnp.sin(self.n * t) - 3 * self.n * t) * state[4] / self.n
        z = state[2] * jnp.cos(self.n * t) + state[5] / self.n * jnp.sin(self.n * t)
        return jnp.linalg.norm([x, y, z]) - self.collision_radius

    def get_array(self, state: jnp.ndarray) -> float:
        """Gets entire trajectory array
        """
        vmapped_get_future_state = vmap(self.get_future_state, (None, 0), 0)
        phi_array = vmapped_get_future_state(state, jnp.linspace(0, self.steps * self.dt, self.steps + 1))
        return phi_array


class ConstraintCWHMaxDistance(ConstraintModule):
    """CWH chief collision avoidance constraint

    Parameters
    ----------
    r_max: float
        maximum relative distance. m
    a_max: float
        Maximum braking acceleration for deputy spacecraft. m/s^2
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.005, 0, 0.05])
    """

    def __init__(self, r_max: float, a_max: float, alpha: ConstraintStrengthener = None, **kwargs):
        self.r_max = r_max
        self.a_max = a_max

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.002, 0, 0.0002])
        super().__init__(alpha=alpha, **kwargs)

    def _compute(self, state: jnp.ndarray) -> float:
        delta_p = state[0:3]
        delta_v = state[3:6]
        mag_delta_p = jnp.linalg.norm(delta_p)
        h = jnp.sqrt(2 * self.a_max * (self.r_max - mag_delta_p)) - delta_p.T @ delta_v / mag_delta_p
        return h

    def _phi(self, state: jnp.ndarray) -> float:
        delta_p = state[0:3]
        return self.r_max - jnp.linalg.norm(delta_p)
