"""This module implements RTA methods for the single-agent inspection problem with 3D CWH dynamics models"""

from collections import OrderedDict
from typing import Union

import jax.numpy as jnp
import numpy as np
import scipy
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import safe_autonomy_simulation.sims.spacecraft.defaults as defaults

from run_time_assurance.constraint import (
    ConstraintMagnitudeStateLimit,
    ConstraintModule,
    ConstraintStrengthener,
    HOCBFConstraint,
    PolynomialConstraintStrengthener,
)
from run_time_assurance.rta import (
    CascadedRTA,
    DiscreteASIFModule,
    ExplicitASIFModule,
    RTAModule,
)
from run_time_assurance.state import RTAStateWrapper
from run_time_assurance.utils import to_jnp_array_jit
from run_time_assurance.zoo.cwh.docking_3d import ConstraintCWH3dRelativeVelocity
from run_time_assurance.zoo.cwh.utils import generate_cwh_matrices

CHIEF_RADIUS_DEFAULT = 5  # chief radius of collision [m] (collision freedom)
DEPUTY_RADIUS_DEFAULT = 5  # deputy radius of collision [m] (collision freedom)
V0_DEFAULT = 0.2  # maximum docking speed [m/s] (dynamic velocity constraint)
V1_COEF_DEFAULT = 2  # velocity constraint slope [-] (dynamic velocity constraint)
V0_DISTANCE_DEFAULT = 0  # distance where v0 is applied
R_MAX_DEFAULT = 1000  # max distance from chief [m] (translational keep out zone)
FOV_DEFAULT = (
    60 * jnp.pi / 180
)  # sun avoidance angle [rad] (translational keep out zone)
U_MAX_DEFAULT = 1  # Max thrust [N] (avoid actuation saturation)
VEL_LIMIT_DEFAULT = 1  # Maximum velocity limit [m/s] (Avoid aggressive maneuvering)
DELTA_V_LIMIT_DEFAULT = 20  # Delta v limit [m/s]
SUN_VEL_DEFAULT = -defaults.N_DEFAULT  # Speed of sun rotation in x-y plane


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
    def delta_v(self) -> float:
        """Getter for delta_v component"""
        return self.vector[7]

    @delta_v.setter
    def delta_v(self, val: float):
        """Setter for delta_v component"""
        self.vector[7] = val


class Inspection1v1RTA(ExplicitASIFModule):
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
        v0 of v_limit = v0 + v1*n*||r-v0_distance||
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r-v0_distance||
    v0_distance: float
        NMT safety constraint minimum distance where v0 is applied. By default 0.
    r_max : float, optional
        maximum relative distance from chief, by default R_MAX_DEFAULT
    fov : float, optional
        sensor field of view, by default FOV_DEFAULT
    vel_limit : float, optional
        max velocity magnitude, by default VEL_LIMIT_DEFAULT
    delta_v_limit : float, optional
        maximum delta_v used limit, by default DELTA_V_LIMIT_DEFAULT
    sun_vel : float, optional
        velocity of sun in x-y plane (rad/sec), by default SUN_VEL_DEFAULT
    use_hocbf : bool, optional
        True to use example HOCBF for collision avoidance, by default False
    control_bounds_high : float, optional
        upper bound of allowable control. Pass a list for element specific limit. By default U_MAX_DEFAULT
    control_bounds_low : float, optional
        lower bound of allowable control. Pass a list for element specific limit. By default -U_MAX_DEFAULT
    """

    def __init__(  # pylint:disable=too-many-arguments
        self,
        *args,
        m: float = defaults.M_DEFAULT,
        n: float = defaults.N_DEFAULT,
        chief_radius: float = CHIEF_RADIUS_DEFAULT,
        deputy_radius: float = DEPUTY_RADIUS_DEFAULT,
        v0: float = V0_DEFAULT,
        v1_coef: float = V1_COEF_DEFAULT,
        v0_distance: float = V0_DISTANCE_DEFAULT,
        r_max: float = R_MAX_DEFAULT,
        fov: float = FOV_DEFAULT,
        vel_limit: float = VEL_LIMIT_DEFAULT,
        delta_v_limit: float = DELTA_V_LIMIT_DEFAULT,
        sun_vel: float = SUN_VEL_DEFAULT,
        use_hocbf: bool = False,
        control_bounds_high: Union[
            float, list, np.ndarray, jnp.ndarray
        ] = U_MAX_DEFAULT,
        control_bounds_low: Union[
            float, list, np.ndarray, jnp.ndarray
        ] = -U_MAX_DEFAULT,
        **kwargs,
    ):
        self.m = m
        self.n = n
        self.chief_radius = chief_radius
        self.deputy_radius = deputy_radius
        self.v0 = v0
        self.v1_coef = v1_coef
        self.v1 = self.v1_coef * self.n
        self.v0_distance = v0_distance
        self.r_max = r_max
        self.fov = fov
        self.vel_limit = vel_limit
        self.delta_v_limit = delta_v_limit
        self.sun_vel = sun_vel
        self.use_hocbf = use_hocbf

        self.u_max = U_MAX_DEFAULT
        vmax = min(self.vel_limit, self.v0 + self.v1 * self.r_max)
        self.a_max = (
            self.u_max / self.m - 3 * self.n**2 * self.r_max - 2 * self.n * vmax
        )
        A, B = generate_cwh_matrices(self.m, self.n, mode="3d")
        self.A = jnp.array(A)
        self.B = jnp.array(B)

        self.control_dim = self.B.shape[1]

        self._pred_state_fn = jit(self._pred_state, static_argnames=["step_size"])
        self._pred_state_cwh_fn = jit(
            self._pred_state_cwh, static_argnames=["step_size"]
        )

        super().__init__(
            *args,
            control_dim=self.control_dim,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            **kwargs,
        )

    def _setup_constraints(self) -> OrderedDict:
        constraint_dict = OrderedDict(
            [
                (
                    "rel_vel",
                    ConstraintCWH3dRelativeVelocity(
                        v0=self.v0, v1=self.v1, v0_distance=self.v0_distance, bias=-1e-3
                    ),
                ),
                (
                    "sun",
                    ConstraintCWHConicKeepOutZone(
                        a_max=self.a_max,
                        fov=self.fov,
                        get_pos=self.get_pos_vector,
                        get_vel=self.get_vel_vector,
                        get_cone_vec=self.get_sun_vector,
                        cone_ang_vel=jnp.array([0, 0, self.sun_vel]),
                        bias=-1e-3,
                    ),
                ),
                (
                    "r_max",
                    ConstraintCWHMaxDistance(
                        r_max=self.r_max, a_max=self.a_max, bias=-1e-3
                    ),
                ),
                (
                    "PSM",
                    ConstraintPassivelySafeManeuver(
                        collision_radius=self.chief_radius + self.deputy_radius,
                        m=self.m,
                        n=self.n,
                        dt=1,
                        steps=100,
                    ),
                ),
                (
                    "x_vel",
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.vel_limit,
                        state_index=3,
                        alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]),
                        bias=-0.001,
                    ),
                ),
                (
                    "y_vel",
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.vel_limit,
                        state_index=4,
                        alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]),
                        bias=-0.001,
                    ),
                ),
                (
                    "z_vel",
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.vel_limit,
                        state_index=5,
                        alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]),
                        bias=-0.001,
                    ),
                ),
            ]
        )

        if self.use_hocbf:
            constraint_dict["chief_collision_hocbf"] = HOCBFConstraint(
                HOCBFExampleChiefCollision(
                    collision_radius=self.chief_radius + self.deputy_radius
                ),
                relative_degree=2,
                state_transition_system=self.state_transition_system,
                alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.1]),
            )
        else:
            constraint_dict["chief_collision"] = ConstraintCWHChiefCollision(
                collision_radius=self.chief_radius + self.deputy_radius,
                a_max=self.a_max,
            )

        return constraint_dict

    def _pred_state(
        self, state: jnp.ndarray, step_size: float, control: jnp.ndarray
    ) -> jnp.ndarray:
        sol = odeint(
            self.compute_state_dot, state, jnp.linspace(0.0, step_size, 11), control
        )
        return sol[-1, :]

    def compute_state_dot(self, x, t, u):
        """Computes state dot for ODE integration"""
        xd = self.A @ x[0:6] + self.B @ u + 0 * t
        delta_v = jnp.sum(jnp.abs(u)) / (self.m)
        return jnp.concatenate((xd, jnp.array([self.sun_vel, delta_v])))

    def _pred_state_cwh(
        self, state: jnp.ndarray, step_size: float, control: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicted state for only CWH equations"""
        sol = odeint(
            self.compute_state_dot_cwh, state, jnp.linspace(0.0, step_size, 11), control
        )
        return sol[-1, :]

    def compute_state_dot_cwh(self, x, t, u):
        """Computes state dot for ODE integration (only CWH equations)"""
        xd = self.A @ x[0:6] + self.B @ u + 0 * t
        return xd

    def state_transition_system(self, state: jnp.ndarray) -> jnp.ndarray:
        xd = self.A @ state[0:6]
        return jnp.concatenate((xd, jnp.array([self.sun_vel, 0])))

    def state_transition_input(self, state: jnp.ndarray) -> jnp.ndarray:
        return jnp.vstack((self.B, jnp.zeros((2, 3))))

    def _get_state(self, input_state) -> jnp.ndarray:
        assert isinstance(
            input_state, (np.ndarray, jnp.ndarray)
        ), "input_state must be an RTAState or numpy array."
        input_state = np.array(input_state)

        if len(input_state) < 8:
            input_state = np.concatenate((input_state, np.array([0.0, 0.0])))
            self.sun_vel = 0

        return to_jnp_array_jit(input_state)

    def get_sun_vector(self, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get vector pointing from sun to chief"""
        return -jnp.array([jnp.cos(state[6]), jnp.sin(state[6]), 0.0])

    def get_pos_vector(self, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get position vector"""
        return state[0:3]

    def get_vel_vector(self, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get velocity vector"""
        return state[3:6]


class DiscreteInspection1v1RTA(DiscreteASIFModule):
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
        v0 of v_limit = v0 + v1*n*||r-v0_distance||
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r-v0_distance||
    v0_distance: float
        NMT safety constraint minimum distance where v0 is applied. By default 0.
    r_max : float, optional
        maximum relative distance from chief, by default R_MAX_DEFAULT
    fov : float, optional
        sensor field of view, by default FOV_DEFAULT
    vel_limit : float, optional
        max velocity magnitude, by default VEL_LIMIT_DEFAULT
    delta_v_limit : float, optional
        maximum delta_v used limit, by default DELTA_V_LIMIT_DEFAULT
    sun_vel : float, optional
        velocity of sun in x-y plane (rad/sec), by default SUN_VEL_DEFAULT
    control_bounds_high : float, optional
        upper bound of allowable control. Pass a list for element specific limit. By default U_MAX_DEFAULT
    control_bounds_low : float, optional
        lower bound of allowable control. Pass a list for element specific limit. By default -U_MAX_DEFAULT
    """

    def __init__(
        self,
        *args,
        m: float = defaults.M_DEFAULT,
        n: float = defaults.N_DEFAULT,
        chief_radius: float = CHIEF_RADIUS_DEFAULT,
        deputy_radius: float = DEPUTY_RADIUS_DEFAULT,
        v0: float = V0_DEFAULT,
        v1_coef: float = V1_COEF_DEFAULT,
        v0_distance: float = V0_DISTANCE_DEFAULT,
        r_max: float = R_MAX_DEFAULT,
        fov: float = FOV_DEFAULT,
        vel_limit: float = VEL_LIMIT_DEFAULT,
        delta_v_limit: float = DELTA_V_LIMIT_DEFAULT,
        sun_vel: float = SUN_VEL_DEFAULT,
        control_bounds_high: Union[
            float, list, np.ndarray, jnp.ndarray
        ] = U_MAX_DEFAULT,
        control_bounds_low: Union[
            float, list, np.ndarray, jnp.ndarray
        ] = -U_MAX_DEFAULT,
        **kwargs,
    ):
        self.m = m
        self.n = n
        self.chief_radius = chief_radius
        self.deputy_radius = deputy_radius
        self.v0 = v0
        self.v1_coef = v1_coef
        self.v1 = self.v1_coef * self.n
        self.v0_distance = v0_distance
        self.r_max = r_max
        self.fov = fov
        self.vel_limit = vel_limit
        self.delta_v_limit = delta_v_limit
        self.sun_vel = sun_vel

        self.u_max = U_MAX_DEFAULT
        vmax = min(self.vel_limit, self.v0 + self.v1 * self.r_max)
        self.a_max = (
            self.u_max / self.m - 3 * self.n**2 * self.r_max - 2 * self.n * vmax
        )
        A, B = generate_cwh_matrices(self.m, self.n, mode="3d")
        self.A = jnp.array(A)
        self.B = jnp.array(B)

        self.control_dim = self.B.shape[1]

        super().__init__(
            *args,
            control_dim=self.control_dim,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            **kwargs,
        )

    def _setup_constraints(self) -> OrderedDict:
        constraint_dict = OrderedDict(
            [
                (
                    "chief_collision",
                    ConstraintCWHChiefCollision(
                        collision_radius=self.chief_radius + self.deputy_radius,
                        a_max=self.a_max,
                    ),
                ),
                (
                    "rel_vel",
                    ConstraintCWH3dRelativeVelocity(
                        v0=self.v0,
                        v1=self.v1,
                        v0_distance=self.v0_distance,
                        bias=-1e-3,
                        alpha=PolynomialConstraintStrengthener([0, 0.01, 0, 0.001]),
                    ),
                ),
                (
                    "sun",
                    ConstraintCWHConicKeepOutZone(
                        a_max=self.a_max,
                        fov=self.fov,
                        get_pos=self.get_pos_vector,
                        get_vel=self.get_vel_vector,
                        get_cone_vec=self.get_sun_vector,
                        cone_ang_vel=jnp.array([0, 0, self.sun_vel]),
                        bias=-1e-3,
                        alpha=PolynomialConstraintStrengthener([0, 0.001, 0, 0.001]),
                    ),
                ),
                (
                    "r_max",
                    ConstraintCWHMaxDistance(
                        r_max=self.r_max,
                        a_max=self.a_max,
                        bias=-1e-3,
                        alpha=PolynomialConstraintStrengthener([0, 0.001, 0, 0.001]),
                    ),
                ),
                (
                    "PSM",
                    ConstraintPassivelySafeManeuver(
                        collision_radius=self.chief_radius + self.deputy_radius,
                        m=self.m,
                        n=self.n,
                        dt=1,
                        steps=100,
                        alpha=PolynomialConstraintStrengthener([0, 0.001, 0, 0.001]),
                    ),
                ),
                (
                    "x_vel",
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.vel_limit,
                        state_index=3,
                        alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]),
                        bias=-0.001,
                    ),
                ),
                (
                    "y_vel",
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.vel_limit,
                        state_index=4,
                        alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]),
                        bias=-0.001,
                    ),
                ),
                (
                    "z_vel",
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.vel_limit,
                        state_index=5,
                        alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]),
                        bias=-0.001,
                    ),
                ),
            ]
        )
        return constraint_dict

    def next_state_differentiable(self, state, step_size, control):
        next_state = (
            self.discrete_a_mat(step_size) @ state[0:6]
            + self.discrete_b_mat(step_size) @ control
        )
        return jnp.concatenate(
            (next_state, jnp.array([state[6] - self.sun_vel * step_size, 0.0]))
        )  # TODO: fix fuel derivative

    def discrete_a_mat(self, dt):
        """Discrete CWH dynamics"""
        c = jnp.cos(self.n * dt)
        s = jnp.sin(self.n * dt)
        return jnp.array(
            [
                [4 - 3 * c, 0, 0, 1 / self.n * s, 2 / self.n * (1 - c), 0],
                [
                    6 * (s - self.n * dt),
                    1,
                    0,
                    -2 / self.n * (1 - c),
                    1 / self.n * (4 * s - 3 * self.n * dt),
                    0,
                ],
                [0, 0, c, 0, 0, 1 / self.n * s],
                [3 * self.n * s, 0, 0, c, 2 * s, 0],
                [-6 * self.n * (1 - c), 0, 0, -2 * s, 4 * c - 3, 0],
                [0, 0, -self.n * s, 0, 0, c],
            ]
        )

    def discrete_b_mat(self, dt):
        """Discrete CWH dynamics"""
        c = jnp.cos(self.n * dt)
        s = jnp.sin(self.n * dt)
        return (
            jnp.array(
                [
                    [1 / self.n**2 * (1 - c), 2 / self.n**2 * (self.n * dt - s), 0],
                    [
                        -2 / self.n**2 * (self.n * dt - s),
                        4 / self.n**2 * (1 - c) - 3 / 2 * dt**2,
                        0,
                    ],
                    [0, 0, 1 / self.n**2 * (1 - c)],
                    [1 / self.n * s, 2 / self.n * (1 - c), 0],
                    [-2 / self.n * (1 - c), 4 / self.n * s - 3 * dt, 0],
                    [0, 0, 1 / self.n * s],
                ]
            )
            / self.m
        )

    def get_sun_vector(self, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get vector pointing from sun to chief"""
        return -jnp.array([jnp.cos(state[6]), jnp.sin(state[6]), 0.0])

    def get_pos_vector(self, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get position vector"""
        return state[0:3]

    def get_vel_vector(self, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get velocity vector"""
        return state[3:6]


class SwitchingDeltaVLimitRTA(RTAModule):
    """Explicit Simplex RTA Filter for delta v use
    Switches to NMT tracking backup controller

    Parameters
    ----------
    m : float, optional
        mass in kg of spacecraft, by default M_DEFAULT
    n : float, optional
        orbital mean motion in rad/s of current Hill's reference frame, by default N_DEFAULT
    delta_v_limit : float, optional
        maximum delta_v used limit, by default DELTA_V_LIMIT_DEFAULT
    n : int, optional
        Number of steps in backup trajectory. By default 500.
    control_bounds_high : float, optional
        upper bound of allowable control. Pass a list for element specific limit. By default U_MAX_DEFAULT
    control_bounds_low : float, optional
        lower bound of allowable control. Pass a list for element specific limit. By default -U_MAX_DEFAULT
    """

    def __init__(
        self,
        *args,
        m: float = defaults.M_DEFAULT,
        n: float = defaults.N_DEFAULT,
        delta_v_limit: float = DELTA_V_LIMIT_DEFAULT,
        n_steps: int = 500,
        control_bounds_high: Union[float, np.ndarray] = U_MAX_DEFAULT,
        control_bounds_low: Union[float, np.ndarray] = -U_MAX_DEFAULT,
        **kwargs,
    ):
        self.m = m
        self.n = n
        self.delta_v_limit = delta_v_limit
        self.n_steps = n_steps

        self.error_integral = np.zeros(6)

        # LQR Gain Matrices
        Q = np.eye(12) * 1e-5
        R = np.eye(3) * 1e5
        C = np.eye(6)
        self.A, self.B = generate_cwh_matrices(m, self.n, mode="3d")

        A_int = np.vstack(
            (np.hstack((self.A, np.zeros((6, 6)))), np.hstack((C, np.zeros((6, 6)))))
        )
        B_int = np.vstack((self.B, np.zeros((6, 3))))
        # Solve the Algebraic Ricatti equation for the given system
        P = scipy.linalg.solve_continuous_are(A_int, B_int, Q, R)
        # Construct the constain gain matrix, K
        K = np.linalg.inv(R) @ (np.transpose(B_int) @ P)
        self.K_1 = K[:, 0:6]
        self.K_2 = K[:, 6:]

        self.latched = False

        super().__init__(
            *args,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            **kwargs,
        )

    def _pred_state(
        self, state: np.ndarray, step_size: float, control: np.ndarray
    ) -> np.ndarray:
        xd = self.A @ state[0:6] + self.B @ control
        sun_dot = np.array([-self.n])
        delta_v_dot = np.array([np.sum(np.abs(control)) / (self.m)])
        return state + np.concatenate((xd, sun_dot, delta_v_dot)) * step_size

    def compute_filtered_control(
        self, input_state, step_size: float, control_desired: np.ndarray
    ) -> np.ndarray:
        if not self.latched:
            pred_state = self._pred_state(input_state, step_size, control_desired)
            error_integral = self.error_integral
            for _ in range(self.n_steps):
                ub, error_integral = self.backup_control(
                    pred_state, step_size, error_integral
                )
                pred_state = self._pred_state(pred_state, step_size, ub)
                if pred_state[7] > self.delta_v_limit:
                    ub, self.error_integral = self.backup_control(
                        input_state, step_size, self.error_integral
                    )
                    self.latched = True
                    self.intervening = True
                    out = ub
            out = control_desired
        else:
            ub, self.error_integral = self.backup_control(
                input_state, step_size, self.error_integral
            )
            out = ub
        return out

    def backup_control(self, state, step_size, error_integral):
        """LQT backup controller to eNMT"""
        error = np.array(
            [
                0,
                0,
                0,
                state[3] - self.n / 2 * state[1],
                state[4] + 2 * self.n * state[0],
                0,
            ]
        )
        backup_action = -self.K_1 @ error - self.K_2 @ error_integral
        error_integral = error_integral + error * step_size
        return np.clip(backup_action, -1, 1), error_integral


class InspectionCascadedRTA(CascadedRTA):
    """Combines ASIF Inspection RTA with switching-based delta v limit"""

    def _setup_rta_list(self):
        return [Inspection1v1RTA(), SwitchingDeltaVLimitRTA()]


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

    def __init__(
        self,
        collision_radius: float,
        a_max: float,
        alpha: ConstraintStrengthener = None,
        **kwargs,
    ):
        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.01, 0, 0.001])
        super().__init__(
            alpha=alpha,
            params={"collision_radius": collision_radius, "a_max": a_max},
            **kwargs,
        )

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        delta_p = state[0:3]
        mag_delta_p = jnp.linalg.norm(delta_p)
        h = lax.cond(
            mag_delta_p >= params["collision_radius"],
            self.positive_distance_constraint,
            self.negative_distance_constraint,
            state,
            params,
        )
        return h

    def _phi(self, state: jnp.ndarray, params: dict) -> float:
        delta_p = state[0:3]
        return jnp.linalg.norm(delta_p) - params["collision_radius"]

    def positive_distance_constraint(self, state, params):
        """Constraint value when sqrt component is real"""
        delta_p = state[0:3]
        delta_v = state[3:6]
        mag_delta_p = jnp.linalg.norm(delta_p)
        return (
            jnp.sqrt(2 * params["a_max"] * (mag_delta_p - params["collision_radius"]))
            + delta_p.T @ delta_v / mag_delta_p
        )

    def negative_distance_constraint(self, state, params):
        """Constraint value when sqrt component is imaginary"""
        delta_p = state[0:3]
        delta_v = state[3:6]
        mag_delta_p = jnp.linalg.norm(delta_p)
        return (
            -jnp.sqrt(2 * params["a_max"] * (-mag_delta_p + params["collision_radius"]))
            + delta_p.T @ delta_v / mag_delta_p
        )


class ConstraintCWHConicKeepOutZone(ConstraintModule):
    """CWH sun avoidance constraint
    Assumes deputy is always pointing at chief

    Parameters
    ----------
    a_max: float
        Maximum braking acceleration for each deputy spacecraft. m/s^2
    fov: float
        sensor field of view. radians
    sun_vel: float
        sun velocity vector
    get_sun_vec_fn : function
        function to get vector pointing from chief to sun
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.01, 0, 0.05])
    """

    def __init__(
        self,
        a_max: float,
        fov: float,
        get_pos,
        get_vel,
        get_cone_vec,
        cone_ang_vel: jnp.ndarray,
        alpha: ConstraintStrengthener = None,
        **kwargs,
    ):
        self.get_pos = get_pos
        self.get_vel = get_vel
        self.get_cone_vec = get_cone_vec
        self.cone_ang_vel = cone_ang_vel

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.001, 0, 0.0001])
        super().__init__(alpha=alpha, params={"fov": fov, "a_max": a_max}, **kwargs)

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        pos = self.get_pos(state)
        vel = self.get_vel(state)
        cone_vec = self.get_cone_vec(state)
        cone_unit_vec = cone_vec / jnp.linalg.norm(cone_vec)
        cone_ang_vel = self.cone_ang_vel
        theta = params["fov"] / 2

        pos_cone = pos - jnp.dot(pos, cone_unit_vec) * cone_unit_vec
        mult = jnp.cos(theta) * (
            jnp.linalg.norm(pos_cone) - jnp.tan(theta) * jnp.dot(pos, cone_unit_vec)
        )
        proj = (
            pos
            + mult * jnp.sin(theta) * cone_unit_vec
            + mult
            * jnp.cos(theta)
            * (jnp.dot(pos, cone_unit_vec) * cone_unit_vec - pos)
            / jnp.linalg.norm(pos_cone)
        )
        vel_proj = jnp.cross(cone_ang_vel, proj)

        delta_p = pos - proj
        delta_v = vel - vel_proj
        mag_delta_p = jnp.linalg.norm(pos) * jnp.sin(
            jnp.arccos(jnp.dot(cone_unit_vec, pos / jnp.linalg.norm(pos))) - theta
        )

        h = lax.cond(
            mag_delta_p >= 0,
            self.positive_distance_constraint,
            self.negative_distance_constraint,
            delta_p,
            delta_v,
            mag_delta_p,
            params["a_max"],
        )
        return h

    def _phi(self, state: jnp.ndarray, params: dict) -> float:
        pos = self.get_pos(state)
        cone_vec = self.get_cone_vec(state)
        p_hat = pos / jnp.linalg.norm(pos)
        c_hat = cone_vec / jnp.linalg.norm(cone_vec)
        h = jnp.arccos(jnp.dot(p_hat, c_hat)) - params["fov"] / 2
        return h

    def positive_distance_constraint(self, delta_p, delta_v, mag_delta_p, a_max):
        """Constraint value when sqrt component is real"""
        return jnp.sqrt(2 * a_max * mag_delta_p) + delta_p.T @ delta_v / mag_delta_p

    def negative_distance_constraint(self, delta_p, delta_v, mag_delta_p, a_max):
        """Constraint value when sqrt component is imaginary"""
        return -jnp.sqrt(-2 * a_max * mag_delta_p) + delta_p.T @ delta_v / mag_delta_p


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

    def __init__(
        self,
        collision_radius: float,
        m: float,
        n: float,
        dt: float,
        steps: int,
        alpha: ConstraintStrengthener = None,
        **kwargs,
    ):
        self.n = n
        self.steps = steps
        A, _ = generate_cwh_matrices(m, n, mode="3d")
        self.A = jnp.array(A)

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.01, 0, 0.001])
        super().__init__(
            alpha=alpha,
            params={"collision_radius": collision_radius, "dt": dt},
            **kwargs,
        )

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        vmapped_get_future_state = vmap(self.get_future_state, (None, 0, None), 0)
        phi_array = vmapped_get_future_state(
            state,
            jnp.linspace(params["dt"], self.steps * params["dt"], self.steps),
            params["collision_radius"],
        )
        return jnp.min(phi_array)

    def get_future_state(self, state, t, collision_radius):
        """Gets future state using closed form CWH dynamics (http://www.ae.utexas.edu/courses/ase366k/cw_equations.pdf)"""
        x = (
            (4 - 3 * jnp.cos(self.n * t)) * state[0]
            + jnp.sin(self.n * t) * state[3] / self.n
            + 2 / self.n * (1 - jnp.cos(self.n * t)) * state[4]
        )
        y = (
            6 * (jnp.sin(self.n * t) - self.n * t) * state[0]
            + state[1]
            - 2 / self.n * (1 - jnp.cos(self.n * t)) * state[3]
            + (4 * jnp.sin(self.n * t) - 3 * self.n * t) * state[4] / self.n
        )
        z = state[2] * jnp.cos(self.n * t) + state[5] / self.n * jnp.sin(self.n * t)
        return jnp.linalg.norm(jnp.array([x, y, z])) - collision_radius

    def get_array(self, state: jnp.ndarray, params: dict) -> float:
        """Gets entire trajectory array"""
        vmapped_get_future_state = vmap(self.get_future_state, (None, 0, None), 0)
        phi_array = vmapped_get_future_state(
            state,
            jnp.linspace(0, self.steps * params["dt"], self.steps + 1),
            params["collision_radius"],
        )
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

    def __init__(
        self, r_max: float, a_max: float, alpha: ConstraintStrengthener = None, **kwargs
    ):
        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.01, 0, 0.001])
        super().__init__(alpha=alpha, params={"r_max": r_max, "a_max": a_max}, **kwargs)

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        delta_p = state[0:3]
        mag_delta_p = jnp.linalg.norm(delta_p)
        h = lax.cond(
            params["r_max"] >= mag_delta_p,
            self.positive_distance_constraint,
            self.negative_distance_constraint,
            state,
            params,
        )
        return h

    def _phi(self, state: jnp.ndarray, params: dict) -> float:
        delta_p = state[0:3]
        return params["r_max"] - jnp.linalg.norm(delta_p)

    def positive_distance_constraint(self, state, params):
        """Constraint value when sqrt component is real"""
        delta_p = state[0:3]
        delta_v = state[3:6]
        mag_delta_p = jnp.linalg.norm(delta_p)
        return (
            jnp.sqrt(2 * params["a_max"] * (params["r_max"] - mag_delta_p))
            - delta_p.T @ delta_v / mag_delta_p
        )

    def negative_distance_constraint(self, state, params):
        """Constraint value when sqrt component is imaginary"""
        delta_p = state[0:3]
        delta_v = state[3:6]
        mag_delta_p = jnp.linalg.norm(delta_p)
        return (
            -jnp.sqrt(2 * params["a_max"] * (-params["r_max"] + mag_delta_p))
            - delta_p.T @ delta_v / mag_delta_p
        )


class HOCBFExampleChiefCollision(ConstraintModule):
    """Example HOCBF for CWH chief collision avoidance constraint

    Parameters
    ----------
    collision_radius: float
        radius of collision for chief spacecraft. m
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.01, 0, 0.01])
    """

    def __init__(
        self, collision_radius: float, alpha: ConstraintStrengthener = None, **kwargs
    ):
        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.01, 0, 0.01])
        super().__init__(
            alpha=alpha, params={"collision_radius": collision_radius}, **kwargs
        )

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        delta_p = state[0:3]
        return jnp.linalg.norm(delta_p) - params["collision_radius"]
