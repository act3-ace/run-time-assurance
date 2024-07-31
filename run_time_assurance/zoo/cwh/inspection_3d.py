"""This module implements RTA methods for the multiagent inspection problem with 3D CWH dynamics models"""

from collections import OrderedDict
from functools import partial
from typing import Any, Union

import jax.numpy as jnp
import numpy as np
import scipy
from jax import lax, vmap
import safe_autonomy_simulation.sims.spacecraft.defaults as defaults

from run_time_assurance.constraint import (
    ConstraintMagnitudeStateLimit,
    ConstraintModule,
    ConstraintStrengthener,
    PolynomialConstraintStrengthener,
)
from run_time_assurance.rta import ExplicitASIFModule, RTAModule
from run_time_assurance.utils import to_jnp_array_jit
from run_time_assurance.zoo.cwh.docking_3d import ConstraintCWH3dRelativeVelocity
from run_time_assurance.zoo.cwh.inspection_1v1 import (
    CHIEF_RADIUS_DEFAULT,
    DEPUTY_RADIUS_DEFAULT,
    FOV_DEFAULT,
    R_MAX_DEFAULT,
    SUN_VEL_DEFAULT,
    U_MAX_DEFAULT,
    V0_DEFAULT,
    V0_DISTANCE_DEFAULT,
    V1_COEF_DEFAULT,
    VEL_LIMIT_DEFAULT,
    ConstraintCWHChiefCollision,
    ConstraintCWHConicKeepOutZone,
    ConstraintCWHMaxDistance,
    ConstraintPassivelySafeManeuver,
)
from run_time_assurance.zoo.cwh.utils import generate_cwh_matrices

NUM_DEPUTIES_DEFAULT = 5  # Number of deputies for inspection problem


class InspectionRTA(ExplicitASIFModule):
    """
    Implements Explicit Optimization RTA for the 3d Inspection problem

    Utilizes Explicit Active Set Invariance Filter algorithm

    Parameters
    ----------
    num_deputies : int, optional
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
        num_deputies: int = NUM_DEPUTIES_DEFAULT,
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
        sun_vel: float = SUN_VEL_DEFAULT,
        control_bounds_high: Union[
            float, list, np.ndarray, jnp.ndarray
        ] = U_MAX_DEFAULT,
        control_bounds_low: Union[
            float, list, np.ndarray, jnp.ndarray
        ] = -U_MAX_DEFAULT,
        **kwargs,
    ):
        self.num_deputies = num_deputies
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
        self.sun_vel = sun_vel

        self.u_max = U_MAX_DEFAULT
        vmax = min(self.vel_limit, self.v0 + self.v1 * self.r_max)
        self.a_max = (
            self.u_max / self.m - 3 * self.n**2 * self.r_max - 2 * self.n * vmax
        )
        A, B = generate_cwh_matrices(self.m, self.n, mode="3d")
        self.A = jnp.array(A)
        self.B = jnp.array(B)

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
                        bias=-2e-3,
                        alpha=PolynomialConstraintStrengthener([0, 0, 0, 0.1]),
                    ),
                ),
                (
                    "chief_sun",
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
                ("r_max", ConstraintCWHMaxDistance(r_max=self.r_max, a_max=self.a_max)),
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
        for i in range(1, self.num_deputies):
            constraint_dict[f"deputy_collision_{i}"] = ConstraintCWHDeputyCollision(
                collision_radius=self.deputy_radius * 2, a_max=self.a_max, deputy=i
            )
            constraint_dict[f"deputy_sun_{i}"] = ConstraintCWHConicKeepOutZone(
                a_max=self.a_max,
                fov=self.fov,
                get_pos=partial(self.get_p1_p2, i),
                get_vel=partial(self.get_v1_v2, i),
                get_cone_vec=self.get_sun_vector,
                cone_ang_vel=jnp.array([0, 0, self.sun_vel]),
                bias=-1e-3,
                alpha=PolynomialConstraintStrengthener([0, 0, 0, 0.1]),
            )
            constraint_dict[f"deputy_PSM_{i}"] = ConstraintPSMDeputy(
                collision_radius=self.chief_radius + self.deputy_radius,
                m=self.m,
                n=self.n,
                dt=1,
                steps=100,
                deputy=i,
            )
        return constraint_dict

    def state_transition_system(self, state: jnp.ndarray) -> jnp.ndarray:
        cwh = self.A_n @ state[0 : self.num_deputies * 6]
        return jnp.concatenate((cwh, jnp.array([self.sun_vel])))

    def state_transition_input(self, state: jnp.ndarray) -> jnp.ndarray:
        cwh = self.B_n
        return jnp.vstack((cwh, jnp.zeros((1, 3))))

    def _get_state(self, input_state) -> jnp.ndarray:
        assert isinstance(
            input_state, (np.ndarray, jnp.ndarray)
        ), "input_state must be an RTAState or numpy array."
        input_state = np.array(input_state)

        if len(input_state) < 6 * self.num_deputies + 1:
            input_state = np.concatenate((input_state, np.array([0.0])))
            self.sun_vel = 0

        return to_jnp_array_jit(input_state)

    def get_sun_vector(self, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get vector pointing from chief to sun"""
        return -jnp.array([jnp.cos(state[-1]), jnp.sin(state[-1]), 0.0])

    def get_pos_vector(self, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get position vector"""
        return state[0:3]

    def get_vel_vector(self, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get velocity vector"""
        return state[3:6]

    def get_p1_p2(self, i: int, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get p1-p2 vector"""
        pos = state[0:3] - state[6 * i : 6 * i + 3]
        sign = self.get_sign(i, state)
        return sign * pos

    def get_v1_v2(self, i: int, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get v1-v2 vector"""
        vel = state[3:6] - state[6 * i + 3 : 6 * i + 6]
        sign = self.get_sign(i, state)
        return sign * vel

    def get_sign(self, i, state):
        """Gets sign for pos/vel"""
        pos = state[0:3] - state[6 * i : 6 * i + 3]
        cone_vec = self.get_sun_vector(state)
        p_hat = pos / jnp.linalg.norm(pos)
        c_hat = cone_vec / jnp.linalg.norm(cone_vec)
        return lax.cond(
            jnp.arccos(jnp.dot(p_hat, c_hat)) > jnp.pi / 2, lambda x: -1, lambda x: 1, 0
        )


class ConstraintCWHDeputyCollision(ConstraintModule):
    """CWH deputy collision avoidance constraint

    Parameters
    ----------
    collision_radius: float
        radius of collision for deputy spacecraft. m
    a_max: float
        Maximum braking acceleration for each deputy spacecraft. m/s^2
    deputy: int
        number of the deputy to avoid
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.005, 0, 0.05])
    """

    def __init__(
        self,
        collision_radius: float,
        a_max: float,
        deputy: int,
        alpha: ConstraintStrengthener = None,
        **kwargs,
    ):
        self.deputy = deputy

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.01, 0, 0.001])
        super().__init__(
            alpha=alpha,
            params={"collision_radius": collision_radius, "a_max": a_max},
            **kwargs,
        )

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        delta_p = state[0:3] - state[int(self.deputy * 6) : int(self.deputy * 6 + 3)]
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
        delta_p = state[0:3] - state[int(self.deputy * 6) : int(self.deputy * 6 + 3)]
        return jnp.linalg.norm(delta_p) - params["collision_radius"]

    def positive_distance_constraint(self, state, params):
        """Constraint value when sqrt component is real"""
        delta_p = state[0:3] - state[int(self.deputy * 6) : int(self.deputy * 6 + 3)]
        delta_v = (
            state[3:6] - state[int(self.deputy * 6 + 3) : int(self.deputy * 6 + 6)]
        )
        mag_delta_p = jnp.linalg.norm(delta_p)
        return (
            jnp.sqrt(4 * params["a_max"] * (mag_delta_p - params["collision_radius"]))
            + delta_p.T @ delta_v / mag_delta_p
        )

    def negative_distance_constraint(self, state, params):
        """Constraint value when sqrt component is imaginary"""
        delta_p = state[0:3] - state[int(self.deputy * 6) : int(self.deputy * 6 + 3)]
        delta_v = (
            state[3:6] - state[int(self.deputy * 6 + 3) : int(self.deputy * 6 + 6)]
        )
        mag_delta_p = jnp.linalg.norm(delta_p)
        return (
            -jnp.sqrt(4 * params["a_max"] * (-mag_delta_p + params["collision_radius"]))
            + delta_p.T @ delta_v / mag_delta_p
        )


class ConstraintPSMDeputy(ConstraintModule):
    """Passively Safe Maneuver Constraint
    Assures that deputy will never collide with another deputy if there is a fault and u=0 for both

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
    deputy: int
        number of the deputy to avoid
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
        deputy: int,
        alpha: ConstraintStrengthener = None,
        **kwargs,
    ):
        self.n = n
        self.steps = steps
        self.deputy = deputy
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

        xd = (
            (4 - 3 * jnp.cos(self.n * t)) * state[int(self.deputy * 6)]
            + jnp.sin(self.n * t) * state[int(self.deputy * 6) + 3] / self.n
            + 2 / self.n * (1 - jnp.cos(self.n * t)) * state[int(self.deputy * 6) + 4]
        )
        yd = (
            6 * (jnp.sin(self.n * t) - self.n * t) * state[int(self.deputy * 6) + 0]
            + state[int(self.deputy * 6) + 1]
            - 2 / self.n * (1 - jnp.cos(self.n * t)) * state[int(self.deputy * 6) + 3]
            + (4 * jnp.sin(self.n * t) - 3 * self.n * t)
            * state[int(self.deputy * 6) + 4]
            / self.n
        )
        zd = state[int(self.deputy * 6) + 2] * jnp.cos(self.n * t) + state[
            int(self.deputy * 6) + 5
        ] / self.n * jnp.sin(self.n * t)

        return jnp.linalg.norm(jnp.array([x - xd, y - yd, z - zd])) - collision_radius

    def get_array(self, state: jnp.ndarray, params: dict) -> float:
        """Gets entire trajectory array"""
        vmapped_get_future_state = vmap(self.get_future_state, (None, 0), 0)
        phi_array = vmapped_get_future_state(
            state,
            jnp.linspace(0, self.steps * params["dt"], self.steps + 1),
            params["collision_radius"],
        )
        return phi_array


class CombinedInspectionRTA(RTAModule):
    """
    Combines each individual deputy RTA into one module

    Parameters
    ----------
    num_deputies : int, optional
        number of deputies in simulation, by default NUM_DEPUTIES_DEFAULT
    deputy_rta: RTAModule, optional
        RTA used by each individual deputy
    control_bounds_high : float, optional
        upper bound of allowable control. Pass a list for element specific limit. By default U_MAX_DEFAULT
    control_bounds_low : float, optional
        lower bound of allowable control. Pass a list for element specific limit. By default -U_MAX_DEFAULT
    """

    def __init__(
        self,
        *args: Any,
        num_deputies: int = NUM_DEPUTIES_DEFAULT,
        deputy_rta: RTAModule = InspectionRTA(),
        control_bounds_high: Union[float, int, list, np.ndarray] = U_MAX_DEFAULT,
        control_bounds_low: Union[float, int, list, np.ndarray] = -U_MAX_DEFAULT,
        **kwargs: Any,
    ):
        self.num_deputies = num_deputies
        self.deputy_rta = deputy_rta
        super().__init__(
            *args,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            **kwargs,
        )

    def compute_filtered_control(
        self, input_state: Any, step_size: float, control_desired: np.ndarray
    ) -> np.ndarray:
        u_act = np.zeros(self.num_deputies * 3)
        for i in range(self.num_deputies):
            u_des = control_desired[3 * i : 3 * i + 3]
            x_new = self.get_agent_state(input_state, i)
            u_act[3 * i : 3 * i + 3] = self.deputy_rta.filter_control(
                x_new, step_size, u_des
            )
        return u_act

    def get_agent_state(self, state, i):
        """Places current agent state at beginning of array"""
        x_i = state[6 * i : 6 * i + 6]
        x_old = np.delete(state, np.s_[6 * i : 6 * i + 6], 0)
        x = np.concatenate((x_i, x_old))
        return x
