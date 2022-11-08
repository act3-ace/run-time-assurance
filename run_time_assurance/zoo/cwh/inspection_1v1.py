"""This module implements RTA methods for the single-agent inspection problem with 3D CWH dynamics models
"""
from collections import OrderedDict
from typing import Dict, Tuple, Union

import jax.numpy as jnp
import numpy as np
import scipy
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
from safe_autonomy_dynamics.cwh.point_model import M_DEFAULT, N_DEFAULT, generate_cwh_matrices

from run_time_assurance.constraint import (
    ConstraintMagnitudeStateLimit,
    ConstraintMaxStateLimit,
    ConstraintModule,
    ConstraintStrengthener,
    PolynomialConstraintStrengthener,
)
from run_time_assurance.controller import RTABackupController
from run_time_assurance.rta import CascadedRTA, ExplicitASIFModule, ExplicitSimplexModule
from run_time_assurance.state import RTAStateWrapper
from run_time_assurance.utils import to_jnp_array_jit
from run_time_assurance.zoo.cwh.docking_3d import ConstraintCWH3dRelativeVelocity

CHIEF_RADIUS_DEFAULT = 5  # chief radius of collision [m] (collision freedom)
DEPUTY_RADIUS_DEFAULT = 5  # deputy radius of collision [m] (collision freedom)
V0_DEFAULT = 0.2  # maximum docking speed [m/s] (dynamic velocity constraint)
V1_COEF_DEFAULT = 2  # velocity constraint slope [-] (dynamic velocity constraint)
V0_DISTANCE_DEFAULT = 0  # distance where v0 is applied
R_MAX_DEFAULT = 1000  # max distance from chief [m] (translational keep out zone)
FOV_DEFAULT = 60 * jnp.pi / 180  # sun avoidance angle [rad] (translational keep out zone)
U_MAX_DEFAULT = 1  # Max thrust [N] (avoid actuation saturation)
VEL_LIMIT_DEFAULT = 1  # Maximum velocity limit [m/s] (Avoid aggressive maneuvering)
FUEL_LIMIT_DEFAULT = 0.25  # Maximum fuel use limit [kg]
FUEL_THRESHOLD_DEFAULT = 0.1  # Fuel use threshold, to switch to backup controller [kg]
GRAVITY = 9.81  # Gravity at Earth's surface [m/s^2]
SPECIFIC_IMPULSE_DEFAULT = 220  # Specific Impulse of thrusters [s]
SUN_VEL_DEFAULT = -N_DEFAULT  # Speed of sun rotation in x-y plane


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
    fuel_limit : float, optional
        maximum fuel used limit, by default FUEL_LIMIT_DEFAULT
    fuel_switching_threshold : float, optional
        fuel threshold at which to switch to backup controller
    gravity : float, optional
        gravity at Earth's surface, by default GRAVITY
    isp : float, optional
        Specific impulse of thrusters in seconds, by default SPECIFIC_IMPULSE_DEFAULT
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
        m: float = M_DEFAULT,
        n: float = N_DEFAULT,
        chief_radius: float = CHIEF_RADIUS_DEFAULT,
        deputy_radius: float = DEPUTY_RADIUS_DEFAULT,
        v0: float = V0_DEFAULT,
        v1_coef: float = V1_COEF_DEFAULT,
        v0_distance: float = V0_DISTANCE_DEFAULT,
        r_max: float = R_MAX_DEFAULT,
        fov: float = FOV_DEFAULT,
        vel_limit: float = VEL_LIMIT_DEFAULT,
        fuel_limit: float = FUEL_LIMIT_DEFAULT,
        fuel_switching_threshold: float = FUEL_THRESHOLD_DEFAULT,
        gravity: float = GRAVITY,
        isp: float = SPECIFIC_IMPULSE_DEFAULT,
        sun_vel: float = SUN_VEL_DEFAULT,
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
        self.v0_distance = v0_distance
        self.r_max = r_max
        self.fov = fov
        self.vel_limit = vel_limit
        self.fuel_limit = fuel_limit
        self.fuel_switching_threshold = fuel_switching_threshold
        self.gravity = gravity
        self.isp = isp
        self.sun_vel = sun_vel

        self.u_max = U_MAX_DEFAULT
        self.a_max = self.u_max / self.m - (3 * self.n**2 + 2 * self.n * self.v1) * self.r_max - 2 * self.n * self.v0
        A, B = generate_cwh_matrices(self.m, self.n, mode="3d")
        self.A = jnp.array(A)
        self.B = jnp.array(B)

        self.control_dim = self.B.shape[1]

        self._pred_state_fn = jit(self._pred_state, static_argnames=['step_size'])
        self._pred_state_cwh_fn = jit(self._pred_state_cwh, static_argnames=['step_size'])

        super().__init__(
            *args, control_dim=self.control_dim, control_bounds_high=control_bounds_high, control_bounds_low=control_bounds_low, **kwargs
        )

    def _setup_constraints(self) -> OrderedDict:
        constraint_dict = OrderedDict(
            [
                ('chief_collision', ConstraintCWHChiefCollision(collision_radius=self.chief_radius + self.deputy_radius, a_max=self.a_max)),
                ('rel_vel', ConstraintCWH3dRelativeVelocity(v0=self.v0, v1=self.v1, v0_distance=self.v0_distance, bias=-1e-3)),
                (
                    'sun',
                    ConstraintCWHConicKeepOutZone(
                        a_max=self.a_max,
                        fov=self.fov,
                        get_pos=self.get_pos_vector,
                        get_vel=self.get_vel_vector,
                        get_cone_vec=self.get_sun_vector,
                        cone_ang_vel=jnp.array([0, 0, self.sun_vel]),
                        bias=-1e-3
                    )
                ),
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
                        limit_val=self.vel_limit, state_index=3, alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]), bias=-0.001
                    )
                ),
                (
                    'y_vel',
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.vel_limit, state_index=4, alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]), bias=-0.001
                    )
                ),
                (
                    'z_vel',
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.vel_limit, state_index=5, alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]), bias=-0.001
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
        return jnp.concatenate((xd, jnp.array([self.sun_vel, fuel])))

    def _pred_state_cwh(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        """Predicted state for only CWH equations
        """
        sol = odeint(self.compute_state_dot_cwh, state, jnp.linspace(0., step_size, 11), control)
        return sol[-1, :]

    def compute_state_dot_cwh(self, x, t, u):
        """Computes state dot for ODE integration (only CWH equations)
        """
        xd = self.A @ x[0:6] + self.B @ u + 0 * t
        return xd

    def state_transition_system(self, state: jnp.ndarray) -> jnp.ndarray:
        xd = self.A @ state[0:6]
        return jnp.concatenate((xd, jnp.array([self.sun_vel, 0])))

    def state_transition_input(self, state: jnp.ndarray) -> jnp.ndarray:
        return jnp.vstack((self.B, jnp.zeros((2, 3))))

    def _get_state(self, input_state) -> jnp.ndarray:
        assert isinstance(input_state, (np.ndarray, jnp.ndarray)), ("input_state must be an RTAState or numpy array.")
        input_state = np.array(input_state)

        if len(input_state) < 8:
            input_state = np.concatenate((input_state, np.array([0., 0.])))
            self.sun_vel = 0

        return to_jnp_array_jit(input_state)

    def get_sun_vector(self, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get vector pointing from sun to chief"""
        return -jnp.array([jnp.cos(state[6]), jnp.sin(state[6]), 0.])

    def get_pos_vector(self, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get position vector"""
        return state[0:3]

    def get_vel_vector(self, state: jnp.ndarray) -> jnp.ndarray:
        """Function to get velocity vector"""
        return state[3:6]


class SwitchingFuelLimitRTA(ExplicitSimplexModule):
    """Explicit Simplex RTA Filter for fuel use
    Switches to NMT tracking backup controller

    Parameters
    ----------
    m : float, optional
        mass in kg of spacecraft, by default M_DEFAULT
    n : float, optional
        orbital mean motion in rad/s of current Hill's reference frame, by default N_DEFAULT
    fuel_limit : float, optional
        maximum fuel used limit, by default FUEL_LIMIT_DEFAULT
    fuel_switching_threshold : float, optional
        fuel threshold at which to switch to backup controller
    gravity : float, optional
        gravity at Earth's surface, by default GRAVITY
    isp : float, optional
        Specific impulse of thrusters in seconds, by default SPECIFIC_IMPULSE_DEFAULT
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
        fuel_limit: float = FUEL_LIMIT_DEFAULT,
        fuel_switching_threshold: float = FUEL_THRESHOLD_DEFAULT,
        gravity: float = GRAVITY,
        isp: float = SPECIFIC_IMPULSE_DEFAULT,
        control_bounds_high: Union[float, np.ndarray] = U_MAX_DEFAULT,
        control_bounds_low: Union[float, np.ndarray] = -U_MAX_DEFAULT,
        backup_controller: RTABackupController = None,
        jit_compile_dict: Dict[str, bool] = None,
        **kwargs
    ):
        self.m = m
        self.n = n
        self.fuel_limit = fuel_limit
        self.fuel_switching_threshold = fuel_switching_threshold
        self.gravity = gravity
        self.isp = isp

        if backup_controller is None:
            backup_controller = NMTBackupController(m=self.m, n=self.n, fuel_limit=self.fuel_limit)

        if jit_compile_dict is None:
            jit_compile_dict = {'pred_state': True}

        super().__init__(
            *args,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            backup_controller=backup_controller,
            jit_compile_dict=jit_compile_dict,
            **kwargs
        )

    def _setup_constraints(self) -> OrderedDict:
        return OrderedDict([('fuel', ConstraintMaxStateLimit(limit_val=self.fuel_switching_threshold, state_index=7))])

    def _pred_state(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        m_f = state[7]
        m_dot_f = jnp.sum(jnp.abs(control)) / (self.gravity * self.isp)
        return jnp.array([0, 0, 0, 0, 0, 0, 0, m_f + m_dot_f * step_size])


class InspectionCascadedRTA(CascadedRTA):
    """Combines ASIF Inspection RTA with switching-based fuel limit
    """

    def _setup_rta_list(self):
        return [Inspection1v1RTA(), SwitchingFuelLimitRTA()]


class NMTBackupController(RTABackupController):
    """Simple LQR controller to guide the deputy to the closest elliptical Natural Motion Trajectory (eNMT)

    Parameters
    ----------
    m : float, optional
        mass in kg of spacecraft, by default M_DEFAULT
    n : float, optional
        orbital mean motion in rad/s of current Hill's reference frame, by default N_DEFAULT
    fuel_limit : float, optional
        maximum fuel used limit, by default FUEL_LIMIT_DEFAULT
    """

    def __init__(self, m: float = M_DEFAULT, n: float = N_DEFAULT, fuel_limit: float = FUEL_LIMIT_DEFAULT):
        self.n = n
        self.fuel_limit = fuel_limit

        # LQR Gain Matrices
        Q = np.eye(6) * 0.05
        R = np.eye(3) * 1000
        A, B = generate_cwh_matrices(m, n, mode="3d")
        # Solve the Algebraic Ricatti equation for the given system
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        # Construct the constain gain matrix, K
        K = np.linalg.inv(R) @ (np.transpose(B) @ P)
        self.K = to_jnp_array_jit(-K)

        super().__init__()

    def _generate_control(
        self,
        state: jnp.ndarray,
        step_size: float,
        controller_state: Union[jnp.ndarray, Dict[str, jnp.ndarray], None] = None
    ) -> Tuple[jnp.ndarray, None]:

        backup_action = lax.cond(state[7] >= self.fuel_limit, self.zero_u, self.lqr, state[0:6])

        return backup_action, None

    def zero_u(self, state):
        """Zero control if fuel used is above limit
        """
        return jnp.array([0., 0., 0.]) + 0 * jnp.sum(state)

    def lqr(self, state):
        """LQR control to nearest eNMT.
        Used when fuel use is above threshold but below limit
        """
        state_des = jnp.array([0, 0, 0, state[3] - self.n / 2 * state[1], state[4] + 2 * self.n * state[0], state[5]])
        return self.K @ state_des


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
        **kwargs
    ):
        self.a_max = a_max
        self.fov = fov
        self.get_pos = get_pos
        self.get_vel = get_vel
        self.get_cone_vec = get_cone_vec
        self.cone_ang_vel = cone_ang_vel

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.001, 0, 0.0001])
        super().__init__(alpha=alpha, **kwargs)

    def _compute(self, state: jnp.ndarray) -> float:
        pos = self.get_pos(state)
        vel = self.get_vel(state)
        cone_vec = self.get_cone_vec(state)
        cone_unit_vec = cone_vec / jnp.linalg.norm(cone_vec)
        cone_ang_vel = self.cone_ang_vel
        theta = self.fov / 2

        pos_cone = pos - jnp.dot(pos, cone_unit_vec) * cone_unit_vec
        mult = jnp.cos(theta) * (jnp.linalg.norm(pos_cone) - jnp.tan(theta) * jnp.dot(pos, cone_unit_vec))
        proj = pos + mult * jnp.sin(theta) * cone_unit_vec + mult * jnp.cos(theta) * (jnp.dot(pos, cone_unit_vec) * cone_unit_vec -
                                                                                      pos) / jnp.linalg.norm(pos_cone)
        vel_proj = jnp.cross(cone_ang_vel, proj)

        delta_p = pos - proj
        delta_v = vel - vel_proj
        mag_delta_p = jnp.linalg.norm(pos) * jnp.sin(jnp.arccos(jnp.dot(cone_unit_vec, pos / jnp.linalg.norm(pos))) - theta)

        h = jnp.sqrt(2 * self.a_max * mag_delta_p) + delta_p.T @ delta_v / mag_delta_p
        return h

    def _phi(self, state: jnp.ndarray) -> float:
        pos = self.get_pos(state)
        cone_vec = self.get_cone_vec(state)
        p_hat = pos / jnp.linalg.norm(pos)
        c_hat = cone_vec / jnp.linalg.norm(cone_vec)
        h = jnp.arccos(jnp.dot(p_hat, c_hat)) - self.fov / 2
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
            alpha = PolynomialConstraintStrengthener([0, 0.01, 0, 0.001])
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
