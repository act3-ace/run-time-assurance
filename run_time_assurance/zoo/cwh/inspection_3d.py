"""This module implements RTA methods for the multiagent inspection problem with 3D CWH dynamics models
"""
from collections import OrderedDict

import numpy as np
from safe_autonomy_dynamics.base_models import BaseLinearODESolverDynamics
from safe_autonomy_dynamics.cwh import M_DEFAULT, N_DEFAULT, generate_cwh_matrices

from run_time_assurance.constraint import (
    ConstraintMagnitudeStateLimit,
    ConstraintModule,
    ConstraintStrengthener,
    PolynomialConstraintStrengthener,
)
from run_time_assurance.rta import ExplicitASIFModule
from run_time_assurance.state import RTAState

NUM_DEPUTIES_DEFAULT = 5  # Number of deputies for inspection problem
# (1. Communication) Doesn't apply here, attitude requirement for pointing at earth
CHIEF_RADIUS_DEFAULT = 10  # chief radius of collision [m] (2. collision freedom)
DEPUTY_RADIUS_DEFAULT = 5  # deputy radius of collision [m] (2. collision freedom)
V0_DEFAULT = 0.2  # maximum docking speed [m/s] (3. dynamic velocity constraint)
V1_COEF_DEFAULT = 4  # velocity constraint slope [-] (3. dynamic velocity constraint)
# (4. Fuel limit) Doesn't apply here, consider using latched RTA to travel to NMT
R_MAX_DEFAULT = 1000  # max distance from chief [m] (5. translational keep out zone)
THETA_DEFAULT = np.pi / 6  # sun avoidance angle [rad] (5. translational keep out zone)
# (6. Attitude keep out zone) Doesn't apply here, no attitude model
# (7. Limited duration attitudes) Doesn't apply here, no attitude model
# (8. Passively safe maneuvers) **??** Ensure if u=0 after safe action is taken, deputy would not colide with chief?
# (9. Maintain battery charge) Doesn't apply here, attitude requirement for pointing solar panels at sun
U_MAX_DEFAULT = 1  # Max thrust [N] (10. avoid actuation saturation)
X_VEL_LIMIT_DEFAULT = 2  # Maximum velocity limit [m/s] (11. Avoid aggressive maneuvering)
Y_VEL_LIMIT_DEFAULT = 2  # Maximum velocity limit [m/s] (11. Avoid aggressive maneuvering)
Z_VEL_LIMIT_DEFAULT = 2  # Maximum velocity limit [m/s] (11. Avoid aggressive maneuvering)


class Inspection3dState(RTAState):
    """RTA state for inspection 3d RTA (only current deputy's state)"""

    @property
    def x(self) -> float:
        """Getter for x position"""
        return self._vector[0]

    @x.setter
    def x(self, val: float):
        """Setter for x position"""
        self._vector[0] = val

    @property
    def y(self) -> float:
        """Getter for y position"""
        return self._vector[1]

    @y.setter
    def y(self, val: float):
        """Setter for y position"""
        self._vector[1] = val

    @property
    def z(self) -> float:
        """Getter for z position"""
        return self._vector[2]

    @z.setter
    def z(self, val: float):
        """Setter for z position"""
        self._vector[2] = val

    @property
    def x_dot(self) -> float:
        """Getter for x velocity component"""
        return self._vector[3]

    @x_dot.setter
    def x_dot(self, val: float):
        """Setter for x velocity component"""
        self._vector[3] = val

    @property
    def y_dot(self) -> float:
        """Getter for y velocity component"""
        return self._vector[4]

    @y_dot.setter
    def y_dot(self, val: float):
        """Setter for y velocity component"""
        self._vector[4] = val

    @property
    def z_dot(self) -> float:
        """Getter for z velocity component"""
        return self._vector[5]

    @z_dot.setter
    def z_dot(self, val: float):
        """Setter for z velocity component"""
        self._vector[5] = val


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
        control_bounds_high: float = U_MAX_DEFAULT,
        control_bounds_low: float = -U_MAX_DEFAULT,
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

        self.e_hat = np.array([1, 0, 0])
        self.u_max = control_bounds_high
        self.a_max = self.u_max / self.m - (3 * self.n**2 + 2 * self.n * self.v1) * self.r_max - 2 * self.n * self.v0

        self.A, self.B = generate_cwh_matrices(self.m, self.n, mode="3d")
        self.dynamics = BaseLinearODESolverDynamics(A=self.A, B=self.B, integration_method="RK45")

        super().__init__(*args, control_bounds_high=control_bounds_high, control_bounds_low=control_bounds_low, **kwargs)

    def _setup_constraints(self) -> OrderedDict:
        OD = OrderedDict(
            [
                ('rel_vel', ConstraintCWHRelativeVelocity(v0=self.v0, v1=self.v1)),
                ('chief_collision', ConstraintCWHChiefCollision(collision_radius=self.chief_radius, a_max=self.a_max)),
                ('sun', ConstraintCWHSunAvoidance(a_max=self.a_max, theta=self.theta, e_hat=self.e_hat)),
                (
                    'x_vel',
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.x_vel_limit, state_index=3, grad_len=6, alpha=PolynomialConstraintStrengthener([0, 0.05, 0, 0.1])
                    )
                ),
                (
                    'y_vel',
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.y_vel_limit, state_index=4, grad_len=6, alpha=PolynomialConstraintStrengthener([0, 0.05, 0, 0.1])
                    )
                ),
                (
                    'z_vel',
                    ConstraintMagnitudeStateLimit(
                        limit_val=self.z_vel_limit, state_index=5, grad_len=6, alpha=PolynomialConstraintStrengthener([0, 0.05, 0, 0.1])
                    )
                )
            ]
        )
        for i in range(self.num_deputies - 1):
            OD[f'deputy_collision_{i+1}'] = ConstraintCWHDeputyCollision(
                collision_radius=self.deputy_radius, a_max=self.a_max, deputy=i + 1
            )
        return OD

    def gen_rta_state(self, vector: np.ndarray) -> Inspection3dState:
        return Inspection3dState(vector=vector)

    def _pred_state(self, state: RTAState, step_size: float, control: np.ndarray) -> Inspection3dState:
        next_state_vec, _ = self.dynamics.step(step_size, state.vector, control)
        return next_state_vec

    def state_transition_system(self, state: RTAState) -> np.ndarray:
        return self.A @ state.vector[0:6]

    def state_transition_input(self, state: RTAState) -> np.ndarray:
        return np.copy(self.B)


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

    def __init__(self, v0: float, v1: float, alpha: ConstraintStrengthener = None):
        self.v0 = v0
        self.v1 = v1

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.05, 0, 0.5])
        super().__init__(alpha=alpha)

    def _compute(self, state: RTAState) -> float:
        state_vec = state.vector
        return float((self.v0 + self.v1 * np.linalg.norm(state_vec[0:3])) - np.linalg.norm(state_vec[3:6]))

    def grad(self, state: RTAState) -> np.ndarray:
        state_vec = state.vector
        a = self.v1 / np.linalg.norm(state_vec[0:3])
        denom = np.linalg.norm(state_vec[3:6])
        if denom == 0:
            denom = 0.000001
        b = -1 / denom
        g = np.array([a * state_vec[0], a * state_vec[1], a * state_vec[2], b * state_vec[3], b * state_vec[4], b * state_vec[5]])
        return g


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

    def __init__(self, collision_radius: float, a_max: float, alpha: ConstraintStrengthener = None):
        self.collision_radius = collision_radius
        self.a_max = a_max

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.005, 0, 0.05])
        super().__init__(alpha=alpha)

    def _compute(self, state: RTAState) -> float:
        state_vec = state.vector
        delta_p = state_vec[0:3]
        delta_v = state_vec[3:6]
        mag_delta_p = np.linalg.norm(delta_p)
        h = np.sqrt(2 * self.a_max * (mag_delta_p - self.collision_radius)) + delta_p.T @ delta_v / mag_delta_p
        return float(h)

    def grad(self, state: RTAState) -> np.ndarray:
        x = state.vector
        dp = x[0:3]
        dv = x[3:6]
        mdp = np.linalg.norm(x[0:3])
        mdv = dp.T @ dv
        a = self.a_max / (mdp * np.sqrt(2 * self.a_max * (mdp - self.collision_radius))) - mdv / mdp**3
        return np.array([x[0] * a + x[3] / mdp, x[1] * a + x[4] / mdp, x[2] * a + x[5] / mdp, x[0] / mdp, x[1] / mdp, x[2] / mdp])


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

    def __init__(self, collision_radius: float, a_max: float, deputy: float, alpha: ConstraintStrengthener = None):
        self.collision_radius = collision_radius
        self.a_max = a_max
        self.deputy = deputy

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.005, 0, 0.05])
        super().__init__(alpha=alpha)

    def _compute(self, state: RTAState) -> float:
        state_vec = state.vector

        delta_p = state_vec[0:3] - state_vec[int(self.deputy * 6):int(self.deputy * 6 + 3)]
        delta_v = state_vec[3:6] - state_vec[int(self.deputy * 6 + 3):int(self.deputy * 6 + 6)]
        mag_delta_p = np.linalg.norm(delta_p)
        h = np.sqrt(4 * self.a_max * (mag_delta_p - self.collision_radius)) + delta_p.T @ delta_v / mag_delta_p
        return float(h)

    def grad(self, state: RTAState) -> np.ndarray:
        i = self.deputy
        x = state.vector
        dp = x[0:3] - x[int(i * 6):int(i * 6 + 3)]
        dv = x[3:6] - x[int(i * 6 + 3):int(i * 6 + 6)]
        mdp = np.linalg.norm(dp)
        mdv = dp.T @ dv
        a = self.a_max / (mdp * np.sqrt(self.a_max * (mdp - self.collision_radius))) - mdv / mdp**3
        g = np.array(
            [
                (x[0] - x[6 * i]) * a + (x[3] - x[6 * i + 3]) / mdp, (x[1] - x[6 * i + 1]) * a + (x[4] - x[6 * i + 4]) / mdp,
                (x[2] - x[6 * i + 2]) * a + (x[5] - x[6 * i + 5]) / mdp, (x[0] - x[6 * i]) / mdp, (x[1] - x[6 * i + 1]) / mdp,
                (x[2] - x[6 * i + 2]) / mdp
            ]
        )
        return g


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

    def __init__(self, a_max: float, theta: float, e_hat: np.ndarray, alpha: ConstraintStrengthener = None):
        self.a_max = a_max
        self.theta = theta
        self.e_hat = e_hat
        self.e_hat_vel = np.array([0, 0, 0])

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.01, 0, 0.05])
        super().__init__(alpha=alpha)

    def _compute(self, state: RTAState) -> float:
        try:
            state_vec = state.vector
        except AttributeError:
            state_vec = state

        p = state_vec[0:3]
        p_es = p - np.dot(p, self.e_hat) * self.e_hat
        a = np.cos(self.theta) * (np.linalg.norm(p_es) - np.tan(self.theta) * np.dot(p, self.e_hat))
        p_pr = p + a * np.sin(self.theta) * self.e_hat + a * np.cos(self.theta
                                                                    ) * (np.dot(p, self.e_hat) * self.e_hat - p) / np.linalg.norm(p_es)

        h = np.sqrt(2 * self.a_max * np.linalg.norm(p - p_pr)
                    ) + np.dot(p - p_pr, state_vec[3:6] - self.e_hat_vel) / np.linalg.norm(p - p_pr)
        return float(h)

    def grad(self, state: RTAState) -> np.ndarray:
        # Numerical approximation
        x = state.vector[0:6]
        gh = []
        delta = 10e-4
        Delta = 0.5 * delta * np.eye(len(x))
        for i in range(len(x)):
            gh.append((self._compute(x + Delta[i]) - self._compute(x - Delta[i])) / delta)

        return np.array(gh)
