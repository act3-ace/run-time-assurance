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


# (1. Communication) Doesn't apply here, attitude requirement for pointing at earth
CHIEF_RADIUS_DEFAULT = 10 # chief radius of collision [m] (2. collision freedom)
DEPUTY_RADIUS_DEFAULT = 5 # deputy radius of collision [m] (2. collision freedom)
V0_DEFAULT = 0.2 # maximum docking speed [m/s] (3. dynamic velocity constraint)
V1_COEF_DEFAULT = 4 # velocity constraint slope [-] (3. dynamic velocity constraint)
# (4. Fuel limit) Doesn't apply here, consider using latched RTA to travel to NMT
R_MAX_DEFAULT = 1000 # max distance from chief [m] (5. translational keep out zone)
THETA_DEFAULT = np.pi/6 # sun avoidance angle [rad] (5. translational keep out zone)
# (6. Attitude keep out zone) Doesn't apply here, no attitude model
# (7. Limited duration attitudes) Doesn't apply here, no attitude model
# (8. Passively safe maneuvers) **??** Ensure if u=0 after safe action is taken, deputy would not colide with chief?
# (9. Maintain battery charge) Doesn't apply here, attitude requirement for pointing solar panels at sun
U_MAX_DEFAULT = 1 # Max thrust [N] (10. avoid actuation saturation)
X_VEL_LIMIT_DEFAULT = 10 # Maximum velocity limit [m/s] (11. Avoid aggressive maneuvering)
Y_VEL_LIMIT_DEFAULT = 10 # Maximum velocity limit [m/s] (11. Avoid aggressive maneuvering)
Z_VEL_LIMIT_DEFAULT = 10 # Maximum velocity limit [m/s] (11. Avoid aggressive maneuvering)


class Inspection3dState(RTAState):
    pass


class InspectionRTA(ExplicitASIFModule):
    def __init__(
        self,
        *args,
        m: float = M_DEFAULT,
        n: float = N_DEFAULT,
        v0: float = V0_DEFAULT,
        v1_coef: float = V1_COEF_DEFAULT,
        x_vel_limit: float = X_VEL_LIMIT_DEFAULT,
        y_vel_limit: float = Y_VEL_LIMIT_DEFAULT,
        z_vel_limit: float = Z_VEL_LIMIT_DEFAULT,
        chief_radius: float = CHIEF_RADIUS_DEFAULT,
        r_max: float = R_MAX_DEFAULT,
        u_max: float = U_MAX_DEFAULT,
        theta: float = THETA_DEFAULT,
        control_bounds_high: float = U_MAX_DEFAULT,
        control_bounds_low: float = -U_MAX_DEFAULT,
        **kwargs
    ):
        self.m = m
        self.n = n
        self.v0 = v0
        self.v1_coef = v1_coef
        self.v1 = self.v1_coef * self.n
        self.e_hat = np.array([1, 0, 0])

        self.A, self.B = generate_cwh_matrices(self.m, self.n, mode="3d")
        self.dynamics = BaseLinearODESolverDynamics(A=self.A, B=self.B, integration_method="RK45")

        self.chief_radius = chief_radius
        self.r_max = r_max
        self.u_max = u_max
        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit
        self.z_vel_limit = z_vel_limit
        self.theta = theta

        self.a_max = self.u_max / self.m - (3*self.n**2 + 2*self.n*self.v1) * self.r_max - 2*self.n*self.v0

        super().__init__(*args, control_bounds_high=control_bounds_high, control_bounds_low=control_bounds_low, **kwargs)

    def _setup_constraints(self) -> OrderedDict:
        return OrderedDict(
            [
                ('rel_vel', ConstraintCWHRelativeVelocity(v0=self.v0, v1=self.v1)),
                ('chief_collision', ConstraintCWHChiefCollision(collision_radius=self.chief_radius, a_max=self.a_max)),
                ('sun', ConstraintCWHSunAvoidance(a_max=self.a_max, theta=self.theta, e_hat=self.e_hat))
            ]
        )

    def gen_rta_state(self, vector: np.ndarray) -> Inspection3dState:
        return Inspection3dState(vector=vector)

    def _pred_state(self, state: RTAState, step_size: float, control: np.ndarray) -> Inspection3dState:
        next_state_vec, _ = self.dynamics.step(step_size, state.vector, control)
        return next_state_vec

    def state_transition_system(self, state: RTAState) -> np.ndarray:
        return self.A @ state.vector

    def state_transition_input(self, state: RTAState) -> np.ndarray:
        return np.copy(self.B)


class ConstraintCWHRelativeVelocity(ConstraintModule):
    def __init__(self, v0: float, v1: float, alpha: ConstraintStrengthener = None):
        self.v0 = v0
        self.v1 = v1

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.05, 0, 0.5])
        super().__init__(alpha=alpha)

    def _compute(self, state: RTAState) -> float:
        try:
            state_vec = state.vector
        except Exception:
            state_vec = state
        return float((self.v0 + self.v1 * np.linalg.norm(state_vec[0:3])) - np.linalg.norm(state_vec[3:6]))

    def grad(self, state: RTAState) -> np.ndarray:
        state_vec = state.vector
        a = self.v1/np.linalg.norm(state_vec[0:3])
        denom = np.linalg.norm(state_vec[3:6])
        if denom == 0:
            denom = 0.000001
        b = -1/denom
        g = np.array([a*state_vec[0], a*state_vec[1], a*state_vec[2], b*state_vec[3], b*state_vec[4], b*state_vec[5]])
        return g


class ConstraintCWHChiefCollision(ConstraintModule):
    def __init__(self, collision_radius: float, a_max: float, alpha: ConstraintStrengthener = None):
        self.collision_radius = collision_radius
        self.a_max = a_max

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.01, 0, 0.1])
        super().__init__(alpha=alpha)

    def _compute(self, state: RTAState) -> float:
        state_vec = state.vector
        delta_p = state_vec[0:3]
        delta_v = state_vec[3:6]
        mag_delta_p = np.linalg.norm(delta_p)
        h = np.sqrt(2*self.a_max*(mag_delta_p-self.collision_radius)) + delta_p.T @ delta_v / mag_delta_p
        return float(h)

    def grad(self, state: RTAState) -> np.ndarray:
        x = state.vector
        dp = x[0:3]
        dv = x[3:6]
        mdp = np.linalg.norm(x[0:3])
        mdv = dp.T @ dv
        a = self.a_max/(mdp*np.sqrt(2*self.a_max*(mdp-self.collision_radius))) - mdv/mdp**3
        return np.array([x[0]*a+x[3]/mdp, x[1]*a+x[4]/mdp, x[2]*a+x[5]/mdp, x[0]/mdp, x[1]/mdp, x[2]/mdp])


class ConstraintCWHSunAvoidance(ConstraintModule):
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
        except Exception:
            state_vec = state

        p = state_vec[0:3]
        p_es = p - np.dot(p, self.e_hat)*self.e_hat
        # p_pr = p + np.sin(self.theta)*np.cos(self.theta)*(np.linalg.norm(p_es)*self.e_hat + np.dot(p, self.e_hat)*p_es) - np.cos(self.theta)**2*p_es - np.sin(self.theta)**2*np.dot(p, self.e_hat)*self.e_hat
        a = np.cos(self.theta)*(np.linalg.norm(p_es)-np.tan(self.theta)*np.dot(p, self.e_hat))
        p_pr = p + a*np.sin(self.theta)*self.e_hat + a*np.cos(self.theta)*(np.dot(p, self.e_hat)*self.e_hat-p)/np.linalg.norm(p_es)
        
        h = np.sqrt(2*self.a_max*np.linalg.norm(p-p_pr)) + np.dot(p-p_pr, state_vec[3:6]-self.e_hat_vel) / np.linalg.norm(p-p_pr)
        return float(h)

    def grad(self, state: RTAState) -> np.ndarray:
        x = state.vector
        gh = []
        delta = 10e-4
        Delta = 0.5*delta*np.eye(len(x))
        for i in range(len(x)):
            gh.append((self._compute(x+Delta[i])-self._compute(x-Delta[i]))/delta)

        return np.array(gh)