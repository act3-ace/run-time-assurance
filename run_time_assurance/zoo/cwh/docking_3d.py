"""This module implements RTA methods for the docking problem with 3D CWH dynamics models
"""
from collections import OrderedDict

import constraint
import numpy as np
import scipy
from safe_autonomy_dynamics.base_models import BaseLinearODESolverDynamics
from safe_autonomy_dynamics.cwh import M_DEFAULT, N_DEFAULT, generate_cwh_matrices

from run_time_assurance.constraint import (
    ConstraintMagnitudeStateLimit,
    ConstraintModule,
    ConstraintStrengthener,
    PolynomialConstraintStrengthener,
)
from run_time_assurance.rta import ExplicitASIFModule, ExplicitSimplexModule, ImplicitASIFModule, ImplicitSimplexModule, RTABackupController
from run_time_assurance.state import RTAState
from run_time_assurance.zoo.cwh.docking_2d import V0_DEFAULT, X_VEL_LIMIT_DEFAULT, Y_VEL_LIMIT_DEFAULT

Z_VEL_LIMIT_DEFAULT = 10
V1_COEF_DEFAULT = 4
V1_DEFAULT = V1_COEF_DEFAULT * N_DEFAULT


class Docking3dState(RTAState):
    """RTA state for cwh docking 3d RTA"""

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


class Docking3dRTAMixin:
    """Mixin class provides 3D docking RTA util functions
    Must call mixin methods using the RTA interface methods
    """

    def _setup_docking_properties(self, m: float, n: float, v1_coef: float):
        """Initializes docking specific properties from other class members"""
        self.v1 = v1_coef * n
        self.A, self.B = generate_cwh_matrices(m, n, mode="3d")
        self.dynamics = BaseLinearODESolverDynamics(A=self.A, B=self.B, integration_method="RK45")

    def _setup_docking_constraints(self, v0: float, v1: float, x_vel_limit: float, y_vel_limit: float, z_vel_limit: float) -> OrderedDict:
        """generates constraints used in the docking problem"""
        return OrderedDict(
            [
                ('rel_vel', ConstraintDocking3dRelativeVelocity(v0=v0, v1=v1)),
                ('x_vel', ConstraintMagnitudeStateLimit(limit_val=x_vel_limit, state_index=3)),
                ('y_vel', ConstraintMagnitudeStateLimit(limit_val=y_vel_limit, state_index=4)),
                ('z_vel', ConstraintMagnitudeStateLimit(limit_val=z_vel_limit, state_index=5)),
            ]
        )

    def _docking_pred_state(self, state: RTAState, step_size: float, control: np.ndarray) -> np.ndarray:
        """Predicts the next state given the current state and control action"""
        next_state_vec, _ = self.dynamics.step(step_size, state.vector, control)
        return next_state_vec

    def _docking_f_x(self, state: RTAState) -> np.ndarray:
        """Computes the system contribution to the state transition: f(x) of dx/dt = f(x) + g(x)u"""
        return self.A @ state.vector

    def _docking_g_x(self, _: RTAState) -> np.ndarray:
        """Computes the control input contribution to the state transition: g(x) of dx/dt = f(x) + g(x)u"""
        return np.copy(self.B)


class Docking3dExplicitSwitchingRTA(ExplicitSimplexModule, Docking3dRTAMixin):
    """Implements Explicit Switching RTA for the 3d Docking problem

    Parameters
    ----------
    m : float, optional
        mass in kg of spacecraft, by default M_DEFAULT
    n : float, optional
        orbital mean motion in rad/s of current Hill's reference frame, by default N_DEFAULT
    v0 : float, optional
        Maximum safe docking velocity in m/s, by default V0_DEFAULT
        v0 of v_limit = v0 + v1*n*||r||
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r||
    x_vel_limit : float, optional
        max velocity magnitude in the x direction, by default X_VEL_LIMIT_DEFAULT
    y_vel_limit : float, optional
        max velocity magnitude in the y direction, by default Y_VEL_LIMIT_DEFAULT
    z_vel_limit : float, optional
        max velocity magnitude in the z direction, by default Z_VEL_LIMIT_DEFAULT
    control_bounds_high : Union[float, np.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default 1
    control_bounds_low : Union[float, np.ndarray], optional
        lower bound of allowable control. Pass a list for element specific limit. By default -1
    backup_controller : RTABackupController, optional
        backup controller object utilized by rta module to generate backup control.
        By default Docking2dStopLQRBackupController
    """

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
        control_bounds_high: float = 1,
        control_bounds_low: float = -1,
        backup_controller: RTABackupController = None,
        **kwargs
    ):
        self.m = m
        self.n = n
        self.v0 = v0
        self.v1_coef = v1_coef

        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit
        self.z_vel_limit = z_vel_limit

        if backup_controller is None:
            backup_controller = Docking3dStopLQRBackupController(m=self.m, n=self.n)

        super().__init__(
            *args,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            backup_controller=backup_controller,
            **kwargs
        )

    def _setup_properties(self):
        self._setup_docking_properties(self.m, self.n, self.v1_coef)

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints(self.v0, self.v1, self.x_vel_limit, self.y_vel_limit, self.z_vel_limit)

    def gen_rta_state(self, vector: np.ndarray) -> Docking3dState:
        return Docking3dState(vector=vector)

    def _pred_state(self, state: RTAState, step_size: float, control: np.ndarray) -> Docking3dState:
        return self.gen_rta_state(self._docking_pred_state(state, step_size, control))


class Docking3dImplicitSwitchingRTA(ImplicitSimplexModule, Docking3dRTAMixin):
    """Implements Implicit Switching RTA for the 3d Docking problem

    Parameters
    ----------
    backup_window : float
        Duration of time in seconds to evaluate backup controller trajectory
    m : float, optional
        mass in kg of spacecraft, by default M_DEFAULT
    n : float, optional
        orbital mean motion in rad/s of current Hill's reference frame, by default N_DEFAULT
    v0 : float, optional
        Maximum safe docking velocity in m/s, by default V0_DEFAULT
        v0 of v_limit = v0 + v1*n*||r||
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r||
    x_vel_limit : float, optional
        max velocity magnitude in the x direction, by default X_VEL_LIMIT_DEFAULT
    y_vel_limit : float, optional
        max velocity magnitude in the y direction, by default Y_VEL_LIMIT_DEFAULT
    z_vel_limit : float, optional
        max velocity magnitude in the z direction, by default Z_VEL_LIMIT_DEFAULT
    control_bounds_high : Union[float, np.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default 1
    control_bounds_low : Union[float, np.ndarray], optional
        lower bound of allowable control. Pass a list for element specific limit. By default -1
    backup_controller : RTABackupController, optional
        backup controller object utilized by rta module to generate backup control.
        By default Docking2dStopLQRBackupController
    """

    def __init__(
        self,
        *args,
        backup_window: float = 5,
        m: float = M_DEFAULT,
        n: float = N_DEFAULT,
        v0: float = V0_DEFAULT,
        v1_coef: float = V1_COEF_DEFAULT,
        x_vel_limit: float = X_VEL_LIMIT_DEFAULT,
        y_vel_limit: float = Y_VEL_LIMIT_DEFAULT,
        z_vel_limit: float = Z_VEL_LIMIT_DEFAULT,
        control_bounds_high: float = 1,
        control_bounds_low: float = -1,
        backup_controller: RTABackupController = None,
        **kwargs
    ):
        self.m = m
        self.n = n
        self.v0 = v0
        self.v1_coef = v1_coef

        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit
        self.z_vel_limit = z_vel_limit

        if backup_controller is None:
            backup_controller = Docking3dStopLQRBackupController(m=self.m, n=self.n)

        super().__init__(
            *args,
            backup_window=backup_window,
            backup_controller=backup_controller,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            **kwargs
        )

    def _setup_properties(self):
        self._setup_docking_properties(self.m, self.n, self.v1_coef)

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints(self.v0, self.v1, self.x_vel_limit, self.y_vel_limit, self.z_vel_limit)

    def gen_rta_state(self, vector: np.ndarray) -> Docking3dState:
        return Docking3dState(vector=vector)

    def _pred_state(self, state: RTAState, step_size: float, control: np.ndarray) -> Docking3dState:
        return self.gen_rta_state(self._docking_pred_state(state, step_size, control))


class Docking3dExplicitOptimizationRTA(ExplicitASIFModule, Docking3dRTAMixin):
    """
    Implements Explicit Optimization RTA for the 3d Docking problem

    Utilizes Explicit Active Set Invariance Function algorithm

    Parameters
    ----------
    m : float, optional
        mass in kg of spacecraft, by default M_DEFAULT
    n : float, optional
        orbital mean motion in rad/s of current Hill's reference frame, by default N_DEFAULT
    v0 : float, optional
        Maximum safe docking velocity in m/s, by default V0_DEFAULT
        v0 of v_limit = v0 + v1*n*||r||
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r||
    x_vel_limit : float, optional
        max velocity magnitude in the x direction, by default X_VEL_LIMIT_DEFAULT
    y_vel_limit : float, optional
        max velocity magnitude in the y direction, by default Y_VEL_LIMIT_DEFAULT
    z_vel_limit : float, optional
        max velocity magnitude in the z direction, by default Z_VEL_LIMIT_DEFAULT
    control_bounds_high : Union[float, np.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default 1
    control_bounds_low : Union[float, np.ndarray], optional
        lower bound of allowable control. Pass a list for element specific limit. By default -1
    """

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
        control_bounds_high: float = 1,
        control_bounds_low: float = -1,
        **kwargs
    ):
        self.m = m
        self.n = n
        self.v0 = v0
        self.v1_coef = v1_coef

        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit
        self.z_vel_limit = z_vel_limit

        super().__init__(*args, control_bounds_high=control_bounds_high, control_bounds_low=control_bounds_low, **kwargs)

    def _setup_properties(self):
        self._setup_docking_properties(self.m, self.n, self.v1_coef)

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints(self.v0, self.v1, self.x_vel_limit, self.y_vel_limit, self.z_vel_limit)

    def gen_rta_state(self, vector: np.ndarray) -> Docking3dState:
        return Docking3dState(vector=vector)

    def _pred_state(self, state: RTAState, step_size: float, control: np.ndarray) -> Docking3dState:
        return self.gen_rta_state(self._docking_pred_state(state, step_size, control))

    def state_transition_system(self, state: RTAState) -> np.ndarray:
        return self._docking_f_x(state)

    def state_transition_input(self, state: RTAState) -> np.ndarray:
        return self._docking_g_x(state)


class Docking3dImplicitOptimizationRTA(ImplicitASIFModule, Docking3dRTAMixin):
    """
    Implements Implicit Optimization RTA for the 3d Docking problem

    Utilizes Implicit Active Set Invariance Function algorithm

    Parameters
    ----------
    backup_window : float
        Duration of time in seconds to evaluate backup controller trajectory
    num_check_all : int
        Number of points at beginning of backup trajectory to check at every sequential simulation timestep.
        Should be <= backup_window.
        Defaults to 0 as skip_length defaults to 1 resulting in all backup trajectory points being checked.
    skip_length : int
        After num_check_all points in the backup trajectory are checked, the remainder of the backup window is filled by
        skipping every skip_length points to reduce the number of backup trajectory constraints. Will always check the
        last point in the backup trajectory as well.
        Defaults to 1, resulting in no skipping.
    m : float, optional
        mass in kg of spacecraft, by default M_DEFAULT
    n : float, optional
        orbital mean motion in rad/s of current Hill's reference frame, by default N_DEFAULT
    v0 : float, optional
        Maximum safe docking velocity in m/s, by default V0_DEFAULT
        v0 of v_limit = v0 + v1*n*||r||
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r||
    x_vel_limit : float, optional
        max velocity magnitude in the x direction, by default X_VEL_LIMIT_DEFAULT
    y_vel_limit : float, optional
        max velocity magnitude in the y direction, by default Y_VEL_LIMIT_DEFAULT
    z_vel_limit : float, optional
        max velocity magnitude in the z direction, by default Z_VEL_LIMIT_DEFAULT
    control_bounds_high : Union[float, np.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default 1
    control_bounds_low : Union[float, np.ndarray], optional
        lower bound of allowable control. Pass a list for element specific limit. By default -1
    backup_controller : RTABackupController, optional
        backup controller object utilized by rta module to generate backup control.
        By default Docking2dStopLQRBackupController
    """

    def __init__(
        self,
        *args,
        backup_window: float = 5,
        num_check_all: int = 5,
        skip_length: int = 1,
        m: float = M_DEFAULT,
        n: float = N_DEFAULT,
        v0: float = V0_DEFAULT,
        v1_coef: float = V1_COEF_DEFAULT,
        x_vel_limit: float = X_VEL_LIMIT_DEFAULT,
        y_vel_limit: float = Y_VEL_LIMIT_DEFAULT,
        z_vel_limit: float = Z_VEL_LIMIT_DEFAULT,
        control_bounds_high: float = 1,
        control_bounds_low: float = -1,
        backup_controller: RTABackupController = None,
        **kwargs
    ):
        self.m = m
        self.n = n
        self.v0 = v0
        self.v1_coef = v1_coef

        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit
        self.z_vel_limit = z_vel_limit

        if backup_controller is None:
            backup_controller = Docking3dStopLQRBackupController(m=self.m, n=self.n)

        super().__init__(
            *args,
            backup_window=backup_window,
            num_check_all=num_check_all,
            skip_length=skip_length,
            backup_controller=backup_controller,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            **kwargs
        )

    def _setup_properties(self):
        self._setup_docking_properties(self.m, self.n, self.v1_coef)

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints(self.v0, self.v1, self.x_vel_limit, self.y_vel_limit, self.z_vel_limit)

    def gen_rta_state(self, vector: np.ndarray) -> Docking3dState:
        return Docking3dState(vector=vector)

    def _pred_state(self, state: RTAState, step_size: float, control: np.ndarray) -> Docking3dState:
        return self.gen_rta_state(self._docking_pred_state(state, step_size, control))

    def compute_jacobian(self, state: RTAState) -> np.ndarray:
        return self.A + self.B @ self.backup_controller.compute_jacobian(state)

    def state_transition_system(self, state: RTAState) -> np.ndarray:
        return self._docking_f_x(state)

    def state_transition_input(self, state: RTAState) -> np.ndarray:
        return self._docking_g_x(state)


class Docking3dStopLQRBackupController(RTABackupController):
    """Simple LQR controller to bring velocity to zero for 3d CWHSpacecraft

    Parameters
    ----------
    m : float, optional
        mass in kg of spacecraft, by default M_DEFAULT
    n : float, optional
        orbital mean motion in rad/s of current Hill's reference frame, by default N_DEFAULT
    """

    def __init__(self, m: float = M_DEFAULT, n: float = N_DEFAULT):
        # LQR Gain Matrices
        self.Q = np.multiply(.050, np.eye(6))
        self.R = np.multiply(1000, np.eye(3))

        self.A, self.B = generate_cwh_matrices(m, n, mode="3d")

        # Solve the Algebraic Ricatti equation for the given system
        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        # Construct the constain gain matrix, K
        self.K = np.linalg.inv(self.R) @ (np.transpose(self.B) @ P)

    def generate_control(self, state: RTAState, step_size) -> np.ndarray:
        state_vec = state.vector
        state_des = np.copy(state_vec)
        state_des[3:] = 0

        error = state_vec - state_des
        backup_action = -self.K @ error

        return backup_action

    def compute_jacobian(self, state: RTAState):
        return -self.K


class Docking3dENMTTrackingBackupController(RTABackupController):
    """Tracking LQR controller that tracks closest eNMT for 3d CWHSpacecraft

    Parameters
    ----------
    m : float, optional
        mass in kg of spacecraft, by default M_DEFAULT
    n : float, optional
        orbital mean motion in rad/s of current Hill's reference frame, by default N_DEFAULT
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r||
    """

    def __init__(self, m: float = M_DEFAULT, n: float = N_DEFAULT, v1_coef: float = V1_COEF_DEFAULT):
        # LQR Gain Matrices
        self.Q = np.eye(12) * 1e-5
        self.R = np.eye(3) * 1e7

        self.num_enmt_points = 10
        self.eps = 1.5

        self.error_integral = np.zeros(6)
        self.error_integral_saved = self.error_integral

        self.n = n
        self.v1 = v1_coef * self.n

        C = np.eye(6)
        self.A, self.B = generate_cwh_matrices(m, self.n, mode="3d")

        A_int = np.vstack((np.hstack((self.A, np.zeros((6, 6)))), np.hstack((C, np.zeros((6, 6))))))
        B_int = np.vstack((self.B, np.zeros((6, 3))))
        # Solve the Algebraic Ricatti equation for the given system
        P = scipy.linalg.solve_continuous_are(A_int, B_int, self.Q, self.R)
        # Construct the constain gain matrix, K
        self.K = np.linalg.inv(self.R) @ (np.transpose(B_int) @ P)
        self.K_1 = self.K[:, 0:6]
        self.K_2 = self.K[:, 6:]

        def safety_constraint(theta1, theta2):
            if (np.tan(theta2)**2 + 4 * np.cos(theta1)**2) / np.sin(theta1)**2 <= (self.v1 / self.n)**2 - 4:
                return True
            return False

        problem = constraint.Problem()
        problem.addVariable('theta1', np.linspace(np.pi / 10, np.pi / 2 - np.pi / 10, self.num_enmt_points))
        problem.addVariable('theta2', np.linspace(0.001, np.pi, self.num_enmt_points))
        problem.addConstraint(safety_constraint, ['theta1', 'theta2'])
        self.solutions = problem.getSolutions()

        enmts = []
        self.z_coefs = []
        for _, solution in enumerate(self.solutions):
            theta1 = solution['theta1']
            theta2 = solution['theta2']
            z_coef = 1 / np.sin(theta1) * np.sqrt(np.tan(theta2)**2 + 4 * np.cos(theta1)**2)
            for i1 in range(self.num_enmt_points):
                self.z_coefs.append(z_coef)
                psi = i1 / self.num_enmt_points * 2 * np.pi
                nu = np.arctan(2 * np.cos(theta1) / np.tan(theta2)) + psi
                x_NMT = np.array(
                    [
                        np.sin(nu),
                        2 * np.cos(nu),
                        z_coef * np.sin(psi),
                        self.n * np.cos(nu),
                        -2 * self.n * np.sin(nu),
                        self.n * z_coef * np.cos(psi)
                    ]
                )
                enmts.append(x_NMT)

        self.enmts = np.array(enmts)

    def reset(self):
        self.error_integral = np.zeros(6)
        self.error_integral_saved = self.error_integral

    def generate_control(self, state: RTAState, step_size) -> np.ndarray:
        state_vec = state.vector

        # TODO this feels like a bug, we should only look for a new x_des if we lose the track
        state_desired = self.find_eNMT(state_vec)

        # check if controller is tracking trajectory
        if np.linalg.norm(state_vec[0:3] - state_desired[0:3]) <= self.eps:
            state_desired = state_desired + (self.A @ state_desired) * step_size

        error = state_vec - state_desired
        backup_action = -self.K_1 @ error - self.K_2 @ self.error_integral
        self.error_integral = self.error_integral + error * step_size

        return backup_action

    def compute_jacobian(self, state: RTAState) -> np.ndarray:
        return -self.K_1

    def find_eNMT(self, state_vec: np.ndarray) -> np.ndarray:
        """finds closest precomputed eNMT trajectory point

        Parameters
        ----------
        state_vec : np.ndarray
            current state vector of the system

        Returns
        -------
        np.ndarray
            closest eNMT point
        """
        b = state_vec[0] / (np.sin(np.arctan(2 * state_vec[0] / state_vec[1])))
        if b > 4868.5:
            b = b / b * 4868.5
        dist = []
        nmts = []

        for i in range(len(self.z_coefs)):
            if b * self.z_coefs[i] <= 9737:
                x_nmt = self.enmts[i, :] * b
                nmts.append(x_nmt)
                dist.append(np.linalg.norm(state_vec[0:3].flatten() - x_nmt[0:3]))

        return nmts[np.argmin(dist)]

    def save(self):
        self.error_integral_saved = self.error_integral

    def restore(self):
        self.error_integral = self.error_integral_saved


class ConstraintDocking3dRelativeVelocity(ConstraintModule):
    """CWH NMT velocity constraint

    Parameters
    ----------
    v0: float
        NMT safety constraint velocity upper bound constatnt component where ||v|| <= v0 + v1*distance. m/s
    v1: float
        NMT safety constraint velocity upper bound distance proportinality coefficient where
        ||v|| <= v0 + v1*distance. m/s
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.05, 0, 0.1])
    """

    def __init__(self, v0: float, v1: float, alpha: ConstraintStrengthener = None):
        self.v0 = v0
        self.v1 = v1

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.05, 0, 0.1])
        super().__init__(alpha=alpha)

    def _compute(self, state: RTAState) -> float:
        state_vec = state.vector
        return float((self.v0 + self.v1 * np.linalg.norm(state_vec[0:3])) - np.linalg.norm(state_vec[3:6]))

    def grad(self, state: RTAState) -> np.ndarray:
        state_vec = state.vector
        Hs = np.array(
            [
                [2 * self.v1**2, 0, 0, 0, 0, 0], [0, 2 * self.v1**2, 0, 0, 0, 0], [0, 0, 2 * self.v1**2, 0, 0, 0], [0, 0, 0, -2, 0, 0],
                [0, 0, 0, 0, -2, 0], [0, 0, 0, 0, 0, -2]
            ]
        )

        ghs = Hs @ state_vec
        ghs[0] = ghs[0] + 2 * self.v1 * self.v0 * state_vec[0] / np.linalg.norm(state_vec[0:3])
        ghs[1] = ghs[1] + 2 * self.v1 * self.v0 * state_vec[1] / np.linalg.norm(state_vec[0:3])
        ghs[2] = ghs[2] + 2 * self.v1 * self.v0 * state_vec[2] / np.linalg.norm(state_vec[0:3])
        return ghs
