"""This module implements RTA methods for the docking problem with 2D CWH dynamics models
"""
from collections import OrderedDict
from typing import Union

import numpy as np
import scipy
from safe_autonomy_dynamics.base_models import BaseLinearODESolverDynamics
from safe_autonomy_dynamics.cwh import M_DEFAULT, N_DEFAULT, generate_cwh_matrices

from run_time_assurance.constraint import ConstraintModule, ConstraintMagnitudeStateLimit, ConstraintStrengthener, PolynomialConstraintStrengthener
from run_time_assurance.rta import ExplicitASIFModule, ExplicitSimplexModule, ImplicitASIFModule, ImplicitSimplexModule, RTABackupController
from run_time_assurance.state import RTAState

X_VEL_LIMIT_DEFAULT = 10
Y_VEL_LIMIT_DEFAULT = 10
V0_DEFAULT = 0.2
V1_COEF_DEFAULT = 2
V1_DEFAULT = V1_COEF_DEFAULT * N_DEFAULT


class Docking2dState(RTAState):
    """RTA state for cwh docking 2d RTA"""

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
    def x_dot(self) -> float:
        """Getter for x velocity component"""
        return self._vector[2]

    @x_dot.setter
    def x_dot(self, val: float):
        """Setter for x velocity component"""
        self._vector[2] = val

    @property
    def y_dot(self) -> float:
        """Getter for y velocity component"""
        return self._vector[3]

    @y_dot.setter
    def y_dot(self, val: float):
        """Setter for y velocity component"""
        self._vector[3] = val


class Docking2dRTAMixin:
    """Mixin class provides 2D docking RTA util functions
    Must call mixin methods using the RTA interface methods
    """

    def _setup_docking_properties(self, m: float, n: float, v1_coef: float):
        """Initializes docking specific properties from other class members"""
        self.v1 = v1_coef * n
        self.A, self.B = generate_cwh_matrices(m, n, mode="2d")
        self.dynamics = BaseLinearODESolverDynamics(A=self.A, B=self.B, integration_method="RK45")

    def _setup_docking_constraints(self, v0: float, v1: float, x_vel_limit: float, y_vel_limit: float) -> OrderedDict:
        """generates constraints used in the docking problem"""
        return OrderedDict(
            [
                ('rel_vel', ConstraintDocking2dRelativeVelocity(v0=v0, v1=v1)),
                ('x_vel', ConstraintMagnitudeStateLimit(limit_val=x_vel_limit, state_index=2)),
                ('y_vel', ConstraintMagnitudeStateLimit(limit_val=y_vel_limit, state_index=3)),
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


class Docking2dExplicitSwitchingRTA(ExplicitSimplexModule, Docking2dRTAMixin):
    """Implements Explicit Switching RTA for the 2d Docking problem

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
        control_bounds_high: Union[float, np.ndarray] = 1,
        control_bounds_low: Union[float, np.ndarray] = -1,
        backup_controller: RTABackupController = None,
        **kwargs
    ):
        self.m = m
        self.n = n
        self.v0 = v0
        self.v1_coef = v1_coef

        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit

        if backup_controller is None:
            backup_controller = Docking2dStopLQRBackupController(m=self.m, n=self.n)

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
        return self._setup_docking_constraints(self.v0, self.v1, self.x_vel_limit, self.y_vel_limit)

    def gen_rta_state(self, vector: np.ndarray) -> Docking2dState:
        return Docking2dState(vector=vector)

    def _pred_state(self, state: RTAState, step_size: float, control: np.ndarray) -> Docking2dState:
        return self.gen_rta_state(self._docking_pred_state(state, step_size, control))


class Docking2dImplicitSwitchingRTA(ImplicitSimplexModule, Docking2dRTAMixin):
    """Implements Implicit Switching RTA for the 2d Docking problem

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
        control_bounds_high: Union[float, np.ndarray] = 1,
        control_bounds_low: Union[float, np.ndarray] = -1,
        backup_controller: RTABackupController = None,
        **kwargs
    ):

        self.m = m
        self.n = n
        self.v0 = v0
        self.v1_coef = v1_coef

        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit
        if backup_controller is None:
            backup_controller = Docking2dStopLQRBackupController(m=self.m, n=self.n)

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
        return self._setup_docking_constraints(self.v0, self.v1, self.x_vel_limit, self.y_vel_limit)

    def gen_rta_state(self, vector: np.ndarray) -> Docking2dState:
        return Docking2dState(vector=vector)

    def _pred_state(self, state: RTAState, step_size: float, control: np.ndarray) -> Docking2dState:
        return self.gen_rta_state(self._docking_pred_state(state, step_size, control))


class Docking2dExplicitOptimizationRTA(ExplicitASIFModule, Docking2dRTAMixin):
    """
    Implements Explicit Optimization RTA for the 2d Docking problem

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
        control_bounds_high: Union[float, np.ndarray] = 1,
        control_bounds_low: Union[float, np.ndarray] = -1,
        **kwargs
    ):

        self.m = m
        self.n = n
        self.v0 = v0
        self.v1_coef = v1_coef

        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit
        super().__init__(*args, control_bounds_high=control_bounds_high, control_bounds_low=control_bounds_low, **kwargs)

    def _setup_properties(self):
        self._setup_docking_properties(self.m, self.n, self.v1_coef)

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints(self.v0, self.v1, self.x_vel_limit, self.y_vel_limit)

    def gen_rta_state(self, vector: np.ndarray) -> Docking2dState:
        return Docking2dState(vector=vector)

    def _pred_state(self, state: RTAState, step_size: float, control: np.ndarray) -> Docking2dState:
        return self.gen_rta_state(self._docking_pred_state(state, step_size, control))

    def state_transition_system(self, state: RTAState) -> np.ndarray:
        return self._docking_f_x(state)

    def state_transition_input(self, state: RTAState) -> np.ndarray:
        return self._docking_g_x(state)


class Docking2dImplicitOptimizationRTA(ImplicitASIFModule, Docking2dRTAMixin):
    """
    Implements Implicit Optimization RTA for the 2d Docking problem

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
        control_bounds_high: Union[float, np.ndarray] = 1,
        control_bounds_low: Union[float, np.ndarray] = -1,
        backup_controller: RTABackupController = None,
        **kwargs
    ):
        self.m = m
        self.n = n
        self.v0 = v0
        self.v1_coef = v1_coef

        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit
        if backup_controller is None:
            backup_controller = Docking2dStopLQRBackupController(m=self.m, n=self.n)

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
        return self._setup_docking_constraints(self.v0, self.v1, self.x_vel_limit, self.y_vel_limit)

    def gen_rta_state(self, vector: np.ndarray) -> Docking2dState:
        return Docking2dState(vector=vector)

    def _pred_state(self, state: RTAState, step_size: float, control: np.ndarray) -> Docking2dState:
        return self.gen_rta_state(self._docking_pred_state(state, step_size, control))

    def compute_jacobian(self, state: RTAState) -> np.ndarray:
        return self.A + self.B @ self.backup_controller.compute_jacobian(state)

    def state_transition_system(self, state: RTAState) -> np.ndarray:
        return self._docking_f_x(state)

    def state_transition_input(self, state: RTAState) -> np.ndarray:
        return self._docking_g_x(state)


class Docking2dStopLQRBackupController(RTABackupController):
    """Simple LQR controller to bring velocity to zero for 2d CWHSpacecraft

    Parameters
    ----------
    m : float, optional
        mass in kg of spacecraft, by default M_DEFAULT
    n : float, optional
        orbital mean motion in rad/s of current Hill's reference frame, by default N_DEFAULT
    """

    def __init__(self, m: float = M_DEFAULT, n: float = N_DEFAULT):
        # LQR Gain Matrices
        self.Q = np.multiply(.050, np.eye(4))
        self.R = np.multiply(1000, np.eye(2))

        self.A, self.B = generate_cwh_matrices(m, n, mode="2d")

        # Solve the Algebraic Ricatti equation for the given system
        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        # Construct the constain gain matrix, K
        self.K = np.linalg.inv(self.R) @ (np.transpose(self.B) @ P)

    def generate_control(self, state: RTAState, step_size: float) -> np.ndarray:
        state_vec = state.vector
        state_des = np.copy(state_vec)
        state_des[2:] = 0

        error = state_vec - state_des
        backup_action = -self.K @ error

        return backup_action

    def compute_jacobian(self, state: RTAState):
        return -self.K


class ConstraintDocking2dRelativeVelocity(ConstraintModule):
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

        return float((self.v0 + self.v1 * np.linalg.norm(state_vec[0:2])) - np.linalg.norm(state_vec[2:4]))

    def grad(self, state: RTAState) -> np.ndarray:
        state_vec = state.vector

        Hs = np.array([[2 * self.v1**2, 0, 0, 0], [0, 2 * self.v1**2, 0, 0], [0, 0, -2, 0], [0, 0, 0, -2]])

        ghs = Hs @ state_vec
        ghs[0] = ghs[0] + 2 * self.v1 * self.v0 * state_vec[0] / np.linalg.norm(state_vec[0:2])
        ghs[1] = ghs[1] + 2 * self.v1 * self.v0 * state_vec[1] / np.linalg.norm(state_vec[0:2])
        return ghs
