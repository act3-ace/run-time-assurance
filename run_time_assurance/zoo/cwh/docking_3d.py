"""This module implements RTA methods for the docking problem with 3D CWH dynamics models"""

from collections import OrderedDict
from typing import Dict, Tuple, Union

import jax.numpy as jnp
import scipy
import safe_autonomy_simulation.dynamics as dynamics
import safe_autonomy_simulation.sims.spacecraft.defaults as defaults

from run_time_assurance.constraint import (
    ConstraintMagnitudeStateLimit,
    ConstraintModule,
    ConstraintStrengthener,
    PolynomialConstraintStrengthener,
)
from run_time_assurance.controller import RTABackupController
from run_time_assurance.rta import (
    ExplicitASIFModule,
    ExplicitSimplexModule,
    ImplicitASIFModule,
    ImplicitSimplexModule,
)
from run_time_assurance.state import RTAStateWrapper
from run_time_assurance.utils import norm_with_delta
from run_time_assurance.zoo.cwh.docking_2d import (
    V0_DEFAULT,
    X_VEL_LIMIT_DEFAULT,
    Y_VEL_LIMIT_DEFAULT,
)
from run_time_assurance.zoo.cwh.utils import generate_cwh_matrices

Z_VEL_LIMIT_DEFAULT = 10
V1_COEF_DEFAULT = 4
V1_DEFAULT = V1_COEF_DEFAULT * defaults.N_DEFAULT
V0_DISTANCE_DEFAULT = 0


class Docking3dState(RTAStateWrapper):
    """RTA state for cwh docking 3d RTA"""

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


class Docking3dRTAMixin:
    """Mixin class provides 3D docking RTA util functions
    Must call mixin methods using the RTA interface methods
    """

    def _setup_docking_properties(
        self,
        m: float,
        n: float,
        v1_coef: float,
        jit_compile_dict: Dict[str, bool],
        integration_method: str,
    ):
        """Initializes docking specific properties from other class members"""
        self.v1 = v1_coef * n

        A, B = generate_cwh_matrices(m, n, mode="3d")
        self.A = jnp.array(A)
        self.B = jnp.array(B)

        self.dynamics = dynamics.LinearODEDynamics(
            A=A,
            B=B,
            integration_method=integration_method,
            use_jax=True
        )

        assert integration_method in (
            "Euler",
            "RK45",
        ), f"Invalid integration method {integration_method}, must be 'Euler' or 'RK45'"

        # jit_compile_dict.setdefault("pred_state", True)
        # jit_compile_dict.setdefault("integrate", True)

    def _setup_docking_constraints(
        self,
        v0: float,
        v1: float,
        v0_distance: float,
        x_vel_limit: float,
        y_vel_limit: float,
        z_vel_limit: float,
    ) -> OrderedDict:
        """generates constraints used in the docking problem"""
        return OrderedDict(
            [
                (
                    "rel_vel",
                    ConstraintCWH3dRelativeVelocity(
                        v0=v0, v1=v1, v0_distance=v0_distance, bias=-1e-4
                    ),
                ),
                (
                    "x_vel",
                    ConstraintMagnitudeStateLimit(limit_val=x_vel_limit, state_index=3),
                ),
                (
                    "y_vel",
                    ConstraintMagnitudeStateLimit(limit_val=y_vel_limit, state_index=4),
                ),
                (
                    "z_vel",
                    ConstraintMagnitudeStateLimit(limit_val=z_vel_limit, state_index=5),
                ),
            ]
        )

    def _docking_pred_state(
        self, state: jnp.ndarray, step_size: float, control: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the next state given the current state and control action"""
        out, _ = self.dynamics.step(step_size, state, control)
        return out

    def _docking_f_x(self, state: jnp.ndarray) -> jnp.ndarray:
        """Computes the system contribution to the state transition: f(x) of dx/dt = f(x) + g(x)u"""
        return self.A @ state

    def _docking_g_x(self, _: jnp.ndarray) -> jnp.ndarray:
        """Computes the control input contribution to the state transition: g(x) of dx/dt = f(x) + g(x)u"""
        return jnp.copy(self.B)


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
        v0 of v_limit = v0 + v1*n*||r-v0_distance||
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r-v0_distance||
    v0_distance: float
        NMT safety constraint minimum distance where v0 is applied. By default 0.
    x_vel_limit : float, optional
        max velocity magnitude in the x direction, by default X_VEL_LIMIT_DEFAULT
    y_vel_limit : float, optional
        max velocity magnitude in the y direction, by default Y_VEL_LIMIT_DEFAULT
    z_vel_limit : float, optional
        max velocity magnitude in the z direction, by default Z_VEL_LIMIT_DEFAULT
    control_bounds_high : Union[float, list, np.ndarray, jnp.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default 1
    control_bounds_low : Union[float, list, np.ndarray, jnp.ndarray], optional
        lower bound of allowable control. Pass a list for element specific limit. By default -1
    backup_controller : RTABackupController, optional
        backup controller object utilized by rta module to generate backup control.
        By default Docking2dStopLQRBackupController
    jit_compile_dict: Dict[str, bool], optional
        Dictionary specifying which subroutines will be jax jit compiled. Behavior defined in self.compose()
    integration_method: str, optional
        Integration method to use, either 'RK45' or 'Euler'
    """

    def __init__(
        self,
        *args,
        m: float = defaults.M_DEFAULT,
        n: float = defaults.N_DEFAULT,
        v0: float = V0_DEFAULT,
        v1_coef: float = V1_COEF_DEFAULT,
        v0_distance: float = V0_DISTANCE_DEFAULT,
        x_vel_limit: float = X_VEL_LIMIT_DEFAULT,
        y_vel_limit: float = Y_VEL_LIMIT_DEFAULT,
        z_vel_limit: float = Z_VEL_LIMIT_DEFAULT,
        control_bounds_high: float = 1,
        control_bounds_low: float = -1,
        backup_controller: RTABackupController = None,
        jit_compile_dict: Dict[str, bool] = None,
        integration_method: str = "RK45",
        **kwargs,
    ):
        self.m = m
        self.n = n
        self.v0 = v0
        self.v1_coef = v1_coef
        self.v0_distance = v0_distance

        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit
        self.z_vel_limit = z_vel_limit
        self.integration_method = integration_method

        if backup_controller is None:
            backup_controller = Docking3dStopLQRBackupController(m=self.m, n=self.n)

        if jit_compile_dict is None:
            jit_compile_dict = {"constraint_violation": True}

        super().__init__(
            *args,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            backup_controller=backup_controller,
            jit_compile_dict=jit_compile_dict,
            **kwargs,
        )

    def _setup_properties(self):
        self._setup_docking_properties(
            self.m, self.n, self.v1_coef, self.jit_compile_dict, self.integration_method
        )

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints(
            self.v0,
            self.v1,
            self.v0_distance,
            self.x_vel_limit,
            self.y_vel_limit,
            self.z_vel_limit,
        )

    def _pred_state(
        self, state: jnp.ndarray, step_size: float, control: jnp.ndarray
    ) -> jnp.ndarray:
        return self._docking_pred_state(state, step_size, control)


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
        v0 of v_limit = v0 + v1*n*||r-v0_distance||
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r-v0_distance||
    v0_distance: float
        NMT safety constraint minimum distance where v0 is applied. By default 0.
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
    jit_compile_dict: Dict[str, bool], optional
        Dictionary specifying which subroutines will be jax jit compiled. Behavior defined in self.compose()
    integration_method: str, optional
        Integration method to use, either 'RK45' or 'Euler'
    """

    def __init__(
        self,
        *args,
        backup_window: float = 5,
        m: float = defaults.M_DEFAULT,
        n: float = defaults.N_DEFAULT,
        v0: float = V0_DEFAULT,
        v1_coef: float = V1_COEF_DEFAULT,
        v0_distance: float = V0_DISTANCE_DEFAULT,
        x_vel_limit: float = X_VEL_LIMIT_DEFAULT,
        y_vel_limit: float = Y_VEL_LIMIT_DEFAULT,
        z_vel_limit: float = Z_VEL_LIMIT_DEFAULT,
        control_bounds_high: float = 1,
        control_bounds_low: float = -1,
        backup_controller: RTABackupController = None,
        jit_compile_dict: Dict[str, bool] = None,
        integration_method: str = "RK45",
        **kwargs,
    ):
        self.m = m
        self.n = n
        self.v0 = v0
        self.v1_coef = v1_coef
        self.v0_distance = v0_distance

        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit
        self.z_vel_limit = z_vel_limit
        self.integration_method = integration_method

        if backup_controller is None:
            backup_controller = Docking3dStopLQRBackupController(m=self.m, n=self.n)

        if jit_compile_dict is None:
            jit_compile_dict = {"constraint_violation": True}

        super().__init__(
            *args,
            backup_window=backup_window,
            backup_controller=backup_controller,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            jit_compile_dict=jit_compile_dict,
            **kwargs,
        )

    def _setup_properties(self):
        self._setup_docking_properties(
            self.m, self.n, self.v1_coef, self.jit_compile_dict, self.integration_method
        )

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints(
            self.v0,
            self.v1,
            self.v0_distance,
            self.x_vel_limit,
            self.y_vel_limit,
            self.z_vel_limit,
        )

    def _pred_state(
        self, state: jnp.ndarray, step_size: float, control: jnp.ndarray
    ) -> jnp.ndarray:
        return self._docking_pred_state(state, step_size, control)


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
        v0 of v_limit = v0 + v1*n*||r-v0_distance||
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r-v0_distance||
    v0_distance: float
        NMT safety constraint minimum distance where v0 is applied. By default 0.
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
    jit_compile_dict: Dict[str, bool], optional
        Dictionary specifying which subroutines will be jax jit compiled. Behavior defined in self.compose()
    """

    def __init__(
        self,
        *args,
        m: float = defaults.M_DEFAULT,
        n: float = defaults.N_DEFAULT,
        v0: float = V0_DEFAULT,
        v1_coef: float = V1_COEF_DEFAULT,
        v0_distance: float = V0_DISTANCE_DEFAULT,
        x_vel_limit: float = X_VEL_LIMIT_DEFAULT,
        y_vel_limit: float = Y_VEL_LIMIT_DEFAULT,
        z_vel_limit: float = Z_VEL_LIMIT_DEFAULT,
        control_bounds_high: float = 1,
        control_bounds_low: float = -1,
        jit_compile_dict: Dict[str, bool] = None,
        **kwargs,
    ):
        self.m = m
        self.n = n
        self.v0 = v0
        self.v1_coef = v1_coef
        self.v0_distance = v0_distance

        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit
        self.z_vel_limit = z_vel_limit

        if jit_compile_dict is None:
            jit_compile_dict = {"generate_barrier_constraint_mats": True}

        super().__init__(
            *args,
            control_dim=3,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            jit_compile_dict=jit_compile_dict,
            **kwargs,
        )

    def _setup_properties(self):
        self._setup_docking_properties(
            self.m, self.n, self.v1_coef, self.jit_compile_dict, "RK45"
        )

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints(
            self.v0,
            self.v1,
            self.v0_distance,
            self.x_vel_limit,
            self.y_vel_limit,
            self.z_vel_limit,
        )

    def state_transition_system(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._docking_f_x(state)

    def state_transition_input(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._docking_g_x(state)


class Docking3dImplicitOptimizationRTA(ImplicitASIFModule, Docking3dRTAMixin):
    """
    Implements Implicit Optimization RTA for the 3d Docking problem

    Utilizes Implicit Active Set Invariance Function algorithm

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
        v0 of v_limit = v0 + v1*n*||r-v0_distance||
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r-v0_distance||
    v0_distance: float
        NMT safety constraint minimum distance where v0 is applied. By default 0.
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
    jit_compile_dict: Dict[str, bool], optional
        Dictionary specifying which subroutines will be jax jit compiled. Behavior defined in self.compose()
    integration_method: str, optional
        Integration method to use, either 'RK45' or 'Euler'
    """

    def __init__(
        self,
        *args,
        backup_window: float = 5,
        m: float = defaults.M_DEFAULT,
        n: float = defaults.N_DEFAULT,
        v0: float = V0_DEFAULT,
        v1_coef: float = V1_COEF_DEFAULT,
        v0_distance: float = V0_DISTANCE_DEFAULT,
        x_vel_limit: float = X_VEL_LIMIT_DEFAULT,
        y_vel_limit: float = Y_VEL_LIMIT_DEFAULT,
        z_vel_limit: float = Z_VEL_LIMIT_DEFAULT,
        control_bounds_high: float = 1,
        control_bounds_low: float = -1,
        backup_controller: RTABackupController = None,
        jit_compile_dict: Dict[str, bool] = None,
        integration_method: str = "RK45",
        **kwargs,
    ):
        self.m = m
        self.n = n
        self.v0 = v0
        self.v1_coef = v1_coef
        self.v0_distance = v0_distance

        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit
        self.z_vel_limit = z_vel_limit

        if backup_controller is None:
            backup_controller = Docking3dStopLQRBackupController(m=self.m, n=self.n)

        if jit_compile_dict is None:
            jit_compile_dict = {"generate_ineq_constraint_mats": True}

        super().__init__(
            *args,
            control_dim=3,
            backup_window=backup_window,
            backup_controller=backup_controller,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            jit_compile_dict=jit_compile_dict,
            integration_method=integration_method,
            **kwargs,
        )

    def _setup_properties(self):
        self._setup_docking_properties(
            self.m, self.n, self.v1_coef, self.jit_compile_dict, self.integration_method
        )

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints(
            self.v0,
            self.v1,
            self.v0_distance,
            self.x_vel_limit,
            self.y_vel_limit,
            self.z_vel_limit,
        )

    def state_transition_system(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._docking_f_x(state)

    def state_transition_input(self, state: jnp.ndarray) -> jnp.ndarray:
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

    def __init__(self, m: float = defaults.M_DEFAULT, n: float = defaults.N_DEFAULT):
        # LQR Gain Matrices
        self.Q = jnp.multiply(0.050, jnp.eye(6))
        self.R = jnp.multiply(1000, jnp.eye(3))

        self.A, self.B = generate_cwh_matrices(m, n, mode="3d")

        # Solve the Algebraic Ricatti equation for the given system
        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        # Construct the constain gain matrix, K
        self.K = jnp.linalg.inv(self.R) @ (jnp.transpose(self.B) @ P)

        super().__init__()

    def _generate_control(
        self,
        state: jnp.ndarray,
        step_size: float,
        controller_state: Union[jnp.ndarray, Dict[str, jnp.ndarray], None] = None,
    ) -> Tuple[jnp.ndarray, None]:
        state_des = jnp.copy(state)
        state_des = state_des.at[3:].set(0)

        error = state - state_des
        backup_action = -self.K @ error

        return backup_action, None


class ConstraintCWH3dRelativeVelocity(ConstraintModule):
    """CWH 3d NMT velocity constraint

    Parameters
    ----------
    v0: float
        NMT safety constraint velocity upper bound constant component where ||v|| <= v0 + v1*(distance-v0_distance). m/s
    v1: float
        NMT safety constraint velocity upper bound distance proportinality coefficient where
        ||v|| <= v0 + v1*(distance-v0_distance). 1/s
    v0_distance: float
        NMT safety constraint minimum distance where v0 is applied. By default 0.
    delta: float
        Small postiive value summed inside the vector norm sqrt operation to make constraint differentiable at 0
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 0.05, 0, 0.5])
    """

    def __init__(
        self,
        v0: float,
        v1: float,
        v0_distance: float = 0,
        delta: float = 1e-5,
        alpha: ConstraintStrengthener = None,
        **kwargs,
    ):
        self.delta = delta

        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 0.05, 0, 0.005])
        super().__init__(
            alpha=alpha,
            params={"v0": v0, "v1": v1, "v0_distance": v0_distance},
            **kwargs,
        )

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        return (
            params["v0"]
            + params["v1"]
            * (norm_with_delta(state[0:3], self.delta) - params["v0_distance"])
        ) - norm_with_delta(state[3:6], self.delta)
