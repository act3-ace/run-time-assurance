"""This module implements RTA methods for the 1D integrator problem applied to spacecraft docking"""

from collections import OrderedDict
from typing import Dict, Tuple, Union

import jax.numpy as jnp
import numpy as np
from safe_autonomy_simulation.entities.integrator import (
    M_DEFAULT,
    PointMassIntegratorDynamics,
)

from run_time_assurance.constraint import (
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


class Integrator1dDockingState(RTAStateWrapper):
    """RTA state for Integrator 1d docking RTA"""

    @property
    def x(self) -> float:
        """Getter for x position"""
        return self.vector[0]

    @x.setter
    def x(self, val: float):
        """Setter for x position"""
        self.vector[0] = val

    @property
    def x_dot(self) -> float:
        """Getter for x velocity component"""
        return self.vector[1]

    @x_dot.setter
    def x_dot(self, val: float):
        """Setter for x velocity component"""
        self.vector[1] = val


class Integrator1dDockingRTAMixin:
    """Mixin class provides Integrator 1D docking RTA util functions
    Must call mixin methods using the RTA interface methods
    """

    def _setup_docking_properties(
        self, m: float, jit_compile_dict: Dict[str, bool], integration_method: str
    ):
        """Initializes docking specific properties from other class members"""
        self.dynamics = PointMassIntegratorDynamics(
            m=m,
            mode="1d",
            integration_method=integration_method,
            use_jax=True
        )
        self.A = jnp.array(self.dynamics.A)
        self.B = jnp.array(self.dynamics.B)

        assert (
            integration_method in ("RK45", "Euler")
        ), f"Invalid integration method {integration_method}, must be either 'RK45' or 'Euler'"

        # jit_compile_dict.setdefault("pred_state", True)
        # jit_compile_dict.setdefault("integrate", True)

    def _setup_docking_constraints_explicit(self) -> OrderedDict:
        """generates explicit constraints used in the docking problem"""
        return OrderedDict(
            [("rel_vel", ConstraintIntegrator1dDockingCollisionExplicit())]
        )

    def _setup_docking_constraints_implicit(self) -> OrderedDict:
        """generates implicit constraints used in the docking problem"""
        return OrderedDict(
            [
                (
                    "rel_vel",
                    ConstraintIntegrator1dDockingCollisionImplicit(
                        subsample_constraints_num_least=1
                    ),
                )
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


class Integrator1dDockingExplicitSwitchingRTA(
    ExplicitSimplexModule, Integrator1dDockingRTAMixin
):
    """Implements Explicit Switching RTA for the Integrator 1d Docking problem

    Parameters
    ----------
    control_bounds_high : Union[float, np.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default 1
    control_bounds_low : Union[float, np.ndarray], optional
        lower bound of allowable control. Pass a list for element specific limit. By default -1
    backup_controller : RTABackupController, optional
        backup controller object utilized by rta module to generate backup control.
        By default Integrator1dDockingBackupController
    integration_method: str, optional
        Integration method to use, either 'RK45' or 'Euler'
    """

    def __init__(
        self,
        *args,
        m: float = M_DEFAULT,
        control_bounds_high: Union[float, np.ndarray] = 1,
        control_bounds_low: Union[float, np.ndarray] = -1,
        backup_controller: RTABackupController = None,
        jit_compile_dict: Dict[str, bool] = None,
        integration_method: str = "Euler",
        **kwargs,
    ):
        self.m = m
        self.integration_method = integration_method

        if backup_controller is None:
            backup_controller = Integrator1dDockingBackupController()

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
            self.m, self.jit_compile_dict, self.integration_method
        )

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints_explicit()

    def _pred_state(
        self, state: jnp.ndarray, step_size: float, control: jnp.ndarray
    ) -> jnp.ndarray:
        return self._docking_pred_state(state, step_size, control)


class Integrator1dDockingImplicitSwitchingRTA(
    ImplicitSimplexModule, Integrator1dDockingRTAMixin
):
    """Implements Implicit Switching RTA for the Integrator 1d Docking problem

    Parameters
    ----------
    backup_window : float
        Duration of time in seconds to evaluate backup controller trajectory
    platform_name : str, optional
        name of the platform this rta module is attaching to, by default "deputy"
    control_bounds_high : Union[float, np.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default 1
    control_bounds_low : Union[float, np.ndarray], optional
        lower bound of allowable control. Pass a list for element specific limit. By default -1
    backup_controller : RTABackupController, optional
        backup controller object utilized by rta module to generate backup control.
        By default Integrator1dDockingBackupController
    integration_method: str, optional
        Integration method to use, either 'RK45' or 'Euler'
    """

    def __init__(
        self,
        *args,
        backup_window: float = 2,
        m: float = M_DEFAULT,
        control_bounds_high: Union[float, np.ndarray] = 1,
        control_bounds_low: Union[float, np.ndarray] = -1,
        backup_controller: RTABackupController = None,
        jit_compile_dict: Dict[str, bool] = None,
        integration_method: str = "Euler",
        **kwargs,
    ):
        self.m = m
        self.integration_method = integration_method

        if backup_controller is None:
            backup_controller = Integrator1dDockingBackupController()

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
            self.m, self.jit_compile_dict, self.integration_method
        )

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints_implicit()

    def _pred_state(
        self, state: jnp.ndarray, step_size: float, control: jnp.ndarray
    ) -> jnp.ndarray:
        return self._docking_pred_state(state, step_size, control)


class Integrator1dDockingExplicitOptimizationRTA(
    ExplicitASIFModule, Integrator1dDockingRTAMixin
):
    """
    Implements Explicit Optimization RTA for the Integrator 1d Docking problem

    Utilizes Explicit Active Set Invariance Function algorithm

    Parameters
    ----------
    platform_name : str, optional
        name of the platform this rta module is attaching to, by default "deputy"
    control_bounds_high : Union[float, np.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default 1
    control_bounds_low : Union[float, np.ndarray], optional
        lower bound of allowable control. Pass a list for element specific limit. By default -1
    """

    def __init__(
        self,
        *args,
        m: float = M_DEFAULT,
        control_bounds_high: Union[float, np.ndarray] = 1,
        control_bounds_low: Union[float, np.ndarray] = -1,
        jit_compile_dict: Dict[str, bool] = None,
        **kwargs,
    ):
        self.m = m
        if jit_compile_dict is None:
            jit_compile_dict = {"generate_barrier_constraint_mats": True}

        super().__init__(
            *args,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            control_dim=1,
            jit_compile_dict=jit_compile_dict,
            **kwargs,
        )

    def _setup_properties(self):
        self._setup_docking_properties(self.m, self.jit_compile_dict, "RK45")

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints_explicit()

    def state_transition_system(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._docking_f_x(state)

    def state_transition_input(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._docking_g_x(state)


class Integrator1dDockingImplicitOptimizationRTA(
    ImplicitASIFModule, Integrator1dDockingRTAMixin
):
    """
    Implements Implicit Optimization RTA for the Integrator 1d Docking problem

    Utilizes Implicit Active Set Invariance Function algorithm

    Parameters
    ----------
    backup_window : float
        Duration of time in seconds to evaluate backup controller trajectory
    platform_name : str, optional
        name of the platform this rta module is attaching to, by default "deputy"
    control_bounds_high : Union[float, np.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default 1
    control_bounds_low : Union[float, np.ndarray], optional
        lower bound of allowable control. Pass a list for element specific limit. By default -1
    backup_controller : RTABackupController, optional
        backup controller object utilized by rta module to generate backup control.
        By default Integrator1dDockingBackupController
    integration_method: str, optional
        Integration method to use, either 'RK45' or 'Euler'
    """

    def __init__(
        self,
        *args,
        backup_window: float = 2,
        m: float = M_DEFAULT,
        control_bounds_high: Union[float, np.ndarray] = 1,
        control_bounds_low: Union[float, np.ndarray] = -1,
        backup_controller: RTABackupController = None,
        jit_compile_dict: Dict[str, bool] = None,
        integration_method: str = "Euler",
        **kwargs,
    ):
        self.m = m
        self.integration_method = integration_method
        if backup_controller is None:
            backup_controller = Integrator1dDockingBackupController()

        if jit_compile_dict is None:
            jit_compile_dict = {"generate_ineq_constraint_mats": True}

        super().__init__(
            *args,
            backup_window=backup_window,
            integration_method=integration_method,
            backup_controller=backup_controller,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            control_dim=1,
            jit_compile_dict=jit_compile_dict,
            **kwargs,
        )

    def _setup_properties(self):
        self._setup_docking_properties(
            self.m, self.jit_compile_dict, self.integration_method
        )

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints_implicit()

    def state_transition_system(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._docking_f_x(state)

    def state_transition_input(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._docking_g_x(state)


class Integrator1dDockingBackupController(RTABackupController):
    """Max braking backup controller to bring velocity to zero for 1d Integrator"""

    def _generate_control(
        self,
        state: jnp.ndarray,
        step_size: float,
        controller_state: Union[jnp.ndarray, Dict[str, jnp.ndarray], None] = None,
    ) -> Tuple[jnp.ndarray, None]:
        return jnp.array([-1]), None


class ConstraintIntegrator1dDockingCollisionExplicit(ConstraintModule):
    """Integrator 1d docking explicit chief collision avoidance constraint

    Parameters
    ----------
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 1, 0, 30])
    """

    def __init__(self, alpha: ConstraintStrengthener = None, **kwargs):
        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 1, 0, 30])
        super().__init__(alpha=alpha, **kwargs)

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        return -2 * state[0] - state[1] ** 2


class ConstraintIntegrator1dDockingCollisionImplicit(ConstraintModule):
    """Integrator 1d docking implicit chief collision avoidance constraint

    Parameters
    ----------
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 10, 0, 30])
    """

    def __init__(self, alpha: ConstraintStrengthener = None, **kwargs):
        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 10, 0, 30])
        super().__init__(alpha=alpha, **kwargs)

    def _compute(self, state: jnp.ndarray, params: dict) -> float:
        return -state[0]
