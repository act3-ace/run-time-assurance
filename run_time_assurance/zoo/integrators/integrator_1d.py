"""This module implements RTA methods for the 1D integrator problem applied to spacecraft docking
"""
from collections import OrderedDict
from typing import Dict, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
from safe_autonomy_dynamics.base_models import BaseLinearODESolverDynamics
from safe_autonomy_dynamics.integrators import M_DEFAULT, generate_dynamics_matrices

from run_time_assurance.constraint import ConstraintModule, ConstraintStrengthener, PolynomialConstraintStrengthener
from run_time_assurance.controller import RTABackupController
from run_time_assurance.rta import ExplicitASIFModule, ExplicitSimplexModule, ImplicitASIFModule, ImplicitSimplexModule
from run_time_assurance.state import RTAStateWrapper
from run_time_assurance.utils import to_jnp_array_jit


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

    def _setup_docking_properties(self, m: float, jit_compile_dict: Dict[str, bool], integration_method: str):
        """Initializes docking specific properties from other class members"""
        A, B = generate_dynamics_matrices(m=m, mode='1d')
        self.A = jnp.array(A)
        self.B = jnp.array(B)
        self.dynamics = BaseLinearODESolverDynamics(A=A, B=B, integration_method=integration_method)

        if integration_method == 'RK45':
            jit_compile_dict.setdefault('pred_state', False)
            jit_compile_dict.setdefault('integrate', False)
            if jit_compile_dict.get('pred_state'):
                raise ValueError('pred_state uses RK45 integration and can not be compiled using jit')
            if jit_compile_dict.get('integrate'):
                raise ValueError('integrate uses RK45 integration and can not be compiled using jit')
        elif integration_method in ('Euler', 'RK45_JAX'):
            jit_compile_dict.setdefault('pred_state', True)
            jit_compile_dict.setdefault('integrate', True)
        else:
            raise ValueError('integration_method must be either RK45_JAX, RK45, or Euler')

    def _setup_docking_constraints_explicit(self) -> OrderedDict:
        """generates explicit constraints used in the docking problem"""
        return OrderedDict([('rel_vel', ConstraintIntegrator1dDockingCollisionExplicit())])

    def _setup_docking_constraints_implicit(self) -> OrderedDict:
        """generates implicit constraints used in the docking problem"""
        return OrderedDict([('rel_vel', ConstraintIntegrator1dDockingCollisionImplicit())])

    def _docking_pred_state(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray, integration_method: str) -> jnp.ndarray:
        """Predicts the next state given the current state and control action"""
        if integration_method == 'RK45':
            next_state_vec, _ = self.dynamics.step(step_size, np.array(state), np.array(control))
            out = to_jnp_array_jit(next_state_vec)
        elif integration_method == 'Euler':
            state_dot = self._docking_f_x(state) + self._docking_g_x(state) @ control
            out = state + state_dot * step_size
        elif integration_method == 'RK45_JAX':
            sol = odeint(self._docking_state_dot, state, jnp.linspace(0., step_size, 11), control)
            out = sol[-1, :]
        else:
            raise ValueError('integration_method must be either RK45_JAX, RK45, or Euler')
        return out

    def _docking_f_x(self, state: jnp.ndarray) -> jnp.ndarray:
        """Computes the system contribution to the state transition: f(x) of dx/dt = f(x) + g(x)u"""
        return self.A @ state

    def _docking_g_x(self, _: jnp.ndarray) -> jnp.ndarray:
        """Computes the control input contribution to the state transition: g(x) of dx/dt = f(x) + g(x)u"""
        return jnp.copy(self.B)

    def _docking_state_dot(
        self,
        state: jnp.ndarray,
        t: float,  # pylint: disable=unused-argument
        control: jnp.ndarray
    ) -> jnp.ndarray:
        """Computes the instantaneous time derivative of the state vector for jax odeint"""
        return self._docking_f_x(state) + self._docking_g_x(state) @ control


class Integrator1dDockingExplicitSwitchingRTA(ExplicitSimplexModule, Integrator1dDockingRTAMixin):
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
        Integration method to use, either 'RK45_JAX', 'RK45', or 'Euler'
    """

    def __init__(
        self,
        *args,
        m: float = M_DEFAULT,
        control_bounds_high: Union[float, np.ndarray] = 1,
        control_bounds_low: Union[float, np.ndarray] = -1,
        backup_controller: RTABackupController = None,
        jit_compile_dict: Dict[str, bool] = None,
        integration_method: str = 'Euler',
        **kwargs
    ):
        self.m = m
        self.integration_method = integration_method

        if backup_controller is None:
            backup_controller = Integrator1dDockingBackupController()

        if jit_compile_dict is None:
            jit_compile_dict = {'constraint_violation': True}

        super().__init__(
            *args,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            backup_controller=backup_controller,
            jit_compile_dict=jit_compile_dict,
            **kwargs
        )

    def _setup_properties(self):
        self._setup_docking_properties(self.m, self.jit_compile_dict, self.integration_method)

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints_explicit()

    def _pred_state(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        return self._docking_pred_state(state, step_size, control, self.integration_method)


class Integrator1dDockingImplicitSwitchingRTA(ImplicitSimplexModule, Integrator1dDockingRTAMixin):
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
        Integration method to use, either 'RK45_JAX', 'RK45', or 'Euler'
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
        integration_method: str = 'Euler',
        **kwargs
    ):
        self.m = m
        self.integration_method = integration_method

        if backup_controller is None:
            backup_controller = Integrator1dDockingBackupController()

        if jit_compile_dict is None:
            jit_compile_dict = {'constraint_violation': True}

        super().__init__(
            *args,
            backup_window=backup_window,
            backup_controller=backup_controller,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            jit_compile_dict=jit_compile_dict,
            **kwargs
        )

    def _setup_properties(self):
        self._setup_docking_properties(self.m, self.jit_compile_dict, self.integration_method)

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints_implicit()

    def _pred_state(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        return self._docking_pred_state(state, step_size, control, self.integration_method)


class Integrator1dDockingExplicitOptimizationRTA(ExplicitASIFModule, Integrator1dDockingRTAMixin):
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
        **kwargs
    ):
        self.m = m
        if jit_compile_dict is None:
            jit_compile_dict = {'generate_barrier_constraint_mats': True}

        super().__init__(
            *args,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            control_dim=1,
            jit_compile_dict=jit_compile_dict,
            **kwargs
        )

    def _setup_properties(self):
        self._setup_docking_properties(self.m, self.jit_compile_dict, 'RK45_JAX')

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints_explicit()

    def state_transition_system(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._docking_f_x(state)

    def state_transition_input(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._docking_g_x(state)


class Integrator1dDockingImplicitOptimizationRTA(ImplicitASIFModule, Integrator1dDockingRTAMixin):
    """
    Implements Implicit Optimization RTA for the Integrator 1d Docking problem

    Utilizes Implicit Active Set Invariance Function algorithm

    Parameters
    ----------
    backup_window : float
        Duration of time in seconds to evaluate backup controller trajectory
    num_check_all : int
        Number of points at beginning of backup trajectory to check at every sequential simulation timestep.
        Should be <= backup_window.
        Defaults to 0 as skip_length defaults to 1 resulting in all backup trajectory points being checked.
    subsample_constraints_num_least: int
        subsample the backup trajectory down to the points with the N least constraint function outputs
        i.e. the n points closest to violating a safety constraint
        ....
    skip_length : int
        After num_check_all points in the backup trajectory are checked, the remainder of the backup window is filled by
        skipping every skip_length points to reduce the number of backup trajectory constraints. Will always check the
        last point in the backup trajectory as well.
        Defaults to 1, resulting in no skipping.
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
        Integration method to use, either 'RK45_JAX', 'RK45', or 'Euler'
    """

    def __init__(
        self,
        *args,
        backup_window: float = 2,
        num_check_all: int = 0,
        subsample_constraints_num_least: int = 1,
        skip_length: int = 1,
        m: float = M_DEFAULT,
        control_bounds_high: Union[float, np.ndarray] = 1,
        control_bounds_low: Union[float, np.ndarray] = -1,
        backup_controller: RTABackupController = None,
        jit_compile_dict: Dict[str, bool] = None,
        integration_method: str = 'Euler',
        **kwargs
    ):
        self.m = m
        self.integration_method = integration_method
        if backup_controller is None:
            backup_controller = Integrator1dDockingBackupController()

        if jit_compile_dict is None:
            jit_compile_dict = {'generate_ineq_constraint_mats': True}

        super().__init__(
            *args,
            backup_window=backup_window,
            integration_method=integration_method,
            num_check_all=num_check_all,
            subsample_constraints_num_least=subsample_constraints_num_least,
            skip_length=skip_length,
            backup_controller=backup_controller,
            control_bounds_high=control_bounds_high,
            control_bounds_low=control_bounds_low,
            control_dim=1,
            jit_compile_dict=jit_compile_dict,
            **kwargs
        )

    def _setup_properties(self):
        self._setup_docking_properties(self.m, self.jit_compile_dict, self.integration_method)

    def _setup_constraints(self) -> OrderedDict:
        return self._setup_docking_constraints_implicit()

    def state_transition_system(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._docking_f_x(state)

    def state_transition_input(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._docking_g_x(state)


class Integrator1dDockingBackupController(RTABackupController):
    """Max braking backup controller to bring velocity to zero for 1d Integrator
    """

    def _generate_control(
        self,
        state: jnp.ndarray,
        step_size: float,
        controller_state: Union[jnp.ndarray, Dict[str, jnp.ndarray], None] = None
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

    def __init__(self, alpha: ConstraintStrengthener = None):
        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 1, 0, 30])
        super().__init__(alpha=alpha)

    def _compute(self, state: jnp.ndarray) -> float:
        return -2 * state[0] - state[1]**2


class ConstraintIntegrator1dDockingCollisionImplicit(ConstraintModule):
    """Integrator 1d docking implicit chief collision avoidance constraint

    Parameters
    ----------
    alpha : ConstraintStrengthener
        Constraint Strengthener object used for ASIF methods. Required for ASIF methods.
        Defaults to PolynomialConstraintStrengthener([0, 10, 0, 30])
    """

    def __init__(self, alpha: ConstraintStrengthener = None):
        if alpha is None:
            alpha = PolynomialConstraintStrengthener([0, 10, 0, 30])
        super().__init__(alpha=alpha)

    def _compute(self, state: jnp.ndarray) -> float:
        return -state[0]
