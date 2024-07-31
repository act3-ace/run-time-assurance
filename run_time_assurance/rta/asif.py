"""
This module implements explicit and implicit Active Set Invariance Filter RTA Modules. These RTA modules generate safe
    control actions that are maximially close to the desired (in an L2 norm sense) such that the system does not leave
    the explicitly or implicitly defined safe set.
"""

from __future__ import annotations

import abc
import warnings
from collections import OrderedDict
from typing import Any, Dict, Union

import jax.numpy as jnp
import numpy as np
import quadprog
from jax import jacfwd, jit, lax, vmap
from safe_autonomy_simulation.dynamics import ControlAffineODEDynamics, ODEDynamics
from scipy.optimize import minimize

from run_time_assurance.constraint import ConstraintModule, DirectInequalityConstraint, DiscreteCBFConstraint
from run_time_assurance.controller import RTABackupController
from run_time_assurance.rta.base import BackupControlBasedRTA, ConstraintBasedRTA
from run_time_assurance.utils import SolverError, SolverWarning, to_jnp_array_jit


class BaseOptimizationModule(ConstraintBasedRTA):
    """
    Base class for optimization-based RTA

    Parameters
    ----------
    epsilon : float
        threshold distance between desired action and actual safe action at which the rta is said to be intervening
        default 1e-2
    control_dim : int
        length of control vector
    solver_exception : bool
        When the solver cannot find a solution, True for an exception and False for a warning
    """

    def __init__(self, *args: Any, epsilon: float = 1e-2, control_dim: int, solver_exception: bool = False, **kwargs: Any):
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

        self.control_dim = control_dim
        self.solver_exception = solver_exception

        self.constraints_with_slack = OrderedDict()
        for k, c in self.constraints.items():
            if c.slack_priority_scale is not None:
                self.constraints_with_slack[k] = c
        self.num_slack = len(self.constraints_with_slack)

        self.obj_weight = np.ones(self.control_dim + self.num_slack)
        for i, c in enumerate(self.constraints_with_slack.values()):
            self.obj_weight[i + self.control_dim] = np.sqrt(c.slack_priority_scale)
        self.obj_weight = np.diag(self.obj_weight)

    def monitor(self, desired_control: np.ndarray, actual_control: np.ndarray) -> bool:
        """Determines whether the ASIF RTA module is currently intervening

        Parameters
        ----------
        desired_control : np.ndarray
            desired control vector
        actual_control : np.ndarray
            actual control vector produced by ASIF optimization

        Returns
        -------
        bool
            True if rta module is interveining
        """
        return bool(np.linalg.norm(desired_control - actual_control) > self.epsilon)


class ASIFModule(BaseOptimizationModule):
    """
    Base class for Active Set Invariance Filter Optimization RTA

    Only supports dynamical systems with dynamics in the form of:
        dx/dt = f(x) + g(x)u
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.constraint_names = []
        self.ineq_weight_actuation, self.ineq_constant_actuation = self._generate_actuation_constraint_mats()

        self.direct_inequality_constraints = self._setup_direct_constraints()

        self.direct_inequality_params = dict.fromkeys(self.direct_inequality_constraints.keys())
        self.direct_inequality_enabled_dict = dict.fromkeys(self.direct_inequality_constraints.keys())
        self._update_direct_inequality_params()
        self.constraints_causing_intervention = None
        self.constraint_names += list(self.constraints.keys()) + list(self.direct_inequality_constraints.keys())

    def _update_direct_inequality_params(self):
        """Update the parameters for each constraint"""
        for k, c in self.direct_inequality_constraints.items():
            self.direct_inequality_params[k] = c.params
            self.direct_inequality_enabled_dict[k] = c.enabled

    def compose(self):
        """
        applies jax composition transformations (grad, jit, jacobian etc.)

        jit complilation is determined by the jit_compile_dict constructor parameter
        jit compilation settings:
            generate_barrier_constraint_mats:
                default True
        """
        super().compose()
        if self.jit_enable and self.jit_compile_dict.get('generate_barrier_constraint_mats', True):
            self._generate_barrier_constraint_mats_fn = jit(self._generate_barrier_constraint_mats, static_argnames=['step_size'])
        else:
            self._generate_barrier_constraint_mats_fn = self._generate_barrier_constraint_mats

        if self.jit_enable and self.jit_compile_dict.get('update_ineq_mats', True):
            self._update_ineq_mats_fn = jit(self._update_ineq_mats)
        else:
            self._update_ineq_mats_fn = self._update_ineq_mats

    def _filter_control(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        self._update_direct_inequality_params()
        ineq_weight, ineq_constant = self._generate_barrier_constraint_mats_fn(state, step_size, self.params, self.constraint_enabled_dict)
        ineq_weight, ineq_constant = self._update_ineq_mats_fn(
            ineq_weight, ineq_constant, state, self.direct_inequality_params, self.direct_inequality_enabled_dict
        )
        desired_control = np.array(control, dtype=np.float64)
        actual_control = self._optimize(self.obj_weight, desired_control, ineq_weight, ineq_constant)
        self.intervening = self.monitor(desired_control, actual_control)

        return to_jnp_array_jit(actual_control)

    def _generate_actuation_constraint_mats(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """generates matrices for quadratic program optimization inequality constraint matrices that impose actuator limits
        on optimized control vector

        Returns
        -------
        jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        """
        ineq_weight = jnp.empty((0, self.control_dim + self.num_slack))
        ineq_constant = jnp.empty(0)

        if self.control_bounds_low is not None:
            c, b = get_lower_bound_ineq_constraint_mats(self.control_bounds_low, self.control_dim)
            c = jnp.hstack((c, jnp.zeros((self.control_dim, self.num_slack))))
            ineq_weight = jnp.vstack((ineq_weight, c))
            ineq_constant = jnp.concatenate((ineq_constant, b))
            for i in range(self.control_dim):
                self.constraint_names.append('u' + str(i + 1) + '_min')

        if self.control_bounds_high is not None:
            c, b = get_lower_bound_ineq_constraint_mats(self.control_bounds_high, self.control_dim)
            c *= -1
            b *= -1
            c = jnp.hstack((c, jnp.zeros((self.control_dim, self.num_slack))))
            ineq_weight = jnp.vstack((ineq_weight, c))
            ineq_constant = jnp.concatenate((ineq_constant, b))
            for i in range(self.control_dim):
                self.constraint_names.append('u' + str(i + 1) + '_max')

        return ineq_weight, ineq_constant

    def _update_ineq_mats(
        self, ineq_weight: jnp.ndarray, ineq_constant: jnp.ndarray, state: jnp.ndarray, params: dict, constraint_enabled_dict: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Update inequality matrices before sending to solver: adds any direct inequality constraints

        Parameters
        ----------
        ineq_weight : jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        ineq_constant : jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        params : dict
            Parameters for each constraint
        constraint_enabled_dict : dict
            Enable flag for each constraint

        Returns
        -------
        jnp.ndarray
            updated ineq_weight
        jnp.ndarray
            updated ineq_constant
        """
        for k, c in self.direct_inequality_constraints.items():
            # Compute constraints
            c_params = params[k]
            temp1 = c.ineq_weight(state, c_params)
            temp1 = jnp.concatenate((temp1, jnp.zeros(self.num_slack)))
            temp2 = c.ineq_constant(state, c_params)

            # Return zero if constraint is disabled
            temp1, temp2 = lax.cond(
                constraint_enabled_dict[k],
                self.direct_constraint_enabled,
                self.direct_constraint_disabled,
                temp1,
                temp2,
            )

            # Append to full constraint mats
            ineq_weight = jnp.append(ineq_weight, temp1[:, None], axis=1)
            ineq_constant = jnp.append(ineq_constant, temp2)
        return ineq_weight, ineq_constant

    def direct_constraint_enabled(self, temp1: jnp.ndarray, temp2: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return same when constraint is enabled"""
        return temp1, temp2

    def direct_constraint_disabled(self, temp1: jnp.ndarray, temp2: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return zeros when constraint is disabled"""
        return jnp.zeros(temp1.shape), jnp.zeros(temp2.shape)

    def _optimize(
        self, obj_weight: np.ndarray, obj_constant: np.ndarray, ineq_weight: jnp.ndarray, ineq_constant: jnp.ndarray
    ) -> np.ndarray:
        """Solve ASIF optimization problem via quadratic program

        Parameters
        ----------
        obj_weight : np.ndarray
            matix G of quadprog objective 1/2 x^T G x - a^T x
        obj_constant : np.ndarray
            vector a of quadprog objective 1/2 x^T G x - a^T x
        ineq_weight : jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        ineq_constant : jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b

        Returns
        -------
        np.ndarray
            Actual control solved by QP
        """
        obj_constant = np.concatenate((obj_constant, np.zeros(self.num_slack)))
        try:
            opt = quadprog.solve_qp(
                obj_weight, obj_constant, np.array(ineq_weight, dtype=np.float64), np.array(ineq_constant, dtype=np.float64), 0
            )
            u = opt[0]

            # Get constraints causing intervention
            self.constraints_causing_intervention = []
            for i in opt[5]:
                key = self.constraint_names[i - 1]
                if key not in self.constraints_causing_intervention:
                    self.constraints_causing_intervention.append(key)

        except ValueError as e:
            if e.args[0] == "constraints are inconsistent, no solution":
                if not self.solver_exception:
                    warnings.warn(SolverWarning())
                    u = obj_constant
                else:
                    raise SolverError() from e
            else:
                raise e
        return u[0:self.control_dim]

    @abc.abstractmethod
    def _generate_barrier_constraint_mats(self, state: jnp.ndarray, step_size: float, params: dict,
                                          constraint_enabled_dict: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        """generates matrices for quadratic program optimization inequality constraint matrices corresponding to safety
        barrier constraints

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        step_size : float
            duration of control step
        params : dict
            Parameters for each constraint
        constraint_enabled_dict : dict
            Enable flag for each constraint

        Returns
        -------
        jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def state_transition_system(self, state: jnp.ndarray) -> jnp.ndarray:
        """Computes the system state contribution to the system state's time derivative

        i.e. implements f(x) from dx/dt = f(x) + g(x)u

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system

        Returns
        -------
        jnp.ndarray
            state time derivative contribution from the current system state
        """
        raise NotImplementedError

    @abc.abstractmethod
    def state_transition_input(self, state: jnp.ndarray) -> jnp.ndarray:
        """Computes the control input matrix contribution to the system state's time derivative

        i.e. implements g(x) from dx/dt = f(x) + g(x)u

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system

        Returns
        -------
        jnp.ndarray
            input matrix in state space representation time derivative
        """
        raise NotImplementedError

    def _setup_direct_constraints(self) -> OrderedDict[str, DirectInequalityConstraint]:
        """Initializes and returns direct RTA constraints, to be used in the QP for ASIF.
        By default returns empty dict.

        Returns
        -------
        OrderedDict
            OrderedDict of direct rta contraints with name string keys and DirectInequalityConstraint object values
        """
        return OrderedDict()


class ExplicitASIFModule(ASIFModule):
    """
    Base class implementation of Explicit ASIF RTA

    Only supports dynamical systems with dynamics in the form of:
        dx/dt = f(x) + g(x)u

    Only supports constraints with relative degree difference of 1 between constraint jacobian and
        control input matrix g(x).

    Parameters
    ----------
    epislon : float
        threshold distance between desired action and actual safe action at which the rta is said to be intervening
        default 1e-2
    control_dim : int
        length of control vector
    """

    def _generate_barrier_constraint_mats(self, state: jnp.ndarray, step_size: float, params: dict,
                                          constraint_enabled_dict: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        """generates matrices for quadratic program optimization inequality constraint matrices corresponding to safety
        barrier constraints

        Applies Nagumo's condition to safety constraints

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        step_size : float
            duration of control step
        params : dict
            Parameters for each constraint
        constraint_enabled_dict : dict
            Enable flag for each constraint

        Returns
        -------
        jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        """
        ineq_weight_barrier = jnp.empty((0, self.control_dim + self.num_slack))
        ineq_constant_barrier = jnp.empty(0)

        slack_counter = 0
        for k, c in self.constraints.items():
            # Compute BC
            c_params = params[k]
            grad_x = c.grad(state, c_params)
            temp1 = grad_x @ self.state_transition_input(state)

            # Append slack variables
            slack = jnp.zeros(self.num_slack)
            if c.slack_priority_scale is not None:
                slack = slack.at[slack_counter].set(-1.)
                slack_counter += 1
            temp1 = jnp.concatenate((temp1, slack))

            # Get positive or negative alpha function depending on constraint value
            constraint_val = c(state, c_params)
            alpha = lax.cond(
                constraint_val < 0,
                self.negative_constraint,
                self.positive_constraint,
                c.alpha(constraint_val),
                c.alpha_negative(constraint_val) + c.bias_negative
            )

            # Compute BC
            temp2 = -grad_x @ self.state_transition_system(state) - alpha

            # Return zero if constraint is disabled
            temp1, temp2 = lax.cond(
                constraint_enabled_dict[k],
                self.constraint_enabled,
                self.constraint_disabled,
                temp1,
                temp2,
            )

            # Append to full BC mats
            ineq_weight_barrier = jnp.append(ineq_weight_barrier, temp1[None, :], axis=0)
            ineq_constant_barrier = jnp.append(ineq_constant_barrier, temp2)

        # Append actuation mats
        ineq_weight = jnp.concatenate((self.ineq_weight_actuation, ineq_weight_barrier))
        ineq_constant = jnp.concatenate((self.ineq_constant_actuation, ineq_constant_barrier))

        return ineq_weight.transpose(), ineq_constant

    def positive_constraint(self, val: float, _) -> float:
        """Return c.alpha(constraint_val) when constraint is positive"""
        return val

    def negative_constraint(self, _, val: float) -> float:
        """Return c.alpha_negative(constraint_val) + c.bias_negative when constraint is negative"""
        return val

    def constraint_enabled(self, temp1: jnp.ndarray, temp2: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return BC when constraint is enabled"""
        return temp1, temp2

    def constraint_disabled(self, temp1: jnp.ndarray, temp2: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return zeros when constraint is disabled"""
        return jnp.zeros(temp1.shape), jnp.zeros(temp2.shape)


class ImplicitASIFModule(ASIFModule, BackupControlBasedRTA):
    """
    Base class implementation of implicit ASIF RTA

    Requires a backup controller that provides a jacobian of output wrt state vector

    Only supports dynamical systems with dynamics in the form of:
        dx/dt = f(x) + g(x)u

    Parameters
    ----------
    backup_window : float
        Duration of time in seconds to evaluate backup controller trajectory.
    backup_controller : RTABackupController
        backup controller object utilized by rta module to generate backup control
    integration_method: str, optional
        Integration method to use, either 'RK45', or 'Euler'
    """

    def __init__(
        self,
        *args: Any,
        backup_window: float,
        backup_controller: RTABackupController,
        integration_method: str = 'RK45',
        **kwargs: Any,
    ):
        self.backup_window = backup_window
        self.integration_method = integration_method

        self.ode_dynamics = ASIFODESolver(integration_method, self.state_transition_system, self.state_transition_input)

        super().__init__(*args, backup_controller=backup_controller, **kwargs)

    def reset(self):
        """Resets the rta module to the initial state at the beginning of an episode
        Also calls reset on the backup controller
        """
        super().reset()
        self.reset_backup_controller()

    def compose(self):
        """
        applies jax composition transformations (grad, jit, jacobian etc.)

        jit complilation is determined by the jit_compile_dict constructor parameter
        jit compilation settings:
            generate_ineq_constraint_mats:
                default True
            pred_state:
                default False
            integrate:
                default False
        """
        if self.jit_enable and self.jit_compile_dict.get('jacobian', True):
            self._jacobian = jit(jacfwd(self._backup_state_transition), static_argnums=[1], static_argnames=['step_size'])
        else:
            self._jacobian = jacfwd(self._backup_state_transition)

        self.jit_compile_dict.setdefault('generate_barrier_constraint_mats', False)

        if self.jit_enable and self.jit_compile_dict.get('generate_ineq_constraint_mats', True):
            self._generate_ineq_constraint_mats_fn = jit(self._generate_ineq_constraint_mats, static_argnames=['num_steps'])
        else:
            self._generate_ineq_constraint_mats_fn = self._generate_ineq_constraint_mats

        default_int = True

        if self.jit_enable and self.jit_compile_dict.get('pred_state', default_int):
            self._pred_state_fn = jit(self._pred_state, static_argnames=['step_size'])
        else:
            self._pred_state_fn = self._pred_state

        if self.jit_enable and self.jit_compile_dict.get('integrate', default_int):
            self._integrate_fn = jit(self.integrate, static_argnames=['step_size', 'Nsteps'])
        else:
            self._integrate_fn = self.integrate

        if self.jit_enable:
            self._get_ineq_mats_fn = self._get_constraint_ineq_mats
        else:
            self._get_ineq_mats_fn = self._get_constraint_ineq_mats_no_vmap

        super().compose()

    def jacobian(self, state: jnp.ndarray, step_size: float, controller_state: Union[jnp.ndarray, Dict[str, jnp.ndarray]] = None):
        """Computes Jacobian of system state transition J(f(x) + g(x,u)) wrt x

        Parameters
        ----------
        state : jnp.ndarray
            Current jnp.ndarray of the system at which to evaluate Jacobian
        step_size : float
            simulation integration step size
        controller_state: jnp.ndarray or Dict[str, jnp.ndarray] or None
            internal controller state. For stateful controllers, all states that are modified in the control computation
                (e.g. integral control error buffers) must be contained within controller_state

        Returns
        -------
        jnp.ndarray
            Jacobian matrix of state transition
        """
        return self._jacobian(state, step_size, controller_state)

    def _backup_state_transition(
        self, state: jnp.ndarray, step_size: float, controller_state: Union[jnp.ndarray, Dict[str, jnp.ndarray]] = None
    ):
        return self.state_transition_system(state) + self.state_transition_input(state) @ (
            self.backup_controller.generate_control_with_controller_state(state, step_size, controller_state)[0]
        )

    def _generate_barrier_constraint_mats(self, state: jnp.ndarray, step_size: float, params: dict,
                                          constraint_enabled_dict: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        """generates matrices for quadratic program optimization inequality constraint matrices corresponding to safety
        barrier constraints

        Computes backup trajectory with backup controller and applies Nagumo's condition on the safety constraints at
        points along backup trajectory.

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        step_size : float
            duration of control step
        params : dict
            Parameters for each constraint
        constraint_enabled_dict : dict
            Enable flag for each constraint

        Returns
        -------
        jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        """
        num_steps = int(self.backup_window / step_size) + 1
        traj_states, traj_sensitivities = self._integrate_fn(state, step_size, num_steps)
        ineq_weight, ineq_constant = self._generate_ineq_constraint_mats_fn(
            state, num_steps, traj_states, traj_sensitivities, params, constraint_enabled_dict
        )

        # Fix constraint names:
        idx = 0
        if self.control_bounds_low is not None:
            idx += self.control_dim
        if self.control_bounds_high is not None:
            idx += self.control_dim
        self.constraint_names = self.constraint_names[0:idx]
        for k in self.constraints.keys():
            self.constraint_names += [k] * num_steps
        self.constraint_names += list(self.direct_inequality_constraints.keys())

        return ineq_weight, ineq_constant

    def _generate_ineq_constraint_mats(
        self,
        state: jnp.ndarray,
        num_steps: int,
        traj_states: jnp.ndarray,
        traj_sensitivities: jnp.ndarray,
        params: dict,
        constraint_enabled_dict: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """generates quadratic program optimization inequality constraint matrices corresponding to safety

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        num_steps : int
            number of trajectory steps
        traj_states : jnp.ndarray
            list of rta states from along the trajectory
        traj_sensitivities: jnp.ndarray
            list of trajectory state sensitivities (i.e. jacobian wrt initial trajectory state).
            Elements are jnp.ndarrays with size (n, n) where n = state.size
        params : dict
            Parameters for each constraint
        constraint_enabled_dict : dict
            Enable flag for each constraint

        Returns
        -------
        jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        """

        # Initialize
        num_constraints = len(self.constraints)
        num_points = len(traj_states)

        ineq_weight_barrier = jnp.empty((num_constraints * num_points, self.control_dim + self.num_slack))
        ineq_constant_barrier = jnp.empty(num_constraints * num_points)
        constraint_vals = jnp.empty((num_constraints, num_points))

        traj_states = jnp.array(traj_states)
        traj_sensitivities = jnp.array(traj_sensitivities)

        slack_counter = 0
        # for i in range(num_constraints):
        for i, k in enumerate(self.constraints.keys()):
            # Get BCs for entire trajectory
            point_ineq_weight, point_ineq_constant, point_constraint_vals = self._get_ineq_mats_fn(
                self.constraints[k], state, traj_states, traj_sensitivities, params[k])

            # Return zeros if constraint is disabled
            point_ineq_weight, point_ineq_constant = lax.cond(
                constraint_enabled_dict[k],
                self.constraint_enabled,
                self.constraint_disabled,
                point_ineq_weight, point_ineq_constant
            )

            # Append slack variables
            slack = jnp.zeros((num_points, self.num_slack))
            if self.constraints[k].slack_priority_scale is not None:
                slack = slack.at[:, slack_counter].set(-1.)
                slack_counter += 1
            point_ineq_weight = jnp.hstack((point_ineq_weight, slack))

            # Add to full BC mats
            ineq_weight_barrier = ineq_weight_barrier.at[i * num_points:(i + 1) * num_points, :].set(point_ineq_weight)
            ineq_constant_barrier = ineq_constant_barrier.at[i * num_points:(i + 1) * num_points].set(point_ineq_constant)
            constraint_vals = constraint_vals.at[i, :].set(point_constraint_vals)

        # Subsample
        ineq_weight_barrier, ineq_constant_barrier = self.subsample_constraints(
            ineq_weight_barrier, ineq_constant_barrier, constraint_vals, num_steps
        )

        # Append actuation constraint mats
        ineq_weight = jnp.concatenate((self.ineq_weight_actuation, ineq_weight_barrier))
        ineq_constant = jnp.concatenate((self.ineq_constant_actuation, ineq_constant_barrier))
        return ineq_weight.transpose(), ineq_constant

    def constraint_enabled(self, point_ineq_weight: jnp.ndarray, point_ineq_constant: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return BC when constraint is enabled"""
        return point_ineq_weight, point_ineq_constant

    def constraint_disabled(self, point_ineq_weight: jnp.ndarray, point_ineq_constant: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return zeros when constraint is disabled"""
        return jnp.zeros(point_ineq_weight.shape), jnp.zeros(point_ineq_constant.shape)

    def subsample_constraints(self, ineq_weight_barrier, ineq_constant_barrier, constraint_vals, num_steps):
        """Subsample constraints individually based on the constraint's subsample_config attribute

        Parameters
        ----------
        ineq_weight_barrier : jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        ineq_constant_barrier : jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        constraint_vals : jnp.ndarray
            constraint values along trajectory
        num_steps: int
            number of trajectory steps

        Returns
        -------
        jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        """
        new_ineq_weight_barrier = jnp.empty((0, self.control_dim))
        new_ineq_constant_barrier = jnp.empty(0)

        for i, c in enumerate(self.constraints.values()):
            # Get barrier constraints for each constraint
            tmp_ineq_weight_barrier = ineq_weight_barrier[num_steps * i:num_steps * (i + 1), :]
            tmp_ineq_constant_barrier = ineq_constant_barrier[num_steps * i:num_steps * (i + 1)]
            tmp_constraint_vals = constraint_vals[i]

            # Get points to check from num_check_all and skip_length
            check_points = jnp.hstack(
                (
                    jnp.array(range(0, c.subsample_config.num_check_all)),
                    jnp.array(range(c.subsample_config.num_check_all + 1, num_steps, c.subsample_config.skip_length))
                )
            ).astype(int)
            tmp_ineq_weight_barrier = tmp_ineq_weight_barrier[check_points, :]
            tmp_ineq_constant_barrier = tmp_ineq_constant_barrier[check_points]
            tmp_constraint_vals = tmp_constraint_vals[check_points]

            # Get points to check from subsample_constraints_num_least
            if c.subsample_config.subsample_constraints_num_least is not None:
                constraint_sorted_idxs = jnp.argsort(tmp_constraint_vals)
                use_idxs = constraint_sorted_idxs[0:c.subsample_config.subsample_constraints_num_least]
                tmp_ineq_weight_barrier = tmp_ineq_weight_barrier[use_idxs, :]
                tmp_ineq_constant_barrier = tmp_ineq_constant_barrier[use_idxs]

            # Keep first and/or last point
            if c.subsample_config.keep_first:
                tmp_ineq_weight_barrier = jnp.concatenate((tmp_ineq_weight_barrier, ineq_weight_barrier[num_steps * i, :][None]), axis=0)
                tmp_ineq_constant_barrier = jnp.concatenate((tmp_ineq_constant_barrier, ineq_constant_barrier[num_steps * i][None]), axis=0)
            if c.subsample_config.keep_last:
                tmp_ineq_weight_barrier = jnp.concatenate(
                    (tmp_ineq_weight_barrier, ineq_weight_barrier[num_steps * (i + 1) - 1, :][None]), axis=0
                )
                tmp_ineq_constant_barrier = jnp.concatenate(
                    (tmp_ineq_constant_barrier, ineq_constant_barrier[num_steps * (i + 1) - 1][None]), axis=0
                )

            new_ineq_weight_barrier = jnp.concatenate((new_ineq_weight_barrier, tmp_ineq_weight_barrier), axis=0)
            new_ineq_constant_barrier = jnp.concatenate((new_ineq_constant_barrier, tmp_ineq_constant_barrier), axis=0)

        return new_ineq_weight_barrier, new_ineq_constant_barrier

    def _get_constraint_ineq_mats(
        self, constraint: ConstraintModule, state: jnp.ndarray, traj_states: jnp.ndarray, traj_sensitivities: jnp.ndarray, params: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray, Union[jnp.ndarray, float]]:
        """Computes inequality constraint matrices for a given constraint using vmap

        Parameters
        ----------
        constraint : ConstraintModule
            constraint to create cbf for
        initial_state : jnp.ndarray
            initial state of the backup trajectory
        traj_state : list
            array of trajectory states
        traj_sensitivity : list
            array of trajectory sensitivities
        params : dict
            dict of parameters for the constraint

        Returns
        -------
        jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            constraint values along trajectory
        """
        constraint_vmapped = vmap(self.invariance_constraints, (None, None, 0, 0, None), (0, 0, 0))
        point_ineq_weight, point_ineq_constant, point_constraint_vals = constraint_vmapped(
            constraint, state, traj_states, traj_sensitivities, params
        )
        return point_ineq_weight, point_ineq_constant, point_constraint_vals

    def _get_constraint_ineq_mats_no_vmap(
        self, constraint: ConstraintModule, state: jnp.ndarray, traj_states: jnp.ndarray, traj_sensitivities: jnp.ndarray, params: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Computes inequality constraint matrices for a given constraint without vmap, allowing easier debugging

        Parameters
        ----------
        constraint : ConstraintModule
            constraint to create cbf for
        initial_state : jnp.ndarray
            initial state of the backup trajectory
        traj_state : list
            array of trajectory states
        traj_sensitivity : list
            array of trajectory sensitivities
        params : dict
            dict of parameters for the constraint

        Returns
        -------
        jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            constraint values along trajectory
        """
        point_ineq_weight = []
        point_ineq_constant = []
        point_constraint_vals = []
        for (traj_state, traj_sensitivity) in zip(traj_states, traj_sensitivities):
            w, c, v = self.invariance_constraints(constraint, state, traj_state, traj_sensitivity, params)
            point_ineq_weight.append(w)
            point_ineq_constant.append(c)
            point_constraint_vals.append(v)
        return jnp.array(point_ineq_weight), jnp.array(point_ineq_constant), jnp.array(point_constraint_vals)

    def invariance_constraints(
        self,
        constraint: ConstraintModule,
        initial_state: jnp.ndarray,
        traj_state: jnp.ndarray,
        traj_sensitivity: jnp.ndarray,
        params: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray, float]:
        """Computes safety constraint invariance constraints via Nagumo's condition for a point in the backup trajectory

        Parameters
        ----------
        constraint : ConstraintModule
            constraint to create cbf for
        initial_state : jnp.ndarray
            initial state of the backup trajectory
        traj_state : list
            arbitrary state in the backup trajectory
        traj_sensitivity : list
            backup trajectory state sensitivity (i.e. jacobian relative to the initial state)
        params : dict
            dict of parameters for the constraint

        Returns
        -------
        jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        float
            constraint value at trajectory state
        """
        traj_state_array = jnp.array(traj_state)
        traj_sensitivity_array = jnp.array(traj_sensitivity)

        f_x0 = self.state_transition_system(initial_state)
        g_x0 = self.state_transition_input(initial_state)

        grad_x = constraint.grad(traj_state_array, params)
        ineq_weight = grad_x @ (traj_sensitivity_array @ g_x0)

        ineq_constant = grad_x @ (traj_sensitivity_array @ f_x0) \
            + constraint.alpha(constraint(traj_state_array, params))

        return ineq_weight, -ineq_constant, constraint(traj_state_array, params)

    def integrate(self, state: jnp.ndarray, step_size: float, Nsteps: int) -> tuple[list, list]:
        """Estimate backup trajectory by polling backup controller backup control and integrating system dynamics

        Parameters
        ----------
        state : jnp.ndarray
            initial rta state of the system
        step_size : float
            simulation integration step size
        Nsteps : int
            number of simulation integration steps

        Returns
        -------
        list
            list of rta states from along the trajectory
        list
            list of trajectory state sensitivities (i.e. jacobian wrt initial trajectory state)
        """
        sensitivity = jnp.eye(state.size)

        traj_states = [state.copy()]
        traj_sensitivity = [sensitivity]

        self.backup_controller_save()

        for _ in range(1, Nsteps):
            control = self.backup_control(state, step_size)
            state = self._pred_state_fn(state, step_size, control)
            traj_jac = self.jacobian(state, step_size, self.backup_controller.controller_state)
            sensitivity = sensitivity + (traj_jac @ sensitivity) * step_size

            traj_states.append(state)
            traj_sensitivity.append(sensitivity)

        self.backup_controller_restore()

        return traj_states, traj_sensitivity

    def _pred_state(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        out, _ = self.ode_dynamics.step(step_size, state, control)
        return out


def get_lower_bound_ineq_constraint_mats(bound: Union[int, float, np.ndarray, jnp.ndarray],
                                         vec_len: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes inequality constraint matrices for applying a lower bound to optimization var in quadprog

    Parameters
    ----------
    bound : Union[jnp.ndarray, int, float]
        Lower bound for optimization variable.
        If jnp.ndarray, must be same length as optimization variable. Will be applied elementwise.
        If number, will be applied to all elements.
    vec_len : int
        optimization variable vector length

    Returns
    -------
    jnp.ndarray
        matix C.T of quadprog inequality constraint C.T x >= b
    jnp.ndarray
        vector b of quadprog inequality constraint C.T x >= b
    """
    c = jnp.eye(vec_len)

    if isinstance(bound, jnp.ndarray):
        assert bound.shape == (vec_len, ), f"the shape of bound must be ({vec_len},)"
        b = jnp.copy(bound)
    else:
        b = bound * jnp.ones(vec_len)

    return c, b


class ASIFODESolver(ControlAffineODEDynamics):
    """Control Affine ODE solver for ASIF"""

    def __init__(self, integration_method, state_transition_system, state_transition_input, **kwargs):
        self.asif_state_transition_system = state_transition_system
        self.asif_state_transition_input = state_transition_input
        super().__init__(integration_method=integration_method, **kwargs)

    def state_transition_system(self, state):
        return self.asif_state_transition_system(state)

    def state_transition_input(self, state):
        return self.asif_state_transition_input(state)


class DiscreteASIFModule(BaseOptimizationModule):
    """
    Base class for Active Set Invariance Filter Optimization RTA using Discrete Control Barrier Functions (CBFs)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.opt_bounds = [(self.control_bounds_low, self.control_bounds_high)] * self.control_dim + [(-np.inf, np.inf)] * self.num_slack
        self.scipy_constraints = []
        self.discrete_constraints = OrderedDict()
        slack_counter = 0
        for k, c in self.constraints.items():
            if c.slack_priority_scale is not None:
                slack_idx = slack_counter
                slack_counter += 1
            else:
                slack_idx = None
            temp_c = DiscreteCBFConstraint(c, self.next_state_differentiable, self.control_dim, slack_idx)
            self.discrete_constraints[k] = temp_c
            self.scipy_constraints.append({'type': 'ineq', 'fun': temp_c.cbf, 'jac': temp_c.jac})

    def _filter_control(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        desired_control = np.concatenate((np.array(control, dtype=np.float64), np.zeros((self.num_slack))))
        enabled_constraints = []
        for c, params, enabled in zip(self.scipy_constraints, list(self.params.values()), list(self.constraint_enabled_dict.values())):
            if enabled:
                c['args'] = (state, step_size, params)
                enabled_constraints.append(c)
        res = minimize(
            self.obj, desired_control, method='SLSQP', constraints=enabled_constraints, bounds=self.opt_bounds, args=(desired_control)
        )
        if not res.success:
            error = f'Solver Failure: {res.message}'
            if not self.solver_exception:
                warnings.warn(error)
            else:
                raise SolverError(error)

        actual_control = res.x[0:self.control_dim]

        self.intervening = self.monitor(desired_control[0:self.control_dim], actual_control)

        return to_jnp_array_jit(actual_control)

    def obj(self, control: np.ndarray, desired_control: np.ndarray) -> float:
        """Objective function of the solver (quadratic loss function).

        Parameters
        ----------
        control: np.ndarray
            Control vector of the system, solved for by optimizer
        desired_control: np.ndarray
            Desired control vector of the system

        Returns
        -------
        float:
            Value of objective function
        """
        return 0.5 * control.T @ (self.obj_weight.T @ self.obj_weight) @ control + (-self.obj_weight.T @ desired_control).T @ control

    @abc.abstractmethod
    def next_state_differentiable(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        """Function to compute the next state of the system. Must be differentiable using Jax.

        Parameters
        ----------
        state: jnp.ndarray
            Current rta state of the system
        step_size: float
            Time duration over which filtered control will be applied to actuators
        control: jnp.ndarray
            Control vector of the system

        Returns
        -------
        jnp.ndarray:
            Next state of the system
        """
        raise NotImplementedError


class DiscreteASIFIntegratorModule(DiscreteASIFModule):
    """
    Active Set Invariance Filter Optimization RTA using Discrete Control Barrier Functions (CBFs).
    Uses Differentiable Jax ODE solver, where user only needs to define the state derivative.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.ode_solver = DifferentiableODESolver(self.compute_state_dot)

    def next_state_differentiable(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        """Function to compute the next state of the system. Must be differentiable using Jax.
        By default, uses the Jax ODE solver given the state derivative.
        Can be overwritten by known system dynamics to decrease computation time.

        Parameters
        ----------
        state: jnp.ndarray
            Current rta state of the system
        step_size: float
            Time duration over which filtered control will be applied to actuators
        control: jnp.ndarray
            Control vector of the system

        Returns
        -------
        jnp.ndarray:
            Next state of the system
        """
        next_state, _ = self.ode_solver.step(step_size, state, control)
        return next_state

    @abc.abstractmethod
    def compute_state_dot(self, t: float, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        """Function to compute the state derivative of the system. Must be differentiable using Jax.

        Parameters
        ----------
        t: float
            Current time of the system
        state: jnp.ndarray
            Current rta state of the system
        control: jnp.ndarray
            Control vector of the system

        Returns
        -------
        jnp.ndarray:
            State derivative
        """
        raise NotImplementedError


class DifferentiableODESolver(ODEDynamics):
    """Differentiable ODE solver for Discrete ASIF Module"""

    def __init__(self, state_dot_fn, **kwargs):
        self.state_dot_fn = state_dot_fn
        super().__init__(integration_method='RK45', **kwargs)

    def _compute_state_dot(self, t, state, control):
        return self.state_dot_fn(t, state, control)
