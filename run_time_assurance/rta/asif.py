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
import scipy
from jax import jacfwd, jit, vmap
from jax.experimental.ode import odeint

from run_time_assurance.constraint import ConstraintModule, DirectInequalityConstraint
from run_time_assurance.controller import RTABackupController
from run_time_assurance.rta.base import BackupControlBasedRTA, ConstraintBasedRTA
from run_time_assurance.utils import SolverError, SolverWarning, to_jnp_array_jit


class ASIFModule(ConstraintBasedRTA):
    """
    Base class for Active Set Invariance Filter Optimization RTA

    Only supports dynamical systems with dynamics in the form of:
        dx/dt = f(x) + g(x)u

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
        self.obj_weight = np.eye(self.control_dim)
        self.ineq_weight_actuation, self.ineq_constant_actuation = self._generate_actuation_constraint_mats()

        self.direct_inequality_constraints = OrderedDict()
        for k, c in self.constraints.items():
            if isinstance(c, DirectInequalityConstraint):
                self.direct_inequality_constraints[k] = c
                self.constraints.pop(k)

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
        ineq_weight, ineq_constant = self._generate_barrier_constraint_mats_fn(state, step_size)
        ineq_weight, ineq_constant = self._update_ineq_mats_fn(ineq_weight, ineq_constant, state)
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
        ineq_weight = jnp.empty((0, self.control_dim))
        ineq_constant = jnp.empty(0)

        if self.control_bounds_low is not None:
            c, b = get_lower_bound_ineq_constraint_mats(self.control_bounds_low, self.control_dim)
            ineq_weight = jnp.vstack((ineq_weight, c))
            ineq_constant = jnp.concatenate((ineq_constant, b))

        if self.control_bounds_high is not None:
            c, b = get_lower_bound_ineq_constraint_mats(self.control_bounds_high, self.control_dim)
            c *= -1
            b *= -1
            ineq_weight = jnp.vstack((ineq_weight, c))
            ineq_constant = jnp.concatenate((ineq_constant, b))

        return ineq_weight, ineq_constant

    def _update_ineq_mats(self, ineq_weight: jnp.ndarray, ineq_constant: jnp.ndarray,
                          state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Update inequality matrices before sending to solver: adds any direct inequality constraints

        Parameters
        ----------
        ineq_weight : jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        ineq_constant : jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b

        Returns
        -------
        jnp.ndarray
            updated ineq_weight
        jnp.ndarray
            updated ineq_constant
        """
        for c in self.direct_inequality_constraints.values():
            temp1 = c.ineq_weight(state)
            temp2 = c.ineq_constant(state)
            ineq_weight = jnp.append(ineq_weight, temp1[:, None], axis=1)
            ineq_constant = jnp.append(ineq_constant, temp2)
        return ineq_weight, ineq_constant

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
        try:
            opt = quadprog.solve_qp(
                obj_weight, obj_constant, np.array(ineq_weight, dtype=np.float64), np.array(ineq_constant, dtype=np.float64), 0
            )[0]
        except ValueError as e:
            if e.args[0] == "constraints are inconsistent, no solution":
                if not self.solver_exception:
                    warnings.warn(SolverWarning())
                    opt = obj_constant
                else:
                    raise SolverError() from e
            else:
                raise e
        return opt

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

    @abc.abstractmethod
    def _generate_barrier_constraint_mats(self, state: jnp.ndarray, step_size: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """generates matrices for quadratic program optimization inequality constraint matrices corresponding to safety
        barrier constraints

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        step_size : float
            duration of control step

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

    def _generate_barrier_constraint_mats(self, state: jnp.ndarray, step_size: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """generates matrices for quadratic program optimization inequality constraint matrices corresponding to safety
        barrier constraints

        Applies Nagumo's condition to safety constraints

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        step_size : float
            duration of control step

        Returns
        -------
        jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        """
        ineq_weight_barrier = jnp.empty((0, self.control_dim))
        ineq_constant_barrier = jnp.empty(0)

        for c in self.constraints.values():
            grad_x = c.grad(state)
            temp1 = grad_x @ self.state_transition_input(state)
            temp2 = -grad_x @ self.state_transition_system(state) - c.alpha(c(state))

            ineq_weight_barrier = jnp.append(ineq_weight_barrier, temp1[None, :], axis=0)
            ineq_constant_barrier = jnp.append(ineq_constant_barrier, temp2)

        ineq_weight = jnp.concatenate((self.ineq_weight_actuation, ineq_weight_barrier))
        ineq_constant = jnp.concatenate((self.ineq_constant_actuation, ineq_constant_barrier))

        return ineq_weight.transpose(), ineq_constant


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
    num_check_all : int
        Number of points at beginning of backup trajectory to check at every sequential simulation timestep.
        Should be <= backup_window.
        Defaults to 0 as skip_length defaults to 1 resulting in all backup trajectory points being checked.
    skip_length : int
        After num_check_all points in the backup trajectory are checked, the remainder of the backup window is filled by
        skipping every skip_length points to reduce the number of backup trajectory constraints.
        Defaults to 1, resulting in no skipping.
    subsample_constraints_num_least : int
        subsample the backup trajectory down to the points with the N least constraint function outputs
            i.e. the n points closest to violating a safety constraint
    backup_controller : RTABackupController
        backup controller object utilized by rta module to generate backup control
    integration_method: str, optional
        Integration method to use, either 'RK45_JAX', 'RK45', or 'Euler'
    """

    def __init__(
        self,
        *args: Any,
        backup_window: float,
        num_check_all: int = 0,
        skip_length: int = 1,
        subsample_constraints_num_least: int = None,
        backup_controller: RTABackupController,
        integration_method: str = 'RK45_JAX',
        **kwargs: Any,
    ):
        self.backup_window = backup_window
        self.num_check_all = num_check_all
        self.skip_length = skip_length
        self.integration_method = integration_method

        assert (subsample_constraints_num_least is None) or \
               (isinstance(subsample_constraints_num_least, int) and subsample_constraints_num_least > 0), \
               "subsample_constraints_num_least must be a positive integer or None"
        self.subsample_constraints_num_least = subsample_constraints_num_least
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

        if self.integration_method in ('Euler', 'RK45_JAX'):
            default_int = True
        elif self.integration_method == 'RK45':
            default_int = False
        else:
            raise ValueError('integration_method must be either RK45_JAX, RK45, or Euler')

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

    def _generate_barrier_constraint_mats(self, state: jnp.ndarray, step_size: float) -> tuple[jnp.ndarray, jnp.ndarray]:
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

        Returns
        -------
        jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        """
        num_steps = int(self.backup_window / step_size) + 1
        traj_states, traj_sensitivities = self._integrate_fn(state, step_size, num_steps)
        ineq_weight, ineq_constant = self._generate_ineq_constraint_mats_fn(state, num_steps, traj_states, traj_sensitivities)

        return ineq_weight, ineq_constant

    def _generate_ineq_constraint_mats(self, state: jnp.ndarray, num_steps: int, traj_states: jnp.ndarray,
                                       traj_sensitivities: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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

        Returns
        -------
        jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        """
        constraint_list = list(self.constraints.values())
        num_constraints = len(self.constraints)

        check_points = jnp.hstack(
            (jnp.array(range(0, self.num_check_all)), jnp.array(range(self.num_check_all + 1, num_steps, self.skip_length)))
        ).astype(int)
        num_points = len(check_points)

        ineq_weight_barrier = jnp.empty((num_constraints * num_points, self.control_dim))
        ineq_constant_barrier = jnp.empty(num_constraints * num_points)
        constraint_vals = jnp.empty((num_constraints, num_points))

        traj_states = jnp.array(traj_states)[check_points, :]
        traj_sensitivities = jnp.array(traj_sensitivities)[check_points, :]

        for i in range(num_constraints):
            point_ineq_weight, point_ineq_constant, point_constraint_vals = self._get_ineq_mats_fn(
                constraint_list[i], state, traj_states, traj_sensitivities)
            ineq_weight_barrier = ineq_weight_barrier.at[i * num_points:(i + 1) * num_points, :].set(point_ineq_weight)
            ineq_constant_barrier = ineq_constant_barrier.at[i * num_points:(i + 1) * num_points].set(point_ineq_constant)
            constraint_vals = constraint_vals.at[i, :].set(point_constraint_vals)

        if self.subsample_constraints_num_least is not None:
            constraint_sorted_idxs = jnp.argsort(constraint_vals)
            use_idxs = constraint_sorted_idxs[:, 0:self.subsample_constraints_num_least]

            constraint_rows = jnp.arange(constraint_vals.shape[0])[:, None]
            final_idxs = constraint_rows * constraint_vals.shape[1] + use_idxs

            ineq_weight_barrier = ineq_weight_barrier[final_idxs.flatten(), :]
            ineq_constant_barrier = ineq_constant_barrier[final_idxs.flatten()]

        ineq_weight = jnp.concatenate((self.ineq_weight_actuation, ineq_weight_barrier))
        ineq_constant = jnp.concatenate((self.ineq_constant_actuation, ineq_constant_barrier))
        return ineq_weight.transpose(), ineq_constant

    def _get_constraint_ineq_mats(
        self, constraint: ConstraintModule, state: jnp.ndarray, traj_states: jnp.ndarray, traj_sensitivities: jnp.ndarray
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

        Returns
        -------
        jnp.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            vector b of quadprog inequality constraint C.T x >= b
        jnp.ndarray
            constraint values along trajectory
        """
        constraint_vmapped = vmap(self.invariance_constraints, (None, None, 0, 0), (0, 0, 0))
        point_ineq_weight, point_ineq_constant, point_constraint_vals = constraint_vmapped(
            constraint, state, traj_states, traj_sensitivities
        )
        return point_ineq_weight, point_ineq_constant, point_constraint_vals

    def _get_constraint_ineq_mats_no_vmap(
        self, constraint: ConstraintModule, state: jnp.ndarray, traj_states: jnp.ndarray, traj_sensitivities: jnp.ndarray
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
            w, c, v = self.invariance_constraints(constraint, state, traj_state, traj_sensitivity)
            point_ineq_weight.append(w)
            point_ineq_constant.append(c)
            point_constraint_vals.append(v)
        return jnp.array(point_ineq_weight), jnp.array(point_ineq_constant), jnp.array(point_constraint_vals)

    def invariance_constraints(
        self, constraint: ConstraintModule, initial_state: jnp.ndarray, traj_state: jnp.ndarray, traj_sensitivity: jnp.ndarray
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

        grad_x = constraint.grad(traj_state_array)
        ineq_weight = grad_x @ (traj_sensitivity_array @ g_x0)

        ineq_constant = grad_x @ (traj_sensitivity_array @ f_x0) \
            + constraint.alpha(constraint(traj_state_array))

        return ineq_weight, -ineq_constant, constraint(traj_state_array)

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
        if self.integration_method == 'RK45':
            sol = scipy.integrate.solve_ivp(self.state_dot_fn, (0, step_size), state, args=(control, ))
            next_state_vec = sol.y[:, -1]
            out = to_jnp_array_jit(next_state_vec)
        elif self.integration_method == 'Euler':
            state_dot = self.state_dot_fn(0, state, control)
            out = state + state_dot * step_size
        elif self.integration_method == 'RK45_JAX':
            sol = odeint(self.state_dot_fn_jax, state, jnp.linspace(0., step_size, 11), control)
            out = sol[-1, :]
        else:
            raise ValueError('integration_method must be either RK45_JAX, RK45, or Euler')
        return out

    def state_dot_fn(
        self,
        t: float,  # pylint: disable=unused-argument
        state: Union[np.ndarray, jnp.ndarray],
        control: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        """
        Computes the instantaneous time derivative of the state vector for scipy solve_ivp

        Parameters
        ----------
        t : float
            Time in seconds since the beginning of the simulation step.
            Note, this is NOT the total simulation time but the time within the individual step.
        state : Union[np.ndarray, jnp.ndarray]
            Current state vector at time t.
        control : Union[np.ndarray, jnp.ndarray]
            Control vector.

        Returns
        -------
        Union[np.ndarray, jnp.ndarray]
            Instantaneous time derivative of the state vector.
        """
        return self.state_transition_system(to_jnp_array_jit(state)) + self.state_transition_input(to_jnp_array_jit(state)) @ control

    def state_dot_fn_jax(
        self,
        state: jnp.ndarray,
        t: float,  # pylint: disable=unused-argument
        control: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the instantaneous time derivative of the state vector for jax odeint

        Parameters
        ----------
        state : jnp.ndarray
            Current state vector at time t.
        t : float
            Time in seconds since the beginning of the simulation step.
            Note, this is NOT the total simulation time but the time within the individual step.
        control : jnp.ndarray
            Control vector.

        Returns
        -------
        jnp.ndarray
            Instantaneous time derivative of the state vector.
        """
        return self.state_transition_system(state) + self.state_transition_input(state) @ control


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
