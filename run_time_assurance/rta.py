"""
This module implements the base rta interface and algorithm implementations
Inlcudes base implementations for the following RTA algorithms:
    - Explicit Simplex
    - Implicit Simplex
    - Explicit Active Set Invariance Filter (ASIF)
    - Implicit Active Set Invariance Filter (ASIF)
"""
from __future__ import annotations

import abc
from collections import OrderedDict
from typing import Any, Dict, Optional, Union, cast

import jax.numpy as jnp
import numpy as np
import quadprog
from jax import jacfwd, jit, vmap

from run_time_assurance.constraint import ConstraintModule
from run_time_assurance.controller import RTABackupController
from run_time_assurance.utils import SolverError, SolverWarning, add_dim_jit, jnp_stack_jit, to_jnp_array_jit


class RTAModule(abc.ABC):
    """Base class for RTA modules

    Parameters
    ----------
    control_bounds_high : Union[float, int, list, np.ndarray, jnp.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default None
    control_bounds_low : Union[float, int, list, np.ndarray, jnp.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default None
    jit_compile_dict: Dict[str, bool], optional
        Dictionary specifying which subroutines will be jax jit compiled. Behavior defined in self.compose()
        Useful for implementing versions methods that can't be jit compiled
        Each RTA class will have custom default behavior if not passed
    """

    def __init__(
        self,
        *args: Any,
        control_bounds_high: Union[float, int, list, np.ndarray, jnp.ndarray] = None,
        control_bounds_low: Union[float, int, list, np.ndarray, jnp.ndarray] = None,
        jit_compile_dict: Dict[str, bool] = None,
        **kwargs: Any
    ):
        if isinstance(control_bounds_high, (list, np.ndarray)):
            control_bounds_high = jnp.array(control_bounds_high, float)
            control_bounds_high = cast(jnp.ndarray, control_bounds_high)

        if isinstance(control_bounds_low, (list, np.ndarray)):
            control_bounds_low = jnp.array(control_bounds_low, float)
            control_bounds_low = cast(jnp.ndarray, control_bounds_low)

        self.control_bounds_high = control_bounds_high
        self.control_bounds_low = control_bounds_low

        if jit_compile_dict is None:
            self.jit_compile_dict = {}
        else:
            self.jit_compile_dict = jit_compile_dict

        self.enable = True
        self.intervening = False
        self.control_desired: Optional[np.ndarray] = None
        self.control_actual: Optional[np.ndarray] = None

        super().__init__(*args, **kwargs)

        self._setup_properties()
        self.constraints = self._setup_constraints()
        self.compose()

    def reset(self):
        """Resets the rta module to the initial state at the beginning of an episode
        """
        self.enable = True
        self.intervening = False
        self.control_desired: np.ndarray = None
        self.control_actual: np.ndarray = None

    def _setup_properties(self):
        """Additional initialization function to allow custom initialization to run after baseclass initialization,
        but before constraint initialization"""

    @abc.abstractmethod
    def _setup_constraints(self) -> OrderedDict[str, ConstraintModule]:
        """Initializes and returns RTA constraints

        Returns
        -------
        OrderedDict
            OrderedDict of rta contraints with name string keys and ConstraintModule object values
        """
        raise NotImplementedError()

    def compose(self):
        """
        applies jax composition transformations (grad, jit, jacobian etc.)

        jit complilation is determined by the jit_compile_dict constructor parameter
        """

    @abc.abstractmethod
    def _pred_state(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        """predict the next state of the system given the current state, step size, and control vector"""
        raise NotImplementedError()

    def _get_state(self, input_state) -> jnp.ndarray:
        """Converts the global state to an internal RTA state"""

        assert isinstance(input_state, (np.ndarray, jnp.ndarray)), (
            "input_state must be an RTAState or numpy array. "
            "If you are tying to use a custom state variable, make sure to implement a custom "
            "_get_state() method to translate your custom state to an RTAState")

        if isinstance(input_state, jnp.ndarray):
            return input_state

        return to_jnp_array_jit(input_state)

    def filter_control(self, input_state, step_size: float, control_desired: np.ndarray) -> np.ndarray:
        """filters desired control into safe action

        Parameters
        ----------
        input_state
            input state of environment to RTA module. May be any custom state type.
            If using a custom state type, make sure to implement _get_state to traslate into an RTA state.
            If custom _get_state() method is not implemented, must be an RTAState or numpy.ndarray instance.
        step_size : float
            time duration over which filtered control will be applied to actuators
        control_desired : np.ndarray
            desired control vector

        Returns
        -------
        np.ndarray
            safe filtered control vector
        """
        self.control_desired = np.copy(control_desired)

        if self.enable:
            state = self._get_state(input_state)
            control_actual = self._clip_control(self._filter_control(state, step_size, to_jnp_array_jit(control_desired)))
            self.control_actual = np.array(control_actual)
        else:
            self.control_actual = np.copy(control_desired)

        return np.copy(self.control_actual)

    @abc.abstractmethod
    def _filter_control(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        """custom logic for filtering desired control into safe action

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        step_size : float
            simulation step size
        control : jnp.ndarray
            desired control vector

        Returns
        -------
        jnp.ndarray
            safe filtered control vector
        """
        raise NotImplementedError()

    def generate_info(self) -> dict:
        """generates info dictionary on RTA module for logging

        Returns
        -------
        dict
            info dictionary for rta module
        """
        info = {
            'enable': self.enable,
            'intervening': self.intervening,
            'control_desired': self.control_desired,
            'control_actual': self.control_actual,
        }

        return info

    def _clip_control(self, control: jnp.ndarray) -> jnp.ndarray:
        """clip control vector values to specified upper and lower bounds
        Parameters
        ----------
        control : jnp.ndarray
            raw control vector

        Returns
        -------
        jnp.ndarray
            clipped control vector
        """
        if self.control_bounds_low is not None or self.control_bounds_high is not None:
            control = jnp.clip(control, self.control_bounds_low, self.control_bounds_high)  # type: ignore
        return control


class BackupControlBasedRTA(RTAModule):
    """Base class for backup control based RTA algorithms
    Adds iterfaces for backup controller member

    Parameters
    ----------
    backup_controller : RTABackupController
        backup controller object utilized by rta module to generate backup control
    """

    def __init__(self, *args: Any, backup_controller: RTABackupController, **kwargs: Any):
        self.backup_controller = backup_controller
        super().__init__(*args, **kwargs)

    def backup_control(self, state: jnp.ndarray, step_size: float) -> jnp.ndarray:
        """retrieve safe backup control given the current state

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        step_size : float
            time duration over which backup control action will be applied

        Returns
        -------
        jnp.ndarray
            backup control vector
        """
        control = self.backup_controller.generate_control(state, step_size)

        return self._clip_control(control)

    def reset_backup_controller(self):
        """Resets the backup controller to the initial state at the beginning of an episode
        """
        self.backup_controller.reset()

    def backup_controller_save(self):
        """Save the internal state of the backup controller
        Allows trajectory integration with a stateful backup controller
        """
        self.backup_controller.save()

    def backup_controller_restore(self):
        """Restores the internal state of the backup controller from the last save
        Allows trajectory integration with a stateful backup controller
        """
        self.backup_controller.restore()


class SimplexModule(BackupControlBasedRTA):
    """Base class for simplex RTA modules.
    Simplex methods for RTA utilize a monitor that detects unsafe behavior and a backup controller that takes over to
        prevent the unsafe behavior

    Parameters
    ----------
    backup_controller : RTABackupController
        backup controller object utilized by rta module to generate backup control
    """

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
            constraint_violation:
                default True
            pred_state:
                default False
        """
        super().compose()
        if self.jit_compile_dict.get('constraint_violation', True):
            self._constraint_violation_fn = jit(self._constraint_violation)
        else:
            self._constraint_violation_fn = self._constraint_violation

        if self.jit_compile_dict.get('pred_state', False):
            self._pred_state_fn = jit(self._pred_state, static_argnames=['step_size'])
        else:
            self._pred_state_fn = self._pred_state

    def _filter_control(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        """Simplex implementation of filter control
        Returns backup control if monitor returns True
        """
        self.intervening = self.monitor(state, step_size, control)

        if self.intervening:
            return self.backup_control(state, step_size)

        return control

    def monitor(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> bool:
        """Detects if desired control will result in an unsafe state

        Parameters
        ----------
        state : jnp.ndarray
            Current rta state of the system
        step_size : float
            time duration over which filtered control will be applied to actuators
        control : np.ndarray
            desired control vector

        Returns
        -------
        bool
            return False if desired control is safe and True if unsafe
        """
        return self._monitor(state, step_size, control, self.intervening)

    @abc.abstractmethod
    def _monitor(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray, intervening: bool) -> bool:
        """custom monitor implementation

        Parameters
        ----------
        state : jnp.ndarray
            Current rta state of the system
        step_size : float
            time duration over which filtered control will be applied to actuators
        control : np.ndarray
            desired control vector
        intervening : bool
            Indicates whether simplex rta is currently intervening with the backup controller or not

        Returns
        -------
        bool
            return False if desired control is safe and True if unsafe
        """
        raise NotImplementedError()

    def _constraint_violation(self, states: jnp.ndarray) -> bool:

        constraint_list = list(self.constraints.values())
        num_constraints = len(self.constraints)
        constraint_violations = jnp.zeros(num_constraints)

        for i in range(num_constraints):
            c = constraint_list[i]

            constraint_vmapped = vmap(c.compute, 0, 0)
            traj_constraint_vals = constraint_vmapped(states)

            constraint_violations = constraint_violations.at[i].set(jnp.any(traj_constraint_vals < 0))

        return jnp.any(constraint_violations)


class ExplicitSimplexModule(SimplexModule):
    """Base implementation for Explicit Simplex RTA module
    Switches to backup controller if desired control would evaluate safety constraint at next timestep
    Requires a backup controller which is known safe within the constraint set
    """

    def _monitor(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray, intervening: bool) -> bool:
        pred_state = self._pred_state_fn(state, step_size, control)
        return bool(self._constraint_violation_fn(add_dim_jit(pred_state)))


class ImplicitSimplexModule(SimplexModule):
    """Base implementation for Explicit Simplex RTA module
    Switches to backup controller if desired control would result in a state from which the backup controller cannot
        recover
    This is determined by computing a trajectory under the backup controller and ensuring that the safety constraints
        aren't violated along it

    Parameters
    ----------
    backup_window : float
        Duration of time in seconds to evaluate backup controller trajectory
    backup_controller : RTABackupController
        backup controller object utilized by rta module to generate backup control
    """

    def __init__(self, *args: Any, backup_window: float, backup_controller: RTABackupController, **kwargs: Any):
        self.backup_window = backup_window
        super().__init__(*args, backup_controller=backup_controller, **kwargs)

    def compose(self):
        """
        applies jax composition transformations (grad, jit, jacobian etc.)

        jit complilation is determined by the jit_compile_dict constructor parameter
        jit compilation settings:
            integrate:
                Backup controller trajectory integration
                default False

        See parent class for additional options
        """
        super().compose()
        if self.jit_compile_dict.get('integrate', False):
            self._integrate_fn = jit(self.integrate, static_argnames=['step_size'])
        else:
            self._integrate_fn = self.integrate

    def _monitor(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray, intervening: bool) -> bool:

        traj_states = self._integrate_fn(state, step_size, control)

        return bool(self._constraint_violation_fn(traj_states))

    def integrate(self, state: jnp.ndarray, step_size: float, desired_control: jnp.ndarray) -> jnp.ndarray:
        """Estimate backup trajectory by polling backup controller backup control and integrating system dynamics

        Parameters
        ----------
        state : jnp.ndarray
            initial rta state of the system
        step_size : float
            simulation integration step size
        desired_control : jnp.ndarray
            control desired by the primary controller

        Returns
        -------
        jnp.ndarray
            jax array of implict backup trajectory states.
            Shape (M, N) where M is number of states and N is the dimension of the state vector
        """
        Nsteps = int(self.backup_window / step_size)

        state = self._pred_state_fn(state, step_size, desired_control)
        traj_states = [state]

        self.backup_controller_save()

        for _ in range(Nsteps):
            control = self.backup_control(state, step_size)
            state = self._pred_state_fn(state, step_size, control)
            traj_states.append(state)

        self.backup_controller_restore()

        return jnp_stack_jit(traj_states, axis=0)


class ASIFModule(RTAModule):
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

    def compose(self):
        """
        applies jax composition transformations (grad, jit, jacobian etc.)

        jit complilation is determined by the jit_compile_dict constructor parameter
        jit compilation settings:
            generate_barrier_constraint_mats:
                default True
        """
        super().compose()
        if self.jit_compile_dict.get('generate_barrier_constraint_mats', True):
            self._generate_barrier_constraint_mats_fn = jit(self._generate_barrier_constraint_mats)
        else:
            self._generate_barrier_constraint_mats_fn = self._generate_barrier_constraint_mats

    def _filter_control(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> np.ndarray:
        ineq_weight, ineq_constant = self._generate_barrier_constraint_mats_fn(state, step_size)
        desired_control = np.array(control, dtype=np.float64)
        actual_control = self._optimize(self.obj_weight, desired_control, ineq_weight, ineq_constant)
        self.intervening = self.monitor(desired_control, actual_control)

        return actual_control

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
                    SolverWarning()
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
    """

    def __init__(
        self,
        *args: Any,
        backup_window: float,
        num_check_all: int = 0,
        skip_length: int = 1,
        subsample_constraints_num_least: int = None,
        backup_controller: RTABackupController,
        **kwargs: Any,
    ):
        self.backup_window = backup_window
        self.num_check_all = num_check_all
        self.skip_length = skip_length

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
        super().compose()
        self._jacobian = jacfwd(self._backup_state_transition)
        if self.jit_compile_dict.get('generate_ineq_constraint_mats', True):
            self._generate_ineq_constraint_mats_fn = jit(self._generate_ineq_constraint_mats, static_argnames=['num_steps'])
        else:
            self._generate_ineq_constraint_mats_fn = self._generate_ineq_constraint_mats
        if self.jit_compile_dict.get('pred_state', False):
            self._pred_state_fn = jit(self._pred_state, static_argnames=['step_size'])
        else:
            self._pred_state_fn = self._pred_state
        if self.jit_compile_dict.get('integrate', False):
            self._integrate_fn = jit(self.integrate, static_argnames=['step_size', 'Nsteps'])
        else:
            self._integrate_fn = self.integrate

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
        return self._generate_ineq_constraint_mats_fn(state, num_steps, traj_states, traj_sensitivities)

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
        ineq_weight_barrier = jnp.empty((0, self.control_dim))
        ineq_constant_barrier = jnp.empty(0)

        check_points = jnp.hstack(
            (
                jnp.array(range(0, self.num_check_all + 1)),
                jnp.array(range(self.num_check_all + self.skip_length, num_steps, self.skip_length))
            )
        ).astype(int)

        # resample checkpoints to the trajectory points with the min constraint values
        if self.subsample_constraints_num_least is not None:
            # evaluate constraints at all trajectory points
            constraint_vals = []
            for i in check_points:
                traj_state = traj_states[i]
                constraint_val = min([c(traj_state) for c in self.constraints.values()])
                constraint_vals.append(constraint_val)

            constraint_sorted_idxs = jnp.argsort(constraint_vals)
            check_points = [check_points[i] for i in constraint_sorted_idxs[0:self.subsample_constraints_num_least]]

        traj_states = jnp.array(traj_states)[check_points, :]
        traj_sensitivities = jnp.array(traj_sensitivities)[check_points, :]

        constraint_list = list(self.constraints.values())
        num_constraints = len(self.constraints)
        for i in range(num_constraints):
            constraint_vmapped = vmap(self.invariance_constraints, (None, None, 0, 0), (0, 0))
            point_ineq_weight, point_ineq_constant = constraint_vmapped(constraint_list[i], state, traj_states, traj_sensitivities)
            ineq_weight_barrier = jnp.concatenate((ineq_weight_barrier, point_ineq_weight), axis=0)
            ineq_constant_barrier = jnp.concatenate((ineq_constant_barrier, point_ineq_constant), axis=0)

        ineq_weight = jnp.concatenate((self.ineq_weight_actuation, ineq_weight_barrier))
        ineq_constant = jnp.concatenate((self.ineq_constant_actuation, ineq_constant_barrier))
        return ineq_weight.transpose(), ineq_constant

    def invariance_constraints(
        self, constraint: ConstraintModule, initial_state: jnp.ndarray, traj_state: jnp.ndarray, traj_sensitivity: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        """
        traj_state_array = jnp.array(traj_state)
        traj_sensitivity_array = jnp.array(traj_sensitivity)

        f_x0 = self.state_transition_system(initial_state)
        g_x0 = self.state_transition_input(initial_state)

        grad_x = constraint.grad(traj_state_array)
        ineq_weight = grad_x @ (traj_sensitivity_array @ g_x0)

        ineq_constant = grad_x @ (traj_sensitivity_array @ f_x0) \
            + constraint.alpha(constraint(traj_state_array))

        return ineq_weight, -ineq_constant

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


def get_lower_bound_ineq_constraint_mats(bound: Union[jnp.ndarray, int, float], vec_len: int) -> tuple[jnp.ndarray, jnp.ndarray]:
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
