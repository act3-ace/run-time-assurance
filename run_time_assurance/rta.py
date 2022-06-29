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
from typing import Any, Optional, Union

import numpy as np
import quadprog

from run_time_assurance.state import RTAState


class RTAModule(abc.ABC):
    """Base class for RTA modules

    Parameters
    ----------
    control_bounds_high : Union[float, int, list], optional
        upper bound of allowable control. Pass a list for element specific limit. By default None
    control_bounds_low : Union[float, int, list], optional
        upper bound of allowable control. Pass a list for element specific limit. By default None
    """

    def __init__(
        self,
        *args: Any,
        control_bounds_high: Union[float, int, list, np.ndarray] = None,
        control_bounds_low: Union[float, int, list, np.ndarray] = None,
        **kwargs: Any
    ):
        if isinstance(control_bounds_high, list):
            control_bounds_high = np.array(control_bounds_high, float)

        if isinstance(control_bounds_low, list):
            control_bounds_low = np.array(control_bounds_low, float)

        self.control_bounds_high = control_bounds_high
        self.control_bounds_low = control_bounds_low

        self.enable = True
        self.intervening = False
        self.control_desired: Optional[np.ndarray] = None
        self.control_actual: Optional[np.ndarray] = None

        super().__init__(*args, **kwargs)

        self._setup_properties()
        self.constraints = self._setup_constraints()

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
    def _setup_constraints(self) -> OrderedDict:
        """Initializes and returns RTA constraints

        Returns
        -------
        OrderedDict
            OrderedDict of rta contraints with name string keys and ConstraintModule object values
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _pred_state(self, state: RTAState, step_size: float, control: np.ndarray) -> RTAState:
        """predict the next state of the system given the current state, step size, and control vector"""
        raise NotImplementedError()

    def _get_state(self, input_state) -> RTAState:
        """Converts the global state to an internal RTA state"""

        assert isinstance(input_state, (RTAState, np.ndarray)), (
            "input_state must be an RTAState or numpy array. "
            "If you are tying to use a custom state variable, make sure to implement a custom "
            "_get_state() method to translate your custom state to an RTAState")

        if isinstance(input_state, RTAState):
            return input_state

        return self.gen_rta_state(vector=input_state)

    def filter_control(self, input_state, step_size: float, control: np.ndarray) -> np.ndarray:
        """filters desired control into safe action

        Parameters
        ----------
        input_state
            input state of environment to RTA module. May be any custom state type.
            If using a custom state type, make sure to implement _get_state to traslate into an RTA state.
            If custom _get_state() method is not implemented, must be an RTAState or numpy.ndarray instance.
        step_size : float
            time duration over which filtered control will be applied to actuators
        control : np.ndarray
            desired control vector

        Returns
        -------
        np.ndarray
            safe filtered control vector
        """
        self.control_desired = np.copy(control)

        if self.enable:
            state = self._get_state(input_state)
            self.control_actual = np.copy(self._filter_control(state, step_size, control))
        else:
            self.control_actual = np.copy(control)

        return np.copy(self.control_actual)

    @abc.abstractmethod
    def _filter_control(self, state: RTAState, step_size: float, control: np.ndarray) -> np.ndarray:
        """custom logic for filtering desired control into safe action

        Parameters
        ----------
        state : RTAState
            current rta state of the system
        step_size : float
            simulation step size
        control : np.ndarray
            desired control vector

        Returns
        -------
        np.ndarray
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

    def _clip_control(self, control: np.ndarray) -> np.ndarray:
        """clip control vector values to specified upper and lower bounds
        Parameters
        ----------
        control : np.ndarray
            raw control vector

        Returns
        -------
        np.ndarray
            clipped control vector
        """
        if self.control_bounds_low is not None or self.control_bounds_high is not None:
            control = np.clip(control, self.control_bounds_low, self.control_bounds_high)  # type: ignore
        return control

    def gen_rta_state(self, vector: np.ndarray) -> RTAState:
        """Wraps a numpy array into an rta state

        Parameters
        ----------
        vector : np.array
            state vector

        Returns
        -------
        RTAState
            rta state derived from input state vector
        """
        return RTAState(vector=vector)


class RTABackupController(abc.ABC):
    """Base Class for backup controllers used by backup control based RTA methods
    """

    @abc.abstractmethod
    def generate_control(self, state: RTAState, step_size: float) -> np.ndarray:
        """Generates safe backup control given the current state and step size

        Parameters
        ----------
        state : RTAState
            current rta state of the system
        step_size : float
            time duration over which backup control action will be applied

        Returns
        -------
        np.ndarray
            control vector
        """
        raise NotImplementedError()

    def compute_jacobian(self, state: RTAState, step_size: float) -> np.ndarray:
        """Computes the Jacobian of the backup controller's control output wrt the state. Used by implicit asif methods.

        Parameters
        ----------
        state : RTAState
            Current rta state of the system at which to evaluate the jacobian
        step_size : float
            time duration over which backup control action will be applied

        Returns
        -------
        np.ndarray
            Jacobian of the output control vector wrt the state input
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the backup controller to its initial state for a new episode
        """

    def save(self):
        """Save the internal state of the backup controller
        Allows trajectory integration with a stateful backup controller
        """

    def restore(self):
        """Restores the internal state of the backup controller from the last save
        Allows trajectory integration with a stateful backup controller
        """


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

    def backup_control(self, state: RTAState, step_size: float) -> np.ndarray:
        """retrieve safe backup control given the current state

        Parameters
        ----------
        state : RTAState
            current rta state of the system
        step_size : float
            time duration over which backup control action will be applied

        Returns
        -------
        np.ndarray
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

    def _filter_control(self, state: RTAState, step_size: float, control: np.ndarray) -> np.ndarray:
        """Simplex implementation of filter control
        Returns backup control if monitor returns True
        """
        self.intervening = self.monitor(state, step_size, control)

        if self.intervening:
            return self.backup_control(state, step_size)

        return control

    def monitor(self, state: RTAState, step_size: float, control: np.ndarray) -> bool:
        """Detects if desired control will result in an unsafe state

        Parameters
        ----------
        state : RTAState
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
    def _monitor(self, state: RTAState, step_size: float, control: np.ndarray, intervening: bool) -> bool:
        """custom monitor implementation

        Parameters
        ----------
        state : RTAState
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


class ExplicitSimplexModule(SimplexModule):
    """Base implementation for Explicit Simplex RTA module
    Switches to backup controller if desired control would evaluate safety constraint at next timestep
    Requires a backup controller which is known safe within the constraint set
    """

    def _monitor(self, state: RTAState, step_size: float, control: np.ndarray, intervening: bool) -> bool:
        pred_state = self._pred_state(state, step_size, control)
        for c in self.constraints.values():
            if c(pred_state) < 0:
                return True

        return False


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

    def _monitor(self, state: RTAState, step_size: float, control: np.ndarray, intervening: bool) -> bool:
        Nsteps = int(self.backup_window / step_size) + 1

        next_state = self._pred_state(state, step_size, control)
        traj_states = self.integrate(next_state, step_size, Nsteps)

        for i in range(Nsteps):
            for c in self.constraints.values():
                if c(traj_states[i]) < 0:
                    return True

        return False

    def integrate(self, state: RTAState, step_size: float, Nsteps: int) -> list:
        """Estimate backup trajectory by polling backup controller backup control and integrating system dynamics

        Parameters
        ----------
        state : RTAState
            initial rta state of the system
        step_size : float
            simulation integration step size
        Nsteps : int
            number of simulation integration steps

        Returns
        -------
        list
            list of rta states from along the trajectory
        """
        traj_states = [state.copy()]

        self.backup_controller_save()

        for _ in range(1, Nsteps):
            control = self.backup_control(state, step_size)
            state = self._pred_state(state, step_size, control)
            traj_states.append(state)

        self.backup_controller_restore()

        return traj_states


class ASIFModule(RTAModule):
    """
    Base class for Active Set Invariance Filter Optimization RTA

    Only supports dynamical systems with dynamics in the form of:
        dx/dt = f(x) + g(x)u

    Parameters
    ----------
    epislon : float
        threshold distance between desired action and actual safe action at which the rta is said to be intervening
        default 1e-2
    """

    def __init__(self, *args: Any, epsilon: float = 1e-2, **kwargs: Any):
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

    def _filter_control(self, state: RTAState, step_size: float, control: np.ndarray) -> np.ndarray:
        obj_weight, obj_constant = self._generate_objective_function_mats(control)

        ineq_weight_actuation, ineq_constant_actuation = self._generate_actuation_constraint_mats(control.size)
        ineq_weight_barrier, ineq_constant_barrier = \
            self._generate_barrier_constraint_mats(state, step_size, control.size)

        ineq_weight = np.concatenate((ineq_weight_actuation, ineq_weight_barrier))
        ineq_constant = np.concatenate((ineq_constant_actuation, ineq_constant_barrier))
        actual_control = self._optimize(obj_weight, obj_constant, ineq_weight, ineq_constant)
        self.intervening = self.monitor(control, actual_control)

        return actual_control

    def _generate_objective_function_mats(self, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """generates matrices for quadratic program optimization objective function

        Parameters
        ----------
        control : np.ndarray
            desired control vector

        Returns
        -------
        np.ndarray
            matix G of quadprog objective 1/2 x^T G x - a^T x
        np.ndarray
            vector a of quadprog objective 1/2 x^T G x - a^T x
        """
        obj_weight = np.eye(len(control))
        obj_constant = np.reshape(control, len(control)).astype('float64')
        return obj_weight, obj_constant

    def _generate_actuation_constraint_mats(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """generates matrices for quadratic program optimization inequality constraint matrices that impose actuator limits
        on optimized control vector

        Parameters
        ----------
        dim : int
            length of control vector, corresponds to number of columns in ineq weight matrix

        Returns
        -------
        np.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        np.ndarray
            vector a of quadprog objective 1/2 x^T G x - a^T x
        """
        ineq_weight = np.empty((0, dim))
        ineq_constant = np.empty(0)

        if self.control_bounds_low is not None:
            c, b = get_lower_bound_ineq_constraint_mats(self.control_bounds_low, dim)
            ineq_weight = np.vstack((ineq_weight, c))
            ineq_constant = np.concatenate((ineq_constant, b))

        if self.control_bounds_high is not None:
            c, b = get_lower_bound_ineq_constraint_mats(self.control_bounds_high, dim)
            c *= -1
            b *= -1
            ineq_weight = np.vstack((ineq_weight, c))
            ineq_constant = np.concatenate((ineq_constant, b))

        return ineq_weight, ineq_constant

    def _optimize(self, obj_weight: np.ndarray, obj_constant: np.ndarray, ineq_weight: np.ndarray, ineq_constant: np.ndarray) -> np.ndarray:
        """Solve ASIF optimization problem via quadratic program

        Parameters
        ----------
        obj_weight : np.ndarray
            matix G of quadprog objective 1/2 x^T G x - a^T x
        obj_constant : np.ndarray
            vector a of quadprog objective 1/2 x^T G x - a^T x
        ineq_weight : np.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        ineq_constant : np.ndarray
            vector a of quadprog objective 1/2 x^T G x - a^T x

        Returns
        -------
        np.ndarray
            _description_
        """
        return quadprog.solve_qp(obj_weight, obj_constant, ineq_weight.T, ineq_constant, 0)[0]

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
        if np.linalg.norm(desired_control - actual_control) <= self.epsilon:
            return False

        return True

    @abc.abstractmethod
    def _generate_barrier_constraint_mats(self, state: RTAState, step_size: float, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """generates matrices for quadratic program optimization inequality constraint matrices corresponding to safety
        barrier constraints

        Parameters
        ----------
        state : RTAState
            current rta state of the system
        step_size : float
            duration of control step
        dim : int
            length of control vector, corresponds to number of columns in ineq weight matrix

        Returns
        -------
        np.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        np.ndarray
            vector a of quadprog objective 1/2 x^T G x - a^T x
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def state_transition_system(self, state: RTAState) -> np.ndarray:
        """Computes the system state contribution to the system state's time derivative

        i.e. implements f(x) from dx/dt = f(x) + g(x)u

        Parameters
        ----------
        state : RTAState
            current rta state of the system

        Returns
        -------
        np.ndarray
            state time derivative contribution from the current system state
        """
        raise NotImplementedError

    @abc.abstractmethod
    def state_transition_input(self, state: RTAState) -> np.ndarray:
        """Computes the control input matrix contribution to the system state's time derivative

        i.e. implements g(x) from dx/dt = f(x) + g(x)u

        Parameters
        ----------
        state : RTAState
            current rta state of the system

        Returns
        -------
        np.ndarray
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
    """

    def _generate_barrier_constraint_mats(self, state: RTAState, step_size: float, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """generates matrices for quadratic program optimization inequality constraint matrices corresponding to safety
        barrier constraints

        Applies Nagumo's condition to safety constraints

        Parameters
        ----------
        state : RTAState
            current rta state of the system
        step_size : float
            duration of control step
        dim : int
            length of control vector, corresponds to number of columns in ineq weight matrix

        Returns
        -------
        np.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        np.ndarray
            vector a of quadprog objective 1/2 x^T G x - a^T x
        """
        ineq_weight = np.empty((0, dim))
        ineq_constant = np.empty(0)

        for c in self.constraints.values():
            temp1 = c.grad(state) @ self.state_transition_input(state)
            temp2 = -c.grad(state) @ self.state_transition_system(state) - c.alpha(c(state))

            ineq_weight = np.append(ineq_weight, [temp1], axis=0)
            ineq_constant = np.append(ineq_constant, temp2)

        return ineq_weight, ineq_constant


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

    def _generate_barrier_constraint_mats(self, state: RTAState, step_size: float, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """generates matrices for quadratic program optimization inequality constraint matrices corresponding to safety
        barrier constraints

        Computes backup trajectory with backup controller and applies Nagumo's condition on the safety constraints at
        points along backup trajectory.

        Parameters
        ----------
        state : RTAState
            current rta state of the system
        step_size : float
            duration of control step
        dim : int
            length of control vector, corresponds to number of columns in ineq weight matrix

        Returns
        -------
        np.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        np.ndarray
            vector a of quadprog objective 1/2 x^T G x - a^T x
        """

        ineq_weight = np.empty((0, dim))
        ineq_constant = np.empty(0)

        num_steps = int(self.backup_window / step_size) + 1

        traj_states, traj_sensitivities = self.integrate(state, step_size, num_steps)

        check_points = list(range(0, self.num_check_all + 1)) + \
            list(range(self.num_check_all + self.skip_length, num_steps, self.skip_length))

        # resample checkpoints to the trajectory points with the min constraint values
        if self.subsample_constraints_num_least is not None:
            # evaluate constraints at all trajectory points
            constraint_vals = []
            for i in check_points:
                traj_state = traj_states[i]
                constraint_val = min([c(traj_state) for c in self.constraints.values()])
                constraint_vals.append(constraint_val)

            constraint_sorted_idxs = np.argsort(constraint_vals)
            check_points = [check_points[i] for i in constraint_sorted_idxs[0:self.subsample_constraints_num_least]]

        for i in check_points:
            point_ineq_weight, point_ineq_constant = \
                self.invariance_constraints(state, traj_states[i], traj_sensitivities[i], dim)
            ineq_weight = np.concatenate((ineq_weight, point_ineq_weight), axis=0)
            ineq_constant = np.concatenate((ineq_constant, point_ineq_constant), axis=0)

        return ineq_weight, ineq_constant

    def invariance_constraints(self, initial_state: RTAState, traj_state: RTAState, traj_sensitivity: np.ndarray,
                               dim: int) -> tuple[np.ndarray, np.ndarray]:
        """Computes safety constraint invariance constraints via Nagumo's condition for a point in the backup trajectory

        Parameters
        ----------
        initial_state : RTAState
            initial state of the backup trajectory
        traj_state : RTAState
            arbitrary state in the backup trajectory
        traj_sensitivity : np.ndarray
            backup trajectory state sensitivity (i.e. jacobian relative to the initial state)
        dim : int
            length of control vector

        Returns
        -------
        np.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
        np.ndarray
            vector a of quadprog objective 1/2 x^T G x - a^T x
        """
        f_x0 = self.state_transition_system(initial_state)
        g_x0 = self.state_transition_input(initial_state)

        num_constraints = len(self.constraints)

        ineq_weight = np.zeros((num_constraints, dim))
        ineq_constant = np.zeros(num_constraints)

        for i, constraint in enumerate(self.constraints.values()):
            a = constraint.grad(traj_state) @ (traj_sensitivity @ g_x0)

            b = constraint.grad(traj_state) @ (traj_sensitivity @ f_x0) \
                + constraint.alpha(constraint(traj_state))

            ineq_weight[i, :] = a
            ineq_constant[i] = -b

        return ineq_weight, ineq_constant

    def integrate(self, state: RTAState, step_size: float, Nsteps: int) -> tuple[list, list]:
        """Estimate backup trajectory by polling backup controller backup control and integrating system dynamics

        Parameters
        ----------
        state : RTAState
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
            list of trajectory state sensitivities (i.e. jacobian wrt initial trajectory state).
            Elements are np.ndarrays with size (n, n) where n = state.size
        """
        sensitivity = np.eye(state.size)

        traj_states = [state.copy()]
        traj_sensitivity = [sensitivity]

        self.backup_controller_save()

        for _ in range(1, Nsteps):
            control = self.backup_control(state, step_size)
            state = self._pred_state(state, step_size, control)
            sensitivity = sensitivity + (self.compute_jacobian(state, step_size) @ sensitivity) * step_size

            traj_states.append(state)
            traj_sensitivity.append(sensitivity)

        self.backup_controller_restore()

        return traj_states, traj_sensitivity

    @abc.abstractmethod
    def compute_jacobian(self, state: RTAState, step_size: float) -> np.ndarray:
        """Computes Jacobian of system state transition J(f(x) + g(x,u)) wrt x

        Parameters
        ----------
        state : RTAState
            Current RTAState of the system at which to evaluate Jacobian
        step_size : float
            simulation integration step size

        Returns
        -------
        np.ndarray
            Jacobian matrix of state transition
        """
        raise NotImplementedError()


def get_lower_bound_ineq_constraint_mats(bound: Union[np.ndarray, int, float], vec_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Computes inequality constraint matrices for applying a lower bound to optimization var in quadprog

    Parameters
    ----------
    bound : Union[np.ndarray, int, float]
        Lower bound for optimization variable.
        If np.ndarray, must be same length as optimization variable. Will be applied elementwise.
        If number, will be applied to all elements.
    vec_len : int
        optimization variable vector length

    Returns
    -------
    np.ndarray
            matix C.T of quadprog inequality constraint C.T x >= b
    np.ndarray
        vector a of quadprog objective 1/2 x^T G x - a^T x
    """
    c = np.eye(vec_len)

    if isinstance(bound, np.ndarray):
        assert bound.shape == (vec_len, ), f"the shape of bound must be ({vec_len},)"
        b = np.copy(bound)
    else:
        b = bound * np.ones(vec_len)

    return c, b
