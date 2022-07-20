"""
This module implements the base runtime assurance module interface for filtering actions into safe actions
"""
from __future__ import annotations

import abc
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np

from run_time_assurance.constraint import ConstraintModule
from run_time_assurance.controller import RTABackupController
from run_time_assurance.utils import to_jnp_array_jit


class RTAModule(abc.ABC):
    """Base class for RTA modules

    Parameters
    ----------
    control_bounds_high : Union[float, int, list, np.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default None
    control_bounds_low : Union[float, int, list, np.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default None
    """

    def __init__(
        self,
        *args: Any,
        control_bounds_high: Union[float, int, list, np.ndarray] = None,
        control_bounds_low: Union[float, int, list, np.ndarray] = None,
        **kwargs: Any
    ):
        if isinstance(control_bounds_high, (list)):
            control_bounds_high = np.array(control_bounds_high, float)

        if isinstance(control_bounds_low, (list)):
            control_bounds_low = np.array(control_bounds_low, float)

        self.control_bounds_high = control_bounds_high
        self.control_bounds_low = control_bounds_low

        self.enable = True
        self.intervening = False
        self.control_desired: Optional[np.ndarray] = None
        self.control_actual: Optional[np.ndarray] = None

        super().__init__(*args, **kwargs)

    def reset(self):
        """Resets the rta module to the initial state at the beginning of an episode
        """
        self.enable = True
        self.intervening = False
        self.control_desired: np.ndarray = None
        self.control_actual: np.ndarray = None

    def filter_control(self, input_state: Any, step_size: float, control_desired: np.ndarray) -> np.ndarray:
        """filters desired control into safe action

        Parameters
        ----------
        input_state
            input state of environment to RTA module. May be any custom state type.
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
            control_actual = self._clip_control(self.compute_filtered_control(input_state, step_size, control_desired))
            self.control_actual = np.array(control_actual)
        else:
            self.control_actual = np.copy(control_desired)

        return np.copy(self.control_actual)

    @abc.abstractmethod
    def compute_filtered_control(self, input_state: Any, step_size: float, control_desired: np.ndarray) -> np.ndarray:
        """custom logic for filtering desired control into safe action

        Parameters
        ----------
        input_state : Any
            input state of environment to RTA module. May be any custom state type.
        step_size : float
            simulation step size
        control_desired : np.ndarray
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


class ConstraintBasedRTA(RTAModule):
    """Base class for constraint-based RTA systems

    Parameters
    ----------
    control_bounds_high : Union[float, int, list, np.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default None
    control_bounds_low : Union[float, int, list, np.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default None
    jit_compile_dict: Dict[str, bool], optional
        Dictionary specifying which subroutines will be jax jit compiled. Behavior defined in self.compose()
        Useful for implementing versions methods that can't be jit compiled
        Each RTA class will have custom default behavior if not passed
    """

    def __init__(
        self,
        *args: Any,
        control_bounds_high: Union[float, int, list, np.ndarray] = None,
        control_bounds_low: Union[float, int, list, np.ndarray] = None,
        jit_compile_dict: Dict[str, bool] = None,
        **kwargs: Any
    ):
        super().__init__(*args, control_bounds_high=control_bounds_high, control_bounds_low=control_bounds_low, **kwargs)

        if jit_compile_dict is None:
            self.jit_compile_dict = {}
        else:
            self.jit_compile_dict = jit_compile_dict

        self._setup_properties()
        self.constraints = self._setup_constraints()
        self.compose()

    def compute_filtered_control(self, input_state: Any, step_size: float, control_desired: np.ndarray) -> np.ndarray:
        """filters desired control into safe action

        Parameters
        ----------
        input_state : Any
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
        state = self._get_state(input_state)
        control_actual = self._filter_control(state, step_size, to_jnp_array_jit(control_desired))
        self.control_actual = np.array(control_actual)

        return np.copy(control_actual)

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

    def _setup_properties(self):
        """Additional initialization function to allow custom initialization to run after baseclass initialization,
        but before constraint initialization"""

    def compose(self):
        """
        applies jax composition transformations (grad, jit, jacobian etc.)

        jit complilation is determined by the jit_compile_dict constructor parameter
        """

    @abc.abstractmethod
    def _setup_constraints(self) -> OrderedDict[str, ConstraintModule]:
        """Initializes and returns RTA constraints

        Returns
        -------
        OrderedDict
            OrderedDict of rta contraints with name string keys and ConstraintModule object values
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _pred_state(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        """predict the next state of the system given the current state, step size, and control vector"""
        raise NotImplementedError()

    def _clip_control_jax(self, control: jnp.ndarray) -> jnp.ndarray:
        """jax version of clip control for clipping control vector values to specified upper and lower bounds
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

    def _get_state(self, input_state) -> jnp.ndarray:
        """Converts the global state to an internal RTA state"""

        assert isinstance(input_state, (np.ndarray, jnp.ndarray)), (
            "input_state must be an RTAState or numpy array. "
            "If you are tying to use a custom state variable, make sure to implement a custom "
            "_get_state() method to translate your custom state to an RTAState")

        if isinstance(input_state, jnp.ndarray):
            return input_state

        return to_jnp_array_jit(input_state)


class CascadedRTA(RTAModule):
    """Base class for cascaded RTA systems
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self.rta_list = self._setup_rta_list()
        super().__init__(*args, **kwargs)

    def compute_filtered_control(self, input_state: Any, step_size: float, control_desired: np.ndarray) -> np.ndarray:
        self.intervening = False
        control = control_desired
        for rta in self.rta_list:
            control = np.copy(rta.filter_control(input_state, step_size, control))
            if rta.intervening:
                self.intervening = True

        return control

    @abc.abstractmethod
    def _setup_rta_list(self) -> List[RTAModule]:
        """Setup list of RTA objects

        Returns
        -------
        list
            list of RTA objects in order from lowest to highest priority
            for list of length N, where {i = 1, ..., N}, output of RTA {i} is passed as input RTA {i+1}
        """
        raise NotImplementedError()


class BackupControlBasedRTA(ConstraintBasedRTA):
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

        return self._clip_control_jax(control)

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
