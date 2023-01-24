"""
This module implements the backup controller interface.
These backup controllers are used to provide safe actions for backup controller based rta methods
"""
from __future__ import annotations

import abc
from typing import Dict, Union

import jax.numpy as jnp
from jax import jacfwd, jit


class RTABackupController(abc.ABC):
    """Base Class for backup controllers used by backup control based RTA methods
    """

    def __init__(self, controller_state_initial: Union[jnp.ndarray, Dict[str, jnp.ndarray], None] = None, jit_enable: bool = True):
        self.controller_state_initial = self._copy_controller_state(controller_state_initial)
        self.controller_state_saved = None
        self.jit_enable = jit_enable

        self.controller_state = self._copy_controller_state(self.controller_state_initial)
        self._compose()

    def _compose(self):
        if self.jit_enable:
            self._jacobian = jit(
                jacfwd(self._generate_control, has_aux=True), static_argnums=[1, 2], static_argnames=['step_size', 'controller_state']
            )
            self._generate_control_fn = jit(self._generate_control, static_argnames=['step_size', 'controller_state'])
        else:
            self._jacobian = jacfwd(self._generate_control, has_aux=True)
            self._generate_control_fn = self._generate_control

    def reset(self):
        """Resets the backup controller to its initial state for a new episode
        """
        self.controller_state = self._copy_controller_state(self.controller_state_initial)
        self._compose()

    def _copy_controller_state(self, controller_state: Union[jnp.ndarray, Dict[str, jnp.ndarray], None]):
        if controller_state is None:
            copied_state = None
        elif isinstance(controller_state, jnp.ndarray):
            copied_state = jnp.copy(controller_state)
        elif isinstance(controller_state, dict):
            copied_state = {k: jnp.copy(v) for k, v in controller_state.items()}
        else:
            raise TypeError("controller_state to copy must be one of (jnp.ndarray, Dict[str,jnp.ndarray], None)")

        return copied_state

    def generate_control(self, state: jnp.ndarray, step_size: float) -> jnp.ndarray:
        """Generates safe backup control given the current state and step size

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        step_size : float
            time duration over which backup control action will be applied

        Returns
        -------
        jnp.ndarray
            control vector
        """
        controller_output = self._generate_control_fn(state, step_size, self.controller_state)
        if (not isinstance(controller_output, tuple)) or len(controller_output) != 2:
            raise ValueError('_generate_control should return 2 values: the control vector and the updated controller state')
        control, self.controller_state = controller_output
        return control

    def generate_control_with_controller_state(
        self,
        state: jnp.ndarray,
        step_size: float,
        controller_state: Union[jnp.ndarray, Dict[str, jnp.ndarray]] = None
    ) -> tuple[jnp.ndarray, Union[jnp.ndarray, Dict[str, jnp.ndarray], None]]:
        """Generates safe backup control given the current state, step_size, and internal controller state

        Note that in order to be compatible with jax differentiation and jit compiler, all states that are modified by
            generating control must be contained within the controller_state

        Public interface for _generate_control

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        step_size : float
            time duration over which backup control action will be applied
        controller_state: jnp.ndarray or Dict[str, jnp.ndarray] or None
            internal controller state. For stateful controllers, all states that are modified in the control computation
                (e.g. integral control error buffers) must be contained within controller_state

        Returns
        -------
        jnp.ndarray
            control vector
        jnp.ndarray or Dict[str, jnp.ndarray] or None
            Updated controller_state modified by the control algorithm
            If no internal controller_state is used, return None
        """
        return self._generate_control_fn(state, step_size, controller_state)

    @abc.abstractmethod
    def _generate_control(
        self,
        state: jnp.ndarray,
        step_size: float,
        controller_state: Union[jnp.ndarray, Dict[str, jnp.ndarray]] = None
    ) -> tuple[jnp.ndarray, Union[jnp.ndarray, Dict[str, jnp.ndarray], None]]:
        """Generates safe backup control given the current state, step_size, and internal controller state

        Note that in order to be compatible with jax differentiation and jit compiler, all states that are modified by
            generating control must be contained within the controller_state

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        step_size : float
            time duration over which backup control action will be applied
        controller_state: jnp.ndarray or Dict[str, jnp.ndarray] or None
            internal controller state. For stateful controllers, all states that are modified in the control computation
                (e.g. integral control error buffers) must be contained within controller_state

        Returns
        -------
        jnp.ndarray
            control vector
        jnp.ndarray or Dict[str, jnp.ndarray] or None
            Updated controller_state modified by the control algorithm
            If no internal controller_state is used, return None
        """
        raise NotImplementedError()

    def jacobian(self, state: jnp.ndarray, step_size: float):
        """Computes the jacobian of the of the backup controller control output with respect to the input state

        Parameters
        ----------
        state : jnp.ndarray
            current rta state of the system
        step_size : float
            time duration over which backup control action will be applied

        Returns
        -------
        jnp.ndarray
            jacobian matrix
        """
        return self._jacobian(state, step_size, self.controller_state)[0]

    def save(self):
        """Save the internal state of the backup controller
        Allows trajectory integration with a stateful backup controller
        """
        self.controller_state_saved = self._copy_controller_state(self.controller_state)

    def restore(self):
        """Restores the internal state of the backup controller from the last save
        Allows trajectory integration with a stateful backup controller

        !!! Note stateful backup controllers are not compatible with jax jit compilation
        """
        self.controller_state = self.controller_state_saved
