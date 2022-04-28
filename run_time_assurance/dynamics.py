"""Implements base dynamics for use in RTA modules"""

from __future__ import annotations

import abc
from typing import Union

import numpy as np
import scipy


class BaseDynamics(abc.ABC):
    """
    State transition implementation for a physics dynamics model. Used by entities to compute their next state while stepping.

    Parameters
    ----------
    state_min : float or np.ndarray
        Minimum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
    state_min : float or np.ndarray
        Maximum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
    angle_wrap_centers: np.ndarray
        Enables circular wrapping of angles. Defines the center of circular wrap such that angles are within [center+pi, center-pi].
        When None, no angle wrapping applied.
        When ndarray, each element defines the angle wrap center of the corresponding state element.
        Wrapping not applied when element is NaN.
    """

    def __init__(
        self,
        state_min: Union[float, np.ndarray] = -np.inf,
        state_max: Union[float, np.ndarray] = np.inf,
        angle_wrap_centers: np.ndarray = None,
    ):
        self.state_min = state_min
        self.state_max = state_max
        self.angle_wrap_centers = angle_wrap_centers

    def step(self, step_size: float, state: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes the dynamics state transition from the current state and control input

        Parameters
        ----------
        step_size : float
            Duration of the simation step in seconds.
        state : np.ndarray
            Current state of the system at the beginning of the simulation step.
        control : np.ndarray
            Control vector of the dynamics model.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            tuple of the systems's next state and the state's instantaneous time derivative at the end of the step
        """
        next_state, state_dot = self._step(step_size, state, control)
        next_state = np.clip(next_state, self.state_min, self.state_max)
        next_state = self._wrap_angles(next_state)
        return next_state, state_dot

    def _wrap_angles(self, state):
        wrapped_state = state.copy()
        if self.angle_wrap_centers is not None:
            wrap_idxs = np.logical_not(np.isnan(self.angle_wrap_centers))

            wrapped_state[wrap_idxs] = \
                ((wrapped_state[wrap_idxs] + np.pi) % (2 * np.pi)) - np.pi + self.angle_wrap_centers[wrap_idxs]

        return wrapped_state

    @abc.abstractmethod
    def _step(self, step_size: float, state: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class BaseODESolverDynamics(BaseDynamics):
    """
    State transition implementation for generic Ordinary Differential Equation dynamics models.
    Computes next state through numerical integration of differential equation.

    Parameters
    ----------
    integration_method : string
        Numerical integration method used by dyanmics solver. One of ['RK45', 'Euler'].
        'RK45' is slow but very accurate.
        'Euler' is fast but very inaccurate.
    kwargs
        Additional keyword arguments passed to parent BaseDynamics constructor.
    """

    def __init__(self, integration_method="RK45", **kwargs):
        self.integration_method = integration_method
        super().__init__(**kwargs)

    def compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Computes the instataneous time derivative of the state vector

        Parameters
        ----------
        t : float
            Time in seconds since the beginning of the simulation step.
            Note, this is NOT the total simulation time but the time within the individual step.
        state : np.ndarray
            Current state vector at time t.
        control : np.ndarray
            Control vector.

        Returns
        -------
        np.ndarray
            Instantaneous time derivative of the state vector.
        """
        state_dot = self._compute_state_dot(t, state, control)
        state_dot = self._clip_state_dot_by_state_limits(state, state_dot)
        return state_dot

    @abc.abstractmethod
    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _clip_state_dot_by_state_limits(self, state, state_dot):
        lower_bounded_states = state <= self.state_min
        upper_bounded_state = state >= self.state_max

        state_dot[lower_bounded_states] = np.clip(state_dot[lower_bounded_states], 0, np.inf)
        state_dot[upper_bounded_state] = np.clip(state_dot[upper_bounded_state], -np.inf, 0)

        return state_dot

    def _step(self, step_size, state, control):

        if self.integration_method == "RK45":
            sol = scipy.integrate.solve_ivp(self.compute_state_dot, (0, step_size), state, args=(control, ))

            next_state = sol.y[:, -1]  # save last timestep of integration solution
            state_dot = self.compute_state_dot(step_size, next_state, control)
        elif self.integration_method == "Euler":
            state_dot = self.compute_state_dot(0, state, control)
            next_state = state + step_size * state_dot
        else:
            raise ValueError(f"invalid integration method '{self.integration_method}'")

        return next_state, state_dot


class BaseLinearODESolverDynamics(BaseODESolverDynamics):
    """
    State transition implementation for generic Linear Ordinary Differential Equation dynamics models of the form dx/dt = Ax+Bu.
    Computes next state through numerical integration of differential equation.

    Parameters
    ----------
    kwargs
        Additional keyword arguments passed to parent BaseODESolverDynamics constructor.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, **kwargs):
        self.A = np.copy(A)
        self.B = np.copy(B)
        super().__init__(**kwargs)

    def _update_dynamics_matrices(self, state):
        """
        Updates the linear ODE matrices A, B with current system state before computing derivative.
        Allows non-linear dynamics models to be linearized at each numerical integration interval.
        Directly modifies self.A, self.B.

        Default implementation is a no-op.

        Parameters
        ----------
        state : np.ndarray
            Current state vector of the system.
        """

    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray):
        self._update_dynamics_matrices(state)
        state_dot = np.matmul(self.A, state) + np.matmul(self.B, control)
        return state_dot
