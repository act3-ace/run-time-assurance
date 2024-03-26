"""
This module implements Explicit and Simple RTA modules. These RTA modules utilize a backup controller that is switched
    to when the desired action would causes the system to leave either the explicit or implicit safeset.
"""
from __future__ import annotations

import abc
from typing import Any, Union

import jax.numpy as jnp
from jax import jit, lax, vmap

from run_time_assurance.constraint import ConstraintModule
from run_time_assurance.controller import RTABackupController
from run_time_assurance.rta.base import BackupControlBasedRTA
from run_time_assurance.utils import add_dim_jit, jnp_stack_jit


class SimplexModule(BackupControlBasedRTA):
    """Base class for simplex RTA modules.
    Simplex methods for RTA utilize a monitor that detects unsafe behavior and a backup controller that takes over to
        prevent the unsafe behavior

    Parameters
    ----------
    latch_time : float
        Amount of time to latch onto the backup controller. By default 0 (unlatched)
    """

    def __init__(self, *args: Any, latch_time: float = 0, **kwargs: Any):
        self.latch_time = latch_time
        self.latched_elapsed = 0.
        super().__init__(*args, **kwargs)

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
        if self.jit_enable and self.jit_compile_dict.get('constraint_violation', True):
            self._constraint_violation_fn = jit(self._constraint_violation)
        else:
            self._constraint_violation_fn = self._constraint_violation

        if self.jit_enable and self.jit_compile_dict.get('pred_state', False):
            self._pred_state_fn = jit(self._pred_state, static_argnames=['step_size'])
        else:
            self._pred_state_fn = self._pred_state

        if self.jit_enable:
            self._get_constraint_vals_fn = self._get_constraint_vals
        else:
            self._get_constraint_vals_fn = self._get_constraint_vals_no_vmap

    def _filter_control(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray) -> jnp.ndarray:
        """Simplex implementation of filter control
        If latched, returns backup control.
        Otherwise returns backup control if monitor returns True
        """
        latched = False
        if self.intervening:
            if self.latched_elapsed >= self.latch_time:
                self.latched_elapsed = 0
            else:
                self.latched_elapsed += step_size
                latched = True

        if latched:
            control = self.backup_control(state, step_size)

        else:
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

    def _constraint_violation(self, states: jnp.ndarray, params: dict, constraint_enabled_dict: dict) -> bool:
        """Determine if any constraints are violated

        Parameters
        ----------
        states : jnp.ndarray
            Array of states to check
        params : dict
            Parameters for each constraint
        constraint_enabled_dict : dict
            Enable flag for each constraint

        Returns
        -------
        bool
            return True if constraints are violated, False if not
        """
        # Initialize
        constraint_violations = jnp.zeros(len(self.constraints))

        for i, k in enumerate(self.constraints.keys()):
            # Get constraint
            c = self.constraints[k]

            # Get constraint vals along trajectory
            traj_constraint_vals = self._get_constraint_vals_fn(c, states, params[k])

            # Return zero if constraint is disabled
            traj_constraint_vals = lax.cond(
                constraint_enabled_dict[k],
                self.constraint_enabled,
                self.constraint_disabled,
                traj_constraint_vals,
            )

            # Set constraint violation
            constraint_violations = constraint_violations.at[i].set(jnp.any(traj_constraint_vals < 0))

        return jnp.any(constraint_violations)

    def constraint_enabled(self, traj_constraint_vals: jnp.ndarray) -> jnp.ndarray:
        """Return constraint vals when constraint is enabled"""
        return traj_constraint_vals

    def constraint_disabled(self, traj_constraint_vals: jnp.ndarray) -> jnp.ndarray:
        """Return zeros when constraint is disabled"""
        return jnp.zeros(traj_constraint_vals.shape)

    def _get_constraint_vals(self, constraint: ConstraintModule, states: jnp.ndarray, params: dict) -> Union[float, jnp.ndarray]:
        """Uses vmap to compute constraint values along a trajectory

        Parameters
        ----------
        constraint : ConstraintModule
            Constraint to evaluate
        states : jnp.ndarray
            array of state values
        params : dict
            dict of parameters for the constraint

        Returns
        -------
        [float, jnp.ndarray]
            constraint values along trajectory
        """
        constraint_vmapped = vmap(constraint.compute, (0, None), 0)
        traj_constraint_vals = constraint_vmapped(states, params)
        return traj_constraint_vals

    def _get_constraint_vals_no_vmap(self, constraint: ConstraintModule, states: jnp.ndarray, params: dict) -> jnp.ndarray:
        """Does not use vmap to compute constraint values along a trajectory, allows for easier debugging

        Parameters
        ----------
        constraint : ConstraintModule
            Constraint to evaluate
        states : jnp.ndarray
            array of state values
        params : dict
            dict of parameters for the constraint

        Returns
        -------
        jnp.ndarray
            constraint values along trajectory
        """
        traj_constraint_vals = []
        for state in states:
            traj_constraint_vals.append(constraint.compute(state, params))
        return jnp.array(traj_constraint_vals)


class ExplicitSimplexModule(SimplexModule):
    """Base implementation for Explicit Simplex RTA module
    Switches to backup controller if desired control would evaluate safety constraint at next timestep
    Requires a backup controller which is known safe within the constraint set
    """

    def _monitor(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray, intervening: bool) -> bool:
        pred_state = self._pred_state_fn(state, step_size, control)
        return bool(self._constraint_violation_fn(add_dim_jit(pred_state), self.params, self.constraint_enabled_dict))


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
        if self.jit_enable and self.jit_compile_dict.get('integrate', False):
            self._integrate_fn = jit(self.integrate, static_argnames=['step_size'])
        else:
            self._integrate_fn = self.integrate

    def _monitor(self, state: jnp.ndarray, step_size: float, control: jnp.ndarray, intervening: bool) -> bool:

        traj_states = self._integrate_fn(state, step_size, control)

        return bool(self._constraint_violation_fn(traj_states, self.params, self.constraint_enabled_dict))

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
