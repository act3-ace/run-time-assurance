"""
This module implements sample testing modules.
Used to simulate one or many episodes with varying initial conditions, to track data or determine if RTA assures safety.
"""
from __future__ import annotations

import abc
import csv
import os
from datetime import datetime
from typing import Any, Union

import numpy as np
from scipy.stats import qmc

from run_time_assurance.rta.base import ConstraintBasedRTA, RTAModule
from run_time_assurance.utils import SolverError, to_jnp_array_jit


class BaseSampleTestingModule(abc.ABC):
    """Base Sample Testing module.
    Setup functions for running a simulation with RTA

    Parameters
    ----------
    simulation_time: float
        Total time to simulate each episode
    step_size: float
        Time step for the simulation
    control_dim: int
        Dimension of the control vector
    state_dim: int
        Dimension of the state vector
    """

    def __init__(self, simulation_time: float, step_size: float, control_dim: int, state_dim: int):
        self.simulation_time = simulation_time
        self.step_size = step_size
        self.control_dim = control_dim
        self.state_dim = state_dim

    @abc.abstractmethod
    def _get_initial_state(self) -> np.ndarray:
        """Get an initial state for the system

        Returns
        -------
        np.ndarray
            System state
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _pred_state(self, state: np.ndarray, step_size: float, control: np.ndarray) -> np.ndarray:
        """predict the next state of the system given the current state, step size, and control vector

        Parameters
        ----------
        state: np.ndarray
            Current system state
        step_size: float
            Simulation step size
        control: np.ndarray
            Control vector

        Returns
        -------
        np.ndarray
            Next state
        """
        raise NotImplementedError()

    def _desired_control(self, state: np.ndarray) -> np.ndarray:  # pylint: disable=unused-argument
        """Get the desired control for the system given the current state.

        Parameters
        ----------
        state: np.ndarray
            Current system state

        Returns
        -------
        np.ndarray
            Control vector. By default, returns zero control.
        """
        return np.zeros(self.control_dim)

    def _check_done_conditions(self, state: np.ndarray, time: float) -> bool:  # pylint: disable=unused-argument
        """Check terminal conditions for the episode.

        Parameters
        ----------
        state: np.ndarray
            Current system state
        time: float
            Current simulation time

        Returns
        -------
        bool
            True if done, False if not. By default, terminal condition is reaching maximum simulation time.
        """
        return bool(time >= self.simulation_time)


class DataTrackingSampleTestingModule(BaseSampleTestingModule):
    """Sample testing module for tracking data.

    Parameters
    ----------
    rta: RTAModule
        RTA module to test
    check_init_state: bool
        Resamples initial state if it is initially unsafe. Default True
    """

    def __init__(self, *args, rta: RTAModule, check_init_state: bool = True, **kwargs):
        self.rta = rta
        self.check_init_state = check_init_state
        super().__init__(*args, **kwargs)

    def simulate_episode(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulates one episode given the initial state.
        Runs the episode until "done", by default a time limit.
        Uses a desired controller, which by default returns zero control.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Array tracking the state vector at each time step.
            Array tracking the control vector at each time step.
            Array tracking if RTA is intervening at each time step.
        """

        if self.check_init_state:
            safe_state = False
            check_counter = 0
            while not safe_state:
                check_counter += 1
                if check_counter >= 100:
                    raise ValueError('Unable to find safe initial state. To disable, set check_init_state = False.')
                state = self._get_initial_state()
                safe_state = self.check_if_safe_state(state)
        else:
            state = self._get_initial_state()

        done = False
        current_time = 0.

        state_array = [state]
        control_array = [np.zeros(self.control_dim)]
        intervening_array = [False]

        while not done:
            desired_control = self._desired_control(state)
            rta_control = self.rta.filter_control(state, self.step_size, desired_control)
            state = self._pred_state(state, self.step_size, rta_control)
            current_time += self.step_size
            done = self._check_done_conditions(state, current_time)
            self._update_status(state, current_time)

            state_array.append(state)
            control_array.append(rta_control)
            intervening_array.append(self.rta.intervening)

        return np.array(state_array), np.array(control_array), np.array(intervening_array)

    def run_one_step(self):
        """Run one step, useful for profiling"""
        state = self._get_initial_state()
        desired_control = self._desired_control(state)
        rta_control = self.rta.filter_control(state, self.step_size, desired_control)
        state = self._pred_state(state, self.step_size, rta_control)

    def _update_status(self, state: np.ndarray, time: float):  # pylint: disable=unused-argument
        """Update sim variables at each step, if desired

        Parameters
        ----------
        state: np.ndarray
            Current system state
        time: float
            Current simulation time
        """

    def check_if_safe_state(self, state: np.ndarray):
        """
        Determines if state is safe or not.
        Overwrite this method when checking safety for non ContraintBasedRTA modules.

        Parameters
        ----------
        state: np.ndarray
            Current system state

        Returns
        -------
        bool
            True if state is safe, False if not.
        """
        assert isinstance(self.rta, ConstraintBasedRTA
                          ), ("Must use constraint based rta with default check_state method. To disable, set check_init_state = False.")
        init_state_safe = True
        for c in self.rta.constraints.values():
            if c.phi(to_jnp_array_jit(state), c.params) < 0 or c(to_jnp_array_jit(state), c.params) < 0:
                init_state_safe = False
                break
        return init_state_safe


class SafetyAssuranceSampleTestingModule(BaseSampleTestingModule):
    """Sample testing module for determining if RTA assures safety.
    Runs many simulations and saves results

    Parameters
    ----------
    rta: ConstraintBasedRTA
        RTA module to test. Must be constraint based RTA.
    n_points: int
        Number of episodes to simulate
    """

    def __init__(self, *args, rta: ConstraintBasedRTA, n_points: int, **kwargs):
        self.rta = rta
        self.n_points = n_points
        super().__init__(*args, **kwargs)

    def run_sample_test(self, output_dir: str):
        """Run a sample test experiment.
        For all points, gets an initial state and runs an episode.
        Resamples states if they initially violate safety.
        Saves all results in a csv, where each initial condition is listed along with bool for RTA assuring safety or not.

        Parameters
        ----------
        output_dir: str
            Path to directory to save the results as csv.
        """
        results_array: list = []

        os.makedirs(output_dir, exist_ok=True)

        for n in range(self.n_points):
            rta_assured_safety_bool = None
            while rta_assured_safety_bool is None:
                initial_state = self._get_initial_state()
                initial_state = self._transform_state(initial_state)
                rta_assured_safety_bool = self._simulate_episode(initial_state)

            results_array.append([rta_assured_safety_bool, initial_state.tolist()])

            if ((((n + 1) / self.n_points) * 100) % 10) == 0:
                print(str(round(((n + 1) / self.n_points) * 100, 4)), '% complete')

        filename = datetime.now().strftime('%Y-%m-%d--%H-%M-%S') + ".csv"
        with open(os.path.join(output_dir, filename), "w+", encoding="utf-8") as my_csv:
            newarray = csv.writer(my_csv, delimiter=',')
            newarray.writerows(results_array)

    def _simulate_episode(self, state: np.ndarray) -> Union[bool, None]:
        """Simulates one episode given the initial state.
        Runs the episode until "done", by default a time limit.
        Uses a desired controller, which by default returns zero control.

        Parameters
        ----------
        state: np.ndarray
            Initial state for the episode to begin

        Returns
        -------
        Union[bool, None]
            Returns True if RTA assures safety for the entire episode.
            Returns False if RTA does not assure safety.
            Returns None if the initial state is unsafe.
        """

        for c in self.rta.constraints.values():
            if c.phi(to_jnp_array_jit(state)) < 0 or c(to_jnp_array_jit(state)) < 0:
                return None

        done = False
        current_time = 0.

        while not done:
            desired_control = self._desired_control(state)
            try:
                rta_control = self.rta.filter_control(state, self.step_size, desired_control)
            except SolverError:
                return False
            state = self._pred_state(state, self.step_size, rta_control)
            current_time += self.step_size
            done = self._check_done_conditions(state, current_time)

            for c in self.rta.constraints.values():
                if c.phi(to_jnp_array_jit(state)) < 0:
                    return False

        return True

    def _transform_state(self, state: np.ndarray) -> np.ndarray:
        """Transform the state if necessary

        Parameters
        ----------
        state: np.ndarray
            Current system state

        Returns
        -------
        np.ndarray
            Transformed state. By default no transformation.
        """
        return state


class LatinHypercubeRandomSampleTestingModule(SafetyAssuranceSampleTestingModule):
    """Latin Hypercube Random Sample Testing.

    Parameters
    ----------
    bounds: np.ndarray
        Array of lower and upper bounds corresponding to each state component. Size (2, state_dim).
    multiplier: int
        Multiplier for the total number of steps. Used to initialize more points, in the event that some are initially unsafe
    """

    def __init__(self, *args: Any, bounds: np.ndarray, multiplier: int, **kwargs: Any):
        super().__init__(*args, **kwargs)

        sampler = qmc.LatinHypercube(d=self.state_dim)
        sample = sampler.random(n=self.n_points * multiplier)
        self.all_ics = qmc.scale(sample, bounds[0, :], bounds[1, :])
        self.idx = 0

    def _get_initial_state(self) -> np.ndarray:
        """Get an initial state for the system

        Returns
        -------
        np.ndarray
            System state
        """
        initial_state = self.all_ics[self.idx]
        self.idx += 1
        return initial_state


class ParametricRandomSampleTestingModule(SafetyAssuranceSampleTestingModule):
    """Parametric Random Sample Testing.

    Parameters
    ----------
    state_pdfs: list
        List of strings for probability distribution functions corresponding to each state component. len: state_dim.
        Must be either "gaussian", "constant", or "uniform".
    distribution_params: np.ndarray
        Array of distribution parameters corresponding to each state component. Size (2, state_dim).
        gaussian: (mean, standard deviation)
        constant: (value, None)
        uniform:  (lower bounds, upper bound)
    """

    def __init__(self, *args: Any, state_pdfs: list, distribution_params: np.ndarray, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.state_pdfs = state_pdfs
        self.distribution_params = distribution_params

    def _get_initial_state(self) -> np.ndarray:
        """Get an initial state for the system

        Returns
        -------
        np.ndarray
            System state
        """
        initial_state = np.zeros(self.state_dim)
        for i in range(self.state_dim):
            if self.state_pdfs[i] == 'gaussian':
                initial_state[i] = np.random.normal(self.distribution_params[0][i], self.distribution_params[1][i], 1)
            elif self.state_pdfs[i] == 'constant':
                initial_state[i] = np.array([self.distribution_params[0][i]])
            elif self.state_pdfs[i] == 'uniform':
                initial_state[i] = np.random.uniform(self.distribution_params[0][i], self.distribution_params[1][i])
            else:
                raise ValueError('state_PDFs must be either gaussian, constant, or uniform')
        return initial_state
