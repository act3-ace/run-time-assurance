"""Module for example random sample tests for 2d docking"""

import numpy as np
import safe_autonomy_simulation.dynamics as dynamics
import safe_autonomy_simulation.sims.spacecraft.defaults as defaults

from run_time_assurance.rta.base import ConstraintBasedRTA
from run_time_assurance.utils.sample_testing import (
    LatinHypercubeRandomSampleTestingModule,
    ParametricRandomSampleTestingModule,
)
from run_time_assurance.zoo.cwh.docking_2d import Docking2dExplicitOptimizationRTA
from run_time_assurance.zoo.cwh.utils import generate_cwh_matrices


class Docking2DLatinHypercubeRandomSampleTest(LatinHypercubeRandomSampleTestingModule):
    """Docking 2d latin hypercube random sample test"""

    def __init__(
        self,
        rta: ConstraintBasedRTA = None,
        n_points: int = 100,
        simulation_time: float = 100.0,
        step_size: float = 1.0,
        control_dim: int = 2,
        state_dim: int = 4,
        bounds: np.ndarray = None,
        multiplier: int = 3,
    ):
        if rta is None:
            rta = Docking2dExplicitOptimizationRTA()
        if bounds is None:
            bounds = np.array([[-10000, -10000, -5, -5], [10000, 10000, 5, 5]])

        A, B = generate_cwh_matrices(defaults.M_DEFAULT, defaults.N_DEFAULT, mode="2d")
        self.dynamics = dynamics.LinearODEDynamics(A=A, B=B, integration_method="RK45", use_jax=True)

        super().__init__(
            rta=rta,
            n_points=n_points,
            simulation_time=simulation_time,
            step_size=step_size,
            control_dim=control_dim,
            state_dim=state_dim,
            bounds=bounds,
            multiplier=multiplier,
        )

    def _pred_state(
        self, state: np.ndarray, step_size: float, control: np.ndarray
    ) -> np.ndarray:
        next_state_vec, _ = self.dynamics.step(step_size, state, control)
        return next_state_vec


class Docking2DParametricRandomSampleTest(ParametricRandomSampleTestingModule):
    """Docking 2d parametric random sample test"""

    def __init__(
        self,
        rta: ConstraintBasedRTA = None,
        n_points: int = 100,
        simulation_time: float = 100.0,
        step_size: float = 1.0,
        control_dim: int = 2,
        state_dim: int = 4,
        state_pdfs: list = None,
        distribution_params: np.ndarray = None,
    ):
        if rta is None:
            rta = Docking2dExplicitOptimizationRTA()
        if state_pdfs is None:
            state_pdfs = ["gaussian", "gaussian", "gaussian", "gaussian"]
        if distribution_params is None:
            distribution_params = np.array([[0, 0, 0, 0], [5000, 5000, 2, 2]])

        A, B = generate_cwh_matrices(defaults.M_DEFAULT, defaults.N_DEFAULT, mode="2d")
        self.dynamics = dynamics.LinearODEDynamics(A=A, B=B, integration_method="RK45")

        super().__init__(
            rta=rta,
            n_points=n_points,
            simulation_time=simulation_time,
            step_size=step_size,
            control_dim=control_dim,
            state_dim=state_dim,
            state_pdfs=state_pdfs,
            distribution_params=distribution_params,
        )

    def _pred_state(
        self, state: np.ndarray, step_size: float, control: np.ndarray
    ) -> np.ndarray:
        next_state_vec, _ = self.dynamics.step(step_size, state, control)
        return next_state_vec


if __name__ == "__main__":
    mode = "LatinHypercube"
    if mode == "LatinHypercube":
        mc = Docking2DLatinHypercubeRandomSampleTest()
        mc.run_sample_test("tmp")
    else:
        mc_r = Docking2DParametricRandomSampleTest()
        mc_r.run_sample_test("tmp")
