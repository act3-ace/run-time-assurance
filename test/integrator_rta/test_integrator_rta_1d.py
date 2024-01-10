import numpy as np
import matplotlib.pyplot as plt
import time
import os

from safe_autonomy_dynamics.integrators import M_DEFAULT, generate_dynamics_matrices
from safe_autonomy_dynamics.base_models import BaseLinearODESolverDynamics
from run_time_assurance.zoo.integrators.integrator_1d import Integrator1dDockingExplicitSwitchingRTA, Integrator1dDockingImplicitSwitchingRTA, \
                                                 Integrator1dDockingExplicitOptimizationRTA, Integrator1dDockingImplicitOptimizationRTA
from run_time_assurance.utils.sample_testing import DataTrackingSampleTestingModule


class Env(DataTrackingSampleTestingModule):
    def __init__(self, rta):
        self.u_max = 1  # Actuation constraint
        self.docking_region = 0.1
        self.docking_max_vel = 0.1

        A, B = generate_dynamics_matrices(m=M_DEFAULT, mode='1d')
        self.dynamics = BaseLinearODESolverDynamics(A=A, B=B, integration_method='RK45')

        super().__init__(rta=rta, simulation_time=4, step_size=0.02, control_dim=1, state_dim=2)

    def _desired_control(self, state):
        return np.array([self.u_max])

    def _get_initial_state(self):
        return np.array([-1.75, 0.])

    def _pred_state(self, state, step_size, control):
        # return next_state
        out, _ = self.dynamics.step(step_size, state, control)
        return out
    
    def _check_done_conditions(self, state, time):
        docking_done = abs(state[0]) < self.docking_region and state[1] < self.docking_max_vel
        time_done = super()._check_done_conditions(state, time)
        return bool(docking_done or time_done)

    def plotter(self, array, control, intervening):
        fig = plt.figure(figsize=(20, 10))

        lim_x1 = np.linspace(-2, 0, 10000)
        lim_x2 = np.sqrt(-2*lim_x1)
        ax1 = fig.add_subplot(121)
        ax1.plot(lim_x1, lim_x2, 'k--', linewidth=2, label='Constraint')
        ax1.plot([0, 0], [-1, 0], 'k--', linewidth=2)
        ax1.plot(0, 0, 'k*', markersize=15, label='Desired State')
        ax1.fill_between(lim_x1, 0, lim_x2, color=(244/255, 249/255, 241/255)) # green
        ax1.fill_between(lim_x1, -1, 0, color=(244/255, 249/255, 241/255)) # green
        ax1.fill_between(lim_x1, lim_x2, 10, color=(255/255, 239/255, 239/255)) # red
        ax1.fill_between([0, 1], -1, 10, color=(255/255, 239/255, 239/255)) # red
        ax1.plot(array[0, 0], array[0, 1], 'b*', markersize=15, label='Initial State')
        ax1.plot(array[:, 0], array[:, 1], 'b', linewidth=2, label='Trajectory')
        ax1.set_xlim([-2, 0.1])
        ax1.set_ylim([-0.2, 2.2])
        ax1.set_xlabel(r'$x_1$ (position) [m]')
        ax1.set_ylabel(r'$x_2$ (velocity) [m/s]')
        ax1.set_title('Trajectory')
        ax1.grid(True)
        ax1.legend()

        ax2 = fig.add_subplot(122)
        xlim = len(control[:, 0]) * self.step_size * 1.1
        ax2.plot([0, xlim], [1, 1], 'k--', linewidth=2)
        ax2.plot([0, xlim], [-1, -1], 'k--', linewidth=2)
        ax2.fill_between([0, xlim], -1, 1, color=(244/255, 249/255, 241/255)) # green
        ax2.fill_between([0, xlim], -2, -1, color=(255/255, 239/255, 239/255)) # red
        ax2.fill_between([0, xlim], 1, 2, color=(255/255, 239/255, 239/255)) # red
        ax2.plot(np.array(range(len(control[:, 0])))*self.step_size, control[:, 0], 'b', linewidth=2)
        ax2.grid(True)
        ax2.set_xlim([0, xlim])
        ax2.set_ylim([-1.1, 1.1])
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Control [N]')
        ax2.set_title('Control')  


if __name__ == '__main__':
    plot_fig = True
    save_fig = True
    output_dir = 'figs/1d'

    rtas = [Integrator1dDockingExplicitSwitchingRTA(), Integrator1dDockingImplicitSwitchingRTA(), 
            Integrator1dDockingExplicitOptimizationRTA(), Integrator1dDockingImplicitOptimizationRTA()]
    output_names = ['rta_test_integrator_1d_explicit_switching', 'rta_test_integrator_1d_implicit_switching',
                    'rta_test_integrator_1d_explicit_optimization', 'rta_test_integrator_1d_implicit_optimization']

    os.makedirs(output_dir, exist_ok=True)

    for rta, output_name in zip(rtas, output_names):
        env = Env(rta)
        start_time = time.time()
        x, u, i = env.simulate_episode()
        print(f"Simulation time: {time.time()-start_time:2.3f} sec")
        env.plotter(x, u, i)
        if plot_fig:
            plt.show()
        if save_fig:
            plt.savefig(os.path.join(output_dir, output_name))
