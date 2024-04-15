import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import os

from safe_autonomy_simulation.spacecraft.utils import M_DEFAULT, N_DEFAULT, generate_cwh_matrices
from safe_autonomy_simulation.base_models import BaseLinearODESolverDynamics
from run_time_assurance.zoo.cwh.docking_2d import Docking2dExplicitSwitchingRTA, Docking2dImplicitSwitchingRTA, \
                                                 Docking2dExplicitOptimizationRTA, Docking2dImplicitOptimizationRTA
from run_time_assurance.utils.sample_testing import DataTrackingSampleTestingModule


class Env(DataTrackingSampleTestingModule):
    def __init__(self, rta, random_init=False):
        self.random_init = random_init

        self.u_max = 1  # Actuation constraint
        self.docking_region = 1  # m

        A, B = generate_cwh_matrices(M_DEFAULT, N_DEFAULT, mode="2d")
        self.dynamics = BaseLinearODESolverDynamics(A=A, B=B, integration_method='RK45')

        # Specify LQR gains
        Q = np.eye(4) * 0.05  # State cost
        R = np.eye(2) * 1000  # Control cost

        # Solve ARE
        Xare = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
        self.Klqr = np.array(-scipy.linalg.inv(R)*(B.T*Xare))

        super().__init__(rta=rta, simulation_time=5000, step_size=1, control_dim=2, state_dim=4)

    def _desired_control(self, state):
        # LQR to origin
        u = self.Klqr @ state
        return np.clip(u, -self.u_max, self.u_max)

    def _get_initial_state(self):
        # Random point 10km away from origin
        if self.random_init:
            theta = np.random.rand()*2*np.pi
        else:
            theta = 4.298

        x = np.array([10000*np.cos(theta), 10000*np.sin(theta), 0, 0])
        return x

    def _pred_state(self, state, step_size, control):
        # return next_state
        out, _ = self.dynamics.step(step_size, state, control)
        return out
    
    def _check_done_conditions(self, state, time):
        docking_done = np.linalg.norm(state[0:2]) < self.docking_region
        time_done = super()._check_done_conditions(state, time)
        return bool(docking_done or time_done)

    def plotter(self, array, control, intervening):
        fig = plt.figure(figsize=(15, 10))
        
        ax1 = fig.add_subplot(231)
        RTAon = np.ma.masked_where(intervening != 1, array[:, 1])
        ax1.plot(0, 0, 'k*', markersize=15)
        ax1.plot(array[0, 0], array[0, 1], 'r*', markersize=15)
        ax1.plot(array[:, 0], array[:, 1], 'b', linewidth=2)
        ax1.plot(array[:, 0], RTAon, 'c', linewidth=2)
        ax1.set_aspect('equal')
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.set_title('Trajectory')
        ax1.grid(True)

        ax2 = fig.add_subplot(232)
        v = np.empty([len(array), 2])
        for i in range(len(array)):
            v[i, :] = [np.linalg.norm(array[i, 0:2]), np.linalg.norm(array[i, 2:4])]
        xmax = np.max(v[:, 0])*1.1
        ymax = np.max(v[:, 1])*1.1
        RTAon = np.ma.masked_where(intervening != 1, v[:, 1])
        ax2.fill_between([0, xmax], [self.rta.v0, self.rta.v0 + self.rta.v1*xmax], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax2.fill_between([0, xmax], [0, 0], [self.rta.v0, self.rta.v0 + self.rta.v1*xmax], color=(244/255, 249/255, 241/255))
        ax2.plot([0, xmax], [self.rta.v0, self.rta.v0 + self.rta.v1*xmax], 'k--', linewidth=2)
        ax2.plot(v[:, 0], v[:, 1], 'b', linewidth=2)
        ax2.plot(v[:, 0], RTAon, 'c', linewidth=2)
        ax2.set_xlim([0, xmax])
        ax2.set_ylim([0, ymax])
        ax2.set_xlabel(r'$r_H$ [m]')
        ax2.set_ylabel(r'$v_H$ [m/s]')
        ax2.set_title('Distance Dependent Speed Limit')
        ax2.grid(True)

        ax3 = fig.add_subplot(233)
        ax3.plot(0, 0, 'k*', markersize=15)
        ax3.plot(0, 0, 'r*', markersize=15)
        ax3.plot(0, 0, 'k--', linewidth=2)
        ax3.plot(0, 0, 'b', linewidth=2)
        ax3.plot(0, 0, 'c', linewidth=2)
        ax3.legend(['Chief Position', 'Deputy Initial Position', 'Constraint', 'RTA Not Intervening', 'RTA Intervening'])
        ax3.axis('off')
        ax3.set_xlim([1, 2])
        ax3.set_ylim([1, 2])

        ax4 = fig.add_subplot(234)
        RTAonx = np.ma.masked_where(intervening != 1, array[:, 2])
        RTAony = np.ma.masked_where(intervening != 1, array[:, 3])
        xmax = len(array)*1.1
        ymax = self.rta.x_vel_limit*1.2
        ax4.fill_between([0, xmax], [self.rta.x_vel_limit, self.rta.x_vel_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax4.fill_between([0, xmax], [-ymax, -ymax], [self.rta.x_vel_limit, self.rta.x_vel_limit], color=(255/255, 239/255, 239/255))
        ax4.fill_between([0, xmax], [-self.rta.x_vel_limit, -self.rta.x_vel_limit], [self.rta.x_vel_limit, self.rta.x_vel_limit], color=(244/255, 249/255, 241/255))
        ax4.plot([0, xmax], [self.rta.x_vel_limit, self.rta.x_vel_limit], 'k--', linewidth=2)
        ax4.plot([0, xmax], [-self.rta.x_vel_limit, -self.rta.x_vel_limit], 'k--', linewidth=2)
        ax4.plot(range(len(array)), array[:, 2], 'b', linewidth=2)
        ax4.plot(range(len(array)), RTAonx, 'c', linewidth=2)
        ax4.plot(range(len(array)), array[:, 3], 'r', linewidth=2)
        ax4.plot(range(len(array)), RTAony, 'tab:orange', linewidth=2)
        ax4.set_xlim([0, xmax])
        ax4.set_ylim([-ymax, ymax])
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Velocity [m/s]')
        ax4.set_title('Max Velocity Constraint')
        ax4.grid(True)

        ax5 = fig.add_subplot(235)
        RTAonx = np.ma.masked_where(intervening != 1, control[:, 0])
        RTAony = np.ma.masked_where(intervening != 1, control[:, 1])
        ax5.plot([0, xmax], [1, 1], 'k--', linewidth=2)
        ax5.plot([0, xmax], [-1, -1], 'k--', linewidth=2)
        ax5.plot(range(len(control)), control[:, 0], 'b', linewidth=2)
        ax5.plot(range(len(control)), RTAonx, 'c', linewidth=2)
        ax5.plot(range(len(control)), control[:, 1], 'r', linewidth=2)
        ax5.plot(range(len(control)), RTAony, 'tab:orange', linewidth=2)
        ax5.set_xlim([0, xmax])
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Force [N]')
        ax5.set_title('Actions')
        ax5.grid(True)

        ax6 = fig.add_subplot(236)
        ax6.plot(0, 0, 'k--', linewidth=2)
        ax6.plot(0, 0, 'b', linewidth=2)
        ax6.plot(0, 0, 'c', linewidth=2)
        ax6.plot(0, 0, 'r', linewidth=2)
        ax6.plot(0, 0, 'tab:orange', linewidth=2)
        ax6.legend(['Constraint', r'$v_x/F_x$: RTA Not Intervening', r'$v_x/F_x$: RTA Intervening',
                    r'$v_y/F_y$: RTA Not Intervening', r'$v_y/F_y$: RTA Intervening'])
        ax6.axis('off')
        ax6.set_xlim([1, 2])
        ax6.set_ylim([1, 2])


if __name__ == '__main__':
    plot_fig = True
    save_fig = True
    output_dir = 'figs/2d'

    rtas = [Docking2dExplicitSwitchingRTA(), Docking2dImplicitSwitchingRTA(), 
            Docking2dExplicitOptimizationRTA(), Docking2dImplicitOptimizationRTA()]
    output_names = ['rta_test_docking_2d_explicit_switching', 'rta_test_docking_2d_implicit_switching',
                    'rta_test_docking_2d_explicit_optimization', 'rta_test_docking_2d_implicit_optimization']

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

    # env = Env(Docking2dExplicitOptimizationRTA())
    # env.run_one_step()
    # import cProfile
    # cProfile.run('env.run_one_step()', filename='docking2d.prof')
