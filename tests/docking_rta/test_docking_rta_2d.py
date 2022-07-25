import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import os

from safe_autonomy_dynamics.cwh import M_DEFAULT, N_DEFAULT, generate_cwh_matrices
from run_time_assurance.zoo.cwh.docking_2d import Docking2dExplicitSwitchingRTA, Docking2dImplicitSwitchingRTA, \
                                                 Docking2dExplicitOptimizationRTA, Docking2dImplicitOptimizationRTA


theta_init = 4.298

class Env():
    def __init__(self, random_init=False):
        self.random_init = random_init

        self.dt = 1  # Time step
        self.u_max = 1  # Actuation constraint
        self.docking_region = 1  # m

        self.A, self.B = generate_cwh_matrices(M_DEFAULT, N_DEFAULT, mode="2d")

        # Specify LQR gains
        Q = np.eye(4) * 0.05  # State cost
        R = np.eye(2) * 1000  # Control cost

        # Solve ARE
        Xare = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, Q, R))
        self.Klqr = np.array(-scipy.linalg.inv(R)*(self.B.T*Xare))

    def u_des(self, x):
        # LQR to origin
        u = self.Klqr @ x
        return np.clip(u, -self.u_max, self.u_max)

    def reset(self):
        # Random point 10km away from origin
        if self.random_init:
            theta = np.random.rand()*2*np.pi
        else:
            theta = theta_init

        x = np.array([[10000*np.cos(theta)], [10000*np.sin(theta)], [0], [0]])
        return x, False

    def step(self, x, u):
        # Euler integration
        xd = self.A @ x + self.B @ u
        x1 = x + xd * self.dt

        if np.linalg.norm(x1[0:2, 0]) < self.docking_region:
            done = True
        else:
            done = False

        return x1, done

    def run_one_step(self):
        x, _ = self.reset()
        u_des = self.u_des(x)
        u_safe = np.vstack(self.rta.filter_control(x.flatten(), self.dt, u_des.flatten()))
        x, _ = self.step(x, u_safe)

    def run_episode(self, rta):
        self.rta = rta
        # Track time
        start_time = time.time()
        # Track initial values
        x, done = self.reset()
        array = [x.flatten()]
        control = [self.u_des(x).flatten()]
        intervening = [False]

        # Run episode
        while not done:
            # Get u_des
            u_des = self.u_des(x)
            # Use RTA
            u_safe = np.vstack(self.rta.filter_control(x.flatten(), self.dt, u_des.flatten()))
            # Take step using safe action
            x, done = self.step(x, u_safe)
            # Track values
            array = np.append(array, [x.flatten()], axis=0)
            control = np.append(control, [u_safe.flatten()], axis=0)
            intervening.append(self.rta.intervening)

        # Print final time, plot
        print(f"Simulation time: {time.time()-start_time:2.3f} sec")
        self.plotter(array, control, np.array(intervening))

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


plot_fig = True
save_fig = True
output_dir = 'figs/2d'

rtas = [Docking2dExplicitSwitchingRTA(), Docking2dImplicitSwitchingRTA(), 
        Docking2dExplicitOptimizationRTA(), Docking2dImplicitOptimizationRTA()]
output_names = ['rta_test_docking_2d_explicit_switching', 'rta_test_docking_2d_implicit_switching',
                'rta_test_docking_2d_explicit_optimization', 'rta_test_docking_2d_implicit_optimization']

env = Env()

os.makedirs(output_dir, exist_ok=True)

for rta, output_name in zip(rtas, output_names):
    env.run_episode(rta)
    if plot_fig:
        plt.show()
    if save_fig:
        plt.savefig(os.path.join(output_dir, output_name))

# env = Env()
# env.rta = Docking2dExplicitOptimizationRTA()
# env.run_one_step()
# import cProfile
# cProfile.run('env.run_one_step()', filename='docking2d.prof')
