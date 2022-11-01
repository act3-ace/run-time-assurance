import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
from collections import OrderedDict
import glob
import os
import csv

from safe_autonomy_dynamics.cwh import M_DEFAULT, N_DEFAULT, generate_cwh_matrices
from run_time_assurance.zoo.cwh.inspection_1v1 import U_MAX_DEFAULT, Inspection1v1RTA, SwitchingFuelLimitRTA, InspectionCascadedRTA
from run_time_assurance.utils import to_jnp_array_jit, SolverError

# from jax.config import config
# config.update('jax_disable_jit', True)


class Env():
    def __init__(self, rta, dt=1, time=3000, constraint_keys=[]):
        self.rta = rta
        self.dt = dt  # Time step
        self.time = time # Total sim time
        self.u_max = U_MAX_DEFAULT  # Actuation constraint
        self.inspection_rta = Inspection1v1RTA()

        # Update constraints
        new_constraints = OrderedDict()
        for k in constraint_keys:
            new_constraints[k] = self.rta.constraints[k]
        if len(new_constraints) != 0:
            self.rta.constraints = new_constraints

        # Generate dynamics matrices
        self.A, self.B = generate_cwh_matrices(M_DEFAULT, N_DEFAULT, mode="3d")

        # Specify LQR gains
        Q = np.eye(6) * 0.05  # State cost
        R = np.eye(3) * 1000  # Control cost

        # Solve ARE
        Xare = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, Q, R))
        # Get LQR gain
        self.Klqr = np.array(-scipy.linalg.inv(R)*(self.B.T*Xare))

    def u_des(self, x, x_des=np.array([0, 0, 0, 0, 0, 0, 0, 0])):
        # LQR to desired state
        u = self.Klqr @ (x[0:6] - x_des[0:6])
        return np.clip(u, -self.u_max, self.u_max)

    def reset(self):
        theta = np.random.rand(2) * 2 * np.pi
        p = np.array([np.cos(theta[0])*np.sin(theta[1]), np.sin(theta[0])*np.sin(theta[1]), np.cos(theta[1])]) * 100
        v = np.array([0, 0, 0])
        return np.concatenate((p, v, np.array([0, 0])))

    def run_episode(self, plotter=True, init_state=None):
        # Track time
        start_time = time.time()
        # Track initial values
        if init_state is None:
            x = self.reset()
        else:
            x = init_state

        for c in self.rta.constraints.values():
            if c.phi(x) < 0 or c(x) < 0:
                print("Initial state unsafe")
                return None
        if plotter:
            # Data tracking arrays
            array = [x]
            control = [np.zeros(3)]
            intervening = [False]

        # Run episode
        for t in range(int(self.time/self.dt)):
            if t < 1000:
                x_des = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
            else:
                x_des = np.array([-1500*np.cos(x[6]), -1500*np.sin(x[6]), 0., 0., 0., 0., 0., 0.])
            if init_state is None:
                u_des = self.u_des(x, x_des)
            else:
                u_des = np.array([0, 0, 0])
            u_safe = self.rta.filter_control(x, self.dt, u_des)
            # Take step using safe action
            x = np.array(self.inspection_rta._pred_state_fn(to_jnp_array_jit(x), self.dt, to_jnp_array_jit(u_safe)))
            if plotter:
                # Track data
                array = np.append(array, [x], axis=0)
                control = np.append(control, [u_safe], axis=0)
                intervening.append(self.rta.intervening)

            # for k, c in self.rta.constraints.items():
            #     if c.phi(to_jnp_array_jit(x)) < 0:
            #         print(t, k, c.phi(to_jnp_array_jit(x)))

        # Print final time, plot
        print(f"Simulation time: {time.time()-start_time:2.3f} sec")
        if plotter:
            self.plotter(array, control, np.array(intervening))

    def run_mc(self, x):
        self.rta.solver_exception = True
        # Check if initial state is safe
        for c in self.rta.constraints.values():
            if c.phi(x) < 0 or c(x) < 0:
                # print("Initial state unsafe")
                return None
        
        # Run episode
        for t in range(int(self.time/self.dt)):
            # Use RTA
            try:
                act =  np.array([0, 0, 0])
                # act = np.random.rand(3)*2-1
                # act = self.u_des(x)
                u = self.rta.filter_control(x, self.dt, act)
            except SolverError:
                print('Solver failed')
                return False
            # Take step using safe action (**Euler integration**)
            x = np.array(self.inspection_rta._pred_state_fn(to_jnp_array_jit(x), self.dt, to_jnp_array_jit(u)))
            # Check if current state is safe
            for c in self.rta.constraints.values():
                if c.phi(x) < 0:
                    # print("Current state unsafe, RTA failed")
                    return False
            
        # If all states are safe, returns True
        return True

    def plotter(self, array, control, intervening, paper_plot=False, fast_plot=True):
        if not paper_plot:
            fig = plt.figure(figsize=(15, 15))
            ax1 = fig.add_subplot(331, projection='3d')
            ax2 = fig.add_subplot(332)
            ax3 = fig.add_subplot(333)
            ax4 = fig.add_subplot(334)
            ax5 = fig.add_subplot(335)
            ax6 = fig.add_subplot(336)
            ax7 = fig.add_subplot(337)
            ax8 = fig.add_subplot(338)
            ax9 = fig.add_subplot(339)
            lw = 2
        else:
            plt.rcParams.update({'font.size': 30, 'text.usetex': True, 'figure.figsize': [6.4, 6]})
            lw = 4
            hp = 0.1
        
        if not fast_plot:
            if paper_plot:
                fig = plt.figure()
                ax1 = fig.add_subplot(111, projection='3d')
            for i in range(len(array)-1):
                ax1.plot(array[i:i+2, 0], array[i:i+2, 1], array[i:i+2, 2], color=plt.cm.cool(i/len(array)), linewidth=lw)
                ax1.plot(1000*np.cos(array[i:i+2, 6]), 1000*np.sin(array[i:i+2, 6]), [0, 0], color=plt.cm.Wistia(i/len(array)), linewidth=lw)
            # max = np.max(np.abs(array[:, 0:3]))*1.1
            max = 1100
            ax1.plot(0, 0, 0, 'k*', markersize=15)
            ax1.set_xlabel(r'$x$ [m]')
            ax1.set_ylabel(r'$y$ [m]')
            ax1.set_zlabel(r'$z$ [m]')
            ax1.set_xlim([-max, max])
            ax1.set_ylim([-max, max])
            ax1.set_zlim([-max, max])
            ax1.set_box_aspect((1,1,1))
            ax1.grid(True)
            if paper_plot:
                plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax2 = fig.add_subplot(111)
        v = np.empty([len(array), 2])
        for j in range(len(array)):
            v[j, :] = [np.linalg.norm(array[j, 0:3]), np.linalg.norm(array[j, 3:6])]
        ax2.plot(range(len(array)), v[:, 0], linewidth=lw)
        xmax = len(array)*1.1
        ymax = np.max(v[:, 0])*1.1
        ax2.fill_between([0, xmax], [self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius, self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius], [ymax, ymax], color=(244/255, 249/255, 241/255))
        ax2.fill_between([0, xmax], [0, 0], [self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius, self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius], color=(255/255, 239/255, 239/255))
        ax2.plot([0, xmax], [self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius, self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius], 'k--', linewidth=lw)
        ax2.set_xlim([0, xmax])
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel(r'Relative Dist. ($\vert \vert \mathbf{p} \vert \vert_2$) [m]')
        ax2.set_yscale('log')
        ax2.set_ylim([6, ymax])
        ax2.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax3 = fig.add_subplot(111)
        xmax = np.max(v[:, 0])*1.1
        ymax = np.max(v[:, 1])*1.1
        ax3.plot(v[:, 0], v[:, 1], linewidth=lw)
        ax3.fill_between([0, xmax], [self.inspection_rta.v0, self.inspection_rta.v0 + self.inspection_rta.v1*xmax], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax3.fill_between([0, xmax], [0, 0], [self.inspection_rta.v0, self.inspection_rta.v0 + self.inspection_rta.v1*xmax], color=(244/255, 249/255, 241/255))
        ax3.plot([0, xmax], [self.inspection_rta.v0, self.inspection_rta.v0 + self.inspection_rta.v1*xmax], 'k--', linewidth=lw)
        ax3.set_xlim([0, xmax])
        ax3.set_ylim([0, ymax])
        ax3.set_xlabel(r'Relative Dist. ($\vert \vert \mathbf{p}  \vert \vert_2$) [m]')
        ax3.set_ylabel(r'Relative Vel. ($\vert \vert \mathbf{v}  \vert \vert_2$) [m/s]')
        ax3.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax4 = fig.add_subplot(111)
        xmax = len(array)*1.1
        ymax = np.maximum(self.inspection_rta.fuel_limit, np.max(array[:, 7]))*1.1
        ax4.plot(range(len(array)), array[:, 7], linewidth=lw)
        ax4.fill_between([0, xmax], [self.inspection_rta.fuel_limit, self.inspection_rta.fuel_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax4.fill_between([0, xmax], [0, 0], [self.inspection_rta.fuel_limit, self.inspection_rta.fuel_limit], color=(244/255, 249/255, 241/255))
        ax4.plot([0, xmax], [self.inspection_rta.fuel_limit, self.inspection_rta.fuel_limit], 'k--', linewidth=lw)
        ax4.plot([0, xmax], [self.inspection_rta.fuel_switching_threshold, self.inspection_rta.fuel_switching_threshold], 'r--', linewidth=lw)
        ax4.set_xlim([0, xmax])
        ax4.set_ylim([0, ymax])
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel(r'Fuel Used ($\dot{m}_f$) [kg]')
        ax4.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax5 = fig.add_subplot(111)
        th = self.inspection_rta.fov/2*180/np.pi
        xmax = len(array)*1.1
        h = np.zeros(len(array))
        for j in range(len(array)):
            r_s_hat = np.array([np.cos(array[j, 6]), np.sin(array[j, 6]), 0.])
            r_b_hat = -array[j, 0:3]/np.linalg.norm(array[j, 0:3])
            h[j] = np.arccos(np.dot(r_s_hat, r_b_hat))*180/np.pi
        ymax = np.max(h)*1.1
        ax5.plot(range(len(array)), h, linewidth=lw)
        ax5.fill_between([0, xmax], [th, th], [ymax, ymax], color=(244/255, 249/255, 241/255))
        ax5.fill_between([0, xmax], [0, 0], [th, th], color=(255/255, 239/255, 239/255))
        ax5.plot([0, xmax], [th, th], 'k--', linewidth=lw)
        ax5.set_xlim([0, xmax])
        ax5.set_ylim([0, ymax])
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel(r'Angle to Sun ($\theta_{EZ}}$) [degrees]')
        ax5.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax6 = fig.add_subplot(111)
        ax6.plot(range(len(array)), v[:, 0], linewidth=lw)
        xmax = len(array)*1.1
        ymax = np.maximum(np.max(v[:, 0])*1.1, self.inspection_rta.r_max*1.1)
        ax6.fill_between([0, xmax], [self.inspection_rta.r_max, self.inspection_rta.r_max], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax6.fill_between([0, xmax], [0, 0], [self.inspection_rta.r_max, self.inspection_rta.r_max], color=(244/255, 249/255, 241/255))
        ax6.plot([0, xmax], [self.inspection_rta.r_max, self.inspection_rta.r_max], 'k--', linewidth=lw)
        ax6.set_xlim([0, xmax])
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel(r'Relative Dist. ($\vert \vert \mathbf{p} \vert \vert_2$) [m]')
        ax6.set_ylim([0, ymax])
        ax6.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        # h = []
        # phi = []
        # for i in range(len(array)):
        #     h.append(self.inspection_rta.constraints["sun"](array[i]))
        #     phi.append(self.inspection_rta.constraints["sun"].phi(array[i]))
        # ax7.plot(range(len(h)), h, linewidth=lw, label='h')
        # ax7.plot(range(len(phi)), phi, linewidth=lw, label='phi')
        # ax7.grid(True)
        # ax7.legend()

        if not fast_plot:
            if paper_plot:
                fig = plt.figure()
                ax7 = fig.add_subplot(111)
            for i in range(0, len(array), 3):
                r = self.inspection_rta.constraints["PSM"].get_array(array[i])+self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius
                ax7.plot(range(i, len(r)+i), r, 'c', linewidth=0.5)
            ax7.plot(range(len(array)), v[:, 0], linewidth=lw)
            xmax = len(array)*1.1
            ymax = np.max(v[:, 0])*1.1
            ax7.fill_between([0, xmax], [self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius, self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius], [ymax, ymax], color=(244/255, 249/255, 241/255))
            ax7.fill_between([0, xmax], [0, 0], [self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius, self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius], color=(255/255, 239/255, 239/255))
            ax7.plot([0, xmax], [self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius, self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius], 'k--', linewidth=lw)
            ax7.set_xlim([0, xmax])
            ax7.set_xlabel('Time [s]')
            ax7.set_ylabel(r'Relative Dist. ($\vert \vert \mathbf{p} \vert \vert_2$) [m]')
            ax7.set_yscale('log')
            ax7.set_ylim([6, ymax])
            ax7.grid(True)
            if paper_plot:
                plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax8 = fig.add_subplot(111)
        xmax = len(array)*1.1
        ymax = self.inspection_rta.vel_limit*1.2
        ax8.plot(range(len(array)), array[:, 3], linewidth=lw, label=r'$\dot{x}$')
        ax8.plot(range(len(array)), array[:, 4], linewidth=lw, label=r'$\dot{y}$')
        ax8.plot(range(len(array)), array[:, 5], linewidth=lw, label=r'$\dot{z}$')
        ax8.fill_between([0, xmax], [self.inspection_rta.vel_limit, self.inspection_rta.vel_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax8.fill_between([0, xmax], [-ymax, -ymax], [self.inspection_rta.vel_limit, self.inspection_rta.vel_limit], color=(255/255, 239/255, 239/255))
        ax8.fill_between([0, xmax], [-self.inspection_rta.vel_limit, -self.inspection_rta.vel_limit], [self.inspection_rta.vel_limit, self.inspection_rta.vel_limit], color=(244/255, 249/255, 241/255))
        ax8.plot([0, xmax], [self.inspection_rta.vel_limit, self.inspection_rta.vel_limit], 'k--', linewidth=lw)
        ax8.plot([0, xmax], [-self.inspection_rta.vel_limit, -self.inspection_rta.vel_limit], 'k--', linewidth=lw)
        ax8.set_xlim([0, xmax])
        ax8.set_ylim([-ymax, ymax])
        ax8.set_xlabel('Time [s]')
        ax8.set_ylabel(r'Velocity ($\mathbf{v}$) [m/s]')
        ax8.grid(True)
        ax8.legend()
        if paper_plot:
            plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax9 = fig.add_subplot(111)
        xmax = len(array)*1.1
        ymax = self.inspection_rta.u_max*1.2
        ax9.plot(range(len(control)), control[:, 0], linewidth=lw, label=r'$F_x$')
        ax9.plot(range(len(control)), control[:, 1], linewidth=lw, label=r'$F_y$')
        ax9.plot(range(len(control)), control[:, 2], linewidth=lw, label=r'$F_z$')
        ax9.fill_between([0, xmax], [self.inspection_rta.u_max, self.inspection_rta.u_max], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax9.fill_between([0, xmax], [-ymax, -ymax], [self.inspection_rta.u_max, self.inspection_rta.u_max], color=(255/255, 239/255, 239/255))
        ax9.fill_between([0, xmax], [-self.inspection_rta.u_max, -self.inspection_rta.u_max], [self.inspection_rta.u_max, self.inspection_rta.u_max], color=(244/255, 249/255, 241/255))
        ax9.plot([0, xmax], [self.inspection_rta.u_max, self.inspection_rta.u_max], 'k--', linewidth=lw)
        ax9.plot([0, xmax], [-self.inspection_rta.u_max, -self.inspection_rta.u_max], 'k--', linewidth=lw)
        ax9.set_xlim([0, xmax])
        ax9.set_ylim([-ymax, ymax])
        ax9.set_xlabel('Time [s]')
        ax9.set_ylabel(r'$\mathbf{u}$ [N]')
        ax9.grid(True)
        ax9.legend()
        if paper_plot:
            plt.tight_layout(pad=hp)

        if not paper_plot:
            ax1.set_title('Trajectory')
            ax2.set_title('Safe Separation')
            ax3.set_title('Dynamic Speed Constraint')
            ax4.set_title('Fuel Limit')
            ax5.set_title('Keep Out Zone (Sun Avoidance)')
            ax6.set_title('Keep In Zone')
            ax7.set_title('Passively Safe Maneuvers')
            ax8.set_title('Velocity Limits')
            ax9.set_title('Actuation Constraints')
            fig.tight_layout(h_pad=2)


if __name__ == '__main__':
    rta = Inspection1v1RTA()
    # rta = SwitchingFuelLimitRTA()
    # rta = InspectionCascadedRTA()
    env = Env(rta)

    env.run_episode()
    plt.show()

    # list_of_files = glob.glob('*.csv')
    # latest_file = max(list_of_files, key=os.path.getctime)
    # with open(latest_file, newline='') as csvfile:
    #     array = csv.reader(csvfile, delimiter=',')
    #     for row in array:
    #         if row[0] == 'False':
    #             x = row[1]
    #             x = x.replace('[', '')
    #             x = x.replace(']', '')
    #             x = x.replace(',', '')
    #             x = np.array([float(val) for val in x.split()])
    #             env.run_episode(init_state=x)
    #             plt.show()
