import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
from collections import OrderedDict
import os

from safe_autonomy_simulation.sims.spacecraft.defaults import M_DEFAULT, N_DEFAULT
from safe_autonomy_simulation.dynamics import LinearODEDynamics
from run_time_assurance.zoo.cwh.utils import generate_cwh_matrices
from run_time_assurance.zoo.cwh.inspection_1v1 import U_MAX_DEFAULT, Inspection1v1RTA, InspectionCascadedRTA, DiscreteInspection1v1RTA
from run_time_assurance.utils.sample_testing import DataTrackingSampleTestingModule
from run_time_assurance.utils import to_jnp_array_jit

# from jax.config import config
# config.update('jax_disable_jit', True)


class Env(DataTrackingSampleTestingModule):
    def __init__(self, rta, constraint_keys=[], step_size=1, **kwargs):
        self.u_max = U_MAX_DEFAULT  # Actuation constraint
        self.inspection_rta = Inspection1v1RTA()
        self.state_des = np.array([0, 0, 0, 0, 0, 0])

        # Update constraints
        new_constraints = OrderedDict()
        for k in constraint_keys:
            new_constraints[k] = self.rta.constraints[k]
        if len(new_constraints) != 0:
            self.rta.constraints = new_constraints

        A, B = generate_cwh_matrices(M_DEFAULT, N_DEFAULT, mode="3d")
        self.dynamics = LinearODEDynamics(A=A, B=B, integration_method='RK45', use_jax=True)

        # Specify LQR gains
        Q = np.eye(6) * 0.05  # State cost
        R = np.eye(3) * 1000  # Control cost

        # Solve ARE
        Xare = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
        self.Klqr = np.array(-scipy.linalg.inv(R)*(B.T*Xare))

        super().__init__(rta=rta, simulation_time=3000, step_size=step_size, control_dim=3, state_dim=8, **kwargs)

    def _desired_control(self, state):
        # LQR to origin
        u = self.Klqr @ (state[0:6] - self.state_des)
        return np.clip(u, -self.u_max, self.u_max)

    def _get_initial_state(self):
        # Random point 10km away from origin
        theta = np.random.rand(2) * 2 * np.pi

        p = np.array([np.cos(theta[0])*np.sin(theta[1]), np.sin(theta[0])*np.sin(theta[1]), np.cos(theta[1])]) * 100
        v = np.array([0, 0, 0])
        return np.concatenate((p, v, np.array([0, 0])))

    def _pred_state(self, state, step_size, control):
        out = np.array(self.inspection_rta._pred_state_fn(to_jnp_array_jit(state), step_size, to_jnp_array_jit(control)))
        return out
    
    def _update_status(self, state: np.ndarray, time: float):
        if time >= 1000:
            self.state_des = np.array([-1500*np.cos(state[6]), -1500*np.sin(state[6]), 0., 0., 0., 0.])

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
            plt.rcParams.update({'font.size': 25, 'text.usetex': True})
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
            ax1.set_xlabel(r'$x$ [km]')
            ax1.set_ylabel(r'$y$ [km]')
            ax1.set_zlabel(r'$z$ [km]')
            ax1.set_xticks([-1000, 0, 1000])
            ax1.set_yticks([-1000, 0, 1000])
            ax1.set_zticks([-1000, 0, 1000])
            ax1.set_xticklabels([-1, 0, 1])
            ax1.set_yticklabels([-1, 0, 1])
            ax1.set_zticklabels([-1, 0, 1])
            ax1.set_xlim([-max, max])
            ax1.set_ylim([-max, max])
            ax1.set_zlim([-max, max])
            ax1.set_box_aspect((1,1,1))
            ax1.grid(True)
            if paper_plot:
                ax1.xaxis.labelpad = 10
                ax1.yaxis.labelpad = 10
                ax1.zaxis.labelpad = 10
                plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax2 = fig.add_subplot(111)
            if not fast_plot:
                for i in range(0, len(array), 3):
                    r = self.inspection_rta.constraints["PSM"].get_array(array[i])+self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius
                    ax2.plot(range(i, len(r)+i), r, 'c', linewidth=0.5)
            v = np.empty([len(array), 2])
            for j in range(len(array)):
                v[j, :] = [np.linalg.norm(array[j, 0:3]), np.linalg.norm(array[j, 3:6])]
            ax2.plot(range(0, len(array)*self.step_size, self.step_size), v[:, 0], linewidth=lw, label=r'$\mathbf{p}_{act}$')
            ax2.plot(-1, -1, 'c', linewidth=0.5, label=r'$\mathbf{p}_{NM}$')
            xmax = len(array)*1.1*self.step_size
            ymax = self.inspection_rta.r_max * 1.4
            ax2.fill_between([0, xmax], [self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius, self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius], [self.inspection_rta.r_max, self.inspection_rta.r_max], color=(244/255, 249/255, 241/255))
            ax2.fill_between([0, xmax], [0, 0], [self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius, self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius], color=(255/255, 239/255, 239/255))
            ax2.fill_between([0, xmax], [self.inspection_rta.r_max, self.inspection_rta.r_max], [ymax, ymax], color=(255/255, 239/255, 239/255))
            ax2.plot([0, xmax], [self.inspection_rta.r_max, self.inspection_rta.r_max], 'k--', linewidth=lw)
            ax2.plot([0, xmax], [self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius, self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius], 'k--', linewidth=lw)
            ax2.set_xlim([0, xmax])
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel(r'Relative Dist. ($\vert \vert \mathbf{p} \vert \vert_2$) [m]')
            ax2.set_yscale('log')
            ax2.set_ylim([6, ymax])
            ax2.grid(True)
            ax2.legend()
            plt.tight_layout(pad=hp)
        if not paper_plot:
            v = np.empty([len(array), 2])
            for j in range(len(array)):
                v[j, :] = [np.linalg.norm(array[j, 0:3]), np.linalg.norm(array[j, 3:6])]
            ax2.plot(range(0, len(array)*self.step_size, self.step_size), v[:, 0], linewidth=lw)
            xmax = len(array)*1.1*self.step_size
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
            fig = plt.figure()
            ax3 = fig.add_subplot(111)
        xmax = np.max(v[:, 0])*1.1
        ymax = np.max(v[:, 1])*1.1
        ax3.plot(v[:, 0], v[:, 1], linewidth=lw)
        ax3.fill_between([self.inspection_rta.v0_distance, xmax], [self.inspection_rta.v0, self.inspection_rta.v0 + self.inspection_rta.v1*(xmax-self.inspection_rta.v0_distance)], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax3.fill_between([self.inspection_rta.v0_distance, xmax], [0, 0], [self.inspection_rta.v0, self.inspection_rta.v0 + self.inspection_rta.v1*(xmax-self.inspection_rta.v0_distance)], color=(244/255, 249/255, 241/255))
        ax3.plot([self.inspection_rta.v0_distance, xmax], [self.inspection_rta.v0, self.inspection_rta.v0 + self.inspection_rta.v1*(xmax-self.inspection_rta.v0_distance)], 'k--', linewidth=lw)
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
        xmax = len(array)*1.1*self.step_size
        ymax = np.maximum(self.inspection_rta.delta_v_limit, np.max(array[:, 7]))*1.1
        ax4.plot(range(0, len(array)*self.step_size, self.step_size), array[:, 7], linewidth=lw)
        ax4.fill_between([0, xmax], [self.inspection_rta.delta_v_limit, self.inspection_rta.delta_v_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax4.fill_between([0, xmax], [0, 0], [self.inspection_rta.delta_v_limit, self.inspection_rta.delta_v_limit], color=(244/255, 249/255, 241/255))
        ax4.plot([0, xmax], [self.inspection_rta.delta_v_limit, self.inspection_rta.delta_v_limit], 'k--', linewidth=lw)
        ax4.set_xlim([0, xmax])
        ax4.set_ylim([0, ymax])
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel(r'$\Delta$ V Used [m/s]')
        ax4.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax5 = fig.add_subplot(111)
        th = self.inspection_rta.fov/2*180/np.pi
        xmax = len(array)*1.1*self.step_size
        h = np.zeros(len(array))
        for j in range(len(array)):
            r_s_hat = np.array([np.cos(array[j, 6]), np.sin(array[j, 6]), 0.])
            r_b_hat = -array[j, 0:3]/np.linalg.norm(array[j, 0:3])
            h[j] = np.arccos(np.dot(r_s_hat, r_b_hat))*180/np.pi
        ymax = np.max(h)*1.1
        ax5.plot(range(0, len(array)*self.step_size, self.step_size), h, linewidth=lw)
        ax5.fill_between([0, xmax], [th, th], [ymax, ymax], color=(244/255, 249/255, 241/255))
        ax5.fill_between([0, xmax], [0, 0], [th, th], color=(255/255, 239/255, 239/255))
        ax5.plot([0, xmax], [th, th], 'k--', linewidth=lw)
        ax5.set_xlim([0, xmax])
        ax5.set_ylim([0, ymax])
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel(r'Angle to Sun ($\theta_{EZ}$) [degrees]')
        ax5.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        if not paper_plot:
            ax6.plot(range(0, len(array)*self.step_size, self.step_size), v[:, 0], linewidth=lw)
            xmax = len(array)*1.1*self.step_size
            ymax = np.maximum(np.max(v[:, 0])*1.1, self.inspection_rta.r_max*1.1)
            ax6.fill_between([0, xmax], [self.inspection_rta.r_max, self.inspection_rta.r_max], [ymax, ymax], color=(255/255, 239/255, 239/255))
            ax6.fill_between([0, xmax], [0, 0], [self.inspection_rta.r_max, self.inspection_rta.r_max], color=(244/255, 249/255, 241/255))
            ax6.plot([0, xmax], [self.inspection_rta.r_max, self.inspection_rta.r_max], 'k--', linewidth=lw)
            ax6.set_xlim([0, xmax])
            ax6.set_xlabel('Time [s]')
            ax6.set_ylabel(r'Relative Dist. ($\vert \vert \mathbf{p} \vert \vert_2$) [m]')
            ax6.set_ylim([0, ymax])
            ax6.grid(True)

        if not fast_plot:
            if not paper_plot:
                for i in range(0, len(array), 3):
                    r = self.inspection_rta.constraints["PSM"].get_array(array[i])+self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius
                    ax7.plot(range(i, len(r)+i), r, 'c', linewidth=0.5)
                ax7.plot(range(0, len(array)*self.step_size, self.step_size), v[:, 0], linewidth=lw)
                xmax = len(array)*1.1*self.step_size
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
            fig = plt.figure()
            ax8 = fig.add_subplot(111)
        xmax = len(array)*1.1*self.step_size
        ymax = self.inspection_rta.vel_limit*1.2
        ax8.plot(range(0, len(array)*self.step_size, self.step_size), array[:, 3], linewidth=lw, label=r'$\dot{x}$')
        ax8.plot(range(0, len(array)*self.step_size, self.step_size), array[:, 4], linewidth=lw, label=r'$\dot{y}$')
        ax8.plot(range(0, len(array)*self.step_size, self.step_size), array[:, 5], linewidth=lw, label=r'$\dot{z}$')
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
        xmax = len(array)*1.1*self.step_size
        ymax = self.inspection_rta.u_max*1.2
        ax9.plot(range(0, len(control)*self.step_size, self.step_size), control[:, 0], linewidth=lw, label=r'$F_x$')
        ax9.plot(range(0, len(control)*self.step_size, self.step_size), control[:, 1], linewidth=lw, label=r'$F_y$')
        ax9.plot(range(0, len(control)*self.step_size, self.step_size), control[:, 2], linewidth=lw, label=r'$F_z$')
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
            ax4.set_title('Delta V Limit')
            ax5.set_title('Keep Out Zone (Sun Avoidance)')
            ax6.set_title('Keep In Zone')
            ax7.set_title('Passively Safe Maneuvers')
            ax8.set_title('Velocity Limits')
            ax9.set_title('Actuation Constraints')
            fig.tight_layout(h_pad=2)


class FuelEnv(Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, check_init_state = False, **kwargs)
        self.state_des = np.array([500, 500, 700, 0, 0, 0])
    
    def _get_initial_state(self):
        return np.array([-500., -500., 700., 0, 0, 0, 0, 0])
    
    def _update_status(self, state: np.ndarray, time: float):
        if time % 100 == 0:
            self.state_des *= -1

    def plotter(self, array, control, intervening, paper_plot=False, fast_plot=False):
        if not paper_plot:
            fig = plt.figure(figsize=(15, 6))
            ax1 = fig.add_subplot(131, projection='3d')
            ax4 = fig.add_subplot(132)
            ax9 = fig.add_subplot(133)
            lw = 2
        else:
            plt.rcParams.update({'font.size': 25, 'text.usetex': True})
            lw = 4
            hp = 0.1
        
        if not fast_plot:
            if paper_plot:
                fig = plt.figure()
                ax1 = fig.add_subplot(111, projection='3d')
            for i in range(len(array)-1):
                ax1.plot(array[i:i+2, 0], array[i:i+2, 1], array[i:i+2, 2], color=plt.cm.cool(i/len(array)), linewidth=lw)
            # max = np.max(np.abs(array[:, 0:3]))*1.1
            max = 1100
            ax1.plot(0, 0, 0, 'k*', markersize=15)
            ax1.set_xlabel(r'$x$ [km]')
            ax1.set_ylabel(r'$y$ [km]')
            ax1.set_zlabel(r'$z$ [km]')
            ax1.set_xticks([-1000, 0, 1000])
            ax1.set_yticks([-1000, 0, 1000])
            ax1.set_zticks([-1000, 0, 1000])
            ax1.set_xticklabels([-1, 0, 1])
            ax1.set_yticklabels([-1, 0, 1])
            ax1.set_zticklabels([-1, 0, 1])
            ax1.set_xlim([-max, max])
            ax1.set_ylim([-max, max])
            ax1.set_zlim([-max, max])
            ax1.set_box_aspect((1,1,1))
            ax1.grid(True)
            if paper_plot:
                ax1.xaxis.labelpad = 10
                ax1.yaxis.labelpad = 10
                ax1.zaxis.labelpad = 10
                plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax4 = fig.add_subplot(111)
        xmax = len(array)*1.1
        ymax = np.maximum(self.inspection_rta.delta_v_limit, np.max(array[:, 7]))*1.1
        ax4.plot(range(len(array)), array[:, 7], linewidth=lw)
        ax4.fill_between([0, xmax], [self.inspection_rta.delta_v_limit, self.inspection_rta.delta_v_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax4.fill_between([0, xmax], [0, 0], [self.inspection_rta.delta_v_limit, self.inspection_rta.delta_v_limit], color=(244/255, 249/255, 241/255))
        ax4.plot([0, xmax], [self.inspection_rta.delta_v_limit, self.inspection_rta.delta_v_limit], 'k--', linewidth=lw)
        ax4.set_xlim([0, xmax])
        ax4.set_ylim([0, ymax])
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel(r'$\Delta$ V Used [m/s]')
        ax4.grid(True)
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


if __name__ == '__main__':
    plot_fig = True
    save_fig = True
    output_dir = 'figs/inspection_1v1'

    envs = [Env(Inspection1v1RTA()), Env(DiscreteInspection1v1RTA(), step_size=10), FuelEnv(InspectionCascadedRTA())]
    output_names = ['rta_test_inspection_1v1', 'rta_test_discrete_inspection_1v1', 'rta_test_cascaded_rta']

    os.makedirs(output_dir, exist_ok=True)

    for env, output_name in zip(envs, output_names):
        start_time = time.time()
        x, u, i = env.simulate_episode()
        print(f"Simulation time: {time.time()-start_time:2.3f} sec")
        env.plotter(x, u, i)
        if plot_fig:
            plt.show()
        if save_fig:
            plt.savefig(os.path.join(output_dir, output_name))
