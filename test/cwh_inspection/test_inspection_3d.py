import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
from functools import partial
from jax import jit
from jax.experimental.ode import odeint
import jax.numpy as jnp
from safe_autonomy_simulation.spacecraft.utils import M_DEFAULT, N_DEFAULT, generate_cwh_matrices
from run_time_assurance.zoo.cwh.inspection_3d import NUM_DEPUTIES_DEFAULT, U_MAX_DEFAULT, SUN_VEL_DEFAULT, CombinedInspectionRTA, InspectionRTA
from run_time_assurance.utils.sample_testing import DataTrackingSampleTestingModule
from run_time_assurance.utils import to_jnp_array_jit
import os

# from jax.config import config
# config.update('jax_disable_jit', True)


class Env(DataTrackingSampleTestingModule):
    def __init__(self, rta, **kwargs):
        self.u_max = U_MAX_DEFAULT  # Actuation constraint
        self.deputies = NUM_DEPUTIES_DEFAULT
        self.sun_vel = SUN_VEL_DEFAULT
        self.inspection_rta = InspectionRTA()
        self.state_des = np.zeros(6*self.deputies+1)

        self.A, self.B = generate_cwh_matrices(M_DEFAULT, N_DEFAULT, mode="3d")

        # Specify LQR gains
        Q = np.eye(6) * 0.05  # State cost
        R = np.eye(3) * 1000  # Control cost

        # Solve ARE
        Xare = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, Q, R))
        # Get LQR gain
        self.Klqr = np.array(-scipy.linalg.inv(R)*(self.B.T*Xare))

        super().__init__(rta=rta, simulation_time=2000, step_size=1, control_dim=3*self.deputies, state_dim=6*self.deputies+1, **kwargs)

    def _desired_control(self, state):
        u = np.zeros(3*self.deputies)
        for i in range(self.deputies):
            u[3*i:3*i+3] = self.Klqr @ (state[6*i:6*i+6] - self.state_des[6*i:6*i+6])
        return np.clip(u, -self.u_max, self.u_max)

    def _get_initial_state(self):
        x = np.array([])
        for _ in range(self.deputies):
            theta = np.random.rand(2) * 2 * np.pi
            pos = np.random.rand(1) * 200 + 800
            p = np.array([np.cos(theta[0])*np.sin(theta[1]), np.sin(theta[0])*np.sin(theta[1]), np.cos(theta[1])]) * pos
            v = np.array([0, 0, 0])
            x = np.concatenate((x, p, v))
        x = np.concatenate((x, np.array([0.])))
        self.state_des = -x
        return x

    @partial(jit, static_argnums=0)
    def _pred_state_fn(self, state, step_size, control):
        x1 = jnp.zeros(self.deputies*6+1)
        for i in range(self.deputies):
            sol = odeint(self.compute_state_dot, state[6*i:6*i+6], jnp.linspace(0., step_size, 11), control[3*i:3*i+3])
            x1 = x1.at[6*i:6*i+6].set(sol[-1, :])
        x1 = x1.at[-1].set(state[-1]+SUN_VEL_DEFAULT*step_size)
        return x1
    
    def _pred_state(self, state, step_size, control):
        x1 = self._pred_state_fn(to_jnp_array_jit(state), step_size, to_jnp_array_jit(control))
        return np.array(x1)
    
    def compute_state_dot(self, x, t, u):
        xd = self.A @ x + self.B @ u
        return xd
    
    def check_if_safe_state(self, state: np.ndarray):
        init_state_safe = True
        for i in range(self.deputies):
            x = self.rta.get_agent_state(state, i)
            for c in self.inspection_rta.constraints.values():
                if c.phi(x, c.params) < 0 or c(x, c.params) < 0:
                    init_state_safe = False
                    break
        return init_state_safe
    
    def _update_status(self, state: np.ndarray, time: float):
        if time >= 1250:
            for i in range(self.deputies):
                self.state_des[6*i:6*i+3] = -np.linalg.norm(self.state_des[6*i:6*i+3]) * np.array([np.cos(state[-1]), np.sin(state[-1]), 0.])

    def plotter(self, array, control, intervening, paper_plot=False):
        if not paper_plot:
            fig = plt.figure(figsize=(15, 15))
            ax1 = fig.add_subplot(331, projection='3d')
            ax2 = fig.add_subplot(334)
            ax3 = fig.add_subplot(332)
            ax4 = fig.add_subplot(335)
            ax8 = fig.add_subplot(333)
            ax9 = fig.add_subplot(336)
            ax5 = fig.add_subplot(337)
            ax6 = fig.add_subplot(338)
            lw = 2
        else:
            plt.rcParams.update({'font.size': 30, 'text.usetex': True, 'figure.figsize': [6.4, 6]})
            lw = 4
            hp = 0.1
        
        if paper_plot:
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')
        max = 0
        for i in range(self.deputies):
            ax1.plot(array[:, 6*i], array[:, 6*i+1], array[:, 6*i+2], linewidth=lw)
            max = np.maximum(max, np.max(np.abs(array[:, 6*i:6*i+3]))*1.1)
        ax1.plot(0, 0, 0, 'k*', markersize=15)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.set_zlabel(r'$z$ [m]')
        ax1.set_xlim([-max, max])
        ax1.set_ylim([-max, max])
        ax1.set_zlim([-max, max])
        ax1.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax2 = fig.add_subplot(111)
        xmax = 0
        ymax = 0
        for i in range(self.deputies):
            v = np.empty([len(array), 2])
            for j in range(len(array)):
                v[j, :] = [np.linalg.norm(array[j, 6*i:6*i+3]), np.linalg.norm(array[j, 6*i+3:6*i+6])]
            xmax = np.maximum(xmax, np.max(v[:, 0])*1.1)
            ymax = np.maximum(ymax, np.max(v[:, 1])*1.1)
            ax2.plot(v[:, 0], v[:, 1], linewidth=lw)
        ax2.fill_between([0, xmax], [self.inspection_rta.v0, self.inspection_rta.v0 + self.inspection_rta.v1*xmax], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax2.fill_between([0, xmax], [0, 0], [self.inspection_rta.v0, self.inspection_rta.v0 + self.inspection_rta.v1*xmax], color=(244/255, 249/255, 241/255))
        ax2.plot([0, xmax], [self.inspection_rta.v0, self.inspection_rta.v0 + self.inspection_rta.v1*xmax], 'k--', linewidth=lw)
        ax2.set_xlim([0, xmax])
        ax2.set_ylim([0, ymax])
        ax2.set_xlabel(r'Relative Dist. ($\vert \vert \mathbf{p}_i  \vert \vert_2$) [m]')
        ax2.set_ylabel(r'Relative Vel. ($\vert \vert \mathbf{v}_i  \vert \vert_2$) [m/s]')
        ax2.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax3 = fig.add_subplot(111)
        for i in range(self.deputies):
            v = np.empty([len(array), 2])
            for j in range(len(array)):
                v[j, :] = [np.linalg.norm(array[j, 6*i:6*i+3]), np.linalg.norm(array[j, 6*i+3:6*i+6])]
            ax3.plot(range(len(array)), v[:, 0], linewidth=lw)
        ymax = xmax
        xmax = len(array)*1.1
        ax3.fill_between([0, xmax], [self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius, self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius], [ymax, ymax], color=(244/255, 249/255, 241/255))
        ax3.fill_between([0, xmax], [0, 0], [self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius, self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius], color=(255/255, 239/255, 239/255))
        ax3.plot([0, xmax], [self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius, self.inspection_rta.chief_radius+self.inspection_rta.deputy_radius], 'k--', linewidth=lw)
        ax3.set_xlim([0, xmax])
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel(r'Relative Dist. ($\vert \vert \mathbf{p}_i \vert \vert_2$) [m]')
        ax3.set_yscale('log')
        ax3.set_ylim([6, ymax])
        ax3.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax4 = fig.add_subplot(111)
        th = self.inspection_rta.fov/2*180/np.pi
        xmax = len(array)*1.1
        ymax = 0
        for i in range(self.deputies):
            h = np.zeros(len(array))
            for j in range(len(array)):
                r_s_hat = np.array([np.cos(array[j, -1]), np.sin(array[j, -1]), 0.])
                r_b_hat = -array[j, 6*i:6*i+3]/np.linalg.norm(array[j, 6*i:6*i+3])
                h[j] = np.arccos(np.dot(r_s_hat, r_b_hat))*180/np.pi
            ymax = np.maximum(ymax, np.max(h)*1.1)
            ax4.plot(range(len(array)), h, linewidth=lw)
        ax4.fill_between([0, xmax], [th, th], [ymax, ymax], color=(244/255, 249/255, 241/255))
        ax4.fill_between([0, xmax], [0, 0], [th, th], color=(255/255, 239/255, 239/255))
        ax4.plot([0, xmax], [th, th], 'k--', linewidth=lw)
        ax4.set_xlim([0, xmax])
        ax4.set_ylim([0, ymax])
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel(r'Angle to Sun ($\theta_s$) [degrees]')
        ax4.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax8 = fig.add_subplot(111)
        if self.deputies != 1:
            xmax = len(array)*1.1
            ymax = 0
            for i in range(self.deputies):
                dist = np.empty((self.deputies, len(array)))
                for j in range(self.deputies):
                    if i != j:
                        for k in range(len(array)):
                            dist[j, k] = np.linalg.norm(array[k, 6*i:6*i+3] - array[k, 6*j:6*j+3])
                dist = np.delete(dist, i, 0)
                d_array = np.amin(dist, axis=0)
                ax8.plot(range(len(d_array)), d_array, linewidth=lw)
                ymax = np.maximum(ymax, np.max(d_array)*1.1)
            ax8.fill_between([0, xmax], [self.inspection_rta.deputy_radius*2, self.inspection_rta.deputy_radius*2], [ymax, ymax], color=(244/255, 249/255, 241/255))
            ax8.fill_between([0, xmax], [0, 0], [self.inspection_rta.deputy_radius*2, self.inspection_rta.deputy_radius*2], color=(255/255, 239/255, 239/255))
            ax8.plot([0, xmax], [self.inspection_rta.deputy_radius*2, self.inspection_rta.deputy_radius*2], 'k--', linewidth=lw)
            ax8.set_xlim([0, xmax])
            ax8.set_xlabel('Time [s]')
            ax8.set_ylabel(r'Relative Dist. ($\vert \vert \mathbf{p}_i - \mathbf{p}_j \vert \vert_2$) [m]')
            ax8.set_yscale('log')
            ax8.set_ylim([6, ymax])
            ax8.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax9 = fig.add_subplot(111)
        xmax = len(array)*1.1
        ymax = self.inspection_rta.u_max*1.2
        for i in range(self.deputies*3):
            ax9.plot(range(len(control)), control[:, i], linewidth=lw)
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
        if paper_plot:
            plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax5 = fig.add_subplot(111)
        xmax = len(array)*1.1
        ymax = self.inspection_rta.vel_limit*1.2
        for i in range(self.deputies):
            ax5.plot(range(len(array)), array[:, 6*i+3], linewidth=lw)
            ax5.plot(range(len(array)), array[:, 6*i+4], linewidth=lw)
            ax5.plot(range(len(array)), array[:, 6*i+5], linewidth=lw)
        ax5.fill_between([0, xmax], [self.inspection_rta.vel_limit, self.inspection_rta.vel_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax5.fill_between([0, xmax], [-ymax, -ymax], [self.inspection_rta.vel_limit, self.inspection_rta.vel_limit], color=(255/255, 239/255, 239/255))
        ax5.fill_between([0, xmax], [-self.inspection_rta.vel_limit, -self.inspection_rta.vel_limit], [self.inspection_rta.vel_limit, self.inspection_rta.vel_limit], color=(244/255, 249/255, 241/255))
        ax5.plot([0, xmax], [self.inspection_rta.vel_limit, self.inspection_rta.vel_limit], 'k--', linewidth=lw)
        ax5.plot([0, xmax], [-self.inspection_rta.vel_limit, -self.inspection_rta.vel_limit], 'k--', linewidth=lw)
        ax5.set_xlim([0, xmax])
        ax5.set_ylim([-ymax, ymax])
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel(r'$v$ [m/s]')
        ax5.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        if paper_plot:
            fig = plt.figure()
            ax6 = fig.add_subplot(111)
        th = self.inspection_rta.fov/2*180/np.pi
        xmax = len(array)*1.1
        ymax = 0
        for i in range(self.deputies):
            h = np.zeros((self.deputies, len(array)))
            for k in range(self.deputies):
                if i != k:
                    for j in range(len(array)):
                        r_s_hat = -np.array([np.cos(array[j, -1]), np.sin(array[j, -1]), 0.])
                        p1_p2 = array[j, 6*i:6*i+3] - array[j, 6*k:6*k+3]
                        r_b_hat = p1_p2/np.linalg.norm(p1_p2)
                        h[k, j] = np.arccos(np.dot(r_s_hat, r_b_hat))*180/np.pi
                    ymax = np.maximum(ymax, np.max(h[k, :])*1.1)
                    ax6.plot(range(len(array)), h[k, :], linewidth=lw)
        ax6.fill_between([0, xmax], [th, th], [ymax, ymax], color=(244/255, 249/255, 241/255))
        ax6.fill_between([0, xmax], [0, 0], [th, th], color=(255/255, 239/255, 239/255))
        ax6.plot([0, xmax], [th, th], 'k--', linewidth=lw)
        ax6.set_xlim([0, xmax])
        ax6.set_ylim([0, ymax])
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel(r'Angle to Sun ($\theta_s$) [degrees]')
        ax6.grid(True)
        if paper_plot:
            plt.tight_layout(pad=hp)

        if not paper_plot:
            ax1.set_title('Trajectories')
            ax2.set_title('Dynamic Velocity Constraint')
            ax3.set_title('Chief Collision Constraint')
            ax4.set_title('Translational Keep Out Zone (Sun Avoidance)')
            ax8.set_title('Deputy Collision Constraint')
            ax9.set_title('Actuation Saturation Constraint')
            ax5.set_title(r'Velocity Constraint')
            ax6.set_title(r'Deputy Sun Avoidance')
            fig.tight_layout(h_pad=2)


if __name__ == '__main__':
    plot_fig = True
    save_fig = True
    output_dir = 'figs/inspection_3d'

    env = Env(CombinedInspectionRTA())

    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    x, u, i = env.simulate_episode()
    print(f"Simulation time: {time.time()-start_time:2.3f} sec")
    env.plotter(x, u, i)
    if plot_fig:
        plt.show()
    if save_fig:
        plt.savefig(os.path.join(output_dir, 'rta_test_inspection_3d'))
