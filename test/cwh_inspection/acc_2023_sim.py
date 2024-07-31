import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
from functools import partial
from jax import jit
from jax.experimental.ode import odeint
import jax.numpy as jnp
from safe_autonomy_dynamics.cwh import generate_cwh_matrices
from run_time_assurance.zoo.cwh.inspection_3d import CombinedInspectionRTA, InspectionRTA
from run_time_assurance.utils.sample_testing import DataTrackingSampleTestingModule
import os
from collections import OrderedDict
from run_time_assurance.constraint import (
    ConstraintMagnitudeStateLimit,
    PolynomialConstraintStrengthener,
    HOCBFConstraint
)
from run_time_assurance.zoo.cwh.inspection_1v1 import ConstraintCWHConicKeepOutZone, HOCBFExampleChiefCollision
from run_time_assurance.zoo.cwh.docking_3d import ConstraintCWH3dRelativeVelocity
from run_time_assurance.zoo.cwh.inspection_3d import ConstraintCWHDeputyCollision

class Env(DataTrackingSampleTestingModule):
    def __init__(self, rta, **kwargs):
        self.u_max = 1  # Actuation constraint
        self.deputies = 5
        self.sun_vel = -0.001027
        self.state_des = np.zeros(6*self.deputies+1)
        self.state_des2 = -self._get_initial_state()
        self.flag = True

        self.A, self.B = generate_cwh_matrices(12, 0.001027, mode="3d")

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
        return np.array([-726.02371369,  468.57059501, -276.955184  ,    0.        ,
                0.        ,    0.        ,   -9.92771938, -642.69600232,
            -626.99606681,    0.        ,    0.        ,    0.        ,
                733.01511413, -445.64516233,  133.1485955 ,    0.        ,
                0.        ,    0.        , -841.26561104,   27.0420982 ,
                -3.48942661,    0.        ,    0.        ,    0.        ,
                366.531211  , -660.69545305, -325.17854097,    0.        ,
                0.        ,    0.        ,    1.36751792])

    @partial(jit, static_argnums=0)
    def _pred_state(self, state, step_size, control):
        x1 = jnp.zeros(self.deputies*6+1)
        for i in range(self.deputies):
            sol = odeint(self.compute_state_dot, state[6*i:6*i+6], jnp.linspace(0., step_size, 11), control[3*i:3*i+3])
            x1 = x1.at[6*i:6*i+6].set(sol[-1, :])
        x1 = x1.at[-1].set(state[-1]+self.sun_vel*step_size)
        return x1
    
    def compute_state_dot(self, x, t, u):
        xd = self.A @ x + self.B @ u
        return xd
    
    def check_if_safe_state(self, state: np.ndarray):
        init_state_safe = True
        for i in range(self.deputies):
            x = self.rta.get_agent_state(state, i)
            for c in self.rta.deputy_rta.constraints.values():
                if c.phi(x, c.params) < 0 or c(x, c.params) < 0:
                    init_state_safe = False
                    break
        return init_state_safe
    
    def _update_status(self, state: np.ndarray, time: float):
        if time >= 1250:
            if self.flag:
                self.state_des = self.state_des2
                self.flag = False
            for i in range(self.deputies):
                self.state_des[6*i:6*i+3] = -np.linalg.norm(self.state_des[6*i:6*i+3]) * np.array([np.cos(state[-1]), np.sin(state[-1]), 0.])

    def plotter(self, array, control, intervening):
        plt.rcParams.update({'font.size': 30, 'text.usetex': True, 'figure.figsize': [6.4, 6]})
        lw = 4
        hp = 0.1
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        o = [(3, 0, 30), (3, 0, -15), (3, 0, 55), (3, 0, 15), (3, 0, 0)]
        
        fig = plt.figure(figsize=[9, 7])
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.xaxis.labelpad = 30
        ax1.yaxis.labelpad = 30
        ax1.zaxis.set_tick_params(pad=15)
        ax1.zaxis.labelpad = 40
        ax1.dist = 11
        max = 0
        for i in range(self.deputies):
            ax1.plot(array[:, 6*i], array[:, 6*i+1], array[:, 6*i+2], linewidth=lw)
            max = np.maximum(max, np.max(np.abs(array[:, 6*i:6*i+3]))*1.1)
            ax1.plot(array[0, 6*i], array[0, 6*i+1], array[0, 6*i+2], marker=o[i], markersize=15, color=colors[i])
        ax1.plot(0, 0, 0, 'k*', markersize=20)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.set_zlabel(r'$z$ [m]')
        ax1.set_xlim([-max, max])
        ax1.set_ylim([-max, max])
        ax1.set_zlim([-max, max])
        ax1.grid(True)
        plt.tight_layout(pad=hp)

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
        ax2.fill_between([0, xmax], [self.rta.deputy_rta.v0, self.rta.deputy_rta.v0 + self.rta.deputy_rta.v1*xmax], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax2.fill_between([0, xmax], [0, 0], [self.rta.deputy_rta.v0, self.rta.deputy_rta.v0 + self.rta.deputy_rta.v1*xmax], color=(244/255, 249/255, 241/255))
        ax2.plot([0, xmax], [self.rta.deputy_rta.v0, self.rta.deputy_rta.v0 + self.rta.deputy_rta.v1*xmax], 'k--', linewidth=lw)
        ax2.set_xlim([0, xmax])
        ax2.set_ylim([0, ymax])
        ax2.set_xlabel(r'Relative Dist. ($\vert \vert \mathbf{p}_i  \vert \vert_2$) [m]')
        ax2.set_ylabel(r'Relative Vel. ($\vert \vert \mathbf{v}_i  \vert \vert_2$) [m/s]')
        ax2.grid(True)
        plt.tight_layout(pad=hp)

        fig = plt.figure()
        ax3 = fig.add_subplot(111)
        for i in range(self.deputies):
            v = np.empty([len(array), 2])
            for j in range(len(array)):
                v[j, :] = [np.linalg.norm(array[j, 6*i:6*i+3]), np.linalg.norm(array[j, 6*i+3:6*i+6])]
            ax3.plot(range(len(array)), v[:, 0], linewidth=lw)
        ymax = xmax
        xmax = len(array)*1.1
        ax3.fill_between([0, xmax], [self.rta.deputy_rta.chief_radius+self.rta.deputy_rta.deputy_radius, self.rta.deputy_rta.chief_radius+self.rta.deputy_rta.deputy_radius], [ymax, ymax], color=(244/255, 249/255, 241/255))
        ax3.fill_between([0, xmax], [0, 0], [self.rta.deputy_rta.chief_radius+self.rta.deputy_rta.deputy_radius, self.rta.deputy_rta.chief_radius+self.rta.deputy_rta.deputy_radius], color=(255/255, 239/255, 239/255))
        ax3.plot([0, xmax], [self.rta.deputy_rta.chief_radius+self.rta.deputy_rta.deputy_radius, self.rta.deputy_rta.chief_radius+self.rta.deputy_rta.deputy_radius], 'k--', linewidth=lw)
        ax3.set_xlim([0, xmax])
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel(r'Relative Dist. ($\vert \vert \mathbf{p}_i \vert \vert_2$) [m]')
        ax3.set_yscale('log')
        ax3.set_ylim([6, ymax])
        ax3.grid(True)
        plt.tight_layout(pad=hp)

        fig = plt.figure()
        ax4 = fig.add_subplot(111)
        th = self.rta.deputy_rta.fov/2*180/np.pi
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
        plt.tight_layout(pad=hp)

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
            ax8.fill_between([0, xmax], [self.rta.deputy_rta.deputy_radius*2, self.rta.deputy_rta.deputy_radius*2], [ymax, ymax], color=(244/255, 249/255, 241/255))
            ax8.fill_between([0, xmax], [0, 0], [self.rta.deputy_rta.deputy_radius*2, self.rta.deputy_rta.deputy_radius*2], color=(255/255, 239/255, 239/255))
            ax8.plot([0, xmax], [self.rta.deputy_rta.deputy_radius*2, self.rta.deputy_rta.deputy_radius*2], 'k--', linewidth=lw)
            ax8.set_xlim([0, xmax])
            ax8.set_xlabel('Time [s]')
            ax8.set_ylabel(r'Relative Dist. ($\vert \vert \mathbf{p}_i - \mathbf{p}_j \vert \vert_2$) [m]')
            ax8.set_yscale('log')
            ax8.set_ylim([6, ymax])
            ax8.grid(True)
        plt.tight_layout(pad=hp)

        fig = plt.figure()
        ax9 = fig.add_subplot(111)
        xmax = len(array)*1.1
        ymax = self.rta.deputy_rta.u_max*1.2
        for i in range(self.deputies*3):
            ax9.plot(range(len(control)), control[:, i], linewidth=lw)
        ax9.fill_between([0, xmax], [self.rta.deputy_rta.u_max, self.rta.deputy_rta.u_max], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax9.fill_between([0, xmax], [-ymax, -ymax], [self.rta.deputy_rta.u_max, self.rta.deputy_rta.u_max], color=(255/255, 239/255, 239/255))
        ax9.fill_between([0, xmax], [-self.rta.deputy_rta.u_max, -self.rta.deputy_rta.u_max], [self.rta.deputy_rta.u_max, self.rta.deputy_rta.u_max], color=(244/255, 249/255, 241/255))
        ax9.plot([0, xmax], [self.rta.deputy_rta.u_max, self.rta.deputy_rta.u_max], 'k--', linewidth=lw)
        ax9.plot([0, xmax], [-self.rta.deputy_rta.u_max, -self.rta.deputy_rta.u_max], 'k--', linewidth=lw)
        ax9.set_xlim([0, xmax])
        ax9.set_ylim([-ymax, ymax])
        ax9.set_xlabel('Time [s]')
        ax9.set_ylabel(r'$\mathbf{u}$ [N]')
        ax9.grid(True)
        plt.tight_layout(pad=hp)

        fig = plt.figure()
        ax5 = fig.add_subplot(111)
        xmax = len(array)*1.1
        ymax = self.rta.deputy_rta.vel_limit*1.2
        for i in range(self.deputies):
            ax5.plot(range(len(array)), array[:, 6*i+3], linewidth=lw)
        ax5.fill_between([0, xmax], [self.rta.deputy_rta.vel_limit, self.rta.deputy_rta.vel_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax5.fill_between([0, xmax], [-ymax, -ymax], [self.rta.deputy_rta.vel_limit, self.rta.deputy_rta.vel_limit], color=(255/255, 239/255, 239/255))
        ax5.fill_between([0, xmax], [-self.rta.deputy_rta.vel_limit, -self.rta.deputy_rta.vel_limit], [self.rta.deputy_rta.vel_limit, self.rta.deputy_rta.vel_limit], color=(244/255, 249/255, 241/255))
        ax5.plot([0, xmax], [self.rta.deputy_rta.vel_limit, self.rta.deputy_rta.vel_limit], 'k--', linewidth=lw)
        ax5.plot([0, xmax], [-self.rta.deputy_rta.vel_limit, -self.rta.deputy_rta.vel_limit], 'k--', linewidth=lw)
        ax5.set_xlim([0, xmax])
        ax5.set_ylim([-ymax, ymax])
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel(r'$\dot{x}_i$ [m/s]')
        ax5.grid(True)
        plt.tight_layout(pad=hp)

        fig = plt.figure()
        ax6 = fig.add_subplot(111)
        xmax = len(array)*1.1
        ymax = self.rta.deputy_rta.vel_limit*1.2
        for i in range(self.deputies):
            ax6.plot(range(len(array)), array[:, 6*i+4], linewidth=lw)
        ax6.fill_between([0, xmax], [self.rta.deputy_rta.vel_limit, self.rta.deputy_rta.vel_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax6.fill_between([0, xmax], [-ymax, -ymax], [self.rta.deputy_rta.vel_limit, self.rta.deputy_rta.vel_limit], color=(255/255, 239/255, 239/255))
        ax6.fill_between([0, xmax], [-self.rta.deputy_rta.vel_limit, -self.rta.deputy_rta.vel_limit], [self.rta.deputy_rta.vel_limit, self.rta.deputy_rta.vel_limit], color=(244/255, 249/255, 241/255))
        ax6.plot([0, xmax], [self.rta.deputy_rta.vel_limit, self.rta.deputy_rta.vel_limit], 'k--', linewidth=lw)
        ax6.plot([0, xmax], [-self.rta.deputy_rta.vel_limit, -self.rta.deputy_rta.vel_limit], 'k--', linewidth=lw)
        ax6.set_xlim([0, xmax])
        ax6.set_ylim([-ymax, ymax])
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel(r'$\dot{y}_i$ [m/s]')
        ax6.grid(True)
        plt.tight_layout(pad=hp)

        fig = plt.figure()
        ax7 = fig.add_subplot(111)
        xmax = len(array)*1.1
        ymax = self.rta.deputy_rta.vel_limit*1.2
        for i in range(self.deputies):
            ax7.plot(range(len(array)), array[:, 6*i+5], linewidth=lw)
        ax7.fill_between([0, xmax], [self.rta.deputy_rta.vel_limit, self.rta.deputy_rta.vel_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax7.fill_between([0, xmax], [-ymax, -ymax], [self.rta.deputy_rta.vel_limit, self.rta.deputy_rta.vel_limit], color=(255/255, 239/255, 239/255))
        ax7.fill_between([0, xmax], [-self.rta.deputy_rta.vel_limit, -self.rta.deputy_rta.vel_limit], [self.rta.deputy_rta.vel_limit, self.rta.deputy_rta.vel_limit], color=(244/255, 249/255, 241/255))
        ax7.plot([0, xmax], [self.rta.deputy_rta.vel_limit, self.rta.deputy_rta.vel_limit], 'k--', linewidth=lw)
        ax7.plot([0, xmax], [-self.rta.deputy_rta.vel_limit, -self.rta.deputy_rta.vel_limit], 'k--', linewidth=lw)
        ax7.set_xlim([0, xmax])
        ax7.set_ylim([-ymax, ymax])
        ax7.set_xlabel('Time [s]')
        ax7.set_ylabel(r'$\dot{z}_i$ [m/s]')
        ax7.grid(True)
        plt.tight_layout(pad=hp)


if __name__ == '__main__':
    plot_fig = True
    save_fig = False
    output_dir = 'figs/inspection_3d_acc'
    deputy_rta = InspectionRTA(
        m = 12,
        n = 0.001027,
        chief_radius = 5,
        deputy_radius = 5,
        v0 = 0.2,
        v1_coef = 4,
        v0_distance = 0,
        r_max = 1000,
        fov = 60 * jnp.pi / 180,
        vel_limit = 2,
        sun_vel = -0.001027,
        control_bounds_high = 1,
        control_bounds_low = -1,
    )

    constraint_dict = OrderedDict(
        [
            ('chief_collision', HOCBFConstraint(
                HOCBFExampleChiefCollision(collision_radius=deputy_rta.chief_radius + deputy_rta.deputy_radius, a_max=deputy_rta.a_max),
                relative_degree=2,
                state_transition_system=deputy_rta.state_transition_system,
                alpha=PolynomialConstraintStrengthener([0, 0.01, 0, 0.01])
            )),
            (
                'rel_vel',
                ConstraintCWH3dRelativeVelocity(
                    v0=deputy_rta.v0,
                    v1=deputy_rta.v1,
                    v0_distance=deputy_rta.v0_distance,
                    alpha=PolynomialConstraintStrengthener([0, 0.1, 0, .1])
                )
            ),
            (
                'chief_sun',
                ConstraintCWHConicKeepOutZone(
                    a_max=deputy_rta.a_max,
                    fov=deputy_rta.fov,
                    get_pos=deputy_rta.get_pos_vector,
                    get_vel=deputy_rta.get_vel_vector,
                    get_cone_vec=deputy_rta.get_sun_vector,
                    cone_ang_vel=jnp.array([0, 0, deputy_rta.sun_vel]),
                )
            ),
            (
                'x_vel',
                ConstraintMagnitudeStateLimit(
                    limit_val=deputy_rta.vel_limit, state_index=3, alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]),
                )
            ),
            (
                'y_vel',
                ConstraintMagnitudeStateLimit(
                    limit_val=deputy_rta.vel_limit, state_index=4, alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]),
                )
            ),
            (
                'z_vel',
                ConstraintMagnitudeStateLimit(
                    limit_val=deputy_rta.vel_limit, state_index=5, alpha=PolynomialConstraintStrengthener([0, 0.1, 0, 0.01]),
                )
            )
        ]
    )
    for i in range(1, deputy_rta.num_deputies):
        constraint_dict[f'deputy_collision_{i}'] = ConstraintCWHDeputyCollision(
            collision_radius=deputy_rta.deputy_radius * 2, a_max=deputy_rta.a_max, deputy=i
        )

    deputy_rta.constraints = constraint_dict

    env = Env(CombinedInspectionRTA(deputy_rta=deputy_rta))

    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    x, u, i = env.simulate_episode()
    print(f"Simulation time: {time.time()-start_time:2.3f} sec")
    env.plotter(x, u, i)
    if plot_fig:
        plt.show()
    if save_fig:
        plt.savefig(os.path.join(output_dir, 'rta_test_inspection_3d'))
