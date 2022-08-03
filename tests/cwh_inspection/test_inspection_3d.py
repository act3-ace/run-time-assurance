import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

plt.rcParams.update({'font.size': 23, 'text.usetex': True, 'figure.autolayout': True})

from safe_autonomy_dynamics.cwh.point_model import M_DEFAULT, N_DEFAULT, generate_cwh_matrices
from run_time_assurance.zoo.cwh.inspection_3d import NUM_DEPUTIES_DEFAULT, U_MAX_DEFAULT, InspectionRTA


class Env():
    def __init__(self):
        self.dt = 1  # Time step
        self.time = 2000 # Total sim time
        self.u_max = U_MAX_DEFAULT  # Actuation constraint
        self.deputies = NUM_DEPUTIES_DEFAULT # Number of deputies in simulation

        # Generate dynamics matrices
        self.A, self.B = generate_cwh_matrices(M_DEFAULT, N_DEFAULT, mode="3d")

        # Specify LQR gains
        Q = np.eye(6) * 0.05  # State cost
        R = np.eye(3) * 1000  # Control cost

        # Solve ARE
        Xare = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, Q, R))
        # Get LQR gain
        self.Klqr = np.array(-scipy.linalg.inv(R)*(self.B.T*Xare))

    def u_des(self, x, x_des=np.array([[0], [0], [0], [0], [0], [0]])):
        # LQR to desired state
        u = self.Klqr @ (x - x_des)
        return np.clip(u, -self.u_max, self.u_max).flatten()

    def reset(self):
        # Random point 800-1000 m away from origin
        theta = np.random.rand()*2*np.pi
        gamma = np.random.rand()*2*np.pi
        r = (np.random.rand()-0.5)*200+900
        x = np.array([[r*np.cos(theta)*np.cos(gamma)], [r*np.sin(theta)*np.cos(gamma)], [r*np.sin(gamma)], [0], [0], [0]])
        return x.flatten()

    def step(self, x, u):
        x1 = np.zeros((self.deputies, 6))
        for i in range(self.deputies):
            # Euler integration
            xd = self.A @ np.vstack(x[i, :]) + self.B @ np.vstack(u[i, :])
            x1[i, :] = (np.vstack(x[i, :]) + xd * self.dt).flatten()
        return x1

    def run_one_step(self):
        x = np.zeros((self.deputies, 6))
        for i in range(self.deputies):
            x[i, :] = self.reset()
        u_safe = np.zeros((self.deputies, 3))
        for i in range(self.deputies):
            x_i = x[i, :]
            u_des = self.u_des(np.vstack(x_i))
            x_old = np.delete(x, i, 0)
            x_new = np.row_stack((x_i, x_old))
            # Use RTA
            u_safe[i, :] = self.rta.filter_control(x_new.flatten(), self.dt, u_des)
        # Take step using safe action
        x = self.step(x, u_safe)

    def run_episode(self, rta, plotter=True):
        self.rta = rta
        # Track time
        start_time = time.time()
        # Track initial values
        # x = np.zeros((self.deputies, 6))
        # for i in range(self.deputies):
        #     x[i, :] = self.reset()
        # print(repr(x))
        x = np.array([[-678.84733531,  645.12760716, -129.71259692,    0.        ,
                0.        ,    0.        ],
            [ 674.36453393,  442.90772096,  543.97750877,    0.        ,
                0.        ,    0.        ],
            [-272.12276168, -393.5272131 ,  700.1336778 ,    0.        ,
                0.        ,    0.        ],
            [ 353.64983193,  142.56376478,  879.52164532,    0.        ,
                0.        ,    0.        ],
            [-528.83729321, -436.48202214,  582.09238772,    0.        ,
                0.        ,    0.        ]])
        # Desired state is opposite of initial state
        x_des = -x
        if plotter:
            # Data tracking arrays
            array = [x.flatten()]
            control = [np.zeros(self.deputies*3)]
            intervening = [False]

        # Run episode
        for t in range(int(self.time/self.dt)):
            u_safe = np.zeros((self.deputies, 3))
            # For each deputy
            for i in range(self.deputies):
                x_i = x[i, :]
                # Get u_des (x_des is the origin for the first half of the simulation)
                if t > self.time/2:
                    u_des = self.u_des(np.vstack(x_i), np.vstack(x_des[i, :]))
                else:
                    u_des = self.u_des(np.vstack(x_i))
                # List current deputy's state first
                x_old = np.delete(x, i, 0)
                x_new = np.row_stack((x_i, x_old))
                # Use RTA
                u_safe[i, :] = self.rta.filter_control(x_new.flatten(), self.dt, u_des)
            # Take step using safe action
            x = self.step(x, u_safe)
            if plotter:
                # Track data
                array = np.append(array, [x.flatten()], axis=0)
                control = np.append(control, [u_safe.flatten()], axis=0)
                intervening.append(self.rta.intervening)

        # Print final time, plot
        print(f"Simulation time: {time.time()-start_time:2.3f} sec")
        if plotter:
            self.plotter(array, control, np.array(intervening))

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
            ax7 = fig.add_subplot(339)
            lw = 2
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')
            fig = plt.figure()
            ax2 = fig.add_subplot(111)
            fig = plt.figure()
            ax3 = fig.add_subplot(111)
            fig = plt.figure()
            ax4 = fig.add_subplot(111)
            fig = plt.figure()
            ax8 = fig.add_subplot(111)
            fig = plt.figure()
            ax9 = fig.add_subplot(111)
            fig = plt.figure()
            ax5 = fig.add_subplot(111)
            fig = plt.figure()
            ax6 = fig.add_subplot(111)
            fig = plt.figure()
            ax7 = fig.add_subplot(111)
            lw = 3
        
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

        
        xmax = 0
        ymax = 0
        for i in range(self.deputies):
            v = np.empty([len(array), 2])
            for j in range(len(array)):
                v[j, :] = [np.linalg.norm(array[j, 6*i:6*i+3]), np.linalg.norm(array[j, 6*i+3:6*i+6])]
            xmax = np.maximum(xmax, np.max(v[:, 0])*1.1)
            ymax = np.maximum(ymax, np.max(v[:, 1])*1.1)
            ax2.plot(v[:, 0], v[:, 1], linewidth=lw)
            ax3.plot(range(len(array)), v[:, 0], linewidth=lw)
        ax2.fill_between([0, xmax], [self.rta.v0, self.rta.v0 + self.rta.v1*xmax], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax2.fill_between([0, xmax], [0, 0], [self.rta.v0, self.rta.v0 + self.rta.v1*xmax], color=(244/255, 249/255, 241/255))
        ax2.plot([0, xmax], [self.rta.v0, self.rta.v0 + self.rta.v1*xmax], 'k--', linewidth=lw)
        ax2.set_xlim([0, xmax])
        ax2.set_ylim([0, ymax])
        ax2.set_xlabel(r'Relative Dist. ($\vert \vert \mathbf{p}_i  \vert \vert_2$) [m]')
        ax2.set_ylabel(r'Relative Vel. ($\vert \vert \mathbf{v}_i  \vert \vert_2$) [m/s]')
        ax2.grid(True)

        ymax = xmax
        xmax = len(array)*1.1
        ax3.fill_between([0, xmax], [self.rta.chief_radius+self.rta.deputy_radius, self.rta.chief_radius+self.rta.deputy_radius], [ymax, ymax], color=(244/255, 249/255, 241/255))
        ax3.fill_between([0, xmax], [0, 0], [self.rta.chief_radius+self.rta.deputy_radius, self.rta.chief_radius+self.rta.deputy_radius], color=(255/255, 239/255, 239/255))
        ax3.plot([0, xmax], [self.rta.chief_radius+self.rta.deputy_radius, self.rta.chief_radius+self.rta.deputy_radius], 'k--', linewidth=lw)
        ax3.set_xlim([0, xmax])
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel(r'Relative Dist. ($\vert \vert \mathbf{p}_i \vert \vert_2$) [m]')
        ax3.set_yscale('log')
        ax3.set_ylim([6, ymax])
        ax3.grid(True)

        th = self.rta.theta*180/np.pi
        xmax = len(array)*1.1
        ymax = 0
        for i in range(self.deputies):
            h = np.zeros(len(array))
            for j in range(len(array)):
                # h[j] = -np.dot(array[j, 6*i:6*i+3], self.rta.e_hat)/np.linalg.norm(array[j, 6*i:6*i+3]) + np.cos(self.rta.theta)
                h[j] = np.arccos(np.dot(array[j, 6*i:6*i+3], self.rta.e_hat)/(np.linalg.norm(array[j, 6*i:6*i+3])*np.linalg.norm(self.rta.e_hat)))*180/np.pi
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
            ax8.fill_between([0, xmax], [self.rta.deputy_radius*2, self.rta.deputy_radius*2], [ymax, ymax], color=(244/255, 249/255, 241/255))
            ax8.fill_between([0, xmax], [0, 0], [self.rta.deputy_radius*2, self.rta.deputy_radius*2], color=(255/255, 239/255, 239/255))
            ax8.plot([0, xmax], [self.rta.deputy_radius*2, self.rta.deputy_radius*2], 'k--', linewidth=lw)
            ax8.set_xlim([0, xmax])
            ax8.set_xlabel('Time [s]')
            ax8.set_ylabel(r'Relative Dist. ($\vert \vert \mathbf{p}_i - \mathbf{p}_j \vert \vert_2$) [m]')
            ax8.set_yscale('log')
            ax8.set_ylim([6, ymax])
            ax8.grid(True)

        xmax = len(array)*1.1
        ymax = self.rta.u_max*1.2
        for i in range(self.deputies*3):
            ax9.plot(range(len(control)), control[:, i], linewidth=lw)
        ax9.fill_between([0, xmax], [self.rta.u_max, self.rta.u_max], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax9.fill_between([0, xmax], [-ymax, -ymax], [self.rta.u_max, self.rta.u_max], color=(255/255, 239/255, 239/255))
        ax9.fill_between([0, xmax], [-self.rta.u_max, -self.rta.u_max], [self.rta.u_max, self.rta.u_max], color=(244/255, 249/255, 241/255))
        ax9.plot([0, xmax], [self.rta.u_max, self.rta.u_max], 'k--', linewidth=lw)
        ax9.plot([0, xmax], [-self.rta.u_max, -self.rta.u_max], 'k--', linewidth=lw)
        ax9.set_xlim([0, xmax])
        ax9.set_ylim([-ymax, ymax])
        ax9.set_xlabel('Time [s]')
        ax9.set_ylabel(r'$\mathbf{u}$ [N]')
        ax9.grid(True)

        xmax = len(array)*1.1
        ymax = self.rta.x_vel_limit*1.2
        for i in range(self.deputies):
            ax5.plot(range(len(array)), array[:, 6*i+3], linewidth=lw)
        ax5.fill_between([0, xmax], [self.rta.x_vel_limit, self.rta.x_vel_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax5.fill_between([0, xmax], [-ymax, -ymax], [self.rta.x_vel_limit, self.rta.x_vel_limit], color=(255/255, 239/255, 239/255))
        ax5.fill_between([0, xmax], [-self.rta.x_vel_limit, -self.rta.x_vel_limit], [self.rta.x_vel_limit, self.rta.x_vel_limit], color=(244/255, 249/255, 241/255))
        ax5.plot([0, xmax], [self.rta.x_vel_limit, self.rta.x_vel_limit], 'k--', linewidth=lw)
        ax5.plot([0, xmax], [-self.rta.x_vel_limit, -self.rta.x_vel_limit], 'k--', linewidth=lw)
        ax5.set_xlim([0, xmax])
        ax5.set_ylim([-ymax, ymax])
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel(r'$\dot{x}_i$ [m/s]')
        ax5.grid(True)

        xmax = len(array)*1.1
        ymax = self.rta.y_vel_limit*1.2
        for i in range(self.deputies):
            ax6.plot(range(len(array)), array[:, 6*i+4], linewidth=lw)
        ax6.fill_between([0, xmax], [self.rta.y_vel_limit, self.rta.y_vel_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax6.fill_between([0, xmax], [-ymax, -ymax], [self.rta.y_vel_limit, self.rta.y_vel_limit], color=(255/255, 239/255, 239/255))
        ax6.fill_between([0, xmax], [-self.rta.y_vel_limit, -self.rta.y_vel_limit], [self.rta.y_vel_limit, self.rta.y_vel_limit], color=(244/255, 249/255, 241/255))
        ax6.plot([0, xmax], [self.rta.y_vel_limit, self.rta.y_vel_limit], 'k--', linewidth=lw)
        ax6.plot([0, xmax], [-self.rta.y_vel_limit, -self.rta.y_vel_limit], 'k--', linewidth=lw)
        ax6.set_xlim([0, xmax])
        ax6.set_ylim([-ymax, ymax])
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel(r'$\dot{y}_i$ [m/s]')
        ax6.grid(True)

        xmax = len(array)*1.1
        ymax = self.rta.z_vel_limit*1.2
        for i in range(self.deputies):
            ax7.plot(range(len(array)), array[:, 6*i+5], linewidth=lw)
        ax7.fill_between([0, xmax], [self.rta.z_vel_limit, self.rta.z_vel_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax7.fill_between([0, xmax], [-ymax, -ymax], [self.rta.z_vel_limit, self.rta.z_vel_limit], color=(255/255, 239/255, 239/255))
        ax7.fill_between([0, xmax], [-self.rta.z_vel_limit, -self.rta.z_vel_limit], [self.rta.z_vel_limit, self.rta.z_vel_limit], color=(244/255, 249/255, 241/255))
        ax7.plot([0, xmax], [self.rta.z_vel_limit, self.rta.z_vel_limit], 'k--', linewidth=lw)
        ax7.plot([0, xmax], [-self.rta.z_vel_limit, -self.rta.z_vel_limit], 'k--', linewidth=lw)
        ax7.set_xlim([0, xmax])
        ax7.set_ylim([-ymax, ymax])
        ax7.set_xlabel('Time [s]')
        ax7.set_ylabel(r'$\dot{z}_i$ [m/s]')
        ax7.grid(True)

        if not paper_plot:
            ax1.set_title('Trajectories')
            ax2.set_title('Dynamic Velocity Constraint')
            ax3.set_title('Chief Collision Constraint')
            ax4.set_title('Translational Keep Out Zone (Sun Avoidance)')
            ax8.set_title('Deputy Collision Constraint')
            ax9.set_title('Actuation Saturation Constraint')
            ax5.set_title(r'$\dot{x}$ Velocity Constraint')
            ax6.set_title(r'$\dot{y}$ Velocity Constraint')
            ax7.set_title(r'$\dot{z}$ Velocity Constraint')
            fig.tight_layout(h_pad=2)


# Setup env, RTA, then run episode
env = Env()
rta = InspectionRTA()
env.run_episode(rta)

# times = []
# env.run_episode(rta, plotter=False)
# for i in range(100):
#     st = time.time()
#     env.run_episode(rta, plotter=False)
#     times.append(time.time()-st)
# print(np.mean(times))

# env.rta = rta
# env.run_one_step()
# env.run_one_step()
# env.run_one_step()
# env.run_one_step()

# import cProfile

# cProfile.run('env.run_one_step()', filename='program2.prof')

plt.show()
