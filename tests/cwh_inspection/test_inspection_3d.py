import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

from safe_autonomy_dynamics.cwh import M_DEFAULT, N_DEFAULT, generate_cwh_matrices
from run_time_assurance.zoo.cwh.inspection_3d import U_MAX_DEFAULT, InspectionRTA


class Env():
    def __init__(self):
        self.dt = 1  # Time step
        self.time = 500 # Total sim time
        self.u_max = U_MAX_DEFAULT  # Actuation constraint

        self.A, self.B = generate_cwh_matrices(M_DEFAULT, N_DEFAULT, mode="3d")

        # Specify LQR gains
        Q = np.eye(6) * 0.05  # State cost
        R = np.eye(3) * 1000  # Control cost

        # Solve ARE
        Xare = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, Q, R))
        self.Klqr = np.array(-scipy.linalg.inv(R)*(self.B.T*Xare))

    def u_des(self, x, x_des=np.array([[0], [0], [0], [0], [0], [0]])):
        # LQR to desired state
        u = self.Klqr @ (x - x_des)
        return np.clip(u, -self.u_max, self.u_max)

    def reset(self):
        # Random point 100 m away from origin
        theta = np.random.rand()*2*np.pi
        gamma = np.random.rand()*2*np.pi
        x = np.array([[100*np.cos(theta)*np.cos(gamma)], [100*np.sin(theta)*np.cos(gamma)], [100*np.sin(gamma)], [0], [0], [0]])
        # x = np.array([[50], [-50], [-50], [0], [0], [0]])
        return x

    def step(self, x, u):
        # Euler integration
        xd = self.A @ x + self.B @ u
        x1 = x + xd * self.dt
        return x1

    def run_episode(self, rta):
        self.rta = rta
        # Track time
        start_time = time.time()
        # Track initial values
        x = self.reset()
        array = [x.flatten()]
        control = [self.u_des(x).flatten()]
        intervening = [False]

        # Run episode
        for _ in range(int(self.time/self.dt)):
            # Get u_des
            u_des = self.u_des(x)
            # Use RTA
            u_safe = np.vstack(self.rta.filter_control(x.flatten(), self.dt, u_des.flatten()))
            # Take step using safe action
            x = self.step(x, u_safe)
            # Track values
            array = np.append(array, [x.flatten()], axis=0)
            control = np.append(control, [u_safe.flatten()], axis=0)
            intervening.append(self.rta.intervening)

        # Print final time, plot
        print(f"Simulation time: {time.time()-start_time:2.3f} sec")
        self.plotter(array, control, np.array(intervening))

    def plotter(self, array, control, intervening):
        fig = plt.figure(figsize=(15, 10))
        
        max = np.max(np.abs(array[:, 0:3]))*1.1
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot(0, 0, 0, 'k*', markersize=15)
        ax1.plot(array[0, 0], array[0, 1], array[0, 2], 'r*', markersize=15)
        ax1.plot(array[:, 0], array[:, 1], array[:, 2], 'b', linewidth=2)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.set_zlabel(r'$z$ [m]')
        ax1.set_xlim([-max, max])
        ax1.set_ylim([-max, max])
        ax1.set_zlim([-max, max])
        ax1.set_title('Trajectory')
        ax1.grid(True)

        ax2 = fig.add_subplot(232)
        v = np.empty([len(array), 2])
        for i in range(len(array)):
            v[i, :] = [np.linalg.norm(array[i, 0:3]), np.linalg.norm(array[i, 3:6])]
        xmax = np.max(v[:, 0])*1.1
        ymax = np.max(v[:, 1])*1.1
        ax2.fill_between([0, xmax], [self.rta.v0, self.rta.v0 + self.rta.v1*xmax], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax2.fill_between([0, xmax], [0, 0], [self.rta.v0, self.rta.v0 + self.rta.v1*xmax], color=(244/255, 249/255, 241/255))
        ax2.plot([0, xmax], [self.rta.v0, self.rta.v0 + self.rta.v1*xmax], 'k--', linewidth=2)
        ax2.plot(v[:, 0], v[:, 1], 'b', linewidth=2)
        ax2.set_xlim([0, xmax])
        ax2.set_ylim([0, ymax])
        ax2.set_xlabel(r'$r_H$ [m]')
        ax2.set_ylabel(r'$v_H$ [m/s]')
        ax2.set_title('Distance Dependent Speed Limit')
        ax2.grid(True)

        ax3 = fig.add_subplot(233)
        xmax = len(array)*1.1
        ymax = np.max(v[:, 0])*1.1
        ax3.fill_between([0, xmax], [self.rta.chief_radius, self.rta.chief_radius], [ymax, ymax], color=(244/255, 249/255, 241/255))
        ax3.fill_between([0, xmax], [0, 0], [self.rta.chief_radius, self.rta.chief_radius], color=(255/255, 239/255, 239/255))
        ax3.plot([0, xmax], [self.rta.chief_radius, self.rta.chief_radius], 'k--', linewidth=2)
        ax3.plot(range(len(array)), v[:, 0], 'b', linewidth=2)
        ax3.set_xlim([0, xmax])
        ax3.set_ylim([0, ymax])
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel(r'$r_H$ [m]')
        ax3.set_title('Chief Collision Constraint')
        ax3.grid(True)

        h = np.zeros(len(array))
        for i in range(len(array)):
            h[i] = -np.dot(array[i, 0:3], self.rta.e_hat)/np.linalg.norm(array[i, 0:3]) + np.cos(self.rta.theta)
        ax4 = fig.add_subplot(234)
        xmax = len(array)*1.1
        ymax = np.max(h)*1.1
        ax4.fill_between([0, xmax], [0, 0], [ymax, ymax], color=(244/255, 249/255, 241/255))
        ax4.fill_between([0, xmax], [-ymax*0.1, -ymax*0.1], [0, 0], color=(255/255, 239/255, 239/255))
        ax4.plot([0, xmax], [0, 0], 'k--', linewidth=2)
        ax4.plot(range(len(array)), h, 'b', linewidth=2)
        ax4.set_xlim([0, xmax])
        ax4.set_ylim([0, ymax])
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel(r'$\phi(x)$')
        ax4.set_title('Sun Avoidance Constraint')
        ax4.grid(True)

        # ax4 = fig.add_subplot(234)
        # xmax = len(array)*1.1
        # ymax = self.rta.x_vel_limit*1.2
        # ax4.fill_between([0, xmax], [self.rta.x_vel_limit, self.rta.x_vel_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        # ax4.fill_between([0, xmax], [-ymax, -ymax], [self.rta.x_vel_limit, self.rta.x_vel_limit], color=(255/255, 239/255, 239/255))
        # ax4.fill_between([0, xmax], [-self.rta.x_vel_limit, -self.rta.x_vel_limit], [self.rta.x_vel_limit, self.rta.x_vel_limit], color=(244/255, 249/255, 241/255))
        # ax4.plot([0, xmax], [self.rta.x_vel_limit, self.rta.x_vel_limit], 'k--', linewidth=2)
        # ax4.plot([0, xmax], [-self.rta.x_vel_limit, -self.rta.x_vel_limit], 'k--', linewidth=2)
        # ax4.plot(range(len(array)), array[:, 3], 'b', linewidth=2)
        # ax4.plot(range(len(array)), array[:, 4], 'r', linewidth=2)
        # ax4.plot(range(len(array)), array[:, 5], 'm', linewidth=2)
        # ax4.set_xlim([0, xmax])
        # ax4.set_ylim([-ymax, ymax])
        # ax4.set_xlabel('Time [s]')
        # ax4.set_ylabel('Velocity [m/s]')
        # ax4.set_title('Max Velocity Constraint')
        # ax4.grid(True)

        # ax5 = fig.add_subplot(235)
        # ax5.plot([0, xmax], [1, 1], 'k--', linewidth=2)
        # ax5.plot([0, xmax], [-1, -1], 'k--', linewidth=2)
        # ax5.plot(range(len(control)), control[:, 0], 'b', linewidth=2)
        # ax5.plot(range(len(control)), control[:, 1], 'r', linewidth=2)
        # ax5.plot(range(len(control)), control[:, 2], 'm', linewidth=2)
        # ax5.set_xlim([0, xmax])
        # ax5.set_xlabel('Time [s]')
        # ax5.set_ylabel('Force [N]')
        # ax5.set_title('Actions')
        # ax5.grid(True)


env = Env()
rta = InspectionRTA()
env.run_episode(rta)

plt.show()
