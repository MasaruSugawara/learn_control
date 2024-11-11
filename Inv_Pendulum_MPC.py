#!/usr/bin/env python3

from dynamics_control_simulator import *
from models import *
import multiprocessing as mp
from tqdm import tqdm
import os
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


class Inverted_Pendulum_Model_MPC_Simulator(Simulator):
  def __init__(self, M = 1.0, m = 0.2, l = 1.0):
    model = Inverted_Pendulum_Model(M, m, l)
    sys = System(model)
    sys.update_by_euler = False
    ctrl = MPC_Controller(sys)
    super().__init__(ctrl)

  def plot_result(self):
    plt.figure(figsize=(10, 6))
    # 結果をNumPy配列に変換
    x_array = np.array([x.full().flatten() for x in self.history_x])
    u_array = np.array([u.full().flatten() for u in self.history_u])
    # 結果をプロット
    title_str = 'Inverted Pendulum Model MPC Simulation \n'
    title_str += '(M, m, l) = (%.1f, %.1f, %.1f)' % (self.sys.model.M, self.sys.model.m, self.sys.model.l)
    plt.suptitle(title_str)

    plt.subplot(2, 3, 1)
    plt.plot(self.time, x_array[:, 0], label='x')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(self.time, x_array[:, 1], label='theta')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(self.time, np.append(u_array[:, 0], None), label='Control Input')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(self.time, x_array[:, 2], label='x_dot')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(self.time, x_array[:, 3], label='theta_dot')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.show()

  def update_figure(self, i):
    x_lim_min = -4
    x_lim_max = 4
    y_lim_min = -2
    y_lim_max = 2
    u_scale = 15
    ax = self.ax

    ax.cla()
    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(y_lim_min, y_lim_max)
    ax.set_aspect("equal")
    ax.set_title(f"t={self.time[i]:0.2f}")

    x,theta,_,_ = self.x_array[i]
    u = self.u_array[i]

    l = self.sys.model.l
    points = np.array([
        [x,x-l*np.sin(theta)],
        [0,l*np.cos(theta)]
    ])

    ax.hlines(0,x_lim_min,x_lim_max,colors="black")
    ax.scatter(*points,color="blue", s=50)
    ax.plot(*points, color='blue', lw=2)
    ax.arrow(x,0,u/u_scale,0,width=0.02,head_width=0.06,head_length=0.12,length_includes_head=False,color="green",zorder=3)

    w = 0.2
    h = 0.1
    rect = patches.Rectangle(xy=(x-w/2,-h/2), width=w, height=h,color="black")
    ax.add_patch(rect)

  def animation(self):
    fig = plt.figure(figsize=(12,6))
    self.ax = fig.add_subplot(111)
    self.x_array = np.array([x.full().flatten() for x in self.history_x])
    self.u_array = np.append(np.array([u.full().flatten() for u in self.history_u]), 0.0)
    frames = np.arange(0, len(self.time))
    fps = 1 / self.dt
    ani = FuncAnimation(fig, self.update_figure, frames=frames)
    ani.save("inv_pendulum.gif",writer="pillow",fps=fps)


def single_test():
  dms = Inverted_Pendulum_Model_MPC_Simulator(1.0, 1.0, 0.5)

  dms.ctrl.set_horizon(20, 1.0)
  dms.ctrl.set_constraint(np.array([-np.inf]*4), np.array([np.inf]*4), np.array([-10000.0]), np.array([10000.0]))
  dms.ctrl.set_cost(np.diag([2.5, 10.0, 0.01, 0.01]), np.diag([0.01]), np.diag([2.5, 10.0, 0.01, 0.01]))
  dms.set_aim([0.0, 3.14159, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0])
  dms.ctrl.set_solver()
  try:
    dms.execute_until_stationary(maxT=100, dt=0.01)
  except:
    pass
  dms.plot_result()
  dms.animation()

if __name__ == '__main__':
   single_test()
