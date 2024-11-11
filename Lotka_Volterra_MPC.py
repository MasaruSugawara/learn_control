#!/usr/bin/env python3

from dynamics_control_simulator import *
from models import *
import multiprocessing as mp
from tqdm import tqdm
import os
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


class Lotka_Volterra_Model_MPC_Simulator(Simulator):
  def __init__(self):
    model = Lotka_Volterra_Model()
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
    title_str = 'Lotka Volterra Model MPC Simulation'
    plt.suptitle(title_str)

    plt.subplot(1, 2, 1)
    plt.plot(self.time, x_array[:, 0], label='x')
    plt.plot(self.time, x_array[:, 1], label='y')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(self.time, np.append(u_array[:, 0], None), label='u')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.show()

def single_test():
  dms = Lotka_Volterra_Model_MPC_Simulator()

  dms.ctrl.set_horizon(30, 30.0)
  dms.ctrl.set_constraint(np.array([0.0]*2), np.array([np.inf]*2), np.array([0.0]), np.array([1.0]))
  dms.ctrl.set_cost(np.diag([1.0, 1.0]), np.diag([0.05]), np.diag([1.0, 1.0]))
  dms.set_aim([0.5, 0.7], [1.0, 1.0], [0.0])
  dms.ctrl.set_solver(is_euler=False)
  dms.execute_until_stationary(maxT=1000, dt=1.0, exam_period=100.0)
  dms.plot_result()

if __name__ == '__main__':
   single_test()
