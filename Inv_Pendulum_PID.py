#!/usr/bin/env python3

from dynamics_control_simulator import *
from models import *
import multiprocessing as mp
from tqdm import tqdm
import os

class Inverted_Pendulum_Model_PID_Simulator(Simulator):
  def __init__(self, M = 1.0, m = 0.2, l = 1.0, kp = 0.0, ki = 0.0, kd= 0.0):
    model = Inverted_Pendulum_Model(M, m, l)
    sys = System(model)
    sys.update_by_euler = False
    ctrl = PID_Controller(sys, kp, ki, kd)
    super().__init__(ctrl)

  def plot_result(self):
    plt.figure(figsize=(10, 6))
    # 結果をNumPy配列に変換
    x_array = np.array([x.full().flatten() for x in self.history_x])
    u_array = np.array([u.full().flatten() for u in self.history_u])
    # 結果をプロット
    title_str = 'Inverted Pendulum Model PID Simulation \n'
    title_str += '(M, m, l) = (%.1f, %.1f, %.1f)' % (self.sys.model.M, self.sys.model.m, self.sys.model.l)
    plt.suptitle(title_str)

    plt.subplot(2, 3, 1)
    plt.plot(self.time, x_array[:, 0], label='x')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(self.time, x_array[:, 2], label='theta')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(self.time, np.append(u_array[:, 0], None), label='Control Input')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(self.time, x_array[:, 1], label='x_dot')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(self.time, x_array[:, 3], label='theta_dot')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.show()

def f_err(x, x_ref):
  return x_ref[0] - x[0] - 10*(np.sin(x[1]) - np.sin(x_ref[1]))

def single_test():
  dms = Inverted_Pendulum_Model_PID_Simulator(1.0, 1.0, 0.5, 10.0, 1.0, 1.0)
  dms.ctrl.set_error_func(f_err)
  dms.set_aim([0.0, 3.14159, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0])
  dms.execute_until_stationary()
  dms.plot_result()

single_test()
