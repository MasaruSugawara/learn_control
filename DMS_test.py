#!/usr/bin/env python3

from dynamics_control_simulator import *
import multiprocessing as mp
from tqdm import tqdm
import os

class Damped_Mass_Spring_Model_Const_Simulator(Simulator):
  def __init__(self, m = 1, k = 1, c = 0.5):
    model = Damped_Mass_Spring_Model(m, k, c)
    sys = System(model)
    sys.update_by_euler = False
    ctrl = Const_Controller(sys)
    super().__init__(ctrl)

  def plot_result(self):
    plt.figure(figsize=(10, 6))
    # 結果をNumPy配列に変換
    x_array = np.array([x.full().flatten() for x in self.history_x])
    u_array = np.array([u.full().flatten() for u in self.history_u])
    # 結果をプロット
    plt.plot(self.time, x_array[:, 0], label='Position')
    plt.plot(self.time, x_array[:, 1], label='Velocity')
    plt.plot(self.time, np.append(u_array[:, 0], None), label='Control Input')
#    plt.plot(self.time, self.history_attained, label='Attained')
    plt.xlabel('Time (s)')
    plt.ylabel('State')
    title_str = 'Damped-Mass-Spring Model Const Simulation \n'
    title_str += '(m, k, c) = (%.1f, %.1f, %.1f)' % (self.sys.model.m, self.sys.model.k, self.sys.model.c)
    plt.title(title_str)
    plt.legend()
    plt.grid(True)
    plt.show()

def single_test():
  dms = Damped_Mass_Spring_Model_Const_Simulator(1.0, 1.0, 0.5)
  dms.ctrl.set_gain(10.0)
  dms.set_aim([0, 0], [10, 0], [10])
  dms.execute_until_stationary()
  dms.plot_result()

single_test()
