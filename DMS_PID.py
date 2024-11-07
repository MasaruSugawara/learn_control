#!/usr/bin/env python3

from dynamics_control_simulator import *
from models import *
import multiprocessing as mp
from tqdm import tqdm

class Damped_Mass_Spring_Model_PID_Simulator(Simulator):
  def __init__(self, m = 1, k = 1, c = 0.5, kp = 0.0, ki = 0.0, kd= 0.0):
    model = Damped_Mass_Spring_Model(m, k, c)
    sys = System(model)
    sys.update_by_euler = False
    ctrl = PID_Controller(sys, kp, ki, kd)
    super().__init__(ctrl)

  def plot_result(self):
    plt.figure(figsize=(10, 6))
    # 結果をNumPy配列に変換
    x_array = np.array([x.full().flatten() for x in self.history_x])
    u_array = np.array([u.full().flatten() for u in self.history_u])[:, 0]
    u_array = np.append(u_array, None)
    # 結果をプロット
    plt.plot(self.time, x_array[:, 0], label='Position')
    plt.plot(self.time, x_array[:, 1], label='Velocity')
    plt.plot(self.time, u_array, label='Control Input')
#    plt.plot(self.time, self.history_attained, label='Attained')
    plt.xlabel('Time (s)')
    plt.ylabel('State')
    title_str = 'Damped-Mass-Spring Model PID Simulation\n'
    title_str += '(m, k, c) = (%.1f, %.1f, %.1f); ' % (self.sys.model.m, self.sys.model.k, self.sys.model.c)
    title_str += '(kp, ki, kd) = (%.3f, %.3f, %.3f)\n' % (self.ctrl.Kp, self.ctrl.Ki, self.ctrl.Kd)
    title_str += 'Attained in %.3f sec' % (self.time[-1])
    plt.title(title_str)
    plt.legend()
    plt.grid(True)
    plt.show()

class Simulation_result:
  def __init__(self, kp, ki, kd, x, u, t, duration, is_attained):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.x = x
    self.u = u
    self.time = t
    self.duration = duration
    self.is_attained = is_attained

def simulate(param):
    dms = Damped_Mass_Spring_Model_PID_Simulator(1.0, 1.0, 0.5)
    kp, ki, kd = param
    maxT = 10
    dt = 0.001
    dms.set_aim([0, 0], [10, 0], [10])
    dms.ctrl.set_param(kp, ki, kd)
    dms.ctrl.reset()
    is_attained = dms.execute_until_stationary(maxT, dt, max_u=10.0)
    x_array = np.array([x.full().flatten() for x in dms.history_x])
    u_array = dms.history_u
    res = Simulation_result(kp, ki, kd, x_array, u_array, dms.time, dt*len(x_array), is_attained)    
    
    return res

def parallel_sim(param_list):
    with mp.Pool(mp.cpu_count()) as pool:  # CPUコアの数だけプロセスを作成
        result = list(tqdm(pool.imap(simulate, param_list), total=len(param_list)))  # 並列に適用
    return result

def random_trial():
    best_dur = 100
    n_trial = 1000
    kp = 10.0
    ki_list = np.random.uniform(0.8, 0.9, n_trial)
    kd_list = np.random.uniform(5.0, 8.0, n_trial)
    param_list = [[kp, ki, kd] for ki, kd in zip(ki_list, kd_list)]
    result_list = parallel_sim(param_list)

    for res in result_list:
       if res.is_attained and res.duration < best_dur:
          best_res = res
          best_dur = res.duration
    
    print(best_res.kp, best_res.ki, best_res.kd, best_res.duration)

    plt.figure(figsize=(10, 6))
    plt.xlabel('ki')
    plt.ylabel('kd')
    plt.title('Damped-Mass-Spring Model PID Simulation')
    scatter = plt.scatter([res.ki for res in result_list if res.is_attained], [res.kd for res in result_list if res.is_attained], c=[res.duration for res in result_list if res.is_attained], cmap='viridis')
    plt.colorbar(scatter, label='duration')
    plt.legend()
    plt.grid(True)
    plt.show()

def single_test():
    dms = Damped_Mass_Spring_Model_PID_Simulator(1.0, 1.0, 0.5, 10.0, 0.8525740673895923, 5.249060720959527)
    dms.set_aim([0, 0], [10, 0], [10])
    dms.execute_until_stationary(10, 0.01, max_u=10.0)
    dms.plot_result()

single_test()

