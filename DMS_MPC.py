from dynamics_control_simulator import *
import multiprocessing as mp
from tqdm import tqdm
import os

class Damped_Mass_Spring_Model_MPC_Simulator(Simulator):
  def __init__(self, m = 1, k = 1, c = 0.5):
    model = Damped_Mass_Spring_Model(m, k, c)
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
    plt.plot(self.time, x_array[:, 0], label='Position')
    plt.plot(self.time, x_array[:, 1], label='Velocity')
    plt.plot(self.time, np.append(u_array[:, 0], None), label='Control Input')
#    plt.plot(self.time, self.history_attained, label='Attained')
    plt.xlabel('Time (s)')
    plt.ylabel('State')
    title_str = 'Damped-Mass-Spring Model MPC Simulation '
    title_str += '(m, k, c) = (%.1f, %.1f, %.1f);\n' % (self.sys.model.m, self.sys.model.k, self.sys.model.c)
    title_str += '(T, dt) = (%.3f, %.3f); ' % (self.ctrl.T, self.ctrl.dt)
    title_str += '(Qx, Qv, Ru, Qfx, Qfv) = (%.1f, %.1f, %.1f, %.1f, %.1f)\n' % (self.ctrl.Q[0][0], self.ctrl.Q[1][1], self.ctrl.R[0][0], self.ctrl.Q_f[0][0], self.ctrl.Q_f[1][1])
    title_str += 'Attained in %.3f sec' % (self.time[-1])
    plt.title(title_str)
    plt.legend()
    plt.grid(True)
    plt.show()

class MPC_param:
   def __init__(self, param_horizon, param_cost):
    self.K, self.T = param_horizon
    self.dt = self.T / self.K
    self.Qx, self.Qv, self.Ru, self.Qfx, self.Qfv = param_cost

class Simulation_result:
  def __init__(self, mpc_param, x, u, t, duration, is_attained):
    self.mpc_param = mpc_param
    self.x = x
    self.u = u
    self.time = t
    self.duration = duration
    self.is_attained = is_attained

def simulate(param):
    dms = Damped_Mass_Spring_Model_MPC_Simulator(1.0, 1.0, 0.5)
    maxT = 10
    dt = 0.01

    dms.ctrl.set_horizon(param.K, param.K*param.dt)
    dms.ctrl.set_constraint(np.array([-0.01, -0.01]), np.array([10.01, 10.01]), np.array([-10.0]), np.array([10.0]))
    dms.ctrl.set_cost(np.diag([param.Qx, param.Qv]), np.diag([param.Ru]), np.diag([param.Qfx, param.Qfv]))
    dms.set_aim([0, 0], [10, 0], [10])
    dms.ctrl.set_solver()
    
    try:
      is_attained = dms.execute_until_stationary(maxT, dt, max_u=10.0)
      x_array = np.array([x.full().flatten() for x in dms.history_x])
      u_array = dms.history_u
      ret = Simulation_result(param, x_array, u_array, dms.time, dt*len(x_array), is_attained)
    except:
      ret = None
      
    return ret

def parallel_sim(param_list):
    with mp.Pool(mp.cpu_count()) as pool:  # CPUコアの数だけプロセスを作成
        result = list(tqdm(pool.imap(simulate, param_list), total=len(param_list)))  # 並列に適用
    return result

def random_trial():
    best_dur = 100
    n_trial = 100
    K = 1000
    T = 0.01 * K
    Qx = 100.0
    Qv = 0.0
    Ru = 0.0
    Qfx = 1.0
    Qfv = 1.0
#    X_list = np.random.uniform(0.0, 100.0, n_trial)
    X_list = np.array([50]*n_trial)
    Y_list = np.random.uniform(1000.0, 10000.0, n_trial)
    param_list = [MPC_param((K, 0.01*K), (Qx, Qv, Ru, Qfx, Qfv)) for K, Qx in zip(X_list, Y_list)]
    result_list = parallel_sim(param_list)

    for res in result_list:
       if res is not None and res.is_attained and res.duration < best_dur:
          best_res = res
          best_dur = res.duration
    
    print(vars(best_res.mpc_param), best_res.duration)

    plt.figure(figsize=(10, 6))
    plt.xlabel('K')
    plt.ylabel('Qx')
    plt.title('Damped-Mass-Spring Model MPC Simulation')
    scatter = plt.scatter([res.mpc_param.K for res in result_list if res is not None and res.is_attained], [res.mpc_param.Qx for res in result_list if res is not None and res.is_attained], c=[res.duration for res in result_list if res is not None and res.is_attained], cmap='cool')
    plt.colorbar(scatter, label='duration')
    plt.grid(True)
    plt.show()

def single_test():
  dms = Damped_Mass_Spring_Model_MPC_Simulator(1.0, 1.0, 0.5)
  dms.ctrl.set_horizon(50, 0.5)
  dms.ctrl.set_constraint(np.array([-0.01, -0.01]), np.array([10.01, 10.01]), np.array([0.0]), np.array([10.0]))
  dms.ctrl.set_cost(np.diag([5098.0, 0.0]), np.diag([0.0]), np.diag([1.0, 1.0]))
  dms.set_aim([0, 0], [10, 0], [10])
  dms.ctrl.set_solver()
  dms.execute_until_stationary()
  dms.plot_result()

single_test()
