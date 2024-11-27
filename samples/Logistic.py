import casadi
from casadi.casadi import Function
from ctrl_sim import *

class Logistic_System(Dynamical_System):
  def __init__(self, a = 1.0):
    super().__init__(1, 1, {'a': a}, 1.0)
    self.update_by_euler = True # since discrete system
  
  def make_f(self) -> Function:
    states = casadi.SX.sym("states", self.nx) 
    ctrls = casadi.SX.sym("ctrls", self.nu) 

    a = self.param['a'] # 
    x = states[0];
    u = ctrls[0]
    # x_{n+1}       = a*x_n*(1-x_n) - u
    # x_{n+1} - x_n = -a*x_n^2 + (a-1)*x_n - u

    states_dot = casadi.vertcat(-a*x*x + (a-1)*x - u)

    f = casadi.Function("f",[states,ctrls],[states_dot],['x','u'],['x_dot'])
    return f


def test():
  a = 3.6
  fig = plt.figure(figsize=(20, 8))
  fig.suptitle(f'Logistic system $x_{{n+1}} = {a} x_n(1-x_n) - u$')

  sim = Simulator(Logistic_System(a=a), Const_Controller(1, 1))
  sim.set_aim(np.array([0.1]), np.array([0.577]), np.array([0.0]))
  sim.execute_until_stationary(100, 1.0, 10.0)

  ax = fig.add_subplot(1,3,1)
  ax.set_title('No control')
  sim.sys.plot(ax)
  ax.legend()

  sim = Simulator(Logistic_System(a), PID_Controller(1, 1))
  sim.ctrl.set_pid(np.array([[0.3]]), np.array([[0.0072]]), np.array([[1.0]]))
  sim.ctrl.set_ctrl_range(np.array([0.0]), np.array([1.0]))
  sim.set_aim(np.array([0.1]), np.array([0.577]), np.array([0.0]))
  sim.execute_until_stationary(100, 1.0, 10.0)
  
  ax = fig.add_subplot(1,3,2)
  ax.set_title('PID control')
  sim.sys.plot(ax)
  ax.legend()
  
  sim = Simulator(Logistic_System(a), MPC_Controller(1, 1))
  sim.ctrl.set_model(sim.sys.f)
  sim.ctrl.set_horizon(20, 20)
  sim.ctrl.set_cost(np.array([[1.0]]), np.array([[0.0]]), np.array([[1.0]]))
  sim.ctrl.set_constraint(np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([10.0]))
  sim.set_aim(np.array([0.1]), np.array([0.577]), np.array([0.0]))
  sim.ctrl.set_solver(is_euler = True)
  sim.execute_until_stationary(100, 1.0, 10.0)
  
  ax = fig.add_subplot(1,3,3)
  ax.set_title('MPC control')
  sim.sys.plot(ax)
  ax.legend()
  plt.show()

if __name__ == '__main__':
  test()
