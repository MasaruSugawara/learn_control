import casadi
from abc import abstractmethod
import queue
import numpy as np

class Controller:
  def __init__(self, no: int, nu:int, param: dict = {}, observer_queue_max = 0, discard_old_data = True):
    self.no = no  # dimension of observable
    self.nu = nu  # dimension of control input
    if observer_queue_max >= 0:
      self.observed_que = queue.PriorityQueue(observer_queue_max)
    self.discard_old_data = discard_old_data
    self.param = param
    self.set_ref(casadi.DM.zeros(no), casadi.DM.zeros(nu))

  def set_ref(self, x_ref: np.array, u_ref: np.array):
    self.x_ref = casadi.DM(x_ref)
    self.u_ref = casadi.DM(u_ref)

  def set_param(self, key, param):
    self.param[key] = casadi.DM(param)

  def get_param(self, key):
    val = self.param.get(key, None)
    if val is not None:
      return val.full()

  def put_data(self, d: tuple[float, casadi.DM]):
    t, x = d
    if hasattr(self, 'observed_que'):
      if self.observed_que.full():
        if self.discard_old_data:
          self.observed_que.get()
        else:
          pass # discard new data
      else:
        self.observed_que.put((t, x))

  def get_data(self):
    if hasattr(self, 'observed_que') and not self.observed_que.empty():
      return self.observed_que.get()
    
  @abstractmethod
  def ctrl_out(self):
    pass
  
  @abstractmethod
  def reset(self):
    pass

class Const_Controller(Controller):
  def __init__(self, no: int, nu:int, param: dict = {}):
    super().__init__(no, nu, param, observer_queue_max = -1)
    self.param.setdefault('gain', casadi.DM.zeros(nu))

  def set_gain(self, gain: np.array):
    self.param.update('gain', casadi.DM(gain))

  def ctrl_out(self):
    return self.param.get('gain')
  
  def reset(self):
    pass

class PID_Controller(Controller):
  def __init__(self, no: int, nu:int, param: dict = {}):
    super().__init__(no, nu, param, observer_queue_max = 1)
    self.param.setdefault('kp', casadi.DM.zeros(nu, no))
    self.param.setdefault('ki', casadi.DM.zeros(nu, no))
    self.param.setdefault('kd', casadi.DM.zeros(nu, no))
    self.reset()

  def reset(self):
    self.error_sum = casadi.DM.zeros(self.no)
    self.last_state = None
    self.last_error = None

  def ctrl_out(self):
    d = self.get_data()
    if d is not None:
      t, x = d
      error = self.x_ref - x
      u = self.param.get('kp') @ error
      if self.last_state is not None:
        last_t, _ = self.last_state
        dt = t - last_t
        if dt > 0:
          d_error = (error - self.last_error) / dt
          u += self.param.get('kd') @ d_error

          self.error_sum += error * dt
      u += self.param.get('ki') @ self.error_sum

      self.last_error = error
      self.last_state = d
    return u

class MPC_Controller(Controller):
  def __init__(self, no: int, nu:int, param: dict = {}):
    super().__init__(no, nu, param, observer_queue_max = 1)
    self.param.setdefault('Q', casadi.diag([1.0]*no))
    self.param.setdefault('R', casadi.diag([1.0]*nu))
    self.param.setdefault('Q_f', casadi.diag([1.0]*no))
    self.param.setdefault('x_lb', np.array([-np.inf]*no))
    self.param.setdefault('x_ub', np.array([np.inf]*no))
    self.param.setdefault('u_lb', np.array([-np.inf]*nu))
    self.param.setdefault('u_ub', np.array([np.inf]*nu))
    self.record_opt_history = False
    self.opt_history = []

  def set_horizon(self, horizon_len, period):
    self.K = horizon_len
    self.T = period
    self.dt = period / horizon_len

  def set_cost(self, Q, R, Q_f):
    self.Q = Q
    self.R = R
    self.Q_f = Q_f

  def set_constraint(self, x_lb, x_ub, u_lb, u_ub):
    self.x_lb = x_lb
    self.x_ub = x_ub
    self.u_lb = u_lb
    self.u_ub = u_ub

  def make_euler(self):
    states = casadi.SX.sym("states", self.sys.nx)
    ctrls = casadi.SX.sym("ctrls", self.sys.nu)
    f = self.sys.f
    dt = self.dt
    states_next = states + dt*f(x=states, u=ctrls)["x_dot"]
    Euler = casadi.Function("Euler",[states,ctrls],[states_next],["x","u"],["x_next"])
    return Euler

  def make_RK4(self):
    states = casadi.SX.sym("states", self.sys.nx)
    ctrls = casadi.SX.sym("ctrls", self.sys.nu)
    f = self.sys.f
    dt = self.dt

    r1 = f(x=states,u=ctrls)["x_dot"]
    r2 = f(x=states+dt*r1/2,u=ctrls)["x_dot"]
    r3 = f(x=states+dt*r2/2,u=ctrls)["x_dot"]
    r4 = f(x=states+dt*r3,u=ctrls)["x_dot"]

    states_next = states + dt*(r1+2*r2+2*r3+r4)/6.0

    RK4 = casadi.Function("RK4",[states,ctrls],[states_next],["x","u"],["x_next"])
    return RK4

  def compute_stage_cost(self, x, u):
    x_diff = x - self.sys.x_ref
    u_diff = u - self.sys.u_ref
    cost = (casadi.dot(self.Q @ x_diff, x_diff) + casadi.dot(self.R @ u_diff, u_diff)) / 2
    return cost

  def compute_terminal_cost(self, x):
    x_diff = x - self.sys.x_ref
    cost = casadi.dot(self.Q_f @ x_diff, x_diff) / 2
    return cost
  
  def make_qp(self, X, U, J, G):
    qp = {"x":casadi.vertcat(*X,*U),"f":J,"g":casadi.vertcat(*G)}
    self.S = casadi.qpsol("S","osqp",qp, {
      'error_on_fail': False,
      'osqp': {'verbose': True}
      })
    
  def make_nlp(self, X, U, J, G):
    option = {'print_time':False,'ipopt':{'max_iter':10,'print_level':0}}
    nlp = {"x":casadi.vertcat(*X,*U),"f":J,"g":casadi.vertcat(*G)}
    self.S = casadi.nlpsol("S","ipopt",nlp,option)
    
  def set_solver(self, is_euler = False, is_qp = False):
    if is_euler:
      update_func = self.make_euler()
    else:
      update_func = self.make_RK4()

    X = [casadi.SX.sym(f"x_{k}", self.sys.nx) for k in range(self.K+1)]
    U = [casadi.SX.sym(f"u_{k}", self.sys.nu) for k in range(self.K)]
    G = []

    J = 0
    for k in range(self.K):
        J += self.compute_stage_cost(X[k], U[k]) * self.dt
        eq = X[k+1] - update_func(x=X[k],u=U[k])["x_next"]
        G.append(eq)
    J += self.compute_terminal_cost(X[-1])

    if is_qp:
      self.make_qp(X, U, J, G)
    else:
      self.make_nlp(X, U, J, G)    
    self.x0 = casadi.DM.zeros(self.sys.nx*(self.K+1)+self.sys.nu*self.K)

  def compute_optimal_control(self, S):
    K = self.K
    x_init = self.sys.x.full().ravel().tolist()
    lbx = x_init + self.x_lb.tolist()*K + self.u_lb.tolist()*K
    ubx = x_init + self.x_ub.tolist()*K + self.u_ub.tolist()*K
    lbg = [-1e-8]*self.sys.nx*K
    ubg = [1e-8]*self.sys.nx*K

    res = S(lbx=lbx,ubx=ubx,lbg=lbg,ubg=ubg,x0=self.x0)

    offset = self.sys.nx*(K+1)
    self.x0 = res["x"]
    u_opt = self.x0[offset:offset+self.sys.nu]
    return u_opt
  
  def reshape_opt(self):
    lenx = (self.K + 1) * self.sys.nx
    x_opt = self.x0[:lenx].reshape((self.sys.nx, self.K + 1))
    u_opt = self.x0[lenx:].reshape((self.sys.nu, self.K))
    return [x_opt, u_opt]

  def set_record(self, toggle = True):
    self.record_opt_history = toggle

  def ctrl_out(self, dt):
    u_opt = self.compute_optimal_control(self.S)
    if self.record_opt_history:
      self.opt_history.append(self.reshape_opt())

    return u_opt
  
  def reset(self):
    pass
