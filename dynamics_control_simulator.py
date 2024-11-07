#!/usr/bin/env python3

# This code is an arrangement of casadi_mpc_nyuumon:
# https://github.com/proxima-technology/casadi_mpc_nyuumon.git

import numpy as np
import matplotlib.pyplot as plt
import casadi
import math

class Dynamics_Model:
  def __init__(self, nx, nu):
    self.nx = nx
    self.nu = nu

class System:
  def __init__(self, model):
    self.model = model
    self.f = model.f
    self.nx = model.nx
    self.nu = model.nu
    self.update_by_euler = False

  def set_init(self, x_init):
    self.x = x_init

  def make_integrator(self, dt):
    states = casadi.SX.sym("states", self.nx)
    ctrls = casadi.SX.sym("ctrls", self.nu)
    ode = self.f(x=states, u=ctrls)["x_dot"]
    dae = {"x":states,"p":ctrls,"ode":ode}

    I = casadi.integrator("I", "cvodes", dae, 0, dt)
    return I

  def update(self, u, dt):
    if self.update_by_euler:
      self.x += dt * self.f(x=self.x, u=u)["x_dot"]
    else:
      I = self.make_integrator(dt)
      self.x = I(x0=self.x, p=u)["xf"]

class Controller:
  def __init__(self, sys):
    self.sys = sys

  def set_ref(self, x_ref, u_ref):
    self.x_ref = x_ref
    self.u_ref = u_ref

class Const_Controller(Controller):
  def __init__(self, sys):
    super().__init__(sys)
    self.gain = np.zeros(self.sys.nu)

  def set_gain(self, gain):
    self.gain = casadi.DM(gain)

  def ctrl_out(self, dt):
    return self.gain
  
  def reset(self):
    pass

class Fool_Controller(Controller):
  def __init__(self, sys, gain = 0.3):
    super().__init__(sys)
    self.gain = gain

  def ctrl_out(self):
    if self.sys.x[0] < self.sys.x_ref[0]:
      return self.gain
    else:
      return -self.gain

class PID_Controller(Controller):
  def __init__(self, sys, Kp, Ki, Kd):
    super().__init__(sys)
    self.Kp = Kp
    self.Ki = Ki
    self.Kd = Kd
    self.error_sum = 0
    self.last_error = 0
    self.f_err = lambda x, x_ref: x_ref[0] - x[0]

  def reset(self):
    self.error_sum = 0
    self.last_error = self.f_err(self.sys.x, self.sys.x_ref)
  
  def set_param(self, Kp, Ki, Kd):
    self.Kp = Kp
    self.Ki = Ki
    self.Kd = Kd

  def set_error_func(self, f_err):
    self.f_err = f_err

  def ctrl_out(self, dt):
    error = self.f_err(self.sys.x, self.sys.x_ref)
    self.error_sum += error * dt
    u = self.Kp * error + self.Ki * self.error_sum + self.Kd * (error - self.last_error) / dt
    self.last_error = error
    return u

class MPC_Controller(Controller):
  def __init__(self, sys):
    super().__init__(sys)
    self.Q = casadi.diag([1.0]*sys.nx)
    self.R = casadi.diag([1.0]*sys.nu)
    self.Q_f = casadi.diag([1.0]*sys.nx)
    self.x_lb = np.array([-np.inf]*sys.nx)
    self.x_ub = np.array([np.inf]*sys.nx)
    self.u_lb = np.array([-np.inf]*sys.nu)
    self.u_ub = np.array([np.inf]*sys.nu)

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
    
  def set_solver(self, is_qp = False):
    RK4 = self.make_RK4()
    X = [casadi.SX.sym(f"x_{k}", self.sys.nx) for k in range(self.K+1)]
    U = [casadi.SX.sym(f"u_{k}", self.sys.nu) for k in range(self.K)]
    G = []

    J = 0
    for k in range(self.K):
        J += self.compute_stage_cost(X[k], U[k]) * self.dt
        eq = X[k+1] - RK4(x=X[k],u=U[k])["x_next"]
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

  def ctrl_out(self, dt):
    u_opt = self.compute_optimal_control(self.S)
    return u_opt
  
  def reset(self):
    pass

class Simulator:
  def __init__(self, ctrl):
    self.ctrl = ctrl
    self.sys = ctrl.sys

  def set_aim(self, x_init, x_ref, u_ref):
    self.sys.x = casadi.DM(x_init)
    self.sys.x_ref = casadi.DM(x_ref)
    self.sys.u_ref = casadi.DM(u_ref)

  def execute(self, T, N):
    dt = (T - 0) / (N - 1)
    self.time = np.linspace(0, T, N)
    self.history_x = [self.sys.x]
    self.history_u = []
    for t in self.time[:-1]:
      u = self.ctrl.ctrl_out()
      self.sys.update(u, dt)
      self.history_x.append(self.sys.x)
      self.history_u.append(u)

  def execute_until_stationary(self, maxT = 100, dt = 0.01, exam_period = 1.0, threshold = 0.01, max_u = np.inf):
    t = 0
    attained_time = 0
    is_attained = False
    self.time = [0]
    self.dt = dt
    self.history_x = [self.sys.x]
    self.history_u = []
    self.history_attained = [0.0]
    self.ctrl.reset()
    while t < maxT:
      u = self.ctrl.ctrl_out(dt)
      u = min(casadi.DM(max_u), u)
      u = max(casadi.DM(-max_u), u)
      self.sys.update(u, dt)
      t += dt
      self.time.append(t)
      self.history_x.append(self.sys.x)
      self.history_u.append(u)
      if casadi.norm_2(self.sys.x - self.sys.x_ref) < threshold:
        self.history_attained.append(1.0)
        if is_attained:
          if t - attained_time >= exam_period:
            return True
        else:
          is_attained = True
          attained_time = t
      else:
        self.history_attained.append(0.0)
        is_attained = False
    return False
