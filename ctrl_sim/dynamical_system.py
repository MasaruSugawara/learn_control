import casadi
from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Callable

class Dynamical_System:
  # Dimension of state vectors and control input
  def __init__(self, nx: int, nu: int, param: dict, dt=0.1):
    self.nx = nx                        # dimension of state vector
    self.nu = nu                        # dimension of control input vector
    self.x = casadi.DM.zeros(nx)        # state vector
    self.param = param                  # system's parameter
    self.f = self.make_f()              # equation of the dynamical system
    self.t = 0.0                        # current system time
    self.dt = dt                        # step time 
    self.I = self.make_integrator(dt)   # integrator
    self.update_by_euler = False        # update the system by forward Euler method (for reduced computation time)
    self.observer: Callable[[casadi.DM], casadi.DM] = lambda x: x # observer of the state of the system
    self.history = []                   # history of system state

  # equation of system: x' = f(x, u)
  # where x is a state vector and u is control input.
  # *** make_f() must be defined in subclass *** 
  # f = f(x, u) must be a casadi function object which takes 
  # x: state vector of dim nx and u: control input vector of dim nu
  # as arguments and returns nx-dimensional vector as time differential of state vector
  @abstractmethod
  def make_f(self) -> casadi.Function:
    pass

  # set a state of the system
  def set_init(self, x_init: np.array):
    self.x = casadi.DM(x_init)

  # advance the time of the dynamical system using ODE integrator
  def make_integrator(self, dt: float) -> casadi.Function:
    states = casadi.SX.sym("states", self.nx)
    ctrls = casadi.SX.sym("ctrls", self.nu)
    ode = self.f(x=states, u=ctrls)["x_dot"]
    dae = {"x":states,"p":ctrls,"ode":ode}

    I = casadi.integrator("I", "cvodes", dae, 0, dt)
    return I

  # wipe history data  
  def clear_history(self):
    self.history = []

  # set system time
  # if system time is changed, history should be cleared (but can be optout)
  def set_time(self, t: float, clear_history = True):
    self.t = t
    if clear_history:
      self.clear_history()
  
  # change step time
  def set_dt(self, dt: float):
    if dt > 0:
      self.dt = dt
      self.I = self.make_integrator(dt)

  # compute the next state if control input u were applied
  def next_state(self, u: casadi.DM):
    if self.update_by_euler:
      x_pred = self.x + self.dt * self.f(x=self.x, u=u)["x_dot"]
    else:
      x_pred = self.I(x0=self.x, p=u)["xf"]
      
    return x_pred
  
  # advance the time of system by dt
  def update(self, u: casadi.DM):
    self.history.append((self.t, self.x.full()[:, 0], u.full()[:, 0]))
    x_pred = self.next_state(u)
    self.x = x_pred
    self.t += self.dt
  
  # set parameters
  def set_param(self, param: dict):
    self.param = param
    self.f = self.make_f() # update the equation

  # observe system state
  def observe(self) -> tuple[float, casadi.DM]:
    return (self.t, self.observer(self.x))
  
  def set_observer(self, obs: Callable[[casadi.DM], casadi.DM]):
    self.observer = obs
  
  # plot system state (default)
  def plot_state(self, ax: Axes, idxlist=[]):
    hist = self.history
    t_axis = [v[0] for v in hist]
    t_axis.append(self.t)
    cmap = plt.get_cmap('tab10')
    if not idxlist:
      idxlist = range(self.nx)

    for i in idxlist:
      x_values = [v[1][i] for v in hist]
      x_values.append(self.x.full()[i][0])
      ax.plot(t_axis, x_values, label=f'x_{i}', color=cmap(i))
      ax.scatter(self.t, self.x[i].full()[0], color=cmap(i))

  # plot control history
  def plot_control(self, ax: Axes, idxlist=[]):
    hist = self.history
    t_axis = [v[0] for v in hist]
    cmap = plt.get_cmap('tab10')
    if not idxlist:
      idxlist = range(self.nu)
    for i in idxlist:
      ax.plot(t_axis, [v[2][i] for v in hist], label=f'u_{i}', color=cmap(self.nx + i))

  # plot the history of state and control simultaneously
  def plot(self, ax: Axes, state_idx=[], control_idx=[]):
    self.plot_state(ax, state_idx)
    self.plot_control(ax, control_idx)

# Examples of dynamical systems

class Damped_Mass_Spring_System(Dynamical_System):
  def __init__(self, m = 1, k = 1, c = 0.5):
    super().__init__(2, 1, {'m': m, 'k': k, 'c': c})

  def make_f(self):
    states = casadi.SX.sym("states", self.nx) # [x, x_dot]
    ctrls = casadi.SX.sym("ctrls", self.nu)   # pulling force
    m = self.param['m'] # mass
    k = self.param['k'] # spring constant
    c = self.param['c'] # damper

    A = casadi.DM([
        [0,1],
        [-k/m,-c/m]
    ])
    B = casadi.DM([
        [0],
        [1/m]
    ])
    states_dot = A @ states + B @ ctrls
    f = casadi.Function("f",[states,ctrls],[states_dot],['x','u'],['x_dot'])
    return f

class Lotka_Volterra_System(Dynamical_System):
  def __init__(self, a = 0.06, b = 0.06, c = 0.06, d = 0.06, p = 0.024, q = 0.012):
    super().__init__(2, 1, {'a': a, 'b': b, 'c': c, 'd': d, 'p': p, 'q': q})

  def make_f(self):
    states = casadi.SX.sym("states", self.nx)
    ctrls = casadi.SX.sym("ctrls", self.nu)

    x = states[0]; y = states[1]; u = ctrls[0]

    a = self.param['a']; b = self.param['b']; c = self.param['c']; d = self.param['d'];
    p = self.param['p']; q = self.param['q'];

    # Lotka-Volterra equation with hunting (to control animal population)
    x_dot = a*x   - b*x*y - p*u
    y_dot = c*x*y - d*y   - q*u

    states_dot = casadi.vertcat(x_dot, y_dot)
    f = casadi.Function("f",[states,ctrls],[states_dot],['x','u'],['x_dot'])
    return f
  

class Inverted_Pendulum_System(Dynamical_System):
  def __init__(self, M = 1.0, m = 0.2, l = 1.0):
    super().__init__(4, 1, {'M': M, 'm': m, 'l': l})

  def make_f(self):
    g = casadi.DM(9.81) # gravitational acceleration

    # [x, theta, x', theta'] where x: position of box, 
    # theta: angle of pendulum; theta==0 when standing erect
    states = casadi.SX.sym("states", self.nx) 
    ctrls = casadi.SX.sym("ctrls", self.nu)   # force of pulling the box 

    M = self.param['M'] # mass of box
    m = self.param['m'] # mass of pendulum
    l = self.param['l'] # length of pendulum

    _, theta, x_dot, theta_dot = casadi.vertsplit(states)
    F = ctrls[0]

    sin = casadi.sin(theta)
    cos = casadi.cos(theta)
    det = M+m*sin**2

    x_ddot = (-m*l*sin*theta_dot**2+m*g*sin*cos+F)/det
    theta_ddot = (-m*l*sin*cos*theta_dot**2+(M+m)*g*sin+F*cos)/(l*det)

    states_dot = casadi.vertcat(x_dot,theta_dot,x_ddot,theta_ddot)

    f = casadi.Function("f",[states,ctrls],[states_dot],['x','u'],['x_dot'])
    return f
  