from dynamics_control_simulator import *

class Damped_Mass_Spring_Model(Dynamics_Model):
  def __init__(self, m = 1, k = 1, c = 0.5):
    super().__init__(2, 1)
    self.set_param(m, k, c)

  def set_param(self, m, k, c):
    self.m = m
    self.k = k
    self.c = c
    self.f = self.make_f(m, k, c)

  def make_f(self, m, k, c):
    states = casadi.SX.sym("states", self.nx)
    ctrls = casadi.SX.sym("ctrls", self.nu)

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

class Inverted_Pendulum_Model(Dynamics_Model):
  def __init__(self, M = 1.0, m = 0.2, l = 1.0):
    super().__init__(4, 1)
    self.set_param(M, m, l)

  def set_param(self, M, m, l):
    self.M = M
    self.m = m
    self.l = l
    self.f = self.make_f(M, m, l)

  def make_f(self, M, m, l):
    g = casadi.DM(9.81)
    states = casadi.SX.sym("states", self.nx)
    ctrls = casadi.SX.sym("ctrls", self.nu)

    x = states[0]
    theta = states[1]
    x_dot = states[2]
    theta_dot = states[3]
    F = ctrls[0]

    sin = casadi.sin(theta)
    cos = casadi.cos(theta)
    det = M+m*sin**2

    x_ddot = (-m*l*sin*theta_dot**2+m*g*sin*cos+F)/det
    theta_ddot = (-m*l*sin*cos*theta_dot**2+(M+m)*g*sin+F*cos)/(l*det)

    states_dot = casadi.vertcat(x_dot,theta_dot,x_ddot,theta_ddot)

    f = casadi.Function("f",[states,ctrls],[states_dot],['x','u'],['x_dot'])
    return f
  