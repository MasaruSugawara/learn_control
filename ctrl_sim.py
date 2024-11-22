#!/usr/bin/env python3

# This code is an arrangement of casadi_mpc_nyuumon:
# https://github.com/proxima-technology/casadi_mpc_nyuumon.git

import casadi
from dynamical_system import *
from controller import *

class Simulator:
  def __init__(self, sys: Dynamical_System, ctrl: Controller):
    self.sys = sys
    self.ctrl = ctrl
    self.history = []

  def set_aim(self, x_init, x_ref, u_ref):
    self.sys.set_init(x_init)
    self.ctrl.set_ref(x_ref, u_ref)

  # advance the system by the step time sys.dt
  # return True if advanced, False if failed
  def advance_dt(self):
    sys_state = self.sys.observe()
    self.ctrl.put_data(sys_state)
    u = self.ctrl.ctrl_out()
    if u is not None:
      self.sys.update(u)
      t, x = sys_state
      self.history.append((t, x.full(), u.full()))
      return True
    else:
      return False
  
  # advance the system time by T as N steps
  # return False if failed at some advancing step; otherwise return True
  def execute(self, T: float, N: int):
    if N <= 0:
      return False
    
    dt = (T - 0.0) / N
    self.sys.set_dt(dt)
    for i in range(N):
      if not self.advance_dt():
        return False
    return True

  # execute until observable state vector remain stationary at target x_ref for exam_period
  # return True if attained, False otherwise
  def execute_until_stationary(self, maxT = 100, dt = 0.01, exam_period = 1.0, threshold = 0.01):
    attained_time = 0.0
    is_attained = False
    self.sys.set_time(0.0)
    self.sys.set_dt(dt)
    self.ctrl.reset()
    while self.sys.t < maxT:
      if not self.advance_dt():
        break

      t, x = self.sys.observe()
      if casadi.norm_2(x - self.ctrl.x_ref) < threshold:
        if is_attained:
          if t - attained_time >= exam_period:
            return True
        else:
          is_attained = True
          attained_time = t
      else:
        is_attained = False
    return False
