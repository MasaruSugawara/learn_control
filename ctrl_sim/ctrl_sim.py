#!/usr/bin/env python3
""" ctrl_sim: Simulator
This code is an arrangement of casadi_mpc_nyuumon:
https://github.com/proxima-technology/casadi_mpc_nyuumon.git
"""

import casadi
from .dynamical_system import *
from .controller import *
from abc import abstractmethod

class Compensator:
  def __init__(self, sys: Dynamical_System, ctrl: Controller):
    self.sys = sys
    self.ctrl = ctrl

  @abstractmethod
  def compensate(self, u: casadi.DM) -> casadi.DM:
    pass

class Passthrough_Compensator(Compensator):
  """ Pass-through compensator

  Trivial compensator which does nothing
  """

  def __init__(self, sys, ctrl):
    super().__init__(sys, ctrl)

  def compensate(self, u):
    return u

class Simulator:
  def __init__(self, sys: Dynamical_System, ctrl: Controller):
    self.sys = sys
    self.ctrl = ctrl
    self.cpst = Passthrough_Compensator(sys, ctrl)
    self.history = []

  def set_aim(self, x_init, x_ref, u_ref):
    self.sys.set_init(x_init)
    self.ctrl.set_ref(x_ref, u_ref)

  def set_compensator(self, comp: Compensator):
    self.cpst = comp
  
  def advance_dt(self):
    """ advance the system by dt
    advance the system by the step time sys.dt

    return True if advanced, False if failed
    """
    sys_state = self.sys.observe()
    self.ctrl.put_data(sys_state)
    u = self.ctrl.ctrl_out()
    if u is not None:
      u = self.cpst.compensate(u)
      self.sys.update(u)
      t, x = sys_state
      self.history.append((t, x.full(), u.full()))
      return True
    else:
      return False
  
  def execute(self, T: float, N: int):
    """ execute the system by time T as N steps
    advance the system time by T as N steps.
    return False if failed at some advancing step; otherwise return True.
    """
    if N <= 0:
      return False
    
    dt = (T - 0.0) / N
    self.sys.set_dt(dt)
    for i in range(N):
      if not self.advance_dt():
        return False
    return True

  
  def execute_until_stationary(self, maxT = 100, dt = 0.01, exam_period = 1.0, threshold = 0.01):
    """ execute the system until the state of the sytem become stationaly.
    execute until observable state vector remain stationary at target x_ref for exam_period.
    return True if attained, False otherwise.
    """
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
