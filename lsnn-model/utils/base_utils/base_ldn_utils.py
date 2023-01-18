# This file creates a non-spiking LDN.

import _init_paths

import numpy as np
from nengo.utils.filter_design import cont2discrete

from consts.exp_consts import EXC

class BaseLDN(object):
  """
  Create a non-spiking LDN.
  """
  def __init__(self, rtc, c2d=False, dt=0.001):
    """
    Args:
      rtc <class>: Run Time Constants class.
    """
    self._rtc = rtc
    self._c2d = c2d
    self._dt = dt

    self._order = rtc.ORDER
    self._theta = rtc.THETA
    self._tau = rtc.TAU

    self._init_A_p_and_B_p_matrices()

  def _init_A_p_and_B_p_matrices(self):
    """
    Returns the A-prime and B-prime matrices.
    """
    Q = np.arange(self._order, dtype=EXC.NP_DTYPE)
    R = (2*Q + 1)[:, None]/self._theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i<j, -1, (-1.0)**(i-j+1)) * R
    B = (-1.0)**Q[:, None]*R

    if self._c2d:
      C = np.ones((1, self._order))
      D = np.zeros((1,))
      self.A_p, self.B_p, _, _, _ = cont2discrete(
          (A, B, C, D), dt=self._dt, method="zoh")
    else:
      self.A_p = A*self._tau + np.eye(self._order)
      self.B_p = B*self._tau
