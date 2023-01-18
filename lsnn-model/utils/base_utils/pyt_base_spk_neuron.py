import _init_paths
import sys
import torch

from abc import ABC, abstractmethod
from consts.exp_consts import EXC

class BaseSpikingNeuron():
  def __init__(self, v_threshold=1, gain=1, bias=0):
    self._dtype = EXC.PT_DTYPE
    self._v = torch.as_tensor(0, dtype=self._dtype) # Maybe overridden as vector.
    self._v_thr = torch.as_tensor(v_threshold, dtype=self._dtype)
    self._g = torch.as_tensor(gain, dtype=self._dtype)
    self._b = torch.as_tensor(bias, dtype=self._dtype)

  def spike_and_reset(self):
    """
    Returns spike and resets the neuron.
    """
    spike = torch.as_tensor(self._v > self._v_thr, dtype=self._dtype)
    mask = spike > 0

    if self._v.dim(): # Voltage is a vector here.
      if EXC.HARD_RESET:
        self._v[mask] = 0
      else:
        self._v[mask] = self._v[mask] - self._v_thr
    else: # Voltage is just a scalar here.
      if mask: # `mask` is just a scalar here.
        if EXC.HARD_RESET:
          self._v = 0 # Hard Reset.
        else:
          self._v = self._v - self._v_thr # Soft Reset.

    return spike

  def reset_v(self):
    self._v = torch.zeros_like(self._v, dtype=self._dtype)

  @abstractmethod
  def encode(self, x):
    """
    Abstract method to encode the input x.
    """
    raise NotImplementedError
