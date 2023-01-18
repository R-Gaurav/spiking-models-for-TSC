import nengo
import numpy as np
import torch

from utils.base_utils import log
from utils.pyt_ldn import PyTorchLDN

from consts.exp_consts import EXC

class ExtractSignals(object):
  def __init__(self, rtc):
    """
    Args:
      rtc <class>: Run Time Constants class.
    """
    self._rtc = rtc
    self._pln = PyTorchLDN(rtc)

  def run_pytorch_ldn_and_return_ldn_signals(self, signals):
    """
    Runs the PyTorchLDN and returns the LDN signals for batchwise training.

    Args:
      signals <numpy.ndarray>: A 2D matrix of input signals.
    """
    n_samples, n_steps = signals.shape
    pt_ldn_sigs = torch.zeros(
        n_samples, n_steps, self._rtc.ORDER, dtype=EXC.PT_DTYPE)
    signals = torch.from_numpy(signals)
    for i, sig in enumerate(signals):
      sig = sig.reshape(1, -1)
      pt_ldn_sigs[i, :, :] = self._pln.get_pytorch_ldn_sigs(sig)

    log.INFO("PyTorch LDN signals created of shape: {}, now returning...".format(
        pt_ldn_sigs.shape))
    return pt_ldn_sigs
