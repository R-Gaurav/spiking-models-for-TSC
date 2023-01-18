#
# This file implements the non-spiking variant of LSNN.
#

import _init_paths

import torch
import torch.nn as nn
import torch.nn.functional as F

from consts.exp_consts import EXC

class BWNspkNet(torch.nn.Module):
  def __init__(self, dataset, rtc):
    """
    Initializes the Batch-Wise Non-Spiking Net.
    """
    super().__init__()
    self._num_clss = EXC.NUM_CLASSES[dataset]
    self._n_tsteps = EXC.SIGNAL_DURATION[dataset]
    self._bsize = rtc.BATCH_SIZE
    self._dtype = EXC.PT_DTYPE
    self._debug = rtc.DEBUG

    self._fc1 = torch.nn.Linear(rtc.ORDER, rtc.N_HDN_NEURONS, dtype=self._dtype)
    self._fc2 = torch.nn.Linear(rtc.N_HDN_NEURONS, self._num_clss, dtype=self._dtype)

  def _forward_through_time(self, x):
    """
    Implements forward through timesteps.

    Args:
      x <Tensor>: Batch input of shape: (batch_size, signal_duration, LDN_ORDER)
    """
    all_ts_otps = torch.zeros(
        self._bsize, self._n_tsteps, self._num_clss, dtype=self._dtype)

    for t in range(self._n_tsteps):
      o1 = F.relu(self._fc1(x[:, t, :]))
      o2 = self._fc2(o1)
      all_ts_otps[:, t, :] = o2

    return all_ts_otps

  def forward(self, x):
    """
    Implements the forward method on the batch input x.

    Args:
      x <Tensor>: Batch input for shape
    """
    all_ts_otps = self._forward_through_time(x)
    return all_ts_otps
