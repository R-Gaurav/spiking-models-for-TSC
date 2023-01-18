#
# This file is a benchmark file with no hidden layer in the model. The model
# is still meant to be trained with Surrogate GD.
#

import _init_paths

import torch

from consts.exp_consts import EXC
from src.pyt_layers import EncoderLayer, OutputLayer
from src.extract_signals import ExtractSignals

class LSNN_NHDN(torch.nn.Module):
  def __init__(self, dataset, rtc):
    super().__init__()
    self._num_clss = EXC.NUM_CLASSES[dataset]
    self._n_tsteps = EXC.SIGNAL_DURATION[dataset]
    self._bsize = rtc.BATCH_SIZE
    self._dtype = EXC.PT_DTYPE
    self._debug = rtc.DEBUG

    # Note: Number of neurons in the Encoder layer is by default 2*RTC.ORDER
    self._enc_lyr = EncoderLayer(batch_size=self._bsize, debug=self._debug)
    self._otp_lyr = OutputLayer(
        2*rtc.ORDER, self._num_clss, batch_size=self._bsize, debug=self._debug)

  def _forward_through_time(self, x):
    """
    Implements the forward through timesteps.

    Args:
      x <Tensor>: Batch input of shape: (batch_size, signal_duration, LDN_order).
    """
    all_m_pots = torch.zeros( # Stores the output neurons' membrane potentials.
        self._bsize, self._n_tsteps, self._num_clss, dtype=self._dtype)

    for t in range(self._n_tsteps):
      spikes = self._enc_lyr.encode_inp(x[:, t, :])
      m_pots = self._otp_lyr(spikes)
      all_m_pots[:, t, :] = m_pots

    return all_m_pots

  def forward(self, x):
    """
    Implements the forward method on the batch input x. Note that for each batch
    input, reset the necessary states.

    Args:
      x <Tensor>: Batch input for shape: (batch_size, signal_duration, LDN_ORDER).
    """
    self._enc_lyr.reset_v()
    self._otp_lyr.reset_states()
    all_m_pots = self._forward_through_time(x)

    return all_m_pots
