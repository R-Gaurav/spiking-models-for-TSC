#
# This file implements the LSNN model.
#

import _init_paths

import torch

from consts.exp_consts import EXC
from src.pyt_layers import EncoderLayer, HiddenLayer, OutputLayer

class LSNN(torch.nn.Module):
  def __init__(self, dataset, rtc):
    super().__init__()
    self._num_clss = EXC.NUM_CLASSES[dataset]
    self._n_tsteps = EXC.SIGNAL_DURATION[dataset]
    self._bsize = rtc.BATCH_SIZE
    self._dtype = EXC.PT_DTYPE
    self._debug = rtc.DEBUG

    # Note: Number of neurons in the Encoder layer is by default 2*RTC.ORDER.
    self._enc_lyr = EncoderLayer(batch_size=self._bsize, debug=self._debug)
    self._hdn_lyr = HiddenLayer(
        2*rtc.ORDER, rtc.N_HDN_NEURONS, batch_size=self._bsize, debug=self._debug)
    self._otp_lyr = OutputLayer(
        rtc.N_HDN_NEURONS, self._num_clss, batch_size=self._bsize, debug=self._debug)

  def _forward_through_time(self, x):
    """
    Implements the forward through timesteps.

    Args:
      x <Tensor>: Batch input of shape: (batch_size, signal_duration, LDN_order).
    """
    #all_m_pots = [] # Stores all the output membrane potentials for each timestep.
    # Here all_m_pots has requires_grad = False by default. But since it is
    # populated by membrane potential values which are function of trainable
    # weights, it's requires_grad attribute's value changes to True at the end
    # of the forward pass.
    all_m_pots = torch.zeros(
        self._bsize, self._n_tsteps, self._num_clss, dtype=self._dtype)

    for t in range(self._n_tsteps):
      spikes = self._enc_lyr.encode_inp(x[:, t, :])
      spikes = self._hdn_lyr(spikes)
      m_pots = self._otp_lyr(spikes) # Get the membrane potentials.
      all_m_pots[:, t, :] = m_pots

    return all_m_pots

  def forward(self, x):
    """
    Implements the forward method on the batch input x. Note that for each batch
    input, reset the necessary states.

    Args:
      x <Tensor>: Batch input of shape: (batch_size, signal_duration, LDN_order).
    """
    # Reset the voltage of the Encoding layer, note that current is stateless.
    self._enc_lyr.reset_v()
    # Reset the states (potential and current) of the hidden layer neurons.
    self._hdn_lyr.reset_states()
    # Reset the states (potential and current) of the output layer.
    self._otp_lyr.reset_states()

    all_m_pots = self._forward_through_time(x) # Output membrane potentials.

    return all_m_pots
