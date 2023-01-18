import _init_paths

import torch
import numpy as np

from consts.exp_consts import EXC
from consts.run_time_consts import RTC
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils
from utils.pyt_spk_encoder import TensorEncoder
from utils.pyt_surr_grad_spike import spike_func

class EncoderLayer(TensorEncoder):
  def __init__(self, num_neurons=2*RTC.ORDER, v_threshold=RTC.PYTORCH_VOL_THR,
               gain=RTC.PYTORCH_NEURON_GAIN, bias=RTC.PYTORCH_NEURON_BIAS,
               batch_size=None, debug=False):
    """
    Args:
      num_neurons <int>: Number of neurons in the Encoder Layer. It has to be
                         2*RTC.ORDER as there is +ve and -ve encoder for each dim.
      v_threshold <float>: The voltage threshold.
      gain <float>: Neuron's gain `alpha`.
      bias <float>: Neuron's bias `J_bias`.
      batch_size <int>: Batch size of the input.
    """
    if batch_size is not None:
      tensor_size = (batch_size, num_neurons)
    else:
      tensor_size = (num_neurons, )

    super().__init__(
        tensor_size=tensor_size, v_threshold=v_threshold, gain=gain, bias=bias)
    self._debug = debug
    self._exu = ExpUtils()
    self._t_mat  = torch.zeros(RTC.ORDER, num_neurons, dtype=self._dtype)
    for i in range(0, RTC.ORDER):
      self._t_mat[i, 2*i] = 1 # To make a copy of the scalar.
      self._t_mat[i, 2*i+1] = 1 # To make a copy of the scalar.

  def encode_inp(self, x):
    """
    Encodes each 1-D vector x. Note that each element of the vector x is part of
    a time series, and can be positive or negative. Therefore, each element is
    connected to two neurons - one with positive encoder and one with negative
    encoder.

    Args:
      x <torch.Tensor>: Real valued input of shape (batch_size, RTC.ORDER)
    """
    x = torch.mm(x, self._t_mat) # Transform x.
    if self._debug:
      self._exu.log_debug_requires_grad(
          "EncoderLayer",
          [("v", self._v), ("e", self._e), ("x", x), ("t_mat", self._t_mat)])

    return self.encode(x)

class HiddenLayer(torch.nn.Module):
  """
  Hidden layer with spiking neurons implementation.
  """
  def __init__(self, n_previous, n_hidden, batch_size, dt=1e-3, debug=False):
    """
    Initializes the Hidden Layer.

    Args:
      n_previous <int>: Number of neurons in the previous layer.
      n_hidden <int>: Number of neurons in this hidden layer.
    """
    super().__init__()
    self._debug = debug
    self._dtype = EXC.PT_DTYPE
    self._n_prev = n_previous
    self._n_hidn = n_hidden
    self._exu = ExpUtils()
    self._v = torch.zeros(batch_size, n_hidden, dtype=self._dtype) # Mem potential.
    self._c = torch.zeros(batch_size, n_hidden, dtype=self._dtype) # Mem current.
    self._c_dcy = torch.as_tensor(np.exp(-dt/RTC.PYTORCH_TAU_CUR))
    self._v_thr = torch.as_tensor(RTC.PYTORCH_VOL_THR)
    self._fc = torch.nn.Linear(n_previous, n_hidden, bias=False, dtype=self._dtype)
    # Transpose the dims because internally the Linear layer transposes the
    # weights when multiplying to the input.
    self._fc.weight.data = torch.empty(
        n_hidden, n_previous, dtype=self._dtype).normal_(
        mean=0.0, std=EXC.WEIGHT_SCALE/np.sqrt(n_previous),
        generator=self._exu.get_pytorch_seed()) # Transpose the dims.
    log.INFO("Initial Hidden Wts: {}".format(self._fc.weight.data))

  def reset_states(self):
    """
    Resets the membrane potential and current states.
    """
    self._v = torch.zeros_like(self._v, dtype=self._dtype)
    self._c = torch.zeros_like(self._c, dtype=self._dtype)

  def _spike_and_reset(self):
    """
    Spikes and resets the neuron membrane potential.
    """
    out_s = spike_func(self._v - self._v_thr) # Get the output spikes.
    # Detach the output spikes to not backpropagate through potential reset.
    mask = out_s.detach() > 0
    if EXC.HARD_RESET:
      self._v[mask] = 0
    else:
      self._v[mask] = self._v[mask] - self._v_thr
    return out_s

  def forward(self, x):
    """
    Does the forward computation for the current time t.

    Args:
      x <torch.Tensor>: Spikes from the previous layer of shape:
                        (batch_size, number of neurons in encoding layer).
    """
    x = self._fc(x) # Get the Weights x Spikes output.
    #x = torch.einsum("ab,bc->ac", (x.view(1, -1), self._w1))

    self._c = self._c_dcy*self._c + x # Get new current accounting for the decay.
    self._v = self._v + self._c # Get new voltage, IF neuron => no voltage decay.
    out_s = self._spike_and_reset()

    if self._debug:
      self._exu.log_debug_requires_grad(
          "HiddenLayer", [("v", self._v), ("c", self._c), ("x", x), ("s", out_s)])
      self._exu.log_debug_values(
          "HiddenLayer", [("v", self._v), ("c", self._v), ("s", out_s)])

    return out_s

class OutputLayer(torch.nn.Module):
  """
  Output layer implementation - note that these neurons do not output spikes -
  this is simply a readout layer.
  """
  def __init__(self, n_previous, n_output, batch_size, dt=1e-3, debug=False):
    """
    Initializes the Output layer.

    Args:
      n_previous <int>: Number of neurons in the previous layer.
      n_output <int>: Number of neurons in this output layer.
    """
    super().__init__()
    self._debug = debug
    self._dtype = EXC.PT_DTYPE
    self._n_prev = n_previous
    self._n_otpt = n_output
    self._exu = ExpUtils()
    self._v = torch.zeros(batch_size, n_output, dtype=self._dtype) # Mem potential.
    self._c = torch.zeros(batch_size, n_output, dtype=self._dtype) # Mem current.
    self._c_dcy = torch.as_tensor(np.exp(-dt/RTC.PYTORCH_TAU_CUR))
    self._v_thr = torch.as_tensor(RTC.PYTORCH_VOL_THR)
    self._v_dcy = torch.as_tensor(np.exp(-dt/RTC.PYTORCH_TAU_VOL))
    self._fc = torch.nn.Linear(n_previous, n_output, dtype=self._dtype)
    # Transpose the dims because internally the Linear layer transposes the
    # weights when multiplying to the input.
    self._fc.weight.data = torch.empty(
        n_output, n_previous, dtype=self._dtype).normal_(
        mean=0.0, std=EXC.WEIGHT_SCALE/np.sqrt(n_output),
        generator=self._exu.get_pytorch_seed()) # Transpose the dims.
    log.INFO("Initial Output Wts: {}".format(self._fc.weight.data))

  def reset_states(self):
    """
    Resets the membrane potential and the current states.
    """
    self._v = torch.zeros_like(self._v, dtype=self._dtype)
    self._c = torch.zeros_like(self._c, dtype=self._dtype)

  def forward(self, x):
    """
    Does the forward computation. Note that the neuron doesn't output spike and
    its membrane potential rise with every input spike as well as decays.

    Args:
      x <torch.Tensor>: Spikes from the previous layer of shape:
                        (batch_size, number of neurons in hidden layer).
    """
    x = self._fc(x)
    self._c = self._c_dcy*self._c + x
    self._v = self._v_dcy*self._v + self._c
    #self._v = self._v + self._c # A softmax is taken at the end anyways!

    if self._debug:
      self._exu.log_debug_requires_grad(
          "OutputLayer", [("v", self._v), ("c", self._c)])
      self._exu.log_debug_values("OutputLayer", [("v", self._v), ("c", self._c)])

    return self._v
