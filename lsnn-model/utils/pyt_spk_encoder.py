import _init_paths

import torch

from utils.base_utils.pyt_base_spk_neuron import BaseSpikingNeuron

class Encoder(BaseSpikingNeuron): # Encodes a scalar.
  def __init__(self, encoder=1, v_threshold=1, gain=1, bias=0):
    super().__init__(v_threshold=v_threshold, gain=gain, bias=bias)
    self._e = torch.as_tensor(encoder, dtype=self._dtype)

  def encode(self, x):
    """
    Encodes the input x. It makes sense to call this function in loop for
    a temporal signal. Note that this neuron encoder is IF spiking neuron.

    Args:
      x <Tensor>: The tensor input x at time t.
    """
    self._J = self._g*self._e*x + self._b # It does element wise multiplication.
    self._v += self._J
    # Reectify the voltage `self._v` to 0 or +ve. Don't let it go negative. For
    # a signal encoded with positive and negative encoder neurons, if the signal
    # is positive for long, the positive encoder neuron's `self._v` is regulated
    # as expected, however, the negative encoder neuron's `self._v` is constantly
    # pushed below zero as its `self._J` is negative (`self._g` is positive,
    # `self._e` is negative, and `x` is positive). So when the signal becomes
    # negative, the negative encoder neuron's `self._v` is so low (i.e. below 0)
    # such that the negative encoder neuron doesn't spike for long (even if the
    # signal is negative), as it takes time for it's `self._v` to become positive
    # and eventually cross threshold. So rectifying the `self._v` is crucial.
    # Same stand true for the positive encoder neuron  when the signal is negative
    # for long and then becomes positive.

    # Rectify the Voltage!
    mask = self._v < 0
    if self._v.dim(): # Voltage is a vector here.
      self._v[mask] = 0
    else: # Voltage is scalar here.
      if mask: # `mask` is scalar here.
        self._v = 0

    spike = self.spike_and_reset()
    return spike

class TensorEncoder(Encoder): # Encodes a vector.
  def __init__(self, tensor_size, v_threshold=2, gain=1, bias=0):
    """
    Args:
      tensor_size <int>: Size of the input tensor to be encoded.
    """
    super().__init__(v_threshold=v_threshold, gain=gain, bias=bias)
    self._v = torch.zeros(tensor_size, dtype=self._dtype)
    self._e = torch.zeros(tensor_size, dtype=self._dtype)

    if len(tensor_size) == 2: # (Batch size, Num neurons)
      for i in range(tensor_size[1]):
        self._e[:, i] = -1.0 if i%2 else 1.0
    else: # (Num neurons, )
      for i in range(tensor_size[0]):
        self._e[i] = -1.0 if i%2 else 1.0

    assert self._e.sum() == 0
