import numpy as np
import torch

from utils.base_utils.base_ldn_utils import BaseLDN
from consts.exp_consts import EXC

class PyTorchLDN(BaseLDN):
  def __init__(self, rtc, c2d=True):
    """
    Args:
      rtc <class>: Run Time Constants class.
      c2d <bool>: Use contiuous to discrete transformation if True else don't.
    """
    BaseLDN.__init__(self, rtc, c2d=c2d)

    # Convert Numpy matrices to Torch tensors with requires_grad=False (default).
    self.A_p = torch.from_numpy(self.A_p)
    self.B_p = torch.from_numpy(self.B_p)

  def get_pytorch_ldn_transform(self, u_t, m):
    """
    Returns the LDN transform of a scalar.

    Args:
      u_t <float>: Scalar.
      m <Tensor>: State-vector tensor.
    """
    return torch.mm(self.A_p, m) + torch.mm(self.B_p, u_t)

  def get_pytorch_ldn_sigs(self, u):
    """
    Returns the LDN built with matrix multiplications only - no neurons whatsoever.

    Args:
      u <Tensor>: The input tensor.
    """
    assert u.shape[0] == 1 # One row only with multiple columns in the input.
    if isinstance(u, np.ndarray): # Convert it to torch.Tensor if not already.
      u = torch.from_numpy(u)

    # Initialize the state-vector `m`, requires_grad=False by default.
    m = torch.zeros(self._order, 1, dtype=EXC.PT_DTYPE)
    out = torch.zeros(u.shape[1], self._order, dtype=EXC.PT_DTYPE)

    for i, u_t in enumerate(u[0]):
      u_t = u_t.reshape(1, 1)
      out[i, :] = self.get_pytorch_ldn_transform(u_t, m).reshape(-1)
      m[:, 0] = out[i, :] # Update the state vector `m`.

    return out
