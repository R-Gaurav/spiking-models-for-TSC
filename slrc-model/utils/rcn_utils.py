#
# This
#

import _init_paths

import numpy as np
import nengo
import nengo_loihi

from utils.base_utils import exp_utils as exu
from utils.base_utils import log

class ReservoirComputingNetwork(object):
  """
  This class buils spiking and non-spiking Reservoir Computing Networks (RCN)
  depending on the neuron types.
  """

  def __init__(self, rtc):
    """
    Args:
      order <int>: Number of Legendre Polynomials to represent delay window.
      theta <int>: Length of delay window (in secs) LDN is supposed to represent.
      seed <int>: A random and fixed seed for reproducibility.
      tau <float>: The synaptic filter for LDN connections.
      neuron_type <nengo.NeuronType>: A spiking neuron type or a direct neuron.
      n_spk_neurons <int>: Number of spiking neurons in each Ensemble.
    """
    self._rtc = rtc
    self._order = rtc.ORDER
    self._theta = rtc.THETA
    self._seed = rtc.SEED
    self._tau = rtc.TAU
    self._n_spk_neurons = rtc.N_SPK_NEURONS
    self._ldn_nt = rtc.LDN_NEURON_TYPE
    self._ens_nt = rtc.SIG_ENS_NEURON_TYPE
    log.INFO("LDN Config: ORDER = {0}, THETA = {1}, SEED = {2}, TAU = {3}, "
             "LDN_NEURON_TYPE = {4}, SIG_ENS_NEURON_TYPE: {5}, N_SPK_NEURONS = "
             "{6} ".format(self._order, self._theta, self._seed, self._tau,
             self._ldn_nt, self._ens_nt, self._n_spk_neurons))

    if self._ens_nt != self._ldn_nt:
      log.WARN("LDN Neuron Type {0} is not the same as the Ensemble Neuron Type: "
               "{1}".format(self._ldn_nt, self._ens_nt))

    # Computing the A and B weight matrices for LDN based on `theta` and `order`.
    self._Q = np.arange(self._order, dtype=np.float64)
    self._R = (2*self._Q + 1)[:, None]/self._theta
    j, i = np.meshgrid(self._Q, self._Q)

    self._A = np.where(i < j, -1, (-1.0)**(i-j+1)) * self._R
    self._B = (-1.0)**self._Q[:, None] * self._R

  def build_ldn_net(self, input_dim):
    """
    Builds and returns an LDN network - only the linear memory component.
    Note: Different neuron type produces different pattern of LDN signals.

    Args:
      input_dim <int>: The dimension of the input.
    """
    # To implement the linear system, use `self._n_spk_neurons` spiking neurons
    # to represent each dimension of the system. Each `self._n_spk_neurons`
    # neuron group can be thought of as a noisy approximation of a linear element.
    with nengo.Network(seed=self._seed) as ldn_net:
      # Modifies Nengo Params for optimal performance on Loihi.
      # nengo_loihi.set_defaults()
      ldn_net.input = nengo.Node(size_in=input_dim)
      ldn_net.output = nengo.Node(size_in=self._order)

      # Both input and output size of LDN EnsembleArray is `order`. The transform
      # on the 1D output converts it to an `order` dimensional input.
      log.INFO("From build_ldn_net, ORDER, THETA, LDN_RADIUS: {}, {} {}".format(
               self._order, self._theta, self._rtc.LDN_RADIUS))
      ldn_net.ldn_ena = nengo.networks.EnsembleArray(
          n_neurons = self._n_spk_neurons,
          n_ensembles = self._order,
          neuron_type = self._ldn_nt,
          max_rates = nengo.dists.Uniform(
              self._rtc.LOIHI_MIN_RATE, self._rtc.LOIHI_MAX_RATE),
          intercepts = nengo.dists.Uniform(
              self._rtc.ITRCPT_LOW, self._rtc.ITRCPT_HGH),
          radius=self._rtc.LDN_RADIUS
          )

      # Connnect stimulus input to the LDN's input. Note: B_prime = B*tau
      nengo.Connection(ldn_net.input, ldn_net.ldn_ena.input,
                       transform=self._B*self._tau, synapse=self._tau)
      # Recurrently connect LDN's output to LDN's input.
      # Note: A_prime = A*tau + np.eye(order)
      nengo.Connection(ldn_net.ldn_ena.output, ldn_net.ldn_ena.input,
                       transform=self._A*self._tau + np.eye(self._order),
                       synapse=self._tau
                       )

      # Connect the LDN's output to the networks output Node.
      nengo.Connection(ldn_net.ldn_ena.output, ldn_net.output, synapse=None)

    # Set the seed values of all the objects - Ensembles, Connections, Nodes...
    exu.set_seed_of_all_objects(ldn_net, self._seed)
    return ldn_net

  def build_ldn_to_ens_for_disc_sigs(self, input_dim, sig_len, is_inh):
    """
    Builds a spiking LDN to Ensemble network with or without inhibition on the
    LDN EnsembleArray neurons.

    Args:
      input_dim <int>: The dimension of the Input.
      sig_len <int>: Length of each discontinous signal, which are all same.
      is_inh <bool>: Inhibit the LDN if True else don't.
    """
    if is_inh:
      # This attempts to effectively clean the memory of the LDN between
      # individual input signals. This is supposed to be done for the training
      # and test signals both.
      inh_sig_len = sig_len + self._rtc.INHIBIT_DURTN

      def _inhibit_ldn(t):
        ts = int(t*1000) % inh_sig_len
        if ts == 0 or (ts >= sig_len + 1 and ts <= inh_sig_len-1):
          return self._rtc.INHIBIT_CONST
        else:
          return 0.0

    ldn_net = self.build_ldn_net(input_dim)

    with ldn_net:
      # The LDN EnsembleArray neurons can be inhibited for desired duration
      # between the discontinuous signals.
      if is_inh:
        ldn_net.ldn_inh_node = nengo.Node(_inhibit_ldn)
        # Make Inhbitory Connections.
        for i in range(ldn_net.ldn_ena.n_ensembles):
          nengo.Connection(
              ldn_net.ldn_inh_node, ldn_net.ldn_ena.ensembles[i].neurons, synapse=
              None, transform=[[-1]]*ldn_net.ldn_ena.n_neurons_per_ensemble)

      # Create an Ensemble to represent the LDN outputs.
      log.INFO("From build_ldn_to_ens_for_disc_sigs: MIN_RATE/MAX_RATE/ENS_RAD "
               "{}, {}, {}".format(
               self._rtc.LOIHI_MIN_RATE, self._rtc.LOIHI_MAX_RATE,
               self._rtc.SIG_ENS_RADIUS))
      ldn_net.ldn_sig_ens = nengo.Ensemble(
                                    n_neurons=int(self._n_spk_neurons*self._order/4),
                                    dimensions=self._order,
                                    max_rates=nengo.dists.Uniform(
                                        self._rtc.LOIHI_MIN_RATE,
                                        self._rtc.LOIHI_MAX_RATE),
                                    intercepts=nengo.dists.Uniform(
                                        self._rtc.SIG_ENS_ITRCPT_LOW,
                                        self._rtc.SIG_ENS_ITRCPT_HGH),
                                    neuron_type=self._ens_nt,
                                    radius=self._rtc.SIG_ENS_RADIUS
                                    )
      # Connect the LDN output to the `ldn_sig_ens`.
      nengo.Connection(
          ldn_net.output, ldn_net.ldn_sig_ens, synapse=self._rtc.SYNAPSE)

    exu.set_seed_of_all_objects(ldn_net, self._seed)
    return ldn_net
