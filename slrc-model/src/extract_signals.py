#
# This file extracts and saves Legendre signals from different datasets.
#

import _init_paths

import nengo
import nengo_loihi

from utils.base_utils import exp_utils as exu
from utils.base_utils import log
from utils.rcn_utils import ReservoirComputingNetwork

class ExtractLegendreSignals(object):
  def __init__(self, rtc, exc):
    self._rtc = rtc
    self._exc = exc
    self._seed = rtc.SEED
    self._rcn = ReservoirComputingNetwork(rtc)

  def run_ldn_net_and_collect_signals(self, signal, inp_dim):
    """
    Extracts the Legendre signals directly from the LDN EnsembleArray. Note that
    you should run this function on Nengo simulator, else, on NengoLoihi, the
    output Node from the LDN net becomes a Passthrough node and no output signals
    are recorded. However, if an Ensemble is connected to the output Node from
    the LDN, then the output signals can be collected from the Ensemble
    on NengoLoihi.

    Args:
      signal <np.ndarray>: Input signal whose spikes have to be extracted.
      inp_d <int>: Input dimension of the signal.
    """
    ldn_net = self._rcn.build_ldn_net(inp_dim)
    sig_len = len(signal)

    with ldn_net:
      # Note that `t` starts from 0.001, so a manipulation of `int(t*1000)-1`
      # is required.
      stim = nengo.Node(output=lambda t: signal[int(t*1000)-1])
      nengo.Connection(stim, ldn_net.input, synapse=None)
      probe_stim = nengo.Probe(stim, synapse=None)
      probe_ldn_otp = nengo.Probe(ldn_net.output, synapse=self._rtc.SYNAPSE)

    exu.set_seed_of_all_objects(ldn_net, self._seed)

    if self._rtc.SIMULATOR == self._exc.SIM_NENGO:
      sim = nengo.Simulator(ldn_net, seed=self._seed)
    elif self._rtc.SIMULATOR == self._exc.SIM_NLOIHI:
      nengo_loihi.set_defaults()
      sim = nengo_loihi.Simulator(ldn_net, target=self._rtc.BKND, seed=self._seed)

    with sim:
      sim.run(sig_len/1000.0)

    return sim.data[probe_stim], sim.data[probe_ldn_otp]

  def run_rcn_and_collect_lte_signals(self, signals, input_dim, sig_len, is_inh):
    """
    Runs the RCN to compute the Legendre signals represented by the Ensemble
    connected to the LDN for the Regression baseline.

    Args:
      signals <np.ndarray>: Multiple input signals appended next to each other.
      input_dim <int>: The dimension of the Input.
      sig_len <int>: Length of each discontinous signal, which are all same.
      is_inh <bool>: Inhibit the LDN if True else don't.
    """
    # If the LDN is inhibited, insert zeros between the each indivdiual signal
    # (of length: sig_len) with the number of zeros between each signal being the
    # same as the duration of the inhibition.
    if is_inh:
      signals = exu.insert_zeros(signals, self._rtc.INHIBIT_DURTN, sig_len)

    all_sigs_len = len(signals)
    ret_probes = {}
    lte_net = self._rcn.build_ldn_to_ens_for_disc_sigs(input_dim, sig_len, is_inh)

    with lte_net:
      stim_node = nengo.Node(output=lambda t: signals[int(t*1000)-1])
      nengo.Connection(stim_node, lte_net.input, synapse=None)

      if not(self._rtc.SIMULATOR == self._exc.SIM_NLOIHI and
             self._rtc.BKND == self._exc.LOIHI):
        probe_stim = nengo.Probe(stim_node, synapse=None)
        if is_inh:
          probe_ldn_inh = nengo.Probe(lte_net.ldn_inh_node, synapse=None)

      probe_ens_otp = nengo.Probe(lte_net.ldn_sig_ens, synapse=self._rtc.SYNAPSE)

    # Following way of setting the seed ensures that the introduction of new
    # components does not lead to a change in the seeds of the orginal components
    # in the `lte_net` obtained from `self._rcn.build_ldn_to_ens_for_disc_sigs()`.
    exu.set_seed_of_all_objects(lte_net, self._seed)

    if self._rtc.SIMULATOR == self._exc.SIM_NENGO:
      sim = nengo.Simulator(lte_net, seed=self._seed)
    elif self._rtc.SIMULATOR == self._exc.SIM_NLOIHI:
      nengo_loihi.set_defaults()
      sim = nengo_loihi.Simulator(lte_net, target=self._rtc.BKND, seed=self._seed)

    with sim:
      sim.run(all_sigs_len/1000.0)

    ret_probes["ens_otp"] = sim.data[probe_ens_otp]
    if not(self._rtc.SIMULATOR == self._exc.SIM_NLOIHI and
           self._rtc.BKND == self._exc.LOIHI):
      if is_inh:
        ret_probes["ldn_inh"] = sim.data[probe_ldn_inh]
      ret_probes["stim"] = sim.data[probe_stim]
      ret_probes["ens_otp"] = sim.data[probe_ens_otp]

    return ret_probes
