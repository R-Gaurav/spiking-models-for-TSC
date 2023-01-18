#
# This file does the L2 regularized Regression fitting over the datasets.
#

import _init_paths

import nengo
import nengo_loihi
import numpy as np
import os
import pickle

from src.extract_signals import ExtractLegendreSignals
from utils.base_utils import log
from utils.base_utils.data_prep_utils import DataPrepUtils
from utils.base_utils import exp_utils as exu
from utils.rcn_utils import ReservoirComputingNetwork

class SLRCRegFitting(object):
  def __init__(self, dataset, rtc, exc, drc):
    """
    Args:
      data <str>: Dataset's name, e.g. "ECG5000" as in exp_consts.
    """
    self._index = 0
    self._rtc = rtc
    self._exc = exc
    self._data = dataset
    self._seed = rtc.SEED
    self._sig_len = exc.SIGNAL_LENGTH[dataset]
    self._inp_dim = exc.DATASET_DIMENSIONS[dataset]
    self._dpu = DataPrepUtils(dataset, rtc, exc, drc)
    self._els = ExtractLegendreSignals(rtc, exc)
    self._rcn = ReservoirComputingNetwork(rtc)
    self._train_x, self._train_y, self._test_x, self._test_y = (
        self._dpu.get_nengo_loihi_compatible_x_y_from_dataset(
        exc.CLASSES[dataset]))
    self._interim_data = drc.INTERIM_DIR

  def _get_all_train_y(self):
    """
    Whether the model is inhbited or not, the ground truth labels remain same.
    """
    log.INFO("Interim Data Dir: %s" % self._interim_data)
    if os.path.exists(self._interim_data + "/all_train_y.p"):
      log.INFO("Found and loading the ground truth labels...")
      all_train_y = pickle.load(
          open(self._interim_data + "/all_train_y.p", "rb"))
    else:
      log.INFO("Ground truth labels not found, preparing them..")
      all_train_y = np.hstack(
          [np.repeat(np.argmax(y), self._sig_len) for y in self._train_y]
          ).reshape(-1, 1)
      os.makedirs(self._interim_data, exist_ok=True)
      pickle.dump(all_train_y, open(self._interim_data + "/all_train_y.p", "wb"))

    return all_train_y

  def _get_inhibited_train_x_lte_sigs(self):
    """
    Returns the training all_x_lte signals for regression weight learning
    between the Ensemble and the output Node. Note that the LDN is inhibited
    between the individual signals while extracting the LTE signals.
    """
    log.INFO("LDN to Ensemble signals, with inhibition between inputs")
    if os.path.exists(self._interim_data + "/all_train_x_inh_lte_sigs.p"):
      log.INFO("Found and loading the extracted inhibited LDN to Ens signals...")
      all_lte_signals = pickle.load(
          open(self._interim_data + "/all_train_x_inh_lte_sigs.p", "rb"))
    else:
      log.INFO("Inhibited LDN to Ens sigs not found, extracting them...")
      num_train_xs = self._train_x.shape[0]
      inh_sig_len = self._sig_len + self._rtc.INHIBIT_DURTN
      all_x_signals = [] # Stores all the individual train_x signals.

      # Arrange all the input signals in one row.
      for x in self._train_x:
        all_x_signals = np.hstack((all_x_signals, x))

      log.INFO("Extracting inh LDN signals for all training data")
      all_x_lte = self._els.run_rcn_and_collect_lte_signals(
          all_x_signals, self._inp_dim, self._sig_len, True)["ens_otp"]
      # Store all the extracted LDN to Ensemble signals, removing the 0 input.
      all_lte_signals = np.vstack(
          [all_x_lte[i*inh_sig_len : (i+1)*inh_sig_len][:self._sig_len]
          for i in range(num_train_xs)])

      pickle.dump(all_lte_signals, open(
                  self._interim_data+"/all_train_x_inh_lte_sigs.p", "wb"))

    return all_lte_signals

  def _get_uninhbited_train_x_lte_sigs(self):
    """
    Returns the training all_x_lte signals for regression weight learning
    between the Enesemble and the output Node. Note that the LDN isn't inhibited
    between the individual signals and a new RC net is created for extracting
    the LTE signals for each input signal.
    """
    log.INFO("LDN to Ensemble signals, without inhibition between inputs")
    if os.path.exists(self._interim_data + "all_train_x_lte_sigs.p"):
      log.INFO("Found and loading the un-inhibited LDN to Ensemble sigs...")
      all_lte_signals = pickle.load(
          open(self._interim_data + "all_train_x_lte_sigs.p", "rb"))
    else:
      all_x_signals = []
      log.INFO("Uninhibited LDN to Ens sigs not found, prepping and saving...")
      num_train_xs = self._train_x.shape[0]
      for index in range(num_train_xs):
        log.INFO("Extracting uninh LDN signals for training index: %s" % index)
        ret_probes = self._els.run_rcn_and_collect_lte_signals(
            self._train_x[index], self._inp_dim, self._sig_len, False)
        all_x_signals.append(ret_probes["ens_otp"])

      # Store and save the signals.
      all_lte_signals = np.vstack([lte_sigs for lte_sigs in all_x_signals])
      os.makedirs(self._interim_data, exist_ok=True)
      pickle.dump(all_lte_signals,
                  open(self._interim_data + "/all_train_x_lte_sigs.p", "wb"))

    return all_lte_signals

  def _get_lte_extracted_signals_x_and_true_y(self, is_inh):
    """
    Returns the signals extracted from the Ensemble post the LDN. Depending on
    the value of `is_inh`, the LDN is inhibited between each inputs if True,
    else a new LDN model is created for each input to extract signals.

    Args:
      is_inh <bool>: Use inhibited inputs to extract signals if True else don't.
    """
    # Training dataset y.
    all_train_y = self._get_all_train_y()

    # Trainig dataset x.
    if is_inh == True:
      all_train_x = self._get_inhibited_train_x_lte_sigs()
    else:
      all_train_x = self._get_uninhbited_train_x_lte_sigs()

    return all_train_x, all_train_y

  def do_regression_baseline_with_rcn(self, is_train, is_inh, is_inh_trx):
    """
    Create a copy of the same spiking-network which does regression learning on
    the outermost/readout layer of the RC network.

    Args:
      is_train <bool>: Use the training data if True, else use the Test data.
      is_inh <bool>: Inhibit the Ensemble neurons if True else don't.
      is_inh_trx <bool>: Use the inhibited training LTE inputs to obtain
                         regression weights if True else use un-inhibited inputs.
    """
    all_x, all_y = self._get_lte_extracted_signals_x_and_true_y(is_inh=is_inh_trx)
    # Accuracy to be obtained on which set?
    input_sigs = self._train_x if is_train else self._test_x
    lte_net = self._rcn.build_ldn_to_ens_for_disc_sigs(
        self._inp_dim, self._sig_len, is_inh)
    ret_probes = {}

    log.INFO("Extraction of LTE signals done, now running the simulation...")
    if is_inh:
      # Append zeros column at the end of the `input_sigs` matrix.
      input_sigs = np.pad(input_sigs, [(0, 0), (0, self._rtc.INHIBIT_DURTN)])
      log.INFO("Inhibition is set, input_sigs shape: {}".format(input_sigs.shape))
      final_sig_len = self._sig_len + self._rtc.INHIBIT_DURTN
    else:
      final_sig_len = self._sig_len

    # Create the Net with Regression Fitting. At the compile time, the LTE
    # signals `all_x` and the true labels `all_y` are used to find the connection
    # weights of the readout layer through Least Squares regression.
    with lte_net:
      # Connect the stimulus to the LTE input.
      lte_net.stim = nengo.Node(output=lambda t: input_sigs[
                                self._index, (int(t*1000)-1) % final_sig_len])
      nengo.Connection(lte_net.stim, lte_net.input, synapse=None)

      # Connect the LDN Ensemble to an output Node and do Least Squares Regression
      # Fitting, default `Solver` is `LstsqL2()`.
      lte_net.cls_node = nengo.Node(output=None, size_in=1)
      nengo.Connection(lte_net.ldn_sig_ens, lte_net.cls_node, eval_points=all_x,
                       function=all_y, synapse=self._rtc.SYNAPSE)

      # Probe the required Nodes.
      if not(self._rtc.SIMULATOR == self._exc.SIM_NLOIHI and
             self._rtc.BKND == self._exc.LOIHI):
        if is_inh:
          probe_inh_node = nengo.Probe(lte_net.ldn_inh_node, synapse=None)
        probe_stm_node = nengo.Probe(lte_net.stim, synapse=None)
        probe_ldn_otp = nengo.Probe(lte_net.output, synapse=self._rtc.SYNAPSE)
        probe_lte_otp = nengo.Probe(lte_net.ldn_sig_ens, synapse=self._rtc.SYNAPSE)

      probe_cls_node = nengo.Probe(lte_net.cls_node, synapse=self._rtc.SYNAPSE)

    # Set the seeds!
    exu.set_seed_of_all_objects(lte_net, self._seed)
    # Once the regression weights for the above connection is obtained after
    # compilation, the network just uses them to make predictions on input
    # samples, be it training or test.
    # Run the network.
    self._index = 0
    num_samples = input_sigs.shape[0]

    log.INFO(
        "Number of samples & each sample's length: {}".format(input_sigs.shape))
    # Run the Net.
    if self._rtc.SIMULATOR == self._exc.SIM_NENGO:
      sim = nengo.Simulator(lte_net, seed=self._seed)
    elif self._rtc.SIMULATOR == self._exc.SIM_NLOIHI:
      nengo_loihi.set_defaults()
      sim = nengo_loihi.Simulator(lte_net, target=self._rtc.BKND, seed=self._seed)

    with sim:
      while self._index < num_samples:
        sim.run(final_sig_len/1000.0)
        log.INFO("Simulation run for index: %s done!" % self._index)
        self._index += 1

    ret_probes["cls_node"] = sim.data[probe_cls_node]
    if not(self._rtc.SIMULATOR == self._exc.SIM_NLOIHI and
           self._rtc.BKND == self._exc.LOIHI):
      if is_inh:
        ret_probes["inh_node"] = sim.data[probe_inh_node]
      ret_probes["stm_node"] = sim.data[probe_stm_node]
      ret_probes["ldn_otp"] = sim.data[probe_ldn_otp]
      ret_probes["lte_otp"] = sim.data[probe_lte_otp]

    return ret_probes
