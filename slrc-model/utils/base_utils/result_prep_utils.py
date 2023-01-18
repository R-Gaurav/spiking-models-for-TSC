#
# This file contains the code to prepare the results.
#

import _init_paths
import numpy as np
import os
import pickle
import pandas as pd

from utils.base_utils.data_prep_utils import DataPrepUtils
from utils.base_utils import exp_utils as exu

class ResultPrepUtils(object):
  def __init__(self, dataset, rtc, exc, drc):
    """
    Args:
      dataset <str>: Dataset's name, e.g. "ECG5000" as in exp_consts.
    """
    self._rtc = rtc
    self._exc= exc
    self._data = dataset
    self._sig_len = exc.SIGNAL_LENGTH[dataset]
    self._dpu = DataPrepUtils(dataset, rtc, exc, drc)
    _, self._train_y, _, self._test_y = (
        self._dpu.get_nengo_loihi_compatible_x_y_from_dataset(
        exc.CLASSES[dataset]))

    if dataset == exc.ECG5000:
      self._results_path = drc.RESULTS_DIR+"/"+exc.ECG5000

  def load_regression_otp_pickle_files(self, config, do_inhibit, res_dir):
    """
    Args:
      config <int>: Results from which configuration set up to be loaded?
      do_inhibit <bool>: Load pickle files of the inhibited LDN RC experiment.
      res_dir <str>: Results directory.
    """
    ret = {}
    res_dir = res_dir + "/config_%s/reg_fitting/" % config
    irm_dir = res_dir + "/interim_data/"

    if do_inhibit:
      print("Loading the LDN inhibition set up results of config: %s" % config)
      apdx = "inh_exp_"
    else:
      print("Loading the non-inhibition set up results of config: %s" % config)
      apdx = ""

    # Load inhibitory signal output.
    if do_inhibit:
      ret["inh_otp"] = exu.check_and_load_file(
          res_dir+"%sinhibitory_node_output.p" % apdx)
    # Load the predicted class output.
    ret["pred_cls_otp"] = exu.check_and_load_file(res_dir+"%spredicted.p" % apdx)
    # Load the LDN output.
    ret["ldn_otp"] = exu.check_and_load_file(res_dir+"%sldn_output.p" % apdx)
    # Load the Signal Ensemble output (representing the LDN Signals).
    ret["sig_ens_otp"] = exu.check_and_load_file(res_dir+"%slte_output.p" % apdx)
    # Load the LDN input signals of all the train_x data.
    ret["all_train_x_sigs"] = exu.check_and_load_file(
        irm_dir+"all_train_x_inh_lte_sigs.p")
    # Load the training y ground truths.
    ret["all_train_y"] = exu.check_and_load_file(irm_dir+"all_train_y.p")

    return ret

  def get_mean_scores_of_last_n_tsteps(self, y_pred, n, is_inhibit=False):
    """
    Returns the mean of last n-tsteps scores of the predicted scores for each
    signal.

    Args:
      y_pred <np.array>: Predicted labels.
      n <int>: The number of last time-steps to consider to take mean of scores.
      is_inhibit <bool>: If True, remove the predicted output during inhibtion.
    """
    if len(y_pred.shape) != 1:
      y_pred = y_pred.reshape(-1)

    assert len(y_pred.shape) == 1

    if is_inhibit:
      steps = self._sig_len + self._rtc.INHIBIT_DURTN
    else:
      steps = self._sig_len

    # Calculate last n-tsteps accuracy.
    mean_scores = []
    for i in range(self._sig_len-1, y_pred.shape[0], steps):
      mean_scores.append(np.mean(y_pred[i+1-n:i+1]))

    return np.array(mean_scores)

  def get_accuracy(self, y_pred, is_train=False):
    """
    Returns the training or test accuracy.

    Args:
      y_pred <np.ndarray>: The predicted scores.
      is_train <bool>: Calculate training accuracy if True else test accuracy.
    """
    y_true = self._train_y if is_train else self._test_y
    assert y_pred.shape[0] == y_true.shape[0]

    acc = 0
    for y_t, y_p in zip(y_true, y_pred):
      if np.argmax(y_t) == np.round(y_p):
        acc += 1

    return acc*1.0/y_true.shape[0]

  def get_loihi_results_accuracy(self, pred_vals, true_vals):
    """
    Returns the training, evaluation, and test accuracies. This is for outputs
    shaped as [Number of samples x Signal Length x Number of classes].

    NOTE: Don't use the self._train_y, as they can be shuffled before training.
    self._test_y can be used as the WHOLE test data isn't shuffled.

    Args:
      pred_vals <np.ndarray>: Predicted class scores.
      true_vals <np.ndarray>: True class scores.
      is_test <bool>: Is the `pred_vals` of test samples of train samples.
    """
    acc = 0
    pred_vals = pred_vals[:, :self._sig_len, :]
    if len(true_vals.shape) == 3:
      true_vals = true_vals[:, :self._sig_len, :]
    else:
      true_vals = np.expand_dims(true_vals, axis=1)

    #assert pred_vals.shape == true_vals.shape
    for y_p, y_t in zip(pred_vals, true_vals):
      if np.argmax(y_p[-1, :]) == np.argmax(y_t[-1, :]):
        acc +=1

    return acc*1.0/true_vals.shape[0]

  def get_dataframe_of_loihi_regression_results(self, results):
    """
    Creates a dataframe of RTC config and accuracy results.

    Args:
      results <{}>: A dictionary of config and corresponding accuracy.
    """
    col_names = ["CONFIG", "LOIHI_MIN_RATE", "LOIHI_MAX_RATE", "ORDER", "THETA",
                 "LDN_RADIUS", "SIG_ENS_RADIUS", "ACCURACY"]
    df = []

    for key, value in results.items():
      e = []
      e.append(key[0])
      e.extend([k for k in key[1]])
      e.append(value)
      df.append(e)

    df = pd.DataFrame(df, columns=col_names)
    return df

  def get_accuracies_from_loihi_regression_log_files(self, res_dir, start=0):
    """
    Obtains the already calculated accuracies from the log files.

    Args:
      res_dir <str>: The result dir where all the config files are stored.
      start <int>: Start of the config if any experiments already done.
    """
    def _check_if_accs_arent_equal(config, log_file):
      acc_lst = []
      # Check if all the log files have the same obtained accuracy or not.
      for lf in log_file:
        with open(res_dir+"/config_%s/" % config+"reg_fitting/"+lf) as f:
          lines = f.readlines()
        acc_lst.append(float(lines[-2].split()[-1]))

      acc_lst = np.array(acc_lst)
      if not np.all(acc_lst == acc_lst[0]):
        print("Found config {0} where number of log files = {1} and accuracies "
              "={2}".format(config, acc_lst.shape, acc_lst))
    ############################################################################

    ret = {}
    all_combs = exu.get_combination_for_regression(self._exc)
    for i, comb in enumerate(all_combs):
      config = i+1+start
      #print(config)
      try:
        files = os.listdir(res_dir + "/config_%s/" % config + "reg_fitting/")
      except FileNotFoundError:
        print("Config %s not found." % config)
        continue
      log_file = [_ for _ in files if _.endswith(".log")]

      if len(log_file) !=1:
        #print("Found config {0} where number of log files = {1}".format(
        #      config, log_file))
        _check_if_accs_arent_equal(config, log_file)

      with open(res_dir+"/config_%s/" % config+"reg_fitting/"+log_file[0]) as f:
        lines = f.readlines()

      acc = float(lines[-2].split()[-1])
      ret[(config, comb)] = acc

    return ret
