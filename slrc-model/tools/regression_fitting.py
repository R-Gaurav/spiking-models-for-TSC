#
# This file does the regression fitting of the SLRC model.
#

import _init_paths

import argparse
import numpy as np
import os
import pickle

from consts.run_time_consts import RTC
from consts.exp_consts import EXC
from consts.dir_consts import DRC
from src.slrc_reg_fitting import SLRCRegFitting
from utils.base_utils import log, exp_utils as exu

def execute_one_combination(rtc, exc, drc, otp_dir):
  rbf = SLRCRegFitting(args.dataset, rtc, exc, drc)
  sig_len = exc.SIGNAL_LENGTH[args.dataset]
  ret_probes = rbf.do_regression_baseline_with_rcn(
      args.is_train, args.is_inh, args.is_inh_trx)

  log.INFO("Regression fitting done. Saving results and calculating accuracy...")
  prfx = "inh_exp_" if args.is_inh == True else ""
  predicted = ret_probes["cls_node"]
  pickle.dump(predicted, open(otp_dir + "/%spredicted.p" % prfx, "wb"))

  if not (rtc.SIMULATOR == exc.SIM_NLOIHI and rtc.BKND == exc.LOIHI):
    if args.is_inh:
      inh_otp = ret_probes["inh_node"]
      pickle.dump(inh_otp,
                  open(otp_dir + "/%sinhibitory_node_output.p" % prfx, "wb"))
    stm_inp = ret_probes["stm_node"]
    ldn_otp = ret_probes["ldn_otp"]
    lte_otp = ret_probes["lte_otp"]
    pickle.dump(lte_otp, open(otp_dir + "/%slte_output.p" % prfx, "wb"))
    pickle.dump(stm_inp, open(otp_dir + "/%sstimulus_input.p" % prfx, "wb"))
    pickle.dump(ldn_otp, open(otp_dir + "/%sldn_output.p" % prfx, "wb"))

  # Get class predictions made in the last time-step.
  step = (sig_len + rtc.INHIBIT_DURTN) if args.is_inh else sig_len
  predicted = predicted[ # Take last time-step output for each signal.
      [idx for idx in range(sig_len-1, predicted.shape[0], step)]]

  # Calculate accuracy.
  y = rbf._train_y if args.is_train else rbf._test_y
  acc = 0
  for idx in range(predicted.shape[0]):
    if np.round(predicted[idx]) == np.argmax(y[idx]):
      acc += 1

  if args.is_train:
    log.INFO(
        "Regression fitting ACC for training set: %s" % (acc/predicted.shape[0]))
  else:
    log.INFO(
        "Regression fitting ACC for test set: %s" % (acc/predicted.shape[0]))

  log.INFO("DONE!!!")
  log.RESET()

def setup_logging(rtc, exc, drc, otp_dir):
  log.configure_log_handler(
      "%s/regression_fitting_ts_%s.log" % (otp_dir, exu.get_timestamp()))

  log.INFO("#"*30 + "  C O N F I G S  "+"#"*30)
  pickle.dump(rtc, open(otp_dir+"run_time_consts.p", "wb"))
  keys = list(vars(rtc).keys())
  for key in keys:
    log.INFO("{0}: {1}".format(key, getattr(rtc, key)))
  log.INFO("#"*70)

  if args.is_train:
    log.INFO("Doing regression fitting with accuracy on training set.")
  else:
    log.INFO("Doing regression fitting with accuracy on test set.")

  if args.is_inh:
    log.INFO("Inhibiting the LDN between input signals while "
             "evaluating the weights obtained through regression.")
  else:
    log.INFO("NOT inhibiting the LDN between input signals while evaluating the "
             "weights obtained through regression.")

  if args.is_inh_trx:
    log.INFO("Using the inhibited training inputs to obtain regression weights")
  else:
    log.INFO("NOT using the inhibited inputs to obtain the regression weights.")

def setup_otp_dir(drc, rtc):
  if args.is_train:
    otp_dir = (
        drc.RESULTS_DIR+"/"+args.dataset+"/training/seed_{0}_n_spk_{1}_neurons/"
        "config_{2}/reg_fitting/".format(rtc.SEED, rtc.N_SPK_NEURONS, rtc.CONFIG))
  else:
    otp_dir = (
        drc.RESULTS_DIR+"/"+args.dataset+"/test/seed_{0}_n_spk_{1}_neurons/"
        "config_{2}/reg_fitting/".format(rtc.SEED, rtc.N_SPK_NEURONS, rtc.CONFIG))

  os.makedirs(otp_dir, exist_ok=True)
  return otp_dir

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, required=True, help="Which dataset?")
  parser.add_argument(
      "--is_train", type=int, required=True, choices=[0, 1], help="Train set?")
  parser.add_argument(
      "--is_inh", type=int, required=True, choices=[0, 1], help="Inhbit LDN?")
  parser.add_argument(
      "--is_inh_trx", type=int, required=True, choices=[0, 1],
      help="Use inhibted inputs to do regression?")
  parser.add_argument(
      "--config_num", type=int, required=False, default=0,
      help="Which config number to start with?")

  args = parser.parse_args()
  args.is_train = True if args.is_train == 1 else False
  args.is_inh = True if args.is_inh == 1 else False
  args.is_inh_trx = True if args.is_inh_trx == 1 else False

  # Run the experiment over all the possible combinations mentioned in
  # exp_consts.py for Regression.
  all_combs = exu.get_combination_for_regression(EXC)
  RTC.CONFIG = 0

  for comb in all_combs:
    RTC.CONFIG +=1
    RTC.LOIHI_MIN_RATE = comb[0]
    RTC.LOIHI_MAX_RATE = comb[1]
    RTC.ORDER = comb[2]
    RTC.THETA = comb[3]
    RTC.LDN_RADIUS = comb[4]
    RTC.SIG_ENS_RADIUS = comb[5]

    if RTC.CONFIG < args.config_num:
      continue

    # Valid only for testing.
    if not args.is_train:
      if RTC.CONFIG not in EXC.TEST_CONFIGS:
        continue

    otp_dir = setup_otp_dir(DRC, RTC)
    DRC.INTERIM_DIR = otp_dir + "/interim_data/"
    setup_logging(RTC, EXC, DRC, otp_dir)
    execute_one_combination(RTC, EXC, DRC, otp_dir)
