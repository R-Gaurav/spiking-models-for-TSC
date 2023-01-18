#
# This file does the PyTorch Batchwise Surrogate Gradient based learning.
#

import _init_paths

import argparse
import numpy as np
import os
import pickle
import sys

from consts.dir_consts import DRC
from consts.run_time_consts import RTC
from consts.exp_consts import EXC
from src.pyt_train_eval_lsnn_and_lsnn_nhdn_model import PTTrainEvalModel
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils

def run_net(rtc, otp_dir):
  pte = PTTrainEvalModel(args.dataset, rtc)
  log.INFO("Starting the PyTorch training...")
  loss_history = pte.train_model(args.epochs, otp_dir)
  #pickle.dump(loss_history, open(otp_dir + "/training_loss_history.p", "wb"))
  log.INFO("Training done, now finally evaluating on the entire test set...")
  acc, all_outputs = pte.evaluate_model(
      num_samples=EXC.NUM_TEST_SAMPLES[args.dataset], ldn_path=otp_dir,
      final_eval=True)
  log.INFO("Test accuracy: {}".format(acc))
  pickle.dump(all_outputs, open(otp_dir + "/test_outputs.p", "wb"))
  log.INFO("Performing cleanup...")
  log.INFO("Deleting train and test X_ldn_sigs.p and Y.p files...")
  os.remove(otp_dir+"/test_X_ldn_sigs.p")
  os.remove(otp_dir+"/test_Y.p")
  os.remove(otp_dir+"/train_X_ldn_sigs.p")
  os.remove(otp_dir+"/train_Y.p")
  log.INFO("Files removed...Exp Done!")

def setup_logging(rtc, otp_dir):
  log.configure_log_handler(
      "%s/pytorch_training_evaluation_%s_%s.log"
      % (otp_dir, rtc.PYTORCH_MODEL_NAME, exu.get_timestamp()))
  keys = list(vars(rtc).keys())
  log.INFO("#"*30 + " C O N F I G " + "#"*30)
  for key in keys:
    log.INFO("{0}: {1}".format(key, getattr(rtc, key)))
  log.INFO("#"*70)

def setup_otp_dir(rtc):
  if args.meta_cfg_dir:
    otp_dir = drc._results_path + "/pytorch_train_eval/{0}/{1}/config_{2}/".format(
        rtc.PYTORCH_MODEL_NAME, args.meta_cfg_dir, rtc.CONFIG)
  else:
    otp_dir = drc._results_path + "/pytorch_train_eval/{0}/config_{1}/".format(
        rtc.PYTORCH_MODEL_NAME, rtc.CONFIG)
  os.makedirs(otp_dir, exist_ok=True)
  return otp_dir

def call_one_combination(rtc):
  otp_dir = setup_otp_dir(rtc)
  setup_logging(rtc, otp_dir)
  run_net(rtc, otp_dir)
  log.RESET()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, required=True, help="Which dataset?")
  parser.add_argument("--epochs", type=int, required=True, help="Training epochs?")
  parser.add_argument("--is_all_combs", type=int, required=False, choices=[0, 1],
                      default=0, help="Search over all hyper-params combinations?")
  parser.add_argument("--config_num", type=int, required=False, default=0,
                      help="Which config number to start with?")
  parser.add_argument("--meta_cfg_dir", type=str, required=False, default=None,
                      help="Meta config name?")

  args = parser.parse_args()
  exu = ExpUtils()
  drc = DRC(args.dataset)

  if not args.is_all_combs:
    call_one_combination(RTC)
    sys.exit("One combination experiment done!")
  else:
    all_combs = exu.get_combinations_for_lsnn_model(EXC)
    RTC.CONFIG = 0
    for comb in all_combs:
      RTC.CONFIG += 1
      if RTC.CONFIG < args.config_num:
        continue

      RTC.PYTORCH_LR = comb[0]
      RTC.PYTORCH_TAU_CUR = comb[1]
      RTC.PYTORCH_TAU_VOL = comb[2]
      RTC.PYTORCH_VOL_THR = comb[3]
      RTC.PYTORCH_NEURON_GAIN = comb[4]
      RTC.PYTORCH_NEURON_BIAS = comb[5]

      call_one_combination(RTC)

    sys.exit("All combination experiments done!")
