#
# This file contains modular helper/util functions for the experiment.
#

import _init_paths

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle

def get_timestamp():
  now = datetime.datetime.now()
  now = "%s" % now
  return "T".join(now.split())

def append_zeros(signal, num_zeros):
  return np.pad(signal, (0, num_zeros))

def insert_zeros(signals, num_zeros, sig_len):
  ret = []
  len_all_sigs = len(signals)
  for i in range(0, len_all_sigs, sig_len):
    sig_with_zeros = append_zeros(signals[i:i+sig_len], num_zeros)
    ret = np.hstack([ret, sig_with_zeros])

  return ret[:-num_zeros]

def set_seed_of_all_objects(net, seed):
  for i, obj in enumerate(net.all_objects):
    if not obj.seed:
      obj.seed = seed + i

def plot(vals, fs=16, fn=None, d_label=None, x_label=None, y_label=None):
  plt.figure(figsize=(6, 4))
  plt.xlabel(x_label, fontsize=fs)
  plt.ylabel(y_label, fontsize=fs)
  plt.xticks(fontsize=fs)
  plt.yticks(fontsize=fs)
  plt.plot(vals, label=d_label)
  if d_label:
    plt.legend(framealpha=0.325, fontsize=fs)

  if fn:
    plt.savefig(fn, dpi=450, bbox_inches="tight")

def get_combination_for_regression(exc):
  all_lists = [exc.LST_MIN_RATE, exc.LST_MAX_RATE, exc.LST_ORDER, exc.LST_THETA,
               exc.LST_LDN_RADIUS, exc.LST_SIG_ENS_RADIUS]
  all_combs = itertools.product(*all_lists)
  return all_combs

def check_and_load_file(file_path):
  if os.path.exists(file_path):
    return pickle.load(open(file_path, "rb"))
  else:
    return None
