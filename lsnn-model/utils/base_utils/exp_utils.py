import _init_paths

import datetime
import itertools
import numpy as np
import torch

from utils.base_utils import log
from consts.run_time_consts import RTC

class ExpUtils(object):

  def __init__(self):
    pass

  def get_timestamp(self):
    now = datetime.datetime.now()
    now = "%s" % now
    return "T".join(now.split())

  def get_combinations_for_lsnn_model(self, exc):
    all_lists = [exc.PYTORCH_LR_LST, exc.PYTORCH_TAU_CUR_LST,
                 exc.PYTORCH_TAU_VOL_LST, exc.PYTORCH_VOL_THR_LST,
                 exc.PYTORCH_NEURON_GAIN_LST, exc.PYTORCH_NEURON_BIAS_LST]
    all_combs = itertools.product(*all_lists)
    return all_combs

  def get_combinations_for_lsnn_nspk(self, exc):
    all_lists = [exc.PT_NSPK_ORDER_LST, exc.PT_NSPK_THETA_LST, exc.PT_NSPK_LR_LST]
    all_combs = itertools.product(*all_lists)
    return all_combs

  def log_debug_requires_grad(self, cls, params):
    for param in params:
      log.DEBUG(
          f"Class {cls}, {param[0]}: requires_grad = {param[1].requires_grad}")

  def log_debug_values(self, cls, params):
    for param in params:
      log.DEBUG(f"(Class {cls}, {param[0]}: values = {param[1].detach()}")

  def get_pytorch_seed(self):
    gen = torch.Generator()
    return gen.manual_seed(RTC.SEED)
