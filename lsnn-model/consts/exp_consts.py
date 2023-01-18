import numpy as np
import torch

class EXC(object):
  NP_DTYPE = np.float64
  PT_DTYPE = torch.float64
  WEIGHT_SCALE = 2.0
  HARD_RESET = True # Do hard reset of voltage if True else do soft reset.

  ECG5000 = "ECG5000"
  FORDA = "FORDA"
  FORDB = "FORDB"
  WAFER = "WAFER"
  EQUAKES = "EQUAKES"

  SIGNAL_DURATION = {
    ECG5000: 140,
    FORDA: 500,
    FORDB: 500,
    WAFER: 152,
    EQUAKES: 512
  }
  NUM_CLASSES = {
    ECG5000: 2,
    FORDA: 2,
    FORDB: 2,
    WAFER: 2,
    EQUAKES: 2
  }

  NUM_TEST_SAMPLES = {
    ECG5000: 4500,
    FORDA: 1320,
    FORDB: 810,
    WAFER: 6150, # Originally: 6164, discard last 14 to suit the batch size in net.
    EQUAKES: 138 # Originally: 139, discard last 1 to suit the batch in net.
  }

  # PyTorch Hyper-parameters for LSNN -------- PRE ANALYSIS = 972 Combs.
  #PYTORCH_LR_LST = [0.01, 0.001, 0.0001, 0.05, 0.005, 0.0005]
  #PYTORCH_TAU_CUR_LST = [5e-3, 10e-3, 15e-3]
  #PYTORCH_TAU_VOL_LST = [10e-3, 20e-3, 30e-3]
  #PYTORCH_VOL_THR_LST = [1, 1.5]
  #PYTORCH_NEURON_GAIN_LST = [1, 2, 4]
  #PYTORCH_NEURON_BIAS_LST = [0, 0.5, 1]

  # FOR ECG5000 and WAFER
  # PyTorch Hyper-parameters for LSNN -------- POST ANALYSIS = 324 Combs.
  #PYTORCH_LR_LST = [0.01, 0.05, 0.005]
  #PYTORCH_TAU_CUR_LST = [5e-3, 10e-3, 15e-3]
  #PYTORCH_TAU_VOL_LST = [10e-3, 20e-3, 30e-3]
  #PYTORCH_VOL_THR_LST = [1, 1.5]
  #PYTORCH_NEURON_GAIN_LST = [1, 2, 4]
  #PYTORCH_NEURON_BIAS_LST = [0, 0.5]

  # For EQUAKES, FORDA, and FORDB.
  # PyTorch Hyper-parameters for LSNN -------- POST ANALYSIS = 96 Combs.
  PYTORCH_LR_LST = [0.01, 0.005] #, 0.001]
  PYTORCH_TAU_CUR_LST = [5e-3, 10e-3]
  PYTORCH_TAU_VOL_LST = [20e-3, 30e-3]
  PYTORCH_VOL_THR_LST = [1, 1.5]
  PYTORCH_NEURON_GAIN_LST = [2, 4]
  PYTORCH_NEURON_BIAS_LST = [0, 0.5]

  # PyTorch Non-Spiking Net hyper-parameters.
  # FORDA, FORDB, EQUAKES.
  #PT_NSPK_ORDER_LST = [10, 12, 16, 24]
  #PT_NSPK_THETA_LST = [0.025, 0.05, 0.1, 0.15]
  #PT_NSPK_LR_LST = [0.01, 0.005]
