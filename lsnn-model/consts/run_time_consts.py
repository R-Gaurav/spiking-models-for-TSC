import nengo

class RTC(object):
  """
  Run Time Constants.
  """
  SEED = 9
  CONFIG = 1

  BATCH_SIZE = 23 # 50 for LSNN - ECG5000 and WAFER, 23 for EQUAKES.
  #BATCH_SIZE = 18 # 45 for LSNN 40 for FORDA, 18 for FORDB.

  # Number of test samples to evaluate during training epochs.
  #TEST_EVAL_SIZE = 4500 # For ECG5000
  #TEST_EVAL_SIZE = 1320 # For FORDA
  #TEST_EVAL_SIZE = 810 # For FORDB
  #TEST_EVAL_SIZE = 6150 # For WAFER.
  TEST_EVAL_SIZE = 138 # For EQUAKES

  DEBUG = False
  NORMALIZE_DATASET = False

  # LDN Constants.
  ORDER = 12
  THETA = 0.10 # ECG5000 -> 0.10, 0.12, 0.14
  #THETA = 0.04
  TAU = 0.1

  # PyTorch Model Constants. LSNN | LSNN_NHDN
  PYTORCH_MODEL_NAME = "LSNN"
  #PYTORCH_MODEL_NAME = "LSNN_NHDN"
  N_HDN_NEURONS = 3*ORDER

  PYTORCH_LR = 5e-2
  PYTORCH_TAU_CUR = 5e-3
  PYTORCH_TAU_VOL = 10e-3
  PYTORCH_VOL_THR = 1.0
  PYTORCH_NEURON_GAIN = 1.0 # For calculating J.
  PYTORCH_NEURON_BIAS = 0.0 # For calculating J.
