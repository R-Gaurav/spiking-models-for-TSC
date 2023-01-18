class EXC(object):
  ECG5000 = "ECG5000"
  SIM_NENGO = "sim_nengo"
  SIM_NLOIHI = "sim_nloihi"
  LOIHI = "loihi"

  # Each dataset's one sample's signal length.
  SIGNAL_LENGTH = {
    ECG5000: 140,
  }

  NUM_TRAIN_SAMPLES = {
    ECG5000: 500,
  }

  NUM_TEST_SAMPLES = {
    ECG5000: 4500,
  }

  CLASSES = {
    ECG5000: 2,
  }

  DATASET_DIMENSIONS = {
    ECG5000: 1
  }

  # ORDER = 10 on POHOIKI.
  # ORDER = 8 on NAHUKU32
  # ORDER = 6 on LOIHI_2H.

  LST_ORDER = [6] # Don't forget to change the RESULT_DIR accordingly.
  LST_THETA = [0.10, 0.12, 0.14]

  LST_MIN_RATE = [40, 60, 80]
  LST_MAX_RATE = [100, 120, 140]
  LST_LDN_RADIUS = [0.5, 1.0, 1.5]
  LST_SIG_ENS_RADIUS = [0.5, 1.0, 1.5]
  LST_ERR_ENS_RADIUS = [0.5, 1.0, 1.5]

  # Regression Test Configs.
  TEST_CONFIGS = [205, 214, 242, 158]
