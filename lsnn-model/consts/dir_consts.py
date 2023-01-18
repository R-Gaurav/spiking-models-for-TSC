import _init_paths
from consts.exp_consts import EXC

BASE_DIR = "/home/rgaurav/Documents/Projects/"
RESULTS_DIR = BASE_DIR + "/ExpResults/final_results/LSNN/ECG5000/seed_9/results/"
DATA_DIR = BASE_DIR + "/lsnn_model/data/"

class DRC(object):

  def __init__(self, dataset):
    """
    Initializes the dataset paths.

    Args:
      dataset <str>: The dataset to work with e.g. ECG5000
    """
    if dataset == EXC.ECG5000:
      self._dataset = dataset
      self._data_path = DATA_DIR +"/ECG5000/"
      self._results_path = RESULTS_DIR +"/ECG5000/"
      self._pre_proc_data_path = DATA_DIR + "/ECG5000/pre_processed_data/"
      self._train_set = "ECG5000_TRAIN.arff"
      self._test_set = "ECG5000_TEST.arff"
      self._do_shuffle = True

    elif dataset == EXC.FORDA:
      self._dataset = dataset
      self._data_path = DATA_DIR +"/FORDA/"
      self._results_path = RESULTS_DIR +"/FORDA/"
      self._pre_proc_data_path = DATA_DIR + "/FORDA/pre_processed_data/"
      self._train_set = "FordA_TRAIN.arff"
      self._test_set = "FordA_TEST.arff"
      self._do_shuffle = True

    elif dataset == EXC.FORDB:
      self._dataset = dataset
      self._data_path = DATA_DIR + "/FORDB/"
      self._results_path = RESULTS_DIR + "/FORDB/"
      self._pre_proc_data_path = DATA_DIR + "/FORDB/pre_processed_data/"
      self._train_set = "FordB_TRAIN.arff"
      self._test_set = "FordB_TEST.arff"
      self._do_shuffle = True

    elif dataset == EXC.WAFER:
      self._dataset = dataset
      self._data_path = DATA_DIR + "/WAFER/"
      self._results_path = RESULTS_DIR + "/WAFER/"
      self._pre_proc_data_path = DATA_DIR + "/WAFER/pre_processed_data/"
      self._train_set = "Wafer_TRAIN.arff"
      self._test_set = "Wafer_TEST.arff"
      self._do_shuffle = True

    elif dataset == EXC.EQUAKES:
      self._dataset = dataset
      self._data_path = DATA_DIR + "/EQUAKES/"
      self._results_path = RESULTS_DIR + "/EQUAKES/"
      self._pre_proc_data_path = DATA_DIR + "/EQUAKES/pre_processed_data/"
      self._train_set = "Earthquakes_TRAIN.arff"
      self._test_set = "Earthquakes_TEST.arff"
      self._do_shuffle = True
