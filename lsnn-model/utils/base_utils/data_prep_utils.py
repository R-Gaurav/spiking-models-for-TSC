import _init_paths

import numpy as np
import os
import pickle
from scipy.io import arff, loadmat
from sklearn.utils import shuffle
import torch

from utils.base_utils import log
from consts.dir_consts import DRC
from consts.exp_consts import EXC

class DataPrepUtils(DRC):
  def __init__(self, dataset, rtc):
    """
    Args:
      dataset <str>: Dataset's name, e.g. "FordA" as in exp_consts.
      rtc <class>: Run Time Constants class.
    """
    super().__init__(dataset)

    self._rtc = rtc
    self._sig_len = EXC.SIGNAL_DURATION[dataset]
    self._num_clss = EXC.NUM_CLASSES[dataset]

  def _load_arff_dataset(self, dataset):
    log.INFO("Loading dataset: {}".format(dataset))
    raw_data, meta_data = arff.loadarff(self._data_path+"/%s" % dataset)
    cols = [x for x in meta_data]
    data = np.zeros([raw_data.shape[0],len(cols)]) # Shape: rows x cols.
    for i,col in zip(range(len(cols)),cols):
      data[:,i]=raw_data[col] # raw_data[col] is a column vector of shape (rows).

    log.INFO("Dataset loaded, it's shape: {}".format(data.shape))
    return data[:, :len(cols)-1], data[:, len(cols)-1] # Last column is the y-value.

  def _normalize_values(self, values):
    """
    Normalizes the values between -1 and 1.

    Args:
      values <np.ndarray>: A one dimensional vector to be normalized.
    """
    min_val, max_val = np.min(values), np.max(values)
    return (2*(values-min_val) / (max_val-min_val)) - 1

  def get_x_y_from_dataset(self):
    """
    Return train_x, train_y, test_x, test_y data from the chosen dataset in
    `self._data_path`.

    Returns:
      np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """
    train_x, train_y = self._load_arff_dataset(self._train_set)
    test_x, test_y = self._load_arff_dataset(self._test_set)

    if self._do_shuffle: # Shuffle both the training/test data.
      train_x, train_y = shuffle(train_x, train_y)#, random_state=self._rtc.SEED)
      test_x, test_y = shuffle(test_x, test_y)#, random_state=self._rtc.SEED)

    return train_x, train_y, test_x, test_y

  def make_dataset_binary_classification(self, train_y, test_y):
    """
    In ECG5000 dataset, the normal class is 1 and all the other classes are
    abnormal classes. In FORDA dataset, the binary classes are -1 and 1. In
    Earthquakes dataset, the classes are 1 and 0.

    Args:
      train_y <np.ndarray>: The training lables.
      test_y <np.ndarray>: The test labels.
    """
    train_y[np.where(train_y != 1)] = 2
    test_y[np.where(test_y != 1)] = 2

    return train_y, test_y

  def get_experiment_compatible_x_y_from_dataset(self, do_normalize=False):
    """
    Returns experiment compatible train_x, train_y, test_x, test_y dataset.

    Args:
      do_normalize <bool>: Normalize the x part of the dataset if True else don't.
    """
    train_x, train_y, test_x, test_y = self.get_x_y_from_dataset()
    if self._dataset == EXC.FORDA:
      # Remove the last sample from training set to suit the batch size.
      train_x, train_y = train_x[:-1], train_y[:-1]

    if do_normalize:
      log.INFO("Normalizing the dataset!")
      for i in range(train_x.shape[0]):
        train_x[i] = self._normalize_values(train_x[i])
      for i in range(test_x.shape[0]):
        test_x[i] = self._normalize_values(test_x[i])

    if self._dataset in [EXC.ECG5000, EXC.FORDA, EXC.FORDB, EXC.WAFER,
                         EXC.EQUAKES]:
      num_samples, num_tsteps = train_x.shape
      assert num_samples == train_y.shape[0]
      train_y, test_y = self.make_dataset_binary_classification(train_y, test_y)
      assert num_samples == train_y.shape[0]
      train_y = np.eye(self._num_clss)[train_y.astype(np.int)-1]
      test_y = np.eye(self._num_clss)[test_y.astype(np.int)-1]

    return train_x, train_y, test_x, test_y
