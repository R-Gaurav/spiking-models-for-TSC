import _init_paths

import numpy as np
from scipy.io import arff
from sklearn.utils import shuffle

from utils.base_utils import log

class DataPrepUtils(object):
  def __init__(self, dataset, rtc, exc, drc):
    """
    Args:
      dataset <str>: Dataset's name, e.g. "ECG5000" as in exp_consts.
    """
    self._rtc = rtc
    self._exc = exc
    self._data = dataset
    if self._data in [exc.ECG5000]:
      self._do_shuffle = True
    else:
      self._do_shuffle = False

    if self._data == exc.ECG5000:
      self._data_path = drc.ECG5000
      self._train_dataset = "ECG5000_TRAIN.arff"
      self._test_dataset = "ECG5000_TEST.arff"

  def get_x_y_from_dataset(self):
    """
    Return train_x, train_y, test_x, test_y data from the chosen dataset in
    `self._data_path`.

    Returns:
      np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """
    train_x, train_y = self._load_arff_dataset(self._train_dataset)
    test_x, test_y = self._load_arff_dataset(self._test_dataset)

    if self._do_shuffle: # Shuffle only the training data.
      train_x, train_y = shuffle(train_x, train_y, random_state=self._rtc.SEED)

    return train_x, train_y, test_x, test_y

  def _load_arff_dataset(self, dataset):
    log.INFO("Loading dataset: {}".format(dataset))
    raw_data, meta_data = arff.loadarff(self._data_path+"/%s" % dataset)
    cols = [x for x in meta_data]
    data = np.zeros([raw_data.shape[0],len(cols)]) # Shape: rows x cols.
    for i,col in zip(range(len(cols)),cols):
      data[:,i]=raw_data[col] # raw_data[col] is a column vector of shape (rows).

    log.INFO("Dataset loaded, it's shape: {}".format(data.shape))
    return data[:, :len(cols)-1], data[:, len(cols)-1] # Last column is the y-value.

  def make_ecg5000_binary_classification(self, train_y, test_y):
    """
    In ECG5000 dataset, the normal class is 1 and all the other classes are
    abnormal classes.

    Args:
      train_y <np.ndarray>: The training lables.
      test_y <np.ndarray>: The test labels.
    """
    assert self._data == self._exc.ECG5000

    train_y[np.where(train_y != 1)] = 2
    test_y[np.where(test_y != 1)] = 2

    return train_y, test_y

  def get_nengo_loihi_compatible_x_y_from_dataset(self, num_clss):
    """
    Returns the NengoLoihi compatible train_x, train_y dataset.

    Args:
      num_clss <int>: Number of classes in the training samples. May be different
                      than the actual number of classes in the raw dataset.
    """
    train_x, train_y, test_x, test_y = self.get_x_y_from_dataset()
    num_spls, num_tstps = train_x.shape[0], train_x.shape[1]
    assert num_spls == train_y.shape[0]

    if self._data == self._exc.ECG5000:
      train_y, test_y = self.make_ecg5000_binary_classification(train_y, test_y)
      assert num_spls == train_y.shape[0]
      train_y = np.eye(num_clss)[train_y.astype(np.int)-1]
      test_y = np.eye(num_clss)[test_y.astype(np.int)-1]

      return train_x, train_y, test_x, test_y
