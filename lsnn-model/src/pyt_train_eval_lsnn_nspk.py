import _init_paths

import os
import pickle
import torch
import numpy as np
import sys

from consts.exp_consts import EXC
from src.extract_signals import ExtractSignals
from src.pyt_bwise_nspk_net import BWNspkNet
from utils.base_utils.data_prep_utils import DataPrepUtils
from utils.base_utils import log

class PTNspkTrainEvalModel(object):
  def __init__(self, dataset, rtc):
    self._rtc = rtc
    self._data = dataset
    self._lr = rtc.PYTORCH_LR
    self._batch_size = rtc.BATCH_SIZE
    self._do_normalize = rtc.NORMALIZE_DATASET
    self._test_eval_size = rtc.TEST_EVAL_SIZE
    self._dpu = DataPrepUtils(dataset, rtc)
    self._exs = ExtractSignals(rtc)

    # Get the Non-Spiking model.
    assert rtc.PYTORCH_MODEL_NAME == "BW_NSPK"
    self._model = BWNspkNet(dataset, rtc)

  def get_batches_of_x_y_from_ldn_sigs(self, is_train, num_samples=None,
                                       ldn_path=None):
    """
    Returns batches of x and y - training and test LDN signals.
    Note that the training data is not saved because it is shuffled for every
    training iteration.

    Args:
      is_train <bool>: Return batches of training data if True else test data.
      num_samples <int>: Number of samples to train/test upon.
      ldn_path <str>: Path/to/the/LDN/sigs/extracted/from/test/data.
    """
    log.INFO("Obtaining experiment compatible X-Y data...")

    if is_train == True and os.path.exists(ldn_path+"/train_X_ldn_sigs.p"):
      log.INFO("Found the already extracted LDN sigs of complete train data.")
      X_ldn = pickle.load(open(ldn_path+"/train_X_ldn_sigs.p", "rb"))
      Y = pickle.load(open(ldn_path+"/train_Y.p", "rb"))
    #if is_train == False and num_samples == EXC.NUM_TEST_SAMPLES[self._data] and (
    #    os.path.exists(ldn_path+"/test_X_ldn_sigs.p")):
    elif is_train == False and os.path.exists(ldn_path+"/test_X_ldn_sigs.p"):
      log.INFO("Found the already extracted LDN sigs of complete test data.")
      X_ldn = pickle.load(open(ldn_path+"/test_X_ldn_sigs.p", "rb"))
      Y = pickle.load(open(ldn_path+"/test_Y.p", "rb"))

    else:
      tr_x, tr_y, te_x, te_y = self._dpu.get_experiment_compatible_x_y_from_dataset(
          do_normalize=self._do_normalize)
      if is_train:
        X, Y = tr_x, tr_y
        log.INFO("Returning training data X, Y of shape: {0}, {1}".format(
                 X.shape, Y.shape))
        print(X.shape, Y.shape)
      else:
        X, Y = te_x, te_y
        log.INFO("Returning test data X, Y of shape: {0}, {1}".format(
                 X.shape, Y.shape))

      if num_samples:
        X, Y = X[:num_samples], Y[:num_samples]

      log.INFO("Data X and Y shape: {0}, {1}".format(X.shape, Y.shape))
      log.INFO("Obtaining the LDN signals from the signals X...")
      X_ldn = self._exs.run_pytorch_ldn_and_return_ldn_signals(X)
      assert X_ldn.shape[0] == X.shape[0]

      if is_train:
        log.INFO("Saving the extracted LDN sigs of the complete train data and Y...")
        pickle.dump(X_ldn, open(ldn_path+"/train_X_ldn_sigs.p", "wb"))
        pickle.dump(Y, open(ldn_path+"/train_Y.p", "wb"))
      #if is_train == False and num_samples == EXC.NUM_TEST_SAMPLES[self._data]:
      elif not is_train:
        log.INFO("Saving the extracted LDN sigs of the complete test data and Y...")
        pickle.dump(X_ldn, open(ldn_path+"/test_X_ldn_sigs.p", "wb"))
        pickle.dump(Y, open(ldn_path+"/test_Y.p", "wb"))

    for i in range(0, X_ldn.shape[0], self._batch_size):
      yield(
          torch.as_tensor(X_ldn[i : i+self._batch_size], dtype=EXC.PT_DTYPE),
          torch.as_tensor(Y[i : i+self._batch_size], dtype=EXC.PT_DTYPE))

  def train_model(self, epochs, ldn_path=None):
    """
    Trains the model.

    Args:
      epochs <int>: Number of epochs to train for.
      ldn_path <str>: Path/to/the/LDN/sigs/extracted/from/train or test/data.
    """
    optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
    log_softmax = torch.nn.LogSoftmax(dim=1)
    loss_func = torch.nn.NLLLoss()
    loss_history = []

    for e in range(epochs):
      self._model.train() # Set the model in train mode.
      log.INFO("Starting epoch: %s" % (e+1))
      # Delete the already existing training LDN files every 20th epoch to force
      # shuffling of the training set.
      if (e+1)%20 == 0:
        log.INFO("Epoch: %s, remove stale training LDN signals X and Y." % (e+1))
        os.remove(ldn_path + "/train_X_ldn_sigs.p")
        os.remove(ldn_path + "/train_Y.p")

      batches = self.get_batches_of_x_y_from_ldn_sigs(True, ldn_path=ldn_path)
      batch_losses = []
      for tr_x, tr_y in batches:
        output = self._model(tr_x)
        max_otp, _ = torch.max(output, 1)
        log_max_otp = log_softmax(max_otp)
        loss_value = loss_func(log_max_otp, torch.argmax(tr_y, dim=1))

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        batch_losses.append(loss_value.item())

      epoch_loss = torch.mean(torch.as_tensor(batch_losses))
      log.INFO("Epoch {0} loss: {1}".format(e+1, epoch_loss))
      loss_history.append(epoch_loss)

      # Evaluate on test set.
      eval_acc = self.evaluate_model(
          num_samples=self._test_eval_size, ldn_path=ldn_path)[0]
      log.INFO("Epoch {0} intermediate test accuracy: {1}".format(e+1, eval_acc))

    return loss_history

  def evaluate_model(self, num_samples=None, ldn_path=None):
    """
    Evaluates the trained model on the entire test set if `num_samples`=None,
    otherwise tests on the specified number of `num_samples`.

    Call it after calling the `train_model()`.

    Args:
      num_samples <int>: Number of test samples to evaluate upon.
      ldn_path <str>: Path/to/the/LDN/sigs/extracted/from/test/data.
    """
    log.INFO("Obtaining test X and Y...")
    batches = self.get_batches_of_x_y_from_ldn_sigs(
        False, num_samples, ldn_path) # is_train=False => Get test data.
    acc = []
    all_outputs = []
    self._model.eval() # Set the model in eval() mode. Call train() to train ltr.
    with torch.no_grad():
      for te_x, te_y in batches:
        output = self._model(te_x)
        all_outputs.append(output)
        max_over_nsteps, _ = torch.max(output, 1) # Max over time.
        _, pred_cls = torch.max(max_over_nsteps, 1) # Max over output units.
        _, true_cls = torch.max(te_y, 1)
        temp = torch.as_tensor(pred_cls == true_cls, dtype=EXC.PT_DTYPE).detach()
        acc.append(torch.mean(temp))

    log.INFO("Evaluation done, now returning results...")
    return torch.mean(torch.as_tensor(acc, dtype=EXC.PT_DTYPE)), all_outputs
