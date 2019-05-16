import collections
import numpy as np
import os
import tensorflow as tf

from specs import loss_specs, network_specs, dataset_specs


def get_dataset_specs(config_name,
                      train_batch_size,
                      val_batch_size,
                      test_batch_size,
                      input_seq_len,
                      pred_seq_len,
                      loss_spec,
                      img_res):
  """Returns the dataset specs associated with a given config.

  Args:
    config_name: The name of the config being loaded.
    train_batch_size: The batch size to use for training.
    val_batch_size: The batch size to use for validation.
  Returns:
    all_specs: A tuple of train and validation dataset specs.
  Raises:
    ValueError if config_name is unrecognized.
  """

  if "moving_mnist" in config_name:
    dataset_name = "moving_mnist"
  elif config_name == "bouncing_balls":
    dataset_name = "bouncing_balls"
  elif config_name == "kth_basic":
    dataset_name = "kth"
  elif config_name == "h36":
    dataset_name = "h36"
  elif config_name == "ucf101_rgb":
    dataset_name = "ucf101"
  elif config_name == "ucf101_grayscale":
    dataset_name = "ucf101"
  elif config_name[0:7] == "reacher":
    dataset_name = "reacher"
  elif config_name == "bair":
    dataset_name = "bair"
  elif config_name == "top":
    dataset_name = "top"
  elif config_name == "gridworld":
    dataset_name = "gridworld"
  else:
    raise ValueError("Unknown config_name: {}.".format(config_name))

  l_get_data_spec = lambda phase, batch_size: dataset_specs.get_data_spec(
      config_name,
      phase,
      batch_size,
      input_seq_len,
      pred_seq_len,
      loss_spec,
      img_res=img_res)

  dataset_spec_train = l_get_data_spec("train", train_batch_size)
  dataset_spec_val = l_get_data_spec("val", val_batch_size)
  dataset_spec_test = l_get_data_spec("test", test_batch_size)

  return dataset_name, dataset_spec_train, dataset_spec_val, dataset_spec_test


def get_loss_spec(config_name):
  """Returns the loss spec associated with a given configuration.

  Args:
    config_name: The name of the config being loaded.
  Returns:
    loss_spec: A loss spec.
  Raises:
    ValueError if config_name is unrecognized.
  """
  try:
    loss_spec = getattr(loss_specs, config_name)
  except:
    raise ValueError("Unknown config_name: {}.".format(config_name))

  return loss_spec


def get_network_specs(config_name):
  """Returns the network specs associated with a given config.

  Args:
    config_name: The name of the config being loaded.
  Returns:
    network_spec: A namedtuple, each element of which is the spec for a network
      component in the config being loaded.
  Raises:
    ValueError if config_name is unrecognized.
  """
  try:
    network_spec = getattr(network_specs, config_name)
  except:
    raise ValueError("Unknown config_name: {}.".format(config_name))

  return network_spec
