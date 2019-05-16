"""Simple script to train a simple predictive LSTM on moving MNIST."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import glob
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

import configs
from data.moving_mnist import data_handler
from specs import dataset_specs
from viz import viz_utils

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("num_training_iterations", 100000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("num_testing_iterations", 0,
                        "Number of iterations to test on. Zero means run on all test data")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (train losses only).")
tf.flags.DEFINE_integer("validation_interval", 1000,
                        "Iterations between validation and full reports "
                        "(including images).")
tf.flags.DEFINE_integer("checkpoint_interval", 1000,
                        "Checkpointing step interval. If 0, no checkpoints")
tf.flags.DEFINE_string("learning_rate_reduce_criterion", "plateau",
                       "Criterion for reducing learning rate: "
                       "scheduled or plateau.")
tf.flags.DEFINE_float("plateau_scale_criterion", 2e-3,
                      "The criterion for determining a plateau in the "
                      "validation loss, as a fraction of the validation loss.")
tf.flags.DEFINE_integer("plateau_min_delay", 20000,
                        "The minimum number of batches to wait before decaying "
                        "the learning rate, if using a plateau criterion.")
tf.flags.DEFINE_integer("reduce_learning_rate_interval", 20000,
                        "Iterations between learning rate reductions.")
tf.flags.DEFINE_integer("train_batch_size", 50,
                        "Batch size for training.")
tf.flags.DEFINE_integer("val_batch_size", 50,
                        "Batch size for validation.")
tf.flags.DEFINE_integer("test_batch_size", 20,
                        "Batch size for testing. ")
tf.flags.DEFINE_integer("input_seq_len", 0,
                        "Length of the past sequence. If zero, defaults to a"
                        "dataset dependent value.")
tf.flags.DEFINE_integer("pred_seq_len", 0,
                        "Length of the future sequence. If zero, defaults to a"
                        "dataset dependent value.")
tf.flags.DEFINE_integer("test_sequence_repeat", 0,
                        "How often should val/test sequences be repeated to see variation.")
tf.flags.DEFINE_float("max_grad_norm", 5,
                      "Gradient clipping norm limit. If <=0, "
                      "no clipping is applied.")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Optimizer learning rate.")
tf.flags.DEFINE_float("reduce_learning_rate_multiplier", 1.0,
                      "Learning rate is multiplied by this when reduced.")
tf.flags.DEFINE_float("increase_pp_lambda_multiplier", 1.0,
                      "Pred. potential weight is increased by this factor.")
tf.flags.DEFINE_integer("increase_pp_lambda_interval", 20000,
                        "Iterations between pred. potential lambda increases.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-8,
                      "Epsilon used for Adam optimizer.")
tf.flags.DEFINE_float("regularizer_weight", 1e-4,
                      "Weight for weight decay regularizer.")
tf.flags.DEFINE_string("base_dir", "~/logs",
                       "Base directory for checkpoints and summaries.")
tf.flags.DEFINE_string("network_config_name", "simple_conv_lstm_conv",
                       "The network architecture configuration to use.")
tf.flags.DEFINE_string("dataset_config_name", "moving_mnist_basic",
                       "The dataset configuration to use.")
tf.flags.DEFINE_string("loss_config_name", "mnist_bce_image_latent",
                       "The loss configuration to use.")
tf.flags.DEFINE_boolean("create_new_subdir", False,
                        "If True, creates a new subdirectory in base_dir. "
                        "Set to False to reload. Defaults to True.")
tf.flags.DEFINE_string("plateau_criterion_loss_name",
                       "total_loss_val",
                       "Loss to use as criterion for decaying learning rate.")
tf.flags.DEFINE_integer("output_buffer_size", 100,
                        "The size of the training buffer size for the output "
                        "prefetch queue, as a multiple of the batch size. E.g. "
                        "a value of 10 will produce a buffer of size 10 * "
                        "batch_size. Defaults to 100.")
tf.flags.DEFINE_integer("shuffle_buffer_size", 5,
                        "The size of the training buffer size for the shuffle "
                        "queue, as a multiple of the batch size. E.g. a value "
                        "of 10 will produce a buffer of size 10 * batch_size. "
                        "Defaults to 100.")
tf.flags.DEFINE_boolean("kth_downsample", False,
                        "If True, downsamples videos to half-resolution."
                        "I.e. 64x64 for KTH or 128x128 for UCF.")
tf.flags.DEFINE_boolean("compute_predictive_potential", False,
                        "If True, compute the predictive potential for ground "
                        "truth and predicted sequences. If False, these terms "
                        "are not computed. Note that PPs can be computed and "
                        "not optimized, e.g. for monitoring. Defaults to False.")
tf.flags.DEFINE_string("optimizer_type", "adam",
                       "Which optimizer to use. Can be "
                       "adam, sgd, momentum, or nesterov. Defaults to adam.")
tf.flags.DEFINE_boolean("use_recursive_image", True,
                        "If True, uses recursive image skips when using a "
                        "recursive network. Ignored if network is not recursive.")
tf.flags.DEFINE_boolean("show_encoder_predictions", False,
                        "If True, shows encoder predictions in tensorboard as "
                        "opposed to reconstructions.")

# GAN
tf.flags.DEFINE_boolean("use_gan", False,
                        "If True, a GAN is used for training.")
tf.flags.DEFINE_boolean("use_image_gan_rec", False,
                        "If True and use_gan is specified, a GAN on "
                        "reconstructed images is used.")
tf.flags.DEFINE_boolean("use_video_gan_rec", False,
                        "If True and use_gan is specified, a GAN on "
                        "reconstructed videos is used.")
tf.flags.DEFINE_boolean("use_image_gan_pred", True,
                        "If True and use_gan is specified, a GAN on predicted "
                        "images is used.")
tf.flags.DEFINE_boolean("use_video_gan_pred", True,
                        "If True and use_gan is specified, a GAN on predicted "
                        "videos is used.")
tf.flags.DEFINE_boolean("conditional_gan", False,
                        "If True, the video/motion discriminator on predicted "
                        "images is conditioned on the past true images as well.")
tf.flags.DEFINE_boolean("is_patchGAN", False,
                        "If True, a patchGAN is used on images instead of a "
                        "standard GAN.")
tf.flags.DEFINE_boolean("latent_vgan", False,
                        "If True, the future video gan operates on latents "
                        "instead of images.")
tf.flags.DEFINE_float("relative_gan_lr", 1e-3,
                        "The coefficient on the GAN loss. Default is 1e-3")
tf.flags.DEFINE_float("relative_vgan_lr", 1e-2,
                        "The coefficient on the video GAN loss. Default is 1e-2")
tf.flags.DEFINE_string("imgan_discriminator_spec", "small",
                        "Specification for the image gan discriminator. Can be "
                        "'small', 'med' or 'big'.")
tf.flags.DEFINE_float("dicriminator_regularizer_weight", 0,
                      "Weight for weight decay regularizer for discriminator networks.")

# Variational
tf.flags.DEFINE_boolean("use_variational", False,
                        "If True, sampling from learned prior is used.")
tf.flags.DEFINE_boolean("teacher_forcing", True,
                        "Works with variational LSTM. If False, prediction and "
                        "prior networks observe generated latents in the future "
                        "at training. If True, they observe the true frames.")
tf.flags.DEFINE_boolean("fixed_prior", False,
                        "If True, unit Gaussian is used as fixed prior distribution.")
tf.flags.DEFINE_boolean("infer_actions", False,
                        "If True, regression network for inferring actions is added.")
tf.flags.DEFINE_boolean("action_inference_with_image", False,
                        "If True, image is concatenated to latent for action regression.")
tf.flags.DEFINE_float("lr_action_inference", 1e-3,
                      "Learning rate for the action inference network. Used when "
                      "infer_actions is true.")
tf.flags.DEFINE_boolean("enforce_composable_actions", False,
                        "If True, loss on composed sequences is enforced.")
tf.flags.DEFINE_integer("comp_seq_length", 2,
                        "Defines length of composable sequences.")



def get_batch(data,
              batch_size,
              seq_len_total,
              input_seq_len,
              pred_seq_len,
              im_height,
              im_width,
              channels):
  """Returns a batch of the data, partitioned into input and predicted seqs.

  Args:
    data: The data generating object.
    batch_size: The number of sequences in each batch.
    seq_len_total: The total number of images in each sequence, including both
      input and prediction sequences.
    input_seq_len: The length of input sequences, in frames.
    pred_seq_len: The length of prediction sequences, in frames.
    im_height: The height of the data, in pixels.
    im_width: The width of the data, in pixels.
  Returns:
    input_seqs: A batch of sequences, ready to use as network input. An array
      of shape [input_seq_len, batch_size, 1, im_height, im_width],
      dtype float32.
    pred_seqs: A batch of prediction sequences, ready to use as a network
      target. An array of shape
      [pred_seq_len, batch_size, 1, im_height, im_width], dtype float32.
  """

  # GetBatch returns an array of size
  # batch x (seq_len_total * im_height * im_width)
  batch_data, action_label = data.GetBatch()

  # Split out sequence, channel, and spatial dimensions
  batch_data = batch_data.reshape(
      [batch_size, seq_len_total, channels, im_height, im_width])

  # Convert to time major
  batch_data = batch_data.transpose([1, 0, 2, 3, 4])
  input_seqs = batch_data[:input_seq_len, ...]
  pred_seqs = batch_data[input_seq_len:(input_seq_len + pred_seq_len), ...]

  return input_seqs, pred_seqs, action_label


def video_random_flip_left_right(video, seed=None):
  """Randomly flip a video horizontally (left to right).
  With a 1 in 2 chance, outputs the contents of `video` flipped along the
  third dimension, which is `width`.  Otherwise output the video as-is.
  Args:
    video: A 4-D tensor of shape `[time, height, width, channels].`
    seed: A Python integer. Used to create a random seed. See
      @{tf.set_random_seed}
      for behavior.
  Returns:
    A 4-D tensor of the same type and shape as `video`.
  Raises:
    ValueError: if the shape of `video` not supported.
  """
  video = ops.convert_to_tensor(video, name='video')
  shape = video.get_shape()
  if shape.ndims == 3 or shape.ndims is None:
    video_out = tf.image.random_flip_up_down(video, seed=seed)
  elif shape.ndims == 4:
    uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
    mirror_cond = math_ops.less(uniform_random, .5)
    video_out = control_flow_ops.cond(mirror_cond,
                                   lambda: array_ops.reverse(video, [2]),
                                   lambda: video)
    video_out.set_shape(video.get_shape())
  else:
    raise ValueError('\'video\' must have either 3 or 4 dimensions.')

  return video_out


def video_augmentation(video, dataset_config, phase="train"):
  """Runs augmentation function on the video input.

  Args:
    video: A float32 Tensor of shape [time, height, width, channels].
    dataset_config: The config for the current dataset.
    phase: The training phase.
  returns:
    video_augmented: The video with the config's specified augmentation applied.
  """

  with tf.name_scope("dataset_augment_{}".format(phase)):
    crop_cond = (dataset_config.input_height != dataset_config.im_height or
                 dataset_config.input_width != dataset_config.im_width)

    if crop_cond:
      if phase == "train":
        offset_height = tf.random_uniform(
            [1,],
            minval=0,
            maxval=dataset_config.input_height - dataset_config.im_height + 1,
            dtype=tf.int32)
        offset_width = tf.random_uniform(
            [1,],
            minval=0,
            maxval=dataset_config.input_width - dataset_config.im_width + 1,
            dtype=tf.int32)
      else:
        # Fixed, central crop
        offset_height = [int(
            (dataset_config.input_height - dataset_config.im_height + 1) / 2)]
        offset_width = [int(
            (dataset_config.input_width - dataset_config.im_width + 1) / 2)]

      offset_height = offset_height[0]
      offset_width = offset_width[0]
      target_height = dataset_config.im_height
      target_width = dataset_config.im_width

      video = tf.image.crop_to_bounding_box(
          video,
          offset_height=offset_height,
          offset_width=offset_width,
          target_height=target_height,
          target_width=target_width)

    if phase == "train":
      if dataset_config.flip_lr:
        video = video_random_flip_left_right(video)

  return video


def get_video_parse_function(dataset_config, dataset_name, phase):
  """"Returns a video parse function based on the config and phase."""
  if isinstance(dataset_config, dataset_specs.ReacherConfig):
      decode_actions = True
  else:
      decode_actions = False

  def video_parse_function(example_proto):
    """Parses and preprocesses the features from a video dataset tfrecord."""
    # TODO(drewjaegle): extend this if we need classes, etc.
    # TODO(drewjaegle): put this in a single place (it's read and written)
    features = {
        "video": tf.VarLenFeature(dtype=tf.string),
        "video_length": tf.FixedLenFeature(dtype=tf.int64, shape=[],
                                           default_value=10),
        "video_height": tf.FixedLenFeature(dtype=tf.int64, shape=[],
                                           default_value=128),
        "video_width": tf.FixedLenFeature(dtype=tf.int64, shape=[],
                                          default_value=128),
        "video_channels": tf.FixedLenFeature(dtype=tf.int64, shape=[],
                                             default_value=1),
    }

    if decode_actions:
        features.update({
            "actions": tf.VarLenFeature(dtype=tf.string),
            "actions_summed": tf.VarLenFeature(dtype=tf.string),
            "actions_absolute": tf.VarLenFeature(dtype=tf.string),
            "num_actions": tf.FixedLenFeature(dtype=tf.int64, shape=[],
                                              default_value=1)
        })

    parsed_features = tf.parse_single_example(example_proto, features)
    
    def decode_raw_tensor(name, shape, type):
        tensor = tf.sparse_tensor_to_dense(parsed_features[name], default_value="")
        if type == 'uint8':
            tensor = tf.cast(tf.decode_raw(tensor, tf.uint8), tf.float32)
            # Rescale tensor from [0, 255] to [-1, 1]
            tensor = tensor * (2 / 255) - 1
        elif type == 'float32':
            tensor = tf.decode_raw(tensor, tf.float32)
        else:
            raise ValueError("Only uint8 and float32 are supported tfRecord tensor types.")
        tensor = tf.reshape(tensor, shape)
        return tensor
        
    video = decode_raw_tensor("video", 
                              shape=tf.stack([parsed_features["video_length"],
                                              parsed_features["video_height"],
                                              parsed_features["video_width"],
                                              parsed_features["video_channels"]]),
                              type='uint8')

    if decode_actions:
        if dataset_config.action_coding == "absolute":
            feature_name = "actions_absolute"
        elif dataset_config.action_coding == "summed":
            feature_name = "actions_summed"
        elif dataset_config.action_coding == "relative":
            feature_name = "actions"
        else:
            raise ValueError("Reacher action_coding needs to be [relative/absolute/summed]")
        actions = decode_raw_tensor(feature_name,
                                    shape=tf.stack([parsed_features["video_length"],
                                                    parsed_features["num_actions"]]),
                                    type='float32')
    else:
        actions = None
    

    # Sample if the video is bigger than needed
    # datasets can have chunks bigger than num_frames now so this does not
    # depend on dataset_config.chunked_examples
    if phase == "train" or phase == "test":
      # unchunked test means that we have to sample to get enough data
      # the sampling is deterministic for test (tf.set_random_seed is called)
      start_index = tf.random_uniform(
          [1,],
          minval=0,
          maxval=(parsed_features["video_length"] -
                  dataset_config.num_frames + 1),
          dtype=tf.int64)
    else:
      # this is to make sure validation is deterministic
      start_index = [0,]
    video = video[
        start_index[0]:start_index[0] + dataset_config.num_frames, ...]
    if actions is not None:
        actions = actions[
                    start_index[0]:start_index[0] + dataset_config.num_frames, ...]

    if not decode_actions:
        # can only augment video data if no action annotation
        video = video_augmentation(video, dataset_config, phase=phase)

    if dataset_name == "ucf101" and dataset_config.channels == 1:
      # Convert videos to
      video = tf.image.rgb_to_grayscale(video)

    video.set_shape(
        [dataset_config.num_frames,
         dataset_config.im_height,
         dataset_config.im_width,
         dataset_config.channels])

    if actions is not None:
        actions.set_shape(
            [dataset_config.num_frames,
             dataset_config.num_actions])

    if FLAGS.kth_downsample:
      video = tf.image.resize_images(
          video,
          tf.convert_to_tensor(
              [int(dataset_config.im_height / 2),
               int(dataset_config.im_width / 2)],
              dtype=tf.int32))

    # Reshape to TCHW from THWC
    video = tf.transpose(video, [0, 3, 1, 2])

    # Split images to input and predict
    input_sequence = video[:dataset_config.input_seq_len, ...]
    predict_sequence = video[
        dataset_config.input_seq_len:(dataset_config.input_seq_len +
                                      dataset_config.pred_seq_len),
        ...]

    if actions is None:
        actions = []    # map function does not support None outputs

    return input_sequence, predict_sequence, actions

  return video_parse_function


def get_video_dataset(
    dataset_config,
    dataset_name,
    batch_size,
    num_threads=8,
    output_buffer_size=None,
    shuffle_buffer_size=None,
    phase="train"):
  """Returns the dataset associated with a config.

  See goo.gl/zx4Gq9 for advice on setting buffer sizes.

  Args:
    dataset_config: A namedtuple for the current session.
    dataset_name: A string with the name of the dataset.
    batch_size: The number of sequences in a batch.
    num_threads: The number of threads used to preprocess the data.
    output_buffer_size: The size of the output buffer, as a multiple of the
      batch size - e.g. for batch_size 10, a value of 5 will return a buffer
      of size 50=5*10. If None, returns 10 * batch_size. Defaults to None.
    shuffle_buffer_size: The size of the shuffle buffer, as a multiple of the
      batch size - e.g. for batch_size 10, a value of 5 will return a buffer
      of size 50=5*10. If None, returns 10 * batch_size. Defaults to None.
    phase: The phase of training, "train" or "val".
  """
  if output_buffer_size is None:
    # Used for preloading - relatively small is okay
    output_buffer_size = 10 * batch_size
  if shuffle_buffer_size is None:
    # Large enough, assuming random shards
    shuffle_buffer_size = 10 * batch_size

  filenames = glob.glob(os.path.join(dataset_config.data_dir, "*.tfrecord"))

  map_fn = get_video_parse_function(dataset_config, dataset_name, phase)
  dataset = tf.data.TFRecordDataset(filenames)
  if phase == "train":
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.map(
      map_fn,
      num_parallel_calls=num_threads)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(output_buffer_size)

  batched_dataset = dataset.batch(batch_size)

  if FLAGS.kth_downsample:
    im_height = int(dataset_config.im_height / 2)
    im_width = int(dataset_config.im_width / 2)
  else:
    im_height = dataset_config.im_height
    im_width = dataset_config.im_width

  dataset_shapes = {
    "input": [dataset_config.input_seq_len,
              batch_size,
              dataset_config.channels,
              im_height,
              im_width],
    "predict": [dataset_config.pred_seq_len,
                batch_size,
                dataset_config.channels,
                im_height,
                im_width]}

  if isinstance(dataset_config, dataset_specs.ReacherConfig):
      dataset_shapes.update({
          "actions": [dataset_config.input_seq_len + dataset_config.pred_seq_len,
                      batch_size,
                      dataset_config.num_actions]
      })

  dataset_size = dataset_config.n_examples

  return batched_dataset, dataset_shapes, dataset_size


def build_dataset(
    dataset_name,
    dataset_spec,
    batch_size,
    phase):
  """Builds a dataset for training/validation.

  Args:
    dataset_name: The name of the dataset to use.
    dataset_name: The spec of the dataset to use. This may be phase dependent.
    batch_size: The batch size.
    phase: The phase of training, e.g. 'train' or 'val'.
  Returns:
    data_tuple: A namedtuple with TF Dataset API Dataset and other fields
      useful for running an network on the data.
  Raises:
    ValueError if the configuration specifies a dataset that is not configured.
  """
  # tf.data Dataset, shape dictionary, and size of dataset
  DatasetTuple = collections.namedtuple(
    "DatasetTuple",
    "dataset_name dataset shapes dataset_size"
  )
  handler = None    # set default value
  if dataset_name == "moving_mnist":
    get_batch, dataset_shapes, dataset_size, handler = build_batch_generator(
        dataset_spec, batch_size)

    dataset = tf.data.Dataset.from_generator(
        get_batch,
        (tf.float32, tf.float32, tf.float32),
        (tf.TensorShape(dataset_shapes["input"]),
         tf.TensorShape(dataset_shapes["predict"]),
         tf.TensorShape(dataset_shapes["actions"])))
  elif dataset_name == "kth" or \
          dataset_name == "reacher":
    if phase == "train":
      output_buffer_size = FLAGS.output_buffer_size
      shuffle_buffer_size = FLAGS.shuffle_buffer_size
    else:
      output_buffer_size = 1
      shuffle_buffer_size = 1

    dataset, dataset_shapes, dataset_size = get_video_dataset(
        dataset_spec,
        dataset_name,
        batch_size,
        phase=phase,
        output_buffer_size=FLAGS.output_buffer_size,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size)
  elif dataset_name == "ucf101":
    if phase == "train":
      output_buffer_size = FLAGS.output_buffer_size
      shuffle_buffer_size = FLAGS.shuffle_buffer_size
    else:
      output_buffer_size = 1
      shuffle_buffer_size = 1

    dataset, dataset_shapes, dataset_size = get_video_dataset(
        dataset_spec,
        dataset_name,
        batch_size,
        phase=phase,
        output_buffer_size=FLAGS.output_buffer_size,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size)
  else:
    raise ValueError("Dataset {} not yet configured.".format(dataset_name))

  data_tuple = DatasetTuple(
      dataset_name,
      dataset,
      dataset_shapes,
      dataset_size)

  return data_tuple, handler

def run_step(sess,
             monitor_values,
             opt_steps,
             train_ops_gan,
             train_steps_gan,
             monitor_index,
             phase,
             feed_dict_elems,
             use_gan,
             metrics_only=False):
  """Runs the steps appropriate for the given phase (train/val).

  Args:
    sess: The current session.
    monitor_values: A dict containing all Tensors whose values are monitored,
      but which do not control optimization.
    opt_steps: The non-GAN training ops to run.
    train_ops_gan: The GAN training ops to run.
    train_steps_gan: A namedtuple of GAN train steps.
    monitor_index: A nested dict allocating each Tensor to an optimization
      phase and specifying its type (loss, scalar, image, or hist).
    phase: The phase of training: 'train' or 'val'.
    feed_dict_elems: A dictionary of values to pass to a feed_dict.
    use_gan: If True, GAN alternating minimization run. Otherwise, all ops run
      together.
    metrics_only: If True, only loss and metric values will be reported.
      Otherwise, all values will be reported (including images, etc.). Defaults
      to False.
  Returns:
    run_output: A dictionary of evaluated values.
  Raises:
    ValueError: if phase is not 'train' or 'val'.
  """
  if phase == "train":
    sess_dict = get_sess_dict(monitor_values,
                              opt_steps,
                              train_ops_gan,
                              monitor_index,
                              phase,
                              metrics_only=metrics_only,
                              use_gan=use_gan)
  elif phase == "val":
    sess_dict = get_sess_dict(monitor_values,
                              None,
                              None,
                              monitor_index,
                              phase,
                              metrics_only=metrics_only,
                              use_gan=use_gan)
  else:
    raise ValueError("Unknown phase. Must be 'train' or 'val'.")

  if use_gan:
    # Run generator training steps.
    for _ in range(train_steps_gan.generator_train_steps):
      # Make sure this output is correct
      run_output = sess.run(
          sess_dict["generator"],
          feed_dict=feed_dict_elems)

    # Run discriminator training steps.
    if train_steps_gan.discriminator_train_steps > 1:
      print("For > 1, need to modify sess_dict here. "
            "Should only be logging on the last discriminator step. "
            "Should hold for validation and training.")

    for _ in range(train_steps_gan.discriminator_train_steps):
      run_output = sess.run(
          sess_dict["discriminator"],
          feed_dict=feed_dict_elems)
    sess.run(sess_dict["global_step"])  # Update the global step
  else:
    run_output = sess.run(sess_dict, feed_dict=feed_dict_elems)

  return run_output


def setup_decay_lr(reduce_criterion,
                   scheduled_reduce_interval,
                   plateau_min_delay,
                   plateau_scale_criterion):
  """Builds a closure for checking and decaying the learning rate.

  Args:
    reduce_criterion: The criterion for reducing the learning rate (a string).
    scheduled_reduce_interval: The interal at which to reduce the learning rate,
      if using a scheduled reduce_criterion.
    plateau_min_delay: The minimum number of batches to wait before decaying
      the learning rate, if using a plateau criterion.
    plateau_scale_criterion: The criterion for determining a plateau in the
      validation loss, as a fraction of the validation loss. Typically a small
      value in (0, 1].
  Returns:
    decay_lr: A function that will call or not call the reduce_learning_rate
      based on the chosen criterion and the relevant status.
  """
  # Keep track of state variables to update
  state_variables = {
    "previous_iteration": 0,
    "time_since_plateau": 0,
    "plateau_criterion_loss_old": np.inf,
  }


  def decay_lr(current_iteration, plateau_criterion_loss):
    """A function that reduces the learning rate when appropriate.

    Args:
      current_iteration: The current training iteration.
      plateau_criterion_loss: The current value of the loss value monitored
        for plateau-based learning rate decay.

    Returns:
      reduce_lr: True when learning rate should be reduced, False otherwise.
    """
    reduce_lr = False

    time_since_last = current_iteration - state_variables["previous_iteration"]

    if reduce_criterion == "scheduled":
      state_variables["time_since_plateau"] += time_since_last
      if state_variables["time_since_plateau"] >= scheduled_reduce_interval:
        reduce_lr = True
        state_variables["time_since_plateau"] = 0

    elif reduce_criterion == "plateau":
      delay_satisfied = (state_variables["time_since_plateau"] >
                         plateau_min_delay)
      at_plateau = (
          state_variables["plateau_criterion_loss_old"] -
          plateau_criterion_loss) < (plateau_scale_criterion *
                                     plateau_criterion_loss)
      if delay_satisfied and at_plateau:
        reduce_lr = True
        state_variables["time_since_plateau"] = 0
      else:
        state_variables["time_since_plateau"] += time_since_last
      state_variables["plateau_criterion_loss_old"] = plateau_criterion_loss
    else:
      raise ValueError("Unknown learning rate reduce_criterion {}".format(
          reduce_criterion))

    state_variables["previous_iteration"] = current_iteration
    return reduce_lr

  return decay_lr


def build_batch_generator(config,
                          batch_size):
  """Returns a Dataset generator for moving MNIST or reacher examples.

  Args:
    config: The config file for the dataset.
    batch_size: The number of sequences per batch.
  Returns:
    batch_generator: A generator for tf.data.Dataset.from_generator batching.
    dataset_shapes: A dictionary specifying the shapes of the input and
      predict data Tensors.
    dataset_size: The number of examples in the dataset.
  """
  handler = data_handler.ChooseDataHandler(config)

  dataset_size = handler.GetDatasetSize()
  seq_len_total = handler.GetSeqLength()
  im_height = handler.GetImageSize()
  im_width = im_height

  input_seq_len = config.input_seq_len
  pred_seq_len = config.pred_seq_len

  def batch_generator():
    """A generator that yields a batch of data each time it's called.
    """
    while True:
      input_seqs, pred_seqs, action_seqs = get_batch(
          handler,
          batch_size,
          seq_len_total,
          input_seq_len,
          pred_seq_len,
          im_height,
          im_width,
          config.channels)
      yield (input_seqs, pred_seqs, action_seqs)

  dataset_shapes = {
    "input": [input_seq_len, batch_size, config.channels, im_height, im_width],
    "predict": [pred_seq_len, batch_size, config.channels, im_height, im_width],
    "actions": [seq_len_total, batch_size, config.num_actions]
  }
  return batch_generator, dataset_shapes, dataset_size, handler


def get_sess_dict(monitor_values,
                  opt_steps,
                  train_ops_gan,
                  monitor_index,
                  phase,
                  metrics_only=True,
                  use_gan=False):
  """Returns the appropriate dictionary of Tensors to pass to a run call.

  Args:
    monitor_values: A dict containing all Tensors whose values are monitored,
      but which do not control optimization.
    opt_steps: A dict containing all non-GAN Tensors controlling optimization.
    train_ops_gan: A namedtuple with fields for GAN optimization.
    monitor_index: A nested dict allocating each Tensor to an optimization
      phase and specifying its type (loss, scalar, image, or hist).
    phase: The phase of training: 'train' or 'val'.
    metrics_only: If True, only monitor values corresponding to losses and
      metrics will be returned. If False, all values for this phase will be
      returned. Defaults to True.
    use_gan: If True, returns a nested sess_dict with values for generator and
      discriminator phases. Otherwise, returns a non-nested sess_dict for a
      single optimization step.
  Returns:
    sess_dict: The sess_dict for the current run call.
  """
  sess_names = []

  for type_name, type_values in monitor_index[phase].items():
    if metrics_only:
      update_sess = (type_name == "loss") or (type_name == "metric")
    else:
      update_sess = True

    if update_sess:
      sess_names.extend(type_values)

  sess_dict = dict((key_i, val_i) for key_i, val_i in monitor_values.items()
                   if key_i in sess_names)

  if use_gan:
    # All non-training ops run with discriminator
    sess_dict = {
        "generator": {},
        "discriminator": sess_dict,
        "global_step": {},
    }

  if phase == "train":
    # Add training ops.
    # NB: for GAN, losses are already incorporated into gan train_ops,
    #  so opt_steps can be ignored.
    if use_gan:
      sess_dict["generator"]["generator_train_op"] = train_ops_gan.generator_train_op
      sess_dict["discriminator"].update(train_ops_gan.discriminator_train_op)
      sess_dict["global_step"]["inc_global_step"] = train_ops_gan.global_step_inc_op
    else:
      # Directly use the optimization stuff
      sess_dict.update(opt_steps)

  return sess_dict


def update_full_loss_vals(monitor_index,
                          full_loss_vals=None,
                          val_output=None,
                          phase="val"):
  """Updates the accumulated values of tracked losses.

  Args:
    monitor_index: A nested dictionary of Tensors, sorted by type.
    full_loss_vals: A dictionary of current state of the tracked losses. If
      None, the dictionary is initialized with all values set to zero.
      Defaults to None.
    val_output: The current evaluated values of a sess.run call. If None,
      values are not updated. Defaults to None.
  """
  tracking_fields = monitor_index[phase]["loss"] + monitor_index[phase]["metric"]

  if full_loss_vals is None:
    full_loss_vals = {}
    for tracking_field_i in tracking_fields:
      full_loss_vals[tracking_field_i] = 0

  if val_output is not None:
    for tracking_field_i in tracking_fields:
      full_loss_vals[tracking_field_i] += val_output[tracking_field_i]

  return full_loss_vals


def log_sess_output(sess_output,
                    monitor_index,
                    logger,
                    iteration,
                    dataset_name,
                    base_dir,
                    n_seqs=5,
                    phase="train",
                    build_seq_ims=True,
                    repeat=0):
  """Logs the session output.

  Args:
    sess_output: A dictionary of evaluated values (not tensors) to log.
    monitor_index: The index of Tensor phase and types
    logger: A Logger object.
    iteration: The current training iteration.
    dataset_name: A string with the name of the dataset.
    n_seq: Number of sequences that should be logged at max. Defaults to 5.
    phase: The current phase, "train" or "val".
    build_seq_ims: If True, will be build sequence images from the ground truth
      and estimates returned by evaluation.. Defaults to True.
    base_dir: Base directory for saving visualization results.
  Raises:
    ValueError if phase is not "train" or "val".
  """
  for type_key, type_vals in monitor_index[phase].items():
    for type_ind in type_vals:
      if type_key in ["scalar", "metric", "loss"]:
        if type_ind in sess_output:
          logger.log_scalar(
              tag=type_ind,
              value=sess_output[type_ind],
              step=iteration)
      elif type_key == "hist":
        if type_ind in sess_output:
          logger.log_histogram(
              tag=type_ind,
              values=sess_output[type_ind],
              step=iteration)
      elif type_key == "sum":
        if type_ind in sess_output:
          logger.log_summary(sess_output[type_ind], step=iteration)

  # Build sequence image to save
  if build_seq_ims:
    if FLAGS.show_encoder_predictions:
      input_estimate_str = 'encoder_rnn_image_predictions'
    else:
      input_estimate_str = "image_reconstructions"
      
    if phase == "train":
      im_keys = {
        "input": "input_images",
        "predict": "predict_images",
        "input_estimate": input_estimate_str,
        "predict_estimate": "image_predictions",
      }
      seq_tag = "image_sequence_estimates"
    elif phase == "val" or phase == "test":
      im_keys = {
        "input": "input_images_val",
        "predict": "predict_images_val",
        "input_estimate": input_estimate_str + "_val",
        "predict_estimate": "image_predictions_val",
      }
      seq_tag = "image_sequence_estimates_val"
    else:
      raise ValueError("Unknown phase: {}".format(phase))

    if tf.flags.FLAGS.enforce_composable_actions:
      if phase == "train":
        im_keys.update({"comp_seq": "decoded_seq_comb", "comp_idxs": "comp_idxs"})
      elif phase == "val" or phase == "test":
        im_keys.update({"comp_seq": "decoded_seq_comp_val", "comp_idxs": "comp_idxs_val"})

    for _, val in im_keys.items():
      if val not in sess_output:
        raise ValueError('Tensor value {} not in sess.run output.'.format(val))

    input_estimate = sess_output[im_keys["input_estimate"]]
    if FLAGS.show_encoder_predictions:
      # for plotting purposes, I append the sequence with a blank image
      input_length = input_estimate.shape[0] + 1
      input_estimate = viz_utils.pad_sequence(input_estimate,
                                              np.arange(input_length)[1:],
                                              input_length)

    n_seqs = min(n_seqs, sess_output[im_keys["input"]].shape[1])
    plot_seqs = [np.concatenate((sess_output[im_keys["input"]], sess_output[im_keys["predict"]])),
                 np.concatenate((input_estimate, sess_output[im_keys["predict_estimate"]]))]
    if tf.flags.FLAGS.enforce_composable_actions:
        comp_seq = sess_output[im_keys["comp_seq"]]
        comp_idxs = np.asarray(sess_output[im_keys["comp_idxs"]], dtype=np.int16)
        comp_seq = viz_utils.pad_sequence(comp_seq, comp_idxs, plot_seqs[0].shape[0])
        plot_seqs.append(comp_seq)
    im_seqs = viz_utils.generate_sequence_images(
        input_seq_list=plot_seqs,
        dataset_name=dataset_name,
        n_seqs=n_seqs)

    if tf.flags.FLAGS.use_variational and (phase == "val" or phase == "test"):
      # Trajectory transplantation
      trajectory_transplantation = True
      if repeat>0:
        def downsample(x):
          # Do not plot repeating sequences
          idxs = np.linspace(0, n_seqs - 1, int(n_seqs/repeat), dtype=np.int32)
          return np.take(x, idxs, axis=1)
        n_seqs_trans = int(n_seqs/repeat)
      else:
        def downsample(x):
          return x
        n_seqs_trans = n_seqs
      first_row = sess_output["image_predictions_val"]
      first_row = np.repeat(first_row[:,0:1], first_row.shape[1], axis=1)
      plot_seqs = [downsample(first_row),
                   downsample(sess_output["transplanted_seq_predict_val"])]
      im_seqs_trans = viz_utils.generate_sequence_images(
        input_seq_list=plot_seqs,
        dataset_name=dataset_name,
        n_seqs=n_seqs_trans)
      seq_tag_trans = "transplanted_trajectory_val"
    else:
      trajectory_transplantation = False
      
    if phase == "val":
      viz_utils.generate_sequence_videos(
          sess_output[im_keys["input"]],
          sess_output[im_keys["predict"]],
           input_estimate,
          sess_output[im_keys["predict_estimate"]],
          iteration,
          base_dir,
          dataset_name=dataset_name,
          n_seqs=n_seqs,
          phase=phase)

    # Generate sprite image on first iteration
    if (iteration == 0) & (dataset_name == "moving_mnist") & (phase == "val"):
      viz_utils.generate_sprite_image(
          sess_output[im_keys["input"]],
          sess_output[im_keys["predict"]],
          base_dir)

    # if variational store plots of latent variances
    if "prior_dists_encoder" in sess_output:
        # combine encoder and decoder, extract stdDev
        # use teacher_forcing output for predictor to match ground truth actions
        latents = np.concatenate((sess_output["prior_dists_encoder"],
                                  sess_output["prior_dists_predictor_tf"]), axis=0)
        latent_dim = latents.shape
        latents = latents[..., int(latent_dim[-1]/2):]
        latents = np.exp(latents).mean(axis=tuple(range(2, len(latent_dim))))
        # offset one because inference network outputs for previous frame
        viz_utils.generate_latent_variance_plots(
            latents,
            sess_output["actions_true"][:-1, ...],
            iteration,
            base_dir,
            n_seqs,
            phase)

    logger.log_images(
      tag=seq_tag,
      images=im_seqs,
      step=iteration
    )
    if trajectory_transplantation:
      logger.log_images(
        tag=seq_tag_trans,
        images=im_seqs_trans,
        step=iteration
      )


def repeat_data_sequences(test_sequence_repeat, input_tensors):
    # repeat validation sequences to see variability
    # ensure that repeated sequences are next to each other
    extended_tensors = []
    for input_tensor in input_tensors:
        input_shape = input_tensor.get_shape().as_list()
        repeats = [1] * len(input_shape)
        repeats[1] = test_sequence_repeat
        updated_shapes = [i * r for (i, r) in zip(input_shape, repeats)]
        expanded_set = tf.expand_dims(input_tensor, -1)
        repeated_set = tf.tile(expanded_set, [1] + repeats)
        extended_tensors.append(tf.reshape(repeated_set, updated_shapes))
    return extended_tensors


def get_images(data_tuple):
  """Returns image Tensors by iterating a dataset.

  Args:
    data_tuple: A DatasetTuple for data.
  Returns:
    input_images: A tensor of the image sequence to take as input.
    predict_images: A tensor of the image sequence to predict.
  """

  im_iterator = data_tuple.dataset.make_one_shot_iterator()
  if len(data_tuple.shapes) == 3:   # dataset with action output
    input_images, predict_images, actions = im_iterator.get_next()
  elif len(data_tuple.shapes) == 2:   # dataset with action output
    input_images, predict_images, _ = im_iterator.get_next()
  else:
    raise NotImplementedError('Only support dataset with 2 or 3 outputs!')

  # Convert TFRecord datasets to TNCHW from NTCHW
  # (moving_mnist is already like this)
  if data_tuple.dataset_name != "moving_mnist":
    input_images = tf.transpose(input_images, [1, 0, 2, 3, 4])
    predict_images = tf.transpose(predict_images, [1, 0, 2, 3, 4])
  input_images.set_shape(data_tuple.shapes["input"])
  predict_images.set_shape(data_tuple.shapes["predict"])

  if len(data_tuple.shapes) == 3:
    if data_tuple.dataset_name != "moving_mnist":
        actions = tf.transpose(actions, [1, 0, 2])
    actions.set_shape(data_tuple.shapes["actions"])
  else:
    actions = None
  return input_images, predict_images, actions


# Number of rows in projection matrix. Equal to the number of high dimensional points
def build_projector(
    data_dict,
    batch_size,
    checkpoint_dir,
    dataset_name,
    embedding="latents_true"):
  """Builds a projector for visualizing embeddings."""

  if embedding == "latents_true":
    latents = tf.concat(
        [data_dict['past_latents_true'],
         data_dict['future_latents_true']], 0)
  else:
    raise ValueError("Unknown embedding type.")

  with tf.name_scope("embedding_projection"):
    # reshape latent tensor to the embedding variable shape
    latent_shape = latents.get_shape()
    n_embeddings = np.prod(latent_shape[:2])
    latents = tf.reshape(
        latents,
        [n_embeddings,
         np.prod(latent_shape[2:])])

    # assign tensor values to embedding_var
    # initialize embedding variable
    embedding_var = tf.get_variable(
        "embedding_var",
        latents.get_shape())
    tf.assign(embedding_var, latents)

  # create tsv file for one validation batch
  metadata_path = os.path.join(checkpoint_dir, "metadata.tsv")
  with open(metadata_path, "w") as metadata_file:
    metadata_file.write("Frame_id\tSequence_id\n")
    for it_frame in range(latent_shape[0]):
      for it_seq in range(batch_size):
        metadata_file.write(
            "frame " + str(it_frame) + "\t" + "sequence " + str(it_seq) + "\n")

  # create metafile that links tsv with checkpoint variable saves
  config_emb = projector.ProjectorConfig()
  embedding = config_emb.embeddings.add()
  embedding.tensor_name = embedding_var.name
  embedding.metadata_path = metadata_path
  project_writer = tf.summary.FileWriter(checkpoint_dir)

  return embedding, projector, config_emb, project_writer


def train(num_training_iterations,
          report_interval,
          validation_interval,
          reduce_learning_rate_interval,
          increase_pp_lambda_interval,
          learning_rate_reduce_criterion,
          plateau_scale_criterion,
          plateau_min_delay,
          learning_rate_init,
          reduce_learning_rate_multiplier,
          base_dir,
          dataset_config_name,
          train_batch_size,
          val_test_batch_size,
          test_sequence_repeat,
          network_config_name,
          loss_config_name,
          regularizer_weight,
          max_grad_norm,
          optimizer_epsilon,
          checkpoint_interval,
          increase_pp_lambda_multiplier,
          plateau_criterion_loss_name,
          use_variational,
          fixed_prior,
          infer_actions,
          is_test=False):
  """Run the training of the LSTM model on moving MNIST.

  Args:
    num_training_iterations: The number of batches to run during training.
    report_interval: The period (in batches) at which train stats will be
      reported.
    validation_interval: The period (in batches) at which validation will be run.
    reduce_learning_rate_interval: The period (in batches) at which the learning
      rate will be decayed.
    increase_pp_lambda_interval: The period (in batches) at which the
      predictive potential weight will be increased.
    learning_rate_reduce_criterion: The criterion used for reducing the learning
      rate. Either "scheduled" or "plateau".
    plateau_scale_criterion: The criterion for determining a plateau in the
      validation loss, as a fraction of the validation loss. Typically a small
      value in (0, 1].
    plateau_min_delay: The minimum number of batches to wait before decaying
      the learning rate, if using a plateau criterion.
    learning_rate_init: The initial learning rate.
    reduce_learning_rate_multiplier: The factor used to reduce the learning
      rate when appropriate.
    base_dir: The base_dir for model checkpoint and tensorboard output.
    dataset_config_name: The dataset configuration to use.
    train_batch_size: Batch size for training.
    val_test_batch_size: Batch size for validation.
    test_sequence_repeat: How often val/test sequences should be repeated.
    network_config_name: The network architecture configuration to use.
    loss_config_name: The loss configuration to use.
    regularizer_weight: Weight for weight decay regularizer.
    max_grad_norm: Gradient clipping norm limit.
    optimizer_epsilon: Epsilon used for Adam optimizer.
    checkpoint_interval: Checkpointing step interval.
    increase_pp_lambda_multiplier: pp_lambda is increased by this factor
      when updated.
    plateau_criterion_loss_name: A string specifying which loss to use as a
      criterion for decreasing the learning rate. When this loss plateaus,
      the learning rate will decay.
    use_variational: If True use variational RNN implementation.
  Raises:
    ValueError: If the total sequence length is too short to accomodate the
      input and predicted subsequences.
  """

  time_0 = time.time()

  network_spec = configs.get_network_specs(
    network_config_name)
  loss_spec = configs.get_loss_spec(loss_config_name)

  if test_sequence_repeat > 0:
      if val_test_batch_size % test_sequence_repeat != 0:
          raise ValueError("Test sequence repeat must evenly divide val/test batch size!")
      val_test_fetch_batch_size = int(val_test_batch_size / test_sequence_repeat)
  else:
      val_test_fetch_batch_size = val_test_batch_size

  dataset_name, data_spec_train, data_spec_val, data_spec_test = configs.get_dataset_specs(
      dataset_config_name,
      train_batch_size,
      val_test_fetch_batch_size,
      val_test_fetch_batch_size,
      FLAGS.input_seq_len,
      FLAGS.pred_seq_len,
      loss_spec)

  if is_test:
    data_spec_val_test=data_spec_test
    val_test_phase="test"
  else:
    data_spec_val_test=data_spec_val
    val_test_phase="val"
    
  train_data, _ = build_dataset(
    dataset_name,
    data_spec_train,
    train_batch_size,
    phase="train")
  val_test_data, val_data_handler = build_dataset(
      dataset_name,
      data_spec_val_test,
      val_test_fetch_batch_size,
      phase=val_test_phase)

  input_images, predict_images, actions = get_images(train_data)
  input_images_val_test, predict_images_val_test, actions_val_test = get_images(val_test_data)

  full_data_train = tf.concat(
      [input_images, predict_images],
      axis=0)
  full_data_val_test = tf.concat(
      [input_images_val_test, predict_images_val_test],
      axis=0)

  with tf.Session() as sess:
      N_ITER = 200
      n_actions = actions.get_shape().as_list()[2]
      data_shape = full_data_train.get_shape().as_list()
      errors = np.empty([data_shape[0], data_shape[1], N_ITER, n_actions])
      for i in range(N_ITER):
          print("Iteration %d" % i)
          sess_output = sess.run({"imgs": full_data_train, "actions": actions})
          images = sess_output["imgs"]
          acts = sess_output["actions"]
          for seq in range(images.shape[1]):
              for timestep in range(images.shape[0]-1):
                  img = (np.transpose(images[timestep+1, seq], (1, 2, 0)) + 1) / 2
                  gt_angles = acts[timestep, seq]
                  # print(gt_angles * 180 / np.pi)
                  angles = viz_utils.compute_angle(img)
                  for joint in range(n_actions):
                    errors[timestep, seq, i, joint] = \
                        np.abs(gt_angles[joint] - angles[joint]) * 180 / np.pi

      errors = errors.reshape([-1, n_actions])
      print(np.mean(errors, axis=0))
      errors = errors.reshape([-1])
      errors = errors[errors<40]
      import matplotlib.pyplot as plt
      plt.figure()
      plt.hist(errors)
      plt.xlabel("Angle Error [deg]")
      plt.ylabel("# of Images")
      # plt.savefig("/tmp/hist.png")
      plt.savefig("/tmp/hist.eps", format='eps', dpi=180)
      plt.show()



def main(unused_argv):
  if FLAGS.create_new_subdir:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_dir = os.path.join(os.path.expanduser(FLAGS.base_dir), timestamp)
  else:
    base_dir = os.path.expanduser(FLAGS.base_dir)

  train(
      num_training_iterations=FLAGS.num_training_iterations,
      report_interval=FLAGS.report_interval,
      validation_interval=FLAGS.validation_interval,
      reduce_learning_rate_interval=FLAGS.reduce_learning_rate_interval,
      increase_pp_lambda_interval=FLAGS.increase_pp_lambda_interval,
      learning_rate_reduce_criterion=FLAGS.learning_rate_reduce_criterion,
      plateau_scale_criterion=FLAGS.plateau_scale_criterion,
      plateau_min_delay=FLAGS.plateau_min_delay,
      learning_rate_init=FLAGS.learning_rate,
      reduce_learning_rate_multiplier=FLAGS.reduce_learning_rate_multiplier,
      base_dir=base_dir,
      dataset_config_name=FLAGS.dataset_config_name,
      train_batch_size=FLAGS.train_batch_size,
      val_test_batch_size=FLAGS.val_batch_size,
      test_sequence_repeat=FLAGS.test_sequence_repeat,
      network_config_name=FLAGS.network_config_name,
      loss_config_name=FLAGS.loss_config_name,
      regularizer_weight=FLAGS.regularizer_weight,
      max_grad_norm=FLAGS.max_grad_norm,
      optimizer_epsilon=FLAGS.optimizer_epsilon,
      checkpoint_interval=FLAGS.checkpoint_interval,
      increase_pp_lambda_multiplier=FLAGS.increase_pp_lambda_multiplier,
      plateau_criterion_loss_name=FLAGS.plateau_criterion_loss_name,
      use_variational=FLAGS.use_variational,
      fixed_prior=FLAGS.fixed_prior,
      infer_actions=FLAGS.infer_actions)


if __name__ == "__main__":
  tf.app.run(main=main)
