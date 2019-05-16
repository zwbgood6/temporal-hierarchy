"Handles all dataset related issues like loading, queuing, augmentation..."
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
import sys
import collections
import glob
import os
from recordclass import recordclass
from collections import namedtuple
from utils import AttrDict, shape

sys.path.append("..")
import configs
from specs import dataset_specs
from architectures import th_utils
from data.moving_mnist import data_handler
import utils


FLAGS = tf.flags.FLAGS


class VideoDataHandler:
  def __init__(self, loss_spec, is_test=False):
    self.is_test = is_test
    if self.is_test:
      self.val_test_batch_size = FLAGS.test_batch_size
    else:
      self.val_test_batch_size = FLAGS.val_batch_size
    # fetch less data if seqs should be repeated at validation time
    if FLAGS.test_sequence_repeat > 0:
      if self.val_test_batch_size % FLAGS.test_sequence_repeat != 0:
        raise ValueError("Test sequence repeat must evenly divide val/test batch size!")
      self.val_test_fetch_batch_size = int(self.val_test_batch_size / FLAGS.test_sequence_repeat)
    else:
      self.val_test_fetch_batch_size = self.val_test_batch_size
    self.render_fcn = None    # used in case of coord based prediction
    self.render_shape = None
    self.loss_spec = loss_spec

  def video_random_flip_left_right(self, video, seed=None):
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

  def video_augmentation(self, video, dataset_config, phase="train"):
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
            [1, ],
            minval=0,
            maxval=dataset_config.input_height - dataset_config.im_height + 1,
            dtype=tf.int32)
          offset_width = tf.random_uniform(
            [1, ],
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
          video = self.video_random_flip_left_right(video)

    return video

  def parse_bair(self, example_proto, dataset_config):
    # specify feature templates that frame numbers will be filled into
    img_shape = (dataset_config.input_res, dataset_config.input_res, dataset_config.channels,)
    feature_templates_and_shapes = {
      "images": {"name": "{}/image_aux1/encoded", "shape": img_shape, "type": tf.string},
      "actions": {"name": "{}/action", "shape": (dataset_config.num_actions,), "type": tf.float32},
      "abs_actions": {"name": "{}/endeffector_pos", "shape": (dataset_config.num_actions - 1,), "type": tf.float32}
    }

    parsed_seqs = self.parse_bair_styled_dataset(example_proto, dataset_config, 
                                                 feature_templates_and_shapes)
    return parsed_seqs["images"], parsed_seqs["actions"], parsed_seqs["abs_actions"]

  @staticmethod
  def parse_top(example_proto, dataset_config):
    # specify feature templates that frame numbers will be filled into
    img_shape = (dataset_config.input_res, dataset_config.input_res, dataset_config.channels,)
    feature_templates_and_shapes = {
      "goal_timestep": {"name": "goal_timestep", "shape": (1,), "type": tf.int64, "dim": 0},
      "images": {"name": "{}/image_view0/encoded", "shape": img_shape, "type": tf.string, "dim": 1},
      "actions": {"name": "{}/action", "shape": (dataset_config.num_actions,), "type": tf.float32, "dim": 1},
      "abs_actions": {"name": "{}/is_key_frame", "shape": (1,), "type": tf.int64, "dim": 1},
      "is_key_frame": {"name": "{}/is_key_frame", "shape": (1,), "type": tf.int64, "dim": 1}
    }

    parsed_seqs = VideoDataHandler.parse_bair_styled_dataset(example_proto, dataset_config,
                                                 feature_templates_and_shapes)
    data_tensors = AttrDict({"goal_timestep": parsed_seqs["goal_timestep"]})
    parsed_seqs["actions"] = parsed_seqs["actions"] / 2  # rescale actions
    return parsed_seqs["images"], parsed_seqs["actions"], \
            parsed_seqs["abs_actions"], parsed_seqs["is_key_frame"], data_tensors

  @staticmethod
  def parse_bair_styled_dataset(example_proto, dataset_config,
                                feature_templates_and_shapes):
    """Parses the BAIR dataset, fuses individual frames to tensors."""
    features = {}  # fill all features in feature dict
    for key, feat_params in feature_templates_and_shapes.items():
      for frame in range(dataset_config.max_seq_length):
        if feat_params["type"] == tf.string:
          feat = tf.VarLenFeature(dtype=tf.string)
        else:
          feat = tf.FixedLenFeature(dtype=feat_params["type"],
                                    shape=feat_params["shape"])
        features.update({feat_params["name"].format(frame): feat})
    parsed_features = tf.parse_single_example(example_proto, features)

    # decode frames and stack in video
    
    def process_feature(feat_params, frame):
      feat_tensor = parsed_features[feat_params["name"].format(frame)]
      if feat_params["type"] == tf.string:
        feat_tensor = tf.sparse_tensor_to_dense(feat_tensor, default_value="")
        # feat_tensor = tf.decode_raw(feat_tensor, tf.float32)
        feat_tensor = tf.cast(tf.decode_raw(feat_tensor, tf.uint8), tf.float32)
        feat_tensor = tf.reshape(feat_tensor, feat_params["shape"])
        if dataset_config.input_res != dataset_config.im_width:
          feat_tensor = tf.image.resize_images(feat_tensor, (dataset_config.im_height,
                                                             dataset_config.im_width))
      return feat_tensor
    
    parsed_seqs = {}
    for key, feat_params in feature_templates_and_shapes.items():
      if feat_params["dim"] == 0:
        feat_tensor = process_feature(feat_params, 0)
        parsed_seqs.update({key: feat_tensor})
      else:
        frames = []
        for frame in range(dataset_config.max_seq_length):
          feat_tensor = process_feature(feat_params, frame)
          frames.append(feat_tensor)
        parsed_seqs.update({key: tf.stack(frames)})

    return parsed_seqs
  
  def parse_kth_styled_dataset(self, example_proto, dataset_config, decode_actions):
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
      actions_abs = decode_raw_tensor("actions_absolute",
                                      shape=tf.stack([parsed_features["video_length"],
                                                      parsed_features["num_actions"]]),
                                      type='float32')
    else:
      actions, actions_abs = None, None
  
    video_length = parsed_features["video_length"]
    
    return video, actions, actions_abs, video_length

  def get_video_parse_function(self, dataset_config, dataset_name, phase):
    """"Returns a video parse function based on the config and phase."""
    if isinstance(dataset_config, (dataset_specs.ReacherConfig, dataset_specs.BAIRConfig)):
      decode_actions = True
    else:
      decode_actions = False

    def video_parse_function(example_proto):
      """Parses and preprocesses the features from a video dataset tfrecord."""
      if dataset_name == "bair":
        video, actions, actions_abs = self.parse_bair(example_proto, dataset_config)
        video_length = dataset_config.max_seq_length
        data_tensors = AttrDict()
      if dataset_name == "top":
        video, actions, actions_abs, is_key_frame, data_tensors = self.parse_top(example_proto, dataset_config)
        video_length = dataset_config.max_seq_length
      else:
        video, actions, actions_abs, video_length  = self.parse_kth_styled_dataset(
          example_proto, dataset_config, decode_actions)
        data_tensors = AttrDict()

      if "rescale_size" in dataset_config._asdict():
        if dataset_config.rescale_size == "0..1":
          video = video / 255
        else:
          # Rescale frames from [0, 255] to [-1, 1]
          video = video * (2 / 255) - 1

      # Sample if the video is bigger than needed
      if phase == "train" and dataset_name != "top":
        start_index = tf.random_uniform(
          [1, ],
          minval=0,
          maxval=video_length -
                  dataset_config.num_frames * FLAGS.video_length_downsample + 1,
          dtype=tf.int64)
      else:
        # this is to make sure validation and test are deterministic
        # TODO do we have enough data testing this way
        start_index = [0, ]
      
      frame_range = tf.range(start_index[0], start_index[0] + dataset_config.num_frames * FLAGS.video_length_downsample,
                             FLAGS.video_length_downsample)
      video = tf.gather(video, frame_range, axis=0)
      
      if actions is not None:
        actions = tf.gather(actions, frame_range, axis=0)
        actions_abs = tf.gather(actions_abs, frame_range, axis=0)

      if not decode_actions:
        # can only augment video data if no action annotation
        video = self.video_augmentation(video, dataset_config, phase=phase)

      if dataset_name == "ucf101" and dataset_config.channels == 1:
        # Convert videos to
        video = tf.image.rgb_to_grayscale(video)

      video.set_shape(
        [dataset_config.num_frames,
         dataset_config.im_height,
         dataset_config.im_width,
         dataset_config.channels])

      if actions is not None:
        if dataset_name == "bair":
          num_abs_actions = dataset_config.num_actions - 1
        elif dataset_name == "top":
          num_abs_actions = 1
        else:
          num_abs_actions = dataset_config.num_actions
        actions.set_shape(
          [dataset_config.num_frames,
           dataset_config.num_actions])
        actions_abs.set_shape(
          [dataset_config.num_frames,
           num_abs_actions])

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
        actions, actions_abs = tf.convert_to_tensor([]), tf.convert_to_tensor(
          [])  # map function does not support None outputs
      return input_sequence, predict_sequence, actions, actions_abs, data_tensors

    return video_parse_function

  def get_video_dataset(
      self,
      dataset_config,
      dataset_name,
      batch_size,
      num_threads=20,
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

    filenames = glob.glob(os.path.join(dataset_config.data_dir, "*.tfrecord")) \
                + glob.glob(os.path.join(dataset_config.data_dir, "*.tfrecords"))
    if phase == "train" and tf.flags.FLAGS.n_shards != 0:
      filenames = filenames[0:tf.flags.FLAGS.n_shards]
    dataset = tf.data.TFRecordDataset(filenames)

    map_fn = self.get_video_parse_function(dataset_config, dataset_name, phase)
    if phase == "train":
      dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(
      map_fn,
      num_parallel_calls=num_threads)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(output_buffer_size)

    batched_dataset = dataset.batch(batch_size, drop_remainder=True) # dropping here is fine since we repeat

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

    if isinstance(dataset_config, (dataset_specs.ReacherConfig, dataset_specs.BAIRConfig, dataset_specs.TOPConfig)):
      if dataset_name == "bair":
        num_abs_actions = dataset_config.num_actions - 1
      elif dataset_name == "top":
        num_abs_actions = 1
      else:
        num_abs_actions = dataset_config.num_actions
      dataset_shapes.update({
        "actions": [dataset_config.input_seq_len + dataset_config.pred_seq_len,
                    batch_size,
                    dataset_config.num_actions],
        "actions_abs": [dataset_config.input_seq_len + dataset_config.pred_seq_len,
                        batch_size,
                        num_abs_actions]
      })

    dataset_size = dataset_config.n_examples

    return batched_dataset, dataset_shapes, dataset_size

  def get_batch(self,
                data,
                batch_size,
                seq_len_total,
                input_seq_len,
                pred_seq_len,
                im_height,
                im_width,
                channels,
                image_input):
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
    batch_data, action_label, abs_action_label = data.GetBatch()

    # Split out sequence, channel, and spatial dimensions
    if image_input:
      batch_data = batch_data.reshape(
        [batch_size, seq_len_total, channels, im_height, im_width])
    else:
      # for non-image data output shape is stored in im_height
      batch_data = batch_data.reshape([batch_size, seq_len_total] + im_height)

    # Convert to time major
    batch_data = batch_data.transpose([1, 0] + list(range(len(batch_data.shape)))[2:])
    input_seqs = batch_data[:input_seq_len, ...]
    pred_seqs = batch_data[input_seq_len:(input_seq_len + pred_seq_len), ...]

    return input_seqs, pred_seqs, action_label, abs_action_label

  def build_batch_generator(self,
                            config,
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
    # TODO: make data loading with coordinate obs less hacky by implementing getOutputSize in data_handler
    image_input = True
    if isinstance(config, dataset_specs.MovingMNISTConfig):
      handler = data_handler.ChooseDataHandler(config)
    elif isinstance(config, dataset_specs.GridworldConfig):
      from data.gridworld.gridworld_data_handler import GridWorldDataHandler
      handler = GridWorldDataHandler(config)
    elif isinstance(config, dataset_specs.BouncingBallsConfig):
      from data.bouncing_balls.bouncing_balls_data_handler import BouncingBallsDataHandler
      handler = BouncingBallsDataHandler(config)
      image_input = config.image_input    # this dataset has potential to have non-img observations
      self.render_fcn = handler.render_fcn
      self.render_shape = handler.render_shape
    else:
      raise ValueError("Config type is not supported for on-the-fly data generation.")

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
        input_seqs, pred_seqs, action_seqs, abs_action_seqs = self.get_batch(
          handler,
          batch_size,
          seq_len_total,
          input_seq_len,
          pred_seq_len,
          im_height,
          im_width,
          config.channels,
          image_input)
        data_tensors = AttrDict()
        yield (input_seqs, pred_seqs, action_seqs, abs_action_seqs, data_tensors)

    if image_input:
      obs_shape = [config.channels, im_height, im_width]
    else:
      obs_shape = im_height   # for coordinate observations the shape is stored in the im_height
    dataset_shapes = {
      "input": [input_seq_len, batch_size] + obs_shape,
      "predict": [pred_seq_len, batch_size] + obs_shape,
      "actions": [seq_len_total, batch_size, config.num_actions],
      "actions_abs": [seq_len_total, batch_size, config.num_actions]
    }
    return batch_generator, dataset_shapes, dataset_size, handler

  def build_dataset(
      self,
      dataset_name,
      dataset_spec,
      batch_size,
      phase):
    """Builds a dataset for training/validation.

    Args:
      dataset_name: The name of the dataset to use.
      dataset_spec: The spec of the dataset to use. This may be phase dependent.
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
    handler = None  # set default value
    if dataset_name == "moving_mnist" \
        or dataset_name == "gridworld" \
        or dataset_name == "bouncing_balls":
      get_batch, dataset_shapes, dataset_size, handler = self.build_batch_generator(
        dataset_spec, batch_size)

      dataset = tf.data.Dataset.from_generator(
        get_batch,
        (tf.float32, tf.float32, tf.float32, tf.float32, {}),
        (tf.TensorShape(dataset_shapes["input"]),
         tf.TensorShape(dataset_shapes["predict"]),
         tf.TensorShape(dataset_shapes["actions"]),
         tf.TensorShape(dataset_shapes["actions_abs"]),
         {}))
    elif dataset_name == "kth" or \
            dataset_name == "h36" or \
            dataset_name == "ucf101" or \
            dataset_name == "xreacher" or \
            dataset_name == "bair" or \
            dataset_name == "top":

      dataset, dataset_shapes, dataset_size = self.get_video_dataset(
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

  def get_images(self, data_tuple):
    """Returns image Tensors by iterating a dataset.

    Args:
      data_tuple: A DatasetTuple for data.
    Returns:
      input_images: A tensor of the image sequence to take as input.
      predict_images: A tensor of the image sequence to predict.
    """

    im_iterator = data_tuple.dataset.make_one_shot_iterator()
    if len(data_tuple.shapes) == 4:  # dataset with action output
      input_images, predict_images, actions, actions_abs, data_tensors = im_iterator.get_next()
    elif len(data_tuple.shapes) == 2:  # dataset with action output
      input_images, predict_images, _, _, data_tensors = im_iterator.get_next()
    else:
      raise NotImplementedError('Only support dataset with 2 or 3 outputs!')

    # Convert TFRecord datasets to TNCHW from NTCHW
    # (moving_mnist is already like this)
    is_TNCHW = data_tuple.dataset_name in ["moving_mnist", "gridworld", "bouncing_balls"]
    if not is_TNCHW:
      input_images = tf.transpose(input_images, [1, 0, 2, 3, 4])
      predict_images = tf.transpose(predict_images, [1, 0, 2, 3, 4])
    input_images.set_shape(data_tuple.shapes["input"])
    predict_images.set_shape(data_tuple.shapes["predict"])

    if len(data_tuple.shapes) == 4:
      if not is_TNCHW:
        actions = tf.transpose(actions, [1, 0, 2])
        actions_abs = tf.transpose(actions_abs, [1, 0, 2])
      actions.set_shape(data_tuple.shapes["actions"])
      actions_abs.set_shape(data_tuple.shapes["actions_abs"])
    else:
      actions, actions_abs = tf.convert_to_tensor([]), tf.convert_to_tensor([])
    
    for key, tensor in data_tensors.items():
      data_tensors[key] = utils.tf_swapaxes(tensor, 0, 1)

    data_tensors = AttrDict(data_tensors)
    data_tensors.input_images = input_images
    data_tensors.predict_images = predict_images
    data_tensors.actions = actions
    data_tensors.actions_abs = actions_abs
    return data_tensors

  def repeat_data_sequences(self, test_sequence_repeat, input_tensor):
    # repeat validation sequences to see variability
    # ensure that repeated sequences are next to each other
    input_shape = shape(input_tensor)
    if input_shape[0] != 0:
      repeats = [1] * len(input_shape)
      repeats[1] = test_sequence_repeat
      updated_shapes = [i * r for (i, r) in zip(input_shape, repeats)]
      expanded_set = tf.expand_dims(input_tensor, -1)
      repeated_set = tf.tile(expanded_set, [1] + repeats)
      return tf.reshape(repeated_set, updated_shapes)
    else:
      return input_tensor

  def create_multi_dataset(self, data_spec, batch_size, phase):
    def gen_dataset(spec, bs):
      data, handler = self.build_dataset(
        self.dataset_name,
        spec,
        bs,
        phase=phase)
      data_tensors = self.get_images(data)
      return data, handler, data_tensors

    def batch_concat_tuple(tensor_dicts_list, batch_dim=1):
      keys = tensor_dicts_list[0].keys()
      values = map(lambda d: d.values(), tensor_dicts_list)
      output = zip(*values)
      output = [tf.concat(o, axis=batch_dim) for o in output]
      return dict(zip(keys, output))

    if isinstance(data_spec, dataset_specs.TOPConfig) and isinstance(data_spec.data_dir, list):
      data_dirs = data_spec.data_dir
      tensor_dicts_list = []
      for data_dir in data_dirs:
        sub_spec = data_spec; sub_spec.data_dir = data_dir
        data, handler, tensors = gen_dataset(sub_spec, int(batch_size / len(data_dirs)))
        tensor_dicts_list.append(tensors)
      return data, handler, batch_concat_tuple(tensor_dicts_list)
    else:
      return gen_dataset(data_spec, batch_size)


  def fetch_data(self, is_hierarchical):
    if is_hierarchical:
        self._pred_img_fetch_len = th_utils.get_future_input_length()
    else:
        self._rollout_len = FLAGS.pred_seq_len
        self._pred_img_fetch_len = FLAGS.pred_seq_len

    self.dataset_name, self.data_spec_train, data_spec_val, data_spec_test = configs.get_dataset_specs(
      FLAGS.dataset_config_name,
      FLAGS.train_batch_size,
      self.val_test_fetch_batch_size,
      self.val_test_fetch_batch_size,
      input_seq_len=FLAGS.input_seq_len,
      pred_seq_len=self._pred_img_fetch_len,
      loss_spec=self.loss_spec,
      img_res=FLAGS.input_img_res)

    if self.is_test:
      self.data_spec_val_test = data_spec_test
      val_test_phase = "test"
    else:
      self.data_spec_val_test = data_spec_val
      val_test_phase = "val"

    self.train_data, _, train_data_tensors = self.create_multi_dataset(self.data_spec_train,
                                                                       FLAGS.train_batch_size,
                                                                       phase="train")
    self.val_test_data, self.val_data_handler, val_test_data_tensors = self.create_multi_dataset(self.data_spec_val_test,
                                                                       self.val_test_fetch_batch_size,
                                                                       phase=val_test_phase)
    train_data_tensors = AttrDict(train_data_tensors)
    val_test_data_tensors = AttrDict(val_test_data_tensors)
    if FLAGS.test_sequence_repeat > 0:
      val_test_data_tensors = AttrDict(utils.map_dict(
        lambda value: self.repeat_data_sequences(FLAGS.test_sequence_repeat, value), val_test_data_tensors))

    if not self.is_test:
      return train_data_tensors, val_test_data_tensors
    else:
      return val_test_data_tensors

  def maybe_turnoff_randomness(self):
    if isinstance(self.val_data_handler, data_handler.BouncingMNISTDataHandler):
      self.val_data_handler.TurnOffRandomness()

  def maybe_turnon_randomness(self):
    if isinstance(self.val_data_handler, data_handler.BouncingMNISTDataHandler):
      self.val_data_handler.TurnOnRandomness()

  def get_num_actions(self):
    try:
      return self.data_spec_train.num_actions
    except:
      return  None

  def get_input_len(self, phase):
    if phase == "train":
      return self.data_spec_train.input_seq_len
    else:
      return self.data_spec_val_test.input_seq_len

  def get_channels(self):
    return self.data_spec_train.channels

  def get_input_img_shape(self):
    return self.train_data[2]["input"][2:]

  def get_dataset_name(self):
    return self.dataset_name

  def get_dataset_spec(self, phase):
    if phase == "train":
      return self.data_spec_train
    else:
      return self.data_spec_val_test

  def get_dataset_size(self, phase):
    if phase == "train":
      return self.train_data.dataset_size
    else:
      return self.val_test_data.dataset_size

  def get_fetched_batch_size(self, phase):
    if phase == "train":
      return FLAGS.train_batch_size
    else:
      return self.val_test_fetch_batch_size

  def get_render_fcn(self):
    return self.render_fcn

  def get_render_shape(self):
    return self.render_shape
