"""Tools for parsing generic video datasets (KTH, KITTI, UCF, etc.)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import tensorflow as tf


def get_parse_function(config):
  """Returns a parse function based on the config.

  Returns only video-related features, ignoring other features, such as class.
  Assumes video is stored as uint8, returns video as float32 in [-1, 1].
  """

  def video_parse_function(example_proto):
    """Parses and preprocesses the features from a video tfrecord."""

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

    parsed_features = tf.parse_single_example(example_proto, features)

    video = tf.sparse_tensor_to_dense(parsed_features["video"], default_value="")
    video = tf.cast(tf.decode_raw(video, tf.uint8),
                    tf.float32)

    # Rescale video from [0, 255] to [-1, 1]
    video = tf.reshape(video,
        tf.stack([parsed_features["video_length"], parsed_features["video_height"],
         parsed_features["video_width"], parsed_features["video_channels"]]))
    video = video * (2 / 255) - 1


    start_index = tf.random_uniform(
        1,
        minval=0,
        maxval=parsed_features["video_length"] - config.num_frames + 1,
        dtype=tf.int64)
    import pdb; pdb.set_trace()  # Make sure OK
    video = tf.gather(video, start_index, config.num_frames)

    import pdb; pdb.set_trace()  # Make sure this works
    # start_index = 0
    # video = video[start_index:start_index + config.num_frames, ...]

    # TODO(drewjaegle): do we need these config fields?
    # Otherwise, we need to ensure the tensor values match the config values.
    video.set_shape(
        [config.num_frames,
         config.im_height,
         config.im_width,
         config.im_channels])

    # Reshape to NCHW from NHWC
    video = tf.transpose(video, [0, 3, 1, 2])

    # TODO(drewjaegle): Allow resampling of images here

    # Split images to input (10 frames) and predict (10 frames)
    input_sequence = video[:config.input_seq_len, ...]
    predict_sequence = video[config.input_seq_len:, ...]

    # TODO(drewjaegle): Do any preprocessing needed (i.e. downsample to 64)
    return input_sequence, predict_sequence

  return video_parse_function


def get_video_dataset(
    config,
    batch_size,
    num_threads=8,
    output_buffer_size=None,
    shuffle_buffer_size=None,
    phase="train"):
  """Returns the video dataset associated with a config.

  See goo.gl/zx4Gq9 for advice on setting buffer sizes.

  Args:
    config: A Config namedtuple for the current session.
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

  filenames = glob.glob(os.path.join(config.data_dir, "*.tfrecord"))

  map_fn = get_parse_function(config)
  dataset = tf.data.TFRecordDataset(filenames)

  if phase == "train":
    dataset = dataset.shuffle(shuffle_buffer_size)

  dataset = dataset.map(
      map_fn,
      num_parallel_calls=num_threads)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(output_buffer_size)

  batched_dataset = dataset.batch(batch_size)

  dataset_shapes = {
    "input": [config.input_seq_len,
              batch_size,
              1,
              config.im_height,
              config.im_width],
    "predict": [config.input_seq_len,
                batch_size,
                1,
                config.im_height,
                config.im_width]
  }

  dataset_size = config.n_examples

  return batched_dataset, dataset_shapes, dataset_size


# See the elif dataset_name == "kth" block in train_simple_rnn for the
# standard interface.

# Each should be required to supply its own features?
