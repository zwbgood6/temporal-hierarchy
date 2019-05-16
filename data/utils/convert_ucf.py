"""Convert UCF data from t7 to tfrecords."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import cv2
import glob
import numpy as np
import os
import random

UCF_FILE_SIZE = (256, 256)

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# For from scratch parsing: read in the file list from the CSV folder, using
# trainlist01.csv and testlist01.csv. These contain filenames and class labels.



# TODO(drewjaegle): code for reading from avi
# See extractMovieFrames.lua in the compositional_motion directory.
# Generalize the avi loading from convert_kth into a generic method.
# Read the CSV files trainlist01 and testlist01, classlist for class value.
# Then shuffle the order from this
# Save out as below.

def write_example(video, video_class, writer):
  """Writes a single example to a shard."""
  video_raw = video.tostring()

  feature = {}
  feature["video"] = _bytes_feature(video_raw)
  feature["class"] = _int64_feature(video_class)
  feature["video_length"] = _int64_feature(video.shape[0])
  feature["video_height"] = _int64_feature(video.shape[1])
  feature["video_width"] = _int64_feature(video.shape[2])
  if video.ndim == 4:
    video_channels = video.shape[3]
  else:
    video_channels = 1
  feature["video_channels"] = _int64_feature(video_channels)

  example = tf.train.Example(features=tf.train.Features(feature=feature))
  writer.write(example.SerializeToString())

def get_new_shard_writer(shard_dir, shard_idx, shard_writer):
  if shard_writer is not None:
    shard_writer.close()

  shard_path = os.path.join(
      shard_dir,
      "{:05d}.tfrecord".format(shard_idx))
  print("Starting new shard: {}".format(shard_path))
  shard_writer = tf.python_io.TFRecordWriter(shard_path)
  return shard_writer


def resize_all_frames(video_in, h_new, w_new):
  """Resizes all frames of a video formatted as NHWC or NHW.

  Input:
    video_in: The video to resize.
    h_new: The target height of the resized video.
    w_new: The target width of the resized video.
  Returns:
    video_out: The resized video.
  Raises:
    ValueError if video is not 3 or 4 dimensional.
  """
  if video_in.ndim == 3:
    video_out = np.zeros(
        (video_in.shape[0], h_new, w_new),
        dtype=video_in.dtype)
  elif video_in.ndim == 4:
    video_out = np.zeros(
        (video_in.shape[0], h_new,
         w_new, video_in.shape[3]),
        dtype=video_in.dtype)
  else:
    raise ValueError("Video must be 3 (NHW) or 4 (NHWC) dimensional.")

  for frame_idx in range(video_in.shape[0]):
    video_out[frame_idx, ...] = cv2.resize(
        video_in[frame_idx, ...], (h_new, w_new))

  return video_out

def make_ucf_tfrecords(pkl_dir, tfrecord_dir):
  """Converts UCF pickled files to tfrecords."""
  pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))
  random.shuffle(pkl_files)  # To randomize tfrecord order

  test_dir = os.path.join(tfrecord_dir, "test")
  train_dir = os.path.join(tfrecord_dir, "train")

  if not(os.path.exists(test_dir)):
    os.makedirs(test_dir)
  if not(os.path.exists(train_dir)):
    os.makedirs(train_dir)

  # For online validation, we'll just reuse a fixed subset of the test files,
  # using fixed indices for each (may as well just using first 20 frames of each)

  n_ex_per_tfrecord = 100  # videos per tfrecord file
  train_idx = 0
  test_idx = 0
  train_shard_idx = 0
  test_shard_idx = 0
  train_writer = None
  test_writer = None
  n_files = len(pkl_files)

  for idx, file_i in enumerate(pkl_files):
    print("Processing file {} of {}: {}".format(idx, n_files, file_i))
    with open(file_i, "rb") as file_in:
      data_in = pickle.load(file_in)

    video_class = data_in["class"] - 1  # Change to 0 index

    video_in = data_in["frames"]
    # Convert images from [0, 1] (in pkl files) to [0, 255], save as uint8
    video_in = resize_all_frames(video_in, UCF_FILE_SIZE[0], UCF_FILE_SIZE[1])
    video_in = (video_in * 255).astype(np.uint8)

    # Use first train split
    if data_in["train"][0]:
      if train_idx % n_ex_per_tfrecord == 0:
        train_writer = get_new_shard_writer(
            train_dir,
            train_shard_idx,
            train_writer)
        train_shard_idx += 1
      writer = train_writer
      train_idx += 1
    else:
      if test_idx % n_ex_per_tfrecord == 0:
        test_writer = get_new_shard_writer(
            test_dir,
            test_shard_idx,
            test_writer)
        test_shard_idx += 1
      writer = test_writer
      test_idx += 1

    write_example(video_in, video_class, writer)

  # Close both train and test writers
  if train_writer is not None:
    train_writer.close()
  if test_writer is not None:
    test_writer.close()

def convert_t7(t7_dir, pkl_dir):
  """Converts UCF t7 videos to pickled files."""
  t7_files = glob.glob(os.path.join(t7_dir, "*.t7"))

  if not(os.path.exists(pkl_dir)):
    os.makedirs(pkl_dir)

  TENSOR_KEY = "frames"
  n_files = len(t7_files)

  for idx, file_i in enumerate(t7_files):
    print("Processing file {}: {} of {}.".format(file_i, idx, n_files))
    # Strip the filename
    filename_raw = os.path.splitext(os.path.basename(file_i))[0]
    # Create the output_file name
    output_filename = os.path.join(pkl_dir, "{}.pkl".format(filename_raw))

    # Load the raw data
    input_data = load_lua(file_i)
    output_data = {}

    for key_j in input_data.keys():
      if key_j == TENSOR_KEY:
        output_data[key_j] = input_data[key_j].numpy()
      else:
        output_data[key_j] = input_data[key_j]

    with open(output_filename, "wb") as out_file:
      pickle.dump(output_data, out_file)


if __name__ == "__main__":
  UCF_DIR = "/NAS/data/UCF101Dataset"
  T7_DIR = os.path.join(UCF_DIR, "torch/frames")
  PKL_DIR = os.path.join(UCF_DIR, "pkl")
  TFRECORD_DIR = os.path.join(UCF_DIR, "tfrecord")

  convert_t7_to_pkl = False  # Requires pytorch
  convert_pkl_to_tfrecord = True  # Requires tensorflow

  if convert_t7_to_pkl:
    from torch.utils.serialization import load_lua
    convert_t7(T7_DIR, PKL_DIR)
  if convert_pkl_to_tfrecord:
    import tensorflow as tf
    make_ucf_tfrecords(PKL_DIR, TFRECORD_DIR)
