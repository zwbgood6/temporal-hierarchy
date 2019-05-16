"""Utilities to convert UCF data to tfrecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import random

import data.utils.video_utils

random.seed(a=123)

UCFVideoTuple = collections.namedtuple(
    "UCFVideoTuple",
    "filename action_class")

# Original is 320 x 240, so keep at 320 x 256 for augmentation
UCF_FRAME_OUTPUT_DIMS = data.utils.video_utils.FrameDims(height=256, width=320)
UCF_TEST_EXAMPLE_LENGTH = 20  # Test examples always use this length

UCF_NON_VIDEO_FEATURES =  {"action_class": "int64",}
UCF_CLASSLIST_CSV = "classlist.csv"
UCF_TRAIN_CSV = "trainlist01.csv"
UCF_TEST_CSV = "testlist01.csv"

UCF_N_SHARDS_TRAIN = 50
UCF_N_SHARDS_TEST = 20
UCF_N_SHARDS_VAL = 1


def _parse_ucf_filename_classes(classlist_file):
  """Parses the filenames and classes of the UCF dataset."""
  filename_classes = {}
  with open(classlist_file, "rb") as csv_file:
    split_reader = csv.reader(csv_file, delimiter=",")
    for row in split_reader:
      filename_classes[row[0]] = row[1]

  return filename_classes


def _parse_ucf_split(split_file, filename_classes, video_dir):
  """Parses the video tuples of a UCF split."""
  split_filenames = []
  with open(split_file, "rb") as csv_file:
    split_reader = csv.reader(csv_file)
    for split_entry in split_reader:
      if not isinstance(split_entry, basestring):
        split_entry = split_entry[0]

      split_filenames.append(split_entry)

  # And write out with the class associated with each entry
  split_video_tuples = [
      UCFVideoTuple(
          filename=os.path.join(video_dir, filename_i),
          action_class=int(filename_classes[filename_i]) - 1)  # 0-indexed
      for filename_i in split_filenames]
  return split_video_tuples


def build_ucf_dataset(
    video_dir,
    csv_dir,
    tfrecord_dir,
    train_example_length,
    n_val_examples,
    shuffle_examples=True,
    preload_video=False):
  """Builds shards with randomly ordered sequences."""

  filename_classes = _parse_ucf_filename_classes(
      os.path.join(csv_dir, UCF_CLASSLIST_CSV))
  video_tuples_train = _parse_ucf_split(
      os.path.join(csv_dir, UCF_TRAIN_CSV),
      filename_classes,
      video_dir)
  video_tuples_test = _parse_ucf_split(
      os.path.join(csv_dir, UCF_TEST_CSV),
      filename_classes,
      video_dir)

  out_dir = data.utils.video_utils.make_timestamp_dir(tfrecord_dir)

  # Grabs the frames (a list of video data arrays) and the read order
  # a list of tuples of (video_id, start_frame, end_frame)
  frame_data_train, read_order_train = data.utils.video_utils.get_video_data(
      video_tuples_train,
      example_length=train_example_length,
      shuffle_examples=shuffle_examples,
      frame_output_dims=UCF_FRAME_OUTPUT_DIMS,
      keep_frames=preload_video)

  # Build the tfrecords for each:
  data.utils.video_utils.build_phase_shards(
      frame_data=frame_data_train,
      video_tuples=video_tuples_train,
      read_order=read_order_train,
      base_dir=out_dir,
      n_shards=UCF_N_SHARDS_TRAIN,
      phase="train",
      non_video_features=UCF_NON_VIDEO_FEATURES,
      frame_output_dims=UCF_FRAME_OUTPUT_DIMS)

  print("Training shards written.")

  # Do the same for test
  frame_data_test, read_order_test = data.utils.video_utils.get_video_data(
      video_tuples_test,
      example_length=train_example_length,
      shuffle_examples=True,
      frame_output_dims=UCF_FRAME_OUTPUT_DIMS,
      keep_frames=preload_video)

  # This should not be chunked - it will be way too large
  data.utils.video_utils.build_phase_shards(
      frame_data=frame_data_test,
      video_tuples=video_tuples_test,
      read_order=read_order_test,
      base_dir=out_dir,
      n_shards=UCF_N_SHARDS_TEST,
      phase="test",
      non_video_features=UCF_NON_VIDEO_FEATURES,
      frame_output_dims=UCF_FRAME_OUTPUT_DIMS)

  # Additionally, build a tfrecord for a small visualization/validation set
  # This one should be chunked
  frame_data_val, read_order_val = data.utils.video_utils.get_video_data(
      video_tuples_test,
      example_length=UCF_TEST_EXAMPLE_LENGTH,
      shuffle_examples=True,
      frame_output_dims=UCF_FRAME_OUTPUT_DIMS,
      keep_frames=preload_video)

  read_order_val = read_order_val[:n_val_examples]
  data.utils.video_utils.build_phase_shards(
      frame_data=frame_data_val,
      video_tuples=video_tuples_test,
      read_order=read_order_val,
      base_dir=out_dir,
      n_shards=UCF_N_SHARDS_VAL,
      phase="val",
      non_video_features=UCF_NON_VIDEO_FEATURES,
      frame_output_dims=UCF_FRAME_OUTPUT_DIMS)

  print("Test and validation shards written.")


if __name__ == "__main__":
  # Build full dataset
  ucf_base_dir = "/NAS/data/UCF101Dataset/"
  video_dir = os.path.join(ucf_base_dir, "avi")
  csv_dir = os.path.join(ucf_base_dir, "csv")
  tfrecord_dir = os.path.join(ucf_base_dir, "tfrecord")
  train_example_length = None  # None to write whole videos as examples
  shuffle_examples = True  # Randomize order of sequence in shards
  n_val_examples = 1000
  preload_video = False  # Too much data - load at write time

  build_ucf_dataset(
      video_dir,
      csv_dir,
      tfrecord_dir,
      train_example_length=train_example_length,
      n_val_examples=n_val_examples,
      shuffle_examples=shuffle_examples,
      preload_video=preload_video)
