"""Utilities to convert KTH data to tfrecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import glob
import os
import random

import data.utils.video_utils

random.seed(a=123)

KTHVideoTuple = collections.namedtuple(
    "KTHVideoTuple",
    "filename person_id action_class video_type")

KTH_ACTION_CLASSES = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "walking": 4,
    "running": 5}

KTH_PERSON_IDS_TRAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
KTH_PERSON_IDS_TEST = [17, 18, 19, 20, 21, 22, 23, 24, 25]
KTH_VIDEO_TYPES = {"d1": 0, "d2": 1, "d3": 2, "d4": 3}
# Images are downsampled to this size
KTH_FRAME_OUTPUT_DIMS = data.utils.video_utils.FrameDims(height=128, width=160)
KTH_TEST_EXAMPLE_LENGTH = None  # Test examples always use this length

KTH_NON_VIDEO_FEATURES =  {
    "person_id": "int64",
    "action_class": "int64",
    "video_type": "int64"}

def _kth_rgb_list_to_grayscale(frame_list):
  """Chops a list of RGB frames down to grayscale."""
  # All frames are the same, so this is OK
  return [frame[..., 0] for frame in frame_list]

def parse_kth_filename(filename):
  """Parses the KTH filename for person ID, action class, and video type."""

  # Format is /blah/personXX_ACTION_VIDTYPE.ext
  kth_filename = os.path.splitext(os.path.basename(filename))[0]
  person_id, action_class, video_type, _ = kth_filename.split("_")
  person_id = int(person_id[-2:]) - 1  # 0-index
  action_class = KTH_ACTION_CLASSES[action_class]
  video_type = KTH_VIDEO_TYPES[video_type]

  return person_id, action_class, video_type


def build_all_shards(
    all_video_tuples,
    tfrecord_dir,
    train_example_length,
    n_val_examples,
    shuffle_examples=True):
  """Builds shards with randomly ordered sequences."""

  # Grab files for each person and split into train, test file lists
  video_tuples_train = [
      vt for vt in all_video_tuples if vt.person_id in KTH_PERSON_IDS_TRAIN]
  video_tuples_test = [
      vt for vt in all_video_tuples if vt.person_id in KTH_PERSON_IDS_TEST]

  n_shards_train = len(KTH_PERSON_IDS_TRAIN)
  n_shards_test = len(KTH_PERSON_IDS_TEST)
  n_shards_val = 1

  out_dir = data.utils.video_utils.make_timestamp_dir(tfrecord_dir)

  # Grabs the frames (a list of video data arrays) and the read order
  # a list of tuples of (video_id, start_frame, end_frame)
  frame_data_train, read_order_train = data.utils.video_utils.get_video_data(
      video_tuples_train,
      example_length=train_example_length,
      shuffle_examples=shuffle_examples,
      frame_output_dims=KTH_FRAME_OUTPUT_DIMS)

  frame_data_train = _kth_rgb_list_to_grayscale(frame_data_train)

  # Build the tfrecords for each:
  data.utils.video_utils.build_phase_shards(
      frame_data=frame_data_train,
      video_tuples=video_tuples_train,
      read_order=read_order_train,
      base_dir=out_dir,
      n_shards=n_shards_train,
      phase="train",
      non_video_features=KTH_NON_VIDEO_FEATURES)

  print("Training shards written.")

  # Do the same for test
  # Grab:
  # person20_walking_d1_uncomp.avi, 553
  # person25_running_d4_uncomp.avi, 14
  # test_files = ["person20_walking_d1_uncomp.avi", "person25_running_d4_uncomp.avi"]
  # video_tuples_test = [vt for vt in video_tuples_test if os.path.basename(vt.filename) in test_files]
  frame_data_test, read_order_test = data.utils.video_utils.get_video_data(
      video_tuples_test,
      example_length=KTH_TEST_EXAMPLE_LENGTH,
      shuffle_examples=False,
      frame_output_dims=KTH_FRAME_OUTPUT_DIMS)

  frame_data_test = _kth_rgb_list_to_grayscale(frame_data_test)

  # TODO(drewjaegle): remove this!!!
  # SEQ_LEN_TMP = 30
  # # 0: person25_running_d4_uncomp.avi - frame 553
  # read_order_test[0] = video_utils.ReadOrderTuple(
  #     video_id=0, start_frame=14, end_frame=14 + SEQ_LEN_TMP)
  # # 1: person20_walking_d1_uncomp.avi - frame 14
  # read_order_test[1] = video_utils.ReadOrderTuple(
  #     video_id=1, start_frame=553, end_frame=553 + SEQ_LEN_TMP)
  # n_shards_test = 1

  data.utils.video_utils.build_phase_shards(
      frame_data=frame_data_test,
      video_tuples=video_tuples_test,
      read_order=read_order_test,
      base_dir=out_dir,
      n_shards=n_shards_test,
      phase="test",
      non_video_features=KTH_NON_VIDEO_FEATURES)

  # Additionally, build a tfrecord for a small visualization/validation set
  read_order_val = read_order_test[:n_val_examples]
  data.utils.video_utils.build_phase_shards(
      frame_data=frame_data_test,
      video_tuples=video_tuples_test,
      read_order=read_order_val,
      base_dir=out_dir,
      n_shards=n_shards_val,
      phase="val",
      non_video_features=KTH_NON_VIDEO_FEATURES)

  print("Test and validation shards written.")


def build_dataset(
    dataset_dir,
    tfrecord_dir,
    extension="avi",
    person_ids=None,
    train_example_length=20,
    shuffle_examples=False,
    n_val_examples=1000):
  """Builds the full set of tfrecords for the KTH dataset.

  Args:
    dataset_dir: The directory containing the KTH dataset videos.
    tfrecord_dir: The directory to write the KTH dataset tfrecords.
    extension: The extension of video files. Defaults to "avi".
    person_ids: An optional list of person IDs of the KTH dataset to include.
      If None, videos of all person IDs will be included. Defaults to None.
    shuffle_examples: If True, shuffles all examples to random shards.
      Otherwise, reads avi files out serially to a shard.
  """
  all_filenames = glob.glob(
      os.path.join(dataset_dir, "*.{}".format(extension)))

  all_video_tuples = []
  for filename in all_filenames:
    person_id, action_class, video_type = parse_kth_filename(filename)

    all_video_tuples.append(
        KTHVideoTuple(filename=filename,
                      person_id=person_id,
                      action_class=action_class,
                      video_type=video_type))

  if person_ids is not None:
    all_video_tuples = [
        vt for vt in all_video_tuples if vt.person_id in person_ids]

  build_all_shards(
      all_video_tuples,
      tfrecord_dir,
      train_example_length=train_example_length,
      n_val_examples=n_val_examples,
      shuffle_examples=shuffle_examples)


if __name__ == "__main__":
  # Build full dataset
  kth_base_dir = "/NAS/data/KTH_action/"
  dataset_dir = os.path.join(kth_base_dir, "avi")
  tfrecord_dir = os.path.join(kth_base_dir, "tfrecords")
  person_ids = None  # None for full dataset
  train_example_length = None  # None to write whole videos as examples
  shuffle_examples = True  # Randomize order of sequence in shards
  n_val_examples = 1000

  build_dataset(
      dataset_dir,
      tfrecord_dir,
      person_ids=person_ids,
      train_example_length=train_example_length,
      n_val_examples=n_val_examples,
      shuffle_examples=shuffle_examples)
