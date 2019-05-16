"""Utilities for video dataset creation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import cv2
import datetime
import glob
import numpy as np
import os
import random
import tensorflow as tf
import tqdm
from multiprocessing.pool import Pool, ThreadPool

FrameDims = collections.namedtuple(
    "FrameDims",
    "height width")

ReadOrderTuple = collections.namedtuple(
    "ReadOrderTuple",
    "video_id start_frame end_frame")


def make_timestamp_dir(base_dir):
  timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  out_dir = os.path.join(base_dir, timestamp)
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  return out_dir


def _dims_match(frame_in, frame_output_dims):
  """Checks if the height and width of a frame match the target."""
  return (frame_in.shape[0] == frame_output_dims.height and
          frame_in.shape[1] == frame_output_dims.width)

def standardize_frame(frame_in, frame_output_dims):
  """Standardizes input frame width and height, and removes dummy channels.

  Args:
    frame_in: A video frame.
    frame_output_dims: A namedtuple with the target height and width
      for the frame.
  """
  if _dims_match(frame_in, frame_output_dims):
    frame_out = frame_in
  else:
    # Resize to the canonical width and height
    # NB: OpenCV takes new size as (X, Y), not (Y, X)!!!
    frame_out = cv2.resize(
        frame_in.astype(np.float32),
        (frame_output_dims.width, frame_output_dims.height)).astype(
            frame_in.dtype)

  if frame_out.shape[-1] == 1:
    # Chop off redundant dimensions
    frame_out = frame_out[..., 0]
  elif frame_out.shape[-1] == 3:
    # Convert OpenCV's BGR to RGB
    frame_out = frame_out[..., ::-1]
  else:
    raise ValueError("Unexpected frame shape!")

  return frame_out


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_video_capture(path):
  assert os.path.isfile(path)
  cap = None
  if path:
    cap = cv2.VideoCapture(path)
  return cap


def get_next_frame(cap):
  ret, frame = cap.read()
  if not ret:
    return None
  return np.asarray(frame)


def read_video(
    video_path,
    frame_output_dims,
    downsample=1,
    keep_frames=True):
  """Reads a video from a file and returns as an array."""

  all_frames = []
  # Open video
  cap = get_video_capture(video_path)
  frame_n = 0

  while(cap.isOpened()):
    frame = get_next_frame(cap)
    if frame is not None:
      frame_n += 1
      if keep_frames and frame_n % downsample == 0:  # Preprocess the frames
        frame = standardize_frame(frame, frame_output_dims)
        all_frames.append(frame)
    else:
      cap.release()

  all_frames = np.asarray(all_frames)
  return all_frames


def get_video_read_order(video_id, video_length, example_length):
  """Returns a tuple of video index and all possible examples."""
  if example_length is not None:
    # Grab all chunks of the video
    last_start_frame = video_length - example_length + 1

    read_order = [ReadOrderTuple(video_id=video_id,
                                 start_frame=start_frame,
                                 end_frame=start_frame + example_length)
                  for start_frame in range(0, last_start_frame)]
  else:
    # Return the whole video
    read_order = [ReadOrderTuple(
        video_id=video_id,
        start_frame=None,
        end_frame=None)]

  return read_order


def get_video_data(
    video_tuples,
    example_length,
    shuffle_examples,
    frame_output_dims,
    keep_frames=True):
    """Loads in raw video data and optionally randomizes order.

    Args:
    video_tuples: A list of namedtuples for the dataset.
    example_length: The video length of each example. If None, each example is
      the entire video.
    shuffle_examples: If True, example are written in shuffled order. Otherwise,
      they are written in the order read in.
    frame_output_dims: A namedtuple specifying the height and width of output
      frames. Frames will be resized to these dimensions if necessary.
    keep_frames: If True, loaded video is returned. Otherwise, only the
      read_order is built here and frame_data must be read in later.

    Returns:
      frame_data: A list of videos loaded. The list is empty if preload_video
        is False.
      read_order: A list of ReadOrderTuples built from the loaded video.
    """
    
    # TODO rework this. The chunked datasets should never be used. Then there is also no need for this function

    frame_data = [] if keep_frames else None
    read_order = []
    for tuple_i, video_tuple in enumerate(video_tuples):
      if example_length is not None or keep_frames:
        video = read_video(
            video_tuple.filename,
            frame_output_dims,
            keep_frames=keep_frames)
      
      if example_length is not None:
        read_order.extend(
          get_video_read_order(tuple_i, len(video), example_length))
      else:
        read_order.extend(
          get_video_read_order(tuple_i, None, None))
        
      if keep_frames:
        frame_data.append(video)

    # And shuffle read_order over the full dataset
    if shuffle_examples:
      random.shuffle(read_order)

    return frame_data, read_order


def write_example(
    video_tuple,
    video,
    writer=None,
    action_seq=None,
    summed_action_seq=None,
    absolute_action_seq=None,
    non_video_features=None):
  """Writes a single example to a shard. If no writer is given, returns the example"""

  feature = {}
  for feature_key, feature_type in non_video_features.items():
    if feature_type == "int64":
      feature_val = _int64_feature(getattr(video_tuple, feature_key))
    elif feature_type == "bytes":
      feature_val = _bytes_feature(getattr(video_tuple, feature_key))
    else:
      raise ValueError(
          "Unknown feature type {} for key {}".format(
              feature_type, feature_key))
    feature[feature_key] = feature_val

  feature["video_length"] = _int64_feature(video.shape[0])
  feature["video_height"] = _int64_feature(video.shape[1])
  feature["video_width"] = _int64_feature(video.shape[2])
  video_raw = video.tostring()
  feature["video"] = _bytes_feature(video_raw)
  if video.ndim == 4:
    video_channels = video.shape[3]
  else:
    video_channels = 1
  feature["video_channels"] = _int64_feature(video_channels)
  if action_seq is not None:
      actions_raw = action_seq.tostring()
      feature["actions"] = _bytes_feature(actions_raw)
      feature["num_actions"] = _int64_feature(action_seq.shape[1])
  if summed_action_seq is not None:
      if summed_action_seq.shape[1] != action_seq.shape[1]:
          raise ValueError("Action sequence and summed version need to have same number of actions.")
      actions_raw = summed_action_seq.tostring()
      feature["actions_summed"] = _bytes_feature(actions_raw)
  if absolute_action_seq is not None:
      if absolute_action_seq.shape[1] != action_seq.shape[1]:
          raise ValueError("Action sequence and absolute version need to have same number of actions.")
      actions_raw = absolute_action_seq.tostring()
      feature["actions_absolute"] = _bytes_feature(actions_raw)

  example = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
  
  if writer is not None:
    writer.write(example)
  return example


def build_one_shard(
    frame_data,
    video_tuples,
    read_order,
    shard_path,
    non_video_features=None,
    frame_output_dims=None,
    downsample_video=1,
    n_threads=1):
  """Builds a video dataset record shard."""
  
  # Initialize the shard
  writer = tf.python_io.TFRecordWriter(shard_path)
  
  def process_one_video(example_i):
    if frame_data is not None:
      frame_data_i = frame_data[example_i.video_id]
    else:
      frame_data_i = read_video(
        video_tuples[example_i.video_id].filename, frame_output_dims, downsample=downsample_video)
    
    if example_i.end_frame is not None:
      # Write a chunk of the video
      result = write_example(
        video_tuples[example_i.video_id],
        frame_data_i[
        example_i.start_frame:example_i.end_frame, ...],
        non_video_features=non_video_features)
    else:
      # Write the whole video out
      result = write_example(
        video_tuples[example_i.video_id],
        frame_data_i,
        non_video_features=non_video_features)
    
    return result
  
  # Note: as the threadpooling enables parallel reading and writing,
  # it is faster even if there is only one producer thread
  
  tp = ThreadPool(n_threads)
  results = tp.imap(process_one_video, read_order)  # an entire chunk must be processed before return
  # if chuncksize=1, tqdm will return online results

  for example in tqdm.tqdm(results):
    writer.write(example)

  tp.close()
  tp.join()
  
  if writer is not None:  # TODO why is this needed?
    writer.close()



def build_phase_shards(
    frame_data,
    video_tuples,
    read_order,
    base_dir,
    n_shards,
    phase,
    non_video_features={},
    frame_output_dims=None,
    downsample_video=1,
    n_threads=1):
  """Writes the shards for one phase of a shuffled dataset."""

  out_dir = os.path.join(base_dir, phase)
  if not os.path.exists(out_dir):
      os.makedirs(out_dir)
  print("Building shards for phase {} to {}.".format(phase, out_dir))

  shard_pts = np.ceil(np.linspace(0, len(read_order), n_shards + 1)).astype(int)
  for shard_i in range(0, n_shards):
    shard_path = os.path.join(out_dir,
                              "{}_of_{}.tfrecord".format(shard_i, n_shards))
    read_order_i = read_order[shard_pts[shard_i]:shard_pts[shard_i + 1]]

    print("Building shard {} of {}...".format(shard_i, n_shards))
    build_one_shard(
        frame_data,
        video_tuples,
        read_order_i,
        shard_path,
        non_video_features=non_video_features,
        frame_output_dims=frame_output_dims,
        downsample_video=downsample_video,
        n_threads=n_threads)
    n_written_shard = shard_pts[shard_i + 1] - shard_pts[shard_i]
    print("Shard built with {} examples.".format(n_written_shard))

  with open(os.path.join(out_dir, 'README.txt'), 'a') as readme:
    readme.write('{}\n'.format(len(read_order)))

