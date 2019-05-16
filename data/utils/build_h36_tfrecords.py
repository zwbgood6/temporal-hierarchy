"""Utilities to convert KTH data to tfrecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import glob
import os
import random
import tensorflow as tf

import video_utils


tf.flags.DEFINE_integer("n_threads", 1,
                        "Number of threads used")
tf.flags.DEFINE_integer("n_files_write", 0,
                        "Number of files written to each phase. If 0, all files are written.")
tf.flags.DEFINE_string("base_dir", "~/logs",
                       "Number of threads used")

FLAGS = tf.flags.FLAGS

random.seed(a=123)

H36VideoTuple = collections.namedtuple(
    "H36VideoTuple",
    "filename")

class H36VideoBuilder():
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
    self.out_dir = video_utils.make_timestamp_dir(self.tfrecord_dir)
    # Note: one shard per subject in the dataset is created (not necessarily containing videos of that subject)
    
    self.extension = "mp4"
    self.ids_train = [1, 5, 6, 7]
    self.ids_val = [8]
    self.ids_test = [9, 11]
  
  def build_phase(
      self,
      phase,
      IDs,
      shuffle_examples=True):
    """ Builds one phase of the dataset.
    
    :param phase: string "train"/"val"/"test
    :param IDs: a list of person IDs
    :param shuffle_examples: whether the examples will be shuffled
    """
    
    # Fetch names
    filenames = []
    for id in IDs:
      filenames = filenames + glob.glob(
          os.path.join(self.base_dir, "S{}".format(id), "Videos", "*.{}".format(self.extension)))
      
    if FLAGS.n_files_write > 0:
      filenames = filenames[:FLAGS.n_files_write]
  
    video_tuples = map(lambda x: H36VideoTuple(filename=x), filenames)  # this can be extended to include metadata
  
    # Randomizes the read order
    _, read_order = video_utils.get_video_data(
        video_tuples,
        example_length=None,
        shuffle_examples=shuffle_examples,
        frame_output_dims=None,  # This doesn't load the videos
        keep_frames=False)
  
    # Build the tfrecords:
    video_utils.build_phase_shards(
        frame_data=None,
        video_tuples=video_tuples,
        read_order=read_order,
        base_dir=self.out_dir,
        n_shards=len(IDs),
        phase=phase,
        frame_output_dims=self.frame_dims,
        downsample_video=self.downsample_video,
        n_threads=self.n_threads)
  
    print("{} shards written with {} examples total.".format(phase, len(read_order)))
  
  def build_dataset(
        self,
        shuffle_train_examples):
    """Builds the full set of tfrecords.
  
    Args:
      shuffle_train_examples: If True, shuffles train examples to random shards.
        Otherwise, reads the files out serially to a shard.
    """
    
    self.build_phase(
        "train",
        self.ids_train,
        shuffle_examples=shuffle_train_examples)

    self.build_phase(
        "val",
        self.ids_val,
        shuffle_examples=False)

    self.build_phase(
        "test",
        self.ids_test,
        shuffle_examples=False)

def main(unused_argv):
  # Build full dataset
  
  base_dir = "/NAS/data/h36m/"
  tfrecord_dir = os.path.join(base_dir, "tfrec_video")

  # TODO Add jpeg compression (unnecessary for the downsampled dataset)
  # TODO if the memory is not cleaned properly in the ThreadPool, implement a custom solution
  # here is an example: https://codeistry.wordpress.com/2018/03/10/ordered-producer-consumer/
  
  # base_dir = "/Users/Oleh/Code/PredictionProject/temporal_hierarchy/temp"
  # tfrecord_dir = os.path.join(base_dir, "tfrec_video")
  
  shuffle_train_examples = True  # Randomize order of sequence in shards
  builder = H36VideoBuilder(downsample_video=8,
                            base_dir=base_dir,
                            tfrecord_dir=tfrecord_dir,
                            frame_dims=video_utils.FrameDims(height=64, width=64),  # downsample images
                            n_threads=FLAGS.n_threads)

  builder.build_dataset(shuffle_train_examples=shuffle_train_examples)


if __name__ == "__main__":
  tf.app.run(main=main)
