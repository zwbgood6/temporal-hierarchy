import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath('../data'))
sys.path.insert(0, os.path.abspath('../specs'))
from data.utils import video_utils
from specs import dataset_specs
from reacher_data_handler import DataHandler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--base_dir",
                        help="base directory for storing generated data")
    parser.add_argument("-nt", "--num_train_samples", type=int,
                        help="number of training sequences")
    parser.add_argument("-nv", "--num_val_samples", type=int,
                        help="number of validation sequences")
    parser.add_argument("-ns", "--num_shard_samples", type=int,
                        help="number of sequences per shard")
    parser.add_argument("-nj", "--num_joints", type=int,
                        help="number of reacher joints")
    parser.add_argument("--sequence_length", type=int,
                        help="number of images per sequence", default=15)
    parser.add_argument("-i", "--index", type=int,
                        help="current index in data generation")
    args = parser.parse_args()
    return args


def build_reacher_dataset(phase, args, n_shards):
    if args.num_joints == 1:
        dataset_name = "reacher_oneJoint"
    elif args.num_joints == 2:
        dataset_name = "reacher"
    else:
        raise ValueError("Currently only 1 and 2 joints supported for reacher!")
    dataset_config = dataset_specs.get_data_spec(dataset_name,
                                                 dataset_phase=phase,
                                                 batch_size=None,  # value not used for tfRecord gen
                                                 input_seq_len=args.sequence_length - 1,
                                                 pred_seq_len=1)
    handler = DataHandler(dataset_config)
    if not dataset_config.num_actions == handler.GetNumActions():
        raise ValueError("Reacher dataset config number of actions need to correspond to "
                         "actual number of actions in environment. Currently %d vs. %d."
                         % (dataset_config.num_actions, handler.GetNumActions()))

    outfolder = os.path.join(args.base_dir, phase)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    env = handler.MakeEnvironment()
    shard_file = os.path.join(outfolder, "reacher_%d_of_%d.tfrecord" % (args.index+1, n_shards))
    writer = tf.python_io.TFRecordWriter(shard_file)
    for seq_idx in range(args.num_shard_samples):
        img_seq, action_seq, summed_action_seq, absolute_action_seq = \
            handler.CreateSequence(env,
                                   args.sequence_length,
                                   resolution=dataset_config.im_height)

        # cast sequences in correct formats for encoding in tfRecords
        img_seq = img_seq.astype(np.uint8)
        action_seq = action_seq.astype(np.float32)
        summed_action_seq = summed_action_seq.astype(np.float32)
        absolute_action_seq = absolute_action_seq.astype(np.float32)
        video_utils.write_example(video_tuple=[],  # not needed as no non_video_features
                                  video=img_seq,
                                  action_seq=action_seq,
                                  summed_action_seq=summed_action_seq,
                                  absolute_action_seq=absolute_action_seq,
                                  writer=writer,
                                  non_video_features={})  # no non_video_features
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    n_shards_train = int(np.ceil(args.num_train_samples / args.num_shard_samples))
    n_shards_val = int(np.ceil(args.num_val_samples / args.num_shard_samples))

    # sanity checking of input arguments
    if not args.sequence_length > 1:
        raise ValueError("Sequence length needs to be > 1. Currently given: %d." % args.sequence_length)
    if not (args.num_train_samples % args.num_shard_samples == 0 and
            args.num_val_samples % args.num_shard_samples == 0):
        raise ValueError("Number of sequences per shard needs to evenly divide "
                         "number of desired samples in train/val dataset.")
    if not args.index < np.max((n_shards_train, n_shards_val)):
        raise ValueError("Index must be smaller than max number of shards, which is %d" %
                         np.max((n_shards_train, n_shards_val)))

    # build reacher dataset
    if args.index < n_shards_train:
        build_reacher_dataset("train", args, n_shards_train)
    if args.index < n_shards_val:
        build_reacher_dataset("val", args, n_shards_val)