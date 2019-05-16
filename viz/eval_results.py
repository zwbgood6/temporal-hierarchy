"""Visualize the results of a test run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import os
import tensorflow as tf
import numpy as np
import h5py

import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import params
from result_viz import gen_composed_hierarchical_seqs, gen_hierarchical_plot_imgs
import logger


FLAGS = tf.flags.FLAGS
N_VIS_SEQS = 10


def hdf2dict(filename):
    """Parses a HDF5 results file into a dictionary."""
    eval_results = dict()
    h5f = h5py.File(filename, 'r')
    for dataset in h5f.values():
        if "global_step" in dataset.name:
            eval_results.update({dataset.name[1:]: h5f[dataset.name][()]})       # log the only scalar separately
        else:
            eval_results.update({dataset.name[1:]: h5f[dataset.name][:]})       # first symbol in name is "/"
    h5f.close()
    return eval_results


def main(unused_argv):
    # open HDF5 file and load content
    base_dir = os.path.expanduser(FLAGS.base_dir)
    results_file = os.path.join(base_dir, "test", "eval_results.h5")
    eval_results = hdf2dict(results_file)

    # gen logger to visualize results in tensorboard
    summary_dir = os.path.join(base_dir, "summaries")
    viz_logger = logger.Logger(summary_dir)

    # generate hierarchical sequence plots
    composed_seqs, keyframe_idxs = gen_composed_hierarchical_seqs(eval_results["decoded_low_level_frames"],
                                                                  eval_results["high_level_rnn_output_dt"],
                                                                  N_VIS_SEQS)
    plot_imgs = gen_hierarchical_plot_imgs(eval_results["input_images_true"],
                                           eval_results["predict_images_true"],
                                           composed_seqs,
                                           None,
                                           eval_results["decoded_keyframes"],
                                           keyframe_idxs,
                                           eval_results["attention_weights"],
                                           eval_results["gt_keyframe_idxs"])
    viz_logger.log_images("image_predictions_test", plot_imgs, eval_results["global_step"])

    viz_logger.writer.flush()
    print("Visualization logged to Tensorboard!")


if __name__ == "__main__":
  tf.app.run(main=main)
