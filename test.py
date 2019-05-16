"""Run the specified model on test data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

import params
import configs
from architectures.graph_builder import GraphBuilder
from specs.network_specs import HierarchicalNetworkSpec
from data import data_handler
from viz import viz_utils

FLAGS=tf.flags.FLAGS


def get_gt_images(input_images, predict_images, dh):
  """Renders output images if coordinate-based prediction."""
  predict_len = FLAGS.pred_seq_len
  if FLAGS.image_input:
    return {"input_images_true": input_images, "predict_images_true": predict_images[:predict_len]}
  else:
    render_fcn = dh.get_render_fcn()
    input_render_shape = input_images.get_shape().as_list()[:2] + dh.get_render_shape()
    predict_render_shape = predict_images[:predict_len].get_shape().as_list()[:2] + dh.get_render_shape()
    rendered_input = tf.py_func(render_fcn, [input_images], tf.float32)
    rendered_predict = tf.py_func(render_fcn, [predict_images[:predict_len]], tf.float32)
    return {"input_images_true": tf.reshape(rendered_input, input_render_shape),
            "predict_images_true": tf.reshape(rendered_predict, predict_render_shape)}


def test(base_dir):
  network_spec = configs.get_network_specs(FLAGS.network_config_name)
  loss_spec = configs.get_loss_spec(FLAGS.loss_config_name)
  is_hierarchical = isinstance(network_spec, HierarchicalNetworkSpec)

  # load data
  is_hierarchical = isinstance(network_spec, HierarchicalNetworkSpec)
  dh = data_handler.VideoDataHandler(loss_spec, is_test=True)
  data_tensors = dh.fetch_data(is_hierarchical)

  # build model architecture
  gb = GraphBuilder(network_spec, loss_spec, dh, is_hierarchical)
  print("##################################")
  print("!!! USES TRAINING MODEL !!!")
  print("##################################")
  model_output_test = gb.build_model(data_tensors, "train")

  # post-activate all image outputs
  if FLAGS.image_input:
    model_output_test["decoded_keyframes"] = tf.nn.sigmoid(model_output_test["decoded_keyframes"])
    model_output_test["decoded_low_level_frames"] = tf.nn.sigmoid(model_output_test["decoded_low_level_frames"])

  # log global step
  global_step_op = tf.train.get_or_create_global_step()

  # get fetch_dict
  fetch_dict = dict((key_i, val_i) for key_i, val_i in model_output_test.items())
  fetch_dict.update(get_gt_images(data_tensors.input_images, data_tensors.predict_images, dh))
  fetch_dict.update({"gt_keyframe_idxs": data_tensors.actions_abs})

  # construct feed_dict
  tau = gb.tau
  feed_dict_elems = { tau: FLAGS.tau }

  # configure variable loader
  variables_checkpoint = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  saver = tf.train.Saver(var_list=variables_checkpoint,
                         save_relative_paths=True)
  tf.add_to_collection(tf.GraphKeys.SAVERS, saver)

  # start session
  with tf.train.SingularMonitoredSession(checkpoint_dir=base_dir) as sess:
    global_step = sess.run(global_step_op)
    tf.set_random_seed(1)
    n_test_batches = \
      int(np.floor(dh.get_dataset_size("test") / dh.get_fetched_batch_size("test")))
    dh.maybe_turnoff_randomness()

    # setup output logging
    log_keys = ["decoded_keyframes",
                "decoded_low_level_frames",
                "high_level_rnn_output_dt",
                "input_images_true",
                "predict_images_true",
                "attention_weights",
                "gt_keyframe_idxs"]
    eval_saver = viz_utils.EvalSaver(log_keys=log_keys,
                                     base_dir=base_dir,
                                     test_batch_size=FLAGS.test_batch_size,
                                     global_step=global_step)

    for _ in tqdm(range(n_test_batches)):
      test_output = sess.run(fetch_dict, feed_dict=feed_dict_elems)
      eval_saver.log_results(test_output)
    eval_saver.dump_results()
    print("Finished test.")
    

def main(unused_argv):
  base_dir = os.path.expanduser(FLAGS.base_dir)
  if not os.path.exists(base_dir):
    raise ValueError("Base dir %s does not exist!" % base_dir)
  params.dump_params(base_dir, file_extension="test")
  test(base_dir=base_dir)


if __name__ == "__main__":
  tf.app.run(main=main)
