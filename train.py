"""Simple script to train a simple predictive LSTM on moving MNIST."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import time
import sys
import pipes

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import params
import configs
import logger
from architectures.graph_builder import GraphBuilder
from specs.network_specs import HierarchicalNetworkSpec
from data import data_handler
from viz import viz_utils
from viz.result_viz import log_sess_output

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.flags.FLAGS



def run_step(sess,
             monitor_values,
             opt_steps,
             train_ops_gan,
             train_steps_gan,
             monitor_index,
             phase,
             feed_dict_elems,
             use_gan,
             metrics_only=False):
  """Runs the steps appropriate for the given phase (train/val).

  Args:
    sess: The current session.
    monitor_values: A dict containing all Tensors whose values are monitored,
      but which do not control optimization.
    opt_steps: The non-GAN training ops to run.
    train_ops_gan: The GAN training ops to run.
    train_steps_gan: A namedtuple of GAN train steps.
    monitor_index: A nested dict allocating each Tensor to an optimization
      phase and specifying its type (loss, scalar, image, or hist).
    phase: The phase of training: 'train' or 'val'.
    feed_dict_elems: A dictionary of values to pass to a feed_dict.
    use_gan: If True, GAN alternating minimization run. Otherwise, all ops run
      together.
    metrics_only: If True, only loss and metric values will be reported.
      Otherwise, all values will be reported (including images, etc.). Defaults
      to False.
  Returns:
    run_output: A dictionary of evaluated values.
  Raises:
    ValueError: if phase is not 'train' or 'val'.
  """
  if phase == "train":
    sess_dict = get_sess_dict(monitor_values,
                              opt_steps,
                              train_ops_gan,
                              monitor_index,
                              phase,
                              metrics_only=metrics_only,
                              use_gan=use_gan)
  elif phase == "val":
    sess_dict = get_sess_dict(monitor_values,
                              None,
                              None,
                              monitor_index,
                              phase,
                              metrics_only=metrics_only,
                              use_gan=use_gan)
  else:
    raise ValueError("Unknown phase. Must be 'train' or 'val'.")

  if use_gan:
    # Run generator training steps.
    for _ in range(train_steps_gan.generator_train_steps):
      # Make sure this output is correct
      run_output = sess.run(
          sess_dict["generator"],
          feed_dict=feed_dict_elems)
    
    # Run discriminator training steps.
    if train_steps_gan.discriminator_train_steps > 1:
      print("For > 1, need to modify sess_dict here. "
            "Should only be logging on the last discriminator step. "
            "Should hold for validation and training.")

    for _ in range(train_steps_gan.discriminator_train_steps):
      run_output = sess.run(
          sess_dict["discriminator"],
          feed_dict=feed_dict_elems)
    sess.run(sess_dict["global_step"])  # Update the global step
  else:
    run_output = sess.run(sess_dict, feed_dict=feed_dict_elems)

  return run_output


def get_sess_dict(monitor_values,
                  opt_steps,
                  train_ops_gan,
                  monitor_index,
                  phase,
                  metrics_only=True,
                  use_gan=False):
  """Returns the appropriate dictionary of Tensors to pass to a run call.

  Args:
    monitor_values: A dict containing all Tensors whose values are monitored,
      but which do not control optimization.
    opt_steps: A dict containing all non-GAN Tensors controlling optimization.
    train_ops_gan: A namedtuple with fields for GAN optimization.
    monitor_index: A nested dict allocating each Tensor to an optimization
      phase and specifying its type (loss, scalar, image, or hist).
    phase: The phase of training: 'train' or 'val'.
    metrics_only: If True, only monitor values corresponding to losses and
      metrics will be returned. If False, all values for this phase will be
      returned. Defaults to True.
    use_gan: If True, returns a nested sess_dict with values for generator and
      discriminator phases. Otherwise, returns a non-nested sess_dict for a
      single optimization step.
  Returns:
    sess_dict: The sess_dict for the current run call.
  """
  sess_names = []

  for type_name, type_values in monitor_index[phase].items():
    if metrics_only:
      update_sess = (type_name == "loss") or \
                    (type_name == "metric") or \
                    (type_name == "fetch_no_log")
    else:
      update_sess = True

    if update_sess:
      sess_names.extend(type_values)

  sess_dict = dict((key_i, val_i) for key_i, val_i in monitor_values.items()
                   if key_i in sess_names)

  if use_gan:
    # All non-training ops run with discriminator
    sess_dict = {
        "generator": {},
        "discriminator": sess_dict,
        "global_step": {},
    }

  if phase == "train":
    # Add training ops.
    # NB: for GAN, losses are already incorporated into gan train_ops,
    #  so opt_steps can be ignored.
    if use_gan:
      sess_dict["generator"]["generator_train_op"] = train_ops_gan.generator_train_op
      sess_dict["discriminator"].update(train_ops_gan.discriminator_train_op)
      sess_dict["global_step"]["inc_global_step"] = train_ops_gan.global_step_inc_op
    else:
      # Directly use the optimization stuff
      sess_dict.update(opt_steps)

  return sess_dict


def update_full_loss_vals(monitor_index,
                          full_loss_vals=None,
                          val_output=None,
                          phase="val"):
  """Updates the accumulated values of tracked losses.

  Args:
    monitor_index: A nested dictionary of Tensors, sorted by type.
    full_loss_vals: A dictionary of current state of the tracked losses. If
      None, the dictionary is initialized with all values set to zero.
      Defaults to None.
    val_output: The current evaluated values of a sess.run call. If None,
      values are not updated. Defaults to None.
  """
  tracking_fields = monitor_index[phase]["loss"] + monitor_index[phase]["metric"]

  if full_loss_vals is None:
    full_loss_vals = {}
    for tracking_field_i in tracking_fields:
      full_loss_vals[tracking_field_i] = 0

  if val_output is not None:
    for tracking_field_i in tracking_fields:
      full_loss_vals[tracking_field_i] += val_output[tracking_field_i]

  return full_loss_vals


def get_dummy_dataset_spec():
    _, data_spec_train, _, _ = configs.get_dataset_specs(
        FLAGS.dataset_config_name,
        FLAGS.train_batch_size,
        -1,
        -1,
        FLAGS.input_seq_len,
        FLAGS.pred_seq_len,
        None,
        None)
    return data_spec_train


def run_validation(dh, sess, monitor_values,
                   train_steps_gan, monitor_index, feed_dict_elems, base_dir,
                   np_logger, global_train_iteration, checkpoint_dir, decay_lr, is_hierarchical):
  full_loss_vals = update_full_loss_vals(monitor_index)
  n_val_batches = \
    int(np.floor(dh.get_dataset_size("val") / dh.get_fetched_batch_size("val")))

  # turn off randomness in validation data to keep val sets comparable
  dh.maybe_turnoff_randomness()

  for val_batch_idx in tqdm(range(n_val_batches)):
    metrics_only = val_batch_idx != n_val_batches - 1
    # Only grab images and other non-metric values on the last batch. Grab losses and metrics always
    # For test always get the images because we're saving them
    val_output = run_step(
      sess=sess,
      monitor_values=monitor_values,
      opt_steps=None,
      train_ops_gan=None,
      train_steps_gan=train_steps_gan,
      monitor_index=monitor_index,
      phase="val",  # Never pass "test" here
      feed_dict_elems=feed_dict_elems,
      use_gan=False,
      metrics_only=metrics_only)

    # Store the inference samples for PCA
    if FLAGS.save_z_samples:
      viz_utils.store_latent_samples(val_output,
                                     base_dir,
                                     val_batch_idx,
                                     n_val_batches,
                                     global_train_iteration,
                                     store_angle_regressor=FLAGS.train_action_regressor,
                                     store_comp_latents=False)

    # Update loss vals for full epoch
    full_loss_vals = update_full_loss_vals(monitor_index,
                                           full_loss_vals=full_loss_vals,
                                           val_output=val_output)

  # turn randomness back on for data handlers to have random train batches
  dh.maybe_turnon_randomness()

  for loss_name_i in full_loss_vals:
    full_loss_vals[loss_name_i] /= n_val_batches
    val_output[loss_name_i] = full_loss_vals[loss_name_i]

  # if output seqs are repeated show results for at least two seqs
  if FLAGS.test_sequence_repeat > 0:
    n_seq_val_log = 30
  else:
    n_seq_val_log = 10

  # TODO(oleh) save all generated .avis
  log_sess_output(val_output,
                  monitor_index,
                  np_logger,
                  global_train_iteration,
                  dh.get_dataset_name(),
                  checkpoint_dir,
                  n_seqs=n_seq_val_log,
                  phase="val",
                  build_seq_ims=True,
                  repeat=FLAGS.test_sequence_repeat,
                  is_hierarchical=is_hierarchical)  # at test time this is handled differently

  tf.logging.info("Train iteration %d: Validation prediction loss %f",
                  global_train_iteration,
                  val_output["total_loss_val"])

  reduce_lr = decay_lr(
    global_train_iteration,
    plateau_criterion_loss=val_output[FLAGS.plateau_criterion_loss_name])
  return reduce_lr


def build_restore_saver():
  # create variable list for loading generator + enc/decoder
  load_prefixes = ["generator/image_encoder", "generator/image_decoder",
                   "generator/low_level_rnn", "generator/generator/low_level_rnn"]
  load_vars = []
  for key in load_prefixes:
      load_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, key)

  # create saver to load vars
  restore_saver = tf.train.Saver(var_list=load_vars)
  return restore_saver


def load_pretrained_ll(sess, restore_saver):
  assert FLAGS.load_ll_ckpt_dir, "Need to specify path to pretrained weights for freezing low level net!"
  # get checkpoint file
  ckpt_path = tf.train.latest_checkpoint(FLAGS.load_ll_ckpt_dir)
  # restore vars
  restore_saver.restore(sess, save_path=ckpt_path)


def train(base_dir):
  """Run the training of the LSTM model."""

  time_0 = time.time()
  logging_global_step = tf.train.get_or_create_global_step()
  network_spec = configs.get_network_specs(FLAGS.network_config_name)
  loss_spec = configs.get_loss_spec(FLAGS.loss_config_name)
  is_hierarchical = isinstance(network_spec, HierarchicalNetworkSpec)

  # load data
  dh = data_handler.VideoDataHandler(loss_spec)
  train_data_tensors, val_test_data_tensors = dh.fetch_data(is_hierarchical)

  # build model architecture
  gb = GraphBuilder(network_spec, loss_spec, dh, is_hierarchical)
  model_output_train = gb.build_model(train_data_tensors, "train")
  model_output_val_test = gb.build_model(val_test_data_tensors, "val")

  # setup losses
  monitor_values, monitor_index, opt_steps, train_ops_gan, train_steps_gan, \
  learning_rate, decay_lr = gb.build_losses(model_output_train,
                                            model_output_val_test,
                                            train_data_tensors, val_test_data_tensors,
                                            logging_global_step)

  # prepare logging + load variables
  checkpoint_dir = base_dir
  summary_dir = os.path.join(base_dir, "summaries")
  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)
  variables_checkpoint=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  saver = tf.train.Saver(var_list=variables_checkpoint,
                         save_relative_paths=True)
  tf.add_to_collection(tf.GraphKeys.SAVERS, saver) # this will make the session call it
  if FLAGS.checkpoint_interval > 0:
    hooks = [
        tf.train.CheckpointSaverHook(
            checkpoint_dir=checkpoint_dir,
            save_steps=FLAGS.checkpoint_interval,
            saver=saver),
    ]
  else:
    hooks = None

  # restore variables
  if FLAGS.freeze_ll:       # only restore partial vars from pretrain if no checkpoint exists
    if tf.train.latest_checkpoint(checkpoint_dir) is None:
      restore_dir = None
      restore_saver = build_restore_saver()
    else:
      restore_dir = checkpoint_dir
  else:
    restore_dir = checkpoint_dir    # if all vars should be restored -> handled by MonitoredSession

  # start session
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True       # avoid taking up more space than needed
  with tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=restore_dir, config=sess_config) as sess:
    if restore_dir is None:
      load_pretrained_ll(sess, restore_saver)      # load part of vars
    start_iteration = sess.run(logging_global_step)
    reduce_lr = False
    np_logger = logger.Logger(summary_dir, dh.data_spec_train)

    reduce_tau = gb.reduce_tau
    feed_dict_elems = {
        learning_rate: FLAGS.learning_rate
    }

    # training loop
    for local_train_iteration in range(start_iteration, FLAGS.num_training_iterations):
      global_train_iteration = sess.run(logging_global_step)
      if global_train_iteration != local_train_iteration:
        raise ValueError("global_step must be updated exactly once per iteration!")

      if time.time() - time_0 > 3600:
        # it takes ~25 seconds before the first iteration runs.
        # The overhead for restarting a job (save, load, warmup) should be around a minute
        print("Timeout after 1 hour!")
        time_0 = time.time()
        #return # if we return, sess.close() gets called, which saves the checkpoint
      
      if global_train_iteration % FLAGS.validation_interval == 0 and global_train_iteration != 0 and FLAGS.validate:
        reduce_lr = run_validation(dh, sess, monitor_values,
                                   train_steps_gan, monitor_index, feed_dict_elems, base_dir,
                                   np_logger, global_train_iteration, checkpoint_dir, decay_lr, is_hierarchical)

      if reduce_lr:
        if FLAGS.reduce_learning_rate_multiplier != 1.0:
          raise ValueError("Learning rate decay is broken as the learning rate value"
                           " isn't saved between training restarts.")
        feed_dict_elems[learning_rate] *= FLAGS.reduce_learning_rate_multiplier
        tf.logging.info(
            "Reducing the learning rate to %f.",
            feed_dict_elems[learning_rate])
        reduce_lr = False

      # update dt softmax temp
      if global_train_iteration % FLAGS.tau_schedule_step == 0 and global_train_iteration != 0:
        sess.run(reduce_tau)

      # Run training ops
      report_all = (global_train_iteration % FLAGS.validation_interval == 0)
      report_losses = (global_train_iteration % FLAGS.report_interval == 0)

      if FLAGS.debug_main_loop:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        import pdb; pdb.set_trace()
        
      train_output = run_step(
          sess=sess,
          monitor_values=monitor_values,
          opt_steps=opt_steps,
          train_ops_gan=train_ops_gan,
          train_steps_gan=train_steps_gan,
          monitor_index=monitor_index,
          phase="train",
          feed_dict_elems=feed_dict_elems,
          use_gan=False,
          metrics_only=(not report_all))

      
      if report_all or report_losses:
        log_sess_output(train_output,
                        monitor_index,
                        np_logger,
                        global_train_iteration,
                        dh.get_dataset_name(),
                        checkpoint_dir,
                        phase="train",
                        build_seq_ims=report_all,
                        is_hierarchical=is_hierarchical)

      tf.logging.info("Train iteration: %d, total training loss: %f.,"
                      " prediction: %f.",
                      global_train_iteration,
                      train_output["total_loss"],
                      train_output["total_loss"]) # HACK


def setup_base_dir():
  if FLAGS.create_new_subdir:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_dir = os.path.join(os.path.expanduser(FLAGS.base_dir), timestamp)
  else:
    base_dir = os.path.expanduser(FLAGS.base_dir)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  return base_dir


def save_git(base_dir):
  # save code revision
  print('Save git commit and diff to {}/git.txt'.format(base_dir))
  cmds = ["echo `git rev-parse HEAD` >> {}".format(
    os.path.join(base_dir, 'git.txt')),
    "git diff >> {}".format(
      os.path.join(base_dir, 'git.txt'))]
  print(cmds)
  os.system("\n".join(cmds))


def save_cmd(base_dir):
  train_cmd = 'python ' + ' '.join([sys.argv[0]] + [pipes.quote(s) for s in sys.argv[1:]])
  train_cmd += '\n'
  print('\n' + '*' * 80)
  print('Training command:\n' + train_cmd)
  print('*' * 80 + '\n')
  with open(os.path.join(base_dir, "cmd.txt"), "a+") as f:
    f.write(train_cmd)


def main(unused_argv):
  base_dir = setup_base_dir()
  params.dump_params(base_dir)
  save_git(base_dir)
  save_cmd(base_dir)
  train(base_dir=base_dir)


if __name__ == "__main__":
  tf.app.run(main=main)
