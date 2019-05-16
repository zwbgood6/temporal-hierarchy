"Builds the computational graph, i.e. the network architecture and losses."

import tensorflow as tf
import numpy as np
import sonnet as snt

from architectures import sequence_architectures, acssp, sssp
from specs.network_specs import HierarchicalNetworkSpec, ActionConditionedSpec, StochasticSingleStepSpec
import losses
from architectures import th_utils
from utils import AttrDict


FLAGS = tf.flags.FLAGS


def get_model(network_spec):
  if isinstance(network_spec, HierarchicalNetworkSpec):
    return sequence_architectures.TemporalHierarchyModel
  elif isinstance(network_spec, ActionConditionedSpec):
    return acssp.ActionConditionedSingleStepPredictor
  elif isinstance(network_spec, StochasticSingleStepSpec):
    if FLAGS.use_svg_kf_detection:
      return sssp.StochasticSingleStepPredictorKFDetect
    else:
      return sssp.StochasticSingleStepPredictor
  else:
    raise ValueError("No corresponding network type for given spec.")


class GraphBuilder():
  def __init__(self,
               network_spec,
               loss_spec,
               data_handler,
               is_hierarchical):
    self.generator_scope = "generator"
    self.data_handler = data_handler
    self.network_spec = network_spec
    self.loss_spec = loss_spec
    self.is_hierarchical = is_hierarchical
    self.has_image_input = False if (FLAGS.dataset_config_name == "bouncing_balls" and not FLAGS.image_input) else True
    num_actions = data_handler.get_num_actions()
    with tf.name_scope("softmax_temp"):
      self.tau = tf.Variable(FLAGS.tau, name="tau", trainable=False)
      self.reduce_tau = self.tau.assign(tf.maximum(self.tau * FLAGS.reduce_temp_multiplier, FLAGS.tau_min))
    model = get_model(network_spec)
    self.network = model(
      network_spec,
      channels=data_handler.get_channels(),
      input_image_shape=data_handler.get_input_img_shape(),
      output_activation=loss_spec.image_output_activation,
      backprop_elstm_to_encoder=loss_spec.backprop_elstm_to_encoder,
      use_recursive_image=FLAGS.use_recursive_image,
      num_actions=num_actions,
      has_image_input=self.has_image_input,
      render_fcn=None if self.has_image_input else data_handler.get_render_fcn(),
      render_shape=None if self.has_image_input else data_handler.get_render_shape(),
      tau=self.tau,
      infer_actions=FLAGS.train_action_regressor,
      name=self.generator_scope)

  def setup_decay_lr(self,
                     reduce_criterion,
                     scheduled_reduce_interval,
                     plateau_min_delay,
                     plateau_scale_criterion):
    """Builds a closure for checking and decaying the learning rate.

    Args:
      reduce_criterion: The criterion for reducing the learning rate (a string).
      scheduled_reduce_interval: The interal at which to reduce the learning rate,
        if using a scheduled reduce_criterion.
      plateau_min_delay: The minimum number of batches to wait before decaying
        the learning rate, if using a plateau criterion.
      plateau_scale_criterion: The criterion for determining a plateau in the
        validation loss, as a fraction of the validation loss. Typically a small
        value in (0, 1].
    Returns:
      decay_lr: A function that will call or not call the reduce_learning_rate
        based on the chosen criterion and the relevant status.
    """
    # Keep track of state variables to update
    state_variables = {
      "previous_iteration": 0,
      "time_since_plateau": 0,
      "plateau_criterion_loss_old": np.inf,
    }

    def decay_lr(current_iteration, plateau_criterion_loss):
      """A function that reduces the learning rate when appropriate.

      Args:
        current_iteration: The current training iteration.
        plateau_criterion_loss: The current value of the loss value monitored
          for plateau-based learning rate decay.

      Returns:
        reduce_lr: True when learning rate should be reduced, False otherwise.
      """
      reduce_lr = False

      time_since_last = current_iteration - state_variables["previous_iteration"]

      if reduce_criterion == "scheduled":
        state_variables["time_since_plateau"] += time_since_last
        if state_variables["time_since_plateau"] >= scheduled_reduce_interval:
          reduce_lr = True
          state_variables["time_since_plateau"] = 0

      elif reduce_criterion == "plateau":
        delay_satisfied = (state_variables["time_since_plateau"] >
                           plateau_min_delay)
        at_plateau = (
                       state_variables["plateau_criterion_loss_old"] -
                       plateau_criterion_loss) < (plateau_scale_criterion *
                                                  plateau_criterion_loss)
        if delay_satisfied and at_plateau:
          reduce_lr = True
          state_variables["time_since_plateau"] = 0
        else:
          state_variables["time_since_plateau"] += time_since_last
        state_variables["plateau_criterion_loss_old"] = plateau_criterion_loss
      else:
        raise ValueError("Unknown learning rate reduce_criterion {}".format(
          reduce_criterion))

      state_variables["previous_iteration"] = current_iteration
      return reduce_lr

    return decay_lr
  
  def build_model(self,
                  data_tensors,
                  phase):
    model_output = self.network(
      data_tensors,
      n_frames_input=self.data_handler.get_input_len(phase),
      n_frames_predict=FLAGS.pred_seq_len,
      is_training=True if phase == "train" else False)
    return model_output

  def build_losses(self,
                   model_output_train,
                   model_output_val_test,
                   train_data, val_test_data,
                   global_step):

    pred_seq_len = FLAGS.pred_seq_len
    has_image_input = False if (FLAGS.dataset_config_name == "bouncing_balls" and not FLAGS.image_input) else True
    if has_image_input:
      def process_input(data):
        data["reconstruct"] = data.pop("input_images")
        data["predict"] = data.pop("predict_images")[:pred_seq_len]
    else:
      render_fcn = self.data_handler.get_render_fcn()
      input_render_shape = train_data.input_images.get_shape().as_list()[:2] + self.data_handler.get_render_shape()
      predict_render_shape = train_data.predict_images[:pred_seq_len].get_shape().as_list()[:2] + self.data_handler.get_render_shape()
      def process_input(data):
        data["reconstruct_coord"] = data.pop("input_images")
        data["predict_coord"] = data.pop("predict_images")[:pred_seq_len]
        rendered_input = tf.py_func(render_fcn, [data.reconstruct_coord], tf.float32)
        rendered_predict = tf.py_func(render_fcn, [data.predict_coord], tf.float32)
        data["reconstruct"] = tf.reshape(rendered_input, input_render_shape)
        data["predict"] = tf.reshape(rendered_predict, predict_render_shape)
        
    input_train = AttrDict(train_data)  # copy the dict
    input_val_test = AttrDict(val_test_data)

    process_input(input_train)
    process_input(input_val_test)

    with tf.name_scope("learning_rate"):
      learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")

    with tf.name_scope("sequence_length_loss_weight"):
      if FLAGS.schedule_seq_len_weight:
        seq_len_loss_weight = tf.train.exponential_decay(
            1e-9, tf.cast(global_step, tf.float32), FLAGS.seq_len_schedule_step, 1.5, staircase=True,
        )
        seq_len_loss_weight = tf.minimum(seq_len_loss_weight, FLAGS.sequence_length_term)
      else:
        seq_len_loss_weight = tf.constant(FLAGS.sequence_length_term)

    if self.network_spec.use_recursive_skips and FLAGS.use_recursive_image \
          or not has_image_input or FLAGS.use_cdna_decoder:
      # Nonlinearity applied in network setup / rendering
      predictions_preactivated = True
    else:
      predictions_preactivated = False

    learning_rates = dict()
    learning_rates["prediction"] = learning_rate
    if FLAGS.infer_actions:
      learning_rates["action"] = FLAGS.lr_action_inference
      learning_rates["latent"] = FLAGS.lr_action_inference
    if (tf.flags.FLAGS.train_action_regressor or
          tf.flags.FLAGS.train_abs_action_regressor) and not FLAGS.use_cdna_model:
      learning_rates["abs_action"] = FLAGS.lr_action_inference
    if FLAGS.regularizer_weight > 0:
      learning_rates["regularization"] = FLAGS.regularizer_weight

    monitor_values, monitor_index, opt_steps, train_ops_gan, train_steps_gan = losses.configure_optimization(
      input_train=input_train,
      input_val=input_val_test,
      model_output_train=model_output_train,
      model_output_val=model_output_val_test,
      learning_rates=learning_rates,
      tau=self.tau,
      regularizer_weight=FLAGS.regularizer_weight,
      seq_len_loss_weight=seq_len_loss_weight,
      loss_spec=self.loss_spec,
      use_gan=False,
      generator_scope=self.generator_scope,
      max_grad_norm=FLAGS.max_grad_norm,
      optimizer_epsilon=FLAGS.optimizer_epsilon,
      data_spec_train=self.data_handler.get_dataset_spec("train"),
      dataset_name=self.data_handler.get_dataset_name(),
      global_step=global_step,
      optimizer_type=FLAGS.optimizer_type,
      predictions_preactivated=predictions_preactivated,
      has_image_input=has_image_input,
      is_hierarchical=self.is_hierarchical)
    monitor_index["test"] = monitor_index["val"]

    decay_lr = self.setup_decay_lr(
      FLAGS.learning_rate_reduce_criterion,
      FLAGS.reduce_learning_rate_interval,
      FLAGS.plateau_min_delay,
      FLAGS.plateau_scale_criterion)

    return monitor_values, monitor_index, opt_steps, train_ops_gan, train_steps_gan, \
            learning_rate, decay_lr

