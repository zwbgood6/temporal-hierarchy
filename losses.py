"""Losses for sequence prediction."""

import collections
from architectures import discriminators, conv_architectures, th_utils
import gan_losses
import numpy as np
import sonnet as snt
import tensorflow as tf
from utils import safe_entropy, safe_bce_loss, tf_swapaxes, add_n_dims, shape
from tensorflow.contrib import distributions
from sonnet.python.modules.basic import merge_leading_dims
from sonnet.python.modules.basic import split_leading_dim
import utils

FLAGS = tf.flags.FLAGS

MonitorTuple = collections.namedtuple("MonitorTuple",
                                      "value name phase type weight")

ReconstructionLossTuple = collections.namedtuple(
    "ReconstructionLossTuple",
    "pixel_loss dssim_loss psnr psnr_first_frame ssim ssim_first_frame pixel_loss_first_frame")

# This is used so that we can keep track of whether there is an optimizer hooked
# to the global step. We do not want multiple optimizers to increment the global
# step as this would result in seemingly bigger (by the multiple) number of passed
# iterations
_global_step_incremented = False

def get_phase_string(phase):
  if phase == "val":
    phase_string = "_val"
  elif phase == "train":
    phase_string = ""

  return phase_string

def optimize(opt_losses,
             learning_rate,
             max_grad_norm,
             optimizer_epsilon,
             optimizer_name,
             global_step=None,
             optimizer_type="adam"):
  """Constructs the optimizer and utilities adds them to the graph.

  Args:
    opt_losses: A dictionary of scalar Tensors. Each value specifies the current
      training loss for one of the optimizers.
    learning_rate: The learning rate Tensor.
    max_grad_norm: The maximum norm for the gradient.
    optimizer_epsilon: The Adam epsilon value.
    optimizer_name: A string that is concatenated to the optimizer key.
    optimizer_type: A string specifying the optimizer to use.
      Defaults to "adam".
  Returns:
    opt_steps: A dictionary of ops, each of which applies gradients to trainable
      parameters.
  """
  global _global_step_incremented
  
  
  with tf.name_scope("optimizer"):
    if _global_step_incremented:
      global_step = None
    else:
      if global_step is None:
        with tf.variable_scope("global_step", reuse=tf.AUTO_REUSE):
          global_step = tf.get_variable(
                name="global_step",
                shape=[],
                dtype=tf.int64,
                initializer=tf.zeros_initializer(),
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])
      _global_step_incremented = True

    if FLAGS.freeze_ll:
      assert not FLAGS.pretrain_ll, "Cannot freeze low level during low level pretraining!"
      train_prefixes = ["generator/generator", "generator/encoder_rnn",     # generator/generator is for RNN inits
                        "generator/inference_rnn", "generator/high_level_rnn"]
      if FLAGS.static_dt:
          train_prefixes += ["generator/static_dt"]
      trainable_variables = []
      for key in train_prefixes:
        trainable_variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, key)
    else:
      trainable_variables = tf.trainable_variables()
    opt_steps = {}
    for loss_key_i, loss_val_i in opt_losses.items():
      grads = tf.gradients(loss_val_i, trainable_variables)
      if max_grad_norm > 0:
        # import pdb
        # def debug_fn(*args):
        #   grads_in = grads
        #   pdb.set_trace()
        # debug_dep = tf.py_func(debug_fn, [grad for grad in grads if grad is not None], tf.float32)
        # debug_dep = 1
        # with tf.control_dependencies([debug_dep]):
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

      if optimizer_type == "adam":
        optimizer = tf.train.AdamOptimizer(
            learning_rate, epsilon=optimizer_epsilon)
      elif optimizer_type == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      elif optimizer_type == "momentum" or optimizer_type == "nesterov":
        use_nesterov = (optimizer_type == "nesterov")
        momentum = 0.5
        optimizer = tf.train.MomentumOptimizer(
            learning_rate, momentum=momentum, use_nesterov=use_nesterov)
      else:
        raise ValueError("Unknown optimizer.")

      opt_steps[loss_key_i + "_" + optimizer_name + "_optimize"] = optimizer.apply_gradients(
          zip(grads, trainable_variables),
          global_step=global_step)

  return opt_steps

def average_loss(loss_tensor, average_type, batch_dims=[0]):
  """Returns an averaged loss.

  Args:
    loss_tensor: A tensor with the elementwise loss values.
    average_type: A string indicating what type of loss averaging to apply.
      "elemwise" or "batchwise".
    batch_dims: A list specifying which dimensions are considered batch-like.
  Returns:
    loss_scalar: A scalar tensor with the averaged loss.
  """
  if average_type == "elemwise":
    loss_scalar = tf.reduce_mean(loss_tensor)
  elif average_type == "batchwise":
    loss_scalar = tf.reduce_sum(tf.reduce_mean(loss_tensor, axis=batch_dims))

  return loss_scalar

def seq_norm_loss(estimates):
  """Returns a norm loss on the elementwise norm, averaging across images.

  Args:
    estimates: The input sequence to compute norms on.
  Returns:
    loss_scalar: A scalar tensor with the average norm.
  """
  # Flatten to T x B x C
  input_shape = estimates.get_shape()
  estimates_flat = tf.reshape(estimates, [input_shape[0],
                                          input_shape[1],
                                          -1])
  estimates_norm = tf.norm(
      estimates_flat,
      ord="euclidean",
      axis=2)
  loss_scalar = average_loss(estimates_norm, "elemwise")
  return loss_scalar


def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.
  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


# Should be done on images, and averaged across batch and time.
def peak_signal_to_noise_ratio(
    estimates,
    targets,
    min_to_max_range=1.0,
    data_format="TNCHW"):
  """Image quality metric based on maximal signal power vs. power of the noise.

  Args:
    estimates: The estimated tensor.
    targets: The true tensor.
  Returns:
    peak signal to noise ratio (PSNR)
  """

  mse = tf.losses.mean_squared_error(
      labels=targets,
      predictions=estimates,
      reduction=tf.losses.Reduction.NONE)

  if data_format == "TNCHW":
    img_axes = [2, 3, 4]
  else:
    raise ValueError("Unknown data format.")

  mse_img = tf.reduce_mean(mse, axis=img_axes)

  psnr_imgs = 10.0 * tf.log((min_to_max_range ** 2) / mse_img) / tf.log(10.0)
  avg_psnr = tf.reduce_mean(psnr_imgs)
  psnr_first_frame = tf.reduce_mean(psnr_imgs[0, ...])

  return avg_psnr, psnr_first_frame


def split_leading_dim_with_shape(tensor, input_shape_list, n_dims=2):
  """Split the first dimension of a tensor.
  Args:
    tensor: Tensor to have its first dimension split.
    input_shape_list: Reference input shape to look the dimensions of.
    n_dims: Number of dimensions to split.
  Returns:
    The input tensor, with its first dimension split.
  """
  tensor_shape_static = tensor.get_shape()
  tensor_shape_list = tensor_shape_static.as_list()

  new_shape = input_shape_list[:n_dims] + tensor_shape_list[1:]
  return tf.reshape(tensor, new_shape)


def SSIM_sequence(
    estimates,
    targets,
    k1=0.01,
    k2=0.03,
    L=1.0,
    patch_size=7,
    patch_stride=2,
    time_major=True,
    channels=1,
    data_format="NCHW"):
  """Stuctural dissimilarity for sequences. Returns loss and raw versions.

  Takes values between 0 (maximally similar) and 0.5 (maximally dissimilar).

  Note: the weighting is uniform and the metric is computed on grayscale.
  Multiple papers (Finn, Villegas) apply gaussian filters and (Finn, Denton)
  compute average over 3 channels first converting to RGB.

  Args:
    estimates: The estimated images.
    targets: The ground true images.
    k1: A stabilization term to make the mean terms more numerically stable.
      Defaults to 0.01.
    k2: A stabilization term to make the std terms more numerically stable.
      Defaults to 0.03.
    L: The dynamic range of the pixels. Typically 2^#bits per pixel - 1.
      Defaults to 1.
    patch_size: The size of the patches to compare between the two images. The
      original SSIM paper uses a patch size of 8. Defaults to 5.
    patch_stride: The stride between compared patches.
    time_major: If True, input tensors are assumed to be of shape
      [time, batch, ...], otherwise they're assumed to be [batch, time, ...].
      Defaults to True.
    data_format: The image data format, either "NCHW" or "NHWC". Defaults to
      "NCHW".
  Returns:
    avg_dssim: The SSIM as loss: DSSIM = (SSIM-1)/2
    avg_ssim: The average SSIM over all frames.
    avg_ssim_first_frame: The average SSIM for the first frame of each batch.
  """
  #TODO(oleh) add options for gaussian weighting and averaging across channels
  # Get batch and time for final reshape

  if len(targets.get_shape().as_list()) == 6:
    n_dims = 3 # additional dimension, e.g. in temporal hierarchy
    input_shape_bt = targets.get_shape().as_list()[:3]
  else:
    n_dims = 2 # Collapse time and batch dimensions:
    input_shape_bt = targets.get_shape().as_list()[:2]
  targets = merge_leading_dims(targets, n_dims=n_dims)
  estimates = merge_leading_dims(estimates, n_dims=n_dims)
  # NCHW->NHWC
  if data_format == "NCHW":
    targets = tf.transpose(targets, [0, 2, 3, 1])
    estimates = tf.transpose(estimates, [0, 2, 3, 1])
  if channels == 3:
    targets = tf.image.rgb_to_grayscale(targets)
    estimates = tf.image.rgb_to_grayscale(estimates)

  # Make tanh output positive for SSIM computation
  if L == 2:
    estimates += 1
    targets += 1

  # Get height and weight for final reshape
  input_shape_hw = targets.get_shape().as_list()[1:3]
  output_shape = input_shape_bt + [hw / patch_stride for hw in input_shape_hw]

  ksizes = [1, patch_size, patch_size, 1]
  strides = [1, patch_stride, patch_stride, 1]
  rates = [1, 1, 1, 1]

  # Patches are N x H x W x (n_pixels_per_patch)
  target_patches = tf.extract_image_patches(
      targets,
      ksizes=ksizes,
      strides=strides,
      rates=rates,
      padding="SAME")
  estimate_patches = tf.extract_image_patches(
      estimates,
      ksizes=ksizes,
      strides=strides,
      rates=rates,
      padding="SAME")

  u_true, var_true = tf.nn.moments(target_patches, axes=[3])
  u_pred, var_pred = tf.nn.moments(estimate_patches, axes=[3])
  std_true = tf.sqrt(var_true)
  std_pred = tf.sqrt(var_pred)
  mean_product = u_true * u_pred
  covar_true_pred = tf.reduce_mean(
      target_patches * estimate_patches, axis=-1) - mean_product

  c1 = (k1 * L) ** 2
  c2 = (k2 * L) ** 2
  ssim = (2 * mean_product + c1) * (2 * covar_true_pred + c2)
  denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
  ssim /= denom

  dssim = (1.0 - ssim) / 2.0

  # Average across patches at different positions and time/batch
  avg_dssim = tf.reduce_mean(dssim)
  avg_ssim = tf.reduce_mean(ssim)
  ssim_framewise = split_leading_dim_with_shape(ssim, output_shape, n_dims=n_dims)
  avg_ssim_first_frame = tf.reduce_mean(ssim_framewise[0, ...])

  return avg_dssim, avg_ssim, avg_ssim_first_frame


def seq_reconstruction_loss(
    estimates,
    targets,
    weights,
    loss_type="mse",
    output_activation=None,
    average_type="elemwise",
    time_major=True,
    output_channels=1,
    train_pixel_loss=True,
    train_ssim=False,
    dataset_name=None):
  """Defines a reconstruction loss for image sequences.

  Args:
    estimates: The estimated reconstruction.
    targets: The target reconstruction.
    loss_type: A string specifying which loss to use for reconstruction.
      Defaults to "mse".
    output_activation: An activation function to apply to estimates before
      evaluating the loss. Defaults to None.
    average_type: The type of average applied to the reconstruction loss.
      "elemwise" averages over the total number of elements in the estimated
      arrays, while "batchwise" averages only across batches. The former is
      invariant to e.g. # of pixels, image channels, and sequence length.
    time_major: If True, input tensors are assumed to be of shape
      [time, batch, ...], otherwise they're assumed to be [batch, time, ...].
      Defaults to True.
    output_channels: The number of channels in the output image.
    train_pixel_loss: If True, the pixel loss (specified in loss_type)
      is added to the training loss. Defaults to True.
    train_ssim: If True, DSSIM is added to the training loss. Defaults to
      False.

  Returns:
    loss: A scalar Tensor giving the value of the reconstruction loss.
    component_losses: A namedtuple with the raw pixel loss, SSIM, and PSNR
      losses.
    activated_estimates: The results of applying the activation on the
      estimates.
  """

  if not (train_pixel_loss or train_ssim):
    raise ValueError("At least one reconstruction loss type must be True!")

  if time_major:
    batch_dims = [1,]
  else:
    batch_dims = [0,]

  if loss_type == "mse":
    if output_activation is not None:
      activated_estimates = output_activation(estimates)
    else:
      activated_estimates = estimates

    loss_tensor = tf.losses.mean_squared_error(
        labels=targets,
        predictions=activated_estimates,
        weights=weights,
        reduction=tf.losses.Reduction.NONE)
  elif loss_type == "absolute_difference":
    if output_activation is not None:
      activated_estimates = output_activation(estimates)
    else:
      activated_estimates = estimates

    loss_tensor = tf.losses.absolute_difference(
        labels=targets,
        weights=weights,
        predictions=activated_estimates,
        reduction=tf.losses.Reduction.NONE)
  elif loss_type == "binary_cross_entropy":
    if output_activation is tf.nn.sigmoid:
      # Sigmoid cross entropy: sum over channels and pixels,
      #   averaged over batch and seq
      loss_tensor = tf.losses.sigmoid_cross_entropy(
          logits=estimates,
          multi_class_labels=targets,
          weights=weights,
          reduction=tf.losses.Reduction.NONE)
      activated_estimates = tf.nn.sigmoid(estimates)
    elif output_activation is None:
      loss_tensor = safe_bce_loss(estimates, targets, from_logits=False, reduction=tf.losses.Reduction.NONE)
      activated_estimates = estimates
    else:
      raise ValueError("Only `tensorflow.nn.sigmoid`"
                       "activations can be used with the "
                       "'cross-entropy' loss!")
  else:
    raise ValueError("Unknown loss type {}.!".format(loss_type))

  pixel_loss_first_frame = tf.reduce_mean(loss_tensor[0,...])
  pixel_loss = average_loss(
      loss_tensor,
      average_type=average_type,
      batch_dims=batch_dims)

  # Should pass in the dynamic range:
  if dataset_name == "moving_mnist":
    min_to_max_range = 1.0
  else:
    min_to_max_range = 2.0
  psnr, psnr_first_frame = peak_signal_to_noise_ratio(
      estimates=activated_estimates,
      targets=targets,
      min_to_max_range=min_to_max_range)

  dssim_loss, ssim, ssim_first_frame = SSIM_sequence(
      estimates=activated_estimates,
      targets=targets,
      channels=output_channels,
      L=min_to_max_range)

  loss = 0
  # TODO(drewjaegle): add gradient loss here as well
  if train_pixel_loss:
    loss += pixel_loss
  if train_ssim:
    loss += dssim_loss

  component_losses = ReconstructionLossTuple(
      pixel_loss=pixel_loss,
      dssim_loss=dssim_loss,
      psnr=psnr,
      psnr_first_frame=psnr_first_frame,
      ssim=ssim,
      ssim_first_frame=ssim_first_frame,
      pixel_loss_first_frame=pixel_loss_first_frame)

  return loss, activated_estimates, component_losses


def _setup_monitor_index():
  """Builds the structured dictionary used for populating the sess.run dict."""
  phases = ["train", "val"]
  value_types = ["loss", "scalar", "metric", "image", "hist", "sum", "fetch_no_log"]
  # loss - is used for training losses
  # metric - is averaged over the validation set, should be used for non-training losses
  # scalar - reported on the last validation batch only

  monitor_index = {}
  for phase in phases:
    monitor_index[phase] = {}
    for value_type in value_types:
      monitor_index[phase][value_type] = []

  return monitor_index

def _setup_monitor(optimizer_names, monitor_lists, combine_losses):
  """Parses a list of values and metadata into values to run or optimize.

  Args:
    monitor_lists: A list of lists of elements of MonitorTuple namedtuples, each
      giving a Tensor and associated metadata. Used to determine when and how
      to execute and monitor each Tensor.
    combine_losses: If True, losses are added together for joint optimization.
      Otherwise, they are kept distinct for seperate optimization.
  Returns:
    monitor_values: A dictionary of Tensors to monitor. These Tensors are run
      but do not control optimization.
    monitor_index: A nested dictionary specifying the phase and type of each
      Tensor in monitor_values. Determines how and when each is executed.
    opt_losses: A dictionary of Tensors that control optimization.
  """
  monitor_index = _setup_monitor_index()
  opt_losses = {}
  loss_weights = {}
  monitor_values = {}

  for key in optimizer_names:
      monitor_list = monitor_lists[key]
      monitor_index_i = _setup_monitor_index()
      for tuple_i in monitor_list:
        monitor_index_i[tuple_i.phase][tuple_i.type].append(tuple_i.name)
        monitor_index[tuple_i.phase][tuple_i.type].append(tuple_i.name)
        monitor_values[tuple_i.name] = tuple_i.value
        loss_weights[tuple_i.name] = tuple_i.weight

      # Grab training losses, combining if necessary
      if combine_losses:
        total_loss = 0
        total_loss_val = 0
        for loss_name in monitor_index_i["train"]["loss"]:
          total_loss += monitor_values[loss_name] * loss_weights[loss_name]

          # Accumulate validation near equivalent for monitoring
          val_name = "{}_val".format(loss_name)
          if val_name in monitor_index["val"]["loss"]:
            total_loss_val += monitor_values[val_name] * loss_weights[loss_name]
        # Add loss to optimize list and to monitor list
        opt_losses.update({key: {"total_loss": total_loss}})
        if key == "prediction":      # only add main loss as 'total_loss' in monitor
            monitor_index["train"]["loss"].append("total_loss")
            monitor_values["total_loss"] = total_loss
            monitor_index["val"]["loss"].append("total_loss_val")
            monitor_values["total_loss_val"] = total_loss_val
      else:
        opt_losses.update(monitor_index["train"]["loss"])
        if key == "prediction":
            monitor_values["total_loss_val"] = tf.reduce_mean(
                monitor_index["train"]["loss"])

  return monitor_values, monitor_index, opt_losses


def _pack_image_loss(
    monitor_list,
    estimates,
    targets,
    weights,
    loss_type,
    train_ssim,
    loss_weight,
    output_activation,
    output_channels,
    loss_name,
    phase,
    log_images=True,
    apply_loss=True,
    log_targets=False,
    dataset_name=None,
    only_log_final_loss=True):
  """Adds an image loss to the list of values to monitor."""

  image_loss, image_outputs, component_losses = seq_reconstruction_loss(
      estimates=estimates,
      targets=targets,
      weights=weights,
      loss_type=loss_type,
      train_ssim=train_ssim,
      output_activation=output_activation,
      output_channels=output_channels,
      dataset_name=dataset_name)

  phase_string = get_phase_string(phase)

  # Add loss
  monitor_list.append(MonitorTuple(
      value=image_loss,
      name="{}_loss{}".format(loss_name, phase_string),
      phase=phase,
      type="loss" if apply_loss else "metric",
      weight=loss_weight))

  # Add values to monitor
  if log_images:
    monitor_list.append(MonitorTuple(
        value=image_outputs,
        name="{}s{}".format(loss_name, phase_string),
        phase=phase,
        type="image",
        weight=None))
    
  # Add values to monitor
  if log_targets:
    monitor_list.append(MonitorTuple(
        value=targets,
        name="{}s{}".format(loss_name + "_target", phase_string),
        phase=phase,
        type="image",
        weight=None))

  if not only_log_final_loss:
    # Loop through the fields of component_losses
    for loss_field in component_losses._fields:
      monitor_list.append(MonitorTuple(
          value=getattr(component_losses, loss_field),
          name="{}_{}{}".format(loss_name, loss_field, phase_string),
          phase=phase,
          type="metric",
          weight=None))


def _pack_latent_loss(
    monitor_list,
    estimates,
    targets,
    loss_weight,
    loss_name,
    phase,
    weights = 1.0,
    add_loss=True,
    check_nan=False):
  """Adds a latent loss to the monitor list.
  Args
    only_summaries: if False, the loss is not added, but the histograms
    of latents are packed for the tensorboard inspection
  """

  # sanity check input dimensions
  # estimate_length = estimates.get_shape().as_list()[0]
  # target_length = targets.get_shape().as_list()[0]
  # if estimate_length != target_length:
  #     raise ValueError("Latent estimate and target need to have the same length!")

  # gather all the values for which target is not NaN (for which actions are defined)
  if check_nan:
    valid_idxs = tf.logical_not(tf.is_nan(estimates))
    estimates = tf.boolean_mask(estimates, valid_idxs)
    targets = tf.boolean_mask(targets, valid_idxs)
  
  latent_loss = tf.losses.mean_squared_error(
      labels=targets,
      predictions=estimates,
      weights=weights)

  phase_string = get_phase_string(phase)

  if add_loss:
    monitor_list.append(MonitorTuple(
        value=latent_loss,
        name="{}_loss{}".format(loss_name, phase_string),
        phase=phase,
        type="loss",
        weight=loss_weight))
  monitor_list.append(MonitorTuple(
      value=estimates,
      name="{}s_est{}".format(loss_name, phase_string),
      phase=phase,
      type="hist",
      weight=None))
  monitor_list.append(MonitorTuple(
      value=targets,
      name="{}s_true{}".format(loss_name, phase_string),
      phase=phase,
      type="hist",
      weight=None))


def _pack_bce_loss(
        monitor_list,
        estimates_list,
        targets_list,
        loss_weight,
        loss_name,
        phase,
        weights_list,
        from_logits=False):
    """Adds a binary cross-entropy loss to the monitor list.
    Args
      only_summaries: if False, the loss is not added, but the histograms
      of latents are packed for the tensorboard inspection
    """

    bce_losses = []
    for (target, estimate, weight) in zip(targets_list, estimates_list, weights_list):
        bce_losses.append(safe_bce_loss(estimate, target, weight, from_logits, reduction="custom"))
    bce_loss = tf.reduce_mean(tf.stack(bce_losses))

    phase_string = get_phase_string(phase)

    monitor_list.append(MonitorTuple(
        value=bce_loss,
        name="{}_loss{}".format(loss_name, phase_string),
        phase=phase,
        type="loss",
        weight=loss_weight))


def _pack_kldivergence_loss(
    monitor_list,
    estimate,
    target,
    loss_weight,
    loss_name,
    phase,
    weights=1.0):
  """Adds a KL divergence loss between two distributions to the monitor list.
    :argument
      estimate/target: Tensor with first half of last dimension representing
        means, second half standard deviations of Gaussian distributions.
      estimate: estimate distribution, often denoted Q
      target: true distribution, often denoted P
  """

  if phase == "val":
    # In this case the targets (inference) might be missing some or all steps
    target_duration = target.shape.as_list()[0]

    if target_duration == 0:
      return
    if target_duration != estimate.shape.as_list()[0]:
      estimate = target[0:target_duration]
 
  # sanity check input dimensions
  segments = estimate.get_shape().as_list()[0]
  estimate_dim = estimate.get_shape().as_list()[-1]
  target_dim = target.get_shape().as_list()[-1]
  # sanity check that even number of components
  if estimate_dim % 2 != 0:
    raise ValueError("Latent for Normal distribution needs to have even number of elements!")
  if estimate_dim != target_dim:
    raise ValueError("Latent dimensions for estimated and ground truth values must be the same!")

  # separate means and variances
  p_mean = target[..., :int(target_dim / 2)]
  p_log_sig = target[..., int(target_dim / 2):]
  q_mean = estimate[..., :int(estimate_dim / 2)]
  q_log_sig = estimate[..., int(estimate_dim / 2):]

  # compute KL divergence for two Gaussian distributions with exponentiated variances
  # variance is exponentiated to transform [-1,1] -> [0,1] range
  # do the substitution in the equation to avoid log(exp)
  if FLAGS.use_old_kl:
      divergence = (q_log_sig - p_log_sig) + (tf.pow(tf.exp(p_log_sig), 2) + tf.pow(p_mean - q_mean, 2)) \
                   / (2 * tf.pow(tf.exp(q_log_sig), 2)) - 0.5
  else:
      divergence = (p_log_sig - q_log_sig) + (tf.pow(tf.exp(q_log_sig), 2) + tf.pow(q_mean - p_mean, 2)) \
                      / (2* tf.pow(tf.exp(p_log_sig), 2)) - 0.5
  divergence_loss = tf.losses.compute_weighted_loss(divergence, weights, reduction=tf.losses.Reduction.NONE)

  phase_string = get_phase_string(phase)

  monitor_list.append(MonitorTuple(
    value=tf.reduce_mean(divergence_loss),
    name="{}_loss{}".format(loss_name, phase_string),
    phase=phase,
    type="loss",
    weight=loss_weight))
  for i in range(segments):
    monitor_list.append(MonitorTuple(
      value=tf.reduce_mean(divergence_loss[i]),
      name="{}_loss{}/{}".format(loss_name, phase_string,i),
      phase=phase,
      type="metric",
      weight=loss_weight))

  monitor_list.append(MonitorTuple(
    value=q_mean,
    name="{}s_prior_mean{}".format(loss_name, phase_string),
    phase=phase,
    type="hist",
    weight=None))
  monitor_list.append(MonitorTuple(
    value=tf.exp(q_log_sig),
    name="{}s_prior_stdDev{}".format(loss_name, phase_string),
    phase=phase,
    type="hist",
    weight=None))
  monitor_list.append(MonitorTuple(
    value=p_mean,
    name="{}s_inference_mean{}".format(loss_name, phase_string),
    phase=phase,
    type="hist",
    weight=None))
  monitor_list.append(MonitorTuple(
    value=tf.exp(p_log_sig),
    name="{}s_inference_stdDev{}".format(loss_name, phase_string),
    phase=phase,
    type="hist",
    weight=None))


def _pack_action_loss(
        monitor_list,
        estimate,
        target,
        angles,
        loss_weight,
        loss_name,
        phase):
    """Computes MSE between action estimate and targets only for samples for which
        targets are not NaN.
    """
    # sanity check input dimensions
    estimate_length = estimate.get_shape().as_list()[0]
    target_length = target.get_shape().as_list()[0]
    if estimate_length != target_length:
        raise ValueError("Action estimate and target need to have the same length!")

    if tf.flags.FLAGS.ignore_lift_action:
        action_dim = target.get_shape().as_list()[2]
        if (action_dim != 4) and (action_dim != 3):
            raise ValueError("Ignore Lift Action is only possible on the BAIR dataset. "
                             "Action dimension of %d does not fit BAIR dim 4!" % action_dim)
        if action_dim == 4:
          target, estimate = target[..., :-2], estimate[..., :-2]

    # gather all the values for which target is not NaN (for which actions are defined)
    valid_idxs = tf.logical_not(tf.is_nan(target))
    estimate = tf.boolean_mask(estimate, valid_idxs)
    target = tf.boolean_mask(target, valid_idxs)

    difference = tf.abs(estimate - target)
    if angles:
      reverse_difference = 2 * np.pi - tf.boolean_mask(difference, difference > np.pi)
      direct_difference = tf.boolean_mask(difference, difference <= np.pi)
      combined_difference = tf.concat([reverse_difference, direct_difference], axis=0)
    else:
      combined_difference = difference

    def l2(x):
      return tf.reduce_mean(tf.pow(x,2))

    action_loss = l2(combined_difference)

    # compute MSE loss
    # action_loss = tf.losses.mean_squared_error(target, estimate)

    phase_string = get_phase_string(phase)

    monitor_list.append(MonitorTuple(
        value=action_loss,
        name="{}_loss{}".format(loss_name, phase_string),
        phase=phase,
        type="loss",
        weight=loss_weight))
    monitor_list.append(MonitorTuple(
        value=combined_difference,
        name="{}_diffs{}".format(loss_name, phase_string),
        phase=phase,
        type="hist",
        weight=None))


def _pack_f1_loss(monitor_list,
                    estimate,
                    target,
                    loss_name,
                    phase):
    """Computes F1 score between estimated and target keyframe indexes, registers as metric in TB."""
    estimate, target = tf.cast(estimate, dtype=tf.bool), tf.cast(target, dtype=tf.bool)
    def sum_int(x):
        return tf.to_float(tf.reduce_sum(tf.cast(x, tf.int32)))

    true_positives = sum_int(tf.math.logical_and(target, estimate))
    true_negatives = sum_int(tf.math.logical_and(tf.math.logical_not(target), (tf.math.logical_not(estimate))))

    false_positives = sum_int(tf.math.logical_and(tf.math.logical_not(target), estimate))
    false_negatives = sum_int(tf.math.logical_and(target, (tf.math.logical_not(estimate))))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1_score = 2 * precision * recall / (precision + recall + tf.constant(1e-12))

    phase_string = get_phase_string(phase)
    monitor_list.append(MonitorTuple(
        value=f1_score,
        name="{}_score{}".format(loss_name, phase_string),
        phase=phase,
        type="metric",
        weight=1.0))


def build_normal_dist(latent):
  """
  Constructs a distributions.Normal object from an RNN latent prediction.
  :param latent: An RNN-predicted latent, first half is mean, second half variance.
  :return: Normal distribution object.
  """
  latent_dim = latent.get_shape().as_list()[2]
  # sanity check that even number of components
  if latent_dim % 2 != 0:
    raise ValueError("Latent for Normal distribution needs to have even number of elements!")

  return distributions.Normal(loc=latent[:,:,:latent_dim/2], scale=latent[:,:,latent_dim/2:])


def _monitor_entropy(monitor_list, dists, name, phase):
  num_segments, batch_size, num_segment_frames = dists.get_shape().as_list()
  entropy = safe_entropy(dists, axis=2)

  # compute entropy of keyframe idx, use batch to approximate prob distribution
  max_idxs = tf.reduce_mean(tf.one_hot(tf.argmax(dists, axis=-1), depth=num_segment_frames), axis=1)
  max_idx_entropy = safe_entropy(max_idxs, axis=-1)

  phase_string = get_phase_string(phase)
  
  monitor_list.append(MonitorTuple(tf.reduce_mean(entropy),
                                   name + "_avg_entropy" + phase_string, phase, "metric", 0))
  for segment in range(num_segments):
      monitor_list.append(MonitorTuple(tf.reduce_mean(entropy[segment]),
                                       name + "_avg_entropy/seg_%d" % (segment+1) + phase_string, phase, "metric", 0))

  monitor_list.append(MonitorTuple(tf.reduce_mean(max_idx_entropy),
                                   name + "_avg_maxIdx_entropy" + phase_string, phase, "metric", 0))
  for segment in range(num_segments):
      monitor_list.append(MonitorTuple(max_idx_entropy[segment],
                                       name + "_avg_maxIdx_entropy/seg_%d" % (segment + 1) + phase_string, phase, "metric", 0))
    
  # Montitor averages of individual frame dts to see if a specific frame is never produced
  for frame in range(num_segment_frames):
      monitor_list.append(MonitorTuple(tf.reduce_mean(dists[:,:,frame]),
                                       name + "_avg/frame_%d" % (frame + 1) + phase_string, phase, "metric", 0))


def _compute_gt_dt_targets(gt_dts, segment_length, num_segments, n_loss_frames):
  perm_gt_dts = tf.transpose(gt_dts)
  idxs = tf.cast(tf.map_fn(lambda x: tf.cast(tf.where(tf.not_equal(x, 0))[:num_segments], tf.float32), perm_gt_dts)[..., 0], tf.int32)
  target_distributions, within_lossregion = [], []
  for segment in range(num_segments):
    target = tf.one_hot(idxs[:, segment], depth=(segment+1) * segment_length)
    target_distributions.append(target)
    within_lossregion.append(tf.cast(tf.less(idxs[:, segment], n_loss_frames), dtype=tf.float32))
  return target_distributions, within_lossregion

  
def compute_losses(input_data_train,
                   input_data_val,
                   model_output_train,
                   model_output_val,
                   regularizer_weight,
                   seq_len_loss_weight,
                   loss_spec,
                   output_channels,
                   optimizer_names,
                   dataset_name,
                   predictions_preactivated,
                   has_image_input):
  """Computes losses and other values being monitored during train/val.

  Args:
    input_data_train: A dictionary of training input Tensors.
    input_data_val: A dictionary of validation input Tensors.
    model_output_train: A dictionary of training output Tensors, with fields
      corresponding to those in input_data_train.
    model_output_val: A dictionary of validation output Tensors, with fields
      corresponding to those in input_data_val.
    pp_lambda: The weight for the predictive potential loss.
    regularizer_weight: The weight for the regularizer (e.g. L2)
    loss_spec: A namedtuple specifying the loss configuration for training.
    output_channels: The number of output channels in the image.
    compute_predictive_potential: If True, adds PP-based losses to the graph.
  Returns:
    monitor_losses_train: A dictionary of losses to monitor during training.
    monitor_losses_val: A dictionary of losses to monitor during validation.
    opt_losses: A dictionary of losses to optimize. Each element will be
      optimized by a different optimizer.
  """

  monitor_lists = dict((name, []) for name in optimizer_names)

  monitor_lists['regularization'] = [] # Add regularization to the monitor list but not the optimizer

  if predictions_preactivated:
    prediction_activation = None
  else:
    prediction_activation = loss_spec.image_output_activation

  # compute number of segments that are considered for loss
  n_frames_reconstruct, batch_size = shape(input_data_train['reconstruct'])[0:2]
  n_loss_frames = th_utils.get_future_loss_length()

  for phase in ["train", "val"]:
    if phase == "train":
      model_output = model_output_train
      input_data = input_data_train
    else:
      model_output = model_output_val
      input_data = input_data_val

    with tf.name_scope(phase + "losses"):
      _monitor_entropy(monitor_lists["prediction"],
                       model_output["high_level_rnn_output_dt"],
                       name="dt",
                       phase=phase, )

      soft_latent_targets, high_level_weights, propagated_distributions = \
        model_output["predicted"]["soft_latent_targets"], model_output["predicted"]["high_level_weights"], \
        model_output["predicted"]["propagated_distributions"]

      soft_image_targets = th_utils.get_high_level_targets(
        high_level_weights,
        propagated_distributions,
        input_data["predict"])

      # normalize loss with fraction of prob distribution within cutoff/loss range
      # this prevents network from pushing keyframes out of loss region to avoid incurring loss
      if FLAGS.norm_high_level_loss:
        with tf.name_scope('loss_normalization'):
          prob_weigths_in = tf.add_n([tf.reduce_sum(dist[:, :min(shape(dist)[1], n_loss_frames)], axis=1)
                                      for dist in propagated_distributions])   # prob. weight within the loss region
          loss_norm_factor = tf.divide(FLAGS.n_segments, prob_weigths_in)
          high_level_weights = tf.multiply(high_level_weights, loss_norm_factor[None, :, None])
          monitor_lists['prediction'].append(MonitorTuple(value=loss_norm_factor,
                                                          name="high_level_loss_norm_factor" + get_phase_string(phase),
                                                          phase=phase,
                                                          type="hist",
                                                          weight=1.0))
      if FLAGS.high_level_image_term > 0:
          _pack_image_loss(
            monitor_lists["prediction"],
            estimates=model_output["decoded_keyframes"],
            targets=soft_image_targets,
            weights=high_level_weights[:,:,:,None,None],
            loss_type=loss_spec.image_loss_type,
            train_ssim=loss_spec.train_ssim,
            loss_weight=tf.flags.FLAGS.high_level_image_term,
            output_activation=prediction_activation,
            output_channels=output_channels,
            loss_name="high_level_image",
            phase=phase,
            apply_loss=has_image_input,
            dataset_name=dataset_name)

      if FLAGS.high_level_latent_term > 0:
          if not FLAGS.train_hl_latent_swap:
            estimates = model_output["high_level_rnn_output_keyframe"]
            targets = soft_latent_targets
          else:
            estimates = model_output["high_level_rnn_output_keyframe"]
            targets = model_output["predicted"]["hl_swapped_frames"]
            # TODO the weights should be also changed here

          _pack_latent_loss(
            monitor_lists["prediction"],
            estimates=estimates,
            targets=targets,
            weights=high_level_weights,
            loss_weight=tf.flags.FLAGS.high_level_latent_term,
            loss_name="high_level_latent",
            phase=phase)

      # Entropy loss
      # dists = tf.distributions.Categorical(probs=model_output["high_level_rnn_output_dt"])
      # estimates = tf.contrib.bayesflow.entropy.entropy_shannon(dists)
      if FLAGS.entropy_term > 0:
          dists = model_output["high_level_rnn_output_dt"]
          estimates = safe_entropy(dists, axis=-1)
          targets = tf.zeros(estimates.get_shape().as_list())
          _pack_latent_loss(
            monitor_lists["prediction"],
            estimates=estimates,
            targets=targets,
            loss_weight=tf.flags.FLAGS.entropy_term,
            loss_name="entropy",
            phase=phase)

      # Sequence length loss
      if FLAGS.sequence_length_term > 0:
          estimates = propagated_distributions[-1][:, :n_loss_frames]
          targets = tf.zeros(estimates.get_shape().as_list())
          _pack_latent_loss(
            monitor_lists["prediction"],
            estimates=estimates,
            targets=targets,
            loss_weight=seq_len_loss_weight,
            loss_name="sequence_ends_mse",
            phase=phase)

      # Supervise dt
      if FLAGS.supervise_dt_term > 0.0:
        predict_gt_keyframe_idxs = \
            input_data["actions_abs"][-(n_loss_frames+FLAGS.n_frames_segment): -FLAGS.n_frames_segment, :, 0]
        target_dt_dists, dt_sup_weights = _compute_gt_dt_targets(predict_gt_keyframe_idxs,
                                                                 segment_length=FLAGS.n_frames_segment,
                                                                 num_segments=len(propagated_distributions),
                                                                 n_loss_frames=n_loss_frames)
        _pack_bce_loss(
                monitor_lists["prediction"],
                estimates_list=propagated_distributions,
                targets_list=target_dt_dists,
                loss_weight=FLAGS.supervise_dt_term,
                loss_name="supervise_dt",
                phase=phase,
                weights_list=dt_sup_weights)

      # Low-level loss
      assert not (FLAGS.low_level_image_term > 0.0 and FLAGS.gt_target_loss_term > 0.0), \
                "Cannot have both low level image terms!"
      if FLAGS.low_level_image_term > 0.0:
          # NOTE that the loss with weights is not equivalent to the MSE that people normally report
          # TODO add a proper validation metric with hard decisions
          img_targets, img_weights, coord_targets = th_utils.get_low_level_targets(
            model_output["high_level_rnn_output_dt"],
            input_data["predict"],
            propagated_distributions,
            coord_targets=None if has_image_input else input_data["predict_coord"])

          _pack_image_loss(
              monitor_lists["prediction"],
              estimates=model_output["decoded_low_level_frames"],
              targets=img_targets,
              weights=img_weights,
              loss_type=loss_spec.image_loss_type,
              train_ssim=loss_spec.train_ssim,
              loss_weight=tf.flags.FLAGS.low_level_image_term,
              output_activation=prediction_activation,
              output_channels=output_channels,
              loss_name="low_level_image",
              phase=phase,
              apply_loss=has_image_input,
              log_targets=True,
              dataset_name=dataset_name)
          if not has_image_input:
              _pack_latent_loss(
                  monitor_lists["prediction"],
                  estimates=model_output["decoded_low_level_coords"],
                  targets=coord_targets,
                  weights=img_weights[:, :, :, 0, 0],
                  loss_weight=tf.flags.FLAGS.low_level_image_term,
                  loss_name="low_level_coord",
                  phase=phase)

      # GT target loss
      if FLAGS.gt_target_loss_term > 0.0:
          predicted = model_output["decoded_low_level_frames"]
          if FLAGS.activate_before_averaging and prediction_activation is not None:
              predicted = prediction_activation(predicted)

          if FLAGS.predict_to_the_goal:
            if "goal_timestep" not in input_data.keys():
              goal_timestep = tf.ones(batch_size, dtype=tf.int32) * (n_loss_frames - 1)
            else:
              goal_timestep = tf.minimum(input_data.goal_timestep[0] - n_frames_reconstruct, n_loss_frames-1)
              goal_timestep = tf.Print(goal_timestep, [goal_timestep], "goal_timestep", summarize=20)
            goal_timestep =tf.one_hot(goal_timestep, n_loss_frames, axis=0)
            gtt_weights = tf.cumsum(goal_timestep, reverse=True)
            gtt_weights = tf.Print(gtt_weights, [tf.reduce_sum(goal_timestep)], "weights", summarize=20)
          else:
            gtt_weights = tf.ones((n_loss_frames, batch_size), dtype=tf.float32)
            # TODO this is broken with sigmoid

          fs_seqwise_distributions = th_utils.dists_keyframe_to_first_segment(propagated_distributions, n_loss_frames)
          gt_soft_targets = th_utils.get_low_level_gt_targets(
              model_output["high_level_rnn_output_dt"],
              predicted,
              fs_seqwise_distributions,
              n_loss_frames)

          # manually add predicted sequence images
          pred_low_level = prediction_activation(model_output["decoded_low_level_frames"]) \
              if prediction_activation is not None else model_output["decoded_low_level_frames"]
          monitor_lists["prediction"].append(MonitorTuple(
              value=pred_low_level,
              name="low_level_images" + get_phase_string(phase),
              phase=phase,
              type="image",
              weight=None))
          
          _pack_image_loss(
              monitor_lists["prediction"],
              estimates=gt_soft_targets,
              targets=input_data["predict"],
              weights=gtt_weights[:, :, None, None, None],
              loss_type=loss_spec.image_loss_type,
              train_ssim=loss_spec.train_ssim,
              loss_weight=tf.flags.FLAGS.gt_target_loss_term,
              output_activation=prediction_activation if not FLAGS.activate_before_averaging else None,
              output_channels=output_channels,
              loss_name="gt_target_low_level_image",
              phase=phase,
              apply_loss=has_image_input,
              log_targets=True,
              dataset_name=dataset_name)
          
          if FLAGS.decode_actions:
            ll_pred_actions = model_output.predicted.low_level_rnn_output_actions
            ll_gt_action_targets = th_utils.get_low_level_gt_targets(
                model_output["high_level_rnn_output_dt"],
                ll_pred_actions,
                fs_seqwise_distributions,
                n_loss_frames)
            ll_action_true = input_data["actions"][(n_frames_reconstruct - 1):
                                                   (n_frames_reconstruct - 1 + n_loss_frames)]
            
            _pack_latent_loss(
              monitor_lists["prediction"],
              estimates=ll_gt_action_targets,
              targets=ll_action_true,
              weights=gtt_weights[:, :, None],
              loss_weight=tf.flags.FLAGS.ll_actions_term,
              loss_name="ll_action_regression",
              phase=phase)

      # KL-divergence:
      match_var_distributions = tf.flags.FLAGS.kl_divergence_weight
      if match_var_distributions:
        # encoder loss, discard first image
        # predictor loss
        _pack_kldivergence_loss(
          monitor_lists["prediction"],
          estimate=model_output["prior_dists"],
          target=model_output["inference_dists"],
          weights=high_level_weights,
          loss_weight=match_var_distributions,
          loss_name="kl_divergence",
          phase=phase)

      # Low-level KL divergence vs fixed uniform prior
      if FLAGS.ll_kl_term > 0:
        num_seg, _, ll_latent_dim = model_output["predicted"]["ll_inf_dists"].get_shape().as_list()
        dist_dtype = model_output["predicted"]["ll_inf_dists"].dtype
        ll_kl_target = utils.get_fixed_prior((num_seg, batch_size, ll_latent_dim), dist_dtype)
        ll_kl_estimate = model_output["predicted"]["ll_inf_dists"]
        _pack_kldivergence_loss(
          monitor_lists["prediction"],
          estimate=ll_kl_estimate,
          target=ll_kl_target,
          weights=1.0,
          loss_weight=FLAGS.ll_kl_term,
          loss_name="ll_kl_divergence",
          phase=phase)

      if FLAGS.supervise_attention_term > 0:
        if FLAGS.use_full_inf:
          if FLAGS.inference_backwards:
            # Shift targets by one
            targets = model_output["oh_keyframe_idxs"][:,:-1]
            estimates = tf.transpose(model_output["attention_weights"], [2, 0, 1])[:,1:]
          else:
            targets = model_output["oh_keyframe_idxs"]
            estimates = tf.transpose(model_output["attention_weights"], [2, 0, 1])
          _pack_bce_loss(
            monitor_lists["prediction"],
            estimates_list=[estimates],
            targets_list=[targets],
            loss_weight=FLAGS.supervise_attention_term,
            loss_name="supervise_attention",
            weights_list=[1],
            phase=phase)
        else:
          raise ValueError("Supervised attention only works with full inference")
        
      if FLAGS.train_action_regressor:
        targets_list = [input_data["actions"][n_frames_reconstruct-1]]
        n_loss_keyframes = int(np.floor(n_loss_frames / float(FLAGS.n_frames_segment))) + 1
        for i in range(1, n_loss_keyframes):
          idcs = tf.argmax(propagated_distributions[i-1], axis=1, output_type=tf.int32) + n_frames_reconstruct
          targets = tf.batch_gather(tf.transpose(input_data["actions"], [1, 0, 2]), idcs[:,None])[:,0]
          targets_list.append(targets)
        estimates = model_output["regressed_actions"][:n_loss_keyframes]
        targets = tf.stack(targets_list, axis=0)
        
        _pack_latent_loss(
          monitor_lists["prediction"],
          estimates=estimates,
          targets=targets,
          loss_weight=1,
          loss_name="action_regression",
          phase=phase)
  
      # Phase-specific losses
      if phase == "train" and not FLAGS.pretrain_ll:
        regularization_loss = sum(tf.get_collection(
          tf.GraphKeys.REGULARIZATION_LOSSES))
        monitor_lists["regularization"].append(MonitorTuple(regularization_loss,
                                                        "regularization_loss",
                                                        "train", "loss",
                                                        1))

  monitor_values, monitor_index, opt_losses = _setup_monitor(
      optimizer_names,
      monitor_lists,
      loss_spec.combine_losses)

  return monitor_values, monitor_index, opt_losses


def compute_non_hierarchical_losses(input_data_train,
                                   input_data_val,
                                   model_output_train,
                                   model_output_val,
                                   regularizer_weight,
                                   seq_len_loss_weight,
                                   loss_spec,
                                   output_channels,
                                   optimizer_names,
                                   dataset_name,
                                   predictions_preactivated,
                                   has_image_input):
    """Computes losses and other values being monitored during train/val.

    Args:
      input_data_train: A dictionary of training input Tensors.
      input_data_val: A dictionary of validation input Tensors.
      model_output_train: A dictionary of training output Tensors, with fields
        corresponding to those in input_data_train.
      model_output_val: A dictionary of validation output Tensors, with fields
        corresponding to those in input_data_val.
      pp_lambda: The weight for the predictive potential loss.
      regularizer_weight: The weight for the regularizer (e.g. L2)
      loss_spec: A namedtuple specifying the loss configuration for training.
      output_channels: The number of output channels in the image.
      compute_predictive_potential: If True, adds PP-based losses to the graph.
    Returns:
      monitor_losses_train: A dictionary of losses to monitor during training.
      monitor_losses_val: A dictionary of losses to monitor during validation.
      opt_losses: A dictionary of losses to optimize. Each element will be
        optimized by a different optimizer.
    """

    monitor_lists = dict((name, []) for name in optimizer_names)

    monitor_lists['regularization'] = []  # Add regularization to the monitor list but not the optimizer

    if predictions_preactivated:
        prediction_activation = None
    else:
        prediction_activation = loss_spec.image_output_activation

    # compute number of segments that are considered for loss
    n_frames_reconstruct = input_data_train['reconstruct'].get_shape().as_list()[0]

    for phase in ["train", "val"]:
        if phase == "train":
            model_output = model_output_train
            input_data = input_data_train
        else:
            model_output = model_output_val
            input_data = input_data_val

        with tf.name_scope(phase + "losses"):
            # reconstruction loss
            _pack_image_loss(
                monitor_lists["prediction"],
                estimates=model_output["decoded_low_level_frames"],
                targets=input_data["predict"],
                weights=1.0,
                loss_type=loss_spec.image_loss_type,
                train_ssim=loss_spec.train_ssim,
                loss_weight=tf.flags.FLAGS.low_level_image_term,
                output_activation=prediction_activation,
                output_channels=output_channels,
                loss_name="low_level_image",
                phase=phase,
                apply_loss=has_image_input,
                log_targets=True,
                dataset_name=dataset_name)
            if not has_image_input:
                _pack_latent_loss(
                    monitor_lists["prediction"],
                    estimates=model_output["decoded_low_level_coords"],
                    targets=input_data["predict_coord"],
                    weights=1.0,
                    loss_weight=tf.flags.FLAGS.low_level_image_term,
                    loss_name="low_level_coord",
                    phase=phase)

            # KL-divergence:
            match_var_distributions = tf.flags.FLAGS.kl_divergence_weight
            if match_var_distributions and "inference_dists" in model_output:
                # encoder loss, discard first image
                # predictor loss
                _pack_kldivergence_loss(
                    monitor_lists["prediction"],
                    estimate=model_output["prior_dists"],
                    target=model_output["inference_dists"],
                    weights=1.0,
                    loss_weight=match_var_distributions,
                    loss_name="kl_divergence",
                        phase=phase)

            if FLAGS.train_action_regressor:
                _pack_latent_loss(
                    monitor_lists["abs_action"],
                    estimates=model_output["regressed_actions"],
                    targets=input_data["actions"][(n_frames_reconstruct):],
                    loss_weight=1,
                    loss_name="action_regression",
                    phase=phase)

                _pack_latent_loss(
                    monitor_lists["abs_action"],
                    estimates=model_output["regressed_actions_z"],
                    targets=input_data["actions"][(n_frames_reconstruct):],
                    loss_weight=1,
                    loss_name="z_action_regression",
                    phase=phase)

            if "inference_dists_reencode" in model_output:
                # in this case we determine the keyframe indexes from kl divergence of inference dists
                from architectures.sssp import StochasticSingleStepPredictorKFDetect as SSSP
                _, estimated_kf_idxs, _ = SSSP.get_high_kl_keyframes(model_output["decoded_low_level_frames"],
                                                                  model_output["inference_dists"],
                                                                  model_output["prior_dists"])
                target_kf_idxs = input_data["actions_abs"][n_frames_reconstruct:, :, 0]
                _pack_f1_loss(
                    monitor_lists["prediction"],
                    estimate=estimated_kf_idxs,
                    target=target_kf_idxs,
                    loss_name="f1",
                    phase=phase)

                _, estimated_kf_idxs_reenc, _ = SSSP.get_high_kl_keyframes(model_output["decoded_low_level_frames"],
                                                                        model_output["inference_dists_reencode"],
                                                                        model_output["prior_dists"])
                _pack_f1_loss(
                    monitor_lists["prediction"],
                    estimate=estimated_kf_idxs_reenc,
                    target=target_kf_idxs,
                    loss_name="f1_reenc",
                    phase=phase)

            # Phase-specific losses
            if phase == "train" and not FLAGS.pretrain_ll:
                regularization_loss = sum(tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES))
                monitor_lists["regularization"].append(MonitorTuple(regularization_loss,
                                                                    "regularization_loss",
                                                                    "train", "loss",
                                                                    1))

    monitor_values, monitor_index, opt_losses = _setup_monitor(
        optimizer_names,
        monitor_lists,
        loss_spec.combine_losses)

    return monitor_values, monitor_index, opt_losses


def configure_optimization(
    input_train,
    input_val,
    model_output_train,
    model_output_val,
    learning_rates,
    tau,
    regularizer_weight,
    seq_len_loss_weight,
    loss_spec,
    use_gan,
    generator_scope,
    max_grad_norm,
    optimizer_epsilon,
    data_spec_train,
    dataset_name,
    global_step,
    optimizer_type="adam",
    predictions_preactivated=False,
    has_image_input=True,
    is_hierarchical=False):
  """Configures optimization and monitoring on the appropriate losses."""

  if is_hierarchical:
      monitor_values, monitor_index, opt_losses = compute_losses(
          input_train,
          input_val,
          model_output_train,
          model_output_val,
          regularizer_weight,
          seq_len_loss_weight,
          loss_spec,
          data_spec_train.channels,
          optimizer_names=learning_rates.keys(), # define optimizer names
          dataset_name=dataset_name,
          predictions_preactivated=predictions_preactivated,
          has_image_input=has_image_input)
  else:
      monitor_values, monitor_index, opt_losses = compute_non_hierarchical_losses(
          input_train,
          input_val,
          model_output_train,
          model_output_val,
          regularizer_weight,
          seq_len_loss_weight,
          loss_spec,
          data_spec_train.channels,
          optimizer_names=learning_rates.keys(),  # define optimizer names
          dataset_name=dataset_name,
          predictions_preactivated=predictions_preactivated,
          has_image_input=has_image_input)

  def monitor(tb_name, type, dict_name=None):
      if dict_name is None:
          dict_name = tb_name

      monitor_values[tb_name] = model_output_train[dict_name]
      monitor_index["train"][type].append(tb_name)

      monitor_values[tb_name + "_val"] = model_output_val[dict_name]
      monitor_index["val"][type].append(tb_name + "_val")

  has_encoder = tf.flags.FLAGS.input_seq_len != 1

  # Package additional Tensors for monitoring
  monitor_values["input_images"] = input_train["reconstruct"]
  monitor_index["train"]["image"].append("input_images")

  monitor_values["input_images_val"] = input_val["reconstruct"]
  monitor_index["val"]["image"].append("input_images_val")

  monitor_values["predict_images"] = input_train["predict"]
  monitor_index["train"]["image"].append("predict_images")

  monitor_values["predict_images_val"] = input_val["predict"]
  monitor_index["val"]["image"].append("predict_images_val")

  monitor_values["actions"] = input_train["actions"]
  monitor_index["train"]["image"].append("actions")

  monitor_values["actions_val"] = input_val["actions"]
  monitor_index["val"]["image"].append("actions_val")

  if "regressed_actions" in model_output_train:
      monitor_values["regressed_actions"] = model_output_train["regressed_actions"]
      monitor_index["train"]["image"].append("regressed_actions")

      monitor_values["regressed_actions_val"] = model_output_val["regressed_actions"]
      monitor_index["val"]["image"].append("regressed_actions_val")

  monitor_values["actions_abs"] = input_train["actions_abs"]
  monitor_index["train"]["image"].append("actions_abs")

  monitor_values["actions_abs_val"] = input_val["actions_abs"]
  monitor_index["val"]["image"].append("actions_abs_val")

  if is_hierarchical:
      if not predictions_preactivated:
        monitor_values["high_level_images"] = loss_spec.image_output_activation(model_output_train["decoded_keyframes"])
      else:
        monitor_values["high_level_images"] = model_output_train["decoded_keyframes"]
      monitor_index["train"]["image"].append("high_level_images")

      if not predictions_preactivated:
        monitor_values["high_level_images_val"] = loss_spec.image_output_activation(model_output_val["decoded_keyframes"])
      else:
        monitor_values["high_level_images_val"] = model_output_val["decoded_keyframes"]
      monitor_index["val"]["image"].append("high_level_images_val")

      monitor(tb_name="attention_weights", type="image")
      monitor(tb_name="dt", type="hist", dict_name="high_level_rnn_output_dt")

      monitor_values["sequence_length_loss_weight"] = seq_len_loss_weight
      monitor_index["train"]["scalar"].append("sequence_length_loss_weight")


  monitor_values["learning_rate"] = learning_rates["prediction"]
  monitor_index["train"]["scalar"].append("learning_rate")

  monitor_values["tau"] = tau
  monitor_index["train"]["scalar"].append("tau")

  if "decoded_seq_predict_pri" in model_output_train:
    monitor(tb_name="decoded_seq_predict_pri", type="image")

  # monitor distribution values on validation set
  if "prior_dists_encoder" in model_output_val:
      # Todo(oleh) this is unneded. _pack_kl_div already logs (or should) these values
      if has_encoder:
          monitor_values["prior_dists_encoder"] = model_output_val["prior_dists_encoder"]
          monitor_index["val"]["image"].append("prior_dists_encoder")

          monitor_values["inference_dists_encoder"] = model_output_val["inference_dists_encoder"]
          monitor_index["val"]["image"].append("inference_dists_encoder")

      monitor_values["prior_dists_predictor"] = model_output_val["prior_dists_predictor"]
      monitor_index["val"]["image"].append("prior_dists_predictor")

      monitor_values["inference_dists_predictor"] = model_output_val["inference_dists_predictor"]
      monitor_index["val"]["image"].append("inference_dists_predictor")

      monitor_values["prior_dists_predictor_tf"] = model_output_val["prior_dists_predictor_tf"]
      monitor_index["val"]["image"].append("prior_dists_predictor_tf")

      monitor_values["inference_z_samples"] = model_output_train["inference_z_samples"]
      monitor_index["train"]["hist"].append("inference_z_samples")

      monitor_values["inference_z_samples_val"] = model_output_val["inference_z_samples"]
      monitor_index["val"]["fetch_no_log"].append("inference_z_samples_val")

      monitor_values["inference_z_means_val"] = model_output_val["inference_z_means"]
      monitor_index["val"]["fetch_no_log"].append("inference_z_means_val")
      
      monitor_values["inference_z_stds_val"] = model_output_val["inference_z_stds"]
      monitor_index["val"]["fetch_no_log"].append("inference_z_stds_val")

  if "inference_dists_reencode" in model_output_train:
      # in this case we determine the keyframe indexes from kl divergence of inference dists
      from architectures.sssp import StochasticSingleStepPredictorKFDetect as SSSP
      def register_kfs(model_output, phase, name):
          s = "_val" if phase == "val" else ""
          for suffix in ["", "_reencode"]:
              kfs, kf_idxs, kl = SSSP.get_high_kl_keyframes(model_output["decoded_low_level_frames"],
                                                            model_output["inference_dists"+suffix],
                                                            model_output["prior_dists"])
              monitor_values[name+suffix+s] = loss_spec.image_output_activation(kfs) if not predictions_preactivated else kfs
              monitor_index[phase]["image"].append(name+suffix+s)
              monitor_values[name+suffix+"_idxs"+s] = kf_idxs
              monitor_index[phase]["image"].append(name+suffix+"_idxs"+s)
              monitor_values[name + suffix + "_kl" + s] = kl
              monitor_index[phase]["image"].append(name + suffix + "_kl" + s)
      register_kfs(model_output_train, "train", "kl_based_kfs")
      register_kfs(model_output_val, "val", "kl_based_kfs")

  if use_gan:
    opt_steps = None
    # Todo(oleh) merge train_ops_gan into opt_steps
    train_ops_gan, train_steps_gan = gan_losses.build_gan_specs(
        model_output_train,
        model_output_val,
        loss_spec,
        input_train,
        input_val,
        learning_rates["prediction"],
        regularizer_weight,
        opt_losses["prediction"],
        generator_scope,
        monitor_values,
        monitor_index)

  else:
    opt_steps = dict()

    [opt_steps.update(optimize(opt_losses[key],
                               learning_rates[key],
                               max_grad_norm,
                               optimizer_epsilon,
                               key,
                               global_step,
                               optimizer_type=optimizer_type if key != "regularization" else "sgd"))
     for key, _ in opt_losses.items()]

    train_ops_gan = None
    train_steps_gan = None

  return monitor_values, monitor_index, opt_steps, train_ops_gan, train_steps_gan
