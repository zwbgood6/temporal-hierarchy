"""GAN losses and utilities for sequence prediction."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow as tf
import collections

from tensorflow.contrib.training.python.training import training
from tensorflow.contrib.gan.python.namedtuples import GANTrainOps
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import training_util
from tensorflow.python.framework import ops

from architectures import discriminators, mlp_architectures

GANTuple = collections.namedtuple("GANTuple",
                                  "real_data generated_data discriminator_fn")

class GANLosses(
    collections.namedtuple('GANLoss', (
        'generator_losses',
        'discriminator_losses',
        'nongan_losses',
        'gen_nongan_losses'
    ))):
  """GANLoss contains the generator and discriminator loss lists as well as non-gan losses

  Args:
    generator_losses: A list of  tensors for the generator losses.
    discriminator_losses: A list of tensor for the discriminator losses.
    nongan_losses: A tensor for the remaining losses.
    gen_nongan_losses: A sum of all generator and nongan losses. TODO(oleh) restructure so this is computed inside the class
  """

tfgan = tf.contrib.gan

FLAGS=tf.flags.FLAGS

# TODO(drewjaegle): reconcile with original optimize
def gan_optimizers(gen_lr, dis_lr_list):
  # First is generator optimizer, second is discriminator.
  adam_kwargs = {
      'epsilon': 1e-8,
      'beta1': 0.5,
  }
  dis_opt=[]
  for dis_lr in dis_lr_list:
    dis_opt.append(tf.train.AdamOptimizer(dis_lr, **adam_kwargs))

  return (tf.train.AdamOptimizer(gen_lr, **adam_kwargs),
          dis_opt)

def _get_update_ops(kwargs, gen_scope, dis_scopes, check_for_unused_ops=True):
  """Gets generator and discriminator update ops.

  Args:
    kwargs: A dictionary of kwargs to be passed to `create_train_op`.
      `update_ops` is removed, if present.
    gen_scope: A scope for the generator.
    dis_scope: A scope for the discriminator.
    check_for_unused_ops: A Python bool. If `True`, throw Exception if there are
      unused update ops.
  Returns:
    A 2-tuple of (generator update ops, discriminator train ops).
  Raises:
    ValueError: If there are update ops outside of the generator or
      discriminator scopes.
  """
  if 'update_ops' in kwargs:
    update_ops = set(kwargs['update_ops'])
    del kwargs['update_ops']
  else:
    update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))

  all_gen_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS, gen_scope))
  all_dis_ops_list=[]
  all_dis_ops_set=set()
  for dis_scope in dis_scopes:
    all_dis_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS, dis_scope))
    all_dis_ops_set=all_dis_ops_set | all_dis_ops
    all_dis_ops_list.append(all_dis_ops)

  if check_for_unused_ops:
    unused_ops = update_ops - all_gen_ops - all_dis_ops_set
    if unused_ops:
      raise ValueError('There are unused update ops: %s' % unused_ops)

  gen_update_ops = list(all_gen_ops & update_ops)
  dis_update_ops_list=[]
  for all_dis_ops in all_dis_ops_list:
    dis_update_ops = list(all_dis_ops & update_ops)
    dis_update_ops_list.append(dis_update_ops)

  return gen_update_ops, dis_update_ops_list

def gan_train_ops(
    models,
    generator_scope,
    discriminator_scopes,
    losses,
    generator_optimizer,
    discriminator_optimizers,
    check_for_unused_update_ops=True,
    # Optional args to pass directly to the `create_train_op`.
    **kwargs):
  """Returns GAN train ops.
  The highest-level call in TFGAN. It is composed of functions that can also
  be called, should a user require more control over some part of the GAN
  training process.
  Args:
    model: A GANModel.
    loss: A GANLoss.
    generator_optimizer: The optimizer for generator updates.
    discriminator_optimizer: The optimizer for the discriminator updates.
    check_for_unused_update_ops: If `True`, throws an exception if there are
      update ops outside of the generator or discriminator scopes.
    **kwargs: Keyword args to pass directly to
      `training.create_train_op` for both the generator and
      discriminator train op.
  Returns:
    A GANTrainOps tuple of
    (generator_train_op, discriminator_train_op, global_step_inc_op) that can
    be used to train a generator/discriminator pair.
  """
  # Create global step increment op.
  global_step = training_util.get_or_create_global_step()
  global_step_inc = global_step.assign_add(1)

  # Get generator and discriminator update ops. We split them so that update
  # ops aren't accidentally run multiple times. For now, throw an error if
  # there are update ops that aren't associated with either the generator or
  # the discriminator. Might modify the `kwargs` dictionary.
  gen_update_ops, dis_update_ops_list = _get_update_ops(
      kwargs, generator_scope, discriminator_scopes,
      check_for_unused_update_ops)

  with tf.name_scope(generator_scope):
    gen_train_op = training.create_train_op(
        total_loss=losses.gen_nongan_losses,
        optimizer=generator_optimizer,
        variables_to_train=models[0].generator_variables, # the generator variables are always the same
        global_step=None,
        update_ops=gen_update_ops,
        summarize_gradients=True,
        **kwargs)

    gen_summaries=ops.get_collection(ops.GraphKeys.SUMMARIES, generator_scope)

  disc_train_ops={}
  disc_summaries_dict={}
  for i in np.arange(len(models)):
    with tf.name_scope(discriminator_scopes[i]):
      disc_train_op = training.create_train_op(
        total_loss=losses.discriminator_losses[i],
        optimizer=discriminator_optimizers[i],
        variables_to_train=models[i].discriminator_variables,
        global_step=None,
        update_ops=dis_update_ops_list[i],
        summarize_gradients=True,
        **kwargs)
      # annotate the train_ops with meaningful names?
      disc_train_ops[i]=disc_train_op

      disc_summaries_dict[discriminator_scopes[i]]=ops.get_collection(ops.GraphKeys.SUMMARIES, discriminator_scopes[i])

  return GANTrainOps(gen_train_op, disc_train_ops, global_step_inc), gen_summaries, disc_summaries_dict

def get_gan_model(generated_data,
                  real_data,
                  generator_scope,
                  discriminator_scope,
                  discriminator_fn):
  """Manually construct and return a GANModel tuple."""
  discriminator_gen_outputs = discriminator_fn(generated_data)
  discriminator_real_outputs = discriminator_fn(real_data)

  if discriminator_scope is not None:
    discriminator_vars = tf.contrib.framework.get_trainable_variables(
        scope=discriminator_scope)
  else:
    discriminator_vars = None

  if generator_scope is not None:
    generator_vars = tf.contrib.framework.get_trainable_variables(
        scope=generator_scope)
  else:
    generator_vars = None

  # Manually construct GANModel tuple.
  # (oleh) I am using scopes now to handle train_ops and summaries
  gan_model = tfgan.GANModel(
      generator_inputs=None,
      generated_data=generated_data,
      generator_variables=generator_vars,
      generator_scope=None, # is used to get regularization. We do want generator regularization as it is already there.
      generator_fn=None,  # not necessary
      real_data=real_data,
      discriminator_real_outputs=discriminator_real_outputs,
      discriminator_gen_outputs=discriminator_gen_outputs,
      discriminator_variables=discriminator_vars,
      discriminator_scope=tf.name_scope(discriminator_scope), # is used to get regularization
      discriminator_fn=None) # not necessary

  return gan_model

def get_gan_losses(gan_loss_type):
  """Returns the generator and discriminator losses of a specified type.

  Args:
    gan_loss_type: A string specifying the loss type to use.
  Returns:
    generator_loss: A tfgan generator loss op.
    discriminator_loss: A tfgan discriminator loss op.
  Raises:
    ValueError: if gan_loss_type is unknown.
  """
  if gan_loss_type == "acgan":
    generator_loss = tfgan.losses.acgan_generator_loss
    discriminator_loss = tfgan.losses.acgan_discriminator_loss
  elif gan_loss_type == "least_squares":
    generator_loss = tfgan.losses.least_squares_generator_loss
    discriminator_loss = tfgan.losses.least_squares_discriminator_loss
  elif gan_loss_type == "modified":
    generator_loss = tfgan.losses.modified_generator_loss
    discriminator_loss = tfgan.losses.modified_discriminator_loss
  elif gan_loss_type == "minimax":
    generator_loss = tfgan.losses.minimax_generator_loss
    discriminator_loss = tfgan.losses.minimax_discriminator_loss
  elif gan_loss_type == "wasserstein":
    generator_loss = tfgan.losses.wasserstein_generator_loss
    discriminator_loss = tfgan.losses.wasserstein_discriminator_loss
  else:
    raise ValueError("Unknown loss type.")

  return generator_loss, discriminator_loss


def configure_gans(
      gan_tuples,
      gan_loss_types,
      relative_learning_rates,
      generator_scope,
      discriminator_scopes,
      non_gan_losses=0,
      learning_rate=None,
      phase="train"):
  """Sets up the GAN and returns training ops.

  Args:

  Returns:
    train_ops_gan
    train_steps_gan
  """

  disc_losses=[]
  gen_losses=[]
  gan_models=[]
  # The generator and nongan losses are all summed to one loss
  gen_and_nongan_losses=non_gan_losses

  for i_gan in np.arange(len(gan_tuples)):
    gan_model = get_gan_model(
          generated_data=gan_tuples[i_gan].generated_data,
          real_data=gan_tuples[i_gan].real_data,
          generator_scope=generator_scope,
          discriminator_scope=discriminator_scopes[i_gan],
          discriminator_fn=gan_tuples[i_gan].discriminator_fn)
    gan_models.append(gan_model)
    
    # TODO(drewjaegle): add tensor pool fn here if needed (generator history)

    generator_loss, discriminator_loss = get_gan_losses(gan_loss_types[i_gan])

    gan_loss=tfgan.gan_loss(
          gan_model,
          generator_loss_fn=generator_loss,
          discriminator_loss_fn=discriminator_loss,
          add_summaries=False)
    disc_losses.append(gan_loss.discriminator_loss*relative_learning_rates[i_gan])
    gen_losses.append(gan_loss.generator_loss*relative_learning_rates[i_gan])
    gen_and_nongan_losses=gen_and_nongan_losses+gan_loss.generator_loss*relative_learning_rates[i_gan]

  gan_losses=GANLosses(gen_losses, disc_losses, non_gan_losses, gen_and_nongan_losses)

  if phase == "train":
    # Get the GANTrain ops using the custom optimizers and optional
    # discriminator weight clipping.
    # TODO(oleh): Expand to allow separate learning rates
    # Specify this in config instead
    disc_learning_rates=[]
    for i in np.arange(len(gan_tuples)):
      disc_learning_rates.append(learning_rate)

    gen_opt, dis_opt_list = gan_optimizers(learning_rate, disc_learning_rates)

    # gen_scope and dis_scope both required in here - update_ops are pooled
    # from them. Make sure this is okay (check _get_update_ops in /gan/python/train.py)
    # TODO(oleh): make sure this is acting as we expect.
    # Notably: we're manually specifying scopes names here, so they're not used
    # for regularization above and because we're using sonnet to manage scopes
    # for us otherwise. Double check to make sure this is OK!!!

    train_ops_gan, gen_summaries, disc_summaries_dict = gan_train_ops(
        gan_models,
        generator_scope,
        discriminator_scopes,
        gan_losses,
        generator_optimizer=gen_opt,
        discriminator_optimizers=dis_opt_list)

    # Determine the number of generator vs discriminator steps.
    train_steps_gan = tfgan.GANTrainSteps(
        generator_train_steps=1,
        discriminator_train_steps=1)
  else:
    train_ops_gan = None
    train_steps_gan = None
    gen_summaries = None
    disc_summaries_dict = None

  return train_ops_gan, train_steps_gan, gan_losses, gen_summaries, disc_summaries_dict

def build_gan_tuple(true_data,
                    gen_data,
                    latent_discriminator,
                    individual_elements=False,
                    activation_fn=None):
  # Build a GAN spec tuple

  if activation_fn:
    gen_data=activation_fn(gen_data)

  # Flatten the time dimension into batch_size, the sequence is considered by the GAN individually image-by-image
  if individual_elements:
    # Get the input shape.
    input_shape = gen_data.get_shape().as_list()
    # Convert sequence length and batch to batch
    batch_size = input_shape[0] * input_shape[1]
    latent_size = input_shape[2:]
    latent_size.insert(0, batch_size)
    new_size=latent_size

    gen_data = tf.reshape(gen_data, new_size)
    true_data = tf.reshape(true_data, new_size)
  else:
    # this is for the snt.Conv3D
    # TODO(oleh) generalize to different data formats
    true_data = tf.transpose(true_data, [1, 2, 0, 3, 4])
    gen_data = tf.transpose(gen_data, [1, 2, 0, 3, 4])

  return GANTuple(true_data,
      gen_data,
      latent_discriminator)


def build_gan_specs(
    model_output_train,
    model_output_val,
    loss_spec,
    input_train,
    input_val,
    learning_rate,
    regularizer_weight,
    opt_losses,
    generator_scope,
    monitor_values,
    monitor_index):
  # TODO(oleh): Should be specified in a config:
  #   - everything
  #   - architecture options
  #   - relative l rates
  #   - patchgan
  #   - video/image gan
  #   - reconstruction/prediction gan
  #   - conditional gan
  
  # (drewjaegle) For GAN: compute non-reconstruction losses
  # Will just combine for now, but should modify losses.compute_losses as well
  # TODO(drewjaegle): should pack this into dictionaries with corresponding elements
  # for the different real and generated pairs.
  # discriminator_real = discriminator(model_output_train["future_latents_true"])
  # discriminator_generated = discriminator(model_output_train["future_latents_est"])
  
  regularizer_weight=FLAGS.dicriminator_regularizer_weight
  
  discriminator_scopes = []
  names = []
  gan_tuples = []
  gan_tuples_val = []
  gan_loss_types = []
  relative_learning_rates = []

  individual_element_specs = []
  activations = []
  discriminator_fns = []
  data_getters = []
  
  if FLAGS.use_image_gan_rec:
    names.append("reconstructed_image")
    discriminator_scopes.append("discriminator_" + names[-1])
    discriminator = discriminators.DCGANDiscriminator(name=discriminator_scopes[-1],
                                                      filters_spec=FLAGS.imgan_discriminator_spec,
                                                      patchGAN=FLAGS.is_patchGAN,
                                                      regularizer_weight=regularizer_weight)
    def get_real_fake_data_image_gan_rec(input, output):
        return input["reconstruct"], output["decoded_seq_reconstruct"]
    individual_element_specs.append(True)
    activations.append(loss_spec.image_output_activation)
    discriminator_fns.append(discriminator)
    data_getters.append(get_real_fake_data_image_gan_rec)
    relative_learning_rates.append(FLAGS.relative_gan_lr)
  
  if FLAGS.use_image_gan_pred:
    names.append("predicted_image")
    discriminator_scopes.append("discriminator_" + names[-1])
    discriminator = discriminators.DCGANDiscriminator(name=discriminator_scopes[-1],
                                                      filters_spec=FLAGS.imgan_discriminator_spec,
                                                      patchGAN=FLAGS.is_patchGAN,
                                                      regularizer_weight=regularizer_weight)
    def get_real_fake_data_image_gan_pred(input, output):
        return input["predict"], output["decoded_seq_predict"]
    individual_element_specs.append(True)
    activations.append(loss_spec.image_output_activation)
    discriminator_fns.append(discriminator)
    data_getters.append(get_real_fake_data_image_gan_pred)
    relative_learning_rates.append(FLAGS.relative_gan_lr)
    
  if FLAGS.use_video_gan_rec:
    names.append("reconstructed_video")
    discriminator_scopes.append("discriminator_" + names[-1])
    discriminator = discriminators.DCGAN3DDiscriminator(name=discriminator_scopes[-1],
                                                        filters_spec=FLAGS.imgan_discriminator_spec,
                                                        regularizer_weight=regularizer_weight)
    def get_real_fake_data_vgan_rec(input, output):
        return input["reconstruct"], output["decoded_seq_reconstruct"]
    individual_element_specs.append(False)
    activations.append(loss_spec.image_output_activation)
    discriminator_fns.append(discriminator)
    data_getters.append(get_real_fake_data_vgan_rec)
    relative_learning_rates.append(FLAGS.relative_vgan_lr)
    
  if FLAGS.use_video_gan_pred:
    names.append("predicted_video")
    discriminator_scopes.append("discriminator_" + names[-1])
    if FLAGS.latent_vgan:
      discriminator = mlp_architectures.GeneralizedMLP(name=discriminator_scopes[-1],
                                                    regularizer_weight=regularizer_weight)
    else:
      discriminator = discriminators.DCGAN3DDiscriminator(name=discriminator_scopes[-1],
                                                          filters_spec=FLAGS.imgan_discriminator_spec,
                                                          regularizer_weight=regularizer_weight)
    def get_real_fake_data_vgan_pred(input, output):
      if FLAGS.latent_vgan: # operate on latents
        real = output["future_latents_true"]
        fake = output["future_latents_est"]
        past = output["past_latents_true"]
      else:
        real = input["predict"]
        fake = output["decoded_seq_predict"]
        past = input["reconstruct"]
      if FLAGS.conditional_gan: # concatenate the past sequence
        add_past_seq = lambda future: tf.concat([past, future], axis=0)
        return add_past_seq(real), add_past_seq(fake)
      else:
        return real, fake
    individual_element_specs.append(False)
    if FLAGS.latent_vgan:
      activations.append(loss_spec.image_output_activation)
    else:
      activations.append(None)
    discriminator_fns.append(discriminator)
    data_getters.append(get_real_fake_data_vgan_pred)
    relative_learning_rates.append(FLAGS.relative_vgan_lr)
    
  # Build GAN tuples
  for i, spec in enumerate(names):
    gan_loss_types.append("modified")
    real_data, fake_data = data_getters[i](input_train, model_output_train)
    real_data_val, fake_data_val = data_getters[i](input_val, model_output_val)
    gan_tuple_train = build_gan_tuple(
      real_data,
      fake_data,
      discriminator_fns[i],
      individual_elements=individual_element_specs[i],
      activation_fn=activations[i])
    gan_tuple_val = build_gan_tuple(
      real_data_val,
      fake_data_val,
      lambda data, i=i: discriminator_fns[i](data, is_training=False), # the default i makes this work
      individual_elements=individual_element_specs[i],
      activation_fn=activations[i])
    gan_tuples.append(gan_tuple_train)
    gan_tuples_val.append(gan_tuple_val)

  train_ops_gan, train_steps_gan, train_gan_losses, gen_summaries, disc_summaries_list = configure_gans(
    gan_tuples,
    gan_loss_types,
    relative_learning_rates,
    generator_scope,
    discriminator_scopes,
    opt_losses["total_loss"],
    learning_rate,
    phase="train")

  _, _, val_gan_losses, _, _ = configure_gans(
    gan_tuples_val,
    gan_loss_types,
    relative_learning_rates,
    generator_scope,
    discriminator_scopes,
    phase="val")

  for i, name in enumerate(names):
    monitor_values["generator_" + name + "_loss"] =\
      train_gan_losses.generator_losses[i]/relative_learning_rates[i]
    monitor_index["train"]["loss"].append("generator_" + name + "_loss")
    
    monitor_values["discriminator_" + name + "_loss"] =\
      train_gan_losses.discriminator_losses[i]/relative_learning_rates[i]
    monitor_index["train"]["loss"].append("discriminator_" + name + "_loss")
    
    monitor_values["generator_" + name + "_loss_val"] =\
      val_gan_losses.generator_losses[i]/relative_learning_rates[i]
    monitor_index["val"]["loss"].append("generator_" + name + "_loss_val")
    
    monitor_values["discriminator_" + name + "_loss_val"] =\
      val_gan_losses.discriminator_losses[i]/relative_learning_rates[i]
    monitor_index["val"]["loss"].append("discriminator_" + name + "_loss_val")

    for i in range(len(gen_summaries)):
      monitor_values["gen_summary_"+str(i)] = gen_summaries[i]
      monitor_index["train"]["sum"].append("gen_summary_"+str(i))

    for name, disc_summaries in (disc_summaries_list.items()):
      for i in range(len(disc_summaries)):
        monitor_values[name+"_summary_"+str(i)] = disc_summaries[i]
        monitor_index["train"]["sum"].append(name+"_summary_"+str(i))

  return train_ops_gan, train_steps_gan
