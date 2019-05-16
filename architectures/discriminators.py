"""Discriminator architectures for sequence prediction."""

import configs
import numpy as np
import sonnet as snt
import tensorflow as tf

from  architectures import conv_architectures

# TODO(oleh) order those into subclasses

class DCGANDiscriminator(snt.AbstractModule):
  """A multilayer, fully-connected discriminator."""

  def __init__(self,
               filters_spec="big",
               name="dcgan_discriminator",
               patchGAN=False,
               regularizer_weight=0):
    "Constructs a DCGANDiscriminator."

    super(DCGANDiscriminator, self).__init__(name=name)
    # Set defaults for number of units and layers, etc.
    
    if filters_spec=="small":
      filters=[64, 64, 96, 96]
    elif filters_spec=="med":
      filters=[32, 64, 128, 128, 128]
    elif filters_spec=="big":
      filters=None
    self._patchGAN=patchGAN
    if regularizer_weight:
      self._regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=regularizer_weight)}
    else:
      self._regularizers = None
      
    with self._enter_variable_scope():
      self._discriminator = conv_architectures.DCGANEncoder(latent_size=64,
                                                            filters=filters,
                                                            final_activation=tf.nn.leaky_relu,
                                                            regularizers_no_bias=self._regularizers)
      self._last_layer=snt.Linear(1, regularizers=self._regularizers)

  def _build(self,
             input,
             is_training=True):
    """Adds the network into the graph."""

    # TODO(oleh): add initializers, etc.
    # TODO(oleh) make sure the correct format is used
    output, _ = self._discriminator(input, is_training=is_training)
    #output=tf.nn.dropout(output, keep_prob=0.5)
    
    if self._patchGAN:
      output=tf.reduce_mean(output, axis=[2,3])
    else:
      input_shape = output.get_shape()
      output=tf.reshape(output, [input_shape[0], -1])
    
    output = self._last_layer(output)
    logits = output

    return logits

class DCGAN3DDiscriminator(snt.AbstractModule):
  """A multilayer, fully-connected discriminator."""

  def __init__(self,
               filters_spec="big",
               name="dcgan_discriminator",
               regularizer_weight=0):
    "Constructs a DCGANDiscriminator."
  
    super(DCGAN3DDiscriminator, self).__init__(name=name)
    # Set defaults for number of units and layers, etc.
  
    if filters_spec == "small":
      filters = [64, 64, 96, 96]
    elif filters_spec == "med":
      filters = [32, 64, 128, 128, 128]
    elif filters_spec == "big":
      filters = None
    if regularizer_weight:
      self._regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=regularizer_weight)}
    else:
      self._regularizers = None
  
    with self._enter_variable_scope():
      self._discriminator = conv_architectures.DCGAN3DEncoder(latent_size=64,
                                                             filters=filters,
                                                             final_activation=tf.nn.leaky_relu,
                                                             regularizers_no_bias=self._regularizers)
      self._last_layer = snt.Linear(1, regularizers=self._regularizers)

  def _build(self,
             input,
             is_training=True):
    """Adds the network into the graph."""
  
    # TODO(oleh): add initializers, etc.
    # TODO(oleh) make sure the correct format is used
    output = self._discriminator(input, is_training=is_training)
  
    input_shape = output.get_shape()
    output = tf.reshape(output, [input_shape[0], -1])
  
    output = self._last_layer(output)
  
    logits = output
  
    return logits
