import sonnet as snt
import tensorflow as tf

class GeneralizedMLP(snt.AbstractModule):
  """A multilayer, fully-connected discriminator."""
  
  def __init__(self,
               name="fc_discriminator",
               regularizer_weight=0.0,
               use_batchnorm=False,
               layers=None,
               final_activation=None):
    "Constructs a GeneralizedMLP."
    
    super(GeneralizedMLP, self).__init__(name=name)
    # Set defaults for number of units and layers, etc.
    
    if layers is not None:
      self._layers = layers
    else:
      self._layers = [64, 64, 1]  # use default values
    
    if regularizer_weight:
      self._regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=regularizer_weight)}
    else:
      self._regularizers = None
    self._use_batchnorm = False  # use_batchnorm
    self._final_activation = final_activation
  
  def _build(self,
             input,
             is_training=False):
    """Adds the network into the graph."""
    
    # TODO(drewjaegle): add initializers, etc.
    
    input_shape = input.get_shape().as_list()
    input = tf.reshape(input, [input_shape[0], -1])
    
    if self._use_batchnorm:
      bn = lambda x: snt.BatchNorm(axis=[0])(x, is_training)
    else:
      bn = lambda x: x
    discriminator = snt.nets.MLP(
      self._layers,
      activation=lambda x: tf.nn.leaky_relu(bn(x)),
      activate_final=False,
      regularizers=self._regularizers,
      initializers={"w": tf.variance_scaling_initializer(scale=1e-4),
                    "b": tf.constant_initializer(value=0.01)})
    
    logits = discriminator(input)
    if self._final_activation is not None:
      logits = self._final_activation(logits)
    
    return logits


class MemoryMLP(snt.AbstractModule):
  """A tiny wrapper around GeneralizedMLP that saves the last input and applies the network
      on the concatenation of the last and the current input."""
  
  def __init__(self,
               layers,
               regularizer_weight=1.0,
               name="memMLP"):
    super(MemoryMLP, self).__init__(name=name)
    self._layers = layers
    self._regularizer_weight = regularizer_weight
    self._prev_input = None
  
  def reset(self):
    self._prev_input = None
  
  def _build(self,
             input,
             is_training=False):
    network = GeneralizedMLP(layers=self._layers,
                             regularizer_weight=self._regularizer_weight)
    if self._prev_input is None:
      # we need to use network in first iteration otherwise sonnet throws error when it tries to
      # reuse its weights in later iterations
      concat_input = tf.concat([input, input], axis=1)
    else:
      concat_input = tf.concat([self._prev_input, input], axis=1)
    output = network(concat_input, is_training=is_training)
    self._prev_input = input
    return output
