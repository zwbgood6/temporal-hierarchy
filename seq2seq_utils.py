"""A Decoder subclass that allows Sonnet output layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sonnet as snt
import tensorflow as tf

from tensorflow.contrib.seq2seq import BasicDecoderOutput
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.contrib.distributions import Normal
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest


__all__ = [
    "SonnetBasicDecoder",
    "AutoregressiveHelper"
]

class SonnetBasicDecoder(decoder.Decoder):
  """Basic sampling decoder."""

  def __init__(self, cell, helper, initial_state, output_layer=None):
    """Initialize SonnetBasicDecoder.

    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      output_layer: (Optional) An instance of `tf.layers.Layer`, e.g.,
        `tf.layers.Dense` or snt.AbstractModule, e.g. snt.Linear.
        Optional layer to apply to the RNN output prior to storing the result or
        sampling.

    Raises:
      TypeError: if `cell` is not an instance of `RNNCell`, `helper`
        is not an instance of `Helper`, or `output_layer` is not an instance
        of `tf.layers.Layer`.
    """

    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
    if not isinstance(helper, helper_py.Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    if output_layer is not None:
      if isinstance(output_layer, snt.AbstractModule):
        output_layer_type = 'AbstractModule'
      elif isinstance(output_layer, layers_base._Layer):  # pylint: disable=protected-access
        output_layer_type = 'Layer'
      else:
        raise TypeError(
            "output_layer must be a Layer or a "
            "Sonnet AbstractModule, received: %s" % type(output_layer))
    else:
      output_layer_type = None
    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state
    self._output_layer = output_layer
    self._output_layer_type = output_layer_type

  @property
  def batch_size(self):
    return self._helper.batch_size

  def _rnn_output_size(self):
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:
      if self._output_layer_type == 'Layer':
        output_shape_with_unknown_batch = nest.map_structure(
            lambda s: tensor_shape.TensorShape([None]).concatenate(s),
            size)
        layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
            output_shape_with_unknown_batch)
        output_size = nest.map_structure(lambda s: s[1:], layer_output_shape)
      else:
        output_size = self._output_layer.output_size
      return output_size

  @property
  def output_size(self):
    # Return the cell output and the id
    return BasicDecoderOutput(
        rnn_output=self._rnn_output_size(),
        sample_id=tensor_shape.TensorShape([]))

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and int32 (the id)
    dtype = nest.flatten(self._initial_state)[0].dtype
    return BasicDecoderOutput(
        nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        self._helper.sample_ids_dtype)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(name, "SonnetBasicDecoderStep", (time, inputs, state)):
      cell_outputs, cell_state = self._cell(inputs, state)
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)
      sample_ids = self._helper.sample(
          time=time, outputs=cell_outputs, state=cell_state)
      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state,
          sample_ids=sample_ids)
    outputs = BasicDecoderOutput(cell_outputs, sample_ids)
    return (outputs, next_state, next_inputs, finished)


class AutoregressiveHelper(tf.contrib.seq2seq.CustomHelper):
  """A helper to deterministically feed previous output as input."""

  def __init__(self, initial_inputs, sequence_length):
    """Initializer.
    Args:
      initial_inputs: The batch of inputs for the first timestep, which is the
        only timestep not generated by the RNN itself. An array of shape
        [batch_size, ...].
      sequence_length: An int32 vector tensor of the sequence length of each
        sequence in the batch.
    """
    self._initial_inputs = initial_inputs
    self._batch_size = initial_inputs.get_shape()[0]
    self._sample_ids_shape = tensor_shape.TensorShape([])
    self._sample_ids_dtype = dtypes.int32

    self._sequence_length = ops.convert_to_tensor(
        sequence_length, name="sequence_length")
    if self._sequence_length.get_shape().ndims != 1:
      raise ValueError(
          "Expected sequence_length to be a vector, but received shape: %s" %
          self._sequence_length.get_shape())

  @property
  def sample_ids_shape(self):
    return self._sample_ids_shape

  @property
  def sample_ids_dtype(self):
    return dtypes.float32

  @property
  def batch_size(self):
    return self._batch_size

  def initialize(self, name=None):
    # If we're finished, the next_inputs value doesn't matter,
    # so just always give the initial inputs
    finished = math_ops.equal(0, self._sequence_length)

    return (finished, self._initial_inputs)

  def sample(self, time, outputs, state, name=None):
    """Returns dummy sample_ids, for compatibility with BasicDecoder."""
    sample_ids = tf.zeros(self._batch_size, dtype=self.sample_ids_dtype)
    return sample_ids

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    # See Decoder metaclass in Seq2seq's decoder.py for semantics
    # of these outputs.

    # If we're finished, the next_inputs value doesn't matter,
    # so just always give outputs (don't need control flow)
    next_inputs = outputs

    next_time = time + 1
    finished = (next_time >= self._sequence_length)

    return (finished, next_inputs, state)


class DeterministicDecoderOutput(
    collections.namedtuple("DeterministicDecoderOutput", "rnn_output")):
  pass

class DeterministicDecoder(tf.contrib.seq2seq.Decoder):
  """A seq2seq decoder that does not sample."""

  def __init__(self, cell, helper, initial_state, output_layer=None):
    """Initialize DeterministicDecoder.
    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Optional layer to apply to the RNN output prior
        to storing the result.
    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
    if not isinstance(helper, helper_py.Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    if (output_layer is not None
        and not isinstance(output_layer, layers_base.Layer)):
      raise TypeError(
          "output_layer must be a Layer, received: %s" % type(output_layer))
    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state
    self._output_layer = output_layer

  @property
  def batch_size(self):
    """The batch size of input values."""
    return self._helper.batch_size

  def _rnn_output_size(self):
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    return DeterministicDecoderOutput(
        rnn_output=self._rnn_output_size())

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    dtype = nest.flatten(self._initial_state)[0].dtype
    return DeterministicDecoderOutput(
        nest.map_structure(lambda _: dtype, self._rnn_output_size()))

  def initialize(self, name=None):
    """Initialize the decoder.
    Args:
      name: Name scope for any created operations.
    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.
    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.
    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(name, "DeterministicDecoderStep",
                        (time, inputs, state)):
      cell_outputs, cell_state = self._cell(inputs, state)
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)
      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state)
    outputs = DeterministicDecoderOutput(cell_outputs)
    return (outputs, next_state, next_inputs, finished)


class VariationalDecoderOutput(
    collections.namedtuple("VariationalDecoderOutput",
                           "rnn_output inference_dist prior_dist z_samples")):
  pass

class VariationalDecoder(object):
  """A static seq2seq decoder that realizes the sampling from learned prior."""

  def __init__(self,
               cells,
               initial_states,
               is_training,
               seq_len,
               use_conv_lstm,
               first_image,
               input_sequence=None,
               input_latent_sample_sequence=None,
               initial_inputs=None,
               output_layer=None,
               autoregress=False,
               reencode=False,
               encoder_cnn=None,
               decoder_cnn=None,
               encoder_data_format="NCHW",
               fixed_prior=False,
               data_format="NCHW",
               image_activation=None,
               init_inference=False):
    """Initialize VariationalDecoder.
    Args:
      cells: Multiple `RNNCell` instances.
      initial_states: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial states of the RNNCells.
      is_training: True if in training mode.
      seq_len: Desired output sequence length.
      use_conv_lstm: If True convolutional LSTM is used in cells argument.
      input_sequence: (Optional) Sequence of input embeddings that replaces
        autoregressive feeding of previous output as next input.
      input_latent_sample_sequence: (Optional) Sequence of input z latents that are fed to
        the decoder instead of sampling new ones.
      initial_inputs: (Optional) Initial input frame, if None first real
        input is used instead (and no prediction is made for first frame)
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Optional layer to apply to the RNN output prior
        to storing the result.
      autoregress: If true, the prior and decoder LSTMs observe the autoregressively
        generated latents. The inference LSTM still observes the ground truth, if any
      encoder_cnn: (optional) CNN used for autoregressive reencoding.
      decoder_cnn: (optional) CNN used for autoregressive reencoding.
      data_format: (optional) Desired data format of cnn_decoder,
        used to determine whether dimensions need to be resorted prior to
        reencoding in autoregression case. RNN always takes NHWC.
      init_inference: If True, inference network is initialized by running on the
        initial_input.
    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    # super(VariationalDecoder, self).__init__(name=name)
    for _, cell in cells.items():
      if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
        raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
    if (output_layer is not None
        and not isinstance(output_layer, layers_base.Layer)):
      raise TypeError(
          "output_layer must be a Layer, received: %s" % type(output_layer))
    if (initial_inputs is None):
      raise ValueError(
        "Need to give either initial input or input sequence for Variational Decoder!")

    self._cells = cells
    self._initial_states = initial_states
    self._is_training = is_training
    self._seq_len = seq_len
    self._use_conv_lstm = use_conv_lstm
    self._output_layer = output_layer
    self._fixed_prior = fixed_prior
    self._input_sequence = input_sequence
    self._batch_size = initial_inputs.get_shape().as_list()[0]
    self._use_cdna_model = tf.flags.FLAGS.use_cdna_model
    self._autoregress = autoregress
    self._cnn_encoder = encoder_cnn
    self._cnn_decoder = decoder_cnn
    self._encoder_data_format = encoder_data_format
    self._reencode = reencode
    self._input_latent_sample_sequence = input_latent_sample_sequence
    self._prev_inputs = initial_inputs
    self._data_format = data_format
    self._first_image = first_image

    if input_sequence is not None:
      self._input_seq_len = input_sequence.get_shape().as_list()[0]
      if not self._use_conv_lstm:
        self._input_sequence = tf.reshape(self._input_sequence, [self._input_seq_len,
                                                                 self._batch_size,
                                                                 -1])
      if self._input_seq_len != self._seq_len:
        tf.logging.warning(['VariationalDecoder input sequence length and desired output length',
                           ' do not match. Is this desired? They are %d and %d' %
                            (self._input_seq_len, self._seq_len)])
    else:
      self._input_seq_len = 0
    if input_latent_sample_sequence is not None:
      if input_latent_sample_sequence.get_shape().as_list()[0] != self._seq_len:
        raise ValueError("Input Latent sequence must have the same length as the desired"
                         "output sequence")

    def get_size(output_size):
      # This is needed as there is an inconsistency between linear layers and LSTM
      # linear layers return an int, whereas LSTM returns TensorShape
      if isinstance(output_size, int):
        return output_size
      else:
        return output_size.as_list()[-1]
      
    self._inf_output_size = get_size(self._cells['inference'].output_size)
    self._prior_output_size = get_size(self._cells['prior'].output_size)
    self._sample_dim = int(self._inf_output_size*0.5)
    # sanity check sample distribution dimensions
    if (self._inf_output_size % 2 != 0
        or self._inf_output_size != self._prior_output_size):
      raise ValueError("Dimensions of Inference and Prior distribution are not valid, "
                       "they are: %d, %d" % (self._inf_output_size,
                                             self._prior_output_size))
    if init_inference:
      # run inference network on initial_inputs to initialize
      _, self._initial_states['inference'] = \
            self._cells['inference'](self._maybe_encode_inputs(initial_inputs),
                                     self._initial_states['inference'])

    if reencode and not autoregress:
      # (oleh) note that autoregression will also be executed when out of input frames
      # in which case this combination will be meaningful
      raise ValueError("Reencoding is only supported with autoregression.")

    if image_activation is None:
      self._image_activation = lambda x: x
    else:
      self._image_activation = image_activation

  def _lstm2latent(self, lstm_latent):
    """ Transforms LSTM-formatted latent to decoder-compatible latent. """
    output_latent = lstm_latent
    if not self._use_conv_lstm:
      # reshape to spatial latent in NHWC
      # TODO(karl, oleh): no magic numbers!
      # The latent now has spatial dimensions of [1,1] because of compression in DCGAN
      output_latent = tf.reshape(output_latent, [self._batch_size, -1, 1, 1])
    if self._use_conv_lstm and self._data_format == "NCHW":
      # shift channels to the front
      output_latent = tf.transpose(output_latent, [0, 3, 1, 2])
    return output_latent

  def _latent2lstm(self, encoder_latent):
    """ Transforms encoder output latent to lstm-compatible latent. """
    output_latent = encoder_latent
    if self._use_conv_lstm and self._data_format == "NCHW":
      # shift channels back to the back
      output_latent = tf.transpose(output_latent, [0, 2, 3, 1])
    if not self._use_conv_lstm:
      output_latent = tf.reshape(output_latent, [self._batch_size, -1])
    return output_latent

  def decode(self, use_prior_override=False):
    """
    Builds the static RNN decoder.
    Args:
      use_prior_override: Forces decoder to use prior for sampling. Default: False
    :return: Dictionary of decoded time series (VariarionalDecoderOutput object)
    """
    output_latent_sequence, inference_dists, prior_dists, z_samples = [], [], [], []
    output_images = [] if self._reencode else None
    states = self._initial_states
    input_latent_sample = None
    for time_step in range(self._seq_len):
      if (self._input_sequence is not None
            and time_step < self._input_seq_len):
        inputs = self._input_sequence[time_step]
      else:
        inputs = None
      if self._input_latent_sample_sequence is not None:
        input_latent_sample = self._input_latent_sample_sequence[time_step]

      use_inference = True if (not use_prior_override
                               and time_step < self._input_seq_len) else False

      output_latent, inference_dist, prior_dist, states, z_sample = self.step(time_step,
                                                                   inputs,
                                                                   input_latent_sample,
                                                                   states,
                                                                   use_inference,
                                                                   name='VarDecode_%d' % time_step)
      output_latent_sequence.append(output_latent)
      z_samples.append(z_sample)
      if input_latent_sample is None:
        prior_dists.append(prior_dist)  # is None if input sample is given
      
      # update last_inputs
      if self._autoregress or inputs is None:
        if not self._reencode:
          self._prev_inputs = output_latent
        else:
          # de-+encode latent in image space during autoregression
          decoder_latent = self._lstm2latent(output_latent)
          output_img_preactivation = self._cnn_decoder(decoder_latent, self._is_training)
          output_images.append(output_img_preactivation)
          encoder_latent, _ = self._cnn_encoder(self._image_activation(output_img_preactivation), self._is_training)
          self._prev_inputs = self._latent2lstm(encoder_latent)
      else:
        self._prev_inputs = inputs

      if inputs is not None\
              and input_latent_sample is None:
        # (oleh) we might actually want to see validation KL
        inference_dists.append(inference_dist) # Don't append None

    output = VariationalDecoderOutput(rnn_output=tf.stack(output_latent_sequence),
                                      inference_dist=tf.stack(inference_dists),
                                      prior_dist=tf.stack(prior_dists),
                                      z_samples=tf.stack(z_samples))
    if self._reencode:
      output_image_sequence = tf.stack(output_images)
    else:
      output_image_sequence = None
    return output, states, output_image_sequence

  def step(self, time, inputs, input_latent_sample, states, use_inference, name=None):
    """Perform a decoding step.
    Args:
      time: scalar `int32`.
      inputs: A (structure of) input tensors.
      input_latent_sample: Can override sampling of new latent.
      states: A (structure of) state tensors and TensorArrays.
      use_inference: If True overrides checks for inference or prior network usage and
          always uses inference network.
      name: Name scope for any created operations.
    Returns:
      `output_frame, inference_dist, prior_dist, states`.
    """
    cell_outputs, cell_states = dict(), dict()
    if self._prev_inputs is None:
      raise ValueError("Need previous input for VariationalDecoder!")

    with ops.name_scope(name, "VariationalDecoderStep",
                        (time, inputs, states)):

      if input_latent_sample is None:
        # predict inference distribution from current frame if any
        if inputs is not None:
          cell_outputs['inference'], cell_states['inference'] = \
            self._cells['inference'](self._maybe_encode_inputs(inputs), states['inference'])
        else:
          cell_outputs['inference'], cell_states['inference'] = None, None

        # predict learned prior from previous frame
        if not self._fixed_prior:
          cell_outputs['prior'], cell_states['prior'] = \
            self._cells['prior'](self._maybe_encode_inputs(self._prev_inputs), states['prior'])
        else:
          means = tf.zeros([self._batch_size, self._sample_dim])
          log_std_dev = tf.log(tf.constant(1.0, shape=[self._batch_size, self._sample_dim]))
          cell_outputs['prior'] = tf.concat([means, log_std_dev], axis=1)

        # sample from inference or prior distribution
        if use_inference:
          means = cell_outputs['inference'][..., :self._sample_dim]
          std_dev = tf.exp(cell_outputs['inference'][..., self._sample_dim:])
        else:
          means = cell_outputs['prior'][..., :self._sample_dim]
          std_dev = tf.exp(cell_outputs['prior'][..., self._sample_dim:])

        z_dists = Normal(loc=means, scale=std_dev)
        z_sample = tf.squeeze(z_dists.sample([1]))  # sample one sample from each distribution
        if tf.flags.FLAGS.trajectory_space and not tf.flags.FLAGS.trajectory_autoencoding:
          z_sample = tf.concat(
            [z_sample,tf.zeros(z_sample.get_shape().as_list()[:-1]+[1],dtype=tf.float32)], axis=-1)
      else:
        z_sample = input_latent_sample
        cell_outputs['inference'] = None
        cell_outputs['prior'] = None

      # reconstruct output with LSTM and decoder
      if self._use_cdna_model:
        decoder_input = [self._prev_inputs, self._first_image, z_sample, self._is_training]
      else:
        decoder_input = tf.concat((self._prev_inputs, z_sample), axis=-1)
      cell_outputs['output'], cell_states['output'] = \
        self._cells['output'](decoder_input, states['output'])
      if self._output_layer is not None:
        cell_outputs['output'] = self._output_layer(cell_outputs['output'])

    return cell_outputs['output'], cell_outputs['inference'], \
           cell_outputs['prior'], cell_states, z_sample

  def _maybe_encode_inputs(self, inputs):
    # encode input image in case of CDNA model for prior + inference network
    if self._use_cdna_model:
      with tf.variable_scope("prior_inf_encoder"):
        if self._data_format == "NHWC" and self._encoder_data_format == "NCHW":
          encoder_inputs = tf.transpose(inputs, (0, 3, 1, 2))
        else:
          encoder_inputs = inputs
        encoded_input, _ = self._cnn_encoder(encoder_inputs, self._is_training)
        encoded_input = tf.squeeze(encoded_input)
    else:
      encoded_input = inputs
    return encoded_input


