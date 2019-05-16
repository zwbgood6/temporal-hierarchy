"""RNN architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import numpy as np
import sonnet as snt
import tensorflow as tf

from architectures import arch_utils, mlp_architectures
from seq2seq_utils import AutoregressiveHelper, DeterministicDecoder
from seq2seq_utils import VariationalDecoder
from sonnet.python.modules import gated_rnn, layer_norm, rnn_core
from specs import module_specs
from tensorflow.python.ops import array_ops

_DEFAULT_FC_INITIALIZERS = {
    "w": tf.contrib.layers.xavier_initializer(),
    "b": tf.constant_initializer(value=0.01)}
_DEFAULT_FC_REGULARIZERS = {
    "w": tf.contrib.layers.l2_regularizer(scale=1.0),
    "b": tf.contrib.layers.l2_regularizer(scale=1.0)}

# See Jozefowicz et al -
# An Empirical Exploration of Recurrent Network Architectures
# for the effect of bias initialization.

# We don't regularize our RNNs, except the initial state.
# See this discussion: goo.gl/Pg6511
_PEEPHOLE_KEYS = ["w_f_diag", "w_i_diag", "w_o_diag"]
_PROJECTION_KEYS = ["w_h_projection"]
_DEFAULT_LSTM_INITIALIZERS = {
    "w_gates": tf.contrib.layers.xavier_initializer(),
    "w_f_diag": tf.contrib.layers.xavier_initializer(),
    "w_i_diag": tf.contrib.layers.xavier_initializer(),
    "w_o_diag": tf.contrib.layers.xavier_initializer(),
    "b_gates": tf.constant_initializer(value=0.00),
    "w_h_projection": tf.contrib.layers.xavier_initializer()}
_DEFAULT_CONVLSTM_INITIALIZERS = {
    "w_gates": tf.contrib.layers.xavier_initializer_conv2d(),
    "w_f_diag": tf.contrib.layers.xavier_initializer_conv2d(),
    "w_i_diag": tf.contrib.layers.xavier_initializer_conv2d(),
    "w_o_diag": tf.contrib.layers.xavier_initializer_conv2d(),
    "b_gates": tf.constant_initializer(value=0.00),
    "w_h_projection": tf.contrib.layers.xavier_initializer_conv2d()}

_DEFAULT_INITIAL_STATE_INITIALIZERS = {
    "hidden": tf.contrib.layers.xavier_initializer(),
    "cell": tf.contrib.layers.xavier_initializer(),
}
_DEFAULT_CONV_INITIAL_STATE_INITIALIZERS = {
    "hidden": tf.contrib.layers.xavier_initializer_conv2d(),
    "cell": tf.contrib.layers.xavier_initializer_conv2d(),
}
_DEFAULT_INITIAL_STATE_REGULARIZERS = {
    "hidden": tf.contrib.layers.l2_regularizer(scale=1.0),
    "cell": tf.contrib.layers.l2_regularizer(scale=1.0),
}
_DEFAULT_CONV_INITIAL_STATE_REGULARIZERS = {
    "hidden": tf.contrib.layers.l2_regularizer(scale=1.0),
    "cell": tf.contrib.layers.l2_regularizer(scale=1.0),
}


def build_lstm_module(
    module_spec,
    use_peepholes=False,
    use_projection=False):
  """Builds an LSTM module and initial state properties."""

  # This is the default thing to do for LSTMs
  initializers = _get_default_lstm_init(
      use_peepholes=use_peepholes,
      use_projection=use_projection)

  initial_state_initializers = gated_rnn.LSTMState(
      hidden=_DEFAULT_INITIAL_STATE_INITIALIZERS["hidden"],
      cell=_DEFAULT_INITIAL_STATE_INITIALIZERS["cell"])
  initial_state_regularizers = gated_rnn.LSTMState(
      hidden=_DEFAULT_INITIAL_STATE_REGULARIZERS["hidden"],
      cell=_DEFAULT_INITIAL_STATE_REGULARIZERS["cell"])

  if module_spec.smooth_projection:
    projection_initializers = _DEFAULT_FC_INITIALIZERS
    module = SmoothProjectionLSTM(
        module_spec.num_hidden,
        output_size=module_spec.output_size,
        output_nonlinearity=module_spec.sp_output_nonlinearity,
        initializers=initializers,
        projection_initializers=projection_initializers)
  else:
    module = snt.LSTM(
        module_spec.num_hidden,
        initializers=initializers)

  return module, initial_state_initializers, initial_state_regularizers


def build_mlp_initializer(lstm_spec, mlp):
  """Builds LSTM initial_state returning function."""
  def mlp_initializer_getter(mlp_input):
    mlp_output = mlp(mlp_input)
    init_states = []
    cum_idx = 0
    for lstm_module_spec in lstm_spec:
      if not isinstance(lstm_module_spec, module_specs.LSTMSpec):
        continue
      state_size_i = lstm_module_spec.num_hidden
      init_state_i = gated_rnn.LSTMState(hidden=mlp_output[:, cum_idx:cum_idx+state_size_i],
                                         cell=mlp_output[:, cum_idx+state_size_i:cum_idx+2*state_size_i])
      init_states.append(init_state_i)
      cum_idx += 2*state_size_i
    return init_states
  return mlp_initializer_getter


def build_conv_lstm_module(
    module_spec,
    data_format,
    use_bias=True,
    use_peepholes=False,
    use_projection=False):

  # Set up initializers and regularizers
  if use_bias:
    initializers = arch_utils._DEFAULT_CONV_INITIALIZERS
  else:
    initializers = arch_utils._DEFAULT_CONV_INITIALIZERS_NO_BIAS

  initial_state_initializers = (
      _DEFAULT_CONV_INITIAL_STATE_INITIALIZERS["hidden"],
      _DEFAULT_CONV_INITIAL_STATE_INITIALIZERS["cell"])
  initial_state_regularizers = (
      _DEFAULT_CONV_INITIAL_STATE_REGULARIZERS["hidden"],
      _DEFAULT_CONV_INITIAL_STATE_REGULARIZERS["cell"])

  if module_spec.smooth_projection:
    module = SmoothProjectionConv2DLSTM(
        input_shape=module_spec.input_shape,
        output_channels=module_spec.output_channels,
        kernel_shape=module_spec.kernel_shape,
        output_nonlinearity=module_spec.sp_output_nonlinearity,
        stride=module_spec.stride,
        rate=module_spec.rate,
        padding=snt.SAME,
        use_bias=use_bias,
        data_format=data_format,
        initializers=initializers)
  else:
    if data_format != "NHWC":
      raise ValueError("Only NHWC currently supported by convLSTM.")

    module = snt.Conv2DLSTM(
        input_shape=module_spec.input_shape,
        output_channels=module_spec.output_channels,
        kernel_shape=module_spec.kernel_shape,
        stride=module_spec.stride,
        rate=module_spec.rate,
        padding=snt.SAME,
        use_bias=use_bias,
        initializers=initializers)

  return module, initial_state_initializers, initial_state_regularizers


def _get_default_lstm_init(
    use_peepholes=False,
    use_projection=False):
  """Returns LSTM initializers, possibly with some keys removed."""
  # Set up initializers and regularizers
  initializers = copy.deepcopy(_DEFAULT_LSTM_INITIALIZERS)

  if not use_peepholes:
    for p_key in _PEEPHOLE_KEYS:
      initializers.pop(p_key)

  if not use_projection:
    for p_key in _PROJECTION_KEYS:
      initializers.pop(p_key)

  return initializers


class SmoothProjectionConv2DLSTM(gated_rnn.ConvLSTM):
  """A ConvLSTM wrapper with projection and output summation in the core."""
  def __init__(self,
               input_shape,
               output_channels,
               kernel_shape,
               project_and_skip=True,
               projection_layer=None,
               output_nonlinearity=None,
               projection_initializers=None,
               data_format="NHWC",
               name="conv_lstm",
               **kwargs):
    """Construct SmoothProjectionConvLSTM. See 'snt.ConvLSTM' for more details.
    Args:
      project_and_skip: If True, applies 1x1 conv projection to LSTM output and
        adds inputs. If False, equivalent to standard convLSTM.
        Defaults to True.
      projection_layer: A sonnet module to apply to the LSTM output. If None,
        a 1x1 conv layer will be constructed. Defaults to None.
      output_nonlinearity: An optional nonlinearity to apply after the
        project and skip. Accepts None or "tanh".
      projection_initializers: Initializers for the projection layer. Defaults
        to None.
    Raises:
      ValueError: if data_format is not "NHWC".
    """
    conv_ndims = 2
    super(SmoothProjectionConv2DLSTM, self).__init__(
        conv_ndims,
        input_shape,
        output_channels,
        kernel_shape,
        name=name,
        **kwargs)
    self._project_and_skip = project_and_skip
    self._projection_layer = projection_layer
    # TODO(drewjaegle): Refactor convLSTM to allow NCHW input
    if data_format != "NHWC":
      raise ValueError("Only NHWC currently supported by convLSTM.")
    self._data_format = data_format
    self._output_nonlinearity = output_nonlinearity
    self._projection_initializers = projection_initializers

  def _build_projection_layer(self, inputs):
    if self._data_format == "NHWC":
      channel_dim = 3
    elif self._data_format == "NCHW":
      channel_dim = 1

    output_channels = inputs.get_shape()[channel_dim]
    self._projection_layer = snt.Conv2D(
        output_channels=output_channels,
        kernel_shape=1,
        stride=1,
        rate=1,
        padding=snt.SAME,
        use_bias=self._use_bias,
        data_format=self._data_format,
        initializers=self._initializers)


  def _build(self, inputs, state):
    hidden, cell = state
    input_conv = self._convolutions["input"]
    hidden_conv = self._convolutions["hidden"]
    next_hidden = input_conv(inputs) + hidden_conv(hidden)
    gates = tf.split(value=next_hidden, num_or_size_splits=4,
                     axis=self._conv_ndims+1)

    input_gate, next_input, forget_gate, output_gate = gates
    next_cell = tf.sigmoid(forget_gate + self._forget_bias) * cell
    next_cell += tf.sigmoid(input_gate) * tf.tanh(next_input)
    next_hidden = tf.tanh(next_cell) * tf.sigmoid(output_gate)

    if self._project_and_skip:
      if self._output_nonlinearity is None:
        if self._projection_layer is None:
          # TODO(drewjaegle): encapsulate the logic below.
          self._build_projection_layer(inputs)
          # TODO(drewjaegle): add batchnorm - not sure if can pass is_training in
        output = self._projection_layer(next_hidden)
        output += inputs
      elif self._output_nonlinearity == "tanh":
        if self._projection_layer is None:
          self._build_projection_layer(inputs)
        output = tf.tanh(self._projection_layer(next_hidden + inputs))
      else:
        raise ValueError("Requested output nonlinearity not supported.")
    else:
      output = next_hidden

    if self._skip_connection:
      next_hidden = tf.concat([next_hidden, inputs], axis=-1)
    # output, next_hidden, and inputs are NHWC
    return output, (next_hidden, next_cell)

class SmoothProjectionLSTM(snt.LSTM):
  """An LSTM wrapper with projection and output summation in the core."""

  def __init__(self,
               hidden_size,
               project_and_skip=True,
               projection_layer=None,
               output_size=None,
               output_nonlinearity=None,
               projection_initializers=None,
               name="smooth_projection_lstm",
               **kwargs):
    """Construct SmoothProjectionLSTM . See `snt.LSTM` for more details.

    Args:
      project_and_skip: If True, applies projection to LSTM output and add
        inputs. If False, equivalent to standard LSTM. Defaults to True.
      projection_layer: A sonnet module to apply to the LSTM output. If None,
        a linear layer will be constructed. Defaults to None.
      output_nonlinearity: An optional nonlinearity to apply after the
        project and skip. Accepts None or "tanh".
      projection_initializers: Initializers for the projection layer. Defaults
        to None.
      output_size: An optional argument specifying the size of the output.
        Because the projection is part of the core, this is required if the
        projection output differs from the hidden_size. If None, the value of
        hidden_size is used. Defaults to None.
    """
    super(SmoothProjectionLSTM, self).__init__(
        hidden_size,
        name=name,
        **kwargs)

    self._project_and_skip = project_and_skip
    self._projection_layer = projection_layer
    self._output_nonlinearity = output_nonlinearity  # After the project and skip
    self._projection_initializers = projection_initializers

    if output_size is None:
      self._output_size = hidden_size
    else:
      self._output_size = output_size

  @property
  def output_size(self):
    """`tf.TensorShape` indicating the size of the core output."""
    return tf.TensorShape([self._output_size])

  def _build(self, inputs, prev_state):
    """Connects the LSTM module into the graph.
    If this is not the first time the module has been connected to the graph,
    the Tensors provided as inputs and state must have the same final
    dimension, in order for the existing variables to be the correct size for
    their corresponding multiplications. The batch size may differ for each
    connection.
    Args:
      inputs: Tensor of size `[batch_size, input_size]`.
      prev_state: Tuple (prev_hidden, prev_cell).
    Returns:
      A tuple (output, next_state) where 'output' is a Tensor of size
      `[batch_size, hidden_size]` and 'next_state' is a tuple
      (next_hidden, next_cell) where next_hidden and next_cell have size
      `[batch_size, hidden_size]`.
    Raises:
      ValueError: If connecting the module into the graph any time after the
        first time, and the inferred size of the inputs does not match previous
        invocations.
    """
    prev_hidden, prev_cell = prev_state

    # pylint: disable=invalid-unary-operand-type
    if self._hidden_clip_value is not None:
      prev_hidden = tf.clip_by_value(
          prev_hidden, -self._hidden_clip_value, self._hidden_clip_value)
    if self._cell_clip_value is not None:
      prev_cell = tf.clip_by_value(
          prev_cell, -self._cell_clip_value, self._cell_clip_value)
    # pylint: enable=invalid-unary-operand-type

    self._create_gate_variables(inputs.get_shape(), inputs.dtype)

    # pylint false positive: calling module of same file;
    # pylint: disable=not-callable

    # Parameters of gates are concatenated into one multiply for efficiency.
    inputs_and_hidden = tf.concat([inputs, prev_hidden], 1)
    gates = tf.matmul(inputs_and_hidden, self._w_xh)

    if self._use_layer_norm:
      gates = layer_norm.LayerNorm()(gates)

    gates += self._b

    # i = input_gate, j = next_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(value=gates, num_or_size_splits=4, axis=1)

    if self._use_peepholes:  # diagonal connections
      self._create_peephole_variables(inputs.dtype)
      f += self._w_f_diag * prev_cell
      i += self._w_i_diag * prev_cell

    forget_mask = tf.sigmoid(f + self._forget_bias)
    next_cell = forget_mask * prev_cell + tf.sigmoid(i) * tf.tanh(j)
    cell_output = next_cell
    if self._use_peepholes:
      cell_output += self._w_o_diag * cell_output

    next_hidden = tf.tanh(cell_output) * tf.sigmoid(o)

    if self._project_and_skip:
      if self._output_nonlinearity is None:
        if self._projection_layer is None:
          self._projection_layer = snt.Linear(
              self._output_size,
              initializers=self._projection_initializers)
        output = self._projection_layer(next_hidden) + inputs
      elif self._output_nonlinearity == "tanh":
        if self._projection_layer is None:
          self._projection_layer = snt.Linear(
              self._output_size,
              initializers=self._projection_initializers)
        output = tf.tanh(self._projection_layer(next_hidden + inputs))

    return output, gated_rnn.LSTMState(hidden=next_hidden, cell=next_cell)



def _build_rnn_module(
    module_spec,
    data_format,
    use_peepholes=False,
    use_projection=False):
  """Builds a module and its initial state initializers for a deep RNN."""

  if isinstance(module_spec, module_specs.LSTMSpec):
    module, initial_state_initializers, initial_state_regularizers = \
        build_lstm_module(
            module_spec,
            use_peepholes=use_peepholes,
            use_projection=use_projection)
  elif isinstance(module_spec, module_specs.ConvLSTMSpec):
    module, initial_state_initializers, initial_state_regularizers = \
      build_conv_lstm_module(
          module_spec,
          data_format,
          use_peepholes=use_peepholes,
          use_projection=use_projection)
  elif isinstance(module_spec, module_specs.LinearSpec):
    module = LinearLayer(
        module_spec.output_size,
        initializers=_DEFAULT_FC_INITIALIZERS,
        non_linearity=module_spec.non_linearity)
    # module = snt.Linear(
    #     module_spec.output_size,
    #     initializers=_DEFAULT_FC_INITIALIZERS)
    initial_state_initializers = None
    initial_state_regularizers = None
  elif isinstance(module_spec, module_specs.ConvNetSpec):
    pass
  elif isinstance(module_spec, module_specs.MLPSpec):
    module = mlp_architectures.MemoryMLP(module_spec.layers)
    initial_state_initializers = None
    initial_state_regularizers = None
  else:
    raise ValueError("Module not yet supported in RNNs. Maybe add it?")

  return module, initial_state_initializers, initial_state_regularizers


def build_rnn(
    rnn_spec,
    input_image_shape=None,
    channels=None,
    use_skip_connections=False,
    trainable_initial_state=True):
  """Builds an RNN core and a state initializer.

  Args:
    rnn_spec: A specification defining the RNN.

  Returns:
    core: A callable RNN module.
    initial_state_getter: A function returning an initial state
      for the rnn core if called with the batch_size.
  """
  if tf.flags.FLAGS.use_cdna_model and \
      isinstance(rnn_spec, module_specs.CDNASpec):
      core, initial_state_getter = build_cdna_model(
          rnn_spec,
          input_image_shape,
          channels,
          trainable_initial_state=trainable_initial_state)
  else:
      core, initial_state_getter = build_deep_rnn(
          rnn_spec,
          use_skip_connections=use_skip_connections,
          trainable_initial_state=trainable_initial_state)
  return core, initial_state_getter


def build_deep_rnn(
    lstm_spec,
    use_skip_connections=False,
    trainable_initial_state=True,
    data_format="NHWC"):
  """A wrapper around snt.DeepRNN that returns a DeepRNN and initial states.

  Args:
    lstm_spec: A tuple specifying the parameters of the network. Each entry is
      a NamedTuple, with the values of the corresponding LSTM module.
    use_skip_connections: Whether to use skip connections in the
      `snt.DeepRNN`. Default is `True`.
    trainable_initial_state: Whether to use a trainable initial_state for the
      RNN core. Ignored if an initial_state is passed into the network at
      graph construction time. Defaults to True.
    data_format: The format of the input images. Only NHWC supported
      (convLSTM does not support NCHW).
  Returns:
    deep_lstm: An instance of snt.DeepRNN with the specified deep LSTM
      architecture.
    initial_state_getter: A function of batch size returning a nested tuple of
      initial states for the modules of deep_lstm.
  """
  # TODO(drewjaegle): refactor this to use SkipConnectionCore
  modules = []
  initial_state_initializers = []
  initial_state_regularizers = []

  # We assume that peepholes and projection are NOT used
  _USE_PEEPHOLES = False
  _USE_PROJECTION = False

  for module_spec in lstm_spec:
    module_i, initial_state_initializers_i, initial_state_regularizers_i = \
      _build_rnn_module(
          module_spec,
          data_format,
          use_peepholes=_USE_PEEPHOLES,
          use_projection=_USE_PROJECTION)

    # Add this module to the core
    modules.append(module_i)
    if initial_state_initializers_i is not None:
      initial_state_initializers.append(initial_state_initializers_i)
    if initial_state_regularizers_i is not None:
      initial_state_regularizers.append(initial_state_regularizers_i)


  deep_rnn = snt.DeepRNN(
      modules,
      skip_connections=use_skip_connections,
      name="deep_rnn")

  def initial_state_getter(batch_size):
    initial_state = deep_rnn.initial_state(
        batch_size=batch_size,
        trainable=trainable_initial_state,
        trainable_initializers=initial_state_initializers,
        trainable_regularizers=initial_state_regularizers)
    return initial_state

  return deep_rnn, initial_state_getter


def build_cdna_model(
    lstm_spec,
    input_shape,
    channels,
    trainable_initial_state=True,
    data_format="NHWC"):

    if data_format == "NHWC" and input_shape[-1] != channels:
        # correct for inconsistency between dataset data format and CDNA input format
        input_shape = input_shape[1:] + [input_shape[0]]

    from architectures.cdna_model import Finn2016Model
    cdna_model = Finn2016Model(lstm_spec, data_format, input_shape)

    def initial_state_getter(batch_size):
        initial_state = cdna_model.initial_state(
            batch_size=batch_size,
            trainable=trainable_initial_state)
        return initial_state
    return cdna_model, initial_state_getter


def _check_conv_lstm_input(
    conv_lstm_input_shape,
    data_input_shape,
    n_batch_like_dims):
  """Checks that the input data matches the shape expected by the convLSTM.

  Args:
    conv_lstm_input_shape: The shape expected by the convLSTM. Time and batch
      dimensions are not specified.
    data_input_shape: The shape of the input data. Time and batch dimensions
      may be specified.
    n_batch_like_dims: The number of dimensions to ignore of data_input_shape.
  Raises:
    ValueError if data_input_shape non-time and non-batch dims do not match
      the expected shape.
  """
  shapes_match = all(
      [data_input_shape[i + n_batch_like_dims] == conv_lstm_input_shape[i]
       for i in range(len(conv_lstm_input_shape))])
  if not shapes_match:
    raise ValueError("RNN core input_shape does not match data input shape!")


class DeepLSTMSeq2Seq(snt.AbstractModule):
  """A deep LSTM for seq2seq output."""

  def __init__(self,
               core,
               use_conv_lstm=False,
               data_format="NCHW",
               name="deep_lstm_seq2seq"):
    """Constructs a DeepLSTMSeq2Seq.

    Args:
      core: The RNN core to run.
      use_conv_lstm: Whether to use convolutional LSTM.
        Defaults to False (uses standard, fully-connected LSTM).
      data_format: The format of the input images: 'NCHW' or 'NHWC'.
      name: The network name. Defaults to 'deep_lstm_seq2seq'.
    Raises:
      ValueError: If data_format is not supported.
    """
    super(DeepLSTMSeq2Seq, self).__init__(name=name)

    self._core = core
    self._use_conv_lstm = use_conv_lstm
    self._data_format = data_format

  def _format_input(
      self,
      initial_inputs,
      initial_inputs_shape,
      batch_size,
      latent_size):
    """Formats the input data for RNN processing."""

    if self._use_conv_lstm:
      # TODO(drewjaegle): remove once convLSTM supports NCHW
      if self._data_format == "NCHW":
        # NCHW->NHWC
        initial_inputs_formatted = tf.transpose(
            initial_inputs,
            perm=[0, 2, 3, 1])
        lstm_input_shape = [
            initial_inputs_shape[2],
            initial_inputs_shape[3],
            initial_inputs_shape[1]]
      elif self._data_format == "NHWC":
        lstm_input_shape = initial_inputs_shape[1:]
      else:
        raise ValueError("Unsupported data_format.")
    else:
      # Flatten for fully-connected LSTM
      initial_inputs_formatted = tf.reshape(
          initial_inputs, [batch_size, latent_size])
      lstm_input_shape = None

    return initial_inputs_formatted, lstm_input_shape

  def _format_output(
      self,
      output_sequence,
      output_seq_shape,
      seq_len,
      batch_size,
      latent_size):
    """Formats the output of the RNN to match the input."""
    if self._use_conv_lstm:
      if self._data_format == "NCHW":
        # TNHWC->TNCHW
        output_sequence_formatted = tf.transpose(
            output_sequence,
            perm=[0, 1, 4, 2, 3])
      output_sequence_formatted.set_shape(output_seq_shape)
    else:
      output_sequence.set_shape([seq_len, batch_size, latent_size])
      # Restore spatial dimensions
      output_sequence_formatted = tf.reshape(output_sequence, output_seq_shape)

    return output_sequence_formatted

  def _build(self,
             initial_inputs,
             seq_len,
             initial_state,
             is_training):
    """Adds the network into the graph.

    Args:
      initial_inputs: An batch of inputs for the first timestep. A Tensor of
        shape [batch_size, n_channels, height, width], of dtype tf.float32.
      seq_len: The length of the sequence to output. Each batch element will
        have the same output length.
      initial_state: A nested tuple of initial states for the RNN.
      is_training: True if in training mode.
    Returns:
      output_sequence: An batch of outputs of the network. A Tensor of shape
        [seq_len, batch_size, n_channels, height, width], of dtype tf.float32.
    """
    if initial_state is None:
      raise ValueError("initial_state must now be provided.")

    initial_inputs_shape = initial_inputs.get_shape().as_list()
    output_seq_shape = [seq_len] + initial_inputs_shape
    batch_size = initial_inputs_shape[0]
    latent_size = np.prod(initial_inputs_shape[1:])
    batch_sequence_lengths = [seq_len] * batch_size  # List of ints

    initial_inputs, lstm_input_shape = self._format_input(
        initial_inputs,
        initial_inputs_shape,
        batch_size,
        latent_size)

    # Feed previous output to RNN at first timestep
    # and map output of RNN to input at next timestep
    helper = AutoregressiveHelper(
        initial_inputs=initial_inputs,
        sequence_length=batch_sequence_lengths)

    # Manage output of RNN and input at next timesteps
    decoder = DeterministicDecoder(
        cell=self._core,
        helper=helper,
        initial_state=initial_state)
    decoder_output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        output_time_major=True,
        maximum_iterations=seq_len)

    output_sequence = decoder_output.rnn_output

    output_sequence = self._format_output(
        output_sequence,
        output_seq_shape,
        seq_len,
        batch_size,
        latent_size)

    # package multiple outputs in dictionary
    outputs = dict()
    outputs["output_sequence"] = output_sequence

    return outputs, final_state


class VariationalLSTMSeq2Seq(snt.AbstractModule):
  """A deep LSTM for seq2seq output with variational sampling from a learned prior."""

  def __init__(self,
               cores,
               use_conv_lstm=False,
               data_format="NCHW",
               fixed_prior=False,
               name="var_lstm_seq2seq"):
    """Constructs a VariationalLSTMSeq2Seq.

    Args:
      cores: The RNN cores to run.
      use_conv_lstm: Whether to use convolutional LSTM.
        Defaults to False (uses standard, fully-connected LSTM).
      data_format: The format of the input images: 'NCHW' or 'NHWC'.
      name: The network name. Defaults to 'deep_lstm_seq2seq'.
    Raises:
      ValueError: If data_format is not supported.
    """
    super(VariationalLSTMSeq2Seq, self).__init__(name=name)

    self._cores = cores
    self._use_conv_lstm = use_conv_lstm
    self._data_format = data_format
    self._fixed_prior = fixed_prior

  def reset_inference(self):
      """Resets the inference network (so that it can be reinitialized with the first input)."""
      is_mlp = False
      for core in self._cores['inference']._cores:
          if isinstance(core, mlp_architectures.MemoryMLP):
              core.reset()
              is_mlp = True
      return is_mlp

  def _format_input(
      self,
      initial_inputs,
      initial_inputs_shape,
      batch_size,
      latent_size):
    """Formats the input data for RNN processing."""

    if self._use_conv_lstm:
      # TODO(drewjaegle): remove once convLSTM supports NCHW
      if self._data_format == "NCHW":
        # NCHW->NHWC
        initial_inputs_formatted = tf.transpose(
            initial_inputs,
            perm=[0, 2, 3, 1])
        lstm_input_shape = [
            initial_inputs_shape[2],
            initial_inputs_shape[3],
            initial_inputs_shape[1]]
      elif self._data_format == "NHWC":
        lstm_input_shape = initial_inputs_shape[1:]
      else:
        raise ValueError("Unsupported data_format.")
    else:
      # Flatten for fully-connected LSTM
      initial_inputs_formatted = tf.reshape(
          initial_inputs, [batch_size, latent_size])
      lstm_input_shape = None

    return initial_inputs_formatted, lstm_input_shape

  def _format_output(
      self,
      output_sequence,
      output_seq_shape,
      seq_len,
      batch_size,
      latent_size):
    """Formats the output of the RNN to match the input."""
    if self._use_conv_lstm:
      if self._data_format == "NCHW":
        # TNHWC->TNCHW
        output_sequence_formatted = tf.transpose(
            output_sequence,
            perm=[0, 1, 4, 2, 3])
      output_sequence_formatted.set_shape(output_seq_shape)
    else:
      output_sequence.set_shape([seq_len, batch_size, latent_size])
      # Restore spatial dimensions
      output_sequence_formatted = tf.reshape(output_sequence, output_seq_shape)

    return output_sequence_formatted

  def _build(self,
             initial_inputs,
             seq_len,
             initial_state,
             is_training,
             input_embed_seq=None,
             input_latent_samples_seq=None,
             autoregress=False,
             reencode=False,
             encoder_cnn=None,
             decoder_cnn=None,
             use_prior_override=False,
             image_activation=None,
             init_inference=False,
             first_image=None):
    """Adds the network into the graph.

    Args:
      initial_inputs: An batch of inputs for the first timestep. A Tensor of
        shape [batch_size, n_channels, height, width], of dtype tf.float32.
      seq_len: The length of the sequence to output. Each batch element will
        have the same output length.
      initial_state: A nested tuple of initial states for the RNNs.
      is_training: True if in training mode.
      input_embed_seq: (optional) sequence of input embeddings, if None: uses
        autoregression (seq2seq)
      encoder_cnn: (optional) CNN used for autoregressive reencoding
      decoder_cnn: (optional) CNN used for autoregressive reencoding
      use_prior_override: Forces variational decoder to always use prior network.
    Returns:
      output_sequence: An batch of outputs of the network. A Tensor of shape
        [seq_len, batch_size, n_channels, height, width], of dtype tf.float32.
    """
    if initial_state is None:
      raise ValueError("initial_state must now be provided.")

    initial_inputs_shape = initial_inputs.get_shape().as_list()
    output_seq_shape = [seq_len] + initial_inputs_shape
    batch_size = initial_inputs_shape[0]
    latent_size = np.prod(initial_inputs_shape[1:])
    batch_sequence_lengths = [seq_len] * batch_size  # List of ints

    initial_inputs, lstm_input_shape = self._format_input(
        initial_inputs,
        initial_inputs_shape,
        batch_size,
        latent_size)

    # transpose NCHW -> NHWC if needed for input sequence
    if (self._use_conv_lstm
        and self._data_format == "NCHW"):
      if (input_embed_seq is not None):
          input_embed_seq = tf.transpose(input_embed_seq, [0, 1, 3, 4, 2])
      first_image = tf.transpose(first_image, (0, 2, 3, 1))
      data_format = "NHWC"
    else:
        data_format = self._data_format

    if not self._use_conv_lstm and input_embed_seq is not None:
        input_seq_len = input_embed_seq.get_shape().as_list()[0]
        input_embed_seq = tf.reshape(input_embed_seq, [input_seq_len,
                                                       batch_size,
                                                       -1])

    static_decoder = VariationalDecoder(cells=self._cores,
                                        initial_states=initial_state,
                                        is_training=is_training,
                                        seq_len=seq_len,
                                        input_sequence=input_embed_seq,
                                        input_latent_sample_sequence=input_latent_samples_seq,
                                        initial_inputs=initial_inputs,
                                        use_conv_lstm=self._use_conv_lstm,
                                        autoregress=autoregress,
                                        reencode=reencode,
                                        encoder_cnn=encoder_cnn,
                                        decoder_cnn=decoder_cnn,
                                        encoder_data_format=self._data_format,
                                        data_format=data_format,
                                        fixed_prior=self._fixed_prior,
                                        image_activation=image_activation,
                                        init_inference=init_inference,
                                        first_image=first_image)
    decoder_output, final_states, output_image_sequence = static_decoder.decode(use_prior_override)

    output_sequence = decoder_output.rnn_output

    output_sequence = self._format_output(
        output_sequence,
        output_seq_shape,
        seq_len,
        batch_size,
        latent_size)

    # package multiple outputs in dictionary
    outputs = dict()
    outputs["output_sequence"] = output_sequence
    outputs["inference_dists"] = decoder_output.inference_dist
    outputs["prior_dists"] = decoder_output.prior_dist
    outputs["z_samples"] = decoder_output.z_samples

    if reencode:
        outputs["output_image_sequence"] = output_image_sequence

    return outputs, final_states


class DeepLSTM(snt.AbstractModule):
  """A deep LSTM."""

  def __init__(self,
               core,
               use_conv_lstm=False,
               use_dynamic_rnn=True,
               data_format="NCHW",
               name="deep_lstm"):
    """Constructs a DeepLSTM.

    Args:
      core: The RNN core to run.
      use_conv_lstm: Whether to use convolutional LSTM.
        Defaults to False (uses standard, fully-connected LSTM).
      use_dynamic_rnn: Whether to use dynamic RNN unrolling. If `False`, it uses
        static unrolling. Default is `True`.
      data_format: The format of the input images: 'NCHW' or 'NHWC'.
      name: The module's name. Defaults to 'deep_lstm'.
    Raises:
      ValueError: If data_format is not supported.
    """
    super(DeepLSTM, self).__init__(name=name)

    self._core = core
    self._use_conv_lstm = use_conv_lstm
    self._data_format = data_format
    self._use_dynamic_rnn = use_dynamic_rnn

  def _format_input(self, inputs, data_input_shape):
    """Formats the input for RNN processing."""

    if self._use_conv_lstm:
      # TODO(drewjaegle): remove once convLSTM supports NCHW
      if self._data_format == "NCHW":
        # TNCHW->TNHWC
        inputs_formatted = tf.transpose(
            inputs,
            perm=[0, 1, 3, 4, 2])
        # as NHWC, for convLSTM
        lstm_input_shape = [
            data_input_shape[3],
            data_input_shape[4],
            data_input_shape[2]]
      elif self._data_format == "NHWC":
        lstm_input_shape = data_input_shape[2:]
      else:
        raise ValueError("Unsupported data_format.")
    else:
      # Flatten spatial and channel dimensions for LSTM
      inputs_formatted = tf.reshape(
          inputs,
          [data_input_shape[0], data_input_shape[1], -1])
      lstm_input_shape = None

    return inputs_formatted, lstm_input_shape

  def _format_output(self, output_sequence, data_output_shape):
    """Formats the RNN output to match the input."""
    if self._use_conv_lstm:
      # TODO(drewjaegle): remove once convLSTM supports NCHW
      if self._data_format == "NCHW":
        # TNHWC->TNCHW
        output_sequence_formatted = tf.transpose(
            output_sequence,
            perm=[0, 1, 4, 2, 3])
    else:
      output_sequence_formatted = tf.reshape(output_sequence, data_output_shape)

    return output_sequence_formatted

  def _build(self, inputs, initial_state):
    """Adds the DeepLSTM into the graph.

    Args:
      inputs: An batch of input sequences. A Tensor of shape
        [n_frames, batch_size, n_channels, height, width], of dtype tf.float32.
      initial_state: A nested tuple of initial states for the RNN.
    Returns:
      output_sequence: An batch of outputs of the network. A Tensor of shape
        [n_frames, batch_size, n_channels, height, width], of dtype tf.float32.
    """
    if initial_state is None:
      raise ValueError("initial_state must now be provided.")

    data_input_shape = inputs.get_shape().as_list()
    batch_size = data_input_shape[1]
    latent_size = np.prod(data_input_shape[2:])
    inputs, lstm_input_shape = self._format_input(inputs, data_input_shape)

    if self._use_dynamic_rnn:
      output_sequence, final_state = tf.nn.dynamic_rnn(
          cell=self._core,
          inputs=inputs,
          time_major=True,
          initial_state=initial_state)
    else:
      rnn_input_sequence = tf.unstack(inputs)
      output, final_state = tf.contrib.rnn.static_rnn(
          cell=self._core,
          inputs=rnn_input_sequence,
          initial_state=initial_state)
      output_sequence = tf.stack(output)

    output_sequence = self._format_output(
        output_sequence, data_output_shape=data_input_shape)

    return output_sequence, final_state


class LinearLayer(snt.AbstractModule):
  """A thin wrapper around snt.Linear that adds support for output non-linearity."""

  def __init__(self,
               output_size,
               initializers,
               non_linearity,
               name="linear_layer"):
    """Constructs a thin wrapper around snt.Linear that adds support for output non-linearity.

    Args:
      output_size: Size of the output of the layer.
      initializers: Initializers for layer variables.
      non_linearity: Used non-linearity. ['tanh', 'sigmoid', None]
      name: The module's name. Defaults to 'linear_layer'.
    Raises:
      ValueError: If non-linearity is not supported.
    """
    super(LinearLayer, self).__init__(name=name)

    self._output_size = output_size
    self._initializers = initializers

    self._layer = snt.Linear(self._output_size,
                             initializers=self._initializers)
    if non_linearity == 'tanh':
        self._non_linearity = tf.tanh
    elif non_linearity == 'sigmoid':
        self._non_linearity = tf.sigmoid
    elif non_linearity is None:
        self._non_linearity = tf.identity
    else:
        self._non_linearity = non_linearity

  def _build(self, inputs):
    return self._non_linearity(self._layer(inputs))

  @property
  def output_size(self):
    return self._output_size

