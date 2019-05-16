"""Utilities for architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from specs import module_specs
import tensorflow as tf


_DEFAULT_CONV_INITIALIZERS = {
    "w": tf.contrib.layers.xavier_initializer_conv2d(),
    "b": tf.constant_initializer(value=0.01)}
_DEFAULT_CONV_REGULARIZERS = {
    "w": tf.contrib.layers.l2_regularizer(scale=1.0),
    "b": tf.contrib.layers.l2_regularizer(scale=1.0)}

_DEFAULT_CONV_INITIALIZERS_NO_BIAS = {
    "w": tf.contrib.layers.xavier_initializer_conv2d()}
_DEFAULT_CONV_REGULARIZERS_NO_BIAS = {
    "w": tf.contrib.layers.l2_regularizer(scale=1.0)}

def _check_module_is_conv(module_spec):
  """Checks whether a module is a convolutional type.
  Args:
    module_spec: The spec for the module.
  Returns:
    is_conv_type: True if module is a convolutional type, False if module is a
      fully-connected type.
  Raises:
    ValueError if either module_spec is of an unknown type.
  """
  if any([isinstance(module_spec, fc_type)
          for fc_type in module_specs.FC_TYPES]):
    is_conv_type = False
  elif any([isinstance(module_spec, conv_type)
            for conv_type in module_specs.CONV_TYPES]):
    is_conv_type = True
  else:
    raise ValueError("Unknown module_spec type.")

  return is_conv_type

def check_rnn_is_conv(rnn_spec):
  """Checks that an RNN is consistently convolutional or fully connected.
  Args:
    rnn_spec: The spec for the RNN.
  Returns:
    use_conv_lstm: True if conv LSTM, False if fully-connected.
  Raises:
    ValueError if either lstm_spec contains entries of inconsistent type.
  """
  rnn_is_conv = _check_module_is_conv(rnn_spec[0])

  for lstm_spec_i in rnn_spec:
    module_i_is_conv = _check_module_is_conv(lstm_spec_i)
    if not module_i_is_conv == rnn_is_conv:
      raise ValueError("Inconsistent rnn_spec types.")

  return rnn_is_conv

def check_rnns_conv(encoder_spec, decoder_spec, variational):
  """Checks the format of LSTM encoder and decoder specifications.
     Supports multiple sub-specs per spec.

  Args:
    encoder_spec: The spec for the encoder LSTM (can be list of specs).
    decoder_spec: The spec for the decoder LSTM (can be list of specs).
  Returns:
    use_conv_lstm: True if conv LSTM, False if fully-connected.
  Raises:
    ValueError: If the output module of the encoder has a different size than
      the input module of the decoder, or if the LSTMs are of different type.
  """
  is_convs = []
  if not variational:
    encoder_spec = [encoder_spec]
    decoder_spec = [decoder_spec]
  for e_spec, d_spec in zip(encoder_spec, decoder_spec):
    encoder_is_conv = check_rnn_is_conv(e_spec)
    decoder_is_conv = check_rnn_is_conv(d_spec)
    if encoder_is_conv != decoder_is_conv:
      raise ValueError("Inconsistent encoder and decoder LSTM types.")
    is_convs.append(encoder_is_conv)

  # check whether all are equal
  # TODO(karl): support different architecture types in one RNN
  if not is_convs.count(is_convs[0]) == len(is_convs):
    raise ValueError(["All LSTM specifications need to be either fully connected",
                     " or convolutional!"])

  return all(is_convs)

def estimate_next_phi(phi_seq, dphi_seq, mapping="add"):
  """Maps (phi, dphi)_t to phi_{t+1}.

  Args:
    phi_seq: A sequence of states.
    dphi_seq: A sequence of state deltas.
    mapping: A string specifying what type of mapping to use to estimate
      the values of next_phi. Defaults to "add".
  Returns:
    next_phi_seq: A sequence of estimated next states.
  Raises:
    ValueError if mapping type is not known.
  """
  if mapping == "add":
    next_phi_seq = phi_seq + dphi_seq
  else:
    raise ValueError("Unknown mapping {}".format(mapping))

  return next_phi_seq

def parse_phi_seq(phi_seq_raw,
                  n_frames_input,
                  n_stack=2,
                  mapping="add",
                  data_format="NCHW"):
  """Parses the CNN output into states for different times.

  Args:
    phi_seq_raw: The raw latent output by the multiframe CNN. A Tensor of shape
      [(N-n_stack-1), batch_size, C*n_stack , H, W]
      or [..., H, W, C*n_stack].
    n_frames_input: The number of frames used as input before prediction is
      started.
    n_stack: The number of frames to stack as input to the network.
      Defaults to 2.
    mapping: The functional form used to map phi and dphi to phi'. Defaults to
      "add".
    data_format: The format of the input images: 'NCHW' or 'NHWC'.
      Defaults to "NCHW".
  Returns:
    phi_seqs:
    dphi_seqs:
    next_phi_seqs:
  """
  # TODO(drewjaegle): implement more general form of this for longer estimates
  if n_stack > 2:
    raise ValueError("Not implemented for n_stack > 2!!")

  phi_seqs = {}
  dphi_seqs = {}

  # TODO(drewjaegle): modify docstring when done.
  # First, just slice them into the two seqs.
  if data_format == "NCHW":
    n_channels_total = phi_seq_raw.get_shape()[2]
    phi_seq_data = phi_seq_raw[:, :, :(n_channels_total / 2), :, :]
    dphi_seq_data = phi_seq_raw[:, :, (n_channels_total / 2):, :, :]
  elif data_format == "NHWC":
    n_channels_total = phi_seq_raw.get_shape()[2]
    phi_seq_data = phi_seq_raw[..., :(n_channels_total / 2)]
    dphi_seq_data = phi_seq_raw[..., (n_channels_total / 2)]

  next_phi_data = estimate_next_phi(phi_seq_data, dphi_seq_data,
                                    mapping="add")

  # Then, convert to different input types
  phi_seqs = {
    "encoded_seq_reconstruct": phi_seq_data[:n_frames_input, ...],
    "future_latents_true": tf.concat([phi_seq_data[n_frames_input:, ...],
                                      next_phi_data[-1:, ...]], axis=0),
    "encoded_seq_future_inputs": phi_seq_data[(n_frames_input - 1):, ...],
    "match_next": phi_seq_data[1:, ...],
  }
  dphi_seqs = {
    "full_seq": phi_seq_data,
  }
  next_phi_seqs = {
    "match_next": next_phi_data[:-1, ...],
  }

  return phi_seqs, dphi_seqs, next_phi_seqs

def make_time_stack(full_sequence, n_stack=2, data_format="NCHW"):
  """Stacks frames of an input timeseries.

  Args:
    full_seq: The input sequence, of shape [T, N, C, H, W] or [T, N, H, W, C],
      where T is the number of timesteps in the sequence, N the batch size, C
      the number of channels, and H and W the spatial height and width.
    n_stack: The number of consecutive frames to stack. Defaults to 2.
    data_format: The image format, "NCHW" or "NHWC". Defaults to "NCHW"
  Returns:
    stacked_seq: The output sequence, of shape [T-(n_stack-1), N, n_stack*C, H, W]
      or [T-(n_stack-1), N, H, W, n_stack*C]. Images are stacked as
      [t_0, ..., t_C, {t+1}_0, ..., {t+1}_C] in the channel dimension.
  """
  if data_format == "NCHW":
    stack_axis = 2
  else:
    stack_axis = 4

  stacked_seq = tf.concat(
      [full_sequence[:-(n_stack-1), ...],
      full_sequence[(n_stack-1):, ...]], axis=stack_axis)

  return stacked_seq
