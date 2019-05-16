"""Convolutional network architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf
from architectures.cdna_model import CDNADecoder
from utils import AttrDict
from architectures.arch_utils import _DEFAULT_CONV_INITIALIZERS, _DEFAULT_CONV_INITIALIZERS_NO_BIAS,\
  _DEFAULT_CONV_REGULARIZERS, _DEFAULT_CONV_REGULARIZERS_NO_BIAS

_MASK_INITIALIZERS = {
    "w": tf.contrib.layers.xavier_initializer_conv2d(),
    "b": tf.constant_initializer(value=1.0)}  # Like a forget gate: keep the past.


def upsample_image(input_image, data_format):
  """Upsamples a feature map 2x in both dimensions by nearest neighbor.

  Args:
    input_image: A Tensor of shape [batch_size, n_channels, height, width].
    data_format: The format of the input images: 'NCHW' or 'NHWC'.
  Returns:
    upsampled_image: An upsampled Tensor of shape
      [batch_size, n_channels, 2*height, 2*width].
  Raises:
    ValueError: If data_format is not 'NCHW'.
  """
  if data_format != "NCHW":
    raise ValueError("Only NCHW data_format supported.")

  output_shape = input_image.get_shape().as_list()
  output_shape[2] *= 2
  output_shape[3] *= 2
  input_height_width = tf.shape(input_image)[2:]

  # NCHW->NHWC (tf.image.resize_images expects NHWC)
  upsampled_image = tf.transpose(
      input_image,
      [0, 2, 3, 1])

  upsampled_image = tf.image.resize_images(
      upsampled_image,
      size=2 * input_height_width,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
      )

  # NHWC->NCHW
  upsampled_image = tf.transpose(
      upsampled_image,
      [0, 3, 1, 2])

  upsampled_image.set_shape(output_shape)

  return upsampled_image

class DCGANEncoder(snt.AbstractModule):
  """A DCGAN model for 128x128 dimensional input."""

  def __init__(self,
               latent_size=128,
               filters=None,
               final_activation=tf.nn.tanh,
               use_input_batchnorm=False,
               dropout_rate=0.0,
               final_shrink=False,
               data_format="NCHW",
               initializers=None,
               initializers_no_bias=None,
               regularizers=None,
               regularizers_no_bias=None,
               name="dcgan_encoder"):
    """Constructs a DCGAN Encoder.

    Args:
      latent_size: The number of channels in the output layer. Defaults to 128.
      filters: An optional iterable giving the number of filters at each
        layer of the network. If None, uses the default configuration of
        [64, 128, 256, 512, 512].
      final_activation: A nonlinearity applied to the output of the network.
        Defaults to tf.nn.tanh. If None, the output is used as is.
      use_input_batchnorm: If True, batch norm is applied to the input of the
        network. Defaults to True. Batch norm is always applied to hidden
        layers.
      dropout_rate: The rate units are dropped at all layers but the input. If
        set to 0.0, dropout is not used. Defaults to 0.0
      data_format: The format of the input images. 'NHWC' and 'NCHW' supported.
      final_shrink: if True, the final layer shrinks the spatial dimensions by
      4
    Raises:
      ValueError: If data_format is not supported.
    """
    super(DCGANEncoder, self).__init__(name=name)

    if filters is None:
      filters = [64, 128, 256, 512, 512]

    self._initializers = initializers
    self._initializers_no_bias = initializers_no_bias
    self._regularizers = regularizers
    self._regularizers_no_bias = regularizers_no_bias
    self._data_format = data_format
    self._latent_size = latent_size  # Final output
    self._output_channels_list = filters
    self._activation = final_activation
    self._use_input_batchnorm = use_input_batchnorm
    self._dropout_rate = dropout_rate
    self._final_shrink = final_shrink

    if self._data_format == "NCHW":
      self._batchnorm_axis = [0, 2, 3]
    elif self._data_format == "NHWC":
      self._batchnorm_axis = [0, 1, 2]
    else:
      raise ValueError("data_format not supported.")

  def _build(self, inputs, is_training):
    """Adds the network into the graph.

    Args:
      inputs: The network input. For NCHW, A tensor of dtype float32 of shape:
        [batch_size, input_channels, height_in, width_in]
      is_training: True if running in training mode, False otherwise.
    Returns:
      outputs: The network output. For NCHW, a tensor of dtype float32, of shape
        [batch_size, output_channels, height_out, width_out]
    """
    outputs_i = inputs
    skip_connection_filters = []

    if tf.flags.FLAGS.normalization == "batchnorm":
      norm_fn = lambda x: snt.BatchNorm(
        axis=self._batchnorm_axis, update_ops_collection=None)(x, is_training, test_local_stats=False)
    elif tf.flags.FLAGS.normalization == "layernorm":
      norm_fn = lambda x: snt.LayerNorm()(x)
    else:
      raise ValueError(tf.flags.FLAGS.normalization + " not supported")
    
    # DCGAN layers (5 for 128x128)
    # 128->64->32->16->8->4, spatial dims
    use_input_bias = (not self._use_input_batchnorm)
    if use_input_bias:
      input_initializers = self._initializers
      input_regularizers = self._regularizers
    else:
      input_initializers = self._initializers_no_bias
      input_regularizers = self._regularizers_no_bias
    outputs_i = snt.Conv2D(name="first_layer",
                           output_channels=self._output_channels_list[0],
                           kernel_shape=4,
                           stride=2,
                           rate=1,
                           padding=snt.SAME,
                           use_bias=use_input_bias,
                           data_format=self._data_format,
                           initializers=input_initializers,
                           regularizers=input_regularizers)(outputs_i)

    if self._use_input_batchnorm:
      outputs_i = norm_fn(outputs_i)
    outputs_i = tf.nn.leaky_relu(outputs_i, alpha=0.2)
    skip_connection_filters.append(outputs_i)

    # Set up internal layers
    for output_channels in self._output_channels_list[1:]:
      outputs_i = snt.Conv2D(output_channels=output_channels,
                             kernel_shape=4,
                             stride=2,
                             rate=1,
                             padding=snt.SAME,
                             use_bias=False,
                             data_format=self._data_format,
                             initializers=self._initializers_no_bias,
                             regularizers=self._regularizers_no_bias)(outputs_i)

      outputs_i = norm_fn(outputs_i)
      outputs_i = tf.nn.leaky_relu(outputs_i, alpha=0.2)
      skip_connection_filters.append(outputs_i)

      if self._dropout_rate > 0.0:
        outputs_i = tf.layers.dropout(
            outputs_i, rate=self._dropout_rate, training=is_training)

    # Set up output layer, no downsampling
    outputs_i = snt.Conv2D(name="last_layer",
                           output_channels=self._latent_size,
                           kernel_shape=4,
                           stride=1,
                           rate=1,
                           padding=snt.VALID if self._final_shrink else snt.SAME,
                           use_bias=False,
                           data_format=self._data_format,
                           initializers=self._initializers_no_bias,
                           regularizers=self._regularizers_no_bias)(outputs_i)
    outputs_i = norm_fn(outputs_i)

    if self._dropout_rate > 0.0:
      outputs_i = tf.layers.dropout(
          outputs_i, rate=self._dropout_rate, training=is_training)

    if self._activation is not None:
      outputs = self._activation(outputs_i)
    else:
      outputs = outputs_i

    return outputs, skip_connection_filters


class DCGANDecoder(snt.AbstractModule):
  """A DCGAN model for 128x128 dimensional input."""

  def __init__(self,
               output_channels=1,
               filters=None,
               activation_fn=tf.nn.leaky_relu,
               use_output_batchnorm=False,
               dropout_rate=0.0,
               initial_enlarge=False,
               skip_type="skip",
               data_format="NCHW",
               initializers=None,
               initializers_no_bias=None,
               regularizers=None,
               regularizers_no_bias=None,
               image_activation=None,
               name="dcgan_decoder"):
    """Constructs a DCGANDecoder.
    Args:
      output_channels: The number of channels in the output image.
        Defaults to 1.
      filters: An optional iterable giving the number of filters at each
        layer of the network. If None, uses the default configuration of
        [512, 512, 256, 128, 64].
      activation_fn: The nonlinearity applied to all hidden layers. Defaults
        to tf.nn.leaky_relu.
      use_output_batchnorm: If True, batch norm is applied to the output of the
        network. Defaults to True. Batch norm is always applied to all other
        layers.
      dropout_rate: The rate units are dropped at all layers but the input. If
        set to 0.0, dropout is not used. Defaults to 0.0
      skip_type: Type of skip connections to use. If "skip",
        skip_connection_filters are concatenated to the corresponding decoder
        layers. If "res", skip_connection_filters are added to (a subset of) the
        corresponding decoder layers. If None, skips are ignored.
      data_format: The format of the input images. 'NHWC' and 'NCHW' supported.
    Raises:
      ValueError: If data_format is not supported.
    """
    super(DCGANDecoder, self).__init__(name=name)

    if filters is None:
      filters = [512, 512, 256, 128, 64]

    self._activation_fn = activation_fn
    self._initializers = initializers
    self._initializers_no_bias = initializers_no_bias
    self._regularizers = regularizers
    self._regularizers_no_bias = regularizers_no_bias
    self._data_format = data_format
    self._output_channels_list = filters
    self.output_channels = output_channels
    self._use_output_batchnorm = use_output_batchnorm
    self._dropout_rate = dropout_rate
    self._skip_type = skip_type
    self._initial_enlarge = initial_enlarge
    self.image_activation = image_activation

    if not (self._skip_type is None or self._skip_type in ["skip", "res"]):
      raise ValueError("Unknown skip connection type.")

    if self._data_format == "NCHW":
      self._batchnorm_axis = [0, 2, 3]
      self._channel_axis = 1
    elif self._data_format == "NHWC":
      self._batchnorm_axis = [0, 1, 2]
      self._channel_axis = 3
    else:
      raise ValueError("data_format not supported.")
    
    self.use_output_bias = (not self._use_output_batchnorm)
    if self.use_output_bias:
      self.output_initializers = self._initializers
      self.output_regularizers = self._regularizers
    else:
      self.output_initializers = self._initializers_no_bias
      self.output_regularizers = self._regularizers_no_bias
  

  def _connect_res(self, latents_now, latents_skip, res_mask):
    """Adds residual connections as skips to the first subset of filters."""
    skip_filter_shape = latents_skip.get_shape()
    n_skip_filters = skip_filter_shape[self._channel_axis]
    output_channels = latents_now.get_shape()[self._channel_axis]

    if res_mask is not None:
      latents_now = res_mask * latents_skip + (1 - res_mask) * latents_now
    else:
      latents_now += latents_skip

    return latents_now

  def _connect_skips(self, inputs, skips, res_mask):
    """Connects the skip connections into the network."""
    if self._skip_type is not None:
      if self._skip_type == "skip":
        inputs = tf.concat(
            [inputs, skips], axis=self._channel_axis)
      elif self._skip_type == "res":
        inputs = self._connect_res(inputs, skips, res_mask)

    return inputs

  def _build_residual_mask(self, mask_inputs, stride=2):
    """Builds a spatial output mask using the input to the last layer."""
    output_mask_logits = snt.Conv2DTranspose(
      name="output_mask_logits",
      output_channels=1,
      kernel_shape=4,
      stride=stride,
      padding=snt.SAME,
      use_bias=True,
      data_format=self._data_format,
      initializers=_MASK_INITIALIZERS,
      regularizers=self._regularizers)(mask_inputs)

    output_mask = tf.sigmoid(
        output_mask_logits,
        name="output_mask")

    return output_mask

  def _build_layer(self, build_recursive_skips, inputs, output_channels, layer_kwargs, is_training, decoder_skips,
                   skip_connection_filters=None, is_final=False, use_batchnorm=True):
    
    if build_recursive_skips:
      res_mask = self._build_residual_mask(inputs, stride=layer_kwargs.stride)
    else:
      res_mask = None
      
    outputs = snt.Conv2DTranspose(
      output_channels=output_channels,
      kernel_shape=4,
      data_format=self._data_format,
      **layer_kwargs)(inputs)
  
    if use_batchnorm:
      outputs = self.norm_fn(outputs)
  
    if not is_final:
      outputs = self._activation_fn(outputs)
      if skip_connection_filters is not None:
        outputs = self._connect_skips(
          outputs,
          skip_connection_filters,
          res_mask)

      decoder_skips.append(outputs)
    
      if self._dropout_rate > 0.0:
        outputs = tf.layers.dropout(
          outputs, rate=self._dropout_rate, training=is_training)
        
    return res_mask, outputs

  def _build(
      self,
      inputs,
      is_training,
      skip_connection_filters=None,
      build_recursive_skips=False,
      prev_img=None,
      first_img=None,
      goal_img=None):
    """Adds the network into the graph.

    Args:
      inputs: The network input. For NCHW, A tensor of dtype float32 of shape:
        [batch_size, input_channels, height_in, width_in]
      is_training: True if running in training mode, False otherwise.
      skip_connection_filters: An iterable of input skip connections, in order
        from earliest to latest in the encoder architecture.
      build_recursive_skips: If True, returns decoder layers for use as skips.
    Returns:
      outputs: The network output. For NCHW, a tensor of dtype float32, of shape
        [batch_size, output_channels, height_out, width_out]
    """
    outputs_i = inputs
    decoder_skips = []

    if tf.flags.FLAGS.normalization == "batchnorm":
      self.norm_fn = lambda x: snt.BatchNorm(
        axis=self._batchnorm_axis, update_ops_collection=None)(x, is_training, test_local_stats=False)
    elif tf.flags.FLAGS.normalization == "layernorm":
      self.norm_fn = lambda x: snt.LayerNorm()(x)
    else:
      raise ValueError(tf.flags.FLAGS.normalization + " not supported")
  
    # DCGAN layers (5 for 128x128)
    # 4->8->16->32->64->128, spatial dims

    # Set up  layers
    for i, output_channels in enumerate(self._output_channels_list):
      layer_kwargs = AttrDict(use_bias=False,
                              initializers=self._initializers_no_bias,
                              regularizers=self._regularizers_no_bias,
                              padding=snt.SAME,
                              name="conv_2d_transpose",
                              stride=2)
      kwargs = AttrDict()
        
      if i == 0:
        layer_kwargs.update(name="first_layer",
                            padding=snt.VALID if self._initial_enlarge else snt.SAME,
                            stride=1)
        
      if skip_connection_filters is not None:
        kwargs.skip_connection_filters = skip_connection_filters[-(i + 1)]
        
      res_mask, outputs_i = self._build_layer(
        build_recursive_skips, outputs_i, output_channels, layer_kwargs, is_training, decoder_skips, **kwargs)

    layer_kwargs.update(name="last_layer",
                        use_bias=self.use_output_bias,
                        initializers=self.output_initializers,
                        regularizers=self.output_regularizers)
    kwargs.use_batchnorm = self._use_output_batchnorm
    kwargs.is_final = True

    res_mask, outputs = self._build_layer(
      build_recursive_skips, outputs_i, self.output_channels, layer_kwargs, is_training, decoder_skips, **kwargs)
    
    if tf.flags.FLAGS.use_cdna_decoder:
      assert prev_img is not None
      assert first_img is not None
      cdna_module = CDNADecoder(
        num_cdna_kernels=4, data_format=self._data_format, cdna_kernel_size=5, num_final_feats=64,
        output_channels=self.output_channels, stride=2, image_activation=self.image_activation)
      outputs, goal_img_warped = cdna_module(outputs_i, inputs, prev_img, first_img, is_training, goal_img)
      
      return outputs, None, None, goal_img_warped

    if build_recursive_skips:
      decoder_skips = decoder_skips[::-1]  # Get in same order
      return outputs, decoder_skips, res_mask, None
    else:
      return outputs
    
    

class DCGAN3DEncoder(snt.AbstractModule):
  """A DCGAN model for 128x128 dimensional input."""

  def __init__(self,
               latent_size=128,
               filters=None,
               final_activation=tf.nn.tanh,
               use_input_batchnorm=False,
               data_format="NCDHW",
               initializers_no_bias=None,
               regularizers_no_bias=None,
               name="dcgan_encoder"):
    """Constructs a spatiotemporal DCGAN Encoder.

    Args:
      latent_size: The number of channels in the output layer. Defaults to 128.
      filters: An optional iterable giving the number of filters at each
        layer of the network. If None, uses the default configuration of
        [64, 128, 256, 512, 512].
      final_activation: A nonlinearity applied to the output of the network.
        Defaults to tf.nn.tanh. If None, the output is used as is.
      use_input_batchnorm: If True, batch norm is applied to the input of the
        network. Defaults to True. Batch norm is always applied to hidden
        layers.
      data_format: The format of the input images. Only 'NCDHW' supported.
    Raises:
      ValueError: If data_format is not 'NCDHW'.
    """
    super(DCGAN3DEncoder, self).__init__(name=name)

    if initializers_no_bias is None:
      initializers_no_bias = _DEFAULT_CONV_INITIALIZERS_NO_BIAS
    if regularizers_no_bias is None:
      regularizers_no_bias = _DEFAULT_CONV_REGULARIZERS_NO_BIAS
    if filters is None:
      filters = [64, 128, 256, 512, 512]

    self._initializers = initializers_no_bias
    self._regularizers = regularizers_no_bias
    self._data_format = data_format
    self._latent_size = latent_size  # Final output
    self._output_channels_list = filters
    self._activation = final_activation
    self._use_input_batchnorm = use_input_batchnorm

    if self._data_format == "NCDHW":
      self._batchnorm_axis = [0, 2, 3, 4]
    else:
      raise ValueError("data_format must be NCDHW.")

  def _build(self, inputs, is_training):
    """Adds the network into the graph.

    Args:
      inputs: The network input. A tensor of dtype float32, of shape:
        [batch_size, input_channels, height_in, width_in]
      is_training: True if running in training mode, False otherwise.
    Returns:
      outputs: The network output. A tensor of dtype float32, of shape
        [batch_size, output_channels, height_out, width_out]
    """
    outputs_i = inputs

    # DCGAN layers (5 for 128x128)
    # 128->64->32->16->8->4, spatial dims

    outputs_i = snt.Conv3D(name="first_layer",
                           output_channels=self._output_channels_list[0],
                           kernel_shape=4,
                           stride=2,
                           rate=1,
                           padding=snt.SAME,
                           use_bias=False,
                           data_format=self._data_format,
                           initializers=self._initializers,
                           regularizers=self._regularizers)(outputs_i)

    if self._use_input_batchnorm:
      outputs_i = snt.BatchNorm(
        axis=self._batchnorm_axis)(outputs_i, is_training, test_local_stats=False)
    outputs_i = tf.nn.leaky_relu(outputs_i, alpha=0.2)

    # Set up internal layers
    for output_channels in self._output_channels_list[1:]:
      outputs_i = snt.Conv3D(output_channels=output_channels,
                             kernel_shape=4,
                             stride=2,
                             rate=1,
                             padding=snt.SAME,
                             use_bias=False,
                             data_format=self._data_format,
                             initializers=self._initializers,
                             regularizers=self._regularizers)(outputs_i)
      outputs_i = snt.BatchNorm(
        axis=self._batchnorm_axis)(outputs_i, is_training, test_local_stats=False)
      outputs_i = tf.nn.leaky_relu(outputs_i, alpha=0.2)

    # Set up output layer, no downsampling
    outputs_i = snt.Conv3D(name="last_layer",
                           output_channels=self._latent_size,
                           kernel_shape=4,
                           stride=1,
                           rate=1,
                           padding=snt.SAME,
                           use_bias=False,
                           data_format=self._data_format,
                           initializers=self._initializers,
                           regularizers=self._regularizers)(outputs_i)
    outputs_i = snt.BatchNorm(
      axis=self._batchnorm_axis)(outputs_i, is_training, test_local_stats=False)

    if self._activation is not None:
      outputs = self._activation(outputs_i)
    else:
      outputs = outputs_i

    return outputs


class SimpleConvNet(snt.AbstractModule):
  """A simple convolutional network."""

  def __init__(self,
               conv_spec,
               network_type="encoder",
               use_bias=True,
               nonlinearity=tf.nn.leaky_relu,
               skip_type=None,
               data_format="NCHW",
               initializers=None,
               initializers_no_bias=None,
               regularizers=None,
               regularizers_no_bias=None,
               name=None):
    """Constructs a SimpleConvNet.

    Args:
      conv_spec: A tuple specifying the parameters of the network. Each entry is
        a NamedTuple, with the values of the corresponding layer.
      network_type: Determines whether the network is an 'encoder' or 'decoder'.
        The former can specify pooling layers, while the latter can specify
        nearest neighbor upsampling layers. Defaults to 'encoder'.
      use_bias: Use or omit bias parameter in conv layers. Default to True.
      nonlinearity: The point nonlinearity to use after convolutional layers.
        Must be a Tensorflow function mapping Tensors to Tensors of the same
        dimensionality. Defaults to tf.nn.leaky_relu.
      skip_type: Type of skip connections to use. If "skip",
        skip_connection_filters are concatenated to the corresponding decoder
        layers. If "res", skip_connection_filters are added to (a subset of) the
        corresponding decoder layers. If None, skips are ignored. Defaults to
        None (not implemented here).
      data_format: The format of the input images. 'NHWC' and 'NCHW' supported.
    Raises:
      ValueError: If data_format is not supported.
    """
    # TODO(drewjaegle): implement support for skip connections.
    if name is None:
      name = "simple_conv_net_{}".format(network_type)
    super(SimpleConvNet, self).__init__(name=name)

    self._conv_spec = conv_spec
    self._use_bias = use_bias
    self._initializers = initializers
    self._initializers_no_bias = initializers_no_bias
    self._regularizers = regularizers
    self._regularizers_no_bias = regularizers_no_bias
    self._nonlinearity = nonlinearity
    self._data_format = data_format
    self._network_type = network_type
    self._skip_type = skip_type

    if self._network_type not in {"encoder", "decoder"}:
      raise ValueError("Unknown network_type.")

    if self._data_format == "NCHW":
      self._batchnorm_axis = [0, 2, 3]
    elif self._data_format == "NHWC":
      self._batchnorm_axis = [0, 1, 2]
    else:
      raise ValueError("data_format not supported.")

  def _build(
      self,
      inputs,
      is_training,
      skip_connection_filters=None):
    """Adds the network into the graph.

    Args:
      inputs: The network input. For NCHW, A tensor of dtype float32 of shape:
        [batch_size, input_channels, height_in, width_in]
      is_training: True if running in training mode, False otherwise.
      skip_connection_filters: An iterable of input skip connections, in order
        from earliest to latest in the encoder architecture.
    Returns:
      outputs: The network output. For NCHW, A tensor of dtype float32 of shape
        [batch_size, output_channels, height_out, width_out]
      Curently returns None for compatibility with skip_connections interface.
    """
    outputs_i = inputs
    for layer_spec in self._conv_spec.layers:
      if self._use_bias:
        initializers_i = self._initializers_no_bias
        regularizers_i = self._regularizers_no_bias
      else:
        initializers_i = self._initializers
        regularizers_i = self._regularizers

      outputs_i = snt.Conv2D(output_channels=layer_spec.output_channels,
                             kernel_shape=layer_spec.kernel_shape,
                             stride=layer_spec.stride,
                             rate=layer_spec.rate,
                             padding=snt.SAME,
                             use_bias=self._use_bias,
                             data_format=self._data_format,
                             initializers=initializers_i,
                             regularizers=regularizers_i)(outputs_i)
      if layer_spec.use_nonlinearity:
        outputs_i = self._nonlinearity(outputs_i)
      if layer_spec.use_batchnorm:
        outputs_i = snt.BatchNorm(
            axis=self._batchnorm_axis, update_ops_collection=None)(outputs_i, is_training, test_local_stats=False)
      if self._network_type == "encoder":
        if layer_spec.use_pool:
          outputs_i = tf.nn.pool(outputs_i,
                                window_shape=[2, 2],
                                strides=[2, 2],
                                pooling_type="MAX",
                                padding="VALID",
                                data_format=self._data_format)
      else:
        if layer_spec.use_upsample:
          outputs_i = upsample_image(outputs_i, self._data_format)
    outputs = outputs_i

    if self._network_type == "encoder":
      skip_connections = None
      return outputs, skip_connections
    else:
      return outputs


def build_cnn(
    conv_spec,
    network_type,
    decoder_output_channels=1,
    data_format="NCHW",
    regularizer_type="default",
    initializer_type="default",
    image_activation=None):
  """Builds a CNN using the desired conv_spec.

  Args:
    conv_spec: A namedtuple with the parameters of the conv architecture.
    network_type: A string specifying how the conv_spec should be parsed.
    output_channels: Number of output channels of decoder, defaults to 1.
    data_format: The order of dimensions in the data tensor.
    regularizer_type: A string specifying how the regularizers should be set up.
      Defaults to "default", which uses the global conv default.
    initializer_type: A string specifying how the initializers should be set up.
      Defaults to "default", which uses the global conv default.
  """
  if initializer_type == "default":
    initializers = _DEFAULT_CONV_INITIALIZERS
    initializers_no_bias = _DEFAULT_CONV_INITIALIZERS_NO_BIAS
  elif initializer_type is None:
    initializers = None
    initializers_no_bias = None
  else:
    raise ValueError("Unknown initializer setup.")

  if regularizer_type == "default":
    regularizers = _DEFAULT_CONV_REGULARIZERS
    regularizers_no_bias = _DEFAULT_CONV_REGULARIZERS_NO_BIAS
  elif regularizer_type is None:
    regularizers = None
    regularizers_no_bias = None
  else:
    raise ValueError("Unknown regularizer setup.")

  if network_type not in ["encoder", "decoder"]:
    raise ValueError("Unknown network_type")
  if conv_spec.spec_type == "simple_conv_net":
    cnn = SimpleConvNet(
        conv_spec,
        network_type=network_type,
        nonlinearity=tf.nn.leaky_relu,
        data_format=data_format,
        initializers=initializers,
        initializers_no_bias=initializers_no_bias,
        regularizers=regularizers,
        regularizers_no_bias=regularizers_no_bias,
        skip_type=conv_spec.skip_type)

  elif conv_spec.spec_type == "dcgan":
    final_shrink = False
    if network_type == "encoder":
      if conv_spec.layers == "small":
        filters = [64, 64, 96, 96]
      elif conv_spec.layers == "med":
        filters = [32, 64, 128, 128, 128]
      elif conv_spec.layers == "denton_mnist":
        filters = [64, 128, 256, 512]
        final_shrink = True
      elif conv_spec.layers == "denton_mnist_small":
        filters = [64, 64, 96, 96]
        final_shrink = True
      elif conv_spec.layers == "denton_mnist_very_small_res":
        filters = [64, 96, 96]
        final_shrink = True
      elif conv_spec.layers == "ucf_gray_large":
        filters = [64, 128, 256, 256, 512, 512]
      else:
        filters = None
      cnn = DCGANEncoder(
          filters=filters,
          latent_size=conv_spec.dcgan_latent_size,
          final_activation=tf.nn.tanh if tf.flags.FLAGS.activate_latents else None,
          data_format=data_format,
          use_input_batchnorm=conv_spec.dcgan_use_image_bn,
          dropout_rate=conv_spec.dcgan_dropout_rate,
          final_shrink=final_shrink,
          initializers=initializers,
          initializers_no_bias=initializers_no_bias,
          regularizers=regularizers,
          regularizers_no_bias=regularizers_no_bias)

    elif network_type == "decoder":
      initial_enlarge=False
      if conv_spec.layers == "small":
        filters = [96, 96, 64, 64]
      elif conv_spec.layers == "med":
        filters = [128, 128, 128, 64, 32]
      elif conv_spec.layers == "denton_mnist":
        filters = [512, 256, 128, 64]
        initial_enlarge = True
      elif conv_spec.layers == "denton_mnist_small":
        filters = [96, 96, 64, 64]
        initial_enlarge = True
      elif conv_spec.layers == "denton_mnist_very_small_res":
        filters = [96, 96, 64]
        initial_enlarge = True
      elif conv_spec.layers == "ucf_gray_large":
        filters = [512, 512, 256, 256, 128, 64]
      else:
        filters = None
      cnn = DCGANDecoder(
        filters=filters,
        data_format=data_format,
        use_output_batchnorm=conv_spec.dcgan_use_image_bn,
        output_channels=decoder_output_channels,
        dropout_rate=conv_spec.dcgan_dropout_rate,
        initial_enlarge=initial_enlarge,
        initializers=initializers,
        initializers_no_bias=initializers_no_bias,
        regularizers=regularizers,
        regularizers_no_bias=regularizers_no_bias,
        skip_type=conv_spec.skip_type,
        image_activation=image_activation)
  else:
    raise ValueError("Unknown conv_spec type!")

  return cnn