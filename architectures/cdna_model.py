from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
from sonnet.python.modules import basic
from sonnet.python.modules import rnn_core
from sonnet.python.modules import util
import numpy as np
import tensorflow as tf
import sonnet as snt
from architectures.rnn_architectures import build_lstm_module, build_conv_lstm_module
from specs.module_specs import EncodingLayerSpec
from utils import shape

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
RELU_SHIFT = 1e-12

def _build_conv_layer(conv_spec, data_format):
    return snt.Conv2D(output_channels=conv_spec.output_channels,
                         kernel_shape=conv_spec.kernel_shape,
                         stride=conv_spec.stride,
                         rate=conv_spec.rate,
                         padding=snt.SAME,
                         use_bias=True,
                         data_format=data_format,
                         initializers=_DEFAULT_CONV_INITIALIZERS,
                         regularizers=_DEFAULT_CONV_REGULARIZERS)


def _build_conv_t_layer(conv_spec, data_format):
    return snt.Conv2DTranspose(output_channels=conv_spec.output_channels,
                         kernel_shape=conv_spec.kernel_shape,
                         stride=conv_spec.stride,
                         padding=snt.SAME,
                         use_bias=True,
                         data_format=data_format,
                         initializers=_DEFAULT_CONV_INITIALIZERS,
                         regularizers=_DEFAULT_CONV_REGULARIZERS)


def _get_default_conv_spec(num_outputs, stride=1):
    return EncodingLayerSpec(output_channels=num_outputs,
                             kernel_shape=3,
                             stride=stride,
                             rate=1,
                             use_nonlinearity=True,
                             use_batchnorm=True,
                             use_pool=False)


def pad2d_paddings(inputs, size, strides=(1, 1), rate=(1, 1), padding='SAME'):
    """
    From: https://github.com/alexlee-gk/video_prediction
    Computes the paddings for a 4-D tensor according to the convolution padding algorithm.

    See pad2d.

    Reference:
        https://www.tensorflow.org/api_guides/python/nn#convolution
        https://www.tensorflow.org/api_docs/python/tf/nn/with_space_to_batch
    """
    size = np.array(size) if isinstance(size, (tuple, list)) else np.array([size] * 2)
    strides = np.array(strides) if isinstance(strides, (tuple, list)) else np.array([strides] * 2)
    rate = np.array(rate) if isinstance(rate, (tuple, list)) else np.array([rate] * 2)
    if np.any(strides > 1) and np.any(rate > 1):
        raise ValueError("strides > 1 not supported in conjunction with rate > 1")
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 4
    input_size = np.array(input_shape[1:3])
    if padding in ('SAME', 'FULL'):
        if np.any(rate > 1):
            # We have two padding contributions. The first is used for converting "SAME"
            # to "VALID". The second is required so that the height and width of the
            # zero-padded value tensor are multiples of rate.

            # Spatial dimensions of the filters and the upsampled filters in which we
            # introduce (rate - 1) zeros between consecutive filter values.
            dilated_size = size + (size - 1) * (rate - 1)
            pad = dilated_size - 1
        else:
            pad = np.where(input_size % strides == 0,
                           np.maximum(size - strides, 0),
                           np.maximum(size - (input_size % strides), 0))
        if padding == 'SAME':
            # When full_padding_shape is odd, we pad more at end, following the same
            # convention as conv2d.
            pad_start = pad // 2
            pad_end = pad - pad_start
        else:
            pad_start = pad
            pad_end = pad
        if np.any(rate > 1):
            # More padding so that rate divides the height and width of the input.
            # TODO: not sure if this is correct when padding == 'FULL'
            orig_pad_end = pad_end
            full_input_size = input_size + pad_start + orig_pad_end
            pad_end_extra = (rate - full_input_size % rate) % rate
            pad_end = orig_pad_end + pad_end_extra
        paddings = [[0, 0],
                    [pad_start[0], pad_end[0]],
                    [pad_start[1], pad_end[1]],
                    [0, 0]]
    elif padding == 'VALID':
        paddings = [[0, 0]] * 4
    else:
        raise ValueError("Invalid padding scheme %s" % padding)
    return paddings


def pad2d(inputs, size, strides=(1, 1), rate=(1, 1), padding='SAME', mode='CONSTANT'):
    """
    From: https://github.com/alexlee-gk/video_prediction
    Pads a 4-D tensor according to the convolution padding algorithm.

    Convolution with a padding scheme
        conv2d(..., padding=padding)
    is equivalent to zero-padding of the input with such scheme, followed by
    convolution with 'VALID' padding
        padded = pad2d(..., padding=padding, mode='CONSTANT')
        conv2d(padded, ..., padding='VALID')

    Args:
        inputs: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        padding: A string, either 'VALID', 'SAME', or 'FULL'. The padding algorithm.
        mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).

    Returns:
        A 4-D tensor.

    Reference:
        https://www.tensorflow.org/api_guides/python/nn#convolution
    """
    paddings = pad2d_paddings(inputs, size, strides=strides, rate=rate, padding=padding)
    if paddings == [[0, 0]] * 4:
        outputs = inputs
    else:
        outputs = tf.pad(inputs, paddings, mode=mode)
    return outputs


def identity_kernel(kernel_size):
    """From: https: // github.com / alexlee - gk / video_prediction"""
    kh, kw = kernel_size
    kernel = np.zeros(kernel_size)

    def center_slice(k):
        if k % 2 == 0:
            return slice(k // 2 - 1, k // 2 + 1)
        else:
            return slice(k // 2, k // 2 + 1)

    kernel[center_slice(kh), center_slice(kw)] = 1.0
    kernel /= np.sum(kernel)
    return kernel


class CDNA:
    """ Implements Convolutional Neural Advection operation from Finn&Levine 2016 """
    
    def __init__(self, data_format, num_masks):
        self._data_format = data_format
        self._num_masks = num_masks
     
    def apply(self, input_img, first_input_img, scratch_img, cdna_kernels, masks):
        """ The main function """
        transformed_images = self.get_transformed_images(input_img, first_input_img, scratch_img, cdna_kernels)

        return self.compose_images(masks, transformed_images)
    
    def get_transformed_images(self, input_img, first_input_img, scratch_img, cdna_kernels):
        with tf.variable_scope("cdna_warp"):
            warped_imgs = self._apply_cdna_kernels(input_img, cdna_kernels)
          
        with tf.name_scope("stack_transformed_images"):
            transformed_images = [input_img, first_input_img, scratch_img]
            transformed_images.extend(warped_imgs)
        
        return transformed_images
        
    def compose_images(self, masks, transformed_images):
        # stack transformed images
        stack_axis = -1 if self._data_format == "NHWC" else 1
        with tf.name_scope("mask_fusion"):
            masks = tf.split(masks, self._num_masks, axis=stack_axis)
            
            return tf.add_n([transformed_image * mask
                                  for transformed_image, mask in zip(transformed_images, masks)])
        
    def _apply_cdna_kernels(self, input, kernels, dilation_rate=(1, 1)):
        """From: https://github.com/alexlee-gk/video_prediction"""
        if self._data_format == "NCHW":
            input = tf.transpose(input, (0, 2, 3, 1))
        batch_size, height, width, color_channels = input.get_shape().as_list()
        batch_size, kernel_height, kernel_width, num_transformed_images = kernels.get_shape().as_list()
        kernel_size = [kernel_height, kernel_width]
        image_padded = pad2d(input, kernel_size, rate=dilation_rate, padding='SAME', mode='SYMMETRIC')
        # Treat the color channel dimension as the batch dimension since the same
        # transformation is applied to each color channel.
        # Treat the batch dimension as the channel dimension so that
        # depthwise_conv2d can apply a different transformation to each sample.
        kernels = tf.transpose(kernels, [1, 2, 0, 3])
        kernels = tf.reshape(kernels, [kernel_size[0], kernel_size[1], batch_size, num_transformed_images])
        # Swap the batch and channel dimensions.
        image_transposed = tf.transpose(image_padded, [3, 1, 2, 0])
        # Transform image.
        outputs = tf.nn.depthwise_conv2d(image_transposed, kernels, [1, 1, 1, 1], padding='VALID', rate=dilation_rate)
        # Transpose the dimensions to where they belong.
        outputs = tf.reshape(outputs, [color_channels, height, width, batch_size, num_transformed_images])
        if self._data_format == "NHWC":
            outputs = tf.transpose(outputs, [4, 3, 1, 2, 0])
        elif self._data_format == "NCHW":
            outputs = tf.transpose(outputs, [4, 3, 0, 1, 2])
        else:
            raise ValueError("Data format %s currently not supported in CDNA model!" % self._data_format)
        outputs = tf.unstack(outputs, axis=0)
        return outputs

    
class CDNADecoder(snt.AbstractModule):
    """
    Given a latent state, produces an image via the CDNA approach by Finn&Levine 2016.
    https://arxiv.org/pdf/1804.01523.pdf
    """
    
    def __init__(self,
                 num_cdna_kernels,
                 data_format,
                 cdna_kernel_size,
                 num_final_feats,
                 output_channels,
                 stride,
                 image_activation=tf.nn.tanh,
                 name="cdna_module"):
        super(CDNADecoder, self).__init__(name=name)
        
        self.cdna_kernel_size = cdna_kernel_size
        self.num_cdna_kernels = num_cdna_kernels
        self._data_format = data_format
        self._num_masks = 1 + 1 + 1 + num_cdna_kernels
        self.num_final_feats = num_final_feats
        self.output_channels = output_channels
        self.stride = stride
        self.image_activation = image_activation
        
        if self._data_format == "NCHW":
            self._batchnorm_axis = [0, 2, 3]
        elif self._data_format == "NHWC":
            self._batchnorm_axis = [0, 1, 2]
        else:
            raise ValueError("data_format not supported.")
      
        self._setup_modules()
      
    def _setup_modules(self):
        # CDNA kernel linear layer
        with tf.variable_scope("linear_cdna_kernel"):
            kernel_size = self.num_cdna_kernels * np.power(self.cdna_kernel_size, 2)
            self._cdna_kernel_layer = snt.Linear(kernel_size)

        # image from scratch layer
        with tf.variable_scope("conv_image_scratch"):
            self._image_scratch_layer_hidden = _build_conv_t_layer(_get_default_conv_spec(self.num_final_feats, self.stride),
                                                                 self._data_format)
            self._image_scratch_layer = _build_conv_layer(_get_default_conv_spec(self.output_channels),  # RGB image
                                                          self._data_format)

        # mask generation layer
        with tf.variable_scope("conv_mask_gen"):
            # num_masks = (scratch_img) + (prev_img) + (first_img) + (cdna_warp_imgs)
            self._mask_gen_layer_hidden = _build_conv_t_layer(_get_default_conv_spec(self.num_final_feats, self.stride),
                                                            self._data_format)
            self._mask_gen_layer = _build_conv_layer(_get_default_conv_spec(self._num_masks),
                                                     self._data_format)
        
        self.CDNA = CDNA(self._data_format, self._num_masks)
      
    def _build(self, dec_output, embedding, input_img, first_input_img, is_training, goal_img=None):
        # stack transformed images
        stack_axis = -1 if self._data_format == "NHWC" else 1
    
        # scratch image
        with tf.variable_scope("image_from_scratch"):
            scratch_img = self._apply_conv_module(self._image_scratch_layer_hidden, dec_output, is_training)
            scratch_img = self._apply_conv_module(self._image_scratch_layer, scratch_img, is_training,
                                                  activation=self.image_activation, batchnorm=False)
    
        # CDNA kernels
        with tf.variable_scope("cdna_image_transform"):
            with tf.variable_scope("cdna_kernels"):
                cdna_activation = embedding
                cdna_kernels = self._compute_cdna_kernels(cdna_activation)
    
        transformed_images = self.CDNA.get_transformed_images(input_img, first_input_img, scratch_img, cdna_kernels)
        transformed_images_stack = tf.concat(transformed_images, axis=stack_axis)
        
        # mask prediction
        with tf.variable_scope("mask_prediction"):
            with tf.variable_scope("mask_gen_hidden"):
                masks = self._apply_conv_module(self._mask_gen_layer_hidden, dec_output, is_training)
            with tf.variable_scope("mask_gen"):
                masks = tf.concat((masks, transformed_images_stack), axis=stack_axis)  # conditional mask generation
                masks = self._apply_conv_module(self._mask_gen_layer, masks, is_training,
                                                activation=lambda x: tf.nn.softmax(x, axis=stack_axis), batchnorm=False)
    
        gen_image = self.CDNA.compose_images(masks, transformed_images)

        if goal_img is not None:
            goal_img = self.CDNA.apply(goal_img, tf.zeros_like(first_input_img), tf.zeros_like(scratch_img),
                                       cdna_kernels, masks)
        
        return gen_image, goal_img
    
    def _apply_conv_module(self, module, input, is_training, activation=tf.nn.leaky_relu, batchnorm=True):
        output = module(input)
        if batchnorm:
            output = snt.BatchNorm(axis=self._batchnorm_axis)(output, is_training)
        output = activation(output)
        return output

    def _compute_cdna_kernels(self, input):
        batch_size = shape(input)[0]
        cdna_activation = tf.reshape(input, [batch_size, -1])  # flatten input activations
        cdna_kernels = self._cdna_kernel_layer(cdna_activation)
        cdna_kernels = tf.reshape(cdna_kernels, (batch_size,
                                                 self.cdna_kernel_size,
                                                 self.cdna_kernel_size,
                                                 self.num_cdna_kernels))
        # do not append identity kernel, as we only use last input image and it is skipped to the end anyways
        cdna_kernels = tf.nn.relu(cdna_kernels - RELU_SHIFT) + RELU_SHIFT   # make strictly positive
        cdna_kernels /= tf.reduce_sum(cdna_kernels, axis=[1, 2], keep_dims=True)  # normalize spatially
        return cdna_kernels


class Finn2016Model(rnn_core.RNNCore):
    """
    Implements the CDNA model of Finn&Levine (2016).
    https://arxiv.org/pdf/1804.01523.pdf
    """

    def __init__(self,
                 network_spec,
                 data_format,
                 input_shape,
                 name="finn_model"):
        super(Finn2016Model, self).__init__(name=name,)
        self._spec = network_spec
        self._input_shape = input_shape
        self._input_resolution = input_shape[1]  # this dim is independent of data_format
        self._is_recurrent_list, self._module_list, self._init_dict, self._reg_dict = [], [], {}, {}

        if self._data_format == "NCHW":
            raise NotImplementedError("CDNA model does currently not support NCHW data format "
                                      "because conv_LSTM does not support it!")

        self._setup_modules(self._spec)
      
        self.cdna_module = CDNADecoder(data_format=data_format,
                                       num_cdna_kernels=network_spec.num_cdna_kernels,
                                       cdna_kernel_size=network_spec.cdna_kernel_size,
                                       num_final_feats=network_spec.num_final_feats)

    def _setup_modules(self, spec):
        """Builds all required modules."""
        # latent LSTM
        with tf.variable_scope("latent_lstm"):
            self._latent_lstm, state_init, state_reg = build_lstm_module(spec.latent_lstm_spec)
            self._register_recurrent_module(self._latent_lstm, state_init, state_reg, "latent_lstm")

        # encoder-decoder network
        self._encoder_modules = self._build_conv_lstm_stack(spec.encoding_layers, "encoder")
        self._decoder_modules = self._build_conv_lstm_stack(spec.encoding_layers[::-1], "decoder")
        with tf.variable_scope("conv_feat_final"):
            self._final_feature_layer = _build_conv_layer(_get_default_conv_spec(spec.num_final_feats),
                                                          self._data_format)
            self._register_nonrecurrent_module(self._final_feature_layer, "conv_feat_final")

    def _apply_conv_module(self, module, input, is_training, activation=tf.nn.leaky_relu, batchnorm=True):
        output = module(input)
        if batchnorm:
            output = snt.BatchNorm(axis=self._batchnorm_axis)(output, is_training)
        output = activation(output)
        return output
    
    def _apply_lstm_module(self, module, input, state, activation=tf.nn.leaky_relu):
        output, state = module(input, state)
        output = activation(output)
        return output, state
    
    def _build_conv_lstm_stack(self, specs, name_str):
        modules = []
        for i, (conv_spec, lstm_spec) in enumerate(specs):  # layer pairwise: conv, lstm
            with tf.variable_scope("conv_%s_%d" % (name_str, i)):
                conv_module = _build_conv_layer(conv_spec, self._data_format)
                self._register_nonrecurrent_module(conv_module, "conv_%s_%d" % (name_str, i))
            with tf.variable_scope("lstm_%s_%d" % (name_str, i)):
                lstm_module, init, reg = build_conv_lstm_module(lstm_spec,
                                                      self._data_format)
                self._register_recurrent_module(lstm_module, init, reg, "lstm_%s_%d" % (name_str, i))
            modules.append([conv_module, lstm_module])
        return modules

    def _register_nonrecurrent_module(self, module, name):
        self._module_list.append([module, name])
        self._is_recurrent_list.append(False)

    def _register_recurrent_module(self, module, initializer, regularizer, name):
        self._module_list.append([module, name])
        self._init_dict[name], self._reg_dict[name] = initializer, regularizer
        self._is_recurrent_list.append(True)

    def _tile_concat_latent(self, input, latent):
        # latent has dimension (batch_size x N)
        latent = tf.reshape(latent, [latent.shape[0], 1, 1, latent.shape[1]])   # give spatial dims
        spatial_dim = input.shape[2]    # this dim is def not channels, independent of data_format
        latent = tf.tile(latent, [1, spatial_dim, spatial_dim, 1])
        if self._data_format == "NCHW":
            # transpose, tile, re-transpose
            input = tf.transpose(input, (0, 2, 3, 1))
        output = tf.concat((input, latent), axis=-1)
        if self._data_format == "NCHW":
            output = tf.transpose(output, (0, 3, 1, 2))
        return output

    def _concat_skip(self, input, skip_activation):
        if self._data_format == "NCHW":
            return tf.concat((input, skip_activation), axis=1)
        elif self._data_format == "NHWC":
            return tf.concat((input, skip_activation), axis=-1)
        else:
            raise ValueError("Data format %s currently not supported in CDNA model!" % self._data_format)

    def _downsample(self, input, kernel_size):
        return tf.nn.pool(input,
                          window_shape=[kernel_size, kernel_size],
                          pooling_type="AVG",
                          padding="SAME",
                          strides=[2, 2],
                          data_format=self._data_format)

    def _upsample(self, input, i):
        # compute required latent size, every encoding layer reduces resolution by factor 2
        upsample_size = int(self._input_resolution / np.power(2, len(self._encoder_modules) - i - 1))
        return tf.image.resize_bilinear(input, [upsample_size, upsample_size])

    def _build(self, inputs, states):
        input_img, first_input_img, latent, is_training = inputs

        # latent LSTM
        with tf.variable_scope("latent_lstm"):
            latent, states["latent_lstm"] = self._latent_lstm(latent, states["latent_lstm"])

        # encoder-decoder network
        output_i = input_img
        with tf.variable_scope("encoder_decoder_network"):
            skip_activations = []
            for i, (conv_module, lstm_module) in enumerate(self._encoder_modules):
                with tf.variable_scope("encoder_block_%d" % i):
                    output_i = self._tile_concat_latent(output_i, latent)
                    kernel_size = 5 if i == 0 else 3
                    output_i = self._downsample(output_i, kernel_size)
                    output_i = self._apply_conv_module(conv_module, output_i, is_training)
                    output_i = self._tile_concat_latent(output_i, latent)
                    output_i, states["lstm_encoder_%d" % i] = self._apply_lstm_module(lstm_module,
                                                                                      output_i,
                                                                                      states["lstm_encoder_%d" % i])
                    skip_activations.append(output_i)

            for i, (conv_module, lstm_module) in enumerate(self._decoder_modules):
                with tf.variable_scope("decoder_block_%d" % i):
                    if i != 0:
                        output_i = self._concat_skip(output_i, skip_activations[-(i+1)])
                    output_i = self._tile_concat_latent(output_i, latent)
                    output_i = self._apply_conv_module(conv_module, output_i, is_training)
                    output_i = self._tile_concat_latent(output_i, latent)
                    output_i, states["lstm_decoder_%d" % i] = self._apply_lstm_module(lstm_module,
                                                                                      output_i,
                                                                                      states["lstm_decoder_%d" % i])
                    output_i = self._upsample(output_i, i)
            with tf.variable_scope("final_conv_feats"):
                output_i = self._tile_concat_latent(output_i, latent)
                enc_dec_output = self._apply_conv_module(self._final_feature_layer, output_i, is_training)

        gen_image, _ = self.cdna_module(
            enc_dec_output, skip_activations[-1], input_img, first_input_img, is_training)
        return gen_image, states

    def initial_state(self, batch_size, dtype=tf.float32, trainable=False, name=None):
        num_recurrent = sum(self._is_recurrent_list)

        num_initializers = len(self._init_dict)
        if num_initializers != num_recurrent:
            raise ValueError("The number of initializers and recurrent cores should "
                             "be the same. Received %d initializers for %d specified "
                             "recurrent cores."
                             % (num_initializers, num_recurrent))

        with tf.name_scope(self._initial_state_scope(name)):
            initial_state = {}
            for is_recurrent, module_data in zip(self._is_recurrent_list, self._module_list):
                if is_recurrent:
                    module, module_name = module_data
                    module_initial_state = module.initial_state(
                        batch_size, dtype=dtype, trainable=trainable,
                        trainable_initializers=self._init_dict[module_name],
                        trainable_regularizers=self._reg_dict[module_name])
                    initial_state[module_name] = module_initial_state
        return initial_state

    @property
    def state_size(self):
        sizes = []
        for is_recurrent, module_data in zip(self._is_recurrent_list, self._module_list):
            if is_recurrent:
                module, _ = module_data
                sizes.append(module.state_size)
        return tuple(sizes)

    @property
    def output_size(self):
        return tf.TensorShape(self._input_shape)