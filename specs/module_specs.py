"""Module specifications."""
import collections
import tensorflow as tf
from recordclass import recordclass
from recordclass import recordclass

EncodingLayerSpec = recordclass(
    "EncodingLayerSpec",
    "output_channels kernel_shape "
    "stride rate "
    "use_nonlinearity use_batchnorm use_pool")
DecodingLayerSpec = collections.namedtuple(
    "DecodingLayerSpec",
    "output_channels kernel_shape "
    "stride rate "
    "use_nonlinearity use_batchnorm use_upsample")

# For DCGAN: add use_interface_batchnorm (input for encoder, output for decoder)
ConvNetSpec = collections.namedtuple(
  "ConvNetSpec",
  "spec_type " # dcgan or simple_conv_net
  "layers " # Tuple of layers if needed
  "dcgan_latent_size " # Encoder output size, for DCGAN
  "dcgan_output_channels " # Decoder output size, for DCGAN
  "dcgan_use_image_bn "  # Use batchnorm on input (encoder) or output (decoder) image?
  "dcgan_dropout_rate "  # Dropout rate in DCGAN layers (0.0 = no dropout)
  "skip_type"  # skip, res, or None: how skip connection inputs are handled.
)
LSTMSpec = collections.namedtuple(
    "LSTMSpec",
    "num_hidden "  # Number of hidden units
    "smooth_projection "  # If True, use SmoothProjectionLSTM. Otherwise LSTM
    "output_size "  # Output size (# of units)
    "sp_output_nonlinearity"  # SmoothProjectionLSTM nonlinearity after summation with input
)
ConvLSTMSpec = collections.namedtuple(
    "ConvLSTMSpec",
    "input_shape output_channels kernel_shape "
    "stride rate smooth_projection sp_output_nonlinearity")

LinearSpec = recordclass(
    "LinearSpec",
    "output_size non_linearity"
)
StatePredictorSpec = collections.namedtuple(
  "StatePredictorSpec",
  "network_type x_size xt_size"
)
PartitionedRNNSpec = collections.namedtuple(
    "PartitionedRNNSpec",
    "input_shape x_size xt_size "
    "x_predictor_spec xt_predictor_spec x_output_nonlinearity xxt_configuration"
)
MLPSpec = collections.namedtuple(
    "MLPSpec",
    "layers"        # last layer will be added dynamically based on actions in dataset
)

CDNASpec = collections.namedtuple(
    "CDNASpec",
    "num_cdna_kernels cdna_kernel_size num_final_feats encoding_layers latent_lstm_spec"
)


FC_TYPES = [LSTMSpec, LinearSpec, MLPSpec]
CONV_TYPES = [ConvLSTMSpec, ConvNetSpec, EncodingLayerSpec, DecodingLayerSpec, CDNASpec]

_dcgan_use_inout_bn_spec = ConvNetSpec(
    spec_type="dcgan",
    layers=None,
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=True,
    dcgan_dropout_rate=0.0,
    skip_type=None)
_dcgan_large_use_inout_bn_spec = ConvNetSpec(
    spec_type="dcgan",
    layers=None,
    dcgan_latent_size=256,
    dcgan_output_channels=1,
    dcgan_use_image_bn=True,
    dcgan_dropout_rate=0.0,
    skip_type=None)
_dcgan_spec = ConvNetSpec(
    spec_type="dcgan",
    layers=None,
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type=None)
_dcgan_spec_skips = ConvNetSpec(
    spec_type="dcgan",
    layers=None,
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type="skip")
_dcgan_spec_res = ConvNetSpec(
    spec_type="dcgan",
    layers=None,
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type="res")
_dcgan_rgb_spec = ConvNetSpec(
    spec_type="dcgan",
    layers=None,
    dcgan_latent_size=128,
    dcgan_output_channels=3,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type=None)
_dcgan_large_spec = ConvNetSpec(
    spec_type="dcgan",
    layers=None,
    dcgan_latent_size=256,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type=None)
_dcgan_spec_med = ConvNetSpec(
    spec_type="dcgan",
    layers="med",
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type=None)
_dcgan_spec_small = ConvNetSpec(
    spec_type="dcgan",
    layers="small",
    dcgan_latent_size=64,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type=None)
_dcgan_spec_small_res = ConvNetSpec(
    spec_type="dcgan",
    layers="small",
    dcgan_latent_size=64,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type="res")
_dcgan_spec_small_skips = ConvNetSpec(
    spec_type="dcgan",
    layers="small",
    dcgan_latent_size=64,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type="skip")
_dcgan_small_128_spec = ConvNetSpec(
    spec_type="dcgan",
    layers="small",
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0,
    skip_type=None)
_dcgan_small_128_spec_res = ConvNetSpec(
    spec_type="dcgan",
    layers="small",
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0,
    skip_type="res")

# Specs w/ dropout
_dcgan_small_128_spec_drop = ConvNetSpec(
    spec_type="dcgan",
    layers="small",
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.5,
    skip_type=None)
_dcgan_small_128_spec_drop_skips = ConvNetSpec(
    spec_type="dcgan",
    layers="small",
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.5,
    skip_type="skip")
_dcgan_large_spec_drop_res = ConvNetSpec(
    spec_type="dcgan",
    layers=None,
    dcgan_latent_size=256,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.5,
    skip_type="res")
_dcgan_spec_drop = ConvNetSpec(
    spec_type="dcgan",
    layers=None,
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.5,
    skip_type=None)
_dcgan_spec_drop_res = ConvNetSpec(
    spec_type="dcgan",
    layers=None,
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.5,
    skip_type="res")
_dcgan_large_spec_drop = ConvNetSpec(
    spec_type="dcgan",
    layers=None,
    dcgan_latent_size=256,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.5,
    skip_type=None)
_dcgan_large_spec_drop_skips = ConvNetSpec(
    spec_type="dcgan",
    layers=None,
    dcgan_latent_size=256,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.5,
    skip_type="skip")
_dcgan_large_spec_drop_res = ConvNetSpec(
    spec_type="dcgan",
    layers=None,
    dcgan_latent_size=256,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.5,
    skip_type="res")

_dcgan_ucf_spec_drop_skips = ConvNetSpec(
    spec_type="dcgan",
    layers="ucf_gray_large",
    dcgan_latent_size=512,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.5,
    skip_type="skip")
_dcgan_ucf_spec_skips = ConvNetSpec(
    spec_type="dcgan",
    layers="ucf_gray_large",
    dcgan_latent_size=256,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type="skip")

_dcgan_ucf_spec_drop_res = ConvNetSpec(
    spec_type="dcgan",
    layers="ucf_gray_large",
    dcgan_latent_size=512,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.5,
    skip_type="res")
_dcgan_ucf_spec_res = ConvNetSpec(
    spec_type="dcgan",
    layers="ucf_gray_large",
    dcgan_latent_size=256,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type="res")

# Variational - spatial compression in the latent
_dcgan_spec_denton_mnist = ConvNetSpec(
    spec_type="dcgan",
    layers="denton_mnist",
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type=None)
_dcgan_spec_denton_mnist_compressed = ConvNetSpec(
    spec_type="dcgan",
    layers="denton_mnist",
    dcgan_latent_size=20,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type=None)
_dcgan_spec_denton_mnist_skip = ConvNetSpec(
    spec_type="dcgan",
    layers="denton_mnist",
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type="skip")
_dcgan_spec_denton_mnist_res = ConvNetSpec(
    spec_type="dcgan",
    layers="denton_mnist",
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type="res")
_dcgan_spec_denton_mnist_small = ConvNetSpec(
    spec_type="dcgan",
    layers="denton_mnist_small",
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type=None)
_dcgan_spec_denton_mnist_small_skip = ConvNetSpec(
    spec_type="dcgan",
    layers="denton_mnist_small",
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type='skip')
_dcgan_spec_denton_mnist_small_res = ConvNetSpec(
    spec_type="dcgan",
    layers="denton_mnist_small",
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type='res')
_dcgan_spec_denton_mnist_very_small_res = ConvNetSpec(
    spec_type="dcgan",
    layers="denton_mnist_very_small_res",
    dcgan_latent_size=128,
    dcgan_output_channels=1,
    dcgan_use_image_bn=False,
    dcgan_dropout_rate=0.0,
    skip_type='res')

# Convolve + max pool 64x64->4x4, simplest architecture.
_default_conv_enc_spec = ConvNetSpec(
    spec_type="simple_conv_net",
    dcgan_latent_size=None, # Ignored
    dcgan_output_channels=None,  # Ignored
    dcgan_use_image_bn=False, # Ignored
    dcgan_dropout_rate=0.0,
    skip_type=None,
    layers=(
      EncodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          rate=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=False,
          use_batchnorm=True,
          rate=1,
          use_pool=True)))

# 5 layer: 128 -> 4
_kth_conv_enc_spec = ConvNetSpec(
    spec_type="simple_conv_net",
    dcgan_latent_size=None, # Ignored
    dcgan_output_channels=None,  # Ignored
    dcgan_use_image_bn=False, # Ignored
    dcgan_dropout_rate=0.0,
    skip_type=None,
    layers=(
      EncodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          rate=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=False,
          use_batchnorm=True,
          rate=1,
          use_pool=True)))

# Modified version with 2x the number of output channels for 2 frame input
_two_frame_conv_enc_spec = ConvNetSpec(
    spec_type="simple_conv_net",
    dcgan_latent_size=None, # Ignored
    dcgan_output_channels=None,  # Ignored
    dcgan_use_image_bn=False, # Ignored
    dcgan_dropout_rate=0.0,
    skip_type=None,
    layers=(
      EncodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          rate=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=128,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=False,
          use_batchnorm=True,
          rate=1,
          use_pool=True)))

_two_frame_conv_enc_large_spec = ConvNetSpec(
    spec_type="simple_conv_net",
    dcgan_latent_size=None, # Ignored
    dcgan_output_channels=None,  # Ignored
    dcgan_use_image_bn=False, # Ignored
    dcgan_dropout_rate=0.0,
    skip_type=None,
    layers=(
      EncodingLayerSpec(
          output_channels=128,
          kernel_shape=3,
          stride=1,
          rate=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=128,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=128,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=128,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=False,
          use_batchnorm=True,
          rate=1,
          use_pool=True)))

# Larger default
_large_conv_enc_spec = ConvNetSpec(
    spec_type="simple_conv_net",
    dcgan_latent_size=None, # Ignored
    dcgan_output_channels=None,  # Ignored
    dcgan_use_image_bn=False, # Ignored
    dcgan_dropout_rate=0.0,
    skip_type=None,
    layers=(
      EncodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          rate=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=128,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=128,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_pool=True),
      EncodingLayerSpec(
          output_channels=128,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=False,
          use_batchnorm=True,
          rate=1,
          use_pool=True)))

# Convolve + NN upsample 4x4->64x64, simplest architecture.
_default_conv_dec_spec = ConvNetSpec(
    spec_type="simple_conv_net",
    dcgan_latent_size=None, # Ignored
    dcgan_output_channels=None,  # Ignored
    dcgan_use_image_bn=False, # Ignored
    dcgan_dropout_rate=0.0,
    skip_type=None,
    layers=(DecodingLayerSpec(
        output_channels=64,
        kernel_shape=3,
        stride=1,
        rate=1,
        use_nonlinearity=True,
        use_batchnorm=True,
        use_upsample=True),
    DecodingLayerSpec(
        output_channels=64,
        kernel_shape=3,
        stride=1,
        use_nonlinearity=True,
        use_batchnorm=True,
        rate=1,
        use_upsample=True),
    DecodingLayerSpec(
        output_channels=64,
        kernel_shape=3,
        stride=1,
        use_nonlinearity=True,
        use_batchnorm=True,
        rate=1,
        use_upsample=True),
    DecodingLayerSpec(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        use_nonlinearity=False,
        use_batchnorm=False,
        rate=1,
        use_upsample=True)))

# 4x4 -> 128x128
_kth_conv_dec_spec = ConvNetSpec(
    spec_type="simple_conv_net",
    dcgan_latent_size=None, # Ignored
    dcgan_output_channels=None,  # Ignored
    dcgan_use_image_bn=False, # Ignored
    dcgan_dropout_rate=0.0,
    skip_type=None,
    layers=(
      DecodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          rate=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          use_upsample=True),
      DecodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_upsample=True),
      DecodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_upsample=True),
      DecodingLayerSpec(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=True,
          use_batchnorm=True,
          rate=1,
          use_upsample=True),
      DecodingLayerSpec(
          output_channels=1,
          kernel_shape=3,
          stride=1,
          use_nonlinearity=False,
          use_batchnorm=False,
          rate=1,
          use_upsample=True)))

# Convolve + NN upsample 4x4->64x64, simplest architecture.
_large_conv_dec_spec = ConvNetSpec(
    spec_type="simple_conv_net",
    dcgan_latent_size=None, # Ignored
    dcgan_output_channels=None,  # Ignored
    dcgan_use_image_bn=False, # Ignored
    dcgan_dropout_rate=0.0,
    skip_type=None,
    layers=(DecodingLayerSpec(
        output_channels=128,
        kernel_shape=3,
        stride=1,
        rate=1,
        use_nonlinearity=True,
        use_batchnorm=True,
        use_upsample=True),
    DecodingLayerSpec(
        output_channels=128,
        kernel_shape=3,
        stride=1,
        use_nonlinearity=True,
        use_batchnorm=True,
        rate=1,
        use_upsample=True),
    DecodingLayerSpec(
        output_channels=64,
        kernel_shape=3,
        stride=1,
        use_nonlinearity=True,
        use_batchnorm=True,
        rate=1,
        use_upsample=True),
    DecodingLayerSpec(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        use_nonlinearity=False,
        use_batchnorm=False,
        rate=1,
        use_upsample=True)))

_bb_coord_encoder_mlp_spec = MLPSpec(layers=[32, 64, 128])

_bb_coord_decoder_mlp_spec = MLPSpec(layers=[64, 32, 16])  # correct output dim is added in code -> nothing to worry here

# Single LSTM
# Encoder is only used for state (not output),
# so it doesn't need smooth projection
_default_lstm_encoder_spec = (
    LSTMSpec(num_hidden=1024, output_size=1024,
             smooth_projection=False, sp_output_nonlinearity=None),)

_default_lstm_decoder_spec = (
    LSTMSpec(num_hidden=1024, output_size=1024,
             smooth_projection=True, sp_output_nonlinearity=None),)

_lstm_encoder_dcgan_spec = (
    LSTMSpec(num_hidden=2048, output_size=2048,
             smooth_projection=False, sp_output_nonlinearity=None),)

_lstm_decoder_dcgan_spec = (
    LSTMSpec(num_hidden=2048, output_size=2048,
             smooth_projection=False, sp_output_nonlinearity=None),)

_lstm_encoder_dcgan_tanh_spec = (
    LSTMSpec(num_hidden=2048, output_size=2048,
             smooth_projection=True, sp_output_nonlinearity="tanh"),)

_lstm_decoder_dcgan_tanh_spec = (
    LSTMSpec(num_hidden=2048, output_size=2048,
             smooth_projection=True, sp_output_nonlinearity="tanh"),)

# Single LSTM
_large_lstm_encoder_spec = (
    LSTMSpec(num_hidden=4096, output_size=4096,
             smooth_projection=False, sp_output_nonlinearity=None),)

_large_lstm_decoder_spec = (
    LSTMSpec(num_hidden=4096, output_size=4096,
             smooth_projection=True, sp_output_nonlinearity=None),)

_kth_convlstm_dcgan_tanh = (
    ConvLSTMSpec(
        input_shape=(4, 4, 128),
        output_channels=128,
        kernel_shape=3,
        stride=1,
        rate=1,
        smooth_projection=True,
        sp_output_nonlinearity="tanh"),)

# Single ConvLSTM
_default_convlstm_encoder_spec = (
    ConvLSTMSpec(
        input_shape=(4, 4, 64),
        output_channels=64,
        kernel_shape=3,
        stride=1,
        rate=1,
        smooth_projection=False,
        sp_output_nonlinearity=None),)

_default_convlstm_decoder_spec = (
    ConvLSTMSpec(
        input_shape=(4, 4, 64),
        output_channels=64,
        kernel_shape=3,
        stride=1,
        rate=1,
        smooth_projection=True,
        sp_output_nonlinearity=None),)


_kth_lstm_projector_spec = (
    LSTMSpec(num_hidden=1024, output_size=1024,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=1024, non_linearity=None),
)

_mnist_lstm_projector_spec = (
    LSTMSpec(num_hidden=512, output_size=512,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=512, non_linearity=None),
)

_mnist_lstm_projector_large_spec = (
    LSTMSpec(num_hidden=1024,
             output_size=None,  # Unused for snt.LSTM
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=1024, non_linearity=None),  # 4 x 4 x 64 needed for output
)

_kth_lstm_projector_large_spec = (
    LSTMSpec(num_hidden=2048,
             output_size=None,  # Ignored for plain LSTM
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=2048, non_linearity=None),  # 4 x 4 x 256 needed for output
)

_kth_convlstm_128 = (
    ConvLSTMSpec(
        input_shape=(4, 4, 128),
        output_channels=128,
        kernel_shape=3,
        stride=1,
        rate=1,
        smooth_projection=False,
        sp_output_nonlinearity=None),
    ConvLSTMSpec(
        input_shape=(4, 4, 128),
        output_channels=128,
        kernel_shape=3,
        stride=1,
        rate=1,
        smooth_projection=False,
        sp_output_nonlinearity=None),
    ConvLSTMSpec(
        input_shape=(4, 4, 128),
        output_channels=128,
        kernel_shape=3,
        stride=1,
        rate=1,
        smooth_projection=False,
        sp_output_nonlinearity=None),
    ConvLSTMSpec(
        input_shape=(4, 4, 128),
        output_channels=128,
        kernel_shape=3,
        stride=1,
        rate=1,
        smooth_projection=False,
        sp_output_nonlinearity=None),)

_variational_projected_lstm_spec = (
    LSTMSpec(num_hidden=1024, output_size=1024,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=1024, non_linearity=None),
)

_variational_output_convlstm_spec = (
    ConvLSTMSpec(
        input_shape=(4, 4, 96),
        output_channels=64,
        kernel_shape=3,
        stride=1,
        rate=1,
        smooth_projection=False,
        sp_output_nonlinearity=None),)


_variational_output_convlstm_spec_denton = (
    ConvLSTMSpec(
        input_shape=(4, 4, 74),
        output_channels=64,
        kernel_shape=3,
        stride=1,
        rate=1,
        smooth_projection=False,
        sp_output_nonlinearity=None),)


_distribution_convlstm_spec_denton = (
    ConvLSTMSpec(
        input_shape=(4, 4, 64),
        output_channels=20,
        kernel_shape=3,
        stride=1,
        rate=1,
        smooth_projection=False,
        sp_output_nonlinearity=None),)

_variational_output_lstm_spec_denton = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=1024, non_linearity='tanh'),
)

_variational_inference_lstm_spec_denton_compressed = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=128+20, non_linearity='tanh'),      # img_enc_size + variational_latent_size
)

_inf_lstm_spec_20 = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=128+20, non_linearity=None),        # img_enc_size (0 for gtAttentionKeys) + variational_latent_size
)

_ll_inf_lstm_spec_20 = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=0+20, non_linearity=None),
)

_inf_lstm_spec_20_gtKeys = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=0+20, non_linearity=None),        # img_enc_size (0 for gtAttentionKeys) + variational_latent_size
)

_inf_lstm_spec_128_gtKeys = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=0+128, non_linearity=None),        # img_enc_size (0 for gtAttentionKeys) + variational_latent_size
)

_variational_output_lstm_spec_denton_compressed = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=128, non_linearity='tanh'),
)

_lowlevel_lstm_spec = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=128, non_linearity=None),
)

_lowlevel_lstm_spec_dt_ll = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=128+1, non_linearity=None),    # +1 is the prob for that frame (dt), apply tanh on img later
)

_variational_high_level_lstm_spec_denton_compressed = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=128+20, non_linearity='tanh'),
)

_highlevel_lstm_spec_inf20 = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=128+20, non_linearity=None),
)

_highlevel_lstm_spec_inf20_dt_ll = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=128+20, non_linearity=None),
)

_highlevel_lstm_spec_inf128 = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=128+128, non_linearity=None),
)

_variational_output_lstm_spec_denton_compressed_latent = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=20, non_linearity='tanh'),
)

_encoder_lstm_spec_20 = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=20, non_linearity=None),
)

_encoder_lstm_spec_128 = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=128, non_linearity=None),
)

_variational_output_lstm_spec_denton_compressed_md = (
    LinearSpec(output_size=138, non_linearity=tf.nn.leaky_relu),
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=128, non_linearity='tanh'),
)

_variational_output_lstm_spec_denton_compressed_dp = (
    LinearSpec(output_size=138, non_linearity=tf.nn.leaky_relu),
    LinearSpec(output_size=138, non_linearity=tf.nn.leaky_relu),
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=128, non_linearity='tanh'),
)

_distribution_lstm_spec_denton = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=20, non_linearity=None),
)

_distribution_lstm_spec_denton_dp = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=128, non_linearity=tf.nn.leaky_relu),
    LinearSpec(output_size=128, non_linearity=tf.nn.leaky_relu),
    LinearSpec(output_size=20, non_linearity=None),
)

_distribution_mlp_spec = (
    MLPSpec(layers=[256, 128]),
    LinearSpec(output_size=20, non_linearity=None),
)

_initialisation_mlp_spec = MLPSpec(layers=[256, 1024])

_distribution_lstm_spec_denton_small = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=10, non_linearity=None),
)

_distribution_lstm_spec_denton_very_small = (
    LinearSpec(output_size=256, non_linearity=None),
    LSTMSpec(num_hidden=256, output_size=None,
             smooth_projection=False, sp_output_nonlinearity=None),
    LinearSpec(output_size=4, non_linearity=None),
)

# last layer will be added dynamically based on actions in dataset
_action_discriminator_small = MLPSpec(layers=[32, 32])

_INPUT_RES = 64
_FEAT_DEPTH = 32
_LATENT_DIM = 8
_variational_output_cdna_spec_lee_finn = CDNASpec(
    num_cdna_kernels=4,
    cdna_kernel_size=5,
    num_final_feats=_FEAT_DEPTH,
    latent_lstm_spec=LSTMSpec(num_hidden=_LATENT_DIM,
                              output_size=_LATENT_DIM,
                              smooth_projection=False,
                              sp_output_nonlinearity=None),
    encoding_layers=(
        [EncodingLayerSpec(
                output_channels=_FEAT_DEPTH,
                kernel_shape=3,
                stride=1,
                rate=1,
                use_nonlinearity=True,      # in current implementation BN and ReLU are always applied
                use_batchnorm=True,
                use_pool=False),
         ConvLSTMSpec(
                input_shape=(_INPUT_RES/2, _INPUT_RES/2, _FEAT_DEPTH + _LATENT_DIM),
                output_channels=_FEAT_DEPTH,
                kernel_shape=3,
                stride=1,
                rate=1,
                smooth_projection=False,
                sp_output_nonlinearity=None)],
        [EncodingLayerSpec(
                output_channels=_FEAT_DEPTH * 2,
                kernel_shape=3,
                stride=1,
                rate=1,
                use_nonlinearity=True,
                use_batchnorm=True,
                use_pool=False),
         ConvLSTMSpec(
                input_shape=(_INPUT_RES/4, _INPUT_RES/4, _FEAT_DEPTH * 2 + _LATENT_DIM),
                output_channels=_FEAT_DEPTH * 2,
                kernel_shape=3,
                stride=1,
                rate=1,
                smooth_projection=False,
                sp_output_nonlinearity=None)],
        [EncodingLayerSpec(
                output_channels=_FEAT_DEPTH * 4,
                kernel_shape=3,
                stride=1,
                rate=1,
                use_nonlinearity=True,
                use_batchnorm=True,
                use_pool=False),
            ConvLSTMSpec(
                input_shape=(_INPUT_RES / 8, _INPUT_RES / 8, _FEAT_DEPTH * 4 + _LATENT_DIM),
                output_channels=_FEAT_DEPTH * 4,
                kernel_shape=3,
                stride=1,
                rate=1,
                smooth_projection=False,
                sp_output_nonlinearity=None)]
    )
)
