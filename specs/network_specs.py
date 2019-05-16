"""Network specifications."""
import collections
from . import module_specs
import tensorflow as tf

# Architecture specs
ActionConditionedSpec = collections.namedtuple(
    "ActionConditionedSpec",
    "conv_encoder_spec "
    "conv_decoder_spec "
    "encoder_rnn_spec "
    "decoder_rnn_spec "
    "share_past_future_decoder "
    "share_past_future_rnn "
    "use_recursive_skips"
)

StochasticSingleStepSpec = collections.namedtuple(
    "StochasticSingleStepSpec",
    "conv_encoder_spec "
    "conv_decoder_spec "
    "encoder_rnn_spec "
    "inference_rnn_spec "
    "decoder_rnn_spec "
    "action_discriminator_spec "
    "share_past_future_decoder "
    "share_past_future_rnn "
    "use_recursive_skips"
)

HierarchicalNetworkSpec = collections.namedtuple(
    "HierarchicalNetworkSpec",
    "conv_encoder_spec "
    "conv_decoder_spec "
    "encoder_rnn_spec "
    "inference_rnn_spec "
    "high_level_rnn_spec "
    "low_level_rnn_spec "
    "low_level_inf_rnn_spec "
    "low_level_initialisor_spec "
    "action_discriminator_spec "
    "share_past_future_decoder "
    "share_past_future_rnn "
    "use_recursive_skips"
)

actCond_lstm_bb = ActionConditionedSpec(
  conv_encoder_spec=module_specs._dcgan_spec_denton_mnist_small_res,
  conv_decoder_spec=module_specs._dcgan_spec_denton_mnist_small_res,
  encoder_rnn_spec=module_specs._lowlevel_lstm_spec,
  decoder_rnn_spec=module_specs._lowlevel_lstm_spec,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=True)

actCond_lstm_bb_small_res = ActionConditionedSpec(
  conv_encoder_spec=module_specs._dcgan_spec_denton_mnist_very_small_res,
  conv_decoder_spec=module_specs._dcgan_spec_denton_mnist_very_small_res,
  encoder_rnn_spec=module_specs._lowlevel_lstm_spec,
  decoder_rnn_spec=module_specs._lowlevel_lstm_spec,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=True)

actCond_lstm_bb_coord_small_res = ActionConditionedSpec(
  conv_encoder_spec=module_specs._bb_coord_encoder_mlp_spec,
  conv_decoder_spec=module_specs._bb_coord_encoder_mlp_spec,
  encoder_rnn_spec=module_specs._lowlevel_lstm_spec,
  decoder_rnn_spec=module_specs._lowlevel_lstm_spec,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=True)

stochSS_lstm_bb = StochasticSingleStepSpec(
  conv_encoder_spec=module_specs._dcgan_spec_denton_mnist_small_res,
  conv_decoder_spec=module_specs._dcgan_spec_denton_mnist_small_res,
  encoder_rnn_spec=module_specs._lowlevel_lstm_spec,
  inference_rnn_spec=module_specs._ll_inf_lstm_spec_20,
  decoder_rnn_spec=module_specs._lowlevel_lstm_spec,
  action_discriminator_spec=module_specs._action_discriminator_small,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=True)

stochSS_lstm_bb_small_res = StochasticSingleStepSpec(
  conv_encoder_spec=module_specs._dcgan_spec_denton_mnist_very_small_res,
  conv_decoder_spec=module_specs._dcgan_spec_denton_mnist_very_small_res,
  encoder_rnn_spec=module_specs._lowlevel_lstm_spec,
  inference_rnn_spec=module_specs._ll_inf_lstm_spec_20,
  decoder_rnn_spec=module_specs._lowlevel_lstm_spec,
  action_discriminator_spec=module_specs._action_discriminator_small,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=True)

stochSS_lstm_bb_coord_small_res = StochasticSingleStepSpec(
  conv_encoder_spec=module_specs._bb_coord_encoder_mlp_spec,
  conv_decoder_spec=module_specs._bb_coord_encoder_mlp_spec,
  encoder_rnn_spec=module_specs._lowlevel_lstm_spec,
  inference_rnn_spec=module_specs._ll_inf_lstm_spec_20,
  decoder_rnn_spec=module_specs._lowlevel_lstm_spec,
  action_discriminator_spec=module_specs._action_discriminator_small,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=True)

hierarchical_lstm_mnist = HierarchicalNetworkSpec(
  conv_encoder_spec=module_specs._dcgan_spec_denton_mnist_small_res,
  conv_decoder_spec=module_specs._dcgan_spec_denton_mnist_small_res,
  encoder_rnn_spec=module_specs._encoder_lstm_spec_20,
  inference_rnn_spec=module_specs._inf_lstm_spec_20,
  high_level_rnn_spec=module_specs._highlevel_lstm_spec_inf20,
  low_level_rnn_spec=module_specs._lowlevel_lstm_spec,
  low_level_inf_rnn_spec=module_specs._ll_inf_lstm_spec_20,
  low_level_initialisor_spec=module_specs._initialisation_mlp_spec,
  action_discriminator_spec=module_specs._action_discriminator_small,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=True)

hierarchical_lstm_mnist_dt_ll = HierarchicalNetworkSpec(
  conv_encoder_spec=module_specs._dcgan_spec_denton_mnist_small_res,
  conv_decoder_spec=module_specs._dcgan_spec_denton_mnist_small_res,
  encoder_rnn_spec=module_specs._encoder_lstm_spec_20,
  inference_rnn_spec=module_specs._inf_lstm_spec_20,
  high_level_rnn_spec=module_specs._highlevel_lstm_spec_inf20_dt_ll,
  low_level_inf_rnn_spec=module_specs._ll_inf_lstm_spec_20,
  low_level_rnn_spec=module_specs._lowlevel_lstm_spec_dt_ll,
  low_level_initialisor_spec=module_specs._initialisation_mlp_spec,
  action_discriminator_spec=module_specs._action_discriminator_small,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=True)

hierarchical_lstm_kth_small = HierarchicalNetworkSpec(
  conv_encoder_spec=module_specs._dcgan_spec_denton_mnist_res,
  conv_decoder_spec=module_specs._dcgan_spec_denton_mnist_res,
  encoder_rnn_spec=module_specs._encoder_lstm_spec_20,
  inference_rnn_spec=module_specs._inf_lstm_spec_20,
  high_level_rnn_spec=module_specs._highlevel_lstm_spec_inf20,
  low_level_rnn_spec=module_specs._lowlevel_lstm_spec,
  low_level_inf_rnn_spec=module_specs._ll_inf_lstm_spec_20,
  low_level_initialisor_spec=module_specs._initialisation_mlp_spec,
  action_discriminator_spec=module_specs._action_discriminator_small,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=True)

hierarchical_lstm_mnist_small_res = HierarchicalNetworkSpec(
  conv_encoder_spec=module_specs._dcgan_spec_denton_mnist_very_small_res,
  conv_decoder_spec=module_specs._dcgan_spec_denton_mnist_very_small_res,
  encoder_rnn_spec=module_specs._encoder_lstm_spec_20,
  inference_rnn_spec=module_specs._inf_lstm_spec_20,
  high_level_rnn_spec=module_specs._highlevel_lstm_spec_inf20,
  low_level_rnn_spec=module_specs._lowlevel_lstm_spec,
  low_level_inf_rnn_spec=module_specs._ll_inf_lstm_spec_20,
  low_level_initialisor_spec=module_specs._initialisation_mlp_spec,
  action_discriminator_spec=module_specs._action_discriminator_small,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=True)

hierarchical_lstm_mnist_small_res_dt_ll = HierarchicalNetworkSpec(
  conv_encoder_spec=module_specs._dcgan_spec_denton_mnist_very_small_res,
  conv_decoder_spec=module_specs._dcgan_spec_denton_mnist_very_small_res,
  encoder_rnn_spec=module_specs._encoder_lstm_spec_20,
  inference_rnn_spec=module_specs._inf_lstm_spec_20,
  high_level_rnn_spec=module_specs._highlevel_lstm_spec_inf20_dt_ll,
  low_level_rnn_spec=module_specs._lowlevel_lstm_spec_dt_ll,
  low_level_inf_rnn_spec=module_specs._ll_inf_lstm_spec_20,
  low_level_initialisor_spec=module_specs._initialisation_mlp_spec,
  action_discriminator_spec=module_specs._action_discriminator_small,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=True)

hierarchical_lstm_bb_coord = HierarchicalNetworkSpec(
  conv_encoder_spec=module_specs._bb_coord_encoder_mlp_spec,
  conv_decoder_spec=module_specs._bb_coord_decoder_mlp_spec,
  encoder_rnn_spec=module_specs._encoder_lstm_spec_20,
  inference_rnn_spec=module_specs._inf_lstm_spec_20,
  high_level_rnn_spec=module_specs._highlevel_lstm_spec_inf20,
  low_level_rnn_spec=module_specs._lowlevel_lstm_spec,
  low_level_inf_rnn_spec=module_specs._ll_inf_lstm_spec_20,
  low_level_initialisor_spec=module_specs._initialisation_mlp_spec,
  action_discriminator_spec=module_specs._action_discriminator_small,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=False)

hierarchical_lstm_bb_coord_dt_ll = HierarchicalNetworkSpec(
  conv_encoder_spec=module_specs._bb_coord_encoder_mlp_spec,
  conv_decoder_spec=module_specs._bb_coord_decoder_mlp_spec,
  encoder_rnn_spec=module_specs._encoder_lstm_spec_20,
  inference_rnn_spec=module_specs._inf_lstm_spec_20,
  high_level_rnn_spec=module_specs._highlevel_lstm_spec_inf20_dt_ll,
  low_level_rnn_spec=module_specs._lowlevel_lstm_spec_dt_ll,
  low_level_inf_rnn_spec=module_specs._ll_inf_lstm_spec_20,
  low_level_initialisor_spec=module_specs._initialisation_mlp_spec,
  action_discriminator_spec=module_specs._action_discriminator_small,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=False)

hierarchical_lstm_bb_coord_dt_ll_gt_keys = HierarchicalNetworkSpec(
  conv_encoder_spec=module_specs._bb_coord_encoder_mlp_spec,
  conv_decoder_spec=module_specs._bb_coord_decoder_mlp_spec,
  encoder_rnn_spec=module_specs._encoder_lstm_spec_20,
  inference_rnn_spec=module_specs._inf_lstm_spec_20_gtKeys,
  high_level_rnn_spec=module_specs._highlevel_lstm_spec_inf20_dt_ll,
  low_level_rnn_spec=module_specs._lowlevel_lstm_spec_dt_ll,
  low_level_inf_rnn_spec=module_specs._ll_inf_lstm_spec_20,
  low_level_initialisor_spec=module_specs._initialisation_mlp_spec,
  action_discriminator_spec=module_specs._action_discriminator_small,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=False)

hierarchical_lstm_mnist_small_res_gtkeys = HierarchicalNetworkSpec(
  conv_encoder_spec=module_specs._dcgan_spec_denton_mnist_very_small_res,
  conv_decoder_spec=module_specs._dcgan_spec_denton_mnist_very_small_res,
  encoder_rnn_spec=module_specs._encoder_lstm_spec_128,
  inference_rnn_spec=module_specs._inf_lstm_spec_128_gtKeys,
  high_level_rnn_spec=module_specs._highlevel_lstm_spec_inf128,
  low_level_rnn_spec=module_specs._lowlevel_lstm_spec,
  low_level_inf_rnn_spec=module_specs._ll_inf_lstm_spec_20,
  low_level_initialisor_spec=module_specs._initialisation_mlp_spec,
  action_discriminator_spec=module_specs._action_discriminator_small,
  share_past_future_decoder=False,
  share_past_future_rnn=True,
  use_recursive_skips=True)
