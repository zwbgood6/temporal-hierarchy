"""Loss specifications."""
import collections
import tensorflow as tf

loss_fields = [
    "image_prediction",  # Train on future image prediction?
    "latent_prediction",  # Train on future latent prediction?
    "image_reconstruction",  # Train on past image reconstruction?
    "encoder_rnn_image_prediction", # Match Decoder(f([X, Xt])) to Decoder(X')?
]


loss_options = [
    "combine_losses",  # Sum losses or use separate optimizers?
    "image_loss_type",  # which loss to use for image reconstruction
    "image_output_activation",  # to apply to images before evaluating loss
    "backprop_to_targets",  # Backprop from loss to the (encoded) targets?
    "backprop_elstm_to_encoder",  # If True, allows encoding LSTM to backprop to encoding CNN
    "train_ssim"  # If True, minimize DSSIM loss in addition to pixel loss
]

DeprecatedLossConfig = collections.namedtuple(
    "DeprecatedLossConfig", loss_options + loss_fields)

LossConfig = collections.namedtuple(
    "LossConfig", loss_options )

mnist_predict_only = DeprecatedLossConfig(
    image_loss_type="binary_cross_entropy",
    image_output_activation=tf.nn.sigmoid,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=True,
    train_ssim=False,
    # Loss weights:
    image_prediction=1.0,
    latent_prediction=0.0,
    image_reconstruction=0.0,
    encoder_rnn_image_prediction=0.0,
)

# Here: Pre
kth_l2 = DeprecatedLossConfig(
    image_loss_type="mse",
    image_output_activation=tf.tanh,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=True,
    train_ssim=False,
    # Loss weights:
    image_prediction=1.0,
    latent_prediction=0.0,
    image_reconstruction=0.0,
    encoder_rnn_image_prediction=0.0,
)

kth_bce = DeprecatedLossConfig(
    image_loss_type="binary_cross_entropy",
    image_output_activation=tf.nn.sigmoid,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=True,
    train_ssim=False,
    # Loss weights:
    image_prediction=1.0,
    latent_prediction=0.0,
    image_reconstruction=0.0,
    encoder_rnn_image_prediction=0.0,
)

kth_l1 = DeprecatedLossConfig(
    image_loss_type="absolute_difference",
    image_output_activation=tf.tanh,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=True,
    train_ssim=False,
    # Loss weights:
    image_prediction=1.0,
    latent_prediction=0.0,
    image_reconstruction=0.0,
    encoder_rnn_image_prediction=0.0,
)

kth_l1_sigmoid = DeprecatedLossConfig(
    image_loss_type="absolute_difference",
    image_output_activation=tf.nn.sigmoid,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=True,
    train_ssim=False,
    # Loss weights:
    image_prediction=1.0,
    latent_prediction=0.0,
    image_reconstruction=0.0,
    encoder_rnn_image_prediction=0.0,
)

kth_gan = DeprecatedLossConfig(
    image_loss_type="mse",
    image_output_activation=tf.tanh,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=False,
    train_ssim=False,
    # Loss weights:
    image_prediction=0.0,
    latent_prediction=0.0,
    image_reconstruction=0.0,
    encoder_rnn_image_prediction=0.0,
)

mnist_gan = DeprecatedLossConfig(
    image_loss_type="binary_cross_entropy",
    image_output_activation=tf.nn.sigmoid,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=False,
    train_ssim=False,
    # Loss weights:
    image_prediction=0.0,
    latent_prediction=0.0,
    image_reconstruction=0.0,
    encoder_rnn_image_prediction=0.0,
)

gridworld_kl0003ca1 = DeprecatedLossConfig(
    image_loss_type="binary_cross_entropy",
    image_output_activation=tf.nn.sigmoid,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=True,
    train_ssim=False,
    # Loss weights:
    image_prediction=1.0,
    latent_prediction=0.0,
    image_reconstruction=0.0,
    encoder_rnn_image_prediction=1.0,
)

reacher_variational = DeprecatedLossConfig(
    image_loss_type="mse",
    image_output_activation=tf.nn.tanh,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=True,
    train_ssim=False,
    # Loss weights:
    image_prediction=1.0,
    latent_prediction=0.0,
    image_reconstruction=0.0,
    encoder_rnn_image_prediction=1.0,
)

mnist_variational_mse = DeprecatedLossConfig(
    image_loss_type="mse",
    image_output_activation=tf.nn.sigmoid,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=True,
    train_ssim=False,
    # Loss weights:
    image_prediction=1.0,
    latent_prediction=0.0,
    image_reconstruction=0.0,
    encoder_rnn_image_prediction=1.0,
)

mnist_variational = DeprecatedLossConfig(
    image_loss_type="binary_cross_entropy",
    image_output_activation=tf.nn.sigmoid,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=True,
    train_ssim=False,
    # Loss weights:
    image_prediction=1.0,
    latent_prediction=0.0,
    image_reconstruction=0.0,
    encoder_rnn_image_prediction=1.0,
)

hierarchical_mnist = LossConfig(
    image_loss_type="binary_cross_entropy",
    image_output_activation=tf.nn.sigmoid,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=True,
    train_ssim=False,
)

hierarchical_l1 = LossConfig(
    image_loss_type="absolute_difference",
    image_output_activation=tf.nn.sigmoid,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=True,
    train_ssim=False,
)

hierarchical_mse = LossConfig(
    image_loss_type="mse",
    image_output_activation=tf.nn.sigmoid,
    backprop_to_targets=False,
    combine_losses=True,
    backprop_elstm_to_encoder=True,
    train_ssim=False,
)