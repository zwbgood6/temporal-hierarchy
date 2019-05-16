"Definition of all parameters."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

# Main
tf.flags.DEFINE_integer("num_training_iterations", 100000,
                          "Number of iterations to train for.")
tf.flags.DEFINE_integer("num_testing_iterations", 0,
                          "Number of iterations to test on. Zero means run on all test data")
tf.flags.DEFINE_integer("report_interval", 100,
                          "Iterations between reports (train losses only).")
tf.flags.DEFINE_integer("validation_interval", 1000,
                          "Iterations between validation and full reports "
                          "(including images).")
tf.flags.DEFINE_integer("checkpoint_interval", 1000,
                          "Checkpointing step interval.")
tf.flags.DEFINE_integer("train_batch_size", 50,
                          "Batch size for training.")
tf.flags.DEFINE_integer("val_batch_size", 50,
                          "Batch size for validation.")
tf.flags.DEFINE_integer("test_batch_size", 20,
                          "Batch size for testing. ")
tf.flags.DEFINE_integer("input_seq_len", 0,
                          "Length of the past sequence. If zero, defaults to a"
                          "dataset dependent value.")
tf.flags.DEFINE_integer("pred_seq_len", 10,
                          "Length of the future sequence. If zero, defaults to a"
                          "dataset dependent value.")
tf.flags.DEFINE_integer("input_img_res", 64,
                          "Resolution of groundtruth/predicted images (in px)."
                          "(works for MNIST)")
tf.flags.DEFINE_string("base_dir", "~/logs",
                        "Base directory for checkpoints and summaries.")
tf.flags.DEFINE_string("network_config_name", "simple_conv_lstm_conv",
                        "The network architecture configuration to use.")
tf.flags.DEFINE_string("dataset_config_name", "moving_mnist_basic",
                        "The dataset configuration to use.")
tf.flags.DEFINE_string("loss_config_name", "mnist_bce_image_latent",
                        "The loss configuration to use.")
tf.flags.DEFINE_boolean("create_new_subdir", False,
                          "If True, creates a new subdirectory in base_dir. "
                          "Set to False to reload. Defaults to True.")
tf.flags.DEFINE_integer("output_buffer_size", 100,
                          "The size of the training buffer size for the output "
                          "prefetch queue, as a multiple of the batch size. E.g. "
                          "a value of 10 will produce a buffer of size 10 * "
                          "batch_size. Defaults to 100.")
tf.flags.DEFINE_integer("shuffle_buffer_size", 5,
                          "The size of the training buffer size for the shuffle "
                          "queue, as a multiple of the batch size. E.g. a value "
                          "of 10 will produce a buffer of size 10 * batch_size. "
                          "Defaults to 100.")
tf.flags.DEFINE_boolean("kth_downsample", False,
                          "If True, downsamples videos to half-resolution."
                          "I.e. 64x64 for KTH or 128x128 for UCF.")
tf.flags.DEFINE_integer("video_length_downsample", 1,
                          "Subsamples video frames. FPS_new = FPS_base / video_length_downsample")
tf.flags.DEFINE_boolean("debug_main_loop", False,
                          "If True, opens pdb into the main loop right after a training step.")
tf.flags.DEFINE_boolean("validate", True,
                          "If True, validation data will be also processed. This might not"
                          " always be needed on infinite datasets where overfitting is not an issue.")
tf.flags.DEFINE_boolean("debug", False,
                          "If True, debug outputs are printed.")

# Model options
tf.flags.DEFINE_boolean("use_recursive_image", False,
                          "If True, uses recursive image skips when using a "
                          "recursive network. Ignored if network is not recursive.")
tf.flags.DEFINE_boolean("use_gt_attention_keys", False,
                          "If True, uses groundtruth image encodings as keys for inference"
                          "output attention, otherwise learned keys from inference output.")
tf.flags.DEFINE_boolean("inv_mdl_baseline", False,
                          "If True, Agarwal et al. 2016 inverse model regularizer with LSTM is used.")
tf.flags.DEFINE_boolean("no_LSTM_baseline", False,
                          "If True, prediction with simple MLP is used.")
tf.flags.DEFINE_string("normalization", "batchnorm",
                        'Type of normalization. "batchnorm" and "layernorm" supported.')
tf.flags.DEFINE_boolean("activate_latents", True,
                          "If True, will use tanh to activate image latents.")
tf.flags.DEFINE_boolean("use_cdna_decoder", False,
                          "If True, the images are produce via neural advection.")

# Optimization
tf.flags.DEFINE_string("learning_rate_reduce_criterion", "plateau",
                        "Criterion for reducing learning rate: "
                        "scheduled or plateau.")
tf.flags.DEFINE_float("plateau_scale_criterion", 2e-3,
                        "The criterion for determining a plateau in the "
                        "validation loss, as a fraction of the validation loss.")
tf.flags.DEFINE_integer("plateau_min_delay", 20000,
                          "The minimum number of batches to wait before decaying "
                          "the learning rate, if using a plateau criterion.")
tf.flags.DEFINE_integer("reduce_learning_rate_interval", 20000,
                          "Iterations between learning rate reductions.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-8,
                        "Epsilon used for Adam optimizer.")
tf.flags.DEFINE_float("regularizer_weight", 1e-4,
                        "Weight for weight decay regularizer.")
tf.flags.DEFINE_float("max_grad_norm", 5,
                        "Gradient clipping norm limit. If <=0, "
                        "no clipping is applied.")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Optimizer learning rate.")
tf.flags.DEFINE_float("reduce_learning_rate_multiplier", 1.0,
                        "Learning rate is multiplied by this when reduced.")
tf.flags.DEFINE_string("plateau_criterion_loss_name",
                        "total_loss_val",
                        "Loss to use as criterion for decaying learning rate.")
tf.flags.DEFINE_string("optimizer_type", "adam",
                        "Which optimizer to use. Can be "
                        "adam, sgd, momentum, or nesterov. Defaults to adam.")

# Variational
tf.flags.DEFINE_boolean("use_variational", False,
                          "If True, sampling from learned prior is used.")
tf.flags.DEFINE_boolean("teacher_forcing", True,
                          "Works with variational LSTM. If False, prediction and "
                          "prior networks observe generated latents in the future "
                          "at training. If True, they observe the true frames.")
tf.flags.DEFINE_boolean("fixed_prior", False,
                          "If True, unit Gaussian is used as fixed prior distribution.")
tf.flags.DEFINE_float("kl_divergence_weight", 1e-4,
                        "Weight for the KL divergence term, the beta in beta-VAE.")
tf.flags.DEFINE_boolean("use_old_kl", False,
                          "If True, uses old KL implementation.")

# Action inference
tf.flags.DEFINE_boolean("infer_actions", False,
                          "If True, regression network for inferring actions is added.")
tf.flags.DEFINE_boolean("train_action_regressor", False,
                          "If True, regression network for inferring actions "
                          "from images is trained.")
tf.flags.DEFINE_boolean("train_abs_action_regressor", False,
                          "If True, regression network for inferring absolute actions "
                          "from images is trained.")
tf.flags.DEFINE_float("lr_action_inference", 1e-3,
                        "Learning rate for the action inference network. Used when "
                        "infer_actions is true.")
tf.flags.DEFINE_boolean("chance_actions", False,
                          "If true, the action regression network does not observe any input")
tf.flags.DEFINE_boolean("supervised_actions", False,
                          "If true, the gradient is propagated from action/latent regression to"
                          "the prediction network.")
tf.flags.DEFINE_boolean("ignore_lift_action", False,
                          "If True, lift action is ignored for infer action loss in BAIR dataset.")


# Test Data Efficiency
tf.flags.DEFINE_boolean("reinit_action_regression", False,
                          "If true, will reset the action regression network weights.")
tf.flags.DEFINE_boolean("action_conditioned_prediction", False,
                          "If true the prediction model is conditioned on true actions.")
tf.flags.DEFINE_boolean("use_cdna_model", False,
                          "If True, CDNA model of Lee&Finn (2018) is used as decoder RNN core.")
tf.flags.DEFINE_boolean("print_infer_actions", False,
                          "If True, prints action inference loss to cmd.")
tf.flags.DEFINE_integer("n_shards", 0,
                          "If non-zero, only n tfrecord files are read instead of all.")
tf.flags.DEFINE_boolean("is_test", False,
                          "If True, test run is performed.")

# Visualization
tf.flags.DEFINE_integer("test_sequence_repeat", 0,
                          "How often should val/test sequences be repeated to see variation.")
tf.flags.DEFINE_boolean("gen_html_summary", False,
                          "If True, GIF overview in HTML is generated.")
tf.flags.DEFINE_boolean("save_z_samples", False,
                          "If true, will save the inference z samples into a numpy array at validation"
                          ". Slows down the validation.")

# Hierarchical Prediction
tf.flags.DEFINE_integer("n_frames_segment", 10,
                          "Number of low-level decoded image per segment.")
tf.flags.DEFINE_integer("n_segments", 5,
                          "Maximal number of predicted segments.")
tf.flags.DEFINE_boolean("use_full_inf", False,
                          "If True, uses all inference outputs.")
tf.flags.DEFINE_boolean("scenario_based", False,
                          "If True, the computation of ground truth targets will be based on considering"
                          " all possible keyframe placement scenarios.")
tf.flags.DEFINE_boolean("activate_before_averaging", False,
                          "If True, activates the images before averaging for the gt-normalized loss.")
tf.flags.DEFINE_boolean("norm_high_level_loss", False,
                          "If True, high level loss values are normed by the fraction of keyframe "
                          "probability distributions within the loss region.")
tf.flags.DEFINE_boolean("predict_dt_low_level", False,
                          "If True, the temporal offset of the next keyframe is predicted by the "
                          "low level network instead of the high level network.")
tf.flags.DEFINE_boolean("test_hl_latent_swap", False,
                          "If True, high level latents are exchanged with ground truth latent "
                          "at maximum dt value.")
tf.flags.DEFINE_boolean("inference_backwards", True,
                          "It True, the inference LSTM goes backwards.")
tf.flags.DEFINE_boolean("pretrain_ll", False,
                          "It True, the only the low level LSTM is pretrained.")
tf.flags.DEFINE_integer("min_pretrain_seg_len", 0,
                          "Minimal length of random pretraining segments.")
tf.flags.DEFINE_integer("max_pretrain_seg_len", 0,
                          "Maximal length of random pretraining segments.")
tf.flags.DEFINE_boolean("freeze_ll", False,
                          "It True, params of the low level network and encoder+decoder are not updated.")
tf.flags.DEFINE_string("load_ll_ckpt_dir", "",
                        "Path to checkpoint folder that low level network + enc/decoder weights "
                        "should be loaded from.")
tf.flags.DEFINE_boolean("handcrafted_attention", False,
                          "If True, the keyframe encodings are selected instead of attention mechanism.")
tf.flags.DEFINE_boolean("separate_attention_key", False,
                          "If True, the attention key produced by the keyframe network is different from "
                          "the produced frame embedding.")
tf.flags.DEFINE_boolean("ll_mlp", False,
                          "If True, the low level RNN is modeled as a state-less MLP (layer size is parsed "
                          "from low level RNN spec).")
tf.flags.DEFINE_boolean("stateless_predictor", False,
                          "If True, predictor RNN uses a stateless (MLP) core.")
tf.flags.DEFINE_boolean("train_hl_latent_swap", False,
                          "If True, high level latents are exchanged with ground truth latent "
                          "at maximum dt value.")
tf.flags.DEFINE_boolean("ll_svg", False,
                          "If True, the low level RNN is treated as a decoder of a variational model")
tf.flags.DEFINE_boolean("train_hl_latent_soft_swap", False,
                          "If True, high level latents are exchanged with ground truth latent "
                          "averaged by the dt value.")
tf.flags.DEFINE_integer("fixed_dt", 0,
                          "If larger than 0, the predicted dts are substituted with a constant value 'fixed_dt'.")
tf.flags.DEFINE_boolean("goal_conditioned", False,
                          "If true, the predictive model is conditioned on the goal.")
tf.flags.DEFINE_boolean("predict_to_the_goal", False,
                          "If True, the reconstruction loss is only applied up to the goal timestep including it.")
tf.flags.DEFINE_boolean("goal_every_step", False,
                          "If True, the goal is fed as input to high-level LSTM at every step")
tf.flags.DEFINE_boolean("hl_learned_prior", True,
                          "If True, the prior distribution for the variational model is produced from the previous"
                          "LSTM state in the high-level network.")
tf.flags.DEFINE_boolean("decode_actions", False,
                          "If True, the inference LSTM observes and the low-level LSTM produces actions.")
tf.flags.DEFINE_boolean("static_dt", False,
                          "If True, dt prediction is replaced with a variable to learn a static dt scheme.")
tf.flags.DEFINE_boolean("use_svg_kf_detection", False,
                          "If True, uses max KL to determine keyframes.")


# Loss balance terms
tf.flags.DEFINE_float("low_level_image_term", 1.0,
                          "Balance term on the low-level image loss.")
tf.flags.DEFINE_float("gt_target_loss_term", 0.0,
                          "Balance term on the image loss on computed ground truth targets.")
tf.flags.DEFINE_float("high_level_latent_term", 0,
                          "Balance term on the high-level latent loss.")
tf.flags.DEFINE_float("high_level_image_term", 0,
                          "Balance term on the high-level image loss.")

tf.flags.DEFINE_boolean("schedule_seq_len_weight", False,
                          "If true, the weighting factor for the sequence length loss will be "
                          "exponentially increased over training time until it reaches the desired value.")
tf.flags.DEFINE_integer("seq_len_schedule_step", 1000,
                          "Training iterations step size for exponential weight factor increase.")
tf.flags.DEFINE_float("sequence_length_term", 0,
                          "Balance term on the loss that ensures the length of the predicted "
                          "sequence.")

tf.flags.DEFINE_float("entropy_term", 0,
                          "Balance term on the entropy penalty loss.")
tf.flags.DEFINE_float("supervise_dt_term", 0,
                          "Balance term on the dt supervision loss.")
tf.flags.DEFINE_float("supervise_attention_term", 0,
                          "If more than 0, the attention weights are forced to correspond to the keyframes.")
tf.flags.DEFINE_float("ll_kl_term", 0,
                          "Balance term on the low-level inference KL divergence.")
tf.flags.DEFINE_float("ll_actions_term", 0.05,
                          "Balance term on the low-level action decoding, if decode_actions=True.")


# Temperature dt Prediction
tf.flags.DEFINE_float("tau", 1.0,
                          "Dt softmax temperature.")
tf.flags.DEFINE_float("tau_min", 0.1,
                          "Minimal softmax temperature when scheduled decrease is applied.")
tf.flags.DEFINE_integer("tau_schedule_step", 1000,
                          "Training iterations step size for geometric temperature decrease.")
tf.flags.DEFINE_float("reduce_temp_multiplier", 1.0,
                          "Multiplier for temperature schedule step.")

# Datasets
# {{{
# Bouncing Balls
tf.flags.DEFINE_boolean("image_input", True,
                          "If True, uses images as observation modality, "
                          "otherwise ball coordinates.")
tf.flags.DEFINE_boolean("stochastic_angle", False,
                          "If True, balls have stochastic bouncing angles.")
tf.flags.DEFINE_boolean("random_angle", False,
                          "If True, balls have random bouncing angles.")
tf.flags.DEFINE_boolean("stochastic_speed", False,
                          "If True, balls change speed upon bounce.")
tf.flags.DEFINE_boolean("stochastic_bounce", False,
                          "If True, balls bounce at random positions.")
tf.flags.DEFINE_boolean("rand_start_counter", False,
                          "If True, first segment has random length in stochastic bounce.")
tf.flags.DEFINE_boolean("bounce_vertically", False,
                          "If True, balls bounce only up and down.")
tf.flags.DEFINE_integer("num_balls", 1,
                          "Number of bouncing balls.")
tf.flags.DEFINE_float("ball_speed", 2.0,
                          "Speed of the balls in multiples of ball radius/time step.")
tf.flags.DEFINE_float("variance_degrees", 5.0,
                          "Variance of the Gaussian angle noise in case of stochastic angles.")
tf.flags.DEFINE_integer("min_segment_length", 1,
                          "Minimum number of steps before bounce.")
tf.flags.DEFINE_integer("max_segment_length", 100,
                          "Maximum number of steps before bounce.")
# }}}

# Sanity checking
if tf.flags.FLAGS.train_abs_action_regressor and "reacher" in tf.flags.FLAGS.dataset_config_name and \
    not tf.flags.FLAGS.train_action_regressor:
  raise ValueError("If training abs actions on reacher, angle regressor must be turned on!")


# print and dump params
def dump_params(base_dir, file_extension=None):
  print('')
  flags = tf.flags.FLAGS.__flags
  if file_extension is None:
    param_file = open(os.path.join(base_dir, "params.txt"), "w")
  else:
    param_file = open(os.path.join(base_dir, "params_%s.txt" % file_extension), "w")
  for key in flags:
      if tf.VERSION == '1.4.0':
        param_str = "%s : %s" % (key, str(flags[key]))  # FLAGS API changed afterwards
      else:
        param_str = "%s : %s" % (key, str(flags[key].value))
      print(param_str)
      param_file.write(param_str + '\n')
  param_file.close()
  print('')
