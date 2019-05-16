"""Architectures for sequence prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from architectures import arch_utils, conv_architectures, rnn_architectures, mlp_architectures
import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow.contrib.distributions import Normal
from utils import maybe_stop_gradients, pick_and_tile, make_dists_and_sample, debug, shape, AttrDict, make_attr_dict
import utils
from specs import module_specs, network_specs
from architectures import lstm, cores, th_utils

FLAGS = tf.flags.FLAGS


class TemporalHierarchyModel(snt.AbstractModule):
  """An autoencoding + prediction architecture. encoding + LSTM + decoding."""

  def __init__(self,
               network_spec,
               channels=1,
               input_image_shape=None,
               output_activation=None,
               backprop_elstm_to_encoder=False,
               use_recursive_image=True,
               has_image_input=True,
               render_fcn=None,
               render_shape=None,
               data_format="NCHW",
               num_actions=None,
               tau=None,
               infer_actions=False,
               name="conv_embedding_lstm"):
    """Constructs the TH model.

    Args:
      network_spec: A namedtuple specifying the parameters of the network
        modules.
      variational: If True, use variational video prediction with learned prior.
      pred_len: The number of frames to predict after the end of the sequence.
        Defaults to 10.
      channels: Number of image channels. Defaults to 1.
      output_activation: The activation to apply to decoded frames. This is
        only applied to frames that will be re-encoded, to enable use of fused
        losses afterwards. If None, the model output is used directly. Defaults
        to None.
      backprop_elstm_to_encoder: If True, the LSTM encoder is used to train
        the CNN encoding. This allows gradients from prediction losses to flow
        back to the CNN through the encoding LSTM. Defaults to False.
      data_format: The format of the input images: 'NCHW' or 'NHWC'.
      num_actions: Number of actions to infer per input.
      name: The module name. Defaults to 'conv_embedding_lstm'.
    """

    super(TemporalHierarchyModel, self).__init__(name=name)
    self._network_spec = network_spec
    self._conv_encoder_spec = network_spec.conv_encoder_spec
    self._conv_decoder_spec = network_spec.conv_decoder_spec
    self._has_image_input = has_image_input
    self._render_fcn = render_fcn   # coord based prediction render fcn
    self._render_shape = render_shape
    self._tau = tau

    self._share_past_future_decoder = network_spec.share_past_future_decoder
    self._share_past_future_rnn = network_spec.share_past_future_rnn
    self._use_recursive_skips = network_spec.use_recursive_skips
    self._multilayer_skips = self._use_recursive_skips and not FLAGS.use_cdna_decoder
    self._use_recursive_image = use_recursive_image

    self._channels = channels
    self._input_image_shape = input_image_shape
    self._output_activation = output_activation
    self._data_format = data_format
    self._backprop_elstm_to_encoder = backprop_elstm_to_encoder

    self._num_actions = num_actions
    self._train_action_regressor = infer_actions
    if self._train_action_regressor:
      self._action_discriminator_spec = network_spec.action_discriminator_spec

    if tf.flags.FLAGS.action_conditioned_prediction:
      # Any of the variational functionality is not needed here except static decoding
      self._train_action_regressor = False
    elif tf.flags.FLAGS.use_cdna_model:
      self._use_conv_lstm = True
    else:
      decoder_rnn_spec = self._network_spec.decoder_rnn_spec if not isinstance(self._network_spec,
                                                                           network_specs.HierarchicalNetworkSpec) \
                                                             else self._network_spec.high_level_rnn_spec
      self._use_conv_lstm = arch_utils.check_rnns_conv(
        self._network_spec.encoder_rnn_spec,
        decoder_rnn_spec,
        False)
      
    self._setup_network_specs()

  def _apply_recursive_skips(
      self,
      conv_decoder,
      latent,
      is_training,
      first_skips,
      goal_img=None):
    """Recursively decodes latents, using previous decoder to get next skips."""
    decoded_sequence_list = []
    output_mask_list = []
    goal_img_list = []
    skips_prev = first_skips if not FLAGS.use_cdna_decoder else None
    decoded_i = first_skips
    goal_img_transformed = goal_img
    for time_i in range(latent.get_shape()[0]):
      decoded_i, skips_prev, output_mask_i, goal_img_transformed = conv_decoder(
          latent[time_i, ...],
          is_training,
          skips_prev,
          build_recursive_skips=self._multilayer_skips,
          prev_img=decoded_i,
          first_img=first_skips,
          goal_img=goal_img_transformed)
      decoded_sequence_list.append(decoded_i)
      output_mask_list.append(output_mask_i)
      goal_img_list.append(goal_img_transformed)

    return decoded_sequence_list, output_mask_list, goal_img_list

  def _get_conv_decoder(self, decoder_phase):
    """Helper to return decoder associated with a phase."""
    if decoder_phase == "past":
      conv_decoder = self.conv_decoder
    elif decoder_phase == "future":
      conv_decoder = self.conv_decoder_future
    else:
      raise ValueError("Unknown decoder.")

    return conv_decoder

  def _build_image_decoder(
      self,
      latent,
      skip_connections,
      is_training,
      decoder_phase,
      last_input_frame,
      use_recursive_image,
      goal_img=None):
    """Builds recursive or not recursive image decoder"""

    if decoder_phase != "future":
      raise ValueError("Why do you want to do this?")

    if self._use_recursive_skips:
      decoded_seq_predict, goal_imgs_out = self._build_recursive_image_decoder(
        latent,
        skip_connections,
        is_training,
        decoder_phase=decoder_phase,
        last_input_frame=last_input_frame,
        use_recursive_image=use_recursive_image,
        goal_img=goal_img)
    else:
      decoded_seq_predict = self._build_simple_image_decoder(
        latent,
        skip_connections,
        is_training,
        decoder_phase=decoder_phase)

    if goal_img is not None:
      return decoded_seq_predict, goal_imgs_out
    else:
      return decoded_seq_predict

  def _build_simple_image_decoder(
      self,
      latent,
      skip_connections,
      is_training,
      decoder_phase):
    """Adds decoder computation to the graph."""
    conv_decoder = self._get_conv_decoder(decoder_phase)
    decoded_sequence = snt.BatchApply(conv_decoder)(
        latent, is_training, skip_connections)

    return decoded_sequence

  def _build_recursive_image_decoder(
      self,
      latent,
      skip_connections,
      is_training,
      decoder_phase,
      last_input_frame,
      use_recursive_image,
      goal_img=None):
    """Adds recursive decoder computation to the graph."""
    conv_decoder = self._get_conv_decoder(decoder_phase)
    decoded_seq_predict_list, output_mask_list, goal_img_list = self._apply_recursive_skips(
        conv_decoder,
        latent,
        is_training,
        skip_connections,
        goal_img)

    if use_recursive_image:
      decoded_sequence = self._propagate_image_residuals(
          decoded_seq_predict_list,
          output_mask_list,
          true_prev_frame=last_input_frame)
    else:
      decoded_sequence = tf.stack(decoded_seq_predict_list, axis=0)
      
    if goal_img_list[0] is not None:
      goal_imgs_out = tf.stack(goal_img_list, axis=0)
    else:
      goal_imgs_out = None

    return decoded_sequence, goal_imgs_out

  def _build_rnn_core(self, spec):
    rnn_core, rnn_init_state_getter = \
      rnn_architectures.build_rnn(spec)
    return rnn_core, rnn_init_state_getter

  def _build_stateless_core(self, spec):
    mlp_layers = self._rnnSpec2layers(spec)
    rnn_base_core = discriminators.GeneralizedMLP(layers=mlp_layers)
    def rnn_base_core_wrapper(input, state):
      return rnn_base_core(input), None
    def _stateless_core_init(batch_size):
      return None
    return rnn_base_core_wrapper, _stateless_core_init

  @staticmethod
  def _rnnSpec2layers(rnn_spec):
    layers = []
    for element in rnn_spec:
      if isinstance(element, module_specs.LinearSpec):
        layers.append(element.output_size)
      elif isinstance(element, module_specs.LSTMSpec):
        layers.append(element.num_hidden)
      else:
        print(element)
        raise NotImplementedError("Cannot parse RNN spec element to number of layers!")
    return layers

  def _setup_network_specs(self):
    
    # Some variable dimensions
    if self._has_image_input:
      self.encoded_img_size = self._conv_encoder_spec.dcgan_latent_size
    else:
      self.encoded_img_size = self._conv_encoder_spec.layers[-1]  # MLP for coord based prediction

    if FLAGS.use_gt_attention_keys:
      self.z_dist_size = self._network_spec.inference_rnn_spec[-1].output_size
    else:
      self.z_dist_size = self._network_spec.inference_rnn_spec[-1].output_size - self.encoded_img_size
      
    if FLAGS.decode_actions:
      self._network_spec.low_level_rnn_spec[-1].output_size = self._network_spec.low_level_rnn_spec[-1].output_size + \
                                                              self._num_actions
      
    self._network_spec.high_level_rnn_spec[-1].output_size = self._network_spec.high_level_rnn_spec[-1].output_size + \
                                                             tf.flags.FLAGS.n_frames_segment
    
    if FLAGS.separate_attention_key:
      self._network_spec.high_level_rnn_spec[-1].output_size = \
        self._network_spec.high_level_rnn_spec[-1].output_size + self.encoded_img_size
      
      self._network_spec.encoder_rnn_spec[-1].output_size = \
        self._network_spec.encoder_rnn_spec[-1].output_size + self.encoded_img_size


  def _setup_rnn_modules(self):
    
    # Past and future RNNs: cores, initial state getters, and runners
    with tf.variable_scope("encoder_rnn"):
      encoder_rnn_base_core, self._encoder_rnn_init_state_getter = \
          self._build_rnn_core(self._network_spec.encoder_rnn_spec)
      self.encoder_rnn_core = cores.SimpleCore(encoder_rnn_base_core)
      self.encoder_rnn = lstm.CustomLSTM(
          core=self.encoder_rnn_core)

    with tf.variable_scope("inference_rnn"):
      inference_rnn_base_core, self._inference_rnn_init_state_getter = \
          self._build_rnn_core(self._network_spec.inference_rnn_spec)
      self.inference_rnn_core = cores.SimpleCore(inference_rnn_base_core)
      self.inference_rnn = lstm.CustomLSTM(
          core=self.inference_rnn_core,
          backwards=FLAGS.inference_backwards)

    with tf.variable_scope("high_level_rnn"):
      if FLAGS.goal_conditioned:
        high_level_init_mlp = mlp_architectures.GeneralizedMLP(layers=self._network_spec.low_level_initialisor_spec.layers)
        self._high_level_rnn_init_state_getter = rnn_architectures.build_mlp_initializer(
          self._network_spec.high_level_rnn_spec,
          high_level_init_mlp)
        self.init_mlp = mlp_architectures.GeneralizedMLP(
          layers=self._network_spec.action_discriminator_spec.layers + [self.z_dist_size])

      dt_size = FLAGS.n_frames_segment
      high_level_rnn_base_core, _ = \
          self._build_rnn_core(self._network_spec.high_level_rnn_spec)
      self.high_level_rnn_core = cores.KeyInCore(high_level_rnn_base_core,
                                                 self.encoded_img_size,
                                                 dt_size,
                                                 self.z_dist_size,
                                                 self._tau)
      self.high_level_rnn = lstm.CustomLSTM(
          core=self.high_level_rnn_core)


    with tf.variable_scope("low_level_rnn"):
      if FLAGS.ll_mlp:    # build a memory-less RNN based on MLP core
        ll_mlp_layers = self._rnnSpec2layers(self._network_spec.low_level_rnn_spec)
        low_level_rnn_base_core = mlp_architectures.GeneralizedMLP(layers=ll_mlp_layers)
        self.low_level_rnn_core = cores.StateLessInterpolatorCore(low_level_rnn_base_core)
        self.low_level_rnn = lstm.CustomLSTM(core=self.low_level_rnn_core)
      else:   # generic LSTM, initialize state with learned MLP
        low_level_rnn_base_core, _ = \
            self._build_rnn_core(self._network_spec.low_level_rnn_spec)
        self.low_level_rnn_core = cores.SimpleCore(low_level_rnn_base_core)
        self.low_level_rnn = lstm.CustomLSTM(core=self.low_level_rnn_core)
        low_level_init_mlp = mlp_architectures.GeneralizedMLP(layers=self._network_spec.low_level_initialisor_spec.layers)
        self._low_level_rnn_init_state_getter = rnn_architectures.build_mlp_initializer(
          self._network_spec.low_level_rnn_spec,
          low_level_init_mlp)
        if FLAGS.ll_svg:
          ll_inf_rnn_base_core, self._ll_inf_rnn_init_state_getter = \
            self._build_rnn_core(self._network_spec.low_level_inf_rnn_spec)
          self.ll_inf_rnn_core = cores.ResetCore(ll_inf_rnn_base_core)
          self.ll_inf_rnn = lstm.CustomLSTM(
            core=self.ll_inf_rnn_core,
            backwards=False)

  def _setup_encdec_modules(self):
    """Initializes the network components (CNNs, RNNs, etc.)."""

    # CNN encoder
    with tf.variable_scope("image_encoder"):
      if self._has_image_input:
        self.conv_encoder = conv_architectures.build_cnn(
            self._conv_encoder_spec,
            network_type="encoder")
      else:
        self.conv_encoder = mlp_architectures.GeneralizedMLP(layers=self._conv_encoder_spec.layers)

    # CNN decoders
    with tf.variable_scope("image_decoder"):
      if FLAGS.use_cdna_decoder:
        image_activation = self._output_activation
      else:
        image_activation = None
      # set number of output channels according to data input
      if self._has_image_input:
        self.conv_decoder = conv_architectures.build_cnn(
            self._conv_decoder_spec,
            decoder_output_channels=self._channels,
            network_type="decoder",
            image_activation=image_activation)
      else:
        output_shape = self._input_image_shape
        assert len(output_shape) == 1, "For coord the data needs to be single dimensioned!"
        self.conv_decoder = mlp_architectures.GeneralizedMLP(layers=self._conv_decoder_spec.layers + output_shape,
                                                          final_activation=tf.nn.tanh)    # coord range -1...1

      if self._share_past_future_decoder:
        self.conv_decoder_future = self.conv_decoder
      else:
        if self._has_image_input:
          self.conv_decoder_future = conv_architectures.build_cnn(
              self._conv_decoder_spec,
              decoder_output_channels=self._channels,
              network_type="decoder",
              image_activation=image_activation)
        else:
          self.conv_decoder_future = mlp_architectures.GeneralizedMLP(layers=self._conv_decoder_spec.layers + output_shape)

    if self._train_action_regressor:
        with tf.variable_scope("abs_action_regressor"):
            # add number of actions as number of output neurons of action discriminator
            action_discr_layers = self._action_discriminator_spec.layers + [self._num_actions]
            self.action_discriminator = mlp_architectures.GeneralizedMLP(layers=action_discr_layers)

    if FLAGS.static_dt:
        with tf.variable_scope("static_dt", reuse=tf.AUTO_REUSE):
            shape_dt = (FLAGS.n_segments, 1, FLAGS.n_frames_segment)
            uniform_prob = 1.0 / FLAGS.n_frames_segment
            self.static_dt = tf.get_variable("dt", initializer=tf.fill(shape_dt, uniform_prob), trainable=True)


  def _build_encoder(self, full_sequence, n_frames_input, n_frames_predict, is_training):
    """Adds encoder computation to the graph."""
    if not self._has_image_input:
      encoded_seq_full = snt.BatchApply(self.conv_encoder)(full_sequence, is_training)
      skips_seq_full = None
    else:
      encoded_seq_full, skips_seq_full = snt.BatchApply(
          self.conv_encoder)(full_sequence, is_training)

    if len(shape(encoded_seq_full)) == 5:
      encoded_seq_full = encoded_seq_full[:, :, :, 0, 0]
    encoded_seq_reconstruct = encoded_seq_full[:n_frames_input, ...]
    if skips_seq_full is not None:
      # If network returns skip connections, parse them
      encoded_skips_reconstruct = [layer[:n_frames_input, ...]
                                   for layer in skips_seq_full]
    else:
      encoded_skips_reconstruct = None
    future_latents_true = encoded_seq_full[n_frames_input:n_frames_input+n_frames_predict, ...]
    future_latents_true_complete = encoded_seq_full[n_frames_input:, ...]   # with extra images for inference

    encoded = AttrDict()
    encoded["past"] = encoded_seq_reconstruct
    encoded["skips_past"] = encoded_skips_reconstruct
    encoded["future"] = future_latents_true
    encoded["future_complete"] = future_latents_true_complete
    encoded["complete"] = encoded_seq_full

    debug("Encoder", "output", "encoded[past]", shape(encoded["past"]))
    debug("Encoder", "output", "encoded[skips_past]", "list of tensors")
    debug("Encoder", "output", "encoded[future]", shape(encoded["future"]))
    debug("Encoder", "output", "encoded[future_complete]", shape(encoded["future_complete"]))
    if FLAGS.debug: print()

    # Note that the last past frame predicts the first future time, so its
    # encoding is given to the future RNN. Only the last past encoded frame is
    # given as input to the future RNN: other inputs are generated recurrently.
    # encoded["past_sg"] = maybe_stop_gradients(
    #   encoded_seq_full[:(n_frames_input - 1), ...],
    #     stop_criterion=(not self._backprop_elstm_to_encoder)
    # )
    # encoded["future_sg"] = maybe_stop_gradients(
    #   encoded_seq_full[n_frames_input - 1, ...],
    #     stop_criterion=(not self._backprop_elstm_to_encoder)
    # )
    return encoded

  def _build_skips(
      self,
      input_skips,
      input_data,
      n_frames_input):
    """Builds the skip connection inputs for the past and future."""

    if input_skips is not None:
      if self._multilayer_skips:
        skips_predict = [layer[(n_frames_input - 1)] for layer in input_skips]
        skips_reconstruct = [layer[:n_frames_input] for layer in input_skips]
      else:
        # Default use first input image skips for reconstruction
        # Use last input image skips for prediction
        skips_reconstruct = pick_and_tile(input_skips, 0, n_frames_input)
        skips_predict = pick_and_tile(input_skips, -1, FLAGS.n_frames_segment)

      skips = {
          "reconstruct": skips_reconstruct,
          "predict": skips_predict,
          "past_rnn":[scale[:-1, ...] for scale in skips_reconstruct]
      }
    else:
      skips = {
          "reconstruct": None,
          "predict": None,
          "past_rnn": None,}
      
    if FLAGS.use_cdna_decoder:
      skips["predict"] = input_data.input_images[-1]
      
    return skips

  def _propagate_image_residuals(
      self,
      decoded_seq_predict,
      predict_seq_mask,
      true_prev_frame):
    """Computes the final image estimate by adding previous images.

    Args:
      decoded_seq_predict: A list of predicted output image tensors.
      predict_seq_mask: A list of predicted image mask tensors.
      true_prev_frame: The true frame preceding the first predicted image.
    Returns:
      decoded_seq_final: A tensor of final decoded image estimates.
    """
    # Need to sort out _use_image_residuals and _mask_output
    # decoded_sec_predict_mask
    prev_est = true_prev_frame
    decoded_seq_activated = []
    for time_i in range(len(decoded_seq_predict)):
      # Mask: 1 chooses previous frame, 0 chooses next frame
      decoded_time_i = (predict_seq_mask[time_i] * prev_est +
          (1 - predict_seq_mask[time_i]) * decoded_seq_predict[time_i])

      if self._output_activation is not None:
        decoded_time_i = self._output_activation(decoded_time_i)
      prev_est = decoded_time_i
      decoded_seq_activated.append(decoded_time_i)

    decoded_seq_final = tf.stack(decoded_seq_activated, axis=0)

    return decoded_seq_final

  def infer_action_from_img(self, img_seq, is_training, latent_seq=None):
      if latent_seq is not None:
        reencoded = latent_seq
      else:
        reencoded, _ = snt.BatchApply(self.conv_encoder)(img_seq, is_training)
      action_seq = snt.BatchApply(self.abs_action_discriminator)(reencoded)
      if tf.flags.FLAGS.train_action_regressor:
        return tf.constant(2 * np.pi) * tf.sigmoid(action_seq)
      else:
        return action_seq

  @staticmethod
  def get_attention_keys(encoded):
    if FLAGS.use_gt_attention_keys:
      if FLAGS.use_full_inf:
        attention_keys = tf.stop_gradient(encoded["complete"])
      else:
        attention_keys = tf.stop_gradient(encoded["future"])
    else:
      attention_keys = None
    return attention_keys

  def _setup_options(self, is_training):
    params = AttrDict()

    params["action_conditioned_prediction"] = tf.flags.FLAGS.action_conditioned_prediction

    return params

  @staticmethod
  def _swap_high_level_latents(gt_latents, dts):
    # pick high level latent based on max dt
    num_segs, batch_size, seg_length = dts.get_shape().as_list()
    max_dt_idxs = tf.argmax(dts, axis=-1) + 1
    keyframe_idxs = tf.cumsum(max_dt_idxs, axis=0)
    x_idxs = np.asarray(range(batch_size))
    gathered_latents = []
    for t in range(num_segs):
      gather_idxs = tf.stack([keyframe_idxs[t] - 1, x_idxs], axis=-1)
      gathered_latents_t = tf.gather_nd(gt_latents, gather_idxs)
      gathered_latents.append(gathered_latents_t)
    output = tf.stack(gathered_latents, axis=0)
    return output

  @staticmethod
  def _get_keyframe_idxs(gt_keyframe_mask, n_segments):
    gt_keyframe_mask = tf.transpose(gt_keyframe_mask, (1, 0, 2))
    if FLAGS.dataset_config_name == "top":  # force keyframe at end of TOP dataset sequences
      gt_keyframe_mask = tf.cast(gt_keyframe_mask, tf.float32)
      batch_size, seq_len, dim = gt_keyframe_mask.get_shape().as_list()
      gt_keyframe_mask += tf.concat((tf.zeros((batch_size, seq_len-1, dim)), tf.ones((batch_size, 1, dim))), axis=1)
    idxs = tf.cast(
      tf.map_fn(lambda x: tf.cast(tf.where(tf.not_equal(x, 0))[:n_segments], tf.float32),
                gt_keyframe_mask[:, :, 0])[..., 0], tf.int32)
    return idxs

  @staticmethod
  def _gen_rand_keyframe_idxs(batch_size, n_segments):
    rand_seg_lengths = tf.random_uniform((batch_size, n_segments), minval=FLAGS.min_pretrain_seg_len,
                                         maxval=FLAGS.max_pretrain_seg_len, dtype=tf.int32)
    rand_keyframe_idxs = tf.cumsum(rand_seg_lengths, axis=1)
    return rand_keyframe_idxs

  @staticmethod
  def _replace_hl_latents(gt_hl_latents, idxs, n_segments, segment_length, ll_inf_output=None):
    x_idxs = np.asarray(range(gt_hl_latents.get_shape().as_list()[1]))
    target_dt, target_latent, ll_inf_latents = [], [], []
    for segment in range(n_segments):
      dseg = idxs[:, segment] if segment == 0 else idxs[:, segment] - idxs[:, segment-1] - 1
      dt = tf.one_hot(dseg, depth=segment_length)
      target_dt.append(dt)
      gather_idxs = tf.stack([idxs[:, segment], x_idxs], axis=-1)
      target_latent.append(tf.gather_nd(gt_hl_latents, gather_idxs))
      if FLAGS.ll_svg:
        # pick low level inference rnn output at end of segment
        ll_inf_latents.append(tf.gather_nd(ll_inf_output, gather_idxs))
    target_dt = tf.stack(target_dt, axis=0)
    target_latent = tf.stack(target_latent, axis=0)
    if FLAGS.ll_svg:
      ll_inf_latents = tf.stack(ll_inf_latents, axis=0)
    return target_dt, target_latent, ll_inf_latents

  @staticmethod
  def _sample_segment_z(batch_size, inf_latent_dim, is_training, ll_inf_latents=None):
    """Samples from the inference latents during pre-training and from the prior otherwise."""
    def _sample(mu, std_dev):
      """Sample from parametrized Gaussian distribution."""
      z_dists = Normal(loc=mu, scale=std_dev)
      z = tf.squeeze(z_dists.sample([1]))  # sample one sample from each distribution
      return z

    inf_dist_dim = int(inf_latent_dim / 2)
    if FLAGS.pretrain_ll and is_training:
      mu, std_dev = ll_inf_latents[:, :inf_dist_dim], tf.exp(ll_inf_latents[:, inf_dist_dim:])
    else:
      # after pretraining we sample from a uniform prior
      mu, std_dev = tf.zeros((batch_size, inf_dist_dim), dtype=tf.float32), \
                      tf.ones((batch_size, inf_dist_dim), dtype=tf.float32)
    z = _sample(mu, std_dev)
    return z

  @staticmethod
  def _prep_reset_indices(keyframe_idxs):
    batch_size = shape(keyframe_idxs)[0]
    # transform to one-hot
    one_hot_idxs = tf.one_hot(keyframe_idxs, depth=th_utils.get_future_input_length(), axis=0)
    one_hot_idxs = tf.reduce_sum(one_hot_idxs, axis=-1)
    # shift by one to right to reset after keyframe has passed
    reset = tf.concat((tf.zeros((1, batch_size), dtype=one_hot_idxs.dtype), one_hot_idxs[:-1]), axis=0)
    return reset

  def _predict(self,
              input_data,
              n_frames_input,
              n_frames_predict,
              batch_size,
              is_training,
              params,
              encoded,
              input_images,
              predict_images,
              action_sequence,
              abs_action_sequence,
              z_sequence=None):
    """ Runs the prediction in the latent space

    :return:
    """

    predicted = AttrDict()

    n_loss_frames = th_utils.get_future_loss_length()
    n_segments = FLAGS.n_segments
    n_frames_segment = FLAGS.n_frames_segment

    with tf.name_scope("encoder_rnn"):
      encoder_rnn_init_state = self._encoder_rnn_init_state_getter(batch_size)
      encoder_rnn_output, encoder_rnn_final_state = self.encoder_rnn(
          input_seq=encoded["past"],
          initial_state=encoder_rnn_init_state)
      debug("Encoder RNN", "input", "encoded[past]", shape(encoded["past"]))
      debug("Encoder RNN", "output", "encoder_rnn_output", shape(encoder_rnn_output))
      if FLAGS.debug: print()

    with tf.name_scope("inference_rnn"):
      inference_rnn_init_state = self._inference_rnn_init_state_getter(batch_size)
      if FLAGS.decode_actions:
        action_seq_inf = action_sequence if FLAGS.use_full_inf else action_sequence[n_frames_input:]
      else:
        action_seq_inf = None
      inference_rnn_output, _ = self.inference_rnn(
          input_seq=encoded["complete"] if FLAGS.use_full_inf else encoded["future_complete"],
          initial_state=inference_rnn_init_state,
          additional_input_seq=action_seq_inf)
      debug("Inference RNN", "input", "encoded[future_complete]", shape(encoded["future_complete"]))
      debug("Inference RNN", "output", "inference_rnn_output", shape(inference_rnn_output))
      if FLAGS.debug: print()
      # import pdb; pdb.set_trace()

    with tf.name_scope("high_level_rnn"):
      # build initial input dict
      if FLAGS.goal_conditioned:
        goal_latent = utils.batchwise_gather(tensor=encoded["future"], idxs=input_data.goal_timestep, batch_dim=1)
        def LSTM_state_to_latent(state):
          tensor = tf.transpose(tf.convert_to_tensor(state), [2, 0, 1, 3])
          return utils.flatten_end(tensor)
        high_level_rnn_initial_state = self._high_level_rnn_init_state_getter(
          tf.concat([LSTM_state_to_latent(encoder_rnn_final_state), goal_latent], axis=-1))
        
        first_prior_dist = self.init_mlp(tf.concat([encoder_rnn_output[-1], goal_latent], axis=-1))
        if not FLAGS.goal_every_step:
          goal_latent = None
      else:
        high_level_rnn_initial_state = encoder_rnn_final_state
        first_prior_dist = encoder_rnn_output[-1]
        goal_latent = None
    
      high_level_initial_input = {"frame": encoded["past"][-1], "dt": None, "next_prior_dists": first_prior_dist,
                                  "prior_dists": None, "inference_dists": None, "z_sample": None,
                                  "attention_weights": None}
      if FLAGS.separate_attention_key:
        high_level_initial_input["attention_key"] = encoder_rnn_output[-1][:, :self.encoded_img_size]
        high_level_initial_input["next_prior_dists"] = encoder_rnn_output[-1][:, self.encoded_img_size:]

      if not FLAGS.handcrafted_attention:
        attention_idxs = None
      else:
        if FLAGS.use_full_inf:
          # separate idxs into one-hot vectors
          attention_idxs = self._get_keyframe_idxs(abs_action_sequence, n_segments)
          oh_keyframe_idxs = tf.one_hot(
            attention_idxs, depth=n_frames_input+th_utils.get_future_input_length()) # batch x segment x frame
        else:
          raise ValueError("Handcrafted attention only works with full inference")

      if FLAGS.supervise_attention_term > 0:
        if FLAGS.use_full_inf:
          # separate idxs into one-hot vectors
          keyframe_idxs = self._get_keyframe_idxs(abs_action_sequence, n_segments) + 1
          oh_keyframe_idxs = tf.one_hot(
            keyframe_idxs, depth=n_frames_input+th_utils.get_future_input_length()) # batch x segment x frame
        else:
          raise ValueError("Supervised attention only works with full inference")
        
      # Get predictions for all future times.
      high_level_rnn_output, _ = self.high_level_rnn(
          initial_input=high_level_initial_input,  # The inference latents are not passed here
          initial_state=high_level_rnn_initial_state,
          rollout_len=n_segments,
          inference_seq=inference_rnn_output,
          use_inference=is_training,
          inference_attention_keys=self.get_attention_keys(encoded),
          attention_idxs=attention_idxs,
          predict_dt=not FLAGS.predict_dt_low_level,
          z_sequence=z_sequence,
          goal_latent=goal_latent)
      for key, value in high_level_initial_input.items():
        debug("High-level RNN", "input", "high_level_rnn_input[{}]".format(key), "(None)" if value is None else shape(value))
      for key, value in high_level_rnn_output.items():
        debug("High-level RNN", "output", "high_level_rnn_output[{}]".format(key), shape(value))
      if FLAGS.debug: print()

      if FLAGS.static_dt:
        norm_static_dt = tf.nn.softmax(self.static_dt, axis=-1)
        tiled_static_dt = tf.tile(norm_static_dt, (1, batch_size, 1))
        high_level_rnn_output["dt"] = tiled_static_dt

      if FLAGS.fixed_dt > 0:
        shape_dt = shape(high_level_rnn_output["dt"])
        high_level_rnn_output["dt"] = tf.one_hot((FLAGS.fixed_dt - 1) * tf.ones(shape_dt[:-1], dtype=tf.int32),
                                                 FLAGS.n_frames_segment)

    if FLAGS.test_hl_latent_swap:
      if FLAGS.predict_dt_low_level:
        raise NotImplementedError("HL latent swap requires high level dt prediction.")
      else:
        high_level_rnn_output["frame"] = self._swap_high_level_latents(encoded["future_complete"],
                                                                       high_level_rnn_output["dt"])

    if FLAGS.train_hl_latent_swap:
      if FLAGS.predict_dt_low_level:
        raise NotImplementedError("HL latent swap requires high level dt prediction.")
      else:
        hl_swapped_frames = self._swap_high_level_latents(encoded["future_complete"],
                                                          high_level_rnn_output["dt"])

    if FLAGS.handcrafted_attention:
      high_level_rnn_output["attention_weights"] = tf.transpose(oh_keyframe_idxs, [1, 2, 0])

    if FLAGS.pretrain_ll:
      if FLAGS.min_pretrain_seg_len > 0:
        assert FLAGS.max_pretrain_seg_len >= FLAGS.min_pretrain_seg_len, "Max pretrain segment length must be " \
                                                                         "bigger than Min!"
        assert FLAGS.max_pretrain_seg_len <= FLAGS.n_frames_segment, "Max random pretrain segment length needs to " \
                                                                     "be smaller or equal to max segment length"
        pt_keyframe_idxs = self._gen_rand_keyframe_idxs(batch_size.value, n_segments)
      else:
        pt_keyframe_idxs = self._get_keyframe_idxs(
          abs_action_sequence[-th_utils.get_future_input_length():],
            n_segments)
    else:   # for non-pretraining the keyframe idxs are not used so returning dummy
      pt_keyframe_idxs = tf.zeros((batch_size.value, n_segments), dtype=tf.int32)

    if FLAGS.ll_svg:
        # run the low level inference network, reset when keyframe PASSED (shift reset by 1 to right)
        with tf.name_scope("ll_inf_rnn"):
          ll_inf_rnn_init_state = self._ll_inf_rnn_init_state_getter(batch_size)
          reset = self._prep_reset_indices(pt_keyframe_idxs)
          ll_inf_rnn_output, _ = self.ll_inf_rnn(
            input_seq=encoded["future_complete"],
            initial_state=ll_inf_rnn_init_state,
            reset=tf.cast(reset, dtype=tf.bool),
            init_state=ll_inf_rnn_init_state)

    if FLAGS.pretrain_ll:
      high_level_rnn_output["dt"], high_level_rnn_output["frame"], ll_inf_latents = \
        self._replace_hl_latents(encoded["future_complete"],
                                 pt_keyframe_idxs,
                                 n_segments,
                                 n_frames_segment,
                                 ll_inf_rnn_output if FLAGS.ll_svg else None)

    predicted["high_level_weights"], predicted["propagated_distributions"] = \
      th_utils.get_propagated_distributions(high_level_rnn_output["dt"], n_loss_frames)
    predicted["soft_latent_targets"] = th_utils.get_high_level_targets(
      predicted["high_level_weights"],
      predicted["propagated_distributions"],
      tf.stop_gradient(encoded["future"]))

    if FLAGS.train_hl_latent_soft_swap:
      hl_swapped_frames = predicted["soft_latent_targets"]

    low_level_rnn_output_list = []
    low_level_dt_list = []    # used to store dt if it is predicted by low-level network
    low_level_rnn_actions_list = []
    ll_init_inputs = []
    for t in range(n_segments):
      with tf.name_scope("low_level_rnn" + str(t)):
        # Get predictions for all future times.

        # construct initial_input
        if FLAGS.train_hl_latent_swap or FLAGS.train_hl_latent_soft_swap:
          keyframe_latents = hl_swapped_frames
        else:
          keyframe_latents = high_level_rnn_output["frame"]
        start_frame = tf.squeeze(encoded["past"][-1]) if t is 0 else keyframe_latents[t-1]
        end_frame = keyframe_latents[t]
        if not FLAGS.ll_mlp:
          ll_input = (start_frame, end_frame) if FLAGS.predict_dt_low_level else\
                     (start_frame, end_frame, high_level_rnn_output["dt"][t])
          low_level_initialiser_input = tf.concat(ll_input, axis=-1)
          if FLAGS.ll_svg:
            # concatenate segment z latent
            ll_inf_latent_dim = self._network_spec.low_level_inf_rnn_spec[-1].output_size
            z_ll = self._sample_segment_z(batch_size, ll_inf_latent_dim, is_training,
                                          ll_inf_latents[t] if FLAGS.pretrain_ll else None)
            low_level_initialiser_input = tf.concat((low_level_initialiser_input, z_ll), axis=-1)
          initial_state = self._low_level_rnn_init_state_getter(low_level_initialiser_input)
          ll_init_inputs.append(low_level_initialiser_input)

        def add_placeholder(input, size, flag):
          if flag:
            initial_placeholder = tf.constant(0, shape=[batch_size, size], dtype=start_frame.dtype)
            input = tf.concat((start_frame, initial_placeholder), axis=-1)
          return input

        initial_input = add_placeholder(start_frame, 1, FLAGS.predict_dt_low_level and not FLAGS.ll_mlp)
        initial_input = add_placeholder(initial_input, self._num_actions, FLAGS.decode_actions)

        if not FLAGS.ll_mlp:
          low_level_rnn_output, _ = self.low_level_rnn(
            initial_input=initial_input,
            initial_state=initial_state,
            rollout_len=n_frames_segment)
        else:
          low_level_rnn_output, _ = self.low_level_rnn(
            initial_input=initial_input,
            initial_state=None,
            goal=end_frame,
            dt=None if FLAGS.predict_dt_low_level else high_level_rnn_output["dt"][t],
            rollout_len=n_frames_segment)
          
        if FLAGS.decode_actions:
          low_level_rnn_output = low_level_rnn_output[..., :-self._num_actions]
          low_level_rnn_output_actions = low_level_rnn_output[..., -self._num_actions:]
          low_level_rnn_actions_list.append(low_level_rnn_output_actions)

        if FLAGS.predict_dt_low_level:
          low_level_rnn_output_list.append(low_level_rnn_output[:, :, :-1])
          logits = low_level_rnn_output[:, :, -1]
          low_level_dt_list.append(tf.transpose(tf.nn.softmax(logits / self._tau, axis=0)))   # softmax with temperature
        else:
          low_level_rnn_output_list.append(low_level_rnn_output)

        if FLAGS.activate_latents:
          low_level_rnn_output_list[-1] = tf.nn.tanh(low_level_rnn_output_list[-1])

    # Regress actions from the segment boundaries.
    if self._train_action_regressor:
      assert not FLAGS.ll_mlp, "Action regression only works if LSTM is used for low level, not with MLP!"
      regressed_actions = snt.BatchApply(self.action_discriminator)(tf.stop_gradient(tf.stack(ll_init_inputs, axis=0)))

    if FLAGS.predict_dt_low_level:
      high_level_rnn_output["dt"] = tf.stack(low_level_dt_list, axis=0)


    predicted["high_level_rnn_output_keyframe"] = high_level_rnn_output["frame"]
    predicted["high_level_rnn_output_dt"] = high_level_rnn_output["dt"]
    predicted["high_level_rnn_output_z_sample"] = high_level_rnn_output["z_sample"]
    predicted["inference_dists"] = high_level_rnn_output["inference_dists"]
    predicted["prior_dists"] = high_level_rnn_output["prior_dists"]
    predicted["high_level_rnn_output"] = high_level_rnn_output
    predicted["low_level_rnn_output_list"] = low_level_rnn_output_list
    if FLAGS.supervise_attention_term > 0 :
      predicted["oh_keyframe_idxs"] = oh_keyframe_idxs
    if self._train_action_regressor:
      predicted["regressed_actions"] = regressed_actions
    if FLAGS.train_hl_latent_swap:
      predicted["hl_swapped_frames"] = hl_swapped_frames
    if FLAGS.ll_svg and FLAGS.pretrain_ll:
      predicted["ll_inf_dists"] = ll_inf_latents
    if FLAGS.decode_actions:
      predicted.low_level_rnn_output_actions = tf.stack(low_level_rnn_actions_list, axis=0)

    return predicted

  def _decode_images(self,
                    input_data,
                    n_frames_input,
                    is_training,
                    params,
                    encoded,
                    predicted,
                    skips,
                    input_images,
                    predict_images):
    """ Decodes the latents into images and the associated processing

    :param n_frames_input:
    :param n_frames_predict:
    :param is_training:
    :param encoded:
    :param full_sequence:
    :return:
    """

    decoded = AttrDict()
    n_segments = FLAGS.n_segments

    # Decoding phase
    with tf.name_scope("image_decoder"):
      # Prediction
      if self._has_image_input:
        keyframe_enc = tf.expand_dims(tf.expand_dims(predicted["high_level_rnn_output_keyframe"], axis=-1), axis=-1)
        output = self._build_image_decoder(
          keyframe_enc,
          skips["predict"],
          is_training,
          decoder_phase="future",
          last_input_frame=input_images[-1],
          use_recursive_image=self._use_recursive_image,
          goal_img=input_data.goal_image)
        if input_data.goal_image is not None:
          decoded["decoded_keyframes"], goal_imgs_out = output
        else:
          decoded["decoded_keyframes"], goal_imgs_out = output, None
      else:
        assert self._render_fcn is not None, "For coord. based prediction we need a render function!"
        decoded["decoded_keycoords"] = snt.BatchApply(self.conv_decoder)(predicted["high_level_rnn_output_keyframe"],
                                                                         is_training)
        decoded["decoded_keyframes"] = tf.py_func(self._render_fcn, [decoded["decoded_keycoords"]], tf.float32)
        render_shape = decoded["decoded_keycoords"].get_shape().as_list()[:2] + self._render_shape
        decoded["decoded_keyframes"] = tf.reshape(decoded["decoded_keyframes"], render_shape)
        goal_imgs_out = None

    decoded_low_level_list, low_level_coord_list = [], []
    for t in range(n_segments):
      with tf.name_scope("image_decoder" + str(t)):
        # Get predictions for all future times.
        if self._has_image_input:
          # prep input by extending dims
          enc_seq_i = tf.expand_dims(tf.expand_dims(predicted["low_level_rnn_output_list"][t], axis=-1), axis=-1)
          decoded_low_level = self._build_image_decoder(
            enc_seq_i,
            skips["predict"],
            is_training,
            decoder_phase="future",
            last_input_frame=input_images[-1],
            use_recursive_image=self._use_recursive_image)
        else:
          low_level_coord = snt.BatchApply(self.conv_decoder)(predicted["low_level_rnn_output_list"][t],
                                                                         is_training)
          decoded_low_level = tf.py_func(self._render_fcn, [low_level_coord], tf.float32)
          render_shape = low_level_coord.get_shape().as_list()[:2] + self._render_shape
          decoded_low_level = tf.reshape(decoded_low_level, render_shape)
          low_level_coord_list.append(low_level_coord)
        decoded_low_level_list.append(decoded_low_level)

    decoded["low_level_frames"] = tf.stack(decoded_low_level_list, axis=0)
    decoded["low_level_coords"] = tf.stack(low_level_coord_list, axis=0) if low_level_coord_list else None
    decoded.goal_imgs_out = goal_imgs_out

    return decoded

  def _setup_output(self,
                    input_data,
                    n_frames_input,
                    action_sequence,
                    encoded,
                    predicted,
                    decoded,
                    is_training,
                    abs_action_sequence):
    """ Selects the data to output

    :return:
    """

    model_output = AttrDict()

    # Past latents: for reconstruction (true)
    model_output["decoded_low_level_frames"] = decoded["low_level_frames"]
    model_output["decoded_keyframes"] = decoded["decoded_keyframes"]
    if not self._has_image_input:
      model_output["decoded_low_level_coords"] = decoded["low_level_coords"]
      model_output["decoded_keycoords"] = decoded["decoded_keycoords"]

    model_output["high_level_rnn_output_keyframe"] = predicted["high_level_rnn_output_keyframe"]
    model_output["high_level_rnn_output_dt"] = predicted["high_level_rnn_output_dt"]
    model_output["high_level_rnn_output_z_sample"] = predicted["high_level_rnn_output_z_sample"]
    model_output["inference_dists"] = predicted["inference_dists"]
    model_output["prior_dists"] = predicted["prior_dists"]
    model_output["attention_weights"] = predicted["high_level_rnn_output"]["attention_weights"]
    model_output["low_level_rnn_output_list"] = predicted["low_level_rnn_output_list"]
    model_output["predicted"] = predicted

    if FLAGS.supervise_attention_term > 0:
      model_output["oh_keyframe_idxs"] = predicted["oh_keyframe_idxs"]

    if self._train_action_regressor:
      model_output["regressed_actions"] = predicted["regressed_actions"]

    model_output["encoded_future"] = encoded["future"]

    return model_output

  def _build(
      self,
      input_data,
      n_frames_input,
      n_frames_predict,
      is_training):
    """Adds the conv embedding LSTM into the graph.

    Args:
      full_sequence: The sequence of frames used as input to the network.
        A Tensor of shape [n_frames, batch_size, 1, im_height, im_width],
        where n_frames = n_frames_input + n_frames_predict.
      action_sequence: The sequence of actions that transition between frames.
      n_frames_input: The number of frames used as input before prediction is
        started.
      n_frames_predict: The number of frames to predict.
      is_training: True if running in training mode, False otherwise.
    Returns:
      model_output: A dictionary of output tensors.
    """
    # Set options
    batch_size = input_data.input_images.get_shape()[1]
    params = self._setup_options(is_training)
    params.update(batch_size=batch_size,
                  is_training=is_training)

    input_data = AttrDict(input_data)
    if "goal_timestep" not in input_data.keys():
      input_data.goal_timestep = tf.ones(batch_size, dtype=tf.int32) * (n_frames_predict - 1)
    else:
      input_data.goal_timestep = input_data.goal_timestep[0]  # remove first dimension
      input_data.goal_timestep = tf.Print(input_data.goal_timestep, [input_data.goal_timestep], "goal_timestep", summarize=20)
    if "goal_image" not in input_data.keys():
      input_data.goal_image = None
    
    self._setup_encdec_modules()
    self._setup_rnn_modules()

    # First, run the conv net over the full sequence.
    with tf.name_scope("image_encoder"):
      encoded = self._build_encoder(
        tf.concat([input_data.input_images, input_data.predict_images], axis=0), n_frames_input, n_frames_predict, is_training)

    # Run the LSTMs
    kwargs = {}
    if 'z_sequence' in input_data.keys():
        kwargs['z_sequence'] = input_data.z_sequence
    if 'infer_z_inputs' in input_data.keys():
        kwargs['infer_z_inputs'] = input_data.infer_z_inputs
        kwargs['infer_n_zs'] = input_data.infer_n_zs
    predicted = self._predict(
      input_data,
      n_frames_input,
      n_frames_predict,
      batch_size,
      is_training,
      params,
      encoded,
      input_data.input_images,
      input_data.predict_images,
      input_data.actions,
      input_data.actions_abs,
      **kwargs)

    # Assemble skip connections
    skips = self._build_skips(
        encoded["skips_past"],
        input_data,
        n_frames_input)

    # Decode images
    decoded = self._decode_images(
      input_data,
      n_frames_input,
      is_training,
      params,
      encoded,
      predicted,
      skips,
      input_data.input_images,
      input_data.predict_images)

    model_output = self._setup_output(
      input_data,
      n_frames_input,
      input_data.actions,
      encoded,
      predicted,
      decoded,
      is_training,
      input_data.actions_abs)

    return model_output