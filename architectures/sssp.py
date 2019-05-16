from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

from utils import  debug, shape, Gaussian
from architectures.sequence_architectures import TemporalHierarchyModel
from architectures import lstm, cores, mlp_architectures

FLAGS = tf.flags.FLAGS


class StochasticSingleStepPredictor(TemporalHierarchyModel):
  """An SVG-based single step predictor model that consists of
     encoding, LSTM, and decoding modules, takes (image, (z_latent)) sequences
     as input, and produces predicted frame sequences + inferred actions as output."""

  def __init__(self,
               *args,
               **kwargs):
    super(StochasticSingleStepPredictor, self).__init__(*args, **kwargs)
    self._action_conditioned = not self._train_action_regressor    # network can operate with or without action cond.

  def _setup_network_specs(self):
    pass

  def _setup_rnn_modules(self):
    with tf.variable_scope("encoder_rnn"):
      # TODO make the getters nicer (put them inside CustomLSTM?)
      if FLAGS.stateless_predictor:
        encoder_rnn_base_core, self._encoder_rnn_init_state_getter = \
          self._build_stateless_core(self._network_spec.encoder_rnn_spec)
      else:
        encoder_rnn_base_core, self._encoder_rnn_init_state_getter = \
            self._build_rnn_core(self._network_spec.encoder_rnn_spec)
      if self._action_conditioned:
        self.encoder_rnn_core = cores.ActionConditionedCore(encoder_rnn_base_core)
      else:
        self.encoder_rnn_core = cores.SimpleCore(encoder_rnn_base_core)
      self.encoder_rnn = lstm.CustomLSTM(
          core=self.encoder_rnn_core)

    with tf.variable_scope("inference_rnn"):
      if FLAGS.stateless_predictor:
        inference_rnn_base_core, self._inference_rnn_init_state_getter = \
          self._build_stateless_core(self._network_spec.inference_rnn_spec)
      else:
        inference_rnn_base_core, self._inference_rnn_init_state_getter = \
          self._build_rnn_core(self._network_spec.inference_rnn_spec)
      self.inference_rnn_core = cores.SimpleCore(inference_rnn_base_core)
      self.inference_rnn = lstm.CustomLSTM(
        core=self.inference_rnn_core,
        backwards=False)

    with tf.variable_scope("stochastic_decoder_rnn"):
      if FLAGS.stateless_predictor:
        decoder_rnn_base_core, _ = \
          self._build_stateless_core(self._network_spec.decoder_rnn_spec)
      else:
        decoder_rnn_base_core, _ = \
          self._build_rnn_core(self._network_spec.decoder_rnn_spec)
      self.decoder_rnn_core = cores.SVGCore(decoder_rnn_base_core)
      self.decoder_rnn = lstm.CustomLSTM(
        core=self.decoder_rnn_core)

    if not self._action_conditioned:
        with tf.variable_scope("z_action_regressor"):
            # add number of actions as number of output neurons of action discriminator
            action_discr_layers = self._action_discriminator_spec.layers + [self._num_actions]
            self.z_action_discriminator = mlp_architectures.GeneralizedMLP(layers=action_discr_layers)

  def _setup_options(self, is_training):
    params = super(StochasticSingleStepPredictor, self)._setup_options(is_training)

    # TODO: look into what should be filled in here.

    return params

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
               z_sequence=None,
               infer_z_inputs=None,
               infer_n_zs=None):
    """ Runs the prediction in the latent space. """
    
    predicted = dict()

    if shape(encoded["past"])[0] > 1:
      with tf.name_scope("encoder_rnn"):
        encoder_rnn_init_state = self._encoder_rnn_init_state_getter(batch_size)
        encoder_kwargs = {"actions": action_sequence[:n_frames_input]} if self._action_conditioned else {}
        encoder_rnn_output, encoder_rnn_final_state = self.encoder_rnn(
            input_seq=encoded["past"][:-1],
            initial_state=encoder_rnn_init_state,
            **encoder_kwargs)
        debug("Encoder RNN", "input", "encoded[past]", shape(encoded["past"]))
        debug("Encoder RNN", "output", "encoder_rnn_output", shape(encoder_rnn_output))
        if FLAGS.debug: print()
      predicted["seq_past"] = encoder_rnn_output
    else:   # if we condition only on one image we do not need a separate encoding RNN
      encoder_rnn_final_state = self._encoder_rnn_init_state_getter(batch_size)
      predicted["seq_past"] = None

    with tf.name_scope("inference_rnn"):
      inference_rnn_init_state = self._inference_rnn_init_state_getter(batch_size)
      inference_rnn_output, _ = self.inference_rnn(
          input_seq=tf.concat((encoded["past"][-1:], encoded["future_complete"]), axis=0),
          initial_state=inference_rnn_init_state)
      debug("Inference RNN", "input", "encoded[future_complete]", shape(encoded["future_complete"]))
      debug("Inference RNN", "output", "inference_rnn_output", shape(inference_rnn_output))
      if FLAGS.debug: print()

    if infer_z_inputs is not None:
      # infer the first N latents of the z_sequence
      with tf.name_scope("infer_initial_zs"):
        encoded_z_inputs, _ = snt.BatchApply(self.conv_encoder)(infer_z_inputs, is_training)
        encoded_z_inputs = encoded_z_inputs[:,:,:,0,0]
        inference_rnn_output, _ = self.inference_rnn(
          input_seq=encoded_z_inputs,
          initial_state=inference_rnn_init_state)
        z_dim = shape(z_sequence)[-1]
        z_sequence = tf.where(tf.cast(infer_n_zs, tf.bool), inference_rnn_output[1:, :, :z_dim], z_sequence)

    with tf.name_scope("decoder_rnn"):
      decoder_rnn_initial_input = {"frame": encoded["past"][-1],
                                  "prior_dists": None, "inference_dists": None, "z_sample": None}
      encoder_kwargs = {"actions": action_sequence[n_frames_input:]} if self._action_conditioned else {}
      decoder_rnn_output, _ = self.decoder_rnn(
        initial_input=decoder_rnn_initial_input,
        initial_state=encoder_rnn_final_state,
        inference_seq=inference_rnn_output[1:],   # shift by one to get output of next frame
        use_inference=is_training,
        rollout_len=n_frames_predict,
        z_sequence=z_sequence,
        **encoder_kwargs)
      predicted_frames = decoder_rnn_output["frame"]
      debug("Decoder RNN", "output", "decoder_rnn_output", shape(predicted_frames))
      if FLAGS.debug: print()

    if not self._action_conditioned:
      with tf.name_scope("action_regression"):
        assert FLAGS.train_action_regressor, "Need action regressor flag = True for SSSP!"
        act_reg_input = tf.concat((tf.concat((encoded["past"][-1:], predicted_frames[:-1]), axis=0),
                                   predicted_frames), axis=-1)
        regressed_actions = snt.BatchApply(self.action_discriminator)(tf.stop_gradient(act_reg_input))

      with tf.name_scope("z_action_regression"):
        act_reg_input = tf.concat((tf.concat((encoded["past"][-1:], predicted_frames[:-1]), axis=0),
                                   decoder_rnn_output["z_sample"]), axis=-1)
        regressed_actions_z = snt.BatchApply(self.z_action_discriminator)(tf.stop_gradient(act_reg_input))
      predicted["regressed_actions"] = regressed_actions
      predicted["regressed_actions_z"] = regressed_actions_z

    predicted["seq_future"] = predicted_frames
    predicted["z_sample"] = decoder_rnn_output["z_sample"]
    predicted["inference_dists"] = decoder_rnn_output["inference_dists"]
    predicted["prior_dists"] = decoder_rnn_output["prior_dists"]
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
    decoded = dict()

    # Decoding phase
    with tf.name_scope("image_decoder"):

      if self._has_image_input:
        pred_enc = tf.expand_dims(tf.expand_dims(predicted["seq_future"], axis=-1), axis=-1)
        decoded_frames = self._build_image_decoder(
          pred_enc,
          skips["predict"],
          is_training,
          decoder_phase="future",
          last_input_frame=input_images[-1],
          use_recursive_image=self._use_recursive_image)
      else:
        pred_coord = snt.BatchApply(self.conv_decoder)(predicted["seq_future"],
                                                           is_training)
        decoded_frames = tf.py_func(self._render_fcn, [pred_coord], tf.float32)
        render_shape = shape(pred_coord)[:2] + self._render_shape
        decoded_frames = tf.reshape(decoded_frames, render_shape)

    decoded["pred_frames"] = decoded_frames
    decoded["pred_coords"] = pred_coord if not self._has_image_input else None
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
    model_output = {}

    # Past latents: for reconstruction (true)
    model_output["decoded_low_level_frames"] = decoded["pred_frames"]
    if not self._has_image_input:
      model_output["decoded_low_level_coords"] = decoded["pred_coords"]

    model_output["z_sample"] = predicted["z_sample"]
    model_output["inference_dists"] = predicted["inference_dists"]
    model_output["prior_dists"] = predicted["prior_dists"]

    if not self._action_conditioned:
      model_output["regressed_actions"] = predicted["regressed_actions"]
      model_output["regressed_actions_z"] = predicted["regressed_actions_z"]

    model_output["encoded_future"] = encoded["future"]
    return model_output


class StochasticSingleStepPredictorKFDetect(StochasticSingleStepPredictor):
  def __init__(self,
               *args,
               **kwargs):
    super(StochasticSingleStepPredictorKFDetect, self).__init__(*args, **kwargs)

  @staticmethod
  def get_high_kl_keyframes(frames, inference_dists, prior_dists):
    n_dim = int(shape(inference_dists)[-1] / 2)
    batch_size = shape(inference_dists)[1]
    n_kfs = FLAGS.n_segments
    kl = Gaussian(inference_dists[..., :n_dim], inference_dists[..., n_dim:]).kl_divergence(
      Gaussian(prior_dists[..., :n_dim], prior_dists[..., n_dim:]))
    kl = tf.reduce_sum(kl, axis=-1)

    # filter with maxima
    maxima = tf.logical_and(kl[1:-1] > kl[2:], kl[1:-1] > kl[:-2])
    max_shape = shape(maxima)[1:]
    mask = tf.concat([tf.zeros([1]+max_shape, dtype=maxima.dtype), maxima, tf.zeros([1]+max_shape, dtype=maxima.dtype)], axis=0)
    filtered_kl = kl * tf.cast(mask, dtype=kl.dtype)

    kf_idxs = tf.contrib.framework.argsort(filtered_kl, axis=0, direction='DESCENDING')[:n_kfs]
    kf_idxs = tf.contrib.framework.sort(kf_idxs, axis=0)
    kf_idxs_binary = tf.reduce_sum(tf.one_hot(kf_idxs, depth=shape(frames)[0], axis=0), axis=1)
    gather_idxs = tf.reshape(tf.stack((kf_idxs, tf.tile(tf.expand_dims(tf.range(batch_size), axis=0),
                                                        [n_kfs, 1])), axis=-1), (-1, 2))
    gathered_kfs = tf.gather_nd(frames, gather_idxs)
    gathered_kfs = tf.reshape(gathered_kfs, [n_kfs, batch_size] + shape(gathered_kfs)[1:])
    return gathered_kfs, kf_idxs_binary, kl

  def _setup_output(self,
                    input_data,
                    n_frames_input,
                    action_sequence,
                    encoded,
                    predicted,
                    decoded,
                    is_training,
                    abs_action_sequence):
    """ Runs the prediction in the latent space. """
    model_output = super(StochasticSingleStepPredictorKFDetect, self)._setup_output(input_data,
                                                                                    n_frames_input,
                                                                                    action_sequence,
                                                                                    encoded,
                                                                                    predicted,
                                                                                    decoded,
                                                                                    is_training,
                                                                                    abs_action_sequence)
    batch_size = shape(model_output["decoded_low_level_frames"])[1]

    # reencode generate frames and compute inference distributions (mostly for test time)
    if not self._has_image_input:
      reencoded_seq = snt.BatchApply(self.conv_encoder)(model_output["decoded_low_level_coords"], False)
    else:
      reencoded_seq, _ = snt.BatchApply(
          self.conv_encoder)(self._output_activation(model_output["decoded_low_level_frames"]), False)
    if len(shape(reencoded_seq)) == 5:
      reencoded_seq = reencoded_seq[:, :, :, 0, 0]
    with tf.name_scope("reinfer_rnn"):
      inference_rnn_init_state = self._inference_rnn_init_state_getter(batch_size)
      inference_rnn_output, _ = self.inference_rnn(
          input_seq=tf.concat((encoded["past"][-1:], reencoded_seq), axis=0),
          initial_state=inference_rnn_init_state)
    model_output["inference_dists_reencode"] = inference_rnn_output[1:]

    return model_output
