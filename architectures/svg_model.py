"""Architectures for sequence prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from architectures import arch_utils, conv_architectures, rnn_architectures, discriminators
import numpy as np
import sonnet as snt
from specs import module_specs
import tensorflow as tf
from utils import maybe_stop_gradients, pick_and_tile, make_dists_and_sample
from architectures.sequence_architectures import *


class SVG(TemporalHierarchyModel):
  """An autoencoding + prediction architecture. encoding + LSTM + decoding."""

  def __init__(self,
               variational,
               teacher_forcing=True,
               fixed_prior=False,
               *args):
    """Constructs a ConvEmbeddingLSTM.

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

    super(SVG, self).__init__(*args)

    # Variational
    self._variational = variational
    self._fixed_prior = fixed_prior
    self._comp_action_prior = tf.flags.FLAGS.comp_action_prior
    self._produce_two_seqs = self._comp_action_prior # If True, at both training and validation ..
    # two sequences - one sampled from prior and one from inference networks are produced
    self._use_cdna = tf.flags.FLAGS.use_cdna_model
    
    self._teacher_forcing = teacher_forcing

  def _build_rnn_core(self, spec):
    rnn_core, rnn_init_state_getter = dict(), dict()
    for i, name in enumerate(['output', 'inference', 'prior']):
      rnn_core_i, rnn_init_state_getter_i = \
        rnn_architectures.build_rnn(
          spec[i],
          input_image_shape=self._input_image_shape,
          channels=self._channels)
      rnn_core[name] = rnn_core_i
      rnn_init_state_getter[name] = rnn_init_state_getter_i
    return rnn_core, rnn_init_state_getter

  def _setup_rnn_modules(self):
    # Past and future RNNs: cores, initial state getters, and runners
    with tf.name_scope("recurrent_encoder"):
      self.encoder_rnn_core, self._encoder_rnn_init_state_getter = \
          self._build_rnn_core(self._lstm_encoder_spec)
      self.recurrent_encoder = rnn_architectures.VariationalLSTMSeq2Seq(
        cores=self.encoder_rnn_core,
        use_conv_lstm=self._use_conv_lstm,
        data_format=self._data_format,
        fixed_prior=self._fixed_prior)

    with tf.name_scope("recurrent_predictor"):
      if self._share_past_future_rnn:
        self.decoder_rnn_core = self.encoder_rnn_core
      else:
        self.decoder_rnn_core, _ = self._build_rnn_core(
            self._lstm_decoder_spec)

      self.recurrent_predictor = rnn_architectures.VariationalLSTMSeq2Seq(
          cores=self.decoder_rnn_core,
          use_conv_lstm=self._use_conv_lstm,
          data_format=self._data_format,
          fixed_prior=self._fixed_prior)

  def _setup_options(self, is_training):
    params = super(SVG, self)._setup_options(is_training)

    # Whether we want to re-encode the image latents
    reencode = self._variational and self._teacher_forcing

    # This is a flag which is used for more complete visualization
    if self._produce_two_seqs:
      report_prior_as_main = False
    else:
      report_prior_as_main = False if is_training else True

    produce_prior_seq = (self._produce_two_seqs or not is_training)
  
    if tf.flags.FLAGS.action_conditioned_prediction:
      # Any of the variational functionality is not needed here except static decoding
      produce_prior_seq = False
      report_prior_as_main = False
  
    # TODO(karl): enable skip connections also for autoregression
    if reencode and self._use_recursive_skips:
      raise NotImplementedError("Skip connections are currently "
                                "not supported in the variational setup!")
  
    params["report_prior_as_main"] = report_prior_as_main
    params["produce_prior_seq"] = produce_prior_seq
    params["reencode"] = reencode

    return params
  
  def _predict(self,
              n_frames_input,
              n_frames_predict,
              batch_size,
              is_training,
              params,
              encoded,
              full_sequence,
              action_sequence,
              abs_action_sequence):
    """ Runs the prediction in the latent space
    
    :return:
    """

    predicted = dict()

    # Get predictions of all past times.
    with tf.name_scope("recurrent_encoder"):
      # in variational case feed last frame of past series to both networks
      # (because the predictor does not predict an image for the initial input)
      encoded_seq_past = tf.concat((encoded["past_rnn_inputs"],
                                              tf.expand_dims(encoded["future_rnn_inputs"], axis=0)),
                                             axis=0)
      self.encoder_rnn_init_state = dict()
      for _, name in enumerate(['output', 'inference', 'prior']):
        self.encoder_rnn_init_state[name] = \
          self._encoder_rnn_init_state_getter[name](batch_size)
      if n_frames_input > 1:
          encoder_output_collection, encoder_rnn_final_state = self.recurrent_encoder(
            initial_inputs=encoded_seq_past[0],
            initial_state=self.encoder_rnn_init_state,
            seq_len=n_frames_input - 1,   # last frame is given to future RNN, first as initial input
            is_training=is_training,
            input_embed_seq=encoded_seq_past[1:],
            init_inference=True,
            input_latent_samples_seq=action_sequence[0:n_frames_input-1]
              if params["action_conditioned_prediction"] else None,
            encoder_cnn=self.conv_encoder if self._use_cdna else None,
            first_image=encoded_seq_past[0])
          encoder_rnn_output = encoder_output_collection['output_sequence']
          inference_dist_dim = \
              int(encoder_output_collection['inference_dists'].get_shape().as_list()[-1] / 2)
      else:
          encoder_rnn_final_state = self.encoder_rnn_init_state
          encoder_rnn_output = None
          encoder_output_collection = None
      predicted["seq_past"] = encoder_rnn_output

    with tf.name_scope("recurrent_predictor"):
      # feed groundtruth latents for prediction during training only (no autoregression)
      encoded_seq_future = encoded["future_latents_true"]
      if params["produce_prior_seq"]:
        predictor_output_collection_pri, _ = self.recurrent_predictor(
          initial_inputs=encoded["future_rnn_inputs"],
          initial_state=encoder_rnn_final_state,
          seq_len=n_frames_predict,
          is_training=is_training,
          first_image=encoded_seq_past[0],
          input_embed_seq=None,
          autoregress=True,
          reencode=params["reencode"],
          encoder_cnn=self.conv_encoder,
          decoder_cnn=self.conv_decoder,
          image_activation=self._output_activation)
        predicted["seq_future_pri"] = predictor_output_collection_pri['output_sequence']
      predictor_output_collection, _ = self.recurrent_predictor(
          initial_inputs=encoded["future_rnn_inputs"],
          initial_state=encoder_rnn_final_state,
          seq_len=n_frames_predict,
          is_training=is_training,
          input_embed_seq=encoded_seq_future,
          input_latent_samples_seq=action_sequence[n_frames_input-1:-1] if params["action_conditioned_prediction"]
          else None,
          autoregress=not self._teacher_forcing,
          encoder_cnn=self.conv_encoder if self._use_cdna else None,
          first_image=encoded_seq_past[0])
      predicted["inference_dist_dim"] = \
            int(predictor_output_collection['inference_dists'].get_shape().as_list()[-1] / 2)
      predicted["seq_future"] = predictor_output_collection['output_sequence']

    if abs_action_sequence is not None and (tf.flags.FLAGS.train_abs_action_regressor or
                 tf.flags.FLAGS.train_action_regressor):
        if tf.flags.FLAGS.train_action_regressor:
            if n_frames_input > 1:
                img_seq = tf.squeeze(tf.concat([encoded_seq_past[1:],
                                                encoded_seq_future], axis=0))
            else:
                img_seq = tf.squeeze(encoded_seq_future)
        else:
            img_seq = tf.squeeze(tf.concat([encoded_seq_past,
                                            encoded_seq_future], axis=0))
        img_seq = tf.stop_gradient(img_seq)
        regressed_abs_actions = snt.BatchApply(self.abs_action_discriminator)(img_seq)
        if tf.flags.FLAGS.train_action_regressor:
            # scale output 0...2pi
            predicted["regressed_abs_actions"] = tf.constant(2*np.pi) * tf.sigmoid(regressed_abs_actions)

    predicted["predictor_output_collection"] = predictor_output_collection
    predicted["encoder_output_collection"] = encoder_output_collection

    return predicted

  def _decode_images(self,
                    n_frames_input,
                    n_frames_predict,
                    is_training,
                    params,
                    encoded,
                    predicted,
                    skips,
                    full_sequence,
                    abs_action_sequence):
    """ Decodes the latents into images and the associated processing
    
    :param n_frames_input:
    :param n_frames_predict:
    :param is_training:
    :param encoded:
    :param full_sequence:
    :return:
    """

    decoded = dict()
    
    if not self._use_cdna:
        # Assemble skip connections

        with tf.name_scope("image_decoder"):
          # Reconstruction
          decoded["seq_reconstruct"] = self._build_simple_image_decoder(
              encoded["seq_reconstruct"],
              skips["reconstruct"],
              is_training,
              decoder_phase="past")
          # Here: should match decoded_seq_reconstruct[1:]
          # Prediction in the past
          if n_frames_input > 1:
            decoded["seq_predict_past"] = self._build_simple_image_decoder(
                  predicted["seq_past"],
                  skips["past_rnn"],
                  is_training,
                  decoder_phase="past")

          # Prediction
          if not params["report_prior_as_main"]:
            decoded["seq_predict"] = self._build_image_decoder(
                predicted["seq_future"],
                skips["predict"],
                is_training,
                decoder_phase="future",
                last_input_frame=full_sequence[(n_frames_input - 1), ...],
                use_recursive_image=self._use_recursive_image)
          if params["produce_prior_seq"]:
            # Prediction sampled from the prior
            decoded["seq_predict_pri"] = self._build_image_decoder(
                predicted["seq_future_pri"],
                skips["predict"],
                is_training,
                decoder_phase="future",
                last_input_frame=full_sequence[(n_frames_input - 1), ...],
                use_recursive_image=self._use_recursive_image)
            if params["report_prior_as_main"]:
              decoded["seq_predict"] = decoded["seq_predict_pri"]

          if abs_action_sequence is not None \
              and (tf.flags.FLAGS.train_abs_action_regressor or
                     tf.flags.FLAGS.train_action_regressor) \
              and not is_training:
            if not params["report_prior_as_main"]:  # output is already decoded teacher forced seq
              decoded["seq_predict_past"] = decoded["seq_predict"]
            else:  # decode teacher forced sequence here
              decoded_seq_predict_est = self._build_image_decoder(
                predicted["seq_future"],
                skips["predict"],
                is_training,
                decoder_phase="future",
                last_input_frame=full_sequence[(n_frames_input - 1), ...],
                use_recursive_image=self._use_recursive_image)
            if n_frames_input > 1:
              est_img_seq = tf.concat([decoded["seq_predict_past"],
                                       decoded_seq_predict_est], axis=0)
            else:
              est_img_seq = decoded_seq_predict_est
            est_img_seq = tf.stop_gradient(est_img_seq)
            est_img_seq_reencode, _ = snt.BatchApply(self.conv_encoder)(est_img_seq, is_training)
            regressed_abs_actions_est = snt.BatchApply(self.abs_action_discriminator)(est_img_seq_reencode)
            if tf.flags.FLAGS.train_action_regressor:
              # scale output 0...2pi
              decoded["regressed_abs_actions_est"] = tf.constant(2 * np.pi) * tf.sigmoid(regressed_abs_actions_est)

          # infer angles for all predicted sequences
          if tf.flags.FLAGS.train_abs_action_regressor or \
              tf.flags.FLAGS.train_action_regressor:
            decoded["aa_future_seq_true"] = self.infer_action_from_img([], is_training,
                                                            latent_seq=encoded["future_latents_true"])
            decoded["aa_decoded_seq_predict"] = self.infer_action_from_img(decoded["seq_predict"], is_training)
            if n_frames_input > 1:
              decoded["aa_decoded_seq_reconstruct_est"] = self.infer_action_from_img(decoded["seq_predict_past"], is_training)
    else:
        # if use_cdna
        decoded["seq_reconstruct"] = encoded["seq_reconstruct"]
        if n_frames_input > 1:
          decoded["seq_predict_past"] = predicted["seq_past"]
        decoded["seq_predict"] = predicted["seq_future"]
      
    return decoded

  def _setup_output(self,
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

    model_output = super(SVG, self)._setup_output(
      n_frames_input,
      action_sequence,
      encoded,
      predicted,
      decoded,
      is_training,
      abs_action_sequence)

    if self._produce_two_seqs:
      model_output["decoded_seq_predict_pri"] = decoded["seq_predict_pri"]

    model_output["inference_z_samples"] = tf.concat([predicted["encoder_output_collection"]["z_samples"],
                                                     predicted["predictor_output_collection"]["z_samples"]],
                                                    axis=0)
    model_output["inference_z_means"] = tf.concat([
      predicted["encoder_output_collection"]["inference_dists"][..., :predicted["inference_dist_dim"]],
      predicted["predictor_output_collection"]["inference_dists"][..., :predicted["inference_dist_dim"]]],
      axis=0)
    model_output["inference_z_stds"] = tf.concat([
      predicted["encoder_output_collection"]["inference_dists"][..., predicted["inference_dist_dim"]:],
      predicted["predictor_output_collection"]["inference_dists"][..., predicted["inference_dist_dim"]:]],
      axis=0)
    model_output["inference_dists_encoder"] = predicted["encoder_output_collection"]["inference_dists"]
    model_output["prior_dists_encoder"] = predicted["encoder_output_collection"]["prior_dists"]
    model_output["inference_dists_predictor"] = predicted["predictor_output_collection"]["inference_dists"]
    model_output["prior_dists_predictor"] = predicted["predictor_output_collection"]["prior_dists"]
    if not is_training:
      model_output["prior_dists_predictor_tf"] = \
        predicted["predictor_output_collection"]["prior_dists"]

    return model_output
