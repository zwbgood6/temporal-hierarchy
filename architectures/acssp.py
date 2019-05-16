from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

from utils import  debug, shape, AttrDict
from architectures.sequence_architectures import TemporalHierarchyModel
from architectures import lstm, cores

FLAGS = tf.flags.FLAGS


class ActionConditionedSingleStepPredictor(TemporalHierarchyModel):
  """An action conditioned single step predictor model that consists of 
     encoding, LSTM, and decoding modules, takes (image, action) sequences
     as input, and produces predicted frame sequences as output."""

  def __init__(self,
               *args,
               **kwargs):
    super(ActionConditionedSingleStepPredictor, self).__init__(*args, **kwargs)

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
      self.encoder_rnn_core = cores.ActionConditionedCore(encoder_rnn_base_core)
      self.encoder_rnn = lstm.CustomLSTM(
          core=self.encoder_rnn_core)

      with tf.name_scope("decoder_rnn"):
        if self._share_past_future_rnn:
          self.decoder_rnn_core = self.encoder_rnn_core
        else:
          if FLAGS.stateless_predictor:
            decoder_rnn_base_core, _ = \
              self._build_stateless_core(self._network_spec.decoder_rnn_spec)
          else:
            decoder_rnn_base_core, _ = \
              self._build_rnn_core(self._network_spec.decoder_rnn_spec)
          self.decoder_rnn_core = cores.ActionConditionedCore(decoder_rnn_base_core)

        self.decoder_rnn = lstm.CustomLSTM(
          core=self.decoder_rnn_core)

  def _setup_options(self, is_training):
    params = super(ActionConditionedSingleStepPredictor, self)._setup_options(is_training)

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
               abs_action_sequence):
    """ Runs the prediction in the latent space. """
    
    predicted = dict()

    if shape(encoded["past"])[0] > 1:
      with tf.name_scope("encoder_rnn"):
        encoder_rnn_init_state = self._encoder_rnn_init_state_getter(batch_size)
        encoder_rnn_output, encoder_rnn_final_state = self.encoder_rnn(
            input_seq=encoded["past"][:-1],
            initial_state=encoder_rnn_init_state,
            actions=action_sequence[:n_frames_input])
        debug("Encoder RNN", "input", "encoded[past]", shape(encoded["past"]))
        debug("Encoder RNN", "output", "encoder_rnn_output", shape(encoder_rnn_output))
        if FLAGS.debug: print()
      predicted["seq_past"] = encoder_rnn_output
    else:   # if we condition only on one image we do not need a separate encoding RNN
      encoder_rnn_final_state = self._encoder_rnn_init_state_getter(batch_size)
      predicted["seq_past"] = None

    with tf.name_scope("decoder_rnn"):
      decoder_rnn_output, _ = self.decoder_rnn(
        initial_input=encoded["past"][-1],
        initial_state=encoder_rnn_final_state,
        actions=action_sequence[n_frames_input:],
        rollout_len=n_frames_predict)
      debug("Decoder RNN", "output", "decoder_rnn_output", shape(decoder_rnn_output))
      if FLAGS.debug: print()

    predicted["seq_future"] = decoder_rnn_output

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
    decoded = AttrDict()

    # Decoding phase
    with tf.name_scope("image_decoder"):

      if self._has_image_input:
        pred_enc = tf.expand_dims(tf.expand_dims(predicted["seq_future"], axis=-1), axis=-1)
        output = self._build_image_decoder(
          pred_enc,
          skips["predict"],
          is_training,
          decoder_phase="future",
          last_input_frame=input_images[-1],
          use_recursive_image=self._use_recursive_image,
          goal_img=input_data.goal_image)
        if input_data.goal_image is not None:
          decoded_frames, goal_imgs_out = output
        else:
          decoded_frames, goal_imgs_out = output, None
      else:
        pred_coord = snt.BatchApply(self.conv_decoder)(predicted["seq_future"],
                                                           is_training)
        decoded_frames = tf.py_func(self._render_fcn, [pred_coord], tf.float32)
        render_shape = shape(pred_coord)[:2] + self._render_shape
        decoded_frames = tf.reshape(decoded_frames, render_shape)

    decoded["pred_frames"] = decoded_frames
    decoded["pred_coords"] = pred_coord if not self._has_image_input else None
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
    model_output = AttrDict()

    # Past latents: for reconstruction (true)
    model_output["decoded_low_level_frames"] = decoded["pred_frames"]
    if not self._has_image_input:
      model_output["decoded_low_level_coords"] = decoded["pred_coords"]
    model_output.decoded = decoded

    model_output["encoded_future"] = encoded["future"]

    return model_output
