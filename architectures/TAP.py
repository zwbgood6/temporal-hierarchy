from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

from utils import  debug, shape
from architectures.sequence_architectures import TemporalHierarchyModel
from architectures import lstm, cores
from architectures import mlp_architectures, rnn_architectures
import utils
from utils import AttrDict

FLAGS = tf.flags.FLAGS


class TimeAgnosticPredictor(TemporalHierarchyModel):
  """An action conditioned single step predictor model that consists of 
     encoding, LSTM, and decoding modules, takes (image, action) sequences
     as input, and produces predicted frame sequences as output."""

  def __init__(self,
               *args,
               **kwargs):
    super(TimeAgnosticPredictor, self).__init__(*args, **kwargs)

  def _setup_network_specs(self):
  
    self.encoded_img_size = self._conv_encoder_spec.dcgan_latent_size
    
    self._network_spec.high_level_rnn_spec[-1].output_size = \
      self._network_spec.high_level_rnn_spec[-1].output_size + self.encoded_img_size

  def _setup_rnn_modules(self):
    with tf.variable_scope("initializer"):
      high_level_init_mlp = mlp_architectures.GeneralizedMLP(
        layers=self._network_spec.low_level_initialisor_spec.layers)
      self._high_level_rnn_init_state_getter = rnn_architectures.build_mlp_initializer(
        self._network_spec.high_level_rnn_spec,
        high_level_init_mlp)
      self.init_mlp = mlp_architectures.GeneralizedMLP(
        layers=self._network_spec.action_discriminator_spec.layers + [self.z_dist_size + self.encoded_img_size])
  
    with tf.variable_scope("inference_rnn"):
      inference_rnn_base_core, self._inference_rnn_init_state_getter = \
          self._build_rnn_core(self._network_spec.inference_rnn_spec)
      self.inference_rnn_core = cores.SimpleCore(inference_rnn_base_core)
      self.inference_rnn = lstm.CustomLSTM(
          core=self.inference_rnn_core,
          backwards=FLAGS.inference_backwards)

    with tf.variable_scope("high_level_rnn"):
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
  def _setup_options(self, is_training):
    params = super(TimeAgnosticPredictor, self)._setup_options(is_training)

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
    
    predicted = AttrDict()
    
    assert FLAGS.separate_attention_key

    with tf.name_scope("high_level_rnn"):
      # build initial input dict
      goal_latent = utils.batchwise_gather(tensor=encoded["future"], idxs=input_data.goal_timestep, batch_dim=1)
      first_input = tf.concat([encoded.past[-1], goal_latent], axis=-1)
      
      high_level_rnn_initial_state = self._high_level_rnn_init_state_getter(first_input)
    
      init_tensor = self.init_mlp(first_input)
      
      high_level_initial_input = AttrDict({"frame": encoded["past"][-1], "dt": None,
                                           "next_prior_dists": init_tensor[:, self.encoded_img_size:],
                                           "prior_dists": None, "inference_dists": None, "z_sample": None,
                                           "attention_weights": None})
      
      high_level_initial_input["attention_key"] = init_tensor[:, :self.encoded_img_size]
  
      # Get predictions for all future times.
      high_level_rnn_output, _ = self.high_level_rnn(
        initial_input=high_level_initial_input,  # The inference latents are not passed here
        initial_state=high_level_rnn_initial_state,
        rollout_len=FLAGS.n_segments,
        use_inference=is_training,
        inference_attention_keys=self.get_attention_keys(encoded),
        predict_dt=False,
        goal_latent=goal_latent)
      
      for key, value in high_level_initial_input.items():
        debug("High-level RNN", "input", "high_level_rnn_input[{}]".format(key),
              "(None)" if value is None else shape(value))
      for key, value in high_level_rnn_output.items():
        debug("High-level RNN", "output", "high_level_rnn_output[{}]".format(key), shape(value))
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
    model_output["predicted"] = predicted

    model_output["encoded_future"] = encoded["future"]

    return model_output
