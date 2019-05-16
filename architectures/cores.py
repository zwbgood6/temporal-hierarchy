import tensorflow as tf
from sonnet.python.modules import gated_rnn
from tensorflow.contrib.distributions import Normal
from utils import normalize, shape
import utils


class Core(object):
  def __init__(self,
               base_core):
    """
    Initializes the LSTM core.
    :param base_core: The Sonnet LSTM Core object.
    """
    self._base_core = base_core

  def run(self,
          input,
          state):
    """
    Executes one pass of the LSTM core.
    :param input: Input of the last step or initializer.
    :param state: LSTM core state.
    :return:
    """
    raise NotImplementedError("The run function of the Core class needs to be implemented!")
  
  @staticmethod
  def append(input, tensor, step=None):
    if tensor is not None:
      if step is not None:
        input = tf.concat((input, tensor[step]), axis=-1)
      else:
        input = tf.concat((input, tensor), axis=-1)
      
    return input



class SimpleCore(Core):
  """Implements a normal deterministic LSTM core."""
  def __init__(self,
               base_core):
    super(SimpleCore, self).__init__(base_core)

  def run(self,
          input,
          state,
          step=None):
    return self._base_core(input, state)


class StateLessInterpolatorCore(Core):
  """Implements a state-less RNN core based on a generic MLP."""
  def __init__(self,
               base_core):
    super(StateLessInterpolatorCore, self).__init__(base_core)

  def run(self,
          input,
          state,
          goal,
          dt,
          step=None):
    if state is not None:
      raise NotImplementedError("The state-less RNN core does not have a state by definition. "
                                "Received state was not None!")
    mlp_input = tf.concat((input, goal), axis=-1)   # concat input frame and goal frame encoding
    if dt is not None:
      # shift dt by step to the left to account for already predicted time steps
      dt_shifted = tf.pad(dt[:, step:], [[0, 0], [0, step]])
      mlp_input = tf.concat((mlp_input, dt_shifted), axis=-1)
    mlp_output = self._base_core(mlp_input)
    return mlp_output, None   # returned state is None


class SVGCore(Core):
  def __init__(self,
               base_core):
    """
    Implements SVG with stochastic sampling from learned distribution.
    :param base_core: The Sonnet LSTM Core object.
    """
    super(SVGCore, self).__init__(base_core)

  def _sample(self,
              mu,
              std_dev):
    """
    Sample from parametrized Gaussian distribution.
    :param mu: Gaussian mean.
    :param std_dev: Standard deviation of the Gaussian.
    :return: Sample z.
    """
    z_dists = Normal(loc=mu, scale=std_dev)
    z = tf.squeeze(z_dists.sample([1]))  # sample one sample from each distribution
    return z

  def _format_output(self,
                     lstm_output,
                     prior_dists,
                     inference_dists,
                     z_sample):
    """
    Reformat single LSTM output latent into dict of outputs.
    :param lstm_output: LSTM output vector.
    :return: List of next frame encoding, dt and predicted prior.
    """
    next_frame_enc = lstm_output
    if tf.flags.FLAGS.activate_latents:
      next_frame_enc = tf.tanh(next_frame_enc)
    output = {"frame": next_frame_enc,
              "prior_dists": prior_dists, "inference_dists": inference_dists, "z_sample": z_sample}
    return output


  def run(self,
          input,
          state,
          inference_seq,
          use_inference,
          actions=None,
          step=None,
          z_sequence=None):
    """

    :param input:
    :param state:
    :param inference_seq: passed via kwargs, non-optional
    :param use_inference: passed via kwargs, non-optional
    :param inference_attention_keys:
    :param attention_idxs:
    :param predict_dt:
    :param step: Step index in the LSTM exection.
    :return:
    """
    current_frame_enc = tf.squeeze(input["frame"])

    batch_size, prior_latent_size = inference_seq.get_shape().as_list()[-2:]
    dist_dim = int(prior_latent_size / 2)
    inference_dists = inference_seq[step]
    prior_dists = tf.zeros((batch_size, prior_latent_size))
    if use_inference:
      mu, std_dev = inference_dists[:, :dist_dim], tf.exp(inference_dists[:, dist_dim:])
    else:
      mu, std_dev = prior_dists[:, :dist_dim], tf.exp(prior_dists[:, dist_dim:])

    z = self._sample(mu, std_dev) if z_sequence is None else z_sequence[step]

    lstm_input = tf.concat((z, current_frame_enc), axis=1)
    if actions is not None:
      lstm_input = self.append(lstm_input, actions, step)

    lstm_output, new_state = self._base_core(lstm_input, state)
    output = self._format_output(lstm_output, prior_dists, inference_dists, z)
    return output, new_state


class KeyInCore(Core):
  def __init__(self,
               base_core,
               img_enc_size,
               dt_size,
               prior_latent_size,
               tau):
    """
    Implements SVG with stochastic sampling from learned distribution.
    :param base_core: The Sonnet LSTM Core object.
    """
    super(KeyInCore, self).__init__(base_core)
    self._img_enc_size = img_enc_size
    self._dt_size = dt_size
    self._prior_latent_size = prior_latent_size
    self._tau = tau

  def _attend_inference(self,
                        key,
                        inference_seq,
                        inference_attention_keys=None,
                        attention_idxs=None,
                        step=None):
    """
    Use key-based attention to retrieve inference output.
    :param key: key which the attention weight will be computed against.
    :param inference_seq: Output of the inference network. size : seq_len x batch_size x inf_latent_size
    :param inference_attention_keys: (Optional) Encoding of GT frames used as keys for attention.
    :param attention_idxs: (Optional) indices of the desired attention output in the inference_seq.
     size: batch_size x n_segments
    :return: mu, std_dev; parameters of inference sampling distribution.
    """

    seq_len, batch_size, inf_latent_size = inference_seq.get_shape().as_list()
    if inference_attention_keys is not None:
      # use GT encodings as keys and full inference output as message
      inf_keys = inference_attention_keys
      inf_dists = inference_seq
    else:
      inf_keys = inference_seq[:, :, -self._img_enc_size:]  # use fist part of inference output as key
      if tf.flags.FLAGS.activate_latents and not tf.flags.FLAGS.separate_attention_key:
        inf_keys = tf.tanh(inf_keys)
      inf_dists = inference_seq[:, :, :-self._img_enc_size]

    if attention_idxs is None:
      if tf.flags.FLAGS.separate_attention_key:
        # Cosine distance
        inner_key_product = tf.reduce_sum(tf.multiply(normalize(inf_keys), normalize(key)[None]), axis=-1)
      else:
        # Dot product
        inner_key_product = tf.reduce_sum(tf.multiply(inf_keys, key[None]), axis=-1)
      attention_weights = tf.nn.softmax(inner_key_product, axis=0)

      weighted_inference_dists = tf.multiply(attention_weights[:, :, None], inf_dists)
      output_inf_dist = tf.reduce_sum(weighted_inference_dists, axis=0)
    else:
      output_inf_dist = tf.batch_gather(tf.transpose(inf_dists, [1, 0, 2]), attention_idxs[:, step:step + 1])[:, 0]
      attention_weights = tf.zeros((seq_len, batch_size))

    return output_inf_dist, attention_weights

  def _sample(self,
              mu,
              std_dev):
    """
    Sample from parametrized Gaussian distribution.
    :param mu: Gaussian mean.
    :param std_dev: Standard deviation of the Gaussian.
    :return: Sample z.
    """
    z_dists = Normal(loc=mu, scale=std_dev)
    z = tf.squeeze(z_dists.sample([1]))  # sample one sample from each distribution
    return z

  def _format_output(self,
                     lstm_output,
                     prior_dists,
                     inference_dists,
                     z_sample,
                     predict_dt,
                     attention_weights):
    """
    Reformat single LSTM output latent into dict of outputs.
    :param lstm_output: LSTM output vector.
    :return: List of next frame encoding, dt and predicted prior.
    """
    next_frame_enc = lstm_output[:, :self._img_enc_size]
    if tf.flags.FLAGS.activate_latents:
      next_frame_enc = tf.tanh(next_frame_enc)
    if predict_dt:
      logits = lstm_output[:, self._img_enc_size:(self._img_enc_size + self._dt_size)]
      dt = tf.nn.softmax(logits / self._tau)  # softmax with temperature
    else:
      dt = tf.constant(-1)  # dummy value
    next_prior_latent = lstm_output[:, -self._prior_latent_size:]
    output = {"frame": next_frame_enc, "dt": dt, "next_prior_dists": next_prior_latent,
              "prior_dists": prior_dists, "inference_dists": inference_dists, "z_sample": z_sample,
              "attention_weights": attention_weights}
    if tf.flags.FLAGS.separate_attention_key:
      output["attention_key"] = lstm_output[:, -self._img_enc_size:]
    return output

  def run(self,
          input,
          state,
          inference_seq,
          use_inference,
          inference_attention_keys=None,
          attention_idxs=None,
          predict_dt=True,
          step=None,
          z_sequence=None,
          goal_latent=None):
    """

    :param input:
    :param state:
    :param inference_seq: passed via kwargs, non-optional
    :param use_inference: passed via kwargs, non-optional
    :param inference_attention_keys:
    :param attention_idxs:
    :param predict_dt:
    :param step: Step index in the LSTM exection.
    :return:
    """
    current_frame_enc, learned_prior = tf.squeeze(input["frame"]), input["next_prior_dists"]
    if tf.flags.FLAGS.hl_learned_prior:
      prior = learned_prior
    else:
      prior = utils.get_fixed_prior(shape(learned_prior))

    inference_dists, attention_weights = self._attend_inference(
      current_frame_enc if not tf.flags.FLAGS.separate_attention_key else input["attention_key"],
      inference_seq, inference_attention_keys, attention_idxs, step)
    dist_dim = int(self._prior_latent_size / 2)
    if use_inference:
      mu, std_dev = inference_dists[:, :dist_dim], tf.exp(inference_dists[:, dist_dim:])
    else:
      mu, std_dev = prior[:, :dist_dim], tf.exp(prior[:, dist_dim:])

    z = self._sample(mu, std_dev) if z_sequence is None else z_sequence[step]

    lstm_input = self.append(z, current_frame_enc)
    lstm_input = self.append(lstm_input, goal_latent)
    lstm_output, new_state = self._base_core(lstm_input, state)
    output = self._format_output(lstm_output, prior, inference_dists, z, predict_dt, attention_weights)

    return output, new_state


class ResetCore(Core):
  """Implements a deterministic LSTM core that resets the internal state when given a flag."""
  def __init__(self,
               base_core):
    super(ResetCore, self).__init__(base_core)

  def run(self,
          input,
          state,
          reset,
          init_state,
          step=None):
    # reset all states where reset == True
    mod_state = []
    for lstm_mod_idx in range(len(state)):
      mod_cell = tf.where(reset[step], init_state[lstm_mod_idx].cell, state[lstm_mod_idx].cell)
      mod_hidden = tf.where(reset[step], init_state[lstm_mod_idx].hidden, state[lstm_mod_idx].hidden)
      mod_state.append(gated_rnn.LSTMState(hidden=mod_hidden,
                                           cell=mod_cell))
    return self._base_core(input, mod_state)

class ActionConditionedCore(Core):
  def __init__(self, base_core):
    super(ActionConditionedCore, self).__init__(base_core)

  def run(self,
          input,
          state,
          actions,
          step=None):
    input = self.append(input, actions, step)
    return self._base_core(input, state)
