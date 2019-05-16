import tensorflow as tf
import sonnet as snt

class CustomLSTM(snt.AbstractModule):
  def __init__(self,
               core,
               backwards=False):
    """
    Initialize LSTM.
    :param core: contains LSTMCore class object.
    :param backwards: If True: run over sequence back to front.
    """
    super(CustomLSTM, self).__init__(name="custom_lstm")
    self._core = core
    self._backwards = backwards

  def _stack_outputs(self, output_seq):
    """Stacks list elements into tensor."""
    def _stack(tensor):
      output = tf.stack(tensor)
      if self._backwards:
        output = tf.reverse(output, axis=[0])
      return output
    
    if isinstance(output_seq, dict):
      output = dict.fromkeys(output_seq.keys())
      for key in output_seq.keys():
        output[key] = _stack(output_seq[key])
    else:
      output = _stack(output_seq)
      
    return output

  def _build(self,
             initial_state,
             initial_input=None,
             rollout_len=None,
             input_seq=None,
             additional_input_seq=None,
             **kwargs):
    """
    Constructs LSTM for loop around core (either autoregressive or with teacher forcing)
    :param init_state: initial LSTM state.
    :param initial_input: (optional) first input (if not autoregressive, otherwise first seq input is used)
    :param rollout_len: (optional) length of autoregressive rollout (if only first input is given)
    :param input_seq: (optional) input seq (otherwise autoregressive)
    :param kwargs: (optional) additional params passed to core.run()
    :return: LSTM output seq
    """
    if initial_input is not None and input_seq is not None:
      raise ValueError("No initial input needed if input sequence is given!")
    if rollout_len is not None and input_seq is not None:
      raise ValueError("No rollout length is needed if input seq is given, propagates over whole seq!")

    if input_seq is not None:
      rollout_len = input_seq.get_shape().as_list()[0]
    output = initial_input
    state = initial_state
    is_output_dict = True if isinstance(initial_input, dict) else False
    if is_output_dict:
      output_seq = dict.fromkeys(initial_input.keys())
      for key in output_seq.keys():
        output_seq[key] = []
    else:
      output_seq = []
    step_list = range(rollout_len) if not self._backwards else range(rollout_len)[::-1]
    for step_idx in step_list:
      input = output if input_seq is None else input_seq[step_idx]
      if additional_input_seq is not None:
        input = tf.concat([input, additional_input_seq[step_idx]], axis=-1)
      kwargs['step'] = step_idx
      output, state = self._core.run(input, state, **kwargs)
      if is_output_dict:
        for key in output.keys():
          output_seq[key].append(output[key])
      else:
        output_seq.append(output)
    output_seq = self._stack_outputs(output_seq)
    return output_seq, state
