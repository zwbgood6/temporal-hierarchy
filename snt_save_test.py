"""A simple test for saving and restoring specific variables in Sonnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sonnet as snt
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def run_simple_model(restore, save_checkpoint, checkpoint_dir):
  input = tf.placeholder(dtype=tf.float32, shape=(1, 5))
  v1 = snt.Linear(name="v1", output_size=5)
  v2 = snt.Linear(name="v2", output_size=5)

  v1_only = v1(input)
  v2_only = v2(input)
  v1_and_v2 = v2(v1(input))

  # Add ops to save and restore only `v2` using the name "v2"
  # import pdb; pdb.set_trace()  # Need to grab weights and biases
  saver = tf.train.Saver({"v1/b": v1.b, "v1/w": v1.w})

  # class SaveAtEnd(tf.train.SessionRunHook):
  #   def begin(self):
  #     self._saver = tf.train.Saver({"v1/b": v1.b, "v1/w": v1.w})
  #   def after_create_session(self, session):
  #     import pdb; pdb.set_trace()
  #     if restore:
  #       self._saver.restore(session, os.path.join(checkpoint_dir, "my-model"))
  #   def end(self, session):
  #     import pdb; pdb.set_trace()
  #     if save_checkpoint:
  #       self._saver.save(session, os.path.join(checkpoint_dir, "my-model"))
  #
  # save_at_end_hook = SaveAtEnd()

  # Use the saver object normally after that.
  with tf.train.SingularMonitoredSession() as sess:

    # import pdb; pdb.set_trace()
    sess_actual = sess._sess._sess._sess
    # Ah, it looks like restore isn't interacting properly with initialization
    if restore:
      # Initialize v1 since the saver will not.
      saver.restore(sess_actual, os.path.join(checkpoint_dir, "my-model"))
    input_val = np.reshape(np.linspace(0, 10, 5), [1, 5])
    v1_result, v2_result, both_result = sess.run([v1_only, v2_only, v1_and_v2],
                                                 feed_dict={input: input_val})

    print("input : {}".format(input_val))
    print("v1 output : {}".format(v1_result))
    print("v2 output : {}".format(v2_result))
    print("v1->v2 output : {}".format(both_result))

    if save_checkpoint:
      saver.save(sess_actual, os.path.join(checkpoint_dir, "my-model"))


if __name__ == "__main__":
  checkpoint_dir = "/NAS/data/drewjaegle/test_chkpt"
  run_simple_model(restore=False, save_checkpoint=True, checkpoint_dir=checkpoint_dir)
  # run_simple_model(restore=True, save_checkpoint=False, checkpoint_dir=checkpoint_dir)
  run_simple_model(restore=False, save_checkpoint=False, checkpoint_dir=checkpoint_dir)
