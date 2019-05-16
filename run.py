"""Build model graph, load variables from checkpoint, execute prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import params
import configs
from architectures.graph_builder import get_model
from specs.network_specs import HierarchicalNetworkSpec, StochasticSingleStepSpec
import collections
from utils import AttrDict
from servoing.cem.dist_metrics.color_filter_dist_metrics import mask_puck

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.flags.FLAGS


ModelParams = collections.namedtuple(
    "ModelParams",
    "input_seq_len "
    "pred_seq_len "
    "batch_size "
    "channels "
    "input_image_shape "
    "num_actions "
    "num_z "
)


class ModelExecutor(object):
    def __init__(self,
                 checkpoint_dir,
                 network_config_name,
                 model_params,
                 feed_z_sequence=False,
                 infer_initial_zs=False,
                 predicts_actions=False,
                 predicts_warped_prob_img=False,):
        """Builds the model graph and loads the variables from the checkpoint."""
        # initialize a new graph
        self.graph = tf.Graph()
        self._predicts_actions = predicts_actions
        self._predicts_warped_prob_img = predicts_warped_prob_img
        if predicts_warped_prob_img and not FLAGS.use_cdna_decoder:
            raise ValueError("Flow-based loss can only be computed using CDNA decoder!")

        with self.graph.as_default() as g:
            # initialize the model
            network_spec = configs.get_network_specs(network_config_name)
            model = get_model(network_spec)
            self.network = model(
                network_spec,
                channels=model_params.channels,
                input_image_shape=model_params.input_image_shape,
                output_activation=tf.nn.sigmoid,
                backprop_elstm_to_encoder=True,
                use_recursive_image=FLAGS.use_recursive_image,
                num_actions=model_params.num_actions,
                has_image_input=True,
                render_fcn=None,
                render_shape=None,
                tau=1.0,
                infer_actions=self.predicts_actions,
                name="generator")

            # setup input placeholders
            self.input_images = tf.placeholder(dtype=tf.float32, shape=[model_params.input_seq_len,
                                                                       model_params.batch_size] +
                                                                       model_params.input_image_shape)
            self.dummy_gt_pred_frames = tf.placeholder(dtype=tf.float32, shape=[model_params.pred_seq_len,
                                                                       model_params.batch_size] +
                                                                       model_params.input_image_shape)
            num_input_actions = model_params.pred_seq_len if feed_z_sequence else model_params.input_seq_len + model_params.pred_seq_len
            self.actions = tf.placeholder(dtype=tf.float32, shape=[num_input_actions,
                                                                   model_params.batch_size,
                                                                   model_params.num_z if feed_z_sequence else model_params.num_actions])
            if self._predicts_warped_prob_img:
                self.input_prob_images = tf.placeholder(dtype=tf.float32, shape=[model_params.input_seq_len,
                                                                                 model_params.batch_size] +
                                                                                 model_params.input_image_shape[:-1] + [1])

            # build the model graph
            kwargs={
                'n_frames_input': model_params.input_seq_len,
                'n_frames_predict': model_params.pred_seq_len,
                'is_training': False
            }

            input_data = AttrDict(input_images=self.input_images,
                                  predict_images=self.dummy_gt_pred_frames,
                                  actions=self.actions,
                                  actions_abs=tf.zeros_like(self.actions))
            if feed_z_sequence:
                input_data.z_sequence = self.actions
            if infer_initial_zs:
                self.infer_z_inputs = tf.placeholder(dtype=tf.float32, shape=[1 + model_params.pred_seq_len,
                                                                                   model_params.batch_size] +
                                                                                   model_params.input_image_shape)
                self.infer_n_zs = tf.placeholder(dtype=tf.float32, shape=[model_params.pred_seq_len,])
                input_data.infer_z_inputs = self.infer_z_inputs
                input_data.infer_n_zs = self.infer_n_zs
            if self._predicts_warped_prob_img:
                input_data.goal_img = self.input_prob_images
            self.model_output = self.network(input_data, **kwargs)
            # apply output activation separately (would be applied in losses.py normally)
            decoded_frame_key = "decoded_keyframes" if isinstance(network_spec, HierarchicalNetworkSpec) \
                                                    else "decoded_low_level_frames"
            self.model_output["target_decoded_frames"] = tf.nn.sigmoid(self.model_output[decoded_frame_key])

        # setup the session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(graph=self.graph, config=config)

        # load the variables from checkpoint
        ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
        variables_checkpoint = self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(var_list=variables_checkpoint)
        saver.restore(self.sess, save_path=ckpt_path)

    def run(self,
            input_images,
            actions,
            infer_z_inputs=None,
            infer_n_zs=None):
        # setup feed dict
        # note that actions here are actually z sequence if feed_z_sequence
        # flag in on
        dummy_gt_pred_seq = np.random.rand(*self.dummy_gt_pred_frames.get_shape().as_list())
        feed_dict = {
            self.input_images: input_images,
            self.dummy_gt_pred_frames: dummy_gt_pred_seq,
            self.actions: actions
        }
        if infer_z_inputs is not None:
            feed_dict.update({self.infer_z_inputs: infer_z_inputs,
                              self.infer_n_zs: infer_n_zs})
        if self._predicts_warped_prob_img:
            # compute binary puck image from start image
            prob_imgs = np.empty(input_images.shape, dtype=input_images.dtype)
            for i in range(input_images.shape[1]):
                prob_imgs[-1, i] = mask_puck(input_images[-1, i])
            feed_dict.update({self.input_prob_images: prob_imgs})

        # run the model
        run_outputs = self.sess.run(self.model_output, feed_dict=feed_dict)
        outputs = AttrDict(
            pred_frames=run_outputs["target_decoded_frames"],
            pred_actions=run_outputs["regressed_actions"] if self.predicts_actions else None,
            pred_dts=run_outputs["high_level_rnn_output_dt"] if "high_level_rnn_output_dt" in run_outputs else None,
            pred_prob_imgs=run_outputs["goal_imgs_out"] if self._predicts_warped_prob_img else None,
        )
        return outputs

    @property
    def predicts_actions(self):
        return self._predicts_actions

    @property
    def predicts_dts(self):
        return self._predicts_dts


if __name__ == "__main__":
    checkpoint_dir = "/home/karl/logs/top-img-actCond-li_1"
    network_config_name = "actCond_lstm_bb"

    model_params = ModelParams(
        input_seq_len=5,
        pred_seq_len=20,
        batch_size=5,
        channels=3,
        input_image_shape=[3, 64, 64],
        num_actions=8
    )

    model = ModelExecutor(checkpoint_dir, network_config_name, model_params)
    from servoing.cem.dist_metrics.vgg_dist_metrics import vgg_cosine_distance_np

    import numpy as np
    for _ in range(5):
      input_imgs = np.random.rand(model_params.input_seq_len + model_params.pred_seq_len,
                                model_params.batch_size,
                                *model_params.input_image_shape)
      input_actions = np.random.rand(model_params.input_seq_len + model_params.pred_seq_len,
                                   model_params.batch_size, model_params.num_actions)

      output_imgs = model.run(input_imgs, input_actions)
      print(output_imgs.shape)

      ii = np.transpose(input_imgs[-1], (0, 2, 3, 1))
      i2 = np.transpose(output_imgs[-1], (0, 2, 3, 1))

      cost = vgg_cosine_distance_np(i2, ii)
      print(cost)
