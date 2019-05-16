"""Implements simulator base class and multiple options (NN/GT)"""

from __future__ import absolute_import

import numpy as np
import cv2
import collections

from utils import AttrDict
from servoing.cem.env_wrapper import EnvWrapper


class Simulator(object):
    def __init__(self):
        pass

    def rollout(self, start_img, actions):
        raise NotImplementedError

    def _action2sim(self, actions):
        actions[:, :, -2:] = 0 # prevent arm from lifting and rotating
        return actions


class NNSimulator(Simulator):
    def __init__(self,
                 model_params,
                 checkpoint_dir,
                 network_config_name,
                 feed_z_sequence,
                 predict_warped_prob_imgs):
        super(NNSimulator, self).__init__()
        self.model_params = model_params
        self.feed_z_sequence = feed_z_sequence
        from run import ModelExecutor
        self.model = ModelExecutor(checkpoint_dir,
                                   network_config_name,
                                   self.model_params,
                                   feed_z_sequence=feed_z_sequence,
                                   predicts_warped_prob_img=predict_warped_prob_imgs)

    def _input2sim(self, input_img):
        assert input_img.dtype == np.uint8, "Expect uint8 input image!"
        assert len(input_img.shape) == 3, "Expect 3-channel input image!"
        prep_input = input_img.astype(float) / 255
        prep_input = np.repeat(prep_input[None, None, ...], self.model_params.batch_size, axis=1)
        assert prep_input.min() >= 0.0 and prep_input.max() <= 1.0, "Input image to network should be in range [0...1]!"
        assert prep_input.dtype == float, "Input to network should be of type float!"
        assert len(prep_input.shape) == 5, "Input to network should have dimensionality 5! [1, batch_size, channels, res, res]"
        return prep_input.transpose(0, 1, 4, 2, 3)

    def _sim2output(self, output_img):
        assert output_img.dtype == np.float32, "Expect float32 output image!"
        assert len(output_img.shape) == 5, "Expect 5-channel output image!"
        prep_output = (output_img * 255).astype(np.uint8)
        prep_output = prep_output.transpose((1, 0, 3, 4, 2))
        assert prep_output.min() >= 0 and prep_output.max() <= 255, "Rollout output should be in range [0...255]!"
        return prep_output

    def _action2sim(self, actions):
        actions = super(NNSimulator, self)._action2sim(actions)
        padding = np.zeros((actions.shape[0], self.model_params.input_seq_len, actions.shape[2]), dtype=actions.dtype)
        return np.concatenate((padding, actions), axis=1)      # feed dummy values for input sequence

    def _fuse_outputs(self, output_lists, collect_key, concat_axis=1):
        output = []
        for output_list in output_lists:
            output.append(output_list[collect_key])
        return np.concatenate(output, axis=concat_axis)

    def rollout(self, start_img, actions):
        # prepare start image for model rollout
        sim_start_img = self._input2sim(start_img)

        # prepare actions for model rollout
        if not self.feed_z_sequence:
            actions = self._action2sim(actions)

        # compute number of rollouts to break down planning batch size in model batch size
        total_batch_size = actions.shape[0]
        if np.mod(total_batch_size, self.model_params.batch_size) != 0:
            raise ValueError("CEM batch size needs to be evenly dividable by model batch size, are %d and %d!"
                             % (total_batch_size, self.model_params.batch_size))
        n_rollouts = int(total_batch_size / self.model_params.batch_size)

        # perform model rollouts and collect outputs
        output_batches = []
        for i in range(n_rollouts):
            action_batch = actions[i*self.model_params.batch_size : (i+1)*self.model_params.batch_size]
            action_batch = action_batch.transpose(1, 0, 2)
            output_batches.append(self.model.run(sim_start_img, action_batch))

        # build output dict
        outputs = AttrDict(pred_frames=self._sim2output(self._fuse_outputs(output_batches, collect_key="pred_frames")))
        if output_batches[0]["pred_actions"] is not None:        # model predicted actions
            outputs.pred_actions = self._fuse_outputs(output_batches, collect_key="pred_actions").transpose((1, 0, 2))
        if output_batches[0]["pred_dts"] is not None:        # model predicted dts
            outputs.pred_dts = self._fuse_outputs(output_batches, collect_key="pred_dts").transpose((1, 0, 2))
        if output_batches[0]["pred_prob_imgs"] is not None:  # model predicted warped prob images
            outputs.pred_prob_imgs = self._fuse_outputs(output_batches, collect_key="pred_prob_imgs").transpose((1, 0, 3, 4, 2))

        return outputs


GTSimParams = collections.namedtuple(
    "GTSimParams",
    "env_name "
    "num_action_repeat "
    "output_resolution "
    "render_goal "
)

class GTSimulator(Simulator):
    def __init__(self, params):
        super(GTSimulator, self).__init__()
        self.params = params

    def rollout(self, start_img, actions):
        env_state = start_img  # for GT simulator we are overloading the input_img argument
        actions = self._action2sim(actions) # prepare actions for simulator rollout

        # simulate rollouts in batch
        batch_outputs = []
        for batch_idx in range(actions.shape[0]):
            # initialize environment
            env = EnvWrapper.make_env_from_state(self.params.env_name, env_state, self.params.num_action_repeat)

            rollout_frames = []
            for step in range(actions.shape[1]):
                obs, _, done, _ = env.step(actions[batch_idx, step])
                if done:
                    raise ValueError("Episode should never be done!")
                frame = EnvWrapper.render(env, self.params.render_goal)
                rollout_frames.append(cv2.resize(frame, (self.params.output_resolution, self.params.output_resolution),
                                                 interpolation=cv2.INTER_CUBIC))
            batch_outputs.append(np.stack(rollout_frames, axis=0))
        return AttrDict(pred_frames=np.stack(batch_outputs, axis=0))
