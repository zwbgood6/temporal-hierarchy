from OpenGL import GLU
import gym, roboschool
import numpy as np


class DataHandler(object):
    def __init__(self, config):
        """
        Reacher dataset handler for creating trajectory rollouts.
        :param config: Dataset config structure.
        """
        self._seq_length = config.num_frames
        self._resolution = config.im_height
        self._batch_size = config.batch_size
        self._dataset_size = 1000      # infinite data, only important for validation
        self._num_joints = config.num_joints

        if self._num_joints == 1:
            dummy_env = gym.make("RoboschoolOneJointReacher-v1")    # only used for policy init
        elif self._num_joints == 2:
            dummy_env = gym.make("RoboschoolReacher-v1")  # only used for policy init
        else:
            raise ValueError("Number of Joints for reacher can only be 1 or 2!")
        self._num_actions = dummy_env.action_space.shape[0]
        self._max_action = float(config.max_degree/180.0)*np.pi
        if config.policy == "random":
            self._policy = RandomPolicy(dummy_env.observation_space,
                                        dummy_env.action_space,
                                        max_action=self._max_action)
        elif config.policy == "constant":
            self._policy = ConstantPolicy(dummy_env.observation_space,
                                          dummy_env.action_space,
                                          constant=self._max_action)
        else:
            raise NotImplementedError("Policy with keyword %s is not implemented!" % config.policy)

    def GetBatchSize(self):
        return self._batch_size

    def GetImageSize(self):
        return self._resolution

    def GetSeqLength(self):
        return self._seq_length

    def GetDatasetSize(self):
        return self._dataset_size

    def GetNumActions(self):
        return self._num_actions

    def MakeEnvironment(self):
        if self._num_joints == 1:
            return gym.make("RoboschoolOneJointReacher-v1")
        elif self._num_joints == 2:
            return gym.make("RoboschoolReacher-v1")
        else:
            raise ValueError("Number of Joints for reacher can only be 1 or 2!")

    def CreateSequence(self, env, seq_len, resolution):
        img_seq = np.empty((seq_len, resolution, resolution, 3))
        action_seq = np.empty((seq_len, self._num_actions))
        summed_action_seq = np.empty((seq_len, self._num_actions))
        absolute_action_seq = np.empty((seq_len, self._num_actions))
        summed_action = None
        obs = env.reset()
        for n in range(seq_len):
            action = self._policy.act(obs)
            # store observation and action that transforms it into next observation
            img_seq[n, ...] = env.render("rgb_array")
            if summed_action is None:
                summed_action = np.mod(obs[-self._num_actions:] + action, 2*np.pi)
            else:
                summed_action = np.mod(summed_action + action, 2*np.pi)
            # second angle is relative to first by default, convert to absolute
            summed_action_seq[n] = summed_action
            absolute_action_seq[n, 0] = summed_action[0]
            if self._num_joints == 2:
                absolute_action_seq[n, 1] = np.mod(summed_action[0] + summed_action[1]
                                                    + np.pi, 2 * np.pi)
            action_seq[n] = action
            # propagate environment
            obs, r, done, _ = env.step(action)
            # sequence should never end before required number of images is generated
            if done:
                raise ValueError("Reacher trajectory ended too early after %d steps!" % n)
        return img_seq, action_seq, summed_action_seq, absolute_action_seq

    def GetBatch(self):
        # outputs array of dimension: batch_size x seq_len x 3 x resolution^2
        img_batch = np.empty((self._seq_length, self._batch_size, self._resolution, self._resolution, 3))
        action_batch = np.empty((self._seq_length, self._batch_size, self._num_actions))
        env = gym.make("RoboschoolReacher-v1")
        for i in range(self._batch_size):
            img_batch[:, i, ...], action_batch[:, i, :] = \
                self.CreateSequence(env, self._seq_length, self._resolution)
        img_batch = np.transpose(img_batch, [1, 0, 3, 4, 2])   # batch x seq x channels x img_dim^2
        # TODO(karl): remove mean and support multiple actions!
        action_batch = np.mean(action_batch, axis=2)
        return img_batch, action_batch


class RandomPolicy(object):
    def __init__(self,
                 observation_space,
                 action_space,
                 max_action,
                 only_forward=True):
        self._input_size = observation_space.shape[0]
        self._output_size = action_space.shape[0]
        self._max_action = max_action
        self._only_forward = only_forward

    def act(self, observations):
        if self._only_forward:
            return np.random.rand(self._output_size) * self._max_action
        else:
            return (np.random.rand(self._output_size)-0.5) * 2 * self._max_action


class ConstantPolicy(object):
    def __init__(self, observation_space,
                 action_space,
                 constant=0.2):
        self._input_size = observation_space.shape[0]
        self._output_size = action_space.shape[0]
        self._constant = constant

    def act(self, observations):
        output = np.empty(self._output_size)
        output[:] = self._constant
        return output