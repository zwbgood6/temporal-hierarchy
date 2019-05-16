import numpy as np

from servoing.environment import GridworldEnvironment


class GridWorldDataHandler(object):
    def __init__(self, config):
        """
        Reacher dataset handler for creating trajectory rollouts.
        :param config: Dataset config structure.
        """
        self._seq_length = config.num_frames
        self._resolution = config.image_size
        self._batch_size = config.batch_size
        self._dataset_size = 1000      # infinite data, only important for validation

        self._num_actions = config.num_actions
        self._max_action = config.max_action
        self._env = GridworldEnvironment(config)

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

    def CreateSequence(self, seq_len, resolution):
        img_seq = np.empty((seq_len, resolution, resolution, 1))
        action_seq = np.empty((seq_len, self._num_actions))
        abs_action_seq = np.empty((seq_len, self._num_actions))
        img = self._env.reset()
        for n in range(seq_len):
            pos = self._env.getPos()
            action = self._env.act(img)
            next_img, action = self._env.step(action)
            # store observation and action that transforms it into next observation
            img_seq[n] = np.expand_dims(img, -1)
            action_seq[n] = action
            abs_action_seq[n] = pos
            # update image
            img = next_img
        return img_seq, action_seq, abs_action_seq

    def GetBatch(self):
        # outputs array of dimension: batch_size x seq_len x 1 x resolution^2
        img_batch = np.empty((self._seq_length, self._batch_size, self._resolution, self._resolution, 1))
        action_batch = np.empty((self._seq_length, self._batch_size, self._num_actions))
        abs_action_batch = np.empty((self._seq_length, self._batch_size, self._num_actions))
        for i in range(self._batch_size):
            img_batch[:, i], action_batch[:, i, :], abs_action_batch[:, i, :] = \
                self.CreateSequence(self._seq_length, self._resolution)
        img_batch = np.transpose(img_batch, [1, 0, 4, 2, 3])   # batch x seq x channels x img_dim^2
        return img_batch, action_batch, abs_action_batch