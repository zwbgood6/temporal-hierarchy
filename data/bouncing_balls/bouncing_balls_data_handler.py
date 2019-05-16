import numpy as np

from .bouncing_balls import BouncingBalls


class BouncingBallsDataHandler(object):
    def __init__(self, config):
        """
        BouncingBalls dataset handler for creating trajectory rollouts.
        :param config: Dataset config structure.
        """
        self._seq_length = config.num_frames
        self._resolution = config.resolution
        self._batch_size = config.batch_size
        self._dataset_size = 1000      # infinite data, only important for validation
        self._num_actions = config.num_actions   # dummy value for now
        self._image_input = config.image_input

        self._env = BouncingBalls(resolution=config.resolution,
                                  num_balls=config.num_balls,
                                  ball_size=config.ball_size,
                                  ball_speed=config.ball_speed,
                                  agent_speed=config.agent_speed,
                                  stochastic_angle=config.stochastic_angle,
                                  random_angle=config.random_angle,
                                  stochastic_speed=config.stochastic_speed,
                                  stochastic_bounce=config.stochastic_bounce,
                                  rand_start_counter=config.rand_start_counter,
                                  variance_degrees=config.variance_degrees,
                                  bounce_vertically=config.bounce_vertically,
                                  segment_length_limits=config.segment_length_limits,
                                  render=config.image_input,
                                  )

    def GetBatchSize(self):
        return self._batch_size

    def GetImageSize(self):
        if self._image_input:
            return self._resolution
        else:
            return self._env.output_size    # this is a hack to use the same data loading framework as for imgs

    def GetSeqLength(self):
        return self._seq_length

    def GetDatasetSize(self):
        return self._dataset_size

    def GetNumActions(self):
        return self._num_actions

    def CreateSequence(self, seq_len):
        obs_seq = np.empty([seq_len] + self._env.output_size)
        action_seq = np.empty((seq_len, self._num_actions))
        abs_action_seq = np.empty((seq_len, self._num_actions))
        dummy_action = 0
        obs = self._env.reset()
        for n in range(seq_len):
            try:
                next_obs, actions, bounce, _ = self._env.step(dummy_action)
            except ValueError:
                print('Could not find a valid new bouncing angle!')
                print(self._env.unnormalize_vec(obs_seq[:n]))
            # store observation and action that transforms it into next observation
            obs_seq[n] = obs
            action_seq[n] = actions
            abs_action_seq[n] = bounce
            # update image
            obs = next_obs
        abs_action_seq[1:] = abs_action_seq[:-1]
        return obs_seq, action_seq, abs_action_seq

    def GetBatch(self):
        # outputs array of dimension: batch_size x seq_len x obs_dim
        obs_batch = np.empty([self._batch_size, self._seq_length] + self._env.output_size)
        action_batch = np.empty((self._seq_length, self._batch_size, self._num_actions))
        abs_action_batch = np.empty((self._seq_length, self._batch_size, self._num_actions))
        for i in range(self._batch_size):
            obs_batch[i], action_batch[:, i], abs_action_batch[:, i] = \
                self.CreateSequence(self._seq_length)
        return obs_batch, action_batch, abs_action_batch

    @property
    def render_fcn(self):
        return self._env.render_state_vec_batch

    @property
    def render_shape(self):
        return self._env.render_shape


if __name__ == "__main__":
    IMAGE_INPUT = False

    import collections, cv2
    BouncingBallsConfig = collections.namedtuple(
        "BouncingBallsConfig",
        "resolution num_balls ball_size ball_speed agent_speed stochastic_angle random_angle stochastic_speed "
        "variance_degrees bounce_vertically segment_length_limits batch_size input_seq_len pred_seq_len "
        "num_frames channels num_actions stochastic_bounce rand_start_counter image_input")

    spec = BouncingBallsConfig(
        resolution=256,
        num_balls=1,
        ball_size=0.1,
        ball_speed=0.5,
        agent_speed=1,
        stochastic_angle=False,
        random_angle=True,
        stochastic_speed=False,
        stochastic_bounce=True,
        rand_start_counter=True,
        variance_degrees=5,
        bounce_vertically=False,
        segment_length_limits=[5, 8],
        batch_size=1,
        input_seq_len=5,
        pred_seq_len=30,
        num_frames=35,
        channels=1,
        num_actions=5,
        image_input=IMAGE_INPUT)

    dh = BouncingBallsDataHandler(spec)
    batch, _, _ = dh.GetBatch()
    if not IMAGE_INPUT:
        batch = dh.render_fcn(batch)
    for i in range(batch[0].shape[0]):
        img = batch[0, i, 0]
        cv2.imshow('test', img)
        cv2.waitKey(0)
