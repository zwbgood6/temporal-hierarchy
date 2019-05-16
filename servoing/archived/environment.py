from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np

from policy import RandomContinuousPolicy, ConstantContinuousPolicy


class Environment(object):
    def __init__(self):
        pass

    @abstractmethod
    def makeEnvironment(self):
        """Creates an environment instance."""
        return

    @abstractmethod
    def getStartTargetImgPair(self, n_steps):
        """Returns start and goal image (sequence) and input action sequence for servoing
            that are n_steps images apart. Also return initial state difference (e.g. angle)
            Stores state after start image (sequence) for further steps."""
        return

    @abstractmethod
    def step(self, action):
        """Propagates the environment one step, returns image of resulting state and state
            error to target state."""
        return

    @abstractmethod
    def computeCost(self, input_imgs, target_img):
        """Computes the distance of the input_imgs to the target with the environment
            specific distance metric."""
        return

    @abstractmethod
    def checkTargetReached(self, input_img, target_img):
        """Checks whether the input image is close enough to the target image to have reached."""
        return

    def _blendImages(self, img1, img2, alpha=0.3):
        dtype = img1.dtype
        return np.asarray(img1 * (1.0 - alpha) + img2 * alpha, dtype=dtype)

    def plotTrajectory(self, trajectory_imgs, target_img=None, noshow=False, white_separation=False, render=False):
        input_res = trajectory_imgs[0].shape[0]
        concat_input_img = np.concatenate(trajectory_imgs, axis=1)
        if target_img is not None:
            repeated_target_img = np.concatenate([target_img for _ in range(len(trajectory_imgs))], axis=1)
            blended_img = self._blendImages(concat_input_img, repeated_target_img)
        else:
            blended_img = concat_input_img
        # add lines between images
        for i in range(len(trajectory_imgs))[1:]:
            blended_img[:, input_res*i] = 0 if not white_separation else 1
        if render:
            return blended_img
        plt.figure()
        plt.imshow(blended_img)
        if not noshow:
            plt.show()


    def gen_padded_subtrajectory(self, trajectory_imgs, target_img, max_imgs):
        final_height = trajectory_imgs[0].shape[0]
        final_width = max_imgs * final_height
        output_array = np.ones((final_height, final_width, 3), dtype=trajectory_imgs[0].dtype) * 179
        unpadded_array = self.plotTrajectory(trajectory_imgs, target_img, render=True)
        output_array[:, -unpadded_array.shape[1]:] = unpadded_array
        return output_array


    def gen_overview_figure(self, sub_trajectories, executed_traj, noshow=False):
        sub_trajectories.append(executed_traj)
        traj_stack = np.concatenate(sub_trajectories, axis=0)
        plt.figure()
        plt.imshow(traj_stack)
        if noshow:
            return
        plt.show()


class GridworldEnvironment(Environment):
    def __init__(self, config):
        super(GridworldEnvironment, self).__init__()
        self._resolution = config.image_size
        self._input_seq_len = config.input_seq_len
        self._num_actions = config.num_actions
        self._max_action = config.max_action
        self._agent_size = config.agent_size
        if self._agent_size % 2 == 0:
            raise ValueError("Gridworld agent size must be uneven!")
        self._margin = np.uint8((self._agent_size-1)/2)
        self._policy = RandomContinuousPolicy(np.zeros(0), np.zeros((self._num_actions,)), self._max_action)
        self._pos = None        # store current agent position

    def _boundMargin(self, pos):
        return np.asarray(np.clip(pos, self._margin, self._resolution - 1 - self._margin), dtype=np.int8)

    def _samplePos(self):
        raw_pos = np.asarray(np.random.rand(self._num_actions) * (self._resolution-1), dtype=np.int8)
        return self._boundMargin(raw_pos)

    def reset(self):
        self._pos = self._samplePos()
        canvas = self._renderAgent(self._pos)
        return canvas

    def _renderAgent(self, pos, canvas=None):
        if canvas is None:
            canvas = np.zeros((self._resolution, self._resolution), dtype=np.float32)
        # check that its not too close, otherwise correct
        pos = self._boundMargin(pos)
        # print agent on canvas
        canvas[pos[0] - self._margin : pos[0] + self._margin,
               pos[1] - self._margin : pos[1] + self._margin] = 1.0
        return canvas

    def getStartTargetImgPair(self, n_steps):
        seq_len = self._input_seq_len + n_steps
        input_imgs = np.empty((self._input_seq_len, self._resolution, self._resolution))
        input_actions = np.empty((self._input_seq_len, self._num_actions))
        stored_state = None
        obs = self.reset()
        for t in range(seq_len):
            action = np.asarray(self._policy.act(obs), dtype=np.int8)
            new_img, _ = self.step(action)
            if t < self._input_seq_len:
                input_imgs[t] = obs
                input_actions[t] = action
                self._start_state = self._pos
            obs = new_img
        target_img = new_img
        self._target_state = self._pos
        initial_dist = self._target_state - self._start_state
        self._pos = self._start_state        # reset internal state to end of input imgs
        return input_imgs, target_img, input_actions, initial_dist

    def act(self, obs):
        return self._policy.act(obs)

    def step(self, action, compute_error=False):
        new_pos = self._boundMargin(self._pos + action)
        action = new_pos - self._pos
        canvas = self._renderAgent(new_pos)
        if compute_error:
            error = self._target_state - new_pos
        self._pos = new_pos
        if compute_error:
            return canvas, action, error
        else:
            return canvas, action

    def getPos(self):
        return self._pos

    def computeCost(self, input_imgs, target_img):
        return -1.0

    def checkTargetReached(self, input_img, target_img):
        return False


if __name__ == "__main__":
    import collections
    GridworldConfig = collections.namedtuple(
        "GridworldConfig",
        "dataset_type data_file num_frames batch_size image_size "
        "input_seq_len pred_seq_len channels max_action num_actions agent_size"
    )
    spec = GridworldConfig(
        dataset_type=2,
        data_file="",
        num_frames=15,
        batch_size=5,
        image_size=64,
        input_seq_len=5,
        pred_seq_len=10,
        channels=1,
        max_action=10,  # max spatial translation per step in px
        num_actions=2,
        agent_size=5)
    env = GridworldEnvironment(spec)

    input_imgs, target, actions = env.getStartTargetImgPair(5)
    env.plotTrajectory(input_imgs, target, white_separation=True)

