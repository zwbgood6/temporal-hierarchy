from OpenGL import GLU
import gym, roboschool
import numpy as np

from servoing.policy import RandomContinuousPolicy, ConstantContinuousPolicy
from servoing.metrics import vgg_cosine_distance_np
from servoing.environment import Environment

class ReacherEnvironment(Environment):
    def __init__(self, config):
        super(ReacherEnvironment, self).__init__()
        self._cosd_thresh = 0.02    # threshold for target reached
        self._input_seq_len = config.input_seq_len
        self._resolution = config.im_height
        self._num_joints = config.num_joints
        self._max_action = float(config.max_degree / 180.0) * np.pi

        dummy_env = self.makeEnvironment()
        self._num_actions = dummy_env.action_space.shape[0]

        if config.policy == "random":
            self._policy = RandomContinuousPolicy(dummy_env.observation_space,
                                                  dummy_env.action_space,
                                                  max_action=self._max_action,
                                                  only_forward=True)
        elif config.policy == "constant":
            self._policy = ConstantContinuousPolicy(dummy_env.observation_space,
                                                    dummy_env.action_space,
                                                    constant=self._max_action)
        else:
            raise NotImplementedError("Policy with keyword %s is not implemented!" % config.policy)

    def makeEnvironment(self):
        if self._num_joints == 1:
            env = gym.make("RoboschoolOneJointReacher-v1")
        elif self._num_joints == 2:
            env = gym.make("RoboschoolReacher-v1")
        else:
            raise ValueError("Number of Joints for reacher can only be 1 or 2!")
        return env

    def getStartTargetImgPair(self, n_steps):
        seq_len = self._input_seq_len + n_steps
        self.env = self.makeEnvironment()
        input_img_seq = np.empty((self._input_seq_len, self._resolution, self._resolution, 3), dtype=np.uint8)
        input_action_seq = np.empty((self._input_seq_len-1, self._num_actions), dtype=np.float32)
        obs = self.env.reset()
        for n in range(seq_len):
            action = self._policy.act(obs)
            img = self.env.render("rgb_array")
            if n < self._input_seq_len:
                input_img_seq[n] = img
            if n < self._input_seq_len-1:
                input_action_seq[n] = action
            if n == self._input_seq_len - 1:
                self._start_state = obs    # copy final state for further stepping
            state = obs[-self._num_actions:]
            obs, r, done, _ = self.env.step(action)
            if done:
                raise ValueError("Reacher trajectory ended too early after %d steps!" % n)
        target_img = img
        self._target_state = state
        initial_angle_diff = (self._target_state - self._start_state[-self._num_actions:]) * 180/np.pi
        obs, r, done, _ = self.env.step(self._start_state)    # reset to end of input_seq (this is hacked to work)
        return input_img_seq, target_img, input_action_seq, initial_angle_diff

    def step(self, action, compute_error=False):
        obs, r, done, _ = self.env.step(action)
        if done:
            raise ValueError("Reacher trajectory ended too early!")
        if compute_error:
            state_error = (self._target_state - obs[-self._num_actions:]) * 180/np.pi
            return self.env.render("rgb_array"), state_error
        else:
            return self.env.render("rgb_array")

    def _prep_imgs_for_vgg(self, imgs):
        return np.asarray(imgs, dtype=np.float32) / 255

    def computeCost(self, rollout_imgs, target_img_raw):
        if len(rollout_imgs.shape) == 5:
            rollout_imgs = rollout_imgs[-1]
        target_img = self._prep_imgs_for_vgg(target_img_raw)
        cost = vgg_cosine_distance_np(rollout_imgs, target_img, keep_axis=0)
        return cost

    def checkTargetReached(self, input_img, target_img):
        input_img = self._prep_imgs_for_vgg(input_img)
        target_img = self._prep_imgs_for_vgg(target_img)
        cosd = vgg_cosine_distance_np(input_img, target_img)
        return True if cosd <= self._cosd_thresh else False