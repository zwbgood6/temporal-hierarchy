import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class CustomMoveEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, dense_reward=True):
        self.dense_reward = dense_reward

        # initialize ezpickle
        utils.EzPickle.__init__(self)

        # initialize environment
        mujoco_path = os.path.join(os.path.dirname(__file__), 'assets', 'move.xml')
        frame_skip = 5
        mujoco_env.MujocoEnv.__init__(self, mujoco_path, frame_skip)

    def step(self, action, debug=False):
        # print object states before stepping
        if debug:
            print("Printing object states.")
            print("Ball: {}".format(self.get_body_com("ball")))
            print("Goal: {}".format(self.get_body_com("goal")))

        # calculate reward
        ball_location = self.get_body_com("ball")
        goal_location = self.get_body_com("goal")
        goal_dist = np.linalg.norm(np.array(ball_location) - np.array(goal_location))
        if self.dense_reward:
            reward = min(1.0, 0.05 / (goal_dist + 1e-6))
        else:
            # sparse reward:
            # ^
            # 1 ------------\
            # *                  \
            # 0                    \-----------
            # *********** .05 * .1 *********> L2_dist(ball, goal)
            reward = max(0.0, min(1.0, 2.0 - goal_dist / 0.05))
        
        # step the environment
        self.do_simulation(action, self.frame_skip)

        # get observation
        obs = self._get_obs()

        # get if episode completed or not
        done = False

        # generate info
        info = 'keep going' if reward < 1.0 else 'good job'

        return obs, reward, done, info

    def viewer_setup(self):
        # camera distance
        self.viewer.cam.distance = 2
        
        # viewing angle
        self.viewer.cam.lookat[1] = -0.5
        self.viewer.cam.lookat[2] = 1.0

    def reset_model(self):
        qpos = self.init_qpos

        while True:
            self.goal_pos = np.concatenate([
                np.random.uniform(low=-0.8, high=0.8, size=1),
                np.random.uniform(low=-0.8, high=0.8, size=1)
            ])
            self.ball_pos = np.concatenate([
                np.random.uniform(low=-0.8, high=0.8, size=1),
                np.random.uniform(low=-0.8, high=0.8, size=1)
            ])
            if np.linalg.norm(self.goal_pos - self.ball_pos) > 0.3:
                break
        
        qpos[-4:-2] = self.ball_pos
        qpos[-2:] = self.goal_pos

        qvel = self.init_qvel + np.random.uniform(low=-0.005, high=0.005,
                        size=self.model.nv)

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.get_body_com('ball'),
            self.get_body_com('goal'),
        ])
