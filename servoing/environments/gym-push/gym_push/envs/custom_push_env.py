import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class CustomPushEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, dense_reward=True):
        self.dense_reward = dense_reward

        # initialize ezpickle
        utils.EzPickle.__init__(self)

        # initialize environment
        mujoco_path = os.path.join(os.path.dirname(__file__), 'assets', 'pusher.xml')
        frame_skip = 5
        mujoco_env.MujocoEnv.__init__(self, mujoco_path, frame_skip)

    def step(self, action, debug=False):
        # print object states before stepping
        if debug:
            print("Printing object states.")
            print("Robot Arm: {}".format(self.get_body_com("robot")))
            print("Left Gripper: {}".format(self.get_body_com("l_hand")))
            print("Right Gripper: {}".format(self.get_body_com("r_hand")))
            print("Ball: {}".format(self.get_body_com("ball")))
            print("Goal: {}".format(self.get_body_com("goal")))

        # calculate reward
        arm_location = np.array(self.get_body_com("robot"))[:2]
        ball_location = np.array(self.get_body_com("ball"))[:2]
        goal_location = np.array(self.get_body_com("goal"))[:2]
        goal_dist = np.linalg.norm(ball_location - goal_location)
        arm_ball_dist = np.linalg.norm(ball_location - arm_location)

        goal2ball = (ball_location - goal_location)
        ball2arm = (arm_location - ball_location)
        goal2ball /= np.linalg.norm(goal2ball)
        ball2arm /= np.linalg.norm(ball2arm)
        arm_ang = np.arccos(np.clip(np.dot(ball2arm, goal2ball), -1.0, 1.0))
        if(np.isnan(arm_ang)): arm_ang = np.pi
        arm_ang = np.abs(arm_ang - np.pi / 2)

        ac_reg = np.sum(np.array(action) ** 2)

        if self.dense_reward:
            # this reward function is different from that defined in the 
            # official pusher environment in Gym
            # reward in gym: -(goal_dist + 0.5 * arm_ball_dist + 0.1 * ac_reg)
            reward = -(goal_dist + 0.5 * arm_ball_dist + 0.2 * ac_reg)
            # reward =  min(1.0, 0.05 / (goal_dist + 1e-6)) + \
            #           min(1.0, 0.05 / (arm_ball_dist + 1e-6)) \
            #           * (1.0 - arm_ang / np.pi)
        else:
            # sparse reward:
            # ^
            # 1 ------------\
            # *              \
            # 0               \-----------
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
            self.sim.data.qpos.flat[:5],
            self.sim.data.qvel.flat[:5],
            self.get_body_com('robot'),
            self.get_body_com('ball'),
            self.get_body_com('goal'),
        ])
