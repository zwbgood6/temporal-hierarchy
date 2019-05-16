import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from mujoco_py.generated import const

from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm

# from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
def dist_point2seg(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
        return norm(P - A)
    if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
        return norm(P - B)
    return norm(cross(A-B, A-P))/norm(B-A)

class ObstaclePushEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # initialize ezpickle
        utils.EzPickle.__init__(self)

        self.obstacle_config = np.zeros((3,))
        self.init_arm_near_object = False

        # initialize environment
        mujoco_path = os.path.join(os.path.dirname(__file__), 'assets', 'obstacle_pusher.xml')
        frame_skip = 5
        mujoco_env.MujocoEnv.__init__(self, mujoco_path, frame_skip)

    def _loc(self, obj_name):
        return self.get_body_com(obj_name)

    def _loc2(self, obj_name):
        return np.array(self._loc(obj_name))[:2]

    def step(self, action, debug=False):
        # print object states before stepping
        if debug:
            print("Printing object states.")
            print("Robot Arm: {}".format(self._loc("robot")))
            print("Object: {}".format(self._loc("obj")))
            print("Goal: {}".format(self._loc("goal")))

        # calculate reward
        reward = 0
        arm_location = self._loc2("robot")

        locs = {}
        for name in ['goal', 'obj', 'robot']:
            locs[name] = self._loc2("{}".format(name))
        
        # part 1: distance between object part and goal
        reward -= np.linalg.norm(locs['goal'] - locs['obj'])

        # part 2: distance between arm and object part
        reward -= max(0, np.linalg.norm(locs['robot'] - locs['obj']) - 0.04)

        # part 3: arm preferred to be on the ground
        reward -= max(0, self._loc('robot')[-1] - 1.12) * 0.2

        # step the environment
        self.do_simulation(action, self.frame_skip)

        # get observation
        obs = self._get_obs()

        # get if episode completed or not
        done = False

        # generate info
        info = 'keep going' if reward < -0.1 else 'good job'

        return obs, reward, done, info

    def viewer_setup(self):
        # set camera id
        self.viewer.cam.type = const.CAMERA_FIXED
        self.viewer.cam.fixedcamid = 0

        # camera distance
        self.viewer.cam.distance = 2
        
        # viewing angle
        self.viewer.cam.lookat[1] = -0.5
        self.viewer.cam.lookat[2] = 1.0

    def reset_model(self):
        qpos = self.init_qpos

        # select obstacle position and angle
        # obstacle distance from origin is set so that robot arm doesn't overlap with the obstacle
        obstacle_dist = np.random.uniform(low=0.05, high=0.5)
        obstacle_angle = np.random.uniform(low=-np.pi, high=np.pi)
        obstacle_pos = np.array([
            obstacle_dist * np.cos(obstacle_angle - np.pi / 2),
            obstacle_dist * np.sin(obstacle_angle - np.pi / 2),
            obstacle_angle
        ])
        self.obstacle_config = obstacle_pos

        while True:
            # object and goal should have the same rotation
            obj_dist = np.random.uniform(low=0.2, high=1.0)
            obj_angle = np.random.uniform(low=np.pi * 1 / 4, high=np.pi * 3 / 4)
            obj_pos = np.array([obstacle_pos[0] + np.cos(obj_angle + obstacle_angle) * obj_dist, 
                       obstacle_pos[1] + np.sin(obj_angle + obstacle_angle) * obj_dist])

            goal_dist = np.random.uniform(low=0.2, high=1.0)
            goal_angle = np.random.uniform(low=np.pi * 5 / 4, high=np.pi * 7 / 4)
            goal_pos = np.array([obstacle_pos[0] + np.cos(goal_angle + obstacle_angle) * goal_dist, 
                        obstacle_pos[1] + np.sin(goal_angle + obstacle_angle) * goal_dist])

            # requirement 1: object and goal need to be both on table
            obj_and_goal_on_table = np.max(np.abs(np.concatenate([obj_pos, goal_pos]))) < 0.9

            # requirement 2: object and goal need to be occluded by obstacle
            # note: coordinates here use center of obstacle as origin and obstacle as y-axis
            obj_relative = [obj_dist * np.cos(np.pi / 2 - obj_angle),
                            obj_dist * np.sin(np.pi / 2 - obj_angle)]
            goal_relative = [goal_dist * np.cos(np.pi / 2 - goal_angle),
                             goal_dist * np.sin(np.pi / 2 - goal_angle)]
            y_intercept = (goal_relative[0] * obj_relative[1] - goal_relative[1] * obj_relative[0]) / (obj_relative[1] - obj_relative[0])
            obstacle_in_between = np.abs(y_intercept) < 0.4

            if obj_and_goal_on_table and obstacle_in_between:
                break

        # set robot
        if self.init_arm_near_object:
            while True:
                arm_dist = np.random.uniform(low=0.16, high=0.24)
                arm_angle = np.random.uniform(low=0, high=np.pi * 2)
                arm_pos = np.array([obj_pos[0] + np.cos(arm_angle) * arm_dist,
                                    obj_pos[1] + np.sin(arm_angle) * arm_dist])

                arm_on_table = np.max(arm_pos) < 0.9

                if arm_on_table:
                    break
        else:
            arm_pos = np.array([0.0, 0.0])

        goal_display_pos = np.array([10.0, 10.0, 0.0])

        position_config_arr = np.concatenate([arm_pos, np.zeros((3,)), obj_pos, goal_pos, goal_display_pos, obstacle_pos])

        qvel = self.init_qvel + np.random.uniform(low=-0.005, high=0.005,
                        size=self.model.nv)
        
        qpos[-len(position_config_arr):] = position_config_arr
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:5],
            self.sim.data.qvel.flat[:5],
            self._loc('robot'),
            self._loc('obj'),
            self._loc('goal'),
            [self.sim.data.get_joint_qpos('cart_rotate_z')],
            self.obstacle_config
        ])

    def get_goal_image(self):
        qpos, qvel = self.sim.data.qpos, self.sim.data.qvel
        qpos_original = qpos.tolist()
        qpos[-10:-8] = self._loc2('goal')

        self.set_state(qpos, qvel)
        goal_image = self.render(mode='rgb_array')
        self.set_state(np.array(qpos_original), qvel)

        return goal_image

    def get_goal_displayed_image(self):
        qpos, qvel = self.sim.data.qpos, self.sim.data.qvel
        qpos_original = qpos.tolist()
        qpos[-6:-4] = self._loc2('goal')
        qpos[-4] = 0.0

        self.set_state(qpos, qvel)
        goal_displayed_image = self.render(mode='rgb_array')
        self.set_state(np.array(qpos_original), qvel)

        return goal_displayed_image

    def get_centered_arm_image(self):
        qpos, qvel = self.sim.data.qpos, self.sim.data.qvel
        qpos_original = qpos.tolist()
        qpos[-15:-13] = np.array([0, 0])

        self.set_state(qpos, qvel)
        arm_centered_image = self.render(mode='rgb_array')
        self.set_state(np.array(qpos_original), qvel)

        return arm_centered_image
