import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

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

class SingleObjectPushEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # initialize ezpickle
        utils.EzPickle.__init__(self)

        # prev action
        self.prev_action = None

        # initialize environment
        mujoco_path = os.path.join(os.path.dirname(__file__), 'assets', 'single_object_pusher.xml')
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
            print("Left Gripper: {}".format(self._loc("l_hand")))
            print("Right Gripper: {}".format(self._loc("r_hand")))
            print("Object 1: {}".format(self._loc("obj1")))
            print("Goal 1: {}".format(self._loc("goal1")))

        # calculate reward
        reward = 0
        arm_location = self._loc2("robot")
        for obj in [1]:
            locs = {}
            for name in ['goal', 'obj']:
                locs[name + '_part1'] = self._loc2("{}{}_1".format(name, obj))
                locs[name + '_part2'] = self._loc2("{}{}_2".format(name, obj))
                locs[name + '_joint'] = self._loc2("{}{}".format(name, obj))
                locs[name + '_end1'] = locs[name + '_part1'] * 2 - locs[name + '_joint']
                locs[name + '_end2'] = locs[name + '_part2'] * 2 - locs[name + '_joint']
                locs[name + '_center'] = (
                       locs[name + '_part1'] + locs[name + '_part2'] + locs[name + '_joint']) / 3

            obj2target = lambda goal, obj: obj + (obj - goal) / norm(obj - goal) * 0.08
            target1 = obj2target(locs['goal_end1'], locs['obj_end1'])
            target2 = obj2target(locs['goal_end2'], locs['obj_end2'])
            targetj = obj2target(locs['goal_joint'], locs['obj_joint'])

            dist_arm2seg1 = dist_point2seg(targetj, target1, arm_location)
            dist_arm2seg2 = dist_point2seg(targetj, target2, arm_location)
            
            # part 1: distance between object part and goal
            reward -= np.linalg.norm(locs['goal_center'] - locs['obj_center'])

            # part 2: distance between arm and object part
            reward -= max(0, min(dist_arm2seg1, dist_arm2seg2) - 0.04)

        # part 3: arm preferred to be on the ground
        reward -= max(0, self.get_body_com('robot')[2] - 1.12) * 0.2

        # part 4: penalize action
        # reward -= norm(action) * 0.1 

        # part 5: penalize non-smooth action
        # if self.prev_action is not None:
        #     reward -= norm(action - self.prev_action) * 0.1
        # self.prev_action = action

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
        # camera distance
        self.viewer.cam.distance = 2
        
        # viewing angle
        self.viewer.cam.lookat[1] = -0.5
        self.viewer.cam.lookat[2] = 1.0

    def reset_model(self):
        qpos = self.init_qpos

        self.prev_action = None

        goal_obj_pos_arr = []
        for i in range(1):
            goal_obj_pos_arr.append(0)
            while True:
                # object and goal should have the same rotation
                rotation = np.random.uniform(low=-31.4, high=31.4, size=1)
                obj_pos = np.concatenate([
                    rotation,
                    np.random.uniform(low=-0.6, high=0.6, size=1),
                    np.random.uniform(low=-0.6, high=0.6, size=1)
                ])
                goal_pos = np.concatenate([
                    rotation,
                    np.random.uniform(low=-0.6, high=0.6, size=1),
                    np.random.uniform(low=-0.6, high=0.6, size=1)
                ])
                # constrain distance to (0.1, 0.3)
                if np.abs(np.linalg.norm(goal_pos[1:] - obj_pos[1:]) - 0.2) < 0.1:
                    goal_obj_pos_arr.extend(obj_pos.tolist())
                    goal_obj_pos_arr.extend(goal_pos.tolist())
                    break
        
        qpos[-len(goal_obj_pos_arr):] = goal_obj_pos_arr

        qvel = self.init_qvel + np.random.uniform(low=-0.005, high=0.005,
                        size=self.model.nv)

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:5],
            self.sim.data.qvel.flat[:5],
            self._loc('robot'),
            self._loc('obj1'),
            self._loc('obj1_1'),
            self._loc('obj1_2'),
            self._loc('goal1'),
            self._loc('goal1_1'),
            self._loc('goal1_2')
        ])
