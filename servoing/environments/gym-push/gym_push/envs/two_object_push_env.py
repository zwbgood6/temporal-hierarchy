import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class TwoObjectPushEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, dense_reward=True):
        self.dense_reward = dense_reward

        # initialize ezpickle
        utils.EzPickle.__init__(self)

        # initialize environment
        mujoco_path = os.path.join(os.path.dirname(__file__), 'assets', 'two_object_pusher.xml')
        frame_skip = 5
        mujoco_env.MujocoEnv.__init__(self, mujoco_path, frame_skip)

    def _loc(self, obj_name):
        return self.get_body_com(obj_name)

    def step(self, action, debug=False):
        # print object states before stepping
        if debug:
            print("Printing object states.")
            print("Robot Arm: {}".format(self._loc("robot")))
            print("Left Gripper: {}".format(self._loc("l_hand")))
            print("Right Gripper: {}".format(self._loc("r_hand")))
            print("Object 1: {}".format(self._loc("obj1")))
            print("Object 2: {}".format(self._loc("obj2")))
            print("Goal 1: {}".format(self._loc("goal1")))
            print("Goal 2: {}".format(self._loc("goal2")))

        # calculate reward
        reward = 0
        arm_location = self._loc("robot")[:2]
        arm_dists = []
        for obj in [1, 2]:
            goal_dists = []
            for part in [1, 2]:
                goal_location = None
                for name in ['goal', 'obj']:
                    part_location = np.array(self._loc("{}{}_{}".format(name, obj, part)))[:2]
                    joint_location = np.array(self._loc("{}{}".format(name, obj)))[:2]
                    end_location = part_location * 2 - joint_location
                    if name == 'goal':
                        goal_location = end_location
                    if name == 'obj':
                        goal_dist = np.linalg.norm(goal_location - end_location)
                        goal_dists.append(goal_dist)
                        # part 1: distance between object part and goal
                        reward -= goal_dist 

                        # part 2: distance between arm and object part
                        if goal_dist >= 0.05:
                            arm_dists.append(np.linalg.norm(arm_location - end_location))

                        goal_dists.append(goal_dist)

            # part 3: subtask success reward
            if np.mean(goal_dists) < 0.05:
                reward += 10
        if len(arm_dists) > 0:
            reward -= np.min(np.array(arm_dists))

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

        goal_obj_pos_arr = []
        while True:
            goal_obj_pos_arr = []

            # loop through all (object, goal) pairs to generate initial positions
            for i in range(2):
                goal_obj_pos_arr.append(0)

                # repeat until generation of (object, goal) pair is successful
                while True:
                    # object and goal should have the same rotation
                    rotation = (np.random.rand() - 0.5) * 2 * 31.4

                    # sample object's initial position
                    obj_pos = np.array([
                        (np.random.rand() - 0.5) * 1.2,
                        (np.random.rand() - 0.5) * 1.2,
                        rotation
                    ])

                    # sample the distance and angle of goal from the object
                    goal_dist = np.random.rand() * 0.2 + 0.2
                    goal_ang = np.random.rand() * 2 * np.pi

                    # ... then calculate goal's position
                    goal_pos = np.array([
                        obj_pos[0] + np.cos(goal_ang) * goal_dist,
                        obj_pos[1] + np.sin(goal_ang) * goal_dist,
                        rotation
                    ])

                    # check if goal's position is guaranteed to be on the board
                    if np.max(np.abs(goal_pos[:2])) < 0.6:
                        # if so, stop regenerating and go to next object
                        goal_obj_pos_arr.extend(obj_pos.tolist())
                        goal_obj_pos_arr.extend(goal_pos.tolist())
                        break

            # calculate the distances between two objects and two object goals
            goal_obj_pos_arr = np.array(goal_obj_pos_arr)
            obj_dist = np.linalg.norm(goal_obj_pos_arr[1:3] - goal_obj_pos_arr[8:10])
            goal_dist = np.linalg.norm(goal_obj_pos_arr[4:6] - goal_obj_pos_arr[11:13])

            # require that object distance and goal distance must be greater 
            # than a threshold
            if obj_dist > 0.8 and goal_dist > 0.8:
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
            self._loc('obj2'),
            self._loc('obj1_1'),
            self._loc('obj2_1'),
            self._loc('obj1_2'),
            self._loc('obj2_2'),
            self._loc('goal1'),
            self._loc('goal2'),
            self._loc('goal1_1'),
            self._loc('goal2_1'),
            self._loc('goal1_2'),
            self._loc('goal2_2')
        ])
