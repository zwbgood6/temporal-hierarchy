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

class MultigoalObstaclePushEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # initialize ezpickle
        utils.EzPickle.__init__(self)

        # state about whether goals have been reached
        self.num_goals = 3
        self.goal_reached = [0.0] * self.num_goals

        self.obstacle_config = np.zeros((3,))

        # initialize environment
        mujoco_path = os.path.join(os.path.dirname(__file__), 'assets', 'multigoal_obstacle_pusher.xml')
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
            print("Goal: {}, {}, {}".format(self._loc("goal1"), self._loc("goal2"), self._loc("goal3")))

        # calculate reward
        reward = 0
        arm_location = self._loc2("robot")

        locs = {}
        for name in list(map(lambda ix: 'goal' + str(ix + 1), 
                             list(range(self.num_goals)))) + ['obj', 'robot']:
            locs[name] = self._loc2("{}".format(name))
        
        # part 1: min distance between object part and any goal
        reward -= np.min(list(map(lambda ix: np.linalg.norm(locs['goal{}'.format(ix + 1)] - locs['obj']), 
                                  list(range(self.num_goals)))))

        # part 2: distance between arm and object part
        reward -= max(0, np.linalg.norm(locs['robot'] - locs['obj']) - 0.04)

        # part 3: arm preferred to be on the ground
        reward -= max(0, self._loc('robot')[-1] - 1.12) * 0.2

        # part 4: large reward for every goal reached
        reward += np.sum(self.goal_reached) * 10

        # step the environment
        self.do_simulation(action, self.frame_skip)

        # check if any goal is reached, set flag to 1.0
        for ix in range(self.num_goals):
            if np.linalg.norm(locs['goal{}'.format(ix + 1)] - locs['obj']) < 0.16:
                reward += 10

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

            goal_positions = []
            goal_behind_wall = []
            for ix in range(self.num_goals):
                goal_dist = np.random.uniform(low=0.2, high=1.0)
                goal_angle = np.random.uniform(low=np.pi * 5 / 4, high=np.pi * 7 / 4)
                # E[number of balls across the obstacle = 2] <=> P(place across) = 0.873
                if np.random.rand() > 0.873: 
                    goal_angle -= np.pi
                    goal_behind_wall.append(False)
                else:
                    goal_behind_wall.append(True)
                goal_pos = np.array([obstacle_pos[0] + np.cos(goal_angle + obstacle_angle) * goal_dist, 
                            obstacle_pos[1] + np.sin(goal_angle + obstacle_angle) * goal_dist])
                goal_positions.append(goal_pos)
            
            self.goal_behind_wall = goal_behind_wall

            # requirement 1: object and goals need to be both on table
            obj_and_goal_on_table = np.max(np.abs(np.concatenate([obj_pos] + goal_positions))) < 0.9

            if obj_and_goal_on_table:
                break

        goal_image_config_arr = np.concatenate([goal_pos] + goal_positions + [obstacle_pos])
        position_config_arr = np.concatenate([obj_pos] + goal_positions + [obstacle_pos])
        
        qvel = self.init_qvel + np.random.uniform(low=-0.005, high=0.005,
                        size=self.model.nv)
        
        qpos[-len(position_config_arr):] = goal_image_config_arr
        self.set_state(qpos, qvel)

        self.goal_image = self.render(mode='rgb_array')
        
        qpos[-len(position_config_arr):] = position_config_arr
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:5],
            self.sim.data.qvel.flat[:5],
            self._loc('robot'),
            self._loc('obj'),
            self._loc('goal1'),
            self._loc('goal2'),
            self._loc('goal3'),
            [self.sim.data.get_joint_qpos('cart_rotate_z')],
            self.obstacle_config
        ])

    def get_goal_image(self):
        return self.goal_image
