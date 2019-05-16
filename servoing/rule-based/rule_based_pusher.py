# Created by Jingyun Yang on 12/22/2018.
# 
# Implements Rule-Based Pusher (RuleBasedPusher).

import gym
import gym_push
import numpy as np
from numpy.linalg import norm
from enum import Enum
from geometry_utils import *
import math
import os
import os.path as osp
import cv2
import time
import argparse
import moviepy.editor as mpy
from action_repeat import ActionRepeat
import pickle

default_sop_obs_spec = {
    'robot': (-21, -20, -19),
    'obj_joint': [(-18, -17, -16)],
    'obj_part1': [(-15, -14, -13)],
    'obj_part2': [(-12, -11, -10)],
    'goal_joint': [(-9, -8, -7)],
    'goal_part1': [(-6, -5, -4)],
    'goal_part2': [(-3, -2, -1)]
}

default_top_obs_spec = {
    'robot': (-39, -38, -37),
    'obj_joint': [(-36, -35, -34), (-33, -32, -31)],
    'obj_part1': [(-30, -29, -28), (-27, -26, -25)],
    'obj_part2': [(-24, -23, -22), (-21, -20, -19)],
    'goal_joint': [(-18, -17, -16), (-15, -14, -13)],
    'goal_part1': [(-12, -11, -10), (-9, -8, -7)],
    'goal_part2': [(-6, -5, -4), (-3, -2, -1)]
}

default_obp_obs_spec = {
    'robot': (-13, -12, -11, -4), # x, y, z, rotation
    'obj': (-10, -9, -8),
    'goals': [(-7, -6, -5)],
    'obstacle': (-3, -2, -1) # x, y, rotation
}

default_mobp_obs_spec = {
    'robot': (-19, -18, -17, -4), # x, y, z, rotation
    'obj': (-16, -15, -14),
    'goals': [(-13, -12, -11), (-10, -9, -8), (-7, -6, -5)],
    'obstacle': (-3, -2, -1) # x, y, rotation
}

default_ac_spec = {
    'x': 0,
    'y': 1,
    'z': 2,
    'a': 3
}

def save_video(save_path, frames, fps=10.0):
    # frames = np.array(list(map(lambda frame: cv2.resize(frame, (64, 64), interpolation=cv2.INTER_CUBIC), frames)))
    video = mpy.VideoClip(lambda t: frames[int(t * fps)],
                          duration=float(frames.shape[0] / fps))
    video.write_videofile(save_path, fps, verbose=False, progress_bar=False)

class Direction(Enum):
    UP = 1
    LEFT = 2
    DOWN = 3
    RIGHT = 4

class RuleBasedPusher(object):
    def __init__(self, args, obs_spec, ac_spec, output_dir):
        self.env_name = args.env_name
        self.debug = args.debug
        self.log_steps = args.log_steps
        self.seed = args.seed
        self.num_action_repeat = args.num_action_repeat
        self.max_steps = args.max_steps
        self.repush = args.repush
        self.multipush = args.env_name.startswith('Multipush')
        self.subsample_rate = args.subsample_rate

        self.prod = args.prod
        self.prod_ix = args.prod_ix
        self.prod_dump = args.prod_dump
        self.prod_output_steps = args.prod_output_steps

        self.obs_spec = obs_spec
        self.ac_spec = ac_spec
        self.output_dir = output_dir

        self.min_push_dist = args.min_push_dist
        self.max_push_dist = args.max_push_dist

        self.reset()

    def reset(self):
        self.t = 0
        self.frames = []
        self.actions = []
        self.observations = []
        self.arm_positions = []
        self.is_key_frame = []
        self.subtask_lengths = []
        self.stop = False

    # Obtain the longest distance the arm can move with a single action.
    #
    # Args:
    #     ac_dim: dimensionality of action space.
    # 
    # Returns: a floating point number representing the longest xy-plane
    #          distance the arm can move if norm(a) = 1.
    def get_arm_speed(self, ac_dim):
        self.env.reset()
        robot_positions = []
        for i in range(2):
            ac = np.zeros((ac_dim,))
            ac[self.ac_spec['y']] = 1.0
            debug_save = self.debug
            self.debug = False
            pos = self.step(ac)
            self.debug = debug_save
            robot_positions.append(pos['robot'][:2])
        return norm(robot_positions[1] - robot_positions[0])

    # Calculates ratio between the lengths of two parts of object.
    #
    # Args:
    #     pos: dictionary with positions of all parts in the scene.
    #     ix: index of object to query.
    #
    # Returns: a floating point number representing the aspect ratio of the 
    #          specified object.
    def get_aspect_ratio(self, pos, ix):
        width = norm(pos['obj_part2'][ix][:2] - pos['obj_joint'][ix][:2])
        height = norm(pos['obj_part1'][ix][:2] - pos['obj_joint'][ix][:2])
        return height / width

    def get_pushing_direction(self, obj_x_direction, goal_direction, 
                              obj_aspect_ratio):
        angle = angle_between(obj_x_direction, goal_direction)
        diagonal_angle = math.atan(obj_aspect_ratio)
        if angle < diagonal_angle or angle > math.pi * 2 - diagonal_angle:
            return Direction.LEFT
        elif angle < math.pi - diagonal_angle:
            return Direction.DOWN
        elif angle < math.pi + diagonal_angle:
            return Direction.RIGHT
        else:
            return Direction.UP

    def sample_pushing_start_pos(self, pos, ix, direction):
        pos_joint = pos['obj_joint'][ix][:2]
        pos_center = pos['obj_part1'][ix][:2] + pos['obj_part2'][ix][:2] - pos_joint
        
        # decide which object part to push
        if direction == Direction.LEFT or direction == Direction.RIGHT:
            pos_part = pos['obj_part1'][ix][:2]
        else:
            pos_part = pos['obj_part2'][ix][:2]

        pos_end = 2 * pos_part - pos_joint
        pos_center_reflect = 2 * pos_part - pos_center
        
        # randomly sample a position on that part to push
        z = np.random.rand()
        
        obj_ref_point = pos_joint * z + pos_end * (1.0 - z)
        
        # the number 0.5 is selected so that when object reaches this pushing 
        # start position and lowers down to the ground, it will not get stuck
        # to the top of the object.
        if direction == Direction.LEFT or direction == Direction.DOWN:
            # push from outside the L-shaped object
            # center relect is the point that is reflected from center wrt. the object part
            return obj_ref_point - 0.5 * (obj_ref_point - pos_center_reflect)
        else:
            # push from inside the L-shaped object
            return obj_ref_point - 0.5 * (obj_ref_point - pos_center)

    def sample_point_where_object_passes_through_obstacle(self, pos):
       obstacle_center = pos['obstacle'][:2]
       obstacle_angle = pos['obstacle'][-1]
       obstacle_direction = np.array([np.cos(obstacle_angle), np.sin(obstacle_angle)])
       while True:
           rand_dist = np.random.uniform(low=0.6, high=0.9)
           sign = 1.0 if np.random.rand() > 0.5 else -1.0
           intermediate_position = obstacle_center + sign * rand_dist * obstacle_direction
           if self.point_on_table(intermediate_position):
               break
       return intermediate_position

    def sample_subgoals(self, pos, start, end, side='front', num_pushes=3):
        obstacle_center = pos['obstacle'][:2]
        obstacle_angle = pos['obstacle'][-1]

        if num_pushes == 3:
            num_loop_execution = 0
            while True:
                num_loop_execution += 1
                if num_loop_execution > 1000:
                    self.stop = True
                    return None

                # sample first subgoal position
                first_subgoal_dist = np.random.uniform(low=self.min_push_dist, high=self.max_push_dist)
                first_subgoal_angle = np.random.uniform(low=0, high=np.pi * 2)
                first_subgoal_pos = offset_point(start, first_subgoal_dist, first_subgoal_angle)

                # check requirements of first subgoal generation
                if not (
                    self.point_on_table(first_subgoal_pos) \
                        and self.point_on_one_side_of_obstacle(first_subgoal_pos, 
                        obstacle_center, obstacle_angle, side=side, margin=0.2)):
                    continue

                # draw toruses and sample second subgoal position
                second_subgoal_torus = draw_torus(pt2pix(first_subgoal_pos),
                                                  self.min_push_dist,
                                                  self.max_push_dist)
                third_subgoal_torus = draw_torus(pt2pix(end),
                                                 self.min_push_dist,
                                                 self.max_push_dist)
                intersection = compute_intersection([second_subgoal_torus, 
                                                     third_subgoal_torus])
                sampled_intersection_pixel = sample_from_intersect(intersection)
                
                if sampled_intersection_pixel is None:
                    continue

                second_subgoal_pos = pix2pt(sampled_intersection_pixel)
               
                # check requirements of second subgoal generation
                if not (
                    self.point_on_table(second_subgoal_pos) \
                        and self.point_on_one_side_of_obstacle(second_subgoal_pos,
                        obstacle_center, obstacle_angle, side=side, margin=0.2)):
                    continue

                break

            subgoals = [first_subgoal_pos, second_subgoal_pos]
            
        else:
            raise NotImplementedError("Subgoal sampling with number of pushes other than 3 is not implemented.")

        return subgoals
            
    def get_goal_direction(self, pos, obj_key, goal_pos):
        if type(obj_key) is str:
            # single level key
            obj_pos = pos[obj_key][:2]
        else:
            # multiple level key
            obj_pos = pos
            for key in obj_key:
                obj_pos = obj_pos[key]
            obj_pos = obj_pos[:2]

        return goal_pos - obj_pos

    def get_pos(self, obs):
        ix2obs = lambda ix: obs[np.array(ix)]
        pos = dict((k, ix2obs(v)) for k, v in self.obs_spec.items())
        return pos

    def get_arm_angle(self, pos):
        if len(pos['robot']) < 4:
            return None
        else:
            angle = pos['robot'][-1]

            # convert to [-pi, pi)
            while angle >= np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += np.pi

            return angle

    def point_on_table(self, point, margin=0.2):
        return np.max(np.abs(point)) < 1.0 - margin

    # Check whether a point is on one side of obstacle.
    #
    # Args:
    #     point: the point to check
    #     obstacle_pos: position of the obstacle
    #     obstacle_angle: angle of the obstacle
    #     side: 'front' if object not passed obstacle, 'back' otherwise
    #     margin: how long distance point has to be away from obstacle
    def point_on_one_side_of_obstacle(self, point, 
                                      obstacle_pos, obstacle_angle,
                                      side='front', margin=0.1):
        point_relative_angle = angle_between([np.cos(obstacle_angle), np.sin(obstacle_angle)], 
                                             point - obstacle_pos)
        point_distance = np.abs(np.linalg.norm(point - obstacle_pos) * np.sin(point_relative_angle))
        if side == 'front':
            return point_distance > margin and point_relative_angle < np.pi
        else:
            return point_distance > margin and point_relative_angle > np.pi

    def push_complete(self, init_direction, curr_direction):
        if norm(curr_direction) < 0.01:
            return True

        if np.dot(init_direction, curr_direction / norm(curr_direction)) < 0:
            # pushed a bit too far away, success
            return True

        return False

    def step(self, ac):
        self.actions.append(ac)
        self.is_key_frame.append(0)

        obs, rew, done, info = self.env.step(ac)
        pos = self.get_pos(obs)
        self.observations.append(obs)
        self.arm_positions.append(pos['robot'])
        self.current_reward = rew

        if self.debug:
            print('Step {:5d}: rew = {:.3f}'.format(self.t + 1, rew))
        
        rendered_frame = self.env.render(mode='rgb_array')
        self.frames.append(rendered_frame)

        if self.debug and not self.prod:
            cv2.imwrite(osp.join(self.output_dir, "latest.jpg"),
                        cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB))

            if self.t % self.log_steps == 0:
                save_video(osp.join(self.output_dir, 
                                    "traj_{:03d}.mp4".format(self.t)), 
                           np.stack(self.frames))
        
        self.t += 1

        if self.t >= self.max_steps:
            self.stop = True

        return pos

    # Extend a numpy array to prod_output_steps if it's short and clip if it's
    # too long. Used in production setting only.
    def align(self, arr, fill='repeat', length=None):
        if length is None:
            length = self.prod_output_steps
        if arr.shape[0] < length:
            if fill == 'repeat':
                repeat_vec = np.ones((arr.shape[0],), dtype=np.int64)
                repeat_vec[-1] = length - (arr.shape[0] - 1)
                ret = np.repeat(arr, repeat_vec, axis=0)
            elif fill == 'zero':
                fill_arr = np.zeros((length - arr.shape[0], ) + arr.shape[1:])
                ret = np.concatenate((arr, fill_arr), axis=0)
            else:
                raise NotImplementedError()
        else:
            ret = arr[:length]
        return ret

    def run(self):
        env_name = self.env_name
        # use obstacle push environment for multipush experiments
        if env_name.startswith('Multipush'): env_name = env_name[len('Multipush'):]
        elif env_name.startswith('ControlledLength'): env_name = env_name[len('ControlledLength'):]
        elif env_name.startswith('Random'): env_name = env_name[len('Random'):]
        self.env = ActionRepeat(gym.make(env_name), self.num_action_repeat)
        self.ac_dim = self.env.action_space.shape[0]
        self.arm_speed = self.get_arm_speed(self.ac_dim)
        self.env.seed(self.seed)
        np.random.seed(self.seed)

        if env_name.startswith('Random'): self.env.init_arm_near_object = True
        obs = self.env.reset()

        # The purpose of this line of code is just to make gym print its
        # command line output for rendering gym environments and not print them
        # between rollout.
        test_frame = self.env.render(mode='rgb_array')
        
        # save goal image if environment allows it
        if hasattr(self.env.unwrapped, 'get_goal_image'):
            self.goal_image = self.env.unwrapped.get_goal_image()
        else:
            self.goal_image = None

        pos = self.get_pos(obs)
        self.robot_starting_z = pos['robot'][2]

        if self.env_name == 'ObstaclePush-v0' \
            or self.env_name == 'MultigoalObstaclePush-v0' \
            or self.env_name == 'MultipushObstaclePush-v0':
            self.run_obstacle_push(pos)
        elif self.env_name == 'ControlledLengthObstaclePush-v0':
            self.run_controlled_length_obstacle_push(pos)
        elif self.env_name == 'RandomObstaclePush-v0':
            self.run_rand_obstacle_push()
        else:
            self.run_multi_object_push(pos)

        self.save_to_files()

    def run_rand_obstacle_push(self):
        self.reset()

        num_steps_to_run = self.max_steps
        if self.prod:
            num_steps_to_run = self.prod_output_steps * self.subsample_rate

        for i in range(num_steps_to_run):
            ac = np.random.rand(self.ac_dim) * 2 - 1
            ac[self.ac_spec['z']] = 0.0
            ac[self.ac_spec['a']] = 0.0
            self.step(ac)

        # rollout always succeeds
        self.stop = False

    def run_controlled_length_obstacle_push(self, pos):
        self.reset()

        # sample point where object passes through the obstacle
        intermediate_position = self.sample_point_where_object_passes_through_obstacle(pos)

        # sample subgoals before reaching the obstacle
        subgoals_before_obstacle = self.sample_subgoals(pos, pos['obj'][:2], intermediate_position, side='front')
        if self.stop: return

        # sample subgoals after reaching the obstacle
        subgoals_after_obstacle = self.sample_subgoals(pos, intermediate_position, pos['goals'][0][:2], side='back')
        if self.stop: return

        # combine subgoals
        subgoals = subgoals_before_obstacle + [intermediate_position] + subgoals_after_obstacle + [pos['goals'][0][:2]]

        subtask_ix = 0
        for subgoal_pos in subgoals:
            push_direction = subgoal_pos - pos['obj'][:2]
            push_direction /= np.linalg.norm(push_direction)
            push_start_pos = pos['obj'][:2] - push_direction * 0.22
            pos = self.run_push_subtask(subtask_ix, push_start_pos, 'obj', subgoal_pos)
            subtask_ix += 1

            if self.stop: return

    def run_obstacle_push(self, pos):
        # preparation
        num_goals = len(pos['goals'])
        obstacle_center = pos['obstacle'][:2]
        obstacle_angle = pos['obstacle'][-1]

        if self.debug:
            print('There are a total of {} goals.'.format(num_goals))

        self.reset()
            
        # get information of whether goals are in front of or behind obstacle
        # insert False at front to denote start position to simplify indexing
        if num_goals > 1:
            goal_behind_wall = [False] + self.env.unwrapped.goal_behind_wall

        # randomly select goals one at a time and push towards the goal
        previous_ix = -1
        for ix in np.random.permutation(num_goals):
            if self.debug:
                print('----- Task {} -----'.format(ix + 1))

            # sample position for passing through the obstacle
            intermediate_position = self.sample_point_where_object_passes_through_obstacle(pos)

            if self.multipush:
                # sample number of pushes
                num_pushes = 6 # np.random.randint(low=2, high=5)

                # sample zero-indexed push index that reaches position
                # that passes through the obstacle
                # for example, if num_pushes is 4, then eligible indices are
                # 0, 1, and 2 (cannot be last push).
                index_of_push_that_reaches_obstacle = np.random.randint(num_pushes - 1)
                num_subtask_front = index_of_push_that_reaches_obstacle
                num_subtask_back = num_pushes - 2 - num_subtask_front

            subtask_ix = 0

            if self.multipush:
                # add subtasks in front of the obstacle
                for _ in range(num_subtask_front):
                    while True:
                        subgoal_dist = np.random.uniform(low=0.3, high=1.0)
                        subgoal_angle = np.random.uniform(low=np.pi / 4, high=np.pi * 3 / 4)
                        subgoal_pos = offset_point(obstacle_center, subgoal_dist, subgoal_angle + obstacle_angle)

                        dist_start = np.linalg.norm(intermediate_position - pos['obj'][:2])
                        dist_subgoal = np.linalg.norm(intermediate_position - subgoal_pos)
                    
                        if self.point_on_table(subgoal_pos, margin=0.2) and dist_subgoal < dist_start:
                            break
            
                    push_direction = subgoal_pos - pos['obj'][:2]
                    push_direction /= np.linalg.norm(push_direction)
                    push_start_pos = pos['obj'][:2] - push_direction * 0.22
                    pos = self.run_push_subtask(subtask_ix, push_start_pos, 'obj', subgoal_pos)
                    subtask_ix += 1

                    if self.stop: return

            # in single-goal environment, always have intermediate position
            # in multi-goal environment, only push through wall when next goal is 
            #     at different side of the wall
            if num_goals == 1 or goal_behind_wall[previous_ix + 1] != goal_behind_wall[ix + 1]:
                # stage 1: push to intermediate position
                push_direction = intermediate_position - pos['obj'][:2]
                push_direction /= np.linalg.norm(push_direction)
                push_start_pos = pos['obj'][:2] - push_direction * 0.22
                pos = self.run_push_subtask(subtask_ix, push_start_pos, 'obj', intermediate_position)
                subtask_ix += 1

                if self.stop: return

            if self.multipush:
                # add subtasks behind the obstacle
                for _ in range(num_subtask_back):
                    while True:
                        subgoal_dist = np.random.uniform(low=0.3, high=1.0)
                        subgoal_angle = np.random.uniform(low=np.pi * 5 / 4, high=np.pi * 7 / 4)
                        subgoal_pos = offset_point(obstacle_center, subgoal_dist, subgoal_angle + obstacle_angle)
                        
                        dist_start = np.linalg.norm(pos['goals'][ix][:2] - pos['obj'][:2])
                        dist_subgoal = np.linalg.norm(pos['goals'][ix][:2] - subgoal_pos)
                        
                        if self.point_on_table(subgoal_pos, margin=0.2) and dist_subgoal < dist_start:
                            break
            
                    push_direction = subgoal_pos - pos['obj'][:2]
                    push_direction /= np.linalg.norm(push_direction)
                    push_start_pos = pos['obj'][:2] - push_direction * 0.22
                    pos = self.run_push_subtask(subtask_ix, push_start_pos, 'obj', subgoal_pos)
                    subtask_ix += 1

                    if self.stop: return

            # stage 2: push to goal
            push_direction = pos['goals'][ix][:2] - pos['obj'][:2]
            push_direction /= np.linalg.norm(push_direction)
            push_start_pos = pos['obj'][:2] - push_direction * 0.22
            pos = self.run_push_subtask(subtask_ix, push_start_pos, 'obj', pos['goals'][ix][:2])
            subtask_ix += 1

            if self.stop: return

            previous_ix = ix

    def run_multi_object_push(self, pos):
        num_objects = len(pos['obj_joint'])
        if self.debug:
            print('there are a total of {} object(s) to push.'.format(num_objects))

        self.reset()

        for ix in range(num_objects):
            pos_joint = pos['obj_joint'][ix][:2]
            pos_part1 = pos['obj_part1'][ix][:2]
            pos_part2 = pos['obj_part2'][ix][:2]
            pos_goal = pos['goal_joint'][ix][:2]

            obj_x_direction = pos_part2 - pos_joint
            goal_direction = pos_goal - pos_joint
            
            push_direction = self.get_pushing_direction(obj_x_direction, 
                                                        goal_direction,
                                                        self.get_aspect_ratio(pos, ix))
            push_start_pos = self.sample_pushing_start_pos(pos, ix, 
                                                           push_direction)

            pos = self.run_push_subtask(ix, push_start_pos, ('obj_joint', ix), pos_goal)

            if self.stop: break
    
    def run_push_subtask(self, ix, push_start_pos, obj_key, goal_pos, angle_aware=False):
        phase = 0
        start_t = self.t

        # raise arm to air {{{
        phase += 1

        if self.debug:
            print('subtask {} phase {}: raise arm to air.'.format(ix + 1, phase))

        # number 0.2 is selected so the raised distance is more than 
        # the height of the object to be pushed
        robot_z = self.robot_starting_z
        while robot_z < self.robot_starting_z + 0.2: 
            ac = np.zeros((self.ac_dim,))
            ac[self.ac_spec['z']] = 1.0
            pos = self.step(ac)
            robot_z = pos['robot'][2]
            if self.stop: return pos
        # }}}

        if self.stop: return pos

        # [patch] the purpose of this while loop is to repush when the object
        # is stuck on the top of an object
        need_push = True
        while need_push:
            need_push = False
            
            # move arm to starting position {{{
            phase += 1

            if self.debug:
                print('subtask {} phase {}: move arm behind object.'.format(ix + 1, phase))

            pos_robot = pos['robot'][:2]
            init_direction = push_start_pos - pos_robot
            while not self.push_complete(init_direction, push_start_pos - pos_robot):
                move_dir = push_start_pos - pos_robot
                if norm(move_dir) > self.arm_speed:
                    move_dir /= norm(move_dir)
                else:
                    # don't move full step if arm is close
                    move_dir /= self.arm_speed
                ac = np.zeros((self.ac_dim,))
                ac[[self.ac_spec['x'], self.ac_spec['y']]] = move_dir
                ac[self.ac_spec['z']] = 0.2 # slight upward force to prevent dropping
                pos = self.step(ac)
                pos_robot = pos['robot'][:2]
                if self.stop: return pos
            # }}}

            if self.stop: return pos

            # lower arm to ground {{{
            phase += 1

            if self.debug:
                print('subtask {} phase {}: lower arm to ground.'.format(ix + 1, phase))

            # [patch] keep record of timesteps already spent on lowering the arm
            timesteps_spent_lowering_arm = 0
            
            robot_z = pos['robot'][2]
            while robot_z > self.robot_starting_z:
                ac = np.zeros((self.ac_dim,))
                ac[self.ac_spec['z']] = -1.0
                pos = self.step(ac)
                robot_z = pos['robot'][2]
                if self.stop: return pos
                    
                # [patch] if arm lowering didn't complete in 5 timesteps,
                # then the arm is almost surely stuck, so we break and repush
                timesteps_spent_lowering_arm += 1
                if timesteps_spent_lowering_arm >= 5:
                    if self.repush:
                        need_push = True
                    else:
                        self.stop = True
                    break
            # }}}

            if self.stop: return pos
        
        if self.stop: return pos

        # rotate arm to correct direction if needed {{{
        init_direction = self.get_goal_direction(pos, obj_key, goal_pos)
        if angle_aware:
            phase += 1

            if self.debug:
                print('subtask {} phase {}: rotate arm to correct direction.'.format(ix + 1, phase))

            # [patch] keep record of timesteps already spent on rotating the arm
            timesteps_spent_rotating_arm = 0

            goal_direction_angle = np.arctan2(init_direction[1], init_direction[0])
            angle_error = clip_angle_npi_to_pi(goal_direction_angle - self.get_arm_angle(pos))
            while np.abs(angle_error) > 0.2:
                ac = np.zeros((self.ac_dim,))
                if angle_error > 0:
                    ac[self.ac_spec['a']] = 1.0
                else:
                    ac[self.ac_spec['a']] = -1.0
                pos = self.step(ac)
                angle_error = clip_angle_npi_to_pi(goal_direction_angle - self.get_arm_angle(pos))
                if self.stop: return pos

                # [patch] if arm rotation didn't complete in 20 timesteps,
                # then the arm is almost surely stuck, so we break
                timesteps_spent_rotating_arm += 1
                if timesteps_spent_rotating_arm >= 20:
                    self.stop = True
                    break
        # }}}

        if self.stop: return pos

        # push object until goal is reached {{{
        phase += 1

        if self.debug:
            print('subtask {} phase {}: push object until goal is reached.'.format(ix + 1, phase))

        while not self.push_complete(init_direction, self.get_goal_direction(pos, obj_key, goal_pos)):
            move_dir = self.get_goal_direction(pos, obj_key, goal_pos)
            if norm(move_dir) > self.arm_speed:
                move_dir /= norm(move_dir)
            else:
                move_dir /= self.arm_speed
            ac = np.zeros((self.ac_dim,))
            ac[[self.ac_spec['x'], self.ac_spec['y']]] = move_dir
            pos = self.step(ac)
            if self.stop: return pos
        # }}}

        self.is_key_frame[-1] = 1
        self.subtask_lengths.append(self.t - start_t)

        return pos

    def save_to_files(self):
        if len(self.frames) > 0:
            self.frames = np.stack(self.frames)
            self.arm_positions = np.stack(self.arm_positions)
            self.observations = np.stack(self.observations)
            self.actions = np.stack(self.actions)
            self.is_key_frame = np.array(self.is_key_frame)
    
            # perform subsampling if needed
            if self.subsample_rate != 1:
                assert(self.t == len(self.frames))

                # pad to multiple of subsample rate
                ss_rate = self.subsample_rate
                padded_length = int(np.ceil(self.t / ss_rate) * ss_rate)
                self.frames = self.align(self.frames, length=padded_length)
                self.arm_positions = self.align(self.arm_positions, length=padded_length)
                self.observations = self.align(self.observations, length=padded_length)
                self.actions = self.align(self.actions, fill='zero', length=padded_length)
                self.is_key_frame = self.align(self.is_key_frame, fill='zero', length=padded_length).astype(int)
    
                # update sequence length
                ss_len = padded_length // ss_rate
                self.t = ss_len
    
                # perform subsampling
                self.frames = self.frames[(ss_rate - 1)::ss_rate]
                self.arm_positions = self.arm_positions[(ss_rate - 1)::ss_rate]
                self.observations = self.observations[(ss_rate - 1)::ss_rate]
                self.actions = np.array(self.actions)\
                                   .reshape((-1, ss_rate, self.ac_dim))\
                                   .transpose((0, 2, 1))\
                                   .mean(axis=2)
                self.is_key_frame = (np.sum(np.array(self.is_key_frame)\
                                    .reshape((-1, ss_rate)), axis=1) != 0)\
                                    .astype(int)
                self.subtask_lengths = np.floor_divide(self.subtask_lengths, ss_rate)
        
        if self.prod:
            ret = (not self.stop, self.t,
                   self.current_reward,
                   self.prod_output_steps)

            if len(self.frames) > 0:
                aligned_key_frame = self.align(self.is_key_frame, fill='zero').astype(int)
                if self.frames.shape[0] > self.prod_output_steps:
                    aligned_key_frame[-1] = 1
                ret += (
                   self.align(self.frames),
                   self.align(self.arm_positions),
                   self.align(self.observations),
                   self.align(self.actions, fill='zero'),
                   aligned_key_frame,
                   self.subtask_lengths)
            else:
                ret += ([],)

            if self.goal_image is not None:
                ret += (self.goal_image,)

            pickle.dump(ret, open(osp.join(self.output_dir, 
                    "traj_{:05d}.p".format(self.prod_ix)), 'wb'))

            if (not self.prod_dump) and len(self.frames) > 0:
                save_video(osp.join(self.output_dir, "traj_{:05d}.mp4".format(self.prod_ix)), 
                        np.stack(self.frames))

            if self.stop:
                print('traj #{:05d} - failed with reward {:.3f}.'.format(
                    self.prod_ix, self.current_reward))
            else:
                print('traj #{:05d} - succeeded in {:d} steps with reward {:.3f}'.format(
                    self.prod_ix, self.t, self.current_reward))
        else:
            if len(self.frames) > 0:
                save_video(osp.join(self.output_dir, "traj_complete.mp4"), 
                        np.stack(self.frames))

def parse_args(default_only=False):
    parser = argparse.ArgumentParser()

    # exp
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--subsample_rate', type=int, default=1)

    # env
    parser.add_argument('--env_name', type=str, default='SingleObjectPush-v0')
    parser.add_argument('--num_action_repeat', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    # algorithm
    parser.add_argument('--repush', action='store_true', default=False)
    parser.add_argument('--min_push_dist', type=float, default=0.4)
    parser.add_argument('--max_push_dist', type=float, default=0.6)

    # production
    parser.add_argument('--prod', action='store_true', default=False)
    parser.add_argument('--prod_ix', type=int, default=0)
    parser.add_argument('--prod_dir', type=str, default='output')
    parser.add_argument('--prod_dump', action='store_true', default=False)
    parser.add_argument('--prod_output_steps', type=int, default=None)

    if default_only:
        return parser.parse_args([])
    else:
        return parser.parse_args()

def main():
    gym.logger.set_level(gym.logger.ERROR)

    args = parse_args()

    if args.env_name == 'SingleObjectPush-v0':
        obs_spec = default_sop_obs_spec
        ac_spec = default_ac_spec
    elif args.env_name == 'TwoObjectPush-v0':
        obs_spec = default_top_obs_spec
        ac_spec = default_ac_spec
    elif args.env_name == 'ObstaclePush-v0' or args.env_name == 'MultipushObstaclePush-v0' \
            or args.env_name == 'ControlledLengthObstaclePush-v0' \
            or args.env_name == 'RandomObstaclePush-v0':
        obs_spec = default_obp_obs_spec
        ac_spec = default_ac_spec
    elif args.env_name == 'MultigoalObstaclePush-v0':
        obs_spec = default_mobp_obs_spec
        ac_spec = default_ac_spec
    else:
        raise NotImplementedError()

    if args.prod:
        output_dir = args.prod_dir
    else:
        output_dir = 'logs/{}_{}_seed-{}_acrep-{}_{}'.format(
                        args.env_name, args.prefix, args.seed,
                        args.num_action_repeat,
                        time.strftime("%y%m%d-%h%m%s"))

    if not osp.exists(output_dir): os.makedirs(output_dir)
    if not args.prod:
        print('output directory: {}'.format(output_dir))

    rbp = RuleBasedPusher(args, obs_spec, ac_spec, output_dir)

    rbp.run()

if __name__ == '__main__':
    main()
