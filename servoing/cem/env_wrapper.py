"""Wrapper for the environment class that handles all interactions with env."""

from __future__ import absolute_import

import gym
import gym_push
import numpy as np
import cv2

from servoing.cem.action_repeat import ActionRepeat


class EnvWrapper(object):
    def __init__(self):
        pass

    ''' Sample an initial state for a specified environment given a seed. '''

    @staticmethod
    def setup_environment(env_name, seed, init_arm_near_object=False):
        env = gym.make(env_name)
        env.seed(seed)
        np.random.seed(seed)
        if init_arm_near_object: env.init_arm_near_object = True
        env.reset()
        init_state = env.unwrapped.sim.get_state()

        print('observation space:', env.observation_space)
        print('action space:', env.action_space)
        print('  - low:', env.action_space.low)
        print('  - high:', env.action_space.high)
        print()

        env.close()
        return init_state

    @staticmethod
    def make_env_from_state(env_name, env_state, num_action_repeat):
        env = ActionRepeat(gym.make(env_name), num_action_repeat)
        env.reset()
        env.unwrapped.sim.set_state(env_state)
        env.unwrapped.sim.forward()
        return env

    @staticmethod
    def get_start_goal_img(env_name, env_state, num_action_repeat, resolution,
                           high_level_arm_centered, display_goal=False):
        # build environment
        env = ActionRepeat(gym.make(env_name), num_action_repeat)
        env.reset()
        env.unwrapped.sim.set_state(env_state)
        env.unwrapped.sim.forward()

        # get current and goal image
        ll_start_image_original = env.render(mode='rgb_array')
        hl_start_image = env.unwrapped.get_centered_arm_image() if high_level_arm_centered else ll_start_image
        goal_image = env.unwrapped.get_goal_image()

        ll_start_image = cv2.resize(ll_start_image_original,
                                   (resolution, resolution),
                                   interpolation=cv2.INTER_CUBIC)
        hl_start_image = cv2.resize(hl_start_image,
                                   (resolution, resolution),
                                   interpolation=cv2.INTER_CUBIC)
        goal_image = cv2.resize(goal_image,
                                (resolution, resolution),
                                interpolation=cv2.INTER_CUBIC)

        if display_goal:
            ll_start_image_original = env.unwrapped.get_goal_displayed_image()

        env.close()
        return hl_start_image, ll_start_image, goal_image, ll_start_image_original

    @staticmethod
    def get_action_dim(env_name):
        env = gym.make(env_name)
        ac_dim = env.action_space.shape[0]
        env.close()
        return ac_dim

    @staticmethod
    def step_action(prev_state, action, env_name, seed, num_action_repeat, resolution, display_goal=False):
        env = ActionRepeat(gym.make(env_name), num_action_repeat)
        env.seed(seed)
        np.random.seed(seed)
        env.reset()
        env.unwrapped.sim.set_state(prev_state)
        env.unwrapped.sim.forward()

        obs, rew, done, info = env.step(action)

        if done:
            raise ValueError("The simulation should never be done!")

        curr_state = env.unwrapped.sim.get_state()
        original_frame = env.render(mode='rgb_array')
        frame = cv2.resize(original_frame,
                           (resolution, resolution),
                           interpolation=cv2.INTER_CUBIC)

        if display_goal:
            original_frame = env.unwrapped.get_goal_displayed_image()

        env.close()

        return curr_state, frame, original_frame, obs

    @staticmethod
    def render(env, render_goal=False):
        if render_goal:
            return env.unwrapped.get_goal_displayed_image()
        else:
            return env.render(mode="rgb_array")
