'''
Created by Jingyun Yang on 12/10/2018.

Wrapper for gym environment that performs action repeat for every env.step()
call.
'''

import gym
import gym_push

class ActionRepeat(gym.Wrapper):
    def __init__(self, env, num_action_repeat):
        super(ActionRepeat, self).__init__(env)
        self.num_action_repeat = num_action_repeat

    def step(self, action):
        for i in range(self.num_action_repeat):
            observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info
