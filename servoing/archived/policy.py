from abc import abstractmethod
import numpy as np


class Policy(object):
    def __init__(self,
                 observation_space,
                 action_space):
        self._input_size = observation_space.shape[0]
        self._output_size = action_space.shape[0]

    @abstractmethod
    def act(self, observations):
        """Return action based on observations."""
        return


class RandomContinuousPolicy(Policy):
    def __init__(self,
                 observation_space,
                 action_space,
                 max_action,
                 only_forward=False):
        super(RandomContinuousPolicy, self).__init__(observation_space,
                                                     action_space)
        self._max_action = max_action
        self._only_forward = only_forward

    def act(self, observations):
        if self._only_forward:
            return np.random.rand(self._output_size) * self._max_action
        else:
            return (np.random.rand(self._output_size)-0.5) * 2 * self._max_action


class ConstantContinuousPolicy(object):
    def __init__(self,
                 observation_space,
                 action_space,
                 constant):
        super(ConstantContinuousPolicy, self).__init__(observation_space,
                                                       action_space)
        self._constant = constant

    def act(self, observations):
        output = np.empty(self._output_size)
        output[:] = self._constant
        return output