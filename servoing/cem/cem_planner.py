"""Core CEM code that executes simulator, fits Gaussian action distributions and iterates."""

import collections
import numpy as np
from tqdm import trange

from utils import AttrDict


CEMParams = collections.namedtuple(
    "CEMParams",
    "prefix "
    "planning_horizon "
    "cem_default_iter "
    "cem_default_batch_size "
    "elite_frac "
    "initial_variance "
    "action_dim "
    "action_clip_val "
    "reinit_cem_dists "
)


class CEMPlanner(object):
    def __init__(self, cem_params, simulator, distance_metric):
        self.params = cem_params
        self.simulator = simulator
        self.distance_metric = distance_metric

    def _init_action_dists(self, prev_action_dists):
        if prev_action_dists is None or self.params.reinit_cem_dists:
            action_dists = AttrDict(
                mean=np.zeros((self.params.planning_horizon, self.params.action_dim), dtype=float)
            )
        else:
            # step previous action distributions and only append last step with initialization params
            action_dists = AttrDict(
                mean=np.concatenate((prev_action_dists.mean[1:],
                                        np.zeros((1, self.params.action_dim), dtype=float)), axis=0)
            )
        action_dists.var = self.params.initial_variance * \
                       np.ones((self.params.planning_horizon, self.params.action_dim), dtype=float)
        return action_dists

    @staticmethod
    def _sample_gaussian_and_clip(dists, batch_size, action_clip_val):
        assert dists.mean.shape == dists.var.shape, "Mean and variance should have same dimensions!"
        num_frames, action_dim = dists.mean.shape
        normal = np.random.normal(0, 1, [batch_size, num_frames, action_dim])
        sampled_values = normal * np.sqrt(dists.var) + dists.mean
        if action_clip_val is not None:
            sampled_values = np.clip(sampled_values, -action_clip_val, action_clip_val)
        return sampled_values

    def plan_single_step(self, start_img, goal_img, prev_action_dists, cem_iter=None, cem_batch_size=None):
        # prepare the initial action dists
        action_dists = self._init_action_dists(prev_action_dists)
        elite_output_actions_all_iters = []
        elite_frames_all_iters = []

        for cem_iter in trange(cem_iter if cem_iter is not None else self.params.cem_default_iter,
                               desc="CEM" + ((" (" + self.params.prefix + ")") if len(self.params.prefix) > 0 else "")):
            # sample actions of shape (bs, horizon, ac_dim)
            actions = self._sample_gaussian_and_clip(action_dists, 
                                                     cem_batch_size if cem_batch_size is not None else self.params.cem_default_batch_size,
                                                     self.params.action_clip_val)

            # rollout simulator to produce predicted frames with shape (bs, horizon, h, w, c) 
            # frames have uint8 type and value range [0, 255]
            sim_outputs = self.simulator.rollout(start_img, actions)

            # compute distance metric with shape (bs,)
            distances = self.distance_metric(sim_outputs, goal_img)

            # sort outputs wrt to distance metric
            elite_indices = distances.argsort()[:int(len(distances) * self.params.elite_frac)]
            elite_actions = actions[elite_indices] # shape (bs * elite_frac, horizon, ac_dim)
            elite_output_actions = sim_outputs.pred_actions[elite_indices] if "pred_actions" in sim_outputs \
                                else actions[elite_indices]
            elite_frames = sim_outputs.pred_frames[elite_indices]
            elite_distances = distances[elite_indices]

            # record parameters that needs to be recorded for every iteration
            elite_output_actions_all_iters.append(elite_output_actions)
            elite_frames_all_iters.append(elite_frames)

            # refit actions
            action_dists.mean = np.mean(elite_actions, axis=0)
            action_dists.var = np.var(elite_actions, axis=0)

        # prepare outputs
        outputs = AttrDict(
            elite_actions=elite_output_actions,
            elite_frames=elite_frames,
            elite_actions_all_iters=elite_output_actions_all_iters,
            elite_frames_all_iters=elite_frames_all_iters,
            elite_distances=elite_distances,
            action_dists=action_dists,
            )
        return outputs

