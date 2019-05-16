'''
Co-created by Jingyun Yang and Karl Pertsch on 01/28/2019.

Hierarchical CEM planning. This code will only
work with MuJoCo environments.
'''

from __future__ import absolute_import

import tensorflow as tf
from tqdm import trange

from servoing.cem.vis_utils import CEMLogger
from servoing.cem.env_wrapper import EnvWrapper
from servoing.cem.simulator import NNSimulator, GTSimulator, GTSimParams
from servoing.cem.distance_metric import get_distance_metric
from servoing.cem.cem_planner import CEMParams, CEMPlanner
from run import ModelParams
import params


class GoalServer(object):
    """Decides which goals to serve at which time step."""
    def __init__(self, subgoal_list, final_goal, max_subgoal_steps, dist_metric, subgoal_reached_dist):
        self._subgoal_list = subgoal_list
        self._final_goal = final_goal
        self._max_subgoal_steps = max_subgoal_steps
        self._dist_metric = dist_metric
        self._subgoal_reached_dist = subgoal_reached_dist
        self._current_subgoal_pt = 0
        self._current_subgoal_steps = 0

    def get_goal(self, current_img):
        final_goal_dist = self._dist_metric.run(current_img, self._final_goal)
        if final_goal_dist < self._subgoal_reached_dist:
            return None, False, True

        if self._subgoal_list is None:
            replan = self._current_subgoal_steps == 0
            self._current_subgoal_steps += 1
            return self._final_goal, replan, False

        # change goal if reached or tried to reach too long
        if self._current_subgoal_pt < len(self._subgoal_list) and \
           ((self._current_subgoal_steps >= self._max_subgoal_steps) or
            (self._dist_metric.run(current_img, self._subgoal_list[self._current_subgoal_pt]) < self._subgoal_reached_dist)):
            self._current_subgoal_pt += 1
            self._current_subgoal_steps = 0

        replan = self._current_subgoal_steps == 0       # replan each time the goal changes
        self._current_subgoal_steps += 1
        if self._current_subgoal_pt < len(self._subgoal_list):
            return self._subgoal_list[self._current_subgoal_pt], replan, False
        else:
            return self._final_goal, replan, False



def plan_subgoals(start_img, goal_img, args, logger):
    # construct NNSimulator
    hl_model_params = ModelParams(
        input_seq_len=args.high_level_input_seq_len,
        pred_seq_len=args.high_level_planning_horizon,
        batch_size=args.batch_size,
        channels=args.channels,
        input_image_shape=[args.channels, args.image_sidelength, args.image_sidelength],
        num_actions=EnvWrapper.get_action_dim(args.env_name),
        num_z=args.high_level_z_dim,
    )
    hl_simulator = NNSimulator(model_params=hl_model_params,
                               checkpoint_dir=args.high_level_checkpoint_dir,
                               network_config_name=args.high_level_config_name,
                               feed_z_sequence=True,
                               predict_warped_prob_imgs=args.high_level_dist_metric == "flow")

    # get distance metric
    dist_metric = get_distance_metric(args.high_level_dist_metric, False, None)    # do not use dense loss in high level

    # construct CEMPlanner with CEMParams object
    hl_cem_params = CEMParams(
        prefix="HL",
        planning_horizon=args.high_level_planning_horizon,
        cem_default_iter=args.warmup_num_iter,
        cem_default_batch_size=args.warmup_batch_size,
        elite_frac=args.elite_frac,
        initial_variance=args.init_var,
        action_dim=args.high_level_z_dim,
        action_clip_val=None,
        reinit_cem_dists=args.reinit_cem_dists,
    )
    hl_cem_planner = CEMPlanner(cem_params=hl_cem_params,
                                simulator=hl_simulator,
                                distance_metric=dist_metric)

    # run_cem for single step
    cem_outputs = hl_cem_planner.plan_single_step(start_img, goal_img, prev_action_dists=None)

    # log the results
    logger.log_cost_overview(goal_img, cem_outputs.elite_frames, cem_outputs.elite_distances, prefix="HL")
    logger.log_rollout_frames(start_img, cem_outputs.elite_frames, goal_img, cem_outputs.elite_actions, prefix="HL")

    # frames from best sequence == subgoals
    subgoals = cem_outputs.elite_frames[0]
    return subgoals[1:] # skip the first subgoal


def parse_args():
    return tf.flags.FLAGS


def main(unused_argv):
    args = parse_args()
    if args.infer_initial_zs:
        assert args.simulate_from_start, "Only use infer_initial_zs together with simulate_from_start!"

    logger = CEMLogger(args)
    params.dump_params(logger.basedir)

    # get distance metric
    dist_metric = get_distance_metric(args.dist_metric, args.dense_cost, args.final_step_weight)

    # setup environment + get start/goal image
    # images have shape (h, w, c) with uint8 type and value range [0, 255]
    env_init_state = EnvWrapper.setup_environment(args.env_name, args.seed, args.init_arm_near_object)
    hl_start_img, ll_start_img, goal_img, ll_start_img_original = EnvWrapper.get_start_goal_img(args.env_name,
                                                        env_init_state,
                                                        args.num_action_repeat,
                                                        args.image_sidelength,
                                                        args.high_level_arm_centered,
                                                        display_goal=args.display_goal)

    # maybe plan subgoals
    # if subgoal planning is used, a tensor of shape (high_level_planning_horizon, h, w, c)
    # is created with uint8 type and value range [0, 255]
    subgoals = plan_subgoals(hl_start_img, goal_img, args, logger) if args.include_high_level_net else None
    goal_server = GoalServer(subgoals, goal_img, args.max_subgoal_timesteps, dist_metric, args.subgoal_reached_dist)

    if args.high_level_only:
        return 

    # construct Simulator (gt or NN)
    if args.low_level_simulator_name == "gt":
        ll_sim_params = GTSimParams(
            env_name=args.env_name,
            num_action_repeat=args.num_action_repeat,
            output_resolution=args.image_sidelength,
            render_goal=False       # don't render the goal in the predicted images
        )
        ll_simulator = GTSimulator(ll_sim_params)
    elif args.low_level_simulator_name == "net":
        ll_model_params = ModelParams(
            input_seq_len=args.low_level_input_seq_len,
            pred_seq_len=args.planning_horizon,
            batch_size=args.batch_size,
            channels=args.channels,
            input_image_shape=[args.channels, args.image_sidelength, args.image_sidelength],
            num_actions=EnvWrapper.get_action_dim(args.env_name),
            num_z=None,
        )
        ll_simulator = NNSimulator(model_params=ll_model_params,
                                   checkpoint_dir=args.low_level_checkpoint_dir,
                                   network_config_name=args.low_level_config_name,
                                   feed_z_sequence=False,
                                   predict_warped_prob_imgs=args.dist_metric == "flow")
    else:
        raise ValueError("Simulator type %s not supported!" % args.low_level_simulator_name)

    # construct CEMPlanner with CEMParams object
    ll_cem_params = CEMParams(
        prefix="LL",
        planning_horizon=args.planning_horizon,
        cem_default_iter=args.num_iter,
        cem_default_batch_size=args.batch_size,
        elite_frac=args.elite_frac,
        initial_variance=args.init_var,
        action_dim=args.low_level_z_dim,
        action_clip_val=args.low_level_clipping_value,
        reinit_cem_dists=args.reinit_cem_dists,
    )
    ll_cem_planner = CEMPlanner(cem_params=ll_cem_params,
                                simulator=ll_simulator,
                                distance_metric=dist_metric)

    prev_action_dists = None
    current_frame, current_env_state = ll_start_img, env_init_state
    trajectory_images, trajectory_obs, trajectory_actions = [ll_start_img_original], [], []
    for step in trange(args.max_steps, desc='Execution'):
        start_img_i = current_frame if args.low_level_simulator_name == "net" else current_env_state
        goal_img_i, replan, finish = goal_server.get_goal(current_frame)
        if finish: break
        cem_iter = args.warmup_num_iter if replan else None
        cem_batch_size = args.warmup_batch_size if replan else None
        cem_outputs = ll_cem_planner.plan_single_step(start_img_i,
                                                      goal_img_i,
                                                      prev_action_dists,
                                                      cem_iter=cem_iter,
                                                      cem_batch_size=cem_batch_size)
        current_action = cem_outputs.elite_actions[0, 0]

        # log the results
        logger.log_cost_overview(goal_img_i,
                                 cem_outputs.elite_frames,
                                 cem_outputs.elite_distances,
                                 step=step, prefix="LL")
        logger.log_rollout_frames(current_frame,
                                  cem_outputs.elite_frames,
                                  goal_img_i,
                                  cem_outputs.elite_actions,
                                  step=step, prefix="LL")
        logger.log_plan_and_execution_trace(current_frame,
                                            cem_outputs.elite_frames[0],
                                            goal_img_i,
                                            step=step,
                                            max_step=args.max_steps,
                                            rollout_actions=cem_outputs.elite_actions[0],
                                            prefix="LL")
        for cem_iter_ix in range(len(cem_outputs.elite_frames_all_iters)): # range(cem_iters)
            logger.log_rollout_frames(current_frame,
                                      cem_outputs.elite_frames_all_iters[cem_iter_ix],
                                      goal_img_i,
                                      cem_outputs.elite_actions_all_iters[cem_iter_ix],
                                      step=step, prefix="LL_iter{:02d}".format(cem_iter_ix))

        current_env_state, current_frame, current_original_frame, current_obs = EnvWrapper.step_action(current_env_state,
                                                                               current_action,
                                                                               args.env_name,
                                                                               args.seed,
                                                                               args.num_action_repeat,
                                                                               args.image_sidelength,
                                                                               display_goal=args.display_goal)
        prev_action_dists = cem_outputs.action_dists
        trajectory_images.append(current_original_frame)
        trajectory_obs.append(current_obs)
        trajectory_actions.append(current_action)
        logger.dump_video(trajectory_images)

    # log production results
    if args.prod:
        logger.dump_production_results(prod_dir=args.prod_dir,
                                       args=tf.flags.FLAGS.__flags,
                                       init_state=env_init_state,
                                       obs=trajectory_obs,
                                       actions=trajectory_actions,
                                       prod_run=args.prod_ix)


if __name__ == "__main__":
    tf.app.run(main=main)
