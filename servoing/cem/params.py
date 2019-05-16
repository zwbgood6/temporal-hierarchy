from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os


# mode
tf.flags.DEFINE_string('mode', 'th', 'Whether to run '
                'GT simulator or temporal hierarchy mode.')

# exp
tf.flags.DEFINE_string('prefix', 'default', 'Experiment logging prefix.')
tf.flags.DEFINE_integer('log_steps', 10, 'Log interval.')
tf.flags.DEFINE_integer('debug_steps', 10, 'Debug interval.')
tf.flags.DEFINE_integer('subsample_rate', 2, 'Subsampling rate for network input.')
tf.flags.DEFINE_string('train_dir', None, 'Training directory to overwrite.')
tf.flags.DEFINE_boolean('high_level_only', False, 'Run high-level predictions only.')

# env
tf.flags.DEFINE_string('env_name', 'CustomPush-v0', 'Mujoco environment.')
tf.flags.DEFINE_integer('seed', 42, 'Random seed for mujoco and numpy.')
tf.flags.DEFINE_integer('max_steps', 500, 'Horizon for running experiments.')
tf.flags.DEFINE_boolean('init_arm_near_object', False, 'Whether to initialize arm position nearby object.')
tf.flags.DEFINE_boolean('display_goal', False, 'Whether to display goal in trajectory videos.')

# cem
tf.flags.DEFINE_integer('planning_horizon', 20, 'Number of steps to plan.')
tf.flags.DEFINE_boolean('simulate_from_start', False, 'Whether to always simulate from start of episode.')
tf.flags.DEFINE_integer('num_action_repeat', 5, 'Number for action repeat.')
tf.flags.DEFINE_integer('warmup_num_iter', 10, 'Number of warmup iterations.')
tf.flags.DEFINE_integer('warmup_batch_size', 40, 'Batch size for warmup, has to be multiple of batch size.')
tf.flags.DEFINE_integer('num_iter', 2, 'Number of iterations.')
tf.flags.DEFINE_integer('batch_size', 20, 'Batch size for CEM.')
tf.flags.DEFINE_float('elite_frac', 0.25, 'Elite fraction for CEM.')

# running with high level and low level network
tf.flags.DEFINE_string('high_level_checkpoint_dir', '/home/karl/logs/top-img-actCond-li_1',
                       'High-level network checkpoint directory.')
tf.flags.DEFINE_string('high_level_config_name', 'hierarchical_lstm_mnist',
                       'High-level network config name.')
tf.flags.DEFINE_integer('high_level_input_seq_len', 1,
                        'High-level network input sequence length.')
tf.flags.DEFINE_integer('high_level_planning_horizon', 5,
                        'High-level network predict sequence length.')
tf.flags.DEFINE_integer('high_level_z_dim', 10,
                        'High-level network z latent dimension.')
tf.flags.DEFINE_float('subgoal_reached_dist', 5,
                       'If dist is less than this value the subgoal counts as reached')
tf.flags.DEFINE_integer('max_subgoal_timesteps', 10,
                        'Maximal number of timesteps spent on reaching any but the last subgoal.')
tf.flags.DEFINE_integer('max_subgoal_planning_horizon', 8,
                        'Maximal planning horizon for reaching subgoals.')
tf.flags.DEFINE_integer('min_subgoal_planning_horizon', 3,
                        'Minimal planning horizon for reaching subgoal.')
tf.flags.DEFINE_float('subgoal_planning_margin_factor', 1.2,
                        'Factor that the subgoal planning time margin is multiplied with.')
tf.flags.DEFINE_string('low_level_checkpoint_dir', '/home/karl/logs/top-img-actCond-li_1',
                       'Low-level network checkpoint directory.')
tf.flags.DEFINE_string('low_level_config_name', 'actCond_lstm_bb',
                       'Low-level network config name.')
tf.flags.DEFINE_float('low_level_clipping_value', 0.5,
                        'Value for clipping the low level actions.')
tf.flags.DEFINE_float('init_var', 0.3,
                        'Initial variance for CEM.')
tf.flags.DEFINE_integer('low_level_input_seq_len', 5,
                        'Low-level network input sequence length.')
# NOTE: use max_plan_steps instead to set low level predict length
# tf.flags.DEFINE_integer('low_level_pred_seq_len', 20,
#                         'Low-level network predict sequence length.')
# tf.flags.DEFINE_integer('low_level_ac_dim', 8,
#                         'Low-level network action dimension.')
tf.flags.DEFINE_integer('low_level_z_dim', 10,
                        'Low-level network z_sample dimension.')
tf.flags.DEFINE_boolean('high_level_arm_centered', False,
                        'If True, arm is in center for high level..')
tf.flags.DEFINE_boolean('low_level_gt_parallelization', False,
                        'If True, parallelizes gt simulator predictions..')
tf.flags.DEFINE_string('low_level_simulator_name', 'net',
                       'Low-level simulator name (gt or net)')
tf.flags.DEFINE_boolean('include_high_level_net', False,
                        'Whether to include high level network or not.')
tf.flags.DEFINE_boolean('reinit_cem_dists', False,
                        'If True, reinitializes CEM weights in every planning step.')
tf.flags.DEFINE_integer('channels', 3,
                        'Number of channels for images.')
tf.flags.DEFINE_integer('image_sidelength', 64,
                        'Side length of images.')
tf.flags.DEFINE_string('high_level_dist_metric', 'euclidean',
                       'Distance metric (vgg, euclidean or flow) for high level CEM.')
tf.flags.DEFINE_string('dist_metric', 'euclidean',
                       'Distance metric (vgg, euclidean or flow) for low level CEM.')
tf.flags.DEFINE_boolean('parallelize_prediction', False,
                        'Whether to parallelize prediction '
                        + 'across action sequences.')
tf.flags.DEFINE_boolean('infer_initial_zs', False,
                        'If True, infers initial z_variables for execution from start.')
tf.flags.DEFINE_boolean('predict_actions_low_level', False,
                        'If True, directly predicts actions with low level network (planning as inference).')
tf.flags.DEFINE_boolean('dense_cost', False,
                        'If True, cost is a weighted average of individual step costs.')
tf.flags.DEFINE_float('final_step_weight', 10.0,
                       'Cost weight of the final step (all others == 1.0).')

# production  used for dataset collection
tf.flags.DEFINE_boolean('prod', False,
                        'Whether program is used to collect data.')
tf.flags.DEFINE_integer('prod_ix', 0,
                        'Index of data collected.')
tf.flags.DEFINE_string('prod_dir', 'output',
                       'Directory to save generated data.')
tf.flags.DEFINE_boolean('prod_short_path', False,
                        'Whether to use short path name for generated data.')
tf.flags.DEFINE_float('prod_min_reward', None,
                      'Minimum reward to declare success.')
