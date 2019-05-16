import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from train import get_dummy_dataset_spec, rollout_latent_model

from data.reacher.reacher_environment import ReacherEnvironment
from specs import dataset_specs

"""
# Params Action Conditioned
--use_variational=True
--network_config_name=variational_lstm_mlp_inference
--loss_config_name=reacher_variational_kl01_ca1
--create_new_subdir=True
--train_batch_size=2
--val_batch_size=20
--dataset_config_name=reacher_oneJoint
--validation_interval=5
--input_seq_len=5
--use_recursive_image=False
--teacher_forcing=False
--regularizer_weight=0
--fixed_prior=True
--show_encoder_predictions=True
--action_conditioned_prediction=True

# Params Agarwal Baseline
--use_variational=True
--network_config_name=variational_lstm_mlp_inference
--loss_config_name=reacher_variational_kl01_ca1
--create_new_subdir=True
--train_batch_size=2
--val_batch_size=20
--dataset_config_name=reacher_oneJoint
--validation_interval=5
--input_seq_len=5
--use_recursive_image=False
--teacher_forcing=False
--regularizer_weight=0
--fixed_prior=True
--show_encoder_predictions=True
--action_conditioned_prediction=True
--infer_actions=True
--inv_mdl_baseline=True

# Params Agarwal Baseline noLSTM
--use_variational=True
--network_config_name=variational_lstm_mlp_inference
--loss_config_name=reacher_variational_kl01_ca1
--create_new_subdir=True
--train_batch_size=2
--val_batch_size=20
--dataset_config_name=reacher_oneJoint
--validation_interval=5
--input_seq_len=5
--use_recursive_image=False
--teacher_forcing=False
--regularizer_weight=0
--fixed_prior=True
--show_encoder_predictions=True
--action_conditioned_prediction=True
--infer_actions=True
--inv_mdl_baseline=True
--no_LSTM_baseline=True

# Params Ours
--use_variational=True
--network_config_name=variational_lstm_mlp_inference
--loss_config_name=reacher_variational_kl01_ca1
--create_new_subdir=True
--train_batch_size=2
--val_batch_size=20
--dataset_config_name=reacher_oneJoint
--validation_interval=5
--input_seq_len=5
--use_recursive_image=False
--teacher_forcing=False
--regularizer_weight=0
--fixed_prior=True
--show_encoder_predictions=True
--action_conditioned_prediction=False
--enforce_composable_actions=True
--comp_seq_length=4
--comp_action_img_loss=True
--mlp_compose=True
--train_action_regressor=True
--trajectory_space=True
--trajectory_kl_weight=1e-3
--shuffle_comp_latents=True
--concat_mlp_compose=True
"""


_NUM_TASKS = 15             # number of servoing tasks to run

_SERVOING_LENGTH = 5        # how long are the sequences that should be servoed
_MAX_SERVOING_STEPS = 5     # how many servoing steps gets the agent to reach the target
_NUM_SEQS = 10              # how many rollout sequences per refit iteration
_MAX_PRED_HORIZON = 5       # how far in the future do we maximally predict
_CHOOSE_BEST = 3            # how many top sequences to choose for Gaussian refit
_LATENT_DIM = 10            # dimension of the latent (needs to correspond to loaded model)
_REFIT_ITERATIONS = 4       # how many "inner loops" of Gaussian refitting should be run
_ACTION_CONDITIONED = False  # if True, action conditioned servoing is performed
_BATCHED_ROLLOUT = False    # whether rollout should be split across multiple batches for unbiased batch stats
_BATCH_STAT_RATIO = 0.1     # ratio of true seqs vs. dummy seqs per batch, should evenly divide NUM_SEQS
_RANDOM = False              # executes random action as baseline
_ONESTEP = False             # does one-step servoing with Agrawal baseline model

# checkpoint_path = "/home/karl/Downloads/oneJoint_noAct_bestModel/model.ckpt-51658"
# checkpoint_path = "/home/karl/Downloads/oneJoint_act_fixedBN/model.ckpt-98378"
# checkpoint_path = "/home/karl/logs/2018-07-31_07-56-25/model.ckpt-0"
# checkpoint_path = "/home/karl/logs/2018-07-31_13-34-21/model.ckpt-0"

checkpoint_path = "/home/karl/Downloads/efficacy/reacher_traj_1e-8-10sh/model.ckpt-218721"
# checkpoint_path = "/home/karl/Downloads/efficacy/reacher_act_cond_10sh/model.ckpt-31155"
# checkpoint_path = "/home/karl/Downloads/efficacy/reacher_agra_100sh-orig/model.ckpt-9409"

result_dir = "/home/karl/Downloads/efficacy_vis_servoing/ours-10sh"
imgs_filename = "servo_imgs.npy"
error_filename = "servo_errors.npy"


def get_input_and_target(env):
    return env.getStartTargetImgPair(_SERVOING_LENGTH)


def prep_imgs(img, config):
    # invert channels if necessary (NHWC -> NCHW)
    if len(img.shape) == 5 and img.shape[2] != config.channels:
        img = np.transpose(img, (0, 1, 4, 2, 3))
    elif len(img.shape) == 4 and img.shape[1] != config.channels:
        img = np.transpose(img, (0, 3, 1, 2))
    elif len(img.shape) == 3 and img.shape[0] != config.channels:
        img = np.transpose(img, (2, 0, 1))

    # scale image [-1,1]
    if img.dtype == np.uint8:
        img = (np.asarray(img, dtype=np.float32)/255 - 0.5) * 2
    elif img.dtype == np.float32:
        img = img * 2 - 1.0
    else:
        raise ValueError("Unkown image dtype!")
    return img


def decode_img(img):
    if len(img.shape) == 3:
        return (np.transpose(img, (1, 2, 0)) + 1.0)/2
    elif len(img.shape) == 5:
        return (np.transpose(img, (0, 1, 3, 4, 2)) + 1.0)/2


def clip_actions(actions, config):
    if isinstance(config, dataset_specs.ReacherConfig):
        # bound actions [0...max_degree] in radian
        output_actions = np.clip(actions, 0.0, config.max_degree * np.pi / 180)
    else:
        # bound action [-max_action...+max_action]
        output_actions = np.clip(actions, -config.max_action, config.max_action)
        output_actions = np.round(output_actions)
    return output_actions


def transform_zs(input_z, config, first):
    if first:
        if isinstance(config, dataset_specs.ReacherConfig):
            # bound actions [0...max_degree] in radian
            output_z = np.asarray(np.random.uniform(0.0, config.max_degree, size=input_z.shape), dtype=np.float32)
            output_z = output_z * np.pi / 180
        else:
            # bound action [-max_action...+max_action]
            output_z = np.asarray(np.random.uniform(-config.max_action, config.max_action, size=input_z.shape), dtype=np.float32)
            output_z = np.round(output_z)
    else:
        output_z = clip_actions(input_z, config)
    return output_z


def prep_input_vars(config, latent_dim):
    input_image = tf.placeholder(tf.float32, shape=(config.input_seq_len,
                                                    _NUM_SEQS, 3,
                                                    config.im_height,
                                                    config.im_height), name="input_image")
    latent_seq_length = config.input_seq_len-1 + _MAX_PRED_HORIZON if _ACTION_CONDITIONED \
                        else _MAX_PRED_HORIZON
    input_latents = tf.placeholder(tf.float32, shape=(latent_seq_length,
                                                      _NUM_SEQS,
                                                      latent_dim), name="input_latents")
    return input_image, input_latents


def batch_replicate(input, axis):
    return np.repeat(np.expand_dims(input, axis=axis), _NUM_SEQS, axis=axis)


def fit_gaussian(seqs):
    res = [np.mean(seqs, axis=1), np.std(seqs, axis=1)]
    res = [np.repeat(np.expand_dims(s, axis=1), _NUM_SEQS, axis=1) for s in res]
    return res


def vis_cost_img(target_img, costs, imgs):
    plt.figure()
    plt.imshow(target_img)
    plt.title("GOAL")
    for idx in range(len(costs)):
        plt.figure()
        plt.imshow(decode_img(imgs[idx]))
        plt.title(costs[idx])
    plt.show()


def act(env, action, input_imgs, input_actions, config):
    print(action * 180/np.pi)
    next_img, error = env.step(action, compute_error=True)
    if _ONESTEP:
        prepped_img = prep_imgs(next_img, config)
        next_input_img = batch_replicate(np.expand_dims(prepped_img, axis=0), axis=1)
        return next_input_img, None, next_img, error

    next_img_transformed = prep_imgs(next_img, config)
    next_img_transformed = batch_replicate(
        np.expand_dims(next_img_transformed, axis=0), axis=1)
    new_input_imgs = np.concatenate([input_imgs[1:], next_img_transformed], axis=0)
    if _ACTION_CONDITIONED:
        action_transformed = batch_replicate(
            np.expand_dims(action, axis=0), axis=1)
        input_actions = np.concatenate([input_actions[1:], action_transformed], axis=0)
    return new_input_imgs, input_actions, next_img, error


def shift_gaussians(mu, std):
    mu = np.concatenate((mu[1:], np.zeros(([1] + list(mu.shape[1:])))), axis=0)
    std = np.concatenate((std[1:], np.ones(([1] + list(std.shape[1:])))), axis=0)
    return mu, std


def gen_fetch_dict(graph_vars):
    # unpack graph variables
    inferred_actions, inferred_actions_reencode, rollout_latents, decoded_seq, \
    decoded_seq_reencode, z_samples_reencode, decoded_seq_reconstruct, network_input_imgs = graph_vars

    # setup fetch dict
    fetch_dict = {
        "decoded_seq": decoded_seq,
        "decoded_seq_reconstruct": decoded_seq_reconstruct,
        "network_input_imgs": network_input_imgs,
    }
    if not _ACTION_CONDITIONED:
        fetch_dict.update({
            "inferred_actions": inferred_actions,
            "inferred_actions_reencode": inferred_actions_reencode,
            "rollout_latents": rollout_latents,
            "decoded_seq_reencode": decoded_seq_reencode,
            "z_samples_reencode": z_samples_reencode,
        })
    return fetch_dict


def gen_dummy_batch(env):
    # generate dummy servoing tasks (input image sequences)
    input_imgs_s, input_actions_s = [], []
    for i in range(_NUM_SEQS):
        input_imgs, target_img_raw, input_actions, initial_distance = get_input_and_target(env)
        input_imgs_s.append(input_imgs)
        input_actions_s.append(input_actions)
    input_imgs = np.stack(input_imgs_s, axis=1)
    input_actions = np.stack(input_actions_s, axis=1)
    input_imgs = prep_imgs(input_imgs, config)

    # generate dummy input latents
    mu = np.zeros((_MAX_PRED_HORIZON, _NUM_SEQS, latent_dim))
    std = np.ones((_MAX_PRED_HORIZON, _NUM_SEQS, latent_dim))
    z_s = np.random.normal(mu, std)
    if _ACTION_CONDITIONED:
        z_s = transform_zs(z_s, config, first=True)
        input_zs = np.concatenate((input_actions, z_s), axis=0)
    else:
        input_zs = z_s
    return input_imgs, input_zs


def batched_rollout(true_input_imgs, true_input_zs, input_image_ph, input_latents_ph, graph_vars, saver):
    """Splits rollout across batches and adds dummy seqs to balance batch stats."""
    fetch_dict = gen_fetch_dict(graph_vars)

    # initialize output dict with empty lists
    output_dict = dict.fromkeys(fetch_dict.keys())
    for key in output_dict.keys():
        output_dict[key] = []

    # loop sub-batches
    num_sub_batches = int(1 / _BATCH_STAT_RATIO)
    num_true_seqs = int(_NUM_SEQS / num_sub_batches)
    dummy_env = ReacherEnvironment(config)
    for sb_idx in range(num_sub_batches):
        input_imgs, input_zs = gen_dummy_batch(dummy_env)
        # insert true sequences
        input_imgs[:, :num_true_seqs] = true_input_imgs[:, sb_idx*num_true_seqs:(sb_idx+1)*num_true_seqs]
        input_zs[:, :num_true_seqs] = true_input_zs[:, sb_idx*num_true_seqs:(sb_idx + 1)*num_true_seqs]

        # fill feed dict and execute rollout
        feed_dict = {
            input_image_ph: input_imgs,
            input_latents_ph: input_zs,
        }
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            batch_outputs = sess.run(fetch_dict, feed_dict=feed_dict)
            sess.close()

        # extract true results, fuse in output dict, discard dummy rollouts
        for key in batch_outputs.keys():
            output_dict[key].append(batch_outputs[key][:, :num_true_seqs])

    # concatenate output dict entries
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=1)

    return output_dict


def complete_rollout(input_imgs, input_zs, input_image_ph, input_latents_ph, graph_vars, saver):
    fetch_dict = gen_fetch_dict(graph_vars)

    # fill feed dict and execute rollout
    feed_dict = {
        input_image_ph: input_imgs,
        input_latents_ph: input_zs,
    }
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        outputs = sess.run(fetch_dict, feed_dict=feed_dict)
        sess.close()
    return outputs


def run_servoing_task(config, latent_dim):
    env = ReacherEnvironment(config)

    # get start image sequence and target image
    input_imgs, target_img_raw, input_actions, initial_distance = get_input_and_target(env)

    # store all images of trajectory towards goal for later visualization
    trajectory_imgs, trajectory_errors, sub_trajectories = [], [], []
    trajectory_imgs.append(input_imgs[-1])
    trajectory_errors.append(initial_distance)

    input_imgs = prep_imgs(input_imgs, config)
    input_imgs = batch_replicate(input_imgs, axis=1)
    if _ACTION_CONDITIONED:
        input_actions = batch_replicate(input_actions, axis=1)

    mu = np.zeros((_MAX_PRED_HORIZON, _NUM_SEQS, latent_dim))
    std = np.ones((_MAX_PRED_HORIZON, _NUM_SEQS, latent_dim))
    for step_idx in range(_MAX_SERVOING_STEPS):
        tf.reset_default_graph()
        if not _RANDOM:
            # build rollout model and saver for restoring variable values
            input_image, input_latents = prep_input_vars(config, latent_dim)
            graph_vars = rollout_latent_model(input_image, input_latents, config)
            saver = tf.train.Saver()

            for refit_idx in range(_REFIT_ITERATIONS):
                # sample latent sequences
                z_s = np.random.normal(mu, std)
                if _ACTION_CONDITIONED:
                    z_s = transform_zs(z_s, config, first=True if refit_idx == 0 else False)
                    input_zs = np.concatenate((input_actions, z_s), axis=0)
                else:
                    input_zs = z_s

                # roll out prediction model
                if _BATCHED_ROLLOUT:
                    outputs = batched_rollout(input_imgs,
                                              input_zs,
                                              input_image,
                                              input_latents,
                                              graph_vars,
                                              saver)
                else:
                    outputs = complete_rollout(input_imgs,
                                               input_zs,
                                               input_image,
                                               input_latents,
                                               graph_vars,
                                               saver)

                if not _ACTION_CONDITIONED:  # for not action cond use imgs that correspond to inference latents
                    output_img_seq = outputs["decoded_seq_reencode"]
                else:
                    output_img_seq = outputs["decoded_seq"]
                rollout_imgs = decode_img(output_img_seq)

                # compute the cost per rollout, reduce horizon when closing in on target
                rollout_final_idx = min(_MAX_PRED_HORIZON, _MAX_SERVOING_STEPS - (step_idx + 1))
                cost = env.computeCost(rollout_imgs[rollout_final_idx], target_img_raw)

                # sort based on cost: min to max
                sort_idxs = np.argsort(cost)
                best_z_seqs = z_s[:, sort_idxs[:_CHOOSE_BEST]]

                # refit Gaussians to latent data
                mu, std = fit_gaussian(best_z_seqs)
                print(cost[sort_idxs[0]])

            # vis_cost_img(target_img_raw,
            #                  cost[sort_idxs[:_CHOOSE_BEST]],
            #                  output_img_seq[rollout_final_idx, sort_idxs[:_CHOOSE_BEST]])

            if _ACTION_CONDITIONED:
                seq_imgs = [trajectory_imgs[-1]]
                for img_idx in range(rollout_final_idx+1):
                    seq_imgs.append(np.asarray(rollout_imgs[img_idx, sort_idxs[0]] * 255, dtype=np.uint8))
            else:
                seq_imgs = [trajectory_imgs[-1]]
                for img_idx in range(rollout_final_idx+1):
                    seq_imgs.append(np.asarray(decode_img(outputs["decoded_seq_reencode"])[img_idx, sort_idxs[0]] * 255,
                                               dtype=np.uint8))
            sub_trajectories.append(env.gen_padded_subtrajectory(seq_imgs, target_img_raw, _MAX_SERVOING_STEPS+1))

            if not _ACTION_CONDITIONED:
                action = outputs["inferred_actions_reencode"][0, sort_idxs[0]]
                action = clip_actions(action, config)
            else:
                action = z_s[0, sort_idxs[0]]
        else:
            # RANDOM
            action = transform_zs(np.zeros(1), config, True)

        input_imgs, input_actions, new_img, error = act(env, action, input_imgs, input_actions, config)
        trajectory_imgs.append(new_img)
        trajectory_errors.append(error)
        mu, std = shift_gaussians(mu, std)

    print("Reached maximum number of servoing steps!")
    print(trajectory_errors)
    executed_traj = env.plotTrajectory(trajectory_imgs, target_img_raw, render=True)
    env.gen_overview_figure(sub_trajectories, executed_traj, noshow=True)

    return trajectory_imgs, trajectory_errors


def run_single_step_servoing_task(config, latent_dim):
    env = ReacherEnvironment(config)

    # get start image sequence and target image
    input_imgs, target_img_raw, input_actions, initial_distance = get_input_and_target(env)

    # store all images of trajectory towards goal for later visualization
    trajectory_imgs, trajectory_errors, sub_trajectories = [], [], []
    trajectory_imgs.append(input_imgs[-1])
    trajectory_errors.append(initial_distance)

    input_imgs = prep_imgs(input_imgs[-1], config)
    target_img = prep_imgs(target_img_raw, config)
    input_imgs = batch_replicate(np.expand_dims(input_imgs, axis=0), axis=1)
    target_imgs = batch_replicate(np.expand_dims(target_img, axis=0), axis=1)
    if _ACTION_CONDITIONED:
        input_actions = batch_replicate(input_actions, axis=1)
    for step_idx in range(_MAX_SERVOING_STEPS):
        tf.reset_default_graph()
        # build rollout model and saver for restoring variable values
        input_image_pair = tf.placeholder(tf.float32, shape=(2,
                                                        _NUM_SEQS, 3,
                                                        config.im_height,
                                                        config.im_height), name="input_image")
        inferred_action = rollout_latent_model(input_image_pair, None, config)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            outputs = sess.run({"action": inferred_action},
                               feed_dict={input_image_pair: np.concatenate([input_imgs, target_imgs], axis=0)})
            sess.close()

        action = np.asarray(np.squeeze(outputs["action"])[:1], dtype=np.float64)
        action = np.clip(action, 0.0, config.max_degree * np.pi / 180)

        input_imgs, input_actions, new_img, error = act(env, action, input_imgs, input_actions, config)
        trajectory_imgs.append(new_img)
        trajectory_errors.append(error)
    print("Reached maximum number of servoing steps!")
    print(trajectory_errors)
    executed_traj = env.plotTrajectory(trajectory_imgs, target_img_raw, render=True)

    return trajectory_imgs, trajectory_errors


def check_result_files():
    imgs_file = os.path.join(result_dir, imgs_filename)
    error_file = os.path.join(result_dir, error_filename)
    if os.path.isfile(imgs_file) or os.path.isfile(error_file):
        raise ValueError("Result file exists already!")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return imgs_file, error_file


select_seq = 0
if __name__ == "__main__":
    np.random.seed(2)
    # get dataset configuration
    config = get_dummy_dataset_spec()

    if _ACTION_CONDITIONED:
        latent_dim = config.num_actions
    else:
        latent_dim = _LATENT_DIM

    # check result files exist
    imgs_file, error_file = check_result_files()

    trajectory_imgs, trajectory_errors = [], []
    for task in range(_NUM_TASKS):
        print("Servoing task no. %d" % (task+1))
        if _ONESTEP:
            task_imgs, task_errors = run_single_step_servoing_task(config, latent_dim)
        else:
            task_imgs, task_errors = run_servoing_task(config, latent_dim)
        trajectory_imgs.append(task_imgs)
        trajectory_errors.append(task_errors)

        if not _RANDOM:
            plt.savefig("/tmp/task_file_%d.png" % task)
            plt.close()

        # save imgs and errors
        np.save(imgs_file, trajectory_imgs)
        np.save(error_file, trajectory_errors)
    print("")
    print("Finished evaluation of %d servoing tasks!" % _NUM_TASKS)