"""Implements a logger for recording results/debugging info."""

import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys
import pipes
import pickle
import moviepy.editor as mpy

from viz.viz_utils import draw_actions_on_images


class CEMLogger(object):
    def __init__(self, args):
        self._basedir = self._create_base_dir(args)
        self._plan_img_stack = []

    @property
    def basedir(self):
        return self._basedir

    @staticmethod
    def _create_base_dir(args):
        if args.train_dir is not None:
            return args.train_dir

        if args.mode == 'gt':
            plan_steps_str = 'ps_{}'.format(args.planning_horizon)
        elif args.mode == 'th':
            plan_steps_str = 'psh_{}_psl_{}'.format(args.high_level_planning_horizon,
                                                    args.planning_horizon)
        hyperparam_str = '{}_{}_ar_{}_wit_{}_wbs_{}_it_{}_bs_{}_ef_{}_seed_{}{}'.format(
            args.env_name, plan_steps_str, args.num_action_repeat,
            args.warmup_num_iter, args.warmup_batch_size,
            args.num_iter, args.batch_size, args.elite_frac, args.seed,
            '_sfs' if args.simulate_from_start else '')

        base_dir = 'logs/{}/{}_{}'.format(
            args.prefix, hyperparam_str,
            time.strftime("%Y%m%d-%H%M%S"))

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        if args.debug:
            print("Train Dir: {}".format(base_dir))
        return base_dir

    @staticmethod
    def save_git(base_dir):
        # save code revision
        print('Save git commit and diff to {}/git.txt'.format(base_dir))
        cmds = ["echo `git rev-parse HEAD` >> {}".format(
            os.path.join(base_dir, 'git.txt')),
            "git diff >> {}".format(
                os.path.join(base_dir, 'git.txt'))]
        print(cmds)
        os.system("\n".join(cmds))

    @staticmethod
    def save_cmd(base_dir):
        train_cmd = 'python ' + ' '.join([sys.argv[0]] + [pipes.quote(s) for s in sys.argv[1:]])
        train_cmd += '\n'
        print('\n' + '*' * 80)
        print('Training command:\n' + train_cmd)
        print('*' * 80 + '\n')
        with open(os.path.join(base_dir, "cmd.txt"), "a+") as f:
            f.write(train_cmd)

    @staticmethod
    def _maybe_add_step(step):
        return "" if step is None else "_%d" % step

    def log_cost_overview(self, goal_img, frames, costs, prefix="", step=None):
        final_frames = frames[:, -1]
        save_path = os.path.join(self.basedir, prefix + self._maybe_add_step(step) + "_cost_overview.png")
        plt.figure(figsize=(20, 5))
        plt.subplot(251)
        plt.imshow(goal_img, interpolation='bicubic')
        plt.title("GOAL")
        plt.axis("off")
        for i in range(min(9, final_frames.shape[0])):
            plt.subplot(2, 5, i + 2)
            plt.imshow(np.asarray(final_frames[i], dtype=np.uint8), interpolation='bicubic')
            plt.title("Cost: %f" % costs[i])
            plt.axis("off")
        plt.savefig(save_path)

    @staticmethod
    def _draw_actions(frames, actions):
        ndims = frames.ndim
        transpose_forward_ix = tuple(np.array(list(range(ndims))) + ([0] * (ndims - 3) + [2, -1, -1]))
        annotated_frames = frames.transpose(transpose_forward_ix) / 255
        annotated_frames = draw_actions_on_images(annotated_frames, actions)
        transpose_backward_ix = tuple(np.array(list(range(ndims))) + ([0] * (ndims - 3) + [1, 1, -2]))
        return np.asarray(annotated_frames.transpose(transpose_backward_ix) * 255, dtype=np.uint8)

    def log_rollout_frames(self, start_img, rollout_frames, goal_frame, rollout_actions=None, prefix="", step=None):
        log_frames = self._draw_actions(rollout_frames, rollout_actions) if rollout_actions is not None \
                        else rollout_frames
        batch_size, rollout_len, _, res, channels = rollout_frames.shape

        def rep(frame):
            return np.repeat(frame[None, None, ...], batch_size, axis=0)

        log_frames = np.concatenate((rep(start_img), log_frames, rep(goal_frame)), axis=1)
        log_frames = log_frames.transpose(0, 2, 1, 3, 4)
        log_frames = np.reshape(log_frames, (batch_size*res, (rollout_len+2)*res, channels))
        savepath = os.path.join(self.basedir, prefix + self._maybe_add_step(step) + "_rollout_overview.png")
        plt.imsave(savepath, log_frames)

    @staticmethod
    def _pad(img, pre, post):
        assert img.max() <= 1.0, "Input image is assumed to be in range [0...1]"
        img_res = img.shape[0]
        concat_list = []
        if pre > 0:
            pre_imgs = 0.7 * np.ones((img_res, pre * img_res, 3), dtype=np.float32)
            concat_list.append(pre_imgs)
        concat_list.append(img)
        if post > 0:
            post_imgs = 0.7 * np.ones((img_res, post * img_res, 3), dtype=np.float32)
            concat_list.append(post_imgs)
        return np.concatenate(concat_list, axis=1)

    def _dump_plan_img_stack(self, prefix):
        output = np.concatenate(self._plan_img_stack, axis=0)
        savepath = os.path.join(self.basedir, prefix + "_exec_trace.png")
        plt.imsave(savepath, output)

    def log_plan_and_execution_trace(self, start_img, rollout_frames, goal_frame,
                                     step, max_step, rollout_actions=None, prefix=""):
        log_frames = self._draw_actions(rollout_frames, rollout_actions) if rollout_actions is not None \
                        else rollout_frames
        log_frames = np.concatenate(([start_img], log_frames, [goal_frame]), axis=0)
        N, H, W, C = log_frames.shape
        log_frames = log_frames.transpose(1, 0, 2, 3).reshape((H, N * W, C))
        log_frames = log_frames.astype(float) / 255
        log_frames = self._pad(log_frames, step, max_step - step)
        log_frames = (log_frames * 255).astype(np.uint8)
        self._plan_img_stack.append(log_frames)
        self._dump_plan_img_stack(prefix + self._maybe_add_step(step))

    def dump_video(self, vid_imgs, fps=10.0, step=None, prefix=""):
        assert isinstance(vid_imgs, list), "Input stack must be a list of images!"
        assert len(vid_imgs[0].shape) == 3, "Video saving expects image format input with 3 channels."
        vid_imgs = np.stack(vid_imgs, axis=0)
        video = mpy.VideoClip(lambda t: vid_imgs[int(t * fps)],
                              duration=float(vid_imgs.shape[0]) / fps)
        savepath = os.path.join(self.basedir, prefix + self._maybe_add_step(step) + "_trajectory.mp4")
        video.write_videofile(savepath, fps, verbose=False, progress_bar=False)

    @staticmethod
    def dump_production_results(prod_dir, args, init_state, obs, actions, prod_run):
        trajectory_info = {
            'args': args,
            'init_state': init_state,
            'obs': obs,
            'ac': actions,
        }
        savepath = os.path.join(prod_dir, "prod_results_%d.p" % prod_run)
        pickle.dump(trajectory_info, open(savepath, 'wb'))
