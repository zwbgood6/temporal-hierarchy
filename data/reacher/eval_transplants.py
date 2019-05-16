import os

import numpy as np
from tqdm import tqdm

from eval import modebased_eval, plot_time_seqs_shaded_var
from viz.viz_utils import compute_angle

gt_seq_file = "/home/karl/Downloads/reacher_labeling/compActions/test_seqs_gt.npy"
gt_action_file = "/home/karl/Downloads/reacher_labeling/compActions/abs_action_seqs_gt.npy"
est_seq_files = ["/home/karl/Downloads/reacher_labeling/compActions/transplanted_seqs_gt.npy",
                 # "/home/karl/Downloads/reacher_labeling/actCond/test_seqs.npy",
                 "/home/karl/Downloads/reacher_labeling/noComp/transplanted_seqs_gt.npy"]

_GRAPH_LABELS = ["with Comp (ours)",
                 # "action conditioned",
                 "without Comp"]
_MODE = "relative"
_BATCH_SIZE = 4       # batch size during testing (for transplanted latents)

_MAX_ANGLE = 40


def comp_start_state_baseline(gt_angles):
  if "transplant" in os.path.basename(est_seq_files[0]):
    d_gt = np.absolute(gt_angles[:-1] - gt_angles[0])
    d_gt[d_gt > 180] = 360 - d_gt[d_gt > 180]
    d_gt = np.cumsum(d_gt[1:], axis=0)
    d_gt[d_gt > 180] = np.mod(d_gt[d_gt > 180], 360)
    errors = d_gt
  else:
    diffs = np.absolute(gt_angles[1:-1] - gt_angles[:-2])
    diffs[diffs>=180] = 360 - diffs[diffs>180]
    errors = np.cumsum(diffs, axis=0)
  return np.squeeze(errors)


def comp_random_baseline(gt_angles):
  if "transplant" in os.path.basename(est_seq_files[0]):
    d_gt_angles = gt_angles[:-1] - gt_angles[1:]
    d_gt_angles[d_gt_angles>180] = 360 - d_gt_angles[d_gt_angles>180]
    d_rand_angle = np.random.rand(d_gt_angles.shape[0], d_gt_angles.shape[1], 1) * _MAX_ANGLE
    errors = np.absolute(d_gt_angles - d_rand_angle)
  else:
    start_angle = gt_angles[:1]
    rand_movements = np.random.rand(gt_angles.shape[0]-2, gt_angles.shape[1], 1) * _MAX_ANGLE
    rand_angle_vals = np.cumsum(np.concatenate((start_angle, rand_movements), axis=0), axis=0)
    rand_angle_vals[rand_angle_vals>360] = rand_angle_vals[rand_angle_vals>360] - 360
    errors = np.abs(gt_angles[1:-1] - rand_angle_vals[1:])
    errors[errors>180] = 360 - errors[errors>180]
  return np.squeeze(errors)


def pick_transplants(gt_angles, est_angles):
  num_parent_seqs = est_angles.shape[1] / _BATCH_SIZE
  parent_idxs = np.asarray([i * _BATCH_SIZE for i in range(num_parent_seqs)], dtype=int)
  child_idxs = np.concatenate([parent_idxs + i + 1 for i in range(_BATCH_SIZE-1)])
  child_idxs = np.sort(child_idxs)

  parent_seqs = gt_angles[:, parent_idxs]
  child_seqs = est_angles[:, child_idxs]

  parent_seqs = np.repeat(parent_seqs, _BATCH_SIZE-1, axis=1)

  return parent_seqs, child_seqs


if __name__ == "__main__":
  # load GT images and angles
  gt_imgs = np.load(gt_seq_file)
  gt_angles = np.load(gt_action_file)[-gt_imgs.shape[0]:] * 180 / np.pi

  # compute est angles
  est_angles_list = []

  for est_seq_file in est_seq_files:
    est_imgs = np.load(est_seq_file)
    est_angles = np.empty(([gt_angles.shape[0]-1] + list(gt_angles.shape[1:])))
    for idx in tqdm(range(gt_imgs.shape[1])):
      for timestep in range(gt_imgs.shape[0]-1):
        gt_img = (np.transpose(gt_imgs[timestep+1, idx], (1, 2, 0)) + 1) / 2
        gt_angle = gt_angles[timestep, idx]

        # plt.figure()
        # plt.imshow(gt_img)
        # plt.show()

        est_img = (np.transpose(est_imgs[timestep+1, idx], (1, 2, 0)) + 1) / 2
        est_angle = compute_angle(est_img, one_joint=True) * 180 / np.pi
        est_angles[timestep, idx] = est_angle
    if "transplant" in os.path.basename(est_seq_file):
      new_gt_angles, est_angles = pick_transplants(gt_angles, est_angles)
      est_angles_list.append(est_angles)
  gt_angles = new_gt_angles

  errors = modebased_eval(gt_angles[:-1], est_angles_list, _MODE, angular_vals=True)

  errors.append(comp_start_state_baseline(gt_angles))
  _GRAPH_LABELS.append("start state")
  errors.append(comp_random_baseline(gt_angles))
  _GRAPH_LABELS.append("random")

  means = [np.mean(error.flatten()) for error in errors]
  stds = [np.std(error.flatten()) for error in errors]
  for i in range(len(means)):
    print("%s: %f, %f" % (_GRAPH_LABELS[i], means[i], stds[i]))

  plot_time_seqs_shaded_var(errors, _GRAPH_LABELS, "Timestep", _MODE.title() + " Trajectory Error")
