"""Utilities for visualization."""
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py
import itertools

from viz.html_utils import html
from viz.html_utils.ffmpeg_gif import save_gif


def generate_sequence_video(
    targets_rec,
    targets_pred,
    reconstructions,
    predictions,
    output_file,
    dataset_name,
    case_id=0):
  """Generates videos from input sequence images

    Args:
      targets_rec, targets_pred, reconstructions, predictions, case_id:
        Same as function 'generate_sequence_image'
      viz_save_dir: The directory to save generated videos to. Take in
        checkpoint directory now.
    Raises:
      ValueError: if seq_len_total != seq_len_rec + seq_len_pred.
  """
  seq_len_total = reconstructions.shape[0] + predictions.shape[0]
  n_channels = targets_rec.shape[2]
  height = targets_rec.shape[3]
  width = targets_rec.shape[4]

  seq_im = np.zeros([2 * height, seq_len_total * width, n_channels])

  # to seq_len x H x W x C
  targets_rec_to_slice = np.transpose(
      targets_rec[:, case_id, ...], [0, 2, 3, 1])
  targets_pred_to_slice = np.transpose(
      targets_pred[:, case_id, ...], [0, 2, 3, 1])

  reconstructions_to_slice = np.transpose(
      reconstructions, [0, 2, 3, 1])
  predictions_to_slice = np.transpose(
      predictions, [0, 2, 3, 1])

  targets_to_slice = np.concatenate(
      [targets_rec_to_slice, targets_pred_to_slice],
      axis=0)
  estimates_to_slice = np.concatenate(
      [reconstructions_to_slice, predictions_to_slice],
      axis=0)

  targets_to_slice = targets_to_slice[:estimates_to_slice.shape[0]]   # cut targets to size of estimates

  if dataset_name != "moving_mnist":
    # Rescale from [-1, 1] to [0, 1]
    targets_to_slice = targets_to_slice / 2 + 0.5
    estimates_to_slice = estimates_to_slice / 2 + 0.5

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter(output_file, fourcc, 3, (width, 2 * height))

  for i in range(seq_len_total):
    # get estimate and target (ground truth) images at this time step
    est = estimates_to_slice[i, ...]
    tar = targets_to_slice[i, ...]
    comb = np.concatenate([tar, est], axis=0)
    img_comb = np.array(comb * 255, dtype=np.uint8)

    if img_comb.shape[-1] == 1:
      grayscale = True
      # Convert grayscale to color image
      img_comb_color = cv2.cvtColor(img_comb, cv2.COLOR_GRAY2RGB)
    else:
      grayscale = False
      img_comb_color = img_comb

    # Draw a dot indicating reconstruction (green) or prediction (red)
    if grayscale:
      # TODO(drewjaegle): this doesn't work with RGB images. What's going on?
      if i < targets_rec.shape[0]:
        # image, center, radius, color, thickness
        cv2.circle(img_comb_color, (width - 2, 1), 1, (0, 255, 0), -1)
      else:
        cv2.circle(img_comb_color, (width - 2, 1), 1, (0, 0, 255), -1)

    out.write(img_comb_color)

  out.release()


def generate_sequence_videos(
    targets_rec,
    targets_pred,
    reconstructions,
    predictions,
    iteration,
    base_dir,
    n_seqs,
    dataset_name,
    phase="val"):
  """Generates multiple sequence videos."""
  if phase == "val":
    # at validation, make different folders for different sequences
    def get_file(i):
      viz_save_dir = os.path.join(
          base_dir, "videos", "seq_{}".format(i))
      if not os.path.exists(viz_save_dir):
          os.makedirs(viz_save_dir)
      return os.path.join(viz_save_dir, "iter_{}_output.avi".format(iteration))
  elif phase == "test":
    # at test, put all the videos in one folder
    def get_file(i):
      viz_save_dir = os.path.join(base_dir, "test_videos")
      if not os.path.exists(viz_save_dir):
          os.makedirs(viz_save_dir)
      return os.path.join(viz_save_dir, "batch_{}_seq_{}_output.avi".format(iteration, i))
    
  else:
    raise ValueError("do not know what to do at training time")

  for i in range(n_seqs):
    generate_sequence_video(
        targets_rec,
        targets_pred,
        reconstructions,
        predictions[i],
        get_file(i),
        dataset_name,
        case_id=i)


def plot_single_kl_seq(kl, kf_idxs_onehot):
    fig = plt.figure()
    plt.plot(kl)
    kf_idxs = np.where(kf_idxs_onehot)[0]
    for kf_idx in kf_idxs:
        plt.axvline(x=kf_idx, linestyle='--', color='r', alpha=0.5)
    return fig


def close_figs(figs):
    [plt.close(fig) for fig in figs]


def plot_variance_sequence(latents,
                           actions,
                           file_name,
                           seq_ID,
                           n_rows=None,
                           n_cols=None):
    # frames with invalid/no actions are marked with NaN
    if actions is None:
      actions = np.array([[None]],dtype=np.float32)
    valid_action_mask = np.logical_not(np.isnan(actions))
    valid_action_mask = np.any(valid_action_mask, axis=1)   # aggregate along axis

    if n_rows is None:
        fig = plt.figure()
    else:
        subplt = plt.subplot(n_rows, n_cols, seq_ID+1)  # index starts counting at 1
    plt.plot(range(len(latents)), latents)

    # don't plot action timesteps if actions in all frames
    if not np.all(valid_action_mask):
        action_idxs = np.where(valid_action_mask)
        for idx in action_idxs[0]:
            plt.axvline(x=idx, linestyle='--', color='r', alpha=0.5)
    if n_rows is None:
        plt.title("StdDev Sequence %d" % seq_ID)
        plt.xlabel("Step")
        plt.ylabel("Standard Deviation")
        fig.savefig(file_name, bbox_inches="tight")
        plt.close(fig)


def generate_latent_variance_plots(
    latents,
    actions,
    iteration,
    base_dir,
    n_seqs,
    phase="val"):
    if actions is None or actions.shape[0] == 0:
      use_actions = False
    else:
      use_actions = True
      
    """Generates multiple variance timeline plots."""
    if phase == "val":
        # at validation, make different folders for different sequences
        def get_file(i):
            viz_save_dir = os.path.join(
                base_dir, "variance_plots", "seq_{}".format(i))
            if not os.path.exists(viz_save_dir):
                os.makedirs(viz_save_dir)
            return os.path.join(viz_save_dir, "iter_{}_varPlot.png".format(iteration))
    elif phase == "test":
        # at test, put all the sequences in one folder
        def get_file(i):
            viz_save_dir = os.path.join(base_dir, "test_varPlots")
            if not os.path.exists(viz_save_dir):
                os.makedirs(viz_save_dir)
            return os.path.join(viz_save_dir, "batch_{}_seq_{}_varPlot.png".format(iteration, i))
    else:
        raise ValueError("do not know what to do at training time")

    for i in range(n_seqs):
        plot_variance_sequence(
            latents[:, i],
            actions[:, i] if use_actions else None,
            get_file(i),
            seq_ID=i)

    # find number of rows and cols for plot
    if n_seqs % 4 == 0:
        n_rows = 4
        n_cols = n_seqs / 4
    elif n_seqs % 3 == 0:
        n_rows = 3
        n_cols = n_seqs / 3
    else:
        raise ValueError("Number of Variance plotted sequences must be divisible by 3 or 4!")

    fig = plt.figure(figsize=(20, 10))
    for i in range(n_seqs):
        plot_variance_sequence(
            latents[:, i],
            actions[:, i] if use_actions else None,
            get_file(i),
            seq_ID=i,
            n_rows=n_rows,
            n_cols=n_cols)
    fig.savefig(get_file(999))
    plt.close(fig)

def generate_sprite_image(
    targets_rec,
    targets_pred,
    base_dir):
  """Generates sprite image for moving mnist dataset.

  Args:
    targets_rec: The target image sequence for reconstruction. An array of shape
      [seq_len_rec, n_seqs, n_channels, height, width].
    targets_pred: The target image sequence for prediction. An array of shape
      [seq_len_pred, n_seqs, n_channels, height, width].
    base_dir: directory where sprite image will be saved.
  """
  # TODO(drewjaegle): for non-MNIST data, downsample before saving sprite

  seq_len_total = targets_rec.shape[0] + targets_pred.shape[0]
  num_seq = targets_rec.shape[1]
  n_channels = targets_rec.shape[2]
  height = targets_rec.shape[3]
  width = targets_rec.shape[4]

  sprite_im = np.zeros([height * seq_len_total, width * num_seq, n_channels])

  # to seq_len x num_seq x H x W x C
  targets_rec_to_slice = np.transpose(
      targets_rec, [0, 1, 3, 4, 2])
  targets_pred_to_slice = np.transpose(
      targets_pred, [0, 1, 3, 4, 2])

  targets = np.concatenate(
      (targets_rec_to_slice, targets_pred_to_slice), axis=0)

  for sprite_row_index in range(seq_len_total):
    for sprite_col_index in range(num_seq):
      sprite_im[
          (sprite_row_index * height):(height * (sprite_row_index + 1)),
          (sprite_col_index * width):(width * (sprite_col_index + 1)),
          :] = targets[sprite_row_index, sprite_col_index, ...]

  # TODO(drewjaegle): make sure this works for non-MNIST data and RGB data
  im = np.squeeze(sprite_im)
  im[im > 1] = 1
  im[im < 0] = 0
  im = im * 255
  # TODO(drewjaegle): save this sprite image in a subdir
  savedir = os.path.join(base_dir, "sprite.png")
  cv2.imwrite(savedir, im)


def generate_sequence_image(input_seq_list,
                            dataset_name,
                            case_id=0):
  """Generates sequence images for the raw data and estimates.

  Args:
    input_seq_list: A list of N sequences to be plotted top down.
      Each sequence is an array of shape [seq_len_pred, n_seqs, n_channels, height, width].
    dataset_name: A string with the name of the dataset. Used to control
      output image normalization.
    case_id: The sequence ID (batch element) to plot. Defaults to 0.
  Raises:
    ValueError: if seq_len_total != seq_len_rec + seq_len_pred.
  Returns:
    seq_im: An array of shape [N*height, seq_len_total*width, n_channels]
  """
  seq_len_total = input_seq_list[0].shape[0]
  n_channels = input_seq_list[0].shape[2]
  height = input_seq_list[0].shape[3]
  width = input_seq_list[0].shape[4]
  num_seqs = len(input_seq_list)

  seq_im = np.empty([num_seqs * height, seq_len_total * width, n_channels])

  # # to seq_len x H x W x C
  # targets_rec_to_slice = np.transpose(
  #     targets_rec[:, case_id, ...], [0, 2, 3, 1])
  # targets_pred_to_slice = np.transpose(
  #     targets_pred[:, case_id, ...], [0, 2, 3, 1])
  #
  # reconstructions_to_slice = np.transpose(
  #     reconstructions[:, case_id, ...], [0, 2, 3, 1])
  # predictions_to_slice = np.transpose(
  #     predictions[:, case_id, ...], [0, 2, 3, 1])
  #
  # targets_to_slice = np.concatenate(
  #     [targets_rec_to_slice, targets_pred_to_slice],
  #     axis=0)
  # estimates_to_slice = np.concatenate(
  #     [reconstructions_to_slice, predictions_to_slice],
  #     axis=0)

  sliced_seqs = [np.transpose(seq[:, case_id, ...], [0, 2, 3, 1]) for seq in input_seq_list]

  for i in range(seq_len_total):
    for seq_idx in range(num_seqs):
        seq_im[(seq_idx*height):((seq_idx+1)*height),
               (i * width):((i + 1) * width),
               :] = sliced_seqs[seq_idx][i, ...]
    # add separators
    seq_im[:,i * width,:] = 0.5
    seq_im[:,(i + 1) * width - 1,:] = 0.5
  seq_im[0,:,:] = 0.5
  seq_im[-1,:,:] = 0.5
  for seq_idx in range(num_seqs)[1:]:
    seq_im[(seq_idx*height-1):(seq_idx*height+1),:,:] = 0.5


  if dataset_name != "moving_mnist":
    # Rescale from [-1, 1] to [0, 1]
    seq_im = seq_im / 2 + 0.5

  # Just in case, bound to [0, 1]
  seq_im[seq_im > 1] = 1
  seq_im[seq_im < 0] = 0

  return np.squeeze(seq_im)


def generate_sequence_images(
    input_seq_list,
    dataset_name,
    n_seqs):
  """Generates multiple image sequences."""
  return [
      generate_sequence_image(
          input_seq_list,
          dataset_name,
          case_id) for case_id in range(n_seqs)]


def pad_sequence(input_sequence, input_idxs, target_length, color=0.7):
    """Pads input sequence to match target_length. Input_idxs show where input imgs belong."""
    if not input_sequence.shape[0] == len(input_idxs):
        raise ValueError("Input sequence and input indices need to be of same length.")
    input_dims = list(input_sequence.shape[1:])
    output_seq = np.empty([target_length] + input_dims)
    added_inputs = 0
    for idx in range(target_length):
        if idx in input_idxs:
            output_seq[idx] = input_sequence[added_inputs]
            added_inputs += 1
        else:
            output_seq[idx] = np.ones(input_dims) * color
    return output_seq


def mod_angle(raw_angle):
    return np.mod(raw_angle + 2*np.pi, 2*np.pi)

def check_joint(joint_pt, base_pt, edge_img):
    middle_pt = [int((joint_pt[0] - base_pt[0]) / 2 + base_pt[0]),
                 int((joint_pt[1] - base_pt[1]) / 2 + base_pt[1])]
    return edge_img[middle_pt[0], middle_pt[1]]


ARM_LENGTH_PX = 13.0
MIN_DIST_TO_CENTER = 5.0
def compute_angle(img, one_joint=False):
    img = np.uint8(img * 255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    for ks in [4, 3, 2, 1]:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        edges_eroded = cv2.erode(edges_morphed, kernel2)
        idxs = np.where(edges_eroded)
        if idxs[0].size > 10:
            break

    center = edges.shape[0] / 2.0
    center_pt = [center, center]
    dist_to_center = np.sqrt(np.power(idxs[0] - center, 2) + np.power(idxs[1] - center, 2))
    joint_idxs = np.argsort(np.abs(dist_to_center - ARM_LENGTH_PX))
    for ii in range(joint_idxs.size):
        joint_pt = [idxs[0][joint_idxs[ii]], idxs[1][joint_idxs[ii]]]
        if check_joint(joint_pt, center_pt, edges_morphed):
            break
    theta_1 = mod_angle(np.arctan2(joint_pt[1] - center, joint_pt[0] - center))
    if one_joint:
      return theta_1

    dist_to_joint = np.sqrt(np.power(idxs[0] - joint_pt[0], 2) + np.power(idxs[1] - joint_pt[1], 2))
    too_close_to_center = np.where(dist_to_center < MIN_DIST_TO_CENTER)

    sort_idxs = np.argsort(np.abs(dist_to_joint - ARM_LENGTH_PX))
    end_idxs = sort_idxs[~np.in1d(sort_idxs, too_close_to_center)]
    if not end_idxs.size == 0:
        for ii in range(end_idxs.size):
            end_pt = [idxs[0][end_idxs[ii]], idxs[1][end_idxs[ii]]]
            if check_joint(end_pt, joint_pt, edges_morphed):
                break
        theta_2 = mod_angle(np.arctan2(end_pt[1] - joint_pt[1], end_pt[0] - joint_pt[0]))
    else:
        theta_2 = mod_angle(theta_1 - np.pi)
        print(theta_1*180/np.pi)
        print(theta_2*180/np.pi)
        plt.imshow(edges_eroded)
        plt.savefig("/tmp/edges.png")
        plt.imshow(img)
        plt.savefig("/tmp/img.png")
        _ = raw_input("...")
    return [theta_1, theta_2]


# TODO(karl): make font size dependent on image resolution
def annotate_angles(img_seq, angle_seq):
    """Adds angle values in top corner of images."""
    fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    angle_seq = np.asarray(angle_seq * 180 / np.pi, dtype=np.int)
    img_seq = np.asarray((img_seq+1.0) * 255/2, dtype=np.uint8)
    for seq in range(img_seq.shape[1]):
        for timestep in range(img_seq.shape[0]):
            if angle_seq.shape[2] == 2:
              angle_str = "%d:%d" % (angle_seq[timestep, seq, 0], angle_seq[timestep, seq, 1])
            else:
              angle_str = "%d" % (angle_seq[timestep, seq])  # oneJoint reacher
            img = Image.fromarray(np.transpose(img_seq[timestep, seq], (1, 2, 0)))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), angle_str, (0, 0, 0), font=fnt)
            img_seq[timestep, seq] = np.transpose(np.array(img), (2, 0, 1))
    img_seq = np.asarray(img_seq, dtype=np.float32) / 255.0 * 2 - 1
    return img_seq


def prep_gif_seqs_tb(gif_seqs):
    gif_seqs = gif_seqs[:min(len(gif_seqs), 10)]
    max_seq_length = 0
    for seq in gif_seqs:
        max_seq_length = max(max_seq_length, seq.shape[0])
    gif_seqs = [np.concatenate((np.uint8(frame * 255),
                    np.zeros([max_seq_length - frame.shape[0] + 1] + list(frame[0].shape), dtype=np.uint8)), axis=0)
                    for frame in gif_seqs]
    i_shape = gif_seqs[0].shape
    padding = 255 * np.ones((i_shape[0], i_shape[1], 2, i_shape[-1]), dtype=np.uint8)
    frame_list = []
    for _ in gif_seqs:
        frame_list.append(_)
        frame_list.append(padding)
    prep_gif_frame_seqs = [np.concatenate(frame_list, axis=2)]
    return prep_gif_frame_seqs


def init_html(base_dir, iteration, make_parent=False):
    if make_parent:
      web_dir = os.path.join(base_dir, 'web')
      webpage = html.HTML(web_dir,
                        'Experiment name = %s, Iteration: %d ' % (os.path.normpath(base_dir),
                                                                 iteration),
                          reflesh=1,
                          sub_dir='it_%d' % iteration)
    else:
      web_dir = os.path.join(base_dir, 'web/it_%d' % iteration)
      webpage = html.HTML(web_dir,
                        'Experiment name = %s, Iteration: %d ' % (os.path.normpath(base_dir),
                                                                 iteration),
                          reflesh=1)
    return webpage


_MAX_N_GIFS = 20    # do not generate more gifs per evaluation -> slow and no benefit!
def dump_gif_to_html(img_seqs, global_step, webpages, name):
    # dump sequences in gifs in image directory
    img_dir = webpages[0].get_image_dir()
    gif_paths = []
    for img_seq_idx in range(min(len(img_seqs), _MAX_N_GIFS)):
        img_seq = img_seqs[img_seq_idx]
        imgs = [np.uint8((img_seq[i])*255) for i in range(img_seq.shape[0])]
        imgs.append(np.zeros(imgs[0].shape, dtype=np.uint8))
        output_fname = "%s_%d_seq_%d.gif" % (name, global_step, img_seq_idx)
        gif_path = os.path.join(img_dir, output_fname)
        save_gif(gif_path, imgs, fps=3)
        gif_paths.append(output_fname)

    for webpage in webpages:
      webpage.add_table()
      webpage.add_row([name], [len(gif_paths)])
      webpage.add_images(gif_paths, [None]*len(gif_paths), gif_paths, [1]*len(gif_paths), height=256, width=None)
    return webpages


def create_concat_seqs(target_seq, estimate_seq, n_input_frames):
    """Concatenates images to one sequence with GT on top and estimate below."""
    # stack target over estimate
    if not target_seq.shape[-1] == 3:    # transpose to NHWC
        target_seq = np.transpose(target_seq, axes=(0, 1, 3, 4, 2))
        estimate_seq = [np.transpose(e, axes=(0, 2, 3, 1)) for e in estimate_seq]

    fused_seqs = []
    for i in range(len(estimate_seq)):
      target_seq_i = target_seq[:estimate_seq[i].shape[0],i]
      estimate_seq_i = estimate_seq[i][:target_seq_i.shape[0]]
      fused_seq = np.concatenate((target_seq_i, estimate_seq_i), axis=1)

      # add small boundary in between images: red for past, green for future
      middle_px = int(fused_seq.shape[1] / 2)
      if n_input_frames != 1:
          red_shape = np.zeros((n_input_frames, 2,
                                target_seq_i.shape[1], target_seq_i.shape[3]))
          red_shape[..., 0] = 1.0
          fused_seq[:n_input_frames, middle_px - 1:middle_px + 1, ...] = red_shape

      green_shape = np.zeros((estimate_seq_i.shape[0] - n_input_frames, 2,
                            estimate_seq_i.shape[1], estimate_seq_i.shape[3]))
      if green_shape.shape[3] > 1:
          green_shape[..., 1] = 1.0
      else:
          green_shape[..., 0] = 1.0
      fused_seq[n_input_frames:, middle_px - 1:middle_px + 1, ...] = green_shape
      fused_seqs.append(fused_seq)

    return fused_seqs


def stack_seqs(seq_list, swap_channels=False, red_line=None):
  """assumes input dimension 4 (time x img_dims)"""
  num_imgs, _, img_res, _ = seq_list[0].shape
  if swap_channels:
    seq_list = [np.transpose(seq, (0, 2, 3, 1)) for seq in seq_list]   # NCHW -> NHWC
  seq_list = [np.concatenate(np.split(seq, seq.shape[0], axis=0), axis=2)[0] for seq in seq_list]  # concat in time
  output_img = np.concatenate(seq_list, axis=0)   # concat seqs vertically
  # add vertical lines
  for i in range(num_imgs)[1:]:
    output_img[:, i*img_res - 1:i*img_res+1] = 1.0
  # add horizontal lines
  for i in range(len(seq_list))[1:]:
    output_img[i*img_res - 1:i*img_res+1, :] = 1.0
  # Add the loss line
  if red_line is not None:
    output_img[:, red_line*img_res - 1:red_line*img_res+1] = 0.3
  return output_img


def store_latent_samples(val_test_output,
                         base_dir,
                         val_test_batch_idx,
                         n_val_test_batches,
                         global_train_iteration,
                         store_angle_regressor,
                         store_comp_latents):
  def save_latents(folder_name, latents, total_latents):
    folder = os.path.join(base_dir, folder_name)
    if not os.path.exists(folder):
      os.mkdir(folder)
    filename = os.path.join(base_dir, folder_name, "seq_{}".format(global_train_iteration))

    if total_latents is not None:
      st = val_test_batch_idx * latents.shape[1]
      end = (val_test_batch_idx + 1) * latents.shape[1]
      total_latents[:, st:end, :] = latents
    else:
      total_latents = np.repeat(np.zeros(latents.shape), n_val_test_batches, axis=1)
    if val_test_batch_idx == n_val_test_batches - 1:
      np.save(filename, total_latents)
    return total_latents

  if val_test_batch_idx == 0:
    prev_samples = None
    prev_means = None
    prev_actions = None
    prev_stds = None
    prev_aa = None
    prev_comp_z = None
  prev_samples = save_latents(
    "z_samples", val_test_output["inference_z_samples_val"], prev_samples)
  prev_means = save_latents(
    "z_means", val_test_output["inference_z_means_val"], prev_means)
  prev_actions = save_latents(
    "actions", val_test_output["actions_true"][:-1], prev_actions)
  prev_stds = save_latents(
    "stds", val_test_output["inference_z_stds_val"], prev_stds)
  if store_angle_regressor:
    prev_aa = save_latents(
      "aa", val_test_output["abs_actions_true"][:-1], prev_aa)
  if store_comp_latents:
    prev_comp_z = save_latents(
      "comp_z", val_test_output["comp_z_sample_seq_val"], prev_comp_z)
  print("storing z samples")


''' Draw actions on a tensor of images with last few dimensions with shape 
    (c, h, w).

Args:
    images: a tensor of images with pixel value range [0, 1] and type float.
    actions: a tensor of actions.

Returns: output images 
'''
def draw_actions_on_images(images, actions):
  # images have shape [t, bs, c, h, w]; actions have shappe [t, bs, ac_dim]
  images_shape = images.shape
  actions_shape = actions.shape

  # loop shape is [t, bs] - this is how we loop all images in the images tensor
  assert(images_shape[:-3] == actions_shape[:-1])
  loop_shape = images_shape[:-3]

  out_images = images.copy()

  for ix in list(itertools.product(*list(map(range, loop_shape)))):
    # get a single image and change its shape to (h, w, c), range to [0, 255]
    image = (images[ix].transpose((1, 2, 0)) * 255).copy()
    action = actions[ix]

    h, w = images_shape[-2:]
    multiplier = min(h, w) / 32
    thickness = max(1, int(np.round(multiplier / 4)))

    # calculate coordinates to draw action arrow
    action_direction = action[:2] * np.array([1, -1])
    start_pt = np.array([h - float(h) * 5 / 32, float(w) * 5 / 32]).astype(int).tolist()
    end_pt = (start_pt + action_direction * multiplier * 3).astype(int).tolist()
    
    # actual arrow drawing
    white = (255,) * 3
    cv2.arrowedLine(image, tuple(start_pt), tuple(end_pt), white, thickness, cv2.LINE_AA)

    # rectangle around arrow
    r = int(multiplier * 4)
    corner1 = (start_pt[0] - r, start_pt[1] - r)
    corner2 = (start_pt[0] + r, start_pt[1] + r)
    cv2.rectangle(image, corner1, corner2, white, thickness, cv2.LINE_AA)

    # save to out image
    out_images[ix] = image.transpose((2, 0, 1)) / 255
  
  return out_images


class TestSeqSaver(object):
  def __init__(self,
               base_dir,
               store_transplants,
               store_abs_actions,
               action_conditioned_pred):
    self.base_dir = base_dir
    self.store_transplants = store_transplants
    self.store_abs_actions = store_abs_actions
    self.action_cond_pred = action_conditioned_pred
    self.test_batch_seqs_gt, self.test_batch_seqs = [], []
    if self.store_abs_actions:
      self.abs_action_seqs = []
    if self.store_transplants:
      self.transplanted_seqs = []

  def record_test_seqs(self, val_test_output):
    self.test_batch_seqs_gt.append(val_test_output["predict_images_true"])
    self.test_batch_seqs.append(val_test_output["decoded_seq_predict"])
    if self.store_abs_actions:
      self.abs_action_seqs.append(val_test_output["abs_actions_true"])
    if self.store_transplants:
      self.transplanted_seqs.append(val_test_output["transplanted_seq_predict"])

  def write_test_seqs(self):
    test_batch_seqs_gt = np.concatenate(self.test_batch_seqs_gt, axis=1)
    test_batch_seqs = np.concatenate(self.test_batch_seqs, axis=1)
    np.save(os.path.join(self.base_dir, "test_seqs_gt.npy"), test_batch_seqs_gt)
    np.save(os.path.join(self.base_dir, "test_seqs.npy"), test_batch_seqs)
    if self.store_abs_actions:
      abs_action_seqs = np.concatenate(self.abs_action_seqs, axis=1)
      np.save(os.path.join(self.base_dir, "abs_action_seqs_gt.npy"), abs_action_seqs)
    if self.store_transplants:
      transplanted_seqs = np.concatenate(self.transplanted_seqs, axis=1)
      np.save(os.path.join(self.base_dir, "transplanted_seqs_gt.npy"), transplanted_seqs)
    print("Saved test sequences!")


class EvalSaver(object):
    def __init__(self,
                 log_keys,
                 base_dir,
                 test_batch_size,
                 global_step):
        """
        Saves results of evaluation to HDF5 file for later visualization / evaluation.
        :param log_keys: List of strings for network outputs that should be saved.
        :param base_dir: Base directory in which the Saver creates a 'test' directory to store the results.
        :param test_batch_size: Batch size of evaluation runs -> used to find concatenation axis when dumping results.
        :param global_step: training iteration that test is run on
        """
        self._test_batch_size = test_batch_size
        self._global_step = global_step
        self._test_dir = os.path.join(base_dir, "test")
        if not os.path.exists(self._test_dir):
            os.makedirs(self._test_dir)
        self._eval_outputs = dict.fromkeys(log_keys)
        for key in self._eval_outputs:
            self._eval_outputs[key] = []

    def log_results(self, res_dict):
        """Logs the results of a test batch."""
        for key in self._eval_outputs:
            self._eval_outputs[key].append(res_dict[key])

    def dump_results(self, filename="eval_results.h5"):
        """Dumps all logged results to a HDF5 file, concatenates along batch dimension."""
        h5f = h5py.File(os.path.join(self._test_dir, filename), 'w')
        for key in self._eval_outputs:
            # concatenate results along batch dimension
            batch_dim = np.where(np.asarray(self._eval_outputs[key][0].shape) == self._test_batch_size)[0]
            if not batch_dim.size:
                raise ValueError("No batch dimension found in output %s!" % key)
            batch_dim = batch_dim[-1]   # in case there are multiple dims of the same size take the last one as batch
            result = np.concatenate(self._eval_outputs[key], axis=batch_dim)
            h5f.create_dataset(key, data=result)
        h5f.create_dataset("global_step", data=self._global_step)
        h5f.close()
