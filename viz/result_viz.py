import tensorflow as tf
import numpy as np

from viz import viz_utils
from architectures import th_utils


FLAGS = tf.flags.FLAGS

def gen_composed_hierarchical_seqs(low_level_seqs, dt, n_seqs):
  n_segs, batch_size, seg_len = dt.shape
  assert(n_seqs <= batch_size, "Number of requested vis seqs in larger than batch_size!")
  output_seqs = [None for _ in range(n_seqs)]
  keyframe_idxs = [[] for _ in range(n_seqs)]
  max_dt = np.argmax(dt, axis=2)
  for seq_idx in range(n_seqs):
    for seg_idx in range(n_segs):
      seg_subseq = low_level_seqs[seg_idx, :(max_dt[seg_idx, seq_idx]+1), seq_idx]
      if output_seqs[seq_idx] is None:
        output_seqs[seq_idx] = seg_subseq
      else:
        output_seqs[seq_idx] = np.concatenate((output_seqs[seq_idx], seg_subseq), axis=0)
      keyframe_idxs[seq_idx].append(output_seqs[seq_idx].shape[0]-1)
  return output_seqs, keyframe_idxs


def mark_keyframes(seqs, is_keyframe):
  for seq_idx in range(seqs.shape[1]):
    for frame_idx in range(seqs.shape[0]):
      if is_keyframe[frame_idx, seq_idx]:
        # bold frame of keyframe
        seqs[frame_idx, seq_idx, :, :, :2] = 1.0
        seqs[frame_idx, seq_idx, :, :, -2:] = 1.0
        seqs[frame_idx, seq_idx, :, :2, :] = 1.0
        seqs[frame_idx, seq_idx, :, -2:, :] = 1.0
  return seqs


def gen_hierarchical_plot_imgs(gt_seqs_condition, gt_seqs_predict, predicted_seqs,
                               target_seqs, keyframes, keyframe_idxs, attention_weights, gt_keyframe_idxs):
  """
  
  :param gt_seqs_condition:
  :param gt_seqs_predict:
  :param predicted_seqs:
  :param target_seqs:
  :param keyframes:
  :param keyframe_idxs:
  :param attention_weights: array of n_keyframes x n_frames x batch_size
  :return:
  """
  show_gt_keyframes = gt_keyframe_idxs is not None and gt_keyframe_idxs.size > 0  # if exist
  if show_gt_keyframes:
    gt_seqs_predict = mark_keyframes(gt_seqs_predict, gt_keyframe_idxs)
  gt_seqs = np.concatenate((gt_seqs_condition, gt_seqs_predict), axis=0)
  conditioning_offset = gt_seqs_condition.shape[0]
  output_imgs = []
  _, _, _, img_res, _ = gt_seqs.shape
  max_frames = FLAGS.n_frames_segment * FLAGS.n_segments + FLAGS.input_seq_len
  for seq_idx in range(len(predicted_seqs)):
    stack_list = []
    def pad_seq(seq, offset=1):
      idxs = [i + conditioning_offset * offset for i in range(len(seq))]
      return viz_utils.pad_sequence(seq, idxs, max_frames)

    stack_list.append(pad_seq(gt_seqs[:, seq_idx], offset=False))
    # generate keyframe seq
    if keyframe_idxs is not None:
      keyframe_idxs_i = [i + conditioning_offset for i in keyframe_idxs[seq_idx]]
      keyframe_seq_i = viz_utils.pad_sequence(keyframes[:, seq_idx], keyframe_idxs_i, max_frames)
      stack_list.append(keyframe_seq_i)

    stack_list.append(pad_seq(predicted_seqs[seq_idx]))
    if target_seqs is not None:
      stack_list.append(pad_seq(target_seqs[seq_idx]))
      n_loss_frames = th_utils.get_future_loss_length() + conditioning_offset
    else:
      n_loss_frames = None
  
    output_img = viz_utils.stack_seqs(stack_list, swap_channels=True, red_line=n_loss_frames)
    
    # Attention visualization
    if attention_weights is not None:
      n_keys, n_frames, _ = attention_weights.shape
      attention_seq = attention_weights[:, :max_frames, seq_idx]
      if FLAGS.use_full_inf:
        attention_idxs = np.arange(attention_seq.shape[1])
      else:
        attention_idxs = np.arange(attention_seq.shape[1]) + conditioning_offset
      attention_seq = viz_utils.pad_sequence((attention_seq.T),
                                             attention_idxs,
                                             max_frames,
                                             color=0).T
      n_channels = output_img.shape[2]
      att_height = 2
      attention_seq = np.tile(attention_seq.reshape([n_keys, 1, max_frames, 1, 1]),
                              [1, att_height, 1, img_res, n_channels]).reshape([n_keys * att_height, max_frames * img_res, n_channels])
      output_img = np.concatenate([attention_seq, output_img], axis=0)

    if output_img.shape[-1] == 1:
      output_img = output_img[..., 0]   # in case we have a grayscale image reduce channel dimension
    output_imgs.append(output_img)
  return output_imgs


def gen_segment_overviews(segments, dt, n_seqs):
  n_seg, _, seg_len = dt.shape
  output_imgs = []
  for seq_idx in range(n_seqs):
    seg_seq_stack = []
    for seg_idx in range(n_seg):
      segment = segments[seg_idx, :, seq_idx]
      segment = np.transpose(segment, (0, 2, 3, 1))   # put channel to last dim
      for seg_img_idx in range(seg_len):
        segment[seg_img_idx, :2] = dt[seg_idx, seq_idx, seg_img_idx]    # color top of img in gray indicating weight
      segment = np.concatenate(np.split(segment, segment.shape[0], axis=0), axis=2)[0]
      if segment.shape[-1] == 1:
        segment = segment[..., 0]   # remove last channel for grayscale img
      seg_seq_stack.append(segment)
    output_imgs.append(np.concatenate(seg_seq_stack, axis=0))
  return output_imgs


def gen_html_summary(gif_frame_seqs, iteration, base_dir):
  webpage = viz_utils.init_html(base_dir, iteration)
  webpage_parent = viz_utils.init_html(base_dir, iteration, make_parent=True)
  webpages = [webpage, webpage_parent]

  webpage = viz_utils.dump_gif_to_html(gif_frame_seqs, iteration, webpages, "Predictions")

  # save webpage to subfolder for iteration and update global summary
  [wp.save() for wp in webpages]


def log_sess_output(sess_output,
                    monitor_index,
                    logger,
                    iteration,
                    dataset_name,
                    base_dir,
                    n_seqs=5,
                    phase="train",
                    build_seq_ims=True,
                    repeat=0,
                    is_hierarchical=False):
  """Logs the session output.

  Args:
    sess_output: A dictionary of evaluated values (not tensors) to log.
    monitor_index: The index of Tensor phase and types
    logger: A Logger object.
    iteration: The current training iteration.
    dataset_name: A string with the name of the dataset.
    n_seq: Number of sequences that should be logged at max. Defaults to 5.
    phase: The current phase, "train" or "val".
    build_seq_ims: If True, will be build sequence images from the ground truth
      and estimates returned by evaluation.. Defaults to True.
    base_dir: Base directory for saving visualization results.
  Raises:
    ValueError if phase is not "train" or "val".
  """
  for type_key, type_vals in monitor_index[phase].items():
    for type_ind in type_vals:
      if type_key in ["scalar", "metric", "loss"]:
        if type_ind in sess_output:
          logger.log_scalar(
            tag=type_ind,
            value=sess_output[type_ind],
            step=iteration)
      elif type_key == "hist":
        if type_ind in sess_output:
          logger.log_histogram(
            tag=type_ind,
            values=sess_output[type_ind],
            step=iteration)
      elif type_key == "sum":
        if type_ind in sess_output:
          logger.log_summary(sess_output[type_ind], step=iteration)

  if build_seq_ims:
    if phase == "train":
      s = ""
    elif phase == "val":
      s = "_val"
    else:
      raise NotImplementedError("Visualization is currently only implemented for train and val!")

    if ("low_level_images" + s) not in sess_output.keys():
      return

    # Build sequence image to save
    n_seqs = min(n_seqs, sess_output["input_images" + s].shape[1])

    # Draw action arrows if dataset is top
    if FLAGS.dataset_config_name == 'top':
        input_seq_len = sess_output["input_images" + s].shape[0]
        predict_seq_len = sess_output["predict_images" + s].shape[0]
        sess_output["input_images" + s] = viz_utils.draw_actions_on_images(sess_output["input_images" + s],
                                                    sess_output["actions" + s][:input_seq_len])
        sess_output["predict_images" + s] = viz_utils.draw_actions_on_images(sess_output["predict_images" + s],
                                                      sess_output["actions" + s][input_seq_len:input_seq_len + predict_seq_len])

        if 'regressed_actions' + s in sess_output:
            sess_output["low_level_images" + s] = viz_utils.draw_actions_on_images(sess_output["low_level_images" + s],
                                                                                   sess_output["regressed_actions" + s])

    if is_hierarchical:
      composed_seqs, keyframe_idxs = gen_composed_hierarchical_seqs(sess_output["low_level_images" + s],
                                                                    sess_output["dt" + s],
                                                                    n_seqs)
      postfix = "" if phase is "train" else "_val"
      if "gt_target_low_level_images"+postfix in sess_output:
        composed_targets = sess_output["gt_target_low_level_images"+postfix]
        composed_targets = np.split(composed_targets, composed_targets.shape[1], axis=1)
        composed_targets = [ct[:, 0] for ct in composed_targets]
      else:
        composed_targets, _ = gen_composed_hierarchical_seqs(sess_output["low_level_image_targets" + s],
                                                                      sess_output["dt" + s],
                                                                      n_seqs)
    else:
      split_output_seqs = np.split(sess_output["low_level_images" + s],
                                   sess_output["low_level_images" + s].shape[1], axis=1)
      split_output_seqs = [so[:, 0] for so in split_output_seqs[:n_seqs]]

    if is_hierarchical:
      kfs = sess_output["high_level_images" + s]
      kf_idxs = keyframe_idxs
      gt_kf_idxs = sess_output["actions_abs" + s]\
          [-(sess_output["predict_images" + s].shape[0]+FLAGS.n_frames_segment):-FLAGS.n_frames_segment, :, 0]
    elif "kl_based_kfs"+s in sess_output:
      kfs = sess_output["kl_based_kfs" + s][:, :n_seqs]
      kf_idxs = [np.where(sess_output["kl_based_kfs_idxs" + s][:, i])[0] for i in range(n_seqs)]
      gt_kf_idxs = sess_output["actions_abs" + s]\
          [sess_output["input_images" + s].shape[0]:, :, 0]
    else:
      kfs = None
      kf_idxs = None
      gt_kf_idxs = None

    plot_imgs = gen_hierarchical_plot_imgs(sess_output["input_images" + s],
                                           sess_output["predict_images" + s],
                                           composed_seqs if is_hierarchical else split_output_seqs,
                                           composed_targets if is_hierarchical else None,
                                           kfs,
                                           kf_idxs,
                                           sess_output["attention_weights" + s] if is_hierarchical else None,
                                           gt_kf_idxs)

    logger.log_images(
      tag="image_predictions" + s,
      images=plot_imgs,
      step=iteration
    )

    # build kl overview for kl-based keyframes
    if "kl_based_kfs_kl" + s in sess_output:
      for suffix in ["", "_reencode"]:
        kl = sess_output["kl_based_kfs"+suffix+"_kl" + s]
        figs = []
        for idx in range(n_seqs):
          figs.append(viz_utils.plot_single_kl_seq(kl[:, idx], gt_kf_idxs[:, idx]))
        logger.log_figures(
          tag="kl_values" + suffix + s,
          figures=figs,
          step=iteration
        )
        viz_utils.close_figs(figs)


    # build segment overview image
    if is_hierarchical:
      overview_imgs = gen_segment_overviews(sess_output["low_level_images" + s], sess_output["dt" + s], n_seqs)
      logger.log_images(
        tag="segment_overview" + s,
        images=overview_imgs,
        step=iteration
      )

      if "low_level_image_targets" + s in sess_output:
        overview_targets = gen_segment_overviews(sess_output["low_level_image_targets" + s], sess_output["dt" + s], n_seqs)
        logger.log_images(
          tag="segment_targets_overview" + s,
          images=overview_targets,
          step=iteration
        )


    # cut gt seq to output seq length
    gt_seq = np.concatenate((sess_output["input_images" + s], sess_output["predict_images" + s]), axis=0)
    if is_hierarchical:
      est_seqs = []
      for comp_seq, kf_idxs in zip(composed_seqs, keyframe_idxs):
        for kf_idx in kf_idxs:
          comp_seq[kf_idx, :, :2, :] = 1.0    # mark keyframes with white bar at the top
        dummy_seq = np.zeros(([sess_output["input_images" + s].shape[0]] + list(comp_seq.shape[-3:])))
        est_seqs.append(np.concatenate((dummy_seq, comp_seq), axis=0))
    else:
      est_seqs = split_output_seqs

    # log gifs to tensorboard and html
    gif_frame_seqs = viz_utils.create_concat_seqs(gt_seq,
                                                  est_seqs,
                                                  n_input_frames=sess_output["input_images" + s].shape[0] - 1)
    prep_gif_frame_seqs = viz_utils.prep_gif_seqs_tb(gif_frame_seqs)
    logger.log_gifs(
      tag="gif_image_predictions" + s,
      gif_images=prep_gif_frame_seqs,
      step=iteration
    )

    if phase == "val":
      if tf.flags.FLAGS.gen_html_summary:
        gen_html_summary(gif_frame_seqs, iteration, base_dir)
  return


if __name__ == "__main__":
  sess_output = np.load("/home/karl/Downloads/sess_output.npy").item()
