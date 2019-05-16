""" Temporal Hierarchy utils """

import tensorflow as tf
import numpy as np
from utils import batchwise_1d_convolution, tf_swapaxes, add_n_dims, shape

FLAGS = tf.flags.FLAGS

def get_future_loss_length():
  """The length of the ground truth sequence in the future that the prediction loss is be applied to"""
  return FLAGS.pred_seq_len

def get_future_input_length():
  """The length of the ground truth sequence that is fed into the inference network"""
  return FLAGS.pred_seq_len + FLAGS.n_frames_segment


def get_high_level_targets(
      confidence_weights,
      propagated_distributions,
      targets):
  """ Returns soft targets, which are averages of candidate targets by the chance of them being the target

  Note: as this gets called several times, every targets-independent computation should be outside this function

  :param confidence_weights: shape = segments x batchsize
  :param propagated_distributions: list of tensors batchsize x frames_in_distribution, len = segments
  :param targets: tensor time x batchsize x sample_dimensions...
  :return:
  - soft_targets: tensor  segments x batchsize x sample_dimensions...
  """
  
  soft_targets = []
  targets = tf_swapaxes(targets, 0, 1)
  num_segments = len(propagated_distributions)
  targets_shape = targets.get_shape().as_list()
  n_frames_predict = targets_shape[1]
  targets_dim = len(targets_shape) - 2
  
  for t in range(num_segments):
    propagated_distribution = propagated_distributions[t]
    cut = min(propagated_distribution.get_shape().as_list()[1], n_frames_predict)
    
    # alternative: soft_target = tf.einsum('ij,ijklm->jklm', propagated_distribution, targets)
    soft_target = tf.reduce_sum(tf.multiply(add_n_dims(propagated_distribution[:, :cut], targets_dim),
                                            targets[:, :cut]),
                                axis=1)
    # normalize so that it is always a convex combination
    soft_targets.append(soft_target / add_n_dims(confidence_weights[t], targets_dim - 1))
  
  soft_targets = tf.stack(soft_targets, axis=0)
  return soft_targets


def get_propagated_distributions(
      offset_distributions,
      n_loss_frames):
  """ Transforms segment-wise offset into sequence-wise offsets

  :param offset_distributions: tensor segments x batchsize x frames_in_segment
  :return:
  - confidence_weights: tensor of the probabilities that the frame in question will be used for constructing
  the result sequence. shape = segments x batchsize
  - propagated_distributions: list of tensors batchsize x frames_in_distribution, len = segments
  Each tensor is the distribution of possible time positions of the next keyframe in the sequence
  """
  
  num_segments = offset_distributions.get_shape().as_list()[0]
  confidence_weights = []
  propagated_distributions = []
  propagated_distribution = offset_distributions[0]  # batch_size x seg_length
  
  for t in range(num_segments):
    cut = min(propagated_distribution.get_shape().as_list()[1], n_loss_frames)
    confidence_weights.append(tf.reduce_sum(propagated_distribution[:, :cut], axis=1)[:, None])
    # add constant to all 0 values to avoid NaNs
    mask = tf.equal(confidence_weights[-1], 0)
    confidence_weights[-1] += 1e-12 * tf.cast(mask, confidence_weights[-1].dtype)
    
    propagated_distributions.append(propagated_distribution)
    if t + 1 < num_segments:
      # this incorporates the next dt into the current distribution
      propagated_distribution = batchwise_1d_convolution(propagated_distribution,
                                                         tf.pad(offset_distributions[t + 1], [[0, 0], [1, 0]]),
                                                         proper_convolution=True,
                                                         padding="MAX")
  
  confidence_weights = tf.stack(confidence_weights, axis=0)
  return confidence_weights, propagated_distributions


def get_low_level_targets(
      offset_distributions,
      targets,
      propagated_distributions,
      coord_targets):
  """ Returns soft targets, which are averages of candidate targets by the chance of them being the target

  This is the loss from the Temporal Hierarchy model
  :param offset_distributions: tensor segments x batchsize x frames_in_segment
  :param targets: tensor time x batchsize
  :param propagated_distributions: list of tensors batchsize x frames_in_distribution, len = segments
  :return:
  - soft_targets: tensor segments x frames_in_segment x batchsize x image dimensions...
  - confidence_weights: tensor segments x frames_in_segment x batchsize
  """
  
  targets = tf.transpose(targets, [1, 0, 2, 3, 4])
  if coord_targets is not None:
    coord_targets = tf.transpose(coord_targets, [1, 0, 2])
  num_segments = offset_distributions.get_shape().as_list()[0]
  n_frames_segment = offset_distributions.get_shape().as_list()[2]
  n_frames_predict = targets.get_shape().as_list()[1]
  batch_size = targets.get_shape().as_list()[0]
  
  soft_targets, soft_coord_targets = [], []
  confidence_weights = []
  
  # Pad the distributions for every segment so that they are the same size
  # this only doubles the computation needed for the loss, but we don't need nested loops
  propagated_distributions = propagated_distributions[:-1]  # discard the last segment
  for t in range(num_segments - 1):
    propagated_distribution = propagated_distributions[t]
    length = propagated_distribution.get_shape().as_list()[1]
    if length > n_frames_predict:
      propagated_distribution = propagated_distribution[:, :n_frames_predict]
    if length < n_frames_predict:
      propagated_distribution = tf.pad(propagated_distribution, [[0, 0], [0, n_frames_predict - length]])
    propagated_distributions[t] = propagated_distribution
  
  propagated_distributions = tf.stack(propagated_distributions)
  tiled_targets = tf.tile(tf.expand_dims(targets, axis=0), [num_segments, 1, 1, 1, 1, 1])
  if coord_targets is not None:
    tiled_coord_targets = tf.tile(tf.expand_dims(coord_targets, axis=0), [num_segments, 1, 1, 1])
  
  # Transform the distributions for keyframes to distributions of first frames in a segment
  propagated_distributions = tf.pad(propagated_distributions[:, :, :-1], [[0, 0], [0, 0], [1, 0]])
  # Add the first segment
  propagated_distributions = tf.concat([tf.concat([tf.ones([1, batch_size, 1]),
                                                   tf.zeros([1, batch_size, n_frames_predict - 1])], axis=2),
                                        propagated_distributions], axis=0)
  # Construct the loss
  segment_continues = tf.ones((num_segments, batch_size))  # confidence of the segment not ending at this frame
  for s in range(n_frames_segment):
    start = s
    end = n_frames_predict - start
    
    soft_target = tf.reduce_sum(tf.multiply(propagated_distributions[:, :, :end, None, None, None],
                                            tiled_targets[:, :, start:]), axis=2)
    if coord_targets is not None:
      soft_coord_target = tf.reduce_sum(tf.multiply(propagated_distributions[:, :, :end, None],
                                                    tiled_coord_targets[:, :, start:]), axis=2)
    confidence_weight = tf.reduce_sum(propagated_distributions[:, :, :end], axis=2)
    # add constant to all 0 values to avoid NaNs
    mask = tf.equal(confidence_weight, 0)
    confidence_weight += 1e-12 * tf.cast(mask, confidence_weight.dtype)
    
    soft_target = soft_target / confidence_weight[:, :, None, None, None]
    if coord_targets is not None:
      soft_coord_target = soft_coord_target / confidence_weight[:, :, None]
    # The low-level weights are (p(frame is before the last gt frame)) * (p(frame is used in its segment))
    confidence_weight = tf.multiply(confidence_weight, segment_continues)
    segment_continues = segment_continues - offset_distributions[:, :, s]
    
    confidence_weights.append(confidence_weight[:, :, None, None, None])
    soft_targets.append(soft_target)
    if coord_targets is not None:
      soft_coord_targets.append(soft_coord_target)
  
  soft_targets = tf.stack(soft_targets, axis=1)
  confidence_weights = tf.stack(confidence_weights, axis=1)
  if coord_targets is not None:
    soft_coord_targets = tf.stack(soft_coord_targets, axis=1)
  return soft_targets, confidence_weights, soft_coord_targets


def dists_keyframe_to_first_segment(propagated_distributions, n_loss_frames):
  """

  :param propagated_distributions: list of tensors batchsize x frames_in_distribution, len = segments
  :param n_loss_frames:
  :return:
  """
  num_segments = len(propagated_distributions)
  batch_size = shape(propagated_distributions[0])[0]
  
  # Pad the distributions for every segment so that they are the same size
  # this only doubles the computation needed for the loss, but we don't need nested loops
  propagated_distributions = propagated_distributions[:-1]  # discard the last segment
  for t in range(num_segments - 1):
    propagated_distribution = propagated_distributions[t]
    length = propagated_distribution.get_shape().as_list()[1]
    if length > n_loss_frames:
      propagated_distribution = propagated_distribution[:, :n_loss_frames]
    if length < n_loss_frames:
      propagated_distribution = tf.pad(propagated_distribution, [[0, 0], [0, n_loss_frames - length]])
    propagated_distributions[t] = propagated_distribution
  
  propagated_distributions = tf.stack(propagated_distributions)
  
  # Transform the distributions for keyframes to distributions of first frames in a segment
  propagated_distributions = tf.pad(propagated_distributions[:, :, :-1], [[0, 0], [0, 0], [1, 0]])
  # Add the first segment
  propagated_distributions = tf.concat([tf.concat([tf.ones([1, batch_size, 1]),
                                                   tf.zeros([1, batch_size, n_loss_frames - 1])], axis=2),
                                        propagated_distributions], axis=0)
  return propagated_distributions


def get_low_level_gt_targets(
      offset_distributions,
      predictions,
      fs_seqwise_distributions,
      n_loss_frames):
  """ Returns soft targets, which are averages of candidate targets by the chance of them being the target

  This is the loss from the Temporal Hierarchy model
  :param offset_distributions: tensor segments x batchsize x frames_in_segment
  :param predictions: tensor segments x frames_in_segment x batchsize x image dimensions...
  :param fs_seqwise_distributions: tensor segments x batchsize x n_frames_sequence
  :return:
  - soft_targets: tensor time x batchsize x image dimensions...
  """
  
  num_segments, n_frames_segment, batch_size = predictions.get_shape().as_list()[:3]
  frame_dimensions = predictions.get_shape().as_list()[3:]
  targets_dim = len(frame_dimensions)
  
  # Construct the targets
  segment_continues = tf.ones((num_segments, batch_size))  # confidence of the segment not ending at this frame
  soft_gt_targets = tf.zeros([batch_size, n_loss_frames] + frame_dimensions)
  probs = tf.zeros((batch_size, n_loss_frames))
  for s in range(n_frames_segment):
    start = s
    end = n_loss_frames - start
    
    cur_probs = fs_seqwise_distributions[:, :, :end] * segment_continues[:, :, None]
    soft_gt_targets += tf.pad(tf.reduce_sum(predictions[:, s, :, None] * add_n_dims(cur_probs, targets_dim), axis=0),
                              [[0, 0], [start, 0]] + [[0, 0]] * targets_dim)
    probs += tf.pad(tf.reduce_sum(cur_probs, axis=0), [[0, 0], [start, 0]])
    segment_continues = segment_continues - offset_distributions[:, :, s]
  
  soft_gt_targets = soft_gt_targets / add_n_dims(probs, targets_dim)
  
  return tf.transpose(soft_gt_targets, [1, 0] + list(range(2, 2 + len(frame_dimensions))))


def comp_soft_gt_targets(dt, x, n_frames):
  """UNUSED"""
  with tf.name_scope("soft_gt_target_comp"):
    num_segments, batch_size, frames_per_segment = dt.get_shape().as_list()
    n_dim_data = len(x.get_shape().as_list()[3:])
    # create combinatorial matrix of scenarios, add one option (index 0) for "don't care"
    cm = np.array(np.meshgrid(*[range(frames_per_segment + 1) for _ in range(num_segments)])).T.reshape(-1,
                                                                                                        num_segments)
  
    # delete scenarios in which non-zero is chosen after 0 element
    del_idxs = []
    for i in range(cm.shape[0]):
      for j in range(num_segments - 1):
        if cm[i, j] == 0 and np.sum(cm[i, j:]) != 0:
          del_idxs.append(i)
          break
    cm = np.delete(cm, del_idxs, axis=0)
    cm = cm[1:]  # remove first row which has only 0 (no valid scenario)
  
    # make scenario matrix one-hot
    cm_onehot = np.reshape(np.eye(np.max(cm) + 1)[cm], (cm.shape[0], num_segments * (frames_per_segment + 1)))
  
    # compute endframes and sort scenarios
    end_frame_idx = np.cumsum(cm, axis=-1)[:, -1]
    sort_idxs = np.argsort(end_frame_idx)
    end_frame_idx = end_frame_idx[sort_idxs]
    cm = cm[sort_idxs]
    cm_onehot = cm_onehot[sort_idxs]
  
    # compute targets
    target_list = []
    extended_dt = tf.concat((tf.ones(shape=dt.get_shape().as_list()[:-1] + [1], dtype=dt.dtype), dt), axis=-1)
    reshaped_dt = tf.reshape(tf.transpose(extended_dt, (0, 2, 1)),
                             (num_segments * (frames_per_segment + 1), batch_size))
    for f_idx in range(n_frames):
      with tf.name_scope("comp_target_%d" % f_idx):
        # crop out scenario matrix for f_idx and transform to tensor
        end_frame_idxs_i = np.where(end_frame_idx == f_idx + 1)[0]
        cm_i = cm[end_frame_idxs_i, :]
        cm_onehot_i = cm_onehot[end_frame_idxs_i, :]
        cm_tf_i = tf.constant(cm_onehot_i, dtype=tf.float32, name="scn_matrix_%d" % f_idx)
      
        # compute data indices
        extended_cm_i = np.concatenate((cm_i, np.zeros((cm_i.shape[0], 1))), axis=1)
        segment_idx = (extended_cm_i == 0).argmax(axis=1) - 1  # last segment that has not index "don't care"
        frame_idx = cm_i[range(cm_i.shape[0]), segment_idx] - 1  # -1 to correct for don't care bit
        vec_idx = segment_idx * frames_per_segment + frame_idx
        vec_idx_onehot = np.eye(num_segments * frames_per_segment)[vec_idx]
        vec_idx_onehot_tf = tf.constant(vec_idx_onehot, dtype=dt.dtype, name="vec_idx_%d" % f_idx)
      
        # compute scenario probabilities
        with tf.name_scope("scenario_probs_%d" % f_idx):
          sparse_probs = tf.multiply(cm_tf_i[:, :, None], reshaped_dt[None, :, :])
          sparse_probs = tf.where(tf.equal(sparse_probs, 0.0), tf.ones_like(sparse_probs),
                                  sparse_probs)  # fill in sparse holes for multiplication
          probs = tf.reduce_prod(sparse_probs, axis=1)
          norm_factor = tf.reduce_sum(probs, axis=0)
          norm_probs = tf.divide(probs, norm_factor[None, :])
      
        # compute target_frame
        agg_probs = tf.reduce_sum(tf.multiply(norm_probs[:, None, :], vec_idx_onehot_tf[:, :, None]), axis=0)
        agg_probs = tf.reshape(agg_probs, [num_segments, frames_per_segment, batch_size] +
                               [1 for _ in range(n_dim_data)])  # expand to data dimension
        target = tf.reduce_sum(tf.reduce_sum(tf.multiply(agg_probs, x), axis=0), axis=0)
        target_list.append(target)

  return tf.stack(target_list, axis=0)


## This is an unfinished alternative implementation for getting low-level targets for predicted images (uses convolutions)
#
# def get_low_level_targets(
#       offset_distributions,
#       targets,
#       propagated_distributions):
#   """ Returns soft targets, which are averages of candidate targets by the chance of them being the target
#
#   This is the loss from the Temporal Hierarchy model
#   """
#
#   num_segments = offset_distributions.get_shape().as_list()[0]
#   n_frames_predict = targets.get_shape().as_list()[1]
#   #  make sure targets have time as second dimension
#
#   soft_targets = []
#   low_level_weights = []
#
#   for t in range(num_segments):
#     #  you predict further then the sequence goes. what to do with the rest of the frames?
#
#     propagated_distribution = propagated_distributions[t]
#     cut = min(propagated_distribution.get_shape().as_list()[1], n_frames_predict)
#     def flatten_image(image):
#       shape = image.get_shape().as_list()
#       return tf.reshape(image, [shape[0], shape[1], shape[2]*shape[3]*shape[4]]), shape
#     def unflatten_image(image, shape):
#       return tf.reshape(image, [shape[0], -1, shape[2],shape[3],shape[4]])
#
#     flat_targets, pre_shape = flatten_image(targets[:,:cut])
#     soft_target = unflatten_image(batchwise_1d_convolution(propagated_distribution[:,:cut],
#                                                            flat_targets),
#                                   pre_shape)
#     # The low-level weights are (p(frame is before the last gt frame)) * (p(frame is used in its segment))
#     low_level_weights_segment = tf.multiply(cum_sum(propagated_distribution[:,:n_frames_predict-1]),
#                                 (1 - [0, cum_mult(offset_distributions[t])]))
#     low_level_weights.append(low_level_weights_segment)
#     soft_targets.append(soft_target)
#
#   soft_targets = tf.stack(soft_targets)
#   return soft_targets, low_level_weights