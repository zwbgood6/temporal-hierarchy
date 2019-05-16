import numpy as np
import tensorflow as tf


N_LOSS_FRAMES = 10


def comp_soft_gt_targets(dt, x, n_frames):
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


dt = np.asarray([[[0.1, 0.1, 0.2, 0.3, 0.1, 0.2],
                  [0.3, 0.1, 0.1, 0.3, 0.1, 0.1],
                  [0.1, 0.4, 0.2, 0.1, 0.1, 0.1]],
                 [[0.1, 0.1, 0.2, 0.3, 0.1, 0.2],
                  [0.3, 0.1, 0.1, 0.3, 0.1, 0.1],
                  [0.1, 0.4, 0.2, 0.1, 0.1, 0.1]]]
                )
dt = np.transpose(dt, [1, 0, 2])

imgs = np.ones((3, 6, 2, 1, 64, 64), dtype=np.float32)

dt_ph = tf.placeholder(dtype=tf.float32, shape=(3, 2, 6))
imgs_ph = tf.placeholder(dtype=tf.float32, shape=(3, 6, 2, 1, 64, 64))

targets = comp_soft_gt_targets(dt_ph, imgs_ph, N_LOSS_FRAMES)

with tf.Session() as sess:
    output = sess.run(targets, feed_dict={dt_ph: dt, imgs_ph: imgs})
print(output)

