import numpy as np
import h5py


def compute_f1_score(gt_keyframes, pred_keyframes):
    true_positives = np.where(np.logical_and(gt_keyframes, pred_keyframes))[0].shape[0]
    true_negatives = np.where(np.logical_and(np.logical_not(gt_keyframes), (np.logical_not(pred_keyframes))))[0].shape[0]

    false_positives = np.where(np.logical_and(np.logical_not(gt_keyframes), pred_keyframes))[0].shape[0]
    false_negatives = np.where(np.logical_and(gt_keyframes, (np.logical_not(pred_keyframes))))[0].shape[0]

    precision = true_positives / float(true_positives + false_positives)
    recall = true_positives / float(true_positives + false_negatives)

    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return f1



F1_HORIZON = 30
NUM_COND_FRAMES = 5
if __name__ == '__main__':
    filename = '/Users/karl/eval_results_bb.h5'
    f = h5py.File(filename, 'r')

    outputs = {k: f[k] for k in f.keys()}

    # combine pred_keyframes
    dt = outputs["high_level_rnn_output_dt"][:]
    max_idxs = np.argmax(dt, axis=-1)

    # random baseline
    np.random.seed(42)
    # max_idxs = np.random.choice(np.array([6, 7, 8]), size=max_idxs.shape)

    # constant baseline
    # max_idxs = 7 * np.ones(max_idxs.shape, dtype=int)

    # static baseline
    pattern = np.expand_dims(np.array([3, 5, 4, 4, 3, 7]), axis=1)      # push: [3, 4, 4, 4, 4, 5]
    max_idxs = np.repeat(pattern, repeats=max_idxs.shape[1], axis=1)

    cumsum = np.cumsum(max_idxs + 1, axis=0) - 1        # +1-1 so that zero index counts
    one_hot_matrix = np.eye(dt.shape[0] * dt.shape[-1])
    one_hot_vals = one_hot_matrix[cumsum]
    pred_kf_idxs = np.sum(one_hot_vals, axis=0)



    gt_kf_idxs = outputs["gt_keyframe_idxs"][NUM_COND_FRAMES:, :, 0].transpose()
    # gt = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0], dtype=bool)
    # pred = np.logical_not(np.array([0, 1, 0, 0, 1, 1, 0, 0, 0], dtype=bool))

    f1_scores = np.empty((gt_kf_idxs.shape[0]), dtype=np.float32)
    for i in range(gt_kf_idxs.shape[0]):
        f1_scores[i] = compute_f1_score(gt_kf_idxs[i, :F1_HORIZON], pred_kf_idxs[i, :F1_HORIZON])

    print("%f +- %f" % (np.mean(f1_scores), np.var(f1_scores)))