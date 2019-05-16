import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


_GROUNDTRUTH_LABEL_FILE = "/home/karl/Downloads/bair_labeling/groundtruth/labels_gt.npy"
_ESTIMATE_LABEL_FILE = ["/home/karl/Downloads/bair_labeling/cdna_compActions/labels.npy",
                         "/home/karl/Downloads/bair_labeling/cdna_noCompActions/labels.npy",
                         "/home/karl/Downloads/bair_labeling/cdna_action_cond/labels.npy"]
_GRAPH_LABELS = ["withComp (ours)",
                 "noComp",
                 "actionCond"]
_MODE = "relative"      # relative, cumulative or absolute trajectory errors?


def plot_time_seqs_shaded_var(seqs, labels, x_lab, y_lab, savepath=None, x_steps=1.0):
    """Plots multiple sequences with mean and std deviation.
        seqs: list of arrays: time x batch
        x_lab, y_lab: label for x and y axis
    """
    means = [np.mean(seq, axis=1) for seq in seqs]
    stds = [np.std(seq, axis=1) for seq in seqs]

    fig, ax = plt.subplots()
    clrs = sns.color_palette("deep", len(seqs))
    with sns.axes_style("darkgrid"):
        for i in range(len(seqs)):
            epochs = list(np.arange(0, x_steps * seqs[i].shape[0], x_steps))
            ax.plot(epochs, means[i], label=labels[i], c=clrs[i])
            ax.fill_between(epochs, means[i] - stds[i], means[i] + stds[i], alpha=0.3, facecolor=clrs[i])
        ax.legend()
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()


def comp_angular_dist(x1, x2):
    diff = np.absolute(x1 - x2)
    diff[diff > 180] = 360 - diff[diff > 180]
    return diff


def compute_relative_trajectory_error(label, estimates, comp_gt_diff=True, angular_vals=False):
    """Computes the relative trajectory error between movements in a label trajectory
        and the annotated estimates.
        Input are two arrays of dimension: seq_len x num_seqs x label_dim
            label_dim is the dimensionality of the label
        comp_gt_diff: if True, computes difference also for ground_truth labels.
    """
    if comp_gt_diff:
        if angular_vals:
            d_label = comp_angular_dist(label[:-1], label[1:])
        else:
            d_label = label[:-1] - label[1:]
    else:
        d_label = label[:-1]    # interpret label as actions, not absolute positions
    if angular_vals:
        d_estimates = [comp_angular_dist(estimate[:-1], estimate[1:]) for estimate in estimates]
        action_error = [np.linalg.norm(comp_angular_dist(d_label, d_estimate), axis=-1)
                        for d_estimate in d_estimates]
    else:
        d_estimates = [estimate[:-1] - estimate[1:] for estimate in estimates]
        action_error = [np.linalg.norm(d_label-d_estimate, axis=-1) for d_estimate in d_estimates]
    return action_error


def compute_absolute_trajectory_error(label, estimates, angular_vals=False):
    """Computes the absolute trajectory error between positions in a label trajectory
        and the annotated estimates.
        Input are two arrays of dimension: seq_len x num_seqs x label_dim
            label_dim is the dimensionality of the label
    """
    if angular_vals:
        diffs = [comp_angular_dist(label, estimate) for estimate in estimates]
    else:
        diffs = [label - estimate for estimate in estimates]
    action_error = [np.linalg.norm(diff, axis=-1) for diff in diffs]
    return action_error


def compute_cumulative_trajectory_error(label, estimates):
    d_label = label[:-1] - label[1:]
    d_estimates = [estimate[:-1] - estimate[1:] for estimate in estimates]
    cum_d_label = np.cumsum(d_label, axis=1)
    cum_d_est = [np.cumsum(d_est, axis=1) for d_est in d_estimates]
    action_error = [np.linalg.norm(cum_d_label - d_estimate, axis=-1) for d_estimate in cum_d_est]
    return action_error


def load_trajectory_data(gt_label_file, est_label_files):
    gt_labels = np.transpose(np.load(gt_label_file)[:49, :, :2], (1, 0, 2))
    est_labels = [np.transpose(np.load(lf)[:49, :, :2], (1, 0, 2)) for lf in est_label_files]
    est_fails = [np.load(lf)[:49, :, 2] for lf in est_label_files]

    for el in est_labels:
        if gt_labels.shape != el.shape:
            print("Groundtruth label shape:")
            print(gt_labels.shape)
            print("Estimation label shape:")
            print(el.shape)
            raise ValueError("Groundtruth and estimation labels must have the same shape!")

    return gt_labels, est_labels, est_fails


def compute_fail_percentage(fails):
    print("Fail percentages:")
    for f_idx, f in enumerate(fails):
        f = f.flatten()
        print("%s: %f" % (_GRAPH_LABELS[f_idx], np.sum(f) * 100 / float(f.shape[0])))


def modebased_eval(gt_labels, est_labels, mode, angular_vals=False):
    if mode == "relative":
        errors = compute_relative_trajectory_error(gt_labels, est_labels, angular_vals=angular_vals)
    elif mode == "absolute":
        errors = compute_absolute_trajectory_error(gt_labels, est_labels, angular_vals)
    elif mode == "cumulative":
        errors = compute_cumulative_trajectory_error(gt_labels, est_labels)
    else:
        raise ValueError("Mode %s is not supported!" % mode)
    return errors


def eval_trajectories(gt_label_file, est_label_files, mode):
    gt_labels, est_labels, est_fails = load_trajectory_data(gt_label_file, est_label_files)

    # compute error
    errors = modebased_eval(gt_labels, est_labels, mode)
    fail_perc = compute_fail_percentage(est_fails)

    # print means + std
    means = [np.mean(error.flatten()) for error in errors]
    stds = [np.std(error.flatten()) for error in errors]
    medians = [np.median(error.flatten()) for error in errors]
    for i in range(len(means)):
        print("%s: %f, %f : %f" % (_GRAPH_LABELS[i], means[i], stds[i], medians[i]))

    # compute p-value between ours and baseline without comp.
    t_value, p_value = stats.ttest_rel(errors[0].flatten(), errors[1].flatten())
    print("P-Value: %f" % p_value)

    plt.figure()
    x = np.linspace(0, 30)
    y = x
    plt.scatter(errors[0][1:].flatten(), errors[1][1:].flatten())
    plt.scatter(errors[0][0].flatten(), errors[1][0].flatten(), c='r')
    plt.plot(x, y)
    plt.show()

    # plot error mean and variance over test sequences vs timestep
    plot_time_seqs_shaded_var(errors, _GRAPH_LABELS, "Timestep", mode.title() + " Trajectory Error")


if __name__ == "__main__":
    # x = [np.arange(0, 15, 0.1) for _ in range(20)]
    # x = np.stack(x, axis=1)
    # dat = np.sin(x) + np.random.rand(x.shape[0], x.shape[1])*0.5
    # dat2 = np.cos(x)+ np.random.rand(x.shape[0], x.shape[1])*0.5
    # dat3 = 1/np.exp(x)+ np.random.rand(x.shape[0], x.shape[1])*0.8
    #
    # plot_time_seqs_shaded_var([dat, dat2, dat3], ["sin", "cos", "exp"], "time", "amplitude", x_steps=0.1)

    eval_trajectories(_GROUNDTRUTH_LABEL_FILE, _ESTIMATE_LABEL_FILE, mode=_MODE)

