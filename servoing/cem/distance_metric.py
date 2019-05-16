"""Computes CEM metrics with various approaches."""

import numpy as np

from servoing.cem.dist_metrics.vgg_dist_metrics import vgg_cosine_distance_np
from servoing.cem.dist_metrics.color_filter_dist_metrics import color_filter_distance_np, get_object_position


def get_distance_metric(dist_metric_type, dense_cost, final_step_weight):
    if dist_metric_type == "vgg":
        return VGGDistance(dense_cost, final_step_weight)
    elif dist_metric_type == "euclidean":
        return EuclideanDistance(dense_cost, final_step_weight)
    elif dist_metric_type == "flow":
        return FlowDistance(dense_cost, final_step_weight)
    else:
        return ValueError('Distance metric {} not supported.'.format(dist_metric_type))


class DistanceMetric(object):
    def __init__(self, dense_cost, final_step_weight):
        self._dense_cost = dense_cost
        self._final_step_weight = final_step_weight

    def __call__(self, cem_outputs, goal_img):
        raise NotImplementedError

    def run(self, img, goal):
        raise NotImplementedError


class EuclideanDistance(DistanceMetric):
    def __init__(self, dense_cost, final_step_weight):
        super(EuclideanDistance, self).__init__(dense_cost, final_step_weight)

    def __call__(self, cem_outputs, goal_img):
        assert cem_outputs.pred_frames.dtype == np.uint8, "Euclid distance input frames need to be uint8!"
        pred_imgs = cem_outputs.pred_frames.astype(np.float32) / 255
        goal_img = goal_img.astype(np.float32) / 255
        distances = np.empty(pred_imgs.shape[0])
        for i in range(pred_imgs.shape[0]):
            if self._dense_cost:
                distances_per_step = np.empty((pred_imgs.shape[1],))
                for j in range(pred_imgs.shape[1]):
                    distances_per_step[j] = color_filter_distance_np(pred_imgs[i, j], goal_img)
                distances_per_step[-1] *= self._final_step_weight      # weight last time step higher
                distances[i] = np.sum(distances_per_step)
            else:
                distances[i] = color_filter_distance_np(pred_imgs[i, -1], goal_img)
        return distances

    def run(self, img, goal):
        assert img.dtype == np.uint8, "Euclid distance input frames need to be uint8!"
        return color_filter_distance_np(img.astype(np.float32)/255, goal.astype(np.float32)/255)


class VGGDistance(DistanceMetric):
    def __init__(self, dense_cost, final_step_weight):
        super(VGGDistance, self).__init__(dense_cost, final_step_weight)

    def __call__(self, cem_outputs, goal_img):
        assert cem_outputs.pred_frames.dtype == np.uint8, "VGG distance input frames need to be uint8!"
        pred_imgs = cem_outputs.pred_frames[:, -1].astype(np.float32) / 255
        goal_img = goal_img.astype(np.float32) / 255
        distances = np.empty(pred_imgs.shape[0])
        for i in range(pred_imgs.shape[0]):
            if self._dense_cost:
                distances_per_step = np.empty((pred_imgs.shape[1],))
                for j in range(pred_imgs.shape[1]):
                    distances_per_step[j] = vgg_cosine_distance_np(pred_imgs[i, j], goal_img)
                distances_per_step[-1] *= self._final_step_weight      # weight last time step higher
                distances[i] = np.sum(distances_per_step)
            else:
                distances[i] = vgg_cosine_distance_np(pred_imgs[i, -1], goal_img)
        return distances

    def run(self, img, goal):
        assert img.dtype == np.uint8, "VGG distance input frames need to be uint8!"
        return vgg_cosine_distance_np(img.astype(np.float32)/255, goal.astype(np.float32)/255)


class FlowDistance(DistanceMetric):
    def __init__(self, dense_cost, final_step_weight):
        super(FlowDistance, self).__init__(dense_cost, final_step_weight)

    def __call__(self, cem_outputs, goal_img):
        assert len(cem_outputs.pred_prob_frames.shape) == 5, "Expect prob frames array to be [batch, time, x, y, 1]"
        assert cem_outputs.pred_prob_frames.dtype == np.float32 and np.max(cem_outputs.pred_prob_frames) <= 1.0, \
            "Probability frames should be float and in range [0...1]"
        prob_imgs = cem_outputs.pred_prob_frames[..., 0]
        goal_pos = get_object_position(goal_img)
        if goal_pos is None:
            raise ValueError("Could not find object in provided goal image!")

        # compute pixel-wise distance to goal
        res = goal_img.shape[1]
        px_vals = np.linspace(0, res - 1, res)
        pixel_pos_img = np.stack(np.meshgrid(px_vals, px_vals)[::-1], axis=0)
        diffs = np.linalg.norm(pixel_pos_img - goal_pos[:, None, None], axis=0)

        # compute prob weighted sum of diffs
        weighted_diffs = np.multiply(prob_imgs, diffs[None, None, :])
        costs = np.sum(np.sum(weighted_diffs, axis=3), axis=2)
        if self._dense_cost:
            costs[:, -1] *= self._final_step_weight
            return np.sum(costs, axis=-1)
        else:
            return costs[:, -1]


if __name__ == "__main__":
    from utils import AttrDict
    import matplotlib.pyplot as plt
    cem_outputs = AttrDict(prob_frames=np.random.rand(20, 10, 64, 64, 1).astype(np.float32))
    goal_img = plt.imread("/Users/karl/Downloads/seq3.png")[:64, :64, :3]

    dist = FlowDistance(True, 10.0)
    costs = dist(cem_outputs, goal_img)
    print(costs)
