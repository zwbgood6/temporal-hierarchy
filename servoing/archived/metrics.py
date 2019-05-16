# from: https://github.com/alexlee-gk/video_prediction

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

import vgg_network
from tf_utils import PersistentOpEvaluator


def _axis(keep_axis, ndims):
    if keep_axis is None:
        axis = None
    else:
        axis = list(range(ndims))
        try:
            for keep_axis_ in keep_axis:
                axis.remove(keep_axis_)
        except TypeError:
            axis.remove(keep_axis)
        axis = tuple(axis)
    return axis


def normalize_tensor_np(tensor, eps=1e-10):
    norm_factor = np.linalg.norm(tensor, axis=-1, keepdims=True)
    return tensor / (norm_factor + eps)


def cosine_similarity_np(tensor0, tensor1, keep_axis=None):
    tensor0 = normalize_tensor_np(tensor0)
    tensor1 = normalize_tensor_np(tensor1)
    csim = np.sum(tensor0 * tensor1, axis=-1)
    return np.mean(csim, axis=_axis(keep_axis, csim.ndim))


def _with_flat_batch(flat_batch_fn):
    def fn(x, *args, **kwargs):
        shape = tf.shape(x)
        flat_batch_shape = tf.concat([[-1], shape[-3:]], axis=0)
        flat_batch_shape.set_shape([4])
        flat_batch_x = tf.reshape(x, flat_batch_shape)
        flat_batch_r = flat_batch_fn(flat_batch_x, *args, **kwargs)
        r = nest.map_structure(lambda x: tf.reshape(x, tf.concat([shape[:-3], tf.shape(x)[1:]], axis=0)),
                               flat_batch_r)
        return r
    return fn


class _VGGFeaturesExtractor(PersistentOpEvaluator):
    def __init__(self):
        super(_VGGFeaturesExtractor, self).__init__()
        self._image_placeholder = None
        self._feature_op = None
        self._assign_from_values_fn = None
        self._assigned = False

    def initialize_graph(self):
        self._image_placeholder = tf.placeholder(dtype=tf.float32)
        with tf.variable_scope('vgg', reuse=tf.AUTO_REUSE):
            _, self._feature_op = _with_flat_batch(vgg_network.vgg16)(self._image_placeholder)
        self._assign_from_values_fn = vgg_network.vgg_assign_from_values_fn(var_name_prefix='vgg/')

    def run(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("'image' must be a numpy array: %r" % image)
        if image.dtype not in (np.float32, np.float64):
            raise ValueError("'image' dtype must be float32 or float64, but is %r" % image.dtype)
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        sess = tf.get_default_session()
        if not self._assigned:
            self._assign_from_values_fn(sess)
        result = sess.run(self._feature_op, feed_dict={self._image_placeholder: image})
        return result


extract_vgg_features = _VGGFeaturesExtractor()


def get_vgg_features(input_img):
    return extract_vgg_features(input_img)


def vgg_cosine_similarity_np(image0, image1, keep_axis=None):
    features0 = extract_vgg_features(image0)
    features1 = extract_vgg_features(image1)
    csim = 0.0
    for feature0, feature1 in zip(features0, features1):
        csim += cosine_similarity_np(feature0, feature1, keep_axis=keep_axis)
    csim /= len(features0)
    return csim


def vgg_cosine_similarity_with_target_feats(image0, target_feats, keep_axis=None):
    features0 = extract_vgg_features(image0)
    features1 = target_feats
    csim = 0.0
    for feature0, feature1 in zip(features0, features1):
        csim += cosine_similarity_np(feature0, feature1, keep_axis=keep_axis)
    csim /= len(features0)
    return csim


def vgg_cosine_distance_np(image0, image1, keep_axis=None):
    return 1.0 - vgg_cosine_similarity_np(image0, image1, keep_axis=keep_axis)

def vgg_cosine_distance_with_target_feats(image0, target_feats, keep_axis=None):
    return 1.0 - vgg_cosine_similarity_with_target_feats(image0, target_feats, keep_axis=keep_axis)


