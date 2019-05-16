"""General utilies. """
import tensorflow as tf
from tensorflow.contrib.distributions import Normal
import math
import numpy as np

FLAGS = tf.flags.FLAGS

def maybe_stop_gradients(input_tensor, stop_criterion):
  """Wraps a tensor with tf.stop_gradient if a criterion is matched.

  Args:
    input_tensor: The raw, unwrapped tensor.
    stop_criterion: If True, the tensor is wrapped with tf.stop_gradient. If
      False, the original tensor is returned.
  Returns:
    output_tensor: See stop_criterion parameter above.
  """
  if stop_criterion:
    output_tensor = tf.stop_gradient(input_tensor)
  else:
    output_tensor = input_tensor

  return output_tensor

def make_dists_and_sample(latent_sample_seq):
  # latent_sample_seq constists of means and log_stds
  latent_dim = int(latent_sample_seq.get_shape().as_list()[-1]/2)
  latent_dists = Normal(loc=latent_sample_seq[...,:latent_dim],
                        scale=tf.exp(latent_sample_seq[...,latent_dim:]))
  latent_sample_seq = tf.squeeze(latent_dists.sample([1]))  # sample one sample from each distribution
  return latent_dists, latent_sample_seq

def pick_and_tile(tensor_list, idx, len):
  # only works for tensors of len(shape) 5
  if idx == -1:
    picked = [layer[idx:, ...] for layer in tensor_list]
  else:
    picked = [layer[idx:(idx + 1), ...] for layer in tensor_list]
  tiled = [tf.tile(layer, [len, 1, 1, 1, 1]) for layer in picked]
  return tiled

def batchwise_1d_convolution(
      input,
      filter,
      proper_convolution=False,
      padding="SAME"):
  """Uses tf.nn.depthwise_conv2d to compute 1d convolution.
  The filters used is different for every batch

  :param input: tensor of shape batch x spatial
  :param filter: tensor of shape batch x spatial
  :param proper_convolution: if false, computes cross-correlation
  :return:
  """
  batch_size = input.get_shape().as_list()[0]
  if padding=="MAX":
    # pad the sequence so that no information gets discarded
    filter_size = filter.get_shape().as_list()[1] - 1
    pre_size = math.ceil(float(filter_size) / 2)
    post_size = math.floor(float(filter_size) / 2)
    input = tf.concat([tf.zeros((batch_size, pre_size)), input, tf.zeros((batch_size, post_size))], axis=1)

  # We will use the in_channels (4th dimension) of depthwise_conv2d for the batch
  input = tf.expand_dims(tf.expand_dims(tf.transpose(input, [1, 0]), axis=0), axis=0)
  filter = tf.expand_dims(tf.expand_dims(tf.transpose(filter, [1, 0]), axis=0), axis=-1)
  # Convolution not cross-correlation
  if proper_convolution:
    filter = tf.reverse(filter, axis=[1])

  result = tf.nn.depthwise_conv2d(input, filter, strides=[1, 1, 1, 1], padding="SAME")[0][0]
  return tf.transpose(result, [1, 0])

def safe_entropy(dists, axis=None, eps=1e-12):
  """
  Computes entropy even if some entries are 0.
  
  :param dists:
  :param axis:
  :param eps:
  :return:
  """
  return - tf.reduce_sum(dists * safe_prob_log(dists, eps), axis=axis)

def safe_prob_log(tensor, eps=1e-12):
  """
  Safe log of probability values (must be between 0 and 1)
  
  :param tensor:
  :param eps:
  :return:
  """
  return tf.log(tf.clip_by_value(tensor, eps, 1-eps))
  
color2num = dict(
  gray=30,
  red=31,
  green=32,
  yellow=33,
  blue=34,
  magenta=35,
  cyan=36,
  white=37,
  crimson=38
)

def colorize(string, color='cyan', bold=True, highlight=False):
  attr = []
  num = color2num[color]
  if highlight: num += 10
  attr.append(str(num))
  if bold: attr.append('1')
  return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def debug(module, category, name, shape):
  if FLAGS.debug:
    print(colorize("[{} - {}] {} {}".format(module, category, name, shape)))
  
  
def normalize(tensor, axis=-1, eps=1e-6):
  return tensor / (tf.linalg.norm(tensor, axis=axis, keepdims=True) + eps)

def shape(tensor):
  return tensor.get_shape().as_list()


def safe_bce_loss(estimate, target, weights=1, from_logits=True, reduction=tf.losses.Reduction.NONE):
  _EPS = 1e-7
  if not from_logits:
    # TODO (oleh) fix this by computing the loss by hand
    # transform back to logits
    logits = tf.clip_by_value(estimate, _EPS, 1 - _EPS)
    logits = tf.log(logits / (1 - logits))
  else:
    logits = estimate
    
  raw_bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
  if reduction == "custom":
    # TODO weights should be specified for all axes, not excluding axis=1
    weighted_bce_loss = tf.multiply(tf.reduce_mean(raw_bce_loss, axis=1), weights)
    bce_loss = tf.reduce_mean(weighted_bce_loss)
  elif reduction == tf.losses.Reduction.NONE:
    bce_loss = raw_bce_loss
  else:
    raise ValueError("reduction not implemented")
  
  return bce_loss


def tf_swapaxes(tensor, axis1, axis2):
  """
  Analogue of np.swapaxes
  
  :param tensor:
  :param axis1:
  :param axis2:
  :return:
  """
  if axis1 == axis2:
    return tensor
  
  dim = len(tensor.get_shape().as_list())
  new_order = list(range(dim))
  new_order[axis1] = axis2
  new_order[axis2] = axis1
  
  return tf.transpose(tensor, new_order)

def add_n_dims(tensor, n):
  """ Adds n new dimensions of size 1 to the end of the tesnor
  
  :param tensor:
  :param n:
  :return:
  """
  if n == 0:
    return tensor
  
  shape = tensor.get_shape().as_list()
  return tf.reshape(tensor, shape + [1] * n)


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def make_attr_dict(**kwargs):
  return AttrDict(kwargs)
  
def flatten_end(tensor):
  """Flattens all dimensions of the tensor except the first"""
  sh = shape(tensor)
  rest = np.cumprod(sh[1:])[-1]
  return tf.reshape(tensor, [sh[0], rest])

def batchwise_gather(tensor, idxs, batch_dim=None):
  """For every element in the batch, gather corresponding element from the first axis
  
  :param tensor: shape = gather_dim x batchsize x ...
  :param idxs: tensor, shape = batchsize
  :param batch_dim: must be 1 or 2
  :return:
  """
  batch_size = shape(tensor)[batch_dim]
  
  batch_idxs = np.asarray(range(batch_size))
  if batch_dim == 0:
    gather_idxs = tf.stack([batch_idxs, idxs], axis=-1)
  elif batch_dim == 1:
    gather_idxs = tf.stack([idxs, batch_idxs], axis=-1)
  return tf.gather_nd(tensor, gather_idxs)
  
def get_fixed_prior(size, dist_dtype=tf.float32):
  """Returns a vector of means and log stds of specified size"""
  return tf.zeros(size, dtype=dist_dtype)
  
  
def map_dict(fn, d):
  return dict(map(lambda kv: (kv[0], fn(kv[1])), d.items()))


class Gaussian:
  """ Represents a gaussian distribution """

  def __init__(self, mu, log_sigma):
    self.mu = mu
    self.log_sigma = log_sigma
    self._sigma = None

  def sample(self):
    return self.mu + self.sigma * tf.random.normal(self.sigma.get_shape())

  def kl_divergence(self, other):
    """Here self=q and other=p and we compute KL(q, p)"""
    return (other.log_sigma - self.log_sigma) + (self.sigma ** 2 + (self.mu - other.mu) ** 2) \
           / (2 * other.sigma ** 2) - 0.5

  @property
  def sigma(self):
    if self._sigma is None:
      self._sigma = tf.exp(self.log_sigma)
    return self._sigma


  