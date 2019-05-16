"""DataHandler for different types of datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from skimage.transform import resize

# sys.path.append("..")
import data.moving_mnist.config_pb2 as config_pb2

from google.protobuf import text_format

_MNIST_DIR = '/NAS/data/moving_mnist'


def ReadDataProto(fname):
  data_pb = config_pb2.Data()
  with open(fname, 'r') as pbtxt:
    text_format.Merge(pbtxt.read(), data_pb)
  return data_pb


def ChooseDataHandler(data_pb):
  if data_pb.dataset_type == config_pb2.Data.LABELLED:
    return DataHandler(data_pb)
  elif data_pb.dataset_type == config_pb2.Data.UNLABELLED:
    return UnlabelledDataHandler(data_pb)
  elif data_pb.dataset_type == config_pb2.Data.BOUNCING_MNIST:
    return BouncingMNISTDataHandler(data_pb)
  elif data_pb.dataset_type == config_pb2.Data.VIDEO_PATCH:
    return VideoPatchDataHandler(data_pb)
  else:
    raise Exception('Unknown DatasetType.')


class DataHandler(object):
  """DataHandler for labelled datasets.

  Input could be anything from features of convolutional net to raw pixels.
  """

  def __init__(self, data_pb):
    self.data_ = h5py.File(data_pb.data_file)[data_pb.dataset_name]
    self.seq_length_ = data_pb.num_frames
    self.seq_stride_ = data_pb.stride
    self.randomize_ = data_pb.randomize
    self.batch_size_ = data_pb.batch_size
    self.image_size_x_ = data_pb.image_size_x
    self.image_size_y_ = data_pb.image_size_y
    self.patch_size_x_ = data_pb.patch_size_x
    self.patch_size_y_ = data_pb.patch_size_y
    self.sample_times_ = data_pb.sample_times
    self.num_colors_   = data_pb.num_colors

    if self.image_size_x_ == 0:
      self.image_size_x_ = 1
    if self.image_size_y_ == 0:
      self.image_size_y_ = 1
    if self.patch_size_x_ == 0:
      self.patch_size_x_ = self.image_size_x_
    if self.patch_size_y_ == 0:
      self.patch_size_y_ = self.image_size_y_
    if self.num_colors_ == 0:
      self.num_colors_ = self.data_.shape[1]

    if data_pb.mean_file != "":
      f = h5py.File(data_pb.mean_file)
      self.mean_ = f['pixel_mean'].value
      self.std_ = f['pixel_std'].value
      assert self.mean_.shape[0] == self.num_colors_
      f.close()
    else:
      self.mean_ = None
      self.std_ = None

    self.frame_size_ = self.num_colors_ * self.patch_size_y_ * self.patch_size_x_
    assert self.num_colors_ * self.image_size_y_ * self.image_size_x_ == self.data_.shape[1]

    self.x_slack_ = self.image_size_x_ - self.patch_size_x_
    self.y_slack_ = self.image_size_y_ - self.patch_size_y_

    video_boundaries, num_frames = self.GetBoundaries(data_pb.num_frames_file)
    labels = self.GetLabels(data_pb.labels_file)
    assert len(labels) == len(video_boundaries)
    video_ids = self.GetVideoIds(data_pb.video_ids_file)
    if len(video_ids) == 0:
      video_ids = range(len(labels))

    self.num_frames_ = []
    self.video_ind_ = {}
    frame_indices = []
    this_labels = []
    for v, video_id in enumerate(video_ids):
      this_labels.append(labels[video_id])
      start, end = video_boundaries[video_id]
      self.num_frames_.append(num_frames[video_id])
      end = end - self.seq_length_ + 1
      frame_indices.extend(range(start, end, self.seq_stride_))
      for i in range(start, end, self.seq_stride_):
        self.video_ind_[i] = v

    self.num_videos_ = len(video_ids)
    self.dataset_size_ = len(frame_indices)
    print('Dataset size: {}'.format(self.dataset_size_))
    self.frame_indices_ = np.array(frame_indices)
    self.labels_ = np.array(this_labels).reshape(-1, 1)

    self.Reset()
    self.batch_data_  = np.zeros((self.batch_size_, self.seq_length_ * self.frame_size_), dtype=np.float32)
    self.batch_label_ = np.zeros((self.batch_size_, 1), dtype=np.float32)

  # Get the boundaries (start index and end index) of each video
  def GetBoundaries(self, filename):
    boundaries = []
    num_frames = []
    start = 0
    for line in open(filename):
      num_f = int(line.strip())
      num_frames.append(num_f)
      end = start + num_f
      boundaries.append((start, end))
      start = end
    return boundaries, num_frames

  def GetLabels(self, filename):
    labels = []
    if filename != '':
      for line in open(filename):
        labels.append(int(line.strip()))
    return labels

  def GetVideoIds(self, filename):
    video_ids = []
    if filename != '':
      for line in open(filename):
        video_ids.append(int(line.strip()))
    return video_ids

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetDatasetSize(self):
    return self.dataset_size_

  def GetSeqLength(self):
    return self.seq_length_

  def Reset(self):
    self.frame_row_ = 0
    if self.randomize_:
      np.random.shuffle(self.frame_indices_)

  # Crop the patch from image frame
  def Crop(self, data, num_crops=1):
    d = data.reshape((data.shape[0], self.num_colors_, self.image_size_y_, self.image_size_x_))
    if self.x_slack_ > 0:
      x_offset = np.random.choice(self.x_slack_, size=num_crops)
    else:
      x_offset = np.zeros(num_crops, dtype=np.int32)
    if self.y_slack_ > 0:
      y_offset = np.random.choice(self.y_slack_, size=num_crops)
    else:
      y_offset = np.zeros(num_crops, dtype=np.int32)

    crops = np.zeros((num_crops, data.shape[0], self.num_colors_, self.patch_size_y_, self.patch_size_x_))
    seq_length = data.shape[0]
    for i in range(num_crops):
      crops[i, :, :, :, :] = d[:, :,
             y_offset[i]: y_offset[i] + self.patch_size_y_,
             x_offset[i]: x_offset[i] + self.patch_size_x_]
    if self.mean_ is not None:
      for i in range(self.num_colors_):
        crops[:, :, i, :, :] -= self.mean_[i]
        crops[:, :, i, :, :] /= self.std_[i]
    return crops.reshape((num_crops, -1))

  def GetBatch(self, verbose=False):
    batch_size = self.batch_size_
    for j in range(batch_size):
      if verbose:
        sys.stdout.write('\r%d of %d' % (j+1, batch_size))
        sys.stdout.flush()
      ind = j % self.sample_times_
      if ind == 0:
        start = self.frame_indices_[self.frame_row_]
        self.frame_row_ += 1
        if self.frame_row_ == self.dataset_size_:
          self.Reset()
        end = start + self.seq_length_
        crops = self.Crop(self.data_[start:end, :], self.sample_times_)
      self.batch_data_[j, :] = crops[ind, :].reshape(-1)
      self.batch_label_[j, :] = self.labels_[self.video_ind_[start], :]
    if verbose:
      sys.stdout.write('\n')
    return self.batch_data_, self.batch_label_

  def GetResults(self, predictions):
    assert not self.randomize_
    assert predictions.shape[0] == self.dataset_size_
    start = 0
    pooled_correct = 0
    correct = 0

    # pooled_pred are averaged results for all selected frames in the video
    for i in range(self.num_videos_):
      end = start + 1 + max(0, (self.num_frames_[i] - self.seq_length_)/self.seq_stride_)
      correct += (predictions[start:end, :].argmax(axis=1) == self.labels_[i]).sum()
      pooled_pred = predictions[start:end, :].mean(axis=0)
      pooled_correct += pooled_pred.argmax() == self.labels_[i]
      start = end
    return correct / float(self.dataset_size_), pooled_correct / float(self.num_videos_)

  def DisplayData(self,
                  data,
                  rec=None,
                  fut=None,
                  fig=1,
                  case_id=0,
                  output_file=None):
    """
    Plots the raw data along with estimates and optionally saves it disk.

    Args:
      data: The target image sequence. An array of shape
        [n_seqs, seq_len_total, 3, height, width] (for RGB input) or
        [n_seqs, seq_len_total, height, width] (for grayscale input)
      rec: The estimated reconstruction image sequence. An array of shape
        [n_seqs, seq_len_rec, 3, height, width] (for RGB input) or
        [n_seqs, seq_len_rec, height, width] (for grayscale input).
        Ignored if None. Defaults to None.
      fut: The estimated predicted image sequence. An array of shape
        [n_seqs, seq_len_fut, 3, height, width] (for RGB input) or
        [n_seqs, seq_len_fut, height, width] (for grayscale input).
        Ignored if None. Defaults to None.
      fig: The figure ID. Defaults to 1.
      case_id: The sequence ID (batch element) to visualize. Defaults to 0.
      output_file: The file path to save the plot. If None, figure will not be
        saved. Defaults to None.
    """
    name, ext = os.path.splitext(output_file)
    output_file1 = '%s_original%s' % (name, ext)
    output_file2 = '%s_recon%s' % (name, ext)

    if self.num_colors_ == 3:
      d = data[0, :].reshape(self.seq_length_, self.num_colors_, self.patch_size_y_, self.patch_size_x_)
      r = rec[0, :].reshape(-1, self.num_colors_, self.patch_size_y_, self.patch_size_x_)
      f = fut[0, :].reshape(-1, self.num_colors_, self.patch_size_y_, self.patch_size_x_)
      im1 = np.zeros((self.patch_size_y_, self.patch_size_x_, self.num_colors_), dtype=np.uint8)
      im2 = np.zeros((self.patch_size_y_, self.patch_size_x_, self.num_colors_), dtype=np.uint8)
      rec_length = r.shape[0] if rec is not None else 0
      fut_length = f.shape[0] if fut is not None else 0
      plt.figure(2*fig, figsize=(self.seq_length_, 1))
      plt.clf()
      for i in range(self.seq_length_):
        for j in range(self.num_colors_):
          im1[:, :, j] = ((d[i, j, :, :] * self.std_[j]) + self.mean_[j]).astype(np.uint8)
        plt.subplot(1, self.seq_length_, i+1)
        plt.imshow(im1, interpolation="nearest")
        plt.axis('off')
        plt.draw()

      print(output_file1)
      plt.savefig(output_file1, bbox_inches='tight')
      plt.figure(2*fig+1, figsize=(self.seq_length_, 1))
      plt.clf()
      for i in range(self.seq_length_):
        for j in range(self.num_colors_):
          r_i = rec_length - i - 1
          f_i = i - rec_length
          if r_i >= 0:
            im = (r[r_i, j, :, :] * self.std_[j]) + self.mean_[j]
            im = np.minimum(255, np.maximum(im, 0))
            im2[:, :, j] = im.astype(np.uint8)
          if f_i >= 0:
            im = (f[f_i, j, :, :] * self.std_[j]) + self.mean_[j]
            im = np.minimum(255, np.maximum(im, 0))
            im2[:, :, j] = im.astype(np.uint8)
        plt.subplot(1, self.seq_length_, i+1)
        plt.imshow(im2, interpolation="nearest")
        plt.axis('off')
        plt.draw()
      print(output_file2)
      plt.savefig(output_file2, bbox_inches='tight')
    else:
      for i in range(self.seq_length_):
        plt.subplot(1, self.seq_length_, i+1)
        for j in range(self.num_colors_):
          im[:, :, j] = d[i, j, :, :].astype(np.uint8)
        plt.imshow(im)
      plt.draw()
      if output_file is None:
        plt.pause(0.1)
      else:
        print(output_file)
        plt.savefig(output_file, bbox_inches='tight')


class UnlabelledDataHandler(object):
  """DataHandler for unlabelled datasets. Generalizes VideoPatchDataHandler."""

  def __init__(self, data_pb):
    self.seq_length_ = data_pb.num_frames
    self.seq_stride_ = data_pb.stride
    self.randomize_ = data_pb.randomize
    self.batch_size_ = data_pb.batch_size

    self.filenames_ = []
    self.num_frames_ = []
    fnames = []
    num_f = []
    for line in open(data_pb.data_file):
      fnames.append(line.strip())
    for line in open(data_pb.num_frames_file):
      num_f.append(int(line.strip()))
    assert len(num_f) == len(fnames)

    for i in range(len(num_f)):
      if num_f[i] >= self.seq_length_:
        self.num_frames_.append(num_f[i])
        self.filenames_.append(fnames[i])

    self.num_videos_ = len(self.filenames_)
    print('Num videos: {}'.format(self.num_videos_))
    data = h5py.File(self.filenames_[0])[data_pb.dataset_name]
    self.frame_size_ = data.shape[1]
    self.dataset_name_ = data_pb.dataset_name
    frame_indices = []
    self.dataset_size_ = 0
    start = 0
    self.video_ind_ = {}
    for v, f in enumerate(self.num_frames_):
      end = start + f - self.seq_length_ + 1
      frame_indices.extend(range(start, end, self.seq_stride_))
      for i in range(start, end, self.seq_stride_):
        self.video_ind_[i] = v
      start += f
    self.dataset_size_ = len(frame_indices)
    print('Dataset size'.format(self.dataset_size_))
    self.frame_indices_ = np.array(frame_indices)
    self.vid_boundary_ = np.array(self.num_frames_).cumsum()
    self.Reset()
    self.batch_data_  = np.zeros((self.batch_size_, self.seq_length_ * self.frame_size_), dtype=np.float32)

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetDatasetSize(self):
    return self.dataset_size_

  def GetSeqLength(self):
    return self.seq_length_

  def Reset(self):
    self.frame_row_ = 0
    if self.randomize_:
      np.random.shuffle(self.frame_indices_)

  def GetBatch(self, verbose=False):
    batch_size = self.batch_size_
    for j in range(batch_size):
      start = self.frame_indices_[self.frame_row_]
      vid_ind = self.video_ind_[start]
      if vid_ind > 0:
        start -= self.vid_boundary_[vid_ind - 1]
      self.frame_row_ += 1
      if self.frame_row_ == self.dataset_size_:
        self.Reset()
      end = start + self.seq_length_
      f = h5py.File(self.filenames_[vid_ind])
      self.batch_data_[j, :] = f[self.dataset_name_][start:end, :].reshape(-1)
      f.close()
    return self.batch_data_, None

class BouncingMNISTDataHandler(object):
  """DataHandler that creates Bouncing MNIST dataset on the fly."""
  def __init__(self, data_pb):
    self.seq_length_ = data_pb.num_frames
    self.batch_size_ = data_pb.batch_size
    self.image_size_ = data_pb.image_size
    self.num_digits_ = data_pb.num_digits
    self.step_length_ = data_pb.step_length
    self.random_ = data_pb.random
    self.dataset_size_ = 10000  # The dataset is really infinite. This is just for validation.
    self.digit_size_ = int(28 * self.image_size_/64)
    self.frame_size_ = self.image_size_ ** 2
    self._rand_state = None

    try:
      f = h5py.File(os.path.join(_MNIST_DIR, 'mnist.h5'))
    except:
      raise ValueError('Please set the correct path to MNIST dataset.')

    self.data_ = f['train'].value.reshape(-1, 28, 28)
    f.close()
    self.indices_ = np.arange(self.data_.shape[0])
    self.row_ = 0

    # reshape digit size
    if self.digit_size_ != 28:
      reshaped_digits = np.empty((self.data_.shape[0], self.digit_size_, self.digit_size_), dtype=self.data_.dtype)
      for digit_idx in range(self.data_.shape[0]):
        # reshaped_digits[digit_idx, :, :] = spm.imresize(self.data_[digit_idx, :, :],
        #                                                 size=(self.digit_size_, self.digit_size_),
        #                                                 interp="cubic") / 255.0
        reshaped_digits[digit_idx, :, :] = resize(self.data_[digit_idx, :, :],
                                                  output_shape=(self.digit_size_, self.digit_size_),
                                                  mode='constant')    # also does scaling 0...1
      self.data_ = reshaped_digits

  def GetBatchSize(self):
    """Return the number of sequences in a batch."""
    return self.batch_size_

  def GetDims(self):
    """Return the dimensionality of each example (seq_len * im_size)."""
    return self.frame_size_

  def GetDatasetSize(self):
    """Return the number of examples in the dataset."""
    return self.dataset_size_

  def GetImageSize(self):
    return self.image_size_

  def GetSeqLength(self):
    """Return the number of images in each sequence."""
    return self.seq_length_

  def TurnOffRandomness(self):
    """Store state of random generator and set random seed to fixed number."""
    self._rand_state = np.random.get_state()
    np.random.seed(42)
    self.indices_ = np.arange(self.data_.shape[0])
    self.row_ = 0

  def TurnOnRandomness(self):
    """Resets random generator state."""
    if self._rand_state is None:
      raise ValueError("Randomness turned on before it was turned off!")
    np.random.set_state(self._rand_state)

  def Reset(self):
    pass

  def Angle2XY(self, angle):
    return np.cos(angle), np.sin(angle)

  def GetRandomTrajectory(self, batch_size):
    """Generate a random trajectory.

    Args:
      batch_size: The number of sequences in the returned batch.
    Returns:
      seq_y: The sequence of y positions.
      seq_x: The sequence of x positions.
    """
    length = self.seq_length_
    canvas_size = self.image_size_ - self.digit_size_

    # Initial position uniform random inside the box.
    y = np.random.rand(batch_size)
    x = np.random.rand(batch_size)

    # Choose a random velocity.
    theta = np.random.rand(batch_size) * 2 * np.pi
    v_x, v_y = self.Angle2XY(theta)

    seq_y = np.zeros((length, batch_size))
    seq_x = np.zeros((length, batch_size))
    seq_theta = np.empty((length, batch_size))
    seq_theta[:] = np.nan
    for i in range(length):
      # Take a step along velocity.
      y += v_y * self.step_length_
      x += v_x * self.step_length_

      # Bounce off edges.
      for j in range(batch_size):
        if x[j] <= 0:
          x[j] = 0
          if self.random_:
            # random velocity angle between -pi/2 ... pi/2
            theta = np.random.rand() * np.pi - np.pi/2
            v_x[j], v_y[j] = self.Angle2XY(theta)
            seq_theta[i, j] = theta
          else:
            v_x[j] = -v_x[j]
        if x[j] >= 1.0:
          x[j] = 1.0
          if self.random_:
            # random velocity angle between pi/2 ... 3pi/2
            theta = np.random.rand() * np.pi + np.pi/2
            v_x[j], v_y[j] = self.Angle2XY(theta)
            seq_theta[i, j] = theta
          else:
            v_x[j] = -v_x[j]
        if y[j] <= 0:
          y[j] = 0
          if self.random_:
            # random velocity angle between 0 ... pi
            theta = np.random.rand() * np.pi
            v_x[j], v_y[j] = self.Angle2XY(theta)
            seq_theta[i, j] = theta
          else:
            v_y[j] = -v_y[j]
        if y[j] >= 1.0:
          y[j] = 1.0
          if self.random_:
            # random velocity angle between pi ... 2pi
            theta = np.random.rand() * np.pi + np.pi
            v_x[j], v_y[j] = self.Angle2XY(theta)
            seq_theta[i, j] = theta
          else:
            v_y[j] = -v_y[j]
      seq_y[i, :] = y
      seq_x[i, :] = x

    # Scale to the size of the canvas.
    seq_y = (canvas_size * seq_y).astype(np.int32)
    seq_x = (canvas_size * seq_x).astype(np.int32)
    return seq_y, seq_x, seq_theta

  def Overlap(self, a, b):
    """Put b on top of a."""
    return np.maximum(a, b)

  def GetBatch(self, verbose=False):
    """Return a batch of sequence data generated with a random trajectory.

    Args:
      verbose: Ignored.
    Returns:
      data_out: The sequence data for the current batch. An array of shape
        [batch_size, seq_len * im_height * im_width]
      label_out: None, as data is not labeled.
    """
    start_y, start_x, seq_theta = self.GetRandomTrajectory(self.batch_size_ *
                                                           self.num_digits_)
    # minibatch data
    data = np.zeros((self.batch_size_, self.seq_length_,
                     self.image_size_, self.image_size_), dtype=np.float32)

    # use theta as label
    label_out = np.empty((self.seq_length_, self.batch_size_, self.num_digits_))
    label_out[:] = np.nan
    for j in range(self.batch_size_):
      for n in range(self.num_digits_):

        # get random digit from dataset
        ind = self.indices_[self.row_]
        self.row_ += 1
        if self.row_ == self.data_.shape[0]:
          self.row_ = 0
          np.random.shuffle(self.indices_)
        digit_image = self.data_[ind, :, :]

        # generate video
        for i in range(self.seq_length_):
          top = start_y[i, j * self.num_digits_ + n]
          left = start_x[i, j * self.num_digits_ + n]
          bottom = top + self.digit_size_
          right = left + self.digit_size_
          data[j, i, top:bottom, left:right] = self.Overlap(data[j, i,
                                                                 top:bottom,
                                                                 left:right],
                                                            digit_image)
          # copy angle data
          seq_theta_cur = seq_theta[i, j * self.num_digits_ + n]
          label_out[i, j, n] = seq_theta_cur
    data_out = data.reshape(self.batch_size_, -1)
    return data_out, label_out, label_out

  def DisplayData(self, data, rec=None, fut=None, fig=0, case_id=0,
                  output_file=None):
    """Plot the input sequence along with reconstructions and predictions.

    Reconstructions and predictions are plotted together, with Reconstructions
    shown in reverse order.

    Args:
      data: The input sequence. An array of shape
        [batch_size, seq_len * im_height * im_width]
      rec: The reconstruction sequence. An array of shape
        [batch_size, rec_len * im_height * im_width].
        If None, the reconstruction sequence is ignored.
      fut: The predicted future sequence. An array of shape
        [batch_size, (seq_len - rec_len) * im_height * im_width].
        If None, the reconstruction sequence is ignored.
        the future sequence is ignored.
      fig: The figure index of the first plot.
      case_id: The index of the batch to visualize.
      output_file: The path to which sequence plots will be saved. If None,
        sequences will not be saved.
    """
    output_file1 = None
    output_file2 = None

    if output_file is not None:
      name, ext = os.path.splitext(output_file)
      output_file1 = '%s_original%s' % (name, ext)
      output_file2 = '%s_recon%s' % (name, ext)

    # get data
    data = data[case_id, :].reshape(-1, self.image_size_, self.image_size_)
    # get reconstruction and future sequences if exist
    if rec is not None:
      rec = rec[case_id, :].reshape(-1, self.image_size_, self.image_size_)
      enc_seq_length = rec.shape[0]
    if fut is not None:
      fut = fut[case_id, :].reshape(-1, self.image_size_, self.image_size_)
      if rec is None:
        enc_seq_length = self.seq_length_ - fut.shape[0]
      else:
        assert enc_seq_length == self.seq_length_ - fut.shape[0]

    num_rows = 1
    # create figure for original sequence
    plt.figure(2*fig, figsize=(20, 1))
    plt.clf()
    for i in range(self.seq_length_):
      plt.subplot(num_rows, self.seq_length_, i+1)
      plt.imshow(data[i, :, :], cmap=plt.cm.gray, interpolation="nearest")
      plt.axis('off')
    plt.draw()

    if output_file1 is not None:
      print(output_file1)
      plt.savefig(output_file1, bbox_inches='tight')

    # create figure for reconstuction and future sequences
    if rec is not None or fut is not None:
      plt.figure(2*fig+1, figsize=(20, 1))
      plt.clf()
      for i in range(self.seq_length_):
        if rec is not None and i < enc_seq_length:
          plt.subplot(num_rows, self.seq_length_, i + 1)
          plt.imshow(rec[rec.shape[0] - i - 1, :, :],
                     cmap=plt.cm.gray, interpolation="nearest")
        if fut is not None and i >= enc_seq_length:
          plt.subplot(num_rows, self.seq_length_, i + 1)
          plt.imshow(fut[i - enc_seq_length, :, :],
                     cmap=plt.cm.gray, interpolation="nearest")
        plt.axis('off')
      plt.draw()
      if output_file2 is not None:
        print(output_file2)
        plt.savefig(output_file2, bbox_inches='tight')
      else:
        plt.pause(0.1)
    plt.show()

# video patches loaded from some file
class VideoPatchDataHandler(object):
  def __init__(self, data_pb):
    self.seq_length_ = data_pb.num_frames
    self.batch_size_ = data_pb.batch_size
    self.image_size_ = data_pb.image_size
    self.data_file_ = data_pb.data_file
    self.num_frames_ = data_pb.num_frames
    self.num_colors_ = data_pb.num_colors

    self.is_color_ = False
    if self.num_colors_ == 3:
      self.is_color_ = True

    if self.is_color_:
      self.frame_size_ = (self.image_size_ ** 2) * 3
    else:
      self.frame_size_ = self.image_size_ ** 2

    try:
      self.data_ = np.float32(np.load(self.data_file_))
      self.data_ = self.data_ / 255.
    except:
      print('Please set the correct path to the dataset')
      sys.exit()

    self.dataset_size_ = self.data_.shape[0]
    self.row_ = 0

    # hack to cope with varying conditioning/prediction lengths
    # truncate data to desired size
    data_seq_len = self.data_.shape[1]
    if data_seq_len < self.seq_length_:
      print('Desired sequence length of %d is too big for loaded '
            'sequences of length %d.' % (self.seq_length_, self.data_.shape[1]))
      sys.exit(0)
    elif data_seq_len > self.seq_length_:
      self.data_ = self.data_[:, (data_seq_len-self.seq_length_):, ...]

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetDatasetSize(self):
    return self.dataset_size_

  def GetImageSize(self):
    return self.image_size_

  def GetSeqLength(self):
    return self.seq_length_

  def Reset(self):
    self.row_ = 0
    pass

  def GetBatch(self, verbose=False):
    minibatch = self.data_[self.row_:self.row_+self.batch_size_]
    self.row_ = self.row_ + self.batch_size_

    if self.row_ == self.data_.shape[0]:
      self.row_ = 0

    # output dummy action sequences
    label_out = np.random.rand(minibatch.shape[1], minibatch.shape[0])

    return minibatch.reshape(minibatch.shape[0], -1), label_out

  def DisplayData(self, data, rec=None, fut=None, fig=1, case_id=0, output_file=None):
    output_file1 = None
    output_file2 = None

    if output_file is not None:
      name, ext = os.path.splitext(output_file)
      output_file1 = '%s_original%s' % (name, ext)
      output_file2 = '%s_recon%s' % (name, ext)

    # get data
    if self.is_color_:
      data = data[case_id, :]
      data[data>1.] = 1.
      data[data<0.] = 0.
      data = data.reshape(-1, 3, self.image_size_, self.image_size_)
      data = data.transpose(0, 2, 3, 1)
    else:
      data = data[case_id, :].reshape(-1, self.image_size_, self.image_size_)

    # get reconstruction and future sequences if they exist
    if rec is not None:
      if self.is_color_:
        rec = rec[case_id, :]
        rec[rec>1.] = 1.
        rec[rec<0.] = 0.
        rec = rec.reshape(-1, 3, self.image_size_, self.image_size_)
        rec = rec.transpose(0, 2, 3, 1)
      else:
        rec = rec[case_id, :].reshape(-1, self.image_size_, self.image_size_)
      enc_seq_length = rec.shape[0]

    if fut is not None:
      if self.is_color_:
        fut = fut[case_id, :]
        fut[fut>1.] = 1.
        fut[fut<0.] = 0.
        fut = fut.reshape(-1, 3, self.image_size_, self.image_size_)
        fut = fut.transpose(0, 2, 3, 1)
      else:
        fut = fut[case_id, :].reshape(-1, self.image_size_, self.image_size_)
      if rec is None:
        enc_seq_length = self.seq_length_ - fut.shape[0]
      else:
        assert enc_seq_length == self.seq_length_ - fut.shape[0]

    num_rows = 1
    # create figure for original sequence
    plt.figure(2*fig, figsize=(self.num_frames_, 1))
    plt.clf()
    for i in range(self.seq_length_):
      plt.subplot(num_rows, self.seq_length_, i+1)
      if self.is_color_:
        plt.imshow(data[i])
      else:
        plt.imshow(data[i, :, :], cmap=plt.cm.gray, interpolation="nearest")
      plt.axis('off')
    plt.draw()
    if output_file1 is not None:
      print(output_file1)
      plt.savefig(output_file1, bbox_inches='tight')

    # create figure for reconstuction and future sequences
    plt.figure(2*fig+1, figsize=(self.num_frames_, 1))
    plt.clf()
    for i in range(self.seq_length_):
      if rec is not None and i < enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1)
        if self.is_color_:
          plt.imshow(rec[rec.shape[0] - i - 1])
        else:
          plt.imshow(rec[rec.shape[0] - i - 1, :, :], cmap=plt.cm.gray, interpolation="nearest")
      if fut is not None and i >= enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1)
        if self.is_color_:
          plt.imshow(fut[i - enc_seq_length])
        else:
          plt.imshow(fut[i - enc_seq_length, :, :], cmap=plt.cm.gray, interpolation="nearest")
      plt.axis('off')
    plt.draw()
    if output_file2 is not None:
      print(output_file2)
      plt.savefig(output_file2, bbox_inches='tight')
    else:
      plt.pause(0.1)
