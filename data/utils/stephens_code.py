import os
import sys
import numpy as np
import tensorflow as tf
import cv2

import tqdm
import imageio
import glob

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(values):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

os.chdir('/NAS/data/lip_reading_sentences/lip_reading_sentences/trainval')
output_dir = '/NAS/data/group_encoder/tmplipread'
fnames = glob.glob('*/*mp4')
im_lists = []
randinds = np.random.permutation(len(fnames))[:100]

with tf.python_io.TFRecordWriter(os.path.join(output_dir, 'train.tfrecords')) as writer:
  for idx in tqdm.tqdm(randinds):
    fname = fnames[idx]
    vid = imageio.get_reader(fname)
    if len(vid) == 0:
      continue
    vidlen = len(vid)
    frames = np.zeros((vidlen, 224, 224, 3), dtype=np.uint8)
    try:
      for i in range(vidlen):
        frames[i,...] = vid.get_data(i)
    except RuntimeError:
      print("Video {} ({}) could not load".format(fname, idx))
    pass
    features = {}
    features['num_frames']  = _int64_feature(frames.shape[0])
    features['height']      = _int64_feature(frames.shape[1])
    features['width']       = _int64_feature(frames.shape[2])
    features['channels']    = _int64_feature(frames.shape[3])

    # Compress the frames using JPG and store in as a list of strings in 'frames'
    encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame[:,:,[2,1,0]])[1].tobytes())
                      for frame in frames]
    features['frames'] = _bytes_list_feature(encoded_frames)

    tfrecord_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tfrecord_example.SerializeToString())


SEQ_NUM_FRAMES = 2
MAX_RAND = 5
NUM_EPOCHS = 1
BATCH_SIZE = 5

def decode(serialized_example, sess):
  # Prepare feature list; read encoded JPG images as bytes
  features = dict()
  features["frames"] = tf.VarLenFeature(tf.string)
  features["num_frames"] = tf.FixedLenFeature((), tf.int64)

  # Parse into tensors
  parsed_features = tf.parse_single_example(serialized_example, features)

  # Randomly sample offset from the valid range.
  random_offset = tf.random_uniform(
      shape=(), minval=0,
      maxval=parsed_features["num_frames"] - MAX_RAND, dtype=tf.int64)
  #     maxval=parsed_features["num_frames"] - SEQ_NUM_FRAMES, dtype=tf.int64)
  random_dist = tf.random_uniform(
      shape=(), minval=0, maxval=MAX_RAND, dtype=tf.int64)

  offsets = tf.convert_to_tensor([ random_offset, random_offset + random_dist ])
  print('offsets')
  print(offsets)
  # offsets = (tf.range(random_offset, random_offset + SEQ_NUM_FRAMES))

  # Decode the encoded JPG images
  def image_decode(i):
    parsed_i = parsed_features["frames"].values[i]
    decoded = tf.image.decode_jpeg(parsed_i, channels=3)
    return decoded

  images = tf.map_fn(image_decode, offsets, dtype=tf.uint8)
  images_resize = tf.reshape(images, (2, 224, 224, 3))
  ret = images_resize
  return ret


tfrecord_dir = '/NAS/data/group_encoder/tmplipread'
tfrecord_file = os.path.join(tfrecord_dir, 'train.tfrecords')

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())


dataset = tf.data.TFRecordDataset([ tfrecord_file ])
dataset = dataset.repeat(NUM_EPOCHS)
dataset = dataset.map(lambda x: decode(x, sess))
# dataset = dataset.map(preprocess_video)
# The parameter is the queue size
dataset = dataset.shuffle(1000 + 3 * BATCH_SIZE)
dataset = dataset.batch(BATCH_SIZE)

iterator = dataset.make_one_shot_iterator()
next_batch = iterator.get_next()

sess.run(init_op)

# import matplotlib.pyplot as plt
import scipy.misc as misc
batch_idx = 0
while True:
  # Fetch a new batch from the dataset
  print("next_batch")
  batch_videos = sess.run(next_batch)

  for sample_idx in range(BATCH_SIZE):
    for frame_idx in range(SEQ_NUM_FRAMES):
      print('sample_idx: {}, frame_idx: {}'.format(sample_idx, frame_idx))
      imname = 'img_{:03d}_{:03d}_{:03d}.png'
      misc.imsave(imname.format(batch_idx, sample_idx, frame_idx),
                  batch_videos[sample_idx,frame_idx])
  print("End of batch")
  batch_idx += 1
