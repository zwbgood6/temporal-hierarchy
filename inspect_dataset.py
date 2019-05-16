import tensorflow as tf
import os
import glob
import sys
from utils import AttrDict
from specs import machine_dependent_specs as m_specs
import numpy as np

tf.enable_eager_execution()

KTH = False

if KTH:
  # data_dir = "/NAS/data/KTH_action/tfrecords/KTH_unchunked_large/val"
  data_dir = "/Users/Oleh/Code/PredictionProject/temporal_hierarchy/temp/tfrec_video/" + sys.argv[1] + "/train"
else:
  data_dir = m_specs._TOP_DIR + "/train"
  pass
filenames = glob.glob(os.path.join(data_dir, "*.tfrecord"))
raw_dataset = tf.data.TFRecordDataset(filenames)
rec = raw_dataset.take(40).__iter__().next()
if KTH:
  features = {
            "video": tf.VarLenFeature(dtype=tf.string),
            "video_length": tf.FixedLenFeature(dtype=tf.int64, shape=[],
                                               default_value=10),
            "video_height": tf.FixedLenFeature(dtype=tf.int64, shape=[],
                                               default_value=128),
            "video_width": tf.FixedLenFeature(dtype=tf.int64, shape=[],
                                              default_value=128),
            "video_channels": tf.FixedLenFeature(dtype=tf.int64, shape=[],
                                                 default_value=1),
          }
  
  parsed_features = tf.parse_single_example(rec, features)
  
  def decode_raw_tensor(name, shape, type):
    tensor = tf.sparse_tensor_to_dense(parsed_features[name], default_value="")
    if type == 'uint8':
      tensor = tf.cast(tf.decode_raw(tensor, tf.uint8), tf.float32)
    
    elif type == 'float32':
      tensor = tf.decode_raw(tensor, tf.float32)
    else:
      raise ValueError("Only uint8 and float32 are supported tfRecord tensor types.")
    tensor = tf.reshape(tensor, shape)
    return tensor
  
  video = decode_raw_tensor("video",
                            shape=tf.stack([parsed_features["video_length"],
                                            parsed_features["video_height"],
                                            parsed_features["video_width"],
                                            parsed_features["video_channels"]]),
                            type='uint8')
else:
  dataset_config = AttrDict(input_res=64, channels=3, num_actions=4, max_seq_length=40, im_width=64, im_heigh=64)
  def parse_bair_styled_dataset(example_proto, dataset_config,
                                feature_templates_and_shapes):
    """Parses the BAIR dataset, fuses individual frames to tensors."""
    features = {}  # fill all features in feature dict
    for key, feat_params in feature_templates_and_shapes.items():
      for frame in range(dataset_config.max_seq_length):
        if feat_params["type"] == tf.string:
          feat = tf.VarLenFeature(dtype=tf.string)
        else:
          feat = tf.FixedLenFeature(dtype=feat_params["type"],
                                    shape=feat_params["shape"])
        features.update({feat_params["name"].format(frame): feat})
    parsed_features = tf.parse_single_example(example_proto, features)
  
    # decode frames and stack in video
  
    def process_feature(feat_params, frame):
      feat_tensor = parsed_features[feat_params["name"].format(frame)]
      if feat_params["type"] == tf.string:
        feat_tensor = tf.sparse_tensor_to_dense(feat_tensor, default_value="")
        # feat_tensor = tf.decode_raw(feat_tensor, tf.float32)
        feat_tensor = tf.cast(tf.decode_raw(feat_tensor, tf.uint8), tf.float32)
        # Rescale tensor from [0, 255] to [0, 1]
        feat_tensor = feat_tensor / 255
        feat_tensor = tf.reshape(feat_tensor, feat_params["shape"])
        if dataset_config.input_res != dataset_config.im_width:
          feat_tensor = tf.image.resize_images(feat_tensor, (dataset_config.im_height,
                                                             dataset_config.im_width))
      return feat_tensor
  
    parsed_seqs = {}
    for key, feat_params in feature_templates_and_shapes.items():
      if feat_params["dim"] == 0:
        feat_tensor = process_feature(feat_params, 0)
        parsed_seqs.update({key: feat_tensor})
      else:
        frames = []
        for frame in range(dataset_config.max_seq_length):
          feat_tensor = process_feature(feat_params, frame)
          frames.append(feat_tensor)
        parsed_seqs.update({key: tf.stack(frames)})
  
    return parsed_seqs


  def parse_top(example_proto, dataset_config):
    # specify feature templates that frame numbers will be filled into
    img_shape = (dataset_config.input_res, dataset_config.input_res, dataset_config.channels,)
    feature_templates_and_shapes = {
      "goal_timestep": {"name": "goal_timestep", "shape": (1,), "type": tf.int64, "dim": 0},
      "images": {"name": "{}/image_view0/encoded", "shape": img_shape, "type": tf.string, "dim": 1},
      "actions": {"name": "{}/action", "shape": (dataset_config.num_actions,), "type": tf.float32, "dim": 1},
      "abs_actions": {"name": "{}/is_key_frame", "shape": (1,), "type": tf.int64, "dim": 1},
      "is_key_frame": {"name": "{}/is_key_frame", "shape": (1,), "type": tf.int64, "dim": 1}
    }
  
    parsed_seqs = parse_bair_styled_dataset(example_proto, dataset_config,
                                            feature_templates_and_shapes)
    data_tensors = AttrDict({"goal_timestep": parsed_seqs["goal_timestep"]})
    return parsed_seqs["images"], parsed_seqs["actions"], \
           parsed_seqs["abs_actions"], parsed_seqs["is_key_frame"], data_tensors

  concat_ac = []
  for rec in raw_dataset.take(10).__iter__():
      results = parse_top(rec, dataset_config)
      concat_ac.append(results[1])
  concat_ac = np.concatenate(concat_ac, axis=0)
  print(np.var(concat_ac, axis=0))
  
  
  
#
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# imgplot = plt.imshow(video[100])
# plt.show()
#
# import skvideo.io
# import numpy as np
#
# skvideo.io.vwrite("outputvideo.mp4", video)

import pdb; pdb.set_trace()

