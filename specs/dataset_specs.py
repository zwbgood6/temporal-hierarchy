"""Dataset specifications."""
import collections
import numpy as np
import os
import tensorflow as tf
from recordclass import recordclass

try: from specs import machine_dependent_specs as m_specs
except: pass

FLAGS = tf.flags.FLAGS

## MovingMNIST
_LABELLED = 0
_UNLABELLED = 1
_BOUNCING_MNIST = 2
_VIDEO_PATCH = 4
MovingMNISTConfig = collections.namedtuple(
    "MovingMNISTConfig",
    "dataset_type data_file num_frames batch_size channels "
    "image_size num_digits step_length num_colors input_seq_len pred_seq_len "
    "random num_actions angles")

BouncingBallsConfig = collections.namedtuple(
    "BouncingBallsConfig",
    "resolution num_balls ball_size ball_speed agent_speed stochastic_angle random_angle stochastic_speed "
    "variance_degrees bounce_vertically batch_size input_seq_len pred_seq_len num_frames channels "
    "num_actions stochastic_bounce rand_start_counter image_input segment_length_limits")

## KTH
_KTH_FILE_SIZES = [
    10788, 11384, 12348, 9813, 9859, 9551, 11441, 10936,
    10051, 11175, 10390, 12140, 11445, 12586, 11040, 11254,
    11518, 12677, 10948, 10519, 12664, 11463, 10525, 11047, 10174
]
KTHConfig = collections.namedtuple(
    "KTHConfig",
    "data_dir person_ids action_classes video_types "
    "num_frames channels "
    "im_height im_width "  # Output sizes
    "input_seq_len pred_seq_len n_examples flip_lr "
    "input_height input_width rescale_size")  # Raw image sizes

H36Config = collections.namedtuple(
    "H36Config",
    "data_dir "
    "num_frames channels "
    "im_height im_width "  # Output sizes
    "input_seq_len pred_seq_len n_examples flip_lr "
    "input_height input_width rescale_size")  # Raw image sizes

# TODO(drewjaegle): add actions here?
UCFConfig = collections.namedtuple(
    "UCFConfig",
    "data_dir num_frames "
    "channels im_height im_width "
    "input_seq_len pred_seq_len chunked_examples n_examples "
    "flip_lr input_height input_width")

ReacherConfig = collections.namedtuple(
    "ReacherConfig",
    "data_dir num_frames im_height im_width batch_size "
    "policy channels num_actions max_degree action_coding "
    "input_seq_len pred_seq_len n_examples num_joints angles"
)

BAIRConfig = collections.namedtuple(
    "BAIRConfig",
    "data_dir num_frames input_res im_height im_width channels batch_size "
    "input_seq_len pred_seq_len num_actions n_examples max_seq_length angles rescale_size"
)

TOPConfig = recordclass(
    "TOPConfig",
    "data_dir num_frames input_res im_height im_width channels batch_size "
    "input_seq_len pred_seq_len num_actions n_examples max_seq_length angles "
    "flip_lr input_height input_width rescale_size"
)

GridworldConfig = collections.namedtuple(
    "GridworldConfig",
    "dataset_type data_file num_frames batch_size image_size "
    "input_seq_len pred_seq_len channels max_action num_actions agent_size angles"
)

# A generic config for datasets of (unchunked) videos
VideoDatasetConfig = collections.namedtuple(
    "VideoDatasetConfig",
    "data_dir num_frames "
    "im_channels im_height im_width"
    "input_seq_len pred_seq_len n_examples")


def get_data_spec(spec_name,
                  dataset_phase,
                  batch_size,
                  input_seq_len=0,
                  pred_seq_len=0,
                  loss_spec=None,
                  img_res=None,
                  test_chunked=True):
  """Returns a spec with all modifiable fields set appropriately.

  Args:
    spec_name: The name of
    dataset_phase: 'train' or 'val'
    batch_size: The batch size to set for a spec.
  Returns:
    A dataset spec.
  Raises:
    ValueError: If dataset_phase is not one of 'train' or 'val', or if
      spec_name is unknown.
  """
  if dataset_phase not in {"train", "val", "test"}:
    raise ValueError("Unknown dataset_phase {}.".format(dataset_phase))

  if input_seq_len == 0:
    input_seq_len = 10
  if pred_seq_len == 0:
    pred_seq_len = 10
  num_frames = input_seq_len + pred_seq_len

  if loss_spec.image_output_activation == tf.tanh:
    rescale_size = "-1..1"
  elif loss_spec.image_output_activation == tf.nn.sigmoid:
    rescale_size = "0..1"
  else:
    raise ValueError("don't know how to rescale images for this activation")
    
  if "moving_mnist" in spec_name:
    num_digits = 1 if "single" in spec_name else 2
    spec = MovingMNISTConfig(
      dataset_type=_BOUNCING_MNIST,
      data_file="",
      num_frames=num_frames,
      batch_size=batch_size,
      image_size=img_res if img_res is not None else 64,
      num_digits=num_digits,
      step_length=0.1,
      num_colors=0,
      input_seq_len=input_seq_len,
      pred_seq_len=pred_seq_len,
      channels=1,
      random='random' in spec_name,
      num_actions=num_digits,
      angles=True)
  elif spec_name == "bouncing_balls":
    num_balls = FLAGS.num_balls
    ball_size = 0.1   # as fraction of resolution
    ball_speed = FLAGS.ball_speed    # as multiples of ball size
    agent_speed = 1   # as multiples of ball size
    stochastic_angle = FLAGS.stochastic_angle
    random_angle = FLAGS.random_angle
    stochastic_speed = FLAGS.stochastic_speed
    stochastic_bounce = FLAGS.stochastic_bounce
    rand_start_counter = FLAGS.rand_start_counter
    variance_degrees = FLAGS.variance_degrees
    bounce_vertically = FLAGS.bounce_vertically
    segment_length_limits = [FLAGS.min_segment_length, FLAGS.max_segment_length]
    image_input = FLAGS.image_input
    spec = BouncingBallsConfig(
      resolution=img_res if img_res is not None else 64,
      num_balls=num_balls,
      ball_size=ball_size,
      ball_speed=ball_speed,
      agent_speed=agent_speed,
      random_angle=random_angle,
      stochastic_angle=stochastic_angle,
      stochastic_speed=stochastic_speed,
      stochastic_bounce=stochastic_bounce,
      rand_start_counter=rand_start_counter,
      variance_degrees=variance_degrees,
      bounce_vertically=bounce_vertically,
      segment_length_limits=segment_length_limits,
      batch_size=batch_size,
      input_seq_len=input_seq_len,
      pred_seq_len=pred_seq_len,
      num_frames=num_frames,
      channels=1,
      num_actions=2,
      image_input=image_input)
  elif spec_name == "kth_basic":
    person_ids_train = range(0, 17)
    chunked_examples = False

    if chunked_examples:
      print("Using chunked examples is deprecated!!!")
      # NB: using squashed images here
      data_dir_base = os.path.join(m_specs._KTH_DIR, "KTH_chunked_20_random")
      input_height = 128
      input_width = 128
      n_examples_train = np.sum(np.asarray(_KTH_FILE_SIZES)[person_ids_train])
      n_examples_val = 100
    else:
      data_dir_base = os.path.join(m_specs._KTH_DIR, "KTH_unchunked_large")
      input_height = 128
      input_width = 160
      n_examples_train = 383
      n_examples_val = 1000
      if test_chunked:
        n_examples_test = 88289  # test_chunked_len_30
      else:
        n_examples_test = 192   # test_unchunked

    data_dir_train = os.path.join(data_dir_base, "train") # This has video_length larger than 75
    data_dir_val = os.path.join(data_dir_base, "val") # This has video_length 20
    if test_chunked:
      data_dir_test = os.path.join(data_dir_base, "test_chunked_len_30")
    else:
      data_dir_test = os.path.join(data_dir_base, "test_unchunked")
      
    if dataset_phase == "train":
      data_dir = data_dir_train
      person_ids = person_ids_train
      n_examples = n_examples_train
    elif dataset_phase == "val":
      data_dir = data_dir_val
      person_ids = []
      n_examples = n_examples_val
    elif dataset_phase == "test":
      data_dir = data_dir_test
      person_ids = []
      n_examples = n_examples_test
    else:
      raise ValueError("unknown dataset phase")
      
    spec = KTHConfig(
      data_dir=data_dir,
      person_ids=person_ids,
      action_classes=None,
      video_types=None,
      num_frames=num_frames,
      channels=1,
      im_height=128,
      im_width=128,
      input_height=input_height,
      input_width=input_width,
      input_seq_len=input_seq_len,
      pred_seq_len=pred_seq_len,
      n_examples=n_examples,
      flip_lr=dataset_phase == "train",
      rescale_size=rescale_size
    )
  elif spec_name == "h36":
    # The videos have variable length around 120
    
    data_dir = os.path.join(m_specs._H36_DIR, dataset_phase)
    with open(os.path.join(data_dir, "README.txt")) as fp:
      n_examples = int(fp.readline())
    
    spec = H36Config(
      data_dir=data_dir,
      num_frames=num_frames,
      channels=3,
      im_height=64,
      im_width=64,
      input_height=64,
      input_width=64,
      input_seq_len=input_seq_len,
      pred_seq_len=pred_seq_len,
      n_examples=n_examples,
      flip_lr=dataset_phase == "train",
      rescale_size=rescale_size
    )
  elif "ucf101" in spec_name:
    data_dir_base = os.path.join(m_specs._UCF_DIR, "rgb_256x320")
    data_dir = os.path.join(data_dir_base, dataset_phase)
    
    input_height = 256
    input_width = 320
    im_height = 256
    im_width = 256
    channels = 3 if "rgb" in spec_name else 1

    if dataset_phase == "train":
      n_examples = 9537
    elif dataset_phase == "val":
      n_examples = 1000
    else:
      n_examples = 3783
      
    spec = UCFConfig(
        data_dir=data_dir,
        num_frames=num_frames,
        channels=channels,
        im_height=im_height,
        im_width=im_width,
        input_height=input_height,
        input_width=input_width,
        input_seq_len=input_seq_len,
        pred_seq_len=pred_seq_len,
        chunked_examples=dataset_phase == "val",
        n_examples=n_examples,
        flip_lr=dataset_phase == "train"
    )
  elif spec_name[0:7] == "reacher":
      if spec_name == "reacher" or \
                      spec_name == "reacher_relative" or \
                      spec_name == "reacher_oneJoint" or \
                      spec_name == "reacher_oneJoint_balanced" or \
                      spec_name == "reacher_oneJoint_smallShard" or \
                      spec_name == "reacher_oneJoint_verySmallShard" or \
                      spec_name == "reacher_oneJoint_ExtSmallShard" or \
                      spec_name == "reacher_oneJoint_Shard100" or \
                      spec_name == "reacher_oneJoint_Shard10" or \
                      spec_name == "reacher_long":
        action_coding = "relative"
      elif spec_name == "reacher_absolute":
        action_coding = "absolute"
      else:
        raise ValueError("Reacher dataset action coding options "
                         "are [reacher/reacher_relative/reacher_oneJoint/reacher_absolute/reacher_long]")

      if spec_name[0:16] == "reacher_oneJoint":
        _NUM_JOINTS = 1
        
        data_dir_train = "/NAS/data/" + spec_name + "/train"
        data_dir_val = "/NAS/data/" + spec_name + "/val"
      else:
        _NUM_JOINTS = 2
        if spec_name == "reacher_long":
          data_dir_train = "/NAS/data/reacher_long_seqs/train"
          data_dir_val = "/NAS/data/reacher_long_seqs/val"
        else:
          data_dir_train = "/NAS/data/reacher/train"
          data_dir_val = "/NAS/data/reacher/val"
      
      if dataset_phase == "train":
          data_dir = data_dir_train
          n_examples = 100000
      elif dataset_phase == "val" or dataset_phase == "test":
          data_dir = data_dir_val
          n_examples = 1000

      spec = ReacherConfig(
        data_dir=data_dir,
        num_frames=num_frames,
        input_seq_len=input_seq_len,
        pred_seq_len=pred_seq_len,
        batch_size=batch_size,
        im_height=64,
        im_width=64,
        policy="random",
        num_actions=_NUM_JOINTS,
        max_degree=40,  # maximum action [in degree]
        action_coding=action_coding,  # action coding can be [summed/absolute/relative]
        channels=3,
        num_joints=_NUM_JOINTS,
        n_examples=n_examples,
        angles=True)
      
  elif spec_name == "bair":
      _BAIR_TRAIN_SEQUENCES = 41815
      _BAIR_VAL_SEQUENCES = 2304
      _BAIR_TEST_SEQUENCES = 256

      _BAIR_MAX_SEQ_LENGTH = 30

      data_dir_train = "/NAS/data/bair/train"
      data_dir_val = "/NAS/data/bair/val"
      data_dir_test = "/NAS/data/bair/test"

      if num_frames > _BAIR_MAX_SEQ_LENGTH:
          raise ValueError("Maximal sequence length on BAIR is %d, current desired length is %d!"\
                           % (_BAIR_MAX_SEQ_LENGTH, num_frames))

      if dataset_phase == "train":
          data_dir = data_dir_train
          n_examples = _BAIR_TRAIN_SEQUENCES
      elif dataset_phase == "val":
          data_dir = data_dir_val
          n_examples = _BAIR_VAL_SEQUENCES
      elif dataset_phase == "test":
          data_dir = data_dir_test
          n_examples = _BAIR_TEST_SEQUENCES
        
      spec = BAIRConfig(
          data_dir=data_dir,
          num_frames=num_frames,
          input_seq_len=input_seq_len,
          pred_seq_len=pred_seq_len,
          num_actions=4,
          batch_size=batch_size,
          input_res=64,
          im_height=64,
          im_width=64,
          channels=3,
          n_examples=n_examples,
          max_seq_length=_BAIR_MAX_SEQ_LENGTH,
          angles=False,
          rescale_size=rescale_size)

  elif spec_name == "top":
      _TOP_TRAIN_SEQUENCES = 1024
      _TOP_VAL_SEQUENCES = 256
      _TOP_TEST_SEQUENCES = 128

      _TOP_MAX_SEQ_LENGTH = 50

      data_dir = [os.path.join(s, dataset_phase) for s in m_specs._TOP_DIR] if isinstance(m_specs._TOP_DIR, list) \
                        else os.path.join(m_specs._TOP_DIR, dataset_phase)

      if num_frames > _TOP_MAX_SEQ_LENGTH:
          raise ValueError("Maximal sequence length on TOP is %d, current desired length is %d!"\
                           % (_TOP_MAX_SEQ_LENGTH, num_frames))

      if dataset_phase == "train":
          n_examples = _TOP_TRAIN_SEQUENCES
      elif dataset_phase == "val":
          n_examples = _TOP_VAL_SEQUENCES
      elif dataset_phase == "test":
          n_examples = _TOP_TEST_SEQUENCES
        
      spec = TOPConfig(
          data_dir=data_dir,
          num_frames=num_frames,
          input_seq_len=input_seq_len,
          pred_seq_len=pred_seq_len,
          num_actions=4,
          batch_size=batch_size,
          input_res=img_res,
          im_height=img_res,
          im_width=img_res,
          channels=3,
          n_examples=n_examples,
          max_seq_length=_TOP_MAX_SEQ_LENGTH,
          angles=False,
          flip_lr=True,
          input_height=img_res,
          input_width=img_res,
          rescale_size=rescale_size)

  elif spec_name == "gridworld":
    spec = GridworldConfig(
          dataset_type=_BOUNCING_MNIST,
          data_file="",
          num_frames=num_frames,
          batch_size=batch_size,
          image_size=64,
          input_seq_len=input_seq_len,
          pred_seq_len=pred_seq_len,
          channels=1,
          max_action=5,        # max spatial translation per step in px
          num_actions=2,
          agent_size=9,
          angles=False)         # agent spatial dimension
  else:
    raise ValueError("Unknown dataset spec name.")

  return spec
