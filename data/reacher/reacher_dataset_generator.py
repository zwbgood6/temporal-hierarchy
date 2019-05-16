import os
import numpy as np
from tqdm import tqdm


_BASE_DIR = "/home/karl/data/reacher_oneJoint_Shard10"
_NUM_TRAIN_SEQS = 2000
_NUM_VAL_SEQS = 1000
_SEQUENCE_LENGTH = 15
_SEQS_PER_SHARD = 10
_NUM_JOINTS = 1     # 1 or 2


n_shards_train = int(np.ceil(_NUM_TRAIN_SEQS / _SEQS_PER_SHARD))
n_shards_val = int(np.ceil(_NUM_VAL_SEQS / _SEQS_PER_SHARD))
n_shards_max = np.max((n_shards_train, n_shards_val))

print("Generate reacher dataset with %d sequences, broken up in %d shards." %
      (_NUM_TRAIN_SEQS, n_shards_train))
for index in tqdm(range(n_shards_max)):
    os.system('python -W ignore ./reacher_build_tfrecord.py '        # no warnings
              '-d %s -nt %d -nv %d -ns %d -nj %d --sequence_length %d '
              '-i %d >/dev/null'    # suppress all output
              % (_BASE_DIR, _NUM_TRAIN_SEQS, _NUM_VAL_SEQS,
                 _SEQS_PER_SHARD, _NUM_JOINTS, _SEQUENCE_LENGTH, index))
