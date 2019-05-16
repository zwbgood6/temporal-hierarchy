from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import servoing.cem.params
import params
import configs

import tensorflow as tf
from servoing.cem.main import main

if __name__ == '__main__':
    tf.app.run(main=main) 
