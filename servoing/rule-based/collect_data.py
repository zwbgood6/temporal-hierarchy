import gym
import os
import os.path as osp
import time
import datetime
import argparse
import subprocess
from multiprocessing.pool import ThreadPool
import sys
import signal
import atexit
import pickle
import tensorflow as tf
import cv2

import rule_based_pusher
from rule_based_pusher import RuleBasedPusher

cmd_template = ("python3 rule_based_pusher.py --env_name {} --seed {} --prod "
                "--prod_ix {} --prod_dir {} --prod_output_steps {} "
                "--min_push_dist {} --max_push_dist {} --subsample_rate {}")

# logger 

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

def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

class Logger(object):
    def __init__(self):
        self.log = None
        
    def set_logfile(self, logfile):
        self.log = open(logfile, 'a')
        self.log.write('Run at {}\n'.format(time.strftime('%Y/%m/%d %H:%M:%S')))

    def write(self, message, color=None):
        if color is not None:
            print(colorize(message, color, bold=True))
        else:
            print(message)

        if self.log is not None:
            self.log.write('{}\n'.format(message))
            self.log.flush()
            
    def __del__(self):
        if self.log is not None:
            self.log.write('\n\n')
            self.log.flush()
            self.log.close()
        
# helper functions

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env', type=str, default='MultigoalObstaclePush-v0')
    parser.add_argument('--num_traj', type=int, default=128)
    parser.add_argument('--num_thread', type=int, default=8)
    parser.add_argument('--num_records_per_tfrecord_file', type=int, default=128)
    parser.add_argument('--redo_failed', action='store_true', default=False)
    parser.add_argument('--logfile', type=str, default='log.txt')
    parser.add_argument('--out_dir', type=str, default='data_collection')
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--image_sidelength', type=int, default=64)
    parser.add_argument('--episode_length', type=int, default=50)
    parser.add_argument('--min_push_dist', type=float, default=0.4)
    parser.add_argument('--max_push_dist', type=float, default=0.6)
    parser.add_argument('--output_format', type=str, choices=['tfrecord', 'video'], default='tfrecord')
    parser.add_argument('--subsample_rate', type=int, default=1)
    parser.add_argument('--restore_group', type=int, default=0)
    
    return parser.parse_args()

def run_single_program(ix, env_name, num_traj, redo_failed,
                       logger, temp_dir, timeout, episode_length, output_format,
                       min_push_dist, max_push_dist, subsample_rate):
    exec_success = False
    run_ix = 0
    while not exec_success:
        start = time.time()
        
        logger.write('Starting process #{:05d}'.format(ix), 'white')
        cmd = cmd_template.format(env_name, ix + run_ix * num_traj, ix, 
                                  temp_dir, episode_length, 
                                  min_push_dist, max_push_dist,
                                  subsample_rate)
        if output_format == 'tfrecord':
            cmd += ' --prod_dump'
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    
        try:
            (output, err) = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            logger.write('Process #{:05d} timeout. Trying again.'.format(ix), 'red')
            if redo_failed:
                run_ix += 1
            continue
           
        if not p.returncode == 0:
            logger.write('Process #{:05d} exited with error code {:d}.'.format(ix, p.returncode), 'red')
        else:
            end = time.time()
            
            pickle_filename = osp.join(temp_dir, 'traj_{:05d}.p'.format(ix))
            pickle_data = pickle.load(open(pickle_filename, 'rb'))
            success, t, rew = pickle_data[:3]
            subtask_lengths = pickle_data[-2]
            os.remove(pickle_filename)

            if success:
                exec_success = True
                logger.write('Process #{:05d} finished at {:d} steps {:s} with reward {:.3f} in {:.2f}s.'.format(ix, t, '{}'.format(tuple(subtask_lengths)), rew, (end - start)), 'cyan')
            else:
                frames = pickle_data[4]

                if len(frames) > 0:
                    failure_reason = 'failed to complete the task'
                else:
                    failure_reason = 'failed to sample subgoals'

                if redo_failed or len(frames) == 0:
                    run_ix += 1
                    logger.write('Process #{:05d} {:s}. Running again.'.format(ix, failure_reason), 'yellow')
                else:
                    exec_success = True
                    logger.write('Process #{:05d} {:s}.'.format(ix, failure_reason), 'yellow')

        p.kill()

    return pickle_data

def list_to_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

def cleanup():
    os.killpg(0, signal.SIGKILL)
    
# tfrecord utils

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    
def save_as_tfrecord(filename, examples, image_sidelength):
    writer = tf.python_io.TFRecordWriter(filename)
    for example in examples:
        success, total_time, rew, num_timesteps, frames, arm_pos, obs, ac, is_key_frame = example[:9]

        goal_timestep = min(total_time, num_timesteps)

        feature = {'success': _int64_feature(success),
                   'num_timesteps': _int64_feature(num_timesteps),
                   'goal_timestep': _int64_feature(goal_timestep),
                   'final_reward': _float_feature(rew)}

        if len(example) == 11:
            goal_image = example[-1]
            downsampled_goal_image = cv2.resize(goal_image,
                                            dsize=(image_sidelength, image_sidelength),
                                            interpolation=cv2.INTER_CUBIC)
            goal_image_feature = _bytes_feature(tf.compat.as_bytes(downsampled_goal_image.tostring()))
            feature['goal_image'] = goal_image_feature
        
        for t in range(num_timesteps):
            downsampled_frame = cv2.resize(frames[t], 
                                               dsize=(image_sidelength, image_sidelength), 
                                               interpolation=cv2.INTER_CUBIC)
            image_feature = _bytes_feature(tf.compat.as_bytes(downsampled_frame.tostring()))
            
            feature['{}/action'.format(t)] = _float_feature(ac[t].tolist())
            feature['{}/endeffector_pos'.format(t)] = _float_feature(arm_pos[t].tolist())
            feature['{}/image_view0/encoded'.format(t)] = image_feature
            feature['{}/obs'.format(t)] = _float_feature(obs[t].tolist())
            feature['{}/is_key_frame'.format(t)] = _int64_feature(is_key_frame[t])
        
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

# main program

def main():
    start_time = time.time()
    os.setpgrp()
    atexit.register(cleanup)
    args = parse_args()
    
    logger = Logger()
    logger.set_logfile(args.logfile)

    logger.write(args, 'white')
    
    if not osp.exists(args.out_dir): os.makedirs(args.out_dir)

    func = lambda ix: run_single_program(ix, args.env, args.num_traj, args.redo_failed, logger, args.out_dir, args.timeout, args.episode_length, args.output_format, args.min_push_dist, args.max_push_dist, args.subsample_rate)
    for i, ix_list in enumerate(list_to_chunks(list(range(args.num_traj)), args.num_records_per_tfrecord_file)[args.restore_group:], start=args.restore_group):
        logger.write('---- GROUP {:03d} -----'.format(i + 1), 'white')
        tp = ThreadPool(args.num_thread)
        results = tp.map(func, ix_list)
        tp.close()
        tp.join()
        tfrecord_filename = osp.join(args.out_dir, 'traj_{}_to_{}.tfrecord'.format(ix_list[0], ix_list[-1]))
        if args.output_format == 'tfrecord':
            save_as_tfrecord(tfrecord_filename, results, args.image_sidelength)
    
    end_time = time.time()

    logger.write('Program finished in {}.'.format(str(datetime.timedelta(seconds=int(end_time - start_time)))))

if __name__ == '__main__':
    main()
    
