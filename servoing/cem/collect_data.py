import subprocess
from multiprocessing.pool import ThreadPool
import argparse
import sys
import os
import signal
import atexit
import time

# prevent tensorflow from printing logs and warnings
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# command line template
cmd_template = 'python3 servo.py --env_name {} --seed {} --prod --prod_short_path --prod_ix {} --max_steps {}'

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
        pass

    def set_dir(self, cmd_line_args_str):
        # extract the right argument from the list
        for i in range(len(cmd_line_args_str) - 1):
            if "prod_dir" in cmd_line_args_str[i]:
                self.prod_dir = cmd_line_args_str[i+1]
                break
        if not os.path.exists(self.prod_dir):
            os.makedirs(self.prod_dir)

        self.log = open(os.path.join(self.prod_dir, 'log.txt'), 'a')
        self.log.write('\n\nRun at {}\n'.format(time.strftime('%Y/%m/%d %H:%M:%S')))

    def write(self, message, color=None):
        if color is not None:
            print(colorize(message, color, bold=True))
        else:
            print(message)

        self.log.write(message + '\n')
        self.log.flush()

# helper functions

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='RandomObstaclePush-v0')
    parser.add_argument('--num_traj', type=int, default=128)
    parser.add_argument('--num_thread', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--thread_timeout', type=int, default=600)
    parser.add_argument('--min_reward', type=float, default=None)
    parser.add_argument('--redo_failed', action='store_true', default=False)
    parser.add_argument('--start_seed', type=int, default=0)

    return parser.parse_known_args()

def run_single_program(ix, env_name, num_traj, max_steps, timeout, min_reward, 
                       redo_failed, start_seed, args_to_pipe):
    reward_requirement_met = False
    run_ix = 0
    while not reward_requirement_met:
        start = time.time()

        logger.write('Starting process #{:03d}'.format(ix), 'white')
        cmd = cmd_template.format(env_name, 
                                  ix + start_seed + run_ix * num_traj,
                                  ix + start_seed, 
                                  max_steps)
        if min_reward is not None:
            cmd += ' --prod_min_reward {}'.format(min_reward)
        cmd += ' ' + args_to_pipe
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    
        try:
            (output, err) = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            run_ix += 1
            logger.write('Process #{:03d} timeout. Running again.'.format(ix), 'red')
            p.kill()
            continue
    
        if not p.returncode == 0:
            logger.write('Process #{:03d} exited with error code {:d}.'.format(ix, p.returncode), 'red')
        else:
            reward = float(output.splitlines()[-2].decode('ascii').split(' ')[-1])
            end = time.time()
            if min_reward is None or reward >= min_reward:
                reward_requirement_met = True
                logger.write('Process #{:03d} finished with reward {:.3f} in {:.2f}s.'.format(ix, reward, end - start), 'cyan')
            else:
                run_ix += 1
                logger.write('Process #{:03d} finished with reward {:.3f} in {:.2f}s.{:s}'.format(ix, reward, end - start, ' Running again.' if redo_failed else ''), 'yellow')
        
        if not redo_failed: reward_requirement_met = True

        p.kill()

    return

def cleanup():
    os.killpg(0, signal.SIGKILL)

# main program
    
logger = Logger()

def main():
    os.setpgrp()
    atexit.register(cleanup)
    args, args_to_pipe = parse_args()

    logger.set_dir(args_to_pipe)
    
    logger.write('env_name: {}; num_traj: {}; num_thread: {}; max_steps: {}; thread_timeout: {}; min_reward: {}; redo_failed: {}'.format(args.env_name, args.num_traj, args.num_thread, args.max_steps, args.thread_timeout, args.min_reward, 'True' if args.redo_failed else 'False'), 'gray')

    tp = ThreadPool(args.num_thread)
    for i in range(args.num_traj):
        tp.apply_async(run_single_program, (i, args.env_name, args.num_traj, args.max_steps, args.thread_timeout, args.min_reward, args.redo_failed, args.start_seed, ' '.join(args_to_pipe)))
    tp.close()
    tp.join()

    logger.write('Program finished!', 'green')

if __name__ == '__main__':
    main()
