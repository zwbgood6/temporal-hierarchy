'''
Created by Jingyun Yang on 1/17/2018.

Calculates evaluation metrics based on generated rollouts from specified 
trajectory.
'''

from __future__ import absolute_import

import os.path as osp
import numpy as np
from numpy.linalg import norm
import pickle
import argparse
import glob
import os

from servoing.cem.simulator import GTSimParams, GTSimulator

class EvaluatedTrajectory(object):
    def __init__(self, filename, goal_reached_dist=None):
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)
        self._goal_reached_dist = goal_reached_dist
        self._frames = None
        self.obs = np.array(self.data['obs'])
        self.actions = np.array(self.data['ac'])
        self.args = self.data['args']
        self.pos = list(map(lambda t: {
            'robot': self.obs[t][10:13],
            'obj': self.obs[t][13:16],
            'goal': self.goal,
            'obstacle': self.obstacle_config
        }, list(range(len(self.obs)))))
        
    @property
    def obstacle_config(self):
        return self.obs[0][-3:]
    
    @property
    def goal(self):
        return self.obs[0][16:19]
   
    @property
    def frames(self):
        if self._frames is None:
            sim_params = GTSimParams(env_name='ObstaclePush-v0',
                                     num_action_repeat=5,
                                     output_resolution=128,
                                     render_goal=True)
            sim = GTSimulator(sim_params)
            init_state = self.data['init_state']
            # edit MuJoCo environment init state for backward compatibility
            if len(init_state.qpos) == 12:
                from mujoco_py import MjSimState
                new_qpos = np.concatenate([init_state.qpos[:9],
                                           np.array([10, 10, 0]),
                                           init_state.qpos[-3:]])
                new_qvel = np.concatenate([init_state.qvel[:9],
                                           np.array([0, 0, 0]),
                                           init_state.qvel[-3:]])
                init_state = MjSimState(time=0,
                                        qpos=new_qpos,
                                        qvel=new_qvel,
                                        act=init_state.act,
                                        udd_state=init_state.udd_state)
            output_dict = sim.rollout(init_state,
                                      np.expand_dims(self.actions, axis=0))
            self._frames = output_dict.pred_frames[0]
        return self._frames
    
    @property
    def final_goal_distance_around_obstacle(self):
        pos_obj = self.pos[-1]['obj'][:2]
        pos_goal = self.goal[:2]
        pos_obstacle = self.obstacle_config[:2]
        ang_obstacle = self.obstacle_config[-1]

        # transform pos_goal and pos_object to obstacle coordinate
        def coord_trans(c_origin, c_ang):
            def trans(pt):
                residual = pt - c_origin
                dist = norm(residual)
                ang = np.arctan2(residual[1], residual[0]) - c_ang
                return np.array([dist * np.cos(ang), dist * np.sin(ang)])
            return trans

        trans_fn = coord_trans(pos_obstacle, ang_obstacle)
        pos_goal = trans_fn(pos_goal)
        pos_obj = trans_fn(pos_obj)
        
        # object and goal parallel to obstacle
        if pos_goal[1] == pos_obj[1]:
            dist = np.linalg.norm(pos_goal - pos_obj)
        else:
            intercept = pos_goal[0] - (pos_obj[0] - pos_goal[0]) / (pos_obj[1] - pos_goal[1]) * pos_goal[1]
            # no intersection
            if np.abs(intercept) > 0.52:
                dist = np.linalg.norm(pos_goal - pos_obj)
            else:
                # intersection on right side
                if intercept > 0:
                    intermediate = np.array([0.52, 0])
                # intersection on left side
                else:
                    intermediate = np.array([-0.52, 0])
                dist = norm(pos_goal - intermediate) + norm(intermediate - pos_obj)

        return dist

    @property
    def goal_reached(self):
        pos_obj = self.pos[-1]['obj'][:2]
        pos_goal = self.goal[:2]
        final_dist = np.linalg.norm(pos_goal - pos_obj)
        return True if self.final_goal_distance_around_obstacle < self._goal_reached_dist else False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='/home/jingyuny/')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--goal_reached_dist', type=float, default=0.3)

    return parser.parse_args()

def main():
    args = parse_args()

    dd = []
    for dir in ["/home/jingyuny/eval_gt/noHL/eval-noHL-3x200-gt-0_120/", "/home/karl/code/temporal-hierarchy/logs_prod/rand_baseline", "/home/jingyuny/eval_gt/ours_sparse/eval-ours_sparse-3x200-gt-0_120/"]:
        distances = []
        goal_reached = 0.0
        for filename in glob.glob(osp.join(dir, '**/*.p'), recursive=True):
            if args.prefix in filename and filename.endswith(".p"):
                traj = EvaluatedTrajectory(filename, goal_reached_dist=args.goal_reached_dist)
                distances.append(traj.final_goal_distance_around_obstacle)
                print('{}: {}'.format(osp.basename(filename),
                                      distances[-1]))
                if traj.goal_reached:
                    goal_reached += 1
        dd.append(distances)
        print()
        print("#Eval Seqs: {:d}".format(len(distances)))
        print('Mean = {:.3f}'.format(np.mean(distances)))
        print('Var = {:.3f}'.format(np.var(distances)))
        print('Successfully completed %f percent of runs.' % (100.0 * goal_reached / len(distances)))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bins = np.linspace(0.0, 2.0, 50)
    plt.hist(dd[0], bins, alpha=0.33, label="no_HL")
    plt.hist(dd[1], bins, alpha=0.33, label="fixed")
    plt.hist(dd[2], bins, alpha=0.33, label="ours")
    plt.legend(loc='upper right')
    plt.savefig("/home/karl/code/temporal-hierarchy/logs/fig.png")

if __name__ == '__main__':
    main()
