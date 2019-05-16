import matplotlib.pyplot as plt
import glob
import os.path as osp
import numpy as np

from eval import EvaluatedTrajectory
from viz.html_utils.ffmpeg_gif import save_gif

N_COLS = 15
MAX_TRAJS = 120

data_dir = "/home/jingyuny/eval_gt/noHL/eval-noHL-3x200-gt-0_120"

traj_frames = []
traj_filenames = glob.glob(osp.join(data_dir, '*.p'), recursive=True)
traj_filenames = traj_filenames[:MAX_TRAJS] if MAX_TRAJS < len(traj_filenames) else traj_filenames
print("Plotting %d trajectories!" % len(traj_filenames))
for filename in traj_filenames:
    if filename.endswith(".p"):
        traj = EvaluatedTrajectory(filename)
        traj_frames.append(traj.frames)

# compute max traj length
lengths = [t.shape[0] for t in traj_frames]
max_length = max(lengths)

# pad each sequence
frame_shape = traj_frames[0].shape[1:]
res = frame_shape[0]
num_trajs = len(traj_frames)
for i in range(num_trajs):
    length = lengths[i]
    traj_frames[i] = np.concatenate((traj_frames[i], np.zeros([max_length + 1 - length] + list(frame_shape), dtype=traj_frames[0].dtype)), axis=0)

# sort frames in grid
N_ROWS = int(np.floor(num_trajs / N_COLS))
traj_frames = traj_frames[:N_ROWS*N_COLS]

concat_imgs = []
for img in range(max_length):
    img_i = np.empty((N_ROWS * res, N_COLS * res, frame_shape[-1]), dtype=traj_frames[0].dtype)   
    for row in range(N_ROWS):
        for col in range(N_COLS):
            img_i[row*res : (row+1)*res, col*res : (col+1)*res] = traj_frames[row*N_COLS + col][img]
    concat_imgs.append(img_i)

# save resulting gif file
save_path = osp.join("/tmp", "overview_noHL.gif")
save_gif(save_path, concat_imgs, fps=10)
print("Done!")
