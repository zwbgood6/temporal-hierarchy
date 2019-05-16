import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

base_dirs = ["/home/karl/Downloads/bair_labeling/groundtruth",
             "/home/karl/Downloads/bair_labeling/cdna_compActions",
             "/home/karl/Downloads/bair_labeling/cdna_noCompActions",
             "/home/karl/Downloads/bair_labeling/cdna_action_cond"]
set_names = [["groundtruth"],
             ["ours,", "unsupervised"],
             ["Denton & Fergus [8],", "unsupervised"],
             ["Finn & Levine [13],", "supervised"]]
lab_file = "labels.npy"
input_file = "test_seqs.npy"
output_dir = "/home/karl/Downloads/traj_vids"

frame_resolution = 300


def load_labels(outfile_name):
  if os.path.isfile(outfile_name):
    print("Loading already annotated labels!")
    labels = np.load(outfile_name)
    labels = [np.squeeze(l) for l in np.split(labels, labels.shape[0], axis=0)]
    return labels
  else:
    raise ValueError("Could not find the label file!")


def blend_imgs(imgs):
  return imgs[0] * 1.0#0.5 + imgs[9] * 0.5


def draw_trajectory(img, labels, max_labels=None):
  num_labels = labels.shape[0] if max_labels is None else max_labels
  total_num_labels = labels.shape[0]
  img = np.asarray(img, dtype=np.float64).copy()

  # draw lines
  # x = np.linspace(0.0, 1.0, total_num_labels - 1)
  # rgb_colors = cm.get_cmap("plasma")(x)[:, :3]
  # for lab_idx in range(num_labels-1):
  #   start = np.asarray(labels[lab_idx, :2], dtype=np.uint8)
  #   end = np.asarray(labels[lab_idx+1, :2], dtype=np.uint8)
  #   cv2.line(img, (start[0], start[1]), (end[0], end[1]), (rgb_colors[lab_idx, 0],
  #                                                          rgb_colors[lab_idx, 1],
  #                                                          rgb_colors[lab_idx, 2]), 2)

  # draw circles
  x = np.linspace(0.0, 1.0, total_num_labels)
  rgb_colors = cm.get_cmap("plasma")(x)[:, :3]
  for lab_idx in range(num_labels):
    pt = np.asarray(labels[lab_idx, :2], dtype=np.uint8)
    cv2.circle(img, (pt[0], pt[1]), 1, (rgb_colors[lab_idx, 0],
                                        rgb_colors[lab_idx, 1],
                                        rgb_colors[lab_idx, 2]), -1)
  return img


def format_imgs(imgs):
  final_img = None
  for i, dir_name in enumerate(base_dirs):
    res_img = cv2.resize(imgs[i], (frame_resolution, frame_resolution))
    set_name = set_names[i]
    if len(set_name) == 1:
      cv2.putText(res_img, set_name[0],
                  (15,frame_resolution - 35),
                  cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  1.0,
                  (0,255,0),
                  2)
    else:
      cv2.putText(res_img, set_name[0],
                  (15, frame_resolution - 55),
                  cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  1.0,
                  (0, 255, 0),
                  2)
      cv2.putText(res_img, set_name[1],
                  (15, frame_resolution - 35),
                  cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  1.0,
                  (0, 255, 0),
                  2)

    final_img = res_img if final_img is None else np.concatenate((final_img, res_img), axis=1)
  # cv2.imshow('image', final_img[..., ::-1])
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  return final_img[..., ::-1]


if __name__ == "__main__":
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  img_sets, label_sets = [], []
  for base_dir in base_dirs:
    data_file = os.path.join(base_dir, input_file)
    outfile_name = os.path.join(base_dir, lab_file)
    imgs = np.load(data_file)
    imgs = (imgs + 1) / 2
    img_sets.append(np.transpose(imgs, (0, 1, 3, 4, 2)))
    label_sets.append(load_labels(outfile_name))

  seq_len, num_seqs, channels, resolution, _ = img_sets[0].shape

  for seq_idx in tqdm(range(num_seqs)):
    # generate trajectory plot
    output_img_list = []
    for set_idx in range(len(base_dirs)):
      seq_imgs = img_sets[set_idx][:, seq_idx]
      seq_labels = label_sets[set_idx][seq_idx]
      blended_img = blend_imgs(seq_imgs)
      output_img_list.append(draw_trajectory(blended_img, seq_labels))
    video_img_list = [format_imgs(output_img_list)]
    video_img_list.append(video_img_list[0])

    # append the individual images
    for i in range(seq_len):
      step_set = []
      for set_idx in range(len(base_dirs)):
        step_img = img_sets[set_idx][i, seq_idx]
        traj_step_img = draw_trajectory(step_img, label_sets[set_idx][seq_idx], max_labels=i+1)
        step_set.append(traj_step_img)
      video_img_list.append(format_imgs(step_set))

    # save everything to a video
    outfile = os.path.join(output_dir, "seq_%d.avi" % seq_idx)
    img_size = np.asarray(video_img_list[0].shape, dtype=np.uint)[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(outfile, fourcc, 2.0, (img_size[1], img_size[0]))
    for img in video_img_list:
      writer.write(np.asarray(img*255, dtype=np.uint8))
    writer.release()

    # save complete trajectory figure individually
    outfile = os.path.join(output_dir, "seq_%d.png" % seq_idx)
    cv2.imwrite(outfile, np.asarray(video_img_list[0]*255, dtype=np.uint8))

    if seq_idx == 13:
      stored_img_stack = video_img_list

    if seq_idx == 30:
      fused_img_stack = []
      whitespace = np.ones([10, video_img_list[0].shape[1], 3])
      for img1, img2 in zip(stored_img_stack, video_img_list):
        fused_img = np.concatenate([img1, whitespace, img2], axis=0)
        fused_img_stack.append(fused_img)
      # save everything to a video
      outfile = os.path.join(output_dir, "fused_seq.avi")
      img_size = np.asarray(fused_img_stack[0].shape, dtype=np.uint)[:2]
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      writer = cv2.VideoWriter(outfile, fourcc, 2.0, (img_size[1], img_size[0]))
      for img in fused_img_stack:
        writer.write(np.asarray(img * 255, dtype=np.uint8))
      writer.release()

      # save complete trajectory figure individually
      outfile = os.path.join(output_dir, "fused_seq.png")
      cv2.imwrite(outfile, np.asarray(fused_img_stack[0] * 255, dtype=np.uint8))
