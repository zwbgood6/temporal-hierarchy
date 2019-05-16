import cv2
import numpy as np
from tqdm import tqdm
import os
import datetime


tip_coord = []
scale = 4       # how much larger to display the image
base_dir = "./groundtruth"
lab_file = "labels_gt.npy"
input_file = "test_seqs_gt.npy"
data_file = os.path.join(base_dir, input_file)
outfile_name = os.path.join(base_dir, lab_file)


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global tip_coord

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        tip_coord = [x, y]


def annotate_img(img):
    go_back = False
    is_fail = False
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)


    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", img[:,:,::-1])
        key = cv2.waitKey(1) & 0xFF

        # if the 'c' key is pressed, break from the loop
        if key == 32:   # space
            break
        elif key == ord("g"):
            is_fail = False
        elif key == ord("f"):
            is_fail = True
        elif key == ord("r"):
            go_back = True
            break
    cv2.destroyAllWindows()
    return go_back, is_fail


def display_annotation(imgs, labels):
    num_imgs = imgs.shape[0]
    cv2.namedWindow("image")
    for img_idx in range(num_imgs):
        img = imgs[img_idx]
        img = cv2.resize(img, (resolution * scale, resolution * scale))
        scaled_x = int(labels[img_idx, 1]*scale)
        scaled_y = int(labels[img_idx, 0]*scale)
        img[scaled_x-3:scaled_x+3, scaled_y-3:scaled_y+3] = [1.0, 0.0, 0.0]
        cv2.imshow("image", img[:, :, ::-1])
        key = cv2.waitKey(500) & 0xFF   # half a second
    cv2.destroyAllWindows()


def backup_previous_labels():
    if os.path.isfile(outfile_name):
        appended_str = datetime.datetime.now().strftime("_bkp_%Y_%m_%d_%H_%M_%S")
        bkp_filename = os.path.splitext(outfile_name)[0] + appended_str + ".npy"
        os.rename(outfile_name, bkp_filename)


def dump_labels(labels):
    label_file_name = os.path.join(os.path.dirname(data_file), lab_file)
    if len(labels) > 1:
        labels = [np.expand_dims(s, axis=0) for s in labels]
        labels = np.concatenate(labels, axis=0)
    np.save(label_file_name, labels)


def maybe_load_prev_labels():
    if os.path.isfile(outfile_name):
        print("Loading already annotated labels!")
        labels = np.load(outfile_name)
        labels = [np.squeeze(l) for l in np.split(labels, labels.shape[0], axis=0)]
        backup_previous_labels()
        return labels
    else:
        return []


if __name__ == "__main__":
    imgs = np.load(data_file)
    imgs = (imgs+1)/2
    imgs = np.transpose(imgs, (0, 1, 3, 4, 2))

    seq_len, num_seqs, channels, resolution, _ = imgs.shape
    output_label_list = maybe_load_prev_labels()
    start_seq = len(output_label_list)
    for seq in tqdm(range(num_seqs)[start_seq:]):
        labels = np.empty((seq_len, 3))
        step = 0
        while True:
            img = imgs[step, seq]
            img = cv2.resize(img, (resolution*scale, resolution*scale))
            go_back, is_fail = annotate_img(img)
            if go_back:
                step = max(0, step-1)
                continue

            labels[step, 0] = tip_coord[0]/scale
            labels[step, 1] = tip_coord[1]/scale
            labels[step, 2] = 1 if is_fail else 0

            if step == seq_len-1:
                break
            step += 1
        # display_annotation(imgs[:, seq], labels)
        # print(labels)
        output_label_list.append(labels)
        dump_labels(list(output_label_list))

    print("Congrats, you're done with this one!")