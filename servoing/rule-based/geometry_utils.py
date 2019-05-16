import math
from skimage.draw import circle
import numpy as np


# Computes angle between two 2D vectors x and y.
# 
# Args:
#     x: 2D vector.
#     y: another 2D vector.
#
# Returns: value in range [0, 2 * pi) representing counter-clockwise
#          angle from x to y.
def angle_between(x, y):
    ang_x = math.atan2(x[1], x[0])
    ang_y = math.atan2(y[1], y[0])
    return clip_angle_0_to_2pi(ang_y - ang_x)


# Clips angle to [0, 2 * pi)
#
# Args:
#     a: angle in radian.
# 
# Returns: clipped angle value between [0, 2 * pi).
def clip_angle_0_to_2pi(a):
    while a < 0:
        a += math.pi * 2
    while a > math.pi * 2:
        a -= maht.pi * 2
    return a


# Clips angle to [-pi, pi)
#
# Args:
#     a: angle in radian.
# 
# Returns: clipped angle value between [-pi, pi).
def clip_angle_npi_to_pi(a):
    while a < -math.pi:
        a += math.pi * 2
    while a > math.pi:
        a -= math.pi * 2
    return a


# Offsets a specified point by a distance and an angle
#
# Args:
#     p: point to be applied offset
#     d: distance to offset
#     a: angle to offset
def offset_point(p, d, a):
    return np.array([p[0] + np.cos(a) * d, p[1] + np.sin(a) * d])


def pt2pix(pt, frame_size=100):
    """Transfer coordinate value into corresponding pixel value.
    :arg pt: coordinate, range [-1, 1]
    :arg frame_size: returned pixel value in frame_size x frame_size frame
    :return corresponding pixel value in range [0, frame_size-1]
    """
    return np.asarray(np.maximum(0, (0.5*(pt+1) * frame_size) - 1), dtype=np.int32)


def pix2pt(pix, frame_size=100.0):
    """Transfer pixel value into corresponding coordinate value.
    :arg pix: pixel value, range [0, frame_size-1]
    :arg frame_size: input pixel value in frame_size x frame_size frame
    :return corresponding coordinate value
    """
    return 2*(pix / (frame_size-1)) - 1.0


def draw_torus(pix, min_dist, max_dist, frame_size=100):
    """Returns a binary image of a torus around given point with given inner/outer radius.
    :arg pix: pixel coordinates (np.array, shape (2,))
    :arg min_dist: inner radius of torus in coordinate value
    :arg max_dist: outer radius of torus in coordinate value
    :arg frame_size: frame_size in px of returned rendering
    :return binary np.array of size frame_size x frame_size
    """
    torus_img = np.zeros((frame_size, frame_size), dtype=np.int32)
    radius_multiplier = frame_size / 2
    outer_radius = max_dist * radius_multiplier
    inner_radius = min_dist * radius_multiplier
    ox, oy = circle(pix[0], pix[1], radius=outer_radius, shape=torus_img.shape)
    ix, iy = circle(pix[0], pix[1], radius=inner_radius, shape=torus_img.shape)
    torus_img[ox, oy] = 1
    torus_img[ix, iy] = 0
    return torus_img


def compute_intersection(frames):
    """Returns intersection image of all frames in given list.
    :arg frames: list of binary frames of equal shape
    :return intersection frame
    """
    for f in frames:
        assert f.shape == frames[0].shape, "Input images need to have same shape for intersection!"
    assert len(frames) > 1, "Need at least two frames to compute intersection from!"

    intersect = frames[0]
    for f in frames[1:]:
        intersect = np.logical_and(f, intersect)
    return intersect


def sample_from_intersect(frame):
    """Samples one index pair from all True elements in the 2D-array frame.
    :arg frame: boolean 2D numpy array
    :return pixel indices of sampled point
    """
    pix_idxs = np.where(frame)
    if len(pix_idxs[0]) == 0: return None
    sample = np.random.randint(pix_idxs[0].shape[0])
    return np.asarray([pix_idxs[0][sample], pix_idxs[1][sample]])


if __name__ == "__main__":
    pt = np.asarray([0.1, 0.5])
    pt2 = np.asarray([0.1, 0.8])
    pt3 = np.asarray([0.3, 0.65])
    pix = pt2pix(pt)
    pix2 = pt2pix(pt2)
    pix3 = pt2pix(pt3)

    x1 = draw_torus(pix, 10, 20)
    x2 = draw_torus(pix2, 10, 20)
    x3 = draw_torus(pix3, 10, 20)

    ii = compute_intersection([x1, x2, x3])

    for _ in range(100000):
        s_pix = sample_from_intersect(ii)
        if not ii[s_pix[0], s_pix[1]]:
            print("FAIL!")
    print("Done...")

    ptt = pix2pt(s_pix)
