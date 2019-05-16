import numpy as np
import cv2


_PUCK_COLOR = np.array([179, 89, 134])
_MAX_COLOR_DIFF = 100.0
def mask_puck(img):
    assert img.dtype == np.float32, "Input image to color masking should be float!"
    assert np.max(img) <= 1.0, "Input range to masking should be [0...1]"
    copied_img = (np.copy(img) * 255).astype(np.uint8)
    diff_img = np.linalg.norm(copied_img - _PUCK_COLOR, axis=-1)
    diff_mask = diff_img < _MAX_COLOR_DIFF
    mask = np.asarray(diff_mask, dtype=np.float32)
    return mask


''' Compute object position in an image with pixel values between 
    [0, 1] using color filtering.

Args:
    img: an image with shape [h, w, c] and pixel values in range [0, 1].

Returns: a 2d vector describing coordinates of center of object.
'''

def get_object_position(img):
    mask = mask_puck(img)
    if not mask.any():
        return None
    return np.median(np.argwhere(mask == 1.0), 0)

''' Compute distance between objects in the two given images using color 
    filtering.

Args:
    img1, img2: two input images with shape [h, w, c] and pixel values in 
                range [0, 1].

Returns: a positive number indicating distance between objects in the two 
         given images.
'''
def color_filter_distance_np(img1, img2):
    img1_obj_pos = get_object_position(img1)
    img2_obj_pos = get_object_position(img2)
    if img1_obj_pos is None or img2_obj_pos is None:
        return np.Inf
    return np.linalg.norm(img1_obj_pos - img2_obj_pos)

'''	Draws an empty image with a purple object draw as specified location.
    Function used for debugging purposes only.

Args:
    obj_pos: a 2d vector describing coordinates of object.

Returns: an image with shape (500, 500, 3) and pixel value range [0, 1].
'''
def make_debug_image(obj_pos):
    img = np.zeros((500, 500, 3))
    cv2.circle(img, obj_pos, 20, (158, 96, 134), -1)
    return img.astype(float) / 255

if __name__ == '__main__':
    img1 = make_debug_image((30, 30))
    img2 = make_debug_image((330, 430))
    dist = color_filter_distance_np(img1, img1)
