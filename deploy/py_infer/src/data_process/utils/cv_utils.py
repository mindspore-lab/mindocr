import os

import cv2
import numpy as np


def get_hw_of_img(image: np.array):
    """
    get the hw of hwc image
    """
    if len(image.shape) == 3:
        # gbr/rgb
        height, width, _ = image.shape
    elif len(image.shape) == 2:
        # gray
        height, width = image.shape
    else:
        raise TypeError('image is not a image of color/gray')

    return height, width


def crop_box_from_image(image, box):
    if box.shape != (4, 2):
        raise ValueError("shape of crop box must be 4*2")
    box = box.astype(np.float32)
    img_crop_width = int(max(np.linalg.norm(box[0] - box[1]),
                             np.linalg.norm(box[2] - box[3])))
    img_crop_height = int(max(np.linalg.norm(box[0] - box[3]),
                              np.linalg.norm(box[1] - box[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    m = cv2.getPerspectiveTransform(box, pts_std)
    dst_img = cv2.warpPerspective(
        image,
        m, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_width != 0 and dst_img_height / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)

    return dst_img


def img_read(path: str):
    """
    Read a BGR image.
    """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f'Error! Cannot load the image of {path}')

    return img


def img_write(path: str, img: np.ndarray):
    filename = os.path.abspath(path)
    cv2.imencode(os.path.splitext(filename)[1], img)[1].tofile(filename)
