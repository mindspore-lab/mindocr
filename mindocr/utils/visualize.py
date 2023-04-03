import cv2
import numpy as np
from typing import Union, List
import matplotlib.pyplot as plt
from ..data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = ['show_img', 'show_imgs', 'draw_bboxes', 'recover_image']


def show_img(img: np.array, is_bgr_img=True, title='img', show=True, save_path=None):
    color = (len(img.shape) == 3 and img.shape[-1] == 3)
    if is_bgr_img:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #imgs = np.expand_dims(imgs, axis=0)
    plt.figure()
    plt.title('{}_{}'.format(title, 0))
    plt.imshow(img, cmap=None if color else 'gray')
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

def show_imgs(imgs: List[np.array], is_bgr_image=True, title='img', show=True, save_path=None):
    #if len(imgs.shape) not in [2, 4]:
    #    imgs = np.expand_dims(imgs, axis=0)
    plt.figure()
    num_images = len(imgs) #imgs.shape[0]
    for i, img in enumerate(imgs):
        color = (len(img.shape) == 3 and img.shape[-1] == 3)
        if is_bgr_image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(num_images, 1, i+1)
        plt.title('{}_{}'.format(title, i))
        plt.imshow(img, cmap=None if color else 'gray')
    if show:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

def draw_bboxes(image: Union[str, np.array], bboxes: np.array, color=(255, 0, 0), thickness=2):
    ''' image can be str or np.array for image in 'BGR' colorm mode. '''
    if isinstance(image, str):
        image = cv2.imread(image)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    else:
        image = image.copy()

    for box in bboxes:
        box = box.astype(int)
        cv2.polylines(image, [box], True, color, thickness)
    return image

def recover_image(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, is_chw=True, to_bgr=True):
    '''
    recover normalized image for visualization
    img: must be in rgb mode'''
    if img.dtype == 'uint8':
        return img

    if is_chw:
        img = img.transpose((1, 2, 0))

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = (img * std) + mean

    if to_bgr:
        img = img[..., [2,1,0]]
    img = img.astype(np.uint8)

    #print(img.max(), img.min())
    return img

