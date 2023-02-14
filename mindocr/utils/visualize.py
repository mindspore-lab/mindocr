import cv2
import numpy as np
from typing import Union, List
import matplotlib.pyplot as plt

def show_img(img: np.array, img_mode='BGR', title='img'):
    color = (len(img.shape) == 3 and img.shape[-1] == 3)
    if img_mode == 'BGR':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #imgs = np.expand_dims(imgs, axis=0)
    plt.figure()
    plt.title('{}_{}'.format(title, 0))
    plt.imshow(img, cmap=None if color else 'gray')
    plt.show()

def show_imgs(imgs: List[np.array], img_mode='BGR', title='img'):
    #if len(imgs.shape) not in [2, 4]:
    #    imgs = np.expand_dims(imgs, axis=0)
    plt.figure()
    num_images = len(imgs) #imgs.shape[0]
    for i, img in enumerate(imgs):
        color = (len(img.shape) == 3 and img.shape[-1] == 3)
        if img_mode == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(num_images, 1, i+1)
        plt.title('{}_{}'.format(title, i))
        plt.imshow(img, cmap=None if color else 'gray')
    plt.show()

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