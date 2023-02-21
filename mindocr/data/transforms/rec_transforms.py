'''
transform for text recognition tasks.
'''
from __future__ import absolute_import
from __future__ import division

from typing import List
import cv2
import math
import numpy as np

__all__ = ['CTCLabelEncode', 'RecResizeImg']

# TODO: check
class RecLabelEncode(object):
    """ Convert between text-label and text-index 
    Adopted from paddle
    """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 lower=False):

        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = lower

        if character_dict_path is None:
            print(
                "The character_dict_path is None, model can only recognize number and lower letters"
            )
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True
        else:
            self.character_str = []
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char and ' ' not in self.character_str:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        # TODO: returning None will lead to dataset generator fail
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        text_list = []

        # TODO: for char not in the dictionary, skipping may lead to None data. Use a char replacement? refer to mmocr 
        for char in text:
            if char not in self.dict:
                print('WARNING: {} is not in dict'.format(char))
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:

            return None
        return text_list

# TODO: check
class CTCLabelEncode(RecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super().__init__(
            max_text_length, character_dict_path, use_space_char)

    def __call__(self, data):
        '''
        required keys:
            label -> (str), raw label 
        modified keys:
            label -> (numpy array), sequence of character indices padding to max_text_length in shape (sequence_len) 
        added keys:
            label_ace
        '''
        data['text'] = data['label']
        text = data['label']
        text = self.encode(text)
        # TODO: return None will lead to data gen fail. but is it proper to set all 0's?
        if text is None:
            text = []
            #return None

        data['length'] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text, dtype=np.int32)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data['label_ace'] = np.array(label, dtype=np.int32)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


# TODO: reorganize the code for different resize transformation in rec task
def resize_norm_img(img,
                    image_shape,
                    padding=True,
                    interpolation=cv2.INTER_LINEAR):
    '''
    resize image
    Args:
        img: shape (H, W, C)
        image_shape: image shape after resize, in (C, H, W)
        padding: if Ture, resize while preserving the H/W ratio, then pad the blank.

    '''
    imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    if not padding:
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=interpolation)
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
    '''
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    '''
    padding_im = np.zeros((imgH, imgW, c), dtype=np.uint8)
    padding_im[:, 0:resized_w, :] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio

# TODO: check diff from resize_norm_img
def resize_norm_img_chinese(img, image_shape):
    ''' adopted from paddle
    '''
    imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    c = img.shape[2]
    ratio = w * 1.0 / h
    max_wh_ratio = min(max(max_wh_ratio, ratio), max_wh_ratio)
    imgW = int(imgH * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    '''
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    '''
    #padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im = np.zeros((imgH, imgW, c), dtype=np.uint8)
    #padding_im[:, :, 0:resized_w] = resized_image
    padding_im[:, 0:resized_w, :] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio

# TODO: remove infer_mode and character_dict_path if they are not necesary
class RecResizeImg(object):
    ''' adopted from paddle
    resize, convert from hwc to chw, rescale pixel value to -1 to 1
    '''
    def __init__(self,
                 image_shape,
                 infer_mode=False,
                 character_dict_path=None,
                 padding=True,
                 **kwargs):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_dict_path = character_dict_path
        self.padding = padding

    def __call__(self, data):
        img = data['image']
        if self.infer_mode and self.character_dict_path is not None:
            norm_img, valid_ratio = resize_norm_img_chinese(img,
                                                            self.image_shape)
        else:
            norm_img, valid_ratio = resize_norm_img(img, self.image_shape,
                                                    self.padding)
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data
