import cv2
import math
from .random_crop_data import EastRandomCropData
import numpy as np

# default mean/std from ImageNet in RGB order
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data

class DecodeImage(object):
    '''
    img_mode: 'BGR', 'RGB', 'GRAY'
    '''
    def __init__(self, img_mode='BGR', channel_first=False, to_float32=False, **kwargs):
        self.img_mode = img_mode 
        self.to_float32 = to_float32
        self.channel_first = channel_first

    def __call__(self, data):
        # TODO: use more efficient image loader, numpy?
        # TODO: why float32 in modelzoo. uint8 is more efficient
        
        img = cv2.imread(data['img_path'], cv2.IMREAD_COLOR if self.img_mode != 'GRAY' else cv2.IMREAD_GRAYSCALE)
        
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        if self.to_float32:
            img = img.astype('float32')
        data['image'] = img
        data['ori_image'] = img.copy() 
        return data



class TextDetTransform():
    def __init__():
        pass



def create_transforms():
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators


    Return:
        callable 
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]

        op = eval(op_name)(**param)
        ops.append(op)
    return ops

