import cv2 
from .random_crop_data import EastRandomCropData

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
        img = cv2.imread(data['img_path'], 1 if self.img_mode != 'GRAY' else 0).astype('float32') 

        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data['image'] = img

        return data

def ResizeByGrid(object):
    '''
    resize image by ratio so that it's shape is align to grid of denominator
    required key in data: img in shape of (h, w, c) 
    '''
    def __init__(self, denominator=32, isTrain=True):
        self.denominator = denominator

    def __call__(self, data):
        img = data['image']
        polys = data['polys']
        
        denominator = self.denominator
        w_scale = math.ceil(img.shape[1] / denominator) * denominator / img.shape[1]
        h_scale = math.ceil(img.shape[0] / denominator) * denominator / img.shape[0]
        img = cv2.resize(img, dsize=None, fx=w_scale, fy=h_scale)
        if polys is None:
            return img
        if isTrain:
            new_polys = []
            for poly in polys:
                poly[:, 0] = poly[:, 0] * w_scale
                poly[:, 1] = poly[:, 1] * h_scale
                new_polys.append(poly)
            polys = new_polys
        else:
            polys[:, :, 0] = polys[:, :, 0] * w_scale
            polys[:, :, 1] = polys[:, :, 1] * h_scale
        return img, polys
    

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

