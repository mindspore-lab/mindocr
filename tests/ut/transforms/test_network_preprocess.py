import sys
sys.path.append('.')

import numpy as np
from mindocr.data.transforms.transforms_factory import NormalizeImage, ToCHWImage

def test_norm():
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    h, w, c = 8, 8, 3
    bgr_value = np.array([128, 10, 255], dtype=np.uint8)
    img = np.ones([h ,w, c], dtype=np.uint8) 
    img[0, 0] = bgr_value
    norm_fn = NormalizeImage(mean=mean, 
                            std=std, 
                            is_hwc=True, bgr_to_rgb=True)
    
    data = {}
    data['image'] = img
    out_data = norm_fn(data)
    out_img = data['image']
    
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    correct = (bgr_value[::-1].astype(np.float32) - mean)/std
    print(out_img.shape)
    print(out_img[0, 0])
    print(correct)
    print(out_img[0, 0] - correct)
    assert np.allclose(out_img[0,0], correct), 'Incorrect norm'


if __name__ == '__main__':    
    test_norm()
